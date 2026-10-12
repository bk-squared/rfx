#!/usr/bin/env python3
"""Ratchet layer implementations in five callers and outside layers using ASTs.

BASE_SHA/HEAD_SHA (or --base/--head) select Git snapshots. Without HEAD_SHA,
read the working tree. Initial adoption requires a readable base with neither
checker nor baseline. --write-baseline records exact current violations; --base
still rejects increases. Surface sizes and allowed calls are informational.
"""

from __future__ import annotations

import argparse
import ast
from copy import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
BASELINE = "scripts/ci/layer_baseline.json"
CHECKER = "scripts/ci/check_layer_structure.py"
FIVE_FILES = (
    "rfx/api/_execute.py", "rfx/api/__init__.py", "rfx/simulation.py",
    "rfx/nonuniform.py", "rfx/sources/waveguide_port.py",
)
LAYERS = ("rfx.model", "rfx.stepping", "rfx.measurement", "rfx.boundaries")
KERNEL_MODULES = {"rfx.core.yee", "rfx.core.dft_utils"}
RULES = ("I", "S", "M", "A", "G", "D1", "D2")
LOOPS = {"scan", "recorded_scan", "fori_loop", "while_loop", "shard_map",
         "rank_shard_map", "checkpointed_scan"}
KERNEL_NAMES = {"core_step", "_core_step", "step_fn", "_single_step",
                "step_with_previous", "apply_ordered", "slab_neighbor"}
NOT_STEP = {"update_waveguide_port_probe", "update_overlap_dft"}
KERNEL_RE = re.compile(
    r"^(update_e|update_h|update_he|apply_cpml|apply_upml|apply_pec|apply_pmc|"
    r"apply_tfsf|update_tfsf|update_rlc|apply_waveguide_port|inject_|exchange_)"
    r"|^(lorentz|debye)_[a-z]_component$|^make_.*_step$"
)
FIELDS = {"ex", "ey", "ez", "hx", "hy", "hz"}
ALLOW = {
    "asarray", "array", "float32", "float64", "float16", "complex64", "complex128",
    "int32", "int64", "bool_", "dtype", "isfinite", "isnan", "isscalar", "ndim",
    "shape", "result_type", "issubdtype", "can_cast", "all", "any", "size",
    "floating", "integer", "complexfloating", "number", "generic", "ndarray",
    "finfo", "iinfo", "isclose", "allclose", "errstate",
}
ARITHMETIC = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
              ast.MatMult, ast.Pow)
GRIDP = {"_build_grid", "_build_nonuniform_grid", "_assemble_materials",
         "_assemble_materials_nu", "position_to_index", "make_nonuniform_grid",
         "realized_geometry", "_assemble"}
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True,
    ).stdout


class Sources:
    """One consistent source tree, including all import targets at HEAD."""

    def __init__(self, repo: Path, head: str = ""):
        self.repo, self.head = repo, head
        if head:
            self.paths = {p.decode() for p in git(
                repo, "ls-tree", "-r", "--name-only", "-z", head, "--", "rfx/",
            ).split(b"\0") if p.endswith(b".py")}
        else:
            self.paths = {p.relative_to(repo).as_posix()
                          for p in (repo / "rfx").rglob("*.py")}
        self.trees = {}
        self.texts = {}

    def read(self, path: str) -> bytes:
        if self.head:
            return git(self.repo, "show", f"{self.head}:{path}")
        return (self.repo / path).read_bytes()

    def tree(self, path: str) -> ast.Module:
        if path not in self.trees:
            self.texts[path] = self.read(path).decode("utf-8")
            self.trees[path] = ast.parse(self.texts[path], filename=path)
        return self.trees[path]

    def module_path(self, module: str) -> str | None:
        stem = module.replace(".", "/")
        return next((p for p in (stem + ".py", stem + "/__init__.py")
                     if p in self.paths), None)

    def is_function(self, module: str, name: str) -> bool:
        return self._is_function(f"{module}.{name}", 3, set())

    def _is_function(self, qualified: str, hops: int, seen: set[str]) -> bool:
        """Resolve at most three import edges, without executing a re-export."""
        if qualified in seen:
            return False
        seen = seen | {qualified}
        parts = qualified.split(".")
        # The longest real module prefix also handles a module re-export:
        # facade.alias.f -> defining_module.f, where facade imports alias.
        for split in range(len(parts) - 1, 0, -1):
            module = ".".join(parts[:split])
            path = self.module_path(module)
            if path:
                break
        else:
            return False
        name, *tail = parts[split:]

        def bindings(node):
            yield node
            if not isinstance(node, (*FUNCTIONS, ast.ClassDef)):
                for child in ast.iter_child_nodes(node):
                    yield from bindings(child)

        for node in bindings(self.tree(path)):
            if isinstance(node, (*FUNCTIONS, ast.ClassDef)) and node.name == name:
                return isinstance(node, FUNCTIONS) and not tail
            if hops == 0:
                continue
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if (alias.asname or alias.name) == name:
                        target = import_module(path, node) + "." + alias.name
                        return self._is_function(".".join([target, *tail]), hops - 1, seen)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if (alias.asname or alias.name.split(".")[0]) == name:
                        target = alias.name if alias.asname else alias.name.split(".")[0]
                        return self._is_function(".".join([target, *tail]), hops - 1, seen)
        return False


def dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def import_module(path: str, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    package = path.split("/")[:-1]
    if node.level > len(package):
        raise ValueError(f"{path}: relative import escapes the package")
    return ".".join(package[:len(package) - node.level + 1]
                    + ([node.module] if node.module else []))


def layer_of(module: str) -> str | None:
    return next((p for p in LAYERS if module == p or module.startswith(p + ".")), None)


def public_names(sources: Sources, package: str) -> list[str]:
    tree = sources.tree(package.replace(".", "/") + "/__init__.py")
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            names = ast.literal_eval(node.value)
            if not isinstance(names, (list, tuple)) or not all(isinstance(n, str) for n in names):
                raise ValueError(f"{package}: __all__ must be a literal list of names")
            return names
    return []


def units(tree: ast.Module):
    """Partition all code outside function bodies, without double counting.

    Control-flow blocks do not introduce a qualifier. Classes do. A function
    owns its entire subtree, including any nested functions/classes. Remaining
    statements belong to one pseudo-unit per module or class scope.
    """
    found = []

    def scope(body, prefix="", headers=()):
        def partition(node):
            if isinstance(node, FUNCTIONS):
                found.append((prefix + node.name, node, prefix[:-1] or None))
                return None
            if isinstance(node, ast.ClassDef):
                # Class decorators, bases and keywords may also execute code.
                scope(node.body, prefix + node.name + ".",
                      [*node.decorator_list, *node.bases, *node.keywords])
                return None
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)):
                return None
            if not prefix and isinstance(node, (ast.Import, ast.ImportFrom)):
                return None
            result = copy(node)
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    setattr(result, field, [part for child in value
                            if (part := partition(child) if isinstance(child, ast.AST) else child) is not None])
                elif isinstance(value, ast.AST):
                    setattr(result, field, partition(value))
            return result

        remaining = [part for node in body if (part := partition(node)) is not None]
        if remaining or headers:
            name = prefix + "<body>" if prefix else "<module>"
            found.append((name, ast.Module(body=[*headers, *remaining], type_ignores=[]), None))

    scope(tree.body)
    yield from found


def body_without_docstring(node):
    body = node.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        return body[1:]
    return body


def array_call(call: ast.Call) -> bool:
    name = dotted(call.func)
    return name.split(".")[0] in {"jnp", "np", "numpy"} or name.startswith("jax.numpy.")


def unit_hits(node: ast.AST) -> set[str]:
    hits = set()
    for item in ast.walk(node):
        if ((isinstance(item, ast.Name) and item.id in GRIDP)
                or (isinstance(item, ast.Attribute) and item.attr in GRIDP)):
            hits.add("grid")
        if not isinstance(item, ast.Call):
            continue
        name = dotted(item.func)
        last = name.rsplit(".", 1)[-1]
        if (last in LOOPS or ((last in KERNEL_NAMES or KERNEL_RE.search(last))
                             and last not in NOT_STEP)
                or (last == "_replace" and any(k.arg in FIELDS for k in item.keywords))):
            hits.add("S")
        arguments = [*item.args, *(k.value for k in item.keywords)]
        if (".fft." in name or (last == "exp" and any(
                isinstance(n, ast.Constant) and isinstance(n.value, complex)
                for arg in arguments for n in ast.walk(arg)))):
            hits.add("M")
        if array_call(item) and (last not in ALLOW or any(
                isinstance(n, ast.BinOp) and isinstance(n.op, ARITHMETIC)
                for arg in arguments for n in ast.walk(arg))):
            hits.add("G")
        if (last in {"set", "add", "multiply", "min", "max", "divide", "power", "apply"}
                and isinstance(item.func, ast.Attribute)
                and isinstance(item.func.value, ast.Subscript)
                and isinstance(item.func.value.value, ast.Attribute)
                and item.func.value.value.attr == "at"):
            hits.add("A")
    return hits


def import_hits(sources: Sources, path: str, public: dict) -> list[str]:
    """Direct function imports and reads through lexically visible module imports.

    Module bindings are scoped so a local alias in one method does not apply
    to an unrelated method. As with the other rules, dynamic aliases are not
    followed. Private module attributes count even when they name a record.
    """
    hits = set()

    class Reads(ast.NodeVisitor):
        def __init__(self):
            self.bindings = {}

        def scope(self, node):
            # Defaults, decorators, bases and annotations also contain reads,
            # evaluated in the enclosing scope (including postponed annotations).
            for field, value in ast.iter_fields(node):
                if field != "body":
                    for item in value if isinstance(value, list) else [value]:
                        if isinstance(item, ast.AST):
                            self.visit(item)
            previous = self.bindings
            self.bindings = previous.copy()

            def bind(item):
                if isinstance(item, ast.Import):
                    for alias in item.names:
                        self.bindings[alias.asname or alias.name.split(".")[0]] = (
                            alias.name if alias.asname else alias.name.split(".")[0])
                elif isinstance(item, ast.ImportFrom):
                    module = import_module(path, item)
                    layer = layer_of(module)
                    for alias in item.names:
                        key = alias.asname or alias.name
                        self.bindings[key] = None
                        target = f"{module}.{alias.name}"
                        if sources.module_path(target):
                            self.bindings[key] = target
                        private_or_star = layer and (alias.name.startswith("_") or alias.name == "*")
                        if private_or_star or (sources.is_function(module, alias.name) and (
                                module in KERNEL_MODULES or (layer and (
                                    module != layer or alias.name not in public[layer])))):
                            hits.add(f"{module}:{alias.name}")
                elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    self.bindings[item.name] = None
                    return
                elif isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
                    self.bindings[item.id] = None
                for child in ast.iter_child_nodes(item):
                    bind(child)

            if isinstance(node, FUNCTIONS):
                for arg in ast.walk(node.args):
                    if isinstance(arg, ast.arg):
                        self.bindings[arg.arg] = None
            for statement in node.body:
                bind(statement)
            for statement in node.body:
                self.visit(statement)
            self.bindings = previous

        visit_Module = scope
        visit_FunctionDef = scope
        visit_AsyncFunctionDef = scope
        visit_ClassDef = scope

        def visit_Attribute(self, node):
            if isinstance(node.ctx, ast.Load):
                spelling = dotted(node)
                root, *tail = spelling.split(".")
                bound = self.bindings.get(root)
                if bound and tail:
                    module = ".".join([bound, *tail[:-1]])
                    layer = layer_of(module)
                    if layer and (node.attr.startswith("_") or (
                            sources.is_function(module, node.attr)
                            and node.attr not in public[layer])):
                        hits.add(f"{module}:{node.attr}")
            self.generic_visit(node)

    Reads().visit(sources.tree(path))
    return sorted(hits)


def count_file(sources: Sources, path: str, public: dict | None = None) -> dict:
    """The same per-file counter used by the CLI and real-source mutations."""
    rules = {rule: set() for rule in RULES}
    for name, node, cls in units(sources.tree(path)):
        hits = unit_hits(node)
        for rule in "SMAG":
            if rule in hits:
                rules[rule].add(name)
        if path == "rfx/api/__init__.py" and cls == "Simulation":
            short = node.name
            declaration = short.startswith("add_") or short in {"add", "__init__"}
            if declaration and "grid" in hits:
                rules["D1"].add(name)
            elif (not declaration and not (short.startswith("__") and short.endswith("__"))
                  and len(body_without_docstring(node)) > 1
                  and not (short.startswith("_") and "grid" not in hits)):
                rules["D2"].add(name)
    if path in FIVE_FILES:
        if public is None:
            public = {p: public_names(sources, p) for p in LAYERS}
        rules["I"] = set(import_hits(sources, path, public))
    return {r: sorted(rules[r]) for r in RULES}


def count(sources: Sources) -> dict:
    """Count named units and imports without importing any inspected module."""
    public = {package: public_names(sources, package) for package in LAYERS}
    files, outside, unit_counts, allowed = {}, {}, {}, []
    for path in sorted(sources.paths | set(FIVE_FILES)):
        selected = path in FIVE_FILES
        if not selected and any(path.startswith(p.replace(".", "/") + "/") for p in LAYERS):
            continue
        tree = sources.tree(path)
        rules = count_file(sources, path, public)
        if selected:
            files[path] = rules
            unit_counts[path] = sum(1 for _ in units(tree))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and array_call(node)
                        and dotted(node.func).rsplit(".", 1)[-1] in ALLOW):
                    source = ast.get_source_segment(sources.texts[path], node)
                    allowed.append((path, node.lineno, " ".join(source.splitlines())))
        elif rules["S"] or rules["M"]:
            outside[path] = {r: sorted(rules[r]) for r in ("S", "M")}
    return {"baseline": {"files": files, "outside": outside},
            "surface_sizes": {p: len(names) for p, names in public.items()},
            "unit_counts": unit_counts, "allowed": sorted(allowed)}


def baseline(raw: bytes) -> dict:
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != {"files", "outside"}:
        raise ValueError("baseline must contain files and outside")
    if not isinstance(data["files"], dict) or set(data["files"]) != set(FIVE_FILES):
        raise ValueError("baseline must list exactly the five files")
    if not isinstance(data["outside"], dict):
        raise ValueError("outside must map file paths to S and M unit names")
    for path in data["outside"]:
        if (not path.startswith("rfx/") or not path.endswith(".py")
                or any(part in {"", ".", ".."} for part in path.split("/"))
                or path in FIVE_FILES
                or any(path.startswith(p.replace(".", "/") + "/") for p in LAYERS)):
            raise ValueError(f"{path}: outside must list Python files outside the five files and layers")
    for group, expected in (("files", RULES), ("outside", ("S", "M"))):
        for path, rules in data[group].items():
            if not isinstance(rules, dict) or set(rules) != set(expected):
                raise ValueError(f"{path}: baseline must list every rule ({', '.join(expected)})")
            for rule, names in rules.items():
                if (not isinstance(names, list) or not all(isinstance(n, str) and n for n in names)
                        or names != sorted(set(names))):
                    raise ValueError(f"{path} {rule}: names must be sorted and unique")
    return data


def violations(actual: dict, current: dict, previous: dict | None = None) -> list[str]:
    errors = []
    for group, rules in (("files", RULES), ("outside", ("S", "M"))):
        for path in sorted(actual[group].keys() | current[group].keys()):
            for rule in rules:
                found = set(actual[group].get(path, {}).get(rule, []))
                stored = set(current[group].get(path, {}).get(rule, []))
                for name in sorted(found - stored):
                    errors.append(f"{path} {rule}: unlisted violation {name}")
                for name in sorted(stored - found):
                    errors.append(f"{path} {rule}: stale baseline entry {name}")
                if previous is not None:
                    old = set(previous[group].get(path, {}).get(rule, []))
                    for name in sorted(found - old):
                        errors.append(f"{path} {rule}: new violation against BASE {name}")
                    for name in sorted(stored - old):
                        errors.append(f"{path} {rule}: baseline grew against BASE {name}")
    return errors


def report(result: dict):
    print("2.0 condition — five files")
    for path in FIVE_FILES:
        rules = result["baseline"]["files"][path]
        all_units = set().union(*(rules[r] for r in "SMAG"))
        summary = f"{path}: units={len(all_units)}/{result['unit_counts'][path]} imports={len(rules['I'])}"
        if path == "rfx/api/__init__.py":
            summary += f" declarations={len(rules['D1']) + len(rules['D2'])}"
        print(summary + " " + " ".join(f"{r}={len(rules[r])}" for r in RULES))
    print("surface sizes: " + " ".join(f"{p}={n}" for p, n in result["surface_sizes"].items()))
    print("\noutside the layer packages — may only fall, not a 2.0 condition "
          "(PI to decide how far 2.0 reaches)")
    outside = result["baseline"]["outside"]
    for path, rules in sorted(outside.items(), key=lambda item: (
            -sum(len(names) for names in item[1].values()), item[0])):
        print(f"outside {path}: S={len(rules['S'])} M={len(rules['M'])}")
    totals = {r: sum(len(v[r]) for v in outside.values()) for r in ("S", "M")}
    print("outside totals: " + " ".join(f"{r}={n}" for r, n in totals.items()))



def check(repo: Path, base: str = "", head: str = "", *, write_baseline=False, list_allowed=False) -> int:
    sources = Sources(repo, head)
    result = count(sources)
    report(result)
    if list_allowed:
        for path, line, source in result["allowed"]:
            print(f"{path}:{line}: {source}")
    previous = None
    if base:
        if git(repo, "ls-tree", base, "--", BASELINE):
            previous = baseline(git(repo, "show", f"{base}:{BASELINE}"))
        elif git(repo, "ls-tree", base, "--", CHECKER):
            raise ValueError(f"BASE contains {CHECKER} but is missing {BASELINE}")
    actual = result["baseline"]
    if write_baseline:
        (repo / BASELINE).write_text(json.dumps(actual, indent=2) + "\n")
        current = actual
    else:
        current = baseline(sources.read(BASELINE))
    errors = violations(actual, current, previous)
    for error in errors:
        print(error, file=sys.stderr)
    return int(bool(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--base", default=os.environ.get("BASE_SHA", ""))
    parser.add_argument("--head", default=os.environ.get("HEAD_SHA", ""))
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--list-allowed", action="store_true")
    args = parser.parse_args(argv)
    try:
        return check(args.repo, args.base, args.head,
                     write_baseline=args.write_baseline, list_allowed=args.list_allowed)
    except (OSError, ValueError, SyntaxError, subprocess.CalledProcessError) as exc:
        print(f"layer-structure ratchet failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
