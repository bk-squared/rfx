#!/usr/bin/env python3
"""Ratchet layer implementations and reverse imports using ASTs.

BASE_SHA/HEAD_SHA (or --base/--head) select Git snapshots. Without HEAD_SHA,
read the working tree. Initial adoption requires a readable base with neither
checker nor baseline. --write-baseline records exact current violations; --base
still rejects increases. Surface sizes and allowed calls are informational.
Pure same-file unit renames may match BASE; moves and file renames do not.
Attribute names must remain equal, including attributes of rfx modules.
"""

from __future__ import annotations

import argparse
import ast
import builtins
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
FIVE_MODULES = tuple(p.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
                     for p in FIVE_FILES)
CALLER_MODULES = (
    "rfx.api._execute", "rfx.api", "rfx.simulation", "rfx.nonuniform",
    "rfx.sources.waveguide_port",
    # adjoint, ringdown and progress also call implementations in the layers.
    "rfx.adjoint", "rfx.ringdown", "rfx.progress",
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
        self.definitions = {}

    def read(self, path: str) -> bytes:
        if self.head:
            return git(self.repo, "show", f"{self.head}:{path}")
        return (self.repo / path).read_bytes()

    def tree(self, path: str) -> ast.Module:
        if path not in self.trees:
            self.texts[path] = self.read(path).decode("utf-8")
            self.trees[path] = ast.parse(self.texts[path], filename=path)
        return self.trees[path]

    def own(self, path: str) -> set[str]:
        """``own_definitions`` of one file, computed once."""
        if path not in self.definitions:
            self.definitions[path] = own_definitions(self.tree(path))
        return self.definitions[path]

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
        # A module-level __getattr__ serving names from a literal _LAZY_EXPORTS
        # table (name -> module) is an import edge too: a moved kernel stays
        # countable under its old import path.
        for node in self.tree(path).body if hops else ():
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                    and any(isinstance(t, ast.Name) and t.id == "_LAZY_EXPORTS" for t in node.targets)):
                for key, value in zip(node.value.keys, node.value.values):
                    if (isinstance(key, ast.Constant) and key.value == name
                            and isinstance(value, ast.Constant) and isinstance(value.value, str)):
                        return self._is_function(".".join([value.value, name, *tail]), hops - 1, seen)
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


def reverse_imports(sources: Sources, path: str) -> list[str]:
    """Exact caller-module imports in every scope, including lazy exports."""
    hits = set()
    for node in ast.walk(sources.tree(path)):
        if isinstance(node, ast.ImportFrom):
            module = import_module(path, node)
            for alias in node.names:
                if module in CALLER_MODULES:
                    hits.add(f"{module}:{alias.name}")
                elif f"{module}.{alias.name}" in CALLER_MODULES:  # from rfx import simulation
                    hits.add(f"{module}.{alias.name}")
        elif isinstance(node, ast.Import):
            hits.update(alias.name for alias in node.names if alias.name in CALLER_MODULES)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Dict):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "_LAZY_EXPORTS" for t in targets):
                for key, value in zip(node.value.keys, node.value.values):
                    if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                            and isinstance(value, ast.Constant) and value.value in CALLER_MODULES):
                        hits.add(f"{value.value}:{key.value}")
    return sorted(hits)


def top_level(body):
    """Module-level statements, also inside if/try/with/for/while/match; never function or class bodies."""
    for node in body:
        yield node
        if not isinstance(node, (*FUNCTIONS, ast.ClassDef)):
            for field in ("body", "orelse", "finalbody"):
                yield from top_level(getattr(node, field, []))
            for block in (*getattr(node, "handlers", []), *getattr(node, "cases", [])):
                yield from top_level(block.body)


def own_definitions(tree: ast.Module) -> set[str]:
    """Top-level names a module binds by its own code (def, class, assignment), never by import.

    Statements inside if/try/with/for/while/match count; function and class bodies do not.
    """
    names = set()
    for node in top_level(tree.body):
        if isinstance(node, (*FUNCTIONS, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(n.id for target in targets for n in ast.walk(target)
                         if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
    return names


def lazy_exports(tree: ast.Module) -> dict[str, str]:
    """A literal ``_LAZY_EXPORTS`` table (name -> module) that a package ``__getattr__`` serves."""
    table = {}
    for node in top_level(tree.body):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Dict):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "_LAZY_EXPORTS" for t in targets):
                table.update((key.value, value.value) for key, value in zip(node.value.keys, node.value.values)
                             if isinstance(key, ast.Constant) and isinstance(key.value, str)
                             and isinstance(value, ast.Constant) and isinstance(value.value, str))
    return table


def resolve(sources: Sources, module: str, name: str, seen: frozenset = frozenset()) -> tuple | None:
    """What ``module.name`` is, following re-exports without executing anything.

    ("definition", home, name) when a module's own code defines it, ("module", dotted)
    for a module object, None outside rfx/ or when nothing binds it statically (a star
    import, a dynamic read).
    """
    path = sources.module_path(module)
    if path is None or (module, name) in seen:
        return None
    seen = seen | {(module, name)}
    if name in sources.own(path):
        return ("definition", module, name)
    if sources.module_path(f"{module}.{name}"):  # an imported submodule replaces the package attribute
        return ("module", f"{module}.{name}")
    tree = sources.tree(path)
    for node in top_level(tree.body):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    return resolve(sources, import_module(path, node), alias.name, seen)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname == name and sources.module_path(alias.name):
                    return ("module", alias.name)
    lazy = lazy_exports(tree)
    if name in lazy:
        return resolve(sources, lazy[name], name, seen)
    return None


def definition_reads(sources: Sources, path: str) -> set[tuple[str, str]]:
    """(home, name) for every top-level definition a layer file reads from an rfx module.

    Routes: a from-import, through any chain of re-exports (``from rfx import make_probe``,
    one five file re-exporting another); an attribute read through a module binding
    (``from rfx import simulation``, ``import rfx.simulation as kernel``, dotted
    ``rfx.simulation.make_probe``, nested ``api._execute.X``); a ``_LAZY_EXPORTS`` entry.
    Module aliases are collected per file, not per scope (conservative).
    """
    tree = sources.tree(path)
    reads, modules = set(), {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            parent = import_module(path, node)
            for alias in node.names:
                found = resolve(sources, parent, alias.name)
                if found and found[0] == "definition":
                    reads.add(found[1:])
                elif found:
                    modules[alias.asname or alias.name] = found[1]
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                target = alias.name if alias.asname else bound
                if sources.module_path(target):
                    modules[bound] = target
    for name, module in lazy_exports(tree).items():
        found = resolve(sources, module, name)
        if found and found[0] == "definition":
            reads.add(found[1:])
    for node in ast.walk(tree):
        root = node
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(node, ast.Attribute) and isinstance(root, ast.Name):
            head, *attrs = dotted(node).split(".")
            module = modules.get(head)
            for attr in attrs if module else ():
                found = resolve(sources, module, attr)
                if found and found[0] == "module":
                    module = found[1]
                    continue
                if found:
                    reads.add(found[1:])
                break
    return reads


def caller_definitions(sources: Sources, paths) -> dict:
    """Layer reads of a name a caller module defines itself, not one it re-exports.

    A five-file entry is layer code living in a five file (2.0: zero, whatever the
    baseline says); adjoint, ringdown and progress entries are reported only.
    """
    found = {}
    for path in paths:
        entries = sorted(f"{home}:{name}" for home, name in definition_reads(sources, path)
                         if home in CALLER_MODULES)
        if entries:
            found[path] = entries
    return found


def count_file(sources: Sources, path: str, public: dict | None = None) -> dict:
    """The same per-file counter used by the CLI and real-source mutations."""
    if any(path.startswith(p.replace(".", "/") + "/") for p in LAYERS):
        return {"R": reverse_imports(sources, path)}
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
    files, outside, reverse, unit_counts, allowed = {}, {}, {}, {}, []
    for path in sorted(sources.paths | set(FIVE_FILES)):
        selected = path in FIVE_FILES
        if not selected and any(path.startswith(p.replace(".", "/") + "/") for p in LAYERS):
            rules = count_file(sources, path)
            if rules["R"]:
                reverse[path] = rules
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
    return {"baseline": {"files": files, "outside": outside, "reverse": reverse},
            "caller_definitions": caller_definitions(sources, [
                p for p in sorted(sources.paths)
                if any(p.startswith(layer.replace(".", "/") + "/") for layer in LAYERS)]),
            "surface_sizes": {p: len(names) for p, names in public.items()},
            "unit_counts": unit_counts, "allowed": sorted(allowed)}


def baseline(raw: bytes) -> dict:
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) not in (
            {"files", "outside"}, {"files", "outside", "reverse"}):
        raise ValueError("baseline must contain files and outside, optionally reverse")
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
    if "reverse" in data:
        if not isinstance(data["reverse"], dict):
            raise ValueError("reverse must map layer file paths to R import entries")
        for path in data["reverse"]:
            if (not path.endswith(".py")
                    or any(part in {"", ".", ".."} for part in path.split("/"))
                    or not any(path.startswith(p.replace(".", "/") + "/") for p in LAYERS)):
                raise ValueError(f"{path}: reverse must list Python files under layer packages")
    for group, expected in (("files", RULES), ("outside", ("S", "M")), ("reverse", ("R",))):
        for path, rules in data.get(group, {}).items():
            if not isinstance(rules, dict) or set(rules) != set(expected):
                raise ValueError(f"{path}: baseline must list every rule ({', '.join(expected)})")
            for rule, names in rules.items():
                if (not isinstance(names, list) or not all(isinstance(n, str) and n for n in names)
                        or names != sorted(set(names))):
                    raise ValueError(f"{path} {rule}: names must be sorted and unique")
    return data


def bound_names(node):
    """Names introduced in a subtree, excluding imports and scope declarations."""
    names = set()
    declared = set()
    for item in ast.walk(node):
        if isinstance(item, (ast.Global, ast.Nonlocal)):
            declared.update(item.names)
        elif isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
            names.add(item.id)
        elif isinstance(item, ast.arg):
            names.add(item.arg)
        elif isinstance(item, (*FUNCTIONS, ast.ClassDef)) and item is not node:
            names.add(item.name)
        elif isinstance(item, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and item.name:
            names.add(item.name)
        elif isinstance(item, ast.MatchMapping) and item.rest:
            names.add(item.rest)
    return names - declared


def module_bindings(tree, path):
    """Keep only unambiguous top-level definitions and import bindings."""
    bindings = {}

    def record(name, definition):
        bindings[name] = None if name in bindings else definition

    def visit(node, top=False):
        if isinstance(node, (*FUNCTIONS, ast.ClassDef)):
            record(node.name, node if top else None)
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.asname or (alias.name if isinstance(node, ast.ImportFrom)
                                        else alias.name.split(".")[0])
                target = (import_module(path, node), alias.name) if isinstance(node, ast.ImportFrom) else None
                record(name, target if top else None)
            return
        if isinstance(node, ast.Assign) and top:
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                record(node.targets[0].id, node)
                # The RHS may also bind a name through a walrus.
                visit(node.value)
                return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            record(node.id, None)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            record(node.name, None)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            record(node.rest, None)
        # Comprehension and lambda locals are not module bindings.
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    for node in tree.body:
        visit(node, top=True)
    return bindings


class RenameMatcher:
    """Compare ASTs with bijective, definition-checked identifier changes."""

    def __init__(self, head: Sources, base: str):
        self.head, self.base_revision = head, base
        self.base = None
        self.bindings = {}

    def names(self, sources, path):
        key = (sources.head, path)
        if key not in self.bindings:
            self.bindings[key] = module_bindings(sources.tree(path), path)
        return self.bindings[key]

    def same_unit(self, path, old, new):
        if self.base is None:
            # A baseline entry need not have a source file at BASE. Missing
            # source units cannot match, including synthetic baseline-only refs.
            exists = subprocess.run(
                ["git", "cat-file", "-e", f"{self.base_revision}:{path}"],
                cwd=self.head.repo, capture_output=True,
            )
            if exists.returncode:
                return False
            self.base = Sources(self.head.repo, self.base_revision)
        if path not in self.base.paths or path not in self.head.paths:
            return False
        before = {name: node for name, node, _ in units(self.base.tree(path))}
        after = {name: node for name, node, _ in units(self.head.tree(path))}
        if old not in before or new not in after or new in before:
            return False
        old_prefix, _, old_name = old.rpartition(".")
        new_prefix, _, new_name = new.rpartition(".")
        if old_prefix != new_prefix:
            base_names, head_names = self.names(self.base, path), self.names(self.head, path)
            if (not isinstance(base_names.get(old_prefix), ast.ClassDef)
                    or not isinstance(head_names.get(new_prefix), ast.ClassDef)
                    or new_prefix in base_names):
                return False
        return self.compare(before[old], after[new], path, path,
                            (old_name, new_name), set(), 0,
                            own_references=not old_prefix and not new_prefix)

    def definition_pair(self, base_path, head_path, old, new, assumed, depth):
        key = (base_path, head_path, old, new)
        if key in assumed:
            return True
        if depth > 5 or base_path not in self.base.paths or head_path not in self.head.paths:
            return False
        before, after = self.names(self.base, base_path), self.names(self.head, head_path)
        head_at_base = self.names(self.base, head_path) if head_path in self.base.paths else {}
        if old not in before or new not in after or new in head_at_base:
            return False
        a, b = before[old], after[new]
        assumed = assumed | {key}
        if isinstance(a, ast.AST) and isinstance(b, ast.AST):
            return self.compare(a, b, base_path, head_path, (old, new), assumed, depth,
                                own_references=isinstance(a, FUNCTIONS))
        if isinstance(a, tuple) and isinstance(b, tuple):
            if not all(module == "rfx" or module.startswith("rfx.") for module, _ in (a, b)):
                return False
            old_path, new_path = self.base.module_path(a[0]), self.head.module_path(b[0])
            if old_path and new_path:
                return self.definition_pair(old_path, new_path, a[1], b[1], assumed, depth + 1)
        return False

    def compare(self, before, after, base_path, head_path, own, assumed, depth,
                *, own_references=False):
        local_before = bound_names(before) - self.names(self.base, base_path).keys() - set(dir(builtins))
        local_after = bound_names(after) - self.names(self.head, head_path).keys() - set(dir(builtins))
        forward, reverse = {}, {}

        def identifier(a, b, allow_own=False):
            if a in forward and forward[a] != b or b in reverse and reverse[b] != a:
                return False
            forward[a], reverse[b] = b, a
            return (a == b or (allow_own and (a, b) == own)
                    or (a in local_before and b in local_after)
                    or self.definition_pair(base_path, head_path, a, b, assumed, depth + 1))

        def walk(a, b, identifiers=False, allow_own=False):
            if type(a) is not type(b):
                return False
            if isinstance(a, ast.AST):
                if a._fields != b._fields:
                    return False
                # An import's module is not an identifier that can be freely
                # renamed. Third-party imports must match exactly.
                if isinstance(a, (ast.Import, ast.ImportFrom)):
                    if ast.dump(a) == ast.dump(b):
                        return True
                    if not isinstance(a, ast.ImportFrom):
                        return False
                    modules = import_module(base_path, a), import_module(head_path, b)
                    if not all(m == "rfx" or m.startswith("rfx.") for m in modules):
                        return False
                    paths = self.base.module_path(modules[0]), self.head.module_path(modules[1])
                    if not all(paths) or len(a.names) != len(b.names):
                        return False
                    return all(self.definition_pair(*paths, x.name, y.name, assumed, depth + 1)
                               and identifier(x.asname or x.name, y.asname or y.name)
                               for x, y in zip(a.names, b.names))
                for field in a._fields:
                    left, right = getattr(a, field), getattr(b, field)
                    if field == "body" and isinstance(a, (ast.Module, ast.ClassDef, *FUNCTIONS)):
                        left, right = body_without_docstring(a), body_without_docstring(b)
                    if isinstance(a, ast.Attribute) and field == "attr":
                        if left != right:
                            return False
                        continue
                    is_identifier = field in {"id", "name", "arg", "asname", "rest"} and not isinstance(a, ast.Constant)
                    is_identifier |= isinstance(a, (ast.Global, ast.Nonlocal)) and field == "names"
                    own_field = (a is before and isinstance(a, (*FUNCTIONS, ast.ClassDef))
                                 and field == "name") or (
                                     own_references and isinstance(a, ast.Name) and field == "id")
                    if not walk(left, right, is_identifier, own_field):
                        return False
                return True
            if isinstance(a, list):
                return len(a) == len(b) and all(walk(x, y, identifiers) for x, y in zip(a, b))
            if identifiers and isinstance(a, str):
                return identifier(a, b, allow_own)
            return a == b

        return walk(before, after)

    def pairs(self, path, gone, new):
        matches = {(old, name) for old in gone for name in new if self.same_unit(path, old, name)}
        return {(old, name) for old, name in matches
                if sum(o == old for o, _ in matches) == 1
                and sum(n == name for _, n in matches) == 1}


def violations(actual: dict, current: dict, previous: dict | None = None,
               *, renames: RenameMatcher | None = None) -> list[str]:
    errors = []
    if "reverse" not in current:
        errors.append("HEAD baseline must contain reverse")
    for group, rules in (("files", RULES), ("outside", ("S", "M")), ("reverse", ("R",))):
        if group == "reverse" and group not in current:
            continue
        for path in sorted(actual[group].keys() | current[group].keys()):
            for rule in rules:
                found = set(actual[group].get(path, {}).get(rule, []))
                stored = set(current[group].get(path, {}).get(rule, []))
                for name in sorted(found - stored):
                    errors.append(f"{path} {rule}: unlisted violation {name}")
                for name in sorted(stored - found):
                    errors.append(f"{path} {rule}: stale baseline entry {name}")
                if previous is not None and group in previous:
                    old = set(previous[group].get(path, {}).get(rule, []))
                    pairs = (renames.pairs(path, old - found, found - old)
                             if renames and group != "reverse" else set())
                    accepted = {name for _, name in pairs}
                    for before, after in sorted(pairs):
                        print(f"renamed: {path} {rule} {before} -> {after}")
                    for name in sorted(found - old - accepted):
                        errors.append(f"{path} {rule}: new violation against BASE {name}")
                    for name in sorted(stored - old - accepted):
                        errors.append(f"{path} {rule}: baseline grew against BASE {name}")
    return errors


def five_file_definitions(defined: dict) -> list[str]:
    return [f"{path} R: imports {entry}, which the five file defines; move the definition into its layer"
            for path, entries in sorted(defined.items()) for entry in entries
            if entry.partition(":")[0] in FIVE_MODULES]


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
    print("\nreverse imports from layers to callers — may only fall")
    reverse = result["baseline"]["reverse"]
    for path, rules in sorted(reverse.items(), key=lambda item: (-len(item[1]["R"]), item[0])):
        print(f"reverse {path}: R={len(rules['R'])}")
    print(f"reverse totals: R={sum(len(rules['R']) for rules in reverse.values())}")
    defined = result["caller_definitions"]
    if defined:
        print("\nlayer imports of names a caller module defines itself — five files must be 0")
        for path, entries in sorted(defined.items()):
            print(f"defined {path}: {' '.join(entries)}")
        five = sum(e.partition(":")[0] in FIVE_MODULES for entries in defined.values() for e in entries)
        other = sum(len(entries) for entries in defined.values()) - five
        print(f"defined totals: five files={five} other callers={other}")



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
    errors = violations(actual, current, previous,
                        renames=RenameMatcher(sources, base) if previous is not None else None)
    errors += five_file_definitions(result["caller_definitions"])
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
