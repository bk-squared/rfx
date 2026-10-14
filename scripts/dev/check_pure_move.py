"""Compare Python units across Git revisions using an explicit rename map.

IDENTICAL covers the unit bodies only; the new module's imports, globals and
__future__ flags are not compared.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import subprocess


DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
SCOPE_NOTE = (
    "note: IDENTICAL covers the unit bodies only; the new module's imports, "
    "globals and __future__ flags are not compared"
)


def parse_endpoint(value: str) -> tuple[str, str]:
    path, separator, name = value.rpartition(":")
    if not separator or not path or not all(
        part.isidentifier() for part in name.split(".")
    ):
        raise ValueError(f"malformed pair endpoint: {value!r}")
    return path, name


def parse_pair(value: str) -> tuple[tuple[str, str], tuple[str, str]]:
    parts = value.split("=")
    if len(parts) != 2:
        raise ValueError(f"malformed pair: {value!r}")
    return parse_endpoint(parts[0]), parse_endpoint(parts[1])


def add_rename(renames: dict[str, str], value: str) -> None:
    parts = value.split("=")
    if len(parts) != 2 or not all(
        part and all(piece.isidentifier() for piece in part.split("."))
        for part in parts
    ):
        raise ValueError(f"malformed rename: {value!r}")
    old, new = parts
    if old in renames and renames[old] != new:
        raise ValueError(f"conflicting rename for {old}: {renames[old]} and {new}")
    renames[old] = new


class AnnotationStrings(ast.NodeTransformer):
    """Expand parsable strings anywhere within an annotation expression."""

    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self.aliases = aliases or {}

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if not isinstance(node.value, str):
            return node
        try:
            expression = ast.parse(node.value, mode="eval", type_comments=True).body
        except SyntaxError:
            return node
        return ast.copy_location(self.visit(expression), node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        name = (
            node.value.id if isinstance(node.value, ast.Name)
            else node.value.attr if isinstance(node.value, ast.Attribute) else None
        )
        node.value = self.visit(node.value)
        name = self.aliases.get(name, name)
        if name == "Literal":
            return node
        if name == "Annotated":
            if isinstance(node.slice, ast.Tuple):
                if node.slice.elts:
                    node.slice.elts[0] = self.visit(node.slice.elts[0])
            else:
                node.slice = self.visit(node.slice)
            return node
        node.slice = self.visit(node.slice)
        return node


class NormalizeAnnotations(ast.NodeTransformer):
    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self.aliases = aliases or {}

    def generic_visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, (ast.arg, ast.AnnAssign)) and node.annotation is not None:
            node.annotation = AnnotationStrings(self.aliases).visit(node.annotation)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                node.returns = AnnotationStrings(self.aliases).visit(node.returns)
        return super().generic_visit(node)


class RenameIdentifiers(ast.NodeTransformer):
    """Apply simultaneous exact replacements; attributes and keywords stay literal."""

    def __init__(self, renames: dict[str, str]) -> None:
        self.renames = renames

    def generic_visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Name):
            node.id = self.renames.get(node.id, node.id)
        elif isinstance(node, DEFINITIONS):
            node.name = self.renames.get(node.name, node.name)
        elif isinstance(node, ast.alias):
            node.name = self.renames.get(node.name, node.name)
            node.asname = self.renames.get(node.asname, node.asname)
        elif isinstance(node, ast.ImportFrom):
            node.module = self.renames.get(node.module, node.module)
        elif isinstance(node, ast.arg):
            node.arg = self.renames.get(node.arg, node.arg)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            node.name = self.renames.get(node.name, node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            node.names = [self.renames.get(name, name) for name in node.names]
        elif isinstance(node, ast.MatchMapping):
            node.rest = self.renames.get(node.rest, node.rest)
        return super().generic_visit(node)


def addressable_units(tree: ast.Module | None) -> dict[str, list[ast.stmt]]:
    units: dict[str, list[ast.stmt]] = {}

    def collect(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, DEFINITIONS):
                name = prefix + node.name
                units.setdefault(name, []).append(node)
                if isinstance(node, ast.ClassDef):
                    collect(node.body, name + ".")

    if tree is not None:
        collect(tree.body)
    return units


def listed_names(tree: ast.Module | None) -> set[str]:
    names: set[str] = set()

    def bind(target: ast.AST | None) -> None:
        if isinstance(target, ast.Name):
            names.add(target.id)

    def collect(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, DEFINITIONS):
                name = prefix + node.name
                names.add(name)
                if isinstance(node, ast.ClassDef):
                    collect(node.body, name + ".")
                continue
            # Class members contribute definitions, not module-level bindings.
            if prefix:
                continue
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                targets = [node.target]
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                targets = [item.optional_vars for item in node.items]
            for target in targets:
                bind(target)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.asname or (
                        alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name
                    )
                    if name != "*":
                        names.add(name)
            if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While)):
                collect(node.body)
                collect(node.orelse)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                collect(node.body)
            elif isinstance(node, ast.Try):
                collect(node.body)
                for handler in node.handlers:
                    if handler.name:
                        names.add(handler.name)
                    collect(handler.body)
                collect(node.orelse)
                collect(node.finalbody)

    if tree is not None:
        collect(tree.body)
    return names


def identifier_names(unit: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(unit):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, DEFINITIONS):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            names.add(node.name)
            if node.asname:
                names.add(node.asname)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            if node.name:
                names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
    return names


def typing_aliases(tree: ast.Module | None) -> dict[str, str]:
    """Local names bound to typing's Literal or Annotated by an import anywhere in the module."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree) if tree is not None else ():
        if isinstance(node, ast.ImportFrom) and node.module in {"typing", "typing_extensions"}:
            for alias in node.names:
                if alias.name in {"Literal", "Annotated"} and alias.asname:
                    aliases[alias.asname] = alias.name
    return aliases


def renamed_parameters(units: ast.Module, renames: dict[str, str]) -> list[str]:
    """Keyword-callable parameters of the compared definitions that the map renames."""
    found = set()
    for unit in units.body:
        defs = [unit] + (list(unit.body) if isinstance(unit, ast.ClassDef) else [])
        for node in defs:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in [*node.args.args, *node.args.kwonlyargs]:
                    if arg.arg in renames:
                        found.add(f"{arg.arg} -> {renames[arg.arg]}")
    return sorted(found)


def class_context_losses(old_unit: ast.AST, new_unit: ast.AST, old_name: str, new_name: str) -> list[str]:
    old_class = old_name.rsplit(".", 1)[0].rsplit(".", 1)[-1] if "." in old_name else None
    new_class = new_name.rsplit(".", 1)[0].rsplit(".", 1)[-1] if "." in new_name else None
    if old_class == new_class:
        return []
    losses = set()
    if old_class is not None:
        losses |= context_uses(old_unit)
    if new_class is not None:
        losses |= context_uses(new_unit)
    return sorted(losses)


def context_uses(unit: ast.AST) -> set[str]:
    names = identifier_names(unit)
    private_names = names | {
        node.attr for node in ast.walk(unit) if isinstance(node, ast.Attribute)
    }
    losses = {name for name in private_names if name.startswith("__") and not name.endswith("__")}
    if "__class__" in names:
        losses.add("__class__")
    for node in ast.walk(unit):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "super" and not node.args and not node.keywords):
            losses.add("super()")
    return losses


def git(*args: str, cwd: str | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True,
            encoding="utf-8", timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as exc:
        raise ValueError(f"git read error: {exc}") from exc
    if result.returncode:
        raise ValueError(f"git read error: {result.stderr.strip()}")
    return result.stdout


class RevisionFiles:
    def __init__(self, base: str, head: str) -> None:
        resolved = git(
            "rev-parse", "--show-toplevel", "--revs-only", "--end-of-options",
            base + "^{tree}", head + "^{tree}",
        ).splitlines()
        if len(resolved) != 3:
            raise ValueError(f"git read error: cannot resolve trees for {base!r} and {head!r}")
        self.root = resolved[0]
        self.refs = dict(zip((base, head), resolved[1:]))
        self.cache: dict[tuple[str, str], str | None] = {}

    def read(self, ref: str, path: str) -> ast.Module | None:
        key = ref, path
        if key not in self.cache:
            revision = self.refs[ref]
            try:
                self.cache[key] = git("show", revision + ":" + path, cwd=self.root)
            except ValueError:
                entry = git("ls-tree", "-z", "--full-tree", revision, "--",
                            ":(literal)" + path, cwd=self.root)
                if entry:
                    raise
                self.cache[key] = None
        source = self.cache[key]
        return None if source is None else ast.parse(
            source, filename=f"{ref}:{path}", type_comments=True
        )


def first_diff(old: ast.AST, new: ast.AST) -> list[str]:
    diff = difflib.unified_diff(
        ast.unparse(old).splitlines(), ast.unparse(new).splitlines(),
        fromfile="old (renamed)", tofile="new", n=0, lineterm="",
    )
    lines: list[str] = []
    in_hunk = False
    for line in diff:
        if line.startswith("@@"):
            if in_hunk:
                break
            in_hunk = True
        if in_hunk:
            lines.append(line)
            if len(lines) == 20:
                break
    return lines


def check_pairs(files: RevisionFiles, base: str, head: str, pairs: list,
                renames: dict[str, str]) -> int:
    status = 0
    for (old_path, old_name), (new_path, new_name) in pairs:
        old = addressable_units(files.read(base, old_path)).get(old_name)
        new = addressable_units(files.read(head, new_path)).get(new_name)
        label = f"{old_path}:{old_name} -> {new_path}:{new_name}"
        if old is None or new is None:
            missing = "both" if old is None and new is None else "old" if old is None else "new"
            print(f"{label}  MISSING ({missing})")
            status = 1
            continue
        # Each read parses a fresh tree so overlapping pairs cannot mutate each other.
        details = []
        if len(old) != len(new):
            details.append(f"definitions: {len(old)} old, {len(new)} new")
        old = ast.Module(body=old, type_ignores=[])
        new = ast.Module(body=new, type_ignores=[])
        losses = class_context_losses(old, new, old_name, new_name)
        details.extend(f"class context lost: {loss}" for loss in losses)
        old = NormalizeAnnotations(typing_aliases(files.read(base, old_path))).visit(old)
        new = NormalizeAnnotations(typing_aliases(files.read(head, new_path))).visit(new)
        collisions = sorted(set(renames.values()) & identifier_names(old))
        details.extend(f"rename collision: {name} already used in the old unit" for name in collisions)
        details.extend(f"parameter renamed: {item} (callers passing it by keyword break)"
                       for item in renamed_parameters(old, renames))
        old = RenameIdentifiers(renames).visit(old)
        if not details and ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False):
            print(f"{label}  IDENTICAL")
        else:
            print(f"{label}  DIFFERENT")
            for detail in details:
                print(detail)
            for line in first_diff(old, new):
                print(line)
            status = 1

    print(SCOPE_NOTE)
    uncovered = []
    for path in sorted({old[0] for old, _ in pairs}):
        covered = {old[1] for old, _ in pairs if old[0] == path}
        deleted = listed_names(files.read(base, path)) - listed_names(files.read(head, path))
        for name in sorted(deleted):
            if not any(name == unit or name.startswith(unit + ".") for unit in covered):
                uncovered.append(f"{path}:{name}")
    if uncovered:
        print("deleted at HEAD, not in any pair:")
        for name in uncovered:
            print(name)
    else:
        print("deleted at HEAD, not in any pair: none")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--pair", action="append", default=[])
    parser.add_argument("--pairs-file", action="append", default=[])
    parser.add_argument("--rename", action="append", default=[])
    args = parser.parse_args()
    try:
        pair_values = list(args.pair)
        rename_values = list(args.rename)
        for filename in args.pairs_file:
            with open(filename, encoding="utf-8") as stream:
                for line in stream:
                    line = line.partition("#")[0].strip()
                    if line.startswith("rename "):
                        rename_values.append(line[len("rename "):].strip())
                    elif line:
                        pair_values.append(line)
        if not pair_values:
            raise ValueError("at least one pair is required")
        pairs = [parse_pair(value) for value in pair_values]
        renames: dict[str, str] = {}
        for (_, old), (_, new) in pairs:
            old, new = old.rsplit(".", 1)[-1], new.rsplit(".", 1)[-1]
            if old != new:
                add_rename(renames, f"{old}={new}")
        for value in rename_values:
            add_rename(renames, value)
        return check_pairs(RevisionFiles(args.base, args.head), args.base, args.head, pairs, renames)
    except (ValueError, OSError, SyntaxError, UnicodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
