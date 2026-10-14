"""Compare Python units across Git revisions using an explicit rename map."""

from __future__ import annotations

import argparse
import ast
import difflib
import subprocess


DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


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

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if not isinstance(node.value, str):
            return node
        try:
            expression = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return node
        return ast.copy_location(self.visit(expression), node)


class NormalizeAnnotations(ast.NodeTransformer):
    def generic_visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, (ast.arg, ast.AnnAssign)) and node.annotation is not None:
            node.annotation = AnnotationStrings().visit(node.annotation)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                node.returns = AnnotationStrings().visit(node.returns)
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
        return super().generic_visit(node)


def addressable_units(tree: ast.Module | None) -> dict[str, ast.AST]:
    units: dict[str, ast.AST] = {}

    def collect(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, DEFINITIONS):
                name = prefix + node.name
                units[name] = node
                if isinstance(node, ast.ClassDef):
                    collect(node.body, name + ".")

    if tree is not None:
        collect(tree.body)
    return units


def listed_names(tree: ast.Module | None) -> set[str]:
    names = set(addressable_units(tree))
    if tree is not None:
        for node in tree.body:
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


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
        self.root = git("rev-parse", "--show-toplevel").strip()
        self.refs = {
            ref: git("rev-parse", "--verify", "--end-of-options", ref + "^{tree}",
                     cwd=self.root).strip()
            for ref in (base, head)
        }
        self.cache: dict[tuple[str, str], str | None] = {}

    def read(self, ref: str, path: str) -> ast.Module | None:
        key = ref, path
        if key not in self.cache:
            revision = self.refs[ref]
            entry = git("ls-tree", "-z", "--full-tree", revision, "--",
                        ":(literal)" + path, cwd=self.root)
            if not entry:
                self.cache[key] = None
            else:
                self.cache[key] = git("show", revision + ":" + path, cwd=self.root)
        source = self.cache[key]
        return None if source is None else ast.parse(source, filename=f"{ref}:{path}")


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
        old = NormalizeAnnotations().visit(old)
        new = NormalizeAnnotations().visit(new)
        old = RenameIdentifiers(renames).visit(old)
        if ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False):
            print(f"{label}  IDENTICAL")
        else:
            print(f"{label}  DIFFERENT")
            for line in first_diff(old, new):
                print(line)
            status = 1

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
