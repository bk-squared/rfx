"""Select tests with literal file or directory readers of changed not-code paths.

Literal path joins count as the paths they build, without rereading their parts.
Standalone one-component constants match only an exact changed path; other
constants and joined directories can match ancestors, even after deletion.
Absolute paths, parent traversal and code paths are excluded. Imported helpers
propagate readers transitively to their test importers outside tests/contracts.
The literal text of an f-string is read as a constant, so its leading directory
can match. Paths built by helper functions, format, globs or variables, and the
variable parts of f-strings, are invisible; such tests must live in
tests/contracts to run on not-code changes.
"""

from __future__ import annotations

import argparse
import ast
import sys
from functools import lru_cache
from pathlib import Path
from collections.abc import Iterable, Sequence

from changed_paths import is_code_path, normalize


def _candidates(tree: ast.AST) -> set[tuple[str, bool]]:
    """Read trailing literal runs in joins once, excluding body docstrings."""
    excluded = set()
    joined = []

    def string(node):
        return isinstance(node, ast.Constant) and isinstance(node.value, str)

    def parts(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return parts(node.left) + [node.right]
        if isinstance(node, ast.Call) and ast.unparse(node.func) in (
            "Path", "pathlib.Path", "PurePosixPath"
        ) and len(node.args) == 1 and not node.keywords and string(node.args[0]):
            return [node.args[0]]
        return [node]

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and string(node.body[0].value):
                excluded.add(id(node.body[0].value))
        run = []
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            run = parts(node)
        elif isinstance(node, ast.Call) and ast.unparse(node.func) == "os.path.join":
            run = node.args
        trailing = []
        for part in reversed(run):
            if not string(part):
                break
            trailing.insert(0, part)
        if trailing:
            joined.append((node, trailing))
    consumed = set()
    candidates = set()
    for node, run in joined:
        if id(node) in consumed:
            continue
        consumed.update(id(part) for part in run)
        consumed.update(id(child) for child in ast.walk(node) if isinstance(child, ast.BinOp))
        candidates.add(("/".join(part.value for part in run), True))
    for node in ast.walk(tree):
        if string(node) and id(node) not in excluded and id(node) not in consumed:
            candidates.add((node.value, "/" in node.value.replace("\\", "/")))
    return candidates


def _literal_paths(candidates, root, changed):
    paths = set()
    for value, directory in candidates:
        if Path(value.strip().replace("\\", "/")).is_absolute():
            continue
        path = normalize(value).rstrip("/")
        if not path or path in (".", "..") or ".." in Path(path).parts or is_code_path(path):
            continue
        try:
            if path in changed or directory and (
                any(target.startswith(path + "/") for target in changed) or (root / path).exists()
            ):
                paths.add((path, directory))
        except (OSError, ValueError):
            continue
    return paths


def literal_paths(source: str, root: Path, changed: Iterable[str] = ()) -> set[str]:
    """Find joined or standalone not-code readers outside body docstrings."""
    return {path for path, _ in _literal_paths(_candidates(ast.parse(source)), root, set(changed))}


@lru_cache(maxsize=None)
def _scan(root: Path):
    """Cache candidates and module imports once per repository root."""
    entries = {}
    modules = {}
    for file in sorted((root / "tests").rglob("*.py")):
        relative = file.relative_to(root).as_posix()
        if relative.startswith("tests/contracts/"):
            continue
        tree = ast.parse(file.read_text(encoding="utf-8"))
        module = relative[:-3].replace("/", ".")
        modules.setdefault(module.removesuffix(".__init__"), set()).add(relative)
        modules.setdefault(file.stem, set()).add(relative)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = ".".join(module.split(".")[:-node.level] + ([base] if base else []))
                imports.add(base)
                imports.update((base + "." if base else "") + alias.name for alias in node.names)
        entries[relative] = (_candidates(tree), imports, file.name.startswith("test_"))
    return {
        file: (candidates, {target for name in imports for target in modules.get(name, ())}, test)
        for file, (candidates, imports, test) in entries.items()
    }


def select_tests(paths: Iterable[str], root: Path) -> list[str]:
    """Return tests reading changed paths directly or through imported helpers."""
    changed = {normalize(path) for path in paths if normalize(path)}
    entries = _scan(root.resolve())
    matched = {
        file for file, (candidates, imports, test) in entries.items()
        if any(path == reader or directory and path.startswith(reader + "/")
               for reader, directory in _literal_paths(candidates, root, changed) for path in changed)
    }
    while True:
        propagated = {file for file, (_, imports, _) in entries.items()
                      if imports & {name for name in matched if not entries[name][2]}}
        if propagated <= matched:
            break
        matched.update(propagated)
    return sorted(file for file in matched if entries[file][2])


def main(argv: Sequence[str] | None = None) -> int:
    """Read arguments or a NUL-separated diff and print one selected file per line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--stdin", action="store_true")
    parser.add_argument("--null", "-z", action="store_true")
    args = parser.parse_args(argv)
    paths = list(args.paths)
    if args.stdin:
        paths.extend(sys.stdin.read().split("\0") if args.null else sys.stdin.read().splitlines())
    selected = select_tests(paths, Path(__file__).resolve().parents[2])
    if selected:
        print("\n".join(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
