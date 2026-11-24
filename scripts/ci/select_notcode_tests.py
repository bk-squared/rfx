"""Select tests with literal file or directory readers of changed not-code paths.

String constants outside docstrings count when they name existing repository
paths, equal a changed path or name an ancestor of one, even after deletion or
rename. Absolute paths, parent traversal and code paths are excluded.
Paths assembled at run time from parts, helpers or globs are invisible; tests
using those readers must live in tests/contracts to run on not-code changes.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from collections.abc import Iterable, Sequence

from changed_paths import is_code_path, normalize


def literal_paths(source: str, root: Path, changed: Iterable[str] = ()) -> set[str]:
    """Find existing or changed not-code readers outside module and body docstrings."""
    changed = set(changed)
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    docstrings.add(id(value))
    paths = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in docstrings:
            continue
        if Path(node.value.strip().replace("\\", "/")).is_absolute():
            continue
        path = normalize(node.value).rstrip("/")
        if not path or path in (".", "..") or ".." in Path(path).parts or is_code_path(path):
            continue
        try:
            if any(target == path or target.startswith(path + "/") for target in changed) or (root / path).exists():
                paths.add(path)
        except (OSError, ValueError):
            continue
    return paths


def select_tests(paths: Iterable[str], root: Path) -> list[str]:
    """Return sorted test files with exact file or ancestor directory readers."""
    changed = {normalize(path) for path in paths if normalize(path)}
    selected = []
    for test in sorted((root / "tests").rglob("test_*.py")):
        relative = test.relative_to(root).as_posix()
        if relative.startswith("tests/contracts/"):
            continue
        readers = literal_paths(test.read_text(encoding="utf-8"), root, changed)
        if any(
            path == reader or path.startswith(reader + "/")
            for reader in readers for path in changed
        ):
            selected.append(relative)
    return selected


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
