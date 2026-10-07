#!/usr/bin/env python3
"""Select a second local gate session without importing any test modules."""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tokenize
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_FILE_SECONDS = 400
CENTRAL_PATHS = (
    "rfx/api/_execute.py",
    "rfx/simulation.py",
    "rfx/core/",
    "rfx/runners/",
    "rfx/nonuniform.py",
    "rfx/measurement/",
    "rfx/model/",
)
CENTRAL_TEST_DIRS = ("tests/unit/runners/", "tests/unit/autodiff/")
CONTRACT_DIR = "tests/contracts/"


def is_test_file(path: str) -> bool:
    name = PurePosixPath(path)
    return path.startswith("tests/") and name.name.startswith("test_") and name.suffix == ".py"


def module_name(path: str) -> str | None:
    if not path.startswith("rfx/") or not path.endswith(".py"):
        return None
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def directly_imports(source: str, modules: set[str], filename: str = "<test>") -> bool:
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Import):
            if any(alias.name in modules for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            # Top-level re-exports only match changes to rfx/__init__.py.
            if node.module in modules or (node.module != "rfx" and any(
                f"{node.module}.{alias.name}" in modules for alias in node.names
            )):
                return True
    return False


@dataclass(frozen=True)
class Selection:
    files: tuple[str, ...]
    excluded: Mapping[str, float]
    selected_seconds: float
    not_run_count: int
    not_run_seconds: float
    central_skipped_count: int | None
    central_skipped_seconds: float

    def summary(self) -> str:
        lines = [
            f"selected-tests: {len(self.files)} files, recorded {self.selected_seconds:.6f} s",
            f"not run (돌리지 않은 시험): {self.not_run_count} recorded tests, "
            f"{self.not_run_seconds:.6f} s",
            "Counts use .test_durations nodeids; contracts run in stage 1 are excluded. "
            "No pytest collection or marker evaluation is performed.",
            f"excluded by >{MAX_FILE_SECONDS} s rule: {len(self.excluded)} files",
        ]
        if self.central_skipped_count is not None:
            lines.append(
                "central paths changed; full runners/autodiff suite not run (--central-paths disabled): "
                f"{self.central_skipped_count} files, recorded {self.central_skipped_seconds:.6f} s"
            )
        lines.extend(f"  {path}: {seconds:.6f} s" for path, seconds in self.excluded.items())
        return "\n".join(lines) + "\n"


def select_tests(
    changed_paths: Iterable[str], test_sources: Mapping[str, str], durations: Mapping[str, float],
    *, central_paths: bool = False,
) -> Selection:
    """Pure selection over repo-relative paths, source text, and recorded nodeids.

    Missing duration records cost zero; deleted tests are absent from test_sources.
    The summary counts historical nodeids, not a fresh marker-filtered collection.
    """
    changed = set(changed_paths)
    tests = {path for path in test_sources if is_test_file(path)}
    edited = changed & tests
    modules = {name for path in changed if (name := module_name(path)) is not None}
    candidates = set(edited)
    invalid = set()
    for path in tests:
        if path.startswith(CONTRACT_DIR):
            continue
        try:
            imports_changed = directly_imports(test_sources[path], modules, path)
        except SyntaxError:
            # Let pytest report the broken file, even without an import match.
            invalid.add(path)
            candidates.add(path)
        else:
            if imports_changed:
                candidates.add(path)
    central_changed = any(
        path.startswith(prefix) if prefix.endswith("/") else path == prefix
        for path in changed for prefix in CENTRAL_PATHS
    )
    central_tests = {path for path in tests if path.startswith(CENTRAL_TEST_DIRS)}
    if central_changed and central_paths:
        candidates.update(central_tests)
    candidates = {path for path in candidates if not path.startswith(CONTRACT_DIR)}
    totals: dict[str, float] = defaultdict(float)
    for nodeid, seconds in durations.items():
        totals[nodeid.split("::", 1)[0]] += seconds
    excluded = {
        path: totals[path] for path in sorted(candidates)
        if totals[path] > MAX_FILE_SECONDS and path not in edited and path not in invalid
    }
    selected = candidates - excluded.keys()
    not_run = [
        seconds for nodeid, seconds in durations.items()
        if nodeid.split("::", 1)[0] not in selected and not nodeid.startswith(CONTRACT_DIR)
    ]
    return Selection(
        tuple(sorted(selected)), excluded, sum(totals[path] for path in sorted(selected)),
        len(not_run), sum(not_run),
        len(central_tests) if central_changed and not central_paths else None,
        sum(totals[path] for path in sorted(central_tests)),
    )


def changed_paths(base: str, head: str | None, root: Path) -> list[str]:
    """Compare merge-base to head, or to the working tree including untracked files."""
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True)

    merge_base = git("merge-base", base, head or "HEAD").strip()
    paths = git("diff", "--name-only", "--no-renames", "-z", merge_base, *([head] if head else []), "--").split("\0")
    if head is None:
        paths += git("ls-files", "--others", "--exclude-standard", "-z").split("\0")
    return [path for path in paths if path]


def read_test_sources(root: Path) -> dict[str, str]:
    sources = {}
    for path in sorted((root / "tests").rglob("test_*.py")):
        with tokenize.open(path) as handle:
            sources[path.relative_to(root).as_posix()] = handle.read()
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", help="omit to include the working tree")
    parser.add_argument("--durations", type=Path, default=Path(".test_durations"))
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--central-paths", action="store_true",
                        help="include runners/autodiff for central path changes")
    args = parser.parse_args()
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    result = select_tests(
        changed_paths(args.base, args.head, root), read_test_sources(root),
        json.loads(args.durations.read_text(encoding="utf-8")),
        central_paths=args.central_paths,
    )
    if args.summary:
        args.summary.write_text(result.summary(), encoding="utf-8")
    else:
        sys.stderr.write(result.summary())
    for path in result.files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
