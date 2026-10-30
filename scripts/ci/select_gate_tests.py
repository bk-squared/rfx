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

try:
    from scripts.ci.source_scan import scan_source
except ModuleNotFoundError:
    from source_scan import scan_source

MAX_FILE_SECONDS = 400
CENTRAL_PATHS = (
    "rfx/api/_execute.py",
    "rfx/simulation.py",
    "rfx/core/",
    "rfx/runners/",
    "rfx/nonuniform.py",
    "rfx/stepping/",
    "rfx/measurement/",
    "rfx/model/",
)
CENTRAL_TEST_DIRS = ("tests/unit/runners/", "tests/unit/autodiff/")
MODEL_PATHS = (
    "rfx/geometry/", "rfx/model/", "rfx/materials/", "rfx/boundaries/",
    "rfx/mesh_edges.py", "rfx/grid.py", "rfx/_grid_metric.py",
    "rfx/realized_geometry.py", "rfx/realized_placement.py", "rfx/_realized.py",
)
CONTRACT_DIR = "tests/contracts/"
PACKAGE_TEST_ROOT = "tests/unit/"
# Locks and closed-form oracles pin numbers a user receives; any rfx/ change runs them.
ALWAYS_ON_TEST_DIRS = ("tests/oracle/", "tests/locks/")


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


def test_module_name(path: str) -> str | None:
    name = PurePosixPath(path)
    if not path.startswith("tests/") or name.suffix != ".py" or name.name == "conftest.py":
        return None
    parts = list(name.with_suffix("").parts)
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


def model_building_tests(test_sources: Mapping[str, str]) -> set[str]:
    modules = {name: path for path in test_sources
               if (name := test_module_name(path)) is not None}
    importers = defaultdict(set)
    building = set()
    for name, path in modules.items():
        try:
            tree = ast.parse(test_sources[path], filename=path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if ((isinstance(node, ast.Name) and node.id == "Simulation")
                    or (isinstance(node, ast.Attribute) and node.attr == "Simulation")
                    # An aliased import (`Simulation as _Sim`) leaves no Name node.
                    or (isinstance(node, ast.ImportFrom)
                        and any(alias.name == "Simulation" for alias in node.names))):
                building.add(name)
            imported = set()
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module)
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)
            for dependency in imported & modules.keys():
                importers[dependency].add(name)
    pending = list(building)
    while pending:
        for name in importers[pending.pop()] - building:
            building.add(name)
            pending.append(name)
    return {modules[name] for name in building if is_test_file(modules[name])
            and not modules[name].startswith(CONTRACT_DIR)}


@dataclass(frozen=True)
class Selection:
    files: tuple[str, ...]
    excluded: Mapping[str, float]
    partly_added: Mapping[str, float]
    selected_seconds: float
    not_run_count: int
    not_run_seconds: float
    central_skipped_count: int | None
    central_skipped_seconds: float
    package_directory_files: int
    package_directory_seconds: float
    always_on_files: int
    always_on_seconds: float
    test_module_files: int
    test_module_seconds: float
    model_changed: bool
    model_building_files: int
    model_building_seconds: float
    source_scan_count: int
    source_scan_files: int
    source_scan_seconds: float
    source_scan_run_count: int
    source_scan_run_seconds: float
    source_scan_deselected_count: int
    source_scan_deselected_seconds: float

    def summary(self) -> str:
        lines = [
            f"selected-tests: {len({item.split('::', 1)[0] for item in self.files})} files, "
            f"recorded {self.selected_seconds:.6f} s",
            f"package directory tests added: {self.package_directory_files} files, "
            f"recorded {self.package_directory_seconds:.6f} s",
            f"oracle and lock tests added: {self.always_on_files} files, "
            f"recorded {self.always_on_seconds:.6f} s",
            f"test-module importers added: {self.test_module_files} files, "
            f"recorded {self.test_module_seconds:.6f} s",
            *([f"model-building tests added: {self.model_building_files} files, "
               f"recorded {self.model_building_seconds:.6f} s"] if self.model_changed else []),
            f"source-scan tests added: {self.source_scan_count} tests in "
            f"{self.source_scan_files} files, recorded {self.source_scan_seconds:.6f} s",
            f"source-scan tests that run: {self.source_scan_run_count} tests, "
            f"recorded {self.source_scan_run_seconds:.6f} s",
            "source-scan tests deselected by stage-2 markers: "
            f"{self.source_scan_deselected_count} tests, "
            f"recorded {self.source_scan_deselected_seconds:.6f} s",
            f"not run (돌리지 않은 시험): {self.not_run_count} recorded tests, "
            f"{self.not_run_seconds:.6f} s",
            "Counts use .test_durations nodeids; contracts run in stage 1 are excluded. "
            "Markers are read from AST; no pytest collection or test imports are performed.",
            f"excluded by >{MAX_FILE_SECONDS} s rule: {len(self.excluded)} files",
        ]
        if self.central_skipped_count is not None:
            lines.append(
                "central paths changed; full runners/autodiff suite not run (--central-paths disabled): "
                f"{self.central_skipped_count} files, recorded {self.central_skipped_seconds:.6f} s"
            )
        lines.extend(f"  {path}: {seconds:.6f} s" for path, seconds in self.excluded.items())
        lines.append(f"partly added by node id: {len(self.partly_added)} files")
        lines.extend(f"  {path}: whole-file recorded {seconds:.6f} s"
                     for path, seconds in self.partly_added.items())
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
    test_modules = {name for path in changed if (name := test_module_name(path)) is not None}
    test_importers = set()
    candidates = set(edited)
    invalid = set()
    for path in tests:
        if path.startswith(CONTRACT_DIR):
            continue
        try:
            imports_changed = directly_imports(test_sources[path], modules, path)
            if test_modules and directly_imports(test_sources[path], test_modules, path):
                test_importers.add(path)
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
    package_test_dirs = tuple(sorted({
        f"{PACKAGE_TEST_ROOT}{parts[1]}/"
        for path in changed
        if path.startswith("rfx/") and len(parts := PurePosixPath(path).parts) > 2
    }))
    package_added = {path for path in tests if path.startswith(package_test_dirs)} - candidates
    candidates.update(package_added)
    source_changed = any(path.startswith("rfx/") for path in changed)
    always_on_added = {
        path for path in tests if source_changed and path.startswith(ALWAYS_ON_TEST_DIRS)
    } - candidates
    candidates.update(always_on_added)
    test_module_added = test_importers - candidates
    candidates.update(test_module_added)
    model_changed = source_changed and any(
        path.startswith(prefix) if prefix.endswith("/") else path == prefix
        for path in changed for prefix in MODEL_PATHS
    )
    model_added = model_building_tests(test_sources) - candidates if model_changed else set()
    candidates.update(model_added)
    candidates = {path for path in candidates if not path.startswith(CONTRACT_DIR)}
    totals: dict[str, float] = defaultdict(float)
    for nodeid, seconds in durations.items():
        totals[nodeid.split("::", 1)[0]] += seconds
    excluded = {
        path: totals[path] for path in sorted(candidates)
        if totals[path] > MAX_FILE_SECONDS and path not in edited and path not in invalid
    }
    selected = candidates - excluded.keys()
    package_added &= selected
    always_on_added &= selected
    test_module_added &= selected
    model_added &= selected
    source_scan_nodes = set()
    source_scan_files = set()
    source_scan_count = 0
    marked_run = set()
    marked_deselected = set()
    summary_tests = tests if source_changed else selected
    for path in sorted(summary_tests - invalid):
        if path.startswith(CONTRACT_DIR):
            continue
        scan = scan_source(test_sources[path], path)
        marked = [test for test in scan.tests if test.marked]
        for test in marked:
            target = marked_deselected if test.deselected_markers else marked_run
            target.add(f"{path}::{test.name}")
        if path in selected or not source_changed:
            continue
        if scan.module_marked:
            source_scan_nodes.add(path)
        else:
            source_scan_nodes.update(f"{path}::{test.name}" for test in marked)
        if marked or scan.module_marked:
            source_scan_files.add(path)
            source_scan_count += len(marked)
    partly_added = {path: seconds for path, seconds in excluded.items()
                    if path in source_scan_files and path not in source_scan_nodes}
    excluded = {path: seconds for path, seconds in excluded.items()
                if path not in source_scan_files}

    def source_scan_selected(nodeid):
        return (nodeid.split("::", 1)[0] in source_scan_nodes
                or nodeid.split("[", 1)[0] in source_scan_nodes)

    source_scan_seconds = sum(seconds for nodeid, seconds in durations.items()
                              if source_scan_selected(nodeid))
    not_run = [
        seconds for nodeid, seconds in durations.items()
        if nodeid.split("::", 1)[0] not in selected and not nodeid.startswith(CONTRACT_DIR)
        and not source_scan_selected(nodeid)
    ]
    return Selection(
        tuple(sorted(selected | source_scan_nodes)), excluded, partly_added,
        sum(totals[path] for path in sorted(selected)) + source_scan_seconds,
        len(not_run), sum(not_run),
        len(central_tests) if central_changed and not central_paths else None,
        sum(totals[path] for path in sorted(central_tests)),
        len(package_added), sum(totals[path] for path in sorted(package_added)),
        len(always_on_added), sum(totals[path] for path in sorted(always_on_added)),
        len(test_module_added), sum(totals[path] for path in sorted(test_module_added)),
        model_changed, len(model_added), sum(totals[path] for path in sorted(model_added)),
        source_scan_count, len(source_scan_files), source_scan_seconds,
        len(marked_run), sum(seconds for nodeid, seconds in durations.items()
                             if nodeid.split("[", 1)[0] in marked_run),
        len(marked_deselected), sum(seconds for nodeid, seconds in durations.items()
                                    if nodeid.split("[", 1)[0] in marked_deselected),
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
    for path in sorted((root / "tests").rglob("*.py")):
        if path.name == "conftest.py":
            continue
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
