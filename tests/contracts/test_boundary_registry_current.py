"""The boundary registry follows the source, with named unresolved exceptions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.contracts import boundary_registry


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.source_scan


# This list may only shrink: a fix PR removes its entry.
EXCEPTIONS = (
    (
        "rfx/model/realization_reference_area.py",
        "drawn_open_fractions",
        "rfx-known-issues.md: The realization table's face reference does not fold a box drawn past a periodic seam",
        "the reference reads only the domain nodes, so a box past a periodic seam is not folded",
    ),
)


def registry_problems(registry, rows, exceptions):
    """Compare registered sites and require listed exceptions to stay unregistered."""
    excepted = {(file, function) for file, function, ledger, reason in exceptions}
    scanned = {(row["file"], row["function"]) for row in rows}
    registered = {(row["file"], row["function"]) for row in registry["sites"]}
    problems = []
    try:
        boundary_registry.compare_registry(
            registry["sites"],
            [row for row in rows if (row["file"], row["function"]) not in excepted],
            registry["counted_functions"],
        )
    except AssertionError as exc:
        problems.append(str(exc))
    for file, function, ledger, reason in exceptions:
        if (file, function) not in scanned:
            problems.append(f"{file} :: {function}: renamed or removed while listed: remove its entry")
        if (file, function) in registered:
            problems.append(f"{file} :: {function}: registered while listed: remove the row or the entry")
    return problems


def test_boundary_registry_matches_the_source():
    registry = json.loads((ROOT / "tests/contracts/boundary_registry.json").read_text())
    rows = boundary_registry.scan_tree(ROOT)
    problems = registry_problems(registry, rows, EXCEPTIONS)
    assert problems == [], "\n".join(problems)


def test_stale_registry_row_is_detected():
    registry = json.loads((ROOT / "tests/contracts/boundary_registry.json").read_text())
    registry["sites"].append(dict(file="rfx/stale_boundary.py", function="stale_unit", kind="difference"))
    problems = registry_problems(registry, boundary_registry.scan_tree(ROOT), EXCEPTIONS)
    assert len(problems) == 1
    assert "removed sites: rfx/stale_boundary.py :: stale_unit [difference]" in problems[0]


def test_unregistered_scanned_unit_is_detected():
    registry = json.loads((ROOT / "tests/contracts/boundary_registry.json").read_text())
    rows = boundary_registry.scan_tree(ROOT)
    rows += boundary_registry.scan_source(
        "def new_unit(a):\n    return a[1:] - a[:-1]\n", "rfx/new_boundary.py",
    )
    problems = registry_problems(registry, rows, EXCEPTIONS)
    assert len(problems) == 1
    assert "unlisted sites: rfx/new_boundary.py :: new_unit [difference]" in problems[0]


def test_removed_exception_is_detected():
    registry = {"sites": [], "counted_functions": []}
    exception = ("rfx/missing_boundary.py", "missing_unit", "synthetic ledger", "synthetic reason")
    assert registry_problems(registry, [], (exception,)) == [
        "rfx/missing_boundary.py :: missing_unit: renamed or removed while listed: remove its entry"
    ]


def test_registered_exception_is_detected():
    rows = boundary_registry.scan_source(
        "def excepted_unit(a):\n    return a[1:] - a[:-1]\n", "rfx/excepted_boundary.py",
    )
    registry = {"sites": boundary_registry.registry_rows(rows), "counted_functions": []}
    exception = ("rfx/excepted_boundary.py", "excepted_unit", "synthetic ledger", "synthetic reason")
    problems = registry_problems(registry, rows, (exception,))
    assert "rfx/excepted_boundary.py :: excepted_unit: registered while listed: remove the row or the entry" in problems
