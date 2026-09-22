"""The reference maker stays runnable, and honest, on a machine with no openEMS.

``reference/make_openems_reference.py`` is the script that produces this case's
openEMS record from openEMS's own MSL notch filter tutorial.  It is never run by
CI -- the solver runs by hand, on the cluster.  What CI can afford to check, and
what these tests check, is that the script still imports, still agrees with
itself about what it changes, still prints the plan it would build, and still
carries the precedent's sanity helpers unmodified -- all without openEMS.

None of these steps an FDTD, reads a reference record, or asserts anything about
physics.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MAKER = HERE / "reference" / "make_openems_reference.py"
REPO_ROOT = HERE.parents[2]

# The script the maker's sanity helpers were copied from.  It belongs to another
# case (the MSL thru-line phase referee), which is why the helpers are COPIED and
# never imported.  When that case is removed this test goes with it -- the copies
# stay, the comparison has nothing left to compare against.
PRECEDENT = REPO_ROOT / "validation" / "crossval" / "20_msl_phase_referee.py"

# Copied byte for byte, checked below.  Order is the order they appear in.
COPIED_NAMES = [
    "_ensure_openems_numpy_compat",
    "_import_openems",
    "_BAD_STDOUT_PATTERNS",
    "_TRUNCATION_STDOUT_PATTERNS",
    "_ALLOWLISTED_UNUSED_PRIMITIVE_PROPERTIES",
    "_scan_stdout_for_bad_patterns",
    "_run_openems_capturing_stdout",
    "_END_CRITERIA_NOT_REACHED_TEXT",
    "_log_indicates_truncation",
    "_check_excitation_and_trace",
    "_non_physical_guard",
    "_passivity_witness",
    "_build_stage_a_notch_tutorial",
]


def _load_maker():
    spec = importlib.util.spec_from_file_location("_msl_notch_reference_maker", MAKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=4)
def _top_level_sources(path: Path) -> dict:
    """Source text of every top-level function, class and simple assignment.

    Lines are sliced directly rather than via ``ast.get_source_segment``, which
    re-splits the whole file for every node -- 2.2 s per lookup on the 3400-line
    precedent.  The result is the same text: a top-level definition starts at
    column 0, and ``lineno`` points at ``def``/``class``, not at a decorator.
    """
    src = path.read_text()
    lines = src.split("\n")

    def seg(node):
        return "\n".join(lines[node.lineno - 1:node.end_lineno])

    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = seg(node)
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = seg(node)
    return out


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(MAKER), *args],
        capture_output=True, text=True, timeout=120,
    )


def test_self_check_passes_without_openems():
    result = _run("--self-check")
    assert result.returncode == 0, (
        f"--self-check exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "SELF-CHECK PASSED" in result.stdout


def test_dry_run_prints_the_plan_and_the_delta_list():
    result = _run("--dry-run", "--stage", "both")
    assert result.returncode == 0, (
        f"--dry-run exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    out = result.stdout
    assert "MSL_NotchFilter.py" in out, "the dry run does not name the tutorial"
    for stage in ("stage_a", "stage_b_coarse", "stage_b_mid", "stage_b_fine"):
        assert stage in out, f"the dry run does not plan {stage}"
    # P2-9: the delta list the module declares is the one the dry run prints.
    for entry in _load_maker().DELTA_LIST:
        assert entry in out, f"the dry run does not print the delta entry {entry[:60]!r}"


def test_delta_list_says_what_changes():
    delta = _load_maker().DELTA_LIST
    assert len(delta) == 3, f"expected three delta entries, got {len(delta)}"

    # 1: no geometry change at all.
    assert "DELTA 1 (geometry): NONE" in delta[0]
    assert "50000 um" in delta[0], "delta 1 does not name the tutorial's own arm length"

    # 2: the three rung factors, and that the substrate's z lines scale with them.
    assert "1.0" in delta[1] and "0.70711" in delta[1] and "0.5" in delta[1], (
        "delta 2 does not name all three rung factors"
    )
    assert "z lines" in delta[1], "delta 2 does not say the substrate z lines scale"
    assert "4, 6 and 8 cells" in delta[1], (
        "delta 2 does not give the substrate cell count per rung"
    )

    # 3: the boundary list, unchanged from the tutorial.
    assert "['PML_8','PML_8','MUR','MUR','PEC','MUR']" in delta[2], (
        "delta 3 does not name the tutorial's boundary list"
    )


@pytest.mark.parametrize("name", COPIED_NAMES)
def test_copied_helper_is_byte_identical_to_the_precedent(name):
    """The sanity helpers are copies, not edits.

    They cannot be imported (the precedent belongs to another case and will be
    removed), so the only thing keeping them honest is this comparison.  It goes
    away with that file; the copies stay.
    """
    if not PRECEDENT.is_file():
        pytest.skip(f"the precedent {PRECEDENT.name} is gone; the copies stand alone now")
    precedent = _top_level_sources(PRECEDENT)
    maker = _top_level_sources(MAKER)
    assert name in precedent, f"{name} is no longer in {PRECEDENT.name}"
    assert name in maker, f"{name} is no longer in {MAKER.name}"
    assert maker[name] == precedent[name], (
        f"{name} has drifted from its copy in {PRECEDENT.name}. The helpers are "
        f"copied byte for byte on purpose: an edit here is a silent fork of a "
        f"gate whose behaviour another case's record depends on."
    )
