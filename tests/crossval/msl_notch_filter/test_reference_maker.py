"""The reference maker stays runnable, and honest, on a machine with no openEMS.

``reference/make_openems_reference.py`` is the script that produces this case's
openEMS record from openEMS's own MSL notch filter tutorial.  It is never run by
CI -- the solver runs by hand, on the cluster.  What CI can afford to check, and
what these tests check, is that the script still imports, still agrees with
itself about what it changes, still prints the plan it would build, and keeps
no second copy of the shared sanity helpers -- all without openEMS.

None of these steps an FDTD, reads a reference record, or asserts anything about
physics.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MAKER = HERE / "reference" / "make_openems_reference.py"
REPO_ROOT = HERE.parents[2]

# The shared tutorial gate.  The sanity helpers and the verbatim Stage A builder
# live there, one copy for every case that runs openEMS's own MSL notch filter
# tutorial as its reproduce gate.
SHARED_GATE = REPO_ROOT / "tests" / "crossval" / "_openems_tutorial_gate.py"

# The helpers the shared gate holds.  They were copied byte for byte from the MSL
# thru-line phase case's script, and a test here compared the two until that
# case was removed (2026-09-23); the copies stay, and the shared gate is now the
# only one.  Order is the order they appear in.
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


def test_the_maker_keeps_no_second_copy_of_a_shared_helper():
    """One copy, in the shared module -- not two that can drift.

    A helper re-defined in this maker would be a fork of the gate the other
    cases run, and nothing would compare the two.  A bare alias
    (``name = _gate.name``) is allowed; a definition is not.
    """
    tree = ast.parse(MAKER.read_text())
    offenders = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in COPIED_NAMES:
            offenders.append(f"{node.name} (redefined at line {node.lineno})")
        elif (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in COPIED_NAMES):
            v = node.value
            is_alias = (isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name)
                        and v.value.id == "_gate" and v.attr == node.targets[0].id)
            if not is_alias:
                offenders.append(f"{node.targets[0].id} (own value at line {node.lineno})")
    assert not offenders, (
        f"{MAKER.name} carries its own copy of {offenders} -- the one copy "
        f"lives in {SHARED_GATE.name}, and a second one can drift from it."
    )
