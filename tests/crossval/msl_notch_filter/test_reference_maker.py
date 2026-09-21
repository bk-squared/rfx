"""The reference maker stays runnable on a machine with no openEMS.

``reference/make_openems_reference.py`` is the script that produces this case's
openEMS record from openEMS's own MSL notch filter tutorial.  It is never run by
CI -- the solver runs by hand, on the cluster.  What CI can afford to check, and
what this test checks, is that the script still imports, still agrees with itself
about the one declared geometry change, and still prints the plan it would build,
all without openEMS present.  Both subprocesses below take about a second.

Neither of them steps an FDTD, reads a reference record, or asserts anything
about physics.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MAKER = Path(__file__).resolve().parent / "reference" / "make_openems_reference.py"


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


def test_dry_run_names_the_delta_and_the_tutorial():
    result = _run("--dry-run", "--stage", "both")
    assert result.returncode == 0, (
        f"--dry-run exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    out = result.stdout
    assert "MSL_NotchFilter.py" in out, "the dry run does not name the tutorial"
    assert "DELTA 1 (geometry): NONE" in out, (
        "the dry run does not say the geometry is unchanged from the tutorial"
    )
    assert "resolution factor 0.5" in out, (
        "the dry run does not name the one delta, the 0.5 mesh rung"
    )
    assert "50000" in out, "the dry run does not print the tutorial's own MSL_length"
    for stage in ("stage_a", "stage_b_coarse", "stage_b_fine"):
        assert stage in out, f"the dry run does not plan {stage}"
