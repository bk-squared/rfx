"""Issue #946 — cv02's PERSISTED ``exit_code`` must be the process's REAL one.

``validation/crossval/02_ring_resonator.py`` decides its verdict once
(``_rc = _exit_code(...)``), writes ``_rc`` into
``_02_ring_resonator_results/crossval.json`` under ``verdict.exit_code``, and
then returns it from every ``sys.exit``. Today that is correct. The hazard is
structural rather than present: the JSON is written well before the two exit
statements, so ANY exit path added later between the two -- an early return, a
``sys.exit(0)`` in a new branch, an ``atexit`` handler, a swallowed exception
-- would leave a retained artifact claiming a verdict the process did not
return. That artifact is what a scheduled runner's log expires into; a CI lane
or an audit reading ``verdict.exit_code`` would be reading a lie.

So this test does NOT import ``_exit_code`` and compare it with itself. It
RUNS the script and compares the JSON the run wrote against the status the
process actually returned. On this host Meep is absent, so the run takes the
exit-2 (inconclusive) lane; the invariant is lane-independent and the
assertions below are written for whichever code comes back.

Cost: ~26 s wall on a 2026-era CPU host (measured). Marked ``slow`` so it
stays out of the bounded PR gate and runs in the validation lane, which
selects with ``-m "not gpu and not highmem"``.

Sibling shape (NOT covered here, deliberately): ``01_waveguide_bend.py`` has
the identical construction -- its own ``_exit_code`` helper, one ``_rc``,
``verdict.exit_code`` / ``verdict.summary`` in a ``crossval.json`` written
beside the script, one trailing ``sys.exit(_rc)`` -- so the case list below is
one entry wide on purpose and cv01 drops in as a second entry. cv22/23/24
persist ``exit_code`` too but return it out of a ``main()``, which is a
different (and structurally safer) shape.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CROSSVAL_DIR = REPO_ROOT / "validation" / "crossval"

#: (script, extra files it loads from SCRIPT_DIR, artifact path under
#: SCRIPT_DIR). See the module docstring on why cv01 is not here yet.
CASES = [
    pytest.param(
        "02_ring_resonator.py",
        ("comparators/ring_mode_judge.py",),
        "_02_ring_resonator_results/crossval.json",
        id="cv02",
    ),
]

#: The script's own documented convention (its module docstring): 0 = full
#: PASS, 1 = rfx self-check failed, 2 = reference unavailable / inconclusive.
DOCUMENTED_CODES = {0, 1, 2}

TIMEOUT_S = 900


def _stage(tmp_path: Path, script_rel: str, extras) -> Path:
    """Copy the script and its SCRIPT_DIR-relative imports into a scratch dir.

    The script writes its artifact next to ITSELF, and the repo's copy of that
    artifact is the committed live-Meep record (#937). Running from a copy
    leaves the committed record untouched and gives the subprocess a private
    artifact for this test to read back.
    """
    work = tmp_path / "cv"
    work.mkdir(parents=True)
    shutil.copy2(CROSSVAL_DIR / script_rel, work / Path(script_rel).name)
    for extra in extras:
        dst = work / extra
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CROSSVAL_DIR / extra, dst)
    return work


@pytest.mark.slow
@pytest.mark.parametrize("script_rel,extras,artifact_rel", CASES)
def test_persisted_exit_code_equals_the_process_exit_status(
    tmp_path: Path, script_rel: str, extras, artifact_rel: str
) -> None:
    """Run the real script; the JSON it wrote must agree with its own status."""
    work = _stage(tmp_path, script_rel, extras)
    proc = subprocess.run(
        [sys.executable, str(work / Path(script_rel).name)],
        cwd=work, capture_output=True, text=True, timeout=TIMEOUT_S,
    )
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])

    artifact = work / artifact_rel
    assert artifact.is_file(), (
        f"{script_rel} returned {proc.returncode} without writing "
        f"{artifact_rel}; last lines:\n{tail}"
    )
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    verdict = doc["verdict"]

    assert proc.returncode in DOCUMENTED_CODES, (
        f"undocumented exit status {proc.returncode}; last lines:\n{tail}"
    )

    # THE contract (#946): the retained record's code is the process's code.
    assert verdict["exit_code"] == proc.returncode, (
        f"{artifact_rel} records exit_code={verdict['exit_code']} but the "
        f"process returned {proc.returncode} -- an exit path bypassed the "
        f"persisted value. Last lines:\n{tail}"
    )

    # ...and the human-readable half of the record cannot claim a pass while
    # the process failed. Stated as an equivalence so neither direction slips.
    claims_pass = "ALL CHECKS PASSED" in verdict["summary"]
    assert claims_pass == (proc.returncode == 0), (
        f"summary {verdict['summary']!r} vs exit status {proc.returncode}"
    )

    # The exit-2 lane is inconclusive, never green: the record must say so
    # rather than leaving a reader to infer it from the code alone.
    if proc.returncode == 2:
        assert verdict["meep_present"] is False
        assert verdict["judge_passed"] is not True or verdict["exit_code"] == 0

    # Settling witness travels with the record (repo rule: a claims-bearing
    # number is unreadable without it). Presence only -- its VALUE is gated
    # elsewhere, and this lane is inconclusive by construction.
    assert isinstance(doc["measured"]["signal_settling_db"], float)
