"""The execution half of #946: run the real writers and compare the two numbers.

``tests/contracts/test_crossval_exit_code_evidence.py`` pins the mechanism on a
crossval-shaped fixture and pins, statically, that every writer routes through
it. The source contract that shipped in PR #999 -- every ``sys.exit`` takes the
Name ``_rc`` -- is a correct invariant and stays, but it never executes the
script, so a late exit that changes the process status by another route (an
uncaught exception, a ``sys.exit`` inside something the tail calls) passed it.
That is what reopened the issue.

The two cases #946 was written about, cv01 and cv02, were removed on
2026-09-21, and the subprocess replays that drove them through their
``--replay`` path went with them. What executes here is what does not depend on
those two: the write path against every committed record, the refusal that
keeps the forcing knob out of the committed evidence tree, and the two tail
endings (``KeyboardInterrupt``, a ``sys.excepthook`` displaced after ours)
driven on fixtures in ``tmp_path``.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CROSSVAL_DIR = REPO_ROOT / "validation" / "crossval"
HELPER = CROSSVAL_DIR / "_exit_evidence.py"
SELFTEST_ENV = "RFX_EXIT_EVIDENCE_SELFTEST"


#: Every case under validation/crossval/ that puts an exit code into a record.
#: Kept here as well as in the contract test on purpose: this file is the one
#: that EXECUTES, and a writer that appears without a guard has to fail a test
#: somebody runs, not only a test somebody reads.
DECLARED_WRITERS = {
    # cv22 keeps a second out-dir for its diagnostic arms; both hold records
    # this case wrote, and both are therefore its own to amend.
    "22_dispersive_slab_fresnel.py": ("_22_dispersive_results",
                                      "_22_dispersive_diag"),
    "23_lossy_slab_fresnel.py": ("_23_lossy_results",),
    "24_nu_rect_cavity_pozar.py": ("_24_nu_cavity_results",),
    "26_oblique_slab_fresnel.py": ("_26_oblique_results",),
}


# --------------------------------------------------------------------------
# The contract, executed
# --------------------------------------------------------------------------


def _committed_records() -> "list[Path]":
    """Every committed record under a declared writer's out-dir that carries
    an ``exit_code``."""
    out = []
    for dirs in DECLARED_WRITERS.values():
        for name in dirs:
            for record in sorted((CROSSVAL_DIR / name).glob("**/*.json")):
                try:
                    doc = json.loads(record.read_text())
                except (ValueError, UnicodeDecodeError):
                    continue
                verdict = doc.get("verdict") if isinstance(doc, dict) else None
                if isinstance(verdict, dict) and "exit_code" in verdict:
                    out.append(record)
    return out


def test_the_new_writer_reproduces_every_committed_record_byte_for_byte(
        tmp_path: Path) -> None:
    """The migration must not have changed what a re-run produces.

    Most of these writers had no defect to fix; they were migrated for
    uniformity, and the price of that would be real if it moved a byte -- a
    manifest entry that claims a re-run reproduces its record bit-identically
    (#937) depends on it, and cv24's six records carry ``exit_code``/``summary``
    ahead of ``arms`` rather than after.

    Re-running the physics to check that is not available here (Meep, FDTD, an
    8-hour lane). What IS checkable, and is the only thing the migration
    touched, is the write path: given the same document and the same declared
    code, does ``write_record`` emit the same bytes the old ``json.dump`` did?
    So each committed record is taken apart -- its ``exit_code``/``summary``
    removed from the verdict block -- and put back together through the real
    helper, and the result is compared with the file on disk. Any difference
    in key order, indent, separators or escaping fails here.
    """
    helper = _load_helper()
    records = _committed_records()
    assert len(records) >= 40, (
        f"only {len(records)} committed records found; the glob is wrong")

    mismatched = []
    try:
        for index, record in enumerate(records):
            doc = json.loads(record.read_text())
            verdict = doc["verdict"]
            code = verdict["exit_code"]
            summary = verdict.get("summary")
            reserved_first = list(verdict)[:2] == ["exit_code", "summary"]
            rest = {k: v for k, v in verdict.items()
                    if k not in ("exit_code", "summary")}
            doc["verdict"] = (helper.reserve_verdict(**rest) if reserved_first
                              else rest)
            # into tmp_path, never beside the original: this test must not put
            # a file into the evidence tree even for a moment (#967, PR #977).
            produced = tmp_path / f"{index:03d}_{record.name}"
            helper.write_record(str(produced), doc, exit_code=code,
                                summary=summary, arm=False)
            if produced.read_bytes() != record.read_bytes():
                mismatched.append(record.relative_to(REPO_ROOT).as_posix())
    finally:
        helper._reset_for_tests()

    assert mismatched == [], (
        "the migrated writer does not reproduce these committed records: %s"
        % mismatched)


# --------------------------------------------------------------------------
# The knob cannot touch committed evidence
# --------------------------------------------------------------------------

def _load_helper():
    sys.path.insert(0, str(CROSSVAL_DIR))
    try:
        import _exit_evidence
    finally:
        sys.path.remove(str(CROSSVAL_DIR))
    return _exit_evidence


FORCED_WRITE_FIXTURE = '''\
import sys

sys.path.insert(0, {crossval_dir!r})
import _exit_evidence

doc = {{"measured": {{"x": 1}}, "verdict": {{"judge_passed": True}}}}
rc = _exit_evidence.write_record({record!r}, doc, exit_code=0)
print("record written, declared exit", rc)
sys.exit(rc)
'''

INTERRUPTED_FIXTURE = '''\
import sys

sys.path.insert(0, {crossval_dir!r})
import _exit_evidence

_exit_evidence.write_record({record!r}, {{"verdict": {{}}}}, exit_code=0)
raise KeyboardInterrupt
'''

DISPLACED_HOOK_FIXTURE = '''\
import sys

sys.path.insert(0, {crossval_dir!r})
import _exit_evidence

_exit_evidence.write_record({record!r}, {{"verdict": {{}}}}, exit_code=0)
sys.excepthook = lambda *a: None          # installed AFTER, does NOT chain
raise RuntimeError("the tail blew up")
'''


def test_the_forced_exit_knob_refuses_before_it_writes_anything(
        tmp_path: Path) -> None:
    """#967 / PR #977: no test may rewrite a retained crossval record.

    The forcing knob is the one new way to end a case's run abnormally, and so
    the one new way a test could leave a committed record amended. It refuses
    by path -- and the refusal has to come BEFORE the write, not after. The
    first cut checked after, so the refusal arrived as an uncaught exception
    and the finalizer amended the very record the guard existed to protect: a
    real committed record went from `exit_code 0` to `1` with a reconciliation
    block attached (review round 1, P2-2).

    Driven end to end in a subprocess, because the defect lived in the ORDER
    of two statements inside ``write_record`` and calling the guard on its own
    could not see it.

    The write is aimed at a path that does not exist, in the same committed
    directory as a retained record rather than at the record itself. The guard
    is by directory, so the probe proves the same thing -- and a regression then
    leaves one stray file that this test deletes, instead of a clobbered
    retained record it would have to restore. Measured, not hypothetical: the
    first version of this test aimed at a committed record, and verifying it by
    reinstating the defect overwrote that record with the fixture's two-key
    document.
    """
    results = CROSSVAL_DIR / "_22_dispersive_results"
    committed = results / "rfx.json"
    committed_before = committed.read_bytes()
    probe = results / "_exit_evidence_selftest_guard_probe.json"
    assert not probe.exists(), f"stale probe left behind: {probe}"
    fixture = tmp_path / "writer.py"
    fixture.write_text(FORCED_WRITE_FIXTURE.format(
        crossval_dir=str(CROSSVAL_DIR), record=str(probe)))
    env = dict(os.environ)
    env[SELFTEST_ENV] = f"exit:3|{probe}"

    try:
        proc = subprocess.run([sys.executable, str(fixture)],
                              cwd=str(tmp_path), env=env, capture_output=True,
                              text=True, timeout=120)
        written = probe.exists()
    finally:
        probe.unlink(missing_ok=True)

    assert proc.returncode not in (0, 3)
    assert "refuses" in proc.stderr and "#967" in proc.stderr
    # Nothing written, so nothing armed, so nothing amended.
    assert not written, (
        "the guard fired, but only AFTER write_record had already put a file "
        "into the committed evidence tree -- which is the whole defect")
    assert "record written" not in proc.stdout
    assert "EXIT-CODE RECONCILED" not in proc.stderr
    assert committed.read_bytes() == committed_before



def test_a_keyboard_interrupt_is_recorded_as_the_status_a_parent_sees(
        tmp_path: Path) -> None:
    """Ctrl-C is 130, not 1.

    CPython does not exit 1 on ``KeyboardInterrupt``: it restores the default
    SIGINT handler and re-raises the signal at itself, so the parent sees
    128+SIGINT. Recording 1 -- what a generic "uncaught exception" branch
    gives -- would put a status in the record that no parent ever saw
    (review round 1, P3-1).
    """
    record = tmp_path / "crossval.json"
    script = tmp_path / "interrupted.py"
    script.write_text(INTERRUPTED_FIXTURE.format(
        crossval_dir=str(CROSSVAL_DIR), record=str(record)))

    proc = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=120)
    doc = json.loads(record.read_text())

    assert proc.returncode in (130, -2), proc.stderr[-2000:]
    assert doc["verdict"]["exit_code"] == 130
    assert doc["verdict"]["exit_code_declared"] == 0
    assert doc["verdict"]["exit_code_reconciliation"]["observed_via"] == (
        "KeyboardInterrupt")


def test_a_displaced_wrapper_is_announced_rather_than_assumed_harmless(
        tmp_path: Path) -> None:
    """A hook installed after ours, that does not chain, hides the status.

    ``_install`` chains to whatever was there before, so a wrapper installed
    EARLIER is still observed. One installed LATER is observed only if it
    chains -- and if it does not, this module sees nothing and the record
    quietly keeps its declared code, which is the pre-#946 state with no
    signal. So there is a signal (review round 1, P3-3). It cannot tell a
    chaining replacement from a swallowing one, and names what it checked
    rather than claiming harm.
    """
    record = tmp_path / "crossval.json"
    script = tmp_path / "displaced.py"
    script.write_text(DISPLACED_HOOK_FIXTURE.format(
        crossval_dir=str(CROSSVAL_DIR), record=str(record)))

    proc = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=120)

    assert "EXIT-CODE OBSERVATION MAY BE INCOMPLETE" in proc.stderr
    assert "sys.excepthook" in proc.stderr
    # And the honest consequence the warning exists for: the status was not
    # observed, so the record keeps what it declared while the process
    # returned 1. The warning is the only thing standing between that and
    # silence.
    assert proc.returncode == 1
    assert json.loads(record.read_text())["verdict"]["exit_code"] == 0



# --------------------------------------------------------------------------
# No seventh writer without a guard
# --------------------------------------------------------------------------

def _calls_write_record(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "write_record":
            return True
        if isinstance(func, ast.Name) and func.id == "write_record":
            return True
    return False


@pytest.mark.parametrize("script", sorted(DECLARED_WRITERS))
def test_every_declared_writer_routes_through_the_helper(script: str) -> None:
    """One road in, per case, named so a failure says which case left it."""
    path = CROSSVAL_DIR / script
    assert path.is_file(), f"declared writer {script} no longer exists"
    assert _calls_write_record(path), (
        f"{script} persists an exit code without _exit_evidence.write_record, "
        "so nothing will amend that code if the process ends differently "
        "(#946)")


def test_no_committed_record_carries_an_exit_code_from_an_undeclared_case() -> None:
    """The evidence-side half: a seventh writer that already produced output.

    The source-side scan (tests/contracts/test_crossval_exit_code_evidence.py)
    catches a new writer when its code is read. This one catches it when its
    RECORDS are read, which is the side a reviewer actually receives, and it
    does not depend on how the new script spells its write.
    """
    declared_dirs = {d for dirs in DECLARED_WRITERS.values() for d in dirs}
    offenders = []
    for record in sorted(CROSSVAL_DIR.glob("_*/**/*.json")):
        try:
            doc = json.loads(record.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        verdict = doc.get("verdict") if isinstance(doc, dict) else None
        if not isinstance(verdict, dict) or "exit_code" not in verdict:
            continue
        top = record.relative_to(CROSSVAL_DIR).parts[0]
        if top not in declared_dirs:
            offenders.append(record.relative_to(REPO_ROOT).as_posix())
    assert offenders == [], (
        "committed records carry an exit_code from a case that is not a "
        "declared #946 writer -- migrate it to _exit_evidence.write_record "
        "and add it here and to MIGRATED_WRITERS: %s" % offenders)


