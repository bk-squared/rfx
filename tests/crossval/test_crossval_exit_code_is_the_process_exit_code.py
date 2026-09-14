"""The execution half of #946: run the real cases and compare the two numbers.

``tests/contracts/test_crossval_exit_code_evidence.py`` pins the mechanism on a
crossval-shaped fixture and pins, statically, that every writer routes through
it. Neither of those runs cv01 or cv02, and those two are the cases the issue is
about: cv01 writes its record ~90 lines before its ``sys.exit`` with the whole
plotting stage in between, cv02 ~250 lines before its two exits with PART 4's
narrowband visualisation in between. The source contract that shipped in
PR #999 -- every ``sys.exit`` takes the Name ``_rc`` -- is a correct invariant
and stays, but it never executes the script, so a late exit that changes the
process status by another route (an uncaught exception, a ``sys.exit`` inside
something the tail calls) passed it. That is what reopened the issue.

So this file runs them. Each case is driven in a subprocess three ways:

  control      no forcing; the record's code must equal the return code and
               carry NO reconciliation -- the mechanism must be invisible when
               the run agrees with itself.
  exit:3       a divergent ``sys.exit`` fires immediately after the record is
               persisted, the way an added guard would.
  raise        an exception fires there instead, so the process returns 1
               through ``sys.excepthook``.

In every case the assertion is the same one #946 asks for: the number in the
file equals the number the process handed its parent, and when they started out
different the record says so.

Reaching the writer without a solve. cv01 and cv02 are module-body scripts
whose live path runs FDTD (and Meep), which a contract test must not. Both grew
a ``--replay <record.json> --out-dir <dir>`` path for this: it re-judges a
record from the numbers that record retained -- cv01 re-evaluates its three
gates from the stored ``measured`` block, cv02 re-runs ``ring_mode_judge`` on
the stored mode lists -- and re-emits it through the same
``_exit_evidence.write_record`` call the live run uses. Nothing is solved and
nothing is imported from ``rfx``; what is exercised is the decision stage, the
write, and the exit path behind it, which is the whole of what #946 is about.

Replays write into ``tmp_path``. A replay is a re-judge, not a reproduction,
and the committed records under ``validation/crossval/`` are never the target
-- the forced-exit knob refuses outright for any path inside that tree (#967,
PR #977).
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

#: (script, committed record it can replay). Both are the cases #946 names.
REPLAY_CASES = {
    "cv01": (CROSSVAL_DIR / "01_waveguide_bend.py",
             CROSSVAL_DIR / "_01_waveguide_bend_results" / "crossval.json"),
    "cv02": (CROSSVAL_DIR / "02_ring_resonator.py",
             CROSSVAL_DIR / "_02_ring_resonator_results" / "crossval.json"),
}

#: Every case under validation/crossval/ that puts an exit code into a record.
#: Kept here as well as in the contract test on purpose: this file is the one
#: that EXECUTES, and a writer that appears without a guard has to fail a test
#: somebody runs, not only a test somebody reads.
DECLARED_WRITERS = {
    "01_waveguide_bend.py": ("_01_waveguide_bend_results",),
    "02_ring_resonator.py": ("_02_ring_resonator_results",),
    # cv22 keeps a second out-dir for its diagnostic arms; both hold records
    # this case wrote, and both are therefore its own to amend.
    "22_dispersive_slab_fresnel.py": ("_22_dispersive_results",
                                      "_22_dispersive_diag"),
    "23_lossy_slab_fresnel.py": ("_23_lossy_results",),
    "24_nu_rect_cavity_pozar.py": ("_24_nu_cavity_results",),
    "26_oblique_slab_fresnel.py": ("_26_oblique_results",),
}


def _replay(case: str, out_dir: Path, force: "str | None" = None
            ) -> "tuple[int, dict, str]":
    """Run one case's ``--replay`` in a subprocess; return (rc, record, stderr).

    ``force`` is the ``RFX_EXIT_EVIDENCE_SELFTEST`` mode, which fires at the
    first statement after the record is persisted.
    """
    script, source = REPLAY_CASES[case]
    out_dir.mkdir(parents=True, exist_ok=True)
    record = out_dir / "crossval.json"
    env = dict(os.environ)
    env.pop(SELFTEST_ENV, None)
    if force is not None:
        env[SELFTEST_ENV] = f"{force}|{record}"
    proc = subprocess.run(
        [sys.executable, str(script), "--replay", str(source),
         "--out-dir", str(out_dir)],
        cwd=str(out_dir), env=env, capture_output=True, text=True, timeout=600)
    assert record.is_file(), (
        "the record was not persisted at all:\n"
        + proc.stdout[-4000:] + proc.stderr[-4000:])
    return proc.returncode, json.loads(record.read_text()), proc.stderr


# --------------------------------------------------------------------------
# The contract, executed
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(REPLAY_CASES))
def test_an_agreeing_run_leaves_no_reconciliation(case: str,
                                                  tmp_path: Path) -> None:
    """Control. The mechanism must be invisible when nothing diverged.

    Every committed crossval record on ``main`` was produced by a run that
    agreed with itself, so a finalizer that wrote something here would change
    what a re-run produces -- and cv01's and cv02's manifest entries claim a
    re-run reproduces them bit for bit (#937).
    """
    rc, doc, stderr = _replay(case, tmp_path)

    assert doc["verdict"]["exit_code"] == rc
    assert "exit_code_declared" not in doc["verdict"]
    assert "exit_code_reconciliation" not in doc["verdict"]
    assert "summary_declared" not in doc["verdict"]
    assert doc["verdict"]["summary"] in (
        "ALL CHECKS PASSED",
        "[SKIP] Meep reference unavailable — crossval inconclusive (exit 2)",
        "SOME CHECKS FAILED")
    # It armed: an unarmed write announces itself, so silence is the evidence.
    assert "EXIT-CODE EVIDENCE UNARMED" not in stderr
    assert "EXIT-CODE RECONCILED" not in stderr


@pytest.mark.parametrize("case", sorted(REPLAY_CASES))
def test_a_late_sys_exit_makes_the_record_state_the_process_status(
        case: str, tmp_path: Path) -> None:
    """The #907 shape, on the real script: a guard added after the writer.

    3 is deliberately outside this case's own {0, 1, 2} convention, so the
    value in the file can only have come from the process, never from the
    verdict stage re-running its own mapping.
    """
    declared_rc, declared_doc, _ = _replay(case, tmp_path / "control")
    rc, doc, stderr = _replay(case, tmp_path / "forced", force="exit:3")

    assert rc == 3
    assert doc["verdict"]["exit_code"] == rc            # the #946 assertion
    assert doc["verdict"]["exit_code_declared"] == declared_rc
    assert doc["verdict"]["summary_declared"] == declared_doc["verdict"]["summary"]
    reconciliation = doc["verdict"]["exit_code_reconciliation"]
    assert reconciliation == {
        "declared": declared_rc, "actual": 3, "observed_via": "sys.exit",
        "note": reconciliation["note"]}
    assert reconciliation["note"].startswith("the process exited with a status")
    # The amended summary states what happened; it is NOT the case's own
    # pass/fail wording re-run on a code its gate stage never reached.
    assert doc["verdict"]["summary"].startswith("EXIT 3 --")
    assert "EXIT-CODE RECONCILED" in stderr
    # The measurement the case wrote is untouched by the amendment.
    assert doc["measured"] == declared_doc["measured"]


@pytest.mark.parametrize("case", sorted(REPLAY_CASES))
def test_an_uncaught_exception_after_the_write_is_recorded_as_status_one(
        case: str, tmp_path: Path) -> None:
    """The other half of the gap: the tail raises instead of exiting.

    cv01's plotting stage and cv02's PART 4 are ordinary code that can throw.
    ``sys.excepthook`` is what sees it, and the status a parent gets is 1.
    """
    declared_rc, _, _ = _replay(case, tmp_path / "control")
    rc, doc, stderr = _replay(case, tmp_path / "forced", force="raise")

    assert rc == 1
    assert doc["verdict"]["exit_code"] == rc
    if declared_rc == 1:                       # nothing diverged to record
        assert "exit_code_reconciliation" not in doc["verdict"]
        return
    assert doc["verdict"]["exit_code_declared"] == declared_rc
    assert doc["verdict"]["exit_code_reconciliation"]["actual"] == 1
    assert doc["verdict"]["exit_code_reconciliation"]["observed_via"] == (
        "uncaught RuntimeError")
    assert "EXIT-CODE RECONCILED" in stderr


@pytest.mark.parametrize("case", sorted(REPLAY_CASES))
def test_the_replay_re_judges_rather_than_copying_the_source_verdict(
        case: str, tmp_path: Path) -> None:
    """A replay that copied the stored verdict would test nothing.

    The record it writes has to carry its own re-evaluated gate table and say
    where it came from, or the three tests above would pass against a file
    this script merely transcribed.
    """
    _, doc, _ = _replay(case, tmp_path)

    assert doc["replay"]["source"] == str(REPLAY_CASES[case][1])
    assert "no solver ran" in doc["replay"]["note"]
    assert doc["gates"], "the replay produced no gate table"
    assert all(isinstance(v, bool) or v is None for v in doc["gates"].values())


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

    Four of the six writers had no defect to fix; they were migrated for
    uniformity, and the price of that would be real if it moved a byte -- cv01
    and cv02's manifest entries claim a re-run reproduces them bit-identically
    (#937), and cv24's six records carry ``exit_code``/``summary`` ahead of
    ``arms`` rather than after.

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
    directory as cv01's record rather than at the record itself. The guard is
    by directory, so the probe proves the same thing -- and a regression then
    leaves one stray file that this test deletes, instead of a clobbered
    retained record it would have to restore. Measured, not hypothetical: the
    first version of this test aimed at cv01's record, and verifying it by
    reinstating the defect overwrote that record with the fixture's two-key
    document.
    """
    results = CROSSVAL_DIR / "_01_waveguide_bend_results"
    committed = results / "crossval.json"
    committed_before = committed.read_bytes()
    probe = results / "_946_selftest_guard_probe.json"
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


@pytest.mark.parametrize("case", sorted(REPLAY_CASES))
def test_a_replay_refuses_an_out_dir_inside_the_evidence_tree(
        case: str) -> None:
    """The same guard on the other knob that names a write target.

    ``--replay --out-dir`` took its path verbatim, so a replay could be aimed
    at a case's own committed results directory and overwrite the record it
    had just re-judged -- and a re-judge is not a reproduction (cv02's differs
    from the committed one, post-#945 keys) (review round 1, P2-1). The
    refusal is the helper's single spelling of "inside validation/crossval/",
    the same one the self-test knob asks.

    Aimed at a directory that does not exist inside the tree rather than at
    the case's own results directory: the guard is by tree, so the probe
    proves the same thing, and a regression leaves one stray directory this
    test removes instead of a clobbered retained record.
    """
    script, source = REPLAY_CASES[case]
    before = source.read_bytes()
    probe = CROSSVAL_DIR / f"_946_replay_guard_probe_{case}"
    assert not probe.exists(), f"stale probe left behind: {probe}"

    try:
        proc = subprocess.run(
            [sys.executable, str(script), "--replay", str(source),
             "--out-dir", str(probe)],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600)
        written = sorted(q.name for q in probe.glob("*")) if probe.exists() else None
    finally:
        if probe.exists():
            for stray in probe.glob("*"):
                stray.unlink()
            probe.rmdir()

    assert proc.returncode != 0
    assert "--out-dir refuses" in proc.stderr
    assert "#967" in proc.stderr
    assert written is None, (
        f"--out-dir was refused, but only after writing {written} into the "
        "committed evidence tree")
    assert source.read_bytes() == before


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


def test_the_forced_exit_knob_ignores_a_record_it_does_not_name(
        tmp_path: Path) -> None:
    """Naming the absolute path is what keeps a stray variable inert.

    A path that is some other record must not fire -- otherwise the knob could
    be left set in an environment and take down the next case that happens to
    write.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    env_target = tmp_path / "some" / "other" / "record.json"
    script, source = REPLAY_CASES["cv01"]
    env = dict(os.environ)
    env[SELFTEST_ENV] = f"exit:3|{env_target}"

    proc = subprocess.run(
        [sys.executable, str(script), "--replay", str(source),
         "--out-dir", str(out_dir)],
        cwd=str(out_dir), env=env, capture_output=True, text=True, timeout=600)

    assert proc.returncode != 3
    assert "EXIT-CODE SELFTEST IGNORED" in proc.stderr
    doc = json.loads((out_dir / "crossval.json").read_text())
    assert doc["verdict"]["exit_code"] == proc.returncode
    assert "exit_code_reconciliation" not in doc["verdict"]


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


def test_the_source_contract_from_pr_999_still_holds_for_cv02() -> None:
    """The AST invariant is not replaced by this file, it is joined by it.

    PR #999 pinned that every ``sys.exit`` in cv02 takes the Name ``_rc``.
    That is still true and still worth pinning cheaply -- it is what makes the
    single value this file measures a single value. Its home is
    ``tests/crossval/test_cv02_ring_mode_judge.py``; the assertion here is
    that it is still there, so deleting it fails a test in this file too.
    """
    judge_test = (REPO_ROOT / "tests" / "crossval"
                  / "test_cv02_ring_mode_judge.py").read_text()
    assert "def test_cv02_persists_and_exits_through_one_decision_value" in judge_test
