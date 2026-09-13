"""A retained crossval record must state the exit code the process returned.

Issue #946. The crossval writers persist their record BEFORE the run ends, so
that a crash in the plotting stage that follows cannot take the measurement
with it. The price is that the record states an outcome the process has not
reached yet, and an exit path added between the write and the tail makes the
committed file disagree with the run. That is what the abandoned #907 branch
did to cv02:

    q_vacuous            true
    judge_passed         true
    persisted exit_code  0        <- retained evidence claims a pass
    persisted summary    ALL CHECKS PASSED
    actual branch exit   2        <- the process reported inconclusive

Nothing failed, because nothing compared the two. This file compares them.

Two halves:

* BEHAVIOUR -- a crossval-shaped script runs in a subprocess with a forced
  late exit, and the record it left behind is read back and compared against
  the subprocess's own return code. The fixture script writes into tmp_path,
  never into ``validation/crossval``: regenerating committed evidence from a
  test is the #967 defect that PR #977 closed, and this test must not
  reintroduce it under a different name.
* STATIC -- the mechanism only holds if every writer actually routes through
  it. No crossval script may put ``"exit_code"`` into a document by hand, and
  every script that calls ``write_record`` must leave through ``sys.exit``,
  which is the only exit the finalizer can observe.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CROSSVAL_DIR = REPO_ROOT / "validation" / "crossval"
HELPER = CROSSVAL_DIR / "_exit_evidence.py"

# The writers migrated in #946. New cases may join; a case leaving this set
# silently is the regression this assertion catches.
MIGRATED_WRITERS = frozenset({
    "01_waveguide_bend.py",
    "02_ring_resonator.py",
    "22_dispersive_slab_fresnel.py",
    "23_lossy_slab_fresnel.py",
    "24_nu_rect_cavity_pozar.py",
})

# A crossval-shaped script: decide, persist, then do the optional stage that
# may or may not exit differently. `LATE` is spliced in where cv01's plot and
# cv02's narrowband visualisation sit -- after the record is on disk.
FIXTURE = '''\
import os
import sys

sys.path.insert(0, {crossval_dir!r})
import _exit_evidence


def _summary(code):
    return "SUMMARY FOR EXIT %d" % code


def main():
    doc = {{
        "schema": "cv-fixture/v1",
        "measured": {{"modes": [1.0, 2.0]}},
        "verdict": {{"judge_passed": True}},
    }}
    return _exit_evidence.write_record(
        {record!r}, doc, exit_code={declared}, summary=_summary)


rc = main()
print("record written, declared exit", rc)
{late}
sys.exit(rc)
'''


def _run_fixture(tmp_path: Path, declared: int, late: str) -> tuple[int, dict]:
    """Run the fixture script in a subprocess; return (returncode, record)."""
    record = tmp_path / "crossval.json"
    script = tmp_path / "cv_fixture.py"
    script.write_text(FIXTURE.format(
        crossval_dir=str(CROSSVAL_DIR), record=str(record),
        declared=declared, late=late))
    proc = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path),
                          capture_output=True, text=True)
    assert record.is_file(), (
        "the record was not persisted at all:\n" + proc.stdout + proc.stderr)
    return proc.returncode, json.loads(record.read_text())


# --------------------------------------------------------------------------
# BEHAVIOUR
# --------------------------------------------------------------------------

def test_late_exit_path_rewrites_the_persisted_exit_code(tmp_path: Path) -> None:
    """The #907 shape: a guard added AFTER the writer exits 2 on a declared 0."""
    returncode, doc = _run_fixture(tmp_path, declared=0, late="sys.exit(2)")

    assert returncode == 2
    assert doc["verdict"]["exit_code"] == returncode
    # The declared verdict is kept, not deleted -- a reader must be able to
    # see what the gate stage decided and what the process then did.
    assert doc["verdict"]["exit_code_declared"] == 0
    assert doc["verdict"]["summary_declared"] == "SUMMARY FOR EXIT 0"
    assert doc["verdict"]["summary"] == "SUMMARY FOR EXIT 2"
    reconciliation = doc["verdict"]["exit_code_reconciliation"]
    assert reconciliation["declared"] == 0
    assert reconciliation["actual"] == 2
    assert reconciliation["observed_via"] == "sys.exit"
    # Nothing else in the record is touched.
    assert doc["measured"] == {"modes": [1.0, 2.0]}
    assert doc["verdict"]["judge_passed"] is True


def test_a_failing_optional_stage_is_not_reported_as_a_pass(tmp_path: Path) -> None:
    """An exception in the stage after the writer is exit 1, and says so."""
    returncode, doc = _run_fixture(
        tmp_path, declared=0, late="raise RuntimeError('plot stage blew up')")

    assert returncode == 1
    assert doc["verdict"]["exit_code"] == returncode
    assert doc["verdict"]["exit_code_declared"] == 0
    assert (doc["verdict"]["exit_code_reconciliation"]["observed_via"]
            == "uncaught RuntimeError")
    # The measurement itself survives the crash -- that is why the write is
    # early in the first place.
    assert doc["measured"] == {"modes": [1.0, 2.0]}


def test_an_inconclusive_run_that_later_exits_zero_loses_its_pass_claim(
        tmp_path: Path) -> None:
    """The reverse direction: declared 2, process returns 0."""
    returncode, doc = _run_fixture(tmp_path, declared=2, late="sys.exit(0)")

    assert returncode == 0
    assert doc["verdict"]["exit_code"] == 0
    assert doc["verdict"]["exit_code_declared"] == 2


@pytest.mark.parametrize("declared", [0, 1, 2])
def test_agreement_leaves_the_record_byte_identical(tmp_path: Path,
                                                    declared: int) -> None:
    """The normal case writes nothing extra.

    The cv01/cv02 manifest entries claim a re-run on the same commit and rig
    reproduces the committed record bit-identically. A finalizer that stamped
    every run would break that claim, so it only ever writes on disagreement.
    """
    record = tmp_path / "crossval.json"
    script = tmp_path / "cv_fixture.py"
    script.write_text(FIXTURE.format(
        crossval_dir=str(CROSSVAL_DIR), record=str(record),
        declared=declared, late="print('optional stage ran')"))

    proc = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path),
                          capture_output=True, text=True)
    doc = json.loads(record.read_text())

    assert proc.returncode == declared
    assert doc["verdict"]["exit_code"] == declared
    assert doc["verdict"]["summary"] == "SUMMARY FOR EXIT %d" % declared
    for key in ("exit_code_declared", "summary_declared",
                "exit_code_reconciliation"):
        assert key not in doc["verdict"]
    assert "RECONCILED" not in proc.stderr


def test_disagreement_is_announced_on_stderr(tmp_path: Path) -> None:
    """A silently amended record would be its own evidence problem."""
    record = tmp_path / "crossval.json"
    script = tmp_path / "cv_fixture.py"
    script.write_text(FIXTURE.format(
        crossval_dir=str(CROSSVAL_DIR), record=str(record),
        declared=0, late="sys.exit(1)"))

    proc = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path),
                          capture_output=True, text=True)

    assert proc.returncode == 1
    assert "EXIT-CODE RECONCILED" in proc.stderr


def test_os_underscore_exit_is_the_documented_uncoverable_path(
        tmp_path: Path) -> None:
    """``os._exit`` skips atexit, so no in-process finalizer can see it.

    This is asserted rather than left implicit: the record DOES keep its
    declared code here, and the static half of this file is what keeps
    crossval scripts from reaching for that door.
    """
    returncode, doc = _run_fixture(tmp_path, declared=0, late="os._exit(3)")

    assert returncode == 3
    assert doc["verdict"]["exit_code"] == 0  # the boundary, stated
    assert "exit_code_reconciliation" not in doc["verdict"]


CV23 = CROSSVAL_DIR / "23_lossy_slab_fresnel.py"

# Runs the real case, swallows the exit it asked for, then exits 2 -- the
# shape of an exit path added after the record was already persisted. The
# case runs in --smoke with its own --out-dir, so it writes nothing into
# validation/crossval (#967 / PR #977) and evaluates no gate.
LATE_EXIT_WRAPPER = '''\
import runpy
import sys

sys.argv = [{script!r}, "--smoke", "--no-plots", "--arms", "tand1",
            "--out-dir", {out_dir!r}]
try:
    runpy.run_path({script!r}, run_name="__main__")
except SystemExit as exc:
    print("the case asked for exit", exc.code, file=sys.stderr)
sys.exit(2)
'''


def test_a_real_case_record_follows_a_forced_late_exit(tmp_path: Path) -> None:
    """cv23 in a subprocess: the record on disk equals the return code.

    The fixture tests above pin the mechanism; this one pins that a committed
    crossval case is actually wired into it, end to end, through its own
    argument parsing and its own writer.
    """
    out_dir = tmp_path / "cv23_out"
    out_dir.mkdir()
    wrapper = tmp_path / "late_exit_wrapper.py"
    wrapper.write_text(LATE_EXIT_WRAPPER.format(
        script=str(CV23), out_dir=str(out_dir)))

    proc = subprocess.run([sys.executable, str(wrapper)], cwd=str(tmp_path),
                          capture_output=True, text=True)
    record = out_dir / "rfx.json"
    assert record.is_file(), proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(record.read_text())

    assert proc.returncode == 2
    assert doc["verdict"]["exit_code"] == proc.returncode
    assert doc["verdict"]["exit_code_declared"] == 0
    assert doc["verdict"]["summary_declared"].startswith("SMOKE OK")
    assert "EXIT-CODE RECONCILED" in proc.stderr
    # The measurement the case wrote is still the case's own.
    assert doc["arms"]["tand1"]["e2_ok"] in (True, False)


def _load_helper():
    """Import ``_exit_evidence`` the way a crossval script does."""
    sys.path.insert(0, str(CROSSVAL_DIR))
    try:
        import _exit_evidence
    finally:
        sys.path.remove(str(CROSSVAL_DIR))
    return _exit_evidence


def test_write_record_is_the_only_source_of_the_code_in_the_document(
        tmp_path: Path) -> None:
    """The caller cannot hold a second copy that drifts from the exit value."""
    helper = _load_helper()
    record = tmp_path / "nested" / "rec.json"
    doc: dict = {"verdict": {"judge_passed": False}}
    try:
        rc = helper.write_record(str(record), doc, exit_code=1,
                                 summary="SOME CHECKS FAILED")
        assert rc == 1
        assert doc["verdict"]["exit_code"] == 1
        persisted = json.loads(record.read_text())
        assert persisted["verdict"] == {"judge_passed": False, "exit_code": 1,
                                        "summary": "SOME CHECKS FAILED"}
    finally:
        # Never leave this pytest process holding an armed record.
        helper._reset_for_tests()


@pytest.mark.parametrize("code,expected", [
    (None, 0), (0, 0), (2, 2), (True, 1), ("a message", 1), (256, 0), (-1, 255),
])
def test_normalize_matches_what_a_parent_process_sees(code, expected) -> None:
    assert _load_helper().normalize_exit_code(code) == expected


# --------------------------------------------------------------------------
# STATIC -- the mechanism has to be the only road
# --------------------------------------------------------------------------

def _crossval_sources() -> "list[tuple[str, ast.Module]]":
    out = []
    for path in sorted(CROSSVAL_DIR.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        out.append((rel, ast.parse(path.read_text(), filename=rel)))
    return out


def _calls_write_record(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "write_record":
            return True
        if isinstance(func, ast.Name) and func.id == "write_record":
            return True
    return False


def _hand_written_exit_code_keys(tree: ast.Module) -> "list[int]":
    """Line numbers where ``exit_code`` is put into a document by hand."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and key.value == "exit_code":
                    lines.append(key.lineno)
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value == "exit_code"):
                    lines.append(target.lineno)
    return sorted(lines)


def test_no_crossval_script_writes_exit_code_into_a_document_by_hand() -> None:
    """One road into the record, so the arming step cannot be skipped.

    A hand-written ``"exit_code": rc`` is a second copy of the process outcome
    that no finalizer owns -- exactly the shape #946 reports.
    """
    offenders = {
        rel: _hand_written_exit_code_keys(tree)
        for rel, tree in _crossval_sources()
        if _hand_written_exit_code_keys(tree)
    }
    assert offenders == {}, (
        "put the code in through _exit_evidence.write_record(exit_code=...) "
        "instead: %s" % offenders)


def test_every_exit_code_writer_is_migrated() -> None:
    callers = {rel for rel, tree in _crossval_sources()
               if _calls_write_record(tree)}
    names = {Path(rel).name for rel in callers}
    assert MIGRATED_WRITERS <= names, (
        "a crossval case stopped routing its exit code through "
        "_exit_evidence.write_record: %s" % sorted(MIGRATED_WRITERS - names))


def test_writers_leave_through_sys_exit_only() -> None:
    """The finalizer sees ``sys.exit`` and uncaught exceptions, nothing else.

    ``raise SystemExit(n)`` is handled by CPython before ``sys.excepthook``
    runs, and ``os._exit`` skips ``atexit`` entirely, so either one would make
    the persisted code unverifiable for the case that used it. The rest of the
    crossval suite may keep using ``raise SystemExit`` -- those scripts write
    no exit code into a record.
    """
    problems: "dict[str, list[str]]" = {}
    for rel, tree in _crossval_sources():
        if not _calls_write_record(tree):
            continue
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise):
                exc = node.exc
                name = None
                if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                    name = exc.func.id
                elif isinstance(exc, ast.Name):
                    name = exc.id
                if name == "SystemExit":
                    found.append("raise SystemExit at line %d" % node.lineno)
            elif (isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "_exit"):
                found.append("os._exit at line %d" % node.lineno)
            elif isinstance(node, ast.ImportFrom) and node.module == "sys":
                if any(alias.name == "exit" for alias in node.names):
                    found.append("from sys import exit at line %d" % node.lineno)
        if found:
            problems[rel] = found
    assert problems == {}, (
        "these scripts persist an exit code but can leave by a door the "
        "finalizer cannot see: %s" % problems)


def test_the_helper_is_not_itself_a_crossval_case() -> None:
    """It is an underscore-prefixed helper, which the manifest excludes."""
    assert HELPER.is_file()
    assert HELPER.name.startswith("_")
    manifest = json.loads((CROSSVAL_DIR / "manifest.json").read_text())
    scripts = {case["script"] for case in manifest["cases"]}
    assert HELPER.relative_to(REPO_ROOT).as_posix() not in scripts
