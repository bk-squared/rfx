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
  it. No crossval script may put ``"exit_code"`` into a document by hand,
  every script that calls ``write_record`` must leave through ``sys.exit``,
  which is the only exit the finalizer can observe, and the set of files that
  call ``write_record`` must be exactly the set of case scripts -- a shared
  helper doing the write on a case's behalf persists the record unarmed.

The two boundaries the mechanism cannot cover -- ``os._exit`` skipping
``atexit``, and a ``sys.exit`` the script catches itself -- are asserted here
rather than left to the reader, so the guarantee is not read wider than it is.
And the guarantee it DOES make -- a host process's exit status never reaches
an embedded case's record -- is asserted for both ways of embedding a case,
``importlib`` + ``main()`` and ``runpy`` under ``run_name="__main__"``.
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
    # NOT "SUMMARY FOR EXIT 2": the case's own code->text mapping spells the
    # verdicts its gate stage reached, and 2 is not one of them here. Running
    # it on the new code would state a verdict the run never produced.
    assert "SUMMARY FOR EXIT" not in doc["verdict"]["summary"]
    assert doc["verdict"]["summary"].startswith("EXIT 2 --")
    reconciliation = doc["verdict"]["exit_code_reconciliation"]
    assert reconciliation["declared"] == 0
    assert reconciliation["actual"] == 2
    assert reconciliation["observed_via"] == "sys.exit"
    # Nothing else in the record is touched.
    assert doc["measured"] == {"modes": [1.0, 2.0]}
    assert doc["verdict"]["judge_passed"] is True


# A gate stage's own code->text mapping, verbatim in shape from cv01 and cv02
# (01_waveguide_bend.py:_summary, 02_ring_resonator.py:_summary): every exit
# code it knows about gets a sentence describing a verdict THAT STAGE reached.
GATE_MAPPING_FIXTURE = '''\
import sys

sys.path.insert(0, {crossval_dir!r})
import _exit_evidence


def _summary(code):
    if code == 0:
        return "ALL CHECKS PASSED"
    if code == 2:
        return "[SKIP] Meep reference unavailable - crossval inconclusive (exit 2)"
    return "SOME CHECKS FAILED"


doc = {{"verdict": {{"all_gates_ok": {all_gates_ok}, "meep_present": {meep_present}}}}}
rc = _exit_evidence.write_record({record!r}, doc, exit_code={declared},
                                 summary=_summary)
print("declared", rc)
sys.exit({late})
'''


def _run_gate_mapping(tmp_path: Path, declared: int, late: int,
                      all_gates_ok: bool, meep_present: bool):
    record = tmp_path / "crossval.json"
    script = tmp_path / "cv_gate_mapping.py"
    script.write_text(GATE_MAPPING_FIXTURE.format(
        crossval_dir=str(CROSSVAL_DIR), record=str(record), declared=declared,
        late=late, all_gates_ok=all_gates_ok, meep_present=meep_present))
    proc = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path),
                          capture_output=True, text=True)
    assert record.is_file(), proc.stdout + proc.stderr
    return proc.returncode, json.loads(record.read_text())


def test_an_amended_record_never_manufactures_a_pass_headline(
        tmp_path: Path) -> None:
    """cv01's shape: declared 1 with gates failed, late exit 0.

    Regenerating the summary from the case's own mapping would put
    "ALL CHECKS PASSED" next to ``all_gates_ok: false`` in the same document.
    The amended record states what happened instead, and keeps the gate
    stage's own sentence under ``summary_declared``.
    """
    returncode, doc = _run_gate_mapping(
        tmp_path, declared=1, late=0, all_gates_ok=False, meep_present=True)

    assert returncode == 0
    verdict = doc["verdict"]
    assert verdict["exit_code"] == 0
    assert verdict["all_gates_ok"] is False
    assert "ALL CHECKS PASSED" not in verdict["summary"]
    assert verdict["summary_declared"] == "SOME CHECKS FAILED"
    assert verdict["summary"].startswith("EXIT 0 --")


def test_an_amended_record_never_manufactures_a_skip_reason(
        tmp_path: Path) -> None:
    """cv02's shape, the #907 one: declared 0 with Meep present, late exit 2.

    The case's mapping spells exit 2 as "Meep reference unavailable", which is
    a reason, not a status -- and it is false here, in the same document that
    records ``meep_present: true``.
    """
    returncode, doc = _run_gate_mapping(
        tmp_path, declared=0, late=2, all_gates_ok=True, meep_present=True)

    assert returncode == 2
    verdict = doc["verdict"]
    assert verdict["exit_code"] == 2
    assert verdict["meep_present"] is True
    assert "Meep reference unavailable" not in verdict["summary"]
    assert verdict["summary_declared"] == "ALL CHECKS PASSED"
    assert verdict["summary"].startswith("EXIT 2 --")


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


# A sys.exit the script catches itself, after which it finishes normally.
SWALLOWED_EXIT_FIXTURE = '''\
import sys

sys.path.insert(0, {crossval_dir!r})
import _exit_evidence

doc = {{"verdict": {{"judge_passed": True}}}}
rc = _exit_evidence.write_record({record!r}, doc, exit_code=0,
                                 summary="DECLARED EXIT 0")
try:
    sys.exit(2)
except SystemExit:
    pass
print("the exit was swallowed; this process returns 0")
'''


def test_a_swallowed_sys_exit_is_the_documented_last_call_boundary(
        tmp_path: Path) -> None:
    """The wrapper records the last ``sys.exit`` CALL, not the exit status.

    A ``sys.exit`` that something catches and does not re-raise still counts,
    so this process returns 0 while its record is amended to 2. The five
    migrated cases cannot reach this -- each one's last act is
    ``sys.exit(rc)``, and last-call-wins is then the right reading -- but the
    helper is offered as the road for future cases, so the boundary is pinned
    here the same way ``os._exit`` is, instead of living only in prose.
    """
    record = tmp_path / "crossval.json"
    script = tmp_path / "cv_swallowed.py"
    script.write_text(SWALLOWED_EXIT_FIXTURE.format(
        crossval_dir=str(CROSSVAL_DIR), record=str(record)))

    proc = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path),
                          capture_output=True, text=True)
    doc = json.loads(record.read_text())

    assert proc.returncode == 0
    assert doc["verdict"]["exit_code"] == 2  # the boundary, stated
    assert doc["verdict"]["exit_code_declared"] == 0
    assert "EXIT-CODE RECONCILED" in proc.stderr


# A case that hands its write to a module it imports. The record is persisted
# UNARMED -- the pre-#946 state. The notice on stderr is what keeps that from
# being silent.
WRITE_THROUGH_HELPER_MODULE = '''\
import sys

sys.path.insert(0, {crossval_dir!r})
import _exit_evidence


def persist(record, code):
    return _exit_evidence.write_record(
        record, {{"verdict": {{"judge_passed": True}}}}, exit_code=code,
        summary="DECLARED EXIT %d" % code)
'''

CASE_WRITING_THROUGH_A_HELPER = '''\
import sys

sys.path.insert(0, {tmp!r})
import case_write_helper

rc = case_write_helper.persist({record!r}, 0)
print("declared", rc)
sys.exit(2)
'''


def test_a_record_persisted_unarmed_says_so_on_stderr(tmp_path: Path) -> None:
    """Unarmed is the pre-#946 state; it must not also be the silent one.

    ``write_record`` arms only when its caller is the program, so a case that
    routes the write through a module it imports gets a record nothing will
    amend. One line on stderr, naming the module that called from outside
    ``__main__``; the static half below keeps that shape out of
    ``validation/crossval/`` in the first place.
    """
    (tmp_path / "case_write_helper.py").write_text(
        WRITE_THROUGH_HELPER_MODULE.format(crossval_dir=str(CROSSVAL_DIR)))
    record = tmp_path / "crossval.json"
    script = tmp_path / "cv_case.py"
    script.write_text(CASE_WRITING_THROUGH_A_HELPER.format(
        tmp=str(tmp_path), record=str(record)))

    proc = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path),
                          capture_output=True, text=True)
    doc = json.loads(record.read_text())

    assert proc.returncode == 2
    assert doc["verdict"]["exit_code"] == 0          # unarmed: never amended
    assert "exit_code_reconciliation" not in doc["verdict"]
    assert "EXIT-CODE EVIDENCE UNARMED" in proc.stderr
    assert "case_write_helper" in proc.stderr


CV23 = CROSSVAL_DIR / "23_lossy_slab_fresnel.py"
# --smoke evaluates no gate and --out-dir keeps the write out of
# validation/crossval (#967 / PR #977).
CV23_SMOKE = ["--smoke", "--no-plots", "--arms", "tand1", "--out-dir"]


def test_the_committed_case_arms_when_it_is_the_program(tmp_path: Path) -> None:
    """cv23 in a subprocess, run the way evidence is produced: as the program.

    The fixture tests above pin what arming DOES. This one pins that a
    committed crossval case is wired into it end to end -- through its own
    argument parsing and its own writer -- and that the arming predicate still
    accepts the only shape that produces evidence. Arming is visible from
    outside because an unarmed write announces itself on stderr, so the
    absence of that line is the assertion.
    """
    out_dir = tmp_path / "cv23_out"
    out_dir.mkdir()

    proc = subprocess.run([sys.executable, str(CV23), *CV23_SMOKE,
                           str(out_dir)],
                          cwd=str(tmp_path), capture_output=True, text=True)
    record = out_dir / "rfx.json"
    assert record.is_file(), proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(record.read_text())

    assert proc.returncode == 0
    assert doc["verdict"]["exit_code"] == proc.returncode
    assert doc["verdict"]["summary"].startswith("SMOKE OK")
    assert "EXIT-CODE EVIDENCE UNARMED" not in proc.stderr   # it armed,
    assert "EXIT-CODE RECONCILED" not in proc.stderr         # and agreed
    # The measurement the case wrote is still the case's own.
    assert doc["arms"]["tand1"]["e2_ok"] in (True, False)


# A host that runs the case as ``__main__`` through runpy, swallows the exit
# the case asked for, and then returns a status of its own. Two diagnostics
# harnesses embed a case this way: cv0104_dielectric_control_witness.py hands
# over sys.argv as this wrapper does, harminv_record_capture.py keeps its own.
# The wrapper here takes the harder variant -- with argv handed over, nothing
# in sys.argv or sys.modules distinguishes it from a direct run.
RUNPY_HOST = '''\
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


def test_a_case_run_as_main_by_runpy_is_not_armed(tmp_path: Path) -> None:
    """runpy makes ``__name__ == "__main__"`` true; it must not be enough.

    The host here returns 2 for a case that declared 0, so a mechanism that
    armed on the module name alone would write the host's status into the
    case's record -- a manufactured exit code, which is what #946 is about.
    The record keeps what the case itself decided, and the host is told the
    record is unarmed and why.
    """
    out_dir = tmp_path / "cv23_out"
    out_dir.mkdir()
    host = tmp_path / "runpy_host.py"
    host.write_text(RUNPY_HOST.format(script=str(CV23), out_dir=str(out_dir)))

    proc = subprocess.run([sys.executable, str(host)], cwd=str(tmp_path),
                          capture_output=True, text=True)
    record = out_dir / "rfx.json"
    assert record.is_file(), proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(record.read_text())

    assert proc.returncode == 2
    assert doc["verdict"]["exit_code"] == 0  # what the case itself returned
    assert "exit_code_declared" not in doc["verdict"]
    assert "exit_code_reconciliation" not in doc["verdict"]
    assert "EXIT-CODE RECONCILED" not in proc.stderr
    assert "EXIT-CODE EVIDENCE UNARMED" in proc.stderr
    assert "EMBEDDED" in proc.stderr


# Why the arming predicate walks the stack instead of reading sys.argv or
# sys.modules: under runpy both report a direct run.
RUNPY_LOOKALIKE_CASE = '''\
import json
import sys

print(json.dumps(dict(argv0=sys.argv[0], name=__name__,
                      main_module_is_me=sys.modules["__main__"].__dict__ is globals())))
'''

RUNPY_LOOKALIKE_HOST = '''\
import runpy
import sys

sys.argv = [{case!r}]
runpy.run_path({case!r}, run_name="__main__")
'''


def test_runpy_as_main_looks_like_a_direct_run_in_argv_and_sys_modules(
        tmp_path: Path) -> None:
    """The measurement behind the arming predicate, not an assumption.

    ``runpy`` rewrites ``sys.argv[0]`` to the case's own path and installs the
    case as ``sys.modules["__main__"]`` while it runs, so both cheap checks an
    author might reach for report "this is the program". Pinned here so that a
    later simplification to either one fails this test instead of quietly
    reopening the hole the test above closes.
    """
    case = tmp_path / "lookalike_case.py"
    case.write_text(RUNPY_LOOKALIKE_CASE)
    host = tmp_path / "lookalike_host.py"
    host.write_text(RUNPY_LOOKALIKE_HOST.format(case=str(case)))

    proc = subprocess.run([sys.executable, str(host)], cwd=str(tmp_path),
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    seen = json.loads(proc.stdout)

    assert seen["name"] == "__main__"
    assert seen["argv0"] == str(case)          # not the host's path
    assert seen["main_module_is_me"] is True   # not the host's module


# The same case driven as a LIBRARY: imported under its own module name and
# called by a host whose exit status has nothing to do with the case. This is
# how tests/crossval/test_cv23_lossy_slab_gates.py drives it.
EMBEDDED_WRAPPER = '''\
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("cv23_embedded", {script!r})
mod = importlib.util.module_from_spec(spec)
sys.modules["cv23_embedded"] = mod
spec.loader.exec_module(mod)
rc = mod.main(["--smoke", "--no-plots", "--arms", "tand1",
               "--out-dir", {out_dir!r}])
print("the case returned", rc)
sys.exit(2)
'''


def test_a_host_process_exit_status_never_stamps_an_embedded_case(
        tmp_path: Path) -> None:
    """An imported case is not the program, so its record is not armed.

    A pytest run that drives ``main()`` in-process, and a ``pytest.raises(
    SystemExit)`` elsewhere in the same session, must not be able to write
    their own status into a case record.
    """
    out_dir = tmp_path / "cv23_out"
    out_dir.mkdir()
    wrapper = tmp_path / "embedded_wrapper.py"
    wrapper.write_text(EMBEDDED_WRAPPER.format(
        script=str(CV23), out_dir=str(out_dir)))

    proc = subprocess.run([sys.executable, str(wrapper)], cwd=str(tmp_path),
                          capture_output=True, text=True)
    record = out_dir / "rfx.json"
    assert record.is_file(), proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(record.read_text())

    assert proc.returncode == 2
    assert doc["verdict"]["exit_code"] == 0  # what the case itself returned
    assert "exit_code_reconciliation" not in doc["verdict"]
    assert "EXIT-CODE RECONCILED" not in proc.stderr
    # ...and the host is told the record is unarmed, so "nothing happened"
    # and "the mechanism is off here" are not the same silence.
    assert "EXIT-CODE EVIDENCE UNARMED" in proc.stderr


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
        # Called from a test module, not from a program: the record is written
        # and nothing is armed, because this pytest process's exit status is
        # not that record's verdict.
        assert helper.armed_records() == {}
    finally:
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


def test_write_record_is_only_called_from_the_case_scripts_themselves() -> None:
    """No shared helper may do a case's write for it.

    ``write_record`` arms the finalizer only when its caller is running as the
    program, so a case that routes its write through a module it imports gets
    a record nothing will amend -- the pre-#946 state, and the call site is
    then in a file the per-file scan above would happily accept. Keeping this
    set EXACT is the static half of that guard (the runtime half is the
    ``EXIT-CODE EVIDENCE UNARMED`` notice). A new case joins by being added to
    MIGRATED_WRITERS; a helper cannot join without someone reading this.
    """
    callers = {rel for rel, tree in _crossval_sources()
               if _calls_write_record(tree)}
    names = {Path(rel).name for rel in callers}
    assert names == set(MIGRATED_WRITERS), (
        "files under validation/crossval/ calling write_record that are not "
        "declared case writers: %s" % sorted(names - set(MIGRATED_WRITERS)))


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


CV24_RESULTS = CROSSVAL_DIR / "_24_nu_cavity_results"


def test_reserved_verdict_slots_keep_the_committed_key_order(
        tmp_path: Path) -> None:
    """cv24's records carry exit_code/summary BEFORE the rest of the block.

    ``write_record`` fills a key in place when it exists and appends it
    otherwise, so the position is the caller's. cv01/cv02 let it append,
    matching their records; cv24 reserves the slots first. Getting this wrong
    costs no gate -- and produces a re-run diff nobody can explain, which is
    the evidence problem this whole file is about.
    """
    helper = _load_helper()
    record = tmp_path / "rfx.json"
    doc = {"arms": {}, "verdict": helper.reserve_verdict(arms=["uniform"])}
    try:
        helper.write_record(str(record), doc, exit_code=0,
                            summary="SMOKE OK", arm=False)
    finally:
        helper._reset_for_tests()
    produced = list(json.loads(record.read_text())["verdict"])

    assert produced == ["exit_code", "summary", "arms"]
    committed = sorted(CV24_RESULTS.glob("*.json"))
    assert committed, "no committed cv24 record to compare the order against"
    for path in committed:
        assert list(json.loads(path.read_text())["verdict"]) == produced, path


def test_the_case_that_commits_that_order_is_the_one_that_reserves() -> None:
    """cv24 must still go through ``reserve_verdict``.

    The test above pins what the helper produces; this pins that cv24 asks
    for it. Without the reservation cv24 emits arms/exit_code/summary and
    every committed record under _24_nu_cavity_results/ reads
    exit_code/summary/arms.
    """
    tree = ast.parse((CROSSVAL_DIR / "24_nu_rect_cavity_pozar.py").read_text())
    reserved = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ((isinstance(node.func, ast.Attribute)
              and node.func.attr == "reserve_verdict")
             or (isinstance(node.func, ast.Name)
                 and node.func.id == "reserve_verdict"))
    ]
    assert reserved, (
        "24_nu_rect_cavity_pozar.py no longer reserves its verdict slots, so "
        "a re-run reorders the block against its committed records")


def test_the_helper_is_not_itself_a_crossval_case() -> None:
    """It is an underscore-prefixed helper, which the manifest excludes."""
    assert HELPER.is_file()
    assert HELPER.name.startswith("_")
    manifest = json.loads((CROSSVAL_DIR / "manifest.json").read_text())
    scripts = {case["script"] for case in manifest["cases"]}
    assert HELPER.relative_to(REPO_ROOT).as_posix() not in scripts
