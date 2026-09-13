"""Execute cv02's real persistence/summary/exit tail in a subprocess.

Only the expensive field visualization is replaced. The judge, verdict
dictionary expression, exit policy and lexical writer/exit scope are read
from the production code and executed, rather than checking variable names.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "validation/crossval/02_ring_resonator.py"


def _tail_program(board, fault, artifact, *, host_exit=None):
    tree = ast.parse(CASE.read_text())
    exit_policy = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == "_exit_code")
    document = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "_doc" for t in n.targets))
    verdict_expr = next(value for key, value in zip(document.keys, document.values)
                        if isinstance(key, ast.Constant) and key.value == "verdict")
    # Executing the actual tail keeps a new exit path in the regression test.
    tail = tree.body[-1]
    assert isinstance(tail, ast.With), "all post-write case exits must remain in the guarded tail"
    setup = f'''
import sys
from _retained_verdict import retained_verdict
from comparators.ring_mode_judge import ReferenceMode, SolverMode, judge
reference = [ReferenceMode(1., 10.), ReferenceMode(1.2, 10.)]
rfx_modes = [SolverMode(1., 10.), SolverMode(1.2, 10.)]
if {board!r} == "failed":
    rfx_modes = rfx_modes[:1]
if {board!r} == "empty":
    rfx_modes = []
HAVE_MEEP = {board!r} != "no_reference"
meep_modes = reference if HAVE_MEEP else []
matched = []
verdict = judge(reference, rfx_modes, 10., f_min=.5, f_max=1.5)
_rfx_self_ok = bool(rfx_modes)
_artifact = {str(artifact)!r}
'''
    prepare = ast.unparse(exit_policy) + '\n_rc = _exit_code(_rfx_self_ok, HAVE_MEEP, verdict.passed)\n'
    prepare += '_doc = {"verdict": ' + ast.unparse(verdict_expr) + '}\n'
    callback = f'''
def _visualize_comparison():
    global _rc
    import json
    # The evidence is already on disk before optional work can fail.
    assert json.load(open(_artifact))["verdict"]["exit_code"] == _rc
    fault = {fault!r}
    if fault == "late_two":
        _rc = 2
    elif fault == "late_zero":
        _rc = 0
    elif fault == "crash":
        raise RuntimeError("optional visualization failed")
    elif fault == "direct_exit":
        sys.exit(2)
    elif fault == "message_exit":
        sys.exit("optional visualization failed")
    elif fault == "negative_exit":
        sys.exit(-1)
    elif fault == "interrupt":
        raise KeyboardInterrupt
'''
    body = ast.unparse(tail)
    if host_exit is not None:
        body = 'try:\n' + '\n'.join('    '+line for line in body.splitlines())
        body += f'\nexcept SystemExit:\n    pass\nsys.exit({host_exit})\n'
    return setup + prepare + callback + body


def _run(tmp_path, board, fault, host_exit=None):
    artifact = tmp_path / "crossval.json"
    program = tmp_path / "case_tail.py"
    program.write_text(_tail_program(board, fault, artifact, host_exit=host_exit))
    env = dict(os.environ, PYTHONPATH=str(ROOT / "validation/crossval"))
    result = subprocess.run([sys.executable, str(program)], env=env,
                            capture_output=True, text=True, timeout=60)
    assert artifact.exists(), result.stderr
    return result, json.loads(artifact.read_text())["verdict"]


@pytest.mark.parametrize("board,code", [("passed", 0), ("failed", 1),
                                      ("empty", 1), ("no_reference", 2)])
def test_case_record_equals_the_actual_pass_fail_or_missing_reference_exit(tmp_path, board, code):
    result, verdict = _run(tmp_path, board, None)
    assert result.returncode == verdict["exit_code"] == code, result.stderr
    assert (verdict["summary"] == "ALL CHECKS PASSED") == (code == 0)
    assert "declared_exit_code" not in verdict


@pytest.mark.parametrize("board,fault,code", [
    ("passed", "late_two", 2), ("failed", "late_zero", 0),
    ("passed", "crash", 1), ("passed", "direct_exit", 2),
    ("passed", "message_exit", 1), ("passed", "negative_exit", 255),
    ("passed", "interrupt", -2),
])
def test_late_decision_or_exception_cannot_leave_a_false_retained_outcome(tmp_path, board, fault, code):
    result, verdict = _run(tmp_path, board, fault)
    assert result.returncode == verdict["exit_code"] == code, result.stderr
    assert verdict["declared_exit_code"] == (1 if board == "failed" else 0)
    assert verdict["summary"] != "ALL CHECKS PASSED"
    assert "PROCESS OUTCOME CHANGED" in verdict["summary"]


def test_embedding_host_does_not_rewrite_the_cases_own_outcome(tmp_path):
    result, verdict = _run(tmp_path, "passed", None, host_exit=2)
    assert result.returncode == 2
    assert verdict["exit_code"] == 0
    assert verdict["summary"] == "ALL CHECKS PASSED"
