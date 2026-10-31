"""The optional mypy gate distinguishes regressions, improvements and crashes."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "mypy_count", REPO / "scripts/ci/check_mypy_count.py",
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


@pytest.fixture
def run_gate(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gate, "REPO", tmp_path)
    baseline = tmp_path / gate.BASELINE
    baseline.parent.mkdir(parents=True)

    def run(output, returncode=1, limit=1382):
        baseline.write_text(f"{limit}\n")

        def fake_mypy(args, **kwargs):
            assert args == [sys.executable, "-m", "mypy", "rfx"]
            assert kwargs["cwd"] == tmp_path
            return subprocess.CompletedProcess(args, returncode, output)

        monkeypatch.setattr(gate.subprocess, "run", fake_mypy)
        code = gate.main()
        return code, capsys.readouterr()

    return run


def test_equal_count_passes(run_gate):
    code, output = run_gate("Found 1382 errors in 80 files (checked 900 source files)\n")
    assert code == 0
    assert output.out == "mypy errors: 1382 (baseline 1382)\n"


def test_lowering_baseline_by_one_fails(run_gate):
    code, output = run_gate("Found 1382 errors in 80 files (checked 900 source files)\n", limit=1381)
    assert code == 1
    assert "mypy errors: 1382 (baseline 1381)" in output.out


def test_fake_lower_summary_passes_and_requests_baseline_reduction(run_gate):
    code, output = run_gate("Found 1381 errors in 80 files (checked 900 source files)\n")
    assert code == 0
    assert "mypy errors: 1381 (baseline 1382)" in output.out
    assert "Lower the baseline: write 1381 to scripts/ci/mypy_baseline.txt" in output.out


def test_success_counts_as_zero(run_gate):
    code, output = run_gate("Success: no issues found in 900 source files\n", returncode=0)
    assert code == 0
    assert "mypy errors: 0 (baseline 1382)" in output.out


def test_singular_summary(run_gate):
    code, output = run_gate("Found 1 error in 1 file (checked 1 source file)\n", limit=1)
    assert code == 0
    assert "mypy errors: 1 (baseline 1)" in output.out


@pytest.mark.parametrize("output,returncode", [
    ("Traceback: crashed\n", 1),
    ("No summary\n", 0),
    ("Found 1 error in 1 file (checked 1 source file)\n", 2),
    ("Found 1 error in 1 file\ninternal error\n", 1),
])
def test_crash_preserves_output_and_exits_two(run_gate, output, returncode):
    code, captured = run_gate(output, returncode)
    assert code == 2
    assert output in captured.err
    assert captured.out == ""
