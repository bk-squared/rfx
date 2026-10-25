"""local.sh runs and reports every stage-1 step; the selected tests run only after all pass."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Stands in for $PYTHON: each check fails when its FAIL_* variable is set.
FAKE_PYTHON = r'''#!/bin/bash
case "$*" in
  *check_changelog_fragment.py*) exit "${FAIL_CHANGELOG:-0}" ;;
  *check_file_size_ratchet.py*) exit "${FAIL_RATCHET:-0}" ;;
  *"-m pytest tests/contracts"*) printf '%s\n' "$*" > "$LOG_DIR/contracts-args"; exit "${FAIL_CONTRACTS:-0}" ;;
  *select_gate_tests.py*)
    touch "$LOG_DIR/selected"
    while [ "$#" -gt 0 ]; do [ "$1" = --summary ] && : > "$2"; shift; done
    exit 0 ;;
  -) cat > /dev/null; exit 0 ;;
  *) exit 0 ;;
esac
'''


def run_gate(tmp_path, **fail):
    ci = tmp_path / "repo" / "scripts" / "ci"
    ci.mkdir(parents=True)
    shutil.copy(ROOT / "scripts/ci/local.sh", ci / "local.sh")
    (ci / "lint.sh").write_text('exit "${FAIL_RUFF:-0}"\n')
    (ci / "docs_hygiene.sh").write_text("exit 0\n")
    python = tmp_path / "python"
    python.write_text(FAKE_PYTHON)
    python.chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FAIL_", "RFX_GATE_"))}
    env.update(PYTHON=str(python), LOG_DIR=str(logs), **{f"FAIL_{k.upper()}": "1" for k in fail})
    # /bin/bash is 3.2 on macOS: empty arrays under set -u must still work.
    result = subprocess.run(["/bin/bash", str(ci / "local.sh")], env=env, cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    return result, logs


def test_every_failing_stage1_step_is_reported_and_selected_tests_do_not_run(tmp_path):
    result, logs = run_gate(tmp_path, ruff=1, changelog=1, ratchet=1, contracts=1)
    assert result.returncode == 1
    for step in ("ruff", "changelog-fragment", "file-size-ratchet", "contract-tests"):
        assert f"FAILED: {step} (the remaining steps still run)" in result.stdout
    assert "=== [8/10] api-reference ===" in result.stdout
    assert result.stdout.rstrip().endswith(
        "FAILED steps: ruff changelog-fragment file-size-ratchet contract-tests")
    assert "selected-tests === not run: steps 1-9 must pass first" in result.stdout
    assert not (logs / "selected").exists()
    assert "all 10 steps passed" not in result.stdout


def test_contracts_run_after_an_early_failure_without_stopping_at_the_first_test(tmp_path):
    result, logs = run_gate(tmp_path, changelog=1)
    args = (logs / "contracts-args").read_text().split()
    assert "-x" not in args and "--exitfirst" not in args
    assert result.stdout.rstrip().endswith("FAILED steps: changelog-fragment")


def test_all_stage1_steps_pass_then_selected_tests_run(tmp_path):
    result, logs = run_gate(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (logs / "selected").exists()
    assert "nothing selected" in result.stdout
    assert result.stdout.rstrip().endswith("all 10 steps passed")
    assert "FAILED" not in result.stdout
