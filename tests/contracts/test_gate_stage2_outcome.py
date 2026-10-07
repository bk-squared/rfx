"""Lightweight selected-tests outcomes, including a real isolated xdist crash."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts/ci/gate_stage2_outcome.py"
SPEC = importlib.util.spec_from_file_location("gate_stage2_outcome", HELPER)
outcome = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(outcome)
NODE = "test_fake.py::test_crash"


def report(path, kinds):
    suite = ET.Element("testsuite", errors=str(kinds.count("crash")),
                       failures=str(kinds.count("failure")))
    for kind in kinds:
        case = ET.SubElement(suite, "testcase", classname="test_fake", name="test_crash")
        if kind == "crash":
            text = f"worker 'gw0' crashed while running '{NODE}'"
            ET.SubElement(case, "error", message=f'failed on setup with "{text}"').text = text
        elif kind == "failure":
            ET.SubElement(case, "failure", message="assert False").text = "assert False"
    ET.ElementTree(suite).write(path)
    return path


def test_crashes_only(tmp_path):
    assert outcome.classify(1, report(tmp_path / "junit.xml", ["crash", "crash"]),
                            ["test_fake.py"]) == ("retry", ["test_fake.py"], [NODE])


def test_mixed_failures_do_not_retry(tmp_path):
    assert outcome.classify(1, report(tmp_path / "junit.xml", ["crash", "failure"]),
                            ["test_fake.py"])[0] == "fail"


def test_ordinary_failure_does_not_retry(tmp_path):
    assert outcome.classify(1, report(tmp_path / "junit.xml", ["failure"]),
                            ["test_fake.py"])[0] == "fail"


@pytest.mark.parametrize("status", [0, 5])
def test_pass_and_no_collection(tmp_path, status):
    assert outcome.classify(status, report(tmp_path / "junit.xml", ["pass"]), []) == ("pass", [], [])


@pytest.mark.parametrize("status", [1, 2, 3, 4, 137])
def test_missing_report_fails_closed(tmp_path, status):
    assert outcome.classify(status, tmp_path / "missing", ["test_fake.py"])[0] == "fail"


def test_unknown_file_and_invalid_xml_fail_closed(tmp_path):
    path = report(tmp_path / "junit.xml", ["crash"])
    assert outcome.classify(1, path, ["other.py"])[0] == "fail"
    path.write_text("<broken")
    assert outcome.classify(1, path, ["test_fake.py"])[0] == "fail"


def test_real_xdist_crash(tmp_path):
    pytest.importorskip("xdist")
    (tmp_path / "test_fake.py").write_text("import os\ndef test_crash():\n    os._exit(7)\n")
    env = dict(os.environ, PYTEST_ADDOPTS="", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    args = ["-p", "xdist.plugin", "-n", "2", "--max-worker-restart=0", "--dist", "loadfile",
            "-o", "addopts=", "-p", "no:cacheprovider", "--junitxml=result.xml"]
    result = subprocess.run([sys.executable, "-c", f"import pytest; raise SystemExit(pytest.main({args!r}))"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1, result.stdout + result.stderr
    assert outcome.classify(result.returncode, tmp_path / "result.xml", ["test_fake.py"]) == (
        "retry", ["test_fake.py"], [NODE])
    (tmp_path / "test_fake.py").write_text(
        'import os\ndef test_serial():\n    assert "PYTEST_XDIST_WORKER" not in os.environ\n')
    env["PYTEST_ADDOPTS"] = "-n 14 --dist loadfile --max-worker-restart=0"
    args = ["-p", "xdist.plugin", "-n", "0", "-o", "addopts=", "-p", "no:cacheprovider", "test_fake.py"]
    serial = subprocess.run([sys.executable, "-c", f"import pytest; raise SystemExit(pytest.main({args!r}))"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert serial.returncode == 0, serial.stdout + serial.stderr
    assert "1 passed" in serial.stdout


def run_fragment(tmp_path, workers, first=0, retry=0, kinds=()):
    """Execute only the extracted selected-session shell, with pytest replaced."""
    stage = (ROOT / "scripts/ci/local.sh").read_text().split("selected_tests=()", 1)[1]
    stage = "selected_tests=()" + stage.split('\necho\necho "all ', 1)[0]
    (tmp_path / "files").write_text("test_fake.py\nother.py\n")
    (tmp_path / "summary").write_text("selection summary\n")
    report(tmp_path / "fixture.xml", kinds)
    fake = tmp_path / "fake.py"
    fake.write_text('''import json, os, pathlib, shutil, sys
p = pathlib.Path(os.environ["SESSION_DIR"])
log = p / "calls"
calls = json.loads(log.read_text()) if log.exists() else []
calls.append(sys.argv[1:])
log.write_text(json.dumps(calls))
if len(calls) == 1:
    shutil.copyfile(p / "fixture.xml", p / "selected.xml")
sys.exit(int(os.environ["FIRST" if len(calls) == 1 else "RETRY"]))
''')
    env = dict(os.environ, SESSION_DIR=str(tmp_path), REAL_PYTHON=sys.executable,
               FIRST=str(first), RETRY=str(retry), PYTEST_ADDOPTS="-n 14 --dist loadfile")
    env.pop("RFX_GATE_STAGE2_WORKERS", None)
    if workers is not None:
        env["RFX_GATE_STAGE2_WORKERS"] = workers
    prefix = '''set -uo pipefail
selection_dir="$SESSION_DIR"
fail() { echo FAILED; exit 1; }
python_dispatch() {
  if [ "$1" = "-m" ]; then
    "$REAL_PYTHON" "$SESSION_DIR/fake.py" "$@"
  else
    "$REAL_PYTHON" "$@"
  fi
}
PYTHON=python_dispatch
'''
    result = subprocess.run(["/bin/bash", "-uc", prefix + stage], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=10)
    calls = json.loads((tmp_path / "calls").read_text()) if (tmp_path / "calls").exists() else []
    return result, calls


@pytest.mark.parametrize("workers,expected", [(None, []), ("", []), ("8", ["-n", "8", "--dist", "loadfile"])])
def test_worker_override(tmp_path, workers, expected):
    result, calls = run_fragment(tmp_path, workers)
    assert result.returncode == 0, result.stderr
    args = calls[0]
    start = args.index("--strict-markers") + 1
    assert args[start:-1] == expected


@pytest.mark.parametrize("workers", ["abc", "0", "-1", "1.5"])
def test_invalid_workers(tmp_path, workers):
    result, calls = run_fragment(tmp_path, workers)
    assert result.returncode == 1
    assert "must be a positive integer" in result.stderr
    assert calls == []


@pytest.mark.parametrize("retry", [0, 1, 5, 139])
def test_serial_retry_once_and_summary(tmp_path, retry):
    result, calls = run_fragment(tmp_path, "8", first=1, retry=retry, kinds=["crash"])
    assert result.returncode == (0 if retry == 0 else 1), result.stderr
    assert len(calls) == 2
    assert calls[1][2:4] == ["test_fake.py", "-q"]
    assert calls[1][-2:] == ["-n", "0"]
    assert calls[0][calls[0].index("-m", 2) + 1] == calls[1][calls[1].index("-m", 2) + 1]
    assert NODE in result.stdout
    assert f"serial retry exit status: {retry}" in result.stdout


@pytest.mark.parametrize("kinds", [["crash", "failure"], ["failure"]])
def test_shell_does_not_retry_real_failures(tmp_path, kinds):
    result, calls = run_fragment(tmp_path, None, first=1, kinds=kinds)
    assert result.returncode == 1
    assert len(calls) == 1


def test_shell_exit_five_is_unchanged(tmp_path):
    result, calls = run_fragment(tmp_path, None, first=5)
    assert result.returncode == 0
    assert len(calls) == 1
    assert "nothing to run (all selected tests deselected or no tests collected)" in result.stdout
