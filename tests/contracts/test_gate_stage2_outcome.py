"""Fail-closed evidence checks and isolated end-to-end selected-stage crashes."""
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


def collection(tmp_path, nodes=None):
    directory = tmp_path / "collection"
    directory.mkdir(exist_ok=True)
    (directory / "worker.json").write_text(json.dumps([NODE] if nodes is None else nodes))
    return directory


def report(tmp_path, kinds):
    path = tmp_path / "junit.xml"
    suite = ET.Element("testsuite", errors=str(kinds.count("crash")),
                       failures=str(kinds.count("failure")))
    for kind in kinds:
        case = ET.SubElement(suite, "testcase", classname="test_fake", name="test_crash")
        if kind == "crash":
            text = f"worker 'gw0' crashed while running '{NODE}'"
            ET.SubElement(case, "error", message=f'failed on setup with "{text}"').text = text
        elif kind == "failure":
            ET.SubElement(case, "failure", message="assert False").text = "assert False"
        elif kind in ("skip", "xfail"):
            ET.SubElement(case, "skipped", type=kind)
    ET.ElementTree(suite).write(path)
    return path


@pytest.mark.parametrize("kind", ["pass", "skip", "xfail"])
def test_pass(tmp_path, kind):
    assert outcome.classify(0, report(tmp_path, [kind]), collection(tmp_path))["outcome"] == "PASS"


def test_exit_zero_missing_results(tmp_path):
    assert outcome.classify(0, report(tmp_path, ["pass"]),
                            collection(tmp_path, [NODE, "other.py::test_unrun"]))["outcome"] == "FAIL"


@pytest.mark.parametrize("empty", [True, False])
def test_empty_collection(tmp_path, empty):
    directory = collection(tmp_path, []) if empty else tmp_path / "absent"
    assert outcome.classify(0, report(tmp_path, ["pass"]), directory)["outcome"] == "FAIL"


def test_empty_collection_folder(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()
    assert outcome.classify(0, report(tmp_path, ["pass"]), directory)["outcome"] == "FAIL"


def test_crashes_and_missing_files(tmp_path):
    result = outcome.classify(1, report(tmp_path, ["crash"]),
                              collection(tmp_path, [NODE, "other.py::test_unrun"]))
    assert result["outcome"] == "RETRY"
    assert result["files"] == ["other.py", "test_fake.py"]
    assert result["parallel"]
    assert result["nodes"] == [NODE]


@pytest.mark.parametrize("kinds", [["crash", "failure"], ["failure"]])
def test_ordinary_and_mixed_failures(tmp_path, kinds):
    assert outcome.classify(1, report(tmp_path, kinds), collection(tmp_path))["outcome"] == "FAIL"


@pytest.mark.parametrize("status", [0, 1, 2, 3, 4, 137])
def test_missing_report(tmp_path, status):
    assert outcome.classify(status, tmp_path / "missing", collection(tmp_path))["outcome"] == "FAIL"


@pytest.mark.parametrize("status", [2, 3, 4, 137, -9])
def test_abnormal_status_with_valid_crash_evidence(tmp_path, status):
    assert outcome.classify(status, report(tmp_path, ["crash"]), collection(tmp_path))["outcome"] == "FAIL"


def test_declared_problem_count_mismatch(tmp_path):
    path = report(tmp_path, ["crash"])
    tree = ET.parse(path)
    tree.getroot().set("errors", "2")
    tree.write(path)
    assert outcome.classify(1, path, collection(tmp_path))["outcome"] == "FAIL"


@pytest.mark.parametrize("spoof", ["tag", "message", "node"])
def test_spoof_is_not_crash(tmp_path, spoof):
    path = report(tmp_path, ["crash"])
    tree = ET.parse(path)
    problem = tree.find(".//error")
    if spoof == "tag":
        problem.tag = "failure"
    elif spoof == "message":
        problem.set("message", "ordinary failure")
    else:
        problem.text = problem.text.replace("test_fake.py", "unknown.py")
        problem.set("message", f'failed on setup with "{problem.text}"')
    tree.write(path)
    assert outcome.classify(1, path, collection(tmp_path))["outcome"] == "FAIL"


def test_invalid_xml(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text("<broken")
    assert outcome.classify(1, path, collection(tmp_path))["outcome"] == "FAIL"


def test_nothing(tmp_path):
    assert outcome.classify(5, tmp_path / "missing", tmp_path / "absent")["outcome"] == "NOTHING"


def run_fragment(tmp_path, workers="2", fake=False, first=0, retry=0, kinds=None):
    """Execute verbatim local.sh selected-stage fragment, never the entire gate."""
    stage = (ROOT / "scripts/ci/local.sh").read_text().split("selected_tests=()", 1)[1]
    stage = "selected_tests=()" + stage.split('\necho\necho "all ', 1)[0]
    (tmp_path / "summary").write_text("selection summary\n")
    scripts = tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "ci").symlink_to(ROOT / "scripts/ci", target_is_directory=True)
    env = dict(os.environ, SESSION_DIR=str(tmp_path), REAL_PYTHON=sys.executable,
               PYTEST_ADDOPTS="-n 2 --dist loadfile --max-worker-restart=0 -p xdist.plugin",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1",
               PYTEST_DEBUG_TEMPROOT=str(tmp_path),
               FIRST=str(first), RETRY=str(retry))
    env.pop("RFX_GATE_STAGE2_WORKERS", None)
    if workers is not None:
        env["RFX_GATE_STAGE2_WORKERS"] = workers
    prefix = '''set -uo pipefail
selection_dir="$SESSION_DIR"
fail() { echo FAILED; exit 1; }
PYTHON="$REAL_PYTHON"
'''
    if fake:
        (tmp_path / "files").write_text("test_fake.py\n")
        (tmp_path / "fake.py").write_text('''import json, os, pathlib, sys
p = pathlib.Path(os.environ["SESSION_DIR"])
log = p / "calls"
calls = json.loads(log.read_text()) if log.exists() else []
calls.append(sys.argv[1:])
log.write_text(json.dumps(calls))
status = int(os.environ["FIRST" if len(calls) == 1 else "RETRY"])
d = pathlib.Path(os.environ["RFX_GATE_COLLECTION_DIR"])
d.mkdir()
(d / "worker.json").write_text(json.dumps(["test_fake.py::test_crash"]))
xml = '<testsuite failures="0" errors="0"><testcase classname="test_fake" name="test_crash"/></testsuite>'
if status == 1:
    xml = (p / "crash.xml").read_text()
pathlib.Path(next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--junitxml="))).write_text(xml)
sys.exit(status)
''')
        report(tmp_path, ["crash"] if kinds is None else kinds).rename(tmp_path / "crash.xml")
        prefix += '''python_dispatch() {
  if [ "$1" = "-m" ]; then
    "$REAL_PYTHON" "$SESSION_DIR/fake.py" "$@"
  else
    "$REAL_PYTHON" "$@"
  fi
}
PYTHON=python_dispatch
'''
    result = subprocess.run(["/bin/bash", "-uc", prefix + stage], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=45)
    (tmp_path / "stage.log").write_text(result.stdout + result.stderr)
    return result


@pytest.mark.parametrize("workers,expected", [(None, []), ("", []), ("8", ["-n", "8", "--dist", "loadfile"])])
def test_worker_override(tmp_path, workers, expected):
    result = run_fragment(tmp_path, workers, fake=True)
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads((tmp_path / "calls").read_text())[0]
    assert args[3:args.index("-q")] == expected


@pytest.mark.parametrize("workers", ["abc", "0", "08", "01", "-1", "1.5"])
def test_invalid_workers(tmp_path, workers):
    result = run_fragment(tmp_path, workers, fake=True)
    assert result.returncode == 1
    assert "must be a positive integer" in result.stderr
    assert not (tmp_path / "calls").exists()


@pytest.mark.parametrize("retry", [0, 1, 5, 139])
def test_serial_retry_once(tmp_path, retry):
    result = run_fragment(tmp_path, fake=True, first=1, retry=retry)
    assert result.returncode == (0 if retry == 0 else 1), result.stdout + result.stderr
    calls = json.loads((tmp_path / "calls").read_text())
    assert len(calls) == 2
    assert calls[1][3:5] == ["-n", "0"]
    assert NODE in result.stdout
    assert f"retry exit status: {retry}" in result.stdout


def test_shell_nothing(tmp_path):
    result = run_fragment(tmp_path, fake=True, first=5)
    assert result.returncode == 0
    assert "nothing to run (all selected tests deselected or no tests collected)" in result.stdout


def seven_files(tmp_path, failing, persistent):
    # Reviewer run2.sh layout and timings; sentinel variant isolates lost coverage.
    crash = 'os._exit(7)' if persistent else 'time.sleep(0.1); Path("crashed").touch(); os._exit(7)'
    condition = '"PYTEST_XDIST_WORKER" in os.environ'
    if not persistent:
        condition += ' and not Path("crashed").exists()'
    (tmp_path / "test_a.py").write_text(
        'import os, time\nfrom pathlib import Path\n'
        'def test_a1(): time.sleep(0.3)\n'
        f'def test_crash():\n    if {condition}: {crash}\n'
        'def test_a3(): pass\n')
    for letter in "bcdef":
        (tmp_path / f"test_{letter}.py").write_text('import time\n' + ''.join(
            f'def test_{i}(): time.sleep(0.5)\n' for i in range(1, 4)))
    (tmp_path / "test_g.py").write_text(
        'def test_real_failure(): ' + ('assert 0, "REAL FAILURE never run"' if failing else 'pass') + '\n')
    (tmp_path / "files").write_text(''.join(f'test_{letter}.py\n' for letter in "abcdefg"))


@pytest.mark.parametrize("failing", [True, False], ids=["hidden_failure", "recovered"])
def test_real_xdist_unrun_coverage(tmp_path, failing):
    pytest.importorskip("xdist")
    seven_files(tmp_path, failing, persistent=False)
    result = run_fragment(tmp_path)
    assert result.returncode == (1 if failing else 0), result.stdout + result.stderr
    first = json.loads((tmp_path / "first-outcome.json").read_text())
    assert first["collected"] == 19
    assert first["reported"] < 19
    assert first["parallel"]
    retry = json.loads((tmp_path / "retry-outcome.json").read_text())
    assert retry["outcome"] == ("FAIL" if failing else "PASS")
    assert "test_g.py" in first["files"]
    cases = [c for name in ("selected.xml", "retry.xml")
             for c in ET.parse(tmp_path / name).iter("testcase")]
    if failing:
        assert any(c.get("classname") == "test_g" and c.find("failure") is not None for c in cases)
    else:
        successful = {(c.get("classname"), c.get("name")) for c in cases
                      if c.find("failure") is None and c.find("error") is None}
        records = list((tmp_path / "selected-collection").glob("*.json"))
        assert len(records) == 2  # Both workers wrote collection before the crash.
        nodes = json.loads(records[0].read_text())
        assert len(successful) == 19
        assert {outcome.junit_key(n) for n in nodes} == successful


@pytest.mark.parametrize("failing", [True, False])
def test_reviewer_verbatim_repeated_crash_fails(tmp_path, failing):
    pytest.importorskip("xdist")
    seven_files(tmp_path, failing, persistent=True)
    result = run_fragment(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    # Immediate os._exit can also race xdist scheduling into exit 3 (fail closed).
    retry = tmp_path / "retry-outcome.json"
    if retry.exists():
        assert json.loads(retry.read_text())["outcome"] != "PASS"
    else:
        assert json.loads((tmp_path / "first-outcome.json").read_text())["outcome"] == "FAIL"


@pytest.mark.parametrize("kinds", [["failure"], ["crash", "failure"]])
def test_shell_does_not_retry_real_failures(tmp_path, kinds):
    result = run_fragment(tmp_path, fake=True, first=1, kinds=kinds)
    assert result.returncode == 1
    assert len(json.loads((tmp_path / "calls").read_text())) == 1


@pytest.mark.parametrize("workers", ["0", "2"])
def test_real_collection_after_deselection(tmp_path, workers):
    (tmp_path / "test_sample.py").write_text(
        'import pytest\n'
        '@pytest.mark.skip(reason="skip")\ndef test_skip(): pass\n'
        '@pytest.mark.xfail(reason="expected")\ndef test_xfail(): assert False\n'
        '@pytest.mark.parametrize("value", ["a.b::c"])\n'
        'def test_parameter(value): pass\n'
        'def test_deselected(): assert False\n')
    directory = tmp_path / "collection"
    env = dict(os.environ, PYTEST_ADDOPTS="", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               PYTHONPATH=str(ROOT / "scripts/ci"), RFX_GATE_COLLECTION_DIR=str(directory),
               PYTEST_DEBUG_TEMPROOT=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "xdist.plugin", "-n", workers,
         "-p", "rfx_gate_collection", "-o", "addopts=", "-p", "no:cacheprovider",
         "-k", "not deselected", "--junitxml=junit.xml"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    records = list(directory.glob("*.json"))
    assert len(records) == (1 if workers == "0" else 2)
    assert all(len(json.loads(p.read_text())) == 3 for p in records)
    assert outcome.classify(0, tmp_path / "junit.xml", directory)["outcome"] == "PASS"
