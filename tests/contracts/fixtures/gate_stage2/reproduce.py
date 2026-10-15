"""Manual fixture recipe; never invoked by the contract tests.

Measured with Python 3.11.2, pytest 9.1.1, pytest-xdist 3.8.0 on macOS.
Create scratch with `mktemp -d "$PWD/.gate-reports.XXXXXX"`, then run this
script with that directory as its only argument using the gate interpreter.
It writes only into scratch; inspect the XML before replacing frozen fixtures.
Core dumps are disabled. Every worker crash is intentional SIGSEGV.

collection_skip.xml uses the unchanged test_mesh_import.py from 61f40625
(module-level importorskip at line 11), alongside test_pass.py. Trimesh was
absent in both the measurement venv and the gate's recorded pip-freeze.
Frozen XML removes hostname/timestamp and the scratch prefix from skip text;
all cases, outcomes, messages, counters and timings are otherwise preserved.
"""
from pathlib import Path
import os
import resource
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[4]


def main():
    scratch = Path(sys.argv[1]).resolve(strict=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(scratch),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="",
               PYTHONPATH=str(ROOT / "scripts/ci"))
    for kind in ("call", "setup", "teardown", "queued", "mixed", "assertion", "collection_skip"):
        directory = scratch / kind
        directory.mkdir()
        (directory / "pytest.ini").write_text("[pytest]\n")
        if kind == "assertion":
            (directory / "test_assertion.py").write_text(
                "def test_assertion():\n    assert False\n")
        elif kind == "collection_skip":
            source = subprocess.check_output(
                ["git", "show", "61f40625:tests/unit/geometry/test_mesh_import.py"], cwd=ROOT)
            target = directory / "tests/unit/geometry/test_mesh_import.py"
            target.parent.mkdir(parents=True)
            target.write_bytes(source)
            (directory / "test_pass.py").write_text("def test_pass(): pass\n")
        else:
            (directory / "test_a.py").write_text(
                'import os, signal, time\nfrom pathlib import Path\nimport pytest\n'
                'def die():\n    if not Path("crashed").exists():\n'
                '        Path("crashed").touch()\n'
                '        os.kill(os.getpid(), signal.SIGSEGV)\n'
                '@pytest.fixture\ndef fixture():\n'
                + ('    die()\n' if kind == "setup" else '') + '    yield\n'
                + ('    die()\n' if kind == "teardown" else '')
                + 'def test_before(): pass\ndef test_crash(fixture): '
                + ('pass' if kind in ("setup", "teardown") else 'die()')
                + '\ndef test_after(): pass\n')
            for letter in ("bcd" if kind in ("queued", "mixed") else "bc"):
                (directory / f"test_{letter}.py").write_text(
                    'import time\ndef test_one(): time.sleep(0.1)\n'
                    'def test_two(): time.sleep(0.1)\n')
            if kind == "mixed":
                (directory / "test_b.py").write_text(
                    'def test_one():\n    assert False\ndef test_two(): pass\n')
            if kind in ("call", "setup", "teardown"):
                (directory / "test_optional.py").write_text(
                    'import pytest\npytest.skip("optional dependency unavailable", allow_module_level=True)\n')
        command = [sys.executable, "-m", "pytest", "-c", "pytest.ini", "-vv",
                   "-p", "xdist.plugin", "-p", "rfx_gate_collection",
                   "-p", "no:cacheprovider", "--junitxml=raw.xml"]
        if kind not in ("assertion", "collection_skip"):
            command += ["-n", "2", "--dist", "loadfile", "--max-worker-restart=0"]
        run = subprocess.run(command, cwd=directory, env=dict(
            env, RFX_GATE_COLLECTION_DIR=str(directory / "collection")),
            capture_output=True, text=True, timeout=40)
        print(kind, "exit", run.returncode, run.stdout.splitlines()[-1])
        tree = ET.parse(directory / "raw.xml")
        for suite in tree.iter("testsuite"):
            for key in ("timestamp", "hostname"):
                suite.attrib.pop(key, None)
        for skipped in tree.iter("skipped"):
            skipped.text = (skipped.text or "").replace(str(directory) + "/", "")
        ET.indent(tree)
        tree.write(scratch / f"{kind}.xml", encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
