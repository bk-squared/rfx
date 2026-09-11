"""Build-artifact smoke: isolated venv, packaged Studio launch, CPU golden run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request
import venv


def _worker_control(python: Path, root: Path, environment: dict[str, str]) -> dict:
    """Verify actual interpreter ownership in the installed platform/venv."""
    probe = r'''
import json, os, subprocess, sys, time
from pathlib import Path
from rfx.experiments._worker_child import _bind_parent, _python_command
from rfx.experiments.worker import _wait_for_child
assert sys.prefix != sys.base_prefix, 'control must run inside the isolated venv'
ready = Path(sys.argv[1])
child_code = r"""
import ctypes, json, os, sys, time
from pathlib import Path
from rfx.experiments._worker_child import _bind_parent
_bind_parent(int(sys.argv[2]))
ready = Path(sys.argv[1])
temporary = ready.with_suffix('.tmp')
temporary.write_text(json.dumps({'pid': os.getpid(), 'ppid': os.getppid(), 'prefix': sys.prefix}))
temporary.replace(ready)
if sys.platform == 'win32':
    library = ctypes.PyDLL('kernel32')
    library.Sleep.argtypes = [ctypes.c_ulong]
    library.Sleep(30000)
else:
    time.sleep(30)
"""
command, child_env = _python_command([sys.executable, '-c', child_code, str(ready), str(os.getpid())])
child = subprocess.Popen(command, env=child_env)
try:
    startup_deadline = time.monotonic() + 15
    while not ready.exists():
        assert child.poll() is None, 'worker interpreter exited before the control'
        assert time.monotonic() < startup_deadline, 'worker interpreter did not start'
        time.sleep(0.02)
    observed = json.loads(ready.read_text())
    assert observed['pid'] == child.pid, (observed, child.pid)
    assert observed['ppid'] == os.getpid(), (observed, os.getpid())
    assert Path(observed['prefix']).resolve() == Path(sys.prefix).resolve(), observed
    before = time.monotonic()
    outcome = _wait_for_child(child, deadline=before + 0.5, cancelled=lambda: False)
    assert outcome == 'timeout' and child.returncode is not None
    assert time.monotonic() - before < 10
    print(json.dumps({'pid_owned': True, 'venv_preserved': True, 'deadline_enforced': True}))
finally:
    if child.poll() is None:
        child.kill()
    child.wait()
'''
    result = subprocess.run(
        [str(python), "-I", "-c", probe, str(root / "worker-control-ready.json")],
        capture_output=True,
        text=True,
        cwd=root,
        env=environment,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"worker process control failed\n{result.stdout}\n{result.stderr}"
        )
    return json.loads(result.stdout)


def _worker_diagnostics(workspace: Path) -> str:
    lines = []
    for path in sorted((workspace / "runs").glob("*/worker.stderr.log")):
        lines.append(
            f"--- {path.name} ({path.parent.name}) ---\n{path.read_text(encoding='utf-8', errors='replace')[-16000:]}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel")
    parser.add_argument(
        "--fixture", default="tests/fixtures/experiments/patch_antenna_v2.json"
    )
    args = parser.parse_args()
    wheel = Path(args.wheel).expanduser().resolve()
    fixture = Path(args.fixture).expanduser().resolve()
    if not wheel.is_file() or not fixture.is_file():
        parser.error("wheel and fixture must exist")
    with tempfile.TemporaryDirectory(prefix="rfx-clean-smoke-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        python = (
            environment / "Scripts" / "python.exe"
            if sys.platform == "win32"
            else environment / "bin" / "python"
        )
        clean_environment = os.environ.copy()
        clean_environment.pop("PYTHONPATH", None)
        clean_environment["PYTHONNOUSERSITE"] = "1"
        subprocess.run(
            [str(python), "-m", "pip", "install", f"{wheel}[studio]"],
            check=True,
            cwd=root,
            env=clean_environment,
        )
        asset_check = subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                (
                    "from pathlib import Path; import rfx.studio; "
                    "p=Path(rfx.studio.__file__).parent/'static'/'index.html'; "
                    "assert p.is_file(), p; print(p)"
                ),
            ],
            check=True,
            capture_output=True,
            cwd=root,
            env=clean_environment,
            text=True,
        )
        packaged_asset = Path(asset_check.stdout.strip()).resolve()
        if environment.resolve() not in packaged_asset.parents:
            raise RuntimeError(
                f"Studio imported outside isolated environment: {packaged_asset}"
            )
        worker_control = _worker_control(python, root, clean_environment)
        port = 18765
        server = subprocess.Popen(
            [
                str(python),
                "-I",
                "-m",
                "rfx.cli",
                "studio",
                "--no-browser",
                "--port",
                str(port),
                "--workspace",
                str(root / "studio-workspace"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=root,
            env=clean_environment,
            text=True,
        )
        try:
            deadline = time.monotonic() + 30
            health = None
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    stdout, stderr = server.communicate()
                    raise RuntimeError(f"Studio exited early\n{stdout}\n{stderr}")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/api/health", timeout=1
                    ) as response:
                        health = json.load(response)
                    break
                except Exception:
                    time.sleep(0.1)
            if health != {"status": "ok", "mode": "local", "backend": "cpu"}:
                raise RuntimeError(f"Studio health failed: {health}")
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
        golden = subprocess.run(
            [
                str(python),
                "-I",
                "-m",
                "rfx.cli",
                "experiment",
                "run",
                str(fixture),
                "--workspace",
                str(root / "golden-workspace"),
            ],
            check=False,
            capture_output=True,
            cwd=root,
            env=clean_environment,
            text=True,
            timeout=90,
        )
        if golden.returncode != 0:
            diagnostics = _worker_diagnostics(root / "golden-workspace")
            raise RuntimeError(
                f"golden smoke failed\n{golden.stdout}\n{golden.stderr}\n{diagnostics}"
            )
        result = json.loads(golden.stdout)
        if result["state"] != "succeeded":
            raise RuntimeError(f"golden smoke state: {result['state']}")
        print(
            json.dumps(
                {
                    "wheel": str(wheel),
                    "packaged_asset": str(packaged_asset),
                    "studio_health": health,
                    "golden_run_id": result["id"],
                    "golden_state": result["state"],
                    "worker_control": worker_control,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
