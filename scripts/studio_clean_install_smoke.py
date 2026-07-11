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
            raise RuntimeError(f"golden smoke failed\n{golden.stdout}\n{golden.stderr}")
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
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
