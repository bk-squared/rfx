"""Print the P4 baseline through the cluster tool's pytest entry point."""
import json
from pathlib import Path
import subprocess

from tests.unit.preflight._not_judged_scenes import SCENES, build_scene, issue_rows


BASE = "084b5032b7d014f12b65b4b947cf61a840970bf4"


def test_freeze_not_judged_p4(capsys):
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE, "HEAD"],
        cwd=Path(__file__).resolve().parents[3], text=True,
    ).splitlines()
    assert all(name.startswith("tests/") for name in changed), changed
    payload = {
        "base_commit": BASE,
        "scenes": {name: issue_rows(build_scene(name).preflight()) for name in SCENES},
    }
    # The cluster invokes pytest without -s; keep the payload in its stdout.
    with capsys.disabled():
        print("P4-FROZEN-BEGIN", flush=True)
        print(json.dumps(payload, sort_keys=True), flush=True)
        print("P4-FROZEN-END", flush=True)
