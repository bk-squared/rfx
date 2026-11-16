"""Run on main on the cluster: python -m tests.unit.preflight._freeze_not_judged_p4.

Copy this script and _not_judged_scenes.py to main before running. Commit the
resulting JSON with the PR after freezing; never regenerate it on the PR head.
"""
from pathlib import Path
import json
import subprocess

from tests.unit.preflight._not_judged_scenes import SCENES, build_scene, issue_rows


BASE = "084b5032b7d014f12b65b4b947cf61a840970bf4"
DESTINATION = Path(__file__).with_name("data") / "not_judged_p4_frozen.json"


def main():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != BASE:
        raise SystemExit(f"Freeze on the designated main commit {BASE}; HEAD is {head}")
    payload = {
        "base_commit": head,
        "scenes": {name: issue_rows(build_scene(name).preflight()) for name in SCENES},
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
