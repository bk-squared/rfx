"""P4 JSON is produced by the cluster tool running the pytest wrapper
``tests/unit/preflight/test_zz_freeze_not_judged_p4.py`` on branch
``acceleration/2755-pr2-freeze``. The wrapper verifies that only tests differ
from the designated base and prints one JSON line between P4-FROZEN-BEGIN
and P4-FROZEN-END. Save that line as data/not_judged_p4_frozen.json and
commit it with the PR; never regenerate the baseline on the PR head.
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
