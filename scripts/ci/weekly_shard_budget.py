"""Report weekly shard wall time and test ids without recorded durations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# pytest-split 0.11 drops a setup or teardown phase longer than this from a stored price.
SETUP_TEARDOWN_LIMIT_S = 600


def durations_from_log(path: Path) -> dict[str, float]:
    """pytest-split's {nodeid: seconds} rebuilt from conftest's RFX_DURATIONS_LOG lines."""
    durations: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            nodeid, when, seconds = json.loads(line)
        except ValueError:
            continue  # the last line of an aborted session may be cut
        if seconds < 0 or (when in ("setup", "teardown") and seconds > SETUP_TEARDOWN_LIMIT_S):
            continue
        durations[nodeid] = durations.get(nodeid, 0.0) + seconds
    return durations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--limit-minutes", type=float, required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--recorded", type=Path, required=True)
    parser.add_argument(
        "--prune",
        action="store_true",
        help="rewrite --measured keeping only entries that differ from --recorded",
    )
    parser.add_argument(
        "--log",
        type=Path,
        help="conftest's RFX_DURATIONS_LOG file; read when pytest stored no durations",
    )
    args = parser.parse_args()

    elapsed = time.time() - args.start
    limit = args.limit_minutes * 60
    print(
        f"shard wall time: {elapsed / 60:.1f} min of a {args.limit_minutes:g} min "
        f"limit ({elapsed / limit * 100:.1f} %)"
    )
    recorded = json.loads(args.recorded.read_text(encoding="utf-8"))
    ran = None
    try:
        measured = json.loads(args.measured.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print("measured durations file missing")
    except json.JSONDecodeError:
        # pytest was stopped while writing it; the time verdict below still holds.
        print("measured durations file unreadable")
    else:
        # The measured file starts as a copy of the recorded one (pytest-split
        # reads and writes one path); what this shard ran is what changed.
        ran = {k: v for k, v in measured.items() if recorded.get(k) != v}
        if not ran and measured == recorded:
            # pytest-split stores only at session end: the session crashed, was
            # killed or cancelled, or ran nothing.
            print("pytest stored no durations: the session did not reach its end")
            ran = None
    if ran is None and args.log and args.log.exists():
        ran = durations_from_log(args.log)
        print(f"read {len(ran)} tests from the per-test log {args.log}")
    if ran is None:
        print("tests this shard measured: unknown (no per-test log either)")
        unpriced = "unknown"
    else:
        print(f"tests this shard measured: {len(ran)}")
        unpriced = len(ran.keys() - recorded.keys())
        if args.prune:
            args.measured.write_text(
                json.dumps(ran, sort_keys=True, indent=4), encoding="utf-8"
            )
    print(f"tests this shard ran with no recorded duration: {unpriced}")

    if elapsed > args.fraction * limit:
        print(
            f"FAIL: shard used more than {args.fraction * 100:g} % of its job limit "
            "— add a weekly shard or regenerate .test_durations (scripts/ci/DURATIONS.md)"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
