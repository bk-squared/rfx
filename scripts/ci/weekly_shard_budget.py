"""Report weekly shard wall time and test ids without recorded durations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


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
    args = parser.parse_args()

    elapsed = time.time() - args.start
    limit = args.limit_minutes * 60
    print(
        f"shard wall time: {elapsed / 60:.1f} min of a {args.limit_minutes:g} min "
        f"limit ({elapsed / limit * 100:.1f} %)"
    )
    try:
        measured = json.loads(args.measured.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print("measured durations file missing")
        unpriced = "unknown"
    else:
        recorded = json.loads(args.recorded.read_text(encoding="utf-8"))
        unpriced = len(measured.keys() - recorded.keys())
        # The measured file starts as a copy of the recorded one (pytest-split
        # reads and writes one path); what this shard ran is what changed.
        ran = {k: v for k, v in measured.items() if recorded.get(k) != v}
        print(f"tests this shard measured: {len(ran)}")
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
