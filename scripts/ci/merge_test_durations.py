#!/usr/bin/env python3
"""Merge per-shard pytest-split duration files into one .test_durations.

Usage: python scripts/ci/merge_test_durations.py OUT_PATH SHARD_JSON [SHARD_JSON ...]

Every input is a {nodeid: seconds} map written by ``pytest --store-durations``.
The output holds exactly the union of the measured nodeids (later inputs win on
duplicates), sorted, so the committed file lists what the lanes actually run
and nothing that no longer exists.
"""
import json
import sys


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    out, inputs = argv[1], argv[2:]
    merged: dict[str, float] = {}
    for path in inputs:
        with open(path) as f:
            part = json.load(f)
        if not isinstance(part, dict):
            raise SystemExit(f"{path}: not a nodeid->seconds map")
        merged.update({k: float(v) for k, v in part.items()})
    with open(out, "w") as f:
        json.dump(dict(sorted(merged.items())), f, indent=4, sort_keys=True)
        f.write("\n")
    total = sum(merged.values())
    print(f"{out}: {len(merged)} tests, {total:.0f} s total, from {len(inputs)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
