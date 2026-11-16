"""Measure one mesh/record pair; invoke in six independent CPU jobs.

Outputs are campaign records and belong outside the public source tree.
Run from the checkout with PYTHONPATH set to its root.
"""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys

from tests._x64_compat import enable_x64
from tests.oracle._sheet_rs_pole_scene import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h-mm", type=float, choices=(1.5, 1., .75), required=True)
    parser.add_argument("--record-ns", type=float, choices=(12., 24.), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    with enable_x64():
        row = measure(args.h_mm * 1e-3, args.record_ns * 1e-9)
    row["commit"] = sha
    row["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    args.output.write_text(json.dumps(row, sort_keys=True, indent=2) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)
    if not (float("-inf") < row["s"] < float("inf")) or row["s"] == 0:
        raise RuntimeError(f"sheet path does not trace Rs: s={row['s']}")


if __name__ == "__main__":
    main()
