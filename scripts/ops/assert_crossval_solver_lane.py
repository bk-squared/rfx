#!/usr/bin/env python
"""Decide whether a two-solver crossval lane run satisfies #717's closing condition.

#717 item 2 closes on "one green scheduled run that COLLECTS the four gpu-marked
crossval files". A pytest exit code cannot say that, and neither can a count:

  * with no solver installed, all 13 external legs `importorskip` away in 0.99 s TOTAL
    (measured on CPU at 541f703f) and pytest exits 0 with 18 "collected";
  * that green-with-skips outcome is the dishonest coverage claim the issue was filed
    about, and it is what every lane in this repo has produced so far.

So the gate here is, over the run's junit.xml:

  1. the collected node-id set equals the frozen set in
     scripts/ops/crossval_solver_lane_nodes.json, byte for byte;
  2. failures == 0 and errors == 0;
  3. skipped == 0 -- any skip, for any reason;
  4. every one of the 13 external legs reports a junit duration above the floor in
     that file (1.0 s), which is what turns "collected" into "executed".

`scripts/ops/summarize_junit.py` is the parsing idiom this copies, but it tolerates
skips and has no frozen list, so it cannot be the closing gate as written.

Exits nonzero unless every clause holds, and always writes the machine-readable
summary the vessl-jobs rule requires ("the return code, the counts, the failing ids,
and the pinned sha, in fixed filenames").
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "crossval_solver_lane_nodes.json"


def _node_id(case: ET.Element) -> str:
    """Rebuild the pytest node id from junit's classname/name pair."""
    classname = case.get("classname", "")
    name = case.get("name", "")
    if not classname:
        return name
    parts = classname.split(".")
    # junit classname is "tests.crossval.test_x" or "tests.crossval.test_x.TestY"
    for i, part in enumerate(parts):
        if part.startswith("test_") and (i + 1 == len(parts) or parts[i + 1][:1].isupper()):
            path = "/".join(parts[: i + 1]) + ".py"
            rest = parts[i + 1 :]
            return "::".join([path, *rest, name])
    return "::".join([classname, name])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("junit", nargs="+", help="one or more junit.xml files from the run")
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--sha", default="", help="the pinned rfx sha this run exported")
    ap.add_argument("--run-id", default="", help="the VESSL run id, recorded by the submitter")
    args = ap.parse_args()

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    expected = set(frozen["external"]) | set(frozen["rfx_only"])
    external = set(frozen["external"])
    floor = float(frozen["external_duration_floor_s"])

    seen: dict[str, dict[str, object]] = {}
    for path in args.junit:
        root = ET.parse(path).getroot()
        for case in root.iter("testcase"):
            nid = _node_id(case)
            outcome = "passed"
            for child in case:
                tag = child.tag.lower()
                if tag in {"failure", "error", "skipped"}:
                    outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
                    break
            seen[nid] = {"outcome": outcome, "time": float(case.get("time") or 0.0)}

    collected = set(seen)
    problems: list[str] = []

    missing = sorted(expected - collected)
    unexpected = sorted(collected - expected)
    if missing:
        problems.append(f"{len(missing)} frozen node id(s) did not appear: {missing}")
    if unexpected:
        problems.append(
            f"{len(unexpected)} node id(s) appeared that are not in the frozen set "
            f"(regenerate it deliberately, do not widen the gate): {unexpected}"
        )

    failed = sorted(n for n, r in seen.items() if r["outcome"] == "failed")
    errored = sorted(n for n, r in seen.items() if r["outcome"] == "error")
    skipped = sorted(n for n, r in seen.items() if r["outcome"] == "skipped")
    if failed:
        problems.append(f"{len(failed)} failed: {failed}")
    if errored:
        problems.append(f"{len(errored)} errored: {errored}")
    if skipped:
        problems.append(
            f"{len(skipped)} SKIPPED: {skipped}. A skip here means the leg did not run; "
            "that is the exact outcome this lane exists to stop reporting as coverage."
        )

    too_fast = sorted(
        n
        for n in external & collected
        if seen[n]["outcome"] == "passed" and float(seen[n]["time"]) <= floor
    )
    if too_fast:
        problems.append(
            f"{len(too_fast)} external leg(s) passed in <= {floor}s: {too_fast}. "
            "With no solver present all 13 finish in 0.99 s total, so a pass this fast "
            "is evidence the solver did not run, not evidence of agreement."
        )

    summary = {
        "ok": not problems,
        "sha": args.sha,
        "run_id": args.run_id,
        "collected": len(collected),
        "expected": len(expected),
        "passed": sum(1 for r in seen.values() if r["outcome"] == "passed"),
        "failed": failed,
        "errored": errored,
        "skipped": skipped,
        "external_below_duration_floor": too_fast,
        "external_duration_floor_s": floor,
        "durations": {n: seen[n]["time"] for n in sorted(seen)},
        "problems": problems,
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"| node id | outcome | time (s) |")
    print(f"|---|---|---|")
    for nid in sorted(seen):
        mark = "external" if nid in external else "rfx-only"
        print(f"| `{nid}` ({mark}) | {seen[nid]['outcome']} | {seen[nid]['time']:.2f} |")
    print()
    print(f"collected {len(collected)}/{len(expected)}  sha={args.sha or '(unset)'}  run_id={args.run_id or '(unset)'}")

    # One compact, grep-able line LAST, and it is load-bearing rather than
    # cosmetic: the GitHub job that submits this lane reads the run back with
    # `vessl run logs --tail 400 | tail -60`, so a verdict buried above the
    # per-node table never reaches the workflow that has to act on it. The
    # summary JSON lands on the NFS mount, which the GitHub runner cannot read
    # at all. This line is the only part of the verdict that crosses.
    verdict = {
        "ok": summary["ok"],
        "collected": summary["collected"],
        "expected": summary["expected"],
        "passed": summary["passed"],
        "failed": len(failed),
        "errored": len(errored),
        "skipped": len(skipped),
        "below_floor": len(too_fast),
        "sha": args.sha,
    }

    if problems:
        print("", file=sys.stderr)
        for line in problems:
            print(f"GATE FAIL: {line}", file=sys.stderr)
        print("CROSSVAL_SOLVER_LANE_VERDICT " + json.dumps(verdict, sort_keys=True))
        return 1

    print("GATE PASS: all 18 frozen node ids ran, none skipped, every external leg above the duration floor")
    print("CROSSVAL_SOLVER_LANE_VERDICT " + json.dumps(verdict, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
