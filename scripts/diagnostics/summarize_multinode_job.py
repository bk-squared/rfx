#!/usr/bin/env python3
"""Trap-time per-worker and per-job receipts; missing workers remain explicit."""

import argparse
import json
from pathlib import Path
import socket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--process-count", type=int, required=True)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    commits = list(args.output.glob(f"commit.rank{args.rank}.txt"))
    records = []
    for path in sorted(args.output.glob(f"*.rank{args.rank}.json")):
        if path.name.startswith("summary."):
            continue
        data = json.loads(path.read_text())
        records.append({"file": path.name, "status": data.get("status"),
                        "exit_code": data.get("exit_code"), "runs": len(data.get("runs", [])),
                        "exception": data.get("exception")})
    receipt = {"rank": args.rank, "hostname": socket.gethostname(), "exit_code": args.exit_code,
               "expected_tooling_sha": args.sha, "solver_ref": args.sha,
               "actual_sha": commits[0].read_text().strip() if commits else None,
               "provenance_complete": bool(commits), "probes": records}
    if not args.aggregate_only:
        temporary = args.output / f"worker.tmp.rank{args.rank}"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(args.output / f"summary.rank{args.rank}.json")
    ranks = [json.loads(p.read_text()) for p in sorted(args.output.glob("summary.rank*.json"))]
    complete = {r["rank"] for r in ranks} == set(range(args.process_count))
    summary = {"expected_process_count": args.process_count, "workers_complete": complete,
               "expected_tooling_sha": args.sha, "solver_ref": args.sha, "workers": ranks,
               "exit_codes": {str(r["rank"]): r["exit_code"] for r in ranks},
               "probe_count": sum(len(r["probes"]) for r in ranks)}
    temporary = args.output / f"summary.tmp.rank{args.rank}"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(args.output / "summary.json")


if __name__ == "__main__":
    main()
