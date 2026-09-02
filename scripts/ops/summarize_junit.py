"""Fold the shard JUnit files of one weekly GPU-suite run into one markdown report.

Usage::

    python scripts/ops/summarize_junit.py RUN_DIR [--sha SHA] [--runs "id0 id1 ..."]

``RUN_DIR`` is ``.../runs/gpu-suite/<stamp>/`` holding ``shard-<i>/junit.xml``.
Prints markdown to stdout; exit 1 if any shard failed, errored, or is missing.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--sha", default="?")
    ap.add_argument("--runs", default="")
    args = ap.parse_args(argv)
    root = Path(args.run_dir)
    shards = sorted(root.glob("shard-*"))
    tot = dict(tests=0, failures=0, errors=0, skipped=0, time=0.0)
    rows, bad = [], []
    for sd in shards:
        jx = sd / "junit.xml"
        rc = (sd / "rc").read_text().strip() if (sd / "rc").exists() else "missing"
        if not jx.exists():
            rows.append(f"| {sd.name} | — | — | — | — | — | rc={rc} (no junit.xml) |")
            continue
        suite = ET.parse(jx).getroot()
        if suite.tag == "testsuites":
            suite = suite[0] if len(suite) else suite
        n = {k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}
        t = float(suite.get("time", 0.0))
        for k in n:
            tot[k] += n[k]
        tot["time"] += t
        for tc in suite.iter("testcase"):
            for kind in ("failure", "error"):
                if tc.find(kind) is not None:
                    bad.append((sd.name, f"{tc.get('classname')}::{tc.get('name')}", kind,
                                (tc.find(kind).get("message") or "")[:160].replace("\n", " ")))
        rows.append(f"| {sd.name} | {n['tests']} | {n['failures']} | {n['errors']} | {n['skipped']} | {t/60:.1f} min | rc={rc} |")
    print(f"## Weekly GPU suite — main @ {args.sha}\n")
    print(f"Runs: {args.runs or '?'}\n")
    print("| shard | tests | failures | errors | skipped | wall | exit |\n|---|---:|---:|---:|---:|---:|---|")
    print("\n".join(rows))
    print(f"| **total** | **{tot['tests']}** | **{tot['failures']}** | **{tot['errors']}** | **{tot['skipped']}** | **{tot['time']/60:.1f} min** | |\n")
    if bad:
        print("### Failing tests\n")
        print("| shard | test | kind | message |\n|---|---|---|---|")
        for s, name, kind, msg in bad:
            print(f"| {s} | `{name}` | {kind} | {msg} |")
    else:
        print("No failures or errors.")
    missing = [sd.name for sd in shards if not (sd / "junit.xml").exists()]
    return 1 if (bad or missing or not shards) else 0


if __name__ == "__main__":
    sys.exit(main())
