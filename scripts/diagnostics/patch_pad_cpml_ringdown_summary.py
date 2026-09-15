#!/usr/bin/env python3
"""One table row per arm of the #801 ring-down lane.  Reporting only, no verdict.

Reads every ``*.json`` an arm of ``patch_pad_cpml_ringdown.py`` wrote into a directory
and prints the fields the pre-declaration's gate is read from, plus the provenance that
says which tree produced each row.  Kept in the repo because a VESSL ``run:`` block
cannot hold multi-line inline Python (the block is re-indented before it executes).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ORDER = ["main_cpml8", "main_cpml16", "pre1047_cpml8", "pre1047_cpml16",
         "main_cpml8_noulp", "main_cpml8_subpixel", "pre1047_cpml8_subpixel"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--json-out", default=None)
    a = p.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.dir, "*.json"))):
        try:
            r = json.load(open(f))
        except Exception as exc:
            print(f"  (unreadable: {os.path.basename(f)}: {exc!r})")
            continue
        if "tag" not in r or r.get("status") not in ("ran", "dry", "built"):
            continue
        rows.append(r)
    rows.sort(key=lambda r: ORDER.index(r["tag"]) if r["tag"] in ORDER else 99)

    hdr = (f"{'arm':24s} {'cpml':>4s} {'subpx':>5s} {'grid':>16s} {'steps':>6s} "
           f"{'settle dB':>10s} {'settled':>7s} {'growth':>8s} {'NaN@':>7s} {'wall s':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("status") != "ran":
            print(f"{r['tag']:24s}  status={r.get('status')}")
            continue
        print(f"{r['tag']:24s} {str(r.get('cpml_layers')):>4s} "
              f"{str(r.get('subpixel_smoothing')):>5s} "
              f"{str(r.get('raster', {}).get('shape')):>16s} {str(r.get('n_steps')):>6s} "
              f"{r.get('settling_db', float('nan')):10.2f} "
              f"{str(r.get('settled')):>7s} "
              f"{r.get('last30_growth_ratio_worst', float('nan')):8.3f} "
              f"{str(r.get('nonfinite_first_step')):>7s} "
              f"{r.get('wall_s', float('nan')):7.0f}")
    print()
    for r in rows:
        pv = r.get("provenance", {})
        print(f"{r['tag']:24s} rfx tree {str(pv.get('rfx_tree_sha'))[:12]:12s} "
              f"head {str(pv.get('driver_git_sha'))[:12]:12s} dirty={pv.get('driver_dirty')} "
              f"jax {pv.get('jax', {}).get('version')} {pv.get('jax', {}).get('backend')} "
              f"x64={pv.get('jax', {}).get('x64_enabled')} "
              f"domx/dx={r.get('geom', {}).get('domx_over_dx')} "
              f"domy/dx={r.get('geom', {}).get('domy_over_dx')}")
    if a.json_out:
        slim = [{k: v for k, v in r.items() if k != "envelope_trace"} for r in rows]
        with open(a.json_out, "w") as fh:
            json.dump(slim, fh, indent=1)
        print(f"\nsummary written: {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
