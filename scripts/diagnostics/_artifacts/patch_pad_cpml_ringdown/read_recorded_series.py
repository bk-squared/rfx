#!/usr/bin/env python3
"""Re-read #801's OWN recorded probe series with this lane's arithmetic.

The issue's arms were measured on tree fa3a99bd on an rtx4090 (VESSL 369367257192 /
203 / 199) and the raw series were saved to
``scripts/diagnostics/_artifacts/patch_close_20260830/refute_nulltf/*_ts.npz``
in the primary checkout.  Reading those files gives the "before" column without
re-running a GPU campaign on a CPU pod, and it guarantees the before/after columns
are scored by ONE function rather than two.

Read-only: the primary checkout is never written to.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
from patch_pad_cpml_ringdown import envelope_report  # noqa: E402

SRC = ("/root/workspace/byungkwan-workspace/research/rfx/scripts/diagnostics/"
       "_artifacts/patch_close_20260830/refute_nulltf")

ARMS = [
    ("refute_nulltf_n4_pad6", "n=4, +6h lateral pad, cpml 8"),
    ("refute_nulltf_n4_pad8", "n=4, +8h lateral pad, cpml 8"),
    ("refute_nulltf_n4_pad10", "n=4, +10h lateral pad, cpml 8  <- the arm #801 is about"),
    ("refute_nulltf_n4_pad12", "n=4, +12h lateral pad, cpml 8"),
    ("refute_nulltf_n4_pad10cpml16", "n=4, +10h lateral pad, cpml 16  <- the stable control"),
    ("refute_nulltf_n4_cpml16", "n=4, base box, cpml 16"),
    ("refute_nulltf_n4_cpml32", "n=4, base box, cpml 32"),
    ("refute_nulltf_n3_pad10", "n=3, +10h lateral pad, cpml 6"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    for tag, label in ARMS:
        npz = os.path.join(SRC, f"{tag}_ts.npz")
        js = os.path.join(SRC, f"{tag}.json")
        if not os.path.isfile(npz):
            rows.append(dict(tag=tag, label=label, status="series file absent", path=npz))
            continue
        d = np.load(npz)
        ts, dt = np.asarray(d["ts"]), float(d["dt"])
        rep = envelope_report(ts, dt, decimate=128)
        rec = dict(tag=tag, label=label, status="read", source_npz=npz)
        if os.path.isfile(js):
            src = json.load(open(js))
            rec["recorded"] = {k: src.get(k) for k in
                               ("cpml_layers", "pad_h", "n", "periods", "n_steps",
                                "settling_db", "settling_db_per_probe", "settled",
                                "wall_s", "f_L_ghz", "resid_L_pct")}
        rec.update({k: v for k, v in rep.items() if k != "envelope_trace"})
        rec["envelope_trace"] = rep["envelope_trace"]
        rows.append(rec)
    with open(os.path.join(a.out_dir, "recorded_series_reread.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"{'arm':34s} {'cpml':>4s} {'settle dB':>10s} {'recorded dB':>12s} "
          f"{'last30 growth':>14s} {'steps':>7s}")
    for r in rows:
        if r["status"] != "read":
            print(f"{r['tag']:34s}  {r['status']}")
            continue
        rc = r.get("recorded", {})
        print(f"{r['tag']:34s} {str(rc.get('cpml_layers')):>4s} {r['settling_db']:10.2f} "
              f"{rc.get('settling_db', float('nan')):12.2f} "
              f"{r['last30_growth_ratio_worst']:14.3f} {r['n_steps']:7d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
