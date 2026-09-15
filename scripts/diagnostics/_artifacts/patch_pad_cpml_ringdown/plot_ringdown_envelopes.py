#!/usr/bin/env python3
"""Ring-down envelopes for the #801 arms: the curve behind the settling number.

A settling_db headline says a run did not settle; it does not say whether the tail is a
slow physical beat, a truncated transient or one growing eigenvalue of the update
operator.  The curve does, and so does the per-probe growth rate printed beside it: four
spatially separated probes sharing a rate to three significant figures is a single mode
of the discrete operator, not a feature of any one probe's position.

Inputs: the recorded-series re-read (the issue's own arms) and, when present, this lane's
GPU arm JSONs in the same directory.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_recorded(path):
    rows = json.load(open(path))
    return [r for r in rows if r.get("status") == "read"]


def load_arms(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        base = os.path.basename(f)
        if base in ("recorded_series_reread.json", "summary.json",
                    "amplification_at_this_arm.json", "pad_facet_rounding.json"):
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if r.get("status") == "ran" and "envelope_trace" in r:
            out.append(r)
    return out


def panel(ax, rows, title, label_of):
    for r in rows:
        env = np.asarray(r["envelope_trace"], dtype=float)
        dec = int(r.get("envelope_decimate", 64))
        dt = float(r["dt_s"])
        t_ns = np.arange(env.shape[0]) * dec * dt * 1e9
        worst = env.max(axis=1)
        pk = float(np.max(r["peak"]))
        y = 20 * np.log10(np.maximum(worst, 1e-300) / max(pk, 1e-300))
        rate = max((v for v in r["last30_log_rate_per_step"] if v is not None), default=0.0)
        ax.plot(t_ns, y, lw=1.1,
                label=f"{label_of(r)}  {r['settling_db']:+.1f} dB, {rate*1e4:+.2f}e-4/step")
    ax.axhline(-40.0, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("worst-probe envelope re peak (dB)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-90, 20)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.5, loc="lower left")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True,
                   help="directory holding recorded_series_reread.json")
    p.add_argument("--arms-dir", default=None,
                   help="directory holding this lane's arm JSONs (default: --dir)")
    p.add_argument("--arms-title", default="this lane's arms")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    rec_path = os.path.join(a.dir, "recorded_series_reread.json")
    rec = load_recorded(rec_path) if os.path.isfile(rec_path) else []
    arms = load_arms(a.arms_dir or a.dir)

    n = 2 if arms else 1
    fig, axes = plt.subplots(1, n, figsize=(6.6 * n, 4.4), squeeze=False)
    if rec:
        panel(axes[0][0], rec,
              "as recorded on tree fa3a99bd (rtx4090), re-scored here",
              lambda r: r["tag"].replace("refute_nulltf_", ""))
    if arms:
        panel(axes[0][1], arms, a.arms_title, lambda r: r["tag"])
    fig.suptitle("Isolated patch, +10h lateral pad, thin CPML: ring-down envelope "
                 "(dashed = the -40 dB settling bar)", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"figure written: {a.out}  ({len(rec)} recorded arms, {len(arms)} lane arms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
