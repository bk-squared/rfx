"""3-solver |S11|(f) overlay for the X-band inset patch (student comparison tool).

Overlays whatever subset of the three legs is present (degrades gracefully):
  (a) rfx canonical-frame JSON      (freqs_hz + s11 as [re, im] pairs)
  (b) openEMS X-band JSON           (same schema, from openems_patch_inset_xband.py)
  (c) Palace driven CSV             (postpro/patch_s11/port-S.csv: f GHz, |S| dB, arg deg)

The comparison quantity is the dip FREQUENCY and the curve SHAPE.  Dip depth is
NOT comparable across solvers while loss models differ, and dip frequency is
mesh-limited + an unstable argmin — never turn either into a PASS/FAIL gate
(tests/test_issue80_patch_s11_regression.py).  Reference curves live in
evidence/ here and in /root/workspace/lab-shared/rfx-patch-crossval/.

Run:
  python plot_3solver_s11.py \
      --rfx out_canonical_frame/rfx_cf_s11_dx197um.json \
      --openems out_xband/openems_xband_inset_2.4mm.json \
      --palace palace_patch/postpro/patch_s11/port-S.csv \
      --output s11_trend_3solver.png
Defaults point at the committed evidence/ copies, so a bare
`python plot_3solver_s11.py` reproduces the blessed overlay.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_json_leg(path: str):
    """freqs (GHz), |S11| (dB) from an rfx/openEMS lane JSON."""
    with open(path) as f:
        d = json.load(f)
    freqs = np.asarray(d["freqs_hz"], dtype=float) / 1e9
    s11 = np.asarray([complex(re, im) for re, im in d["s11"]])
    db = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-12))
    return freqs, db


def _load_palace_csv(path: str):
    """freqs (GHz), |S11| (dB) from Palace port-S.csv (already in dB)."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rfx",
                   default=os.path.join(SCRIPT_DIR, "evidence/rfx_cf_s11_dx197um.json"))
    p.add_argument("--openems",
                   default=os.path.join(SCRIPT_DIR, "evidence/openems_xband_inset_2.4mm.json"))
    p.add_argument("--palace",
                   default=os.path.join(SCRIPT_DIR, "evidence/palace_port-S.csv"))
    p.add_argument("--output",
                   default=os.path.join(SCRIPT_DIR, "s11_trend_3solver.png"))
    args = p.parse_args()

    legs = []
    for label, path, loader in (
        ("rfx (FDTD, canonical frame)", args.rfx, _load_json_leg),
        ("openEMS (FDTD, independent)", args.openems, _load_json_leg),
        ("Palace (FEM, driven; ABC caveat)", args.palace, _load_palace_csv),
    ):
        if path and os.path.exists(path):
            legs.append((label, *loader(path)))
        else:
            print(f"  [skip] {label}: {path} not found")

    if not legs:
        print("no input curves found — nothing to plot")
        return 2  # inconclusive (missing references), per the lane exit convention

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, f_ghz, db in legs:
        ax.plot(f_ghz, db, label=label, linewidth=1.6)
        i = int(np.argmin(db))
        ax.plot(f_ghz[i], db[i], "o", markersize=4)
        print(f"  {label:36s} dip {db[i]:7.2f} dB @ {f_ghz[i]:.4f} GHz")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|S11| (dB)")
    ax.set_title("X-band inset patch — 3-solver |S11| trend "
                 "(depth not comparable across loss models)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
