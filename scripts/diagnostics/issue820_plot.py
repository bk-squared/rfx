"""Figure for the issue #820 translation probe (named artifact for the R5 / visual-check rule).

Reads the latest ``xsweep``/``ysweep``/``fit``/``vacuum`` JSON emitted by
``issue820_rcs_translation_probe.py`` and writes
``scripts/diagnostics/issue820_results/issue820_translation.png``:

  top    — monostatic sigma(dBsm) vs integer offset, x and y arms, with the
           fitted ``|A exp(-2 j k x0) + B|`` model overlaid on the x arm;
  bottom — the complex backscatter E_theta in the Argand plane per x offset,
           which is where a fixed additive term shows as a circle that does not
           pass through the origin.

Usage:  PYTHONPATH=<worktree> python3 scripts/diagnostics/issue820_plot.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(_SCRIPT_DIR, "issue820_results")


def latest(arm):
    files = sorted(f for f in os.listdir(_OUT) if f.endswith(f"_{arm}.json"))
    if not files:
        return None
    with open(os.path.join(_OUT, files[-1])) as fh:
        return json.load(fh)


def main():
    xs = latest("xsweep")
    if xs is None:
        raise SystemExit("run --arm xsweep first")
    ys = latest("ysweep")
    ft = latest("fit")

    ox = np.array([r["offset_cells"][0] for r in xs["rows"]])
    sx = np.array([r["monostatic_dbsm"] for r in xs["rows"]])
    ex = np.array([complex(*r["E_theta"]) for r in xs["rows"]])

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.2, 8.0))

    ax0.plot(ox, sx, "o-", label=f"x translation (p-p {xs['pp_db']:.3f} dB)")
    if ys is not None:
        oy = np.array([r["offset_cells"][1] for r in ys["rows"]])
        sy = np.array([r["monostatic_dbsm"] for r in ys["rows"]])
        ax0.plot(oy, sy, "s--", label=f"y translation (p-p {ys['pp_db']:.3f} dB)")
    if ft is not None:
        dx = xs["rows"][0]["dx"]
        a = complex(*ft["A"])
        b = complex(*ft["B"])
        k = ft["k_fit"]
        fine = np.linspace(ox.min(), ox.max(), 601)
        model = a * np.exp(-2j * k * fine * dx) + b
        e_inc = xs["rows"][0]["E_inc_abs"]
        db = 10 * np.log10(4 * np.pi * np.abs(model) ** 2 / e_inc ** 2)
        ax0.plot(fine, db, "-", lw=1.0, color="k", alpha=0.6,
                 label=f"|A e^(-2jk x0) + B|, resid {ft['resid']:.3f}")
    ax0.set_xlabel("target offset inside the fixed box (cells)")
    ax0.set_ylabel("monostatic RCS (dBsm)")
    ax0.grid(alpha=0.3)
    ax0.legend(fontsize=8)

    ax1.plot(ex.real, ex.imag, "o-", ms=4)
    for o, e in zip(ox, ex):
        if o % 5 == 0:
            ax1.annotate(f"{o:+d}", (e.real, e.imag), fontsize=7)
    ax1.plot([0], [0], "k+", ms=10)
    if ft is not None:
        b = complex(*ft["B"])
        ax1.plot([b.real], [b.imag], "rx", ms=9,
                 label=f"fitted fixed term B (|B|/|A| = {ft['r']:.3f})")
        vac = latest("vacuum")
        if vac is not None:
            v = complex(*vac["result"]["E_theta"])
            ax1.plot([v.real], [v.imag], "gv", ms=8,
                     label="empty-domain (TFSF leakage) backscatter")
        ax1.legend(fontsize=8)
    ax1.set_xlabel("Re E_theta (backscatter)")
    ax1.set_ylabel("Im E_theta (backscatter)")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(alpha=0.3)

    path = os.path.join(_OUT, "issue820_translation.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"[plot] {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
