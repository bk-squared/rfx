"""Post-process scripts/diagnostics/wr90_port/dump_compare_openems_vs_rfx.py output.

Loads compare_r{R}.json (per-freq |V|, |I|, arg(V), and the underlying
arrays via re-loading) and computes:
  R_V(f) = V_rfx(f) / V_openems(f)
  R_I(f) = I_rfx(f) / I_openems(f)

If rfx's V/I extraction is correct up to a global source-amplitude
gauge, |R_V|/|R_I| are constant across frequency and arg(R_V)=arg(R_I)
constant. Frequency-dependent variation in either is the bug.

Plots:
  - |R_V(f)|, |R_I(f)| normalized to mean (deviation from flat unity)
  - arg(R_V(f)) and arg(R_I(f)) — slope = reference-plane offset, scatter
    = unaccounted phase
"""
from __future__ import annotations
from pathlib import Path
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "out_compare"


def main(R: int, plane: str = "mon_left") -> None:
    j = OUT / f"compare_r{R}_{plane}.json"
    if not j.exists():
        print(f"missing {j}; run _dump_compare_openems_vs_rfx.py compare first")
        sys.exit(1)
    rows = json.loads(j.read_text())
    f = np.array([r["f_GHz"] for r in rows])
    Vr = np.array([r["|V| rfx"] for r in rows])
    Ve = np.array([r["|V| openems"] for r in rows])
    Ir = np.array([r["|I| rfx"] for r in rows])
    Ie = np.array([r["|I| openems"] for r in rows])
    argVr = np.deg2rad(np.array([r["arg(V) rfx [deg]"] for r in rows]))
    argVe = np.deg2rad(np.array([r["arg(V) openems [deg]"] for r in rows]))

    # Magnitude ratios (normalized so mean = 1)
    rv_mag = Vr / Ve
    ri_mag = Ir / Ie
    rv_mag_n = rv_mag / np.mean(rv_mag)
    ri_mag_n = ri_mag / np.mean(ri_mag)

    # Phase delta — wrap into (-pi, pi]
    dphi = np.angle(np.exp(1j * (argVr - argVe)))
    # Linear-detrended phase (remove constant + linear-in-freq slope = ref plane)
    coeff = np.polyfit(f, dphi, 1)
    dphi_resid = dphi - np.polyval(coeff, f)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    ax = axes[0]
    ax.axhline(1.0, color="k", lw=1, ls="--")
    ax.plot(f, rv_mag_n, "o-", color="#27a", label="|V_rfx / V_oe| / mean")
    ax.plot(f, ri_mag_n, "s-", color="#c52", label="|I_rfx / I_oe| / mean")
    ax.set_ylabel("normalized mag ratio")
    ax.set_title(f"rfx vs OpenEMS V/I magnitude ratios (R={R}) — "
                 "freq-dep deviation from flat = freq-dep amplitude bug")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(f, np.degrees(dphi), "o-", color="#27a",
            label="arg(V_rfx) − arg(V_oe)")
    ax.plot(f, np.degrees(np.polyval(coeff, f)), "--", color="#aaa",
            label=f"linear fit (slope={np.degrees(coeff[0]):.1f}°/GHz)")
    ax.set_ylabel("phase delta [deg]")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.axhline(0.0, color="k", lw=1, ls="--")
    ax.plot(f, np.degrees(dphi_resid), "o-", color="#a02",
            label="residual after linear fit")
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel("phase residual [deg]")
    ax.set_title("Linear-detrended phase residual = "
                 "non-linear-in-f phase error")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = OUT / f"vi_ratio_r{R}_{plane}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_png}")

    # Numerical summary
    print(f"\n=== R={R} V/I ratio summary ===")
    print(f"|V_rfx/V_oe| mean: {np.mean(rv_mag):.4g}, "
          f"spread (max-min)/mean: {(np.max(rv_mag)-np.min(rv_mag))/np.mean(rv_mag):.4f}")
    print(f"|I_rfx/I_oe| mean: {np.mean(ri_mag):.4g}, "
          f"spread (max-min)/mean: {(np.max(ri_mag)-np.min(ri_mag))/np.mean(ri_mag):.4f}")
    print(f"phase delta linear slope: {np.degrees(coeff[0]):.2f} deg/GHz "
          f"(=de-embed offset {np.degrees(coeff[0])/360 * 1e9 * 3e8 / 1e3:.1f} mm "
          f"if interpreted as β·Δx)")
    print(f"phase residual after detrend: max |Δφ| = "
          f"{np.max(np.abs(np.degrees(dphi_resid))):.2f} deg")


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    plane = sys.argv[2] if len(sys.argv) > 2 else "mon_left"
    main(R, plane)
