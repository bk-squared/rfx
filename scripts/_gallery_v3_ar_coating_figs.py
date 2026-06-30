#!/usr/bin/env python3
"""Hand-drawn geometry schematic for the AR-coating inverse-design case.

The AR case is a *1-D* normal-incidence problem (the FDTD domain is a single
cell thick transversely), so the generic 2-D ``εr`` heatmap is unreadable — the
structure shows up as a half-filled strip with no labels. This draws the
multilayer stack the way the slab case does: air | 3 matching layers | high-εr
substrate | air, with the incident / reflected / transmitted directions, the
TFSF source plane, and the reflection probe annotated. Layer permittivities are
read from the committed ``optimization.json`` (the converged design).

Usage:  python scripts/_gallery_v3_ar_coating_figs.py [--assets-dir DIR]
Writes ``<assets-dir>/geometry.png``. Run the reconcile pass afterwards to
refresh the manifest sha256.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ASSETS = "docs/public/gallery/assets/ar_coating_design"
EPS_SUB = 12.0  # case constant (high-permittivity substrate)


def _shade(eps: float) -> str:
    """Light→darker blue as εr grows (1 → 12), matching the slab palette family."""
    t = max(0.0, min(1.0, (math.sqrt(eps) - 1.0) / (math.sqrt(EPS_SUB) - 1.0)))
    # interpolate #eaf2fb (air) -> #2f6aa8 (substrate)
    c0 = (0xEA, 0xF2, 0xFB)
    c1 = (0x2F, 0x6A, 0xA8)
    rgb = tuple(int(a + (b - a) * t) for a, b in zip(c0, c1))
    return "#%02x%02x%02x" % rgb


def make_geometry(assets: str = ASSETS) -> str:
    opt = json.load(open(os.path.join(assets, "optimization.json")))
    final = opt["rows"][-1]          # [iter, cost, eps1, eps2, eps3]
    layer_eps = [float(e) for e in final[2:2 + 3]]

    # Synthetic, readable x-extent (mm); widths are schematic, not to scale.
    air_l, lay_w, sub_w, air_r = 22.0, 7.0, 26.0, 22.0
    x = 0.0
    air_lo = x; x += air_l
    layer_x = []
    for _ in layer_eps:
        layer_x.append((x, lay_w)); x += lay_w
    sub_lo = x; x += sub_w
    air_r_hi = x + air_r
    total = air_r_hi

    fig, ax = plt.subplots(figsize=(8.6, 3.7), layout="constrained")
    # air background
    ax.add_patch(Rectangle((0, 0), total, 1.0, facecolor="#eaf2fb",
                           edgecolor="none", zorder=1))
    # three matching layers
    for i, ((lx, lw), eps) in enumerate(zip(layer_x, layer_eps)):
        ax.add_patch(Rectangle((lx, 0), lw, 1.0, facecolor=_shade(eps),
                               edgecolor="#2f6aa8", lw=1.0, zorder=2))
        ax.text(lx + lw / 2, 0.5, f"L{i + 1}\nεr={eps:.2f}", ha="center",
                va="center", fontsize=8.0, rotation=90, zorder=4)
    # substrate
    ax.add_patch(Rectangle((sub_lo, 0), sub_w, 1.0, facecolor=_shade(EPS_SUB),
                           edgecolor="#21527f", lw=1.4, zorder=2))
    ax.text(sub_lo + sub_w / 2, 0.5, f"substrate\nεr = {EPS_SUB:.0f}",
            ha="center", va="center", fontsize=10, color="white", zorder=4)
    ax.text(air_lo + 3, 0.9, "air", fontsize=10, color="#3a5d80", va="top")
    ax.text(sub_lo + sub_w + 3, 0.9, "air", fontsize=10, color="#3a5d80", va="top")

    # incident / reflected / transmitted arrows
    yI, yR = 0.28, 0.72
    ax.add_patch(FancyArrowPatch((3, yI), (layer_x[0][0] - 2, yI),
                 arrowstyle="-|>", mutation_scale=16, color="#1f5fa8", lw=2.0))
    ax.text((3 + layer_x[0][0]) / 2, yI + 0.07, "incident", color="#1f5fa8",
            ha="center", fontsize=9)
    ax.add_patch(FancyArrowPatch((layer_x[0][0] - 2, yR), (3, yR),
                 arrowstyle="-|>", mutation_scale=14, color="#b00000", lw=1.8))
    ax.text((3 + layer_x[0][0]) / 2, yR - 0.10, "reflected (minimized)",
            color="#b00000", ha="center", fontsize=9)
    ax.add_patch(FancyArrowPatch((sub_lo + 2, yI), (sub_lo + sub_w - 3, yI),
                 arrowstyle="-|>", mutation_scale=16, color="#bfeccd", lw=2.2))
    ax.text(sub_lo + sub_w / 2, 0.20, "transmitted", color="white",
            ha="center", va="center", fontsize=9)

    # TFSF source plane + reflection probe (match the page prose)
    x_tfsf = 8.0
    ax.plot([x_tfsf, x_tfsf], [0, 1], ls=(0, (4, 3)), color="#444", lw=1.1, zorder=3)
    ax.text(x_tfsf, 1.02, "TFSF\nsource", ha="center", va="bottom", fontsize=7.5,
            color="#444")
    x_probe = 14.0
    ax.plot([x_probe], [0.5], marker="v", color="#7a3aa8", ms=8, zorder=5)
    ax.text(x_probe, -0.04, "reflection\nprobe", ha="center", va="top",
            fontsize=7.5, color="#7a3aa8")

    ax.set_xlim(0, total)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("propagation axis  x  (normal incidence; widths schematic)")
    ax.set_title("Anti-reflection coating — normal-incidence multilayer stack")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    out = os.path.join(assets, "geometry.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", out, "layer_eps=", [round(e, 3) for e in layer_eps])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-dir", default=ASSETS)
    args = ap.parse_args()
    make_geometry(args.assets_dir)
