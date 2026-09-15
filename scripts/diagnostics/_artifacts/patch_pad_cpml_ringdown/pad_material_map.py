#!/usr/bin/env python3
"""What the SOLVED material arrays hold in the lateral absorber pads of the #801 arm.

No FDTD.  Builds the arm, assembles the materials the run would use, and prints the
permittivity and conductor occupancy along the two lateral axes through the substrate
mid-plane.  The question it answers: does the substrate (and its ground) continue into
the CPML pad, end in a vacuum facet at the interior/pad seam, or do different faces
disagree -- which is what a cross-tree stability comparison must hold fixed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
from patch_pad_cpml_ringdown import build  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--pad", type=int, default=10)
    p.add_argument("--cpml", type=int, default=None)
    p.add_argument("--subpixel", action="store_true")
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    import rfx
    sim, geom = build(a.n, pad_h=a.pad, cpml=a.cpml)
    grid = sim._build_grid()
    cond = np.asarray(sim.conductor_mask(grid), dtype=bool)
    mats = sim._assemble_materials(grid)[0]
    eps = np.asarray(mats.eps_r, dtype=float)
    ncp = int(geom["cpml_layers"])
    dx = geom["dx"]
    (ax, ay), (bx, by) = geom["patch_box"]
    ic = int(round(0.5 * (ax + bx) / dx)) + ncp
    jc = int(round(0.5 * (ay + by) / dx)) + ncp
    ks = np.flatnonzero(cond[ic, jc, :])
    k_gnd, k_patch = int(ks.min()), int(ks.max())
    ksub = int(round(0.5 * (geom["z_sub_lo"] + geom["z_sub_hi"]) / dx)) + ncp

    def runs(v):
        """[(value, first_index, count), ...] so a long row reads as segments."""
        out = []
        for i, x in enumerate(v):
            x = float(x) if not isinstance(x, (bool, np.bool_)) else bool(x)
            if out and out[-1][0] == x:
                out[-1][2] += 1
            else:
                out.append([x, i, 1])
        return [[o[0], o[1], o[2]] for o in out]

    rec = dict(
        tag=a.tag, rfx=rfx.__file__, shape=[int(s) for s in eps.shape],
        cpml_layers=ncp, n=a.n, pad_h=a.pad, subpixel_requested=bool(a.subpixel),
        k_gnd=k_gnd, k_patch=k_patch, k_substrate_mid=ksub,
        interior_index_range=[ncp, int(eps.shape[0]) - ncp - 1],
        eps_x_row_at_substrate=runs(eps[:, jc, ksub]),
        eps_y_row_at_substrate=runs(eps[ic, :, ksub]),
        cond_x_row_at_ground=runs(cond[:, jc, k_gnd]),
        cond_y_row_at_ground=runs(cond[ic, :, k_gnd]),
        eps_z_column_in_xlo_pad=runs(eps[0, jc, :]),
        eps_z_column_in_xhi_pad=runs(eps[-1, jc, :]),
        eps_z_column_interior=runs(eps[ic, jc, :]),
    )
    with open(os.path.join(a.out_dir, f"{a.tag}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"grid {rec['shape']} cpml {ncp} interior idx {rec['interior_index_range']}")
    print(f"k_gnd {k_gnd} k_substrate_mid {ksub} k_patch {k_patch}")
    for k in ("eps_x_row_at_substrate", "eps_y_row_at_substrate",
              "cond_x_row_at_ground", "cond_y_row_at_ground",
              "eps_z_column_in_xlo_pad", "eps_z_column_in_xhi_pad"):
        print(f"{k}: " + "  ".join(f"{v}@{i}x{c}" for v, i, c in rec[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
