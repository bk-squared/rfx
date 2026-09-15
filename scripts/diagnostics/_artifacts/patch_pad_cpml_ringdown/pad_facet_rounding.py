#!/usr/bin/env python3
"""Why one lateral absorber pad is vacuum while the other three carry the substrate.

No FDTD.  The #801 rig declares its domain as ``(38 + 2*pad)*h`` by ``dx = h/n``, and
for some pad values that product carries a one-ULP excess over an exact cell count.
rfx then allocates one MORE cell than the geometry fills, the extra node lies outside
every declared Box, and ``extend_cpml_pad_materials`` replicates THAT node -- vacuum --
through the whole absorber pad on that face.  The substrate then ends in a vacuum facet
at the interior/pad seam: the staircase-lane twin of the smoothed-lane facet #831 was
filed for and #1057 fixed, reached by a different route.

The falsifier run here: subtract one ULP from the declared domain length so the ratio
lands at or below the exact integer, keep everything else, and the facet must go.  If it
does not, the rounding story is wrong and the facet has another cause.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

H = 0.787e-3
EPS_R = 3.38
DOMX_H, DOMY_H, DOMZ_H = 38, 23, 16
ZGND_H, XP0_H = 5, 16
L_H, W_H = 11, 13
F0, BW = 8.5e9, 1.6


def build(n, pad_h, cpml=None, shrink_ulp_x=0, shrink_ulp_y=0):
    """The rig's build(), reduced to what the material map needs, plus an ULP knob."""
    from rfx import Box, Simulation
    from rfx.sources import GaussianPulse
    dx = H / n
    cpml = 2 * n if cpml is None else cpml
    nud = -0.1 * dx
    sx = sy = nud
    px = py = pad_h * H
    domx = DOMX_H * H + 2 * px
    domy = DOMY_H * H + 2 * py
    for _ in range(shrink_ulp_x):
        domx = math.nextafter(domx, 0.0)
    for _ in range(shrink_ulp_y):
        domy = math.nextafter(domy, 0.0)
    domz = DOMZ_H * H
    L, W = L_H * H, W_H * H
    x0, y0 = XP0_H * H + px + sx, (DOMY_H / 2 - W_H / 2) * H + py + sy
    z_gnd = ZGND_H * H + nud
    z_sub_lo = z_gnd + dx
    z_sub_hi = z_sub_lo + H
    z_tr_hi = z_sub_hi + dx
    sim = Simulation(freq_max=15e9, domain=(domx, domy, domz), dx=dx,
                     cpml_layers=cpml, boundary="cpml")
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    sim.add(Box((sx, sy, z_gnd), (domx + sx, domy + sy, z_sub_lo)), material="pec")
    sim.add(Box((sx, sy, z_gnd), (domx + sx, domy + sy, z_sub_hi)), material="ro4003c")
    sim.add(Box((x0, y0, z_sub_hi), (x0 + L, y0 + W, z_tr_hi)), material="pec")
    zm = 0.5 * (z_sub_lo + z_sub_hi)
    sim.add_source(position=(x0 + 0.31 * L, y0 + W / 2 - 0.27 * W, zm),
                   component="ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=F0, bandwidth=BW))
    return sim, dict(dx=dx, cpml=cpml, domx=domx, domy=domy, z_sub_lo=z_sub_lo,
                     z_sub_hi=z_sub_hi, x0=x0, y0=y0, L=L, W=W)


def facet(sim, g):
    grid = sim._build_grid()
    eps = np.asarray(sim._assemble_materials(grid)[0].eps_r, dtype=float)
    dx, ncp = g["dx"], g["cpml"]
    ic = int(round((g["x0"] + g["L"] / 2) / dx)) + ncp
    jc = int(round((g["y0"] + g["W"] / 2) / dx)) + ncp
    ksub = int(round(0.5 * (g["z_sub_lo"] + g["z_sub_hi"]) / dx)) + ncp
    xr, yr = eps[:, jc, ksub], eps[ic, :, ksub]
    return dict(
        shape=[int(v) for v in eps.shape],
        n_cells_x_from_ratio=g["domx"] / dx, n_cells_y_from_ratio=g["domy"] / dx,
        xlo_pad_eps=float(xr[0]), xhi_pad_eps=float(xr[-1]),
        ylo_pad_eps=float(yr[0]), yhi_pad_eps=float(yr[-1]),
        n_vacuum_nodes_x=int((xr < 1.5).sum()), n_vacuum_nodes_y=int((yr < 1.5).sum()),
        facet_faces=[f for f, v in (("x-hi", xr[-1]), ("x-lo", xr[0]),
                                    ("y-hi", yr[-1]), ("y-lo", yr[0])) if v < 1.5],
    )


def main():
    out = []
    for pad in (6, 8, 10, 12):
        for sx, sy, label in ((0, 0, "as declared"),
                              (1, 1, "domain length shrunk by one ULP on both axes")):
            sim, g = build(4, pad, shrink_ulp_x=sx, shrink_ulp_y=sy)
            r = facet(sim, g)
            r.update(pad_h=pad, variant=label, shrink_ulp=(sx, sy))
            out.append(r)
            print(f"pad {pad:2d}h  {label:44s}  grid {r['shape']}  "
                  f"x cells {r['n_cells_x_from_ratio']!r}  y cells {r['n_cells_y_from_ratio']!r}  "
                  f"vacuum facet on {r['facet_faces'] or 'no face'}")
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "pad_facet_rounding.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
