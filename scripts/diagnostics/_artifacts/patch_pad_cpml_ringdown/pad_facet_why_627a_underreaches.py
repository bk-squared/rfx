#!/usr/bin/env python3
"""Why the #627a hi-face fallback does not repair this pad.  No FDTD.

``extend_cpml_pad_materials``' docstring states the rule and its bound:

    "a structure whose hi face lands on (or inside) the domain's last interior node
     loses exactly that one node from its own rasterized mask ... if the naive
     interior-edge column is vacuum ... but the column immediately inside it is not,
     replicate from that inner column instead.  This is bounded to exactly one column
     inward -- the rasterizer's hi-face shortfall for a single box is deterministically
     one node (Box docstring: 'the shortfall is entirely at the hi face'), never more"

That bound is correct for the half-open Box rule alone.  It is not correct once
``Grid.__init__``'s ``int(np.ceil(domain/dx)) + 1`` (rfx/grid.py:153) allocates a cell the
declared domain does not reach, because a one-ULP excess in ``domain/dx`` then adds a
SECOND unfilled node at the hi face.  The fallback looks one column in, finds vacuum
there too, and gives up -- so the pad is filled with vacuum.

This prints the two columns the fallback inspects, for each pad value.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
from patch_pad_cpml_ringdown import build  # noqa: E402


def main():
    rows = []
    for pad in (6, 8, 10, 12):
        sim, geom = build(4, pad_h=pad)
        grid = sim._build_grid()
        eps = np.asarray(sim._assemble_materials(grid)[0].eps_r, dtype=float)
        dx, ncp = geom["dx"], int(geom["cpml_layers"])
        (ax, ay), (bx, by) = geom["patch_box"]
        ic = int(round(0.5 * (ax + bx) / dx)) + ncp
        jc = int(round(0.5 * (ay + by) / dx)) + ncp
        ksub = int(round(0.5 * (geom["z_sub_lo"] + geom["z_sub_hi"]) / dx)) + ncp
        for axis, row, other in (("x", eps[:, jc, ksub], None), ("y", eps[ic, :, ksub], None)):
            n = row.shape[0]
            edge = n - ncp - 1          # outermost INTERIOR node on the hi face
            inner = edge - 1            # the one column #627a is allowed to look at
            declared_cells = (geom["domain"][0] if axis == "x" else geom["domain"][1]) / dx
            rec = dict(
                pad_h=pad, axis=axis,
                declared_cells=declared_cells,
                ceil_added_a_cell=bool(np.ceil(declared_cells) > round(declared_cells)),
                n_nodes=int(n), interior_hi_edge_index=int(edge),
                eps_at_interior_hi_edge=float(row[edge]),
                eps_one_column_inward=float(row[inner]),
                eps_two_columns_inward=float(row[inner - 1]),
                eps_in_hi_pad=float(row[-1]),
                fallback_can_repair=bool(row[edge] <= 1.0 + 1e-9 and row[inner] > 1.0 + 1e-9),
                n_unfilled_nodes_at_hi_face=int(np.sum(row[edge::-1][:4] <= 1.0 + 1e-9)),
            )
            rows.append(rec)
            verdict = ("pad carries the substrate" if rec["eps_in_hi_pad"] > 1.5
                       else "PAD IS VACUUM")
            print(f"pad {pad:2d}h {axis}: declared {declared_cells!r} "
                  f"ceil-added-a-cell={rec['ceil_added_a_cell']!s:5s} | "
                  f"eps at interior hi edge {rec['eps_at_interior_hi_edge']:.2f}, "
                  f"one in {rec['eps_one_column_inward']:.2f}, "
                  f"two in {rec['eps_two_columns_inward']:.2f} | "
                  f"#627a can repair: {rec['fallback_can_repair']!s:5s} -> {verdict}")
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "pad_facet_why_627a_underreaches.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
