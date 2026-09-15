#!/usr/bin/env python3
"""Which edges differ between the pre-#931 rule and the boards that settle.  No FDTD.

The bisect put #801's growth on `a3e4dba4` (#931 stage A), which replaced the rule deciding
which E edges a conductor zeroes.  This says WHICH edges that change moved, for this exact
fixture, so the mechanism paragraph rests on a measurement instead of on a reading of the
commit message.

Three edge sets, all on the same declared geometry (n = 4, +10h lateral pad, cpml 8):

  * ``pre931_volume_GROWS``  -- the rule at ``a3e4dba4^``, restored by monkeypatching
    ``rfx.boundaries.pec._volume_edge_masks`` with the historical body.  The copy used is the
    one already in this repo (``scripts/diagnostics/slow_931_farfield_attribution.py``, whose
    header records it as taken from ``a3e4dba4^:rfx/boundaries/pec.py tangential_edge_masks``).
  * ``main_volume_settles``  -- the shipped rule on the declared board.
  * ``main_sheets_H4_settles`` -- the H4 reconstruction.

What it establishes, and what it does not: the live NORMAL edge is not the discriminator
(``Mz`` is 0 in the growing arm AND in a settling one), while the pre-#931 set is a strict
superset of the H4 set by a one-edge-wide TANGENTIAL ring on each conductor's hi rim.  Whether
that ring is sufficient is a dynamics question, answered by the gate test, not here.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                os.pardir, os.pardir, os.pardir, os.pardir)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import rfx.boundaries.pec as pec  # noqa: E402
from patch_pad_cpml_ringdown import build  # noqa: E402


def legacy_volume_edges(cells, periodic):
    """``a3e4dba4^:rfx/boundaries/pec.py`` tangential_edge_masks, as already copied into
    ``scripts/diagnostics/slow_931_farfield_attribution.py``: a component is PEC iff the body
    has an occupied neighbour along THAT COMPONENT'S OWN axis."""
    return tuple(cells & (pec._shift(cells, a, periodic, +1)
                          | pec._shift(cells, a, periodic, -1))
                 for a in range(3))


def edge_masks(sim, legacy=False):
    grid = sim._build_grid()
    sheets, wires = [], []
    sim._assemble_materials(grid, pec_sheets=sheets, pec_wires=wires)
    cell = None if sheets else np.asarray(sim.conductor_mask(grid), dtype=bool)
    original = pec._volume_edge_masks
    if legacy:
        pec._volume_edge_masks = legacy_volume_edges
    try:
        M = pec.realized_pec_edge_masks(cell, sheets=sheets, wires=wires,
                                        periodic=(False, False, False))
    finally:
        pec._volume_edge_masks = original
    return tuple(np.asarray(m) for m in M)


def main():
    sim_v, geom = build(4, pad_h=10)
    sim_s, _ = build(4, pad_h=10, sheet_conductors=True)
    sets = {
        "pre931_volume_GROWS": edge_masks(sim_v, legacy=True),
        "main_volume_settles": edge_masks(sim_v),
        "main_sheets_H4_settles": edge_masks(sim_s),
    }
    out = {"fixture": "n=4, +10h lateral pad, cpml_layers=8, board as declared",
           "counts": {}, "pec_edges_inside_lateral_pads": {}, "diff_pre931_vs_H4": {}}
    for k, M in sets.items():
        out["counts"][k] = {n: int(m.sum()) for n, m in zip(("Mx", "My", "Mz"), M)}
        out["counts"][k]["total"] = int(sum(int(m.sum()) for m in M))

    ncp = int(geom["cpml_layers"])
    for k, M in sets.items():
        out["pec_edges_inside_lateral_pads"][k] = int(sum(
            int(m[:ncp].sum()) + int(m[-ncp:].sum())
            + int(m[:, :ncp].sum()) + int(m[:, -ncp:].sum()) for m in M))

    a, b = sets["pre931_volume_GROWS"], sets["main_sheets_H4_settles"]
    only_a = only_b = 0
    rims = []
    for name, ma, mb in zip(("Mx", "My", "Mz"), a, b):
        oa, ob = ma & ~mb, ~ma & mb
        only_a += int(oa.sum()); only_b += int(ob.sum())
        sel_all = np.argwhere(oa)
        for k in sorted(set(sel_all[:, 2].tolist())):
            sel = sel_all[sel_all[:, 2] == k]
            rims.append(dict(component=name, k=int(k), n_edges=int(len(sel)),
                             i_lo=int(sel[:, 0].min()), i_hi=int(sel[:, 0].max()),
                             j_lo=int(sel[:, 1].min()), j_hi=int(sel[:, 1].max())))
    out["diff_pre931_vs_H4"] = dict(only_in_pre931=only_a, only_in_H4=only_b, rims=rims)

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "edge_set_comparison.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"{'edge set':32s} {'Mx':>8s} {'My':>8s} {'Mz':>8s} {'total':>9s} {'in pads':>8s}")
    for k in sets:
        c = out["counts"][k]
        print(f"{k:32s} {c['Mx']:8d} {c['My']:8d} {c['Mz']:8d} {c['total']:9d} "
              f"{out['pec_edges_inside_lateral_pads'][k]:8d}")
    d = out["diff_pre931_vs_H4"]
    print(f"\npre-#931 vs H4: only in pre-#931 = {d['only_in_pre931']}, only in H4 = "
          f"{d['only_in_H4']}  (a strict superset by the overhang ring)")
    for r in d["rims"]:
        print(f"   {r['component']} k={r['k']}: {r['n_edges']:4d} edges, "
              f"i {r['i_lo']}..{r['i_hi']}, j {r['j_lo']}..{r['j_hi']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
