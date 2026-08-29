"""D1 (issue #786) — geometry quantization: realized vs declared per rung.

No time stepping. Assembles exactly the materials + PEC mask each rung
compiles, and the SUBPIXEL-SMOOTHED permittivity tensor the ladder
actually solves, then reads realized feature positions off the run's own
node coordinates.

Reports three verdicts:
  D1   the original (base pre-declaration) window, in absolute metres;
  D1b  the same measurement in CELLS (addendum A1);
  D1c  scale-consistency of the smoothed material (addendum A2).

Run:  PYTHONPATH=. python -m validation.research.convergence_floor.d1_geometry
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np

from validation.research.convergence_floor import fixture as fx

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "d1_geometry.json")
WINDOWS = os.path.join(RES, "predeclared_windows_786.json")
ADDENDUM = os.path.join(RES, "predeclared_windows_786_addendum.json")

Z_UP0 = fx.PC_H_SUB + fx.PC_H_TRACE_BAND + fx.PC_AIR1
INTERFACES = {"sub_top": fx.PC_H_SUB, "upper_lo": Z_UP0,
              "upper_hi": Z_UP0 + fx.PC_H_UPPER}
OFFSETS = range(-3, 4)


def realized(scale: float, multiband: bool = False) -> dict:
    from rfx.runners.nonuniform import assemble_materials_nu
    from rfx.geometry.smoothing import compute_smoothed_eps_nonuniform
    from rfx.fidelity import _node_arrays

    prof = (fx.pc_dz_profile_sym(scale) if multiband
            else fx.pc_uniform_profile(scale))
    sim = fx.build_sim(scale, prof)
    grid = sim._build_nonuniform_grid()
    mats, _, _, pec_mask = assemble_materials_nu(sim, grid)
    eps = np.asarray(mats.eps_r, dtype=float)
    pec = np.asarray(pec_mask, dtype=bool)
    _, nodes = _node_arrays(sim, grid, True)
    pads = (int(grid.pad_x_lo), int(grid.pad_y_lo), int(grid.pad_z_lo))
    nodes = tuple(n - n[p] for n, p in zip(nodes, pads))

    dx = fx.PC_DX0 * scale
    dz = fx.PC_DZF0 * scale
    idx = np.nonzero(pec)
    row = {"scale": scale, "multiband": multiband, "dx_m": dx, "dz_m": dz,
           "shape": list(eps.shape), "n_pec_cells": int(pec.sum())}
    for a, (name, lo_d, hi_d, cell) in enumerate(
            [("x", fx.TRACE_X[0], fx.TRACE_X[1], dx),
             ("y", fx.TRACE_Y[0], fx.TRACE_Y[1], dx),
             ("z", fx.TRACE_Z[0], fx.TRACE_Z[1], dz)]):
        i0, i1 = int(idx[a].min()), int(idx[a].max())
        lo_r, hi_r = float(nodes[a][i0]), float(nodes[a][i1])
        row["pec_%s" % name] = {
            "declared_lo_m": lo_d, "declared_hi_m": hi_d,
            "realized_lo_m": lo_r, "realized_hi_m": hi_r,
            "delta_lo_m": abs(lo_r - lo_d), "delta_hi_m": abs(hi_r - hi_d),
            "delta_lo_cells": abs(lo_r - lo_d) / cell,
            "delta_hi_cells": abs(hi_r - hi_d) / cell,
            "realized_extent_m": hi_r - lo_r,
            "declared_extent_m": hi_d - lo_d,
            "realized_extent_cells": (hi_r - lo_r) / cell,
            "declared_extent_cells": (hi_d - lo_d) / cell,
        }

    # Raw assembled eps interfaces, on a column far from the trace.
    col = eps[pads[0] + 1, pads[1] + 1, :]
    zn = nodes[2]

    def span(val, tol=1e-4):
        w = np.nonzero(np.abs(col - val) < tol)[0]
        return (int(w.min()), int(w.max())) if len(w) else (None, None)

    sub_lo, sub_hi = span(fx.PC_EPS_SUB)
    up_lo, up_hi = span(fx.PC_EPS_UPPER)
    raw = {
        "sub_top_realized_m": float(zn[sub_hi + 1]),
        "upper_lo_realized_m": float(zn[up_lo]),
        "upper_hi_realized_m": float(zn[up_hi + 1]),
    }
    raw["delta_m"] = max(abs(raw["sub_top_realized_m"] - fx.PC_H_SUB),
                         abs(raw["upper_lo_realized_m"] - Z_UP0),
                         abs(raw["upper_hi_realized_m"]
                             - (Z_UP0 + fx.PC_H_UPPER)))
    raw["delta_cells"] = raw["delta_m"] / dz
    row["eps_raw"] = raw

    # Subpixel-smoothed permittivity, as an offset -> eps map around every
    # declared interface (D1c). Offsets are in CELLS from the node that
    # sits on the declared interface plane.
    pairs = [(e.shape, sim._resolve_material(e.material_name).eps_r)
             for e in sim._geometry]
    aniso = compute_smoothed_eps_nonuniform(grid, pairs, background_eps=1.0)
    comps = aniso if isinstance(aniso, (tuple, list)) else (aniso,)
    smaps = {}
    for ci, comp in enumerate(comps):
        ax = np.asarray(comp, dtype=float)
        colx = ax[pads[0] + 1, pads[1] + 1, :]
        for iname, zpos in INTERFACES.items():
            k0 = int(round(zpos / dz))
            smaps["comp%d/%s" % (ci, iname)] = {
                str(o): float(colx[k0 + o]) for o in OFFSETS
                if 0 <= k0 + o < len(colx)}
    row["smoothed_offset_maps"] = smaps

    row["delta_max_m"] = max(
        [row["pec_%s" % a][k] for a in "xyz"
         for k in ("delta_lo_m", "delta_hi_m")] + [raw["delta_m"]])
    row["delta_max_cells"] = max(
        [row["pec_%s" % a][k] for a in "xyz"
         for k in ("delta_lo_cells", "delta_hi_cells")] + [raw["delta_cells"]])
    return row


def main():
    warnings.filterwarnings("ignore")
    base = json.load(open(WINDOWS))["D1_geometry_quantization"]
    add = json.load(open(ADDENDUM))
    frac = float(base["attribute_delta_frac_of_dx"])
    exo_cells = float(add["D1b_geometry_quantization_in_cells"]["exonerate_cells"])
    att_cells = float(add["D1b_geometry_quantization_in_cells"]["attribute_cells"])
    tol_rel = float(add["D1c_smoothed_material_scale_consistency"]["tol_rel"])

    rows = []
    for s in sorted(list(fx.SCALES) + [fx.REF_SCALE], reverse=True):
        r = realized(s)
        rows.append(r)
        print("s=%-5s dx=%.5fmm  PEC extent x=%.1f y=%.1f z=%.1f cells  "
              "delta_max=%.3e m = %.2e cells"
              % (s, r["dx_m"] * 1e3,
                 r["pec_x"]["realized_extent_cells"],
                 r["pec_y"]["realized_extent_cells"],
                 r["pec_z"]["realized_extent_cells"],
                 r["delta_max_m"], r["delta_max_cells"]), flush=True)

    dmax_m = max(r["delta_max_m"] for r in rows)
    dmax_c = max(r["delta_max_cells"] for r in rows)

    v_base = ("EXONERATED" if dmax_m < 1e-12 else
              ("ATTRIBUTED-CANDIDATE"
               if any(r["delta_max_m"] >= frac * r["dx_m"] for r in rows)
               else "INCONCLUSIVE"))
    v_b = ("EXONERATED" if dmax_c < exo_cells else
           ("ATTRIBUTED-CANDIDATE" if dmax_c >= att_cells else "INCONCLUSIVE"))

    # D1c: the offset->eps maps must be identical across rungs.
    keys = sorted(rows[0]["smoothed_offset_maps"])
    worst, worst_key = 0.0, None
    for k in keys:
        vals = [r["smoothed_offset_maps"].get(k, {}) for r in rows]
        offs = set().union(*[set(v) for v in vals])
        for o in offs:
            xs = [v[o] for v in vals if o in v]
            if len(xs) < 2:
                continue
            rel = (max(xs) - min(xs)) / max(abs(np.mean(xs)), 1e-30)
            if rel > worst:
                worst, worst_key = rel, "%s@%s" % (k, o)
    v_c = ("EXONERATED" if worst <= tol_rel else "ATTRIBUTED-CANDIDATE")

    out = {
        "issue": 786, "discriminator": "D1", "rows": rows,
        "delta_max_over_all_rungs_m": dmax_m,
        "delta_max_over_all_rungs_cells": dmax_c,
        "D1_base_window_verdict": v_base,
        "D1_base_window_note": (
            "the base window (<1e-12 m) sits BELOW the float32 mesh-storage "
            "floor of ~eps32*L = 1.6e-9 m, so no correctly-realized "
            "geometry can meet it; see the addendum's A1 disclosure"),
        "D1b_cells_verdict": v_b,
        "D1c_smoothed_max_rel_spread": worst,
        "D1c_worst_key": worst_key,
        "D1c_verdict": v_c,
        "verdict": (
            "EXONERATED (D1b: delta_max = %.2e cells < %.0e; D1c: the "
            "smoothed material is self-similar to %.1e relative). Every "
            "rung realizes the identical declared structure; geometry "
            "quantization cannot be the mechanism. D1's own base window "
            "reads %s only because it was written below the float32 "
            "mesh-storage floor." % (dmax_c, exo_cells, worst, v_base)
            if v_b == "EXONERATED" and v_c == "EXONERATED"
            else "D1b=%s D1c=%s" % (v_b, v_c)),
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nD1 base-window letter verdict:", v_base,
          "(delta_max = %.3e m)" % dmax_m)
    print("D1b (cells):", v_b, "(delta_max = %.3e cells)" % dmax_c)
    print("D1c (smoothed self-similarity):", v_c,
          "(max rel spread %.3e at %s)" % (worst, worst_key))
    print("\nD1:", out["verdict"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
