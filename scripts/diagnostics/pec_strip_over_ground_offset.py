#!/usr/bin/env python3
"""The sheet-edge offset for a strip OVER A GROUND WALL, with cells that are
not square.

``pec_sheet_edge_offset.py`` measured the offset on a free fin with square
cells (0.35 cell) and then found it moves when the cell normal to the sheet
differs from the cell along it (0.27 at half, 0.60 at double). A printed
conductor is not a free fin: it lies a few cells above a ground plane and its
normal cell is usually the SMALLER one. This script measures that case.

2-D TEz cross-section of a patch (E in the plane, H out of it): a 44 x 20 mm
PEC box, an air-filled gap h = 1.5 mm between the bottom wall and a PEC strip
of length L centred at x = 22 mm. Both strip ends are free sheet edges, so the
half-wave resonance reads ``L + 2 * offset``. No dielectric: where a
dielectric face lands is a separate first-order error and would mix in.

* reference: square cells 0.25 and 0.125 mm, strip ends ON nodes, nine
  lengths, extrapolated at the measured order (``--order``: a third grid at
  one length);
* arms: normal cell 0.5 mm (three cells in the gap) against an in-plane cell
  of 0.5, 1 and 2 mm -- ratio 1, 0.5 and 0.25 -- the drawn length swept in
  0.125 mm steps. Every arm runs in the same lane (3-D box, magnetic walls in
  z), the square one included.

One record per arm under ``pec_sheet_edge_offset/``; ``--derive`` reads them
and writes ``strip_over_ground.json``.

    python scripts/diagnostics/pec_strip_over_ground_offset.py --reference
    python scripts/diagnostics/pec_strip_over_ground_offset.py --order
    python scripts/diagnostics/pec_strip_over_ground_offset.py --arm 1.0
    python scripts/diagnostics/pec_strip_over_ground_offset.py --derive
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
warnings.filterwarnings("ignore")

from rfx import Box  # noqa: E402
from rfx.api import Simulation  # noqa: E402
from rfx.harminv import harminv  # noqa: E402

A, B, H, XC = 0.044, 0.020, 1.5e-3, 0.022
NORMAL_CELL = 0.5e-3
BAND = (4.8e9, 6.5e9)          # box modes at 3.41 and 6.81 GHz stay outside
#: The reference spans what the arms can be SOLVED as: a 2 mm in-plane cell
#: holds the node span at 24 mm over the whole sweep and adds its offset.
REF_LENGTHS = tuple(np.round(np.arange(23.0e-3, 27.0e-3 + 1e-9, 0.5e-3), 9))
LENGTHS = np.round(np.arange(25.0e-3, 26.875e-3 + 1e-9, 0.125e-3), 9)
DIR = Path(__file__).with_name("pec_sheet_edge_offset")


def strip_resonance(length, *, cell=None, inplane=None):
    """Strongest resonance in BAND (Hz) and the realized strip span (m).

    ``cell`` square cells in the 2-D lane (reference); otherwise the 3-D lane
    with NORMAL_CELL in y and ``inplane`` cells in x between end cells of
    NORMAL_CELL (the lane wants both end cells equal to the base cell)."""
    if cell is not None:
        lz = cell
        sim = Simulation(freq_max=BAND[1], domain=(A, B, lz), boundary="pec",
                         dx=cell, mode="2d_tez")
        z_obs = 0.0
    else:
        from rfx.boundaries.spec import Boundary, BoundarySpec
        base = NORMAL_CELL
        n_end = int(round(inplane / base))
        ends = [base] * n_end
        mid = int(round((A - 2 * n_end * base) / inplane))
        px = np.array(ends + [inplane] * mid + ends)
        assert abs(px.sum() - A) < 1e-12
        lz = 2 * base
        sim = Simulation(
            freq_max=BAND[1], domain=(A, B, lz), dx=base,
            boundary=BoundarySpec(x="pec", y="pec",
                                  z=Boundary(lo="pmc", hi="pmc")),
            dx_profile=px, dy_profile=np.full(int(round(B / base)), base))
        z_obs = base
    sim.add(Box((XC - length / 2, H, 0.0), (XC + length / 2, H, lz)),
            material="pec")
    span = None
    for row in sim.fidelity_report(print_report=False):
        if "pec" in str(row.get("entity")):
            lo, hi = [a["realized_um"] for a in row["axes"]
                      if a["axis"] == "x"][0]
            span = (hi - lo) * 1e-6
    sim.add_source((XC - 0.009, H / 2, z_obs), component="ey")
    sim.add_probe((XC + 0.0085, H / 2, z_obs), component="ey")
    res = sim.run(num_periods=80, compute_s_params=False)
    ts = np.asarray(res.time_series)[:, 0]
    modes = [m for m in harminv(ts[int(0.15 * len(ts)):], float(res.dt), *BAND)
             if m.amplitude > 0]
    best = max(modes, key=lambda m: m.amplitude)
    return float(best.freq), span


def main() -> int:
    DIR.mkdir(exist_ok=True)
    if "--reference" in sys.argv:
        ref = {}
        for cell in (0.25e-3, 0.125e-3):
            for length in REF_LENGTHS:
                f, span = strip_resonance(length, cell=cell)
                ref.setdefault(f"{length:.6f}", {})[f"{cell:.6f}"] = f
                print(length, cell, f, span, flush=True)
        (DIR / "strip_reference.json").write_text(json.dumps(ref, indent=1) + "\n")
        return 0
    if "--order" in sys.argv:
        cells = (0.5e-3, 0.25e-3, 0.125e-3, 0.0625e-3)
        f = [strip_resonance(25.0e-3, cell=c)[0] for c in cells]
        order = [float(np.log2((f[i] - f[i + 1]) / (f[i + 1] - f[i + 2])))
                 if (f[i] - f[i + 1]) * (f[i + 1] - f[i + 2]) > 0 else None
                 for i in range(2)]
        out = {"length_m": 25.0e-3, "cells_m": list(cells), "f_hz": f,
               "order": order}
        print(out)
        (DIR / "strip_reference_order.json").write_text(json.dumps(out, indent=1) + "\n")
        return 0
    if "--arm" in sys.argv:
        inplane = float(sys.argv[sys.argv.index("--arm") + 1]) * 1e-3
        rows = []
        for length in LENGTHS:
            f, span = strip_resonance(float(length), inplane=inplane)
            rows.append({"declared_m": float(length), "node_span_m": span,
                         "f_hz": f})
            print(rows[-1], flush=True)
        (DIR / f"strip_arm_{inplane * 1e3:.3f}mm.json").write_text(
            json.dumps({"inplane_cell_m": inplane,
                        "normal_cell_m": NORMAL_CELL, "rows": rows},
                       indent=1) + "\n")
        return 0
    if "--derive" in sys.argv:
        from scipy.interpolate import CubicSpline
        from scipy.optimize import brentq
        ref = json.loads((DIR / "strip_reference.json").read_text())
        order = json.loads((DIR / "strip_reference_order.json").read_text())
        p = order["order"][-1]
        lengths = np.array(sorted(float(k) for k in ref))
        f_ref = [ref[f"{v:.6f}"]["0.000125"]
                 + (ref[f"{v:.6f}"]["0.000125"] - ref[f"{v:.6f}"]["0.000250"])
                 / (2.0**p - 1.0) for v in lengths]
        curve = CubicSpline(lengths, f_ref)
        out = {"reference": ref, "reference_order": order,
               "reference_extrapolated_hz": dict(zip(map(str, lengths), f_ref)),
               "arms": {}}
        for path in sorted(DIR.glob("strip_arm_*mm.json")):
            arm = json.loads(path.read_text())
            vals, outside = [], 0
            for r in arm["rows"]:
                g = lambda x, f=r["f_hz"]: float(curve(x)) - f   # noqa: E731
                # the curve is extended a little past its ends: a strip
                # solved longer than the longest reference length is the
                # expected case, not an outlier
                lo, hi = lengths[0] - 0.5e-3, lengths[-1] + 0.5e-3
                if g(lo) * g(hi) > 0:
                    outside += 1
                    continue
                l_eff = brentq(g, lo, hi)
                extrapolated = not (lengths[0] <= l_eff <= lengths[-1])
                vals.append(((l_eff - r["node_span_m"])
                             / (2.0 * arm["inplane_cell_m"]), extrapolated))
            inside = [v for v, e in vals if not e]
            arm["derived"] = {
                "ratio_normal_over_inplane":
                    arm["normal_cell_m"] / arm["inplane_cell_m"],
                "rows_inside_reference": len(inside),
                "rows_extrapolated_or_unsolved": len(vals) - len(inside) + outside,
                "offset_cells_mean": float(np.mean(inside)) if inside else None,
                "offset_cells_min": float(np.min(inside)) if inside else None,
                "offset_cells_max": float(np.max(inside)) if inside else None}
            print(path.name, arm["derived"])
            out["arms"][path.stem] = arm
            path.unlink()
        for name in ("strip_reference.json", "strip_reference_order.json"):
            (DIR / name).unlink()
        (DIR / "strip_over_ground.json").write_text(json.dumps(out, indent=1) + "\n")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
