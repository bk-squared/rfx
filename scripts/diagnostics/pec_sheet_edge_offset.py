#!/usr/bin/env python3
"""Where is the electrical edge of a PEC sheet on the Yee grid?

A 2-D PEC box (20 x 10 mm) with a thin PEC fin rising from the bottom wall at
x = 10 mm. The fin's tip is a sheet edge, in both polarizations:
``2d_tez`` (E perpendicular to the edge, the singular case) and ``2d_tmz``
(E parallel to it). The lowest resonance depends smoothly on the fin length,
so it reads the fin's effective length.

* reference: the fin tip ON a node at dx = 0.25 and 0.125 mm, Richardson
  extrapolated (second order); the empty box is checked against the analytic
  value first (comparator before anything else);
* today's uniform grid at dx = 1.0 and 0.5 mm, the drawn length swept through
  the cells in 0.125 mm steps;
* the same sweep on :func:`rfx.mesh_edges.edge_aware_profile`. The
  non-uniform lane ignores ``mode=`` and solves a 3-D box, so TMz runs as a
  one-cell-thick PEC box (Ez survives) and TEz as a two-cell box with
  magnetic walls in z (Ex, Ey, Hz survive); on the uniform grid that 3-D TEz
  arrangement reproduces the 2-D lane's number.

Writes ``pec_sheet_edge_offset/pec_sheet_edge_offset.json``. Runs on CPU,
about 20 minutes.

    python scripts/diagnostics/pec_sheet_edge_offset.py
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
from rfx.grid import C0  # noqa: E402
from rfx.harminv import harminv  # noqa: E402
from rfx.mesh_edges import SheetEdge, edge_aware_profile  # noqa: E402

A, B = 0.020, 0.010
BANDS = {"2d_tmz": (12e9, 30e9), "2d_tez": (2e9, 12e9)}
LENGTHS = np.round(np.arange(4.0e-3, 6.0e-3 + 1e-9, 0.125e-3), 9)
OUT = Path(__file__).with_suffix("") / "pec_sheet_edge_offset.json"


def lowest_resonance(mode, dx, fin_len, *, dy_profile=None, periods=60,
                     pmc_z=False, thickness=0.0):
    """Lowest resonance (Hz) and the realized fin-tip node (m)."""
    lo, hi = BANDS[mode]
    kw = {} if dy_profile is None else {"dy_profile": dy_profile}
    lz = 2 * dx if pmc_z else dx
    if pmc_z:
        from rfx.boundaries.spec import Boundary, BoundarySpec
        sim = Simulation(freq_max=hi, domain=(A, B, lz), dx=dx,
                         boundary=BoundarySpec(x="pec", y="pec",
                                               z=Boundary(lo="pmc", hi="pmc")),
                         **kw)
    else:
        sim = Simulation(freq_max=hi, domain=(A, B, lz), boundary="pec",
                         dx=dx, mode=mode, **kw)
    tip = None
    if fin_len is not None:
        sim.add(Box((A / 2, 0.0, 0.0),
                    (A / 2 + float(thickness), float(fin_len), lz)),
                material="pec")
        for row in sim.fidelity_report(print_report=False):
            if "pec" in str(row.get("entity")):
                tip = [a["realized_um"][1] * 1e-6 for a in row["axes"]
                       if a["axis"] == "y"][0]
    comp = "ez" if mode == "2d_tmz" else "ey"
    z_obs = dx if pmc_z else 0.0
    sim.add_source((0.0063, 0.0031, z_obs), component=comp)
    sim.add_probe((0.0137, 0.0069, z_obs), component=comp)
    res = sim.run(num_periods=periods, compute_s_params=False)
    ts = np.asarray(res.time_series)[:, 0]
    modes = [m for m in harminv(ts[int(0.15 * len(ts)):], float(res.dt), lo, hi)
             if m.amplitude > 0]
    top = max(m.amplitude for m in modes)
    modes = sorted((m for m in modes if m.amplitude > 1e-3 * top),
                   key=lambda m: m.freq)
    return float(modes[0].freq), tip


def edge_aware_sweep(out, modes=("2d_tmz", "2d_tez")):
    for mode in modes:
        for dx in (1e-3, 0.5e-3):
            rows = []
            for length in LENGTHS:
                prof = edge_aware_profile(
                    0.0, B, dx, boundary_cell=dx,
                    sheet_edges=[SheetEdge(float(length), -1)])
                f, tip = lowest_resonance(mode, dx, length,
                                          dy_profile=prof.cells,
                                          pmc_z=(mode == "2d_tez"))
                rows.append({"declared_m": float(length), "tip_node_m": tip,
                             "f_hz": f,
                             "edge_cell_m": float(prof.edge_cells[0]),
                             "cell_min_m": float(prof.cells.min()),
                             "cell_max_m": float(prof.cells.max())})
            out["edge_aware"][f"{mode}|{dx:.6f}"] = rows
    # the 3-D TEz arrangement against the 2-D lane, uniform grid, one length
    f3, _ = lowest_resonance("2d_tez", 1e-3, 4.75e-3, pmc_z=True)
    out["tez_3d_pmc_uniform_check_hz"] = f3


def thick_fin(mode):
    """The same sweep for a fin that is a PEC VOLUME 1 mm thick: its tip is a
    face between two corners, not a sheet edge. Reference at dx = 0.25 and
    0.125 mm (the fin is 4 and 8 cells thick there), uniform grid at 1.0 and
    0.5 mm. Written to its own record."""
    t = 1.0e-3
    out = {"mode": mode, "thickness_m": t, "reference": {}, "uniform": {}}
    for dx in (0.25e-3, 0.125e-3):
        for length in LENGTHS[::2]:
            f, _ = lowest_resonance(mode, dx, length, thickness=t)
            out["reference"].setdefault(f"{length:.6f}", {})[f"{dx:.6f}"] = f
    for d in out["reference"].values():
        d["richardson_hz"] = d["0.000125"] + (d["0.000125"] - d["0.000250"]) / 3.0
    for dx in (1e-3, 0.5e-3):
        rows = []
        for length in LENGTHS:
            f, tip = lowest_resonance(mode, dx, length, thickness=t)
            rows.append({"declared_m": float(length), "tip_node_m": tip, "f_hz": f})
        out["uniform"][f"{dx:.6f}"] = rows
    path = OUT.with_name(f"thick_fin_{mode}.json")
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path.relative_to(_REPO)}")
    return 0


def main() -> int:
    if "--thick-fin" in sys.argv:
        return thick_fin(sys.argv[sys.argv.index("--thick-fin") + 1])
    if "--edge-aware-only" in sys.argv:
        out = json.loads(OUT.read_text())
        edge_aware_sweep(out)
        OUT.write_text(json.dumps(out, indent=1) + "\n")
        print(f"updated {OUT.relative_to(_REPO)}")
        return 0
    import rfx
    out = {"rfx_file": str(Path(rfx.__file__).resolve().relative_to(_REPO)),
           "box_m": [A, B], "lengths_m": LENGTHS.tolist(),
           "empty_box": [], "reference": {}, "uniform": {}, "edge_aware": {}}
    for mode, f_an in (("2d_tmz", C0 / 2 * np.sqrt(1 / A**2 + 1 / B**2)),
                       ("2d_tez", C0 / (2 * A))):
        for dx in (1e-3, 0.5e-3):
            f, _ = lowest_resonance(mode, dx, None)
            out["empty_box"].append({"mode": mode, "dx_m": dx, "f_hz": f,
                                     "analytic_hz": float(f_an),
                                     "rel_err": f / f_an - 1.0})
    for mode in BANDS:
        ref = {}
        for dx in (0.25e-3, 0.125e-3):
            for length in LENGTHS[::2]:            # on a node at both dx
                f, tip = lowest_resonance(mode, dx, length)
                ref.setdefault(f"{length:.6f}", {})[f"{dx:.6f}"] = f
        for key, d in ref.items():
            d["richardson_hz"] = d["0.000125"] + (d["0.000125"] - d["0.000250"]) / 3.0
        out["reference"][mode] = ref
        for dx in (1e-3, 0.5e-3):
            rows = []
            for length in LENGTHS:
                f, tip = lowest_resonance(mode, dx, length)
                rows.append({"declared_m": float(length), "tip_node_m": tip, "f_hz": f})
            out["uniform"][f"{mode}|{dx:.6f}"] = rows
    edge_aware_sweep(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
