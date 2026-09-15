"""cv18 criterion (B), measured for real: a one-cell OVER-aperture iris.

WHY (issue #812 round 2).  cv18's re-gate is justified by a detection table
that is a first-order MODEL (committed in
``validation/crossval/_18_wr90_iris_results/aperture_resolution.json``: the
rfx trace is displaced by the oracle's own response to a one-cell aperture
change).  Round 1 confirmed that model with a live FDTD pair, but reported the
two resulting numbers in prose only -- no artifact carries them.  This driver
re-runs that pair and writes the numbers to JSON so the model's corroboration
has the same provenance the model does.

THE DEFECT.  The symmetric two-fin construction can only realise an EVEN
aperture cell count, so a one-cell error is necessarily asymmetric: the UPPER
fin is drawn one cell short at each rung, making the electrical aperture one
cell too WIDE while ``d_phys``, the record and the oracle all stay at the
declared 7.620 mm.  That is the audit's defect and the campaign's own setup
defect (3) at half its size.

This is a DIAGNOSTIC probe.  It gates nothing, is not imported by the crossval
script, and deliberately does not reuse ``run_point`` -- ``run_point``'s
realized-geometry asserts are the fence that makes the defect impossible on
the gated path, and they must stay that way.  The geometry below is
``run_point``'s, with the upper fin moved by ``--fin-cells-delta`` and the
aperture assert adjusted to demand the defect actually landed.

#931: corners are on node planes here as they are in ``run_point``, and the
asserts read the realized PEC edge set through
``validation/crossval/_wr90_iris_realized.py`` rather than counting open nodes
in a ``rasterize()`` sigma mask.  The emitted aperture fields are renamed and
their VALUES move by one cell for the same geometry (a wall separation is one
larger than the open-node count it replaces); the defect itself -- one cell
too wide -- is unchanged.

FDTD: 2 runs (a/60 and a/30) at one configuration.  Submit on VESSL
(scripts/vessl_issue812_r2_cv17_cv18.yaml); do not run it on a shared laptop.

Usage:
  python scripts/diagnostics/probe_cv18_one_cell_aperture_defect.py \
      --output validation/crossval/_18_wr90_iris_results/one_cell_defect_live.json
  # add --geometry-only to build and rasterize both rungs and exit without
  # solving (seconds; verifies the defect is realised as intended)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CV18 = _REPO_ROOT / "validation/crossval/18_wr90_iris_modematch.py"
_RECORD = _REPO_ROOT / "validation/crossval/_18_wr90_iris_results/rfx.json"

D_PROBE = 7.62e-3          # the audit's configuration
GLEN, FRAC = 0.20, 0.50    # canonical


def _load_cv18():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location("_cv18_module", _CV18)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def defective_run(m, d_phys, cells, fin_cells_delta, geometry_only=False):
    """``run_point``'s geometry with the UPPER fin short by fin_cells_delta."""
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box

    sys.path.insert(0, str(_REPO_ROOT / "validation" / "crossval"))
    from _wr90_iris_realized import (aperture_walls, grid_plane,
                                     realized_edge_masks, wall_plane_runs)

    DX = m.A_WR90 / cells
    d_c = int(round(d_phys / DX))
    t_c = int(round(m.T_IRIS / DX))
    fin_c = (cells - d_c) // 2
    glen_c = int(round(GLEN / DX))
    p1 = int(round(0.040 / DX))
    p2 = glen_c - p1
    iris_lo = int(round(glen_c * FRAC)) - t_c // 2
    sim = Simulation(
        freq_max=float(m.FREQS[-1]) * 1.1,
        domain=(glen_c * DX, m.A_WR90, m.B_WR90), dx=DX,
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                              y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
        cpml_layers=m.cpml_layers_for(DX))
    big = 1.0
    # #931: corners on node planes, tracking run_point. The DEFECT is still
    # the upper fin drawn short by fin_cells_delta cells; only the corner
    # convention moved.
    x_lo = iris_lo * DX
    x_hi = (iris_lo + t_c) * DX
    fin_lo_hi_y = fin_c * DX                                  # lower fin: nominal
    fin_hi_lo_y = (cells - fin_c - fin_cells_delta) * DX      # upper fin: SHORT
    sim.add(Box((x_lo, -big, -big), (x_hi, fin_lo_hi_y, big)), material="pec")
    sim.add(Box((x_lo, fin_hi_lo_y, -big), (x_hi, big, big)), material="pec")
    for x, dr, nm in ((p1 * DX, "+x", "P1"), (p2 * DX, "-x", "P2")):
        sim.add_waveguide_port(x, mode=(1, 0), mode_type="TE", direction=dr,
                               f0=10.3e9, bandwidth=0.41,
                               waveform="modulated_gaussian",
                               freqs=m.FREQS, name=nm)
    grid, edges = realized_edge_masks(sim)
    assert grid.shape[1] == cells + 1, (grid.shape[1], cells)
    x_runs = wall_plane_runs(edges, 0)
    assert len(x_runs) == 1, ("iris count", x_runs)
    (x_wall_lo, x_wall_hi), = x_runs
    assert (x_wall_lo, x_wall_hi) == (grid_plane(grid, 0, iris_lo),
                                      grid_plane(grid, 0, iris_lo + t_c)), (
        "iris walls != drawn", x_runs)
    y_wall_lo, y_wall_hi = aperture_walls(
        edges, 1, (slice(x_wall_lo, x_wall_lo + 1), slice(None), slice(None)))
    # THE DEFECT MUST BE REAL: nominal is d_c cells between the two bounding
    # wall planes; one cell too wide is d_c - fin_cells_delta
    # (fin_cells_delta is negative).
    want = d_c - fin_cells_delta
    assert y_wall_hi - y_wall_lo == want, (
        "defect not realised", y_wall_hi - y_wall_lo, want)
    row = {"d_mm": round(d_phys * 1e3, 3), "cells_per_a": cells,
           "dx_mm": round(DX * 1e3, 4), "glen_m": GLEN, "iris_frac": FRAC,
           "fin_cells_delta": fin_cells_delta,
           # RENAMED for #931 (were nominal_/realized_aperture_nodes, OPEN-node
           # counts): the realized aperture is the distance between the two
           # bounding wall planes, and it is one cell larger than the old
           # count for the same geometry.
           "nominal_aperture_cells": int(d_c),
           "realized_aperture_cells": int(y_wall_hi - y_wall_lo),
           "realized_thickness_cells": int(x_wall_hi - x_wall_lo),
           "iris_wall_nodes": [int(x_wall_lo), int(x_wall_hi)],
           "aperture_wall_nodes": [int(y_wall_lo), int(y_wall_hi)]}
    if geometry_only:
        return row
    t0 = time.time()
    res = sim.compute_waveguide_s_matrix(normalize="flux", num_periods=100.0)
    s = np.asarray(res.s_params)
    row["s11"] = [round(float(v), 5) for v in np.abs(s[0, 0, :])]
    row["s21"] = [round(float(v), 5) for v in np.abs(s[1, 0, :])]
    row["wall_s"] = round(time.time() - t0, 1)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(
        _REPO_ROOT / "validation/crossval/_18_wr90_iris_results/one_cell_defect_live.json"))
    ap.add_argument("--fin-cells-delta", type=int, default=-1,
                    help="upper fin short by this many cells (-1 = one cell "
                         "too WIDE an aperture, the audit's defect)")
    ap.add_argument("--geometry-only", action="store_true")
    args = ap.parse_args()

    m = _load_cv18()
    rec = json.loads(_RECORD.read_text(encoding="utf-8"))
    key = m.config_key(round(D_PROBE * 1e3, 3), GLEN, FRAC)
    gate_cfg = m.GATE_FINE_ABS_PER_CONFIG[key]
    gate_pooled = rec["gates"]["fine_gate_abs"]
    gate_rich = rec["gates"]["richardson_gate_abs"]

    rows = {}
    for cells in (m.FINE_CELLS, m.COARSE_CELLS):
        print(f"== defective run: d={D_PROBE*1e3:.3f} mm, a/{cells} ==", flush=True)
        rows[cells] = defective_run(m, D_PROBE, cells, args.fin_cells_delta,
                                    geometry_only=args.geometry_only)
        print(json.dumps({k: v for k, v in rows[cells].items() if k not in
                          ("s11", "s21")}, indent=1), flush=True)
    if args.geometry_only:
        print("geometry-only: the defect rasterizes as intended; no solve run")
        return 0

    orc = np.asarray(m.oracle_s11(D_PROBE))          # oracle at the DECLARED d
    f = np.asarray(rows[m.FINE_CELLS]["s11"])
    c = np.asarray(rows[m.COARSE_CELLS]["s11"])
    gap = float(np.max(np.abs(f - orc)))
    rich = float(np.max(np.abs(2 * f - c - orc)))

    art = {
        "schema": "rfx.wr90_iris_one_cell_defect_live",
        "schema_version": 1,
        "issue": 812,
        "generated_by": "scripts/diagnostics/probe_cv18_one_cell_aperture_defect.py",
        "runs_fdtd": True,
        "defect": ("upper fin drawn one cell short at EACH rung -> electrical "
                   "aperture one cell too WIDE; d_phys, the record and the "
                   "oracle all stay at the declared d"),
        "config": {"d_mm": round(D_PROBE * 1e3, 3), "glen_m": GLEN,
                   "iris_frac": FRAC, "config_key": key,
                   "fin_cells_delta": args.fin_cells_delta,
                   "fine_gate_abs_per_config": gate_cfg,
                   "pooled_fine_gate_abs": gate_pooled,
                   "richardson_gate_abs": gate_rich},
        "rows": [rows[m.FINE_CELLS], rows[m.COARSE_CELLS]],
        "measured": {
            "fine_gap_abs": round(gap, 5),
            "richardson_dev_abs": round(rich, 5),
            "passes_pooled_fine_gate": bool(gap <= gate_pooled),
            "fails_per_config_fine_gate": bool(gap > gate_cfg),
            "per_config_margin_x": round(gap / gate_cfg, 3),
            "passes_richardson_gate": bool(rich <= gate_rich),
        },
        "model_comparison": {
            "note": ("the committed first-order model for this configuration "
                     "is aperture_resolution.json::pairs[2].one_cell_defect."
                     "over.{fine_gap_abs,richardson_dev_abs}; this row is the "
                     "live measurement of the same defect"),
        },
    }
    Path(args.output).write_text(json.dumps(art, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(art["measured"], indent=1))
    print(f"wrote {args.output}")
    # criterion (B), stated as the exit code: the defect must PASS the old
    # pooled gate and FAIL the new per-config one.
    ok = (art["measured"]["passes_pooled_fine_gate"]
          and art["measured"]["fails_per_config_fine_gate"])
    print("CRITERION (B) LIVE:", "CONFIRMED" if ok else "NOT REPRODUCED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
