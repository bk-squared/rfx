#!/usr/bin/env python3
"""MSL matched-thru |S11| mesh-convergence — the mode-vs-mesh bottleneck test.

Premise-validation for the MSL *eigenmode-source* push (R1/R2): durable memory
(project_msl_eigenmode_path_b_failed) says the matched-thru |S11|~=0.118 floor
"appears to be mesh discretization, not source mode shape" — across 4 prior
eigensolver attempts, a better source mode never beat that floor. Before
committing to a 1-2 week Helmholtz-projected eigensolver, test that claim with
the EXISTING static-Laplace source by refining the mesh:

  * |S11| converges DOWN toward 0 as dx shrinks  -> MESH-limited; an eigenmode
    source cannot beat the floor -> the eigenmode push is futile (R2-STOP).
  * |S11| plateaus near ~0.11 regardless of dx   -> SOURCE-MODE-limited; NEW
    evidence justifying the eigenmode build.

Replicates tests/test_msl_port_integration.py::test_msl_thru_line_passive_gate
exactly (RO4350B, h_sub=254um, W=600um, mode='laplace'), parameterized over dx.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rfx.api import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.geometry.csg import Box  # noqa: E402

# --- thru-line constants (verbatim from the gate test) ---
EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
L_LINE = 10e-3
PORT_MARGIN = 2e-3
F_MAX = 5e9
GATE_F_LO, GATE_F_HI = 3.0e9, 4.5e9


def run_one(dx: float) -> dict:
    lx = L_LINE + 2 * PORT_MARGIN
    ly = W_TRACE + 2 * (2 * H_SUB + 8 * dx)   # same fixed-clearance formula
    lz = H_SUB + 1.5e-3
    sim = Simulation(
        freq_max=F_MAX, domain=(lx, ly, lz), dx=dx, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, H_SUB)), material="ro4350b")
    yc = ly / 2.0
    sim.add(Box((0.0, yc - W_TRACE / 2.0, H_SUB),
                (lx, yc + W_TRACE / 2.0, H_SUB + dx)), material="pec")
    sim.add_msl_port(position=(PORT_MARGIN, yc, 0.0), width=W_TRACE,
                     height=H_SUB, direction="+x", impedance=50.0)
    sim.add_msl_port(position=(PORT_MARGIN + L_LINE, yc, 0.0), width=W_TRACE,
                     height=H_SUB, direction="-x", impedance=50.0)
    t0 = time.time()
    res = sim.compute_msl_s_matrix(n_freqs=30, num_periods=12)
    dt = time.time() - t0
    S = np.asarray(res.S)
    Z0 = np.asarray(res.Z0)
    f = np.asarray(res.freqs)
    m = (f >= GATE_F_LO) & (f <= GATE_F_HI)
    s11 = np.abs(S[0, 0, m]); s21 = np.abs(S[1, 0, m]); z0 = Z0[0, m].real
    return {
        "dx_um": round(dx * 1e6, 2),
        "nz_sub": int(round(H_SUB / dx)),
        "mean_s11": float(s11.mean()),
        "max_s11": float(s11.max()),
        "mean_s21": float(s21.mean()),
        "mean_re_z0": float(z0.mean()),
        "wallclock_s": round(dt, 1),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dx-list-um", default="80,60,40,30,20")
    p.add_argument("--output", default=".omx/physics-gate/2026-06-16-msl-thru-mesh-conv/msl_thru_mesh_convergence.json")
    args = p.parse_args()
    dxs = [float(v) * 1e-6 for v in args.dx_list_um.split(",")]
    rows = []
    for dx in dxs:
        r = run_one(dx)
        rows.append(r)
        print(f"dx={r['dx_um']:5.1f}um nz_sub={r['nz_sub']:2d}  mean|S11|={r['mean_s11']:.4f} "
              f"max|S11|={r['max_s11']:.4f}  mean|S21|={r['mean_s21']:.4f}  "
              f"Re(Z0)={r['mean_re_z0']:.1f}ohm  ({r['wallclock_s']}s)", flush=True)
    # verdict
    s11s = [r["mean_s11"] for r in rows]
    trend = "DECREASING(mesh-limited)" if s11s[-1] < 0.6 * s11s[0] else "PLATEAU(mode-limited)"
    print(f"\nVERDICT: mean|S11| {s11s[0]:.4f} (dx={rows[0]['dx_um']}um) -> {s11s[-1]:.4f} "
          f"(dx={rows[-1]['dx_um']}um)  => {trend}", flush=True)
    out = REPO / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "trend": trend}, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
