#!/usr/bin/env python3
"""Does a patch's resonance stop depending on the in-plane cell size when the
mesh lines follow its edges?

A 29.5 x 38.0 mm PEC sheet on a 60 x 55 mm grounded slab (eps_r 3.38, 1.5 mm),
open domain, soft Ez dipole under the patch, Harminv on the ring-down. Four
arms: in-plane target cell 2 mm and 1 mm, each on a uniform in-plane grid and
on `rfx.mesh_edges.edge_aware_profile`. The z profile is identical in all
four, so only the in-plane lines differ. There is no external reference here:
the reading is self-convergence -- how far each of the three strongest modes
moves when the in-plane cell is halved.

Writes `pec_sheet_edge_offset/patch_self_convergence.json`. CPU, ~20 minutes.
"""
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
import json
import time
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from rfx.api import Simulation
from rfx import Box
from rfx.harminv import harminv
from rfx.mesh_edges import edge_aware_profile, SheetEdge
from rfx.nonuniform import make_band_profile
LX, LY = 0.080, 0.075
L, W, H = 0.0295, 0.0380, 0.0015
GX, GY = 0.060, 0.055
xc, yc = LX/2, LY/2
Z0 = 0.012                                   # ground plane height
def zprofile():
    # 0.5 mm cells through the substrate, 1.5 mm elsewhere; identical for every arm
    return make_band_profile([0.0, Z0, Z0+H, 0.030], [1.5e-3, 0.5e-3, 1.5e-3], max_ratio=1.3, protected=[False, True, False], boundary_cell=1.5e-3)
def arm(scheme, dx):
    ex = [SheetEdge(xc-L/2, +1), SheetEdge(xc+L/2, -1), SheetEdge(xc-GX/2, +1), SheetEdge(xc+GX/2, -1)]
    ey = [SheetEdge(yc-W/2, +1), SheetEdge(yc+W/2, -1), SheetEdge(yc-GY/2, +1), SheetEdge(yc+GY/2, -1)]
    if scheme == "edge":
        px = edge_aware_profile(0.0, LX, dx, sheet_edges=ex, boundary_cell=dx).cells
        py = edge_aware_profile(0.0, LY, dx, sheet_edges=ey, boundary_cell=dx).cells
    else:
        px = np.full(int(round(LX/dx)), dx); py = np.full(int(round(LY/dx)), dx)
    sim = Simulation(freq_max=5e9, domain=(LX, LY, 0.030), boundary="cpml", cpml_layers=8, dx=dx, dx_profile=px, dy_profile=py, dz_profile=zprofile())
    sim.add_material("sub", eps_r=3.38)
    sim.add(Box((xc-GX/2, yc-GY/2, Z0), (xc+GX/2, yc+GY/2, Z0+H)), material="sub")
    sim.add(Box((xc-GX/2, yc-GY/2, Z0), (xc+GX/2, yc+GY/2, Z0)), material="pec")
    sim.add(Box((xc-L/2, yc-W/2, Z0+H), (xc+L/2, yc+W/2, Z0+H)), material="pec")
    sim.add_source((xc-0.009, yc-0.006, Z0+H/2), component="ez")
    sim.add_probe((xc+0.007, yc+0.004, Z0+H/2), component="ez")
    rep = sim.fidelity_report(print_report=False)
    patch = [r for r in rep if "geometry[2]" in str(r["entity"])][0]
    real = {a["axis"]: [round(v/1e3, 4) for v in a["realized_um"]] for a in patch["axes"] if a["axis"] in "xy"}
    t0 = time.time(); res = sim.run(num_periods=40, compute_s_params=False); el = time.time()-t0
    ts = np.asarray(res.time_series)[:, 0]; dt = float(res.dt)
    ms = sorted([m for m in harminv(ts[int(0.2*len(ts)):], dt, 1.5e9, 4.5e9) if m.amplitude > 0], key=lambda m: -m.amplitude)
    return {"scheme": scheme, "dx_mm": dx*1e3, "nx": len(px), "ny": len(py), "cell_min_mm": float(min(px.min(), py.min())*1e3), "realized_patch_mm": real,
            "modes": [(round(m.freq/1e9, 4), float(f"{m.amplitude:.3g}"), round(m.Q, 1)) for m in ms[:4]], "elapsed_s": round(el), "n_steps": len(ts)}
OUT = Path(__file__).with_name("pec_sheet_edge_offset") / "patch_self_convergence.json"
out = []
for scheme, dx in (("uniform", 2e-3), ("edge", 2e-3), ("uniform", 1e-3), ("edge", 1e-3)):
    r = arm(scheme, dx); out.append(r); print(json.dumps(r), flush=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
print("PATCH_DONE")
