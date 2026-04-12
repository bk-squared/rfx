"""Run the canonical openEMS `MSL_NotchFilter.py` tutorial verbatim and
save the S11/S21 spectrum to a `.npz` file for use as the crossval 13
reference.

This file is a **faithful reproduction** of the upstream tutorial
(`thliebig/openEMS:python/Tutorials/MSL_NotchFilter.py`) with three
small surgical changes:

  1. `matplotlib.use("Agg")` so it runs without a display.
  2. `post_proc_only` flag honored via env var so we can skip the
     simulation once the data is cached.
  3. Save `(f, s11, s21)` to `openems_msl_notch_ref.npz` next to this
     script instead of only plotting.

Nothing about the geometry, mesh, or solver configuration has been
altered — this is our ground truth.
"""

import os, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# numpy compat shim for openEMS v0.0.35
for _n in ("float", "int", "complex"):
    if not hasattr(np, _n):
        setattr(np, _n, {"float": float, "int": int, "complex": complex}[_n])

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "openems_msl_notch_ref.npz")

### Setup the simulation
Sim_Path = os.path.join(tempfile.gettempdir(), "NotchFilter")
post_proc_only = (
    os.environ.get("POST_PROC_ONLY", "0") == "1" and os.path.isdir(Sim_Path)
)

unit = 1e-6  # specify everything in um
MSL_length = 50000
MSL_width = 600
substrate_thickness = 254
substrate_epr = 3.66
stub_length = 12e3
f_max = 7e9

### Setup FDTD parameters & excitation function
FDTD = openEMS()
FDTD.SetGaussExcite(f_max / 2, f_max / 2)
FDTD.SetBoundaryCond(["PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"])

### Setup Geometry & Mesh
CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

resolution = C0 / (f_max * np.sqrt(substrate_epr)) / unit / 50  # λ/50
third_mesh = np.array([2 * resolution / 3, -resolution / 3]) / 4

mesh.AddLine("x", 0)
mesh.AddLine("x", MSL_width / 2 + third_mesh)
mesh.AddLine("x", -MSL_width / 2 - third_mesh)
mesh.SmoothMeshLines("x", resolution / 4)
mesh.AddLine("x", [-MSL_length, MSL_length])
mesh.SmoothMeshLines("x", resolution)

mesh.AddLine("y", 0)
mesh.AddLine("y", MSL_width / 2 + third_mesh)
mesh.AddLine("y", -MSL_width / 2 - third_mesh)
mesh.SmoothMeshLines("y", resolution / 4)
mesh.AddLine("y", [-15 * MSL_width, 15 * MSL_width + stub_length])
mesh.AddLine("y", (MSL_width / 2 + stub_length) + third_mesh)
mesh.SmoothMeshLines("y", resolution)

mesh.AddLine("z", np.linspace(0, substrate_thickness, 5))
mesh.AddLine("z", 3000)
mesh.SmoothMeshLines("z", resolution)

### Substrate
substrate = CSX.AddMaterial("RO4350B", epsilon=substrate_epr)
substrate.AddBox(
    [-MSL_length, -15 * MSL_width, 0],
    [+MSL_length, +15 * MSL_width + stub_length, substrate_thickness],
)

### MSL ports
port = [None, None]
pec = CSX.AddMetal("PEC")
port[0] = FDTD.AddMSLPort(
    1, pec,
    [-MSL_length, -MSL_width / 2, substrate_thickness],
    [0,          +MSL_width / 2, 0],
    "x", "z", excite=-1, FeedShift=10 * resolution,
    MeasPlaneShift=MSL_length / 3, priority=10,
)
port[1] = FDTD.AddMSLPort(
    2, pec,
    [MSL_length, -MSL_width / 2, substrate_thickness],
    [0,          +MSL_width / 2, 0],
    "x", "z",
    MeasPlaneShift=MSL_length / 3, priority=10,
)

### Open-circuit stub
pec.AddBox(
    [-MSL_width / 2,  MSL_width / 2,                substrate_thickness],
    [+MSL_width / 2,  MSL_width / 2 + stub_length, substrate_thickness],
    priority=10,
)

### Run
if not post_proc_only:
    FDTD.Run(Sim_Path, cleanup=True)

### Post-process
f = np.linspace(1e6, f_max, 1601)
for p in port:
    p.CalcPort(Sim_Path, f, ref_impedance=50)

s11 = port[0].uf_ref / port[0].uf_inc
s21 = port[1].uf_ref / port[0].uf_inc

np.savez(
    DATA_FILE,
    f=f, s11=s11, s21=s21,
    MSL_length=MSL_length, MSL_width=MSL_width,
    substrate_thickness=substrate_thickness, substrate_epr=substrate_epr,
    stub_length=stub_length, f_max=f_max, unit=unit,
)
print(f"Saved reference S-parameters to {DATA_FILE}")

# Quick plot for visual check
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(f / 1e9, 20 * np.log10(np.abs(s11)), "k-", lw=2, label="$S_{11}$")
ax.plot(f / 1e9, 20 * np.log10(np.abs(s21)), "r--", lw=2, label="$S_{21}$")
ax.set_xlim(0, f_max / 1e9); ax.set_ylim(-40, 5)
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("S-Parameters (dB)")
ax.set_title("openEMS MSL_NotchFilter (upstream tutorial) — reference")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "openems_msl_notch_ref.png"), dpi=150)

# Notch frequency (global S21 minimum)
idx_notch = int(np.argmin(np.abs(s21)))
print(f"openEMS notch: f = {f[idx_notch]/1e9:.3f} GHz, |S21| = {20*np.log10(np.abs(s21[idx_notch])):.2f} dB")
