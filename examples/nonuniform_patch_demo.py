"""Nonuniform-mesh demo: 2.4 GHz FR4 patch antenna — NU runner validation.

Exercises the NU runner (rfx/runners/nonuniform.py) end-to-end:
  - Non-uniform z mesh (fine in substrate, coarser in air)
  - Broadband Gaussian Ez source + Ez probe
  - Harminv resonance extraction

Geometry matches crossval/05_patch_antenna.py exactly.
No OpenEMS code, no comparison bookkeeping — pure NU plumbing check.

Run:
  python examples/nonuniform_patch_demo.py
"""

import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.auto_config import smooth_grading
from rfx.harminv import harminv

C0 = 2.998e8
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "nonuniform_patch_demo")
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Geometry constants (identical to 05_patch_antenna.py)
# =============================================================================
f_design  = 2.4e9
eps_r     = 4.3
h_sub     = 1.5e-3
W         = 38.0e-3
L         = 29.5e-3
gx        = 60.0e-3
gy        = 55.0e-3
air_above = 25.0e-3
air_below = 12.0e-3
probe_inset = 8.0e-3

dx   = 1.0e-3
n_cpml = 8
n_sub  = 6
dz_sub = h_sub / n_sub          # 0.25 mm

n_below = int(math.ceil(air_below / dx))
n_above = int(math.ceil(air_above / dx))

dom_x = gx + 2 * 10e-3
dom_y = gy + 2 * 10e-3
dom_z = air_below + h_sub + air_above

gx_lo = (dom_x - gx) / 2;  gx_hi = gx_lo + gx
gy_lo = (dom_y - gy) / 2;  gy_hi = gy_lo + gy
patch_x_lo = dom_x / 2 - L / 2;  patch_x_hi = dom_x / 2 + L / 2
patch_y_lo = dom_y / 2 - W / 2;  patch_y_hi = dom_y / 2 + W / 2
feed_x = patch_x_lo + probe_inset
feed_y = dom_y / 2

z_gnd_lo  = air_below - dz_sub
z_gnd_hi  = air_below
z_sub_lo  = air_below
z_sub_hi  = air_below + h_sub
z_patch_lo = z_sub_hi
z_patch_hi = z_sub_hi + dz_sub

# =============================================================================
# Analytic reference (Balanis, Ch. 14)
# =============================================================================
eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h_sub / W) ** (-0.5)
delta_L = 0.412 * h_sub * ((eps_eff + 0.3) * (W / h_sub + 0.264)) / \
          ((eps_eff - 0.258) * (W / h_sub + 0.8))
f_an = C0 / (2 * (L + 2 * delta_L) * math.sqrt(eps_eff))

print("=" * 60)
print("Nonuniform-mesh patch antenna demo — NU runner validation")
print("=" * 60)
print(f"  Patch: W={W*1e3:.1f} mm, L={L*1e3:.1f} mm, εr={eps_r}")
print(f"  Analytic f_res = {f_an/1e9:.4f} GHz")

# =============================================================================
# Non-uniform z mesh
# =============================================================================
raw_dz = np.concatenate([
    np.full(n_below, dx),       # air below GP
    np.full(1, dz_sub),         # GP cell
    np.full(n_sub, dz_sub),     # substrate
    np.full(n_above, dx),       # air above patch
])
dz_profile = smooth_grading(raw_dz, max_ratio=1.3)

# Uniform-equivalent cell count (if everything at dz_sub)
nx = int(math.ceil(dom_x / dx))
ny = int(math.ceil(dom_y / dx))
nz_nu = len(dz_profile)
n_cells_nu = nx * ny * nz_nu

nz_uniform_equiv = int(math.ceil(dom_z / dz_sub))
n_cells_uniform = nx * ny * nz_uniform_equiv
ratio = n_cells_uniform / n_cells_nu

print(f"\nMesh:")
print(f"  NU:      {nx} x {ny} x {nz_nu} = {n_cells_nu:,} cells")
print(f"  Uniform: {nx} x {ny} x {nz_uniform_equiv} = {n_cells_uniform:,} cells")
print(f"  Memory saving ratio: {ratio:.2f}x")

# =============================================================================
# Build simulation
# =============================================================================
src_z   = z_sub_lo + dz_sub * 2.5
probe_z = src_z

sim = Simulation(
    freq_max=4e9,
    domain=(dom_x, dom_y, 0),
    dx=dx,
    dz_profile=dz_profile,
    boundary="cpml",
    cpml_layers=n_cpml,
)
sim.add_material("fr4", eps_r=eps_r, sigma=0.0)

# Finite ground plane
sim.add(Box((gx_lo, gy_lo, z_gnd_lo), (gx_hi, gy_hi, z_gnd_hi)), material="pec")
# FR4 substrate
sim.add(Box((gx_lo, gy_lo, z_sub_lo), (gx_hi, gy_hi, z_sub_hi)), material="fr4")
# Patch
sim.add(Box((patch_x_lo, patch_y_lo, z_patch_lo),
            (patch_x_hi, patch_y_hi, z_patch_hi)), material="pec")

# Broadband source at feed point
sim.add_source(
    position=(feed_x, feed_y, src_z),
    component="ez",
    waveform=GaussianPulse(f0=f_design, bandwidth=1.2),
)
# Probe at separate substrate point (avoid source overlap)
sim.add_probe(
    position=(dom_x / 2 + 5e-3, dom_y / 2 + 5e-3, probe_z),
    component="ez",
)

# =============================================================================
# Run
# =============================================================================
print(f"\nRunning NU simulation (num_periods=60)...")
t0 = time.time()
result = sim.run(num_periods=60)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f} s")

# =============================================================================
# Harminv resonance extraction
# =============================================================================
ts = np.asarray(result.time_series).ravel()
dt_val = float(result.dt)

skip = int(len(ts) * 0.3)
signal = ts[skip:]
print(f"\nHarminv on last {len(signal)} samples (ringdown region)...")

modes = harminv(signal, dt_val, 1.5e9, 3.5e9)
modes_good = [m for m in modes if m.Q > 2 and m.amplitude > 1e-8]

if modes_good:
    modes_good.sort(key=lambda m: abs(m.freq - f_an))
    best = modes_good[0]
    f_harminv = float(best.freq)
    Q_harminv  = float(best.Q)
    err_pct    = 100 * abs(f_harminv - f_an) / f_an
else:
    f_harminv = float("nan")
    Q_harminv  = float("nan")
    err_pct    = float("inf")

print(f"\nResults:")
print(f"  Analytic f_res   = {f_an/1e9:.4f} GHz")
print(f"  Harminv f_res    = {f_harminv/1e9:.4f} GHz  (Q = {Q_harminv:.1f})")
print(f"  Error vs analytic = {err_pct:.2f} %")
pass_str = "PASS" if err_pct < 8.0 else "FAIL"
print(f"  Pass criterion (<8%): {pass_str}")

# =============================================================================
# Plot (a): probe Ez time series
# =============================================================================
t_axis = np.arange(len(ts)) * dt_val * 1e9   # ns
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(t_axis, ts, lw=0.7)
ax.axvline(skip * dt_val * 1e9, color="r", ls="--", lw=1.0, label="Harminv start")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Ez (a.u.)")
ax.set_title("Probe Ez — patch ringdown (NU runner)")
ax.legend()
fig.tight_layout()
plot_ts = os.path.join(OUT_DIR, "probe_ez_timeseries.png")
fig.savefig(plot_ts, dpi=120)
plt.close(fig)
print(f"\nPlot (a): {plot_ts}")

# =============================================================================
# Plot (b): dz mesh profile (stem)
# =============================================================================
z_edges = np.concatenate([[0.0], np.cumsum(dz_profile)]) * 1e3   # mm
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
fig2, ax2 = plt.subplots(figsize=(8, 3))
ax2.stem(z_centers, dz_profile * 1e3, markerfmt="C0.", basefmt="k-",
         linefmt="C0-")
ax2.axvspan(air_below * 1e3, (air_below + h_sub) * 1e3,
            alpha=0.15, color="orange", label="FR4 substrate")
ax2.set_xlabel("z (mm)")
ax2.set_ylabel("dz (mm)")
ax2.set_title("Non-uniform z mesh profile")
ax2.legend()
fig2.tight_layout()
plot_mesh = os.path.join(OUT_DIR, "dz_mesh_profile.png")
fig2.savefig(plot_mesh, dpi=120)
plt.close(fig2)
print(f"Plot (b): {plot_mesh}")
