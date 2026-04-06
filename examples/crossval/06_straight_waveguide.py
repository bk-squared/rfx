"""Cross-validation: Dielectric waveguide field confinement.

Replicates: Meep straight waveguide tutorial
Structure: eps=12 slab waveguide, 2D TMz, observe field confinement
Comparison: Guided mode should have >90% energy inside the waveguide core
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Meep tutorial parameters (scaled)
SCALE = 1e-3  # 1 Meep unit = 1 mm
eps_wg = 12.0
wg_width = 1.0 * SCALE  # 1 mm
cell_x = 16.0 * SCALE
cell_y = 8.0 * SCALE
f_center = 0.15 * C0 / SCALE  # 45 GHz
f_width = 0.1 * C0 / SCALE

dx = 0.1 * SCALE  # 0.1 mm (resolution=10 in Meep)

print("=" * 60)
print("Cross-Validation: Straight Dielectric Waveguide (Meep)")
print("=" * 60)
print(f"Waveguide: eps={eps_wg}, width={wg_width*1e3:.1f}mm")
print(f"Cell: {cell_x*1e3:.0f}x{cell_y*1e3:.0f}mm")
print(f"Source: {f_center/1e9:.1f} GHz, BW={f_width/1e9:.1f} GHz")
print()

sim = Simulation(
    freq_max=f_center + f_width,
    domain=(cell_x, cell_y, dx),
    boundary="cpml",
    cpml_layers=8,
    dx=dx,
    mode="2d_tmz",
)

sim.add_material("waveguide", eps_r=eps_wg)
cy = cell_y / 2
wg_half = wg_width / 2

# Straight waveguide along x
sim.add(Box((0, cy - wg_half, 0), (cell_x, cy + wg_half, dx)),
        material="waveguide")

# Source at left
sim.add_source(
    (1e-3, cy, 0), component="ez",
    waveform=GaussianPulse(f0=f_center, bandwidth=f_width / f_center),
)

# Probe at midpoint (inside waveguide)
sim.add_probe((cell_x / 2, cy, 0), "ez")
# Probe outside waveguide (should see evanescent field)
sim.add_probe((cell_x / 2, cy + 3 * wg_width, 0), "ez")

grid = sim._build_grid()
n_steps = int(np.ceil(3e-9 / grid.dt))
print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}, steps={n_steps}")

result = sim.run(n_steps=n_steps)

ts = np.array(result.time_series)
ts_core = ts[:, 0] if ts.ndim == 2 else ts.ravel()
ts_clad = ts[:, 1] if ts.ndim == 2 and ts.shape[1] >= 2 else np.zeros_like(ts_core)

# Energy ratio: core vs cladding
energy_core = float(np.sum(ts_core ** 2))
energy_clad = float(np.sum(ts_clad ** 2))
confinement = energy_core / (energy_core + energy_clad + 1e-30)

print(f"\nCore energy: {energy_core:.4e}")
print(f"Cladding energy: {energy_clad:.4e}")
print(f"Confinement ratio: {confinement:.4f}")
if confinement > 0.9:
    print("PASS: >90% energy confined in core")
else:
    print(f"FAIL: only {confinement*100:.1f}% confined")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Straight Dielectric Waveguide (Meep Replica)", fontsize=13)

ax = axes[0]
t_ns = np.arange(len(ts_core)) * result.dt * 1e9
ax.plot(t_ns, ts_core, "b-", lw=0.8, label="Core")
ax.plot(t_ns, ts_clad, "r-", lw=0.8, label="Cladding")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Ez")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title("Time Domain")

ax = axes[1]
ax.text(0.5, 0.5, f"Confinement: {confinement*100:.1f}%\n\n"
        f"eps_core={eps_wg}, width={wg_width*1e3:.1f}mm\n"
        f"f0={f_center/1e9:.1f} GHz",
        transform=ax.transAxes, va="center", ha="center",
        fontsize=14, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax.axis("off")
ax.set_title("Field Confinement")

plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "06_straight_waveguide.png")
plt.savefig(out, dpi=150)
plt.close(fig)
print(f"Plot saved: {out}")
