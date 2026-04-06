"""Cross-validation: Dipole radiation pattern (2D, E/H plane).

Structure: Hertzian dipole (Ez point source) in free space
Comparison: Far-field pattern should match sin(theta) (donut pattern)
Tests: NTFF far-field extraction from point source
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from rfx import Simulation, compute_far_field
from rfx.sources.sources import GaussianPulse
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

f0 = 5e9
dx = 1e-3  # 1mm
dom = 0.08  # 80mm

print("=" * 60)
print("Cross-Validation: Dipole Radiation Pattern")
print("=" * 60)
print(f"Source: Ez dipole at {f0/1e9:.0f} GHz")
print(f"Domain: {dom*1e3:.0f}mm, dx={dx*1e3:.1f}mm")
print()

sim = Simulation(
    freq_max=f0 * 2,
    domain=(dom, dom, dom),
    boundary="cpml",
    cpml_layers=10,
    dx=dx,
)

center = (dom / 2, dom / 2, dom / 2)
sim.add_source(center, "ez", waveform=GaussianPulse(f0=f0, bandwidth=0.3))
sim.add_probe(center, "ez")

# NTFF box
cpml_thick = 10 * dx
ntff_margin = cpml_thick + 3 * dx
sim.add_ntff_box(
    (ntff_margin,) * 3,
    (dom - ntff_margin,) * 3,
    np.array([f0]),
)

grid = sim._build_grid()
n_steps = int(np.ceil(8e-9 / grid.dt))
print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}, steps={n_steps}")

result = sim.run(n_steps=n_steps)

# Far-field: E-plane (phi=0) and H-plane (phi=pi/2)
theta = np.linspace(0.05, np.pi - 0.05, 37)
phi = np.array([0.0, np.pi / 2])

ff = compute_far_field(result.ntff_data, result.ntff_box, grid, theta, phi)
# Result shape: (n_freqs, n_theta, n_phi) = (1, 37, 2)
power = np.abs(ff.E_theta) ** 2 + np.abs(ff.E_phi) ** 2

# E-plane: all theta at phi=0 (phi index 0)
P_E = power[0, :, 0]  # (37,)
# H-plane: all theta at phi=pi/2 (phi index 1)
P_H = power[0, :, 1]  # (37,)

power_max = float(np.max(power))

# Normalize
P_E_norm = P_E / (np.max(P_E) + 1e-30)
P_H_norm = P_H / (np.max(P_H) + 1e-30)

# Analytical: Hertzian dipole along z → power ∝ sin²(theta)
P_analytical = np.sin(theta) ** 2

# Correlation with analytical pattern
corr_E = np.corrcoef(P_E_norm, P_analytical)[0, 1]
corr_H = np.corrcoef(P_H_norm, P_analytical)[0, 1]

print(f"\nE-plane correlation with sin²(theta): {corr_E:.4f}")
print(f"H-plane correlation with sin²(theta): {corr_H:.4f}")
print(f"Max far-field power: {power_max:.4e}")

if corr_E > 0.95 and power_max > 1e-20:
    print("PASS: dipole pattern matches analytical")
else:
    print(f"FAIL: correlation E={corr_E:.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw=dict(projection="polar"))
fig.suptitle("Dipole Radiation Pattern (NTFF)", fontsize=13)

ax = axes[0]
ax.plot(theta, P_E_norm, "b-", lw=1.5, label="rfx E-plane")
ax.plot(theta, P_analytical, "r--", lw=1, label="sin²(θ)")
ax.set_title("E-plane (φ=0)")
ax.legend(loc="lower right", fontsize=8)

ax = axes[1]
ax.plot(theta, P_H_norm, "b-", lw=1.5, label="rfx H-plane")
ax.plot(theta, P_analytical, "r--", lw=1, label="sin²(θ)")
ax.set_title("H-plane (φ=π/2)")
ax.legend(loc="lower right", fontsize=8)

plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "08_dipole_radiation.png")
plt.savefig(out, dpi=150)
plt.close(fig)
print(f"Plot saved: {out}")
