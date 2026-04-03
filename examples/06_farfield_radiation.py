"""Example 6: Dipole Antenna Far-Field Radiation Pattern

A short dipole antenna (Ez point source) in a CPML-bounded domain.
Demonstrates the near-to-far-field (NTFF) transform workflow:
  1. add_ntff_box() registers a Huygens surface during simulation
  2. compute_far_field() post-processes accumulated DFT data
  3. Radiation pattern in E-plane (phi=0) and H-plane (phi=90°)

3-panel figure:
  Panel 1 — Near-field Ez slice (xy plane at z-mid)
  Panel 2 — Far-field polar pattern, E-plane (phi=0)
  Panel 3 — Far-field polar pattern, H-plane (phi=π/2)

Save: examples/06_farfield_radiation.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from rfx import Simulation, Box, GaussianPulse
from rfx.farfield import compute_far_field, radiation_pattern, directivity

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

F0 = 5e9              # dipole target frequency: 5 GHz
LAMBDA = 3e8 / F0     # ~60 mm
DOMAIN_SIZE = 3 * LAMBDA / 10   # 3 × λ/10 ≈ 18 mm — compact but valid
DX = LAMBDA / 30      # ~2 mm cell size
CPML = 8

DOM = (DOMAIN_SIZE, DOMAIN_SIZE, DOMAIN_SIZE)

# Dipole source at domain centre
SRC = tuple(d / 2 for d in DOM)

print(f"Lambda={LAMBDA*1e3:.1f} mm, domain={DOMAIN_SIZE*1e3:.1f} mm, dx={DX*1e3:.2f} mm")

# ---------------------------------------------------------------------------
# Build simulation
# ---------------------------------------------------------------------------

sim = Simulation(
    freq_max=F0 * 1.5,
    domain=DOM,
    boundary="cpml",
    cpml_layers=CPML,
    dx=DX,
)

# Short Ez dipole at domain centre (soft source, no impedance loading)
sim.add_source(
    SRC,
    "ez",
    waveform=GaussianPulse(f0=F0, bandwidth=0.6),
)

# Probe for time-domain monitoring
sim.add_probe(
    (SRC[0] + DOMAIN_SIZE * 0.1, SRC[1], SRC[2]),
    "ez",
)

# NTFF Huygens box — leave at least 3 cells margin from CPML
ntff_margin = (CPML + 3) * DX
sim.add_ntff_box(
    corner_lo=(ntff_margin, ntff_margin, ntff_margin),
    corner_hi=(DOMAIN_SIZE - ntff_margin,
               DOMAIN_SIZE - ntff_margin,
               DOMAIN_SIZE - ntff_margin),
    freqs=jnp.array([F0]),
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

N_STEPS = 600
print(f"Running {N_STEPS} steps...")
result = sim.run(n_steps=N_STEPS, compute_s_params=False)
print("Simulation complete.")

# ---------------------------------------------------------------------------
# Far-field post-processing
# ---------------------------------------------------------------------------

grid = sim._build_grid()

if result.ntff_data is None or result.ntff_box is None:
    raise RuntimeError("NTFF data not collected — check add_ntff_box() setup")

# Dense angular sampling for smooth pattern plots
theta = np.linspace(0, np.pi, 181)         # elevation: 0° (z+) → 180° (z-)
phi = np.array([0.0, np.pi / 2])           # E-plane (φ=0) and H-plane (φ=90°)

ff = compute_far_field(result.ntff_data, result.ntff_box, grid, theta, phi)

D = directivity(ff)
pat = radiation_pattern(ff)  # (n_freqs, n_theta, n_phi) dB, normalized

print(f"Directivity at {F0/1e9:.1f} GHz: {D[0]:.2f} dBi")

# ---------------------------------------------------------------------------
# Figure: 3-panel layout
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(16, 6))
fig.suptitle(f"Ez Dipole Radiation at {F0/1e9:.1f} GHz  |  D = {D[0]:.1f} dBi",
             fontsize=13)

# -- Panel 1: Near-field Ez slice ------------------------------------------
ax1 = fig.add_subplot(1, 3, 1)

state = result.state
ez = np.asarray(state.ez)
z_mid = ez.shape[2] // 2
slc = ez[:, :, z_mid].T
vmax = float(np.max(np.abs(slc))) or 1.0

im = ax1.imshow(
    slc,
    origin="lower",
    cmap="RdBu_r",
    vmin=-vmax,
    vmax=vmax,
    aspect="equal",
    extent=[0, DOMAIN_SIZE * 1e3, 0, DOMAIN_SIZE * 1e3],
)
fig.colorbar(im, ax=ax1, label="Ez (V/m)", fraction=0.046, pad=0.04)
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")
ax1.set_title("Near-field Ez (xy slice, z-mid)")
ax1.plot(SRC[0] * 1e3, SRC[1] * 1e3, "y*", markersize=12, label="dipole")
ax1.legend(fontsize=9)

# -- Panel 2: E-plane pattern (phi=0) --------------------------------------
ax2 = fig.add_subplot(1, 3, 2, projection="polar")

e_plane = pat[0, :, 0]            # first freq, E-plane cut
e_plane = np.maximum(e_plane, -40)
r2 = e_plane + 40                  # shift so -40 dB → 0

ax2.plot(theta, r2, "b-", linewidth=2)
ax2.plot(-theta + 2 * np.pi, r2, "b-", linewidth=2, alpha=0.5)
ax2.set_theta_zero_location("N")
ax2.set_theta_direction(-1)
ax2.set_title(f"E-plane (φ=0°)\n{F0/1e9:.1f} GHz  D={D[0]:.1f} dBi",
              pad=15, fontsize=10)
ax2.set_ylim(0, 42)

# Add dB scale labels
for r_db, label in [(10, "−30 dB"), (20, "−20 dB"), (30, "−10 dB"), (40, "0 dB")]:
    ax2.text(np.deg2rad(35), r_db, label, fontsize=7, color="gray")

# -- Panel 3: H-plane pattern (phi=90°) ------------------------------------
ax3 = fig.add_subplot(1, 3, 3, projection="polar")

h_plane = pat[0, :, 1]            # first freq, H-plane cut
h_plane = np.maximum(h_plane, -40)
r3 = h_plane + 40

ax3.plot(theta, r3, "r-", linewidth=2)
ax3.plot(-theta + 2 * np.pi, r3, "r-", linewidth=2, alpha=0.5)
ax3.set_theta_zero_location("N")
ax3.set_theta_direction(-1)
ax3.set_title(f"H-plane (φ=90°)\n{F0/1e9:.1f} GHz  D={D[0]:.1f} dBi",
              pad=15, fontsize=10)
ax3.set_ylim(0, 42)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

plt.tight_layout()
out_path = "examples/06_farfield_radiation.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
