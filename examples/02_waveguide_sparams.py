"""Example 2: Waveguide S-Parameter Extraction

Two-port rectangular waveguide with a dielectric obstacle (eps_r=4).
Extracts the 2x2 S-matrix using the high-level Simulation API with
waveguide ports and plots geometry, S-parameters, and a field snapshot.

Expected: above TE10 cutoff (~3.75 GHz for a=40 mm guide),
  |S21| is high (transmission through obstacle) and
  |S11| shows reflection peaks at resonant frequencies of the slab.

Saves: examples/02_waveguide_sparams.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse

# ---- Waveguide geometry ----
Lx = 0.12     # length (m)
Ly = 0.04     # guide width (a = 40 mm  → cutoff TE10 ~3.75 GHz)
Lz = 0.02     # guide height (b = 20 mm)

sim = Simulation(
    freq_max=10e9,
    domain=(Lx, Ly, Lz),
    boundary="cpml",
    cpml_layers=10,
    dx=0.002,
)

# ---- Dielectric obstacle in the middle ----
sim.add_material("obstacle", eps_r=4.0)
sim.add(Box((0.05, 0.0, 0.0), (0.07, Ly, Lz)), material="obstacle")

# ---- Waveguide ports (TE10) ----
freqs = jnp.linspace(4.5e9, 8e9, 30)
sim.add_waveguide_port(
    0.01, direction="+x", mode=(1, 0), mode_type="TE",
    freqs=freqs, f0=6e9, name="port1",
)
sim.add_waveguide_port(
    0.09, direction="-x", mode=(1, 0), mode_type="TE",
    freqs=freqs, f0=6e9, name="port2",
)

# ---- Add a probe for the field-snapshot panel ----
sim.add_probe((Lx / 2, Ly / 2, Lz / 2), component="ez")

# ---- Compute S-matrix ----
print("Running waveguide S-matrix extraction ...")
result = sim.compute_waveguide_s_matrix(num_periods=30)
S = result.s_params          # (2, 2, n_freqs)
f_GHz = np.array(result.freqs) / 1e9

s11_dB = 20 * np.log10(np.maximum(np.abs(S[0, 0, :]), 1e-10))
s21_dB = 20 * np.log10(np.maximum(np.abs(S[1, 0, :]), 1e-10))
s12_dB = 20 * np.log10(np.maximum(np.abs(S[0, 1, :]), 1e-10))

print(f"Frequency range : {f_GHz[0]:.1f} – {f_GHz[-1]:.1f} GHz")
print(f"|S21| mean       : {np.mean(np.abs(S[1, 0, :])):.3f}")
recip = np.mean(np.abs(np.abs(S[1, 0, :]) - np.abs(S[0, 1, :])))
print(f"Reciprocity err : {recip:.4f}")

# ---- Build grid and materials for geometry visualization ----
grid = sim._build_grid()
materials, _, _, _ = sim._assemble_materials(grid)
eps_r_arr = np.asarray(materials.eps_r)

# ---- Run a short sim for field snapshot ----
sim_snap = Simulation(
    freq_max=10e9,
    domain=(Lx, Ly, Lz),
    boundary="cpml",
    cpml_layers=10,
    dx=0.002,
)
sim_snap.add_material("obstacle", eps_r=4.0)
sim_snap.add(Box((0.05, 0.0, 0.0), (0.07, Ly, Lz)), material="obstacle")
sim_snap.add_port(
    (0.01, Ly / 2, Lz / 2), component="ez", impedance=50.0,
    waveform=GaussianPulse(f0=6e9, bandwidth=0.5),
)
sim_snap.add_probe((Lx / 2, Ly / 2, Lz / 2), component="ez")
snap_result = sim_snap.run(n_steps=600, compute_s_params=False)
snap_state = snap_result.state
grid_snap = sim_snap._build_grid()

# ---- 3-panel figure ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Waveguide with Dielectric Obstacle (TE10, eps_r=4)",
             fontsize=13, fontweight="bold")

# Panel 1: Geometry cross-section — eps_r slice at z=center
ax = axes[0]
iz_ctr = grid.nz // 2
eps_slice = eps_r_arr[:, :, iz_ctr]
im = ax.imshow(
    eps_slice.T, origin="lower", cmap="viridis", aspect="auto",
)
fig.colorbar(im, ax=ax, label="eps_r")
# Mark port planes
pad = grid.pad_x
port1_ix = int(round(0.01 / grid.dx)) + pad
port2_ix = int(round(0.09 / grid.dx)) + pad
ax.axvline(port1_ix, color="r", ls="--", lw=1, label="Port 1")
ax.axvline(port2_ix, color="g", ls="--", lw=1, label="Port 2")
ax.set_xlabel("x (cells)")
ax.set_ylabel("y (cells)")
ax.set_title("Geometry: eps_r (z=center)")
ax.legend(fontsize=8)

# Panel 2: |S11|, |S21|, |S12| vs frequency
ax = axes[1]
ax.plot(f_GHz, s11_dB, "b-", lw=1.5, label="|S11|")
ax.plot(f_GHz, s21_dB, "r-", lw=1.5, label="|S21|")
ax.plot(f_GHz, s12_dB, "r--", lw=1.0, alpha=0.6, label="|S12| (reciprocity)")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title("S-Parameters")
ax.set_ylim(-30, 5)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Ez field snapshot at z=center
ax = axes[2]
ez_snap = np.asarray(snap_state.ez)[:, :, grid_snap.nz // 2]
vmax = float(np.max(np.abs(ez_snap))) or 1.0
im2 = ax.imshow(
    ez_snap.T, origin="lower", cmap="RdBu_r",
    vmin=-vmax, vmax=vmax, aspect="auto",
)
fig.colorbar(im2, ax=ax, label="Ez (V/m)")
ax.set_xlabel("x (cells)")
ax.set_ylabel("y (cells)")
ax.set_title("Ez field snapshot (t=600 steps)")

plt.tight_layout()
out_path = "examples/02_waveguide_sparams.png"
plt.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Plot saved: {out_path}")
