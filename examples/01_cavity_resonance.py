"""Example 1: PEC Cavity Resonance

Simulates a rectangular PEC cavity and identifies the TM110 resonant
frequency via FFT, comparing with the analytical solution.

Expected output:
  Analytical TM110: 2.1213 GHz
  Simulated:        ~2.12 GHz (< 0.5% error)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rfx import Grid, GaussianPulse
from rfx.grid import C0
from rfx.core.yee import init_state, init_materials, update_e, update_h
from rfx.boundaries.pec import apply_pec

# Cavity dimensions (meters)
a, b, d = 0.10, 0.10, 0.05

# Analytical TM110 frequency
f_analytical = (C0 / 2) * np.sqrt((1/a)**2 + (1/b)**2)
print(f"Analytical TM110: {f_analytical/1e9:.4f} GHz")

# Set up grid (no CPML — PEC on all faces)
grid = Grid(freq_max=5e9, domain=(a, b, d), dx=0.001, cpml_layers=0)
state = init_state(grid.shape)
materials = init_materials(grid.shape)

# Source and probe positions
pulse = GaussianPulse(f0=f_analytical, bandwidth=0.8)
src = (grid.nx // 3, grid.ny // 3, grid.nz // 2)
probe = (2 * grid.nx // 3, 2 * grid.ny // 3, grid.nz // 2)

# Run simulation
n_steps = grid.num_timesteps(num_periods=80)
ts = np.zeros(n_steps)

for n in range(n_steps):
    t = n * grid.dt
    state = update_h(state, materials, grid.dt, grid.dx)
    state = update_e(state, materials, grid.dt, grid.dx)
    state = apply_pec(state)
    state = state._replace(ez=state.ez.at[src].add(pulse(t)))
    ts[n] = float(state.ez[probe])

# FFT analysis
spectrum = np.abs(np.fft.rfft(ts, n=len(ts) * 8))
freqs = np.fft.rfftfreq(len(ts) * 8, d=grid.dt)
mask = (freqs > f_analytical * 0.5) & (freqs < f_analytical * 1.5)
peak_idx = np.argmax(spectrum * mask)
f_sim = freqs[peak_idx]

error = abs(f_sim - f_analytical) / f_analytical * 100
print(f"Simulated:        {f_sim/1e9:.4f} GHz")
print(f"Error:            {error:.2f}%")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(np.arange(n_steps) * grid.dt * 1e9, ts)
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Ez (V/m)")
ax1.set_title("Time Domain")

ax2.plot(freqs / 1e9, 20 * np.log10(spectrum / max(spectrum.max(), 1e-30)))
ax2.axvline(f_analytical / 1e9, color="r", ls="--", label="Analytical")
ax2.set_xlim(0, 5)
ax2.set_ylim(-60, 5)
ax2.set_xlabel("Frequency (GHz)")
ax2.set_ylabel("Magnitude (dB)")
ax2.set_title("Spectrum")
ax2.legend()

plt.tight_layout()
plt.savefig("examples/01_cavity_resonance.png", dpi=150)
print("Plot saved: examples/01_cavity_resonance.png")
