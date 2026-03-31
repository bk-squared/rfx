"""Example 2: Waveguide S-Parameter Extraction

Two-port rectangular waveguide with a dielectric obstacle.
Extracts the 2x2 S-matrix and plots |S11|, |S21| vs frequency.

Expected: above cutoff, |S21| is high (transmission through obstacle),
|S11| shows reflection peaks at resonant frequencies of the slab.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rfx import Simulation, Box

# Waveguide dimensions
sim = Simulation(
    freq_max=10e9,
    domain=(0.12, 0.04, 0.02),
    boundary="cpml",
    cpml_layers=10,
    dx=0.002,
)

# Dielectric obstacle in the middle (eps_r=4 ~ alumina)
sim.add_material("obstacle", eps_r=4.0)
sim.add(Box((0.05, 0.0, 0.0), (0.07, 0.04, 0.02)), material="obstacle")

# Two waveguide ports
freqs = jnp.linspace(4.5e9, 8e9, 30)
sim.add_waveguide_port(0.01, direction="+x", mode=(1, 0), mode_type="TE",
                       freqs=freqs, f0=6e9, name="port1")
sim.add_waveguide_port(0.09, direction="-x", mode=(1, 0), mode_type="TE",
                       freqs=freqs, f0=6e9, name="port2")

# Compute S-matrix (one-driven-port-at-a-time)
print("Running waveguide S-matrix extraction...")
result = sim.compute_waveguide_s_matrix(num_periods=30)
S = result.s_params
f = np.array(result.freqs) / 1e9

s11 = 20 * np.log10(np.maximum(np.abs(S[0, 0, :]), 1e-10))
s21 = 20 * np.log10(np.maximum(np.abs(S[1, 0, :]), 1e-10))
s12 = 20 * np.log10(np.maximum(np.abs(S[0, 1, :]), 1e-10))

print(f"Frequency range: {f[0]:.1f} - {f[-1]:.1f} GHz")
print(f"|S21| mean: {np.mean(np.abs(S[1,0,:])):.3f}")
print(f"|S12| mean: {np.mean(np.abs(S[0,1,:])):.3f}")
print(f"Reciprocity |S21-S12| mean: {np.mean(np.abs(np.abs(S[1,0,:])-np.abs(S[0,1,:]))):.4f}")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(f, s11, "b-", label="|S11|")
ax.plot(f, s21, "r-", label="|S21|")
ax.plot(f, s12, "r--", alpha=0.5, label="|S12| (reciprocity)")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title("Waveguide S-Parameters (TE10, eps_r=4 obstacle)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-30, 5)

plt.tight_layout()
plt.savefig("examples/02_waveguide_sparams.png", dpi=150)
print("Plot saved: examples/02_waveguide_sparams.png")
