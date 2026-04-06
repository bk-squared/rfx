"""Cross-validation: OpenEMS Rectangular Waveguide tutorial replica.

Replicates: docs.openems.de/python/openEMS/Tutorials/Rect_Waveguide.html
Structure: WR42 rectangular waveguide, TE10 mode, 20-26 GHz
Comparison: S21 transmission and analytical waveguide impedance
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from rfx import Simulation, Box
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# OpenEMS WR42 parameters (exact copy)
# =============================================================================
# WR42: 10.668 x 4.318 mm (standard K-band waveguide)
a = 10.668e-3   # broad wall
b = 4.318e-3    # narrow wall
wg_len = 50e-3  # waveguide length

# TE10 cutoff
f_cutoff = C0 / (2 * a)  # ~14.05 GHz
f_start = 20e9
f_stop = 26e9
n_freq = 51

print("=" * 60)
print("Cross-Validation: OpenEMS Rectangular Waveguide (WR42)")
print("=" * 60)
print(f"WR42: {a*1e3:.3f} x {b*1e3:.3f} mm, length {wg_len*1e3:.0f} mm")
print(f"TE10 cutoff: {f_cutoff/1e9:.2f} GHz")
print(f"Sweep: {f_start/1e9:.0f} - {f_stop/1e9:.0f} GHz ({n_freq} pts)")
print()

# Analytical waveguide impedance
freqs = np.linspace(f_start, f_stop, n_freq)
Z_wg = 120 * np.pi / np.sqrt(1 - (f_cutoff / freqs) ** 2)
Z_te10 = Z_wg * (b / a) * 2  # TE10 wave impedance

# =============================================================================
# rfx simulation
# =============================================================================
dx = 0.5e-3  # 0.5 mm cells

sim = Simulation(
    freq_max=f_stop * 1.2,
    domain=(wg_len, a, b),
    boundary="cpml",
    cpml_layers=10,
    dx=dx,
)

# Waveguide ports at both ends
sim.add_waveguide_port(
    0.005, direction="+x", mode=(1, 0), mode_type="TE",
    freqs=freqs, f0=(f_start + f_stop) / 2, name="port1"
)
sim.add_waveguide_port(
    wg_len - 0.005, direction="-x", mode=(1, 0), mode_type="TE",
    freqs=freqs, f0=(f_start + f_stop) / 2, name="port2"
)

print("Running waveguide S-matrix...")
result = sim.compute_waveguide_s_matrix(num_periods=30, normalize=True)

S = np.array(result.s_params)
f = np.array(result.freqs) / 1e9

s11_dB = 20 * np.log10(np.abs(S[0, 0, :]) + 1e-30)
s21_dB = 20 * np.log10(np.abs(S[1, 0, :]) + 1e-30)

print(f"S21 mean: {np.mean(np.abs(S[1,0,:])):.4f} ({np.mean(s21_dB):.1f} dB)")
print(f"S11 mean: {np.mean(np.abs(S[0,0,:])):.4f} ({np.mean(s11_dB):.1f} dB)")

# Validation: empty waveguide should have |S21| ≈ 1, |S11| ≈ 0
s21_pass = np.mean(np.abs(S[1, 0, :])) > 0.9
s11_pass = np.mean(np.abs(S[0, 0, :])) < 0.1

print("\nValidation:")
print(f"  |S21| > 0.9: {'PASS' if s21_pass else 'FAIL'}")
print(f"  |S11| < 0.1: {'PASS' if s11_pass else 'FAIL'}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("WR42 Waveguide (OpenEMS Tutorial Replica)", fontsize=13)

ax = axes[0]
ax.plot(f, s21_dB, "b-", lw=2, label="|S21|")
ax.plot(f, s11_dB, "r--", lw=1.5, label="|S11|")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("S-parameter (dB)")
ax.set_title("S-Parameters")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-40, 5)

ax = axes[1]
ax.plot(freqs / 1e9, Z_wg, "k--", label="Analytical Z_wg")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Impedance (Ω)")
ax.set_title("TE10 Wave Impedance")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "01_openems_waveguide.png")
plt.savefig(out, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {out}")
