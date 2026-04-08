"""Cross-validation: 2-Port Waveguide Reciprocity

Validates waveguide port S-parameter extraction by checking reciprocity
(S12 = S21) on a straight WR-90 waveguide.

For a passive, linear, reciprocal structure: S12 = S21.
For a lossless straight waveguide: |S21| = |S12| ≈ 1, |S11| ≈ 0.

Uses compute_waveguide_s_matrix(normalize=True) for Yee-dispersion-
corrected multiport extraction.

PASS criteria:
  - |S12 - S21| / |S21| < 1% (reciprocity)
  - |S21| > 0.95 in passband (transmission)
  - |S21|^2 + |S11|^2 ≈ 1 (unitarity / power conservation)

Save: examples/crossval/22_two_port_reciprocity.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from rfx import Simulation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

C0 = 2.998e8

# WR-90 waveguide
a = 22.86e-3    # broad wall (y)
b = 10.16e-3    # narrow wall (z)
L = 80e-3       # total waveguide length (x)

# Frequency range
f_te10 = C0 / (2 * a)   # 6.56 GHz
f_te20 = C0 / a          # 13.11 GHz
f0 = (f_te10 + f_te20) / 2
f_max = 12e9

dx = 1.0e-3  # CPML = 8mm < a = 22.86mm

print("=" * 60)
print("Cross-Validation: 2-Port Waveguide Reciprocity")
print("=" * 60)
print(f"WR-90: a={a*1e3:.2f} mm, b={b*1e3:.2f} mm, L={L*1e3:.0f} mm")
print(f"TE10 cutoff: {f_te10/1e9:.2f} GHz")
print(f"Mesh: dx={dx*1e3:.1f} mm")
print()

# Build simulation
sim = Simulation(
    freq_max=f_max,
    domain=(L, a, b),
    dx=dx,
    boundary="cpml",
    cpml_layers=8,
)

# Waveguide ports
sim.add_waveguide_port(
    x_position=10e-3,
    direction="+x",
    mode=(1, 0),
    mode_type="TE",
    f0=f0,
    bandwidth=0.5,
    probe_offset=5,
    name="port1",
)
sim.add_waveguide_port(
    x_position=L - 10e-3,
    direction="-x",
    mode=(1, 0),
    mode_type="TE",
    probe_offset=5,
    name="port2",
)

# Run multiport S-matrix
print("Running normalized waveguide S-matrix (2 runs)...")
t0 = time.time()
s_result = sim.compute_waveguide_s_matrix(normalize=True, num_periods=20)
elapsed = time.time() - t0
print(f"Simulation time: {elapsed:.1f}s")

S = np.asarray(s_result.s_params)  # (2, 2, N_freq)
freqs = np.asarray(s_result.freqs)
print(f"S-parameters: {S.shape[2]} frequency points")

S11 = S[0, 0, :]
S21 = S[1, 0, :]
S12 = S[0, 1, :]
S22 = S[1, 1, :]

# Focus on passband (above TE10, below TE20)
f_mask = (freqs > f_te10 * 1.1) & (freqs < f_te20 * 0.9)
freqs_pb = freqs[f_mask]
S11_pb = S11[f_mask]
S21_pb = S21[f_mask]
S12_pb = S12[f_mask]
S22_pb = S22[f_mask]

# Diagnostics
mid = len(freqs_pb) // 2
print(f"\nMid-band ({freqs_pb[mid]/1e9:.1f} GHz):")
print(f"  |S11|={np.abs(S11_pb[mid]):.4f}, |S21|={np.abs(S21_pb[mid]):.4f}")
print(f"  |S12|={np.abs(S12_pb[mid]):.4f}, |S22|={np.abs(S22_pb[mid]):.4f}")

# =============================================================================
# Validation
# =============================================================================
PASS = True

# Check 1: Reciprocity |S12 - S21| / |S21|
recip_err = np.abs(S12_pb - S21_pb) / (np.abs(S21_pb) + 1e-30)
recip_mean = np.mean(recip_err)
print(f"\nReciprocity: mean |S12-S21|/|S21| = {recip_mean*100:.3f}%")
if recip_mean < 0.01:
    print(f"  PASS: reciprocity error {recip_mean*100:.3f}% < 1%")
else:
    print(f"  FAIL: reciprocity error {recip_mean*100:.3f}%")
    PASS = False

# Check 2: Transmission |S21| ≈ 1 in passband
s21_mean = np.mean(np.abs(S21_pb))
print(f"Transmission: mean |S21| = {s21_mean:.4f}")
if s21_mean > 0.95:
    print(f"  PASS: |S21| = {s21_mean:.4f} > 0.95")
else:
    print(f"  FAIL: |S21| = {s21_mean:.4f} (expected > 0.95)")
    PASS = False

# Check 3: Unitarity |S11|^2 + |S21|^2 ≈ 1
power = np.abs(S11_pb) ** 2 + np.abs(S21_pb) ** 2
power_mean = np.mean(power)
power_err = np.abs(power_mean - 1.0)
print(f"Power: mean(|S11|²+|S21|²) = {power_mean:.4f}")
if power_err < 0.10:
    print(f"  PASS: power conservation error {power_err:.4f} < 0.10")
else:
    print(f"  FAIL: power conservation error {power_err:.4f}")
    PASS = False

# =============================================================================
# Plot
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("2-Port Reciprocity: WR-90 Straight Waveguide", fontsize=14)

# 1. S-parameter magnitudes
ax1.plot(freqs / 1e9, 20 * np.log10(np.abs(S21) + 1e-30), "r-",
         label="|S21|", linewidth=1.5)
ax1.plot(freqs / 1e9, 20 * np.log10(np.abs(S12) + 1e-30), "r--",
         label="|S12|", linewidth=1, alpha=0.7)
ax1.plot(freqs / 1e9, 20 * np.log10(np.abs(S11) + 1e-30), "b-",
         label="|S11|", linewidth=1.5)
ax1.axvline(f_te10 / 1e9, color="gray", ls=":", alpha=0.3, label="TE10 cutoff")
ax1.set_xlabel("Frequency (GHz)")
ax1.set_ylabel("Magnitude (dB)")
ax1.set_ylim(-40, 5)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_title("S-parameters")

# 2. Reciprocity error
if np.any(f_mask):
    ax2.plot(freqs_pb / 1e9, recip_err * 100, "b-", linewidth=1.5)
    ax2.axhline(1, color="k", ls="--", alpha=0.3, label="1% threshold")
ax2.set_xlabel("Frequency (GHz)")
ax2.set_ylabel("|S12-S21|/|S21| (%)")
ax2.set_ylim(0, 5)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_title("Reciprocity Error")

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "22_two_port_reciprocity.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")

if PASS:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED")
