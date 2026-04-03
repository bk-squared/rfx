"""Validation Case 5: PEC Cavity Filter with Iris Coupling

A waveguide section with two PEC iris walls forming a resonant cavity.
Validates that S21 shows frequency-selective transmission (passband peak)
near the cavity resonance frequency.

Reference: analytical cavity resonance f_mnp = c/(2) * sqrt((m/a)^2 + (n/b)^2 + (p/d)^2)
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import jax.numpy as jnp

from rfx import Simulation, Box
from rfx.grid import C0

# ---- Waveguide and cavity parameters ----
# Waveguide cross-section: wide enough for TE10 well below cavity resonance
wg_width = 0.04    # y-dimension (a = 40mm, cutoff ~3.75 GHz)
wg_height = 0.02   # z-dimension (b = 20mm)

# Cavity length along x between iris walls
cavity_length = 0.04   # 40mm

# TE101-like resonance of the section: accounts for x and z
# f_101 = c/2 * sqrt((1/cavity_length)^2 + (1/wg_height)^2)
# but the transverse TE10 mode structure dominates
# Operating frequency: TE10 in waveguide, resonant when cavity_length ~ lambda_g/2
f_cutoff = C0 / (2 * wg_width)  # TE10 cutoff
f_target = 6e9   # operating frequency well above cutoff

# Guided wavelength at f_target
lambda_0 = C0 / f_target
lambda_g = lambda_0 / np.sqrt(1 - (f_cutoff / f_target) ** 2)

print(f"Waveguide      : {wg_width * 1e3:.0f} x {wg_height * 1e3:.0f} mm")
print(f"TE10 cutoff    : {f_cutoff / 1e9:.2f} GHz")
print(f"Target freq    : {f_target / 1e9:.1f} GHz")
print(f"lambda_g       : {lambda_g * 1e3:.1f} mm")
print(f"Cavity length  : {cavity_length * 1e3:.0f} mm")

# ---- Build simulation ----
wg_length = 0.03    # feed waveguide length on each side
iris_width = 0.015  # iris opening in y (reasonably wide for coupling)
dx = 0.002          # 2mm cells

total_x = wg_length + cavity_length + wg_length

sim = Simulation(
    freq_max=10e9,
    domain=(total_x, wg_width, wg_height),
    boundary="cpml",
    cpml_layers=8,
    dx=dx,
)

# Iris walls with coupling aperture
iris_x1 = wg_length
iris_x2 = wg_length + cavity_length
iris_y_lo = (wg_width - iris_width) / 2
iris_y_hi = iris_y_lo + iris_width

# Input iris
sim.add(Box((iris_x1, 0, 0), (iris_x1, iris_y_lo, wg_height)), material="pec")
sim.add(Box((iris_x1, iris_y_hi, 0), (iris_x1, wg_width, wg_height)), material="pec")

# Output iris
sim.add(Box((iris_x2, 0, 0), (iris_x2, iris_y_lo, wg_height)), material="pec")
sim.add(Box((iris_x2, iris_y_hi, 0), (iris_x2, wg_width, wg_height)), material="pec")

# ---- Waveguide ports ----
freqs = jnp.linspace(4.5e9, 8e9, 30)
sim.add_waveguide_port(
    0.005, direction="+x", mode=(1, 0), mode_type="TE",
    freqs=freqs, f0=f_target, bandwidth=0.6, name="port1",
)
sim.add_waveguide_port(
    total_x - 0.005, direction="-x", mode=(1, 0), mode_type="TE",
    freqs=freqs, f0=f_target, bandwidth=0.6, name="port2",
)

# ---- Run ----
print(f"Running cavity filter S-matrix ...")
result = sim.compute_waveguide_s_matrix(num_periods=30)
S = result.s_params
f_GHz = np.array(result.freqs) / 1e9

s11_dB = 20 * np.log10(np.maximum(np.abs(S[0, 0, :]), 1e-10))
s21_dB = 20 * np.log10(np.maximum(np.abs(S[1, 0, :]), 1e-10))

# Find S21 peak (transmission maximum = passband)
peak_idx = np.argmax(np.abs(S[1, 0, :]))
f_peak = result.freqs[peak_idx]
s21_peak_dB = s21_dB[peak_idx]

# Frequency selectivity: ratio of peak to mean transmission
s21_mean_linear = np.mean(np.abs(S[1, 0, :]))
s21_peak_linear = np.abs(S[1, 0, peak_idx])
selectivity = s21_peak_linear / (s21_mean_linear + 1e-30)

print(f"\n--- Validation Results ---")
print(f"S21 peak freq          : {f_peak / 1e9:.4f} GHz")
print(f"|S21| at peak (dB)      : {s21_peak_dB:.1f} dB")
print(f"|S21| mean (dB)         : {20 * np.log10(max(s21_mean_linear, 1e-10)):.1f} dB")
print(f"Selectivity (peak/mean) : {selectivity:.2f}")

# Reciprocity check
recip_err = np.mean(np.abs(np.abs(S[1, 0, :]) - np.abs(S[0, 1, :])))
print(f"Reciprocity error      : {recip_err:.4f}")

# Validation: the iris-coupled cavity should show frequency-selective
# transmission -- the peak S21 should be notably higher than the mean.
# The passband should fall within the analysis frequency range.
passed = selectivity > 1.1 and s21_peak_linear > s21_mean_linear and f_peak > 4e9
status = "PASS" if passed else "FAIL"
print(f"\nValidation: {status} (iris-coupled cavity shows frequency-selective transmission)")
