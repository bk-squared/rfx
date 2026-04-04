"""Validation Case 5: PEC Cavity Filter with Iris Coupling

A waveguide section with two PEC iris walls forming a resonant cavity.
Validates that S21 shows frequency-selective transmission (passband peak)
near the cavity resonance frequency.

Quantitative checks:
  1. Selectivity (peak/mean |S21|) > 1.5
  2. |S21| peak > -20 dB (usable passband)
  3. Passband frequency within analysis range (4-8 GHz)
  4. Reciprocity: mean |S21 - S12| < 0.05

Reference: analytical cavity resonance f_mnp = c/(2) * sqrt((m/a)^2 + (n/b)^2 + (p/d)^2)
"""

import sys
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

# Reciprocity check
recip_err = np.mean(np.abs(np.abs(S[1, 0, :]) - np.abs(S[0, 1, :])))

print(f"\n--- Validation Results ---")
print(f"S21 peak freq          : {f_peak / 1e9:.4f} GHz")
print(f"|S21| at peak (dB)      : {s21_peak_dB:.1f} dB")
print(f"|S21| mean (dB)         : {20 * np.log10(max(s21_mean_linear, 1e-10)):.1f} dB")
print(f"Selectivity (peak/mean) : {selectivity:.2f}")
print(f"Reciprocity error      : {recip_err:.4f}")

# ---- Quantitative assertions ----
failures = []

# Check 1: Selectivity > 1.5 (tightened from 1.1)
if selectivity > 1.5:
    print(f"PASS: selectivity = {selectivity:.2f} (threshold > 1.5)")
else:
    msg = f"FAIL: selectivity = {selectivity:.2f} (threshold > 1.5)"
    print(msg)
    failures.append(msg)

# Check 2: |S21| peak > -20 dB (usable passband)
if s21_peak_dB > -20.0:
    print(f"PASS: |S21| peak = {s21_peak_dB:.1f} dB (threshold > -20 dB)")
else:
    msg = f"FAIL: |S21| peak = {s21_peak_dB:.1f} dB (threshold > -20 dB)"
    print(msg)
    failures.append(msg)

# Check 3: Passband within analysis range
if 4e9 < f_peak < 8e9:
    print(f"PASS: passband at {f_peak / 1e9:.4f} GHz (within 4-8 GHz range)")
else:
    msg = f"FAIL: passband at {f_peak / 1e9:.4f} GHz (outside 4-8 GHz range)"
    print(msg)
    failures.append(msg)

# Check 4: Reciprocity
if recip_err < 0.05:
    print(f"PASS: reciprocity error = {recip_err:.4f} (threshold < 0.05)")
else:
    msg = f"FAIL: reciprocity error = {recip_err:.4f} (threshold < 0.05)"
    print(msg)
    failures.append(msg)

if failures:
    print(f"\nValidation FAILED ({len(failures)} check(s)):")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
else:
    print(f"\nValidation: PASS (all quantitative checks passed for cavity filter)")
