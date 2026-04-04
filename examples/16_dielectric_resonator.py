"""Validation Case 3: Dielectric Resonator

High-permittivity dielectric sphere in a PEC cavity.
Validates resonant frequency extraction via Harminv against
analytical formula for a dielectric-loaded cavity.

Quantitative checks:
  1. Simulated resonance below empty-cavity TM110
  2. Simulated resonance within 15% of perturbation-theory estimate
  3. Q factor positive (if Harminv available)

Reference: For a dielectric sphere of radius a with eps_r in a PEC cavity,
resonant modes shift proportionally to sqrt(eps_r). We compare against
the known empty-cavity TM110 mode scaled by the dielectric loading.
"""

import sys
import matplotlib
matplotlib.use("Agg")
import numpy as np

from rfx import Simulation, Sphere
from rfx.sources.sources import ModulatedGaussian
from rfx.grid import C0

# ---- Cavity parameters ----
a = 0.06    # cavity x-dimension (m)
b = 0.06    # cavity y-dimension
d = 0.04    # cavity z-dimension

# Empty cavity TM110 frequency
f_empty = (C0 / 2) * np.sqrt((1 / a) ** 2 + (1 / b) ** 2)

# Dielectric sphere parameters
eps_r_sphere = 9.8   # alumina-like
sphere_radius = 0.012  # 12 mm radius
sphere_center = (a / 2, b / 2, d / 2)

# The dielectric loading lowers the resonant frequency.
# Rough estimate: f_loaded ~ f_empty / sqrt(eps_r_eff)
# where eps_r_eff depends on filling fraction.
# Volume filling fraction
V_sphere = (4.0 / 3.0) * np.pi * sphere_radius ** 3
V_cavity = a * b * d
fill_fraction = V_sphere / V_cavity
eps_r_eff = 1.0 + (eps_r_sphere - 1.0) * fill_fraction
f_expected_approx = f_empty / np.sqrt(eps_r_eff)

print(f"Cavity          : {a * 1e3:.0f} x {b * 1e3:.0f} x {d * 1e3:.0f} mm")
print(f"Sphere          : r={sphere_radius * 1e3:.0f} mm, eps_r={eps_r_sphere}")
print(f"Fill fraction   : {fill_fraction * 100:.1f}%")
print(f"Empty TM110     : {f_empty / 1e9:.4f} GHz")
print(f"Expected loaded : {f_expected_approx / 1e9:.4f} GHz (approx)")

# ---- Build simulation ----
sim = Simulation(
    freq_max=f_empty * 1.5,
    domain=(a, b, d),
    boundary="pec",
    dx=0.002,
)

# Dielectric sphere
sim.add_material("alumina", eps_r=eps_r_sphere)
sim.add(Sphere(center=sphere_center, radius=sphere_radius), material="alumina")

# Excitation: soft source offset from center
src_waveform = ModulatedGaussian(f0=f_expected_approx, bandwidth=0.8)
sim.add_source((a / 3, b / 3, d / 2), component="ez", waveform=src_waveform)

# Probe
sim.add_probe((2 * a / 3, 2 * b / 3, d / 2), component="ez")

# Run enough steps for ring-down
grid = sim._build_grid()
n_steps = int(np.ceil(80.0 / (f_expected_approx * grid.dt)))
print(f"Running {n_steps} steps ...")
result = sim.run(n_steps=n_steps)

# ---- Resonance extraction ----
modes = result.find_resonances(
    freq_range=(f_expected_approx * 0.4, f_empty * 1.2),
    probe_idx=0,
)

if modes:
    best = min(modes, key=lambda m: abs(m.freq - f_expected_approx))
    f_sim = best.freq
    Q_sim = best.Q
    print(f"\nHarminv found {len(modes)} modes")
else:
    # FFT fallback
    ts = np.asarray(result.time_series).ravel()
    nfft = len(ts) * 8
    spectrum = np.abs(np.fft.rfft(ts, n=nfft))
    freqs_fft = np.fft.rfftfreq(nfft, d=result.dt)
    band = (freqs_fft > f_expected_approx * 0.4) & (freqs_fft < f_empty * 1.2)
    f_sim = freqs_fft[np.argmax(spectrum * band)]
    Q_sim = float("nan")
    print("\nHarminv found no modes; using FFT peak")

# ---- Validation ----
error_vs_approx = abs(f_sim - f_expected_approx) / f_expected_approx * 100

print(f"\n--- Validation Results ---")
print(f"Empty cavity TM110   : {f_empty / 1e9:.4f} GHz")
print(f"Expected (approx)    : {f_expected_approx / 1e9:.4f} GHz")
print(f"Simulated            : {f_sim / 1e9:.4f} GHz")
print(f"Error vs approx      : {error_vs_approx:.1f}%")
if not np.isnan(Q_sim):
    print(f"Q factor             : {Q_sim:.1f}")
print(f"Frequency lowered    : {'Yes' if f_sim < f_empty else 'No'}")

# ---- Quantitative assertions ----
failures = []

# Check 1: Loaded resonance must be below empty cavity TM110
if f_sim < f_empty and f_sim > 0:
    shift_pct = (f_empty - f_sim) / f_empty * 100
    print(f"PASS: f_sim = {f_sim / 1e9:.4f} GHz < f_empty = {f_empty / 1e9:.4f} GHz (shift = {shift_pct:.1f}%)")
else:
    msg = f"FAIL: f_sim = {f_sim / 1e9:.4f} GHz not below f_empty = {f_empty / 1e9:.4f} GHz"
    print(msg)
    failures.append(msg)

# Check 2: Simulated resonance within 15% of perturbation-theory estimate
if error_vs_approx < 15.0:
    print(f"PASS: resonance error = {error_vs_approx:.1f}% (reference = {f_expected_approx / 1e9:.4f} GHz, threshold = 15%)")
else:
    msg = f"FAIL: resonance error = {error_vs_approx:.1f}% (reference = {f_expected_approx / 1e9:.4f} GHz, threshold = 15%)"
    print(msg)
    failures.append(msg)

# Check 3: Q factor positive (if Harminv succeeded)
if not np.isnan(Q_sim):
    if Q_sim > 0:
        print(f"PASS: Q factor = {Q_sim:.1f} > 0")
    else:
        msg = f"FAIL: Q factor = {Q_sim:.1f} (expected > 0)"
        print(msg)
        failures.append(msg)

if failures:
    print(f"\nValidation FAILED ({len(failures)} check(s)):")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
else:
    print(f"\nValidation: PASS (all quantitative checks passed for dielectric resonator)")
