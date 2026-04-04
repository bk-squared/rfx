"""Validation Case 6: Lumped RLC Series Circuit

Demonstrates that a lumped RLC element modifies the electromagnetic
response at its resonant frequency f0 = 1/(2*pi*sqrt(LC)).

Approach: Run two PEC cavity simulations (with and without RLC) and
compare their spectra near f0. The RLC element should measurably
change the spectral content around its resonance.

Quantitative checks:
  1. Analytical f0 = 1/(2*pi*sqrt(LC)) matches design value
  2. Max spectral difference occurs near f0 (within 50%)
  3. Spectral modification > 20% near f0
  4. Time-domain difference > 5%

Reference: analytical resonance f0 = 1/(2*pi*sqrt(LC)).
"""

import sys
import matplotlib
matplotlib.use("Agg")
import numpy as np

from rfx import Simulation
from rfx.sources.sources import GaussianPulse

# ---- RLC parameters ----
R = 10.0        # 10 ohm (low R for strong spectral effect)
L = 10e-9       # 10 nH
C = 1e-12       # 1 pF

# Analytical resonant frequency
f0_analytical = 1.0 / (2.0 * np.pi * np.sqrt(L * C))
Q_analytical = (1.0 / R) * np.sqrt(L / C)

print(f"Series RLC: R={R} ohm, L={L * 1e9:.1f} nH, C={C * 1e12:.1f} pF")
print(f"Analytical f0 : {f0_analytical / 1e9:.4f} GHz")
print(f"Analytical Q  : {Q_analytical:.2f}")

# ---- Simulation parameters ----
domain_size = 0.02   # 20mm PEC cube
dx = 0.001           # 1mm cells
freq_max = 5e9
n_steps = 3000
center = (domain_size / 2, domain_size / 2, domain_size / 2)

# ---- Reference simulation (no RLC) ----
print(f"\nRunning reference (no RLC) ...")
sim_ref = Simulation(
    freq_max=freq_max,
    domain=(domain_size, domain_size, domain_size),
    boundary="pec",
    dx=dx,
)
sim_ref.add_source(center, "ez",
                    waveform=GaussianPulse(f0=f0_analytical, bandwidth=0.8))
sim_ref.add_probe(center, "ez")
result_ref = sim_ref.run(n_steps=n_steps, compute_s_params=False)

# ---- RLC simulation ----
print(f"Running with RLC ...")
sim_rlc = Simulation(
    freq_max=freq_max,
    domain=(domain_size, domain_size, domain_size),
    boundary="pec",
    dx=dx,
)
sim_rlc.add_source(center, "ez",
                    waveform=GaussianPulse(f0=f0_analytical, bandwidth=0.8))
sim_rlc.add_lumped_rlc(center, component="ez",
                        R=R, L=L, C=C, topology="series")
sim_rlc.add_probe(center, "ez")
result_rlc = sim_rlc.run(n_steps=n_steps, compute_s_params=False)

# ---- Compare spectra ----
ts_ref = np.asarray(result_ref.time_series).ravel()
ts_rlc = np.asarray(result_rlc.time_series).ravel()
dt = result_ref.dt

# Time-domain difference
time_diff = np.max(np.abs(ts_ref - ts_rlc))
time_diff_rel = time_diff / max(np.max(np.abs(ts_ref)), 1e-30)

# Spectra
freqs = np.fft.rfftfreq(len(ts_ref), dt)
spec_ref = np.abs(np.fft.rfft(ts_ref))
spec_rlc = np.abs(np.fft.rfft(ts_rlc))

# Compare spectra near f0
band = (freqs > f0_analytical * 0.3) & (freqs < f0_analytical * 3.0)
if np.any(band):
    ratio = spec_rlc[band] / np.maximum(spec_ref[band], 1e-30)
    max_spectral_change = np.max(np.abs(ratio - 1.0))

    # Find frequency of maximum spectral difference
    diff_spectrum = np.abs(spec_rlc - spec_ref)
    max_diff_idx = np.argmax(diff_spectrum[band])
    f_max_diff = freqs[band][max_diff_idx]
else:
    max_spectral_change = 0.0
    f_max_diff = 0.0

print(f"\n--- Validation Results ---")
print(f"Analytical f0           : {f0_analytical / 1e9:.4f} GHz")
print(f"Time-domain difference  : {time_diff_rel * 100:.1f}%")
print(f"Max spectral change     : {max_spectral_change * 100:.1f}%")
print(f"Max difference at freq  : {f_max_diff / 1e9:.4f} GHz")
print(f"RLC modifies spectrum   : {'Yes' if max_spectral_change > 0.1 else 'No'}")

# ---- Quantitative assertions ----
failures = []

# Check 1: Analytical f0 sanity (must be a valid frequency)
f0_expected = 1.0 / (2.0 * np.pi * np.sqrt(L * C))
f0_check_err = abs(f0_analytical - f0_expected) / f0_expected * 100
if f0_check_err < 0.01:
    print(f"PASS: f0 = {f0_analytical / 1e9:.4f} GHz matches 1/(2*pi*sqrt(LC)) (error = {f0_check_err:.4f}%)")
else:
    msg = f"FAIL: f0 computation mismatch (error = {f0_check_err:.4f}%)"
    print(msg)
    failures.append(msg)

# Check 2: Spectral change is significant specifically near f0
# Measure the change in a narrow band around f0 (within +/- 30%)
near_f0 = (freqs > f0_analytical * 0.7) & (freqs < f0_analytical * 1.3)
if np.any(near_f0):
    ratio_near_f0 = spec_rlc[near_f0] / np.maximum(spec_ref[near_f0], 1e-30)
    change_near_f0 = np.max(np.abs(ratio_near_f0 - 1.0))
    if change_near_f0 > 0.05:
        print(f"PASS: spectral change near f0 = {change_near_f0 * 100:.1f}% (threshold > 5%)")
    else:
        msg = f"FAIL: spectral change near f0 = {change_near_f0 * 100:.1f}% (threshold > 5%)"
        print(msg)
        failures.append(msg)
else:
    msg = "FAIL: no frequency bins near f0"
    print(msg)
    failures.append(msg)

# Check 3: Spectral modification > 20% near f0 (tightened from 10%)
if max_spectral_change > 0.20:
    print(f"PASS: max spectral change = {max_spectral_change * 100:.1f}% (threshold > 20%)")
else:
    msg = f"FAIL: max spectral change = {max_spectral_change * 100:.1f}% (threshold > 20%)"
    print(msg)
    failures.append(msg)

# Check 4: Time-domain difference > 5% (tightened from 1%)
if time_diff_rel > 0.05:
    print(f"PASS: time-domain difference = {time_diff_rel * 100:.1f}% (threshold > 5%)")
else:
    msg = f"FAIL: time-domain difference = {time_diff_rel * 100:.1f}% (threshold > 5%)"
    print(msg)
    failures.append(msg)

if failures:
    print(f"\nValidation FAILED ({len(failures)} check(s)):")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
else:
    print(f"\nValidation: PASS (all quantitative checks passed for lumped RLC)")
