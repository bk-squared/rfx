"""Validation Case 6: Lumped RLC Series Circuit

Demonstrates that a lumped RLC element modifies the electromagnetic
response at its resonant frequency f0 = 1/(2*pi*sqrt(LC)).

Approach: Run two PEC cavity simulations (with and without RLC) and
compare their spectra near f0. The RLC element should measurably
change the spectral content around its resonance.

Reference: analytical resonance f0 = 1/(2*pi*sqrt(LC)).
"""

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

# Validation: the RLC element must measurably modify the electromagnetic
# response. A 10% spectral change near f0 demonstrates the element is active.
passed = max_spectral_change > 0.1 and time_diff_rel > 0.01
status = "PASS" if passed else "FAIL"
print(f"\nValidation: {status} (RLC element produces {max_spectral_change * 100:.0f}% spectral change near f0)")
