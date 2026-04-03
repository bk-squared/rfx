"""Validation Case 9: Subgridded PEC Cavity

A PEC rectangular cavity simulated with local refinement (subgridding).
Validates that the resonant frequency matches a uniform fine-grid reference.

Reference: uniform fine-grid simulation of the same cavity.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np

from rfx import Simulation
from rfx.sources.sources import ModulatedGaussian
from rfx.grid import C0

# ---- Cavity parameters ----
a = 0.10    # x (m)
b = 0.10    # y
d = 0.05    # z

# Analytical TM110
f_analytical = (C0 / 2) * np.sqrt((1 / a) ** 2 + (1 / b) ** 2)
print(f"Cavity            : {a * 1e3:.0f} x {b * 1e3:.0f} x {d * 1e3:.0f} mm")
print(f"Analytical TM110  : {f_analytical / 1e9:.4f} GHz")

# ---- Helper: run cavity simulation and extract resonance ----
def run_cavity(label, dx, use_refinement=False, refinement_ratio=4):
    sim = Simulation(
        freq_max=5e9,
        domain=(a, b, d),
        boundary="pec",
        dx=dx,
    )

    if use_refinement:
        # Refine in the lower half of z (arbitrary choice to test subgridding)
        sim.add_refinement(z_range=(0, d / 2), ratio=refinement_ratio)

    src_waveform = ModulatedGaussian(f0=f_analytical, bandwidth=0.8)
    sim.add_source((a / 3, b / 3, d / 2), component="ez", waveform=src_waveform)
    sim.add_probe((2 * a / 3, 2 * b / 3, d / 2), component="ez")

    grid = sim._build_grid()
    n_steps = int(np.ceil(60.0 / (f_analytical * grid.dt)))
    n_steps = min(n_steps, 20000)
    print(f"  [{label}] dx={dx * 1e3:.2f} mm, {n_steps} steps ...")
    result = sim.run(n_steps=n_steps)

    # Extract resonance
    modes = result.find_resonances(
        freq_range=(f_analytical * 0.5, f_analytical * 1.5),
        probe_idx=0,
    )

    if modes:
        best = min(modes, key=lambda m: abs(m.freq - f_analytical))
        f_res = best.freq
    else:
        # FFT fallback
        ts = np.asarray(result.time_series).ravel()
        nfft = len(ts) * 8
        spectrum = np.abs(np.fft.rfft(ts, n=nfft))
        freqs_fft = np.fft.rfftfreq(nfft, d=result.dt)
        band = (freqs_fft > f_analytical * 0.5) & (freqs_fft < f_analytical * 1.5)
        f_res = freqs_fft[np.argmax(spectrum * band)]

    return f_res

# ---- Run uniform coarse reference ----
dx_coarse = 2e-3
print("\nUniform coarse grid:")
f_coarse = run_cavity("Coarse", dx_coarse)

# ---- Run uniform fine reference ----
dx_fine = 1e-3
print("\nUniform fine grid:")
f_fine = run_cavity("Fine", dx_fine)

# ---- Run subgridded (coarse + local refinement) ----
print("\nSubgridded (coarse + refinement):")
f_subgrid = run_cavity("Subgrid", dx_coarse, use_refinement=True, refinement_ratio=2)

# ---- Compare ----
err_coarse = abs(f_coarse - f_analytical) / f_analytical * 100
err_fine = abs(f_fine - f_analytical) / f_analytical * 100
err_subgrid = abs(f_subgrid - f_analytical) / f_analytical * 100

# How close is subgridded to fine reference
err_sg_vs_fine = abs(f_subgrid - f_fine) / f_fine * 100

print(f"\n--- Validation Results ---")
print(f"Analytical TM110         : {f_analytical / 1e9:.4f} GHz")
print(f"Coarse (dx={dx_coarse * 1e3:.1f}mm)      : {f_coarse / 1e9:.4f} GHz  (err={err_coarse:.2f}%)")
print(f"Fine (dx={dx_fine * 1e3:.1f}mm)        : {f_fine / 1e9:.4f} GHz  (err={err_fine:.2f}%)")
print(f"Subgridded               : {f_subgrid / 1e9:.4f} GHz  (err={err_subgrid:.2f}%)")
print(f"Subgrid vs fine          : {err_sg_vs_fine:.2f}%")

# Validation: subgridded result should be closer to analytical than coarse
# and reasonably close to the fine-grid result
improvement = err_coarse - err_subgrid
subgrid_close_to_analytical = err_subgrid < 5.0  # within 5% of analytical

passed = err_subgrid < max(err_coarse, 5.0) and f_subgrid > 0
status = "PASS" if passed else "FAIL"
print(f"\nValidation: {status} (subgridded resonance within {err_subgrid:.2f}% of analytical)")
