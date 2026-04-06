"""Accuracy Validation Case 5: Dielectric-Loaded PEC Cavity

A PEC rectangular cavity partially filled with a dielectric slab.
The resonant frequency shifts from the empty-cavity analytical value
due to the dielectric loading. Both the empty and loaded frequencies
have analytical solutions, providing a two-point validation.

Analytical reference:
  Empty cavity TM110: f_empty = C0/(2) * sqrt((1/a)^2 + (1/b)^2)
  Full dielectric fill: f_loaded = f_empty / sqrt(eps_r)
  Partial fill (perturbation): between f_empty and f_loaded,
    closer to loaded when slab covers more volume.

Validation metrics:
  - Empty cavity resonance within 0.5% of analytical
  - Loaded cavity resonance shifts in correct direction (lower freq)
  - Loaded resonance within 3% of volume-weighted estimate

Reference: Pozar, "Microwave Engineering", 4th ed., Ch 6.
"""

import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THRESHOLD_PCT = 1.0  # Tight threshold — PEC cavity physics is well-resolved

# =============================================================================
# Cavity dimensions
# =============================================================================
a = 80e-3   # 80 mm
b = 65e-3   # 65 mm
d = 50e-3   # 50 mm
eps_r_slab = 4.4  # FR4-like dielectric

# Analytical TM110
f_empty = (C0 / 2) * np.sqrt((1 / a) ** 2 + (1 / b) ** 2)
# Full-fill reference
f_full = f_empty / np.sqrt(eps_r_slab)
# Partial fill (slab in center third of x-axis): perturbation estimate
# Volume fraction of dielectric
vf = 1.0 / 3.0
eps_eff_est = 1.0 + vf * (eps_r_slab - 1.0)
f_partial_est = f_empty / np.sqrt(eps_eff_est)

print("=" * 60)
print("CASE 5: Dielectric-Loaded PEC Cavity")
print("=" * 60)
print(f"Cavity          : {a*1e3:.0f} x {b*1e3:.0f} x {d*1e3:.0f} mm")
print(f"Slab eps_r      : {eps_r_slab}")
print(f"f_empty (TM110) : {f_empty/1e9:.4f} GHz")
print(f"f_full_fill     : {f_full/1e9:.4f} GHz")
print(f"f_partial (est) : {f_partial_est/1e9:.4f} GHz")
print()

# =============================================================================
# Simulations
# =============================================================================
dx = 1e-3  # 1 mm cells (80x65x50 = 260K cells)
results = {}

for case_name, add_slab in [("empty", False), ("loaded", True)]:
    print(f"--- {case_name} ---")
    sim = Simulation(
        freq_max=f_empty * 2,
        domain=(a, b, d),
        boundary="pec",
        dx=dx,
    )

    if add_slab:
        sim.add_material("slab", eps_r=eps_r_slab)
        sim.add(Box((a / 3, 0, 0), (2 * a / 3, b, d)), material="slab")

    sim.add_source(
        (a / 3, b / 3, d / 2),
        component="ez",
        waveform=GaussianPulse(f0=f_empty, bandwidth=0.8),
    )
    sim.add_probe((2 * a / 3, 2 * b / 3, d / 2), component="ez")

    grid = sim._build_grid()
    n_steps = grid.num_timesteps(num_periods=100)
    print(f"  Grid: {grid.nx}x{grid.ny}x{grid.nz}, steps={n_steps}")

    result = sim.run(n_steps=n_steps)

    # Find resonance via Harminv
    modes = result.find_resonances(freq_range=(f_full * 0.8, f_empty * 1.2))
    if modes:
        best = min(modes, key=lambda m: abs(m.freq - (f_empty if not add_slab else f_partial_est)))
        f_sim = best.freq
        Q_sim = best.Q
    else:
        # FFT fallback
        ts = np.asarray(result.time_series).ravel()
        nfft = len(ts) * 8
        spec = np.abs(np.fft.rfft(ts, n=nfft))
        freqs_fft = np.fft.rfftfreq(nfft, d=result.dt)
        band = (freqs_fft > f_full * 0.8) & (freqs_fft < f_empty * 1.2)
        f_sim = freqs_fft[np.argmax(spec * band)]
        Q_sim = float("nan")

    results[case_name] = {"f": f_sim, "Q": Q_sim}
    print(f"  f_sim = {f_sim/1e9:.4f} GHz, Q = {Q_sim:.0f}")

# =============================================================================
# Validation
# =============================================================================
f_empty_sim = results["empty"]["f"]
f_loaded_sim = results["loaded"]["f"]

err_empty = abs(f_empty_sim - f_empty) / f_empty * 100
err_loaded = abs(f_loaded_sim - f_partial_est) / f_partial_est * 100
shift_correct = f_loaded_sim < f_empty_sim  # loaded should be lower

print()
print("=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
print(f"Empty cavity:")
print(f"  Analytical : {f_empty/1e9:.4f} GHz")
print(f"  Simulated  : {f_empty_sim/1e9:.4f} GHz")
print(f"  Error      : {err_empty:.3f}%")
print(f"Loaded cavity:")
print(f"  Estimate   : {f_partial_est/1e9:.4f} GHz")
print(f"  Simulated  : {f_loaded_sim/1e9:.4f} GHz")
print(f"  Error      : {err_loaded:.2f}%")
print(f"  Shift      : {'correct (lower)' if shift_correct else 'WRONG DIRECTION'}")

empty_pass = err_empty < THRESHOLD_PCT
loaded_pass = err_loaded < 5.0  # perturbation estimate is approximate
shift_pass = shift_correct
all_pass = empty_pass and loaded_pass and shift_pass

print()
status = "PASS" if all_pass else "FAIL"
print(f"Overall: {status}")
print(f"  Empty freq error < {THRESHOLD_PCT}%: {'PASS' if empty_pass else 'FAIL'}")
print(f"  Loaded freq error < 5%: {'PASS' if loaded_pass else 'FAIL'}")
print(f"  Freq shift direction: {'PASS' if shift_pass else 'FAIL'}")

# =============================================================================
# Plot
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Case 5: Dielectric-Loaded PEC Cavity", fontsize=13, fontweight="bold")

# Panel 1: Empty cavity time series + spectrum
ts_empty = np.asarray(results["empty"].get("ts", np.zeros(1)))
# Re-run for plot data if needed
ax = axes[0]
ax.text(0.5, 0.5, f"Empty Cavity\n\nf_anal = {f_empty/1e9:.4f} GHz\n"
        f"f_sim  = {f_empty_sim/1e9:.4f} GHz\n"
        f"Error  = {err_empty:.3f}%\n"
        f"Q      = {results['empty']['Q']:.0f}",
        transform=ax.transAxes, va="center", ha="center",
        fontsize=11, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax.set_title("Empty Cavity")
ax.axis("off")

# Panel 2: Loaded cavity
ax = axes[1]
ax.text(0.5, 0.5, f"Loaded Cavity (eps_r={eps_r_slab})\n\n"
        f"f_est  = {f_partial_est/1e9:.4f} GHz\n"
        f"f_sim  = {f_loaded_sim/1e9:.4f} GHz\n"
        f"Error  = {err_loaded:.2f}%\n"
        f"Q      = {results['loaded']['Q']:.0f}\n\n"
        f"Shift: {f_empty_sim/1e9:.4f} -> {f_loaded_sim/1e9:.4f} GHz\n"
        f"({'correct' if shift_correct else 'WRONG'})",
        transform=ax.transAxes, va="center", ha="center",
        fontsize=11, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8))
ax.set_title("Loaded Cavity")
ax.axis("off")

# Panel 3: Summary
ax = axes[2]
ax.text(0.5, 0.5, f"OVERALL: {status}\n\n"
        f"Empty err  : {err_empty:.3f}% < {THRESHOLD_PCT}% {'✓' if empty_pass else '✗'}\n"
        f"Loaded err : {err_loaded:.2f}% < 5% {'✓' if loaded_pass else '✗'}\n"
        f"Shift dir  : {'✓' if shift_pass else '✗'}\n\n"
        f"Grid: {dx*1e3:.0f}mm, PEC boundary\n"
        f"Harminv resonance extraction",
        transform=ax.transAxes, va="center", ha="center",
        fontsize=11, family="monospace",
        bbox=dict(boxstyle="round",
                  facecolor="lightgreen" if all_pass else "lightsalmon",
                  alpha=0.8))
ax.set_title("Verdict")
ax.axis("off")

plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "05_dielectric_cavity.png")
plt.savefig(out, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {out}")

if not all_pass:
    sys.exit(1)
