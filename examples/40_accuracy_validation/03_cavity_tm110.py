"""Accuracy Validation Case 3: Dielectric-Filled PEC Cavity TM110

PEC rectangular cavity partially or fully filled with dielectric.
Dimensions: a=50 mm, b=40 mm, d=20 mm, eps_r=2.2 (PTFE-like).

Analytical TM110 resonant frequency for a uniformly filled cavity:
  f_110 = c / (2*pi*sqrt(mu*eps)) * sqrt((pi/a)^2 + (pi/b)^2)
        = c / (2*sqrt(eps_r)) * sqrt((1/a)^2 + (1/b)^2)

Validation metric: |f_sim - f_analytical| / f_analytical < 2%

Reference: Pozar, "Microwave Engineering", Ch 6; Harrington, "Time-Harmonic
Electromagnetic Fields", cavity resonator theory.
"""

import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation, Box
from rfx.sources.sources import ModulatedGaussian
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THRESHOLD_PCT = 2.0

# =============================================================================
# Analytical reference
# =============================================================================
a = 50e-3   # x-dimension (m)
b = 40e-3   # y-dimension (m)
d = 20e-3   # z-dimension (m)
eps_r = 2.2  # PTFE / Duroid-like

# TM110 in dielectric-filled PEC cavity
# f_mnp = c/(2*sqrt(eps_r)) * sqrt((m/a)^2 + (n/b)^2 + (p/d)^2)
# TM110: m=1, n=1, p=0
f_analytical = (C0 / (2 * np.sqrt(eps_r))) * np.sqrt((1/a)**2 + (1/b)**2)

# Empty cavity for comparison
f_empty = (C0 / 2) * np.sqrt((1/a)**2 + (1/b)**2)

print("=" * 60)
print("CASE 3: Dielectric-Filled PEC Cavity TM110")
print("=" * 60)
print(f"Cavity         : {a*1e3:.0f} x {b*1e3:.0f} x {d*1e3:.0f} mm")
print(f"Dielectric     : eps_r = {eps_r}")
print(f"f_empty TM110  : {f_empty / 1e9:.4f} GHz")
print(f"f_filled TM110 : {f_analytical / 1e9:.4f} GHz")
print(f"Ratio          : {f_analytical / f_empty:.4f} (expect 1/sqrt(eps_r) = {1/np.sqrt(eps_r):.4f})")
print()

# =============================================================================
# Grid convergence: two resolutions
# =============================================================================
resolutions = [
    ("coarse", 2.0e-3),
    ("medium", 1.0e-3),
]

results_list = []

for label, dx in resolutions:
    print(f"--- Resolution: {label} (dx={dx*1e3:.2f} mm) ---")

    sim = Simulation(
        freq_max=f_empty * 1.5,
        domain=(a, b, d),
        boundary="pec",
        dx=dx,
    )

    # Fill entire cavity with dielectric
    sim.add_material("dielectric", eps_r=eps_r)
    sim.add(Box((0, 0, 0), (a, b, d)), material="dielectric")

    # Excitation: ModulatedGaussian near analytical frequency
    src_waveform = ModulatedGaussian(f0=f_analytical, bandwidth=0.8)
    sim.add_source((a / 3, b / 3, d / 2), component="ez", waveform=src_waveform)
    sim.add_probe((2 * a / 3, 2 * b / 3, d / 2), component="ez")

    # Run enough for Harminv (80+ periods)
    grid = sim._build_grid()
    n_steps = int(np.ceil(100.0 / (f_analytical * grid.dt)))
    print(f"  Steps: {n_steps}, dt={grid.dt*1e12:.3f} ps")

    result = sim.run(n_steps=n_steps)

    # Resonance extraction
    modes = result.find_resonances(
        freq_range=(f_analytical * 0.5, f_analytical * 1.5),
        probe_idx=0,
    )

    if modes:
        best = min(modes, key=lambda m: abs(m.freq - f_analytical))
        f_sim = best.freq
        Q_sim = best.Q
    else:
        # FFT fallback
        ts = np.asarray(result.time_series).ravel()
        nfft = len(ts) * 8
        spectrum = np.abs(np.fft.rfft(ts, n=nfft))
        freqs_fft = np.fft.rfftfreq(nfft, d=result.dt)
        band = (freqs_fft > f_analytical * 0.5) & (freqs_fft < f_analytical * 1.5)
        f_sim = freqs_fft[np.argmax(spectrum * band)]
        Q_sim = float("nan")

    err_pct = abs(f_sim - f_analytical) / f_analytical * 100
    print(f"  f_sim = {f_sim / 1e9:.4f} GHz, error = {err_pct:.3f}%")
    if not np.isnan(Q_sim):
        print(f"  Q = {Q_sim:.1f}")

    results_list.append({
        "label": label,
        "dx": dx,
        "f_sim": f_sim,
        "err_pct": err_pct,
        "Q": Q_sim,
        "n_steps": n_steps,
    })

# =============================================================================
# Summary and pass/fail
# =============================================================================
print()
print("=" * 60)
print("CONVERGENCE SUMMARY")
print("=" * 60)
print(f"{'Resolution':<12} {'dx (mm)':<10} {'f_sim (GHz)':<14} {'Error %':<10}")
print("-" * 46)
for r in results_list:
    print(f"{r['label']:<12} {r['dx']*1e3:<10.2f} {r['f_sim']/1e9:<14.4f} {r['err_pct']:<10.3f}")

best_result = results_list[-1]
final_err = best_result["err_pct"]

print()
print(f"Analytical TM110 : {f_analytical / 1e9:.4f} GHz")
print(f"Best simulation  : {best_result['f_sim'] / 1e9:.4f} GHz")
print(f"Error            : {final_err:.3f}%")
print(f"Threshold        : {THRESHOLD_PCT}%")

# =============================================================================
# Comparison plot
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Case 3: Dielectric-Filled Cavity TM110", fontsize=13, fontweight="bold")

# Panel 1: Grid convergence
ax = axes[0]
dx_vals = [r["dx"] * 1e3 for r in results_list]
err_vals = [r["err_pct"] for r in results_list]
ax.plot(dx_vals, err_vals, "bo-", markersize=8, linewidth=2)
ax.axhline(THRESHOLD_PCT, color="r", ls="--", label=f"Threshold {THRESHOLD_PCT}%")
ax.set_xlabel("dx (mm)")
ax.set_ylabel("Frequency error (%)")
ax.set_title("Grid convergence")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Spectrum from best run
ax = axes[1]
ts_arr = np.asarray(result.time_series).ravel()
nfft = len(ts_arr) * 8
spectrum = np.abs(np.fft.rfft(ts_arr, n=nfft))
freqs_fft = np.fft.rfftfreq(nfft, d=result.dt) / 1e9
spec_db = 20 * np.log10(np.maximum(spectrum / (spectrum.max() or 1.0), 1e-10))
band_mask = (freqs_fft > 1.0) & (freqs_fft < 8.0)
ax.plot(freqs_fft[band_mask], spec_db[band_mask], "b-", lw=1.0)
ax.axvline(f_analytical / 1e9, color="g", ls="--", lw=1.5,
           label=f"Analytical {f_analytical/1e9:.3f} GHz")
ax.axvline(f_empty / 1e9, color="orange", ls=":", lw=1.5,
           label=f"Empty {f_empty/1e9:.3f} GHz")
ax.axvline(best_result["f_sim"] / 1e9, color="r", ls=":", lw=1.5,
           label=f"rfx {best_result['f_sim']/1e9:.3f} GHz")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Normalized (dB)")
ax.set_title("Frequency spectrum (medium grid)")
ax.set_ylim(-60, 5)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Panel 3: Summary
ax = axes[2]
ax.axis("off")
lines = [
    "Dielectric Cavity Validation",
    "-" * 32,
    f"Reference: Pozar Ch 6",
    f"Cavity: {a*1e3:.0f} x {b*1e3:.0f} x {d*1e3:.0f} mm",
    f"eps_r = {eps_r}",
    "",
    f"f_empty  = {f_empty/1e9:.4f} GHz",
    f"f_filled = {f_analytical/1e9:.4f} GHz",
    f"f_sim    = {best_result['f_sim']/1e9:.4f} GHz",
    f"Error    = {final_err:.3f}%",
    f"Threshold= {THRESHOLD_PCT}%",
    f"Verdict  = {'PASS' if final_err < THRESHOLD_PCT else 'FAIL'}",
]
ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
        va="top", ha="left", fontsize=10, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "03_cavity_tm110.png")
plt.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {out_path}")

if final_err < THRESHOLD_PCT:
    print(f"\nPASS: Cavity TM110 error {final_err:.3f}% < {THRESHOLD_PCT}%")
    sys.exit(0)
else:
    print(f"\nFAIL: Cavity TM110 error {final_err:.3f}% >= {THRESHOLD_PCT}%")
    sys.exit(1)
