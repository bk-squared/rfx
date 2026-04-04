"""Accuracy Validation Case 2: Rectangular Waveguide TE10 Propagation

WR-90 standard waveguide (a=22.86 mm, b=10.16 mm).
TE10 cutoff frequency: f_c = c / (2*a) = 6.557 GHz.
Propagation at 10 GHz: beta, guided wavelength, S21 phase.

Analytical reference:
  - f_c = c / (2*a)
  - beta = sqrt(k0^2 - kc^2) where kc = pi/a, k0 = 2*pi*f/c
  - lambda_g = 2*pi / beta
  - Phase shift over length L: phi = beta * L

Validation metric: |f_c_sim - f_c_analytical| / f_c_analytical < 1%

Reference: Pozar, "Microwave Engineering", 4th ed., Ch 3.
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
THRESHOLD_PCT = 1.0

# =============================================================================
# Analytical reference: WR-90
# =============================================================================
a = 22.86e-3   # broad dimension (m)
b = 10.16e-3   # narrow dimension (m)

# TE10 cutoff
f_c_analytical = C0 / (2 * a)
kc = np.pi / a

# Propagation at 10 GHz
f_test = 10.0e9
k0 = 2 * np.pi * f_test / C0
beta_analytical = np.sqrt(k0**2 - kc**2)
lambda_g_analytical = 2 * np.pi / beta_analytical

print("=" * 60)
print("CASE 2: WR-90 Waveguide TE10 Propagation")
print("=" * 60)
print(f"Waveguide      : WR-90, a={a*1e3:.2f} mm, b={b*1e3:.2f} mm")
print(f"f_c (TE10)     : {f_c_analytical / 1e9:.4f} GHz")
print(f"f_test         : {f_test / 1e9:.1f} GHz")
print(f"beta           : {beta_analytical:.2f} rad/m")
print(f"lambda_g       : {lambda_g_analytical * 1e3:.2f} mm")
print()

# =============================================================================
# FDTD simulation: PEC cavity to extract TE10x resonances
# =============================================================================
# Use a PEC-bounded waveguide section. The resonant frequencies of a
# length-L section are: f_mnp = c/2 * sqrt((m/a)^2 + (n/b)^2 + (p/Lx)^2)
# For TE10p: f_10p = c/2 * sqrt((1/a)^2 + (p/Lx)^2)
# As Lx -> large, the lowest TE10p approaches f_c.
# We use multiple resonances to extract the dispersion relation.

resolutions = [
    ("coarse", 2.0e-3),
    ("medium", 1.0e-3),
]

results_list = []

for label, dx in resolutions:
    print(f"--- Resolution: {label} (dx={dx*1e3:.2f} mm) ---")

    # Waveguide length: long enough for several TE10p modes
    Lx = 80e-3  # 80 mm

    sim = Simulation(
        freq_max=12e9,
        domain=(Lx, a, b),
        boundary="pec",
        dx=dx,
    )

    # Excite TE10-like modes with a source offset from center
    src_waveform = ModulatedGaussian(f0=f_c_analytical * 1.3, bandwidth=0.8)
    sim.add_source((Lx / 3, a / 3, b / 2), component="ez", waveform=src_waveform)
    sim.add_probe((2 * Lx / 3, 2 * a / 3, b / 2), component="ez")

    # Run long enough for good frequency resolution
    grid = sim._build_grid()
    n_steps = int(np.ceil(80.0 / (f_c_analytical * grid.dt)))
    print(f"  Steps: {n_steps}, dt={grid.dt*1e12:.3f} ps")

    result = sim.run(n_steps=n_steps)

    # Extract resonances near TE10 cutoff
    modes = result.find_resonances(
        freq_range=(f_c_analytical * 0.7, f_c_analytical * 2.0),
        probe_idx=0,
    )

    if modes:
        # The lowest TE10p mode frequency approaches f_c from above
        # Sort by frequency and take the lowest
        modes_sorted = sorted(modes, key=lambda m: m.freq)
        f_lowest = modes_sorted[0].freq

        # For TE10p modes in a PEC box of length Lx:
        # f_10p = c/2 * sqrt((1/a)^2 + (p/Lx)^2)
        # Extract multiple modes and fit to get f_c
        mode_freqs = np.array([m.freq for m in modes_sorted])

        # Analytical TE10p frequencies for comparison
        p_vals = np.arange(1, len(mode_freqs) + 1)
        f_10p_analytical = (C0 / 2) * np.sqrt((1/a)**2 + (p_vals / Lx)**2)

        # Match simulated modes to analytical p-indices
        # For each sim mode, find best matching analytical p
        best_matches = []
        for f_s in mode_freqs[:6]:  # use up to 6 modes
            errs = np.abs(f_10p_analytical - f_s) / f_10p_analytical
            best_p = np.argmin(errs)
            best_matches.append((p_vals[best_p], f_s, f_10p_analytical[best_p]))

        # Extract cutoff by fitting: f^2 = f_c^2 + (c*p/(2*Lx))^2
        # Least-squares fit for f_c^2
        if len(best_matches) >= 2:
            f_sq = np.array([m[1]**2 for m in best_matches])
            p_sq = np.array([(C0 * m[0] / (2 * Lx))**2 for m in best_matches])
            # f^2 = f_c^2 + p_sq => intercept = f_c^2
            from numpy.polynomial import polynomial as P
            coeffs = np.polyfit(p_sq, f_sq, 1)
            f_c_sq_fit = coeffs[1]
            if f_c_sq_fit > 0:
                f_c_sim = np.sqrt(f_c_sq_fit)
            else:
                f_c_sim = f_lowest
        else:
            f_c_sim = f_lowest
    else:
        # FFT fallback
        ts = np.asarray(result.time_series).ravel()
        nfft = len(ts) * 8
        spectrum = np.abs(np.fft.rfft(ts, n=nfft))
        freqs_fft = np.fft.rfftfreq(nfft, d=result.dt)
        band = (freqs_fft > f_c_analytical * 0.7) & (freqs_fft < f_c_analytical * 1.3)
        f_c_sim = freqs_fft[np.argmax(spectrum * band)]
        mode_freqs = np.array([f_c_sim])
        best_matches = []

    err_pct = abs(f_c_sim - f_c_analytical) / f_c_analytical * 100
    print(f"  f_c_sim = {f_c_sim / 1e9:.4f} GHz, error = {err_pct:.3f}%")

    results_list.append({
        "label": label,
        "dx": dx,
        "f_c_sim": f_c_sim,
        "err_pct": err_pct,
        "n_modes": len(modes) if modes else 0,
        "mode_freqs": mode_freqs if modes else np.array([]),
        "best_matches": best_matches,
        "n_steps": n_steps,
    })

# =============================================================================
# Summary and pass/fail
# =============================================================================
print()
print("=" * 60)
print("CONVERGENCE SUMMARY")
print("=" * 60)
print(f"{'Resolution':<12} {'dx (mm)':<10} {'f_c (GHz)':<14} {'Error %':<10} {'Modes':<8}")
print("-" * 54)
for r in results_list:
    print(f"{r['label']:<12} {r['dx']*1e3:<10.2f} {r['f_c_sim']/1e9:<14.4f} "
          f"{r['err_pct']:<10.3f} {r['n_modes']:<8}")

best_result = results_list[-1]
final_err = best_result["err_pct"]

print()
print(f"Analytical f_c   : {f_c_analytical / 1e9:.4f} GHz")
print(f"Best simulation  : {best_result['f_c_sim'] / 1e9:.4f} GHz")
print(f"Error            : {final_err:.3f}%")
print(f"Threshold        : {THRESHOLD_PCT}%")

# =============================================================================
# Comparison plot
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Case 2: WR-90 Waveguide TE10 Cutoff", fontsize=13, fontweight="bold")

# Panel 1: Grid convergence
ax = axes[0]
dx_vals = [r["dx"] * 1e3 for r in results_list]
err_vals = [r["err_pct"] for r in results_list]
ax.plot(dx_vals, err_vals, "bo-", markersize=8, linewidth=2)
ax.axhline(THRESHOLD_PCT, color="r", ls="--", label=f"Threshold {THRESHOLD_PCT}%")
ax.set_xlabel("dx (mm)")
ax.set_ylabel("Cutoff frequency error (%)")
ax.set_title("Grid convergence")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Dispersion diagram from best run
ax = axes[1]
bm = best_result["best_matches"]
if bm:
    p_plot = np.array([m[0] for m in bm])
    f_sim_plot = np.array([m[1] / 1e9 for m in bm])
    f_ana_plot = np.array([m[2] / 1e9 for m in bm])
    ax.plot(p_plot, f_ana_plot, "gs--", label="Analytical TE10p", markersize=6)
    ax.plot(p_plot, f_sim_plot, "ro-", label="rfx FDTD", markersize=6)
    ax.axhline(f_c_analytical / 1e9, color="b", ls=":", lw=1.5,
               label=f"f_c = {f_c_analytical/1e9:.3f} GHz")
ax.set_xlabel("Mode index p")
ax.set_ylabel("Frequency (GHz)")
ax.set_title("TE10p mode dispersion")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3: Summary
ax = axes[2]
ax.axis("off")
lines = [
    "WR-90 TE10 Validation",
    "-" * 30,
    f"Reference: Pozar Ch 3",
    f"a = {a*1e3:.2f} mm, b = {b*1e3:.2f} mm",
    "",
    f"f_c analytical = {f_c_analytical/1e9:.4f} GHz",
    f"f_c simulated  = {best_result['f_c_sim']/1e9:.4f} GHz",
    f"Error          = {final_err:.3f}%",
    f"Threshold      = {THRESHOLD_PCT}%",
    f"Verdict        = {'PASS' if final_err < THRESHOLD_PCT else 'FAIL'}",
    "",
    f"beta (10 GHz)  = {beta_analytical:.2f} rad/m",
    f"lambda_g       = {lambda_g_analytical*1e3:.2f} mm",
]
ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
        va="top", ha="left", fontsize=10, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "02_waveguide_te10.png")
plt.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {out_path}")

if final_err < THRESHOLD_PCT:
    print(f"\nPASS: TE10 cutoff error {final_err:.3f}% < {THRESHOLD_PCT}%")
    sys.exit(0)
else:
    print(f"\nFAIL: TE10 cutoff error {final_err:.3f}% >= {THRESHOLD_PCT}%")
    sys.exit(1)
