"""Accuracy Validation Case 1: Rectangular Patch Antenna (Balanis)

2.4 GHz rectangular microstrip patch antenna on FR4 substrate.
Patch dimensions from Balanis "Antenna Theory" Ch 14 formulas.

Analytical reference:
  - Patch width:  W = c / (2*f0) * sqrt(2/(eps_r+1))
  - Effective eps: eps_eff from Hammerstad formula
  - Fringe extension: dL from Hammerstad-Jensen
  - Patch length: L = c / (2*f0*sqrt(eps_eff)) - 2*dL
  - Resonant frequency should match f0 within ~5% for FDTD

Validation metric: |f_sim - f0| / f0 < 5%

Reference: C.A. Balanis, "Antenna Theory: Analysis and Design", 4th ed., Ch 14.
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
THRESHOLD_PCT = 5.0

# =============================================================================
# Analytical design (Balanis Ch 14)
# =============================================================================
f0 = 2.4e9           # design frequency
eps_r = 4.4           # FR4 relative permittivity
tan_d = 0.02          # loss tangent
h = 1.6e-3            # substrate thickness (m)

# Patch width (Balanis Eq. 14-6)
W = C0 / (2 * f0) * np.sqrt(2.0 / (eps_r + 1.0))

# Effective permittivity (Hammerstad)
eps_eff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 12.0 * h / W) ** (-0.5)

# Fringe extension (Hammerstad-Jensen)
dL = 0.412 * h * (
    (eps_eff + 0.3) * (W / h + 0.264)
    / ((eps_eff - 0.258) * (W / h + 0.8))
)

# Patch length
L = C0 / (2.0 * f0 * np.sqrt(eps_eff)) - 2.0 * dL

# Analytical resonant frequency (back-check)
f_analytical = C0 / (2.0 * (L + 2 * dL) * np.sqrt(eps_eff))

print("=" * 60)
print("CASE 1: Rectangular Patch Antenna (Balanis)")
print("=" * 60)
print(f"Design frequency : {f0 / 1e9:.4f} GHz")
print(f"Substrate        : FR4, eps_r={eps_r}, tan_d={tan_d}, h={h*1e3:.1f} mm")
print(f"Patch W          : {W * 1e3:.2f} mm")
print(f"Patch L          : {L * 1e3:.2f} mm")
print(f"eps_eff          : {eps_eff:.4f}")
print(f"dL               : {dL * 1e3:.4f} mm")
print(f"Analytical f_res : {f_analytical / 1e9:.4f} GHz")
print()

# =============================================================================
# Use a PEC cavity approach for fast resonance extraction.
# Place the patch inside a PEC box (no CPML overhead). The cavity
# introduces a small perturbation but for resonant frequency comparison
# against the Balanis formula this is acceptable and much faster.
# =============================================================================
resolutions = [
    ("medium", 0.5e-3),
    ("fine",   0.4e-3),  # 4 cells across h=1.6mm substrate
]

results_list = []

for label, dx in resolutions:
    print(f"--- Resolution: {label} (dx={dx*1e3:.2f} mm) ---")

    margin = 5e-3
    dom_x = L + 2 * margin
    dom_y = W + 2 * margin
    dom_z = h + 4e-3  # substrate + air

    sim = Simulation(
        freq_max=f0 * 2.0,
        domain=(dom_x, dom_y, dom_z),
        dx=dx,
        boundary="pec",
    )
    dz_sub = dx  # uniform grid — dx resolves substrate when dx <= h/3

    sigma_sub = 2.0 * np.pi * f0 * 8.854e-12 * eps_r * tan_d
    sim.add_material("substrate", eps_r=eps_r, sigma=sigma_sub)

    # Ground plane (one cell PEC at z=0)
    sim.add(Box((0, 0, 0), (dom_x, dom_y, dx)), material="pec")
    # FR4 substrate
    sim.add(Box((0, 0, dx), (dom_x, dom_y, h)), material="substrate")
    # Patch on top of substrate (one cell thick PEC)
    px0, py0 = margin, margin
    sim.add(Box((px0, py0, h), (px0 + L, py0 + W, h + dx)), material="pec")

    # Source: ModulatedGaussian (zero DC) inside substrate near feed point
    src_x = px0 + L / 3.0
    src_y = py0 + W / 2.0
    src_z = h / 2.0

    sim.add_source(
        (src_x, src_y, src_z),
        component="ez",
        waveform=ModulatedGaussian(f0=f0, bandwidth=0.8),
    )
    sim.add_probe((src_x, src_y, src_z), component="ez")

    # Run ~50 periods at f0 for Harminv
    grid = sim._build_grid()
    n_steps = int(np.ceil(50.0 / (f0 * grid.dt)))
    n_steps = min(n_steps, 30000)
    print(f"  Grid: {grid.nx}x{grid.ny}x{grid.nz}, dx={dx*1e3:.2f}mm, "
          f"steps={n_steps}, dt={grid.dt*1e12:.3f} ps")

    result = sim.run(n_steps=n_steps)

    # Resonance extraction
    modes = result.find_resonances(
        freq_range=(f0 * 0.5, f0 * 1.5),
        probe_idx=0,
    )

    if modes:
        best = min(modes, key=lambda m: abs(m.freq - f0))
        f_sim = best.freq
        Q_sim = best.Q
    else:
        # FFT fallback
        ts_arr = np.asarray(result.time_series).ravel()
        nfft = len(ts_arr) * 8
        spectrum = np.abs(np.fft.rfft(ts_arr, n=nfft))
        freqs_fft = np.fft.rfftfreq(nfft, d=result.dt)
        band = (freqs_fft > f0 * 0.5) & (freqs_fft < f0 * 1.5)
        f_sim = freqs_fft[np.argmax(spectrum * band)]
        Q_sim = float("nan")

    err_pct = abs(f_sim - f0) / f0 * 100
    print(f"  f_sim = {f_sim / 1e9:.4f} GHz, error = {err_pct:.2f}%")
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
    print(f"{r['label']:<12} {r['dx']*1e3:<10.2f} {r['f_sim']/1e9:<14.4f} {r['err_pct']:<10.2f}")

# Use finest resolution for pass/fail
best_result = results_list[-1]
final_err = best_result["err_pct"]

print()
print(f"Analytical reference : {f0 / 1e9:.4f} GHz (Balanis)")
print(f"Best simulation      : {best_result['f_sim'] / 1e9:.4f} GHz")
print(f"Error                : {final_err:.2f}%")
print(f"Threshold            : {THRESHOLD_PCT}%")

# =============================================================================
# Comparison plot
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Case 1: Patch Antenna Resonance (Balanis)", fontsize=13, fontweight="bold")

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
band_mask = (freqs_fft > 1.0) & (freqs_fft < 4.0)
ax.plot(freqs_fft[band_mask], spec_db[band_mask], "b-", lw=1.0)
ax.axvline(f0 / 1e9, color="g", ls="--", lw=1.5, label=f"Balanis {f0/1e9:.2f} GHz")
ax.axvline(best_result["f_sim"] / 1e9, color="r", ls=":", lw=1.5,
           label=f"rfx {best_result['f_sim']/1e9:.3f} GHz")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Normalized (dB)")
ax.set_title("Frequency spectrum (medium grid)")
ax.set_ylim(-60, 5)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3: Summary text
ax = axes[2]
ax.axis("off")
lines = [
    "Patch Antenna Validation",
    "-" * 30,
    f"Reference: Balanis Ch 14",
    f"f_design  = {f0/1e9:.4f} GHz",
    f"f_sim     = {best_result['f_sim']/1e9:.4f} GHz",
    f"Error     = {final_err:.2f}%",
    f"Threshold = {THRESHOLD_PCT}%",
    f"Verdict   = {'PASS' if final_err < THRESHOLD_PCT else 'FAIL'}",
    "",
    f"Patch: L={L*1e3:.2f} mm, W={W*1e3:.2f} mm",
    f"FR4: eps_r={eps_r}, h={h*1e3:.1f} mm",
    f"eps_eff = {eps_eff:.4f}",
]
ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
        va="top", ha="left", fontsize=10, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "01_patch_balanis.png")
plt.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {out_path}")

if final_err < THRESHOLD_PCT:
    print(f"\nPASS: Patch resonance error {final_err:.2f}% < {THRESHOLD_PCT}%")
    sys.exit(0)
else:
    print(f"\nFAIL: Patch resonance error {final_err:.2f}% >= {THRESHOLD_PCT}%")
    sys.exit(1)
