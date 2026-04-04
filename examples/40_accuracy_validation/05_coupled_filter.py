"""Accuracy Validation Case 5: Coupled-Line Bandpass Filter

Parallel-coupled microstrip bandpass filter on FR4.
Two coupled microstrip lines with quarter-wave coupling length
produce frequency-selective energy transfer.

Analytical reference:
  - Coupling length: L_c = lambda_eff / 4 at f0
  - eps_eff from Hammerstad for the line geometry
  - Center frequency: f0 = c / (4 * L_c * sqrt(eps_eff))
  - Coupling occurs near f0 with bandpass behavior

Validation metrics:
  - Spectral peak within 5% of analytical center frequency
  - Bandpass selectivity > 2.0 (peak/mean ratio)
  - Coupling energy ratio > 0.01 (signal transferred to coupled line)

Reference: Pozar, "Microwave Engineering", 4th ed., Ch 8.
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
THRESHOLD_PCT = 5.0

# =============================================================================
# Analytical design
# =============================================================================
f0 = 5e9             # center frequency
eps_r = 4.4           # FR4
tan_d = 0.02
h = 1.6e-3            # substrate thickness

# Line geometry
W_line = 3.0e-3       # line width
gap = 0.5e-3          # coupling gap (tighter coupling)

# Effective permittivity
eps_eff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 12.0 * h / W_line) ** (-0.5)

# Quarter-wave coupling length at f0
lambda_eff = C0 / (f0 * np.sqrt(eps_eff))
L_couple = lambda_eff / 4.0

# Back-compute center frequency
f0_check = C0 / (4 * L_couple * np.sqrt(eps_eff))

print("=" * 60)
print("CASE 5: Coupled-Line Bandpass Filter")
print("=" * 60)
print(f"Design f0       : {f0 / 1e9:.3f} GHz")
print(f"eps_r           : {eps_r}")
print(f"eps_eff         : {eps_eff:.4f}")
print(f"lambda_eff      : {lambda_eff * 1e3:.2f} mm")
print(f"Coupling length : {L_couple * 1e3:.2f} mm (lambda/4)")
print(f"Line width      : {W_line * 1e3:.1f} mm")
print(f"Gap             : {gap * 1e3:.1f} mm")
print(f"f0 check        : {f0_check / 1e9:.3f} GHz")
print()

# =============================================================================
# Grid convergence
# =============================================================================
resolutions = [
    ("coarse", 0.5e-3),
    ("medium", 0.25e-3),
]

results_list = []

for label, dx in resolutions:
    print(f"--- Resolution: {label} (dx={dx*1e3:.2f} mm) ---")

    margin_x = 5e-3
    margin_y = 6e-3
    total_width = 2 * W_line + gap
    dom_x = L_couple + 2 * margin_x
    dom_y = total_width + 2 * margin_y
    dom_z = h + 4e-3

    sim = Simulation(
        freq_max=f0 * 2.5,
        domain=(dom_x, dom_y, dom_z),
        boundary="cpml",
        cpml_layers=8,
        dx=dx,
    )

    sigma_sub = 2.0 * np.pi * f0 * 8.854e-12 * eps_r * tan_d
    sim.add_material("substrate", eps_r=eps_r, sigma=sigma_sub)

    # Ground plane
    sim.add(Box((0, 0, 0), (dom_x, dom_y, 0)), material="pec")
    # Substrate
    sim.add(Box((0, 0, 0), (dom_x, dom_y, h)), material="substrate")

    # Line 1 (input / through)
    y1_lo = (dom_y - total_width) / 2.0
    y1_hi = y1_lo + W_line
    x_start = margin_x
    x_end = margin_x + L_couple
    sim.add(Box((x_start, y1_lo, h), (x_end, y1_hi, h)), material="pec")

    # Line 2 (coupled output)
    y2_lo = y1_hi + gap
    y2_hi = y2_lo + W_line
    sim.add(Box((x_start, y2_lo, h), (x_end, y2_hi, h)), material="pec")

    # Port source on line 1
    feed_x = x_start + 1e-3
    feed_y1 = (y1_lo + y1_hi) / 2.0
    sim.add_port(
        (feed_x, feed_y1, 0),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=f0, bandwidth=0.8),
        extent=h,
    )

    # Probes
    probe_x = x_end - 1e-3
    feed_y2 = (y2_lo + y2_hi) / 2.0
    sim.add_probe((probe_x, feed_y2, h / 2.0), component="ez")  # coupled
    sim.add_probe((probe_x, feed_y1, h / 2.0), component="ez")  # through

    grid = sim._build_grid()
    n_steps = int(np.ceil(10e-9 / grid.dt))
    print(f"  Steps: {n_steps}, dt={grid.dt*1e12:.2f} ps")

    result = sim.run(n_steps=n_steps, compute_s_params=True)

    # Analyze coupled signal
    ts = np.asarray(result.time_series)
    if ts.ndim == 2 and ts.shape[1] >= 2:
        coupled_signal = ts[:, 0]
        through_signal = ts[:, 1]
    else:
        coupled_signal = ts.ravel()
        through_signal = coupled_signal

    # FFT analysis
    nfft = len(coupled_signal) * 4
    dt = result.dt
    freqs_Hz = np.fft.rfftfreq(nfft, d=dt)
    spec_coupled = np.abs(np.fft.rfft(coupled_signal, n=nfft))
    spec_through = np.abs(np.fft.rfft(through_signal, n=nfft))

    # Coupling peak frequency
    band = (freqs_Hz > f0 * 0.3) & (freqs_Hz < f0 * 1.7)
    if np.any(band):
        peak_idx = np.argmax(spec_coupled[band])
        f_peak = freqs_Hz[band][peak_idx]
    else:
        f_peak = 0

    # Coupling energy ratio
    coupled_energy = np.sum(coupled_signal ** 2)
    through_energy = np.sum(through_signal ** 2)
    coupling_ratio = coupled_energy / (through_energy + 1e-30)

    # Selectivity
    band_wide = (freqs_Hz > f0 * 0.2) & (freqs_Hz < f0 * 2.0)
    if np.any(band_wide):
        spec_band = spec_coupled[band_wide]
        selectivity = np.max(spec_band) / (np.mean(spec_band) + 1e-30)
    else:
        selectivity = 0.0

    freq_err = abs(f_peak - f0) / f0 * 100 if f_peak > 0 else 999.0

    print(f"  f_peak          = {f_peak / 1e9:.3f} GHz (error {freq_err:.1f}%)")
    print(f"  Coupling ratio  = {coupling_ratio:.4f}")
    print(f"  Selectivity     = {selectivity:.2f}")

    results_list.append({
        "label": label,
        "dx": dx,
        "f_peak": f_peak,
        "freq_err": freq_err,
        "coupling_ratio": coupling_ratio,
        "selectivity": selectivity,
        "n_steps": n_steps,
        "spec_coupled": spec_coupled,
        "spec_through": spec_through,
        "freqs_Hz": freqs_Hz,
    })

# =============================================================================
# Summary and pass/fail
# =============================================================================
print()
print("=" * 60)
print("CONVERGENCE SUMMARY")
print("=" * 60)
print(f"{'Resolution':<12} {'dx (mm)':<10} {'f_peak (GHz)':<14} {'Freq err %':<12} {'Select.':<10}")
print("-" * 58)
for r in results_list:
    print(f"{r['label']:<12} {r['dx']*1e3:<10.2f} {r['f_peak']/1e9:<14.3f} "
          f"{r['freq_err']:<12.1f} {r['selectivity']:<10.2f}")

best = results_list[-1]
final_err = best["freq_err"]

# Pass/fail criteria
freq_pass = final_err < THRESHOLD_PCT
select_pass = best["selectivity"] > 2.0
coupling_pass = best["coupling_ratio"] > 0.01

all_pass = freq_pass and select_pass and coupling_pass

print()
print(f"Design f0        : {f0 / 1e9:.3f} GHz")
print(f"Simulated peak   : {best['f_peak'] / 1e9:.3f} GHz")
print(f"Freq error       : {final_err:.1f}% ({'PASS' if freq_pass else 'FAIL'}, threshold {THRESHOLD_PCT}%)")
print(f"Selectivity      : {best['selectivity']:.2f} ({'PASS' if select_pass else 'FAIL'}, threshold 2.0)")
print(f"Coupling ratio   : {best['coupling_ratio']:.4f} ({'PASS' if coupling_pass else 'FAIL'}, threshold 0.01)")

# =============================================================================
# Comparison plot
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Case 5: Coupled-Line Bandpass Filter", fontsize=13, fontweight="bold")

# Panel 1: Grid convergence
ax = axes[0]
dx_vals = [r["dx"] * 1e3 for r in results_list]
err_vals = [r["freq_err"] for r in results_list]
ax.plot(dx_vals, err_vals, "bo-", markersize=8, linewidth=2)
ax.axhline(THRESHOLD_PCT, color="r", ls="--", label=f"Threshold {THRESHOLD_PCT}%")
ax.set_xlabel("dx (mm)")
ax.set_ylabel("Center freq error (%)")
ax.set_title("Grid convergence")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Coupled spectrum from best run
ax = axes[1]
freqs_GHz = best["freqs_Hz"] / 1e9
spec_c_db = 20 * np.log10(np.maximum(
    best["spec_coupled"] / (best["spec_through"].max() + 1e-30), 1e-10))
band_plot = (freqs_GHz > 1) & (freqs_GHz < 12)
ax.plot(freqs_GHz[band_plot], spec_c_db[band_plot], "b-", lw=1.0, label="Coupled (S41)")
ax.axvline(f0 / 1e9, color="g", ls="--", lw=1.5, label=f"Design {f0/1e9:.1f} GHz")
ax.axvline(best["f_peak"] / 1e9, color="r", ls=":", lw=1.5,
           label=f"rfx peak {best['f_peak']/1e9:.2f} GHz")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Coupling (dB)")
ax.set_title("Coupled spectrum (medium grid)")
ax.set_ylim(-60, 5)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3: Summary
ax = axes[2]
ax.axis("off")
lines = [
    "Coupled Filter Validation",
    "-" * 30,
    f"Reference: Pozar Ch 8",
    f"f0 = {f0/1e9:.3f} GHz",
    f"L_couple = {L_couple*1e3:.2f} mm",
    f"W = {W_line*1e3:.1f} mm, gap = {gap*1e3:.1f} mm",
    "",
    f"f_peak (sim) = {best['f_peak']/1e9:.3f} GHz",
    f"Freq error   = {final_err:.1f}%",
    f"Selectivity  = {best['selectivity']:.2f}",
    f"Coupling     = {best['coupling_ratio']:.4f}",
    "",
    f"Freq     : {'PASS' if freq_pass else 'FAIL'}",
    f"Select.  : {'PASS' if select_pass else 'FAIL'}",
    f"Coupling : {'PASS' if coupling_pass else 'FAIL'}",
    f"Overall  : {'PASS' if all_pass else 'FAIL'}",
]
ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
        va="top", ha="left", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "05_coupled_filter.png")
plt.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {out_path}")

if all_pass:
    print(f"\nPASS: All coupled filter checks passed")
    sys.exit(0)
else:
    print(f"\nFAIL: One or more coupled filter checks failed")
    sys.exit(1)
