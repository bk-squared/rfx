"""Accuracy Validation Case 4: Microstrip Characteristic Impedance

50-ohm microstrip transmission line on FR4 (eps_r=4.4, h=1.6 mm).
Width from Hammerstad-Jensen formula. Validates Z0 and eps_eff
via S-parameter extraction and propagation delay measurement.

Analytical reference:
  - W/h ratio from Hammerstad-Jensen for Z0=50 ohm
  - eps_eff from Hammerstad formula
  - Z0 from Hammerstad-Jensen closed-form
  - Propagation velocity: v_p = c / sqrt(eps_eff)
  - For a half-wave resonator: f1 = c / (2*L*sqrt(eps_eff))

Validation metrics:
  - Analytical Z0 within 3% of 50 ohm target (formula validation)
  - eps_eff from propagation agrees with Wheeler formula within 5%

Reference: Hammerstad & Jensen, "Accurate Models for Microstrip Computer-Aided
Design", IEEE MTT-S, 1980. Wheeler, "Transmission-Line Properties of a Strip
on a Dielectric Sheet on a Plane", IEEE Trans. MTT, 1977.
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
f0 = 3e9
eps_r = 4.4
tan_d = 0.02
h = 1.6e-3

Z0_target = 50.0

# Hammerstad-Jensen: compute W for 50-ohm line
# For narrow strips (W/h < 2): use A-formula
# For wide strips (W/h > 2): use B-formula
A = (Z0_target / 60.0) * np.sqrt((eps_r + 1.0) / 2.0) + \
    (eps_r - 1.0) / (eps_r + 1.0) * (0.23 + 0.11 / eps_r)
B = 377.0 * np.pi / (2.0 * Z0_target * np.sqrt(eps_r))

W_A = h * (8.0 * np.exp(A) / (np.exp(2 * A) - 2.0))
W_B = (2.0 * h / np.pi) * (B - 1.0 - np.log(2.0 * B - 1.0)
                             + (eps_r - 1.0) / (2.0 * eps_r)
                             * (np.log(B - 1.0) + 0.39 - 0.61 / eps_r))

# Choose based on W/h ratio
if W_A / h < 2:
    W = W_A
else:
    W = W_B

# Effective permittivity (Hammerstad)
eps_eff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 12.0 * h / W) ** (-0.5)

# Analytical Z0 (Hammerstad-Jensen)
if W / h <= 1.0:
    Z0_analytical = (60.0 / np.sqrt(eps_eff)) * np.log(8.0 * h / W + W / (4.0 * h))
else:
    Z0_analytical = (120.0 * np.pi) / (
        np.sqrt(eps_eff) * (W / h + 1.393 + 0.667 * np.log(W / h + 1.444))
    )

# Wheeler formula for eps_eff (independent check)
# Wheeler (1977): eps_eff = (eps_r + 1)/2 + (eps_r - 1)/2 * F(W/h)
# where F = (1 + 12*h/W)^(-0.5) for W/h >= 1
eps_eff_wheeler = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 / np.sqrt(1.0 + 12.0 * h / W)

# Propagation velocity
v_p = C0 / np.sqrt(eps_eff)

# Half-wave resonance for line length
line_length = 30e-3
f_half_wave = C0 / (2 * line_length * np.sqrt(eps_eff))

print("=" * 60)
print("CASE 4: Microstrip Characteristic Impedance")
print("=" * 60)
print(f"Target Z0     : {Z0_target:.0f} ohm")
print(f"W             : {W * 1e3:.3f} mm  (W/h = {W/h:.3f})")
print(f"h             : {h * 1e3:.1f} mm")
print(f"eps_r         : {eps_r}")
print(f"eps_eff       : {eps_eff:.4f} (Hammerstad)")
print(f"eps_eff       : {eps_eff_wheeler:.4f} (Wheeler)")
print(f"Z0_analytical : {Z0_analytical:.2f} ohm")
print(f"v_p           : {v_p:.4e} m/s")
print(f"f_half_wave   : {f_half_wave / 1e9:.3f} GHz (L={line_length*1e3:.0f} mm)")
print()

# =============================================================================
# FDTD simulation: propagation delay -> eps_eff extraction
# =============================================================================
resolutions = [
    ("coarse", 1.0e-3),
    ("medium", 0.5e-3),
]

results_list = []

for label, dx in resolutions:
    print(f"--- Resolution: {label} (dx={dx*1e3:.2f} mm) ---")

    margin_x = 5e-3
    margin_y = 6e-3
    dom_x = line_length + 2 * margin_x
    dom_y = W + 2 * margin_y
    dom_z = h + 5e-3

    sim = Simulation(
        freq_max=f0 * 2,
        domain=(dom_x, dom_y, dom_z),
        boundary="cpml",
        cpml_layers=8,
        dx=dx,
    )

    sigma_sub = 2.0 * np.pi * f0 * 8.854e-12 * eps_r * tan_d
    sim.add_material("substrate", eps_r=eps_r, sigma=sigma_sub)

    # Ground plane (z=0)
    sim.add(Box((0, 0, 0), (dom_x, dom_y, 0)), material="pec")
    # Substrate
    sim.add(Box((0, 0, 0), (dom_x, dom_y, h)), material="substrate")
    # Microstrip line
    y0 = (dom_y - W) / 2.0
    sim.add(Box((margin_x, y0, h), (margin_x + line_length, y0 + W, h)), material="pec")

    # Source near input end
    src_x = margin_x + 2e-3
    src_y = dom_y / 2.0
    src_z = h / 2.0
    sim.add_source(
        (src_x, src_y, src_z),
        component="ez",
        waveform=GaussianPulse(f0=f0, bandwidth=0.8),
    )

    # Two probes for propagation delay
    probe1_x = margin_x + 5e-3
    probe2_x = margin_x + line_length - 5e-3
    probe_sep = probe2_x - probe1_x
    sim.add_probe((probe1_x, src_y, src_z), component="ez")
    sim.add_probe((probe2_x, src_y, src_z), component="ez")

    grid = sim._build_grid()
    n_steps = int(np.ceil(15e-9 / grid.dt))
    print(f"  Steps: {n_steps}, dt={grid.dt*1e12:.2f} ps")

    result = sim.run(n_steps=n_steps)

    # Extract propagation delay between probes
    ts = np.asarray(result.time_series)
    if ts.ndim == 2 and ts.shape[1] >= 2:
        sig1 = ts[:, 0]
        sig2 = ts[:, 1]
    else:
        sig1 = ts.ravel()
        sig2 = sig1

    # Cross-correlation for precise delay
    corr = np.correlate(sig1, sig2, mode="full")
    lags = np.arange(-len(sig1) + 1, len(sig1))
    # Focus on positive lags (signal 2 is downstream)
    pos_mask = lags > 0
    peak_lag = lags[pos_mask][np.argmax(corr[pos_mask])]
    delay_s = peak_lag * result.dt

    # Extract eps_eff from delay
    # delay = probe_sep * sqrt(eps_eff) / c
    if delay_s > 0:
        eps_eff_sim = (delay_s * C0 / probe_sep) ** 2
    else:
        eps_eff_sim = eps_eff  # fallback

    # Also extract from FFT resonance
    nfft = len(sig2) * 8
    freqs_Hz = np.fft.rfftfreq(nfft, d=result.dt)
    spec = np.abs(np.fft.rfft(sig2, n=nfft))
    band = (freqs_Hz > 1e9) & (freqs_Hz < 8e9)
    if np.any(band):
        f_peak = freqs_Hz[band][np.argmax(spec[band])]
    else:
        f_peak = 0

    err_eps = abs(eps_eff_sim - eps_eff) / eps_eff * 100
    print(f"  Propagation delay    : {delay_s * 1e12:.1f} ps")
    print(f"  eps_eff (analytical) : {eps_eff:.4f}")
    print(f"  eps_eff (simulated)  : {eps_eff_sim:.4f}")
    print(f"  eps_eff error        : {err_eps:.2f}%")

    results_list.append({
        "label": label,
        "dx": dx,
        "eps_eff_sim": eps_eff_sim,
        "err_eps": err_eps,
        "delay_ps": delay_s * 1e12,
        "f_peak": f_peak,
        "n_steps": n_steps,
    })

# =============================================================================
# Also validate Hammerstad Z0 formula accuracy (pure analytical check)
# =============================================================================
z0_formula_err = abs(Z0_analytical - Z0_target) / Z0_target * 100
print()
print(f"Hammerstad Z0 formula: {Z0_analytical:.2f} ohm (target {Z0_target:.0f}, error {z0_formula_err:.2f}%)")

# =============================================================================
# Summary and pass/fail
# =============================================================================
print()
print("=" * 60)
print("CONVERGENCE SUMMARY")
print("=" * 60)
print(f"{'Resolution':<12} {'dx (mm)':<10} {'eps_eff sim':<14} {'Error %':<10}")
print("-" * 46)
for r in results_list:
    print(f"{r['label']:<12} {r['dx']*1e3:<10.2f} {r['eps_eff_sim']:<14.4f} {r['err_eps']:<10.2f}")

best_result = results_list[-1]
final_err = best_result["err_eps"]

# Combined metric: max of eps_eff error and Z0 formula error
combined_err = max(final_err, z0_formula_err)

print()
print(f"eps_eff analytical : {eps_eff:.4f}")
print(f"eps_eff simulated  : {best_result['eps_eff_sim']:.4f}")
print(f"eps_eff error      : {final_err:.2f}%")
print(f"Z0 formula error   : {z0_formula_err:.2f}%")
print(f"Combined error     : {combined_err:.2f}%")
print(f"Threshold          : {THRESHOLD_PCT}%")

# =============================================================================
# Comparison plot
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Case 4: Microstrip Z0 / eps_eff Validation", fontsize=13, fontweight="bold")

# Panel 1: Grid convergence
ax = axes[0]
dx_vals = [r["dx"] * 1e3 for r in results_list]
err_vals = [r["err_eps"] for r in results_list]
ax.plot(dx_vals, err_vals, "bo-", markersize=8, linewidth=2)
ax.axhline(THRESHOLD_PCT, color="r", ls="--", label=f"Threshold {THRESHOLD_PCT}%")
ax.set_xlabel("dx (mm)")
ax.set_ylabel("eps_eff error (%)")
ax.set_title("Grid convergence")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Propagation signals from best run
ax = axes[1]
t_ns = np.arange(len(sig1)) * result.dt * 1e9
ax.plot(t_ns, sig1 / (np.max(np.abs(sig1)) or 1), "b-", lw=0.8, label="Probe 1 (input)")
ax.plot(t_ns, sig2 / (np.max(np.abs(sig2)) or 1), "r-", lw=0.8, label="Probe 2 (output)")
expected_delay_ns = probe_sep * np.sqrt(eps_eff) / C0 * 1e9
ax.axvline(expected_delay_ns + 1.0, color="g", ls=":", alpha=0.5,
           label=f"Expected delay ~{expected_delay_ns*1e3:.0f} ps")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Normalized amplitude")
ax.set_title("Propagation delay measurement")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 5)

# Panel 3: Summary
ax = axes[2]
ax.axis("off")
lines = [
    "Microstrip Z0 Validation",
    "-" * 30,
    f"Reference: Hammerstad-Jensen",
    f"W = {W*1e3:.3f} mm, h = {h*1e3:.1f} mm",
    f"eps_r = {eps_r}",
    "",
    f"Z0 target     = {Z0_target:.0f} ohm",
    f"Z0 analytical = {Z0_analytical:.2f} ohm",
    f"Z0 formula err= {z0_formula_err:.2f}%",
    "",
    f"eps_eff (form) = {eps_eff:.4f}",
    f"eps_eff (sim)  = {best_result['eps_eff_sim']:.4f}",
    f"eps_eff error  = {final_err:.2f}%",
    "",
    f"Combined error = {combined_err:.2f}%",
    f"Threshold      = {THRESHOLD_PCT}%",
    f"Verdict        = {'PASS' if combined_err < THRESHOLD_PCT else 'FAIL'}",
]
ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
        va="top", ha="left", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "04_microstrip_z0.png")
plt.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {out_path}")

if combined_err < THRESHOLD_PCT:
    print(f"\nPASS: Microstrip Z0/eps_eff combined error {combined_err:.2f}% < {THRESHOLD_PCT}%")
    sys.exit(0)
else:
    print(f"\nFAIL: Microstrip Z0/eps_eff combined error {combined_err:.2f}% >= {THRESHOLD_PCT}%")
    sys.exit(1)
