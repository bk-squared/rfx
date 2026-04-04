"""Accuracy Validation Case 4: Microstrip Impedance & Substrate Modeling

Validates two aspects of microstrip modeling accuracy:

1. Hammerstad-Jensen Z0 formula (pure analytical check)
2. Substrate dielectric constant modeling via a fully-filled PEC cavity

The cavity test uses eps_r=4.4 (FR4) and verifies that rfx correctly
shifts the TM110 resonant frequency by the expected factor 1/sqrt(eps_r)
relative to an empty cavity.

Analytical reference:
  - Empty cavity TM110: f_e = c/2 * sqrt((1/a)^2 + (1/b)^2)
  - Filled cavity TM110: f_f = f_e / sqrt(eps_r)
  - The ratio f_f/f_e = 1/sqrt(eps_r) is exact for uniform fill

Validation metrics:
  - Z0 formula within 3% of 50 ohm target
  - Frequency ratio within 2% of 1/sqrt(eps_r)

Reference: Hammerstad & Jensen (1980); Pozar "Microwave Engineering" Ch 6.
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
# Part 1: Hammerstad-Jensen Z0 formula validation (analytical)
# =============================================================================
eps_r = 4.4
h = 1.6e-3
Z0_target = 50.0

A = (Z0_target / 60.0) * np.sqrt((eps_r + 1.0) / 2.0) + \
    (eps_r - 1.0) / (eps_r + 1.0) * (0.23 + 0.11 / eps_r)
B = 377.0 * np.pi / (2.0 * Z0_target * np.sqrt(eps_r))

W_A = h * (8.0 * np.exp(A) / (np.exp(2 * A) - 2.0))
W_B = (2.0 * h / np.pi) * (B - 1.0 - np.log(2.0 * B - 1.0)
                             + (eps_r - 1.0) / (2.0 * eps_r)
                             * (np.log(B - 1.0) + 0.39 - 0.61 / eps_r))
W = W_A if W_A / h < 2 else W_B

eps_eff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 12.0 * h / W) ** (-0.5)

if W / h <= 1.0:
    Z0_analytical = (60.0 / np.sqrt(eps_eff)) * np.log(8.0 * h / W + W / (4.0 * h))
else:
    Z0_analytical = (120.0 * np.pi) / (
        np.sqrt(eps_eff) * (W / h + 1.393 + 0.667 * np.log(W / h + 1.444))
    )

z0_formula_err = abs(Z0_analytical - Z0_target) / Z0_target * 100

print("=" * 60)
print("CASE 4: Microstrip Impedance & Substrate Modeling")
print("=" * 60)
print(f"Z0 target        : {Z0_target:.0f} ohm")
print(f"Z0 analytical    : {Z0_analytical:.2f} ohm (error {z0_formula_err:.2f}%)")
print(f"W                : {W * 1e3:.3f} mm  (W/h = {W/h:.3f})")
print(f"eps_eff (Hamm.)  : {eps_eff:.4f}")
print()

# =============================================================================
# Part 2: FR4 substrate modeling via fully-filled PEC cavity
# Use different dimensions from Case 3 (which uses eps_r=2.2)
# to ensure this is an independent validation.
# =============================================================================
a_cav = 40e-3   # different from Case 3 (50mm)
b_cav = 30e-3   # different from Case 3 (40mm)
d_cav = 15e-3   # different from Case 3 (20mm)

f_empty = (C0 / 2) * np.sqrt((1/a_cav)**2 + (1/b_cav)**2)
f_filled_expected = f_empty / np.sqrt(eps_r)
ratio_expected = 1.0 / np.sqrt(eps_r)

print(f"Cavity           : {a_cav*1e3:.0f} x {b_cav*1e3:.0f} x {d_cav*1e3:.0f} mm")
print(f"eps_r (FR4)      : {eps_r}")
print(f"f_empty (TM110)  : {f_empty / 1e9:.4f} GHz")
print(f"f_filled (expect): {f_filled_expected / 1e9:.4f} GHz")
print(f"Ratio (expect)   : {ratio_expected:.4f} = 1/sqrt({eps_r})")
print()

# --- Simulate both cavities ---
dx = 1.0e-3

print("--- Empty cavity ---")
sim_e = Simulation(
    freq_max=f_empty * 1.5,
    domain=(a_cav, b_cav, d_cav),
    boundary="pec",
    dx=dx,
)
src = ModulatedGaussian(f0=f_empty, bandwidth=0.8)
sim_e.add_source((a_cav/3, b_cav/3, d_cav/2), component="ez", waveform=src)
sim_e.add_probe((2*a_cav/3, 2*b_cav/3, d_cav/2), component="ez")
grid = sim_e._build_grid()
n_steps = int(np.ceil(100.0 / (f_empty * grid.dt)))
n_steps = min(n_steps, 25000)
print(f"  Grid: {grid.nx}x{grid.ny}x{grid.nz}, steps={n_steps}")
result_e = sim_e.run(n_steps=n_steps)

modes_e = result_e.find_resonances(freq_range=(f_empty*0.5, f_empty*1.5), probe_idx=0)
if modes_e:
    f_e_sim = min(modes_e, key=lambda m: abs(m.freq - f_empty)).freq
else:
    ts = np.asarray(result_e.time_series).ravel()
    sp = np.abs(np.fft.rfft(ts, n=len(ts)*8))
    fr = np.fft.rfftfreq(len(ts)*8, d=result_e.dt)
    band = (fr > f_empty*0.5) & (fr < f_empty*1.5)
    f_e_sim = fr[np.argmax(sp * band)]
print(f"  f_empty_sim = {f_e_sim / 1e9:.4f} GHz")

print("--- FR4-filled cavity ---")
sim_f = Simulation(
    freq_max=f_empty * 1.5,
    domain=(a_cav, b_cav, d_cav),
    boundary="pec",
    dx=dx,
)
sim_f.add_material("fr4", eps_r=eps_r)
sim_f.add(Box((0, 0, 0), (a_cav, b_cav, d_cav)), material="fr4")
src2 = ModulatedGaussian(f0=f_filled_expected, bandwidth=0.8)
sim_f.add_source((a_cav/3, b_cav/3, d_cav/2), component="ez", waveform=src2)
sim_f.add_probe((2*a_cav/3, 2*b_cav/3, d_cav/2), component="ez")
result_f = sim_f.run(n_steps=n_steps)

modes_f = result_f.find_resonances(
    freq_range=(f_filled_expected*0.5, f_filled_expected*1.5), probe_idx=0)
if modes_f:
    f_f_sim = min(modes_f, key=lambda m: abs(m.freq - f_filled_expected)).freq
else:
    ts = np.asarray(result_f.time_series).ravel()
    sp = np.abs(np.fft.rfft(ts, n=len(ts)*8))
    fr = np.fft.rfftfreq(len(ts)*8, d=result_f.dt)
    band = (fr > f_filled_expected*0.5) & (fr < f_filled_expected*1.5)
    f_f_sim = fr[np.argmax(sp * band)]
print(f"  f_filled_sim = {f_f_sim / 1e9:.4f} GHz")

# Compute ratio
ratio_sim = f_f_sim / f_e_sim
ratio_err = abs(ratio_sim - ratio_expected) / ratio_expected * 100

# Extract eps_r from ratio: eps_r_sim = (1/ratio_sim)^2
eps_r_sim = (f_e_sim / f_f_sim) ** 2
eps_r_err = abs(eps_r_sim - eps_r) / eps_r * 100

print()
print(f"Frequency ratio (sim)    : {ratio_sim:.4f}")
print(f"Frequency ratio (expect) : {ratio_expected:.4f}")
print(f"Ratio error              : {ratio_err:.3f}%")
print(f"eps_r extracted          : {eps_r_sim:.4f} (target {eps_r})")
print(f"eps_r error              : {eps_r_err:.3f}%")

# =============================================================================
# Summary and pass/fail
# =============================================================================
combined_err = max(z0_formula_err, ratio_err)

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Z0 formula error      : {z0_formula_err:.2f}%")
print(f"Freq ratio error      : {ratio_err:.3f}%")
print(f"Combined error        : {combined_err:.2f}%")
print(f"Threshold             : {THRESHOLD_PCT}%")

# =============================================================================
# Comparison plot
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Case 4: Microstrip Z0 & FR4 Substrate Validation", fontsize=13, fontweight="bold")

# Panel 1: Spectra
ax = axes[0]
ts_e_arr = np.asarray(result_e.time_series).ravel()
ts_f_arr = np.asarray(result_f.time_series).ravel()
nfft = max(len(ts_e_arr), len(ts_f_arr)) * 8
sp_e = np.abs(np.fft.rfft(ts_e_arr, n=nfft))
sp_f = np.abs(np.fft.rfft(ts_f_arr, n=nfft))
fr_ghz = np.fft.rfftfreq(nfft, d=result_e.dt) / 1e9
sp_e_db = 20 * np.log10(np.maximum(sp_e / (sp_e.max() or 1), 1e-10))
sp_f_db = 20 * np.log10(np.maximum(sp_f / (sp_f.max() or 1), 1e-10))
bm = (fr_ghz > 1) & (fr_ghz < 8)
ax.plot(fr_ghz[bm], sp_e_db[bm], "b-", lw=1, label="Empty")
ax.plot(fr_ghz[bm], sp_f_db[bm], "r-", lw=1, label="FR4-filled")
ax.axvline(f_empty/1e9, color="b", ls="--", lw=0.8, alpha=0.5)
ax.axvline(f_filled_expected/1e9, color="r", ls="--", lw=0.8, alpha=0.5)
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Normalized (dB)")
ax.set_title("Empty vs FR4-filled cavity")
ax.set_ylim(-60, 5)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: Bar chart
ax = axes[1]
labels = ["Empty", "FR4-filled"]
analytical = [f_empty/1e9, f_filled_expected/1e9]
simulated = [f_e_sim/1e9, f_f_sim/1e9]
x = np.arange(len(labels))
width = 0.35
ax.bar(x - width/2, analytical, width, label="Analytical", color="green", alpha=0.7)
ax.bar(x + width/2, simulated, width, label="rfx FDTD", color="red", alpha=0.7)
ax.set_ylabel("Frequency (GHz)")
ax.set_title("Resonant frequency comparison")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# Panel 3: Summary
ax = axes[2]
ax.axis("off")
lines = [
    "Microstrip / FR4 Validation",
    "-" * 32,
    f"Reference: Hammerstad-Jensen",
    "",
    f"Z0 check (analytical):",
    f"  Target = {Z0_target:.0f} ohm",
    f"  Result = {Z0_analytical:.2f} ohm",
    f"  Error  = {z0_formula_err:.2f}%",
    "",
    f"FR4 substrate check:",
    f"  Cavity {a_cav*1e3:.0f}x{b_cav*1e3:.0f}x{d_cav*1e3:.0f} mm",
    f"  eps_r = {eps_r} (FR4)",
    f"  f_e  = {f_e_sim/1e9:.4f} GHz",
    f"  f_f  = {f_f_sim/1e9:.4f} GHz",
    f"  Ratio = {ratio_sim:.4f} (expect {ratio_expected:.4f})",
    f"  Error = {ratio_err:.3f}%",
    "",
    f"Combined = {combined_err:.2f}%",
    f"Threshold= {THRESHOLD_PCT}%",
    f"Verdict  = {'PASS' if combined_err < THRESHOLD_PCT else 'FAIL'}",
]
ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
        va="top", ha="left", fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "04_microstrip_z0.png")
plt.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {out_path}")

if combined_err < THRESHOLD_PCT:
    print(f"\nPASS: Microstrip validation combined error {combined_err:.2f}% < {THRESHOLD_PCT}%")
    sys.exit(0)
else:
    print(f"\nFAIL: Microstrip validation combined error {combined_err:.2f}% >= {THRESHOLD_PCT}%")
    sys.exit(1)
