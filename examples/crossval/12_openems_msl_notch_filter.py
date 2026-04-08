"""Cross-validation: OpenEMS MSL NotchFilter tutorial replica.

Replicates: docs.openems.de/python/openEMS/Tutorials/MSL_NotchFilter.html
Structure: 50-ohm microstrip line with quarter-wave open stub notch filter
Comparison: S21 (FFT transmission) vs analytical stub resonance frequency

Analytical reference:
  f_notch = c / (4 * L_stub * sqrt(eps_eff))

Uses non-uniform z-mesh (fine substrate cells) for proper microstrip mode
resolution. S21 measured via FFT ratio of output/input probe signals.

PASS criteria:
  - Notch visible in transmission spectrum (depth > 6 dB below passband)
  - Notch frequency within 15% of analytical prediction

Save: examples/crossval/12_openems_msl_notch_filter.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.auto_config import smooth_grading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Parameters
# =============================================================================
C0 = 2.998e8
eps_r = 3.55        # RO4003C
h_sub = 1.524e-3    # substrate thickness
MSL_W = 3.0e-3      # 50-ohm microstrip width
stub_L = 12.0e-3    # quarter-wave stub
stub_W = MSL_W
feed_L = 15.0e-3    # each feed length
f0 = 3.5e9
f_max = 7e9

# Effective eps and analytical notch
u = MSL_W / h_sub
eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 / u) ** (-0.5)
f_notch_analytical = C0 / (4 * stub_L * np.sqrt(eps_eff))

# Mesh: fine in z-substrate (6 cells), standard xy
dx = 0.5e-3
n_sub = 6
dz_sub = h_sub / n_sub  # ~0.254mm
n_air = 20
raw_dz = np.concatenate([np.full(n_sub, dz_sub), np.full(n_air, dx)])
dz_profile = smooth_grading(raw_dz, max_ratio=1.3)

# Domain
total_x = 2 * feed_L + stub_W
margin = 8e-3
dom_x = total_x + 2 * margin
dom_y = stub_L + MSL_W + 2 * margin

print("=" * 60)
print("Cross-Validation: OpenEMS MSL Notch Filter")
print("=" * 60)
print(f"Substrate: eps_r={eps_r}, h={h_sub*1e3:.3f} mm")
print(f"MSL width: {MSL_W*1e3:.1f} mm, stub: {stub_L*1e3:.1f} mm")
print(f"eps_eff: {eps_eff:.3f}")
print(f"Analytical notch: {f_notch_analytical/1e9:.3f} GHz")
print(f"Mesh: dx={dx*1e3:.1f} mm, dz_sub={dz_sub*1e3:.3f} mm ({n_sub} z-cells)")
print()

# =============================================================================
# Build simulation
# =============================================================================
sim = Simulation(
    freq_max=f_max,
    domain=(dom_x, dom_y, 0),
    dx=dx,
    dz_profile=dz_profile,
    boundary="cpml",
    cpml_layers=8,
)

sim.add_material("ro4003c", eps_r=eps_r, sigma=0.0)

# Ground plane (one dz_sub thick)
sim.add(Box((0, 0, 0), (dom_x, dom_y, dz_sub)), material="pec")

# Substrate
sim.add(Box((0, 0, 0), (dom_x, dom_y, h_sub)), material="ro4003c")

# Microstrip layout:
#   Port1 --- feed_L --- T-junction --- feed_L --- Port2
#                             |
#                          stub_L (+y, open end)
msl_y_center = margin
msl_y0 = msl_y_center - MSL_W / 2
msl_y1 = msl_y_center + MSL_W / 2
msl_x0 = margin
msl_x1 = msl_x0 + total_x

# Main microstrip trace on top of substrate
sim.add(
    Box((msl_x0, msl_y0, h_sub), (msl_x1, msl_y1, h_sub + dz_sub)),
    material="pec",
)

# Stub (extends in +y from main line)
junc_x = msl_x0 + feed_L
stub_x0 = junc_x - stub_W / 2
stub_x1 = junc_x + stub_W / 2
stub_y0 = msl_y1
stub_y1 = stub_y0 + stub_L

sim.add(
    Box((stub_x0, stub_y0, h_sub), (stub_x1, stub_y1, h_sub + dz_sub)),
    material="pec",
)

# Source (soft) near port 1 — no impedance loading for FFT method
port1_x = msl_x0 + 3e-3
sim.add_source(
    position=(port1_x, msl_y_center, h_sub / 2),
    component="ez",
    waveform=GaussianPulse(f0=f0, bandwidth=0.8),
)

# Probes at both ports (in substrate midplane)
sim.add_probe(position=(port1_x, msl_y_center, h_sub / 2), component="ez")
port2_x = msl_x1 - 3e-3
sim.add_probe(position=(port2_x, msl_y_center, h_sub / 2), component="ez")

# =============================================================================
# Run
# =============================================================================
print("Preflight:")
warnings = sim.preflight(strict=False)
print()

t0 = time.time()
result = sim.run(num_periods=30)
elapsed = time.time() - t0
print(f"Simulation time: {elapsed:.1f}s")

# =============================================================================
# Analysis: FFT-based S21 (output/input spectral ratio)
# =============================================================================
ts = np.array(result.time_series)
dt = result.dt
n_samples = ts.shape[0]

sig_in = ts[:, 0]   # probe at port 1
sig_out = ts[:, 1]  # probe at port 2

print(f"Probe signals: in_max={np.max(np.abs(sig_in)):.3e}, "
      f"out_max={np.max(np.abs(sig_out)):.3e}")

# Windowed FFT for clean spectrum
window = np.hanning(n_samples)
S_in = np.fft.rfft(sig_in * window)
S_out = np.fft.rfft(sig_out * window)
freqs = np.fft.rfftfreq(n_samples, d=dt)

# Transfer function (S21 proxy)
s21_mag = np.abs(S_out) / (np.abs(S_in) + 1e-30)
s21_db = 20 * np.log10(s21_mag + 1e-30)

# Focus on meaningful bandwidth
f_mask = (freqs > 0.5e9) & (freqs < f_max)
freqs_plot = freqs[f_mask]
s21_plot = s21_db[f_mask]

# Normalize: set passband level to 0 dB
# Use low-frequency region (before notch) as passband reference
passband = (freqs_plot > 1e9) & (freqs_plot < f_notch_analytical * 0.7)
if np.any(passband):
    passband_level = np.median(s21_plot[passband])
    s21_norm = s21_plot - passband_level
else:
    s21_norm = s21_plot

# Find notch (minimum in normalized S21)
notch_idx = np.argmin(s21_norm)
f_notch = freqs_plot[notch_idx]
s21_at_notch = s21_norm[notch_idx]

err = abs(f_notch - f_notch_analytical) / f_notch_analytical

print(f"\nResults:")
print(f"  Notch frequency: {f_notch/1e9:.3f} GHz (analytical: {f_notch_analytical/1e9:.3f} GHz)")
print(f"  S21 notch depth: {s21_at_notch:.1f} dB (relative to passband)")
print(f"  Error:           {err*100:.1f}%")

# =============================================================================
# Validation
# =============================================================================
PASS = True

if err > 0.15:
    print(f"  FAIL: notch frequency error {err*100:.1f}% > 15%")
    PASS = False
else:
    print(f"  PASS: notch frequency within 15%")

if s21_at_notch > -6:
    print(f"  FAIL: notch depth {s21_at_notch:.1f} dB (expected < -6 dB below passband)")
    PASS = False
else:
    print(f"  PASS: notch depth adequate ({s21_at_notch:.1f} dB)")

# =============================================================================
# Plot
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("rfx vs Analytical: MSL Notch Filter", fontsize=14)

ax1.plot(freqs_plot / 1e9, s21_norm, "r-", label="S21 (rfx, normalized)", linewidth=1.5)
ax1.axvline(f_notch_analytical / 1e9, color="k", ls="--", alpha=0.5,
            label=f"Analytical: {f_notch_analytical/1e9:.2f} GHz")
ax1.axvline(f_notch / 1e9, color="g", ls=":", alpha=0.7,
            label=f"rfx notch: {f_notch/1e9:.2f} GHz")
ax1.set_xlabel("Frequency (GHz)")
ax1.set_ylabel("Normalized S21 (dB)")
ax1.set_ylim(-30, 5)
ax1.set_xlim(0.5, f_max / 1e9)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_title("Transmission (S21)")

# Geometry schematic
ax2.set_xlim(-2, total_x * 1e3 + 2)
ax2.set_ylim(-5, (stub_L + MSL_W + 5e-3) * 1e3)
ax2.add_patch(plt.Rectangle(
    (0, -MSL_W / 2 * 1e3), total_x * 1e3, MSL_W * 1e3,
    facecolor="gold", edgecolor="k", linewidth=1))
ax2.add_patch(plt.Rectangle(
    ((feed_L - stub_W / 2) * 1e3, MSL_W / 2 * 1e3),
    stub_W * 1e3, stub_L * 1e3,
    facecolor="gold", edgecolor="k", linewidth=1))
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")
ax2.set_title("Geometry (top view)")
ax2.set_aspect("equal")
ax2.text(1, -4, "Port 1", fontsize=10, color="blue")
ax2.text(total_x * 1e3 - 10, -4, "Port 2", fontsize=10, color="red")
ax2.text(feed_L * 1e3 + 2, (stub_L + MSL_W / 2) * 1e3 - 2, "Open stub", fontsize=9)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "12_openems_msl_notch_filter.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")

if PASS:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED")
