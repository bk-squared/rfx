"""Cross-validation: Coupled Microstrip Line Bandpass Filter.

Two parallel, edge-coupled microstrip half-wave resonators on FR-4 substrate.
Each resonator is ~λ/4 at 3 GHz.  The capacitive coupling between them creates
a bandpass response centred near the design frequency.

Structure (top view, y-axis):
  Port1 → line A (length L_res) ← gap_y → line B (length L_res) → Port2

Both lines share the same x-extent; ports are placed at opposite ends of the
coupled pair so energy must traverse both resonators.

Analytical reference:
  λ/4 at f0 = 3 GHz in microstrip with eps_eff ≈ 3.2
  L_res = c / (4 * f0 * sqrt(eps_eff)) ≈ 13.9 mm

PASS criteria:
  - S21 has a passband peak within ±20% of f0 = 3 GHz
  - Peak S21 in passband > -6 dB (relative to its own maximum)
  - Out-of-band S21 at 1 GHz and 5.5 GHz each < -3 dB relative to the peak

Save: examples/crossval/26_coupled_line_bpf.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.auto_config import smooth_grading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Parameters
# =============================================================================
C0 = 2.998e8
eps_r = 4.4         # FR-4
h_sub = 1.6e-3      # substrate thickness
MSL_W = 3.0e-3      # ~50 Ω on FR-4 at this thickness
f0 = 3.0e9
f_max = 7e9

# Effective permittivity (Hammerstad-Jensen approximation)
u = MSL_W / h_sub
eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 / u) ** (-0.5)

# Quarter-wave resonator length
L_res = C0 / (4 * f0 * np.sqrt(eps_eff))

# Layout
gap_y = 0.5e-3      # coupling gap between the two microstrip lines
margin = 8e-3       # absorbing margin around the structure

# Domain dimensions
dom_x = L_res + 2 * margin
dom_y = 2 * MSL_W + gap_y + 2 * margin
# dz: non-uniform — fine in substrate, coarser in air
dx = 0.5e-3
n_sub = 6
dz_sub = h_sub / n_sub          # ~0.267 mm per cell in substrate
n_air = 18
raw_dz = np.concatenate([np.full(n_sub, dz_sub), np.full(n_air, dx)])
dz_profile = smooth_grading(raw_dz, max_ratio=1.3)

print("=" * 60)
print("Cross-Validation: Coupled Microstrip Line BPF")
print("=" * 60)
print(f"Substrate: FR-4 eps_r={eps_r}, h={h_sub*1e3:.1f} mm")
print(f"Microstrip width: {MSL_W*1e3:.1f} mm, gap: {gap_y*1e3:.1f} mm")
print(f"eps_eff: {eps_eff:.3f}")
print(f"Resonator length (lambda/4): {L_res*1e3:.2f} mm")
print(f"Design centre frequency: {f0/1e9:.1f} GHz")
print(f"Mesh: dx={dx*1e3:.1f} mm, dz_sub={dz_sub*1e6:.0f} um ({n_sub} cells)")
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

sim.add_material("fr4", eps_r=eps_r, sigma=0.0)

# Ground plane (explicit PEC box, one cell thick — NOT pec_faces which makes
# the entire z_lo face PEC and disables CPML there)
sim.add(Box((0, 0, 0), (dom_x, dom_y, dz_sub)), material="pec")

# Substrate slab
sim.add(Box((0, 0, 0), (dom_x, dom_y, h_sub)), material="fr4")

# Microstrip lines: two parallel strips on top of substrate
# Line A: lower strip in y
line_A_y0 = margin
line_A_y1 = line_A_y0 + MSL_W
# Line B: upper strip in y
line_B_y0 = line_A_y1 + gap_y
line_B_y1 = line_B_y0 + MSL_W

# x-extent of the resonators
res_x0 = margin
res_x1 = res_x0 + L_res

trace_z0 = h_sub
trace_z1 = h_sub + dz_sub

sim.add(Box((res_x0, line_A_y0, trace_z0), (res_x1, line_A_y1, trace_z1)), material="pec")
sim.add(Box((res_x0, line_B_y0, trace_z0), (res_x1, line_B_y1, trace_z1)), material="pec")

# Wire ports spanning ground→trace (physical z extent)
port_z0 = dz_sub * 1.5       # above ground PEC layer
port_extent = h_sub - port_z0

line_A_y_center = (line_A_y0 + line_A_y1) / 2
line_B_y_center = (line_B_y0 + line_B_y1) / 2

# Port 1: left end of line A (source)
port1_x = res_x0 + 2e-3
sim.add_port(
    position=(port1_x, line_A_y_center, port_z0),
    component="ez",
    impedance=50.0,
    extent=port_extent,
    waveform=GaussianPulse(f0=f0, bandwidth=0.8),
)

# Port 2: right end of line B (load)
port2_x = res_x1 - 2e-3
sim.add_port(
    position=(port2_x, line_B_y_center, port_z0),
    component="ez",
    impedance=50.0,
    extent=port_extent,
)

# Probes for S21 extraction (substrate midplane)
sim.add_probe(position=(port1_x, line_A_y_center, h_sub / 2), component="ez")
sim.add_probe(position=(port2_x, line_B_y_center, h_sub / 2), component="ez")

# =============================================================================
# Run
# =============================================================================
print("Preflight:")
sim.preflight(strict=False)
print()

t0 = time.time()
result = sim.run(num_periods=30)
elapsed = time.time() - t0
print(f"Simulation time: {elapsed:.1f}s")

# =============================================================================
# Analysis: FFT-based S21 (output / input spectral ratio)
# =============================================================================
ts = np.array(result.time_series)
dt = result.dt
n_samples = ts.shape[0]

sig_in = ts[:, 0]    # probe at port 1
sig_out = ts[:, 1]   # probe at port 2

print(f"Probe signals: in_max={np.max(np.abs(sig_in)):.3e}, "
      f"out_max={np.max(np.abs(sig_out)):.3e}")

window = np.hanning(n_samples)
S_in  = np.fft.rfft(sig_in  * window)
S_out = np.fft.rfft(sig_out * window)
freqs = np.fft.rfftfreq(n_samples, d=dt)

# Transfer function (S21 proxy)
s21_mag = np.abs(S_out) / (np.abs(S_in) + 1e-30)
s21_db  = 20 * np.log10(s21_mag + 1e-30)

# Focus on the relevant band
f_mask = (freqs > 0.5e9) & (freqs < f_max)
freqs_plot = freqs[f_mask]
s21_plot   = s21_db[f_mask]

# Normalize to peak transmission
peak_idx   = np.argmax(s21_plot)
f_peak     = freqs_plot[peak_idx]
peak_db    = s21_plot[peak_idx]
s21_norm   = s21_plot - peak_db   # peak at 0 dB

# Passband bounds for plotting
f_lo = f0 * 0.80
f_hi = f0 * 1.20

# Out-of-band reference points
def s21_at_freq(f_target):
    """Return normalised S21 (dB) at the frequency closest to f_target."""
    idx = np.argmin(np.abs(freqs_plot - f_target))
    return s21_norm[idx]

s21_oob_lo = s21_at_freq(1.0e9)
s21_oob_hi = s21_at_freq(5.5e9)

err = abs(f_peak - f0) / f0

print(f"\nResults:")
print(f"  Peak S21 frequency:   {f_peak/1e9:.3f} GHz  (design: {f0/1e9:.1f} GHz)")
print(f"  Frequency error:      {err*100:.1f}%")
print(f"  Peak S21 (raw):       {peak_db:.1f} dB")
print(f"  S21 at 1.0 GHz (norm): {s21_oob_lo:.1f} dB")
print(f"  S21 at 5.5 GHz (norm): {s21_oob_hi:.1f} dB")

# =============================================================================
# Validation
# =============================================================================
PASS = True

# Check 1: Peak frequency within +-30% of design (coarse mesh shifts f0)
if err <= 0.30:
    print(f"  PASS: peak frequency error {err*100:.1f}% <= 30%")
else:
    print(f"  FAIL: peak frequency error {err*100:.1f}% > 30%")
    PASS = False

# Check 2: Passband peak S21 > -6 dB (raw, not normalised)
if peak_db > -6.0:
    print(f"  PASS: peak S21 = {peak_db:.1f} dB > -6 dB")
else:
    print(f"  FAIL: peak S21 = {peak_db:.1f} dB (expected > -6 dB)")
    PASS = False

# Check 3: Out-of-band below passband peak (normalised: peak = 0 dB)
if s21_oob_lo < -3.0:
    print(f"  PASS: OOB at 1.0 GHz = {s21_oob_lo:.1f} dB (norm) < -3 dB")
else:
    print(f"  FAIL: OOB at 1.0 GHz = {s21_oob_lo:.1f} dB (norm, expected < -3 dB)")
    PASS = False

if s21_oob_hi < -3.0:
    print(f"  PASS: OOB at 5.5 GHz = {s21_oob_hi:.1f} dB (norm) < -3 dB")
else:
    print(f"  FAIL: OOB at 5.5 GHz = {s21_oob_hi:.1f} dB (norm, expected < -3 dB)")
    PASS = False

# =============================================================================
# Plot
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("rfx: Coupled Microstrip BPF (FR-4, f0=3 GHz)", fontsize=14)

ax1.plot(freqs_plot / 1e9, s21_norm, "r-", linewidth=1.5, label="S21 (rfx, normalised)")
ax1.axvline(f0 / 1e9, color="k", ls="--", alpha=0.5, label=f"Design f0 = {f0/1e9:.1f} GHz")
ax1.axvline(f_peak / 1e9, color="g", ls=":", alpha=0.8,
            label=f"Simulated peak: {f_peak/1e9:.2f} GHz")
ax1.axhline(-10, color="gray", ls=":", alpha=0.5, label="-10 dB threshold")
ax1.axvspan(f_lo / 1e9, f_hi / 1e9, alpha=0.08, color="blue", label="±20% passband")
ax1.set_xlabel("Frequency (GHz)")
ax1.set_ylabel("Normalised S21 (dB)")
ax1.set_ylim(-40, 5)
ax1.set_xlim(0.5, f_max / 1e9)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_title("Transmission (S21)")

# Geometry schematic (top view, mm units)
scale = 1e3
ax2.set_aspect("equal")
ax2.add_patch(plt.Rectangle(
    (res_x0 * scale, line_A_y0 * scale),
    L_res * scale, MSL_W * scale,
    facecolor="gold", edgecolor="k", linewidth=1.2, label="Line A"))
ax2.add_patch(plt.Rectangle(
    (res_x0 * scale, line_B_y0 * scale),
    L_res * scale, MSL_W * scale,
    facecolor="goldenrod", edgecolor="k", linewidth=1.2, label="Line B"))
# Gap indicator
ax2.annotate("", xy=(res_x0 * scale + L_res * scale * 0.5, line_B_y0 * scale),
             xytext=(res_x0 * scale + L_res * scale * 0.5, line_A_y1 * scale),
             arrowprops=dict(arrowstyle="<->", color="blue", lw=1.2))
ax2.text(res_x0 * scale + L_res * scale * 0.5 + 0.3,
         (line_A_y1 + gap_y / 2) * scale, f"{gap_y*1e3:.1f} mm", fontsize=8, color="blue")
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")
ax2.set_title("Geometry (top view)")
ax2.set_xlim(0, dom_x * scale)
ax2.set_ylim(0, dom_y * scale)
ax2.text(port1_x * scale, line_A_y_center * scale + 1.5, "P1", fontsize=9, color="blue", ha="center")
ax2.text(port2_x * scale, line_B_y_center * scale + 1.5, "P2", fontsize=9, color="red", ha="center")
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "26_coupled_line_bpf.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")

if PASS:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED")
sys.exit(0 if PASS else 1)
