"""Cross-validation: 3D PEC Cavity Resonance (Analytical)

Validates resonance frequency extraction against exact analytical solution
for a rectangular PEC cavity.

Analytical TM_mnp resonance:
  f_mnp = (c/2) * sqrt((m/a)^2 + (n/b)^2 + (p/d)^2)

For TM110 (m=1, n=1, p=0):
  f_110 = (c/2) * sqrt((1/a)^2 + (1/b)^2)

PASS criteria:
  - Harminv resonance within 1% of analytical f_110
  - At least one clean mode detected with Q > 100

Save: examples/crossval/14_cavity_q_3d.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

C0 = 2.998e8

# Cavity dimensions
a = 50e-3  # x: 50 mm
b = 40e-3  # y: 40 mm
d = 30e-3  # z: 30 mm

# Analytical TM110
f_110 = C0 / 2 * np.sqrt((1 / a) ** 2 + (1 / b) ** 2)

# Also compute TM101, TM011, TM111 for reference
f_101 = C0 / 2 * np.sqrt((1 / a) ** 2 + (1 / d) ** 2)
f_011 = C0 / 2 * np.sqrt((1 / b) ** 2 + (1 / d) ** 2)
f_111 = C0 / 2 * np.sqrt((1 / a) ** 2 + (1 / b) ** 2 + (1 / d) ** 2)

print("=" * 60)
print("Cross-Validation: 3D PEC Cavity Resonance")
print("=" * 60)
print(f"Cavity: {a*1e3:.0f} x {b*1e3:.0f} x {d*1e3:.0f} mm")
print(f"Analytical modes:")
print(f"  TM110: {f_110/1e9:.4f} GHz")
print(f"  TM101: {f_101/1e9:.4f} GHz")
print(f"  TM011: {f_011/1e9:.4f} GHz")
print(f"  TM111: {f_111/1e9:.4f} GHz")
print()

# Mesh
dx = 2.0e-3  # 2 mm cells → 25x20x15 cells
f_max = 8e9  # search up to 8 GHz

sim = Simulation(
    freq_max=f_max,
    domain=(a, b, d),
    dx=dx,
    boundary="pec",
)

# Source at asymmetric position to excite multiple modes
src_pos = (a / 3, b / 4, d / 3)
sim.add_source(src_pos, "ez", waveform=GaussianPulse(f0=5e9, bandwidth=0.8))
sim.add_probe(src_pos, "ez")

# Also place probe at a different point
probe2 = (a * 0.6, b * 0.7, d * 0.5)
sim.add_probe(probe2, "ez")

# Run
t0 = time.time()
result = sim.run(n_steps=4000)
elapsed = time.time() - t0
print(f"Simulation time: {elapsed:.1f}s")

# Harminv resonance extraction
modes = result.find_resonances(freq_range=(2e9, 8e9), probe_idx=0)
print(f"\nHarminv modes found: {len(modes)}")

PASS = True

if modes:
    # Sort by frequency
    modes_sorted = sorted(modes, key=lambda m: m.freq)

    analytical = [
        ("TM110", f_110),
        ("TM101", f_101),
        ("TM011", f_011),
        ("TM111", f_111),
    ]
    analytical.sort(key=lambda x: x[1])

    print(f"\n{'Mode':<10} {'Analytical (GHz)':<18} {'rfx (GHz)':<15} {'Error':<10} {'Q':<10}")
    print("-" * 65)

    matched = 0
    for name, f_ana in analytical:
        # Find closest Harminv mode
        best = min(modes_sorted, key=lambda m: abs(m.freq - f_ana))
        err = abs(best.freq - f_ana) / f_ana
        print(f"{name:<10} {f_ana/1e9:<18.4f} {best.freq/1e9:<15.4f} {err*100:<10.1f}% {best.Q:<10.0f}")
        if err < 0.02:
            matched += 1

    print(f"\nMatched {matched}/{len(analytical)} modes within 2%")

    # Primary check: TM110 (lowest mode)
    best_110 = min(modes_sorted, key=lambda m: abs(m.freq - f_110))
    err_110 = abs(best_110.freq - f_110) / f_110

    if err_110 < 0.01:
        print(f"PASS: TM110 error = {err_110*100:.2f}% < 1%")
    else:
        print(f"FAIL: TM110 error = {err_110*100:.2f}% (expected < 1%)")
        PASS = False

    if best_110.Q > 100:
        print(f"PASS: Q = {best_110.Q:.0f} > 100 (lossless cavity)")
    else:
        print(f"FAIL: Q = {best_110.Q:.0f} (expected > 100 for lossless)")
        PASS = False
else:
    print("FAIL: no modes found")
    PASS = False

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"PEC Cavity {a*1e3:.0f}x{b*1e3:.0f}x{d*1e3:.0f} mm", fontsize=14)

# Time series
ts = np.array(result.time_series)
dt = result.dt
t_us = np.arange(ts.shape[0]) * dt * 1e9
ax1.plot(t_us, ts[:, 0], linewidth=0.5)
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Ez amplitude")
ax1.set_title("Time-domain probe signal")
ax1.grid(True, alpha=0.3)

# Spectrum
fft_sig = ts[:, 0]
window = np.hanning(len(fft_sig))
spec = np.abs(np.fft.rfft(fft_sig * window))
freqs_fft = np.fft.rfftfreq(len(fft_sig), d=dt)
f_mask = (freqs_fft > 1e9) & (freqs_fft < 8e9)

ax2.plot(freqs_fft[f_mask] / 1e9, 20 * np.log10(spec[f_mask] + 1e-30),
         linewidth=0.8, label="rfx spectrum")
for name, f_ana in analytical:
    ax2.axvline(f_ana / 1e9, color="r", ls="--", alpha=0.4, linewidth=0.8)
    ax2.text(f_ana / 1e9, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 0,
             name, fontsize=7, ha="center", va="bottom", color="r")
ax2.set_xlabel("Frequency (GHz)")
ax2.set_ylabel("Spectrum (dB)")
ax2.set_title("Resonance spectrum vs analytical")
ax2.grid(True, alpha=0.3)
# Re-add markers after setting limits
for name, f_ana in analytical:
    ax2.axvline(f_ana / 1e9, color="r", ls="--", alpha=0.4, linewidth=0.8)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "14_cavity_q_3d.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")

if PASS:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED")
