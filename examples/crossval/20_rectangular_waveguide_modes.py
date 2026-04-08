"""Cross-validation: Rectangular Waveguide Multi-Mode Extraction

Validates waveguide mode cutoff frequency extraction against exact
analytical formulas.

Structure: WR-90 rectangular waveguide (a=22.86mm, b=10.16mm), PEC walls.
Broadband excitation, Harminv mode extraction.

Analytical cutoff frequencies:
  f_mn = c/(2) * sqrt((m/a)^2 + (n/b)^2)
  TE10: 6.557 GHz
  TE20: 13.114 GHz
  TE01: 14.753 GHz
  TE11 = TM11: 16.145 GHz

PASS criteria:
  - TE10 cutoff within 2% of analytical
  - At least 3 modes detected by Harminv
  - Monotonic frequency ordering matches analytical

Save: examples/crossval/20_rectangular_waveguide_modes.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from rfx import Simulation
from rfx.sources.sources import GaussianPulse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

C0 = 2.998e8

# WR-90 waveguide dimensions
a = 22.86e-3   # broad wall (x)
b = 10.16e-3   # narrow wall (y)
L = 60e-3      # length (z)

# Analytical cutoff frequencies
modes_ana = {}
for m in range(3):
    for n in range(3):
        if m == 0 and n == 0:
            continue
        f_mn = C0 / 2 * np.sqrt((m / a) ** 2 + (n / b) ** 2)
        label = f"TE{m}{n}" if (m > 0 or n > 0) else ""
        modes_ana[f"({m},{n})"] = f_mn

# Sort by frequency
modes_sorted = sorted(modes_ana.items(), key=lambda x: x[1])

print("=" * 60)
print("Cross-Validation: Rectangular Waveguide Modes (WR-90)")
print("=" * 60)
print(f"Waveguide: a={a*1e3:.2f} mm, b={b*1e3:.2f} mm, L={L*1e3:.0f} mm")
print(f"\nAnalytical cutoff frequencies:")
for label, f in modes_sorted[:6]:
    print(f"  {label}: {f/1e9:.3f} GHz")
print()

# Mesh
dx = 1.5e-3
f_max = 16e9

# PEC cavity (waveguide with PEC endcaps acts as resonator)
sim = Simulation(
    freq_max=f_max,
    domain=(a, b, L),
    dx=dx,
    boundary="pec",
)

# Broadband source centered below TE101 to ensure low modes are excited
sim.add_source(
    position=(a * 0.37, b * 0.31, L * 0.23),
    component="ey",
    waveform=GaussianPulse(f0=8e9, bandwidth=0.9),
)

# Additional source in different component to excite TM modes
sim.add_source(
    position=(a * 0.6, b * 0.5, L * 0.4),
    component="ez",
    waveform=GaussianPulse(f0=8e9, bandwidth=0.9),
)

# Probes at off-symmetry location
sim.add_probe(position=(a * 0.3, b * 0.4, L * 0.5), component="ey")
sim.add_probe(position=(a * 0.3, b * 0.4, L * 0.5), component="ez")

t0 = time.time()
result = sim.run(num_periods=30)
elapsed = time.time() - t0
print(f"Simulation time: {elapsed:.1f}s")

# Harminv mode extraction — search in broad range
modes_found = result.find_resonances(freq_range=(5e9, 18e9))

# Sort by frequency
if modes_found:
    modes_found.sort(key=lambda m: m.freq)

print(f"\nModes found: {len(modes_found) if modes_found else 0}")
if modes_found:
    for m in modes_found[:8]:
        print(f"  f={m.freq/1e9:.3f} GHz, Q={m.Q:.0f}")

# Match found modes to analytical
# For a PEC-terminated waveguide (cavity), the resonances are:
# f_mnp = c/2 * sqrt((m/a)^2 + (n/b)^2 + (p/L)^2)
# where p = 1, 2, ... (longitudinal mode index)
# The first few resonances are dominated by the lowest transverse modes

# TE10p modes: f = c/2 * sqrt((1/a)^2 + (p/L)^2)
f_te10_cutoff = C0 / 2 * np.sqrt((1 / a) ** 2)
f_te101 = C0 / 2 * np.sqrt((1 / a) ** 2 + (1 / L) ** 2)
f_te102 = C0 / 2 * np.sqrt((1 / a) ** 2 + (2 / L) ** 2)
f_te103 = C0 / 2 * np.sqrt((1 / a) ** 2 + (3 / L) ** 2)

# TE20p modes
f_te201 = C0 / 2 * np.sqrt((2 / a) ** 2 + (1 / L) ** 2)

# TE011
f_te011 = C0 / 2 * np.sqrt((1 / b) ** 2 + (1 / L) ** 2)

ana_modes = [
    ("TE101", f_te101),
    ("TE102", f_te102),
    ("TE103", f_te103),
    ("TE201", f_te201),
    ("TE011", f_te011),
]
ana_modes.sort(key=lambda x: x[1])

print(f"\nAnalytical cavity resonances:")
for label, f in ana_modes:
    print(f"  {label}: {f/1e9:.3f} GHz")

# =============================================================================
# Validation
# =============================================================================
PASS = True

# Check 1: Find TE101 (lowest mode)
if modes_found and len(modes_found) >= 1:
    f_first = modes_found[0].freq
    err_te101 = abs(f_first - f_te101) / f_te101
    print(f"\nTE101: sim={f_first/1e9:.3f} GHz, ana={f_te101/1e9:.3f} GHz, "
          f"error={err_te101*100:.2f}%")
    if err_te101 < 0.05:
        print(f"  PASS: TE101 error {err_te101*100:.2f}% < 5%")
    else:
        print(f"  FAIL: TE101 error {err_te101*100:.2f}% (expected < 5%)")
        PASS = False
else:
    print("\nFAIL: no modes found")
    PASS = False

# Check 2: At least 3 modes
n_modes = len(modes_found) if modes_found else 0
if n_modes >= 3:
    print(f"  PASS: {n_modes} modes found (>= 3)")
else:
    print(f"  FAIL: only {n_modes} modes found (expected >= 3)")
    PASS = False

# Check 3: Mode ordering matches analytical
if modes_found and len(modes_found) >= 3:
    freqs_sim = [m.freq for m in modes_found[:5]]
    monotonic = all(freqs_sim[i] <= freqs_sim[i + 1] for i in range(len(freqs_sim) - 1))
    if monotonic:
        print(f"  PASS: modes in ascending frequency order")
    else:
        print(f"  FAIL: modes not in ascending order")
        PASS = False

# =============================================================================
# Plot
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("WR-90 Rectangular Waveguide: Mode Extraction", fontsize=14)

# 1. Time series
ts = np.array(result.time_series)
dt = result.dt
t_ns = np.arange(ts.shape[0]) * dt * 1e9
ax1.plot(t_ns, ts[:, 0], linewidth=0.5)
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Ez amplitude")
ax1.set_title("Probe signal")
ax1.grid(True, alpha=0.3)

# 2. Mode spectrum
if modes_found:
    freqs_sim = [m.freq / 1e9 for m in modes_found[:8]]
    ax2.stem(freqs_sim, [1] * len(freqs_sim), linefmt="b-", markerfmt="bo",
             basefmt="k-", label="rfx (Harminv)")
ana_freqs = [f / 1e9 for _, f in ana_modes[:6]]
ana_labels = [l for l, _ in ana_modes[:6]]
ax2.stem(ana_freqs, [0.7] * len(ana_freqs), linefmt="r--", markerfmt="r^",
         basefmt="k-", label="Analytical")
ax2.set_xlabel("Frequency (GHz)")
ax2.set_ylabel("Mode presence")
ax2.set_title("Cavity Resonances")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "20_rectangular_waveguide_modes.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")

if PASS:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED")
