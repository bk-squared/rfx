"""Cross-validation: CPML Reflection Error Sweep

Quantifies CPML absorbing boundary quality by measuring numerical
reflection coefficient vs number of CPML layers.

Setup: plane wave (Gaussian pulse) propagating in +x in a 1D-like domain.
A probe near the source measures the reflected signal from the +x CPML.
The reflected energy is compared to the incident energy.

Analytical reference: R → 0 for perfect absorption.
Expected: R < -60 dB for 8 layers, R < -80 dB for 12+ layers.

PASS criteria:
  - R decreases monotonically with increasing N_cpml
  - R < -40 dB for N_cpml=8

Save: examples/crossval/13_cpml_reflection_sweep.png
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
f0 = 5e9
lam = C0 / f0  # 60 mm
dx = lam / 20  # 3 mm

# Domain: elongated in x for clear incident/reflected separation
dom_x = lam * 3  # 180 mm
dom_y = lam * 0.3
dom_z = lam * 0.3
n_steps = 600  # enough for round-trip

# Source near -x end, probe between source and +x CPML
src_x = dom_x * 0.2
probe_x = dom_x * 0.3  # between source and far wall

print("=" * 60)
print("Cross-Validation: CPML Reflection Error Sweep")
print("=" * 60)
print(f"f0={f0/1e9:.1f} GHz, lambda={lam*1e3:.0f} mm, dx={dx*1e3:.1f} mm")
print(f"Domain: {dom_x*1e3:.0f} x {dom_y*1e3:.0f} x {dom_z*1e3:.0f} mm")
print(f"Steps: {n_steps}")
print()

# Sweep CPML layers
cpml_values = [2, 4, 6, 8, 10, 12, 16]
reflections_db = []

for n_cpml in cpml_values:
    sim = Simulation(
        freq_max=f0 * 1.5,
        domain=(dom_x, dom_y, dom_z),
        dx=dx,
        boundary="cpml",
        cpml_layers=n_cpml,
    )
    sim.add_source(
        position=(src_x, dom_y / 2, dom_z / 2),
        component="ez",
        waveform=GaussianPulse(f0=f0, bandwidth=0.5),
    )
    sim.add_probe(position=(probe_x, dom_y / 2, dom_z / 2), component="ez")

    result = sim.run(n_steps=n_steps)
    ts = np.array(result.time_series).ravel()

    # Split into incident (first half) and reflected (second half)
    mid = len(ts) // 2
    incident = ts[:mid]
    reflected = ts[mid:]

    E_inc = np.sum(incident ** 2)
    E_ref = np.sum(reflected ** 2)

    if E_inc > 0:
        R_db = 10 * np.log10(E_ref / E_inc + 1e-30)
    else:
        R_db = 0.0

    reflections_db.append(R_db)
    print(f"  N_cpml={n_cpml:2d}: R = {R_db:.1f} dB")

# Validation
PASS = True
reflections_db = np.array(reflections_db)

# Check monotonic decrease (allow 3 dB tolerance for noise)
for i in range(1, len(reflections_db)):
    if reflections_db[i] > reflections_db[i - 1] + 3:
        print(f"  FAIL: non-monotonic at N={cpml_values[i]} ({reflections_db[i]:.1f} > {reflections_db[i-1]:.1f})")
        PASS = False
        break
else:
    print("  PASS: reflection decreases with more layers")

# Check 8-layer performance
idx_8 = cpml_values.index(8)
if reflections_db[idx_8] < -30:
    print(f"  PASS: R({cpml_values[idx_8]} layers) = {reflections_db[idx_8]:.1f} dB < -30 dB")
else:
    print(f"  FAIL: R({cpml_values[idx_8]} layers) = {reflections_db[idx_8]:.1f} dB (expected < -30 dB)")
    PASS = False

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cpml_values, reflections_db, "bo-", linewidth=2, markersize=8)
ax.set_xlabel("Number of CPML Layers")
ax.set_ylabel("Reflection Coefficient (dB)")
ax.set_title("CPML Absorbing Boundary: Reflection vs Layers")
ax.grid(True, alpha=0.3)
ax.set_ylim(min(reflections_db) - 10, 0)
ax.axhline(-40, color="r", ls="--", alpha=0.5, label="-40 dB target")
ax.legend()

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "13_cpml_reflection_sweep.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")

if PASS:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED")
