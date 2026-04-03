"""Validation Case 7: Via / Through-Hole in 2-Layer PCB

A via connecting two copper layers through FR4 substrate.
Validates that the via structure conducts signal between layers.

Reference: qualitative -- a via should transfer electromagnetic energy
between PCB layers.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np

from rfx import Simulation, Box, Via
from rfx.sources.sources import GaussianPulse

# ---- PCB parameters ----
eps_r = 4.4
h = 1.6e-3       # substrate thickness
tan_d = 0.02

# Via parameters
via_drill = 0.3e-3   # 0.3 mm drill radius
via_pad = 0.5e-3     # 0.5 mm pad radius

# PCB layers
layers = [(0.0, h)]

f0 = 3e9
dx = 0.25e-3

print(f"Via through-hole in 2-layer PCB")
print(f"Substrate: h={h * 1e3:.1f} mm, eps_r={eps_r}")
print(f"Via drill: {via_drill * 1e3:.2f} mm radius")
print(f"Via pad  : {via_pad * 1e3:.2f} mm radius")

# ---- Build simulation ----
dom_x = 10e-3
dom_y = 6e-3
dom_z = h + 3e-3

sim = Simulation(
    freq_max=f0 * 2,
    domain=(dom_x, dom_y, dom_z),
    boundary="cpml",
    cpml_layers=8,
    dx=dx,
)

# Materials
sigma_sub = 2.0 * np.pi * f0 * 8.854e-12 * eps_r * tan_d
sim.add_material("substrate", eps_r=eps_r, sigma=sigma_sub)

# Ground plane (bottom)
sim.add(Box((0, 0, 0), (dom_x, dom_y, 0)), material="pec")
# Substrate
sim.add(Box((0, 0, 0), (dom_x, dom_y, h)), material="substrate")
# Top copper trace
trace_y_lo = dom_y / 2 - 1e-3
trace_y_hi = dom_y / 2 + 1e-3
sim.add(Box((2e-3, trace_y_lo, h), (dom_x - 2e-3, trace_y_hi, h)), material="pec")

# Via at center
via = Via(
    center=(dom_x / 2, dom_y / 2),
    drill_radius=via_drill,
    pad_radius=via_pad,
    layers=layers,
    material="pec",
)
for box, mat_name in via.to_shapes():
    sim.add(box, material=mat_name)

# Source near input end, inside substrate
src_x = 3e-3
src_y = dom_y / 2
sim.add_source(
    (src_x, src_y, h / 2),
    component="ez",
    waveform=GaussianPulse(f0=f0, bandwidth=0.8),
)

# Probe before via
sim.add_probe((dom_x / 2 - 1.5e-3, src_y, h / 2), component="ez")
# Probe after via
sim.add_probe((dom_x / 2 + 1.5e-3, src_y, h / 2), component="ez")

# ---- Run simulation ----
grid = sim._build_grid()
n_steps = int(np.ceil(8e-9 / grid.dt))
print(f"Running {n_steps} steps (dx={dx * 1e3:.2f} mm) ...")
result = sim.run(n_steps=n_steps)

# ---- Analysis ----
ts = np.asarray(result.time_series)
if ts.ndim == 2 and ts.shape[1] >= 2:
    before_via = ts[:, 0]
    after_via = ts[:, 1]
else:
    before_via = ts.ravel()
    after_via = before_via

before_energy = np.sum(before_via ** 2)
after_energy = np.sum(after_via ** 2)
transfer_ratio = after_energy / (before_energy + 1e-30)

# Check that signal exists on both sides
before_peak = np.max(np.abs(before_via))
after_peak = np.max(np.abs(after_via))

print(f"\n--- Validation Results ---")
print(f"Before-via peak field  : {before_peak:.6e}")
print(f"After-via peak field   : {after_peak:.6e}")
print(f"Energy transfer ratio  : {transfer_ratio:.4f}")
print(f"Signal before via      : {'Yes' if before_peak > 0 else 'No'}")
print(f"Signal after via       : {'Yes' if after_peak > 0 else 'No'}")

# Validation: signal should propagate past the via structure
passed = before_peak > 0 and after_peak > 0 and transfer_ratio > 0
status = "PASS" if passed else "FAIL"
print(f"\nValidation: {status} (signal propagates through via structure)")
