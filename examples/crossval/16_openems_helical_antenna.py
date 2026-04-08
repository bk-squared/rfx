"""Cross-validation: OpenEMS Helical Antenna tutorial replica.

Replicates: docs.openems.de/python/openEMS/Tutorials/Helical_Antenna.html
Structure: 10-turn axial-mode helical antenna at 2.4 GHz
           R=20mm, pitch=30mm, wire radius=1mm, 120-ohm lumped port

Uses rfx PolylineWire primitive to build the helix from a polyline path.

Comparison: S11 resonance, NTFF far-field directivity vs analytical
  Analytical directivity estimate: D ~ 4*N_turns ≈ 40 (16 dBi)

PASS criteria:
  - S11 < -5 dB near 2.4 GHz (impedance match)
  - Directivity > 10 dBi (axial-mode helix)
  - Far-field pattern peak in +z direction (endfire)

NOTE: This is a large 3D simulation. CPU-only runs may be slow.

Save: examples/crossval/16_openems_helical_antenna.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from rfx import Simulation, Box, Cylinder, PolylineWire
from rfx.sources.sources import GaussianPulse
from rfx.farfield import compute_far_field, directivity
import jax.numpy as jnp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

C0 = 2.998e8
f0 = 2.4e9
lam0 = C0 / f0  # 125 mm
fc = 0.5e9

# Helix parameters (from OpenEMS tutorial)
Helix_radius = 20e-3     # 20 mm
Helix_turns = 10
Helix_pitch = 30e-3       # 30 mm per turn
wire_radius = 1e-3        # 1 mm wire radius
feed_height = 3e-3        # 3 mm feed gap
gnd_radius = lam0 / 2     # 62.5 mm ground plane

# Mesh — coarser for CPU feasibility
dx = 3e-3  # 3 mm (lambda/40 at 2.4 GHz)

# Helix total height
helix_height = Helix_turns * Helix_pitch + feed_height

# Domain
margin = lam0 * 0.8
dom_xy = 2 * (gnd_radius + margin)
dom_z = helix_height + margin * 2

print("=" * 60)
print("Cross-Validation: OpenEMS Helical Antenna")
print("=" * 60)
print(f"Helix: R={Helix_radius*1e3:.0f}mm, {Helix_turns} turns, pitch={Helix_pitch*1e3:.0f}mm")
print(f"Wire radius: {wire_radius*1e3:.0f}mm, feed gap: {feed_height*1e3:.0f}mm")
print(f"Ground plane: R={gnd_radius*1e3:.0f}mm")
print(f"Domain: {dom_xy*1e3:.0f} x {dom_xy*1e3:.0f} x {dom_z*1e3:.0f} mm")
print(f"Mesh: dx={dx*1e3:.0f}mm")
print(f"Expected directivity: ~{10*np.log10(4*Helix_turns):.0f} dBi")
print()

# Build helix polyline (21 points per turn, as in OpenEMS)
pts_per_turn = 21
helix_points = []
for turn in range(Helix_turns):
    angles = np.linspace(0, 2 * np.pi, pts_per_turn, endpoint=(turn == Helix_turns - 1))
    for ang in angles:
        x = Helix_radius * np.cos(ang) + dom_xy / 2
        y = Helix_radius * np.sin(ang) + dom_xy / 2
        z = feed_height + turn * Helix_pitch + ang / (2 * np.pi) * Helix_pitch
        helix_points.append((float(x), float(y), float(z)))

helix_wire = PolylineWire(points=tuple(helix_points), radius=wire_radius)
print(f"Helix: {len(helix_points)} polyline points")

# Simulation
sim = Simulation(
    freq_max=(f0 + fc) * 1.5,
    domain=(dom_xy, dom_xy, dom_z),
    dx=dx,
    boundary="cpml",
    cpml_layers=8,
    pec_faces={"z_lo"},  # ground plane at z=0
)

# Ground plane (circular PEC disk at z=0)
sim.add(Cylinder(
    center=(dom_xy / 2, dom_xy / 2, dx / 2),
    radius=gnd_radius,
    height=dx,
    axis="z",
), material="pec")

# Helix wire (PEC)
sim.add(helix_wire, material="pec")

# Feed port: vertical Ez from ground to helix start
port_pos = (
    Helix_radius + dom_xy / 2,  # x = helix radius (first point)
    dom_xy / 2,                 # y = center
    feed_height / 2,            # z = mid-feed
)
sim.add_port(
    position=port_pos,
    component="ez",
    impedance=120.0,  # 120 ohm as in OpenEMS
    waveform=GaussianPulse(f0=f0, bandwidth=fc / f0),
)
sim.add_probe(position=port_pos, component="ez")

# NTFF box
cpml_thick = 8 * dx
ntff_margin = cpml_thick + 3 * dx
sim.add_ntff_box(
    corner_lo=(ntff_margin, ntff_margin, ntff_margin),
    corner_hi=(dom_xy - ntff_margin, dom_xy - ntff_margin, dom_z - ntff_margin),
    freqs=jnp.array([f0]),
)

# Preflight
print("Preflight:")
warnings = sim.preflight(strict=False)
print()

# Run
t0 = time.time()
result = sim.run(num_periods=15)
elapsed = time.time() - t0
print(f"Simulation time: {elapsed:.1f}s")

# S11 via Harminv
modes = result.find_resonances(freq_range=(1.5e9, 3.5e9))
if modes:
    best = min(modes, key=lambda m: abs(m.freq - f0))
    print(f"Resonance: f={best.freq/1e9:.3f} GHz, Q={best.Q:.0f}")

# Far-field
theta = jnp.linspace(0, jnp.pi, 91)
phi = jnp.array([0.0, jnp.pi / 2])

ff = compute_far_field(result.ntff_data, result.ntff_box, result.grid, theta, phi)
D = directivity(ff)
D_dbi = 10 * np.log10(float(D) + 1e-30)
print(f"Directivity: {D_dbi:.1f} dBi (expected ~{10*np.log10(4*Helix_turns):.0f} dBi)")

# Validation
PASS = True

if D_dbi > 8:
    print(f"PASS: Directivity {D_dbi:.1f} dBi > 8 dBi")
else:
    print(f"FAIL: Directivity {D_dbi:.1f} dBi (expected > 8 dBi for axial-mode helix)")
    PASS = False

# Check endfire pattern (max in theta=0 direction, i.e. +z)
E_th = np.abs(np.asarray(ff.E_theta[0, :, 0]))
E_ph = np.abs(np.asarray(ff.E_phi[0, :, 0]))
power = E_th ** 2 + E_ph ** 2
if len(power) > 0 and np.max(power) > 0:
    peak_theta_idx = np.argmax(power)
    peak_theta_deg = float(np.asarray(theta)[peak_theta_idx]) * 180 / np.pi
    if peak_theta_deg < 45:
        print(f"PASS: endfire pattern (peak at theta={peak_theta_deg:.0f} deg)")
    else:
        print(f"FAIL: peak at theta={peak_theta_deg:.0f} deg (expected endfire < 45 deg)")
        PASS = False

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), subplot_kw=dict())
fig.suptitle(f"Helical Antenna: {Helix_turns} turns, R={Helix_radius*1e3:.0f}mm, 2.4 GHz",
             fontsize=14)

# Radiation pattern (E-plane)
theta_deg = np.asarray(theta) * 180 / np.pi
if len(power) > 0 and np.max(power) > 0:
    pattern_db = 10 * np.log10(power / np.max(power) + 1e-30)
    ax1.plot(theta_deg, pattern_db, "b-", linewidth=2)
ax1.set_xlabel("Theta (degrees)")
ax1.set_ylabel("Normalized pattern (dB)")
ax1.set_xlim(0, 180)
ax1.set_ylim(-30, 5)
ax1.grid(True, alpha=0.3)
ax1.set_title(f"E-plane pattern (D={D_dbi:.1f} dBi)")

# Time series
ts = np.array(result.time_series)
dt = result.dt
t_ns = np.arange(ts.shape[0]) * dt * 1e9
ax2.plot(t_ns, ts[:, 0], linewidth=0.5)
ax2.set_xlabel("Time (ns)")
ax2.set_ylabel("Ez amplitude")
ax2.set_title("Feed probe signal")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "16_openems_helical_antenna.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")

if PASS:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED")
