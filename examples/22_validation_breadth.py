"""Validation breadth: waveguide resonator + field animation.

Tests rfx on structures beyond the patch antenna to verify generality:
1. Rectangular waveguide resonator (TE101 mode, analytical reference)
2. Field animation from snapshots
"""
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import time

from rfx import Simulation, Box, GaussianPulse
from rfx.simulation import SnapshotSpec
from rfx.grid import C0

# ============================================================
# Test 1: Rectangular Waveguide Resonator (TE101)
# ============================================================
print("=" * 60)
print("TEST 1: Waveguide Resonator (TE101)")
print("=" * 60)

# WR-90 waveguide dimensions: 22.86 x 10.16 mm, length 30mm
a = 22.86e-3   # broad wall
b = 10.16e-3   # narrow wall
d = 30e-3      # cavity length

# TE101 resonance: f = c/2 * sqrt((1/a)^2 + (1/d)^2)
f_te101 = C0/2 * np.sqrt((1/a)**2 + (1/d)**2)
print(f"Analytical TE101: {f_te101/1e9:.3f} GHz")

dx = 0.5e-3
sim = Simulation(freq_max=15e9, domain=(a, b, d), boundary='pec', dx=dx)
# Source off-center to excite TE101 (Ey dominant)
sim.add_source((a/3, b/2, d/3), 'ey',
               waveform=GaussianPulse(f0=8e9, bandwidth=0.8))
sim.add_probe((a/3, b/2, d/3), 'ey')
# Snapshots for animation
sim.add_probe((a/2, b/2, d/2), 'ey')  # center probe

grid = sim._build_grid()
print(f"Grid: {grid.shape}, dx={dx*1e3}mm")

t0 = time.time()
result = sim.run(n_steps=3000,
                 snapshot=SnapshotSpec(interval=20, components=("ey",),
                                       slice_axis=1, slice_index=grid.ny//2))
elapsed = time.time() - t0
print(f"Runtime: {elapsed:.1f}s")

# Harminv resonance extraction
modes = result.find_resonances(freq_range=(5e9, 12e9), source_decay_time=0.1e-9)
print(f"Harminv: {len(modes)} modes")
for m in modes[:5]:
    err = abs(m.freq - f_te101) / f_te101 * 100
    print(f"  f={m.freq/1e9:.3f} GHz (err={err:.1f}%), Q={m.Q:.0f}")

if modes:
    best = modes[0]
    # Find the mode closest to TE101
    te101_candidates = [m for m in modes if abs(m.freq - f_te101)/f_te101 < 0.2]
    if te101_candidates:
        best = te101_candidates[0]
    err = abs(best.freq - f_te101) / f_te101 * 100
    print(f"TE101: {best.freq/1e9:.3f} GHz (err={err:.2f}%)")

# ============================================================
# Test 2: Field Animation
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Field Animation")
print("=" * 60)

if result.snapshots and "ey" in result.snapshots:
    from rfx.visualize3d import save_field_animation
    snap_data = {"ey": np.array(result.snapshots["ey"])}
    n_frames = snap_data["ey"].shape[0]
    print(f"Snapshot frames: {n_frames}")

    try:
        out = save_field_animation(snap_data, grid, filename="examples/22_waveguide_anim",
                                    component="ey", fps=10, dpi=72)
        print(f"Animation saved: {out}")
    except Exception as e:
        print(f"Animation error: {e}")
else:
    print("No snapshots available")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
print(f"Waveguide TE101: analytical={f_te101/1e9:.3f} GHz")
if modes:
    print(f"  FDTD: {best.freq/1e9:.3f} GHz, err={err:.2f}%")
print(f"Field animation: {'saved' if result.snapshots else 'N/A'}")
