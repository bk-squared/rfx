"""Verify that fields past the wall in Setup A are TRULY zero, not just
small. Also check Hx at probe location (transverse to z-slice).
"""
from __future__ import annotations
import numpy as np

A = np.load("/tmp/pec_field_trace_A.npz")
ez, ey, hy, hz, ex = A["ez"], A["ey"], A["hy"], A["hz"], A["ex"]
print(f"Snapshot shape (n_snap, Nx, Ny) = {ez.shape}")

# Probe at g=174, wall LEFT FACE g=175.
# Past wall: g=175..240 in Setup A.
print("\n=== max|field| in cells past the wall (g=175..max) ===")
print(f"{'g':>4s}  {'max|ez|':>11s} {'max|ey|':>11s} {'max|hy|':>11s} "
      f"{'max|hz|':>11s} {'max|ex|':>11s}")
for g in [175, 176, 180, 190, 200, 210, 220, 230, 240]:
    if g < ez.shape[1]:
        e = max(abs(ez[:, g, :]).max(), 0)
        ey_v = max(abs(ey[:, g, :]).max(), 0)
        hy_v = max(abs(hy[:, g, :]).max(), 0)
        hz_v = max(abs(hz[:, g, :]).max(), 0)
        ex_v = max(abs(ex[:, g, :]).max(), 0)
        print(f"{g:4d}  {e:11.4e} {ey_v:11.4e} {hy_v:11.4e} "
              f"{hz_v:11.4e} {ex_v:11.4e}")

# Sum of |field|² across cells past wall over time
print("\n=== sum |field|^2 across all cells past wall (per snapshot) ===")
energy_per_snap = np.zeros(ez.shape[0])
for g in range(175, ez.shape[1]):
    energy_per_snap += (ez[:, g, :]**2 + ey[:, g, :]**2 + ex[:, g, :]**2).sum(axis=1)
    energy_per_snap += (hy[:, g, :]**2 + hz[:, g, :]**2).sum(axis=1)
print(f"  Total per-snap energy past wall: max = {energy_per_snap.max():.4e}, "
      f"mean = {energy_per_snap.mean():.4e}")
print(f"  Last snap energy past wall: {energy_per_snap[-1]:.4e}")

# Compare with energy in source-to-probe region
energy_pre = np.zeros(ez.shape[0])
for g in range(20, 175):
    energy_pre += (ez[:, g, :]**2 + ey[:, g, :]**2 + ex[:, g, :]**2).sum(axis=1)
    energy_pre += (hy[:, g, :]**2 + hz[:, g, :]**2).sum(axis=1)
print(f"  Total per-snap energy g=20..174: max = {energy_pre.max():.4e}, "
      f"mean = {energy_pre.mean():.4e}")
print(f"  Last snap energy g=20..174: {energy_pre[-1]:.4e}")

# Also check fields IN the wall cells (g=175, 176)
print("\n=== fields IN wall cells (g=175, 176) — should be tiny via mask ===")
for g in [175, 176]:
    e_max = abs(ez[:, g, :]).max()
    h_max = abs(hy[:, g, :]).max()
    print(f"  g={g}: max|ez|={e_max:.4e}, max|hy|={h_max:.4e}")
