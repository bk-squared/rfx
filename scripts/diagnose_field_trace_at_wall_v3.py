"""Compare Setup A and Setup B in detail at the probe location (g=174 = phys 154mm).

Key finding so far:
  Setup A wall LEFT FACE = g=175 (phys 155 mm). Wall is 2 cells thick (g=175,176).
  Setup B PEC face       = g=176 (phys 156 mm). One cell DOWNSTREAM of A's wall.

Both grids have CPML lo at g=0..19 (phys -20..-1 mm). probe at phys 154 mm = g=174.

Question 1: at g=174 (probe), do A and B give same time-domain Hy(t), Ez(t)?
Question 2: is the difference in PEC face position (g=175 vs g=176) what causes
            the 0.914 vs 1.000 reflection coefficient?
"""
from __future__ import annotations
import numpy as np

A = np.load("/tmp/pec_field_trace_A.npz")
B = np.load("/tmp/pec_field_trace_B.npz")

ez_A, ey_A, hy_A, hz_A, ex_A = A["ez"], A["ey"], A["hy"], A["hz"], A["ex"]
ez_B, ey_B, hy_B, hz_B, ex_B = B["ez"], B["ey"], B["hy"], B["hz"], B["ex"]

dt_est = float(A["dt_estimate"])
snap_int = int(A["snap_interval"])
n_snap = ez_A.shape[0]
t_snap = np.arange(n_snap) * snap_int * dt_est * 1e9  # ns
Ny = ez_A.shape[2]
j_mid = Ny // 2  # mid-y for TE10 sin(pi y/a) peak

print(f"Snapshots: {n_snap}, interval = {snap_int}, dt_est = {dt_est*1e12:.3f} ps")
print(f"Snapshot times span {t_snap[0]:.2f} -- {t_snap[-1]:.2f} ns")
print(f"Y-slice mid: j={j_mid}/{Ny}")

# Show full grid range max|ez|, max|hy| so we can see the standing-wave envelope
print("\n=== max|ez|, max|hy| per g (full grid range, indices 160-180) ===")
print(f"{'g':>4s} {'phys_mm':>8s}  {'max|ez|_A':>11s} {'max|ez|_B':>11s}  "
      f"{'max|hy|_A':>11s} {'max|hy|_B':>11s}")
for g in range(155, min(ez_A.shape[1], ez_B.shape[1]) + 5):
    in_B = g < ez_B.shape[1]
    a_ez = np.abs(ez_A[:, g, :]).max()
    a_hy = np.abs(hy_A[:, g, :]).max()
    if in_B:
        b_ez = np.abs(ez_B[:, g, :]).max()
        b_hy = np.abs(hy_B[:, g, :]).max()
        print(f"{g:4d} {g-20:8d}  {a_ez:11.3e} {b_ez:11.3e}  "
              f"{a_hy:11.3e} {b_hy:11.3e}")
    else:
        print(f"{g:4d} {g-20:8d}  {a_ez:11.3e} {'-':>11s}  "
              f"{a_hy:11.3e} {'-':>11s}")

# The critical comparison: time-domain at probe g=174
# But probe was actually at PHYSICAL location 154mm. With a CPML offset of 20,
# probe = g=174. Let's check what max|ez| at g=174 looks like in both setups.
print(f"\n=== Time-series at g=174 (phys 154 mm = probe location), j_mid={j_mid} ===")
ez_A_t = ez_A[:, 174, j_mid]
ez_B_t = ez_B[:, 174, j_mid]
hy_A_t = hy_A[:, 174, j_mid]
hy_B_t = hy_B[:, 174, j_mid]
ratio_ez = np.abs(ez_A_t).max() / np.abs(ez_B_t).max()
ratio_hy = np.abs(hy_A_t).max() / np.abs(hy_B_t).max()
print(f"  max|ez_A| = {np.abs(ez_A_t).max():.4e}, max|ez_B| = {np.abs(ez_B_t).max():.4e}, "
      f"ratio A/B = {ratio_ez:.4f}")
print(f"  max|hy_A| = {np.abs(hy_A_t).max():.4e}, max|hy_B| = {np.abs(hy_B_t).max():.4e}, "
      f"ratio A/B = {ratio_hy:.4f}")

# Modal V/I — V = -integral(Ey * sin(pi y/a) dy) at i, j from 0 to A_WG
# Probe records something analogous. Let's compute approximate modal V proxy:
# For TE10 in WR-90 with E along y, V = -integral Ey dy weighted by sin(pi y/a).
# Here we don't have ey at probe directly; we have ez (which is z-perp). Hmm.
# Actually for TE10 propagating in +x, primary Ey, primary Hx and Hz. Ez should be ~0
# in interior. Let me check.
print(f"\n  Setup A: max|ez| (z-perp, should be ~0 for TE10) = {np.abs(ez_A[:, 174, :]).max():.4e}")
print(f"  Setup A: max|ey| (transverse, primary)            = {np.abs(ey_A[:, 174, :]).max():.4e}")
print(f"  Setup A: max|hx| (longitudinal, primary)          = NOT IN SNAPSHOT")
print(f"  Setup A: max|hz| (transverse to propagation)      = {np.abs(hz_A[:, 174, :]).max():.4e}")
print(f"  Setup A: max|hy| (perp to mode, should be ~0)     = {np.abs(hy_A[:, 174, :]).max():.4e}")

# Wait — the WR-90 cross-section spans x=propagation, y=22.86mm wide, z=10.16mm tall.
# TE10 = 1 half-wave in y, no variation in z. Dominant fields:
#   Ey (transverse, vertical)
#   Hx (along propagation, transverse to mode)
#   Hz (transverse to mode)
# But wait — that's not right either. For TE10 with cross-section (a, b) and propagation axis z,
# fields are Ey, Hx, Hz with Ey ~ sin(pi x/a). Here propagation is +x, cross-section (y, z) with
# y=A_WG=22.86mm wide. So cross-section coords are y, z with y the "broad" direction.
# For TE_10 with a=22.86mm in y, dominant Ey ~ sin(pi y / a).
# Wait, TE_mn modes in rectangular waveguide propagating in z: H_z ≠ 0, E_z = 0,
#   E_x ~ -(jωμ/k_c²) (m π/a) cos(mπx/a) sin(nπy/b),
#   E_y ~  (jωμ/k_c²) (n π/b) sin(mπx/a) cos(nπy/b).
# For TE_10 (m=1, n=0): E_x ~ 0, E_y ~ -(jωμ a/π) sin(π x/a).
# But here propagation is +x, not +z. So we rotate axes: my x→z', y→x', z→y' or similar.
# In rfx convention with propagation +x, cross-section is (y, z), and TE10 with a=A_WG (along y)
# has Ey ~ sin(pi y / A_WG) only (no z dependence; n=0 along z).

# So the dominant E at probe should be Ey, not Ez. Let me redo my expectations.
# But the current snapshot showed max|ez_A| = 28-100 V/m, max|ey_A| = ~5e-5 V/m.
# That means Ez is dominant?! No wait — let me re-check the array layouts.

# Actually maybe rfx has x=propagation, y=narrow, z=broad? Or arbitrary? Let me check by
# looking at the WR-90 dims: A_WG = 22.86e-3 (broad), B_WG = 10.16e-3 (narrow). With y=A_WG (broad)
# and z=B_WG (narrow), TE10 has dominant Ey, sin(pi y/A_WG). So Ey should be dominant at probe.
# But snapshot shows tiny Ey ~ 5e-5 and big Ez ~ 100 V/m. Contradicts unless slice axis is wrong.

# OH! The snapshot was taken with slice_axis=2 (z), slice_index = Nz//2 = 5.
# Snapshot shape (n_snap, Nx, Ny). But maybe the "ez" component, when sliced along z,
# returns ez at z=5. And ez has shape (Nx, Ny, Nz) — so slicing axis 2 gives (Nx, Ny). Correct.
# But the dominant value being ez means... hmm, maybe rfx swaps y and z conventions?

# OR — maybe the mode_type='TE' with mode=(1,0) in rfx puts mode along z (b dim) instead of y (a dim).
# Need to check rfx convention.

# Print ratio of components at probe to disambiguate
print("\n=== Field component magnitudes at probe (g=174), summed over snapshots and y ===")
for comp_name, fA in [("ex", ex_A), ("ey", ey_A), ("ez", ez_A), ("hy", hy_A), ("hz", hz_A)]:
    print(f"  {comp_name}_A max over time × y at g=174: {np.abs(fA[:, 174, :]).max():.4e}")
