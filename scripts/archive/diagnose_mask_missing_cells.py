"""Identify which cells differ between apply_pec_mask formula and face-style
zeroing at i=175 in Setup A's grid.

Key question: which (j, k) cells have pec_mask=True but mask_ey=False (or
similarly mask_ez=False)? These are the cells where face-style zeros but
apply_pec_mask skips. They're the suspects for the 8% loss.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Build a fake pec_mask matching what Setup A would produce for the 1-cell
# wall at i=175 (Box at 0.155-0.156, full WR-90 cross-section).
A_WG_MM, B_WG_MM = 22.86, 10.16
DX_MM = 1.0
CPML_LAYERS = 20

# Total grid sizes (with CPML+1)
Nx = int(round(200 / DX_MM)) + 2 * CPML_LAYERS + 1
Ny = int(round(A_WG_MM / DX_MM)) + 1  # +1 for the boundary cell
Nz = int(round(B_WG_MM / DX_MM)) + 1
print(f"Grid: Nx={Nx}, Ny={Ny}, Nz={Nz}")

# Box at x=0.155-0.156 (1 cell), y=0..A_WG, z=0..B_WG.
# Standard rasterizer: cell with center inside box is PEC.
# Cell i has center at i*dx in some convention; rfx uses cell center at (i+0.5)*dx?
# Let me try both and see which matches what Setup A used.

# From the snapshot data (Setup A): wall at g=175, 1 cell. So pec_mask[175, j, k] = T
# for j, k in cross-section. But which j, k exactly?
# With Ny=24 (we observed), A_WG=22.86mm, dx=1mm: cells j=0..23.
# Box covers y in [0, A_WG=22.86]: cells with center in this range.
# Likely cell convention: center at (j+0.5)*dx → cells j=0..21 (centers 0.5 to 21.5),
# cell j=22 center at 22.5mm > 22.86 → excluded? No wait 22.5 < 22.86 → included.
# cell j=23 center at 23.5mm > 22.86 → excluded.
# So pec_mask[175, j=0..22, k=0..?, ?] = T.

pec_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
# y range
ny_in = int(round(A_WG_MM / DX_MM))  # 22 cells, j=0..21 fully inside, j=22 partial
nz_in = int(round(B_WG_MM / DX_MM))  # 10 cells, k=0..9
print(f"Box covers j=0..{ny_in-1} ({ny_in} cells), k=0..{nz_in-1} ({nz_in} cells)")
# Conservative: cells with center in (0, A_WG): j=0..21 (centers 0.5 to 21.5)
# Approximate
pec_mask[175, 0:ny_in, 0:nz_in] = True

# Mask formula
def _mask_components(pm):
    mask_ex = pm & (np.roll(pm, 1, axis=0) | np.roll(pm, -1, axis=0))
    mask_ey = pm & (np.roll(pm, 1, axis=1) | np.roll(pm, -1, axis=1))
    mask_ez = pm & (np.roll(pm, 1, axis=2) | np.roll(pm, -1, axis=2))
    return mask_ex, mask_ey, mask_ez


mask_ex, mask_ey, mask_ez = _mask_components(pec_mask)

# Face-style mask: zero ALL j, k at i=175
face_ey = np.zeros_like(pec_mask)
face_ey[175, :, :] = True
face_ez = np.zeros_like(pec_mask)
face_ez[175, :, :] = True

# Find cells where face zeros but mask doesn't
diff_ey = face_ey & ~mask_ey
diff_ez = face_ez & ~mask_ez
print(f"\n=== cells at i=175 where FACE-style zeros but MASK does NOT ===")
print(f"  Ey: {diff_ey.sum()} cells")
print(f"  Ez: {diff_ez.sum()} cells")

if diff_ey.sum() > 0:
    j_idx, k_idx = np.where(diff_ey[175])
    print(f"\n  Ey cells at i=175 NOT zeroed by mask:")
    print(f"    (j, k) pairs (showing first 30):")
    for jj, kk in zip(j_idx[:30], k_idx[:30]):
        # pec_mask at this cell?
        pm_here = bool(pec_mask[175, jj, kk])
        # neighbors in y
        pm_jm1 = bool(pec_mask[175, jj-1 if jj > 0 else -1, kk])
        pm_jp1 = bool(pec_mask[175, (jj+1) % Ny, kk])
        print(f"      j={jj:2d}, k={kk:2d}: pec[{jj}]={pm_here}, "
              f"pec[{jj-1}]={pm_jm1}, pec[{jj+1}]={pm_jp1}")

if diff_ez.sum() > 0:
    j_idx, k_idx = np.where(diff_ez[175])
    print(f"\n  Ez cells at i=175 NOT zeroed by mask:")
    for jj, kk in zip(j_idx[:30], k_idx[:30]):
        pm_here = bool(pec_mask[175, jj, kk])
        pm_km1 = bool(pec_mask[175, jj, kk-1 if kk > 0 else -1])
        pm_kp1 = bool(pec_mask[175, jj, (kk+1) % Nz])
        print(f"      j={jj:2d}, k={kk:2d}: pec[{kk}]={pm_here}, "
              f"pec[{kk-1}]={pm_km1}, pec[{kk+1}]={pm_kp1}")

# Also look at where mask zeros but face doesn't (sanity)
extra_ey = mask_ey & ~face_ey
extra_ez = mask_ez & ~face_ez
print(f"\n  cells where MASK zeros but FACE does NOT:")
print(f"    Ey: {extra_ey.sum()} cells (face-style is per-plane so should be 0)")
print(f"    Ez: {extra_ez.sum()} cells")
