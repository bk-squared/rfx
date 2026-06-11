"""Re-analyze /tmp/pec_field_trace_{A,B}.npz with correct CPML offset.

CPML adds 20 layers on each open boundary, so:
  Setup A (CPML lo+hi):  grid index = physical_x_mm + 20
  Setup B (CPML lo, PEC hi): grid index = physical_x_mm + 20

Wall LEFT FACE at physical 155 mm → grid index 175.
Probe at physical 154 mm → grid index 174.
"""
from __future__ import annotations
import numpy as np

A = np.load("/tmp/pec_field_trace_A.npz")
B = np.load("/tmp/pec_field_trace_B.npz")

ez_A, ey_A, hy_A, hz_A, ex_A = A["ez"], A["ey"], A["hy"], A["hz"], A["ex"]
ez_B, ey_B, hy_B, hz_B, ex_B = B["ez"], B["ey"], B["hy"], B["hz"], B["ex"]
Nx_A, Nx_B = ez_A.shape[1], ez_B.shape[1]
print(f"Setup A: Nx={Nx_A} (= 200 interior + 20 CPML_lo + 20 CPML_hi + 1)")
print(f"Setup B: Nx={Nx_B} (= 156 interior + 20 CPML_lo + 1)")

# Physical-to-grid mapping: g = phys_mm + 20 (CPML_lo offset)
# Setup A wall: phys 155 mm → g = 175 (LEFT FACE), 176 (RIGHT FACE+1 cell-thick)
# Probe       : phys 154 mm → g = 174
# In Setup B, PEC at +x boundary at g = Nx_B-1 = 176, which is phys = 156 mm? Wait...
#
# Setup B has DOMAIN_X = 0.156, so 156 interior cells. With +1 for boundary array
# size, last grid index Nx_B-1 = 176. PEC at the +x face is at grid index 176.
# That corresponds to phys = (176 - 20) = 156 mm. So PEC face at 156 mm, NOT 155 mm.
# This is a 1-cell offset between Setup A (mask at 155 mm) and Setup B (face at 156 mm)!

print("\n=== A vs B: max|field| at CPML-adjusted grid indices around wall ===")
print(f"{'g':>4s} {'phys_mm':>8s}  "
      f"{'ez_A':>9s} {'ez_B':>9s}  "
      f"{'ey_A':>9s} {'ey_B':>9s}  "
      f"{'hy_A':>9s} {'hy_B':>9s}  "
      f"{'hz_A':>9s} {'hz_B':>9s}  "
      f"{'ex_A':>9s} {'ex_B':>9s}")
for g in range(170, min(Nx_A, Nx_B) + 5):
    in_B = g < Nx_B
    phys_mm = g - 20
    row = f"{g:4d} {phys_mm:8d}  "
    for fA, fB in [(ez_A, ez_B), (ey_A, ey_B), (hy_A, hy_B), (hz_A, hz_B), (ex_A, ex_B)]:
        a = np.abs(fA[:, g, :]).max()
        if in_B:
            b = np.abs(fB[:, g, :]).max()
            row += f"{a:9.2e} {b:9.2e}  "
        else:
            row += f"{a:9.2e} {'-':>9s}  "
    print(row)

print("\n=== SETUP A: which cells are zero (PEC wall location) ===")
print("If apply_pec_mask is correctly placing wall at g=175,176,177 (phys 155,156,157):")
print("  - max|ez| at those g should be 0 (Ez tangential to wall in y, full y-span)")
print("  - max|ey| at those g should be 0 (Ey tangential too)")
print("  - max|hy| at those g should be free (no zeroing on H)")
ny_A = ez_A.shape[2]
nz_A_z = 0  # snapshot already z-sliced
print(f"\n  ez_A.shape = {ez_A.shape} (n_snap, Nx, Ny)  Ny={ny_A}")
print(f"  Per-y snapshot: max(over time) of |ez_A[:, g=175, :]|:")
for g in [173, 174, 175, 176, 177, 178]:
    z_max_per_y = np.abs(ez_A[:, g, :]).max(axis=0)  # max over time per y
    print(f"    g={g} (phys={g-20}mm): max|ez| per y = {z_max_per_y}")

print("\n=== Ey snapshot at LEFT FACE position (g=175): per-y max ===")
for g in [173, 174, 175, 176, 177, 178]:
    z_max_per_y = np.abs(ey_A[:, g, :]).max(axis=0)
    print(f"  g={g} (phys={g-20}mm): max|ey| per y = {z_max_per_y}")
