"""Compare V/(Z*I) phase angle in Setup A vs B.2 to identify what shifts |bwd|.

For a perfect standing wave from PEC:
  V(x) = V_0 [e^{-jβx} - e^{+jβx}] (Γ=-1)
  Z*I(x) = V_0 [e^{-jβx} + e^{+jβx}]
  → V / (Z·I) = -j tan(βx)  (purely imaginary)
  → angle(V/(Z·I)) = ±90°

If Setup A's V and (Z·I) are NOT exactly 90° apart, |fwd| ≠ |bwd|.
"""
from __future__ import annotations
import numpy as np

D = np.load("/tmp/VI_compare_A_B2.npz")
freqs = D["freqs"]
Vd_A, Vd_B = D["Vd_A"], D["Vd_B"]
Id_A, Id_B = D["Id_A"], D["Id_B"]
Z_A, Z_B = D["Z_A"], D["Z_B"]
fwd_A, fwd_B = D["fwd_A"], D["fwd_B"]
bwd_A, bwd_B = D["bwd_A"], D["bwd_B"]

print(f"{'f_GHz':>7s} {'|V/(ZI)|_A':>12s} {'|V/(ZI)|_B':>12s} "
      f"{'arg(V/ZI)_A':>13s} {'arg(V/ZI)_B':>13s} "
      f"{'Δarg(deg)':>11s} {'r_A':>8s} {'r_B':>8s}")
for k in range(len(freqs)):
    ratio_A = Vd_A[k] / (Z_A[k] * Id_A[k])
    ratio_B = Vd_B[k] / (Z_B[k] * Id_B[k])
    arg_A = np.degrees(np.angle(ratio_A))
    arg_B = np.degrees(np.angle(ratio_B))
    rA = abs(bwd_A[k]) / max(abs(fwd_A[k]), 1e-30)
    rB = abs(bwd_B[k]) / max(abs(fwd_B[k]), 1e-30)
    delta = arg_A - arg_B
    print(f"{freqs[k]/1e9:7.2f} {abs(ratio_A):12.4f} {abs(ratio_B):12.4f} "
          f"{arg_A:13.4f} {arg_B:13.4f} {delta:11.4f} "
          f"{rA:8.4f} {rB:8.4f}")

# Cross-correlation: Re(V · conj(Z·I))
print("\n=== Re(V · conj(Z·I)) ratio ===")
print(f"{'f_GHz':>7s} {'|V|^2_A':>11s} {'2Re(VZI*)_A':>13s} {'|ZI|^2_A':>11s} "
      f"{'|fwd^2|*4':>11s} {'|bwd^2|*4':>11s}")
for k in range(0, len(freqs), 2):
    V = Vd_A[k]; ZI = Z_A[k] * Id_A[k]
    fwd2_4 = abs(V)**2 + 2 * np.real(V * np.conj(ZI)) + abs(ZI)**2
    bwd2_4 = abs(V)**2 - 2 * np.real(V * np.conj(ZI)) + abs(ZI)**2
    print(f"{freqs[k]/1e9:7.2f} {abs(V)**2:11.3e} "
          f"{2*np.real(V * np.conj(ZI)):13.3e} {abs(ZI)**2:11.3e} "
          f"{fwd2_4:11.3e} {bwd2_4:11.3e}")

print("\n=== Same for B.2 ===")
print(f"{'f_GHz':>7s} {'|V|^2_B':>11s} {'2Re(VZI*)_B':>13s} {'|ZI|^2_B':>11s} "
      f"{'|fwd^2|*4':>11s} {'|bwd^2|*4':>11s}")
for k in range(0, len(freqs), 2):
    V = Vd_B[k]; ZI = Z_B[k] * Id_B[k]
    fwd2_4 = abs(V)**2 + 2 * np.real(V * np.conj(ZI)) + abs(ZI)**2
    bwd2_4 = abs(V)**2 - 2 * np.real(V * np.conj(ZI)) + abs(ZI)**2
    print(f"{freqs[k]/1e9:7.2f} {abs(V)**2:11.3e} "
          f"{2*np.real(V * np.conj(ZI)):13.3e} {abs(ZI)**2:11.3e} "
          f"{fwd2_4:11.3e} {bwd2_4:11.3e}")
