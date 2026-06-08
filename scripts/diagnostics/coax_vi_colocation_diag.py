"""Coaxial V/I extraction localization — Layer 0 (analytic, NO FDTD).

Goal: decide WHERE the |S11|>1 / angle(V/I)!=+-90 bias lives.
  - extractor MATH (radial line integral / azimuthal loop / interp / signs), or
  - FDTD-side co-location (Yee time half-step, axial z stagger, source plane).

Layer 0 feeds the public extractor a SYNTHETIC analytic TEM field for a known
reflection coefficient Gamma. For an ideal coax TEM superposition
  E_r_total  = E_r+ (1 + Gamma)
  H_phi_total= (E_r+/eta) (1 - Gamma)        # reflected wave flips H sign
the exact V/I = Z_TEM (1+Gamma)/(1-Gamma) and, with Z0=Z_TEM, S11 == Gamma
EXACTLY (derivation independent of A, of r_h, of the radial profile).

So: feed Gamma in, the extractor must return S11 == Gamma. Any deviation is a
PURE extractor-math defect (no FDTD involved). We sweep |Gamma|=1 phases (the
lossless-short family) and a matched Gamma=0.
"""
import sys; sys.path.insert(0, "/tmp/rfx-tj")
import numpy as np
from rfx.sources.coaxial_port import (
    coaxial_tem_reference_plane_vi_from_cartesian_plane,
    coaxial_tem_reference_plane_s11,
    SMA_PIN_RADIUS, SMA_OUTER_RADIUS, PTFE_EPS_R,
)

ETA0 = 376.730313668
a, b, epsr = SMA_PIN_RADIUS, SMA_OUTER_RADIUS, PTFE_EPS_R
eta = ETA0 / np.sqrt(epsr)
Z_TEM = (eta / (2.0 * np.pi)) * np.log(b / a)
print(f"a={a*1e3:.3f}mm b={b*1e3:.3f}mm eps_r={epsr}  eta={eta:.3f}  Z_TEM={Z_TEM:.4f} ohm")


def synth_plane(Gamma, *, dx, half_span, profile="1/r"):
    """Build analytic forward+reflected TEM tangential E/H on a Cartesian z-plane.

    profile='1/r' is the exact TEM radial dependence; 'flat' is a control to
    probe interpolation sensitivity. Returns (u, v, e_u, e_v, h_u, h_v)."""
    n = int(round(2 * half_span / dx)) + 1
    u = (np.arange(n) - (n - 1) / 2.0) * dx
    v = u.copy()
    U, V = np.meshgrid(u, v, indexing="ij")
    R = np.sqrt(U**2 + V**2)
    Rsafe = np.where(R > 0, R, 1.0)
    # Forward radial-E amplitude: A/r (A=1). H_phi+ = E_r+/eta.
    if profile == "1/r":
        Er_fwd = 1.0 / Rsafe
    else:
        Er_fwd = np.ones_like(Rsafe)
    Hphi_fwd = Er_fwd / eta
    # superposition scalars
    eF = (1.0 + Gamma)
    hF = (1.0 - Gamma)
    # radial unit r_hat=(u/r, v/r); azimuthal phi_hat=(-v/r, u/r)
    ur, vr = U / Rsafe, V / Rsafe
    e_u = eF * Er_fwd * ur
    e_v = eF * Er_fwd * vr
    h_u = hF * Hphi_fwd * (-vr)
    h_v = hF * Hphi_fwd * (ur)
    # mask the conductors (r<a inside pin, r>b outside shell) to mimic a real
    # field plane; extractor only samples within [a,b] so this is cosmetic.
    return u, v, e_u, e_v, h_u, h_v


def run_case(Gamma, dx, profile="1/r"):
    u, v, e_u, e_v, h_u, h_v = synth_plane(Gamma, dx=dx, half_span=3.0e-3, profile=profile)
    res = coaxial_tem_reference_plane_vi_from_cartesian_plane(
        u, v, e_u, e_v, h_u, h_v,
        center_u_m=0.0, center_v_m=0.0,
        inner_radius=a, outer_radius=b, eps_r=epsr,
    )
    V = complex(res.vi.voltage); I = complex(res.vi.current)
    s11 = complex(coaxial_tem_reference_plane_s11(V, I, Z_TEM))
    zin = V / I
    return V, I, zin, s11


print("\n=== Layer 0a: forward-only Gamma=0 (expect V/I=+Z_TEM real, S11=0) ===")
for dx in (0.05e-3, 0.10e-3, 0.25e-3, 0.5e-3, 0.75e-3):
    V, I, zin, s11 = run_case(0.0 + 0j, dx)
    print(f" dx={dx*1e3:5.3f}mm  V/I={zin.real:8.3f}{zin.imag:+8.3f}j  "
          f"|V/I|={abs(zin):7.3f} (Z_TEM={Z_TEM:.2f})  |S11|={abs(s11):.4f}  ang(V/I)={np.degrees(np.angle(zin)):7.2f}")

print("\n=== Layer 0b: lossless-short family |Gamma|=1, sweep phase (expect |S11|=1, ang(V/I)=+-90) ===")
print(" Gamma_ang  ->  |S11|    S11_ang   ang(V/I)   |V/I|/Z_TEM   (dx=0.10mm)")
for gdeg in (179.0, 150.0, 120.0, 90.0, 60.0, 30.0, 1.0):
    G = np.exp(1j * np.radians(gdeg))
    V, I, zin, s11 = run_case(G, 0.10e-3)
    print(f"  {gdeg:6.1f}     {abs(s11):7.4f}   {np.degrees(np.angle(s11)):7.2f}   "
          f"{np.degrees(np.angle(zin)):7.2f}    {abs(zin)/Z_TEM:8.4f}")

print("\n=== Layer 0c: coarse-grid (dx=0.75mm ~ the FDTD cell) S11 vs exact Gamma ===")
print(" expecting S11==Gamma even on coarse grid IF extractor math is exact")
for gdeg in (179.0, 120.0, 90.0):
    G = np.exp(1j * np.radians(gdeg))
    for dx in (0.10e-3, 0.75e-3):
        V, I, zin, s11 = run_case(G, dx)
        err = abs(s11 - G)
        print(f"  Gamma_ang={gdeg:6.1f} dx={dx*1e3:4.2f}mm  S11={s11.real:+.4f}{s11.imag:+.4f}j "
              f"|S11|={abs(s11):.4f}  |S11-Gamma|={err:.4f}")
