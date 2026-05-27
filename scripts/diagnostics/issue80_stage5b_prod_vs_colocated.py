# ruff: noqa: E741, E702
"""Issue #80 Stage-5b: R5 witness — is the stage-5 co-located loop a real fix?

Stage-5's hand-rolled co-located loop gave a deceptive 1/60 neg-Re(Zin) bins at
dx=80µm vs the established production 38/60. R5 says VERIFY, do not trust. This
script runs the PRODUCTION compute_msl_s_matrix on the identical stub and
reconstructs the co-located b/a from its OWN raw dump:

  (P) production compute_msl_s_matrix  -> S11_prod (b/a at probe0, loop current)
  (C) co-located b/a from raw dump: S11 = (V0 - Z0 I0)/(V0 + Z0 I0), V0=raw_v[0],
      I0=raw_i1 (the loop current, sampled at the SAME plane pxs[0]).

VERIFIED RESULT (2026-05-27): (P) == (C) to 1.3e-7 — production's S11 IS the
co-located b/a. dx=80µm: 1.388/38neg, Z0|I|/|V|=0.84. dx=50.8µm: 2.564/43neg,
Z0|I|/|V|=0.97 (DIVERGES, physical current). The stage-5 harness's 1/60 was a
STANDING-WAVE SAMPLING-POSITION artifact (its V landed a few° on the Re>0 side:
mean ∠(V/I) -79° vs production's -83°, |V| differing 4×). The established
'loop is structurally non-passive on reflectors' verdict STANDS.
"""
from __future__ import annotations

import warnings

import numpy as np

from rfx import Box, Simulation
from rfx.sources import GaussianPulse
from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff

_EPS_R, _H_SUB, _W_TRACE = 3.66, 254e-6, 600e-6
_F_MAX, _PM, _SL = 12e9, 2e-3, 6e-3
_Z0 = 50.0


def build_stub(dx):
    lx = _PM + _SL + _PM
    ly = _W_TRACE + 2 * (2 * _H_SUB + 8 * dx)
    lz = _H_SUB + 0.6e-3
    sim = Simulation(freq_max=_F_MAX, domain=(lx, ly, lz), dx=dx,
                     cpml_layers=8, boundary="cpml")
    sim.add_material("ro", eps_r=_EPS_R)
    sim.add(Box((0, 0, 0), (lx, ly, _H_SUB)), material="ro")
    y_c = ly / 2
    tl, th = y_c - _W_TRACE / 2, y_c + _W_TRACE / 2
    xe = _PM + _SL
    sim.add(Box((0, tl, _H_SUB), (xe, th, _H_SUB + dx)), material="pec")
    sim.add(Box((xe, tl, 0), (xe + dx, th, _H_SUB + dx)), material="pec")
    sim.add_msl_port(position=(_PM, y_c, 0.0), width=_W_TRACE, height=_H_SUB,
                     direction="+x", impedance=_Z0,
                     waveform=GaussianPulse(f0=6e9, bandwidth=1.6), eps_r_sub=_EPS_R)
    return sim, y_c


def main():
    z0_hj, eps_eff = hammerstad_jensen_z0_eps_eff(_W_TRACE, _H_SUB, _EPS_R)
    print("\n=== Stage-5b: production vs co-located loop S11 (same stub) ===")
    print(f"Z0_HJ={z0_hj:.1f}Ω  eps_eff={eps_eff:.3f}")
    for dx in (80e-6, 50.8e-6):
        # (P) production extractor with a raw dump
        sim, y_c = build_stub(dx)
        dump = f"/tmp/issue80_s5b_{int(dx*1e6)}.npz"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.preflight()
            resP = sim.compute_msl_s_matrix(n_freqs=81, num_periods=60.0,
                                            raw_3probe_dump_path=dump)
        f = np.asarray(resP.freqs, float)
        S11P = np.asarray(resP.S)[0, 0, :]
        ZinP = _Z0 * (1 + S11P) / (1 - S11P + 1e-30)

        # (C) co-located b/a from the SAME raw dump: V = raw_v[probe0], I = raw_i1
        d = np.load(dump, allow_pickle=True)
        V0 = np.asarray(d["raw_v"])[0, 0, 0, :]   # probe-0 voltage
        I0 = np.asarray(d["raw_i1"])[0, 0, :]     # loop current (sampled @ pxs[0])
        # b/a co-located with analytic Z0
        S11C = (V0 - z0_hj * I0) / (V0 + z0_hj * I0 + 1e-30)
        ZinC = z0_hj * (1 + S11C) / (1 - S11C + 1e-30)
        # phase diff V vs I at the shared plane (the passivity tell)
        dphi = np.degrees(np.angle(V0 / (I0 + 1e-30)))

        sl = slice(len(f)//8, -len(f)//8)
        print(f"\n--- dx={dx*1e6:.1f}µm ---")
        print(f"  (P) prod      : max|S11|={np.max(np.abs(S11P)[sl]):.3f}  "
              f"ReZin<0:{int(np.sum(ZinP.real[sl]<0))}/{len(f[sl])}")
        print(f"  (C) coloc b/a : max|S11|={np.max(np.abs(S11C)[sl]):.3f}  "
              f"ReZin<0:{int(np.sum(ZinC.real[sl]<0))}/{len(f[sl])}  "
              f"(Z0_HJ={z0_hj:.1f}Ω, V&I same plane pxs[0])")
        print("    f[GHz]  |S11|P |S11|C  ReZinP  ReZinC  ∠(V/I)°")
        for i in range(len(f)//8, len(f)-len(f)//8, max(1,(len(f)*3//4)//8)):
            print(f"    {f[i]/1e9:5.2f}  {abs(S11P[i]):.3f}  {abs(S11C[i]):.3f}  "
                  f"{ZinP[i].real:7.1f} {ZinC[i].real:7.1f}  {dphi[i]:7.1f}")

    print("\n=== READING ===")
    print("If (C) passive where (P) not: the non-passivity is in the production")
    print("S11 construction (γ/α N-probe fit feeding Z0, or V/I plane), NOT the")
    print("loop current per se. ∠(V/I) near ±90° => co-located pair is ~lossless-OK.")


if __name__ == "__main__":
    main()
