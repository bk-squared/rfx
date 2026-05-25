"""Issue #80 acceptance check — edge-fed patch S11 resonance frequency.

Runs the GitHub issue #80 reproduction (edge-fed Hammerstad patch on
RO4003C, 50 ohm microstrip feed) through ``compute_msl_s_matrix`` on the
Fix-A/B/C branch and checks whether the |S11| minimum now lands at the
analytic Balanis resonance 9.21 +/- 0.20 GHz. Pre-fix it landed at
10.11 GHz. This is acceptance criterion 1 of issue #80.

S11 = gamma/alpha is a pure voltage-wave amplitude ratio (it does NOT
use Z0), so the Fix-C N-probe voltage decomposition is what this tests.
The separate Z0-extraction error (contaminated I1, ~74 vs ~54 ohm) does
not enter S11 and is tracked as a distinct follow-up.

Exit 0 = PASS (dip in [9.01, 9.41] GHz), exit 1 = FAIL.
"""
from __future__ import annotations

import sys

import numpy as np

from rfx import Box, Simulation
from rfx.sources import GaussianPulse

EPS_R = 3.38
H_SUB = 0.787e-3
W = 10.129e-3
L = 8.595e-3
W_MSL = 1.8e-3
L_MSL = 8.0e-3
PORT_MARGIN = 5.0e-3
DX = 0.197e-3
DOM_X = 29.747e-3
DOM_Y = 18.130e-3
DOM_Z = 12.787e-3
Y_C = DOM_Y / 2.0

TARGET_GHZ = 9.21
TOL_GHZ = 0.20


def main() -> int:
    sim = Simulation(
        freq_max=15e9, domain=(DOM_X, DOM_Y, DOM_Z),
        dx=DX, cpml_layers=8, boundary="cpml",
    )
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    # PEC ground plane.
    sim.add(Box((0, 0, 4e-3), (DOM_X, DOM_Y, 4e-3 + DX)), material="pec")
    # RO4003C substrate.
    sim.add(Box((0, 0, 4e-3 + DX), (DOM_X, DOM_Y, 4e-3 + DX + H_SUB)),
            material="ro4003c")
    # 50 ohm microstrip feed trace.
    sim.add(Box((0, Y_C - W_MSL / 2, 4e-3 + DX + H_SUB + DX),
                (PORT_MARGIN + L_MSL, Y_C + W_MSL / 2,
                 4e-3 + DX + H_SUB + 2 * DX)),
            material="pec")
    # Edge-fed patch.
    sim.add(Box((PORT_MARGIN + L_MSL, Y_C - W / 2, 4e-3 + DX + H_SUB + DX),
                (PORT_MARGIN + L_MSL + L, Y_C + W / 2,
                 4e-3 + DX + H_SUB + 2 * DX)),
            material="pec")
    # Wider, higher-centre source than the default
    # GaussianPulse(f0=freq_max/2=7.5GHz, bw=0.8) — that default rolls off
    # ~exp(-6.25) ≈ 0.002 at 15 GHz, starving the upper part of the
    # frequency sweep of signal. The previous long-window run
    # (369367239037) had max|S11|=1.527 at 11.96 GHz — exactly the
    # low-SNR tail. f0=8.5 GHz, bw=1.6 puts the spectral peak near
    # ~10 GHz and gives ~14 GHz 1/e width, covering the full 1.5-15 GHz
    # sweep with usable SNR (~77% of peak amplitude at 15 GHz vs 0.2%).
    sim.add_msl_port(
        position=(PORT_MARGIN, Y_C, 4e-3 + DX),
        width=W_MSL, height=H_SUB, direction="+x", impedance=50.0,
        waveform=GaussianPulse(f0=8.5e9, bandwidth=1.6),
    )

    # Preflight (user directive 2026-05-20: never ignore preflight). The
    # patch geometry currently emits 0 warnings on this mesh; surface it
    # anyway so any future regression is visible in the run log.
    print("=== sim.preflight() ===", flush=True)
    sim.preflight()

    # num_periods 200: long-window diagnostic for the truncation
    # hypothesis (issue #80 stage S1 post-mortem). At the patch's
    # Q~30–50 around 9 GHz, 25 periods (~3.3 ns) leaves significant
    # ring-down energy in the DFT window — V (Ez) and I (Hy/Hz) leak
    # differently and corrupt the V·I-split denominator a=(V+Z0·I)/2.
    # 200 periods (~27 ns) is comfortably >60 dB down. If |S11| becomes
    # bounded and smooth with dip near 9.21 GHz, truncation was the
    # upstream cause; if not, keep diagnosing.
    res = sim.compute_msl_s_matrix(n_freqs=81, num_periods=200.0)

    freqs = np.asarray(res.freqs, dtype=float)
    s11 = np.abs(np.asarray(res.S)[0, 0, :])
    z0 = np.asarray(res.Z0)[0, :]

    i_dip = int(np.argmin(s11))
    f_dip = freqs[i_dip] / 1e9
    s11_dip_db = 20.0 * np.log10(max(float(s11[i_dip]), 1e-12))
    s11_max = float(np.max(s11))

    print("=== issue #80 acceptance — patch S11 (stage S1: V·I split) ===")
    print(f"ISSUE80: S11 minimum = {s11_dip_db:.1f} dB at {f_dip:.3f} GHz")
    print(f"ISSUE80: target = {TARGET_GHZ} +/- {TOL_GHZ} GHz (analytic Balanis)")
    print(f"ISSUE80: pre-fix reference = 10.11 GHz (wrong)")
    print(f"ISSUE80: max|S11| = {s11_max:.3f} (headline — must be <= 1 for "
          f"a passive patch; pre-S1 Fix-C blew up to ~8.6)")
    print(f"ISSUE80: Z0[0] median Re = {np.median(z0.real):.2f} ohm")
    # full |S11|(f) trace for the log
    for f, a in zip(freqs / 1e9, s11):
        print(f"ISSUE80-TRACE: {f:7.3f} GHz  |S11|={a:.5f}")

    ok_dip = (TARGET_GHZ - TOL_GHZ) <= f_dip <= (TARGET_GHZ + TOL_GHZ)
    ok_passive = s11_max <= 1.0 + 0.05
    ok = ok_dip and ok_passive
    print(f"ISSUE80: ACCEPTANCE-1 (resonance) {'PASS' if ok_dip else 'FAIL'} "
          f"(dip at {f_dip:.3f} GHz)")
    print(f"ISSUE80: ACCEPTANCE-headline (|S11|<=1) "
          f"{'PASS' if ok_passive else 'FAIL'} (max|S11| = {s11_max:.3f})")
    print(f"ISSUE80: OVERALL {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
