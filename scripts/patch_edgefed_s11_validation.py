"""Issue #80 acceptance check — edge-fed patch |S11| passivity (historical harness).

Runs the GitHub issue #80 reproduction (edge-fed Hammerstad patch on
RO4003C, 50 ohm microstrip feed) through ``compute_msl_s_matrix`` and dumps
the full |S11|(f) trace. The exit code gates PASSIVITY only (max|S11| <=
1.05 — the issue #80 defect was |S11| > 1).

HISTORY. The original acceptance criterion 1 ("|S11| dip at the analytic
Balanis 9.21 +/- 0.20 GHz"; pre-fix the dip sat at 10.11 GHz) is retired
twice over and is no longer gated here:

  * the |S11| dip of a directly edge-fed patch is the OFF-RESONANCE match
    point, not the resonance — reading the dip as the resonance is a
    category error (issue #118);
  * 9.21 GHz is Balanis on the DESIGN dimensions, realized on no mesh, and
    the 9.32 GHz fed resonance it seemed to confirm was two errors
    cancelling — the #702 sheet-node material fix moved the fed TM010 to
    8.16 GHz on the harminv-gate board (issue #782).

The committed gates are tests/locks/test_patch_edgefed_s11_passivity.py
(passivity + edge-fed signature) and
tests/locks/test_patch_edgefed_resonance_harminv.py (signed resonance
envelopes). This script stays as a runnable trace dump + passivity check
on the same geometry.

S11 = gamma/alpha is a pure voltage-wave amplitude ratio (it does NOT
use Z0), so the Fix-C N-probe voltage decomposition is what this tests.
The separate Z0-extraction error (contaminated I1, ~74 vs ~54 ohm) does
not enter S11 and is tracked as a distinct follow-up.

Exit 0 = PASS (max|S11| <= 1.05), exit 1 = FAIL.
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

    # Preflight (user directive 2026-05-20: never ignore preflight). This
    # fixture emits several advisories on this mesh (off-lattice design edges,
    # sheet-cavity electrical thickness, the +25% substrate column under the
    # port) — they are part of any number quoted from this run.
    print("=== sim.preflight() ===", flush=True)
    sim.preflight()

    # num_periods 200: long-window diagnostic for the truncation
    # hypothesis (issue #80 stage S1 post-mortem). At the patch's
    # Q~30–50 around 9 GHz, 25 periods (~3.3 ns) leaves significant
    # ring-down energy in the DFT window — V (Ez) and I (Hy/Hz) leak
    # differently and corrupt the V·I-split denominator a=(V+Z0·I)/2.
    # 200 periods (~27 ns) is comfortably >60 dB down. If |S11| becomes
    # bounded and smooth, truncation was the upstream cause; if not, keep
    # diagnosing. (The 2026-05 note here expected the dip near 9.21 GHz —
    # retired: the dip is the match point (#118), and 9.21 GHz predates the
    # #702 sheet-node material fix (#782).)
    res = sim.compute_msl_s_matrix(n_freqs=81, num_periods=200.0)

    freqs = np.asarray(res.freqs, dtype=float)
    s11 = np.abs(np.asarray(res.S)[0, 0, :])
    z0 = np.asarray(res.Z0)[0, :]

    i_dip = int(np.argmin(s11))
    f_dip = freqs[i_dip] / 1e9
    s11_dip_db = 20.0 * np.log10(max(float(s11[i_dip]), 1e-12))
    s11_max = float(np.max(s11))

    print("=== issue #80 acceptance — patch S11 (stage S1: V·I split) ===")
    print(f"PATCH-EDGEFED: S11 minimum = {s11_dip_db:.1f} dB at {f_dip:.3f} GHz")
    print("PATCH-EDGEFED: dip is reported, NOT gated — it is the off-resonance "
          "match point (issue #118); the retired 9.21 GHz Balanis target "
          "predates #702 (issue #782)")
    print(f"PATCH-EDGEFED: max|S11| = {s11_max:.3f} (headline — must be <= 1 for "
          f"a passive patch; pre-S1 Fix-C blew up to ~8.6)")
    print(f"PATCH-EDGEFED: Z0[0] median Re = {np.median(z0.real):.2f} ohm")
    # full |S11|(f) trace for the log
    for f, a in zip(freqs / 1e9, s11):
        print(f"PATCH-EDGEFED-TRACE: {f:7.3f} GHz  |S11|={a:.5f}")

    ok_passive = s11_max <= 1.0 + 0.05
    print(f"PATCH-EDGEFED: dip at {f_dip:.3f} GHz (reported, not gated — "
          "off-resonance match point, issue #118)")
    print(f"PATCH-EDGEFED: ACCEPTANCE (|S11| <= 1.05) "
          f"{'PASS' if ok_passive else 'FAIL'} (max|S11| = {s11_max:.3f})")
    return 0 if ok_passive else 1


if __name__ == "__main__":
    sys.exit(main())
