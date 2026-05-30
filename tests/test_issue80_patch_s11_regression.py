"""Issue #80 — edge-fed patch S11 regression gate (NEW-2, 2026-05-26).

The ONLY committed MSL S-matrix gate before this file was a 50 ohm THRU-LINE
passivity check (``test_msl_port_integration.py``). A thru-line cannot exhibit
the patch resonance issue #80 is about, so a regression that reintroduces
``|S11| > 1`` on a *reflecting* patch passed every committed test — the headline
"9.2 GHz fix" was assert-by-claim only (see
``docs/agent-memory/rfx-known-issues.md`` NEW-2).

This test pins the patch case as a committed gate. It encodes the *physical*
acceptance criteria (passive patch ``|S11| <= 1`` and resonance dip near the
analytic Balanis 9.21 GHz) and is marked ``xfail(strict=True)``.

HISTORY (2026-05-26): the original V·I-split extractor gave max|S11| ~ 1.44
(> 1, non-physical), then attributed to a mismeasured closed Ampere-loop current.
UPDATE (2026-05-29, PR #99): the extractor was rewritten to assemble S from the
voltage-only spatial fit (S = gamma/alpha, current-FREE), lowering the patch
|S11| to ~1.166 — better, but still > 1.05, so this stays xfail.
UPDATE (2026-05-30): the residual was then mis-attributed to source near-field
probe placement ("Fix B"). That is now FALSIFIED (see the ``xfail`` reason below
and ``docs/research_notes/20260530_issue80_nearfield_falsified.md``): the
spatial-fit extractor is robust on any clean single-mode line at any SWR, so the
patch |S11|>1 is single-mode model mismatch at the non-TEM patch-junction feed
field — confirm via a GPU raw-probe dump on the patch geometry.

When the real fix lands this test will XPASS; ``strict=True`` turns the
unexpected pass into a failure that forces removing the xfail and promoting it to
a real green gate. DO NOT loosen the assertions to force a pass (R5).

Marked ``gpu`` + ``slow``: the geometry is a full patch antenna at dx=0.197 mm
over a ~30x18x13 mm domain with num_periods=200 — GPU-scale, run by the VESSL
validation harness, excluded from the default CPU suite.
"""
from __future__ import annotations

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.sources import GaussianPulse

# --- issue #80 reproduction geometry (mirrors scripts/issue80_patch_s11_validation.py) ---
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

TARGET_GHZ = 9.21       # analytic Balanis resonance
TOL_GHZ = 0.20
PASSIVE_TOL = 1.05      # |S11| <= 1 + numerical slack


def _build_patch_sim() -> Simulation:
    sim = Simulation(
        freq_max=15e9, domain=(DOM_X, DOM_Y, DOM_Z),
        dx=DX, cpml_layers=8, boundary="cpml",
    )
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    sim.add(Box((0, 0, 4e-3), (DOM_X, DOM_Y, 4e-3 + DX)), material="pec")
    sim.add(Box((0, 0, 4e-3 + DX), (DOM_X, DOM_Y, 4e-3 + DX + H_SUB)),
            material="ro4003c")
    sim.add(Box((0, Y_C - W_MSL / 2, 4e-3 + DX + H_SUB + DX),
                (PORT_MARGIN + L_MSL, Y_C + W_MSL / 2,
                 4e-3 + DX + H_SUB + 2 * DX)),
            material="pec")
    sim.add(Box((PORT_MARGIN + L_MSL, Y_C - W / 2, 4e-3 + DX + H_SUB + DX),
                (PORT_MARGIN + L_MSL + L, Y_C + W / 2,
                 4e-3 + DX + H_SUB + 2 * DX)),
            material="pec")
    sim.add_msl_port(
        position=(PORT_MARGIN, Y_C, 4e-3 + DX),
        width=W_MSL, height=H_SUB, direction="+x", impedance=50.0,
        waveform=GaussianPulse(f0=8.5e9, bandwidth=1.6),
    )
    return sim


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "issue #80 patch S11. The extractor was fixed (PR #99: S now assembled "
        "from the voltage-only spatial fit, not the Z0-mismatch-corrupted V·I "
        "split), which lowers the patch |S11| from ~1.44 to ~1.166 (GPU run "
        "369367240214) — but it is still >1.05, so this stays xfail. "
        "NOTE (2026-05-30): an earlier version of this reason blamed the residual "
        "on source reactive near-field probe placement and proposed 'Fix B' "
        "(increase n_probe_offset). That diagnosis is FALSIFIED by three CPU "
        "experiments (docs/research_notes/20260530_issue80_nearfield_falsified.md, "
        "scripts/diagnostics/issue80_{nearfield_offset_sweep,span_conditioning,"
        "highswr_span}.py): on a matched line |S11| is FLAT vs offset (past "
        "lambda/2pi), and on a CLEAN high-SWR open stub (|Gamma|~1) the extractor "
        "returns |S11|=1.000-1.002 at every span and offset — even at the worst "
        "3-probe conditioning (cond~32). So the residual is NEITHER near-field NOR "
        "generic fit conditioning; the spatial-fit extractor is robust on any "
        "clean single-mode line at any SWR. The patch's |S11|>1 is best explained "
        "by SINGLE-MODE MODEL MISMATCH at the short patch feed: the reference "
        "plane sits in the patch-junction's non-TEM higher-order/fringing field, "
        "which V = alpha*e^{-jbx} + gamma*e^{+jbx} cannot represent, so lstsq "
        "returns the non-physical |gamma|>|alpha|. (num_periods=200 settling of "
        "the high-Q resonance is a secondary, not-yet-excluded alternative.) "
        "Confirming this needs a GPU raw-probe dump on the actual patch; CPU "
        "proxies are clean single-mode lines and cannot reproduce it. The honest "
        "resolution is to FLAG (this strict xfail + the runtime honesty guard), "
        "not to manufacture a <=1 number. XPASS => the real fix landed; remove "
        "this xfail and promote to a green gate. Do NOT loosen the assertions."
    ),
)
def test_issue80_patch_s11_passive_and_resonant():
    """Patch |S11| must be passive AND resonate near 9.21 GHz. Currently fails."""
    sim = _build_patch_sim()

    # R: never ignore preflight — surface any warning before trusting |S| numbers.
    sim.preflight()

    res = sim.compute_msl_s_matrix(n_freqs=81, num_periods=200.0)

    freqs = np.asarray(res.freqs, dtype=float)
    s11 = np.abs(np.asarray(res.S)[0, 0, :])
    z0 = np.asarray(res.Z0)[0, :]

    i_dip = int(np.argmin(s11))
    f_dip_ghz = freqs[i_dip] / 1e9
    s11_max = float(np.max(s11))

    # --- R5 witnesses: full trace + impedance, never a bare headline ---
    print(f"\n[ISSUE80-REG] max|S11| = {s11_max:.4f}  dip @ {f_dip_ghz:.3f} GHz")
    print(f"[ISSUE80-REG] Z0[0] median Re = {np.median(z0.real):.2f} ohm "
          f"(analytic Hammerstad-Jensen ~50.6 ohm)")
    for f, a in zip(freqs / 1e9, s11):
        print(f"[ISSUE80-REG-TRACE] {f:7.3f} GHz  |S11|={a:.5f}")

    # --- Physical acceptance (the eventual-green criteria) ---
    assert s11_max <= PASSIVE_TOL, (
        f"non-passive patch: max|S11| = {s11_max:.4f} > {PASSIVE_TOL} "
        "(closed-loop current mismeasured). DO NOT loosen — fix the extractor."
    )
    assert (TARGET_GHZ - TOL_GHZ) <= f_dip_ghz <= (TARGET_GHZ + TOL_GHZ), (
        f"resonance dip at {f_dip_ghz:.3f} GHz, expected "
        f"{TARGET_GHZ} +/- {TOL_GHZ} GHz (analytic Balanis)."
    )
