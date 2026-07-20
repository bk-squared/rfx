"""Issue #80 / #118 — edge-fed patch S11: passivity gate + edge-fed match-point physics.

Issue #80 (non-physical ``|S11| > 1``) is FIXED (#116, ``n_probe_offset`` floor clears the
source fringing transient); this gate pins that fix GREEN. Issue #118 then asked the |S11|
*dip* to converge to the analytic Balanis 9.21 GHz resonance under mesh refinement. That is
the WRONG PHYSICS for a *directly edge-fed* patch and is NOT achievable by refining the mesh:

  * The patch radiating-edge input resistance at the TM010 resonance is high (~hundreds of
    ohms; analytic ~230-470, witness Re(Zin) peaks > 1.5 kohm at coarse mesh), so a 50-ohm
    line is BADLY MATCHED *at* resonance — |S11| is HIGH (~0.9) there, NOT a dip. The |S11|
    minimum is the OFF-RESONANCE MATCH point where Re(Zin) sweeps down through ~50 ohm, which
    sits ABOVE the resonance (witness: dip ~11 GHz vs resonance Im(Zin)=0 ~9.5-9.6 GHz and
    Harminv 9.32 GHz — ~1.6 GHz apart). Reading the |S11| dip as the resonance is a category
    error (the R5 surface-metric trap).
  * The dip frequency is BOTH mesh-limited AND an unstable argmin over a shallow double-minimum
    |S11| curve, so it must not be a gate. Convergence analysis (3 real mesh points 197/98.4/
    78.7 um -> 10.50/9.80/9.70 GHz, decelerating): the dip asymptotes at ~9.48 GHz — at/above
    the old [9.01,9.41] band edge — and even the most optimistic linear fit needs n_sub~37
    (dx~21 um, ~7e8 cells, >300 GPU-h) to enter the band. Infeasible; mesh refinement is a
    diminishing-returns R2 loop here, not a fix.

The RESONANCE itself is correct and is validated by the port-INDEPENDENT companion gate
``test_issue80_patch_resonance_harminv.py`` (Harminv ring-down 9.32 GHz == OpenEMS S11 9.20
== analytic Balanis 9.21). So this gate asserts the two things the |S11| curve *can* robustly
witness, plus a soft bound:

  (1) PASSIVITY:        max|S11| <= 1.05            (the #80 fix; 1.008 at this mesh, 0.985 finer)
  (2) EDGE-FED SIGNATURE: |S11| > 0.70 across the analytic resonance band [9.0, 9.42] GHz
      => the patch is poorly matched at its resonance => the |S11| dip is NOT the resonance.
  (3) (soft) the global |S11| minimum lies ABOVE the resonance band (it is the match point).

Evidence: ``scripts/diagnostics/issue118_match_vs_resonance_witness.py`` derives Zin(f) =
Z0(1+S11)/(1-S11) on this exact geometry and shows Im(Zin)=0 (resonance) well below the dip.

RING-DOWN SETTLING WITNESS (repo mandatory rule; issue #402)
------------------------------------------------------------
The DFT-extracted |S11| is only trustworthy from a drained record. A cavity Ez probe is now
added so the internal ``compute_msl_s_matrix`` run records a point-probe time series and the
framework's #332 energy witness computes; the test captures that advisory and asserts it does
NOT fire (domain drained below -40 dB of peak). The witness fires on the SLOWEST-draining
monitored field — here the feed-line standing wave, which is exactly the settling the S11
extraction depends on. FINDING (issue #402): the previous ``num_periods=200`` left that
standing wave at -36.2 dB of peak (N_SUB=2 witness) — ABOVE the -40 dB bar — because the
badly-matched patch reflects strongly and the feed drains slowly. Under-settling also inflated
|S11| slightly (max|S11| = 1.062 at np=120 -> 1.001 at np=200 -> 0.989 at np=280, N_SUB=2), so
the OLD 1.008 reading was a truncation-biased over-estimate, still passive but not settled. The
gated ``num_periods`` is raised to 280 (N_SUB=2 ladder: 120->-23.8, 200->-36.2, 280->settled;
~0.16 dB/period); the physics VERDICTS (passivity <= 1.05, edge-fed signature) are unchanged —
the settled max|S11| = 0.989 sits further inside the passive envelope, not outside it. The
GPU harness confirms the witness at the committed N_SUB=4 mesh.

Marked ``gpu`` + ``slow``: dx=0.197 mm over a ~30x18x13 mm domain with num_periods=280 —
GPU-scale, run by the VESSL validation harness, excluded from the default CPU suite. The
resonance physics has CPU coverage via the Harminv companion gate (``slow``).
"""
from __future__ import annotations

import warnings

import jax.numpy as jnp
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

PASSIVE_TOL = 1.05          # |S11| <= 1 + numerical slack (the #80 passivity fix)
RES_BAND_GHZ = (9.0, 9.42)  # analytic Balanis resonance neighbourhood
RES_BAND_S11_MIN = 0.70     # patch must be POORLY matched here (edge-fed signature)


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
    # #402 settling witness: the framework's #332 ring-down witness evaluates the
    # tail-vs-peak envelope of a POINT-PROBE time series, so the internal
    # compute_msl_s_matrix run needs at least one probe or the witness has no
    # data and can never fire. This Ez probe under the patch (0.7·L along it)
    # supplies that series; its slow ring-down tail is the settling the DFT-based
    # S11 extraction depends on. (The test below guards that this probe stays
    # present, so removing it fails loudly rather than silently disarming #332.)
    x_patch0 = PORT_MARGIN + L_MSL
    sim.add_probe(
        position=(x_patch0 + 0.7 * L, Y_C - 0.2 * W, 4e-3 + DX + H_SUB * 0.5),
        component="ez",
    )
    return sim


@pytest.mark.gpu
@pytest.mark.slow
def test_issue80_patch_s11_passive_and_edge_fed_match():
    """Patch |S11| is passive AND shows the edge-fed signature (poorly matched at the
    resonance; dip is the off-resonance match point above it). Resonance frequency itself
    is validated by the Harminv companion gate."""
    sim = _build_patch_sim()

    # #402 guard: the settling assertion below is only meaningful if a point probe
    # exists to feed the #332 witness. Without it, #332 has no series, never fires,
    # and `assert not _trunc` becomes silently always-green. Fail loudly instead.
    assert getattr(sim, "_probes", None), (
        "settling-witness probe missing — #332 ring-down witness cannot evaluate; "
        "the settling assertion would be vacuous (issue #402)"
    )

    # R: never ignore preflight — surface any warning before trusting |S| numbers.
    advisories = [str(a) for a in sim.preflight()]
    print(f"\n[ISSUE80/118-REG] preflight advisories ({len(advisories)}) — quoted verbatim:")
    for a in advisories:
        print(f"  ! {a}")

    freqs = np.linspace(6e9, 14e9, 81)
    # #402: num_periods=200 left the ring-down at -36.2 dB of peak (N_SUB=2
    # witness) — above the -40 dB bar; the badly-matched patch reflects strongly
    # so it drains slowly. 280 clears the bar on the N_SUB=2 ladder. CAVEAT: 280
    # is validated only at N_SUB=2 (CPU); the committed gate runs N_SUB=4 (gpu),
    # never exercised on a CPU/committed path. This is fail-safe by design — if
    # N_SUB=4 drains slower and 280 does NOT settle, #332 fires and this test goes
    # RED (surfacing the truncation), it does not pass silently. Re-pin from the
    # first N_SUB=4 GPU run if it reds.
    with warnings.catch_warnings(record=True) as _settling:
        warnings.simplefilter("always")
        res = sim.compute_msl_s_matrix(freqs=jnp.asarray(freqs), num_periods=280.0)
    _trunc = [
        str(w.message) for w in _settling
        if "#332" in str(w.message) or "ring-down truncated" in str(w.message)
    ]
    print("[ISSUE80/118-SETTLING] framework #332 ring-down energy witness: "
          f"{_trunc if _trunc else 'no truncation advisory — domain drained below -40 dB of peak'}")
    assert not _trunc, (
        "ring-down NOT settled at the gated num_periods — the DFT-extracted |S11| may carry "
        f"truncation error (issue #402; framework #332 witness fired): {_trunc}. Raise "
        "num_periods; do NOT trust the |S11| envelope from a truncated record."
    )

    fr = np.asarray(res.freqs, dtype=float) / 1e9
    s = np.asarray(res.S)[0, 0, :]
    z0 = np.asarray(res.Z0)[0, :]
    zin = z0 * (1.0 + s) / (1.0 - s)
    s11 = np.abs(s)

    i_dip = int(np.argmin(s11))
    f_dip = fr[i_dip]
    s11_max = float(np.max(s11))
    band = (fr >= RES_BAND_GHZ[0]) & (fr <= RES_BAND_GHZ[1])
    s11_res_band_min = float(np.min(s11[band]))

    # --- R5 witnesses: full trace (|S11|, Re/Im Zin), never a bare headline ---
    print(f"\n[ISSUE80/118-REG] max|S11| = {s11_max:.4f}  dip @ {f_dip:.3f} GHz "
          f"(|S11|={s11[i_dip]:.4f}, the off-resonance MATCH point)")
    print(f"[ISSUE80/118-REG] min|S11| over resonance band {RES_BAND_GHZ} GHz = "
          f"{s11_res_band_min:.4f} (HIGH => poorly matched at resonance, edge-fed signature)")
    print(f"[ISSUE80/118-REG] Z0[0] median Re = {np.median(z0.real):.2f} ohm "
          f"(analytic Hammerstad-Jensen ~50.6 ohm)")
    for f, a, zr, zi in zip(fr, s11, zin.real, zin.imag):
        print(f"[ISSUE80/118-TRACE] {f:7.3f} GHz  |S11|={a:.5f}  "
              f"Re(Zin)={zr:9.2f}  Im(Zin)={zi:9.2f}")

    # --- (1) passivity: the issue #80 fix (was |S11|=1.44/8.94, now <= 1.05) ---
    assert s11_max <= PASSIVE_TOL, (
        f"non-passive patch: max|S11| = {s11_max:.4f} > {PASSIVE_TOL}. "
        "DO NOT loosen — this guards the #116 n_probe_offset passivity fix."
    )

    # --- (2) edge-fed signature: the patch is POORLY matched at its resonance, so the
    #         |S11| dip cannot be (and is not) the resonance. This is the physically robust
    #         negation of the old mis-specified "dip near 9.21" assertion (issue #118). ---
    assert s11_res_band_min > RES_BAND_S11_MIN, (
        f"min|S11| = {s11_res_band_min:.4f} over the resonance band {RES_BAND_GHZ} GHz is "
        f"unexpectedly LOW (<= {RES_BAND_S11_MIN}). A directly edge-fed patch must be poorly "
        "matched at its TM010 resonance (high edge resistance); a deep dip there would mean "
        "the geometry/feed changed. Resonance frequency is checked by the Harminv companion."
    )

    # --- (3) soft: the |S11| minimum (match point) lies ABOVE the resonance band ---
    assert f_dip > RES_BAND_GHZ[1], (
        f"|S11| dip at {f_dip:.3f} GHz is inside/below the resonance band — expected the "
        "off-resonance match point ABOVE it. The exact dip frequency is mesh-limited and an "
        "unstable argmin (see #118); only the lower bound is asserted."
    )
