"""Issue #80 / #118 / #782 — edge-fed patch S11: passivity gate + edge-fed match-point physics.

Issue #80 (non-physical ``|S11| > 1``) is FIXED (#116, ``n_probe_offset`` floor clears the
source fringing transient); this gate pins that fix GREEN. Issue #118 then asked the |S11|
*dip* to converge to the analytic Balanis resonance under mesh refinement. That is the
WRONG PHYSICS for a *directly edge-fed* patch and is NOT achievable by refining the mesh:

  * The patch radiating-edge input resistance at the TM010 resonance is high (analytic
    ~hundreds of ohms; this gate's own provenance run measures the port-plane Re(Zin)
    peak at 4.3 kohm), so a 50-ohm line is BADLY MATCHED *at* resonance — |S11| is HIGH
    there, NOT a dip. The |S11| minimum is the OFF-RESONANCE MATCH point, which sits
    ABOVE the resonance (measured on this board post-#702: dip 10.100 GHz vs
    antiresonance crossing 8.819 GHz). Reading the |S11| dip as the resonance is a
    category error (the R5 surface-metric trap).
  * The dip frequency is BOTH mesh-limited AND an unstable argmin over a shallow
    |S11| curve, so it must not be a gate. The pre-#702 convergence analysis (3 real
    mesh points 197/98.4/78.7 um -> dips 10.50/9.80/9.70 GHz, decelerating; measured
    on the pre-#702 tree, so the absolute numbers are retired with it) showed even the
    most optimistic fit needs ~7e8 cells to reach the then-target band. Mesh
    refinement is a diminishing-returns R2 loop here, not a fix.

THE BOARD THIS GATE MEASURES ("Board S", issue #782 one-mesh anchor rule)
-------------------------------------------------------------------------
``DX = 0.197 mm`` exactly — NOT ``H_SUB/4 = 196.75 um`` — so this fixture rasterizes to
a 44 x 51-cell patch = 8.668 x 10.047 mm on a 4-cell (788 um) substrate. That is a
DIFFERENT realized board from the Harminv companion gate's 43 x 51 at h/4
(``tests/locks/test_patch_edgefed_resonance_harminv.py``), and every constant below was
measured on THIS board; no companion number (e.g. its fed TM010 8.16131 GHz at N=260)
is reused here. Mixing dimensions or anchors across the two boards describes a board
that exists on no mesh — the ~2-point error class #782 documents. The raster is locked
by the fast companion test in this file.

The RESONANCE FREQUENCY itself is not gated here; the companion gate pins it as SIGNED
regression envelopes on its own realized raster (Leg A isolated-patch offset
-6.17 +/- 1.125 %, Leg B feed pull -6.109 +/- 0.986 %). The historical chain this file
used to cite as validation — "Harminv 9.32 == OpenEMS 9.20 == Balanis 9.21" — was two
errors of opposite sign cancelling (the pre-#702 sheet-node vacuum vs the feed pull) and
is retired (#702, #782).

WHAT IS GATED, AND THE BAND'S PROVENANCE (measured 2026-09-01, post-#702 tree)
------------------------------------------------------------------------------
Evidence run: ``scripts/diagnostics/patch_edgefed_s11_band_repin.py`` — this exact
config (same builder import, freqs 6-14 GHz x 81, num_periods = 280), CPU
(``JAX_PLATFORMS=cpu``), both arms settled (#332 witness silent), every preflight
advisory captured (7 in the CPU provenance run — three are environment x64 notes — including the +25.2 % substrate column under the
MSL port — the half-open rasterizer rounds the off-node substrate face UP under the
port; part of this board's identity, quoted in the run log). Per-bin traces:
``docs/design_notes/patch_edgefed_s11_band_repin_results.json``; predeclaration and
scoring: ``docs/design_notes/issue782_retired_resonance_predeclaration.md``. Measured:

    max|S11| = 0.9921
    port-plane antiresonance: Im(Zin) zero-crossing at 8.8189 GHz, Re(Zin) peak
        4326 ohm at the 8.8 GHz bin (2067 ohm at 8.9)
    min|S11| over (8.4, 9.2) GHz = 0.8794
    global dip 10.100 GHz, |S11| = 0.4426 (the off-resonance match point)

  (1) PASSIVITY:          max|S11| <= 1.05      (the #80 fix; 0.9921 measured settled)
  (2) EDGE-FED SIGNATURE: |S11| > 0.70 across RES_BAND_GHZ = (8.4, 9.2)
      => the patch is poorly matched at its resonance => the dip is NOT the resonance.
  (2b) IN-BAND RESONANCE WITNESS: an Im(Zin) = 0 crossing exists inside the band —
      the resonance the band names is actually there. This is what makes (2)
      falsifiable: without it the band gate would pass over any dead spectral region
      (that is exactly how the retired (9.0, 9.42) band failed, #782).
  (2c) the in-band max Re(Zin) exceeds 500 ohm — the high-impedance antiresonance
      that MAKES "poorly matched at resonance" the right physics (measured 4326).
  (3) (soft) the global |S11| minimum lies ABOVE the band (measured 10.100 > 9.2).

The crossing is a PORT-PLANE observable — the antiresonance seen through the feed
line, reference-plane dependent — so it is an existence witness inside a band, never a
frequency gate, and it must not be quoted as "the TM010 modal frequency" (the
companion's modal numbers live on Board H). Identity of the in-band feature — never
amplitude rank: the y-centred feed parity-suppresses TM001 (Board-S realized-raster
Balanis 8.0016 GHz shows only as a non-crossing Re(Zin) ~ 30 ohm wiggle at
7.9-8.1 GHz, below the band), and the 4.3-kohm Re peak is the edge-fed patch
antiresonance class the pre-#702 witness also saw (Re peaks > 1.5 kohm); the modal
labelling chain (spatial parity across a probe cross, windowed-DFT nodal check,
single-dimension perturbation, Balanis both modes per realized raster) lives in the
companion gate and the #782 ledger record.

DISCRIMINATION (#702 falsifier, scored in the predeclaration note): with
``resample_sheet_node_materials`` disabled — the bit-exact pre-#702 physics — the same
config puts the antiresonance at 9.5-9.6 GHz (Re(Zin) peak 5255 ohm at 9.6, OUT of
band; crossings 9.108 / 9.325 / 9.629 / 11.366, dip 11.400). (2c) FAILS on that
physics: in-band max Re(Zin) reads 9.2 ohm vs the 500-ohm floor (470x separation from
the main arm's 4326). (2b) ALONE WOULD NOT discriminate — the retired physics has a
low-impedance crossing at 9.108 GHz inside the band, on the negative-Re shoulder below
its antiresonance — which is exactly why (2c) exists. The pinned band is stub-locked: the fed line is an open
stub, ~ +0.85 pp per node of stub length (~0.07 GHz) — a FEED_LEN / PORT_MARGIN /
DOM_X / raster / DX change invalidates this band; re-pin from a fresh evidence run.

RING-DOWN SETTLING WITNESS (repo mandatory rule; issue #402)
------------------------------------------------------------
The DFT-extracted |S11| is only trustworthy from a drained record. A cavity Ez probe is
added so the internal ``compute_msl_s_matrix`` run records a point-probe time series and
the framework's #332 energy witness computes; the test captures that advisory and asserts
it does NOT fire (domain drained below -40 dB of peak). The witness fires on the
SLOWEST-draining monitored field — here the feed-line standing wave, which is exactly the
settling the S11 extraction depends on. FINDING (issue #402): ``num_periods=200`` left
that standing wave at -36.2 dB of peak (N_SUB=2 witness) — ABOVE the -40 dB bar —
because the badly-matched patch reflects strongly and the feed drains slowly.
Under-settling also inflated |S11| slightly (max|S11| = 1.062 at np=120 -> 1.001 at
np=200 -> 0.989 at np=280, N_SUB=2), so older readings were truncation-biased
over-estimates, still passive but not settled. The gated ``num_periods`` stays at 280;
the 2026-09-01 provenance run confirms the witness is silent there on this exact config.

Marked ``gpu`` + ``slow``: dx=0.197 mm over a ~30x18x13 mm domain with num_periods=280 —
GPU-scale, run by the VESSL validation harness, excluded from the default CPU suite.
(The band provenance run above is the same config on CPU; the companion Harminv history
measured CPU-vs-GPU agreement < 4e-8 relative on this fixture family.) The realized-
raster lock below is fast and runs in the default lane.
"""
from __future__ import annotations

LOCK_PROVENANCE = {
    "fixture": "docs/design_notes/patch_edgefed_s11_band_repin_results.json",
    "generator": "scripts/diagnostics/patch_edgefed_s11_band_repin.py",
    "commit": "c7527cb",
    "date": "2026-09-01",
    "run_id": "local",
    "host": "cpu, JAX_PLATFORMS=cpu (os / jax version not recorded in #840)",
    "pinned_until": "2027-02-28",
}

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.sources import GaussianPulse

# --- issue #80 reproduction geometry (mirrors scripts/patch_edgefed_s11_validation.py) ---
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

PASSIVE_TOL = 1.05           # |S11| <= 1 + numerical slack (the #80 passivity fix)
RES_BAND_GHZ = (8.4, 9.2)    # measured antiresonance neighbourhood on THIS board
#                              (crossing 8.8189 GHz, 2026-09-01 provenance run; the
#                              pre-#702 band (9.0, 9.42) is retired — issue #782)
RES_BAND_S11_MIN = 0.70      # poorly matched there (measured in-band min 0.8794)
RES_BAND_RE_ZIN_MIN_OHM = 500.0  # in-band antiresonance Re peak (measured 4326 ohm)

# Realized patch raster this band was pinned on: 44 x 51 cells = 8.668 x 10.047 mm at
# dx = 197 um ("Board S"). NOT the harminv companion's 43 x 51 at h/4 (issue #782).
RASTER_CELLS = (44, 51)


def _patch_box() -> Box:
    return Box((PORT_MARGIN + L_MSL, Y_C - W / 2, 4e-3 + DX + H_SUB + DX),
               (PORT_MARGIN + L_MSL + L, Y_C + W / 2,
                4e-3 + DX + H_SUB + 2 * DX))


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
    sim.add(_patch_box(), material="pec")
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


def _im_zin_crossings_ghz(fr_ghz, zin):
    """All Im(Zin) sign-change frequencies, linearly interpolated, in GHz."""
    y = np.asarray(zin.imag, dtype=float)
    idx = np.where(np.diff(np.sign(y)) != 0)[0]
    return [float(fr_ghz[i] - y[i] * (fr_ghz[i + 1] - fr_ghz[i]) / (y[i + 1] - y[i]))
            for i in idx]


def _gate_readings(fr_ghz, s, z0):
    """Every quantity the assertions below read, from one (freqs, S11, Z0) trace.

    Module-level on purpose: ``scripts/diagnostics/patch_edgefed_s11_band_repin.py``'s
    falsifier replay evaluates THIS function on the saved main/retired traces, so the
    committed assertions and the discrimination evidence cannot drift apart.
    """
    fr_ghz = np.asarray(fr_ghz, dtype=float)
    s = np.asarray(s)
    z0 = np.asarray(z0)
    zin = z0 * (1.0 + s) / (1.0 - s)
    s11 = np.abs(s)
    band = (fr_ghz >= RES_BAND_GHZ[0]) & (fr_ghz <= RES_BAND_GHZ[1])
    crossings = _im_zin_crossings_ghz(fr_ghz, zin)
    i_dip = int(np.argmin(s11))
    return dict(
        s11=s11, zin=zin,
        max_s11=float(np.max(s11)),
        f_dip_ghz=float(fr_ghz[i_dip]),
        s11_at_dip=float(s11[i_dip]),
        band_min_s11=float(np.min(s11[band])),
        band_max_re_zin=float(np.max(zin.real[band])),
        crossings_ghz=crossings,
        band_crossings_ghz=[c for c in crossings
                            if RES_BAND_GHZ[0] <= c <= RES_BAND_GHZ[1]],
    )


def test_realized_raster_is_the_board_this_band_was_pinned_on():
    """RES_BAND_GHZ was measured on the 44 x 51-cell realization of this fixture
    (issue #782 one-mesh anchor rule). A raster change silently moves the physics
    out from under the pinned band — fail loudly here instead. Fast, no FDTD."""
    sim = _build_patch_sim()
    grid = sim._build_grid()
    mask = np.asarray(_patch_box().mask(grid), dtype=bool)
    assert mask.any(), "patch rasterizes to ZERO cells — it is not in the solve"
    occ = np.where(mask)
    raster = tuple(int(occ[a].max()) - int(occ[a].min()) + 1 for a in range(2))
    assert raster == RASTER_CELLS, (
        f"realized patch raster {raster} cells != {RASTER_CELLS} — this is not the "
        f"board RES_BAND_GHZ = {RES_BAND_GHZ} was measured on (Board S, "
        f"8.668 x 10.047 mm at dx = {DX * 1e6:.1f} um; issue #782). Re-pin the band "
        "from a fresh evidence run (scripts/diagnostics/patch_edgefed_s11_band_repin.py) "
        "before touching these constants."
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_patch_edgefed_s11_passive_and_match():
    """Patch |S11| is passive AND shows the edge-fed signature (poorly matched at the
    resonance, whose antiresonance witness sits IN the gated band; the dip is the
    off-resonance match point above it). Modal frequencies are validated by the
    Harminv companion gate on its own board."""
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
    print(f"\n[PATCH-EDGEFED/118-REG] preflight advisories ({len(advisories)}) — quoted verbatim:")
    for a in advisories:
        print(f"  ! {a}")

    freqs = np.linspace(6e9, 14e9, 81)
    # #402: num_periods=200 left the ring-down at -36.2 dB of peak (N_SUB=2
    # witness) — above the -40 dB bar; the badly-matched patch reflects strongly
    # so it drains slowly. 280 clears the bar (N_SUB=2 ladder 120->-23.8,
    # 200->-36.2, 280->settled; and the 2026-09-01 CPU provenance run of THIS
    # config confirms the witness silent at 280). If #332 fires, this test goes
    # RED (surfacing the truncation); it does not pass silently.
    with warnings.catch_warnings(record=True) as _settling:
        warnings.simplefilter("always")
        res = sim.compute_msl_s_matrix(freqs=jnp.asarray(freqs), num_periods=280.0)
    _trunc = [
        str(w.message) for w in _settling
        if "#332" in str(w.message) or "ring-down truncated" in str(w.message)
    ]
    print("[PATCH-EDGEFED/118-SETTLING] framework #332 ring-down energy witness: "
          f"{_trunc if _trunc else 'no truncation advisory — domain drained below -40 dB of peak'}")
    assert not _trunc, (
        "ring-down NOT settled at the gated num_periods — the DFT-extracted |S11| may carry "
        f"truncation error (issue #402; framework #332 witness fired): {_trunc}. Raise "
        "num_periods; do NOT trust the |S11| envelope from a truncated record."
    )

    fr = np.asarray(res.freqs, dtype=float) / 1e9
    s = np.asarray(res.S)[0, 0, :]
    z0 = np.asarray(res.Z0)[0, :]
    g = _gate_readings(fr, s, z0)

    # --- R5 witnesses: full trace (|S11|, Re/Im Zin), never a bare headline ---
    print(f"\n[PATCH-EDGEFED/118-REG] max|S11| = {g['max_s11']:.4f}  dip @ "
          f"{g['f_dip_ghz']:.3f} GHz (|S11|={g['s11_at_dip']:.4f}, the off-resonance "
          "MATCH point)")
    print(f"[PATCH-EDGEFED/118-REG] min|S11| over resonance band {RES_BAND_GHZ} GHz = "
          f"{g['band_min_s11']:.4f} (HIGH => poorly matched at resonance); in-band "
          f"max Re(Zin) = {g['band_max_re_zin']:.0f} ohm")
    print(f"[PATCH-EDGEFED/118-REG] Im(Zin)=0 crossings = "
          f"{[round(c, 4) for c in g['crossings_ghz']]} GHz "
          f"(in band: {[round(c, 4) for c in g['band_crossings_ghz']]})")
    print(f"[PATCH-EDGEFED/118-REG] Z0[0] median Re = {np.median(z0.real):.2f} ohm "
          f"(analytic Hammerstad-Jensen ~50.6 ohm)")
    for f, a, zr, zi in zip(fr, g["s11"], g["zin"].real, g["zin"].imag):
        print(f"[PATCH-EDGEFED/118-TRACE] {f:7.3f} GHz  |S11|={a:.5f}  "
              f"Re(Zin)={zr:9.2f}  Im(Zin)={zi:9.2f}")

    # --- (1) passivity: the issue #80 fix (was |S11|=1.44/8.94, now <= 1.05) ---
    assert g["max_s11"] <= PASSIVE_TOL, (
        f"non-passive patch: max|S11| = {g['max_s11']:.4f} > {PASSIVE_TOL}. "
        "DO NOT loosen — this guards the #116 n_probe_offset passivity fix."
    )

    # --- (2) edge-fed signature: the patch is POORLY matched at its resonance, so the
    #         |S11| dip cannot be (and is not) the resonance. The physically robust
    #         negation of the historical "dip near the Balanis value" mis-spec (#118),
    #         now over the band measured on THIS board (#782). ---
    assert g["band_min_s11"] > RES_BAND_S11_MIN, (
        f"min|S11| = {g['band_min_s11']:.4f} over the resonance band {RES_BAND_GHZ} GHz "
        f"is unexpectedly LOW (<= {RES_BAND_S11_MIN}). A directly edge-fed patch must be "
        "poorly matched at its TM010 resonance (high edge resistance); a deep dip there "
        "would mean the geometry/feed changed. Modal frequencies are checked by the "
        "Harminv companion."
    )

    # --- (2b) in-band resonance witness: the resonance the band names is THERE.
    #          Liveness only — (2b) does NOT discriminate the #702 retirement: the
    #          bit-exact pre-#702 physics also has an in-band crossing (9.108 GHz, a
    #          low-impedance negative-Re shoulder), so this assertion PASSES on the
    #          retired arm. The discriminating assertion is (2c): the retired arm
    #          reads in-band max Re(Zin) 9.2 ohm vs the 500-ohm floor (470x; evidence
    #          docs/design_notes/issue782_retired_resonance_predeclaration.md). ---
    assert g["band_crossings_ghz"], (
        f"no Im(Zin)=0 crossing inside the resonance band {RES_BAND_GHZ} GHz — the band "
        "no longer contains the fixture's antiresonance, so assertion (2) would be "
        f"testing a dead spectral region (the #782 failure class). Measured crossings: "
        f"{[round(c, 4) for c in g['crossings_ghz']]} GHz. Either the physics moved "
        "(re-pin from a fresh evidence run, with a written root cause) or the extraction "
        "changed."
    )

    # --- (2c) the in-band antiresonance is the HIGH-IMPEDANCE kind — the mechanism
    #          behind (2); a low-impedance in-band crossing would be a different
    #          feature wearing the band's name. ---
    assert g["band_max_re_zin"] > RES_BAND_RE_ZIN_MIN_OHM, (
        f"in-band max Re(Zin) = {g['band_max_re_zin']:.0f} ohm <= "
        f"{RES_BAND_RE_ZIN_MIN_OHM:.0f} — the band's crossing is not the edge-fed "
        "patch antiresonance (measured 4326 ohm on the 2026-09-01 provenance run; "
        "pre-#702 witness class saw > 1.5 kohm)."
    )

    # --- (3) soft: the |S11| minimum (match point) lies ABOVE the resonance band ---
    assert g["f_dip_ghz"] > RES_BAND_GHZ[1], (
        f"|S11| dip at {g['f_dip_ghz']:.3f} GHz is inside/below the resonance band — "
        "expected the off-resonance match point ABOVE it (measured 10.100 GHz on the "
        "provenance run). The exact dip frequency is mesh-limited and an unstable "
        "argmin (see #118); only the lower bound is asserted."
    )
