"""cv05 canonical patch far-field — rfx vs committed openEMS thirds-rule reference.

ENVELOPE REGRESSION LOCK, not an accuracy claim.  This test runs the validated
rfx side of the canonical patch far-field lane (the openEMS "Simple Patch
Antenna" tutorial geometry: 32 x 40 mm patch, eps_r = 3.38 / 1.524 mm substrate,
60 x 60 mm ground plane, probe feed 6 mm off centre) and gates it against the
COMMITTED openEMS reference fixture
``tests/fixtures/patch_canonical_farfield_e4/patch_farfield_openems.json`` —
our source-built openEMS run of the canonical thirds-rule recipe
(``scripts/diagnostics/patch_tutorial_openems.py``): f_res = 2.4221 GHz,
Q = 20.1, broadside D = 6.79 dBi.  Full provenance, the reproduce-gate status,
and the honest labelling of those numbers as OUR run's measured values (not an
openEMS-published citation) live in the fixture ``meta`` block.

Every tolerance below is a MEASURED envelope plus stated margin (rfx numbers
from ``scripts/diagnostics/patch_tutorial_rfx.py`` and
``examples/tutorials/patch_antenna_demo.py`` on this same geometry):

* broadside directivity: measured |rfx - openEMS| = 0.0659 dB at the
  DESIGN-mode bin (6.7241 vs 6.7900 dBi) on the #931 sheet-declared board
  (VESSL 369367259302) — the gate stays at 1.0 dB (``D_ABS_TOL_DB`` below),
  which is 15x the measurement: the derivation rule gives 0.1 dB and the
  constant is HELD at its pre-#931 value, because a rerun that happens to land
  closer is not a reason to narrow a regression gate. The pre-#931 board
  measured 0.60 dB here.
  DOCSTRING CORRECTION 2026-08-31 (#812 Phase 0, prose only — no constant
  changed): this bullet previously read "measured 0.08 dB in the research
  frame (6.71 vs 6.79 dBi, num_periods=250) and 0.11 dB in this test's lean
  frame (6.68 dBi, num_periods=90) — locked at 0.5 dB". Those were the
  wrong-mode (40 mm cross-mode) numbers and the pre-#693 tolerance; PR #716 /
  commit 12d2f00 repinned the constant to 1.0 dB on 2026-08-27 but left this
  bullet and the gate's own docstring quoting the retired values;
* E-/H-plane beam peaks: measured -1 deg / -4 deg (openEMS 0 / 0) — locked at
  15 deg from broadside;
* resonance (RE-DERIVED 2026-09-07 on the #931 sheet-declared board, VESSL
  369367259302): the design mode — the 32 mm feed-axis mode, rfx ring-down
  2.5070 GHz — reads high by +3.51% vs the openEMS 2.4221 GHz at dx = 2 mm.
  On the pre-#931 board it read +11.3%, and the difference is the mechanism
  #693 named: the one-cell PEC ground's own cell sat inside the modelled
  cavity as vacuum and diluted eps_eff. The band is re-derived by the same
  rule that shaped the old one (measured ± 5 percentage points, rounded
  outward to whole percent) and is now [-2%, +9%].

  **THE GATE IS NO LONGER SIGN-LOCKED, AND THAT IS A CHANGE OF KIND.** The old
  [+6%, +16%] excluded zero, so it asserted "rfx reads HIGH here" as part of
  the characterization. [-2%, +9%] contains zero: what it now asserts is that
  the coarse-dx offset stays inside ±one band-width of the measured +3.51%,
  with the HIGH side still nine times wider than the low. This was written
  down in advance as the outcome that would need saying rather than
  re-tuning (docs/design_notes/931_migration/XA-manifest.json.md §5), and it
  is said here rather than absorbed by picking a narrower rule after seeing
  the number. It remains a discretization-bias regression lock, NOT an
  rfx-vs-openEMS accuracy claim.

  History, retired: the pre-#693 docstring said "-8.6% LOW" because the old
  selector tracked the 40 mm CROSS mode (2.2147 GHz); its dx-ladder claims
  (-6% at dx=1, -3% extrapolated) are wrong-mode data (a per-mode ladder is
  tracked in #693).

The -40 dB ring-down settling witness is asserted in-test before any gated
number (num_periods=110; the num_periods=90 demo of this fixture measured
-36.5 dB, i.e. just under the bar — 110 clears it with margin). Measured
-42.9 dB on the sheet-declared board, so the numbers above are quotable; a
re-derivation from a run that did not clear this bar would not be.

Humble-crossval note: all distances are stated factually as properties of our
coarse fixture; nothing here ranks the solvers.

The FDTD-running gates are marked ``slow`` (~12 min CPU); the fixture
provenance/integrity gates are fast and run in the default selection.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from rfx import (
    Box,
    GaussianPulse,
    Simulation,
    compute_far_field_jax,
    directivity,
    harminv,
    smooth_grading,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO_ROOT / "tests/fixtures/patch_canonical_farfield_e4/patch_farfield_openems.json"

C0 = 2.998e8
EPS0 = 8.8541878e-12

# ---- Geometry: identical to the openEMS Simple_Patch tutorial (fixture meta) ----
PATCH_W = 32.0e-3
PATCH_L = 40.0e-3
SUB_EPS_R = 3.38
SUB_THICK = 1.524e-3
TAN_DELTA = 1.0e-3
GP_SIZE = 60.0e-3
FEED_OFFSET_X = -6.0e-3
F_DESIGN = 2.2e9
SIGMA_SUB = 2 * math.pi * 2.45e9 * EPS0 * SUB_EPS_R * TAN_DELTA

# ---- Mesh/frame: the validated lean frame of examples/tutorials/patch_antenna_demo.py ----
DX = 2.0e-3
N_CPML = 8
N_SUB = 4
DZ_SUB = SUB_THICK / N_SUB
MARGIN_XY = 85.0e-3
AIR_BELOW = 30.0e-3
AIR_ABOVE = 84.0e-3
# 110 periods: the num_periods=90 run of this exact fixture measured a -36.5 dB
# end-of-run envelope (under the -40 dB bar); the radiating mode's Q ~ 59 decays
# ~0.46 dB/period, so 110 lands near -45 dB — settled with margin.
NUM_PERIODS = 110
NTFF_FREQS = np.array([2.0e9, 2.1e9, 2.2e9, 2.3e9, 2.4e9, 2.5e9, 2.8e9])

# ---- Measured-envelope gates (repinned 2026-08-27, issue #693 root cause) ----
# The pre-#693 envelopes were pinned against the WRONG member of the patch's
# near-degenerate mode pair: openEMS 2.4221 GHz is the 32 mm (feed-axis)
# mode, whose rfx counterpart is the UPPER ring-down mode (2.6954 GHz,
# radiated-power max, broadside, 6.5x amplitude), not the 40 mm cross mode
# (2.2147 GHz) the old F_GUESS selector picked. Full written root cause with
# the evidence chain: issue #693 (2026-08-27 comments).
#
# MECHANISM (measured decomposition, #693 closure): the +11.3% was NOT
# primarily a dx-convergence bias — the ladder measured it GROWING at finer
# dx. It was the one-plane REALIZATION: the ground's own cell sat inside the
# cavity as vacuum (+15%p, series-capacitance dilution of eps_eff; the old
# preflight sheet-cavity check printed it as +84.5% electrical thickness),
# plus the one-plane length cost, minus ~4%p in-plane staircase.
#
# #931: that mechanism is gone, at its source. The foil is declared a SHEET on
# the laminate's own faces and a sheet owns no cell, so there is no vacuum
# ground cell to dilute anything and the two reserved fine cells are out of
# the mesh. Every constant below was measured on the geometry that no longer
# exists, so none was carried across: VESSL 369367259302 re-solved the case
# and the record is tests/fixtures/patch_canonical_farfield_e4/
# canonical_farfield_e4_measured_369367259302.json, which carries the
# arithmetic for every value here. Settling -42.9 dB, so the run is quotable.
# The #740 two_plane arm's -4.7% (dx=2) was the PREDICTION to check against;
# measured +3.51%, i.e. the direction it gave (down from +11.3%) but not the
# crossing. Recorded as a prediction that got the sign of the CHANGE right
# and the sign of the RESULT wrong -- not repaired, and not re-fitted.
_ENVELOPES_REDERIVED_FOR_931 = True
_MIGRATION_RUN = ("VESSL 369367259302 — the canonical patch re-solved on the "
                  "sheet-declared board (crossval-A, #931). Record: tests/"
                  "fixtures/patch_canonical_farfield_e4/canonical_farfield_"
                  "e4_measured_369367259302.json; producer + rule in its meta")

D_ABS_TOL_DB = 1.0          # measured 0.0659 dB (rfx 6.7241 vs openEMS 6.7900)
                            # on the sheet board; ceil(0.0659 x 1.5, .1) = 0.1,
                            # HELD at 1.0 — a rerun may not narrow a gate.
PEAK_ANGLE_TOL_DEG = 15.0   # measured -1 / -4 deg vs openEMS 0 / 0. Broadside
                            # is a symmetry statement, not a realization one,
                            # so this one survives the change.
F_RES_REL_LO = -0.02        # measured +3.51% on the sheet board (2.5070 GHz
F_RES_REL_HI = +0.09        # vs the committed openEMS 2.4221), re-derived as
                            # measured +- 5 pp rounded outward to whole percent
                            # — the same rule that shaped [+6%, +16%] around a
                            # measured +11.3%. THIS BAND STRADDLES ZERO, so the
                            # gate is no longer SIGN-locked: see the docstring.
SETTLING_BAR_DB = -40.0


def _require_rederived_envelopes():
    if not _ENVELOPES_REDERIVED_FOR_931:
        import pytest as _pytest
        _pytest.skip(
            "the gated envelopes on this file were measured on the pre-#931 "
            "board (a one-cell PEC ground with its own vacuum cell inside the "
            "cavity). The board is now sheet-declared and those numbers cannot "
            "be translated; re-derive them from the new run and set "
            f"_ENVELOPES_REDERIVED_FOR_931 = True in that commit. {_MIGRATION_RUN}")


def _load_reference() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Fast gates: committed-fixture provenance and integrity (no FDTD).
# --------------------------------------------------------------------------


def test_reference_fixture_pins_the_recorded_openems_numbers():
    """The committed reference numbers are the recorded 2026-07-11 GOOD-run values."""
    ref = _load_reference()
    assert ref["f_res_ghz"] == 2.4221
    assert ref["Q"] == 20.1
    assert ref["directivity_dbi"] == 6.79
    assert ref["broadside_ok"] is True
    assert ref["s11_dip_ghz"] == 2.43 and ref["s11_min_db"] == -27.85
    # full principal-plane cuts are committed (92-point psi grid, max-normalized)
    for key in ("psi_deg", "E_plane_norm", "H_plane_norm"):
        assert len(ref[key]) == 92
    assert max(ref["E_plane_norm"]) == 1.0 and max(ref["H_plane_norm"]) == 1.0


def test_reference_fixture_provenance_and_reproduce_gate_are_recorded():
    """Provenance is part of the reference: recipe, producing script, run id, and an
    HONEST reproduce-gate record. The gate's LETTER was closed 2026-07-18 by the
    literal-tutorial-first external re-verification (unmodified upstream tutorial run
    + external post-processing of its own port data), and the fixture must carry both
    that closure AND the history that the record-first artifact was once retroactive.
    Values stay labelled as OUR runs' measurements — envelope lock, not a
    published-number match. Fails closed if someone strips the honesty block."""
    meta = _load_reference()["meta"]
    assert "AddEdges2Grid" in meta["recipe"] and "thirds-rule" in meta["recipe"]
    assert meta["source_script"].startswith("scripts/diagnostics/patch_tutorial_openems.py")
    assert (_REPO_ROOT / "scripts/diagnostics/patch_tutorial_openems.py").is_file()
    assert (_REPO_ROOT / "scripts/diagnostics/patch_tutorial_rfx.py").is_file()
    assert meta["run"]["vessl_run_good"] == "369367246713"
    rg = meta["reproduce_gate"]
    assert rg["status"].startswith("satisfied")
    assert "retroactively" in rg["status"] or "retroactively" in rg["detail"]
    assert "ENVELOPE LOCK" in rg["consequence"]
    assert rg["follow_up"].startswith("COMPLETED 2026-07-18")
    assert meta["comparison_kind"] == "committed_external_reference_envelope_lock"
    # the literal tutorial's own reproduced number (the reproduce-gate letter)
    lit = meta["literal_tutorial_reproduction"]
    assert lit["f_s11_dip_ghz"] == 2.4325 and lit["s11_min_db"] == -33.0
    assert lit["vessl_run"] == "369367247478"
    # the fresh independent-build re-verification of the committed numbers
    fresh = meta["fresh_reverification"]
    assert fresh["f_res_ghz"] == 2.4222 and fresh["directivity_dbi"] == 6.80
    # the reference run itself clears the settling bar this test enforces on rfx
    assert meta["run"]["end_energy_db"] <= SETTLING_BAR_DB


def test_reference_fixture_geometry_matches_this_test():
    """Geometry identity: the rfx build below and the committed openEMS reference
    describe the same antenna (guards fixture/test divergence)."""
    geo = _load_reference()["meta"]["geometry"]
    assert geo["patch_mm"] == [PATCH_W * 1e3, PATCH_L * 1e3]
    assert geo["sub_eps_r"] == SUB_EPS_R
    assert geo["sub_thick_mm"] == SUB_THICK * 1e3
    assert geo["tan_delta"] == TAN_DELTA
    assert geo["gp_mm"] == [GP_SIZE * 1e3, GP_SIZE * 1e3]
    assert geo["feed_x_mm"] == FEED_OFFSET_X * 1e3


# --------------------------------------------------------------------------
# Slow gates: run the rfx side and lock the measured envelope.
# --------------------------------------------------------------------------


def _build_simulation() -> Simulation:
    """The validated lean-frame build (mirrors patch_antenna_demo.build_simulation)."""
    n_below = int(math.ceil(AIR_BELOW / DX))
    n_above = int(math.ceil(AIR_ABOVE / DX))
    dom_x = GP_SIZE + 2 * MARGIN_XY
    dom_y = GP_SIZE + 2 * MARGIN_XY
    cx, cy = dom_x / 2, dom_y / 2

    # #931: the fine band is N_SUB cells, not 1 + N_SUB + 1. The two extra
    # fine cells existed to give the ground and the patch each a cell of their
    # own, because a one-cell PEC Box was the only way to ask for a wall — and
    # the ground's cell then sat INSIDE the modelled cavity as vacuum, which is
    # the #693 "vacuum ground cell" and the measured +15%p of the +11.3%
    # resonance bias. Under the ownership contract the foil is declared a
    # SHEET on a node plane (design note §1.3), a sheet owns no cell, and the
    # cells it used to need are not in the mesh at all.
    raw_dz = np.concatenate([
        np.full(n_below, DX),
        np.full(N_SUB, DZ_SUB),
        np.full(n_above, DX),
    ])
    dz_profile = smooth_grading(raw_dz, max_ratio=1.3)
    edges = np.insert(np.cumsum(dz_profile), 0, 0.0)
    z_total = float(edges[-1])

    # Register the stack to the BUILT mesh (#325 lesson): derive z coordinates
    # from where smooth_grading actually put the fine band.
    fine = np.where(np.isclose(dz_profile, DZ_SUB, rtol=1e-6))[0]
    assert len(fine) >= N_SUB, "graded mesh lost the fine band"
    f0 = int(fine[0])
    z_sub_lo, z_sub_hi = edges[f0], edges[f0 + N_SUB]

    centers = 0.5 * (edges[:-1] + edges[1:])
    sub_cells = int(np.sum((centers >= z_sub_lo) & (centers < z_sub_hi)))
    assert sub_cells == N_SUB, (
        f"substrate landed on {sub_cells} cells instead of {N_SUB} — "
        "stack is mis-registered to the graded mesh (#325 class)"
    )

    sim = Simulation(
        freq_max=4e9,
        domain=(dom_x, dom_y, 0),
        dx=DX,
        dz_profile=dz_profile,
        boundary="cpml",
        cpml_layers=N_CPML,
    )
    sim.add_material("sub", eps_r=SUB_EPS_R, sigma=SIGMA_SUB)

    gx_lo, gx_hi = cx - GP_SIZE / 2, cx + GP_SIZE / 2
    gy_lo, gy_hi = cy - GP_SIZE / 2, cy + GP_SIZE / 2
    # Ground and patch are 35 um copper foil on a 1.524 mm laminate — foil, so
    # SHEETS on the substrate's own faces (#931 §1.3). Drawn zero-thickness at
    # the exact node planes the graded mesh produced, so nothing is snapped.
    sim.add_thin_conductor(
        Box((gx_lo, gy_lo, z_sub_lo), (gx_hi, gy_hi, z_sub_lo)))
    sim.add(Box((gx_lo, gy_lo, z_sub_lo), (gx_hi, gy_hi, z_sub_hi)), material="sub")
    sim.add_thin_conductor(
        Box((cx - PATCH_W / 2, cy - PATCH_L / 2, z_sub_hi),
            (cx + PATCH_W / 2, cy + PATCH_L / 2, z_sub_hi)))

    feed_x, feed_y = cx + FEED_OFFSET_X, cy
    src_z = z_sub_lo + DZ_SUB * 1.5
    sim.add_source(
        position=(feed_x, feed_y, src_z),
        component="ez",
        waveform=GaussianPulse(f0=F_DESIGN, bandwidth=1.2),
    )
    sim.add_probe(position=(feed_x + 4e-3, feed_y + 4e-3, src_z), component="ez")

    # NTFF box: side/top faces keep lambda/2-class clearance; the bottom face
    # sits 6 mm below the ground plane (the domain has only 30 mm of air below)
    # — preflight flags exactly that face, and the placement is backed by the
    # 0.08 dB solver-to-solver cross-check rather than silence.
    pad = (N_CPML + 3) * DX
    box_lo = (pad, pad, max(pad, z_sub_lo - 3 * DX))
    box_hi = (dom_x - pad, dom_y - pad, z_total - pad)
    sim.add_ntff_box(corner_lo=box_lo, corner_hi=box_hi, freqs=NTFF_FREQS)
    return sim


def _principal_cut(power_f: np.ndarray, phi_index_pos: int, phi_index_neg: int):
    """Compose a -90..+90 deg cut from two opposite azimuth columns."""
    n_half = 91
    pos = power_f[:n_half, phi_index_pos]
    neg = power_f[:n_half, phi_index_neg][::-1]
    angle = np.concatenate([-np.arange(90, 0, -1.0), np.arange(0, 91, 1.0)])
    return angle, np.concatenate([neg[:-1], pos])


def test_the_board_realizes_the_planes_it_declares():
    """BUILD-TIME (no solve): the sheet planes ARE the laminate's own faces.

    #931 §1.3. Two things have to be true and neither is visible in a
    directivity number: the foil realizes ONE wall on each laminate face (not
    a slab with a second wall a cell away), and the substrate between them is
    N_SUB cells of laminate with nothing else in it. The #693 defect this file
    documents was precisely the second one failing — a vacuum cell inside the
    cavity, which cost +15%p of resonance and was invisible to every gate here
    for months.

    The falsifier arm is the one that would have caught it: NO wall one fine
    cell below the laminate floor, which is where the old one-cell ground Box
    put its only wall.
    """
    from tests._realized_pec import (assert_no_wall_at, assert_normal_edge_live,
                                     assert_sheet_owns_no_cell,
                                     assert_sheet_planes, realize, wall_planes)

    sim = _build_simulation()
    realized = realize(sim)
    assert len(realized.sheets) == 2, (
        f"the canonical patch declares {len(realized.sheets)} sheet(s); "
        "ground and patch are both foil")
    assert_sheet_owns_no_cell(realized, what="canonical patch foil")
    assert_normal_edge_live(realized, what="canonical patch foil")

    z_nodes = realized.nodes(2)
    planes = sorted(int(sp.plane) for sp in realized.sheets)
    assert len(planes) == 2 and planes[1] - planes[0] == N_SUB, (
        f"ground and patch sit {planes[1] - planes[0]} cells apart, not "
        f"{N_SUB}: the laminate is not the cavity")
    z_lo, z_hi = float(z_nodes[planes[0]]), float(z_nodes[planes[1]])
    assert z_hi - z_lo == pytest.approx(SUB_THICK, rel=1e-9), (
        f"realized cavity {1e3 * (z_hi - z_lo):.4f} mm against a declared "
        f"{1e3 * SUB_THICK:.4f} mm laminate")
    assert_sheet_planes(realized, 2, [z_lo, z_hi], what="canonical patch foil")

    # The eps between the two planes is laminate all the way — no vacuum cell.
    eps = np.asarray(realized.materials.eps_r)
    i_mid, j_mid = eps.shape[0] // 2, eps.shape[1] // 2
    column = eps[i_mid, j_mid, planes[0]:planes[1]]
    assert np.allclose(column, SUB_EPS_R, rtol=1e-6), (
        f"the modelled cavity carries eps {sorted(set(column.tolist()))} "
        f"instead of {SUB_EPS_R} — the #693 vacuum ground cell is back")

    # Falsifier: nothing one fine cell below the floor, which is where the old
    # one-cell PEC ground Box stood its only wall.
    assert planes[0] - 1 not in wall_planes(realized, 2), (
        "a wall is realized one fine cell below the laminate floor — the "
        "#693 geometry")
    assert_no_wall_at(realized, 2, [float(z_nodes[planes[0] - 1])],
                      what="canonical patch board")


@pytest.fixture(scope="module")
def rfx_run():
    """One FDTD run shared by all slow gates. Preflight advisories are quoted
    verbatim (they are part of the result) and the settling witness is computed
    before any frequency is extracted."""
    sim = _build_simulation()
    advisories = [str(issue) for issue in sim.preflight()]
    print(f"\npreflight advisories ({len(advisories)}) — quoted verbatim:")
    for advisory in advisories:
        print(f"  ! {advisory}")

    t0 = time.time()
    result = sim.run(num_periods=NUM_PERIODS)
    wall_s = time.time() - t0

    ts = np.asarray(result.time_series).ravel()
    dt = float(result.dt)
    envelope = np.abs(ts)
    peak = float(np.max(envelope))
    tail = float(np.max(envelope[int(len(envelope) * 0.95):]))
    end_db = 20 * math.log10(max(tail, 1e-300) / peak)
    print(f"settling witness: end-of-run envelope {end_db:.1f} dB of peak (bar {SETTLING_BAR_DB} dB)")

    modes = [
        m
        for m in harminv(ts[int(len(ts) * 0.3):], dt, 1.0e9, 3.5e9)
        if m.Q > 2 and m.amplitude > 1e-8
    ]
    modes.sort(key=lambda m: m.freq)
    print("harminv ring-down modes:")
    for m in modes:
        print(f"  f = {m.freq / 1e9:.4f} GHz | Q = {m.Q:.1f} | amp = {m.amplitude:.3g}")

    theta = np.linspace(0, np.pi, 181)
    phi = np.linspace(0, 2 * np.pi, 49)
    ff = compute_far_field_jax(result.ntff_data, result.ntff_box, result.grid, theta, phi)
    d_dbi = np.asarray(directivity(ff))
    power = np.abs(np.asarray(ff.E_theta)) ** 2 + np.abs(np.asarray(ff.E_phi)) ** 2
    dth = np.gradient(theta)
    dph = np.gradient(phi)
    p_rad = np.sum(
        power * np.sin(theta)[None, :, None] * dth[None, :, None] * dph[None, None, :],
        axis=(1, 2),
    )
    p_rel_db = 10 * np.log10(p_rad / p_rad.max())
    peak_theta_deg = np.degrees(theta[np.argmax(np.max(power, axis=2), axis=1)])
    for k, f in enumerate(NTFF_FREQS):
        print(
            f"  {f / 1e9:.1f} GHz: P_rad {p_rel_db[k]:6.1f} dB | beam peak "
            f"theta = {peak_theta_deg[k]:5.1f} deg | D = {d_dbi[k]:.2f} dBi"
        )

    # Radiating-mode ID: broadside beam AND the radiated-power peak — never
    # amplitude rank, never distance to a textbook estimate (both shortcuts
    # have mis-identified modes on this exact structure).
    broadside = peak_theta_deg <= PEAK_ANGLE_TOL_DEG
    k_star = int(np.argmax(np.where(broadside, p_rel_db, -np.inf))) if broadside.any() else -1
    radiating = (
        min(modes, key=lambda m: abs(m.freq - NTFF_FREQS[k_star])) if (modes and k_star >= 0) else None
    )

    cuts = {}
    if k_star >= 0:
        angle_xz, cut_xz = _principal_cut(power[k_star], 0, 24)   # phi = 0 / 180 deg (E-plane)
        angle_yz, cut_yz = _principal_cut(power[k_star], 12, 36)  # phi = 90 / 270 deg (H-plane)
        cuts = {
            "E_peak_deg": float(angle_xz[int(np.argmax(cut_xz))]),
            "H_peak_deg": float(angle_yz[int(np.argmax(cut_yz))]),
        }

    print(f"FDTD wall: {wall_s:.0f} s")
    return {
        "advisories": advisories,
        "end_db": end_db,
        "modes": modes,
        "broadside_any": bool(broadside.any()),
        "k_star": k_star,
        "peak_theta_deg": peak_theta_deg,
        "d_dbi": d_dbi,
        "p_rel_db": p_rel_db,
        "f_radiating_hz": float(radiating.freq) if radiating is not None else float("nan"),
        "q_radiating": float(radiating.Q) if radiating is not None else float("nan"),
        "cuts": cuts,
        "wall_s": wall_s,
    }


@pytest.mark.slow
def test_settling_witness_clears_the_bar(rfx_run):
    """Ring-down settling witness (repo rule): no gated number is quotable from a
    truncated run. num_periods=110 measured ~-45 dB; bar is -40 dB."""
    assert rfx_run["end_db"] < SETTLING_BAR_DB, (
        f"end-of-run envelope {rfx_run['end_db']:.1f} dB does not clear the "
        f"{SETTLING_BAR_DB} dB settling bar — raise NUM_PERIODS before trusting any gate"
    )


@pytest.mark.slow
def test_radiating_mode_is_broadside(rfx_run):
    """Far-field mode ID: a broadside bin exists, the radiated-power peak is
    broadside, and both principal-plane beam peaks sit within 15 deg of
    broadside (measured: rfx -1/-4 deg, openEMS reference 0/0 deg)."""
    assert rfx_run["broadside_any"], (
        f"no monitored bin shows a broadside beam; peak thetas = {rfx_run['peak_theta_deg']}"
    )
    k = rfx_run["k_star"]
    assert rfx_run["peak_theta_deg"][k] <= PEAK_ANGLE_TOL_DEG
    ref = _load_reference()
    for plane, key in (("E", "E_peak_deg"), ("H", "H_peak_deg")):
        assert abs(rfx_run["cuts"][key]) <= PEAK_ANGLE_TOL_DEG, (
            f"{plane}-plane beam peak {rfx_run['cuts'][key]:.1f} deg off broadside "
            f"(openEMS reference: {ref[f'{plane}_plane_peak_deg']} deg)"
        )


@pytest.mark.slow
def test_directivity_within_committed_envelope(rfx_run):
    """|D_rfx - D_openEMS| <= D_ABS_TOL_DB = 1.0 dB at the radiating bin.
    Measured on the #931 sheet-declared board (VESSL 369367259302): 0.0659 dB,
    rfx 6.7241 vs openEMS 6.7900 dBi. The gate is HELD at 1.0 dB rather than
    re-derived downward — the file's rule, round-UP(measured x 1.5), gives
    0.1 dB, and a rerun that lands closer is not an argument for narrowing a
    regression gate. So the headroom is now 15x the measurement instead of
    1.7x; it still fails closed if the far-field lane regresses, and it now
    also fails closed a long way before the pre-#931 board's own 0.60 dB.

    DOCSTRING CORRECTION 2026-08-31 (#812 Phase 0, prose only — the asserted
    constant is unchanged): this docstring previously read "<= 0.5 dB ...
    Measured: 0.08 dB in the research frame (6.71 vs 6.79 dBi), 0.11 dB in
    this lean frame (6.68 dBi at num_periods=90)". Those figures were taken at
    the WRONG member of the near-degenerate TM pair (the 40 mm cross mode);
    PR #716 / commit 12d2f00 repinned D_ABS_TOL_DB 0.5 -> 1.0 on 2026-08-27
    and recorded the design-mode measurement at line 111, but this docstring
    was left behind."""
    _require_rederived_envelopes()
    ref = _load_reference()
    d_rfx = float(rfx_run["d_dbi"][rfx_run["k_star"]])
    diff = abs(d_rfx - ref["directivity_dbi"])
    assert diff <= D_ABS_TOL_DB, (
        f"broadside D {d_rfx:.2f} dBi vs committed openEMS {ref['directivity_dbi']:.2f} dBi: "
        f"|diff| {diff:.2f} dB exceeds the {D_ABS_TOL_DB} dB measured envelope"
    )


@pytest.mark.slow
def test_mode_pair_present_with_aspect_ratio(rfx_run):
    """Physics lock born from #693: the 32x40 patch supports a near-degenerate
    TM mode pair whose frequency ratio tracks the aspect ratio (TL model
    f32/f40 = 1.232 at these eps_eff; measured 2.5070/2.0346 = 1.2322 on the
    #931 sheet board, 2.6954/2.2147 = 1.217 before it).  The band below is
    NOT re-derived: 1.2322 sits inside [1.15, 1.30] with margin, the quantity
    is an aspect-ratio statement rather than a realization one, and the
    producer's rule would have moved the band by +-0.01 in both directions —
    widening the high side of a gate for no measured reason.  Any
    selector or realization change that loses one member — the failure mode
    that produced the wrong-mode envelope — fails here by name instead of
    surfacing as an inexplicable offset sign flip."""
    _require_rederived_envelopes()
    fs = sorted(m.freq for m in rfx_run["modes"] if 1.8e9 < m.freq < 3.1e9)
    assert len(fs) >= 2, (
        f"mode pair lost: only {[f/1e9 for f in fs]} GHz in 1.8-3.1 GHz — "
        "the 32x40 patch must show both TM members (#693)")
    ratio = fs[-1] / fs[0]
    assert 1.15 <= ratio <= 1.30, (
        f"mode-pair ratio {ratio:.3f} outside the aspect-locked band "
        f"[1.15, 1.30] (modes {[round(f/1e9, 4) for f in fs]} GHz)")


@pytest.mark.slow
def test_f_res_inside_documented_coarse_dx_envelope(rfx_run):
    """ENVELOPE REGRESSION LOCK, not accuracy: at dx = 2 mm the rfx DESIGN
    mode (32 mm feed-axis; far-field-identified) reads +3.51% against the
    committed openEMS reference on the #931 sheet-declared board (2.5070 vs
    2.4221 GHz, VESSL 369367259302). Locked at [-2%, +9%], the measured value
    +- 5 percentage points rounded outward — the same rule that shaped the
    pre-#931 [+6%, +16%] around a measured +11.3%.

    WHAT THIS GATE NOW ASSERTS, AND WHAT IT NO LONGER DOES. The old band
    excluded zero, so passing it was also a statement that rfx reads HIGH
    here; the deleted mechanism (#693: the one-cell ground's own cell inside
    the cavity as vacuum) was what put the whole band on one side. [-2%, +9%]
    contains zero, so the SIGN is no longer part of the characterization. What
    survives is the magnitude envelope and the asymmetry: the high side is
    nine times the low side, so a run drifting back toward the pre-#931 +11.3%
    still fails, and so does one that swings 2% low. A wrong member of the
    mode pair fails here as it always did. XA-manifest.json.md 5 pre-declared
    that a band around zero would be a different kind of lock and would have
    to be SAID rather than re-tuned into looking like the old one; this
    docstring is that saying.

    History, retired: the pre-#693 [-12%, -5%] envelope was pinned on the
    40 mm cross mode."""
    _require_rederived_envelopes()
    ref = _load_reference()
    f_ref = ref["f_res_ghz"] * 1e9
    rel = (rfx_run["f_radiating_hz"] - f_ref) / f_ref
    assert F_RES_REL_LO <= rel <= F_RES_REL_HI, (
        f"radiating mode {rfx_run['f_radiating_hz'] / 1e9:.4f} GHz vs committed openEMS "
        f"{ref['f_res_ghz']} GHz: offset {rel * 100:+.1f}% outside the documented "
        f"dx=2 mm bias envelope [{F_RES_REL_LO * 100:.0f}%, {F_RES_REL_HI * 100:.0f}%] "
        "(this envelope is a discretization-bias regression lock, NOT an accuracy claim)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-q", "-m", "slow or not slow"])
