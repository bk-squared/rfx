"""Gate MATH for the cv05/cv15 mode-resolved resonance selector (issue #812).

Both patch cross-vals used to pick their gated resonance as
``argmin |f - f_analytic|`` and then report it against that same anchor.  This
file pins the replacement: identification of every ring-down mode against the
DECLARED TM_mn0 spectrum, with the design member required to be found.

Everything here is a pure function over frequency lists -- no FDTD, no solver
-- following ``tests/test_crossval_gate_logic.py`` and
``tests/test_crossval_cv15_wall_planes.py``'s precedent for this crossval
directory.  The live-FDTD reproductions live in
``docs/design_notes/20260901_patch_mode_identification_predeclaration.md``'s
measurement log and in ``tests/fixtures/patch_mode_identification/``.

The tolerance under test is DERIVED, not fitted:
``tol = sqrt(min adjacent declared member ratio) - 1`` -- the largest tolerance
for which "nearest declared member" is unique.  No measured frequency enters
it.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARATORS = REPO_ROOT / "validation" / "crossval" / "comparators"
CV15_PATH = REPO_ROOT / "validation" / "crossval" / "15_patch_antenna_rt5880.py"
sys.path.insert(0, str(COMPARATORS))

from patch_mode_identification import (   # noqa: E402
    declared_cavity_spectrum,
    identification_tolerance,
    identify_patch_modes,
    members_in_band,
    microstrip_eps_eff_and_dl,
)

# ---- the two cases' DECLARED geometry (constants, not measurements) ----
CV05 = dict(eps_r=4.3, h=1.5e-3, a=29.5e-3, b=38.0e-3, c0=2.998e8,
            band=(1.5e9, 3.5e9))
CV15 = dict(eps_r=2.2, h=3.175e-3, a=40.0e-3, b=50.0e-3, c0=2.99792458e8,
            band=(1.6e9, 3.4e9))


def _members(case):
    return members_in_band(
        declared_cavity_spectrum(case["eps_r"], case["h"], case["a"], case["b"],
                                 c0=case["c0"]),
        *case["band"])


def _load_cv15():
    spec = importlib.util.spec_from_file_location("_cv15_mode_id", CV15_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# The declared spectrum IS each script's own closed form
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case,expected_hz", [(CV05, 2423509824.70113),
                                             (CV15, 2415595433.5060616)])
def test_design_member_reproduces_the_scripts_own_closed_form(case, expected_hz):
    """TM100 of the declared spectrum must BE the case's single-mode Balanis
    anchor. If these two ever diverge the identification would be judging a
    different antenna from the one the script reports against."""
    eps_eff, dl = microstrip_eps_eff_and_dl(case["eps_r"], case["h"], case["b"])
    hand = case["c0"] / (2 * (case["a"] + 2 * dl) * math.sqrt(eps_eff))
    members = _members(case)
    assert members[(1, 0)] == pytest.approx(hand, rel=1e-12)
    assert members[(1, 0)] == pytest.approx(expected_hz, rel=1e-9)


def test_cv15_declared_constants_are_the_scripts_own():
    """``CV15`` above must BE cv15's declared constants, not a copy that has
    drifted -- every cv15 number in this file is derived from them.

    cv15 itself ships NO spectral gate (this lane landed cv15 as a STOP; see
    the design note section 6.9), so there is no module-side ``declared_modes()``
    to compare against -- the constants are the contract."""
    cv15 = _load_cv15()
    assert (cv15.EPS_R, cv15.H_SUB, cv15.L_PATCH, cv15.W_PATCH, cv15.C0) == \
        (CV15["eps_r"], CV15["h"], CV15["a"], CV15["b"], CV15["c0"])
    assert (cv15.F_LO, cv15.F_HI) == CV15["band"]
    assert _members(CV15)[(1, 0)] == pytest.approx(
        cv15.f_res_analytic()[0], rel=1e-12)
    assert not hasattr(cv15, "_mode_id_ok"), (
        "cv15 has grown a spectral gate -- criterion (B) was never met for it "
        "(design note section 6.9); re-open the STOP before shipping one")


# --------------------------------------------------------------------------
# The tolerance is derived from geometry alone
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case,expected_pct", [(CV05, 12.4988), (CV15, 10.6198)])
def test_identification_tolerance_is_the_derived_value(case, expected_pct):
    """tol = sqrt(min adjacent declared ratio) - 1, the largest tolerance for
    which "nearest declared member" is unique. Pinned to the value the design
    note pre-declared, and re-derived here from the ratios themselves."""
    members = _members(case)
    fs = sorted(members.values())
    r_min = min(f2 / f1 for f1, f2 in zip(fs, fs[1:]))
    tol = identification_tolerance(members)
    assert tol == pytest.approx(math.sqrt(r_min) - 1.0, rel=1e-12)
    assert tol * 100 == pytest.approx(expected_pct, abs=1e-3)


@pytest.mark.parametrize("case", [CV05, CV15])
def test_tolerance_windows_of_adjacent_members_do_not_overlap(case):
    """The defining property: at the derived tolerance no frequency can be
    inside two members' windows, so "nearest member" is never a coin flip."""
    members = _members(case)
    tol = identification_tolerance(members)
    fs = sorted(members.values())
    for f1, f2 in zip(fs, fs[1:]):
        assert f1 * (1 + tol) <= f2 / (1 + tol) * (1 + 1e-12)


# --------------------------------------------------------------------------
# Positive control: the declared spectrum identifies itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", [CV05, CV15])
def test_declared_spectrum_identifies_itself(case):
    members = _members(case)
    ident = identify_patch_modes(list(members.values()), members)
    assert ident.ok, ident.reasons
    assert ident.f_design == pytest.approx(members[(1, 0)])


# --------------------------------------------------------------------------
# FALSIFIER F1 (pre-declared): a design mode +24 % wrong must FAIL, by name
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", [CV05, CV15])
def test_design_mode_24_percent_high_fails_because_the_member_is_not_found(case):
    """The audit's cv05 defect: the true design mode is +24 % off its declared
    place. The OLD selector promoted whichever mode sat nearest the anchor and
    nothing failed. Here the drifted mode is assigned to a NEIGHBOUR (or to no
    member at all) and TM100 is reported MISSING -- the mode was not found, not
    merely mis-measured."""
    members = _members(case)
    f_bad = members[(1, 0)] * 1.24
    # the other in-plane axis is untouched by a resonant-length error
    measured = [members[(0, 1)], f_bad, math.hypot(f_bad, members[(0, 1)])]
    ident = identify_patch_modes(measured, members)
    assert not ident.ok
    assert any("DESIGN member TM100" in r and "has NO measured mode" in r
               for r in ident.reasons), ident.reasons
    assert ident.f_design is None
    # and the mode that drifted is NOT silently reported as the resonance
    assert all(o != (1, 0) for _f, o, _r in ident.assignments)


def test_two_modes_claiming_one_member_is_ambiguous_not_a_free_pick():
    """Injectivity: two ring-down modes inside one member's window must not let
    the gate pick the nicer one -- that is the anchored selector again."""
    members = _members(CV15)
    f = members[(1, 0)]
    ident = identify_patch_modes([members[(0, 1)], f * 0.98, f * 1.02], members)
    assert not ident.ok
    assert any("claimed by 2 measured modes" in r for r in ident.reasons)
    assert ident.f_design is None


def test_single_mode_verdict_is_refused():
    """G3: a verdict resting on one mode is what the anchored selector gave."""
    members = _members(CV15)
    ident = identify_patch_modes([members[(1, 0)]], members)
    assert not ident.ok
    assert any("second in-plane axis" in r for r in ident.reasons)


def test_mode_inside_the_span_matching_no_member_fails():
    members = _members(CV15)
    fs = sorted(members.values())
    midpoint = math.sqrt(fs[0] * fs[1])   # the worst case by construction
    ident = identify_patch_modes([members[(0, 1)], members[(1, 0)], midpoint],
                                 members)
    assert not ident.ok
    assert any("matches NO declared member" in r for r in ident.reasons)


def test_modes_above_the_span_are_reported_but_not_gated():
    """Higher members the in-band declared set does not model must not be able
    to fail the gate by existing."""
    members = _members(CV15)
    high = max(members.values()) * 1.5
    ident = identify_patch_modes(list(members.values()) + [high], members)
    assert ident.ok, ident.reasons
    assert (high, None, None) in ident.assignments


# --------------------------------------------------------------------------
# The stated blindness, pinned as a test so it cannot be quietly overclaimed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1.02, 1.068, 1.10])
def test_common_mode_dilation_is_invisible_and_that_is_documented(scale):
    """Design note section 5, pinned: every observable here is a ratio of a
    measured frequency to a declared one, so a UNIFORM rescale of the whole
    spectrum shifts every residual by exactly the scale factor and leaves every
    dimensionless observable (the mode-pair RATIO included) unchanged.

    The middle scale is the measured cv15 #740 vacuum-ground-cell dilation,
    ``cv15_ringdown_spectra.json::measured_common_mode_dilation.mean`` (pinned
    to that key by ``test_cv15_740_defect_is_a_common_mode_dilation``).

    This is a statement about an EXACT dilation only. The real #740 dilation is
    not exact, and what its residual spread does to the mode-pair ratio is a
    separate, measured question -- see
    ``cv15_mode_pair_ratio_band.json::declared_anchored_band``."""
    members = _members(CV15)
    measured = [f * scale for f in members.values()]
    ident = identify_patch_modes(measured, members)
    assert ident.ok, ident.reasons
    for _f, order, rel in ident.assignments:
        if order is not None:
            assert rel == pytest.approx(scale - 1.0, rel=1e-9)
    # the pair ratio -- the audit's proposed cv15 instrument -- is invariant
    fs = sorted(f for f in measured)
    assert fs[1] / fs[0] == pytest.approx(
        sorted(members.values())[1] / sorted(members.values())[0], rel=1e-12)


# ==========================================================================
# Committed MEASURED spectra: criteria (A) and (B), re-runnable without a solve
# ==========================================================================

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "patch_mode_identification"


def _fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_cv05_correct_build_passes_with_margin():
    """(A) cv05's committed configuration, run through its own script: the
    design member TM100 is FOUND, its residual is more than 3x inside the
    derived identification tolerance, and TM110 resolves the second in-plane
    axis.

    Frequencies: ``cv05_ringdown_spectra.json::runs.baseline.modes``. The
    reported resonance is the SAME mode the anchored selector returned (the
    lowest in-band mode), so no physics verdict cv05 publishes moves."""
    runs = _fixture("cv05_ringdown_spectra.json")["runs"]
    members = _members(CV05)
    freqs = [m["freq"] for m in runs["baseline"]["modes"]]
    ident = identify_patch_modes(freqs, members)
    assert ident.ok, ident.reasons
    # the same mode the OLD argmin|f - f_analytic| selector would have returned
    assert ident.f_design == min(freqs, key=lambda f: abs(f - members[(1, 0)]))
    rel = ident.f_design / members[(1, 0)] - 1
    assert abs(rel) < ident.tol / 3.0             # margin, not a squeaker
    assert (1, 1) in {o for _f, o, _r in ident.assignments if o}


@pytest.mark.parametrize("run,f_ghz,rel", [
    # values are the committed fixture's (rebuilt on the lab cluster, jax 0.6.2,
    # VESSL 369367257743 -- the round-1 macOS build read 2.87308 / 2.98956 /
    # 3.11259 here, i.e. within 2e-4 relative at two of three lengths and
    # 1.3e-3 at 22.0 mm; design note 6.11)
    ("patch_len_22p5mm", 2.87272, +0.1854),
    ("patch_len_22p0mm", 2.99346, +0.2352),   # the audit's +24% point
    ("patch_len_21p0mm", 3.11189, +0.2840),
])
def test_cv05_mis_realized_resonant_length_fails_for_the_stated_reason(run, f_ghz, rel):
    """(B) LIVE FDTD reproductions: the patch's realized resonant length is
    mis-built while every declaration -- including the anchor -- still says
    29.5 mm. Frequencies and the realized-cell census:
    ``cv05_ringdown_spectra.json::runs`` and ``::_realized_x_cell_census``.

    The audit's +24% point is measured directly by the 22-cell realization
    (``runs.patch_len_22p0mm``); the parametrization's ``rel`` column is
    re-derived from the fixture in the body below, so these digits are checked,
    not asserted. Exactly +24.00% is pinned algebraically by
    ``test_design_mode_24_percent_high_fails_because_the_member_is_not_found``.

    In every case the drifted design mode is captured by a NEIGHBOURING
    declared member -- the cross-mode capture the audit described -- and the
    gate reports TM100 MISSING rather than reporting the neighbour as the
    resonance."""
    data = _fixture("cv05_ringdown_spectra.json")["runs"][run]
    members = _members(CV05)
    freqs = [m["freq"] for m in data["modes"]]
    assert freqs[0] / 1e9 == pytest.approx(f_ghz, abs=1e-5)
    assert freqs[0] / members[(1, 0)] - 1 == pytest.approx(rel, abs=5e-4)
    ident = identify_patch_modes(freqs, members)
    assert not ident.ok
    assert ident.f_design is None
    assert any("DESIGN member TM100" in r and "has NO measured mode" in r
               for r in ident.reasons), ident.reasons
    # the drifted mode did NOT become the reported resonance
    assert all(o != (1, 0) for _f, o, _r in ident.assignments)


def test_cv15_correct_build_would_pass_with_margin():
    """(A) for cv15, recorded even though cv15 ships no spectral gate: the
    correct (two_plane) build's ring-down identifies every in-band declared
    member with margin against the derived tolerance.

    Source: ``cv15_ringdown_spectra.json::two_plane_ground.modes`` -- the live
    reproduction through cv15's production ``build_rfx_sim(two_plane=True)``.
    (A) alone is cosmetic; (B) is what cv15 could not meet."""
    fx = _fixture("cv15_ringdown_spectra.json")
    members = _members(CV15)
    ident = identify_patch_modes(
        [m["freq_hz"] for m in fx["two_plane_ground"]["modes"]], members)
    assert ident.ok, ident.reasons
    worst = max(abs(r) for _f, o, r in ident.assignments if o is not None)
    assert worst < 0.5 * ident.tol        # margin, not a squeaker


def test_cv15_reproduction_ringdown_matches_the_committed_leg():
    """The reproduction the two cv15 fixtures rest on is the same ring-down as
    the leg #768 committed -- so reverting this lane's regeneration of
    ``_15_patch_results/rfx.json`` (design note section 6.9) costs no evidence.

    The leg carries no mode list; f0 is the field the two share."""
    fx = _fixture("cv15_ringdown_spectra.json")["two_plane_ground"]
    leg = json.loads(
        (REPO_ROOT / "validation/crossval/_15_patch_results/rfx.json")
        .read_text(encoding="utf-8"))
    assert "modes" not in leg          # the committed leg is #768's, untouched
    assert fx["f_harminv_hz"] == pytest.approx(leg["f_harminv_hz"], rel=1e-7)


def test_cv15_740_defect_is_a_common_mode_dilation():
    """The pre-declared scope limit, MEASURED on a live reproduction of the
    #740 one-plane ground realization through cv15's production builder.

    The three declared members move together to within the fixture's recorded
    half spread, so the mode-resolved identification PASSES on the defect --
    criterion (B) is not met and cv15 lands as a STOP. #740's detector is
    ``assert_realized_stack``, whose refusal is committed in the fixture.

    What this test does NOT say is that no dimensionless spectral test could
    fire: the ratio does move, and by how much is
    ``cv15_mode_pair_ratio_band.json`` (next test), not this one."""
    fx = _fixture("cv15_ringdown_spectra.json")
    members = _members(CV15)

    good = [m["freq_hz"] for m in fx["two_plane_ground"]["modes"]]
    bad = [m["freq_hz"] for m in fx["one_plane_ground_740_defect"]["modes"]]
    assert identify_patch_modes(good, members).ok
    ident_bad = identify_patch_modes(bad, members)
    assert ident_bad.ok, "the spectral gate is blind to #740 -- by construction"
    assert ident_bad.f_design / 1e9 == pytest.approx(2.47186, abs=1e-5)

    # it IS a dilation: every member moves by the same factor
    dil = fx["measured_common_mode_dilation"]
    per = dil["per_member"]
    assert dil["mean"] == pytest.approx(
        sum(per.values()) / len(per), rel=1e-12)
    assert dil["half_spread"] == pytest.approx(
        (max(per.values()) - min(per.values())) / 2, rel=1e-12)
    for i, name in enumerate(("TM010", "TM100", "TM110")):
        assert per[name] == pytest.approx(sorted(bad)[i] / sorted(good)[i],
                                          rel=1e-12)
    # every residual under the defect is the dilation, and all stay in tolerance
    for _f, order, rel in ident_bad.assignments:
        if order is not None:
            assert abs(rel) < ident_bad.tol
    assert dil["half_spread"] < 0.5 * ident_bad.tol

    # the instrument that DOES see it, quoted from the same reproduction
    assert fx["one_plane_ground_740_defect"]["assert_realized_stack"].startswith(
        "RuntimeError: assert_realized_stack:")
    assert "no electric wall at z_sub_lo" in \
        fx["one_plane_ground_740_defect"]["assert_realized_stack"]


def test_cv15_one_plane_reproduction_matches_the_committed_prefix_leg():
    """The live one-plane reproduction and the committed pre-fix leg
    (`rfx_one_plane_ground_b29f9de7.json`) are the same defect: their ring-down
    f0 agree to 6e-9 relative."""
    fx = _fixture("cv15_ringdown_spectra.json")["one_plane_ground_740_defect"]
    committed = json.loads(
        (REPO_ROOT / "validation/crossval/_15_patch_results"
         / "rfx_one_plane_ground_b29f9de7.json").read_text(encoding="utf-8"))
    assert committed["f_harminv_hz"] == pytest.approx(
        fx["committed_prefix_leg_f_harminv_hz"], rel=1e-15)
    assert fx["f_harminv_hz"] == pytest.approx(committed["f_harminv_hz"], rel=1e-8)


def test_committed_prefix_leg_still_fails_the_live_judge():
    """The committed pre-fix leg fails cv15's live judge -- on the
    realized-wall-plane check, NOT on any measurement of its spectrum. Pinned so
    the distinction cannot be blurred in a later summary: #740 is caught by
    geometry, and nothing this lane wrote catches it."""
    cv15 = _load_cv15()
    leg = json.loads(
        (REPO_ROOT / "validation/crossval/_15_patch_results"
         / "rfx_one_plane_ground_b29f9de7.json").read_text(encoding="utf-8"))
    sc_ok, sc_detail = cv15._stack_check_ok(leg.get("stack_check"))
    assert sc_ok is False and "missing stack_check" in sc_detail


# ==========================================================================
# The withdrawn absolute (design note section 6.9): the ratio band that CAN
# separate the two realizations, and why it still is not one we may adopt
# ==========================================================================


def test_mode_pair_ratio_band_census_reproduces_from_the_committed_spectra():
    """Round 1 published "a ratio band cannot fire on #740 at ANY width that
    admits the correct build". It is false. This test re-derives every number
    in ``cv15_mode_pair_ratio_band.json`` from the ring-down fixture and the
    declared geometry, so the refutation is mechanical rather than asserted:
    the admissible half-width interval is non-empty.

    Emitter: ``scripts/diagnostics/build_patch_mode_pair_ratio_band_census.py``
    (no FDTD)."""
    band = _fixture("cv15_mode_pair_ratio_band.json")
    fx = _fixture("cv15_ringdown_spectra.json")
    members = _members(CV15)
    r_decl = members[(1, 0)] / members[(0, 1)]
    assert band["declared"]["pair_ratio"] == pytest.approx(r_decl, rel=1e-12)
    assert band["declared"]["identification_tolerance"] == pytest.approx(
        identification_tolerance(members), rel=1e-12)

    widths = {}
    for key, leg in (("correct_build", "two_plane_ground"),
                     ("defect_740", "one_plane_ground_740_defect")):
        fs = sorted(m["freq_hz"] for m in fx[leg]["modes"])
        r = fs[1] / fs[0]
        assert band["measured"][key]["pair_ratio"] == pytest.approx(r, rel=1e-12)
        assert band["measured"][key]["residual_vs_declared"] == pytest.approx(
            r / r_decl - 1.0, rel=1e-12)
        widths[key] = abs(r / r_decl - 1.0)

    dab = band["declared_anchored_band"]
    assert dab["min_half_width_admitting_correct_build"] == pytest.approx(
        widths["correct_build"], rel=1e-12)
    assert dab["max_half_width_still_rejecting_740"] == pytest.approx(
        widths["defect_740"], rel=1e-12)
    # THE REFUTATION: the interval is non-empty
    assert widths["correct_build"] < widths["defect_740"]
    assert dab["admissible_interval_is_nonempty"] is True
    assert dab["detection_ratio"] == pytest.approx(
        widths["defect_740"] / widths["correct_build"], rel=1e-12)
    assert dab["upper_endpoint_over_identification_tolerance"] == pytest.approx(
        widths["defect_740"] / identification_tolerance(members), rel=1e-12)


def test_a_band_from_the_census_interval_does_separate_the_two_realizations():
    """Constructive form of the same refutation: take a half-width strictly
    inside the census interval, apply it as a gate, and watch it PASS the
    correct build and FAIL the #740 realization.

    This is what makes the round-1 absolute false. It is NOT an endorsement:
    the width used here has no provenance but the two measurements it judges
    (burned data, the burned-data rule: a threshold may not be set from the measurements it will judge) -- see the artifact's ``verdict``."""
    band = _fixture("cv15_mode_pair_ratio_band.json")
    fx = _fixture("cv15_ringdown_spectra.json")
    members = _members(CV15)
    r_decl = members[(1, 0)] / members[(0, 1)]
    lo = band["declared_anchored_band"]["min_half_width_admitting_correct_build"]
    hi = band["declared_anchored_band"]["max_half_width_still_rejecting_740"]
    w = math.sqrt(lo * hi)          # geometric midpoint of the interval

    def fires(leg):
        fs = sorted(m["freq_hz"] for m in fx[leg]["modes"])
        return abs(fs[1] / fs[0] / r_decl - 1.0) > w

    assert not fires("two_plane_ground")            # correct build admitted
    assert fires("one_plane_ground_740_defect")     # #740 rejected


def test_the_census_interval_is_far_tighter_than_anything_derivable():
    """Why the interval above is not adopted, stated as an assertion rather
    than as prose: its upper endpoint is orders inside the only window this
    lane can derive from declared geometry alone."""
    band = _fixture("cv15_mode_pair_ratio_band.json")
    assert band["declared_anchored_band"][
        "upper_endpoint_over_identification_tolerance"] < 0.05
    assert "burned-data" in band["verdict"]
    assert "STOP" in band["verdict"]


def test_cv05_constants_in_this_file_are_the_scripts_own():
    """Round-2 review: the CV05 dict above was a hand copy of
    05_patch_antenna.py's constants, pinned nowhere (the script cannot be
    imported without solving). Read them out of the source text instead, so
    a change to the case's declared geometry, c0 or harminv band fails
    here rather than silently detuning the spectrum this file re-derives."""
    import re
    src = (REPO_ROOT / "validation" / "crossval" / "05_patch_antenna.py").read_text(encoding="utf-8")
    def const(name):
        m = re.search(r"^%s\s*=\s*([0-9.eE+-]+)" % re.escape(name), src, re.M)
        assert m, name
        return float(m.group(1))
    assert const("C0") == CV05["c0"]
    assert const("eps_r") == CV05["eps_r"]
    assert const("h_sub") == CV05["h"]
    assert const("L") == CV05["a"]
    assert const("W") == CV05["b"]
    m = re.search(r"^HARMINV_F_LO,\s*HARMINV_F_HI\s*=\s*([0-9.eE+-]+),\s*([0-9.eE+-]+)", src, re.M)
    assert m and (float(m.group(1)), float(m.group(2))) == CV05["band"]


def test_cv05_38mm_build_is_not_caught_on_the_cluster_host_a_fired_falsifier():
    """(B) at L = 38.0 mm (-25 %, a square patch) fired on the macOS build
    (two poles: 1.81365 GHz -> TM010, 3.5769 -> none; TM100 MISSING) and
    does NOT fire on the committed cluster build: a third, weak pole appears
    at 2.6096 GHz (+7.68 % of TM100, amplitude an order below the others)
    inside the identification window and is named TM100. Recorded as a
    fired falsifier against the INSTRUMENT (a spurious low-amplitude pole
    in the design window is accepted), not softened: no amplitude floor is
    added after the fact. The three other mis-realized lengths still fail
    by name (test above); the audit's +24 % point is among them."""
    data = _fixture("cv05_ringdown_spectra.json")["runs"]["patch_len_38p0mm"]
    members = _members(CV05)
    modes = sorted(data["modes"], key=lambda m: m["freq"])
    freqs = [m["freq"] for m in modes]
    assert len(freqs) == 3
    assert freqs[0] / 1e9 == pytest.approx(1.81334, abs=1e-5)
    assert freqs[1] / 1e9 == pytest.approx(2.60961, abs=1e-5)
    assert freqs[1] / members[(1, 0)] - 1 == pytest.approx(+0.0768, abs=5e-4)
    amps = [m["amplitude"] for m in modes]
    assert amps[1] < 0.1 * max(amps), "the accepted pole is the weak one"
    ident = identify_patch_modes(freqs, members)
    assert ident.ok is True and ident.f_design == pytest.approx(freqs[1])
    assert [o for _f, o, _r in ident.assignments] == [(0, 1), (1, 0), None]
