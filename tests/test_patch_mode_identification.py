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
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
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


def test_cv15_module_declared_modes_matches_this_construction():
    """cv15's own ``declared_modes()`` must be the same spectrum -- the gate in
    ``compare()`` re-derives it from the module's constants, so a drift between
    the module and this test would silently re-scope the gate."""
    cv15 = _load_cv15()
    assert cv15.declared_modes() == _members(CV15)
    assert cv15.declared_modes()[(1, 0)] == pytest.approx(
        cv15.f_res_analytic()[0], rel=1e-12)


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

    1.068 is the measured cv15 #740 vacuum-ground-cell dilation. This test
    exists so that "the mode-pair gate closes #740" can never be written: it
    does not, and #740's detector is ``assert_realized_stack`` (PR #768)."""
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


# --------------------------------------------------------------------------
# cv15's leg-level gate
# --------------------------------------------------------------------------


def test_cv15_leg_without_a_mode_list_fails_and_says_it_is_a_schema_failure():
    """A leg predating the selector carries no spectrum: FAIL, not skip. The
    detail must say SCHEMA so this is never quoted as a physics measurement."""
    cv15 = _load_cv15()
    ok, detail = cv15._mode_id_ok({"f_harminv_hz": 2.4e9})
    assert ok is False
    assert detail.startswith("SCHEMA:")


def test_cv15_leg_gate_accepts_a_correct_declared_spectrum():
    cv15 = _load_cv15()
    members = cv15.declared_modes()
    leg = {"modes": [{"freq_hz": f, "Q": 20.0, "amplitude": 1e-3}
                     for f in members.values()]}
    ok, detail = cv15._mode_id_ok(leg)
    assert ok is True, detail
    assert "TM100" in detail


def test_cv15_leg_gate_fails_a_24_percent_design_mode_error():
    cv15 = _load_cv15()
    members = cv15.declared_modes()
    f_bad = members[(1, 0)] * 1.24
    leg = {"modes": [{"freq_hz": f, "Q": 20.0, "amplitude": 1e-3}
                     for f in (members[(0, 1)], f_bad,
                               math.hypot(f_bad, members[(0, 1)]))]}
    ok, detail = cv15._mode_id_ok(leg)
    assert ok is False
    assert "DESIGN member TM100" in detail
