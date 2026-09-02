"""Unit tests for the #589 label-independent ladder witnesses (no FDTD).

``scripts/diagnostics/coax_msl_ladder_witnesses.py`` is a pure-NumPy post-
processor for the raw ladder voltages and flux spectra the settled-run driver
dumps. Every witness here is exercised on a SYNTHETIC ladder whose travel
direction, reflection magnitude and propagation constant are known by
construction, and -- the point of a witness -- each criterion is shown to be
DECIDABLE: the same helper is fed a field that satisfies it and a field that
violates it, so a criterion that can only ever say "survives" is caught here
rather than after a GPU run.

Sign contract under test (frozen): the probe DFT kernel is ``exp(-j 2 pi f t)``
(rfx/probes/probes.py:93,408,572,707,987), so a wave travelling toward +axis is
``exp(-j beta z)`` and its adjacent-pair phase slope
``angle(V[p+1] conj(V[p]))/dz`` is NEGATIVE.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "diagnostics"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling fixture module

import coax_msl_ladder_witnesses as W  # noqa: E402

from test_coax_msl_transition import (  # noqa: E402
    DX, FLUX_BOX_X_3, FLUX_BOX_Y_3, FLUX_BOX_Z_3, FLUX_COAX_PATCH_3,
    FLUX_COAX_Z_3, FREQS_2, JUNCTION_X, LX_2, LY, LZ_2, N_GND, Y_C,
    _attempt3_scratch_flux_entries,
)

C0 = 299792458.0
EPS_COAX = 2.1
FREQS = np.asarray(FREQS_2, dtype=float)                       # 6/8/10 GHz
BETA_COAX = 2.0 * np.pi * FREQS * np.sqrt(EPS_COAX) / C0       # 182/243/304 rad/m
BETA_MSL = 2.0 * np.pi * FREQS * np.sqrt(3.0) / C0             # a stand-in eps_eff

# Attempt-3 ladder geometry (the real one; see the driver's dump schema).
Z_COAX = np.arange(6) * 2 * DX + 0.9e-3        # 0.9 .. 1.9 mm, junction at 2.5 mm
REF_COAX = 2.5e-3
X_MSL = np.arange(9) * 10 * DX + 2.6e-3        # 2.6 .. 10.6 mm, junction at 1.0 mm
REF_MSL = JUNCTION_X
SETTLED_DB = np.asarray([-128.5, -118.8])      # the attempt-3 settled run's own figures
UNSETTLED_DB = np.asarray([-1.6, -1.8])        # the 300-step smoke's own figures


def _plant(pos, ref, gamma, toward_junction, amp_toward, amp_away):
    """Two-wave ladder ``(n_probes, n_f)``.

    ``toward_junction`` is '+axis' or '-axis'; the wave travelling toward
    +axis is ``exp(-gamma * (pos - ref))`` (phase decreasing with pos).
    Amplitudes are quoted AT the reference plane.
    """
    pos = np.asarray(pos, dtype=float)[:, None]
    r = pos - float(ref)
    plus = np.exp(-np.asarray(gamma)[None, :] * r)
    minus = np.exp(+np.asarray(gamma)[None, :] * r)
    if toward_junction == "+axis":
        return np.asarray(amp_toward)[None, :] * plus + np.asarray(amp_away)[None, :] * minus
    return np.asarray(amp_toward)[None, :] * minus + np.asarray(amp_away)[None, :] * plus


# ---------------------------------------------------------------------------
# W1 -- phase-slope direction
# ---------------------------------------------------------------------------
def test_phase_slope_sign_convention_is_the_probe_dft_kernel():
    """A +axis wave built as exp(-j beta z) has NEGATIVE adjacent-pair slope."""
    plus = np.exp(-1j * BETA_COAX[None, :] * Z_COAX[:, None])
    minus = np.exp(+1j * BETA_COAX[None, :] * Z_COAX[:, None])
    s_plus = W.phase_slopes(plus, Z_COAX)
    s_minus = W.phase_slopes(minus, Z_COAX)
    assert np.allclose(s_plus, -BETA_COAX[None, :], rtol=1e-10)
    assert np.allclose(s_minus, +BETA_COAX[None, :], rtol=1e-10)
    assert W.travel_direction_from_slope(-1.0) == "+axis"
    assert W.travel_direction_from_slope(+1.0) == "-axis"
    assert W.toward_junction_axis("above") == "+axis"
    assert W.toward_junction_axis("below") == "-axis"
    with pytest.raises(ValueError):
        W.toward_junction_axis("sideways")


def test_phase_slopes_reject_malformed_ladders():
    with pytest.raises(ValueError):
        W.phase_slopes(np.ones((3, 2), dtype=complex), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        W.phase_slopes(np.ones((3, 2), dtype=complex), np.array([3.0, 2.0, 1.0]))


@pytest.mark.parametrize("dominant, expected", [("away", "away"), ("toward", "toward")])
def test_w1_reads_the_dominant_travel_direction_on_a_coax_ladder(dominant, expected):
    """DECIDABLE both ways: the same helper reports 'away' on a ladder whose
    dominant wave leaves the junction and 'toward' on one where it arrives."""
    big, small = 11.0, 1.0
    toward, away = ((small, big) if dominant == "away" else (big, small))
    v = _plant(Z_COAX, REF_COAX, 1j * BETA_COAX, "+axis",
               toward * np.ones(3), away * np.ones(3))
    rows = W.w1_ladder(v, Z_COAX, junction_side="above", beta_analytic=BETA_COAX,
                       freqs=FREQS)
    for r, beta in zip(rows, BETA_COAX):
        assert r["dominant_relative_to_junction"] == expected
        assert r["sign_consistency"] == 1.0
        sign = +1.0 if expected == "away" else -1.0
        assert np.sign(r["mean_slope_rad_per_m"]) == sign
        assert abs(abs(r["mean_slope_rad_per_m"]) / beta - 1.0) < 0.30


def test_echo_aware_slope_tolerance_is_the_two_wave_bound():
    """The 15%-at-all-pairs criterion the design proposed is unattainable at
    the design's own 9% echo: a CLEAN two-wave field already deviates by
    2g/(1-g) = 19.8%. The helper's tolerance must bound the real deviation."""
    g = 0.09
    v = _plant(X_MSL, REF_MSL, 1j * BETA_MSL, "-axis",
               np.ones(3), g * np.ones(3))
    rows = W.w1_ladder(v, X_MSL, junction_side="below", beta_analytic=BETA_MSL,
                       freqs=FREQS)
    worst = max(r["worst_pair_rel_dev_from_beta"] for r in rows)
    assert worst > 0.15, worst                       # the design's kill is unreachable
    tol = W.echo_aware_slope_tolerance(g, margin=0.0)
    assert np.isclose(tol, 2 * g / (1 - g))
    assert worst <= tol + 1e-9, (worst, tol)         # the replacement bounds it


def _w1_rows(amp_toward, amp_away):
    n = len(FREQS)
    v = _plant(Z_COAX, REF_COAX, 1j * BETA_COAX, "+axis",
               amp_toward * np.ones(n), amp_away * np.ones(n))
    return W.w1_ladder(v, Z_COAX, junction_side="above", beta_analytic=BETA_COAX, freqs=FREQS)


def test_w1_h1_sign_witness_resolves_both_ways_on_a_settled_clean_ladder():
    """The verdict must be reachable in BOTH directions -- a witness that can
    only say one thing is not a falsifier.

    POST-#822 each direction carries the OPPOSITE conclusion from the one
    pinned before the fix: the assembler labels the wave LEAVING the
    junction ``b_out``, so a dominant outgoing wave at the undriven coax
    port agrees with the labels (H1 KILLED) and a dominant arriving one
    does not (H1 supported)."""
    clean = [1e-12] * len(FREQS)
    away = W.w1_h1_sign_witness(_w1_rows(1.0, 11.0), fit_residual_by_bin=clean,
                                settling_db=SETTLED_DB)
    toward = W.w1_h1_sign_witness(_w1_rows(11.0, 1.0), fit_residual_by_bin=clean,
                                  settling_db=SETTLED_DB)
    for r in away:
        assert r["verdict_resolved"] is True, r
        assert "H1 KILLED" in r["verdict"]
    for r in toward:
        assert r["verdict_resolved"] is True, r
        assert "H1 supported" in r["verdict"]


@pytest.mark.parametrize("settling, resid, why", [
    (UNSETTLED_DB, 1e-12, "not settled"),
    (None, 1e-12, "not settled"),
    (SETTLED_DB, 0.5, "not two-wave"),
])
def test_w1_h1_sign_witness_is_unresolved_when_a_precondition_fails(settling, resid, why):
    rows = W.w1_h1_sign_witness(_w1_rows(1.0, 11.0),
                                fit_residual_by_bin=[resid] * len(FREQS),
                                settling_db=settling)
    for r in rows:
        assert r["verdict_resolved"] is False, r
        assert r["verdict"].startswith("UNRESOLVED")
        assert any(why in f for f in r["preconditions_failed"]), r["preconditions_failed"]


def test_w1_h1_sign_witness_is_unresolved_on_extractor_floor():
    """The 300-step smoke's own coax-ladder-under-MSL-drive shape: pairwise
    slopes that flip sign along the ladder and a |mean|/beta of ~0.04. The
    pre-fix witness printed a resolved verdict at all three bins from
    exactly this; it must now print UNRESOLVED."""
    rng = np.random.default_rng(589)
    n = len(FREQS)
    noise = (rng.standard_normal((len(Z_COAX), n)) + 1j * rng.standard_normal((len(Z_COAX), n)))
    rows = W.w1_ladder(noise, Z_COAX, junction_side="above",
                       beta_analytic=BETA_COAX, freqs=FREQS)
    out = W.w1_h1_sign_witness(rows, fit_residual_by_bin=[1e-12] * n, settling_db=SETTLED_DB)
    for r in out:
        assert r["verdict_resolved"] is False, r
        assert r["verdict"].startswith("UNRESOLVED")


def test_settling_precondition_is_the_repo_minus_40_db_rule():
    assert W.SETTLING_DB_MAX == -40.0
    assert W.settling_precondition(SETTLED_DB)[0] is True
    assert W.settling_precondition(UNSETTLED_DB)[0] is False
    assert W.settling_precondition([-41.0, -39.0])[0] is False   # worst drive governs
    assert W.settling_precondition(None)[0] is False             # unknown is NOT a pass
    assert W.settling_precondition([np.nan, -100.0])[0] is False


# ---------------------------------------------------------------------------
# W2 -- SWR
# ---------------------------------------------------------------------------
def test_w2_recovers_gamma_when_the_span_covers_half_a_wavelength():
    g = 0.30
    pos = np.linspace(0.0, 3.0 * np.pi / BETA_MSL[0], 401)
    v = _plant(pos, 0.0, 1j * BETA_MSL, "-axis", np.ones(3), g * np.ones(3))
    rows = W.w2_ladder(v, pos, beta_analytic=BETA_MSL, freqs=FREQS)
    assert all(r["valid_as_abs_gamma"] for r in rows)
    for r in rows:
        assert abs(r["swr_abs_gamma"] - g) < 1e-3, r


def test_w2_says_lower_bound_only_on_the_short_coax_ladder():
    """The coax ladder spans 1 mm vs lambda/2 ~ 17 mm: W2 there is a LOWER
    bound and cannot bound the feed-end echo from above (review blocker a)."""
    g = 0.09
    v = _plant(Z_COAX, REF_COAX, 1j * BETA_COAX, "+axis", np.ones(3), g * np.ones(3))
    rows = W.w2_ladder(v, Z_COAX, beta_analytic=BETA_COAX, freqs=FREQS)
    for r in rows:
        assert not r["valid_as_abs_gamma"]
        assert "LOWER BOUND ONLY" in r["reading"]
        assert r["swr_abs_gamma"] < g          # strictly below the true value
        assert r["span_over_half_lambda"] < 0.1


# ---------------------------------------------------------------------------
# W3 -- subset fits, and the two repaired kill criteria
# ---------------------------------------------------------------------------
def test_w3_subset_fits_recover_a_planted_msl_ladder_and_h4_kill_fires():
    """H4's repaired kill (subset residual < 0.02 AND Im(gamma)/beta in
    [0.8, 1.3]) MUST fire on a clean two-wave MSL ladder -- otherwise H4
    could never die and the witness would be worthless."""
    g = 0.09
    v = _plant(X_MSL, REF_MSL, 1j * BETA_MSL, "-axis", np.ones(3), g * np.ones(3))
    rows = W.w3_subsets(v, X_MSL, ref=REF_MSL, junction_side="below",
                        beta_analytic=BETA_MSL, freqs=FREQS,
                        subsets=W.MSL_SUBSETS, label="msl")
    full = [r for r in rows if r["subset"] == "msl[all]"]
    assert len(full) == len(FREQS)
    for r, beta in zip(full, BETA_MSL):
        assert r["fit_residual"] < 1e-6, r
        assert abs(r["gamma_im"] / beta - 1.0) < 0.02, r
        assert abs(r["abs_reflection_at_ref"] - g) < 1e-6, r
    kill = W.h4_kill_from_w3(rows, FREQS)
    assert all(k["h4_killed_at_bin"] for k in kill), kill


def test_h4_kill_does_not_fire_on_a_non_two_wave_ladder():
    """The other side of the same criterion: a ladder carrying three modes
    (a parallel-plate/near-field contaminant) does not satisfy it."""
    rng = np.random.default_rng(589)
    v = _plant(X_MSL, REF_MSL, 1j * BETA_MSL, "-axis", np.ones(3), 0.09 * np.ones(3))
    contaminant = 0.8 * np.exp(-1j * 3.1 * BETA_MSL[None, :] * (X_MSL[:, None] - REF_MSL))
    v = v + contaminant + 0.05 * rng.standard_normal(v.shape)
    rows = W.w3_subsets(v, X_MSL, ref=REF_MSL, junction_side="below",
                        beta_analytic=BETA_MSL, freqs=FREQS,
                        subsets=W.MSL_SUBSETS, label="msl")
    kill = W.h4_kill_from_w3(rows, FREQS)
    assert not any(k["h4_killed_at_bin"] for k in kill), kill


def test_w3_h7_impact_criterion_is_attainable_on_a_lossless_coax_ladder():
    """H7's repaired kill is an IMPACT test (|Gamma_ref| shift when alpha is
    forced to 0), not 'Re(gamma) < 3 /m': on a lossless 1 mm ladder the fitted
    Re(gamma) is noise, but the referral bias it causes is < 1%."""
    v = _plant(Z_COAX, REF_COAX, 1j * BETA_COAX, "+axis",
               np.ones(3), 0.93 * np.ones(3))
    rows = W.w3_subsets(v, Z_COAX, ref=REF_COAX, junction_side="above",
                        beta_analytic=BETA_COAX, freqs=FREQS,
                        subsets=W.COAX_SUBSETS, label="coax")
    impact = W.h7_impact_from_w3(rows, FREQS)
    assert all(r["h7_killed_at_bin"] for r in impact), impact
    assert all(r["worst_alpha0_rel_shift"] < W.H7_ALPHA_IMPACT_MAX for r in impact)


def test_w3_h7_impact_criterion_survives_when_alpha_really_matters():
    """Decidable the other way: plant a genuinely lossy line (alpha = 90 /m
    over the 1.6 mm referral) and the alpha=0 refit moves |Gamma_ref| by more
    than 1%, so H7 survives."""
    gamma = 90.0 + 1j * BETA_COAX
    v = _plant(Z_COAX, REF_COAX, gamma, "+axis", np.ones(3), 0.5 * np.ones(3))
    rows = W.w3_subsets(v, Z_COAX, ref=REF_COAX, junction_side="above",
                        beta_analytic=BETA_COAX, freqs=FREQS,
                        subsets=W.COAX_SUBSETS, label="coax")
    impact = W.h7_impact_from_w3(rows, FREQS)
    assert not any(r["h7_killed_at_bin"] for r in impact), impact


# ---------------------------------------------------------------------------
# W4 -- flux box
# ---------------------------------------------------------------------------
def _flux_case(coax_z22, msl_frac=0.60):
    """One consistent set of six faces: outflow sums exactly to the inflow.

    ``msl_frac`` is ``msl_x20 / coax_z22``; ``top_z36`` absorbs the balance so
    the box still closes exactly, which lets a test drive the sign of the H1
    discriminator (``msl_x20``) INDEPENDENTLY of the sign of ``coax_z22``.
    """
    n = len(FREQS)
    one = np.ones(n)
    faces = {
        "coax_z22": coax_z22 * one,
        "msl_x20": msl_frac * coax_z22 * one,
        "top_z36": (0.80 - msl_frac) * coax_z22 * one,
        "xlo_x05": -0.10 * coax_z22 * one,
        "ylo_y03": -0.05 * coax_z22 * one,
        "yhi_y31": 0.05 * coax_z22 * one,
        "msl_x20_full": 0.75 * coax_z22 * one,
    }
    return faces


class _ExtractorOut:
    """The two amplitudes ``coaxial_line_reflection_from_plane_voltages``
    returns: ``forward_amp`` travels TOWARD the reference plane."""

    def __init__(self, forward, backward):
        self.forward_amp = forward
        self.backward_amp = backward


def _ab_for_flux(net_coax, net_msl_out):
    """a_inc/b_out (2, 2, n_f) planted by PHYSICS, labelled by PRODUCTION.

    The caller states two physical facts about the coax-drive column -- the
    net power flowing TOWARD the junction on the coax ladder (``net_coax``;
    +z points at the coax junction, so this is the net +z power) and the net
    power LEAVING the junction on the MSL ladder (``net_msl_out``; +x points
    away from the MSL junction) -- and the pair is then handed to the code's
    two arrays by production's OWN rule for this lane (``a = forward_amp``,
    the extractor's branch travelling toward the reference plane, which
    here IS the junction; ``b = backward_amp`` -- see the Notes of
    :func:`rfx.api._sparams._assemble_coax_msl_transition_from_voltages`).

    Planting straight onto ``a``/``b`` would re-encode the assembler's label
    constant in the test -- the round-trip issue #822 exists to eliminate.
    That is exactly how this file stayed green while
    :func:`coax_msl_ladder_witnesses.net_plus_axis_power` still carried the
    pre-#822 formula: the helper's constant and the fixture's constant
    cancelled, so the pair could not notice that one of them was wrong for
    the geometry.

    One branch is planted at zero so the tiny (1e-15 W) net survives in
    double precision -- ``|b|^2 = 1 + net`` against ``|a|^2 = 1`` would be
    rounded away, which is itself the reason the flux side is read in
    ``exact_f64`` and never as a difference of two O(1) numbers.
    """
    n = len(FREQS)
    a = np.zeros((2, 2, n), dtype=complex)
    b = np.zeros((2, 2, n), dtype=complex)
    # Coax ladder: junction (= reference plane) ABOVE the probes, so the
    # extractor's forward_amp -- the branch travelling toward the reference
    # plane -- is the +z wave, i.e. the one travelling toward the junction.
    toward_c = np.sqrt(net_coax) if net_coax >= 0 else 0.0
    away_c = 0.0 if net_coax >= 0 else np.sqrt(-net_coax)
    assert REF_COAX > Z_COAX.max()
    out_c = _ExtractorOut(toward_c, away_c)
    a[0, 0], b[0, 0] = out_c.forward_amp, out_c.backward_amp
    # MSL ladder: junction BELOW the probes, so forward_amp is the -x wave
    # (toward the junction) and the +x wave -- the one carrying net_msl_out
    # away from it -- is backward_amp.
    assert REF_MSL < X_MSL.min()
    out_m = _ExtractorOut(0.0, np.sqrt(net_msl_out))
    a[1, 0], b[1, 0] = out_m.forward_amp, out_m.backward_amp
    # MSL-drive column: not read by any assertion here, but it must keep the
    # 2x2 incident matrix non-singular (the end-to-end test re-solves S from
    # it). Physical placement: the MSL port is the driven one, so its
    # INCIDENT wave is the unit amplitude and the coax port only radiates.
    a[1, 1] = 1.0
    b[0, 1] = 1.0
    return a, b


@pytest.mark.parametrize("msl_frac, expect", [(+0.60, "H1 KILLED"), (-0.60, "H1 supported")])
def test_w4_h1_verdict_follows_the_sign_of_msl_x20_the_non_driven_port(msl_frac, expect):
    """The H1 flux discriminator is the sign of ``msl_x20`` under the coax
    drive, and BOTH branches are reachable with the SAME (passivity-required)
    ``coax_z22 > 0``. That is the whole point: coax_z22's sign is fixed by the
    geometry for any passive junction, so a verdict keyed on it can only ever
    say the same thing every time; msl_x20's sign is a property of the DUT.

    POST-#822 the two conclusions are attached to the OPPOSITE signs from
    the ones this test pinned before the fix: the assembler now labels the
    MSL ladder's +x (away-from-junction) branch ``b_out``, so power leaving
    the undriven port AGREES with the labels and kills H1. The measured
    sign did not move; the labels did."""
    coax_z22 = +1.17e-15
    a, b = _ab_for_flux(net_coax=+1.17e-15, net_msl_out=1.76e-15)
    out = W.w4_flux({"coax": _flux_case(coax_z22, msl_frac=msl_frac)},
                    a_inc=a, b_out=b, freqs=FREQS, settling_db=SETTLED_DB)
    rows = out["per_drive"]["coax"]["rows"]
    assert len(rows) == len(FREQS)
    for r in rows:
        assert r["h1_verdict_resolved"] is True, r["h1_flux_verdict"]
        assert expect in r["h1_flux_verdict"], r["h1_flux_verdict"]
        assert r["h1_discriminator"].startswith("sign(msl_x20)")
        assert abs(r["closure_rel_residual"]) < 1e-12, r
        assert r["R1_within_calibration_band"] is True          # planted exact
        assert np.isclose(r["R1_coax_z22_over_net_code_coax"], 1.0)
        assert np.isclose(r["R2_msl_x20_over_abs_msl_outgoing_sq"],
                          msl_frac * coax_z22 / 1.76e-15)


def test_w4_h1_verdict_is_not_keyed_on_coax_z22():
    """Regression pin for the review blocker. With coax_z22 held at the SAME
    positive value passivity requires, flipping only msl_x20 flips the verdict;
    and the coax_z22 line is stated as a passivity CHECK that is confirmatory
    only, never as evidence for H1."""
    a, b = _ab_for_flux(net_coax=+1.17e-15, net_msl_out=1.76e-15)
    verdicts = {}
    for msl_frac in (+0.60, -0.60):
        out = W.w4_flux({"coax": _flux_case(+1.17e-15, msl_frac=msl_frac)},
                        a_inc=a, b_out=b, freqs=FREQS, settling_db=SETTLED_DB)
        r = out["per_drive"]["coax"]["rows"][0]
        verdicts[msl_frac] = r["h1_flux_verdict"]
        assert "CONFIRMATORY ONLY" in r["coax_z22_passivity_check"]
        assert "not evidence for H1" in r["coax_z22_passivity_check"].lower() or \
               "NOT evidence for H1" in r["coax_z22_passivity_check"]
        assert r["coax_z22_sign_is_passivity_entailed"] is True
        assert "h1_sign_verdict" not in r          # the old, non-falsifiable key is gone
    assert "H1 KILLED" in verdicts[+0.60]        # labels agree with the flux
    assert "H1 supported" in verdicts[-0.60]     # labels disagree with it


def test_w4_h1_verdict_is_unresolved_on_an_unsettled_run():
    """The 300-step smoke's own numbers: settling -1.6/-1.8 dB and a box that
    does not close. No hypothesis verdict may be emitted from that state."""
    faces = _flux_case(+1.17e-15)
    a, b = _ab_for_flux(net_coax=+1.17e-15, net_msl_out=1.76e-15)
    out = W.w4_flux({"coax": faces}, a_inc=a, b_out=b, freqs=FREQS,
                    settling_db=UNSETTLED_DB)
    for r in out["per_drive"]["coax"]["rows"]:
        assert r["h1_verdict_resolved"] is False
        assert r["h1_flux_verdict"].startswith("UNRESOLVED")
        assert any("not settled" in f for f in r["h1_preconditions_failed"]), r


def test_w4_h1_verdict_is_unresolved_when_settling_is_absent():
    """An unknown settling is a FAILED precondition, not a pass."""
    a, b = _ab_for_flux(net_coax=+1.17e-15, net_msl_out=1.76e-15)
    out = W.w4_flux({"coax": _flux_case(+1.17e-15)}, a_inc=a, b_out=b, freqs=FREQS)
    for r in out["per_drive"]["coax"]["rows"]:
        assert r["h1_verdict_resolved"] is False
        assert "UNRESOLVED" in r["h1_flux_verdict"]


def test_w4_h1_verdict_is_unresolved_when_the_discriminator_is_below_the_box_error():
    """|msl_x20| must beat the box's own closure error by the declared factor;
    a settled run whose box closes but whose msl_x20 is smaller than the
    residual is UNRESOLVED, not a verdict."""
    faces = _flux_case(+1.0e-15, msl_frac=1e-4)      # msl_x20 = 1e-19
    faces["yhi_y31"] = faces["yhi_y31"] + 0.03e-15   # break closure by 3% (< 5% band)
    a, b = _ab_for_flux(net_coax=+1.0e-15, net_msl_out=1.0e-15)
    out = W.w4_flux({"coax": faces}, a_inc=a, b_out=b, freqs=FREQS, settling_db=SETTLED_DB)
    for r in out["per_drive"]["coax"]["rows"]:
        assert abs(r["closure_rel_residual"]) <= W.W4_CLOSURE_REL_MAX, r
        assert r["h1_verdict_resolved"] is False
        assert any("below the box's own error" in f for f in r["h1_preconditions_failed"]), r


def test_w4_closure_residual_catches_a_broken_box():
    faces = _flux_case(1.0e-15)
    faces["top_z36"] = faces["top_z36"] * 3.0          # a face that does not belong
    a, b = _ab_for_flux(net_coax=1.0e-15, net_msl_out=1.0e-15)
    out = W.w4_flux({"coax": faces}, a_inc=a, b_out=b, freqs=FREQS, settling_db=SETTLED_DB)
    for r in out["per_drive"]["coax"]["rows"]:
        assert abs(r["closure_rel_residual"]) > W.W4_CLOSURE_REL_MAX, r
        assert r["h1_verdict_resolved"] is False       # a leaking box decides nothing


def test_w4_r1_outside_the_calibration_band_is_flagged():
    """R1 is a Z0 CALIBRATION ratio with a >= 10% band, never an identity
    (review blocker 2): a 30% mismatch must be reported as outside it."""
    a, b = _ab_for_flux(net_coax=1.0e-15, net_msl_out=1.0e-15)
    out = W.w4_flux({"coax": _flux_case(1.3e-15)}, a_inc=a, b_out=b, freqs=FREQS,
                    settling_db=SETTLED_DB)
    for r in out["per_drive"]["coax"]["rows"]:
        assert r["R1_within_calibration_band"] is False
        assert np.isclose(r["R1_coax_z22_over_net_code_coax"], 1.3)
        # R1 is label-blind: it is out of band here yet the H1 verdict, which
        # does not consume it, still resolves off the sign of msl_x20.
        assert r["h1_verdict_resolved"] is True
        assert "H1 KILLED" in r["h1_flux_verdict"]


def test_w4_msl_drive_expectations_are_reported_not_asserted():
    a, b = _ab_for_flux(net_coax=-1.0e-15, net_msl_out=1.0e-15)
    out = W.w4_flux({"msl": _flux_case(-1.0e-15)}, a_inc=a, b_out=b, freqs=FREQS,
                    settling_db=SETTLED_DB)
    for r in out["per_drive"]["msl"]["rows"]:
        exp = r["msl_drive_expectations"]
        assert exp["coax_z22_negative"] is True
        assert exp["msl_x20_negative"] is True
        assert "h1_flux_verdict" not in r          # H1's flux witness is coax-drive only
        assert "h1_sign_verdict" not in r


def test_net_plus_axis_power_uses_the_per_ladder_reference_side():
    """POST-#822 ``a_code`` is the branch travelling TOWARD the reference
    plane, which on this lane IS the junction: the +z wave on the coax
    ladder (junction above) and the -x wave on the MSL ladder (junction
    below). The net +axis power is therefore ``|a|^2-|b|^2`` on the coax
    ladder and ``|b|^2-|a|^2`` on the MSL one -- the helper must not use one
    formula for both, and it must use the POST-fix one: with ``a_code``
    still read as the away-from-junction branch these two numbers came out
    +3 and -3, and every quantity derived from them (R1, R2net, the S-side
    ratio, and the coax-drive net +z power passivity forbids to be
    negative) inverted.
    """
    n = len(FREQS)
    a = np.zeros((2, 2, n), dtype=complex)
    b = np.zeros((2, 2, n), dtype=complex)
    a[:, :, :] = 1.0
    b[:, :, :] = 2.0
    k = 0
    assert np.isclose(W.net_plus_axis_power(a[:, :, k], b[:, :, k], 0, 0), -3.0)
    assert np.isclose(W.net_plus_axis_power(a[:, :, k], b[:, :, k], 1, 0), +3.0)


def test_net_plus_axis_power_recovers_the_planted_physical_net_power():
    """The label round trip, closed against PHYSICS rather than against a
    constant (issue #822 review).

    ``_ab_for_flux`` plants a stated net +z power at the coax ladder and a
    stated net +x power at the MSL ladder and lets PRODUCTION's own
    geometric split decide which array each amplitude lands in; the witness
    must read those same two numbers back, sign included. This is the pin
    the old ``net_plus_axis_power`` could not pass once the assembler's
    labels were corrected: it returned both with the sign flipped.
    """
    net_coax, net_msl_out = +1.17e-15, 1.76e-15
    a, b = _ab_for_flux(net_coax=net_coax, net_msl_out=net_msl_out)
    # atol=0.0 is load-bearing: these powers are O(1e-15), far below
    # np.isclose's DEFAULT atol=1e-8, which would make the comparison
    # vacuous -- it would pass even with the sign inverted.
    for k in range(len(FREQS)):
        assert np.isclose(W.net_plus_axis_power(a[:, :, k], b[:, :, k], 0, 0),
                          net_coax, rtol=1e-12, atol=0.0)
        assert np.isclose(W.net_plus_axis_power(a[:, :, k], b[:, :, k], 1, 0),
                          net_msl_out, rtol=1e-12, atol=0.0)
    # ... and a coax ladder whose net +z power is negative (the sign
    # passivity forbids under the coax drive) is reported as negative, not
    # silently turned positive by the label mapping.
    a_neg, b_neg = _ab_for_flux(net_coax=-1.0e-15, net_msl_out=net_msl_out)
    for k in range(len(FREQS)):
        assert np.isclose(W.net_plus_axis_power(a_neg[:, :, k], b_neg[:, :, k], 0, 0),
                          -1.0e-15, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# label-swap counterfactual
# ---------------------------------------------------------------------------
def test_label_swap_counterfactual_is_exactly_the_matrix_inverse():
    rng = np.random.default_rng(0)
    n = len(FREQS)
    a = rng.standard_normal((2, 2, n)) + 1j * rng.standard_normal((2, 2, n))
    b = rng.standard_normal((2, 2, n)) + 1j * rng.standard_normal((2, 2, n))
    cf = W.label_swap_counterfactual(a, b)
    assert cf["max_abs_dev_from_inv_s_code"] < 1e-9
    for k in range(n):
        assert np.allclose(cf["s_swap"][:, :, k],
                           np.linalg.inv(cf["s_code"][:, :, k]), atol=1e-10)


# ---------------------------------------------------------------------------
# end-to-end: the npz the driver writes -> tables
# ---------------------------------------------------------------------------
def _synthetic_dump():
    n = len(FREQS)
    coax = np.stack([
        _plant(Z_COAX, REF_COAX, 1j * BETA_COAX, "+axis", np.ones(n), 0.93 * np.ones(n)),
        _plant(Z_COAX, REF_COAX, 1j * BETA_COAX, "+axis", np.ones(n), 11.0 * np.ones(n)),
    ])
    msl = np.stack([
        _plant(X_MSL, REF_MSL, 1j * BETA_MSL, "-axis", 0.2 * np.ones(n), np.ones(n)),
        _plant(X_MSL, REF_MSL, 1j * BETA_MSL, "-axis", np.ones(n), 0.09 * np.ones(n)),
    ])
    a, b = _ab_for_flux(net_coax=1.17e-15, net_msl_out=1.76e-15)
    return {
        "freqs": FREQS, "a_inc": a, "b_out": b,
        "beta_coax_analytic": BETA_COAX, "beta_msl_analytic": BETA_MSL,
        "coax_ladder_v": coax, "coax_ladder_z_m": Z_COAX,
        "msl_ladder_v": msl, "msl_ladder_x_m": X_MSL,
        "ref_coax_m": REF_COAX, "ref_msl_m": REF_MSL,
        "flux_by_drive": {"coax": _flux_case(1.17e-15), "msl": _flux_case(-1.0e-15)},
        "settling_db": SETTLED_DB,
    }


def test_compute_witnesses_end_to_end_and_npz_round_trip(tmp_path):
    d = _synthetic_dump()
    w = W.compute_witnesses(d)
    # W1 on the coax ladder under the MSL drive: dominant wave leaves the junction
    for r in w["W1_h1_sign_witness"]:
        assert r["relative_to_junction"] == "away"
        assert r["preconditions_failed"] == [], r
        assert r["verdict_resolved"] is True
        assert "H1 KILLED" in r["verdict"]          # post-#822 labels agree
    # and the W4 discriminator resolves off msl_x20, not off coax_z22
    for r in w["W4_flux"]["per_drive"]["coax"]["rows"]:
        assert r["h1_verdict_resolved"] is True
        assert "H1 KILLED" in r["h1_flux_verdict"]
    assert isinstance(w["W4_flux"], dict)
    lines = W.format_tables(w)
    assert any(line.startswith("=== W1") for line in lines)
    assert any("H1 sign" in line for line in lines)
    assert any("H1 discriminator" in line for line in lines)
    assert any("passivity check" in line for line in lines)

    payload = {k: np.asarray(v) for k, v in d.items() if k != "flux_by_drive"}
    for drive, spectra in d["flux_by_drive"].items():
        for name, arr in spectra.items():
            payload[f"flux__{drive}__{name}"] = np.asarray(arr)
    path = tmp_path / "synthetic.ladders.npz"
    np.savez(path, **payload)
    back = W.load_npz(path)
    w2 = W.compute_witnesses(back)
    assert (w2["W1_h1_sign_witness"][0]["coax_ladder_msl_drive_mean_slope"]
            == w["W1_h1_sign_witness"][0]["coax_ladder_msl_drive_mean_slope"])
    assert set(back["flux_by_drive"]) == {"coax", "msl"}


def test_compute_witnesses_without_ladders_still_reports_w4(tmp_path):
    """The path the driver takes when the production return_ladder_voltages
    keyword is not on the checkout: W1-W3 SKIPPED, W4 + counterfactual live."""
    d = _synthetic_dump()
    d["coax_ladder_v"] = None
    d["msl_ladder_v"] = None
    w = W.compute_witnesses(d)
    assert w["W1_W2_W3"].startswith("SKIPPED")
    assert isinstance(w["W4_flux"], dict)
    assert "label_swap_counterfactual" in w
    assert any("SKIPPED" in line for line in W.format_tables(w))


# ---------------------------------------------------------------------------
# the attempt-3 flux box the driver passes with --flux
# ---------------------------------------------------------------------------
def test_attempt3_flux_entries_are_six_consistent_faces_of_one_box():
    entries = {m.name: m for m in _attempt3_scratch_flux_entries()}
    assert set(entries) == set(W.FLUX_BOX_FACES) | {"msl_x20_full"}

    (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = FLUX_BOX_X_3, FLUX_BOX_Y_3, FLUX_BOX_Z_3
    assert z_lo == N_GND * DX                     # the ground plane closes the box below
    assert entries["xlo_x05"].coordinate == x_lo
    assert entries["msl_x20"].coordinate == x_hi
    assert entries["ylo_y03"].coordinate == y_lo
    assert entries["yhi_y31"].coordinate == y_hi
    assert entries["top_z36"].coordinate == z_hi
    assert entries["coax_z22"].coordinate == FLUX_COAX_Z_3

    def extent(name, axis):
        m = entries[name]
        # tangential axes: x-normal -> (y, z), y-normal -> (x, z), z-normal -> (x, y)
        order = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[m.axis]
        i = order.index(axis)
        return (m.center[i] - m.size[i] / 2.0, m.center[i] + m.size[i] / 2.0)

    for name in ("msl_x20", "xlo_x05"):
        assert np.allclose(extent(name, "y"), (y_lo, y_hi))
        assert np.allclose(extent(name, "z"), (z_lo, z_hi))
    for name in ("ylo_y03", "yhi_y31"):
        assert np.allclose(extent(name, "x"), (x_lo, x_hi))
        assert np.allclose(extent(name, "z"), (z_lo, z_hi))
    assert np.allclose(extent("top_z36", "x"), (x_lo, x_hi))
    assert np.allclose(extent("top_z36", "y"), (y_lo, y_hi))

    # bottom face: inside the coax, strictly between the top coax probe (1.9 mm)
    # and the junction node (2.5 mm), and wide enough to cover the whole shell.
    assert Z_COAX[-1] < FLUX_COAX_Z_3 < N_GND * DX
    assert np.allclose(extent("coax_z22", "x"),
                       (JUNCTION_X - FLUX_COAX_PATCH_3 / 2, JUNCTION_X + FLUX_COAX_PATCH_3 / 2))
    assert np.allclose(extent("coax_z22", "y"),
                       (Y_C - FLUX_COAX_PATCH_3 / 2, Y_C + FLUX_COAX_PATCH_3 / 2))

    # the comparator is the SAME plane, full domain, and is not a box face
    assert entries["msl_x20_full"].size is None
    assert entries["msl_x20_full"].coordinate == entries["msl_x20"].coordinate
    assert "msl_x20_full" not in W.FLUX_CLOSURE_SIGNS

    # every face inside the attempt-3 domain
    for m in entries.values():
        limit = {"x": LX_2, "y": LY, "z": LZ_2}[m.axis]
        assert 0.0 < m.coordinate < limit, m.name


def test_h7_impact_rule_is_not_a_re_gamma_threshold():
    """The impact rule and the review-REJECTED ``Re(gamma) < 3 /m`` rule are
    numerically near-identical on the coax ladder itself -- ``2 alpha L`` with
    L = 1.6 mm (ladder centroid 1.4 mm -> reference plane 2.5 mm, plus the
    half-span) is 1% exactly at alpha ~ 3 /m -- which is precisely why the
    threshold cannot be transplanted: the SAME alpha on a ladder with a longer
    referral (the MSL geometry, centroid 6.6 mm -> reference plane 1.0 mm) is a
    2%+ bias on |Gamma| while ``Re(gamma) < 3`` would call it immaterial. The
    witness therefore reports the IMPACT, and this test pins that they differ.
    """
    alpha = 2.0                                  # below the rejected 3 /m threshold
    v = _plant(X_MSL, REF_MSL, alpha + 1j * BETA_MSL, "-axis",
               np.ones(3), 0.5 * np.ones(3))
    rows = W.w3_subsets(v, X_MSL, ref=REF_MSL, junction_side="below",
                        beta_analytic=BETA_MSL, freqs=FREQS, subsets=(), label="msl")
    assert all(abs(r["gamma_re"] - alpha) < 0.05 for r in rows), rows
    impact = W.h7_impact_from_w3(rows, FREQS)
    assert all(r["worst_alpha0_rel_shift"] > W.H7_ALPHA_IMPACT_MAX for r in impact), impact
    assert not any(r["h7_killed_at_bin"] for r in impact)


def test_h7_impact_on_the_coax_ladder_is_noise_limited_and_says_so():
    """MEASURED, report-only: at the coax ladder's own measured fit residual
    (~2e-4 on the settled attempt-3 run) the alpha=0 refit shift is already of
    the same order as the 1% criterion, and the 4-probe subsets are worse than
    the full 6-probe ladder. The H7 verdict must therefore be read next to the
    printed shift, not as a bare boolean. Values here (seed 0, 1e-4 relative
    complex noise, planted lossless line, |Gamma| = 0.93):

        full 6-probe ladder : shift 0.515% / 1.130% / 0.055%  (6/8/10 GHz)
        4-probe subsets     : shift 4.754% / 1.358% / 0.575%
    """
    rng = np.random.default_rng(0)
    v = _plant(Z_COAX, REF_COAX, 1j * BETA_COAX, "+axis", np.ones(3), 0.93 * np.ones(3))
    v = v * (1.0 + 1e-4 * (rng.standard_normal(v.shape)
                           + 1j * rng.standard_normal(v.shape)))
    full = W.h7_impact_from_w3(
        W.w3_subsets(v, Z_COAX, ref=REF_COAX, junction_side="above",
                     beta_analytic=BETA_COAX, freqs=FREQS, subsets=(), label="coax"),
        FREQS)
    subs = W.h7_impact_from_w3(
        W.w3_subsets(v, Z_COAX, ref=REF_COAX, junction_side="above",
                     beta_analytic=BETA_COAX, freqs=FREQS,
                     subsets=W.COAX_SUBSETS, label="coax"),
        FREQS)
    full_shift = [r["worst_alpha0_rel_shift"] for r in full]
    sub_shift = [r["worst_alpha0_rel_shift"] for r in subs]
    assert np.allclose(full_shift, [0.00515, 0.01130, 0.00055], atol=5e-5), full_shift
    assert np.allclose(sub_shift, [0.04754, 0.01358, 0.00575], atol=5e-5), sub_shift
    assert all(s >= f for s, f in zip(sub_shift, full_shift))
    assert max(full_shift) > W.H7_ALPHA_IMPACT_MAX / 2.0    # same order as the criterion
