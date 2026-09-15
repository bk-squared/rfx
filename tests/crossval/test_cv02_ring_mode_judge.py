"""Falsifiers for the cv02 ring-resonator judge (issue #812, Phase 1).

Two-sided, because the audit challenged the *instrument* and not the physics:

* **(A)** the case still passes on today's rfx output — the modes one
  unmodified run of ``02_ring_resonator.py`` produced on 2026-08-31, judged
  against the Meep tutorial's own published harminv output for this geometry;
* **(B)** the new judge FAILS on each defect the shipped judge was measured
  blind to — a missing mode, a mode displaced past the matcher window, and a
  wrong Q — every one of which the shipped judge is shown here to PASS.

Pre-declaration:
``docs/design_notes/20260831_cv02_ring_judge_predeclaration.md``.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib.util
import io
import json
import math
import re
import sys
import tokenize
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
JUDGE_PATH = REPO_ROOT / "validation/crossval/comparators/ring_mode_judge.py"
TRIALS_PATH = REPO_ROOT / "scripts/diagnostics/cv02_judge_tautology_trials.py"
TRIALS_JSON = REPO_ROOT / "tests/fixtures/cv02_ring_judge/tautology_trials_200k.json"
LADDER_JSON = REPO_ROOT / "tests/fixtures/cv02_ring_judge/harminv_decimation_ladder.json"

#: Ceiling for the LIVE decimation re-measurement (relative Q error). Not a
#: physics gate: it separates "orders of magnitude below the withdrawn claim"
#: from "anywhere near it", loosely enough that a different LAPACK cannot red
#: it. Committed ladder values are ~1e-12 to ~1e-13; the smallest figure the
#: withdrawn claim asserted is 2.4e-3.
LIVE_ERROR_CEILING = 1e-8
SCRIPT_PATH = REPO_ROOT / "validation/crossval/02_ring_resonator.py"
PREDECLARATION_PATH = (
    REPO_ROOT / "docs/design_notes/20260831_cv02_ring_judge_predeclaration.md")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rmj = _load("cv02_ring_mode_judge", JUDGE_PATH)
trials_mod = _load("cv02_judge_tautology_trials", TRIALS_PATH)

# --- the case's own numbers -------------------------------------------------

# Meep "Modes of a Ring Resonator" tutorial, published harminv output for the
# geometry this case builds (n=3.4, r=1, w=1, resolution=10, fcen=0.15,
# df=0.1). This is the reference the case is DEFINED against; it is quoted
# here, not measured here -- Meep is not installed on the lane host.
MEEP_REFERENCE = [
    rmj.ReferenceMode(0.118101575043663, 80.683059081382),
    rmj.ReferenceMode(0.147162555528154, 316.29272471914),
    rmj.ReferenceMode(0.175246750722663, 1677.48461212767),
]

# One unmodified run of validation/crossval/02_ring_resonator.py, 2026-08-31,
# commit 649b2cf, this host, printed in harminv-amplitude order.
RFX_TODAY = [
    rmj.SolverMode(0.147213, 357.6, 9.475536e-08),
    rmj.SolverMode(0.175298, 1864.1, 2.396352e-08),
    rmj.SolverMode(0.118068, 86.5, 6.126424e-09),
]

# Printed by the same run: "rfx harminv record T = 291.0 (Meep units)".
RECORD_T = 291.0
F_MIN, F_MAX = 0.10, 0.20


def _judge(rfx_modes, reference=None, record_T=RECORD_T):
    return rmj.judge(list(reference or MEEP_REFERENCE), list(rfx_modes),
                     record_T, f_min=F_MIN, f_max=F_MAX)


def _legacy(rfx_modes, reference=None):
    reference = list(reference or MEEP_REFERENCE)
    return rmj.legacy_shipped_judge([m.freq for m in reference],
                                    [m.Q for m in reference],
                                    [m.freq for m in rfx_modes])


# --- the defect the audit measured: the gate is its own matcher -------------


def test_shipped_headline_gate_is_entailed_by_its_matcher() -> None:
    """`mean_err < 5%` cannot fail: it averages the very quantity the matcher
    thresholded at 5%.

    The algebraic statement, checked by construction below: every element of
    ``errs`` is ``best_diff * 100`` for a pair the matcher admitted only when
    ``best_diff < 0.05``, so ``max(errs) < 5`` and hence ``mean(errs) < 5``
    for EVERY possible input. #812 measured this over 200,000 random trials
    (max ``mean_err`` 4.9997%, zero failures).
    """
    rng = np.random.default_rng(4242)
    worst = 0.0
    for _ in range(3000):
        n_ref = int(rng.integers(2, 6))
        ref_f = np.sort(rng.uniform(F_MIN, F_MAX, n_ref))
        ref_Q = rng.uniform(10.0, 5000.0, n_ref)
        rfx_f = rng.uniform(F_MIN, F_MAX, int(rng.integers(1, 9)))
        _, mean_err, _ = rmj.legacy_shipped_judge(ref_f, ref_Q, rfx_f)
        if mean_err is not None:
            worst = max(worst, mean_err)
            assert mean_err < 5.0
    # The matcher window is reachable, so the bound is tight, not vacuous.
    assert worst > 4.0, worst


def test_regated_judge_breaks_the_entailment_on_the_same_trial_stream() -> None:
    """Same generator, both judges. The shipped judge cannot fail; the new one
    fails, and its mean error is no longer bounded by 5%."""
    rng = np.random.default_rng(trials_mod.__dict__.get("SEED", 812))
    legacy_failures = new_failures = new_mean_ge_5 = 0
    legacy_max_mean = new_max_mean = 0.0
    clean = clean_new_failures = 0

    for _ in range(20_000):
        ref_f, ref_Q, rfx_f, rfx_Q, kinds = trials_mod.draw_trial(rng)
        passed, mean_err, _ = rmj.legacy_shipped_judge(ref_f, ref_Q, rfx_f)
        legacy_failures += int(not passed)
        if mean_err is not None:
            legacy_max_mean = max(legacy_max_mean, mean_err)

        verdict = _judge(
            [rmj.SolverMode(float(f), float(q)) for f, q in zip(rfx_f, rfx_Q)],
            reference=[rmj.ReferenceMode(float(f), float(q))
                       for f, q in zip(ref_f, ref_Q)],
            record_T=trials_mod.RECORD_T,
        )
        new_failures += int(not verdict.passed)
        if verdict.mean_err_pct is not None:
            new_max_mean = max(new_max_mean, verdict.mean_err_pct)
            new_mean_ge_5 += int(verdict.mean_err_pct >= 5.0)
        if np.all(kinds == "inside"):
            clean += 1
            clean_new_failures += int(not verdict.passed)

    # Shipped judge: the tautology, reproduced.
    assert legacy_failures == 0
    assert legacy_max_mean < 5.0

    # New judge: the entailment is gone in both senses -- the metric exceeds
    # 5% and the verdict fails.
    assert new_max_mean > 5.0
    assert new_mean_ge_5 > 0
    assert new_failures > 0.25 * 20_000

    # ...and it does not fail on defect-free trials, which is criterion (A)
    # restated as a property rather than a single measurement.
    assert clean > 1000
    assert clean_new_failures == 0


# --- criterion (A): today's rfx output still passes -------------------------


def test_criterion_a_todays_rfx_modes_pass_every_new_gate() -> None:
    verdict = _judge(RFX_TODAY)
    assert verdict.passed, verdict.gates
    assert verdict.gates == {
        "unmatched": True, "count": True,
        "mean_err": True, "max_err": True, "q": True,
    }
    assert verdict.n_matched == 3
    assert verdict.n_unmatched == 0
    assert verdict.surplus == []
    # Measured 2026-08-31: mean 0.031%, max 0.034% against a 5% gate.
    assert verdict.mean_err_pct == pytest.approx(0.031, abs=0.005)
    assert verdict.max_err_pct == pytest.approx(0.034, abs=0.005)
    assert verdict.max_err_pct < 0.05 * rmj.FREQ_TOL_PCT   # 150x of margin


def test_criterion_a_q_gate_exclusion_follows_the_record_not_the_answer() -> None:
    """Modes 1 and 2 are Q-gated and pass; mode 3's decay is not in the record.

    T/tau is computed from the REFERENCE Q and the record length only. #812
    published 0.376 / 0.086 using rfx's own Q; with the reference Q the same
    two modes read 0.426 / 0.096 -- the same side of the 1/4 cut, so the
    published exclusion survives the substitution.
    """
    verdict = _judge(RFX_TODAY)
    gated = [row for row in verdict.rows if row.q_gated]
    ungated = [row for row in verdict.rows if not row.q_gated]
    assert [round(row.ref_freq, 3) for row in gated] == [0.118, 0.147]
    assert [round(row.ref_freq, 3) for row in ungated] == [0.175]

    t_over_tau = {round(row.ref_freq, 3): row.t_over_tau
                  for row in verdict.rows}
    assert t_over_tau[0.118] == pytest.approx(1.339, abs=0.005)
    assert t_over_tau[0.147] == pytest.approx(0.426, abs=0.005)
    assert t_over_tau[0.175] == pytest.approx(0.096, abs=0.005)

    windows = {round(row.ref_freq, 3): row.q_window for row in verdict.rows}
    assert windows[0.118] == pytest.approx(0.747, abs=0.005)
    assert windows[0.147] == pytest.approx(2.350, abs=0.005)
    assert all(row.q_pass for row in gated)


# --- criterion (B): defects the shipped judge passes ------------------------


def test_b1_missing_mode_passes_the_shipped_judge_and_fails_the_new_one() -> None:
    """rfx never finds the 0.175 mode. Shipped: `matched = 2 >= 2`, mean over
    the survivors 0.03%, PASS. New: that reference mode is UNMATCHED."""
    defect = [m for m in RFX_TODAY if m.freq < 0.17]
    legacy_passed, legacy_mean, legacy_n = _legacy(defect)
    assert legacy_passed is True
    assert legacy_n == 2
    assert legacy_mean < 0.05

    verdict = _judge(defect)
    assert not verdict.passed
    assert verdict.gates["unmatched"] is False
    assert verdict.n_unmatched == 1
    assert [row.ref_freq for row in verdict.rows if not row.matched] == [
        pytest.approx(0.175246750722663)
    ]


def test_b2_displaced_mode_passes_the_shipped_judge_and_fails_the_new_one() -> None:
    """One mode 20% off -- a real, large frequency error.

    The shipped matcher deletes it (0.20 > 0.05 window), so the reported
    ``mean_err`` is computed over the two GOOD modes only: the judge announces
    a 0.03% mean frequency error for a solver that put one of its three modes
    20% away, and exits 0.
    """
    defect = [
        rmj.SolverMode(0.147213, 357.6),
        rmj.SolverMode(0.175246750722663 * 1.20, 1864.1),
        rmj.SolverMode(0.118068, 86.5),
    ]
    legacy_passed, legacy_mean, legacy_n = _legacy(defect)
    assert legacy_passed is True
    assert legacy_n == 2
    assert legacy_mean < 0.05        # reports 0.03% while a mode is 20% off

    # Deleting the WORST-matched mode makes the shipped score strictly better
    # than the defect-free run's: the metric moves the wrong way with the
    # defect, which is what "cannot fail for its stated reason" costs.
    worst_displaced = [
        rmj.SolverMode(0.147162555528154 * 1.20, 357.6),
        rmj.SolverMode(0.175298, 1864.1),
        rmj.SolverMode(0.118068, 86.5),
    ]
    assert _legacy(worst_displaced)[0] is True
    assert _legacy(worst_displaced)[1] < _legacy(RFX_TODAY)[1]
    assert not _judge(worst_displaced).passed

    # +20% puts the mode outside the harminv search band entirely, so the new
    # judge catches it as a missing reference mode.
    assert not _judge(defect).passed

    # In-band displacement is the sharper case: the mode is ASSIGNED rather
    # than hidden, and fails on its own error instead of vanishing from the
    # average.
    inband = [
        rmj.SolverMode(0.147213, 357.6),
        rmj.SolverMode(0.175246750722663 * 0.88, 1864.1),
        rmj.SolverMode(0.118068, 86.5),
    ]
    assert _legacy(inband)[0] is True and _legacy(inband)[2] == 2
    verdict = _judge(inband)
    assert not verdict.passed
    assert verdict.gates["unmatched"] is True      # it is assigned, not hidden
    assert verdict.gates["max_err"] is False
    assert verdict.max_err_pct == pytest.approx(12.0, abs=0.1)
    # ...and the MEAN still passes at 4.0%: one mode 12% wrong averaged against
    # two exact ones is inside the published 5% budget. That null space is why
    # G4 applies the same published number per mode -- with G3 alone this
    # defect would have survived the decoupling.
    assert verdict.gates["mean_err"] is True
    assert verdict.mean_err_pct == pytest.approx(4.02, abs=0.05)


@pytest.mark.parametrize("shift_pct", [5.5, 8.0, 12.0, 20.0, 40.0])
def test_b2_scan_every_displacement_past_the_window_is_now_caught(
    shift_pct: float,
) -> None:
    """The blind spot is the whole half-line beyond the matcher window, not one
    point in it."""
    defect = [
        rmj.SolverMode(0.147213, 357.6),
        rmj.SolverMode(0.175246750722663 * (1.0 + shift_pct / 100.0), 1864.1),
        rmj.SolverMode(0.118068, 86.5),
    ]
    assert _legacy(defect)[0] is True
    assert not _judge(defect).passed


def test_b3_wrong_q_on_the_gated_mode_passes_the_shipped_judge_and_fails() -> None:
    """Mode 2's Q is five times too low -- the size of radiation-loss error a
    curved boundary's discretization can produce. (Which discretization detail
    produces the REAL cv02 gap is unresolved; this one is injected, so nothing
    here rests on that.) The shipped judge gates no Q at all; the corrected
    transformed interval rejects this low-Q side."""
    defect = [
        rmj.SolverMode(0.147213, 357.6 / 5.0),
        rmj.SolverMode(0.175298, 1864.1),
        rmj.SolverMode(0.118068, 86.5),
    ]
    assert _legacy(defect)[0] is True            # Q never enters the old gate

    verdict = _judge(defect)
    assert not verdict.passed
    assert verdict.gates["q"] is False
    assert verdict.gates["mean_err"] is True     # frequencies are untouched
    row = next(r for r in verdict.rows if round(r.ref_freq, 3) == 0.147)
    assert row.q_pass is False


def test_b3_the_q_gate_is_two_sided() -> None:
    """An over-damped mode (Q too LOW) fails too, so a leaky boundary is caught
    as well as a lossless one.

    Two-sided is NOT symmetric. #945 replaced the symmetric ``+-log1p(s)``
    window with the transformed rate interval, whose sides differ and whose
    high-Q side is unbounded for ``s >= 1``
    (:func:`rate_interval_to_log_q_bounds`). The low-Q side stays finite at
    every ``s``, which is why this direction is gated at all; the earlier
    wording here claimed symmetry and was wrong after #999.
    """
    defect = [
        rmj.SolverMode(0.147213, 357.6 / 5.0),
        rmj.SolverMode(0.175298, 1864.1),
        rmj.SolverMode(0.118068, 86.5),
    ]
    assert _legacy(defect)[0] is True
    assert _judge(defect).gates["q"] is False


def test_b3_the_ungated_mode_is_honestly_ungated() -> None:
    """Mode 3's Q is NOT gated -- deliberately. A gate there would measure the
    run length (T/tau = 0.096), which is the failure #812 warned about."""
    defect = [
        rmj.SolverMode(0.147213, 357.6),
        rmj.SolverMode(0.175298, 1864.1 * 100.0),
        rmj.SolverMode(0.118068, 86.5),
    ]
    verdict = _judge(defect)
    assert verdict.gates["q"] is True
    assert verdict.passed
    row = next(r for r in verdict.rows if round(r.ref_freq, 3) == 0.175)
    assert row.q_gated is False and row.q_pass is None


# --- structural properties of the new judge ---------------------------------


def test_the_assignment_contains_no_tolerance() -> None:
    """The decoupling itself: a mode 30% away is still ASSIGNED (and then
    fails on error), where the shipped matcher would have dropped it."""
    ref = [rmj.ReferenceMode(0.15, 300.0)]
    assert rmj.assign([m.freq for m in ref], [0.195]) == [0]
    assert rmj.assign([m.freq for m in ref], []) == [None]
    # Two references, one rfx mode: the assignment serves the closer one and
    # reports the other as unmatched rather than shrinking the comparison.
    assert rmj.assign([0.12, 0.18], [0.1805]) == [None, 0]


def test_q_window_is_tau_ref_over_record_length() -> None:
    """``s = tau_ref/T``, with ``tau = Q_ref/(pi f_ref)``. No fitted constant.

    The name says the formula and nothing more, deliberately. This test used
    to be called ``..._is_the_record_length_resolution_limit``; #907 measured
    that reading false (a damped exponential has no ``1/T`` separation limit
    to import), so the claim is withdrawn here as well as in the module
    docstring, the report header, the artifact note and the pre-declaration.
    What survives is the algebra: no constant in this function was fitted to
    the board it judges.
    """
    for freq, q in [(0.118101575043663, 80.683059081382),
                    (0.147162555528154, 316.29272471914),
                    (0.175246750722663, 1677.48461212767)]:
        tau = q / (math.pi * freq)
        t_over_tau, window = rmj.q_window(freq, q, RECORD_T)
        assert t_over_tau == pytest.approx(RECORD_T / tau, rel=1e-12)
        assert window == pytest.approx(tau / RECORD_T, rel=1e-12)
    # A longer record buys a tighter window, linearly. That 1/T scaling is
    # what makes the window shrink faster than the rfx-vs-Meep Q gap does --
    # the "Known limitation" in q_window, i.e. a defect, NOT the virtue this
    # comment used to claim. Pinned here so the scaling cannot change silently.
    _, w1 = rmj.q_window(0.147162555528154, 316.29272471914, RECORD_T)
    _, w2 = rmj.q_window(0.147162555528154, 316.29272471914, 4 * RECORD_T)
    assert w2 == pytest.approx(w1 / 4.0, rel=1e-12)


def test_q_rate_interval_maps_to_asymmetric_q_bounds() -> None:
    """The inverse decay-rate map has a finite lower and one-sided upper
    bound; the old symmetric ``log1p(s)`` gate rejected the latter too soon.
    """
    record = 20.0 / math.pi  # tau = 10/pi, hence s = tau/T = 0.5
    lower, upper = rmj.q_log_bounds(1.0, 10.0, record)
    assert lower == pytest.approx(-math.log(1.5))
    assert upper == pytest.approx(-math.log(0.5))
    assert lower < 0.0 < upper

    # Ratios infinitesimally inside either transformed endpoint are admissible
    # (using an interior epsilon avoids making the test depend on libm's last
    # bit when the endpoint is reconstructed through a logarithm).
    for ratio in ((1.0 / 1.5) * (1.0 + 1e-9), 2.0 * (1.0 - 1e-9)):
        verdict = rmj.judge(
            [rmj.ReferenceMode(1.0, 10.0)],
            [rmj.SolverMode(1.0, 10.0 * ratio)],
            record,
            f_min=0.5,
            f_max=1.5,
        )
        assert verdict.gates["q"] is True

    # At s >= 1, the rate interval reaches zero and the high-Q side is
    # unbounded; the low-Q side remains bounded by the positive-rate edge.
    record = 5.0 / math.pi  # tau = 10/pi, hence s = 2
    lower, upper = rmj.q_log_bounds(1.0, 10.0, record)
    assert lower == pytest.approx(-math.log(3.0))
    assert upper == math.inf
    high_q = rmj.judge(
        [rmj.ReferenceMode(1.0, 10.0)],
        [rmj.SolverMode(1.0, 1.0e9)],
        record,
        f_min=0.5,
        f_max=1.5,
    )
    low_q = rmj.judge(
        [rmj.ReferenceMode(1.0, 10.0)],
        [rmj.SolverMode(1.0, 2.0)],
        record,
        f_min=0.5,
        f_max=1.5,
    )
    assert high_q.gates["q"] is True
    assert low_q.gates["q"] is False

# --- #945: the transform, as a pure function on synthetic intervals ---------
#
# ``rate_interval_to_log_q_bounds`` takes the declared rate scale ``s`` and
# this pair's frequency shift ``log_freq_ratio = ln(f_rfx/f_ref)`` -- no mode
# list, no record, no solver -- so both arguments can be exercised directly.
# The tests below drive the shift at its default ``0.0``, i.e. the pure
# fixed-frequency image, which is the whole of the FIRST #945 claim: that
# image of ``[1-s, 1+s]`` under ``Q = pi f / alpha`` is ``[1/(1+s), 1/(1-s)]``,
# asymmetric for every ``s > 0`` and unbounded above at ``s >= 1``. The
# frequency shift -- the second #945 claim -- is driven by the
# ``test_issue945_*`` block further down.


@pytest.mark.parametrize("s", [1e-12, 1e-6, 0.01, 0.1, 0.25, 0.5, 0.798,
                               0.9, 0.999, 0.999999])
def test_rate_bounds_invert_the_rate_interval_exactly(s: float) -> None:
    """Round trip: exponentiating each log-Q bound returns the rate edge that
    produced it. ``exp(-lower) = 1+s`` and ``exp(-upper) = 1-s``."""
    lower, upper = rmj.rate_interval_to_log_q_bounds(s)
    assert math.exp(-lower) == pytest.approx(1.0 + s, rel=1e-12)
    assert math.exp(-upper) == pytest.approx(1.0 - s, rel=1e-9, abs=1e-15)
    # the interval straddles equality, and the upper side is the wider one
    assert lower < 0.0 < upper
    assert upper > -lower


@pytest.mark.parametrize("s,q_ratio_upper", [(0.25, 4.0 / 3.0),
                                             (0.5, 2.0),
                                             (0.798, 1.0 / 0.202)])
def test_rate_bounds_upper_side_is_one_over_one_minus_s(s, q_ratio_upper
                                                        ) -> None:
    """The #945 table, in the shipped function. The symmetric window this
    replaced capped the high-Q side at ``1+s``; the transform's own upper edge
    is ``1/(1-s)``, larger for every ``s > 0`` because ``(1+s)(1-s) < 1``."""
    _lower, upper = rmj.rate_interval_to_log_q_bounds(s)
    assert math.exp(upper) == pytest.approx(q_ratio_upper, rel=1e-12)
    old_symmetric_cap = math.log1p(s)
    assert upper > old_symmetric_cap
    assert math.exp(upper) > (1.0 + s)


def test_rate_bounds_lower_side_matches_the_old_symmetric_window() -> None:
    """The defect was one-sided. The rfx-too-lossy edge is unchanged, so this
    fix cannot mask a Q that is too LOW -- criterion (B) of the #812 re-gate
    (a wrong Q must still fail) keeps its low-side detection intact."""
    for s in (0.05, 0.25, 0.5, 0.798, 2.0, 4.0):
        lower, _upper = rmj.rate_interval_to_log_q_bounds(s)
        assert lower == pytest.approx(-math.log1p(s), rel=1e-15)


def test_rate_bounds_upper_side_diverges_as_s_approaches_one() -> None:
    """The s -> 1 limit. The admissible rate interval's lower edge ``1-s``
    reaches zero, so an arbitrarily small decay rate -- an arbitrarily large Q
    -- is admitted, and the upper log bound grows without limit."""
    # binary eps so that 1-(1-eps) is eps exactly and the expected value is
    # not itself a rounding of a decimal literal
    eps_values = tuple(2.0 ** -k for k in (7, 14, 26, 40, 52))
    uppers = [rmj.rate_interval_to_log_q_bounds(1.0 - eps)[1]
              for eps in eps_values]
    # strictly increasing, and each step is the exact -log(eps)
    assert uppers == sorted(uppers)
    for eps, upper in zip(eps_values, uppers):
        assert upper == pytest.approx(-math.log(eps), rel=1e-12)
    assert uppers[-1] > 36.0
    # and the limit itself is attained, not approached: exactly at s = 1 the
    # bound is +inf, so the function is continuous from below into infinity.
    assert rmj.rate_interval_to_log_q_bounds(1.0)[1] == math.inf
    assert uppers[-1] < rmj.rate_interval_to_log_q_bounds(1.0)[1]
    # the low-Q side stays finite there -- only one side is unbounded
    assert rmj.rate_interval_to_log_q_bounds(1.0)[0] == pytest.approx(
        -math.log(2.0))


@pytest.mark.parametrize("s", [1.0, 1.0 + 1e-12, 2.0, 2.8295108704304397,
                               4.0, 1e6])
def test_rate_bounds_have_no_finite_upper_side_above_s_one(s: float) -> None:
    """``s >= 1`` is cv02's operating regime, not an edge case: the committed
    live board's mode 2 runs at ``s = 2.8295``. The stated policy admits any
    positive rate below ``alpha_ref``, so there is no finite Q ceiling to
    impose; the low-Q side stays bounded by the positive-rate edge ``1+s``."""
    lower, upper = rmj.rate_interval_to_log_q_bounds(s)
    assert upper == math.inf
    assert lower == pytest.approx(-math.log1p(s), rel=1e-15)
    assert math.isfinite(lower)


def test_rate_bounds_degenerate_and_invalid_inputs() -> None:
    """``s = 0`` is a point interval (equality only); ``s = inf`` is no record
    and therefore no restriction; a negative or NaN scale is a caller bug and
    is refused rather than absolute-valued into a plausible answer."""
    assert rmj.rate_interval_to_log_q_bounds(0.0) == (0.0, 0.0)
    assert rmj.rate_interval_to_log_q_bounds(math.inf) == (-math.inf, math.inf)
    for bad in (-1e-12, -1.0, float("nan")):
        with pytest.raises(ValueError):
            rmj.rate_interval_to_log_q_bounds(bad)


def test_q_log_bounds_composes_the_policy_scale_with_the_transform() -> None:
    """The per-mode entry point is exactly ``q_window`` feeding the pure
    transform -- no second, divergent copy of the algebra."""
    for freq, q, record in [(0.118, 80.7, 291.0), (0.147, 316.3, 291.0),
                            (0.175, 1677.5, 3385.0), (1.0, 10.0, 0.0)]:
        _t_over_tau, window = rmj.q_window(freq, q, record)
        assert (rmj.q_log_bounds(freq, q, record)
                == rmj.rate_interval_to_log_q_bounds(window))


def test_issue945_a_mode_on_the_declared_rate_bound_is_admitted() -> None:
    """The reproduction from #945, driven through ``judge``.

    ``f_ref = 1/pi``, ``Q_ref = 10``, ``T = 20`` gives ``tau = 10``,
    ``s = 0.5``, ``alpha_ref = 0.1`` and ``1/T = 0.05``. A mode whose decay
    rate differs by EXACTLY the tolerance the gate claims to allow
    (``alpha_rfx = 0.05``, hence ``Q_rfx = 20``) used to be REJECTED, because
    the symmetric window stopped at ``ln(1+s) = 0.405`` while the mode sits at
    ``ln 2 = 0.693``. Both rate edges are now admitted, and a mode past either
    edge is still rejected -- the gate did not become one-sided, it became the
    right two sides."""
    f_ref, q_ref, record = 1.0 / math.pi, 10.0, 20.0
    band = dict(f_min=0.2, f_max=0.5)
    alpha_ref = math.pi * f_ref / q_ref

    def _q_from_rate(alpha: float) -> float:
        return math.pi * f_ref / alpha

    def _gate(q_rfx: float) -> "rmj.Verdict":
        return rmj.judge([rmj.ReferenceMode(f_ref, q_ref)],
                         [rmj.SolverMode(f_ref, q_rfx)], record,
                         min_matched=1, **band)

    assert alpha_ref == pytest.approx(0.1)
    # the record resolves this mode, so the row really is Q-gated
    row = _gate(q_ref).rows[0]
    assert row.q_gated is True
    assert row.q_window == pytest.approx(0.5)

    slow = _q_from_rate(alpha_ref - 1.0 / record)          # Q = 20, the issue
    assert slow == 20.0
    assert _gate(slow).gates["q"] is True
    # what the old symmetric window said about the very same mode
    assert abs(math.log(slow / q_ref)) > math.log1p(0.5)

    # The mirror edge, on the rfx-too-lossy side, taken as Q_ref/(1+s) rather
    # than through alpha: the endpoints are reconstructed through a logarithm,
    # so re-deriving them by a different arithmetic route lands one ULP off
    # and a boundary row flips. That is the endpoint's numerical sensitivity,
    # not the gate's, and the interior checks below are what pin the gate.
    fast = q_ref / (1.0 + 0.5)
    assert _gate(fast).gates["q"] is True

    # a step outside either declared rate edge still fails
    assert _gate(slow * (1.0 + 1e-9)).gates["q"] is False
    assert _gate(fast * (1.0 - 1e-9)).gates["q"] is False
    # and comfortably inside, both sides pass
    assert _gate(slow * 0.99).gates["q"] is True
    assert _gate(fast * 1.01).gates["q"] is True


# --- #907: the three ingredients, named and printed -------------------------


def test_q_gate_ingredients_separate_derived_from_declared_policy() -> None:
    """#907's ask, in executable form: the estimator scale, the transform and
    the discretization budget are three separate quantities with three
    different epistemic statuses, and the module says which is which."""
    kinds = {item.name: item.kind for item in rmj.Q_GATE_INGREDIENTS}
    assert kinds == {
        "estimator_uncertainty": "declared-policy",
        "rate_to_q_transform": "derived",
        "discretization_budget": "absent",
    }
    for item in rmj.Q_GATE_INGREDIENTS:
        assert item.quantity and item.basis and item.source
    # the absent one must not be dressed up as a value
    budget = [i for i in rmj.Q_GATE_INGREDIENTS
              if i.name == "discretization_budget"][0]
    assert "none declared" in budget.quantity
    # and the consequence is stated, not left to the reader
    assert "consistency heuristic" in rmj.Q_GATE_CHARACTER
    assert "NOT a Q-accuracy guarantee" in rmj.Q_GATE_CHARACTER


def test_the_report_prints_which_ingredients_are_policy() -> None:
    """The crossval log is where a reader meets this gate. The split has to be
    in the printed report, not only in the source."""
    text = rmj.format_report(_judge(RFX_TODAY))
    assert "Q gate ingredients" in text
    for item in rmj.Q_GATE_INGREDIENTS:
        assert item.name in text
        assert f"[{item.kind:>15}]" in text
    assert "consistency heuristic" in text
    # the old claim that the window is not a chosen value is gone
    assert "not chosen values" not in text
    assert "DECLARED rate scale" in text


def test_q_window_docstring_withdraws_the_1_over_T_resolution_claim() -> None:
    """The docstring used to argue the window from a Fourier-style ``1/T``
    resolution limit. #907 measured that claim false for this estimator, so
    the withdrawal has to live where the number is produced."""
    doc = rmj.q_window.__doc__
    assert "DECLARED-POLICY" in doc
    assert "refuted" in doc
    assert "decimate=False" in doc
    assert "Known limitation" in doc          # the T-dependence still stands
    # and the derived half is not tarred with it
    assert "derived" in rmj.rate_interval_to_log_q_bounds.__doc__


def test_harminv_error_field_is_the_decay_restated() -> None:
    """Why ingredient 1 is a stated policy and not a fit residual (#907, item 1).

    The obvious alternative to a declared envelope is to take the estimator's
    own residual. rfx's harminv reports one, ``HarminvMode.error``, but it is
    not a residual: ``rfx/harminv.py`` computes
    ``err = 1 - min(|lam|, 1/|lam|)``, and for a decaying pole
    ``|lam| = exp(-alpha * dt_eff)``, so

        error  ==  1 - exp(-decay * dt_eff)

    exactly -- a strictly increasing function of the reported decay rate, i.e.
    of ``1/Q`` at fixed ``f``. It carries no information about how well the
    pole fits the record. A tolerance built from it would scale with the very
    quantity the Q gate judges, which is #812's self-referential gate class.

    Measured here rather than argued, on a clean single damped exponential at
    cv02's own step, on both harminv paths. On the decimated path the identity
    holds against the DECIMATED step, which is the same statement.
    """
    from rfx.harminv import harminv

    dt = 2.3350677933821873e-16           # cv02's step
    freq = 1.2e14
    n_samples = 1500                      # enough for the decimation to fire

    for decimate in (False, "auto"):
        for q_in in (80.0, 1700.0):
            alpha = math.pi * freq / q_in
            t = np.arange(n_samples) * dt
            signal = np.exp(-alpha * t) * np.cos(2 * math.pi * freq * t)
            modes = harminv(signal, dt, freq * 0.5, freq * 1.5,
                            decimate=decimate)
            assert modes, f"no modes at Q={q_in}, decimate={decimate!r}"
            mode = max(modes, key=lambda m: m.amplitude)
            # dt_eff is dt on the undecimated path and an integer multiple of
            # it otherwise; recover it from the identity rather than from the
            # decimation rule, then check it IS that integer multiple.
            dt_eff = -math.log1p(-mode.error) / mode.decay
            factor = dt_eff / dt
            assert factor == pytest.approx(round(factor), rel=1e-6)
            assert round(factor) >= 1
            if decimate is False:
                assert round(factor) == 1
            assert mode.error == pytest.approx(
                1.0 - math.exp(-mode.decay * dt_eff), rel=1e-12)
            # monotone in the decay, so it ranks modes by Q, not by fit
            # quality: the longer-lived mode reports the SMALLER "error"
            assert mode.error > 0.0

def test_the_decimation_penalty_claim_does_not_reproduce() -> None:
    """The second empirical prop under ingredient 1, withdrawn and pinned.

    ``q_window`` briefly read: "what DOES degrade at short records is the
    decimated path cv02 actually runs (3.49% there, 0.24% at the 0.25 cut)".
    Those figures came from a comment on #907 and were never pinned by any
    artifact, test or table in this repo. Re-measured on the configuration
    that sentence names, they do not reproduce, and at the shorter of the two
    rungs they cannot: no decimation stage fires there at all, so there is no
    decimated path to measure.

    Both halves are checked here -- the committed ladder
    (``tests/fixtures/cv02_ring_judge/harminv_decimation_ladder.json``,
    regenerate with ``scripts/diagnostics/cv02_harminv_decimation_ladder.py``)
    and a live re-measurement of the two rungs that carry the claim, so the
    artifact cannot rot into a number nobody re-runs.
    """
    from rfx.harminv import _decimation_plan, harminv

    doc = json.loads(LADDER_JSON.read_text(encoding="utf-8"))
    rows = {row["t_over_tau"]: row for row in doc["rows"]}
    claimed = doc["withdrawn_claim"]["asserted_relative_q_error"]
    assert set(claimed) == {"0.0822", "0.25"}

    # (a) the shorter rung: 'auto' chooses no decimation, in BOTH bands, so
    #     "the decimated path ... 3.49% there" has no path to be about.
    short = rows[0.0822]
    for band in short["bands"].values():
        assert band["decimation_fires"] is False
        assert band["decimation_factors"] == []
        assert band["dt_eff_over_dt"] == 1.0
        assert band["auto"]["relative_q_error"] == band["no_decimation"][
            "relative_q_error"]

    # (b) the 0.25 cut: decimation DOES fire, and the decimated path is not
    #     worse than the undecimated one -- the opposite of the claim's sign.
    cut = rows[0.25]
    for band in cut["bands"].values():
        assert band["decimation_fires"] is True
        assert band["dt_eff_over_dt"] > 1.0
        assert band["auto"]["relative_q_error"] <= band["no_decimation"][
            "relative_q_error"]

    # (c) every measured error on every rung is orders of magnitude below what
    #     was asserted. Stated as a ratio so the assertion says "not close".
    for key, asserted in claimed.items():
        row = rows[float(key)]
        for band in row["bands"].values():
            assert band["auto"]["relative_q_error"] * 1e6 < asserted

    # (d) both bands land on the SAME decimation plan on every rung. That is a
    #     plan-stability check, NOT an independent witness -- and the two are
    #     easy to confuse here, so: f_max enters harminv only through the
    #     decimation target and a pass-band filter, the two targets factor to
    #     one plan, and the two columns are then the same computation reporting
    #     the same error to every digit. Agreement by construction is not
    #     evidence. The independent witness is legs (a)/(b), auto vs
    #     decimate=False -- different sample counts, different pencil sizes.
    #     What (d) buys is that a plan differing between bands would make every
    #     other leg band-contingent; that is worth pinning, and it is all.
    for row in doc["rows"]:
        plans = {tuple(b["decimation_factors"]) for b in row["bands"].values()}
        assert len(plans) == 1, (row["t_over_tau"], plans)
        errors = {b["auto"]["relative_q_error"] for b in row["bands"].values()}
        assert len(errors) == 1, (row["t_over_tau"], errors)

    # (e) live, on the two rungs the claim named: the artifact is reproducible
    #     from today's rfx, not just a file somebody committed once.
    #
    #     The decimation PLAN is integer and deterministic, so it is compared
    #     exactly. The Q error is not: it is a ~1e-12 residue of an SVD, and a
    #     different LAPACK reds an exact comparison for no physical reason
    #     (the cross-machine-float class from the PR #119 slow-suite fixes).
    #     What the claim turns on is the ORDER, so that is what is asserted --
    #     LIVE_ERROR_CEILING sits four decades above anything measured here and
    #     three below the smallest figure the withdrawn claim asserted.
    dt = doc["signal"]["dt_s"]
    freq = doc["signal"]["freq_hz"]
    q_true = doc["signal"]["Q"]
    tau = doc["signal"]["tau_s"]
    band = doc["bands"]["withdrawn_claim_band"]
    for key in ("0.0822", "0.25"):
        t_over_tau = float(key)
        n_samples = int(round(t_over_tau * tau / dt))
        stored = rows[t_over_tau]["bands"]["withdrawn_claim_band"]
        assert n_samples == stored["n_samples"]
        factors, _ = _decimation_plan(n_samples, dt, band["f_max_hz"], "auto")
        assert list(factors) == stored["decimation_factors"]
        t = np.arange(n_samples) * dt
        signal = np.exp(-t / tau) * np.cos(2 * math.pi * freq * t)
        modes = harminv(signal, dt, band["f_min_hz"], band["f_max_hz"],
                        decimate="auto")
        assert modes
        best = min(modes, key=lambda m: abs(m.freq - freq))
        measured = abs(best.Q - q_true) / q_true
        assert measured < LIVE_ERROR_CEILING, (key, measured)
        assert stored["auto"]["relative_q_error"] < LIVE_ERROR_CEILING
        assert LIVE_ERROR_CEILING < claimed[key] / 1e3


def test_q_window_docstring_withdraws_the_decimation_penalty_claim() -> None:
    """The withdrawal has to live where the number was quoted, and the two
    figures must not survive anywhere in the cv02 surfaces."""
    doc = rmj.q_window.__doc__
    assert "WITHDRAWN" in doc
    assert "harminv_decimation_ladder.json" in doc
    assert "UNMEASURED" in doc
    basis = next(i.basis for i in rmj.Q_GATE_INGREDIENTS
                 if i.name == "estimator_uncertainty")
    # the prop is not cited as a motivation any more ("motivated by ... and by
    # the measured degradation ..."); it survives only as a named withdrawal
    assert "and by the measured degradation" not in basis
    assert "WITHDRAWN" in basis
    assert "UNMEASURED" in basis
    assert "harminv_decimation_ladder.json" in basis
    # the figures survive in exactly one place -- quoted inside the sentence
    # that withdraws them -- and nowhere on the script
    judge_source = JUDGE_PATH.read_text(encoding="utf-8")
    assert judge_source.count("3.49%") == 1
    assert "briefly read" in judge_source
    assert "3.49" not in SCRIPT_PATH.read_text(encoding="utf-8")


def _flatten_prose(text: str, *, python_source: bool = False) -> str:
    """The prose of ``text``, on one line.

    ``python_source=True`` keeps only the prose of the file, in source order:
    comments and triple-quoted strings. Every other token is dropped --
    identifiers, so a test's own name is not read as an assertion, and
    ordinary string literals, so the sentences pinned below (ordinary
    literals, every one of them) are not part of any surface they judge.

    Each comment token loses its OWN leading ``#`` before the join, so a claim
    re-wrapped across three comment lines reads as one sentence. Joining first
    and stripping afterwards left every hash but the first inline, which let a
    revert walk past this guard by moving one line break (review of
    2026-09-14). Quotes and rst/markdown emphasis go the same way, and
    whitespace collapses, so re-wrapping a surface is free and rewording it is
    not.
    """
    if python_source:
        def _prose(tok: tokenize.TokenInfo) -> str | None:
            if tok.type == tokenize.COMMENT:
                return tok.string.lstrip("#")
            if (tok.type == tokenize.STRING
                    and tok.string.lstrip("rbuRBUf").startswith(('"""', "'''"))):
                return tok.string
            return None

        text = " ".join(
            prose
            for prose in (_prose(tok) for tok
                          in tokenize.generate_tokens(io.StringIO(text).readline))
            if prose is not None)
    lines = [line.strip().lstrip("#").strip() for line in text.splitlines()]
    flat = " ".join(lines)
    for markup in ('"', "'", "`", "*"):
        flat = flat.replace(markup, " ")
    return re.sub(r"\s+", " ", flat).strip()


#: Sentence boundary on the flattened form. Decimals and version strings have
#: no space after the dot, so they do not split; ``e.g.`` splits, which only
#: makes the check below stricter.
_SENTENCE = re.compile(r"(?<=[.;:])\s+")

#: Key phrases of the mechanism #907 retracted. Spelled out rather than
#: assembled from fragments: ordinary string literals are not part of any
#: surface ``_flatten_prose`` produces, so this cannot match itself. If that
#: ever stopped being true these literals would appear as unpinned sentences
#: and the test would go red, which is the safe direction.
_RETRACTED_MECHANISM = ("staircas", "discretization offset", "subpixel")


def _mechanism_sentences(flat: str) -> tuple[str, ...]:
    """The sentences of ``flat`` that name the retracted mechanism."""
    return tuple(sentence for sentence in _SENTENCE.split(flat)
                 if any(key in sentence for key in _RETRACTED_MECHANISM))


#: Every sentence, on every cv02 surface, that is allowed to name the
#: mechanism #907 retracted -- flattened by ``_flatten_prose``. Adding a
#: sentence, deleting one, or changing the words of one reds the test below.
_MECHANISM_SENTENCES = {
    "ring_mode_judge.py": (
        "This paragraph used to assert a staircased ring boundary and"
        " subpixel treatment as the cause;",
    ),
    "02_ring_resonator.py": (
        "An earlier version of this comment called it a discretization"
        " offset ;",
    ),
    "predeclaration note": (
        "a staircased dielectric ring s radiation loss is precisely the"
        " quantity a curved boundary gets wrong, and no gate looks at it.",
    ),
    # the ingredient strings are persisted into every future crossval.json,
    # so the mechanism may not enter them at all.
    "persisted gate ingredients": (),
    # The second entry here belonged to
    # ``test_verdict_lane_q_gate_is_run_length_contingent``, deleted 2026-09-15
    # when #907 declared the envelope (see the comment block above that test's
    # replacement). The retraction it carried is unchanged and still stated on
    # ring_mode_judge.py and 02_ring_resonator.py; only its copy here is gone.
    "this test file": (
        "#907 (2026-09-10) retracted the attribution of the rfx-vs-Meep Q gap"
        " to the ring s staircased curved boundary and its subpixel"
        " treatment, as an overclaim:",
    ),
}


def test_the_staircasing_attribution_is_withdrawn_everywhere() -> None:
    """#907 (2026-09-10) retracted the attribution of the rfx-vs-Meep Q gap to
    the ring's staircased curved boundary and its subpixel treatment, as an
    overclaim: a counterexample moves Q while every frequency stays inside the
    two solvers' observed mutual agreement, so the frequency agreement cannot
    pin the geometry, and the log decomposition cannot settle it either.

    ONE rule, after three rounds in which a cleverer reading of the prose was
    each time walked past by a cleverer re-assertion (an exact-phrase check, a
    +/-6-line proximity window, then a sentence-order rule paired with a
    copula check): on each surface below, the sentences naming the retracted
    mechanism must be EXACTLY the ones pinned in ``_MECHANISM_SENTENCES``.
    Nothing is read for meaning.

    So this is a revert tripwire, not a proof that the prose is honest: it
    cannot judge a new sentence, only notice that one appeared. Editing a
    pinned sentence -- including editing it for the better -- reds this test
    on purpose, so the retraction is re-read before it moves. Re-wrapping is
    free, because each surface is flattened to one line first.
    """
    surfaces = {
        "ring_mode_judge.py": _flatten_prose(
            JUDGE_PATH.read_text(encoding="utf-8"), python_source=True),
        "02_ring_resonator.py": _flatten_prose(
            SCRIPT_PATH.read_text(encoding="utf-8"), python_source=True),
        "predeclaration note": _flatten_prose(
            PREDECLARATION_PATH.read_text(encoding="utf-8")),
        "persisted gate ingredients": _flatten_prose(
            " ".join([rmj.Q_GATE_CHARACTER]
                     + [str(value) for item in rmj.Q_GATE_INGREDIENTS
                        for value in dataclasses.asdict(item).values()])),
        "this test file": _flatten_prose(
            Path(__file__).read_text(encoding="utf-8"), python_source=True),
    }
    assert set(surfaces) == set(_MECHANISM_SENTENCES)
    for name, flat in surfaces.items():
        found = _mechanism_sentences(flat)
        pinned = _MECHANISM_SENTENCES[name]
        assert found == pinned, (
            name,
            "unpinned: %r" % ([s for s in found if s not in pinned],),
            "gone: %r" % ([s for s in pinned if s not in found],))
    # and the judge no longer prescribes the floor that was refused
    assert "encoding the expected" not in rmj.q_window.__doc__
    assert "certify the agreement" in rmj.q_window.__doc__


def test_gated_row_retains_the_signed_ratio_and_the_bounds_that_judged_it(
) -> None:
    """An asymmetric interval cannot be audited from an absolute ``|ln Q|``:
    the side decides. The row keeps the signed value and both bounds, so the
    retained artifact re-derives its own verdict without the judge.

    The bounds it keeps are this pair's own -- the transform evaluated at
    ``ln(f_rfx/f_ref)`` (#945, second correction), NOT the fixed-frequency
    image. Asserting the shifted pair is the point: with the unshifted one
    stored, a reader re-deriving the verdict from the row would get a
    different inequality from the one the judge evaluated."""
    verdict = _judge(RFX_TODAY)
    gated = verdict.q_gated_rows
    assert gated
    for row in gated:
        assert row.q_log_ratio == pytest.approx(abs(row.q_log_ratio_signed))
        lower, upper = rmj.q_log_bounds(row.ref_freq, row.ref_Q,
                                        verdict.record_length,
                                        rfx_freq=row.rfx_freq)
        assert row.q_log_lower == lower and row.q_log_upper == upper
        assert row.q_pass is (lower <= row.q_log_ratio_signed <= upper)
        # and the stored pair really is the shifted one -- non-vacuous only
        # because these rows have a nonzero frequency disagreement
        assert row.q_log_freq_term != 0.0
        unshifted_lower, _unshifted_upper = rmj.q_log_bounds(
            row.ref_freq, row.ref_Q, verdict.record_length)
        assert row.q_log_lower != unshifted_lower
        assert row.q_log_lower == pytest.approx(
            unshifted_lower + row.q_log_freq_term)
    # ungated rows carry no bounds at all rather than misleading defaults
    for row in verdict.rows:
        if not row.q_gated:
            assert row.q_log_ratio_signed is None
            assert row.q_log_lower is None and row.q_log_upper is None
            assert row.q_log_freq_term is None
            assert row.q_log_rate_ratio_signed is None


# --- #945 reopened: the frequency term the fixed-frequency inversion drops --
#
# ``rate_interval_to_log_q_bounds`` inverts the rate interval correctly AT ONE
# FREQUENCY. The two solvers report two. cv02 gates the frequency disagreement
# separately at 5%; without the ``ln(f_rfx/f_ref)`` term every bit of that
# disagreement is charged to the Q gate as well.


def test_issue945_a_frequency_error_is_not_charged_to_the_q_gate() -> None:
    """The PI's counterexample, driven through ``judge``. RED before the fix.

    ``f_ref = 1``, ``Q_ref = 100``, ``T = tau/0.02`` gives ``s = 0.02``. The
    rfx mode sits 4.9% low in frequency -- INSIDE the 5% frequency gate -- and
    its decay rate is EXACTLY the reference's, i.e. dead centre of the
    declared rate interval ``[1-s, 1+s]``. Because ``Q = pi f / alpha``, an
    equal rate at a 4.9% lower frequency IS a 4.9% lower Q, so the
    fixed-frequency reading saw ``ln(Q_rfx/Q_ref) = -0.0502`` -- exactly
    ``ln(f_rfx/f_ref)`` -- against bounds ``[-0.0198, +0.0202]`` and failed a
    mode that is on the reference's own decay rate."""
    f_ref, q_ref = 1.0, 100.0
    alpha_ref = math.pi * f_ref / q_ref
    s = 0.02
    record = (q_ref / (math.pi * f_ref)) / s          # T = tau / s

    f_rfx = 0.951                                     # 4.9% low
    q_rfx = math.pi * f_rfx / alpha_ref               # SAME decay rate

    # a second, exact pair so the count gate is met without a second defect
    verdict = rmj.judge(
        [rmj.ReferenceMode(f_ref, q_ref), rmj.ReferenceMode(2.0, 100.0)],
        [rmj.SolverMode(f_rfx, q_rfx), rmj.SolverMode(2.0, 100.0)],
        record, f_min=0.5, f_max=2.5)
    row = verdict.rows[0]

    # the premise: the window really is s = 0.02 and the row really is gated
    assert row.q_window == pytest.approx(s)
    assert row.q_gated is True
    # the frequency error is inside the gate cv02 judges frequency with
    assert row.freq_err_pct == pytest.approx(4.9, abs=1e-9)
    assert verdict.gates["max_err"] is True
    # the decay rate is dead centre of the declared interval
    alpha_rfx = math.pi * f_rfx / q_rfx
    assert alpha_rfx / alpha_ref == pytest.approx(1.0, abs=1e-15)
    assert row.q_log_rate_ratio_signed == pytest.approx(0.0, abs=1e-12)
    # what the fixed-frequency reading saw, and where it came from
    assert row.q_log_ratio_signed == pytest.approx(math.log(f_rfx / f_ref))
    assert row.q_log_freq_term == pytest.approx(math.log(f_rfx / f_ref))
    unshifted = rmj.rate_interval_to_log_q_bounds(s)
    assert not (unshifted[0] <= row.q_log_ratio_signed <= unshifted[1]), (
        "the counterexample is vacuous unless the UNSHIFTED interval rejects "
        "it -- that rejection is the defect")
    # and the shipped gate admits it
    assert row.q_pass is True
    assert verdict.gates["q"] is True
    assert verdict.passed is True


def test_issue945_the_q_gate_still_bites_at_exact_frequency() -> None:
    """The mirror of the test above: the fix must not turn the gate off.

    Same rig, frequency EXACT (so the term is zero and nothing is subtracted),
    decay rate 1.5x the declared scale off. The row must FAIL -- on both
    sides, because the interval is two-sided."""
    f_ref, q_ref = 1.0, 100.0
    alpha_ref = math.pi * f_ref / q_ref
    s = 0.02
    record = (q_ref / (math.pi * f_ref)) / s

    for direction in (+1.0, -1.0):
        alpha_rfx = alpha_ref * (1.0 + direction * 1.5 * s)
        q_rfx = math.pi * f_ref / alpha_rfx            # frequency EXACT
        verdict = rmj.judge(
            [rmj.ReferenceMode(f_ref, q_ref), rmj.ReferenceMode(2.0, 100.0)],
            [rmj.SolverMode(f_ref, q_rfx), rmj.SolverMode(2.0, 100.0)],
            record, f_min=0.5, f_max=2.5)
        row = verdict.rows[0]
        assert row.q_gated is True
        assert row.freq_err_pct == pytest.approx(0.0)
        assert row.q_log_freq_term == 0.0
        assert row.q_log_rate_ratio_signed == pytest.approx(
            math.log1p(direction * 1.5 * s))
        assert row.q_pass is False, direction
        assert verdict.gates["q"] is False
        assert verdict.passed is False

    # and 0.5x the scale, the same way, passes -- so the failures above are
    # the gate biting and not the rig being broken
    for direction in (+1.0, -1.0):
        alpha_rfx = alpha_ref * (1.0 + direction * 0.5 * s)
        verdict = rmj.judge(
            [rmj.ReferenceMode(f_ref, q_ref), rmj.ReferenceMode(2.0, 100.0)],
            [rmj.SolverMode(f_ref, math.pi * f_ref / alpha_rfx),
             rmj.SolverMode(2.0, 100.0)],
            record, f_min=0.5, f_max=2.5)
        assert verdict.gates["q"] is True, direction


def test_issue945_the_row_reports_the_frequency_term_and_the_rate_term(
) -> None:
    """Both halves of the subtraction are on the row and in the report.

    A reader must be able to see WHAT was subtracted, not just that the
    verdict changed: the row carries ``q_log_freq_term = ln(f_rfx/f_ref)``
    and ``q_log_rate_ratio_signed = ln(alpha_rfx/alpha_ref)``, they satisfy
    the identity that defines them, and the rate term is inside the DECLARED
    interval ``[log1p(-s), log1p(s)]`` exactly when the row passes."""
    verdict = _judge(RFX_TODAY)
    gated = verdict.q_gated_rows
    assert gated
    for row in gated:
        # the identity: ln(alpha_rfx/alpha_ref) = ln(f) - ln(Q)
        assert row.q_log_freq_term == pytest.approx(
            math.log(row.rfx_freq / row.ref_freq), rel=1e-15)
        assert row.q_log_rate_ratio_signed == pytest.approx(
            row.q_log_freq_term - row.q_log_ratio_signed, rel=1e-15)
        # and it is the quantity the DECLARED interval bounds
        s = row.q_window
        rate_lower = -math.inf if s >= 1.0 else math.log1p(-s)
        rate_upper = math.log1p(s)
        in_rate_interval = (
            rate_lower <= row.q_log_rate_ratio_signed <= rate_upper)
        assert row.q_pass is in_rate_interval

    # the printed report says what was subtracted, per row
    text = rmj.format_report(verdict)
    assert "shifted by ln(f_rfx/f_ref)" in text
    assert "ln(alpha_rfx/alpha_ref)" in text
    for row in gated:
        assert f"{row.q_log_freq_term:+.6f}" in text

    # and the persisted row carries both keys (the script uses asdict)
    payload = json.loads(json.dumps(
        [dataclasses.asdict(row) for row in verdict.rows]))
    for was, row in zip(payload, verdict.rows):
        assert was["q_log_freq_term"] == (
            None if row.q_log_freq_term is None else row.q_log_freq_term)
        assert "q_log_rate_ratio_signed" in was


def test_issue945_the_frequency_term_is_refused_on_an_unphysical_frequency(
) -> None:
    """``ln(f_rfx/f_ref)`` needs two positive frequencies. A non-positive or
    non-finite one is a caller bug and is refused rather than clamped into a
    plausible answer -- the same rule :func:`rate_interval_to_log_q_bounds`
    already applies to a negative rate scale."""
    assert rmj.log_frequency_term(0.15, None) == 0.0
    assert rmj.log_frequency_term(0.15, 0.15) == 0.0
    assert rmj.log_frequency_term(1.0, 2.0) == pytest.approx(math.log(2.0))
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            rmj.log_frequency_term(1.0, bad)
        with pytest.raises(ValueError):
            rmj.log_frequency_term(bad, 1.0)
    for bad in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError):
            rmj.rate_interval_to_log_q_bounds(0.5, bad)


def test_the_persisted_ingredient_payload_round_trips_through_json() -> None:
    """The script writes the ingredients and the per-row bounds into the
    retained record with ``dataclasses.asdict`` + ``json.dump``. An
    unserializable field would only surface on a live Meep host, so it is
    checked here instead."""
    verdict = _judge(RFX_TODAY)
    payload = {
        "q_gate_character": rmj.Q_GATE_CHARACTER,
        "q_gate_ingredients": [dataclasses.asdict(item)
                               for item in rmj.Q_GATE_INGREDIENTS],
        "assignment": [dataclasses.asdict(row) for row in verdict.rows],
    }
    back = json.loads(json.dumps(payload, indent=1))
    assert [i["kind"] for i in back["q_gate_ingredients"]] == [
        "declared-policy", "derived", "absent"]
    gated = [r for r in back["assignment"] if r["q_gated"]]
    assert gated
    for row in gated:
        assert row["q_log_lower"] is not None
        assert row["q_log_ratio_signed"] is not None
    # the s >= 1 row keeps its unbounded high-Q side through the round trip
    # (Python's json writes Infinity; other committed crossval records do too)
    unbounded = [r for r in gated if r["q_window"] >= 1.0]
    assert unbounded
    assert all(r["q_log_upper"] == math.inf for r in unbounded)

CV02_RECORD = REPO_ROOT / "validation/crossval/_02_ring_resonator_results/crossval.json"

#: WHICH retained record the guard below is a guard against. The check is only
#: a check because this record was written BEFORE the asymmetric transform
#: existed -- re-driving today's judge on it then answers "did the transform
#: move an answer that was already committed?". Regenerate the record with
#: today's judge and the comparison becomes the judge against its own output,
#: which cannot detect a verdict move at all. So the identity is pinned here:
#: a regeneration reds this test loudly instead of quietly voiding it.
#:
#: If you regenerated the record on purpose, do NOT simply retype these two
#: strings. Re-establish the guard against a copy of the record that predates
#: the transform under test (the commit named below still carries it), or
#: delete the guard and say in the PR body that the falsifier is gone.
CV02_RECORD_PIN = {
    "commit": "296cabada23343eede7beb05a0a4f98f7bb64adf",
    "date_utc": "2026-09-06T17:07:57Z",
}


def test_the_committed_record_is_still_the_pre_transform_one() -> None:
    """The guard below is void if the record moved. Fail loudly when it does.

    Nothing else in the suite notices a regenerated
    ``_02_ring_resonator_results/crossval.json``: the verdict comparison would
    keep passing while silently comparing today's judge against a record that
    same judge had just written.
    """
    doc = json.loads(CV02_RECORD.read_text(encoding="utf-8"))
    assert {k: doc[k] for k in CV02_RECORD_PIN} == CV02_RECORD_PIN, (
        "the retained cv02 record is not the one the transform guard was "
        "established against -- read CV02_RECORD_PIN before touching this"
    )


def test_the_committed_record_reproduces_its_own_verdict_through_the_judge(
) -> None:
    """The #945 transform must not move a verdict that is already committed.

    The retained record carries both mode lists, the band, the record length
    and the gate table it was written with. Re-driving today's judge on those
    stored inputs has to return the same five gates. This is the falsifier for
    "widening the high-Q side changed a committed answer": if it ever fires,
    the record is NOT to be regenerated to make it green -- the divergence is
    the finding.

    Which record that is, and why regenerating it would void this check
    rather than refresh it: :data:`CV02_RECORD_PIN` and the test above.
    """
    doc = json.loads(CV02_RECORD.read_text(encoding="utf-8"))
    assert {k: doc[k] for k in CV02_RECORD_PIN} == CV02_RECORD_PIN
    f_min, f_max = doc["rig"]["band_c_over_a"]
    verdict = rmj.judge(
        [rmj.ReferenceMode(m["freq_c_over_a"], m["Q"])
         for m in doc["measured"]["meep_modes"]],
        [rmj.SolverMode(m["freq_c_over_a"], m["Q"], m["amplitude"])
         for m in doc["measured"]["rfx_modes"]],
        doc["rig"]["record_length_meep_units"],
        f_min=f_min, f_max=f_max,
    )
    assert verdict.gates == doc["gates"]
    assert verdict.passed is doc["verdict"]["judge_passed"]
    # per row, too -- a gate table can agree while a row's reason changed
    stored = doc["measured"]["assignment"]
    assert len(verdict.rows) == len(stored)
    for row, was in zip(verdict.rows, stored):
        assert row.ref_freq == pytest.approx(was["ref_freq"])
        assert row.q_gated is was["q_gated"]
        assert row.q_pass is was["q_pass"]
        assert row.q_window == pytest.approx(was["q_window"])
        if was["q_log_ratio"] is not None:
            assert row.q_log_ratio == pytest.approx(was["q_log_ratio"])
    # and the record really does exercise the s >= 1 branch this PR is about,
    # so the check above is not vacuous
    assert any(r.q_gated and r.q_window >= 1.0 for r in verdict.rows)

def test_script_persists_the_ingredient_split_and_drops_the_old_claim(
) -> None:
    """Revert-proof, on the script: the retained artifact must carry the named
    ingredients and the heuristic reading, and must not re-assert that the Q
    window is derived rather than chosen."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "q_gate_ingredients" in source
    assert "Q_GATE_INGREDIENTS" in source
    assert "Q_GATE_CHARACTER" in source
    assert "not a chosen number" not in source
    assert "consistency heuristic" in source


def test_the_retained_record_keeps_its_superseded_note_and_a_pointer_to_that(
) -> None:
    """The one surface the withdrawal could not reach, and the pointer that
    stands in for reaching it.

    ``_02_ring_resonator_results/crossval.json`` still carries the framing this
    PR withdraws -- its ``gate_limits.note`` calls the window "not a chosen
    number" -- and it CANNOT be regenerated to fix that: the record is pinned
    (:data:`CV02_RECORD_PIN`) because it is the only copy predating the
    transform, so re-driving today's judge on it is the falsifier for "the
    transform moved a committed verdict". Regenerating would void the
    falsifier to fix a sentence.

    So the correction lives beside the record instead, on the surface a reader
    arrives from: the manifest entry that points at it. This test keeps the two
    tied together -- if the record's note is ever fixed at the source, the
    pointer can go with it; while the note stands, the pointer must too.
    """
    record = json.loads(CV02_RECORD.read_text(encoding="utf-8"))
    note = record["gate_limits"]["note"]
    manifest = json.loads(
        (REPO_ROOT / "validation/crossval/manifest.json").read_text(
            encoding="utf-8"))
    case = next(c for c in manifest["cases"] if c["id"] == "02_ring_resonator")
    if "not a chosen number" in note:
        assert "superseded by ring_mode_judge.Q_GATE_INGREDIENTS" in (
            case["claim_scope"]), (
            "the retained record still calls the Q window 'not a chosen "
            "number'; the manifest must say that wording is superseded")


def test_reference_side_carries_the_same_q_floor_as_rfx() -> None:
    """The shipped script filtered rfx modes (Q > 1) and the reference not at
    all, so a Meep harminv artefact used to enter as a full-weight mode -- and,
    under the new unmatched gate, would have failed the case for it."""
    junk = MEEP_REFERENCE + [rmj.ReferenceMode(0.13, 0.4)]
    assert len(rmj.admit(junk, F_MIN, F_MAX)) == 3
    assert _judge(RFX_TODAY, reference=junk).passed


def test_surplus_rfx_mode_is_reported_not_gated() -> None:
    """A mode rfx finds and Meep does not is recorded, not converted into a
    failure this lane cannot verify against a live Meep."""
    verdict = _judge(RFX_TODAY + [rmj.SolverMode(0.16, 500.0)])
    assert verdict.passed
    assert [round(m.freq, 3) for m in verdict.surplus] == [0.16]
    assert "SURPLUS" in rmj.format_report(verdict)


def test_the_inline_tautology_is_gone_from_the_script() -> None:
    """Revert-proof: the shipped matcher window must not come back inline."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "best_diff" not in source
    assert "ring_mode_judge" in source
    # ...and it is still reachable, under a name that says what it is.
    assert "best_diff" in JUDGE_PATH.read_text(encoding="utf-8")


def test_committed_200k_rerun_reproduces_the_audit_and_refutes_it() -> None:
    """The full re-run of #812's harness, committed alongside the judge.

    Regenerate with::

        python scripts/diagnostics/cv02_judge_tautology_trials.py \
            --trials 200000 \
            --output tests/fixtures/cv02_ring_judge/tautology_trials_200k.json
    """
    import json

    data = json.loads(TRIALS_JSON.read_text(encoding="utf-8"))
    assert data["trials"] == 200_000

    # The audit's finding, reproduced: through the shipped judge the mean gate
    # never fires, and mean_err never reaches 5% (#812 saw 4.9997%).
    assert data["legacy"]["failures"] == 0
    assert data["legacy"]["trials_with_mean_err_ge_5pct"] == 0
    assert 4.9 <= data["legacy"]["max_mean_err_pct"] < 5.0

    # The same stream through the new judge: the bound is gone and so is the
    # tautology.
    assert data["regated"]["max_mean_err_pct"] > 5.0
    assert data["regated"]["trials_with_mean_err_ge_5pct"] > 10_000
    assert data["regated"]["failures"] > 50_000
    for gate in ("unmatched", "mean_err", "max_err"):
        assert data["regated"]["failures_by_gate"][gate] > 0, gate

    # ...while defect-free trials still pass, on both judges.
    assert data["defect_free_trials"]["count"] > 50_000
    assert data["defect_free_trials"]["regated_failures"] == 0
    assert data["defect_free_trials"]["legacy_failures"] == 0


def test_report_renders_every_row_kind_without_crashing() -> None:
    """UNMATCHED and SURPLUS cannot co-occur -- the assignment always uses
    ``min(n_ref, n_rfx)`` pairs -- so each is rendered from its own verdict."""
    short = [rmj.SolverMode(0.147213 * 0.85, 2.0),
             rmj.SolverMode(0.118068, 86.5)]
    text = rmj.format_report(_judge(short))
    assert "UNMATCHED" in text
    assert "FAIL: gate unmatched" in text and "FAIL: gate q" in text
    assert "not gated" in text

    long = rmj.format_report(_judge(RFX_TODAY + [rmj.SolverMode(0.16, 500.0)]))
    assert "SURPLUS" in long
    assert "FAIL" not in long


# --- cv02 audit G2: per-mode ring-down settling witness ---------------------
#
# cv02 is an open (UPML) claims-bearing Harminv case, so the repo rule
# ("Ring-down settling witness" in rfx/CLAUDE.md) requires its mode numbers to
# be quoted with how far the record's end energy sits below the post-source
# peak. The script recorded none. These tests pin the witness math -- all of it
# derived from a mode's own (f, Q) and the run's own record length, none pinned
# to this geometry -- and the fact that the run length now scales with the
# radiation-limited highest-Q mode instead of a fixed step count.


def test_amplitude_tau_is_Q_over_pi_f() -> None:
    """tau = Q/(pi f), the same decay time the Q window uses. No fitted number."""
    for f, q in [(0.118, 80.0), (5.25e13, 1677.0), (1.0, 1.0)]:
        assert rmj.amplitude_tau(f, q) == pytest.approx(q / (math.pi * f),
                                                        rel=1e-12)
    # A non-decaying / non-physical mode has infinite tau (never truncated).
    assert math.isinf(rmj.amplitude_tau(0.15, 0.0))
    assert math.isinf(rmj.amplitude_tau(0.0, 100.0))


def test_energy_db_per_efold_is_the_exponential_decay_constant() -> None:
    """-8.6859 dB per amplitude e-folding is 10*log10(e**-2) -- a property of
    exponential decay, not a chosen threshold."""
    assert rmj.ENERGY_DB_PER_EFOLD == pytest.approx(10.0 * math.log10(math.e ** -2))
    assert rmj.ENERGY_DB_PER_EFOLD == pytest.approx(-8.6859, abs=1e-3)


def test_mode_settling_energy_db_is_t_over_tau_times_the_constant() -> None:
    """Per-mode witness: T/tau e-foldings and the energy end/peak they imply."""
    f, q = 5.0e13, 300.0
    tau = q / (math.pi * f)
    record = 2.5 * tau                      # 2.5 amplitude e-foldings
    row = rmj.mode_settling(f, q, record)
    assert row.t_over_tau == pytest.approx(2.5, rel=1e-12)
    assert row.energy_db == pytest.approx(2.5 * rmj.ENERGY_DB_PER_EFOLD, rel=1e-12)
    # 2.5 e-foldings of amplitude is exp(-5) of energy = -21.7 dB, closed form.
    assert row.energy_db == pytest.approx(10.0 * math.log10(math.exp(-5.0)),
                                          rel=1e-9)
    assert row.observed is True


def test_mode_settling_flags_truncation_at_the_judges_own_floor() -> None:
    """`observed` uses the judge's Q-gating floor (Q_RECORD_MIN_EFOLDS = 1/4):
    a record shorter than that has not seen the decay and is flagged, not
    trusted -- the same cut the judge gates on, reused as a report flag."""
    f, q = 5.0e13, 300.0
    tau = q / (math.pi * f)
    just_under = rmj.mode_settling(f, q, 0.20 * tau)   # < 1/4 e-folding
    just_over = rmj.mode_settling(f, q, 0.30 * tau)    # > 1/4 e-folding
    assert just_under.observed is False
    assert just_over.observed is True
    assert just_under.t_over_tau < rmj.Q_RECORD_MIN_EFOLDS <= just_over.t_over_tau


def test_slowest_mode_sets_the_record_length() -> None:
    """The record length scales with the SLOWEST (highest-Q) mode's tau, so a
    fixed number of e-foldings of it guarantees at least that many of every
    faster mode. This is the runtime rule that replaces a fixed step count."""
    modes = [rmj.SolverMode(0.118, 80.0),
             rmj.SolverMode(0.147, 316.0),
             rmj.SolverMode(0.175, 1677.0)]      # highest Q, slowest decay
    taus = [rmj.amplitude_tau(m.freq, m.Q) for m in modes]
    assert rmj.slowest_amplitude_tau(modes) == pytest.approx(max(taus))
    # 0.175 mode is the slowest here, so it caps every other mode's e-foldings.
    assert max(taus) == taus[2]
    for target in (0.25, 1.0, 4.6):
        L = rmj.record_length_for_efolds(modes, target)
        assert L == pytest.approx(target * max(taus), rel=1e-12)
        # every faster mode gets >= `target` e-foldings over that same record
        for m, tau in zip(modes, taus):
            assert L / tau >= target - 1e-12
    # No decaying mode -> nothing to scale from -> None (caller falls back).
    assert rmj.slowest_amplitude_tau([]) is None
    assert rmj.record_length_for_efolds([], 1.0) is None


def test_signal_settling_db_matches_a_closed_form_decay() -> None:
    """Closed form, not a mirror: for ``exp(-t/tau)`` the peak is the first
    sample (=1) and the tail mean is a geometric series, so the expected dB is
    written out in full below and never touches the implementation's array
    arithmetic."""
    n, dt, tau = 20000, 1.0, 2500.0
    sig = np.exp(-np.arange(n) * dt / tau)
    db = rmj.signal_settling_db(sig, tail_fraction=0.1)

    # P = exp(-2t/tau); peak = P[0] = 1. Tail = the last n_tail samples,
    # starting at t0 = (n - n_tail)*dt; sum_{k<n_tail} exp(-2(t0+k dt)/tau)
    # = exp(-2 t0/tau) * (1 - q**n_tail)/(1 - q) with q = exp(-2 dt/tau).
    n_tail = n // 10
    q = math.exp(-2.0 * dt / tau)
    t0 = (n - n_tail) * dt
    tail_mean = (math.exp(-2.0 * t0 / tau)
                 * (1.0 - q ** n_tail) / (1.0 - q) / n_tail)
    assert db == pytest.approx(10.0 * math.log10(tail_mean), rel=1e-9)
    assert db < -40.0                          # this record IS well settled

    # Empty / dead records are NaN, never a false 0 dB pass.
    assert math.isnan(rmj.signal_settling_db(np.zeros(100)))
    assert math.isnan(rmj.signal_settling_db(np.array([])))


def test_signal_settling_db_reads_minus_3dB_on_an_undecayed_tone() -> None:
    """The documented 3 dB offset against the per-mode column, made
    executable: the whole-signal witness divides a single-sample max of
    ``A**2 sin**2`` (~A**2) by a MEAN of ``A**2 sin**2`` (~A**2/2), so a tone
    that never settles reads 10*log10(1/2) = -3.01 dB, while the per-mode
    envelope witness reads 0.0 dB for the same non-decaying mode. Comparing the
    two columns without subtracting 3 dB compares different quantities."""
    t = np.arange(20000) * 1.0
    tone = np.sin(2 * math.pi * 0.02 * t)
    db = rmj.signal_settling_db(tone)
    assert db == pytest.approx(10.0 * math.log10(0.5), abs=0.05)
    # same non-decaying mode, per-mode envelope column: exactly 0 dB
    assert rmj.mode_settling(0.02, 0.0, 20000.0).energy_db == 0.0
    # and the report says which frame its peak came from
    text = rmj.format_settling_report([], db, 1.0, peak_offset_after_source=0.0)
    assert "post-source peak" in text
    late = rmj.format_settling_report([], db, 1.0,
                                      peak_offset_after_source=94.0)
    assert "ALREADY-DECAYED" in late and "optimistic" in late


def test_format_settling_report_states_the_physical_limitation() -> None:
    """The report names the -40 dB rule as a documented physical limitation,
    not a gate, and prints the per-mode e-foldings."""
    rows = [rmj.mode_settling(0.175, 1677.0, 1.0 * rmj.amplitude_tau(0.175, 1677.0)),
            rmj.mode_settling(0.118, 80.0, 1.0 * rmj.amplitude_tau(0.175, 1677.0))]
    text = rmj.format_settling_report(rows, -26.0, 1.0)
    assert "PHYSICAL LIMITATION" in text
    assert "not a gate" in text
    assert "T/tau" in text
    assert "4.61 e-foldings" in text     # -40 dB / -8.686 dB per e-folding


def test_script_records_a_per_mode_settling_witness() -> None:
    """Revert-proof: the script must wire the per-mode settling witness and, on
    the Meep-absent lane, take its run length from the BOUNDED planner rather
    than from the old magic 450 constant or from the unbounded primitive."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "mode_settling" in source
    assert "format_settling_report" in source
    assert "SETTLE_TARGET_EFOLDS" in source
    # the driven portion is skipped by the computed source-off time on the
    # tau-scaled lane (the Meep verdict lane keeps its calibrated 40% window).
    assert "source_off_time" in source
    # The run length comes from plan_record (band-filtered + clamped at the
    # resolvable-tau bound), NOT from record_length_for_efolds, which is
    # unbounded over a raw harminv mode list and let a band-edge artefact ask
    # for a ~2.3e7-step run.
    assert "plan_record(" in source
    assert "record_length_for_efolds(" not in source
    assert "RECORD_LADDER_BUDGET" in source


def test_script_does_not_claim_the_tau_scaling_on_the_verdict_lane() -> None:
    """The tau-scaled record exists ONLY on the Meep-absent, no-verdict lane.
    The script must say so where the policy is written, and must still hand the
    verdict lane Meep's own record length and the calibrated 40% skip."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "the cv02 verdict lane does not use the tau-scaled record" in source
    assert "meep_total_t = sim_meep.meep_time()" in source
    assert "skip = int(len(ts) * 0.4)" in source


# --- cv02 review2 F1: the record rule must be band-limited and bounded ------
#
# The tau-scaled record rule as first written took max(tau) over EVERY mode
# rfx's harminv returned. harminv deliberately searches a 10%-widened band
# (rfx/harminv.py: freq <= 1.1*f_max) so the requested band is interior to the
# search, so that pool contained modes no gate ever reads, at the one place
# harminv is least reliable. On this board the band edge returns f=0.2027 with
# a Q that swings between 1.0e3 and 1.0e6 depending on the window; at 1.0e6 the
# unfiltered rule asks for 1.6e6 Meep units of record (~2.3e7 steps, ~4500x the
# committed run). These tests pin both guards.

BAND = dict(f_min=0.10, f_max=0.20)          # the judge's band for cv02

# The band-edge mode the live run actually produced, at both Qs it has read.
BAND_EDGE_SPUR = [rmj.SolverMode(0.202707, 1.03e6, 3.5e-10),
                  rmj.SolverMode(0.202707, 1006.3, 7.3e-10)]


@pytest.mark.parametrize("spur", BAND_EDGE_SPUR)
def test_out_of_band_mode_cannot_set_the_record_length(spur) -> None:
    """A mode outside the judge's band is reported, never scaled off."""
    present = 385.0                                   # bootstrap free decay
    plan = rmj.plan_record(list(RFX_TODAY) + [spur],
                           record_after_source=present, target_efolds=1.0,
                           **BAND)
    assert spur in plan.out_of_band
    assert spur not in plan.kept and spur not in plan.unresolved
    # what the UNFILTERED primitive would have asked for, for contrast
    unfiltered = rmj.record_length_for_efolds(list(RFX_TODAY) + [spur], 1.0)
    assert unfiltered >= rmj.amplitude_tau(spur.freq, spur.Q)
    assert plan.length < unfiltered
    # and the plan is still clamped by the record in hand
    assert plan.length <= plan.cap == present / rmj.Q_RECORD_MIN_EFOLDS


def test_one_rung_can_never_exceed_the_resolvable_tau_bound() -> None:
    """The hard bound, over absurd Qs: whatever a mode's Q reads, one rung asks
    for at most ``T / Q_RECORD_MIN_EFOLDS`` -- the published floor inverted. A
    tau above that bound was not measured by this record (T/tau < the floor),
    so it cannot be scaled off; it is re-measured on the next rung instead."""
    present = 385.0
    for q in (1.5, 1e2, 1e4, 1e6, 1e12):
        for target in (0.25, 1.0, 4.61):
            modes = [rmj.SolverMode(0.147, 355.0), rmj.SolverMode(0.175, q)]
            plan = rmj.plan_record(modes, record_after_source=present,
                                   target_efolds=target, **BAND)
            assert present <= plan.length <= present / rmj.Q_RECORD_MIN_EFOLDS
            tau_hi = rmj.amplitude_tau(0.175, q)
            if tau_hi > plan.cap:
                assert [m for m in plan.unresolved if m.Q == q]
                assert plan.length == pytest.approx(plan.cap)
            else:
                assert [m for m in plan.kept if m.Q == q]


def test_ladder_converges_on_the_slowest_in_band_tau_and_stops() -> None:
    """Driven as the script drives it: rung after rung on a fixed mode set, the
    ladder climbs by at most 4x, converges on ``target * slowest in-band tau``,
    and then reports no further extension (``extend`` False) so the caller
    stops. The out-of-band artefact never enters."""
    modes = list(RFX_TODAY) + [BAND_EDGE_SPUR[0]]
    tau_slow = max(rmj.amplitude_tau(m.freq, m.Q)
                   for m in rmj.admit(modes, **BAND))
    present, lengths = 385.0, []
    for _ in range(6):
        plan = rmj.plan_record(modes, record_after_source=present,
                               target_efolds=1.0, **BAND)
        assert plan.length <= present / rmj.Q_RECORD_MIN_EFOLDS
        lengths.append(plan.length)
        if not plan.extend:
            break
        present = plan.length
    assert lengths == sorted(lengths)                    # monotone
    assert present == pytest.approx(tau_slow, rel=1e-9)  # the in-band slowest
    assert plan.extend is False                          # and it stops
    assert len(lengths) <= 4                             # 385 -> 1540 -> 3385


def test_plan_record_falls_back_when_nothing_is_resolved() -> None:
    """No in-band resolved mode -> no scaling: keep the record in hand (the
    caller then reports zero/unresolved modes) rather than invent a length."""
    present = 10.0
    plan = rmj.plan_record([rmj.SolverMode(0.5, 1e6)],   # out of band
                           record_after_source=present, target_efolds=1.0,
                           **BAND)
    assert plan.slowest_tau is None
    assert plan.length == pytest.approx(present)
    assert plan.extend is False


def test_in_band_low_q_mode_is_not_labelled_out_of_band() -> None:
    """An in-band mode dropped on the MIN_Q floor is reported in its own
    bucket, not as OUT-OF-BAND. The two rejections have different causes (one
    is outside the band the judge scores, one is inside it and too lossy to be
    a mode), and the printed rung must not confuse them. Neither can set the
    record length."""
    lossy = rmj.SolverMode(0.15, rmj.MIN_Q / 2.0, 1e-8)   # in band, under MIN_Q
    edge = BAND_EDGE_SPUR[0]                              # f = 0.2027, out of band
    plan = rmj.plan_record(list(RFX_TODAY) + [lossy, edge],
                           record_after_source=385.0, target_efolds=1.0,
                           **BAND)
    assert lossy in plan.below_min_q and lossy not in plan.out_of_band
    assert edge in plan.out_of_band and edge not in plan.below_min_q
    assert lossy not in plan.kept and lossy not in plan.unresolved
    text = rmj.format_record_plan(plan)
    low_q_line = [ln for ln in text.splitlines() if "LOW-Q" in ln]
    assert len(low_q_line) == 1 and "0.15" in low_q_line[0]
    assert "OUT-OF-BAND" in text and "0.2027" in text
    # and neither rejected mode moved the length
    clean = rmj.plan_record(list(RFX_TODAY), record_after_source=385.0,
                            target_efolds=1.0, **BAND)
    assert plan.length == pytest.approx(clean.length)


def test_format_record_plan_prints_every_mode_and_its_verdict() -> None:
    """The rejected modes are printed, not swallowed -- an out-of-band mode
    that shortens the run must be visible in the log."""
    text = rmj.format_record_plan(
        rmj.plan_record(list(RFX_TODAY) + [BAND_EDGE_SPUR[0]],
                        record_after_source=385.0, target_efolds=1.0, **BAND))
    assert "OUT-OF-BAND" in text and "0.2027" in text
    assert "UNRESOLVED" in text
    assert "resolvable-tau bound" in text


# --- #907 (2026-09-15): the q gate is a DECLARED consistency envelope -------
#
# What used to live here: ``test_verdict_lane_q_gate_is_run_length_contingent``
# (#896 review2 F2), a CHARACTERIZATION test whose own docstring said "the
# assertion ``longer.gates['q'] is False`` is the bug, not the requirement"
# and "when the contingency is fixed this test must be INVERTED ... or deleted
# along with it".
#
# #907 closed neither way, and that is why it is gone rather than flipped.
# INVERTING it means asserting that a longer, better-settled record keeps the
# PASS, which only a floor in the judge produces -- and the disposition #907
# actually took refuses the floor (see ``Q_GATE_INGREDIENTS[2]``: the exact
# finite-record Cramer-Rao bound is ~15x below the gap a floor would have to
# license, the frequency-transport route is short at every mode, and a floor
# large enough would have to be read off the gap). So the behaviour it pinned
# is unchanged; what changed is its STATUS. A characterization test is a
# promissory note against a future repair, and there is no repair pending any
# more. Leaving it under that name would tell the next reader a fix is coming
# when the repo has declared it is not.
#
# Every assertion it made that is still true moved into
# ``test_the_q_gate_is_the_only_gate_that_moves_with_record_length`` below,
# widened from two record lengths on one board to five on both.


def test_the_q_gate_is_declared_a_permanent_consistency_envelope() -> None:
    """#907's remaining half, landed as option (b) of the issue's own two.

    The issue asked for either a derived ingredient 1 plus a declared
    ingredient 3, or "a permanent statement that the cv02 Q gate is a
    two-solver consistency envelope". This pins the second, on every surface
    a reader arrives from, and pins that it was NOT done by quietly turning
    ingredient 3 into a number.
    """
    budget = [i for i in rmj.Q_GATE_INGREDIENTS
              if i.name == "discretization_budget"][0]
    # the hole is still a hole -- no floor was back-filled
    assert budget.kind == "absent"
    assert "none declared" in budget.quantity
    assert "DECLARED PERMANENT" in budget.basis
    # ... and the standard for ever filling it is still written down, in full
    for required in ("SNR / model-order uncertainty",
                     "source-free Meep reference record",
                     "CONVERGED"):
        assert required in budget.basis, required
    # the character says permanent, and still says what it is not
    assert "PERMANENT" in rmj.Q_GATE_CHARACTER
    assert "consistency envelope" in rmj.Q_GATE_CHARACTER
    assert "consistency heuristic" in rmj.Q_GATE_CHARACTER
    assert "NOT a Q-accuracy guarantee" in rmj.Q_GATE_CHARACTER
    # the module docstring carries the derivation, not just the verdict
    module_doc = rmj.__doc__
    assert "PERMANENT DECLARATION" in module_doc
    assert "Cramer-Rao" in module_doc
    # and the case says it where a reader of the script meets it. Flattened,
    # because the declaration is a wrapped comment block and a line break must
    # not be able to hide it.
    script = _flatten_prose(SCRIPT_PATH.read_text(encoding="utf-8"),
                            python_source=True)
    assert "DECLARED PERMANENT" in script
    assert "consistency envelope" in script
    # the known limitation stays where the number is produced
    assert "Known limitation" in rmj.q_window.__doc__
    assert "not being repaired" in rmj.q_window.__doc__


#: The ``q`` gate's verdict at five record lengths on both boards, measured
#: through the unmodified judge on 2026-09-15. FIVE of the ten cells are
#: False. This table is the declared envelope's behaviour, not a defect
#: awaiting repair (#907); a change to it is a change to a claims-bearing
#: gate and must be argued, not absorbed.
Q_GATE_BY_RECORD_LENGTH = {
    ("tutorial", 260.9798793571893): True,
    ("tutorial", 291.0): True,
    ("tutorial", 3385.0): False,
    ("tutorial", 15600.0): False,
    ("tutorial", 1e6): False,
    ("live937", 260.9798793571893): True,
    ("live937", 291.0): True,
    ("live937", 3385.0): True,
    ("live937", 15600.0): False,
    ("live937", 1e6): False,
}


def _live_board():
    """The committed #937 board, read from the retained record."""
    doc = json.loads(CV02_RECORD.read_text(encoding="utf-8"))
    ref = [rmj.ReferenceMode(m["freq_c_over_a"], m["Q"])
           for m in doc["measured"]["meep_modes"]]
    rfx = [rmj.SolverMode(m["freq_c_over_a"], m["Q"], m["amplitude"])
           for m in doc["measured"]["rfx_modes"]]
    return ref, rfx


def test_the_q_gate_is_the_only_gate_that_moves_with_record_length() -> None:
    """The declared envelope, executable — and the successor to the deleted
    characterization test (see the comment block above).

    Two things at once, and the second is the reason the first is tolerable:

    * the ``q`` gate IS run-length contingent, in exactly the five of ten
      cells pinned in :data:`Q_GATE_BY_RECORD_LENGTH`. The gate's scale is
      ``s = tau_ref/T``; shrinking it with ``T`` is the declared policy
      (ingredient 1), not an accident;
    * NOTHING else moves. ``unmatched``, ``count``, ``mean_err``,
      ``max_err``, every row's ``|ln Q|`` and every row's frequency error are
      bit-identical across a 3800x range in ``T`` on both boards. The
      physics the case measures does not depend on the record length; only
      the tolerance does.

    That pairing is the whole content of "consistency envelope": the ``q``
    verdict is a statement about the window, and the window is declared.
    """
    boards = {"tutorial": (list(MEEP_REFERENCE), list(RFX_TODAY)),
              "live937": _live_board()}
    seen, gated = {}, {}
    for name, (ref, rfx) in boards.items():
        baseline = None
        for (board, record_T) in Q_GATE_BY_RECORD_LENGTH:
            if board != name:
                continue
            verdict = rmj.judge(list(ref), list(rfx), record_T,
                                f_min=F_MIN, f_max=F_MAX)
            seen[(name, record_T)] = verdict.gates["q"]
            frozen = {
                "gates": {g: verdict.gates[g] for g in
                          ("unmatched", "count", "mean_err", "max_err")},
                "mean_err_pct": round(verdict.mean_err_pct, 12),
                # |ln Q| is a property of the two mode lists, so it is
                # reported for EVERY row regardless of whether that row was
                # gated -- computed here from the pair, not read from the
                # verdict, precisely so the growing gated set below cannot
                # hide a change in it.
                "lnQ": {round(r.ref_freq, 9): round(
                    abs(math.log(r.rfx_Q / r.ref_Q)), 12)
                    for r in verdict.rows},
                "freq_err": {round(r.ref_freq, 9): round(r.freq_err_pct, 12)
                             for r in verdict.rows},
            }
            gated[(name, record_T)] = tuple(
                round(r.ref_freq, 6) for r in verdict.rows if r.q_gated)
            for row in verdict.rows:
                if row.q_log_ratio is not None:
                    assert row.q_log_ratio == pytest.approx(
                        frozen["lnQ"][round(row.ref_freq, 9)])
            if baseline is None:
                baseline = frozen
            assert frozen == baseline, (name, record_T)
        assert all(baseline["gates"].values())
    assert seen == Q_GATE_BY_RECORD_LENGTH
    assert sum(1 for v in seen.values() if v is False) == 5
    # The one other thing T moves, and it moves the RIGHT way: the admission
    # cut Q_RECORD_MIN_EFOLDS lets MORE modes be judged as the record grows,
    # never fewer. So a longer record reds the gate while judging strictly
    # more of the physics -- which is what makes the contingency awkward and
    # is exactly why it is declared rather than hidden.
    for name in boards:
        lengths = sorted(T for (b, T) in gated if b == name)
        for shorter, longer in zip(lengths, lengths[1:]):
            assert set(gated[(name, shorter)]) <= set(gated[(name, longer)])
        assert len(gated[(name, lengths[0])]) < len(gated[(name, lengths[-1])])
    # the flip is the window alone, and it is the FASTEST mode that flips
    ref, rfx = boards["tutorial"]
    short = rmj.judge(list(ref), list(rfx), 291.0, f_min=F_MIN, f_max=F_MAX)
    long_ = rmj.judge(list(ref), list(rfx), 3385.0, f_min=F_MIN, f_max=F_MAX)
    fast_s = [r for r in short.rows if r.ref_freq < 0.12][0]
    fast_l = [r for r in long_.rows if r.ref_freq < 0.12][0]
    assert fast_l.q_window < fast_s.q_window
    assert fast_s.q_pass is True and fast_l.q_pass is False
    assert fast_l.q_log_ratio == pytest.approx(fast_s.q_log_ratio)


def test_cv02_persists_and_exits_through_one_decision_value() -> None:
    """The retained record and every process exit must share ``_rc``.

    cv02 writes its JSON before the visualization tail.  A later ad-hoc
    ``sys.exit(2)`` can therefore make a record claim PASS while the process
    reports inconclusive (the failure found on the abandoned #907 branch).
    This source contract keeps that invariant cheap and always-on: adding an
    exit path with a literal or another expression fails review immediately.

    It is no longer the only contract, and it was never sufficient on its own:
    reading the source cannot see an exit the process takes by another route
    (an uncaught exception in the tail, a ``sys.exit`` inside something the
    tail calls), which is why #946 was reopened after PR #999.  The executing
    contract lives in
    ``tests/crossval/test_crossval_exit_code_is_the_process_exit_code.py``,
    which runs cv02 in a subprocess with a forced late exit and compares the
    persisted code with the subprocess's own return code.  This test keeps its
    cheap half of the job: ``_rc`` is one value, so there is one value for
    that file to measure.

    What changed in the record shape (#946): ``exit_code`` and ``summary`` are
    no longer written into ``_doc`` by hand.  ``_exit_evidence.write_record``
    puts them into the verdict block and returns the code, so the script
    cannot hold a second copy that drifts from the value it exits with -- and
    ``_rc`` is now the name that write binds.
    """
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"),
                     filename=str(SCRIPT_PATH))
    exit_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "exit"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "sys"
    ]
    assert exit_calls, "cv02 must retain an explicit process exit"
    assert all(
        len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "_rc"
        for node in exit_calls
    ), "every cv02 exit path must use the persisted _rc decision"

    # No hand-written "exit_code" key survives: the one road into the record
    # is write_record, and it is what binds _rc.
    hand_written = [
        key.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and key.value == "exit_code"
    ]
    assert hand_written == [], (
        "cv02 puts an exit code into a document by hand at line(s) "
        f"{hand_written}; route it through _exit_evidence.write_record (#946)")

    writes = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "write_record"
    ]
    assert len(writes) == 2, (
        "cv02 must write its record in exactly two places -- the live run and "
        f"--replay -- found {len(writes)}")
    assert all(len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
               for node in writes)
    assert {node.targets[0].id for node in writes} == {"_rc", "rc"}


# --- #907: the exact annulus, and why neither ingredient closes -------------

ANNULUS_SCRIPT = REPO_ROOT / "scripts/diagnostics/cv02_exact_annulus_qnm.py"
ANNULUS_JSON = REPO_ROOT / "tests/fixtures/cv02_ring_judge/exact_annulus_qnm.json"
annulus = _load("cv02_exact_annulus_qnm", ANNULUS_SCRIPT)

#: The continuum anchor for cv02's geometry, to the digits two fully
#: independent 4x4 Bessel/Hankel matching implementations agree on. Committed
#: so the numbers survive the script, and so a reader can check the leverage
#: argument without running anything.
EXACT_ANNULUS = {
    3: (0.11819169, 77.2554),
    4: (0.14743103, 343.9165),
    5: (0.17577937, 1634.2056),
}


# Convergence witnesses, not reproducible values: at a converged root these sit a
# few ulp above zero, so their last bits are a property of the machine and not of
# the physics. Compared against their bound, never against each other.
_RESIDUAL_FIELDS = ("normalized_residual",)


def test_the_exact_annulus_oracle_reproduces_its_frozen_fixture() -> None:
    """The oracle is a diagnostic, so its output has to be checkable without
    it: the committed fixture is the artifact, the script is how it was made.

    Also pins the published digits directly, because those are what the #907
    argument is quoted from and what an independent implementation can be
    checked against.
    """
    frozen = json.loads(ANNULUS_JSON.read_text(encoding="utf-8"))
    fresh = annulus.build()
    assert fresh["schema"] == frozen["schema"]
    assert fresh["geometry"] == frozen["geometry"]
    for key, was in frozen["modes"].items():
        now = fresh["modes"][key]
        assert set(now) == set(was)
        for field, value in was.items():
            if field in _RESIDUAL_FIELDS:
                # Deliberately NOT compared by value. The equilibrated residual
                # at a converged root is a few times machine epsilon, so what it
                # reports is "the solve converged", not a number the code
                # promises to reproduce: the last bits depend on the BLAS, the
                # accumulation order and the arrival path to the root, and they
                # move by tens of percent between two runs that agree on every
                # physical digit. An earlier revision of this test compared it
                # like any other float and the `abs=1e-14` below swallowed a
                # 33-52 % relative disagreement on exactly these three entries,
                # which is how a fixture that does not re-derive from its own
                # script passed a reproduction gate. It is checked against its
                # bound instead, immediately below.
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # `abs` is gone with the residuals: every field still compared
                # here is a physical quantity of order 0.1 to 1e3, so `rel`
                # governs and no absolute floor can mask a disagreement.
                assert now[field] == pytest.approx(value, rel=1e-6)
            else:
                assert now[field] == value
        f_pub, q_pub = EXACT_ANNULUS[int(key)]
        assert now["f_exact"] == pytest.approx(f_pub, abs=5e-9)
        assert now["Q_exact"] == pytest.approx(q_pub, abs=5e-5)
        for field in _RESIDUAL_FIELDS:
            assert now[field] < annulus.RESIDUAL_MAX
            assert was[field] < annulus.RESIDUAL_MAX


def test_the_annulus_residual_is_invariant_under_row_and_column_rescaling(
) -> None:
    """Why the convergence criterion is NOT ``|det| < 1e-8``.

    Writing a matching equation in different units rescales a ROW; writing a
    basis function with a different constant rescales a COLUMN. Neither moves
    a root or changes any physics, and both multiply the determinant. So a
    bare determinant threshold is not decidable across implementations --
    three correct implementations of this same 4x4 report ``|det|`` at the
    same roots seven orders of magnitude apart.

    The row/column-equilibrated smallest singular value is invariant under
    exactly that class of rescalings, which is what makes it quotable. This
    test measures both: the residual must not move, and the determinant must.
    """
    def residual_of(mat):
        mat = mat.copy()
        for _ in range(annulus.EQUILIBRATION_SWEEPS):
            mat = mat / np.linalg.norm(mat, axis=1)[:, None]
            mat = mat / np.linalg.norm(mat, axis=0)[None, :]
        return float(np.linalg.svd(mat, compute_uv=False)[-1])

    rng = np.random.default_rng(907)
    for m in EXACT_ANNULUS:
        root = annulus.find_mode(m)
        # The invariance is measured OFF the root, where the residual is
        # O(0.03) and not floating-point noise; at the root every quantity
        # here is at machine epsilon and agreement there proves nothing.
        off = root * 1.01
        matrix = annulus.matching_matrix(off, m)
        rows = np.exp(rng.normal(0.0, 6.0, 4))
        cols = np.exp(rng.normal(0.0, 6.0, 4))
        scaled = (rows[:, None] * matrix) * cols[None, :]

        # the module's own criterion IS the equilibrated smallest singular
        # value -- swapping in a bare determinant reds this line
        assert annulus.normalized_residual(off, m) == pytest.approx(
            residual_of(matrix), rel=1e-12)
        assert residual_of(matrix) > 1e-3
        # ... and it does not move when the same physics is written with
        # different equation and basis constants
        assert residual_of(scaled) == pytest.approx(residual_of(matrix),
                                                    rel=1e-9)
        # ... while the determinant moves by orders of magnitude on exactly
        # that rewrite, which is why a |det| threshold is not decidable
        ratio = abs(np.linalg.det(scaled)) / abs(np.linalg.det(matrix))
        assert ratio > 1e3 or ratio < 1e-3
        # at the root, both forms are singular to machine precision
        assert annulus.normalized_residual(root, m) < annulus.RESIDUAL_MAX
        assert residual_of(annulus.matching_matrix(root, m)) < \
            annulus.RESIDUAL_MAX


def test_the_annulus_leverage_is_a_choice_not_an_analytic_property() -> None:
    """``|dlnQ/dlnf|`` for this annulus is not a number; it is an interval.

    Two error channels, measured on the exact solution:

    * index: ``dlnf/dlnn`` ~ -0.95, ``dlnQ/dlnn`` = +4.65/+6.59/+8.57, so
      ``|dlnQ/dlnf|`` = 4.97 / 6.91 / 8.85;
    * uniform radius: ``dlnf/dlnR = -1`` EXACTLY and ``dlnQ/dlnR = 0``
      EXACTLY -- Maxwell is scale invariant, so scaling the whole annulus
      moves every frequency and no Q at all. Leverage 0.

    A discretization budget of the form "a permitted frequency error,
    transported into Q" therefore has to DECLARE which channel a solver's
    geometry error lives in. That is an assumption about the solver, not a
    property of the annulus, and it is one reason #907's ingredient 3 stays
    absent.
    """
    frozen = json.loads(ANNULUS_JSON.read_text(encoding="utf-8"))["modes"]
    expected_index = {3: 4.9744, 4: 6.9058, 5: 8.8544}
    for key, mode in frozen.items():
        assert mode["dlnf_dlnR"] == pytest.approx(-1.0, abs=1e-6)
        assert mode["dlnQ_dlnR"] == pytest.approx(0.0, abs=1e-6)
        assert mode["leverage_radius_channel"] == pytest.approx(0.0, abs=1e-6)
        assert mode["leverage_index_channel"] == pytest.approx(
            expected_index[int(key)], abs=1e-3)
        assert mode["dlnQ_dlnn"] > 0.0 and mode["dlnf_dlnn"] < 0.0


def test_no_exact_annulus_frequency_budget_covers_the_observed_q_gap() -> None:
    """The measured half of #907's disposition, branch 2.

    Transport the frequency disagreement that is actually there through the
    LARGEST of the two leverage channels, two ways, and neither reaches the
    Q gap it would have to license:

    * the rfx-vs-Meep frequency DIFFERENCE -> 0.0026 / 0.0021 / 0.0032,
      against observed |lnQ| 0.058 / 0.048 / 0.076: short by ~23x;
    * both solvers' FULL error against the continuum, added (the most
      generous reading available) -> 0.013 / 0.023 / 0.051: still short at
      every mode, though only by 1.5x at m = 5.

    Reported as a negative result, not rounded up: the second branch gets
    within a factor 1.5 of the slowest mode's gap, so "frequency error at
    maximum leverage" is not absurd -- it is unproven, and proving it needs
    the converged ladder that does not exist.
    """
    frozen = json.loads(ANNULUS_JSON.read_text(encoding="utf-8"))["modes"]
    tight, generous = [], []
    for key in sorted(frozen, key=int):
        mode = frozen[key]
        gap = abs(mode["rfx_minus_meep_dlnQ"])
        assert mode["transported_rfx_minus_meep"] < gap
        assert mode["transported_each_solver_vs_exact"] < gap
        tight.append(gap / mode["transported_rfx_minus_meep"])
        generous.append(gap / mode["transported_each_solver_vs_exact"])
    assert min(tight) > 20.0
    assert min(generous) > 1.4
    # and the observed gaps themselves are the committed record's, unchanged
    assert [round(abs(frozen[k]["rfx_minus_meep_dlnQ"]), 4)
            for k in sorted(frozen, key=int)] == [0.0582, 0.0479, 0.0761]


def _sigma_ln_q_crb(alpha: float, record_T: float, n_samples: int,
                    rho: float) -> float:
    """Exact finite-record Cramer-Rao sigma on ``ln Q`` for a damped
    exponential, amplitude treated as an unknown nuisance.

    ``e_n = sum_k t_k**n exp(-2 alpha t_k)``;
    ``var(alpha) = (1 / 2 rho) * e0 / (e0 e2 - e1**2)``;
    ``sigma_lnQ = sqrt(var(alpha)) / alpha`` because ``Q = pi f / alpha``.
    """
    t = np.arange(n_samples) * (record_T / n_samples)
    w = np.exp(-2.0 * alpha * t)
    e0, e1, e2 = w.sum(), (t * w).sum(), (t * t * w).sum()
    return float(math.sqrt((1.0 / (2.0 * rho)) * e0 / (e0 * e2 - e1 * e1))
                 / alpha)


def test_the_finite_record_cramer_rao_bound_does_not_license_the_q_gap(
) -> None:
    """The measured half of #907's disposition, branch 1 -- and a guard
    against the mistake that made an earlier draft of it wrong.

    The long-record limit of this bound is ``2 sqrt(dt / (rho tau))``, which
    is ``T``-independent. Evaluating THAT at cv02's record is wrong by up to
    54x, because cv02's slowest mode runs at ``T/tau = 0.082`` -- nowhere
    near the limit. The exact finite-``T`` Fisher information is what this
    test uses, and it establishes three things:

    1. the exact bound is 1.7x / 7.0x / 54x LARGER than the saturated form
       at cv02's own record, so the saturated form must not be quoted here;
    2. it is nonetheless ~15x SMALLER than the slowest mode's observed
       ``|lnQ| = 0.076`` at a per-sample SNR of 1e4, so estimator noise does
       not license the gap. (``rho`` on the real multi-mode record is
       UNMEASURED; 1e4 is an assumption, stated, and the required ``rho``
       for the bound to reach the gap is only 1.1 / 8.0 / 44.7 -- ordinary
       SNRs, which is exactly why this cannot be settled without measuring
       it. That measurement is the first of ingredient 3's three missing
       artifacts.);
    3. in cv02's regime it scales as ``T**-1.5``, FASTER than the gate's
       ``tau/T``. A derived ingredient 1 would tighten this gate with record
       length, not widen it -- the sharper form of what #907 opened on.
    """
    record = json.loads(CV02_RECORD.read_text(encoding="utf-8"))
    record_T = record["rig"]["record_length_meep_units"]
    n_samples = record["rig"]["harminv_span_samples"]
    dt = record_T / n_samples
    assert (record_T, n_samples) == (260.9798793571893, 3728)

    ratios, gaps = [], []
    for row in record["measured"]["assignment"]:
        tau = row["ref_Q"] / (math.pi * row["ref_freq"])
        alpha = 1.0 / tau
        exact = _sigma_ln_q_crb(alpha, record_T, n_samples, rho=1e6)
        saturated = 2.0 * math.sqrt(dt / (1e6 * tau))
        ratios.append(exact / saturated)
        gaps.append(abs(math.log(row["rfx_Q"] / row["ref_Q"])))
    # (1) the saturated form understates the bound, badly, at this record
    assert [round(r, 1) for r in ratios] == [1.7, 7.0, 54.2]

    # (2) the slowest mode, at a stated rho, is still far under its own gap
    slow = record["measured"]["assignment"][-1]
    alpha_slow = math.pi * slow["ref_freq"] / slow["ref_Q"]
    sigma = _sigma_ln_q_crb(alpha_slow, record_T, n_samples, rho=1e4)
    assert sigma == pytest.approx(0.00509, abs=5e-5)
    assert gaps[-1] / sigma > 14.0
    # the rho that WOULD reach each gap is ordinary, not absurd -- say so
    required = [1e6 * (_sigma_ln_q_crb(
        math.pi * r["ref_freq"] / r["ref_Q"], record_T, n_samples, 1e6) / g)**2
        for r, g in zip(record["measured"]["assignment"], gaps)]
    assert [round(x, 1) for x in required] == [1.1, 8.0, 44.7]

    # (3) T**-1.5 in this regime, and it does saturate eventually -- which is
    #     why the asymptotic form is right somewhere and wrong here
    short = _sigma_ln_q_crb(alpha_slow, record_T, n_samples, rho=1e4)
    longer = _sigma_ln_q_crb(alpha_slow, 2.0 * record_T, 2 * n_samples,
                             rho=1e4)
    exponent = math.log(short / longer) / math.log(2.0)
    assert 1.35 < exponent < 1.55
    very_long = _sigma_ln_q_crb(alpha_slow, 200.0 * record_T,
                                200 * n_samples, rho=1e4)
    assert very_long == pytest.approx(
        2.0 * math.sqrt(dt / (1e4 / alpha_slow)), rel=0.02)
