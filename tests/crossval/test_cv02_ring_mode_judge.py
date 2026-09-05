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

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
JUDGE_PATH = REPO_ROOT / "validation/crossval/comparators/ring_mode_judge.py"
TRIALS_PATH = REPO_ROOT / "scripts/diagnostics/cv02_judge_tautology_trials.py"
TRIALS_JSON = REPO_ROOT / "tests/fixtures/cv02_ring_judge/tautology_trials_200k.json"
SCRIPT_PATH = REPO_ROOT / "validation/crossval/02_ring_resonator.py"


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
    """Mode 2's Q wrong by 5x -- exactly the radiation-loss error a staircased
    curved boundary produces. The shipped judge gates no Q at all."""
    defect = [
        rmj.SolverMode(0.147213, 357.6 * 5.0),
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
    """An over-damped mode (Q too LOW) fails too -- the log form is symmetric,
    so a leaky boundary is caught as well as a lossless one."""
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


def test_q_window_is_the_record_length_resolution_limit() -> None:
    """delta_Q/Q = tau/T, with tau = Q_ref/(pi f_ref). No fitted constant."""
    for freq, q in [(0.118101575043663, 80.683059081382),
                    (0.147162555528154, 316.29272471914),
                    (0.175246750722663, 1677.48461212767)]:
        tau = q / (math.pi * freq)
        t_over_tau, window = rmj.q_window(freq, q, RECORD_T)
        assert t_over_tau == pytest.approx(RECORD_T / tau, rel=1e-12)
        assert window == pytest.approx(tau / RECORD_T, rel=1e-12)
    # A longer record buys a tighter window, linearly. This is the property
    # that makes the gate track the instrument instead of a chosen number.
    _, w1 = rmj.q_window(0.147162555528154, 316.29272471914, RECORD_T)
    _, w2 = rmj.q_window(0.147162555528154, 316.29272471914, 4 * RECORD_T)
    assert w2 == pytest.approx(w1 / 4.0, rel=1e-12)


def test_no_admitted_q_gate_tolerates_more_than_a_factor_five() -> None:
    """Consequence of the 1/4 e-folding cut: the loosest window this rule can
    ever issue is tau/T = 4, so the widest admitted band is a factor
    1 + 4 = 5. Any Q error strictly larger than 5x fails on every gated mode.
    """
    widest = math.log(1.0 + 1.0 / rmj.Q_RECORD_MIN_EFOLDS)
    assert widest == pytest.approx(math.log(5.0))
    assert math.log(5.001) > widest


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
    short = [rmj.SolverMode(0.147213 * 0.85, 357.6 * 5.0),
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


def test_format_record_plan_prints_every_mode_and_its_verdict() -> None:
    """The rejected modes are printed, not swallowed -- an out-of-band mode
    that shortens the run must be visible in the log."""
    text = rmj.format_record_plan(
        rmj.plan_record(list(RFX_TODAY) + [BAND_EDGE_SPUR[0]],
                        record_after_source=385.0, target_efolds=1.0, **BAND))
    assert "OUT-OF-BAND" in text and "0.2027" in text
    assert "UNRESOLVED" in text
    assert "resolvable-tau bound" in text


# --- cv02 review2 F2: the verdict lane's Q gate is run-length contingent -----


def test_verdict_lane_q_gate_is_run_length_contingent() -> None:
    """Why the Meep (verdict) lane keeps its calibrated record instead of the
    tau-scaled one -- executable, so the qualification cannot rot.

    The judge's Q window ``tau_ref/T`` is a record-length RESOLUTION bound: it
    shrinks as 1/T. The rfx-vs-Meep Q gap is a discretization offset and does
    not. So on the very same (frozen) mode pair the q gate passes at the
    committed record and fails at a longer, better-settled one, with the
    frequency gates and the |ln Q| values unchanged. That is a comparator
    defect filed against the judge, NOT a licence to lengthen this lane's
    record, and NOT something this witness change fixed."""
    committed = _judge(RFX_TODAY)                      # T = RECORD_T = 291
    longer = rmj.judge(MEEP_REFERENCE, RFX_TODAY, 3385.0,
                       f_min=0.1, f_max=0.2)           # 1 e-fold of tau_slow

    assert committed.gates["q"] is True
    assert longer.gates["q"] is False
    # nothing about the physics moved: same modes, same errors, same |ln Q|
    for gate in ("unmatched", "count", "mean_err", "max_err"):
        assert committed.gates[gate] is longer.gates[gate] is True
    assert longer.mean_err_pct == pytest.approx(committed.mean_err_pct)
    ln_committed = {round(r.ref_freq, 6): r.q_log_ratio
                    for r in committed.rows if r.q_log_ratio is not None}
    ln_longer = {round(r.ref_freq, 6): r.q_log_ratio
                 for r in longer.rows if r.q_log_ratio is not None}
    for freq, value in ln_committed.items():
        assert ln_longer[freq] == pytest.approx(value)
    # the flip is the window alone, and it is the FASTEST mode that flips
    fast_c = [r for r in committed.rows if r.ref_freq < 0.12][0]
    fast_l = [r for r in longer.rows if r.ref_freq < 0.12][0]
    assert fast_l.q_window < fast_c.q_window
    assert fast_l.q_log_ratio > fast_l.q_window >= 0.0
    assert fast_c.q_log_ratio < fast_c.q_window
    # and the limitation is written where the window is derived
    assert "Known limitation" in rmj.q_window.__doc__
