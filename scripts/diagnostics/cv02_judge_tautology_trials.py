#!/usr/bin/env python3
"""Re-run issue #812's tautology harness against BOTH cv02 judges.

#812 measured the shipped cv02 judge with **200,000 random trials through the
verbatim judge**: maximum ``mean_err`` ever observed **4.9997%**, **zero
failures**. That is not a statement about rfx; it is a statement about the
judge. The matcher admits a pair only when ``best_diff < 0.05`` and the verdict
averages ``best_diff * 100`` over exactly those pairs, so ``mean_err < 5.0``
holds for every possible input — the headline gate is entailed by the matcher.

This script drives the same trial stream through the shipped judge
(``legacy_shipped_judge``, kept verbatim) and through the replacement
(``judge``) and prints both distributions. The replacement must break the
entailment: ``mean_err >= 5%`` must become reachable and the verdict must fail.

The stream deliberately satisfies the shipped judge's ONLY live gate — every
trial is constructed with at least two rfx modes inside the matcher window — so
the shipped judge's failure count is zero for the same reason it was zero in
the audit, and the two judges differ only in what they do with the rest.

Q is drawn identical on both sides here: this harness isolates the FREQUENCY
tautology, so the Q gate is trivially satisfied and cannot be what separates
the two judges.

Usage::

    python scripts/diagnostics/cv02_judge_tautology_trials.py [--trials 200000]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
JUDGE_PATH = REPO_ROOT / "validation/crossval/comparators/ring_mode_judge.py"

# The cv02 harminv search band and the record length the case actually runs
# with (printed by 02_ring_resonator.py: "rfx harminv record T = 291.0").
F_MIN, F_MAX = 0.10, 0.20
RECORD_T = 291.0

# Trial-stream shape. Pre-declared alongside the judge; none of it is fitted to
# an outcome. "inside" reproduces the audit's stream (an rfx mode inside the
# matcher window); "displaced" and "missing" are the two defect classes the
# shipped judge is blind to, at rates that keep a majority of trials clean.
P_INSIDE, P_DISPLACED, P_MISSING = 0.70, 0.15, 0.15
DISPLACED_MIN, DISPLACED_MAX = 0.05, 0.40
# Strictly greater than twice the largest in-window shift
# (2 * 0.05 * REF_MAX = 0.01905): below that, two adjacent reference modes can
# CROSS under defect-free perturbation and the assignment legitimately swaps
# them, which is an ambiguity of the stream, not a property of either judge.
MIN_SEPARATION = 0.021

# Reference modes are drawn from a band inset by the matcher window, so that an
# "inside" partner is still inside the harminv search band. Without the inset a
# defect-FREE trial at the band edge can push its partner out of [F_MIN, F_MAX]
# and read as a missing mode -- an artefact of the stream, not of either judge.
REF_MIN, REF_MAX = F_MIN / 0.95, F_MAX / 1.05


def load_judge():
    spec = importlib.util.spec_from_file_location("cv02_ring_mode_judge",
                                                  JUDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cv02_ring_mode_judge"] = module
    spec.loader.exec_module(module)
    return module


def draw_trial(rng):
    """One trial: reference modes, and an rfx mode list built from them.

    Guarantees >= 2 reference modes carry an rfx partner inside the shipped
    matcher's 5% window, so the shipped judge's ``len(matched) >= 2`` gate --
    its only gate that can fail -- always passes.
    """
    n_ref = int(rng.integers(2, 5))
    # Reference modes separated by at least MIN_SEPARATION, drawn by the
    # standard order-statistic shift so no rejection loop is needed. The
    # separation keeps the assignment problem unambiguous (it exceeds the
    # shipped matcher's widest window, 0.05 * F_MAX = 0.010), so the two judges
    # are compared on detection rather than on tie-breaking.
    span = (REF_MAX - REF_MIN) - (n_ref - 1) * MIN_SEPARATION
    ref_f = (REF_MIN + np.sort(rng.uniform(0.0, span, n_ref))
             + MIN_SEPARATION * np.arange(n_ref))
    ref_Q = 10.0 ** rng.uniform(1.0, np.log10(5000.0), n_ref)

    while True:
        kinds = rng.choice(("inside", "displaced", "missing"), size=n_ref,
                           p=(P_INSIDE, P_DISPLACED, P_MISSING))
        if int(np.sum(kinds == "inside")) >= 2:
            break

    rfx_f, rfx_Q = [], []
    for f, q, kind in zip(ref_f, ref_Q, kinds):
        if kind == "missing":
            continue
        if kind == "inside":
            eps = rng.uniform(-0.05, 0.05)
        else:
            eps = rng.choice((-1.0, 1.0)) * rng.uniform(DISPLACED_MIN,
                                                        DISPLACED_MAX)
        rfx_f.append(f * (1.0 + eps))
        rfx_Q.append(q)          # Q identical on both sides by construction
    return ref_f, ref_Q, np.array(rfx_f), np.array(rfx_Q), kinds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=812)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rmj = load_judge()
    rng = np.random.default_rng(args.seed)

    legacy_fail = legacy_max_mean = 0
    legacy_mean_ge_5 = 0
    new_fail = new_mean_ge_5 = 0
    new_max_mean = 0.0
    new_fail_unmatched = new_fail_mean = new_fail_max = 0
    clean_trials = clean_new_fail = clean_legacy_fail = 0
    legacy_max_mean = 0.0

    for _ in range(args.trials):
        ref_f, ref_Q, rfx_f, rfx_Q, kinds = draw_trial(rng)

        passed, mean_err, _ = rmj.legacy_shipped_judge(ref_f, ref_Q, rfx_f)
        if not passed:
            legacy_fail += 1
        if mean_err is not None:
            legacy_max_mean = max(legacy_max_mean, mean_err)
            if mean_err >= 5.0:
                legacy_mean_ge_5 += 1

        reference = [rmj.ReferenceMode(float(f), float(q))
                     for f, q in zip(ref_f, ref_Q)]
        solver = [rmj.SolverMode(float(f), float(q))
                  for f, q in zip(rfx_f, rfx_Q)]
        verdict = rmj.judge(reference, solver, RECORD_T,
                            f_min=F_MIN, f_max=F_MAX)
        if verdict.mean_err_pct is not None:
            new_max_mean = max(new_max_mean, verdict.mean_err_pct)
            if verdict.mean_err_pct >= 5.0:
                new_mean_ge_5 += 1
        if not verdict.passed:
            new_fail += 1
            new_fail_unmatched += int(not verdict.gates["unmatched"])
            new_fail_mean += int(not verdict.gates["mean_err"])
            new_fail_max += int(not verdict.gates["max_err"])

        if np.all(kinds == "inside"):
            clean_trials += 1
            clean_new_fail += int(not verdict.passed)
            clean_legacy_fail += int(not passed)

    result = {
        "trials": args.trials,
        "seed": args.seed,
        "record_T_meep_units": RECORD_T,
        "legacy": {
            "failures": legacy_fail,
            "max_mean_err_pct": legacy_max_mean,
            "trials_with_mean_err_ge_5pct": legacy_mean_ge_5,
        },
        "regated": {
            "failures": new_fail,
            "max_mean_err_pct": new_max_mean,
            "trials_with_mean_err_ge_5pct": new_mean_ge_5,
            "failures_by_gate": {
                "unmatched": new_fail_unmatched,
                "mean_err": new_fail_mean,
                "max_err": new_fail_max,
            },
        },
        "defect_free_trials": {
            "count": clean_trials,
            "regated_failures": clean_new_fail,
            "legacy_failures": clean_legacy_fail,
        },
    }
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
