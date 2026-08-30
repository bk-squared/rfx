"""D4c ON THE D5 RUNGS (issue #786) — the instrument check the decisive rungs shipped without.

WHY THIS FILE EXISTS. D4c ran the independent-estimator check on D0's
eleven records only (s = 1.5, 1.0, 0.75, 0.6, 0.5 in both arms plus the
s = 0.25 reference). The three rungs D5 ADDED -- 3/s = 7, 8, 9 -- carry
THREE of the four descending steps that the turn-over verdict rests on,
and they were never put through any instrument check at all. This module
supplies it.

THE WINDOW IS NOT NEW. It is the D4c pair frozen in
``results/predeclared_windows_786.json`` before any measurement, read from
that file and applied unchanged:

    estimators AGREE          iff max pairwise spread(E2,E3,E4) <= 1 MHz
    EXONERATE the extraction  iff |E1 - consensus| <= 1 MHz at that rung
    ATTRIBUTE to (4a)         iff |E1 - consensus| >= 10 MHz at that rung
    INCONCLUSIVE              in between; CONSENSUS-UNAVAILABLE if the
                              three independent estimators do not agree

Nothing is widened; the only change is WHICH rungs the rule is evaluated
on. This file is committed BEFORE it is run.

WHAT IS RE-RUN, AND WHY. D5 stored ``f_target`` but not the raw probe
records, so the three rungs must be time-stepped again to obtain a record
to re-estimate from (~6 min of CPU in total). That re-run is also a free
determinism check: ``f_target`` must come back bit-identical to
``d5_turnover.json``, and the harness reports the difference. It does NOT
re-judge D5: the turn-over verdict stays as committed.

WHAT THIS CHECK CAN AND CANNOT SHOW. It bounds the EXTRACTION's own
disagreement on the identical record. It says nothing about the
discretization error at those rungs, and it is not a claim about the
continuum limit.

Run:  PYTHONPATH=. python -m validation.research.convergence_floor.d5_instrument_check
"""

from __future__ import annotations

import json
import os

import numpy as np

from validation.research.convergence_floor import fixture as fx
from validation.research.convergence_floor import estimators as est

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "d5_instrument_check.json")
NPZ = os.path.join(RES, "d5_records.npz")

# the three rungs D5 added; the SAME lattice-valid scales, no new geometry
D5_SCALES = (3.0 / 7.0, 0.375, 1.0 / 3.0)


def main():
    fx.quiet_third_party_warnings()
    win = json.load(open(os.path.join(
        RES, "predeclared_windows_786.json")))["D4_reference_quality"][
            "D4c_independent_estimators"]
    spread_ok = float(win["spread_hz"])
    attribute = float(win["attribute_hz"])
    exonerate = float(win["exonerate_hz"])

    d5 = json.load(open(os.path.join(RES, "d5_turnover.json")))
    committed = {r["scale"]: r["f_target"] for r in d5["rows"]}

    records = {}
    if os.path.exists(NPZ):
        records = dict(np.load(NPZ))

    rows = []
    for s in D5_SCALES:
        key = "UC_%s" % s
        if key in records:
            ts = records[key]
            dt = float(records[key + "__dt"][0])
            e1 = committed[s]
            wall = float("nan")
        else:
            r = fx.measure(s, multiband=False, keep_series=True)
            ts = np.asarray(r.pop("_series"))
            dt = float(r["dt"])
            e1 = r["f_target"]
            wall = r["wallclock_s"]
            records[key] = ts
            records[key + "__dt"] = np.array([dt])
        cons = est.consensus(ts, dt, fx.BAND)
        agree = cons["spread_hz"] <= spread_ok
        delta = abs(e1 - cons["mean_hz"])
        verdict = (("ATTRIBUTE-4a" if delta >= attribute else
                    ("EXONERATE-4a" if delta <= exonerate
                     else "INCONCLUSIVE")) if agree
                   else "CONSENSUS-UNAVAILABLE")
        rows.append({
            "arm": "UC", "scale": s, "three_over_s": round(3.0 / s),
            "n_samples": int(len(ts)), "dt": dt,
            "E1_hz": e1,
            "E1_minus_committed_d5_hz": e1 - committed[s],
            **cons,
            "E1_minus_consensus_hz": e1 - cons["mean_hz"],
            "estimators_agree": bool(agree),
            "verdict": verdict,
            "wallclock_s": wall,
        })
        print("UC 3/s=%-3d s=%-9.6f N=%6d  E1=%.6f  E2=%.6f E3=%.6f "
              "E4=%.6f  spread=%.1f kHz  E1-cons=%+8.3f MHz  "
              "E1-d5=%+.1f Hz  %s"
              % (rows[-1]["three_over_s"], s, len(ts), e1 / 1e9,
                 cons["values_hz"]["E2"] / 1e9, cons["values_hz"]["E3"] / 1e9,
                 cons["values_hz"]["E4"] / 1e9, cons["spread_hz"] / 1e3,
                 rows[-1]["E1_minus_consensus_hz"] / 1e6,
                 rows[-1]["E1_minus_committed_d5_hz"], verdict), flush=True)

    np.savez_compressed(NPZ, **records)
    spreads = [r["spread_hz"] for r in rows]
    out = {
        "issue": 786,
        "discriminator": "D4c applied to the D5 rungs (3/s = 7, 8, 9)",
        "window_source": ("results/predeclared_windows_786.json :: "
                          "D4_reference_quality.D4c_independent_estimators "
                          "(frozen, unchanged)"),
        "window": {"spread_hz": spread_ok, "exonerate_hz": exonerate,
                   "attribute_hz": attribute},
        "rows": rows,
        "reproduces_d5_bit_identically": bool(
            all(r["E1_minus_committed_d5_hz"] == 0.0 for r in rows)),
        "max_abs_E1_minus_committed_d5_hz": float(
            max(abs(r["E1_minus_committed_d5_hz"]) for r in rows)),
        "estimator_spread_range_khz": [float(min(spreads) / 1e3),
                                       float(max(spreads) / 1e3)],
        "max_abs_E1_minus_consensus_hz": float(
            max(abs(r["E1_minus_consensus_hz"]) for r in rows)),
        "verdicts": {("3/s=%d" % r["three_over_s"]): r["verdict"]
                     for r in rows},
        "scope": ("bounds the EXTRACTION's disagreement on the identical "
                  "record at these three rungs; says nothing about the "
                  "discretization error there, and is not a claim about "
                  "the continuum limit"),
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nestimator spread over the three D5 rungs: %.1f - %.1f kHz"
          % tuple(out["estimator_spread_range_khz"]))
    print("max |E1 - consensus| = %.1f kHz (exonerate window %.0f kHz)"
          % (out["max_abs_E1_minus_consensus_hz"] / 1e3, exonerate / 1e3))
    print("reproduces d5_turnover.json bit-identically:",
          out["reproduces_d5_bit_identically"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
