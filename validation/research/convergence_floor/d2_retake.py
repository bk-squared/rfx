"""D2 RE-TAKE (issue #786) — the SAME frozen rule, applied to the NINE-rung ladder.

WHY THIS FILE EXISTS. ``d2_edge.py`` evaluated the frozen D2 rule on the
five pre-D5 scales only (s = 1.5, 1.0, 0.75, 0.6, 0.5) and recorded
EXONERATED-as-the-floor-mechanism. D5 then measured three further
lattice-valid rungs (3/s = 7, 8, 9) and the reference rung, and on that
nine-rung ladder BOTH clauses of the exonerate branch fail:

    f(s) is NOT monotone (d5_turnover.json: sign_changes = 1), and the
    error against the D4b reference INCREASES over the four finest rungs.

So the letter verdict recorded in ``d2_edge.json`` does not hold on the
data the lane ended up with. This module RE-TAKES it.

WHAT IS AND IS NOT CHANGED. The RULE is not rewritten. It is read from
``results/predeclared_windows_786.json`` (frozen before any measurement)
and its branch logic is reproduced here verbatim from ``d2_edge.py``:

    EXONERATE-as-floor  iff f(s) monotone AND err-vs-D4b decreasing at
                        every rung AND p_trace >= 1.0
    ATTRIBUTE-partial   iff p_trace in [1.0, 1.6] AND p_smooth >= 1.8
    INCONCLUSIVE        iff p_trace < 1.0, or otherwise

The only thing that changes is the SCALE SET the rule is evaluated on:
the nine rungs of the full uniform ladder instead of the five that
existed when D2 first ran. No window is widened, no threshold is moved,
and ``d2_edge.json`` is left exactly as it was committed (this writes a
separate file, ``d2_edge_retake.json``, which names it as superseded).

NO NEW FDTD. Every number here is read off committed JSON:
``d5_turnover.json`` (the nine-rung f(s)) and ``d4_reference_b.json``
(f_inf, the independent Richardson reference, and p_smooth from
``d2_edge.json``'s D4a-twin control, which is unchanged by D5).

WHAT SURVIVES THE RE-TAKE. The physical ARGUMENT behind the D2 window --
a Meixner wedge of exponent nu reduces the convergence ORDER and cannot
by itself manufacture a non-vanishing floor -- is a theory statement,
derived in the pre-declaration from the 3*pi/2 conductor wedge. It is
reported here as an ARGUMENT. It is NOT a measured exoneration, and this
lane has no measurement that exonerates the edge on the full ladder.
The smooth-field control stands as measured: p_smooth = 2.0001 analytic
/ 1.9707 measured on the exact-reference twin.

Run:  PYTHONPATH=. python -m validation.research.convergence_floor.d2_retake
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "d2_edge_retake.json")
WINDOWS = os.path.join(RES, "predeclared_windows_786.json")


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        raise SystemExit("missing %s -- run the earlier discriminators first"
                         % name)
    return json.load(open(p))


def main():
    win = load("predeclared_windows_786.json")["D2_edge_singularity"]
    p_lo, p_hi = float(win["p_trace_lo"]), float(win["p_trace_hi"])
    p_smooth_min = float(win["p_smooth_min"])

    d5 = load("d5_turnover.json")
    d4b = load("d4_reference_b.json")["D4b"]["uniform"]
    d2 = load("d2_edge.json")

    scales = np.asarray(d5["ladder_scales"], dtype=float)
    f = np.asarray(d5["ladder_f_hz"], dtype=float)
    h = 0.25e-3 * scales                      # dz_fine, the same h D4b used
    f_inf = float(d4b["f_inf_hz"])
    err = np.abs(f_inf - f)

    # --- the three clauses of the frozen rule, on nine rungs -----------
    df = np.diff(f)
    trace_mono = bool(np.all(df > 0) or np.all(df < 0))
    derr = np.diff(err)
    trace_err_decreasing = bool(np.all(derr < 0))
    # p_trace by the SAME estimator D4b/d2_edge used: log-log slope of the
    # error against the independent reference.
    p_trace = float(np.polyfit(np.log(h), np.log(err), 1)[0])
    p_smooth = float(d2["p_smooth_used"])

    # --- branch logic, verbatim from d2_edge.py ------------------------
    if trace_mono and trace_err_decreasing and p_trace >= 1.0:
        verdict = ("EXONERATED as the FLOOR mechanism (p_trace=%.2f)"
                   % p_trace)
        if p_lo <= p_trace <= p_hi and p_smooth >= p_smooth_min:
            verdict += " + ORDER-REDUCTION ATTRIBUTED"
    elif p_trace < 1.0:
        verdict = "INCONCLUSIVE (p_trace=%.2f < 1.0)" % p_trace
    else:
        verdict = ("INCONCLUSIVE (p_trace=%.2f, monotone=%s, err "
                   "decreasing=%s)" % (p_trace, trace_mono,
                                       trace_err_decreasing))

    attribute_partial = bool(p_lo <= p_trace <= p_hi
                             and p_smooth >= p_smooth_min)

    out = {
        "issue": 786,
        "discriminator": "D2",
        "kind": "RE-TAKE of the D2 letter verdict on the nine-rung ladder",
        "provenance": {
            "supersedes": "results/d2_edge.json :: verdict",
            "superseded_verdict": d2["verdict"],
            "why": ("d2_edge.py evaluated the frozen rule on the five "
                    "pre-D5 scales only; D5 added 3/s = 7, 8, 9 and the "
                    "nine-rung ladder falsifies both clauses of the "
                    "exonerate branch (f is non-monotone; the error "
                    "against the D4b reference increases over the four "
                    "finest rungs)."),
            "rule_source": "results/predeclared_windows_786.json "
                           ":: D2_edge_singularity (frozen, NOT rewritten)",
            "rule_text": {
                "exonerate_as_floor": win["exonerate_as_floor"],
                "attribute_partial": win["attribute_partial"],
                "inconclusive": win["inconclusive"],
            },
            "data_source": ["results/d5_turnover.json (nine-rung f(s))",
                            "results/d4_reference_b.json (f_inf, uniform)",
                            "results/d2_edge.json (p_smooth, unchanged)"],
            "new_fdtd_runs": 0,
        },
        "ladder": {
            "scales": [float(x) for x in scales],
            "three_over_s": [float(round(3.0 / x)) for x in scales],
            "h_mm": [float(x * 1e3) for x in h],
            "f_hz": [float(x) for x in f],
            "err_vs_f_inf_mhz": {("%.6f" % s): float(e / 1e6)
                                 for s, e in zip(scales, err)},
        },
        "clauses": {
            "f_monotone": trace_mono,
            "err_decreasing_at_every_rung": trace_err_decreasing,
            "err_increases_at_scales": [float(scales[i + 1])
                                        for i in range(len(derr))
                                        if derr[i] > 0],
            "p_trace_loglog_9rung": p_trace,
            "p_trace_window": [p_lo, p_hi],
            "p_smooth": p_smooth,
            "p_smooth_min": p_smooth_min,
            "exonerate_branch_satisfied": bool(
                trace_mono and trace_err_decreasing and p_trace >= 1.0),
            "attribute_branch_satisfied": attribute_partial,
        },
        "five_rung_values_for_comparison": {
            "p_trace_loglog_5rung": d4b["p_from_loglog_vs_f_inf"],
            "f_monotone_5rung": d2["D2_p_trace_from_D4b"]["f_monotone"],
            "err_decreasing_5rung":
                d2["D2_p_trace_from_D4b"][
                    "err_vs_independent_reference_decreasing"],
        },
        "verdict": verdict,
        "verdict_letter": ("EXONERATED" if "EXONERATED" in verdict
                           else ("ATTRIBUTED" if attribute_partial
                                 else "INCONCLUSIVE")),
        "argument_not_measurement": (
            "The wedge THEORY stated in the pre-declaration still stands on "
            "its own: a Meixner exponent nu = 2/3 at the trace's 3*pi/2 "
            "conductor wedges predicts a leading O(h^{4/3}) frequency error, "
            "i.e. a REDUCED ORDER, and an order reduction cannot by itself "
            "produce a non-vanishing floor. That is an ARGUMENT from theory, "
            "not a measured exoneration. On the nine-rung ladder this lane "
            "has NO measurement that exonerates candidate (2), and D6 could "
            "not identify the low-order exponent either (RMS_M1 = 4.361 MHz "
            "against a 1 MHz window; a = 0.5 fits better than 4/3)."),
        "smooth_control_unchanged": {
            "p_smooth_analytic": d2["D2B_empty_box_smooth_control"][
                "p_smooth_analytic"],
            "p_smooth_measured": d2["D2B_empty_box_smooth_control"][
                "p_smooth_measured"],
            "note": ("measured on the exact-reference vacuum twin, which "
                     "has an EXACT reference at every rung and is not "
                     "affected by the ladder-anchor problem"),
        },
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)

    print("D2 re-take on the nine-rung ladder (rule unchanged, frozen)")
    for s, hh, e in zip(scales, h, err):
        print("  3/s=%-3.0f s=%-9.6f h=%.6f mm  err_vs_f_inf=%9.3f MHz"
              % (round(3.0 / s), s, hh * 1e3, e / 1e6))
    print("  f monotone                 : %s" % trace_mono)
    print("  err decreasing every rung  : %s (increases at %s)"
          % (trace_err_decreasing,
             out["clauses"]["err_increases_at_scales"]))
    print("  p_trace (log-log, 9 rungs) : %.4f   [5-rung value was %.4f]"
          % (p_trace, d4b["p_from_loglog_vs_f_inf"]))
    print("  p_smooth                   : %.4f (>= %.1f)"
          % (p_smooth, p_smooth_min))
    print("\nD2 (re-taken):", verdict)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
