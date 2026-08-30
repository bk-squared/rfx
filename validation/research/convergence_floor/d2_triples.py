"""D2-A successive-triple order estimates (issue #786, third-review repair).

WHY THIS EXISTS. §5 and §8.1 of the design note say the trace-free P-C
control's "successive-triple ratios put its order near 0.4-0.5". No
committed artifact held those ratios: ``d2_edge.json`` carries only
``p_nls`` (the 3-parameter NLS fit, 4.256, which the note itself reports
as poorly converged) and ``p_smooth_loglog`` (0.370, a log-log fit of
|f - f_inf| against an f_inf that came out of that same NLS fit). This
module computes the actual successive-triple orders and commits them, so
the sentence is either backed or corrected against a number on disk.

NO NEW SIMULATION. Every input is read from committed JSON
(``d2_edge.json`` for the D2-A ladder, ``d4_reference_a.json`` for the
D4a/D2-B twin). Nothing here defines or moves a pre-declared window: the
triple orders are a DIAGNOSTIC, reported and not judged, exactly as §5
says of the D2-A order.

METHOD. For f(h) = f_inf + C h^p on three rungs h1 > h2 > h3,

    (f1 - f2) / (f2 - f3) = (h1^p - h2^p) / (h2^p - h3^p)

has one root p > 0 whenever the sequence is monotone; the ladder's h
ratios are NOT constant (1.5, 1.333, 1.25, 1.2), so the textbook
constant-ratio formula p = log2(R) does not apply and the root is found
numerically.

INSTRUMENT CHECK (the reason the twin arm is here). The same estimator is
run on the D4a twin's ANALYTIC discretization error sequence, whose order
is known independently to be 2.0001 from the closed-form discrete
eigenfrequency. If the estimator does not return ~2 there, its output on
D2-A means nothing.

Run:  PYTHONPATH=. python -m validation.research.convergence_floor.d2_triples
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import brentq

from validation.research.convergence_floor import fixture as fx

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "d2a_triple_orders.json")


def triple_order(h: np.ndarray, f: np.ndarray) -> dict:
    """One (h1>h2>h3, f1,f2,f3) triple -> the p of f_inf + C h^p."""
    h1, h2, h3 = (float(x) for x in h)
    f1, f2, f3 = (float(x) for x in f)
    d12, d23 = f1 - f2, f2 - f3
    if d12 == 0.0 or d23 == 0.0 or (d12 > 0) != (d23 > 0):
        return {"p": None, "why": "non-monotone or degenerate triple",
                "ratio": None}
    ratio = d12 / d23

    def g(p: float) -> float:
        return (h1 ** p - h2 ** p) / (h2 ** p - h3 ** p) - ratio

    lo, hi = 1e-6, 40.0
    if g(lo) * g(hi) > 0:
        return {"p": None, "why": "no root in (1e-6, 40)", "ratio": ratio}
    p = brentq(g, lo, hi, xtol=1e-12, rtol=1e-12)
    C = d12 / (h1 ** p - h2 ** p)
    return {"p": float(p), "ratio": float(ratio), "C": float(C),
            "f_inf_hz": float(f1 - C * h1 ** p),
            "h_mm": [h1 * 1e3, h2 * 1e3, h3 * 1e3]}


def ladder_triples(scales, freqs, label) -> dict:
    order = np.argsort(-np.asarray(scales, float))     # coarse -> fine
    s = np.asarray(scales, float)[order]
    f = np.asarray(freqs, float)[order]
    h = fx.PC_DZF0 * s
    trips = []
    for i in range(len(s) - 2):
        t = triple_order(h[i:i + 3], f[i:i + 3])
        t["scales"] = [float(x) for x in s[i:i + 3]]
        trips.append(t)
    ps = [t["p"] for t in trips if t["p"] is not None]
    return {
        "label": label,
        "scales_coarse_to_fine": [float(x) for x in s],
        "dz_fine_mm": [float(x) * 1e3 for x in h],
        "f_hz": [float(x) for x in f],
        "triples": trips,
        "p_range": [min(ps), max(ps)] if ps else None,
        "p_drifts_upward": bool(
            len(ps) > 1 and all(ps[i] < ps[i + 1] for i in range(len(ps) - 1))),
    }


def main() -> None:
    fx.quiet_third_party_warnings()
    d2 = json.load(open(os.path.join(RES, "d2_edge.json")))
    rows = d2["D2A_no_trace"]["rows"]
    out = {
        "issue": 786,
        "what": "successive-triple order estimates, DIAGNOSTIC, not judged; "
                "no window is defined or moved here and no simulation is run",
        "d2a": ladder_triples([r["scale"] for r in rows],
                              [r["f_target"] for r in rows],
                              "D2-A trace-free P-C box, 11.79 GHz line"),
        "committed_comparators_in_d2_edge_json": {
            "p_nls": d2["D2A_no_trace"]["p_nls"],
            "p_nls_rms_residual_hz": d2["D2A_no_trace"]["rms_residual_hz"],
            "p_smooth_loglog": d2["D2A_no_trace"]["p_smooth_loglog"],
            "note": "p_smooth_loglog fits |f - f_inf| with the f_inf that "
                    "came out of the same poorly-converged NLS fit, so it "
                    "is not independent of p_nls",
        },
    }

    twin_path = os.path.join(RES, "d4_reference_a.json")
    if os.path.exists(twin_path):
        t = json.load(open(twin_path))["D4a"]
        tr = [r for r in t["rows"] if r["scale"] in fx.SCALES]
        out["instrument_check_twin"] = ladder_triples(
            [r["scale"] for r in tr],
            [r["f_discrete_exact_hz"] for r in tr],
            "D4a twin, ANALYTIC discrete eigenfrequency (known p = 2.0001)")
        out["instrument_check_twin"]["known_order"] = t["p_smooth_analytic"]

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    for key in ("instrument_check_twin", "d2a"):
        if key not in out:
            continue
        b = out[key]
        print("%s:" % b["label"])
        for t in b["triples"]:
            print("  s=%-22s ratio=%.4f  p=%s"
                  % (",".join("%g" % x for x in t["scales"]), t["ratio"],
                     ("%.4f" % t["p"]) if t["p"] is not None else t["why"]))
        print("  p range:", b["p_range"], " drifts upward:",
              b["p_drifts_upward"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
