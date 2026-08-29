"""D2 (issue #786) — edge singularity.

Two smooth-field controls, both in the same machinery as the W4R ladder:

  D2-A  the SAME box, dielectric stack, port pair, probe, T_TOTAL,
        subpixel setting and scale set, with ONLY the PEC trace deleted.
        Bring-up (recorded in the note, §5): with the trace gone the
        4.0-6.5 GHz band is EMPTY -- the trace is what puts a resonance
        there -- and the lowest line the anti-symmetric port excites is
        at 11.79 GHz. The control therefore tracks THAT line, in the
        pre-declared control band 10-13 GHz. It is a dielectric-loaded
        box mode: no metal edge anywhere in the interior.
  D2-B  the D4a empty-box twin, which is a smooth-field ladder at 5.54
        GHz -- the SAME band, dt and record length as the W4R rung --
        and carries an EXACT reference, so its order needs no fit.

p_trace comes from D4b (the with-trace ladder judged against its own
independent Richardson reference), so run D4b first.

SUPERSEDED VERDICT (post-review, 2026-08-30). This module evaluates the
frozen D2 rule on the FIVE scales that existed when it first ran
(s = 1.5, 1.0, 0.75, 0.6, 0.5). D5 later added 3/s = 7, 8, 9, and on the
resulting nine-rung ladder both clauses of the exonerate branch fail
(f is non-monotone; the error against the D4b reference rises over the
four finest rungs) and p_trace falls to 0.95, which is the rule's own
INCONCLUSIVE clause. The lane's D2 verdict is therefore the RE-TAKEN one
in ``d2_retake.py`` / ``results/d2_edge_retake.json``; the verdict string
this module writes into ``d2_edge.json`` is kept unedited as the
five-rung record it is. Run d2_retake AFTER d5.

Run:  PYTHONPATH=. python -m validation.research.convergence_floor.d2_edge
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np

from validation.research.convergence_floor import fixture as fx
from validation.research.convergence_floor import estimators as est

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "d2_edge.json")
WINDOWS = os.path.join(RES, "predeclared_windows_786.json")

CTRL_BAND = (10.0e9, 13.0e9)      # frozen in the note §5 addendum
CTRL_SEARCH = (9.0e9, 14.0e9)


def control_rung(scale: float) -> dict:
    prof = fx.pc_uniform_profile(scale)
    sim = fx.build_sim(scale, prof, with_trace=False)
    sim.preflight()
    import time
    t0 = time.time()
    res = sim.run(n_steps=fx.n_steps_for(scale, float(prof.min())),
                  subpixel_smoothing=fx.SUBPIXEL)
    wall = time.time() - t0
    modes = sorted([m for m in res.find_resonances(freq_range=CTRL_SEARCH)
                    if abs(m.Q) > fx.Q_MIN], key=lambda m: m.freq)
    f, dom, info = fx.target_line(modes, CTRL_BAND)
    return {"scale": scale, "f_target": f, "dominance": dom,
            "in_band": info["in_band"], "dt": float(res.dt),
            "wallclock_s": wall,
            "modes": [(float(m.freq), float(m.Q), float(abs(m.amplitude)))
                      for m in modes[:6]]}


def main():
    warnings.filterwarnings("ignore")
    win = json.load(open(WINDOWS))["D2_edge_singularity"]
    p_lo, p_hi = float(win["p_trace_lo"]), float(win["p_trace_hi"])
    p_smooth_min = float(win["p_smooth_min"])

    rows = [control_rung(s) for s in sorted(fx.SCALES, reverse=True)]
    for r in rows:
        print("no-trace s=%-5s f=%.6f GHz dom=%s wall=%.0fs"
              % (r["scale"], r["f_target"] / 1e9, r["dominance"],
                 r["wallclock_s"]), flush=True)

    h = np.array([fx.PC_DZF0 * r["scale"] for r in rows])
    f = np.array([r["f_target"] for r in rows])
    monotone = bool(all(f[i] < f[i + 1] for i in range(len(f) - 1))
                    or all(f[i] > f[i + 1] for i in range(len(f) - 1)))
    fit = est.fit_power_law(h, f)
    err = np.abs(f - fit["f_inf_hz"])
    p_smooth_A = est.fit_order_loglog(h, err)

    out = {"issue": 786, "discriminator": "D2",
           "control_band_hz": list(CTRL_BAND),
           "D2A_no_trace": {
               "rows": rows, "monotone": monotone,
               "f_inf_hz": fit["f_inf_hz"], "p_nls": fit["p"],
               "rms_residual_hz": fit["rms_residual_hz"],
               "p_smooth_loglog": p_smooth_A,
               "err_vs_f_inf_mhz": {str(r["scale"]): float(e) / 1e6
                                    for r, e in zip(rows, err)}}}

    # p_trace and the D4a smooth control, if their arms have already run.
    p_trace = None
    trace_mono = None
    trace_err_decreasing = None
    b = os.path.join(RES, "d4_reference_b.json")
    if os.path.exists(b):
        d4b = json.load(open(b))["D4b"]["uniform"]
        p_trace = d4b["p_from_loglog_vs_f_inf"]
        fv = d4b["f_hz"]
        trace_mono = bool(all(fv[i] < fv[i + 1] for i in range(len(fv) - 1)))
        e = [d4b["err_vs_f_inf_mhz"][str(s)] for s in d4b["scales"]]
        trace_err_decreasing = bool(all(e[i] > e[i + 1]
                                        for i in range(len(e) - 1)))
        out["D2_p_trace_from_D4b"] = {
            "p_trace": p_trace, "f_monotone": trace_mono,
            "err_vs_independent_reference_decreasing": trace_err_decreasing,
            "err_mhz": d4b["err_vs_f_inf_mhz"]}
    a = os.path.join(RES, "d4_reference_a.json")
    if os.path.exists(a):
        d4a = json.load(open(a))["D4a"]
        out["D2B_empty_box_smooth_control"] = {
            "p_smooth_analytic": d4a["p_smooth_analytic"],
            "p_smooth_measured": d4a["p_smooth_measured"]}

    p_smooth = max([x for x in (p_smooth_A,
                                out.get("D2B_empty_box_smooth_control", {})
                                .get("p_smooth_measured"))
                    if x is not None and np.isfinite(x)], default=float("nan"))
    if p_trace is None:
        verdict = "PENDING (run D4b first for p_trace)"
    elif trace_mono and trace_err_decreasing and p_trace >= 1.0:
        verdict = ("EXONERATED as the FLOOR mechanism: with-trace f(s) is "
                   "monotone and its error against the independent "
                   "reference decreases at every rung (p_trace=%.2f). A "
                   "wedge exponent can only reduce the ORDER; it cannot "
                   "produce a non-vanishing floor." % p_trace)
        if p_lo <= p_trace <= p_hi and p_smooth >= p_smooth_min:
            verdict += (" ORDER-REDUCTION ALSO ATTRIBUTED: p_trace=%.2f is "
                        "in [%.1f, %.1f] (Meixner 2*nu=4/3) while the "
                        "smooth control gives p_smooth=%.2f >= %.1f."
                        % (p_trace, p_lo, p_hi, p_smooth, p_smooth_min))
    elif p_trace < 1.0:
        verdict = "INCONCLUSIVE (p_trace=%.2f < 1.0)" % p_trace
    else:
        verdict = ("INCONCLUSIVE (p_trace=%.2f, monotone=%s, err decreasing=%s)"
                   % (p_trace, trace_mono, trace_err_decreasing))
    out["p_smooth_used"] = p_smooth
    out["verdict"] = verdict
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nD2:", verdict)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
