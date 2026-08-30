"""D5 (issue #786) — is the s=0.25 rung ISOLATED, or is the ladder TURNING OVER?

PRE-DECLARATION AND HARNESS IN ONE FILE, committed BEFORE it is run
(``main()`` writes the windows file first and refuses to run if the
windows file already exists with different content).

WHY D5 EXISTS. D1 exonerated geometry quantization (delta 7e-6 cells),
D3 exonerated port loading (coupling span 0.6-3.5 kHz against a
predicted exactly-zero), and D4a/D4c exonerated the extraction
instrument (eps_instr 6-17 kHz on an exact-reference twin; three
independent estimators agree with the incumbent to 0.07 MHz on the
s=0.25 record itself). So the s=0.25 record really does contain
5.520821 GHz, 31.4 MHz below what a single-power-law fit to the five
coarser rungs extrapolates to (D4b). Two mutually exclusive readings
remain, and no discriminator so far separates them:

  T. TURN-OVER. f(h) is the sum of TWO error terms of opposite sign and
     different order (e.g. a positive O(h^4/3) edge term and a negative
     O(h^2) bulk term). Such a curve rises, peaks, and then descends to
     f_inf from above. The five-rung ladder would then be entirely
     PRE-ASYMPTOTIC, its power-law fit would extrapolate to the wrong
     limit, and the s=0.25 rung would be a perfectly good solve sitting
     past the peak.
  I. ISOLATED. f(s) keeps rising through every scale the ladder can
     resolve and only the s=0.25 rung drops -- a rung-local anomaly,
     i.e. a solver-side defect at fine grids.

THE MEASUREMENT. The uniform arm at the three lattice-valid scales
between 0.5 and 0.25 that the ladder skipped: s in {3/7, 0.375, 1/3}.
Lattice validity is the SAME arithmetic D1 verified -- 3/s and 6/s must
both be integers so every declared feature realizes exactly -- and these
are 7/14, 8/16, 9/18, continuing the ladder's own 2/4, 3/6, 4/8, 5/10,
6/12 and the reference's 12/24. No new geometry, no new instrument.

THE WINDOW (structural; no magnitude, so nothing in it can come from the
symptom numbers). Order the full uniform ladder by decreasing s:
1.5, 1.0, 0.75, 0.6, 0.5, 3/7, 0.375, 1/3, 0.25, and take the sequence
of consecutive differences.

  TURN-OVER CONFIRMED  iff the differences have EXACTLY ONE sign change
      AND the descending branch holds at least TWO rungs (so the turn is
      resolved by more than the reference rung alone). Reading T: the
      s=0.25 rung is on-curve, and the ladder is pre-asymptotic.
  ISOLATED ANOMALY     iff the differences have TWO OR MORE sign changes,
      OR the descending branch holds exactly one rung (only s=0.25
      falls). Reading I: escalate as a rung-local solver anomaly.
  INCONCLUSIVE         otherwise (e.g. no sign change at all).

Optional extension under the SAME window: s = 0.3 (3/s = 10, 6/s = 20).

WHAT A TURN-OVER VERDICT DOES AND DOES NOT ESTABLISH (post-review,
2026-08-30). Confirming the turn-over establishes that the s=0.25 anchor
is on-curve and past the maximum, hence that |f(s) - f(0.25)| is not an
error sequence and the "~4e-3 floor" figure is not an error. It does NOT
establish that f(h) converges to the physical resonance: this lane has no
external reference, and a consistency floor (f_floor != f_physical) is
fully compatible with a turn-over. The ledger's measured
DIELECTRIC-interface staircasing floor for this fixture class stands as
evidence that a real floor may exist here.

Run:  PYTHONPATH=. python -m validation.research.convergence_floor.d5_predeclare_and_run [--with-0.3]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from validation.research.convergence_floor import fixture as fx

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
WIN = os.path.join(RES, "predeclared_windows_786_D5.json")
OUT = os.path.join(RES, "d5_turnover.json")

EXT_SCALES = (3.0 / 7.0, 0.375, 1.0 / 3.0)
OPT_SCALE = 0.3


def windows() -> dict:
    return {
        "issue": 786, "discriminator": "D5", "predeclared_utc": "2026-08-30",
        "question": ("is the s=0.25 rung ISOLATED (rung-local solver "
                     "anomaly) or is the ladder TURNING OVER (two error "
                     "terms of opposite sign, ladder pre-asymptotic)?"),
        "scales_added": list(EXT_SCALES),
        "optional_scale": OPT_SCALE,
        "lattice_validity": ("3/s and 6/s integer -- the same arithmetic D1 "
                             "verified; 7/14, 8/16, 9/18 (and 10/20)"),
        "derivation": "structural (a shape test; no magnitude window)",
        "turn_over_confirmed": ("exactly ONE sign change in the consecutive "
                                "differences of f over the full ladder "
                                "ordered by decreasing s, AND the descending "
                                "branch holds >= 2 rungs"),
        "isolated_anomaly": ("TWO OR MORE sign changes, OR the descending "
                             "branch holds exactly one rung"),
        "inconclusive": "otherwise",
    }


def main():
    fx.quiet_third_party_warnings()
    os.makedirs(RES, exist_ok=True)
    w = windows()
    if os.path.exists(WIN):
        old = json.load(open(WIN))
        if old != w:
            raise SystemExit("D5 windows on disk differ from the code -- "
                             "refusing to run (windows are frozen)")
    else:
        with open(WIN, "w") as fh:
            json.dump(w, fh, indent=1)
        print("wrote", WIN)
    if "--windows-only" in sys.argv:
        return

    scales = list(EXT_SCALES) + ([OPT_SCALE] if "--with-0.3" in sys.argv
                                 else [])
    d0 = json.load(open(os.path.join(RES, "d0_reproduction.json")))
    known = {r["scale"]: r["f_target"] for r in d0["rows"]
             if not r["multiband"]}

    rows = []
    if os.path.exists(OUT):
        rows = json.load(open(OUT)).get("rows", [])
        for r in rows:
            known[r["scale"]] = r["f_target"]
    have = {r["scale"] for r in rows}
    for s in scales:
        if s in have:
            continue
        r = fx.measure(s, multiband=False)
        rows.append(r)
        known[s] = r["f_target"]
        print("UC s=%.6f (3/s=%.0f) f=%.6f GHz dom=%s cells=%d wall=%.0fs"
              % (s, 3.0 / s, r["f_target"] / 1e9, r["dominance"],
                 r["cells"], r["wallclock_s"]), flush=True)

    ss = sorted(known, reverse=True)
    f = np.array([known[s] for s in ss])
    diffs = np.diff(f)
    signs = np.sign(diffs)
    changes = int(np.sum(signs[1:] != signs[:-1]))
    # descending branch = the trailing run of negative differences
    desc = 0
    for d in diffs[::-1]:
        if d < 0:
            desc += 1
        else:
            break
    peak_scale = ss[int(np.argmax(f))]
    if changes == 1 and desc >= 2:
        verdict = ("TURN-OVER CONFIRMED: one sign change, descending branch "
                   "holds %d rungs (peak at s=%.4f). The s=0.25 rung is "
                   "ON-CURVE and the 5-rung ladder is PRE-ASYMPTOTIC."
                   % (desc, peak_scale))
    elif changes >= 2 or desc == 1:
        verdict = ("ISOLATED ANOMALY: sign changes=%d, descending branch=%d "
                   "rung(s). Escalate as a rung-local solver anomaly."
                   % (changes, desc))
    else:
        verdict = "INCONCLUSIVE (sign changes=%d, descending=%d)" % (changes,
                                                                     desc)
    out = {"issue": 786, "discriminator": "D5", "rows": rows,
           "ladder_scales": ss,
           "ladder_f_hz": [float(x) for x in f],
           "consecutive_diffs_hz": [float(x) for x in diffs],
           "sign_changes": changes, "descending_branch_rungs": desc,
           "peak_scale": peak_scale, "verdict": verdict}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print()
    for s, x in zip(ss, f):
        print("  s=%-9.6f 3/s=%-4.0f f=%.6f GHz" % (s, 3.0 / s, x / 1e9))
    print("\nD5:", verdict)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
