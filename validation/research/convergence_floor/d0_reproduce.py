"""D0 (issue #786) — reproduce the W4R symptom with the copied fixture.

Re-runs the uniform and multiband ladders and the s=0.25 reference, and
compares every rung with PR #785's committed ``w4r_supraconvergence.json``
under the frozen 1 kHz determinism tolerance. Also stores every rung's raw
probe record so D4c can re-extract from the identical data for free.

Run:  PYTHONPATH=. python -m validation.research.convergence_floor.d0_reproduce
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from validation.research.convergence_floor import fixture as fx

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "d0_reproduction.json")
SERIES = os.path.join(RES, "d0_records.npz")
WINDOWS = os.path.join(RES, "predeclared_windows_786.json")

# PR #785 committed values (branch agent/multiband-nu-envelope,
# validation/research/multiband_nu/results/w4r_supraconvergence.json).
# Transcribed here so this lane does not depend on that branch being
# checked out; D0's whole job is to reproduce them.
PR785 = {
    ("UC", 0.25): 5520820807.318383,
    ("UC", 0.5): 5543558293.505646,
    ("UC", 0.6): 5542444437.5726,
    ("UC", 0.75): 5533066155.959281,
    ("UC", 1.0): 5502114526.260705,
    ("UC", 1.5): 5404546169.206672,
    ("MB", 0.5): 5538672695.290948,
    ("MB", 0.6): 5536597841.756963,
    ("MB", 0.75): 5525828863.445815,
    ("MB", 1.0): 5492732260.12526,
    ("MB", 1.5): 5391247629.600432,
}


def rejudge():
    """Recompute the deltas in an existing d0_reproduction.json against the
    full-precision PR #785 table (no FDTD re-run). Used once, because the
    first pass transcribed all but the reference rung rounded to 1 Hz."""
    with open(WINDOWS) as fh:
        tol = float(json.load(fh)["D0_reproduction"]["tol_hz"])
    out = json.load(open(OUT))
    for r in out["rows"]:
        key = ("MB" if r["multiband"] else "UC", r["scale"])
        ref = PR785.get(key)
        r["pr785_f_target"] = ref
        r["delta_vs_pr785_hz"] = (abs(r["f_target"] - ref)
                                  if ref is not None else None)
        r["reproduced"] = (r["delta_vs_pr785_hz"] is not None
                           and r["delta_vs_pr785_hz"] <= tol)
        print("%s s=%-5s f=%.9f GHz  d(PR785)=%.4f Hz  %s"
              % (key[0], r["scale"], r["f_target"] / 1e9,
                 r["delta_vs_pr785_hz"], r["reproduced"]))
    out["reproduced_all"] = all(r["reproduced"] for r in out["rows"])
    out["max_delta_vs_pr785_hz"] = max(r["delta_vs_pr785_hz"]
                                       for r in out["rows"])
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("reproduced_all =", out["reproduced_all"],
          " max delta = %.4f Hz" % out["max_delta_vs_pr785_hz"])


def main():
    fx.quiet_third_party_warnings()
    if "--rejudge" in sys.argv:
        return rejudge()
    with open(WINDOWS) as fh:
        win = json.load(fh)
    tol = float(win["D0_reproduction"]["tol_hz"])

    rows, records = [], {}
    order = ([(False, fx.REF_SCALE)]
             + [(False, s) for s in sorted(fx.SCALES, reverse=True)]
             + [(True, s) for s in sorted(fx.SCALES, reverse=True)])
    for mb, s in order:
        r = fx.measure(s, multiband=mb, keep_series=True)
        key = ("MB" if mb else "UC", s)
        records["%s_%s" % key] = r.pop("_series")
        records["%s_%s__dt" % key] = np.array([r["dt"]])
        ref = PR785.get(key)
        r["pr785_f_target"] = ref
        r["delta_vs_pr785_hz"] = (abs(r["f_target"] - ref)
                                  if ref is not None else None)
        r["reproduced"] = (r["delta_vs_pr785_hz"] is not None
                           and r["delta_vs_pr785_hz"] <= tol)
        rows.append(r)
        print("%s s=%-5s f=%.6f GHz  d(PR785)=%s Hz  dom=%s  wall=%.0fs"
              % (key[0], s, r["f_target"] / 1e9,
                 ("%.1f" % r["delta_vs_pr785_hz"]
                  if r["delta_vs_pr785_hz"] is not None else "n/a"),
                 r["dominance"], r["wallclock_s"]), flush=True)

    ok = all(r["reproduced"] for r in rows)
    # The symptom, restated on this lane's own data.
    uc = {r["scale"]: r["f_target"] for r in rows if not r["multiband"]}
    ladder = sorted([s for s in uc if s != fx.REF_SCALE], reverse=True)
    fseq = [uc[s] for s in ladder]
    monotone = all(fseq[i] < fseq[i + 1] for i in range(len(fseq) - 1))
    f_ref = uc[fx.REF_SCALE]
    err = {s: abs(uc[s] - f_ref) for s in ladder}
    err_monotone = all(err[ladder[i]] > err[ladder[i + 1]]
                       for i in range(len(ladder) - 1))

    out = {
        "issue": 786, "discriminator": "D0",
        "tol_hz": tol, "rows": rows,
        "reproduced_all": bool(ok),
        "uniform_f_sequence_ghz": {str(s): uc[s] / 1e9 for s in ladder},
        "uniform_f_monotone_increasing_as_s_falls": bool(monotone),
        "reference_f_ghz": f_ref / 1e9,
        "reference_below_two_finest_rungs": bool(
            f_ref < uc[ladder[-1]] and f_ref < uc[ladder[-2]]),
        "abs_error_vs_reference_mhz": {str(s): err[s] / 1e6 for s in ladder},
        "abs_error_monotone_decreasing": bool(err_monotone),
        "verdict": ("REPRODUCED: the |f - f_ref| sequence is NON-monotone "
                    "while f(s) itself is monotone" if ok and monotone
                    and not err_monotone else
                    ("REPRODUCED (all rungs match PR #785)" if ok
                     else "NOT REPRODUCED -- STOP")),
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    np.savez_compressed(SERIES, **records)
    print("\n", out["verdict"])
    print(" f(s) monotone increasing as s falls:", monotone)
    print(" |f-f_ref| monotone decreasing:", err_monotone)
    print(" reference below the two finest rungs:",
          out["reference_below_two_finest_rungs"])
    print("wrote", OUT, "and", SERIES)


if __name__ == "__main__":
    main()
