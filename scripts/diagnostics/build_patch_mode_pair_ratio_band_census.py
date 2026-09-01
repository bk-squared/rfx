#!/usr/bin/env python
"""Census: which mode-pair RATIO band widths separate cv15's correct build
from the issue-#740 realization?  (issue #812, cv05/cv15 lane, round 2)

WHY THIS EXISTS
---------------
Round 1 of this lane published a negative ABSOLUTE in
``docs/design_notes/20260901_patch_mode_identification_predeclaration.md``
sections 5 and 6.5: that the audit's proposed mode-pair ratio band "cannot
fire on #740 at any width that admits the correct build".  A reviewer refuted
it.  This script is the search that REFUTES rather than the one that failed to
confirm: it enumerates the admissible band half-widths in closed form from the
already-committed measurement, and reports the interval endpoints as numbers so
the design note can cite artifact keys instead of restating digits.

NO FDTD.  It post-processes ``tests/fixtures/patch_mode_identification/
cv15_ringdown_spectra.json`` (the committed live reproduction of the #740
realization through cv15's production builder) plus cv15's own declared
constants.  Every value it writes is re-derived and pinned by
``tests/test_patch_mode_identification.py``.

Usage::

    python scripts/diagnostics/build_patch_mode_pair_ratio_band_census.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "validation", "crossval",
                                "comparators"))

from patch_mode_identification import (   # noqa: E402
    declared_cavity_spectrum, identification_tolerance, members_in_band,
)

FIXTURE = os.path.join(REPO_ROOT, "tests", "fixtures",
                       "patch_mode_identification", "cv15_ringdown_spectra.json")
OUT = os.path.join(REPO_ROOT, "tests", "fixtures",
                   "patch_mode_identification", "cv15_mode_pair_ratio_band.json")

# cv15's DECLARED constants (validation/crossval/15_patch_antenna_rt5880.py).
CV15 = dict(eps_r=2.2, h=3.175e-3, a=40.0e-3, b=50.0e-3, c0=2.99792458e8,
            f_lo=1.6e9, f_hi=3.4e9)

PAIR = ((0, 1), (1, 0))   # the mode PAIR the audit proposed banding: TM010, TM100


def _git(*args):
    try:
        return subprocess.run(("git",) + args, cwd=REPO_ROOT, text=True,
                              capture_output=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


def _pair_ratio(freqs):
    """TM100/TM010 from a sorted-ascending measured ring-down list.

    The two lowest in-band modes ARE TM010 and TM100 in both realizations --
    that is what the identification in the fixture establishes; this function
    only reads them off, it does not re-decide the assignment.
    """
    fs = sorted(float(f) for f in freqs)
    return fs[1] / fs[0], fs[0], fs[1]


def main():
    fx = json.load(open(FIXTURE, encoding="utf-8"))
    members = members_in_band(
        declared_cavity_spectrum(CV15["eps_r"], CV15["h"], CV15["a"], CV15["b"],
                                 c0=CV15["c0"]),
        CV15["f_lo"], CV15["f_hi"])
    r_decl = members[PAIR[1]] / members[PAIR[0]]
    tol_id = identification_tolerance(members)

    out = {}
    for key, leg in (("correct_build", "two_plane_ground"),
                     ("defect_740", "one_plane_ground_740_defect")):
        r, f_lo, f_hi = _pair_ratio(m["freq_hz"] for m in fx[leg]["modes"])
        out[key] = dict(
            source_fixture_key=leg,
            TM010_hz=f_lo, TM100_hz=f_hi, pair_ratio=r,
            residual_vs_declared=r / r_decl - 1.0,
            abs_residual_vs_declared=abs(r / r_decl - 1.0),
        )

    w_admit = out["correct_build"]["abs_residual_vs_declared"]
    w_reject = out["defect_740"]["abs_residual_vs_declared"]
    w_measured_anchored = abs(out["defect_740"]["pair_ratio"]
                              / out["correct_build"]["pair_ratio"] - 1.0)

    doc = {
        "_what": ("Closed-form census of mode-pair RATIO band half-widths for "
                  "cv15 (TM100/TM010): which widths admit the correct build AND "
                  "reject the issue-#740 one-plane-ground realization. No FDTD; "
                  "post-processes the committed live reproduction."),
        "_why": ("This artifact is the evidence that REFUTES the negative "
                 "absolute published in round 1 of this lane (design note "
                 "sections 5 and 6.5, withdrawn in section 6.9): 'a ratio band "
                 "cannot fire on #740 at any width that admits the correct "
                 "build'. The admissible-width interval below is NON-EMPTY, so "
                 "that claim was false. What the measurement does support is "
                 "recorded in 'verdict'."),
        "_command": ("python scripts/diagnostics/"
                     "build_patch_mode_pair_ratio_band_census.py"),
        "_inputs": {
            "ringdown_fixture": ("tests/fixtures/patch_mode_identification/"
                                 "cv15_ringdown_spectra.json"),
            "declared_constants": dict(CV15),
            "pair": ["TM010", "TM100"],
        },
        "_provenance": {
            "repo_commit": _git("rev-parse", "HEAD"),
            "repo_dirty": bool(_git("status", "--porcelain")),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "declared": {
            "TM010_hz": members[PAIR[0]],
            "TM100_hz": members[PAIR[1]],
            "pair_ratio": r_decl,
            "identification_tolerance": tol_id,
        },
        "measured": out,
        "declared_anchored_band": {
            "_definition": ("gate |r_measured / r_declared - 1| <= w, with "
                            "r_declared from the declared geometry alone"),
            "min_half_width_admitting_correct_build": w_admit,
            "max_half_width_still_rejecting_740": w_reject,
            "admissible_interval_is_nonempty": bool(w_admit < w_reject),
            "detection_ratio": w_reject / w_admit,
            "upper_endpoint_over_identification_tolerance": w_reject / tol_id,
        },
        "measured_anchored_band": {
            "_definition": ("gate |r_measured / r_correct_build - 1| <= w, i.e. "
                            "a band centred on the correct build's own measured "
                            "ratio -- admits it for every w >= 0"),
            "defect_offset_from_correct_build": w_measured_anchored,
            "max_half_width_still_rejecting_740": w_measured_anchored,
        },
        "verdict": (
            "A declared-anchored mode-pair ratio band with half-width w both "
            "admits the correct build and rejects the #740 realization for every "
            "w in [min_half_width_admitting_correct_build, "
            "max_half_width_still_rejecting_740). That interval is non-empty, so "
            "the round-1 absolute is false and is withdrawn. It is nonetheless "
            "not a window this lane may adopt: both endpoints are properties of "
            "the two measurements the band would judge (burned-data, SPEC-00 "
            "0.2.2), the interval's upper endpoint is "
            "upper_endpoint_over_identification_tolerance of the tightest window "
            "derivable from declared geometry alone, and the run-to-run "
            "reproducibility of the measured pair ratio has not been established "
            "at that scale by any committed run. Criterion (B) for cv15 is "
            "therefore NOT met by a window whose provenance this lane can supply: "
            "cv15 lands as a STOP."),
    }
    with open(OUT, "w", encoding="utf-8") as fp:
        json.dump(doc, fp, indent=2)
        fp.write("\n")
    print(f"wrote {os.path.relpath(OUT, REPO_ROOT)}")
    print(json.dumps(doc["declared_anchored_band"], indent=2))


if __name__ == "__main__":
    main()
