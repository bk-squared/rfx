"""#812 round 2 — the matched-frequency operands for cv03's dispersion claims.

Round 1 published a comparison whose two operands were at DIFFERENT
frequencies: a measured ``n_eff`` taken at DFT bin 24 was divided by the
analytic ``n_eff`` at ``f = fcen`` exactly.  ``n_eff`` rises with ``f``, so the
half-bin mismatch inflated the denominator and inverted the sign of the
deviation.  This builder emits every operand of that comparison, at matched
frequencies, into a committed JSON so the design notes can reference keys
instead of restating digits.

It runs NO FDTD.  The analytic side is the closed-form slab oracle already
committed in ``validation/crossval/comparators/slab_te_dispersion.py``; the
frequency grid is parsed out of the committed crossval script so that a drift
in the case makes this artifact stale rather than silently wrong.  The measured
side is carried forward verbatim from round 1's recorded table and is labelled
with that provenance -- see ``measured.provenance`` in the output.

Run:
    PYTHONPATH=<worktree> python scripts/diagnostics/cv03_flux/build_cv03_matched_frequency.py
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CASE = os.path.join(REPO, "validation", "crossval",
                    "03_straight_waveguide_flux.py")
OUT = os.path.join(REPO, "docs", "design_notes",
                   "issue812_cv03_dispersion_matched_frequency.json")

sys.path.insert(0, os.path.join(REPO, "validation", "crossval", "comparators"))
from slab_te_dispersion import slab_te0_neff          # noqa: E402

# Round 1's recorded time-of-flight table, verbatim from section 8.1 of
# docs/design_notes/issue812_cv03_dispersion_regate_predeclaration.md.
# Its driver was never committed, so these carry NOTE provenance, not harness
# provenance; regenerating them is what the reported VESSL job exists for.
ROUND1_ROWS = [
    # key, sx (a), DFT window (a/c0), fit window (Meep x), recorded n_eff, |B/A|
    ("sx40_dft150_before_round_trip", 40.0, 150.0, [-16.0, -8.0], 2.84338, 0.0002),
    ("sx40_dft400_after_round_trip",  40.0, 400.0, [-16.0, -8.0], 2.84318, 0.4979),
    ("sx16_dft400_recipe_baseline",   16.0, 400.0, [-4.0, +4.0],  2.84175, 0.5311),
    ("sx16_dft150_no_round_trip_yet", 16.0, 150.0, [-4.0, +4.0],  2.84474, 0.5233),
]

# The committed case run's own per-bin deviations over the gated band, as
# recorded in the criterion-(A) section of the results note.  Used here only as
# a cross-check that fixes the semantics of round 1's "measured n_eff" column.
RESULTS_NOTE_BAND_DEV_PCT = [
    0.013, 0.177, 0.160, 0.052, 0.128, 0.172, 0.101,
    0.132, 0.199, 0.150, 0.160, 0.262, 0.229, 0.180,
]


_NUM = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"


def _scalar(name):
    m = re.search(rf"^{name}\s*=\s*({_NUM})", open(CASE).read(), re.M)
    if m is None:
        raise SystemExit(f"{name} not found in {CASE} -- the case has drifted")
    return float(m.group(1))


def build():
    fcen = _scalar("fcen")
    df = _scalar("df")
    n_freqs = int(_scalar("n_freqs"))
    eps_core = _scalar("RECIPE_EPS_WG")
    eps_clad = _scalar("RECIPE_EPS_CLAD")
    width = _scalar("RECIPE_WG_WIDTH")
    tol_neff = _scalar("tol_neff")
    c0 = _scalar("C0")
    a = 1.0e-6

    f = np.linspace(fcen - df / 2.0, fcen + df / 2.0, n_freqs)
    neff = np.array([slab_te0_neff(eps_core, eps_clad, width * a,
                                   2.0 * np.pi * (x * c0 / a) / c0) for x in f])
    band = np.abs(f - fcen) <= 0.15 * df
    band_idx = [int(i) for i in np.nonzero(band)[0]]
    carrier = int(np.argmin(np.abs(f - fcen)))       # the case's own estimator bin

    rows = []
    for key, sx, dft_t, win, n_meas, b_over_a in ROUND1_ROWS:
        matched = float(neff[carrier])
        rows.append({
            "key": key,
            "sx_a": sx,
            "dft_window_a_over_c0": dft_t,
            "fit_window_meep_x_a": win,
            "b_over_a_recorded": b_over_a,
            "n_eff_rfx_carrier_bin": n_meas,
            "n_eff_analytic_matched": matched,
            "dev_pct_matched": 100.0 * (n_meas / matched - 1.0),
            "n_eff_analytic_at_fcen_exact": float(
                slab_te0_neff(eps_core, eps_clad, width * a,
                              2.0 * np.pi * (fcen * c0 / a) / c0)),
            "dev_pct_round1_mismatched": 100.0 * (
                n_meas / slab_te0_neff(eps_core, eps_clad, width * a,
                                       2.0 * np.pi * (fcen * c0 / a) / c0) - 1.0),
        })

    baseline = next(r for r in rows if r["key"] == "sx16_dft400_recipe_baseline")
    xcheck = {
        "what": ("fixes the semantics of round 1's 'measured n_eff' column: "
                 "recomputing the recipe-baseline row at the carrier bin must "
                 "reproduce the carrier-bin entry of the per-bin deviation list "
                 "the results note recorded from the committed case run"),
        "recomputed_dev_pct": baseline["dev_pct_matched"],
        "results_note_band_dev_pct_at_carrier_bin":
            RESULTS_NOTE_BAND_DEV_PCT[band_idx.index(carrier)],
        "abs_difference_pct": abs(
            baseline["dev_pct_matched"]
            - RESULTS_NOTE_BAND_DEV_PCT[band_idx.index(carrier)]),
    }

    refl_free = next(r for r in rows
                     if r["key"] == "sx40_dft150_before_round_trip")

    doc = {
        "schema": "issue812-cv03-matched-frequency-v1",
        "issue": 812,
        "lane": "cv03",
        "generator": "scripts/diagnostics/cv03_flux/build_cv03_matched_frequency.py",
        "runs_fdtd": False,
        "grid": {
            "source": "parsed from validation/crossval/03_straight_waveguide_flux.py",
            "fcen_c_over_a": fcen,
            "df_c_over_a": df,
            "n_freqs": n_freqs,
            "band_half_width_c_over_a": 0.15 * df,
            "band_bin_indices": band_idx,
            "carrier_bin_index": carrier,
            "carrier_bin_f_c_over_a": float(f[carrier]),
            "fcen_minus_carrier_bin_c_over_a": float(fcen - f[carrier]),
        },
        "oracle": {
            "recipe": {"eps_core": eps_core, "eps_clad": eps_clad,
                       "thickness_a": width},
            "n_eff_at_fcen_exact": rows[0]["n_eff_analytic_at_fcen_exact"],
            "n_eff_at_carrier_bin": float(neff[carrier]),
            "n_eff_band_mean": float(neff[band].mean()),
            "band_f_c_over_a": [float(v) for v in f[band]],
            "band_n_eff": [float(v) for v in neff[band]],
            # n_eff rising with f is WHY a half-bin frequency mismatch could
            # invert the sign of the round-1 deviation.
            "band_n_eff_rises_with_f": bool(np.all(np.diff(neff[band]) > 0.0)),
            "band_n_eff_increase_over_band": float(
                neff[band][-1] - neff[band][0]),
        },
        "measured": {
            "provenance": ("verbatim from section 8.1 of "
                           "docs/design_notes/issue812_cv03_dispersion_regate_"
                           "predeclaration.md (round 1). The driver that "
                           "produced these was not committed, so the measured "
                           "operands carry note provenance; the analytic "
                           "operands and every derived deviation in this file "
                           "are recomputed here from committed code."),
            "statistic": ("two-wave-fit n_eff at DFT bin "
                          f"{carrier} (f = {float(f[carrier]):.7f} c/a)"),
            "rows": rows,
        },
        "column_semantics_cross_check": xcheck,
        "round1_error": {
            "locations": [
                "docs/design_notes/issue812_cv03_dispersion_regate_predeclaration.md:300-303",
                "docs/design_notes/issue812_cv03_dispersion_regate_results.md:139-143",
            ],
            "published_dev_pct": -0.026,
            "published_margin_factor_vs_g1": 77,
            "mechanism": ("the measured operand is the carrier DFT bin, the "
                          "analytic operand was taken at f = fcen exactly; "
                          "n_eff is increasing in f, so the half-bin offset "
                          "both inflated the denominator and inverted the sign"),
            "reproduces_published_value": refl_free["dev_pct_round1_mismatched"],
            "corrected_dev_pct": refl_free["dev_pct_matched"],
            "margin_factor_withdrawn_because": (
                "G1's statistic is max |n_eff_rfx/n_eff_analytic - 1| over the "
                "gated band, not a carrier-bin deviation, so no margin factor "
                "against G1 may be read off a single bin. Criterion (A)'s "
                "margin is the committed case run's own band-max against "
                "tol_neff, which the results note already records."),
            "g1_tol_pct": 100.0 * tol_neff,
        },
        "falsified_conclusion": {
            "round1_claim": ("the recipe baseline's band-max deviation is what "
                             "the 16a domain and 400 a/c0 window cost, not what "
                             "the solver's dispersion costs"),
            "refutation": ("at matched frequency the carrier-bin deviation is "
                           "one-signed and of the same order in all four "
                           "configurations, and the reflection-free "
                           "configuration is NOT the smallest of them"),
            "dev_pct_matched_by_row": {r["key"]: r["dev_pct_matched"]
                                       for r in rows},
            "reflection_free_minus_baseline_pct":
                refl_free["dev_pct_matched"] - baseline["dev_pct_matched"],
            "all_same_sign": bool(len({np.sign(r["dev_pct_matched"])
                                       for r in rows}) == 1),
        },
    }
    return doc


def main():
    doc = build()
    with open(OUT, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"wrote {OUT}")
    r = doc["round1_error"]
    print(f"  published {r['published_dev_pct']:+.3f}%  "
          f"reproduced {r['reproduces_published_value']:+.4f}%  "
          f"corrected {r['corrected_dev_pct']:+.4f}%")
    x = doc["column_semantics_cross_check"]
    print(f"  column-semantics cross-check: {x['recomputed_dev_pct']:.4f}% vs "
          f"{x['results_note_band_dev_pct_at_carrier_bin']:.3f}% "
          f"(|diff| {x['abs_difference_pct']:.5f} pp)")


if __name__ == "__main__":
    main()
