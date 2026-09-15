#!/usr/bin/env python3
"""#498 — zero-compute re-analysis of the two runs this lane has already paid for.

    PYTHONPATH=. JAX_PLATFORMS=cpu python scripts/diagnostics/mixed_anchor_reanalysis.py

No solver. No VESSL. No GPU. Pure NumPy over two artifacts that are already on
``main``; it finishes in about a second.

SOURCES OF RECORD (both committed, both read-only here)
  A. scripts/diagnostics/_mixed_refplane_logs/measurement_369367257597_60p.json
     — the reference-plane-instrumented mixed run, VESSL 369367257597,
       tree 38c7552b, jax 0.4.33, numpy 1.26.4, complex64, wall 1366.78 s.
  B. scripts/diagnostics/_probe_fed_msl_referee_logs/stage2_369367257643_referee.json
     — the openEMS external referee Stage 2, VESSL 369367257643
       (Stage 1 reproduce gate: VESSL 369367257598 / 369367251705 / 369367246713).
  Predeclaration: docs/design_notes/mixed_refplane_predeclaration.md

REPORT-ONLY. THIS SCRIPT PINS NOTHING.
Predeclaration §10 ("what must NOT be pinned by any of this"), restated verbatim
in :data:`DO_NOT_PIN`, binds every number below — in particular §10 item 1 (no
lumped/wire diagonal value may be pinned until the PI sequencing decision on
#776/#778 + the parked #683 flip) and §10 item 11 (a "consistent" branch must
never be written up as "vindicated").

WHAT IT REPORTS, AND WHAT EACH REPORT IS WORTH
  §1  The MSL-diagonal anchor sweep against the same-run plane referee M2.
      The agreement at the measured Zc is real, and it is ANCHOR-CIRCULAR: M2
      is itself built by ``refplane_split`` at that same measured Zc, so the
      sweep bounds the anchor to roughly 57–58 Ω and NO TIGHTER. It is not
      independent evidence that 57.93 Ω is the right anchor.
  §2  The #460 Kurokawa renormalization of the openEMS comparator. It is a real
      comparator bug fix and it does NOT restore passivity: balance
      1.0106–1.0579, |S21| > 1 at 48/48 bins, and a residual that DRIFTS with
      frequency — the signature #498's own 2026-08-03 comment records as
      falsifying a constant per-port rescale. Normalization was one defect;
      fixing it exposes a second (the 2026-09-04 audit's D1 branch).
  §3  The flux bracket 1 − (1 − |S22|²)·r1/r0, and the competing hypothesis the
      box-widening plan omitted: r1 > 1 is non-physical ON ITS OWN, and a
      box-independent third instrument (the two committed full-cross-section
      planes at x = 1.44 / 2.56 mm) localizes about half the excess to the box
      and about half to the planes. Neither half is box WIDTH, and widening the
      box has the wrong sign for both.
  §4  rfx vs openEMS, symmetrized. REPORTED, never a verdict — the comparator is
      still non-passive after §2.

Nothing here is a gate, a reference, or a verdict on rfx.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MEAS_JSON = (REPO / "scripts" / "diagnostics" / "_mixed_refplane_logs"
             / "measurement_369367257597_60p.json")
REFEREE_JSON = (REPO / "scripts" / "diagnostics" / "_probe_fed_msl_referee_logs"
                / "stage2_369367257643_referee.json")
PREDECLARATION = REPO / "docs" / "design_notes" / "mixed_refplane_predeclaration.md"
OUT_DIR = REPO / "scripts" / "diagnostics" / "_mixed_anchor_reanalysis"

VESSL_IDS = {
    "mixed_refplane_measurement": 369367257597,
    "openems_referee_stage2": 369367257643,
    "openems_referee_stage1": 369367257598,
    "openems_stage1_a1_reproduction": 369367251705,
    "openems_stage1_a2_reproduction": 369367246713,
}

# Predeclaration §10, verbatim headings. Nothing this script prints may move any
# of these, and nothing downstream of it may pin them either.
DO_NOT_PIN = [
    "1. Any lumped/wire (lw) diagonal value — the shipped 0.38–0.40, the "
    "predicted 0.21 +/- 0.03, sqrt(n_live)-rescaled variants, or anything the "
    "external referee reads. The PI sequencing decision on #776/#778 + the "
    "parked #683 uniform flip is undecided; F4 is a prediction, not a result.",
    "2. reciprocity_tol = 0.06 — stays exactly as shipped.",
    "3. The docs' 9.0 % / 55 % quotes, and the lane's known_limits text.",
    "4. The mixed lane's reference_plane_cells rejection itself.",
    "5. |S22| = 0.02–0.03 — must not become a reference or 'validated' number.",
    "6. Zc_meas / beta from this run — must not replace the analytic "
    "Hammerstad–Jensen anchor anywhere in shipped code.",
    "7. Any openEMS number — comparator leg only; never a gate, never a "
    "reference fixture.",
    "8. cond(A), settling_db, wall clock, the inter-surface offsets — "
    "reported, never gated.",
    "9. num_periods = 60 / the 5-bin frequency set / n_probes = 3.",
    "10. No snapshot re-capture, no reference regeneration, no tolerance edit, "
    "no support-matrix status change.",
    "11. F2's 'consistent' branch must not be written up as 'vindicated'.",
]

# The anchors the sweep is run over. All four are already recorded on main;
# none of them is proposed as a replacement for the shipped analytic anchor.
ANCHORS_OHM = {
    "hj_declared_600x254um": 47.89479996289313,   # the SHIPPED analytic anchor
    "hj_realized_cell_560x320um": 57.463,
    "zc_measured_two_plane": None,                # filled from the run itself
    "hj_realized_node_480x320um": 62.652,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _cx(pairs) -> np.ndarray:
    a = np.asarray(pairs, dtype=float)
    return a[..., 0] + 1j * a[..., 1]


def _band(x) -> list:
    x = np.asarray(x, dtype=float)
    return [float(np.min(x)), float(np.max(x))]


def _row(vals, fmt="%9.5f") -> str:
    return " ".join(fmt % v for v in np.asarray(vals, dtype=float))


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"MISSING ARTIFACT: {path}\n"
                         "This script reads committed artifacts only; it "
                         "cannot regenerate them and must not try.")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# §1 — the MSL-diagonal anchor sweep, and why it is anchor-circular
# ---------------------------------------------------------------------------
def section1_anchor(meas: dict) -> dict:
    f_ghz = np.asarray(meas["fixture"]["freqs_hz"], dtype=float) / 1e9
    mc = meas["msl_channel"]
    n_f = len(f_ghz)
    v = _cx(np.asarray(mc["v0_msl"]).reshape(2, 1, n_f, 2))
    i = _cx(np.asarray(mc["i_msl"]).reshape(2, 1, n_f, 2))
    msl_run = int(meas["refplane"]["msl_drive_run"])
    V, I = v[msl_run, 0], i[msl_run, 0]

    zin = V / I
    zc_meas = float(np.mean(np.real(_cx(meas["refplane"]["runs"][msl_run]["zc"]))))
    anchors = dict(ANCHORS_OHM)
    anchors["zc_measured_two_plane"] = float(
        np.real(_cx(meas["refplane"]["runs"][0]["zc"]))[0])

    m2 = np.asarray(meas["refplane"]["runs"][msl_run]["out_over_inc"]["0"],
                    dtype=float)
    sweep = {}
    for name, z in anchors.items():
        g = np.abs((V - z * I) / (V + z * I))
        sweep[name] = {
            "z_ohm": float(z),
            "abs_S22": g.tolist(),
            "band": _band(g),
            "dev_vs_M2_pct": (np.abs(g - m2) / m2 * 100.0).tolist(),
            "max_dev_vs_M2_pct": float(np.max(np.abs(g - m2) / m2 * 100.0)),
        }

    print("\n" + "=" * 78)
    print("§1  MSL DIAGONAL vs THE SAME-RUN PLANE REFEREE M2  (VESSL %d)"
          % VESSL_IDS["mixed_refplane_measurement"])
    print("=" * 78)
    print("  f [GHz]                   ", _row(f_ghz, "%9.2f"))
    print("  Zin = V0/I at MSL probe 0 (x = 4.72 mm), MSL-driven run:")
    print("     Re                     ", _row(np.real(zin)))
    print("     Im                     ", _row(np.imag(zin)))
    print("     |Im/Re| [%]            ",
          _row(np.abs(np.imag(zin) / np.real(zin)) * 100.0))
    print("  -> the issue body's '|V/I| ~ 591 ohm and strongly reactive' does "
          "NOT reproduce on this fixture.")
    print("     That reading was the July near-open end-fed geometry; the "
          "premise is retired with a number.")
    print()
    print("  |S22| = |(V0 - Z*I)/(V0 + Z*I)| at each anchor, against M2:")
    for name, rec in sweep.items():
        print("    %-30s Z=%8.4f  " % (name, rec["z_ohm"])
              + _row(rec["abs_S22"], "%8.4f")
              + "   max dev vs M2 %6.2f%%" % rec["max_dev_vs_M2_pct"])
    print("    %-30s            " % "M2 (plane referee, out_over_inc[0])"
          + _row(m2, "%8.4f"))
    print("    %-30s            " % "abs_S22_raw (SHIPPED, HJ anchor)"
          + _row(mc["abs_S22_raw"], "%8.4f"))
    print()
    print("  READING — stated as a bound, not as a result:")
    print("   * M2 is built by refplane_split(v, i_corr, zc, sign) at the SAME")
    print("     two-plane measured Zc this sweep re-derives at "
          "(mixed_refplane_measurement.py:503-506, :519-524). The agreement is")
    print("     therefore ANCHOR-CIRCULAR: it checks that two V/I ratios on one")
    print("     low-loss line are mutually consistent with ONE Zc. It is NOT an")
    print("     independent measurement that 57.93 ohm is the right anchor.")
    print("   * The realized-CELL Hammerstad-Jensen anchor 57.463 ohm reproduces")
    print("     M2 essentially as well (max dev %.2f%% vs %.2f%%)."
          % (sweep["hj_realized_cell_560x320um"]["max_dev_vs_M2_pct"],
             sweep["zc_measured_two_plane"]["max_dev_vs_M2_pct"]))
    print("     The sweep bounds the anchor to ~57-58 ohm and NO TIGHTER.")
    print("   * What IS settled by it: the shipped declared-board anchor")
    print("     47.895 ohm is 3.5-5.2x away from M2 at every bin, which is far")
    print("     outside the ~57-58 ohm band. 'The algebra is correct' is NOT")
    print("     established here and is not claimed.")
    print("   * Predeclaration §10 item 6 stands: Zc_meas must not replace the")
    print("     analytic anchor in shipped code. Nothing here proposes that.")

    return {
        "freqs_ghz": f_ghz.tolist(),
        "zin_msl_probe0_ohm": [[float(z.real), float(z.imag)] for z in zin],
        "zin_im_over_re_pct":
            (np.abs(np.imag(zin) / np.real(zin)) * 100.0).tolist(),
        "issue_591_ohm_premise": "DOES NOT REPRODUCE (Re Zin 45.1-49.6 ohm, "
                                 "|Im/Re| 1.7-3.9%)",
        "M2_plane_referee": m2.tolist(),
        "abs_S22_shipped": list(mc["abs_S22_raw"]),
        "zc_measured_mean_ohm": zc_meas,
        "anchor_sweep": sweep,
        "reading": (
            "ANCHOR-CIRCULAR. M2 is built by refplane_split at the same "
            "measured Zc, so the 0.2-3.6% agreement is a consistency check of "
            "one anchor propagated to two planes, not independent evidence "
            "for 57.93 ohm. The realized-cell HJ anchor 57.463 ohm reproduces "
            "M2 comparably. The sweep bounds the anchor to ~57-58 ohm and no "
            "tighter. It does NOT establish 'the MSL extractor algebra is "
            "correct'."),
    }


# ---------------------------------------------------------------------------
# §2 — the openEMS comparator's missing power-wave normalization
# ---------------------------------------------------------------------------
def section2_openems(referee: dict) -> dict:
    legs = referee["stage2"]["legs"]
    out = {}
    print("\n" + "=" * 78)
    print("§2  openEMS COMPARATOR: THE MISSING KUROKAWA sqrt(Re Z_j/Re Z_i)  "
          "(VESSL %d)" % VESSL_IDS["openems_referee_stage2"])
    print("=" * 78)
    print("  probe_fed_msl_openems_referee.py assembled S as bare "
          "uf_ref_i/uf_inc_j across two")
    print("  ports with UNEQUAL reference impedances (lumped Feed_R pinned at "
          "50.0; MSL port's own")
    print("  measured ZL recorded per bin). That is the #460 class the mixed "
          "lane fixes internally.")
    for name, leg in legs.items():
        f = np.asarray(leg["freqs_hz"], dtype=float) / 1e9
        s11, s21, s12 = _cx(leg["s11"]), _cx(leg["s21"]), _cx(leg["s12"])
        rz = np.real(_cx(leg["z0_msl_measured_ohm"]))
        s21n = s21 * np.sqrt(50.0 / rz)
        s12n = s12 * np.sqrt(rz / 50.0)
        bal_raw = np.asarray(leg["passivity_balance_verbatim"], dtype=float)
        bal_new = np.abs(s11) ** 2 + np.abs(s21n) ** 2
        ratio_raw = np.abs(s21) / np.abs(s12)
        ratio_new = np.abs(s21n) / np.abs(s12n)
        over = bal_new > 1.05
        rec = {
            "role": leg.get("role"),
            "dx_m": leg.get("dx_m"),
            "n_bins": int(f.size),
            "re_z_msl_band_ohm": _band(rz),
            "verbatim_balance_reproduced_from_s11_s21_maxdiff":
                float(np.max(np.abs(bal_raw - (np.abs(s11) ** 2
                                               + np.abs(s21) ** 2)))),
            "abs_s21_raw_band": _band(np.abs(s21)),
            "abs_s12_raw_band": _band(np.abs(s12)),
            "reciprocity_ratio_raw_band": _band(ratio_raw),
            "abs_s21_renormalized_band": _band(np.abs(s21n)),
            "reciprocity_ratio_renormalized_band": _band(ratio_new),
            "balance_raw_band": _band(bal_raw),
            "balance_renormalized_band": _band(bal_new),
            "n_bins_balance_over_1p05": int(np.count_nonzero(over)),
            "freqs_ghz_balance_over_1p05": f[over].tolist(),
            "n_bins_abs_s21_renormalized_over_unity":
                int(np.count_nonzero(np.abs(s21n) > 1.0)),
            "residual_drift_per_bin": ratio_new.tolist(),
            "residual_drift_band": _band(ratio_new),
            "residual_drift_min_at_ghz": float(f[int(np.argmin(ratio_new))]),
            "residual_drift_monotone_after_min": bool(
                np.all(np.diff(ratio_new[int(np.argmin(ratio_new)):]) > 0)),
            "freqs_ghz": f.tolist(),
            "abs_s21_renormalized": np.abs(s21n).tolist(),
            "balance_renormalized": bal_new.tolist(),
        }
        out[name] = rec
        print("\n  --- %s (dx = %g m, %s) ---" % (name, leg["dx_m"],
                                                  leg.get("role")))
        print("      Re Z_msl (openEMS MSLPort.ZL, the independent witness): "
              "%.3f .. %.3f ohm" % tuple(rec["re_z_msl_band_ohm"]))
        print("      passivity_balance_verbatim reproduced from s11/s21 to "
              "%.1e  (it IS |S11|^2+|S21|^2)"
              % rec["verbatim_balance_reproduced_from_s11_s21_maxdiff"])
        print("      BEFORE:  |S21| %.4f..%.4f  |S12| %.4f..%.4f  "
              "|S21|/|S12| %.4f..%.4f  balance %.4f..%.4f"
              % (*rec["abs_s21_raw_band"], *rec["abs_s12_raw_band"],
                 *rec["reciprocity_ratio_raw_band"], *rec["balance_raw_band"]))
        print("      AFTER :  |S21| %.4f..%.4f                        "
              "|S21|/|S12| %.4f..%.4f  balance %.4f..%.4f"
              % (*rec["abs_s21_renormalized_band"],
                 *rec["reciprocity_ratio_renormalized_band"],
                 *rec["balance_renormalized_band"]))
        print("      per-bin trace (R5 — the full %d bins, not a band):"
              % rec["n_bins"])
        print("        f[GHz]  " + _row(f, "%7.2f"))
        print("        |S21n|  " + _row(np.abs(s21n), "%7.4f"))
        print("        balance " + _row(bal_new, "%7.4f"))
        print("        drift   " + _row(ratio_new, "%7.4f"))
        print("      -> balance over 1.05 at %d of %d bins%s"
              % (rec["n_bins_balance_over_1p05"], rec["n_bins"],
                 (" (f = " + ", ".join("%.1f" % x
                                       for x in rec["freqs_ghz_balance_over_1p05"])
                  + " GHz)") if rec["n_bins_balance_over_1p05"] else ""))
        print("      -> |S21_renormalized| > 1 at %d of %d bins"
              % (rec["n_bins_abs_s21_renormalized_over_unity"], rec["n_bins"]))
        print("      -> residual %.4f .. %.4f, minimum at %.1f GHz, monotone "
              "above it: %s"
              % (*rec["residual_drift_band"], rec["residual_drift_min_at_ghz"],
                 rec["residual_drift_monotone_after_min"]))

    print()
    print("  READING — this is a REPORT, not a gate:")
    print("   * The Kurokawa factor is a REAL comparator bug and it is fixed "
          "(probe_fed_msl_openems_referee.py,")
    print("     kurokawa_renormalize / --renormalize-json). It removes most of "
          "the reciprocity asymmetry.")
    print("   * It does NOT restore passivity. On the comparator leg the "
          "balance is 1.0106-1.0579 —")
    print("     still above 1.05 at 5 of 48 bins — and |S21| stays above 1 at "
          "ALL 48 bins.")
    print("   * The residual DRIFTS with frequency (1.0101 near 1 GHz -> "
          "1.0302 at 5 GHz, monotone above")
    print("     the minimum). #498's 2026-08-03 comment records that exact "
          "signature as falsifying a")
    print("     constant per-port impedance rescale: 'the required factor "
          "drifts 1.677->1.605 ... whatever")
    print("     closes it needs a frequency-dependent term, not only a "
          "Kurokawa sqrt(Z) correction.'")
    print("   * On the reported-only dx=80um leg the SAME correction drives "
          "|S21|/|S12| to 0.65-0.83,")
    print("     i.e. it does not make that leg reciprocal either — a witness "
          "against 'normalization")
    print("     explains everything'.")
    print("   * VERDICT: normalization was ONE defect of the comparator. "
          "Fixing it EXPOSES a second one.")
    print("     That second defect is the 2026-09-04 audit's own D1 branch, so "
          "the openEMS leg stays")
    print("     CLOSED. No new VESSL slot is justified by this. Phase is "
          "untouched by a positive real")
    print("     scale, so the ~1.29 mm de-embedding-length defect is also "
          "still open.")
    return out


# ---------------------------------------------------------------------------
# §3 — the flux bracket, and the competing hypothesis the box plan omitted
# ---------------------------------------------------------------------------
def section3_flux(meas: dict, sec1: dict) -> dict:
    f_ghz = np.asarray(meas["fixture"]["freqs_hz"], dtype=float) / 1e9
    lw = next(f for f in meas["flux"] if f["drive"] == "lw")
    ml = next(f for f in meas["flux"] if f["drive"] == "msl")

    def g(fl, key, exact=True):
        return np.asarray(fl[key + ("_exact_f64" if exact else "_default")],
                          dtype=float)

    box_lw, pm_lw = g(lw, "box_net"), g(lw, "plane_msl")
    px_lw, mx_lw = g(lw, "plane_px_2p56mm"), g(lw, "plane_mx_1p44mm")
    box_ml, pm_ml = g(ml, "box_net"), g(ml, "plane_msl")
    px_ml, mx_ml = g(ml, "plane_px_2p56mm"), g(ml, "plane_mx_1p44mm")

    r0 = pm_lw / box_lw          # P_arr,msl / P_net,lw   (lw drive)
    r1 = box_ml / pm_ml          # P_arr,lw  / P_net,msl  (MSL drive)

    zc = sec1["anchor_sweep"]["zc_measured_two_plane"]["z_ohm"]
    s22_zc = np.asarray(sec1["anchor_sweep"]["zc_measured_two_plane"]["abs_S22"],
                        dtype=float)
    s22_ship = np.asarray(sec1["abs_S22_shipped"], dtype=float)

    def bracket(s22, ratio):
        return 1.0 - (1.0 - np.asarray(s22, dtype=float) ** 2) * ratio

    # --- the box-independent third instrument -----------------------------
    # The two committed full-cross-section x-planes at 1.44 and 2.56 mm bound a
    # slab that contains the whole lw port. Power balance over that slab uses
    # NO face of the 5-face box.
    q_lw = px_lw - mx_lw              # net power OUT through the slab's x faces
    q_ml = (-px_ml) + mx_ml           # net power IN  through the slab's x faces
    box_over_slab_lw = box_lw / q_lw
    box_over_slab_ml = np.abs(box_ml) / q_ml
    r0_planes = pm_lw / q_lw
    r1_planes = q_ml / np.abs(pm_ml)

    # --- what a common multiplicative instrument bias would have to be -----
    need_bracket = np.sqrt((r1 / r0) * (1.0 - s22_zc ** 2))   # plane_msl scale-up
    need_physical = float(np.max(r1))                          # r1 <= 1
    m_needed = float(np.max(need_bracket))
    resid_after_box = np.sqrt((r1_planes / r0_planes) * (1.0 - s22_zc ** 2))

    print("\n" + "=" * 78)
    print("§3  THE FLUX BRACKET, AND THE HYPOTHESIS THE BOX-WIDENING PLAN "
          "OMITTED")
    print("=" * 78)
    print("  All numbers from the committed exact_f64 flux faces of VESSL %d."
          % VESSL_IDS["mixed_refplane_measurement"])
    print("  r0 = plane_msl(lw drive) / box_net(lw drive)")
    print("  r1 = box_net(MSL drive)  / plane_msl(MSL drive)")
    print()
    print("  f [GHz]        " + _row(f_ghz, "%10.2f"))
    print("  r0             " + _row(r0, "%10.5f"))
    print("  r1             " + _row(r1, "%10.5f"))
    print("  r1/r0          " + _row(r1 / r0, "%10.5f"))
    print("  |S22| @ Zc     " + _row(s22_zc, "%10.5f"))
    print("  bracket @Zc    " + _row(bracket(s22_zc, r1 / r0), "%+10.5f"))
    print("  bracket @ship  " + _row(bracket(s22_ship, r1 / r0), "%+10.5f"))
    print("  bracket @|S|=0 " + _row(bracket(np.zeros_like(r0), r1 / r0),
                                     "%+10.5f"))
    print("  NEGATIVE at every bin at EVERY anchor, including |S22| = 0: no "
          "real |S00| closes it.")
    print()
    print("  (A) r1 > 1 IS NON-PHYSICAL ON ITS OWN — no box geometry needed to "
          "see it.")
    print("      In the MSL drive the only source sits at x = 5.52 mm, EAST of "
          "the MSL reference")
    print("      plane. Let R be everything west of that plane. The net inward "
          "flux across the plane")
    print("      IS the total time-averaged dissipation in R (resistor + "
          "whatever CPML lies in R);")
    print("      the net inward flux through the lw box IS the dissipation "
          "inside the box, and the")
    print("      box (x 1.76-2.24, y 1.26-1.74, z 0-0.494 mm) is a SUBREGION "
          "of R. Dissipation in a")
    print("      subregion cannot exceed dissipation in the region containing "
          "it, so r1 <= 1 with no")
    print("      losslessness assumption at all. Measured: %.5f .. %.5f — a "
          "%.2f%% violation."
          % (float(np.min(r1)), float(np.max(r1)), (need_physical - 1) * 100))
    print("      Sign check on the plan's mechanism: a WIDER box captures MORE, "
          "so box_net grows on")
    print("      both drives; r0 = plane/box falls, r1 = box/plane rises, and "
          "r1/r0 rises as the")
    print("      square. Widening therefore makes r1 MORE non-physical and the "
          "bracket MORE negative.")
    print()
    print("  (B) A BOX-INDEPENDENT THIRD INSTRUMENT. The run also committed "
          "two FULL-cross-section")
    print("      x-planes at 1.44 mm and 2.56 mm (plane_mx / plane_px). They "
          "bound a slab containing")
    print("      the entire lw port and share NO face with the 5-face box.")
    print("      MSL drive, nested-region power budget. D(x<X) = net WESTWARD "
          "flux at X = the total")
    print("      dissipation west of X (the only source is at x = 5.52 mm). "
          "It must be >= 0 and must")
    print("      not DECREASE as X grows. Normalised to D(x<4.72 mm) = 1:")
    print("        D(x<1.44)  " + _row(-mx_ml / np.abs(pm_ml), "%10.5f")
          + "   <-- NEGATIVE: %.2f%% of the scale" % float(
              np.max(np.abs(mx_ml / np.abs(pm_ml))) * 100))
    print("        D(x<2.56)  " + _row(np.abs(px_ml) / np.abs(pm_ml), "%10.5f")
          + "   <-- EXCEEDS D(x<4.72) = 1")
    print("        D(box)     " + _row(np.abs(box_ml) / np.abs(pm_ml), "%10.5f")
          + "   <-- the box, a SUBREGION, exceeds both")
    print("      lw drive, same planes, source INSIDE the slab: eastward flux "
          "falls 2.3587e-28 ->")
    print("      2.3540e-28 from 2.56 to 4.72 mm (-0.20%, physical) and "
          "box_net >= the sum of the")
    print("      slab's two outflows. Every lw-drive budget is consistent; "
          "all three violations")
    print("      above are on the MSL drive only.")
    print()
    print("      box_net / (slab x-face budget), the tightest legitimate bound "
          "using both planes:")
    print("        lw drive  " + _row(box_over_slab_lw, "%10.5f"))
    print("        MSL drive " + _row(box_over_slab_ml, "%10.5f"))
    print("      The same factor on two independent drives, to %.3f%%. On the "
          "MSL drive the box is a"
          % float(np.max(np.abs(box_over_slab_lw - box_over_slab_ml)) * 100))
    print("      SUBREGION of that slab, so this ratio MUST be <= 1 — it is a "
          "hard violation, the box")
    print("      reading %.3f%% high. On the lw drive the source is inside and "
          "the same excess is"
          % float((np.max(box_over_slab_ml) - 1.0) * 100))
    print("      degenerate with real y/z leakage, so it is not a violation "
          "there — only a match.")
    print("      Mechanism available at zero compute: the box's +/-x faces and "
          "its +y face are")
    print("      PIERCED BY PEC. The realized trace is y in [1.28, 1.76] mm "
          "while the box faces sit")
    print("      at y = 1.26 / 1.74 mm, centred on the DECLARED y_c = 1.50 mm, "
          "not the realized 1.52 —")
    print("      so one of the seven metal node rows is excluded asymmetrically "
          "and the face integrals")
    print("      are sampled at a PEC edge. That is a CLOSURE defect of the "
          "5-face surface, not a")
    print("      width defect, and widening the faces does not remove it.")
    print()
    print("  (C) REPLACE THE BOX WITH THE PLANE BUDGET AND THE BRACKET IS "
          "STILL NEGATIVE.")
    print("      r0* = plane_msl(lw)/slab_out(lw); r1* = "
          "slab_in(MSL)/|plane_msl(MSL)|")
    print("      r0*            " + _row(r0_planes, "%10.5f"))
    print("      r1*            " + _row(r1_planes, "%10.5f"))
    print("      r1*/r0*        " + _row(r1_planes / r0_planes, "%10.5f"))
    print("      bracket* @Zc   " + _row(bracket(s22_zc,
                                                 r1_planes / r0_planes),
                                         "%+10.5f"))
    print("      That removes %.0f-%.0f%% of the log-excess in r1/r0 and the "
          "bracket is STILL negative"
          % (float(np.min(1 - np.log(r1_planes / r0_planes)
                          / np.log(r1 / r0)) * 100),
             float(np.max(1 - np.log(r1_planes / r0_planes)
                          / np.log(r1 / r0)) * 100)))
    print("      at all 5 bins, and r1* = %.5f is STILL > 1. So roughly half "
          "the anomaly is in the"
          % float(np.max(r1_planes)))
    print("      box and roughly half is in the plane instruments themselves.")
    print()
    print("  (D) WHAT BIAS MAGNITUDE WOULD BE NEEDED (the #838 family, asked "
          "and answered).")
    print("      r1/r0 = P_box(msl)*P_box(lw) / "
          "(P_plane_msl(msl)*P_plane_msl(lw)), so a COMMON")
    print("      multiplicative factor m on plane_msl (true = m x recorded) "
          "scales it by 1/m^2:")
    print("        m >= %.4f  (%.2f%%) makes the bracket positive at all 5 bins"
          % (m_needed, (m_needed - 1) * 100))
    print("        m >= %.4f  (%.2f%%) is independently required just to make "
          "r1 <= 1"
          % (need_physical, (need_physical - 1) * 100))
    print("      The two thresholds agree to %.2f%% — the single bias that "
          "removes the non-physical"
          % (abs(need_physical - m_needed) * 100.0))
    print("      gain in r1 also flips the bracket's sign, with NO box change "
          "at all.")
    print("      Equivalently a common factor b on box_net: b <= %.4f flips "
          "the bracket, b <= %.4f"
          % (float(1.0 / m_needed), float(1.0 / need_physical)))
    print("      makes r1 physical. After the box is replaced by the plane "
          "budget a residual")
    print("      m* >= %.4f (%.2f%%) is still required."
          % (float(np.max(resid_after_box)),
             (float(np.max(resid_after_box)) - 1) * 100))
    print("      HONEST CAVEAT ON #838: #838's measured ~3x over-read is in "
          "the MSL port's POWER-WAVE")
    print("      normalization (_b_msl = (V0 - Z_hj*I)/(2*sqrt(Z_hj))), and "
          "its own six-face flux box")
    print("      closed to -0.04/+0.06/+0.20%. plane_msl here is a POYNTING "
          "flux, not a power wave, so")
    print("      #838's stated mechanism does not bias it directly. What is "
          "measured above is a")
    print("      ~1.3% box-vs-plane and a ~1.2% plane-vs-plane POYNTING "
          "inconsistency. #838 remains")
    print("      the nearest open precedent for 'a constant in a port "
          "normalization', not a proven cause.")
    print()
    print("  (E) DTYPE WITNESS (R5). Every surface above was recorded twice "
          "by the driver, in the")
    print("      lane default and in exact_f64. Max relative spread between "
          "the two:")
    for lbl, fl in (("lw ", lw), ("msl", ml)):
        spread = {k: float(np.max(np.abs(g(fl, k, False) - g(fl, k))
                                  / np.abs(g(fl, k))))
                  for k in ("box_net", "plane_msl", "plane_px_2p56mm",
                            "plane_mx_1p44mm")}
        print("        %s  " % lbl + "  ".join("%s %.2e" % (k, v)
                                               for k, v in spread.items()))
    print("      The two large surfaces are tight to <= 5e-05. plane_mx on the "
          "MSL drive spreads by")
    print("      2.2e-02, but it carries only ~1% of the slab budget, so it "
          "moves that budget by")
    print("      ~0.02% — two orders below the 1.31% violation in (B). The "
          "findings do not rest on it.")
    print()
    print("  (F) CONSEQUENCE FOR THE PROPOSED WIDER-BOX RUN: it must NOT be "
          "launched as designed.")
    print("      Net flux through a CLOSED surface in a source-free lossless "
          "region is invariant to")
    print("      widening it, which is exactly what a 2% two-box agreement "
          "gate asserts; and the")
    print("      under-capture mechanism has the wrong sign for both (A) and "
          "(D). A box run is")
    print("      justified only after someone names a mechanism by which a "
          "closed-surface flux moves")
    print("      by the ~2.2% the bracket needs. None is named here.")

    return {
        "freqs_ghz": f_ghz.tolist(),
        "r0": r0.tolist(), "r1": r1.tolist(), "r1_over_r0": (r1 / r0).tolist(),
        "r1_is_non_physical": bool(np.any(r1 > 1.0)),
        "r1_max_violation_pct": float((need_physical - 1.0) * 100.0),
        "abs_S22_at_zc": s22_zc.tolist(),
        "zc_ohm": zc,
        "bracket_at_zc": bracket(s22_zc, r1 / r0).tolist(),
        "bracket_at_shipped": bracket(s22_ship, r1 / r0).tolist(),
        "bracket_at_zero_s00": bracket(np.zeros_like(r0), r1 / r0).tolist(),
        "msl_drive_nested_budget_normalized_to_D_at_4p72mm": {
            "D_x_lt_1p44mm": (-mx_ml / np.abs(pm_ml)).tolist(),
            "D_x_lt_2p56mm": (np.abs(px_ml) / np.abs(pm_ml)).tolist(),
            "D_x_lt_4p72mm": [1.0] * int(f_ghz.size),
            "D_box": (np.abs(box_ml) / np.abs(pm_ml)).tolist(),
            "required": "0 <= D(x<1.44) <= D(x<2.56) <= D(x<4.72), and "
                        "D(box) <= D(x<4.72) since the box is a subregion",
            "violations": "D(x<1.44) is negative; D(x<2.56) exceeds "
                          "D(x<4.72); D(box) exceeds both",
        },
        "box_over_slab_budget_lw": box_over_slab_lw.tolist(),
        "box_over_slab_budget_msl": box_over_slab_ml.tolist(),
        "r0_planes_only": r0_planes.tolist(),
        "r1_planes_only": r1_planes.tolist(),
        "bracket_planes_only_at_zc":
            bracket(s22_zc, r1_planes / r0_planes).tolist(),
        "log_excess_fraction_attributable_to_box":
            (1 - np.log(r1_planes / r0_planes) / np.log(r1 / r0)).tolist(),
        "plane_msl_scale_up_needed_for_positive_bracket": m_needed,
        "plane_msl_scale_up_needed_for_r1_physical": need_physical,
        "box_scale_down_needed_for_positive_bracket": float(1.0 / m_needed),
        "box_scale_down_needed_for_r1_physical": float(1.0 / need_physical),
        "residual_plane_side_scale_after_box_removed":
            float(np.max(resid_after_box)),
        "default_vs_f64_max_rel_dev": {
            "lw": {k: float(np.max(np.abs(g(lw, k, False) - g(lw, k))
                                   / np.abs(g(lw, k))))
                   for k in ("box_net", "plane_msl", "plane_px_2p56mm",
                             "plane_mx_1p44mm")},
            "msl": {k: float(np.max(np.abs(g(ml, k, False) - g(ml, k))
                                    / np.abs(g(ml, k))))
                    for k in ("box_net", "plane_msl", "plane_px_2p56mm",
                              "plane_mx_1p44mm")},
        },
        "reading": (
            "r1 = 1.0248-1.0257 > 1 is non-physical on its own. A "
            "box-independent three-plane budget puts ~52% of the log-excess "
            "in the box and ~48% in the planes; the bracket stays negative at "
            "all 5 bins with the box removed from the arithmetic. A common "
            "2.20% plane_msl scale-up flips the bracket and a 2.57% one is "
            "independently required for r1 <= 1. Box WIDTH is not the "
            "mechanism, and widening has the wrong sign. DO NOT LAUNCH the "
            "wider-box run as designed."),
    }


# ---------------------------------------------------------------------------
# §4 — rfx vs openEMS, symmetrized. Reported, never a verdict.
# ---------------------------------------------------------------------------
def section4_cross(meas: dict, referee: dict) -> dict:
    f_rfx = np.asarray(meas["fixture"]["freqs_hz"], dtype=float)
    s = _cx(np.asarray(meas["s_matrix"]["S_raw"]).reshape(2, 2, f_rfx.size, 2))
    rfx_gm = np.sqrt(np.abs(s[1, 0]) * np.abs(s[0, 1]))

    leg = referee["stage2"]["legs"]["comparator_dx50um"]
    f_oe = np.asarray(leg["freqs_hz"], dtype=float)
    oe_gm_all = np.sqrt(np.abs(_cx(leg["s21"])) * np.abs(_cx(leg["s12"])))
    idx = [int(np.argmin(np.abs(f_oe - f))) for f in f_rfx]
    oe_gm = oe_gm_all[idx]

    print("\n" + "=" * 78)
    print("§4  rfx vs openEMS, SYMMETRIZED — REPORTED, NOT A VERDICT")
    print("=" * 78)
    print("  f [GHz]                " + _row(f_rfx / 1e9, "%9.2f"))
    print("  rfx  sqrt(|S10||S01|)  " + _row(rfx_gm))
    print("  oe   sqrt(|S21||S12|)  " + _row(oe_gm))
    print("  ratio rfx/openEMS      " + _row(rfx_gm / oe_gm))
    print("  The committed artifact's RAW comparison band is abs_s21_ratio "
          "[0.8082, 0.8195];")
    print("  symmetrizing removes ~15 of those 19 points. That is a statement "
          "about the")
    print("  COMPARATOR's asymmetry, not evidence that the two solvers agree.")
    print("  §2 shows the openEMS leg is STILL non-passive after the "
          "normalization fix, so")
    print("  predeclaration §7.4 and the lane's 'absolute |S| stays "
          "UNVALIDATED' sentence both")
    print("  stand unchanged. No openEMS number is pinned (§10 item 7).")
    return {
        "freqs_ghz": (f_rfx / 1e9).tolist(),
        "rfx_geometric_mean_s21": rfx_gm.tolist(),
        "openems_geometric_mean_s21_nearest_bin": oe_gm.tolist(),
        "openems_bins_ghz_used": (f_oe[idx] / 1e9).tolist(),
        "ratio_rfx_over_openems": (rfx_gm / oe_gm).tolist(),
        "status": "REPORTED ONLY. Not a verdict on agreement; the comparator "
                  "is non-passive even after the #460 fix (§2).",
    }


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--no-write", action="store_true",
                    help="print everything, write nothing")
    args = ap.parse_args(argv)

    meas = _load(MEAS_JSON)
    referee = _load(REFEREE_JSON)

    print("=" * 78)
    print("#498 ZERO-COMPUTE RE-ANALYSIS — REPORT-ONLY, PINS NOTHING")
    print("=" * 78)
    print("  sources (committed, read-only):")
    print("    %s   [VESSL %d]" % (MEAS_JSON.relative_to(REPO),
                                   VESSL_IDS["mixed_refplane_measurement"]))
    print("    %s   [VESSL %d]" % (REFEREE_JSON.relative_to(REPO),
                                   VESSL_IDS["openems_referee_stage2"]))
    print("    %s" % PREDECLARATION.relative_to(REPO))
    print("  tree of the measurement run: %s / jax %s / numpy %s / %s"
          % (meas["bookkeeping"]["git_sha"][:8],
             meas["bookkeeping"]["jax_version"],
             meas["bookkeeping"]["numpy"],
             meas["bookkeeping"]["field_dtype"]))
    print("  settling witness (both drives, rule %s dB): %s"
          % (meas["witnesses"]["settling_rule_db"],
             meas["witnesses"]["settling_db"]))
    print("  NO SOLVER RUNS HERE. No VESSL, no GPU, no openEMS import.")
    print("\n  PREDECLARATION §10 — WHAT MUST NOT BE PINNED BY ANY OF THIS:")
    for item in DO_NOT_PIN:
        print("    " + item)

    sec1 = section1_anchor(meas)
    sec2 = section2_openems(referee)
    sec3 = section3_flux(meas, sec1)
    sec4 = section4_cross(meas, referee)

    doc = {
        "what_this_is": "#498 zero-compute re-analysis of two committed "
                        "artifacts. REPORT-ONLY; pins nothing; moves no gate.",
        "sources": {"measurement": str(MEAS_JSON.relative_to(REPO)),
                    "openems_referee": str(REFEREE_JSON.relative_to(REPO)),
                    "predeclaration": str(PREDECLARATION.relative_to(REPO))},
        "vessl_run_ids_read_from_the_artifacts": VESSL_IDS,
        "do_not_pin": DO_NOT_PIN,
        "section1_anchor": sec1,
        "section2_openems_kurokawa": sec2,
        "section3_flux_bracket": sec3,
        "section4_cross_solver_reported_only": sec4,
        "headline": (
            "(1) The MSL anchor agreement is ANCHOR-CIRCULAR and bounds the "
            "anchor to ~57-58 ohm, no tighter. "
            "(2) The openEMS comparator's missing Kurokawa factor is a real "
            "bug and fixing it leaves balance 1.0106-1.0579, |S21| > 1 at "
            "48/48 bins, and a frequency-DRIFTING residual: normalization was "
            "one defect, a second remains, the openEMS leg stays closed. "
            "(3) The lw-diagonal flux bracket is negative because r1 > 1 is "
            "non-physical and the box and plane Poynting instruments disagree "
            "by ~1.3% and ~1.2%; box WIDTH is not the mechanism and the "
            "wider-box run must not be launched as designed."),
    }
    if not args.no_write:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "anchor_reanalysis.json"
        dest.write_text(json.dumps(doc, indent=2, default=str))
        print("\n=== Written to %s ===" % dest)
    print("\nREPORT-ONLY. Nothing above is pinned, gated, or promoted. "
          "Predeclaration §10 items 1 and 11 in particular:")
    print("no lw-diagonal value is pinned, and no branch is written up as "
          "'vindicated'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
