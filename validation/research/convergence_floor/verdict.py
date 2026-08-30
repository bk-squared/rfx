"""Issue #786 — synthesis: apply the frozen windows, apportion, verdict.

Reads only the committed discriminator outputs; computes nothing new
except the apportionment arithmetic the pre-declaration fixed.

SCOPE DISCIPLINE (post-review revision, 2026-08-30). This file writes the
lane's human-readable verdict, so it is where an overclaim does the most
damage. Three rules it now obeys:

1. This lane never measured a floor's ABSENCE. It has no external
   reference: nothing here pins f_inf independently of a fitted model.
   What it established is narrower -- PR #785's |f(s) - f(0.25)| was not
   an ERROR sequence, because the anchor sits past a turn in f(h). A
   consistency floor is fully compatible with everything measured: f(h)
   may turn over and still converge to some f_floor != f_physical. The
   authoritative ledger records a MEASURED non-vanishing floor for this
   fixture class (rfx-known-issues.md, "DIELECTRIC-interface staircasing
   (~49%, eps_r-coupled) ... giving a non-vanishing FLOOR for thin
   high-eps_r layers"), and that entry stands as the standing evidence
   that a real floor may exist here.
2. D2's letter verdict is the RE-TAKEN one, evaluated on the nine-rung
   ladder under the unchanged frozen rule (d2_edge_retake.json). The
   five-rung verdict is carried alongside it, labelled superseded.
3. Every uncertainty number is quoted with the object it was measured on.
   eps_instr = 8.4 kHz is the EMPTY-VACUUM-BOX twin's extraction error;
   the fixture's own records give an independent-estimator spread of
   215.5 kHz at the reference rung and 72-230 kHz across records. Where
   the fixture's uncertainty is meant, the fixture's number is quoted.
   The ratio between the two is DERIVED from those fields, not typed
   (third review: a typed "9.7x to 25.7x ... 25.7x at the reference rung"
   did not match the artifact it sat next to).
4. Every control is quoted with what it shares with the fixture and what
   it does not (``smooth_field_control_scope``), and the accuracy
   envelope is quoted with the fixture class it was measured on and the
   classes it does NOT bind (``accuracy_envelope.scope_of_the_binding``).
   Both were added after the third review found the twin described as
   "the same box ... the same port" (it is neither) and the envelope
   binding #715's PATCH baseline (no patch was run in this lane).

Run:  PYTHONPATH=. python -m validation.research.convergence_floor.verdict
"""

from __future__ import annotations

import json
import os

import numpy as np

from validation.research.convergence_floor import fixture as fx
from validation.research.convergence_floor import d4_reference as d4r

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "verdict.json")


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _cells_fixture(scale: float) -> int:
    """Realized cell count of the W4R fixture at one ladder rung."""
    dx = fx.PC_DX0 * scale
    return (int(round(fx.PC_A / dx)) * int(round(fx.PC_B / dx))
            * len(fx.pc_uniform_profile(scale)))


def _cells_twin(scale: float) -> int:
    """Realized cell count of the D4a exact-reference twin at one rung."""
    dx = fx.PC_DX0 * scale
    dz = fx.PC_DZF0 * scale
    return (int(round(d4r.TWIN_A / dx)) * int(round(d4r.TWIN_B / dx))
            * int(round(d4r.TWIN_LZ / dz)))


def main():
    win = load("predeclared_windows_786.json")
    d0 = load("d0_reproduction.json")
    d1 = load("d1_geometry.json")
    d2 = load("d2_edge.json")
    d3 = load("d3_port.json")
    d4a = load("d4_reference_a.json")
    d4b = load("d4_reference_b.json")
    d4c = load("d4_reference_c.json")
    d5 = load("d5_turnover.json")
    d6 = load("d6_two_term.json")

    out = {"issue": 786, "verdicts": {}}
    out["verdicts"]["D0"] = d0["verdict"] if d0 else "NOT RUN"
    if d0:
        out["verdicts"]["D0_scope"] = (
            "SCOPE CORRECTION (2026-08-30, second review). The string above "
            "is d0_reproduction.json's verbatim verdict and its "
            "'f(s) itself is monotone' clause was evaluated on the five FIT "
            "rungs only. It is FALSE on the ladder this lane ended up with: "
            "d5_turnover.json records sign_changes = 1 and "
            "d2_edge_retake.json records f_monotone = false, and f(s) is "
            "not monotone even across D0's own uniform arm including the "
            "s = 0.25 reference (5.404546 / 5.502115 / 5.533066 / 5.542444 "
            "/ 5.543558 / 5.520821 GHz). What D0 establishes, and all it "
            "establishes, is BIT-IDENTICAL REPRODUCTION (0.0000 Hz over "
            "all eleven rungs). The source JSON is left unedited so its "
            "provenance stands; read this field with it."
        )
    out["verdicts"]["D1"] = d1["verdict"] if d1 else "NOT RUN"
    d2r = load("d2_edge_retake.json")
    if d2r:
        out["verdicts"]["D2"] = d2r["verdict"]
        out["verdicts"]["D2_scope"] = (
            "RE-TAKEN on the nine-rung ladder under the UNCHANGED frozen "
            "rule (results/d2_edge_retake.json). Neither the exonerate "
            "branch nor the attribute branch is satisfied.")
        out["verdicts"]["D2_superseded_five_rung"] = d2["verdict"] if d2 \
            else "NOT RUN"
    else:
        out["verdicts"]["D2"] = d2["verdict"] if d2 else "NOT RUN"
    out["verdicts"]["D3"] = d3["verdict"] if d3 else "NOT RUN"
    out["verdicts"]["D4a"] = d4a["D4a"]["verdict"] if d4a else "NOT RUN"
    if d4b:
        out["verdicts"]["D4b"] = {k: v["verdict"]
                                  for k, v in d4b["D4b"].items()}
    if d4c:
        out["verdicts"]["D4c"] = {("%s_%s" % (r["arm"], r["scale"])):
                                  r["verdict"] for r in d4c["D4c"]["rows"]}

    # --- apportionment (rule frozen in the pre-declaration) -----------
    app = {}
    if d4b and d4c:
        f_inf = d4b["D4b"]["uniform"]["f_inf_hz"]
        ref_rows = [r for r in d4c["D4c"]["rows"]
                    if r["arm"] == "UC" and r["scale"] == fx.REF_SCALE]
        if ref_rows:
            r = ref_rows[0]
            e1 = r["E1_hz"]
            cons = r["mean_hz"]
            d_total = abs(e1 - f_inf)
            d_instr = abs(e1 - cons)
            d_phys = abs(cons - f_inf)
            app = {
                "f_inf_hz": f_inf,
                "f_E1_at_ref_hz": e1,
                "f_consensus_at_ref_hz": cons,
                "consensus_spread_hz": r["spread_hz"],
                "Delta_total_hz": d_total,
                "Delta_instr_hz": d_instr,
                "Delta_phys_hz": d_phys,
                "instr_pct": 100.0 * d_instr / d_total if d_total else None,
                "phys_pct": 100.0 * d_phys / d_total if d_total else None,
                "remedy_licensed_4a": bool(d_total and
                                           d_instr >= 0.5 * d_total),
            }
            # The two magnitudes above do NOT partition Delta_total
            # (0.2118 + 100.2118 = 100.42 %): the instrument term is
            # SIGNED, and here it points the same way as f_inf, so the
            # consensus sits FURTHER from f_inf than E1 does. The signed
            # decomposition is exact and does partition.
            # direction of the total: u = sign(f_inf - E1)
            u = 1.0 if (f_inf - e1) >= 0 else -1.0
            phys_signed = u * (f_inf - cons)     # E1 -> consensus -> f_inf
            instr_signed = u * (cons - e1)
            app["signed_decomposition"] = {
                "identity": ("Delta_total = u*(f_inf - E1) = u*(f_inf - "
                             "consensus) + u*(consensus - E1), "
                             "u = sign(f_inf - E1)"),
                "u": u,
                "Delta_phys_signed_hz": phys_signed,
                "Delta_instr_signed_hz": instr_signed,
                "sum_hz": phys_signed + instr_signed,
                "closes_to_hz": abs(phys_signed + instr_signed - d_total),
                "phys_pct": 100.0 * phys_signed / d_total,
                "signed_instr_pct": 100.0 * instr_signed / d_total,
                "sum_pct": 100.0 * (phys_signed + instr_signed) / d_total,
                "why_the_magnitudes_do_not_partition": (
                    "the consensus lies 0.069 MHz FURTHER from f_inf than "
                    "E1 does, so the instrument step points against the "
                    "total and the two MAGNITUDES sum to 100.42 %, not "
                    "100 %. Only the signed form partitions: "
                    "+100.21 % (physics) - 0.21 % (instrument) = 100 %. "
                    "Either way (4a) carries 0.21 % of Delta_total in "
                    "magnitude and the remedy-licence branch (4a "
                    "dominates iff >= 50 %) is unaffected."),
            }
    out["verdicts"]["D5"] = d5["verdict"] if d5 else "NOT RUN"
    out["verdicts"]["D6"] = d6["verdict"] if d6 else "NOT RUN"
    out["apportionment"] = app

    # --- accuracy envelope for this fixture class ---------------------
    if d5:
        ss = d5["ladder_scales"]
        fv = np.array(d5["ladder_f_hz"])
        # Resolved range = every rung the ladder actually paid for,
        # excluding the coarsest (s=1.5), whose trace is 12 cells wide.
        keep = [i for i, s_ in enumerate(ss) if s_ <= 1.0]
        fk = fv[keep]
        sk = [ss[i] for i in keep]
        env = {
            "resolved_scales": sk,
            "dz_fine_mm": [fx.PC_DZF0 * s_ * 1e3 for s_ in sk],
            "f_min_hz": float(fk.min()), "f_max_hz": float(fk.max()),
            "spread_hz": float(fk.max() - fk.min()),
            "spread_relative": float((fk.max() - fk.min()) / fk.mean()),
            "peak_scale": d5["peak_scale"],
            "monotone": bool(d5["sign_changes"] == 0),
            "finest_scale": min(sk),
            "finest_f_hz": float(fk[-1]),
        }
        if d6:
            for name in ("M0", "M1", "free"):
                f_inf = d6[name]["f_inf_hz"]
                env["%s_f_inf_hz" % name] = f_inf
                env["%s_finest_minus_f_inf_hz" % name] = float(fk[-1] - f_inf)
                env["%s_finest_rel_error" % name] = float(
                    abs(fk[-1] - f_inf) / f_inf)
                env["%s_rms_hz" % name] = d6[name]["rms_hz"]
            # --- the ADMISSIBLE model set, stated once and used for
            # every continuum-limit sentence, so the set and the
            # "finest rung is between X and Y" statement cannot drift
            # apart. The free-exponent fit is EXCLUDED and named as
            # excluded: its two exponents collapse onto each other
            # (a = 0.872, b = 0.895, amplitudes +4.9/-5.7 cancelling),
            # which is a numerical derivative, not two physical terms.
            grid = d6["diagnostics_reported_not_judged"][
                "fixed_bulk_order_2_fits"]
            adm = {"M0 single power law, 9 rungs":
                   {"f_inf_hz": d6["M0"]["f_inf_hz"],
                    "rms_hz": d6["M0"]["rms_hz"]}}
            for k, v in grid.items():
                adm["h^%s + h^2 (bulk order 2 = D4a measured)" % k[2:]] = {
                    "f_inf_hz": v["f_inf_hz"], "rms_hz": v["rms_hz"]}
            finest = float(fk[-1])
            offs = {k: finest - v["f_inf_hz"] for k, v in adm.items()}
            best = min(adm, key=lambda k: adm[k]["rms_hz"])
            f5 = None
            if d4b:
                f5 = d4b["D4b"]["uniform"]["f_inf_hz"]
            env["continuum_limit"] = {
                "determined": False,
                "admissible_models_nine_rung": adm,
                "excluded": {
                    "free-exponent f_inf + A h^a - B h^b": {
                        "f_inf_hz": d6["free"]["f_inf_hz"],
                        "rms_hz": d6["free"]["rms_hz"],
                        "why_excluded": ("degenerate: a = 0.872, b = 0.895 "
                                         "collapse onto each other with "
                                         "amplitudes +4.9/-5.7 that cancel "
                                         "-- a numerical derivative, not "
                                         "two physical terms")},
                    "five-rung single power law (D4b)": {
                        "f_inf_hz": f5,
                        "why_excluded": ("fitted on the ascending branch "
                                         "ONLY; reported for comparison, "
                                         "not admissible as a limit for "
                                         "the full ladder")}},
                "f_inf_spread_hz": float(max(v["f_inf_hz"] for v in adm.values())
                                         - min(v["f_inf_hz"]
                                               for v in adm.values())),
                "f_inf_spread_relative": float(
                    (max(v["f_inf_hz"] for v in adm.values())
                     - min(v["f_inf_hz"] for v in adm.values())) / finest),
                "finest_rung_minus_f_inf_hz": {k: float(v)
                                               for k, v in offs.items()},
                "finest_rung_offset_range_hz": [float(min(offs.values())),
                                                float(max(offs.values()))],
                "finest_rung_abs_rel_error_range": [
                    float(min(abs(v) for v in offs.values()) / finest),
                    float(max(abs(v) for v in offs.values()) / finest)],
                "best_fitting_admissible_model": best,
                "best_fitting_finest_minus_f_inf_hz": float(offs[best]),
                "statement": (
                    "Over the ADMISSIBLE nine-rung model set the finest "
                    "rung lies between %.1f MHz below and %.1f MHz above "
                    "the extrapolated limit, i.e. its own absolute accuracy "
                    "is %.1e at best and %.1e at worst. The best-fitting "
                    "admissible model (%s, RMS %.2f MHz) is the one that "
                    "puts it %.1f MHz ABOVE the limit -- the earlier "
                    "'11 MHz above / 17 MHz below' range excluded it and "
                    "is withdrawn."
                    % (abs(min(offs.values())) / 1e6,
                       max(offs.values()) / 1e6,
                       min(abs(v) for v in offs.values()) / finest,
                       max(abs(v) for v in offs.values()) / finest,
                       best, adm[best]["rms_hz"] / 1e6,
                       offs[best] / 1e6)),
            }
        out["accuracy_envelope"] = env

    # --- the ladder, re-judged against the independent reference ------
    if d4b:
        u = d4b["D4b"]["uniform"]
        errs = [u["err_vs_f_inf_mhz"][str(s)] for s in u["scales"]]
        out["ladder_against_independent_reference"] = {
            "scales": u["scales"],
            "err_mhz": u["err_vs_f_inf_mhz"],
            "monotone_decreasing": bool(all(errs[i] > errs[i + 1]
                                            for i in range(len(errs) - 1))),
            "p_loglog": u["p_from_loglog_vs_f_inf"],
            "p_nls": u["p"],
        }
        if "multiband" in d4b["D4b"]:
            m = d4b["D4b"]["multiband"]
            out["ladder_against_independent_reference"]["multiband"] = {
                "err_mhz": m["err_vs_f_inf_mhz"],
                "p_loglog": m["p_from_loglog_vs_f_inf"], "p_nls": m["p"]}

    # --- D4c on the D5 rungs (the instrument check they shipped without)
    d5i = load("d5_instrument_check.json")
    if d5i:
        out["verdicts"]["D4c_on_D5_rungs"] = d5i["verdicts"]
        out["d5_rung_instrument_check"] = {
            "window_source": d5i["window_source"],
            "estimator_spread_range_khz": d5i["estimator_spread_range_khz"],
            "max_abs_E1_minus_consensus_hz":
                d5i["max_abs_E1_minus_consensus_hz"],
            "reproduces_d5_bit_identically":
                d5i["reproduces_d5_bit_identically"],
            "scope": d5i["scope"],
        }

    # --- what the fixture's own uncertainty is, on the fixture ---------
    if d4a and d4c:
        uc = [r for r in d4c["D4c"]["rows"] if r["arm"] == "UC"]
        sp = [r["spread_hz"] for r in uc]
        allsp = [r["spread_hz"] for r in d4c["D4c"]["rows"]
                 if r["verdict"] != "CONSENSUS-UNAVAILABLE"]
        ref = [r for r in uc if r["scale"] == fx.REF_SCALE]
        out["instrument_uncertainty"] = {
            "twin_eps_instr_hz": {
                "what": ("|f_extract - f_disc| on the EMPTY VACUUM BOX twin, "
                         "which has an exact discrete eigenfrequency; this "
                         "is the extraction machinery's error on a smooth "
                         "single-line record, NOT the fixture's"),
                "per_rung_hz": d4a["D4a"]["eps_instr_hz"],
                "at_reference_rung_hz": [
                    r["eps_instr_E1_hz"] for r in d4a["D4a"]["rows"]
                    if r["scale"] == fx.REF_SCALE][0],
            },
            "fixture_estimator_spread_hz": {
                "what": ("max pairwise spread of the three INDEPENDENT "
                         "estimators E2/E3/E4 on the fixture's OWN stored "
                         "records -- the number to quote when the "
                         "fixture's extraction uncertainty is meant"),
                "uniform_arm_range_hz": [float(min(sp)), float(max(sp))],
                "all_arms_range_hz": [float(min(allsp)), float(max(allsp))],
                "at_reference_rung_hz": ref[0]["spread_hz"] if ref else None,
                "d5_rungs_range_hz": [x * 1e3 for x in
                                      d5i["estimator_spread_range_khz"]]
                if d5i else None,
            },
        }
        # BL3 (third review): the ratio is now DERIVED from the two fields
        # in this same JSON object, not typed. The second review's
        # "9.7x to 25.7x ... 25.7x at the reference rung" was not what the
        # artifact supports: all_arms_range_hz / twin(0.25) = 8.6x-27.2x,
        # uniform_arm_range_hz / twin(0.25) = 9.7x-25.5x, and the
        # reference rung is 25.5x, not 25.7x.
        twin_ref = out["instrument_uncertainty"]["twin_eps_instr_hz"][
            "at_reference_rung_hz"]
        spread = out["instrument_uncertainty"]["fixture_estimator_spread_hz"]
        ratios = {
            "denominator_hz": float(twin_ref),
            "denominator_is": "the twin's extraction error at s = 0.25",
            "all_records_with_a_consensus": [
                float(spread["all_arms_range_hz"][0] / twin_ref),
                float(spread["all_arms_range_hz"][1] / twin_ref)],
            "uniform_arm": [
                float(spread["uniform_arm_range_hz"][0] / twin_ref),
                float(spread["uniform_arm_range_hz"][1] / twin_ref)],
            "at_reference_rung": float(
                spread["at_reference_rung_hz"] / twin_ref),
        }
        out["instrument_uncertainty"]["ratio_fixture_over_twin"] = ratios
        out["instrument_uncertainty"]["rule"] = (
            "quote %.1f kHz ONLY for the twin; quote the fixture's own "
            "%.1f kHz (reference rung), %.1f-%.1f kHz (uniform arm) or "
            "%.0f-%.0f kHz (all records with a consensus) wherever the "
            "fixture's uncertainty is meant -- %.1fx to %.1fx the twin's "
            "number across all records with a consensus (%.1fx to %.1fx on "
            "the uniform arm), and %.1fx at the reference rung"
            % (twin_ref / 1e3, spread["at_reference_rung_hz"] / 1e3,
               spread["uniform_arm_range_hz"][0] / 1e3,
               spread["uniform_arm_range_hz"][1] / 1e3,
               spread["all_arms_range_hz"][0] / 1e3,
               spread["all_arms_range_hz"][1] / 1e3,
               ratios["all_records_with_a_consensus"][0],
               ratios["all_records_with_a_consensus"][1],
               ratios["uniform_arm"][0], ratios["uniform_arm"][1],
               ratios["at_reference_rung"]))

    # --- BL1 (third review): the smooth-field control's ACTUAL scope ---
    # The design note's 7 and the PR body said the p = 2.0001 control ran
    # "on a smooth fixture in the same box ... same port". It did not.
    # The twin is a DIFFERENT enclosure and a DIFFERENT excitation; what it
    # shares is the per-rung DISCRETIZATION and the INSTRUMENT. Emitted
    # from the two modules' own frozen constants so it cannot drift.
    out["smooth_field_control_scope"] = {
        "control": "D4a / D2-B exact-reference twin",
        "result_it_carries": "p = 2.0001 analytic (1.9707 measured); "
                             "45 kHz total error at s = 0.25",
        "shared_with_the_W4R_fixture": {
            "dx_per_rung_m": "fx.PC_DX0 * scale (identical)",
            "dz_per_rung_m": "fx.PC_DZF0 * scale, uniform (identical)",
            "dt_and_n_steps": "identical at every rung -- both use "
                              "fx.n_steps_for(scale, dz) and the same CFL",
            "record_length_s": fx.T_TOTAL,
            "source_waveform": dict(fx.WAVEFORM),
            "source_kind": "Ez, amplitude_kind='current' (additive)",
            "analysis_band_hz": list(d4r.TWIN_BAND),
            "extraction": "fx.modes_of + fx.target_line, i.e. the same "
                          "Result.find_resonances path and the same "
                          "Q_MIN = %g dominance rule" % fx.Q_MIN,
            "subpixel_smoothing": fx.SUBPIXEL,
        },
        "NOT_shared_with_the_W4R_fixture": {
            "enclosure_m": {
                "fixture": [fx.PC_A, fx.PC_B, fx.PC_TOTAL_H],
                "twin": [d4r.TWIN_A, d4r.TWIN_B, d4r.TWIN_LZ],
                "comment": "27 x 22.5 x 13.5 mm vs 38.25 x 38.25 x 1.5 mm "
                           "-- a different box, not the same box",
            },
            "materials": {
                "fixture": "eps_r = 4.3 substrate + eps_r = 2.2 upper "
                           "layer + a PEC trace",
                "twin": "vacuum throughout, no dielectric, no trace",
            },
            "port": {
                "fixture": "an antisymmetric PAIR of Ez current sources at "
                           "%s and %s m, probe at %s m"
                           % (fx.SRC_P, fx.SRC_M, fx.PRB),
                "twin": "a SINGLE Ez current source at %s m, probe at %s m"
                        % (d4r.TWIN_SRC, d4r.TWIN_PRB),
                "comment": "a different port: different count, different "
                           "symmetry selection, different coordinates",
            },
            "mode_observed": {
                "fixture": "x-odd / y-even half-wave trace line near "
                           "5.5 GHz",
                "twin": "the empty-box TM110 near 5.542 GHz",
            },
            "cells_at_the_reference_rung": {
                "fixture": _cells_fixture(fx.REF_SCALE),
                "twin": _cells_twin(fx.REF_SCALE),
            },
        },
        "what_it_therefore_licenses": (
            "the Yee update, the dz_profile path at these dx/dz (both "
            "sides run UNIFORM profiles), the additive current port as "
            "a mechanism, the time stepping over this record length, "
            "and the find_resonances extraction are each "
            "second-order-clean at every rung of THIS ladder, reference "
            "rung included"),
        "what_it_does_NOT_license": (
            "any statement about the fixture's box, its dielectric stack, "
            "its rasterized conductor edge, or its port pair. It is a "
            "control on the MACHINERY at matched discretization, not a "
            "smooth twin of the fixture."),
    }

    # --- the ledger cross-check: REFUTED by this lane's own data -------
    if d5:
        L_EFF = 13.5e-3          # the trace's resonant length
        dx_hi = fx.PC_DX0 * 1.0
        dx_lo = fx.PC_DX0 * 0.25
        f_ref = 5.520820807318383e9
        pred_hi = dx_hi / L_EFF
        pred_lo = dx_lo / L_EFF
        pred_change_hz = (pred_hi - pred_lo) * f_ref
        meas = out["accuracy_envelope"]["spread_hz"]
        out["ledger_cross_check"] = {
            "status": "REFUTED -- reported as a FINDING, not as support",
            "ledger_entry": ("rfx-known-issues.md, 'patch-resonance "
                             "discretization error envelope (measured, "
                             "tutorial patch)': f_res bias ~ -dx/L_eff"),
            "applied_to_this_fixture": {
                "L_eff_mm": L_EFF * 1e3,
                "dx_range_mm": [dx_lo * 1e3, dx_hi * 1e3],
                "predicted_bias_range_pct": [-pred_hi * 100, -pred_lo * 100],
                "predicted_CHANGE_over_the_dx_range_pct":
                    (pred_hi - pred_lo) * 100,
                "predicted_CHANGE_hz": pred_change_hz,
            },
            "measured_change_hz": meas,
            "over_prediction_factor": float(pred_change_hz / meas),
            "finding": (
                "The ledger law predicts the bias CHANGES by %.2f %% "
                "= %.0f MHz across this lane's dx range (0.75 -> 0.1875 mm "
                "on a 13.5 mm resonant length). This lane MEASURED %.2f MHz "
                "of refinement variation over that same range. The law "
                "over-predicts this fixture's measured variation by %.1fx, "
                "so the two do NOT corroborate each other; the earlier "
                "'same order, both agree on the 1e-2 band' cross-check is "
                "WITHDRAWN and must not be used to widen the envelope. "
                "Whether the discrepancy is the fixture (an enclosed "
                "3-layer stripline-class line, not an open patch), the "
                "law's L_eff, or a limit of the law itself is a separate "
                "question and deserves its own issue."
                % ((pred_hi - pred_lo) * 100, pred_change_hz / 1e6,
                   meas / 1e6, pred_change_hz / meas)),
        }

    # BL2 is emitted here, after the ledger cross-check, because the
    # extension condition quotes that cross-check's measured
    # over-prediction factor.
    if "accuracy_envelope" in out:
        _env = out["accuracy_envelope"]
        # --- BL2 (third review): WHAT THE ENVELOPE BINDS, narrowed to the
        # fixture actually measured. The design note's 7 bound "#715's
        # patch cross-validation baseline" to the >= 7.5e-3 number. This
        # lane measured ONE enclosed 3-layer microstrip LINE, not a patch;
        # a patch is a different radiator (open boundary, different
        # L_eff, different edge inventory) and nothing here was run on
        # one. The binding is therefore stated for the measured class
        # only, with the extension condition named.
        out["accuracy_envelope"]["scope_of_the_binding"] = {
            "measured_on": {
                "count": 1,
                "fixture": ("a single enclosed 3-layer microstrip-class "
                            "LINE: a %g x %g x %g mm PEC trace on a %g mm "
                            "eps_r = %g substrate under a %g mm air gap, a "
                            "%g mm eps_r = %g upper layer and a second "
                            "%g mm air gap, all inside a %g x %g x %g mm "
                            "PEC box"
                            % ((fx.TRACE_X[1] - fx.TRACE_X[0]) * 1e3,
                               (fx.TRACE_Y[1] - fx.TRACE_Y[0]) * 1e3,
                               fx.PC_H_TRACE_BAND * 1e3,
                               fx.PC_H_SUB * 1e3, fx.PC_EPS_SUB,
                               fx.PC_AIR1 * 1e3,
                               fx.PC_H_UPPER * 1e3, fx.PC_EPS_UPPER,
                               fx.PC_AIR2 * 1e3,
                               fx.PC_A * 1e3, fx.PC_B * 1e3,
                               fx.PC_TOTAL_H * 1e3)),
                "observable": ("the x-odd / y-even half-wave line near "
                               "5.5 GHz, uniform mesh, "
                               "subpixel_smoothing=True, Harminv ring-down"),
                "refinement_range": ("dz_fine 0.25 -> 0.0625 mm, "
                                     "dx 0.75 -> 0.1875 mm"),
            },
            "binds": ("absolute-frequency claims on THIS fixture, and on "
                      "fixtures of the same class -- an enclosed, "
                      "uniform-mesh, rasterized microstrip line on a "
                      "layered dielectric stack, extracted from a "
                      "ring-down -- over this refinement range: state them "
                      "against >= %.2e (measured refinement variation), "
                      "not against the finest rung, and do not claim a "
                      "continuum limit better than ~%.0e without an "
                      "external reference."
                      % (_env["spread_relative"],
                         _env["continuum_limit"]["f_inf_spread_relative"]
                         if "continuum_limit" in _env else 2e-2)),
            "does_NOT_bind": (
                "#715's PATCH cross-validation baseline. No patch was run "
                "in this lane. A patch is a different object: an open "
                "radiating boundary instead of a PEC enclosure, a "
                "different L_eff, a different edge inventory, and a "
                "different mode. Quoting this number against #715's "
                "baseline would be binding a patch to a number measured "
                "on one non-patch fixture."),
            "what_would_extend_it_to_715": (
                "run this same ladder -- same nine lattice-valid rungs, "
                "same uniform mesh, same extraction, same turn-over check "
                "via ladder_guard -- on #715's own patch geometry, and "
                "report ITS measured non-monotone spread. Two further "
                "conditions, both of which this lane's own data show "
                "matter: (a) the ladder must span the turn-over, since "
                "the whole finding here is that a ladder read on one side "
                "of a maximum reports arithmetic, not error; (b) the "
                "ledger's -dx/L_eff staircase law must not be used as the "
                "bridge -- this lane MEASURED it over-predicting this "
                "fixture's variation by %.1fx, so it does not transport "
                "an envelope between fixture classes."
                % (out["ledger_cross_check"]["over_prediction_factor"]
                   if "ledger_cross_check" in out else float("nan"))),
        }

    # --- the claims this lane withdraws, and what replaces them --------
    out["claims_withdrawn"] = {
        "there_is_no_floor": {
            "where_it_appeared": [
                "commit d5edc34 subject line ('the floor is not a floor')",
                "design note wording carried into the verdict summary",
            ],
            "why_withdrawn": (
                "This lane never measured a floor's ABSENCE. It has no "
                "external reference -- design note 8.2 concedes 'no "
                "independent solver or measurement pins f_inf' -- and a "
                "consistency floor is fully compatible with every number "
                "here: f(h) can turn over and still converge to some "
                "f_floor != f_physical. The authoritative ledger records a "
                "MEASURED non-vanishing floor for this fixture class "
                "(DIELECTRIC-interface staircasing, ~49 %, eps_r-coupled, "
                "2nd -> 1st order for thin high-eps_r layers), which is "
                "standing evidence that a real floor may exist here."),
            "replaced_by": (
                "PR #785's |f(s) - f(0.25)| was not an ERROR sequence, "
                "because the anchor sits four rungs past a turn-over in "
                "f(h) (D5: exactly one sign change, maximum at "
                "dz_fine = 0.125 mm, descending branch of four rungs). The "
                "22.7 MHz figure is therefore not an error, and the "
                "'~4e-3 floor' as stated in the issue is not established. "
                "WHETHER f(h) CONVERGES TO THE PHYSICAL ANSWER IS NOT "
                "DETERMINED BY THIS LANE."),
        },
        "D2_measured_exoneration": {
            "why_withdrawn": ("the frozen rule's exonerate branch was "
                              "evaluated on five pre-D5 scales; on the "
                              "nine-rung ladder both its clauses fail and "
                              "p_trace = 0.95 < 1.0 triggers the rule's own "
                              "INCONCLUSIVE clause"),
            "replaced_by": ("D2 INCONCLUSIVE on the nine-rung ladder. The "
                            "Meixner wedge reasoning survives as an "
                            "ARGUMENT from theory (an order reduction "
                            "cannot by itself make a non-vanishing floor), "
                            "not as a measured exoneration."),
        },
        "ledger_cross_check_as_support": {
            "why_withdrawn": "refuted by this lane's own data (5.6x "
                             "over-prediction); see ledger_cross_check",
            "replaced_by": "the discrepancy, reported as a finding",
        },
        "finest_rung_11_above_to_17_below": {
            "why_withdrawn": ("the stated range excluded the "
                              "best-fitting admissible model (h^1/2 + h^2, "
                              "RMS 2.64 MHz), which puts the finest rung "
                              "80 MHz ABOVE the limit"),
            "replaced_by": "accuracy_envelope.continuum_limit.statement",
        },
    }

    # --- the four candidates, at the precision the data support --------
    out["candidate_verdicts"] = {
        "(1) geometry quantization": {
            "letter": "EXONERATED by the D1b addendum window",
            "machine": out["verdicts"]["D1"],
            "caveat": ("the BASE window (delta < 1e-12 m) is below the "
                       "float32 mesh-storage floor and its letter verdict "
                       "is INCONCLUSIVE, reported unchanged, not widened; "
                       "D1c fired by the letter and is recorded as "
                       "ATTRIBUTED-CANDIDATE with a post-hoc diagnosis "
                       "(s = 1.5 halo only). The exoneration rests on the "
                       "addendum window: 6.8e-6 cells worst over six "
                       "rungs, and realized PEC extents that are exact "
                       "integer cell counts."),
        },
        "(2) edge singularity": {
            "letter": "INCONCLUSIVE on the nine-rung ladder (re-taken)",
            "argument_not_measurement": (
                "a Meixner wedge reduces the convergence ORDER and cannot "
                "by itself produce a non-vanishing floor -- theory, from "
                "the pre-declared 3*pi/2 wedge, not a measurement"),
            "smooth_field_control_as_measured": (
                "p = 2.0001 analytic / 1.9707 measured on the "
                "exact-reference vacuum twin -- a DIFFERENT box and a "
                "DIFFERENT port at matched dx/dz/dt/record/band/extraction; "
                "see smooth_field_control_scope for exactly what it does "
                "and does not share with the fixture"),
            "D6": "the Meixner exponent 4/3 is NOT confirmed (RMS 4.36 MHz "
                  "against a 1 MHz window; a = 0.5 fits better at 2.64 MHz)",
        },
        "(3) port / probe loading": {
            "letter": "EXONERATED",
            "number": "span 3.540 kHz (s = 0.75) and 0.623 kHz (s = 0.5) "
                      "over drive x0.01...x100, moved ports, moved probe, "
                      "against a first-principles predicted exact zero",
        },
        "(4) reference quality": {
            "letter": "ATTRIBUTED -- but NOT as instrument noise",
            "what_is_established": (
                "the s = 0.25 record really does contain 5.520821 GHz "
                "(three independent estimators agree with the incumbent to "
                "0.069 MHz on that record) and the anchor is invalid "
                "because it lies past the MAXIMUM of a non-monotone f(h). "
                "That makes PR #785's error column arithmetic, not error."),
            "what_is_NOT_established": (
                "that the ladder converges to the physical resonance. No "
                "external reference exists in this lane; every f_inf here "
                "is a fitted extrapolation and they span 97 MHz."),
        },
    }

    out["headline"] = [
        "D0: the symptom REPRODUCES bit-identically (0.0000 Hz over all "
        "eleven rungs) -- it is a property of the code, not of a run.",
        "D5: f(h) TURNS OVER inside the ladder -- exactly one sign change, "
        "maximum at dz_fine = 0.125 mm, four descending rungs -- so PR "
        "#785's |f(s) - f(0.25)| was never an error sequence and the "
        "22.7 MHz 'floor' figure is not an error.",
        "Three candidates are cleanly exonerated, each against its own "
        "pre-declared window, at 1.2-2.5 orders of margin: geometry "
        "quantization 6.8e-6 cells vs a 1e-3 cell window (146x = 2.2 "
        "orders), port loading 3.5 kHz vs a 1 MHz window against a "
        "predicted exact zero (282x = 2.5 orders), and the extraction "
        "instrument on the reference record 0.069 MHz vs a 1 MHz window "
        "(14.5x = 1.2 orders, the weakest of the three). An earlier "
        "draft said '3-5 orders'; that was arithmetic error, corrected "
        "here against the same comparators.",
        "The smooth-field control converges at p = 2.0001 on an "
        "exact-reference vacuum twin at the same dx, dz, dt, band, record "
        "length and extraction -- but in a DIFFERENT box (38.25 x 38.25 x "
        "1.5 mm vacuum, not the fixture's 27 x 22.5 x 13.5 mm stack) and "
        "with a DIFFERENT port (one Ez source, not the fixture's "
        "antisymmetric pair). It licenses the MACHINERY at matched "
        "discretization; it is not a smooth twin of the fixture, and the "
        "earlier 'same box / same port' framing is withdrawn "
        "(smooth_field_control_scope).",
        "D2 is INCONCLUSIVE on the nine-rung ladder under its own frozen "
        "rule; the wedge argument stands as theory, not as measurement.",
        "WHETHER f(h) CONVERGES TO THE PHYSICAL ANSWER IS NOT DETERMINED "
        "BY THIS LANE: there is no external reference, the admissible "
        "extrapolations span 97 MHz = 1.8e-2, and the ledger's measured "
        "DIELECTRIC-interface staircasing floor for this fixture class "
        "remains fully compatible with everything measured here.",
        "ladder_guard.py states the precondition that catches the original "
        "misreading, and fires on PR #785's own ladder.",
    ]

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
