"""Issue #786 — PRE-DECLARED discriminator windows (SPEC-00 0.2-2).

This module is committed BEFORE any discriminator measurement exists.
It emits ``results/predeclared_windows_786.json``; every later script
reads its windows from that file and may never widen them.

BURNED-DATA DISCIPLINE. The W4R numbers in issue #786 (116.3 / 18.7 /
12.2 / 21.6 / 22.7 MHz, u_ref = 7.58 MHz, "~4.1e-3") are the SYMPTOM.
Not one window below is derived from them. Each window's provenance is
recorded in the JSON under ``derivation``:

* ``arithmetic``  — exact integer commensurability of the declared
  geometry with the ladder's cell sizes (D1). No simulation involved.
* ``first_principles`` — the Cramer-Rao bound of a single-sinusoid
  frequency estimate on a float32 record (D4, and the 1/10 MHz pair
  reused by D3), and the additive-source theorem (D3).
* ``wedge_theory`` — the Meixner edge condition exponent for a 3*pi/2
  conductor wedge (D2).
* ``prior_provenance`` — the ledger's measured patch-resonance
  staircase envelope (context only; never the numeric window itself).

Run:  PYTHONPATH=. python -m validation.research.convergence_floor.predeclare
"""

from __future__ import annotations

import json
import os

import numpy as np

from validation.research.convergence_floor import fixture as fx

OUT = os.path.join(os.path.dirname(__file__), "results",
                   "predeclared_windows_786.json")

# --- D4/D3 instrument-resolution pair (FIRST PRINCIPLES) --------------
# Probe records are float32: relative precision eps32 = 2^-24 = 5.96e-8.
# Cramer-Rao bound for the frequency of a single sinusoid in white noise
# of relative amplitude sigma, N samples spanning T:
#     sigma_f = sqrt(6) / (pi * T * sigma_inv * sqrt(N)),  sigma_inv=1/sigma
# With T = T_TOTAL = 20 ns, N = 700 (the post-decimation sample count the
# incumbent estimator actually uses) and sigma = eps32:
EPS32 = 2.0 ** -24
CRB_T = fx.T_TOTAL
CRB_N = 700
CRB_HZ = float(np.sqrt(6.0) / (np.pi * CRB_T) * EPS32 / np.sqrt(CRB_N))
# Round-off accumulates at worst like sqrt(n_steps) over ~1e5 steps, and
# the estimator is nonlinear rather than efficient; inflate by 1e3 and
# 1e4 to set a SOUND band and an INSTRUMENT-LIMITED band with a decade
# of declared-inconclusive between them:
INSTR_SOUND_HZ = 1.0e6      # ~ 1e3 * (sqrt(n_steps) * CRB)
INSTR_LIMITED_HZ = 10.0e6   # 1 decade above

# --- D2 wedge exponent (Meixner) --------------------------------------
# The W4R trace is a 1.5 mm THICK PEC block: its edges are 90 deg
# conductor corners, exterior (field) wedge angle 3*pi/2, so the field
# singularity exponent is nu = pi/(2*pi - theta) with theta = pi/2:
WEDGE_NU = float(np.pi / (2 * np.pi - np.pi / 2))     # = 2/3
WEDGE_ORDER = 2.0 * WEDGE_NU                          # = 4/3


def geometry_commensurability(scales) -> dict:
    """ARITHMETIC (no simulation): is every declared feature coordinate an
    integer multiple of dx(s) = 0.75*s mm and dz(s) = 0.25*s mm?"""
    xy_feats = {
        "trace_x_lo": fx.TRACE_X[0], "trace_x_hi": fx.TRACE_X[1],
        "trace_y_lo": fx.TRACE_Y[0], "trace_y_hi": fx.TRACE_Y[1],
        "domain_a": fx.PC_A, "domain_b": fx.PC_B,
        "src_p_x": fx.SRC_P[0], "src_m_x": fx.SRC_M[0],
        "src_y": fx.SRC_P[1], "probe_x": fx.PRB[0],
    }
    z_feats = {
        "sub_top": fx.PC_H_SUB,
        "trace_top": fx.PC_H_SUB + fx.PC_H_TRACE_BAND,
        "upper_lo": fx.PC_H_SUB + fx.PC_H_TRACE_BAND + fx.PC_AIR1,
        "upper_hi": (fx.PC_H_SUB + fx.PC_H_TRACE_BAND + fx.PC_AIR1
                     + fx.PC_H_UPPER),
        "total_h": fx.PC_TOTAL_H, "src_z": fx.SRC_P[2],
    }
    out = {}
    for s in scales:
        dx = fx.PC_DX0 * s
        dz = fx.PC_DZF0 * s
        rows = {}
        for name, v in xy_feats.items():
            rows[name] = float(abs(v - round(v / dx) * dx))
        for name, v in z_feats.items():
            rows[name] = float(abs(v - round(v / dz) * dz))
        out[str(s)] = {"dx_m": dx, "dz_m": dz,
                       "max_residual_m": max(rows.values()),
                       "residuals_m": rows}
    return out


def build() -> dict:
    scales = list(fx.SCALES) + [fx.REF_SCALE]
    commens = geometry_commensurability(scales)
    exact_everywhere = all(v["max_residual_m"] < 1e-12
                           for v in commens.values())
    return {
        "issue": 786,
        "predeclared_utc": "2026-08-30",
        "note": ("Windows frozen BEFORE any discriminator measurement. "
                 "Widening any of them invalidates this lane."),
        "symptom_is_not_a_target": {
            "burned": ["w4r_supraconvergence.json err_hz column",
                       "u_ref_hz = 7.58e6", "the ~4.1e-3 relative floor"],
            "rule": ("no window below may be a function of these; each "
                     "window carries its own `derivation` key"),
        },

        "D0_reproduction": {
            "what": ("re-run the PR #785 uniform + multiband ladders and "
                     "the s=0.25 reference with the copied fixture"),
            "pass": "every rung matches the PR #785 f_target to <= 1e3 Hz",
            "derivation": "determinism (identical code, identical CPU JAX)",
            "on_fail": ("STOP: the copy is not faithful; report the "
                        "irreproducibility as the finding"),
            "tol_hz": 1.0e3,
        },

        "D1_geometry_quantization": {
            "what": ("read the realized PEC node span (x,y,z) and the "
                     "realized dielectric interface planes out of the "
                     "assembled arrays at every rung; delta = |realized - "
                     "declared|"),
            "derivation": "arithmetic",
            "arithmetic_prediction": {
                "exact_at_every_rung": bool(exact_everywhere),
                "max_residual_over_all_rungs_m":
                    max(v["max_residual_m"] for v in commens.values()),
                "per_scale": commens,
            },
            "exonerate": ("delta_max(s) < 1e-12 m at EVERY rung including "
                          "s=0.25 -> every rung solves the identical "
                          "continuum structure; quantization cannot be the "
                          "mechanism"),
            "attribute": ("delta_max(s) >= 0.25*dx(s) at one or more rungs "
                          "AND |Pearson rho| >= 0.8 between the per-rung "
                          "residual and the realized-vs-declared electrical-"
                          "length delta"),
            "inconclusive": "0 < delta_max < 0.25*dx(s), or |rho| < 0.8",
            "attribute_delta_frac_of_dx": 0.25,
            "attribute_rho_min": 0.8,
        },

        "D2_edge_singularity": {
            "what": ("identical box / stack / port pair / probe / T / "
                     "scales, PEC trace DELETED (with_trace=False). The "
                     "surviving in-band line is a dielectric-loaded box "
                     "mode with no metal edge in the interior."),
            "derivation": "wedge_theory",
            "wedge": {"conductor_corner_deg": 90,
                      "field_wedge_angle_rad": 1.5 * np.pi,
                      "nu": WEDGE_NU,
                      "predicted_frequency_error_order": WEDGE_ORDER},
            "attribute_partial": (
                "p_trace in [1.0, 1.6] (consistent with 2*nu = 4/3) AND "
                "p_smooth >= 1.8: removing the edge restores the smooth-"
                "field order"),
            "exonerate_as_floor": (
                "f(s) of the with-trace ladder is MONOTONE in s and its "
                "error against the D4b reference DECREASES at every rung "
                "with fitted p_trace >= 1.0 -- a wedge exponent produces a "
                "reduced ORDER, never a non-vanishing floor"),
            "inconclusive": "p_trace < 1.0, or the no-trace ladder is itself non-monotone",
            "p_trace_lo": 1.0, "p_trace_hi": 1.6, "p_smooth_min": 1.8,
        },

        "D3_port_loading": {
            "what": ("at s = 0.75 (and s = 0.5): (a) src_amp 1.0, "
                     "(b) 0.01, (c) 100.0, (d) source pair moved to "
                     "(9.0,11.25,0.75)/(18.0,11.25,0.75) mm -- same "
                     "symmetry class, different physical coupling, "
                     "(e) probe moved to (15.75,11.25,0.75) mm"),
            "derivation": "first_principles",
            "theorem": (
                "rfx soft sources are ADDITIVE: E += Cb*w(t) "
                "(rfx/simulation.py::make_soft_source). An additive "
                "forcing term in a linear time-invariant system leaves the "
                "system operator, hence every eigenfrequency, EXACTLY "
                "unchanged. Predicted coupling-induced df = 0."),
            "exonerate": "max pairwise |df| over (a)-(e) <= 1.0 MHz",
            "attribute": ("monotone dependence of f on coupling strength "
                          "with span >= 10 MHz over (a)-(d)"),
            "inconclusive": "1 MHz < span < 10 MHz",
            "exonerate_hz": INSTR_SOUND_HZ,
            "attribute_hz": INSTR_LIMITED_HZ,
        },

        "D4_reference_quality": {
            "instrument_resolution_derivation": {
                "derivation": "first_principles",
                "float32_eps": EPS32,
                "T_s": CRB_T, "N_samples": CRB_N,
                "cramer_rao_hz": CRB_HZ,
                "sound_hz": INSTR_SOUND_HZ,
                "limited_hz": INSTR_LIMITED_HZ,
                "rationale": (
                    "CRB is ~1.5 Hz; inflating by 1e3 for accumulated "
                    "round-off over ~1e5 steps and estimator "
                    "inefficiency gives ~1 MHz. A decade above that "
                    "(10 MHz = 1.8e-3 of 5.5 GHz) is declared "
                    "instrument-limited. Context only (NOT the window's "
                    "source): the ledger's measured staircase-edge "
                    "envelope for patch resonance is -dx/L_eff, i.e. "
                    "2.9e-2 at dx=1mm on a 32mm patch -- an instrument "
                    "error at 1.8e-3 would already be ~6 % of the "
                    "physics envelope this fixture class is used to "
                    "bound."),
            },
            "D4a_exact_reference_twin": {
                "what": (
                    "empty vacuum PEC box, Lx = Ly = 38.25 mm, "
                    "Lz = 1.5 mm, dx = 0.75*s mm, dz = 0.25*s mm -> dt, "
                    "n_steps, analysis band and record length IDENTICAL "
                    "to the W4R uniform rung at the same s. Mode TM110 "
                    "(Ez, no z variation), f_exact = 5.5421 GHz, inside "
                    "the same BAND. The EXACT discrete leapfrog "
                    "eigenfrequency f_disc(s) = arcsin((c0 dt/2) "
                    "sqrt(mu_x+mu_y))/(pi dt) is computed from the same "
                    "difference operator rfx builds "
                    "(analytic_dispersion.operator_eigenvalues)."),
                "observable": "eps_instr(s) = |f_extract(s) - f_disc(s)|",
                "sound": "eps_instr(s) <= 1.0 MHz -> extraction SOUND at that rung",
                "attribute": ("eps_instr(s) >= 10.0 MHz -> extraction "
                              "INSTRUMENT-LIMITED at that rung; (4) "
                              "attributed there"),
                "inconclusive": "1 MHz < eps_instr < 10 MHz",
                "validity_gate": ("the twin is only admissible if its "
                                  "in-band line is single and its "
                                  "dominance is >= 10 at every rung"),
            },
            "D4b_independent_reference": {
                "model": "f(h) = f_inf - C * h**p, h = dz_fine(s) = 0.25*s mm",
                "fit": ("nonlinear least squares on the 5 ladder rungs "
                        "s in {1.5,1.0,0.75,0.6,0.5} (5 points, 3 "
                        "parameters); the s=0.25 rung is NOT in the fit"),
                "outlier_verdict": (
                    "the s=0.25 reference rung is an OUTLIER (mechanism 4 "
                    "attributed) iff |f(0.25) - f_pred(0.25)| >= 5 * RMS "
                    "residual of the 5-point fit AND sign(f(0.25)-f(0.5)) "
                    "= -sign(f(0.5)-f(0.6)) (it breaks the monotone trend)"),
                "vindicated": ("|f(0.25) - f_pred(0.25)| <= 3 * RMS "
                               "residual"),
                "inconclusive": "between 3x and 5x RMS residual",
                "outlier_k": 5.0, "vindicated_k": 3.0,
            },
            "D4c_independent_estimators": {
                "what": ("re-extract f from the SAME stored probe record "
                         "of every rung with estimators that do not share "
                         "find_resonances's un-antialiased [::step] "
                         "subsampling"),
                "E1": "Result.find_resonances (incumbent)",
                "E2": ("FFT bandpass over BAND of the ring-down + Hilbert "
                       "analytic-signal phase-slope linear fit"),
                "E3": ("rfx.harminv.harminv on the full ring-down with "
                       "anti-aliased decimation (decimate='auto')"),
                "E4": ("4-parameter damped-sinusoid nonlinear least "
                       "squares on the bandpassed ring-down"),
                "consensus_rule": ("E2/E3/E4 agree iff max pairwise spread "
                                   "<= 1.0 MHz; their mean is then the "
                                   "trustworthy value at that rung"),
                "attribute": "|E1 - consensus| >= 10.0 MHz at a rung",
                "exonerate": "|E1 - consensus| <= 1.0 MHz at a rung",
                "spread_hz": INSTR_SOUND_HZ,
                "attribute_hz": INSTR_LIMITED_HZ,
                "exonerate_hz": INSTR_SOUND_HZ,
            },
        },

        "apportionment": {
            "rule": (
                "Delta_total = |f_E1(0.25) - f_inf| (the reference rung's "
                "departure from the D4b independent reference). "
                "Delta_instr = |f_E1(0.25) - f_consensus(0.25)| is charged "
                "to (4a). Delta_phys = |f_consensus(0.25) - f_inf| is the "
                "remaining solver/physics part, apportioned between (1), "
                "(2) and (3) by their own verdicts. Report both as "
                "percentages of Delta_total; a mechanism whose own "
                "discriminator EXONERATES it may not be charged any part."),
        },

        "remedy_licence": {
            "4a_dominates": ("Delta_instr >= 0.5*Delta_total -> the remedy "
                             "is in the extraction instrument: fix the "
                             "record handling in find_resonances and "
                             "re-run the ladder; the floor must vanish "
                             "(demonstrated, not asserted)"),
            "1_dominates": "scale-consistent geometry realization / ladder design rule",
            "3_dominates": "fix the observable",
            "2_dominates_and_irreducible": (
                "NO code fix. Produce the exact envelope statement "
                "(number + scope + what it bounds) and state that #715's "
                "baseline must be quoted against it."),
        },

        "stop_conditions": [
            "D0 fails -> STOP and report irreproducibility",
            ("every discriminator returns INCONCLUSIVE -> STOP, report "
             "inconclusive, implement no remedy"),
        ],
    }


def main():
    out = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", OUT)
    a = out["D1_geometry_quantization"]["arithmetic_prediction"]
    print("D1 arithmetic prediction: exact at every rung =",
          a["exact_at_every_rung"],
          " max residual = %.3e m" % a["max_residual_over_all_rungs_m"])
    d4 = out["D4_reference_quality"]["instrument_resolution_derivation"]
    print("D4 CRB = %.3g Hz -> sound <= %.3g Hz, limited >= %.3g Hz"
          % (d4["cramer_rao_hz"], d4["sound_hz"], d4["limited_hz"]))
    print("D2 wedge nu = %.4f -> predicted order %.4f"
          % (WEDGE_NU, WEDGE_ORDER))


if __name__ == "__main__":
    main()
