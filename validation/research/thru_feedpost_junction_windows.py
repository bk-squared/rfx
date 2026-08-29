"""THRU feed-post attempt-4 harness — analytic junction windows (closing lane).

Pre-declaration (binding, committed BEFORE this file existed):
    docs/design_notes/thru_feedpost_junction_windows_predeclaration.md

This harness builds NO Simulation of its own: it imports the verified
attempt-3 apparatus verbatim (fixtures, raw drives, plane channels,
fits) and changes ONLY the window constants named in the attempt-4
pre-declaration. Nothing here may widen a window; a miss is a miss.

Run (from THIS worktree):

  PYTHONPATH=$PWD JAX_PLATFORMS=cpu .venv/bin/python \
      validation/research/thru_feedpost_junction_windows.py --windows   # sec. 3-4
  ... thru_feedpost_junction_windows.py --extract                       # sec. 5
  ... thru_feedpost_junction_windows.py --extract --band                # sec. 6
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thru_feedpost_deembed import (  # noqa: E402  (battery-verbatim fixture)
    BAND_FREQS, BAND_N_STEPS, BAND_PULSE, C0, LINE_L, RAW_WORST_MEASURED,
    Z0, F_X1_SV_MAX, F_X2_RECIP, F_X3_S21_BAND,
    build_thru, check_re_positive, run_thru, z_in_from_diag,
)
from thru_feedpost_twoseg_extraction import (  # noqa: E402  (attempt-3 appar.)
    EXTRACT_FREQS, EXTRACT_N_STEPS, EXTRACT_PULSE, N_REFPLANE,
    SINGLEPOST_CODES, THRU_CODES,
    F_V1_CORR, F_V1_COND, F_V1_SIG_L, F_V1_SIG_TAU,
    F_V2_CORR, F_V2_COND, F_V2_SIG_L, F_V2_SIG_TAU,
    F_C_L_NH, F_C_TAU_PS,
    build_singlepost, fit_singlepost, fit_thru, fit_verdicts,
    instrument_verdicts, plane_channels, raw_drive, wholeport_channels,
    _verdict_table,
)
from rfx.deembed import deembed_line_segment  # noqa: E402
from rfx.sources.sources import GaussianPulse  # noqa: E402

# ---------------------------------------------------------------- frozen --
# Attempt-4 windows (pre-declaration sections 3-5). The F-I/F-V/F-C
# windows are imported verbatim from the attempt-3 harness above.
F_P4_L_NH = (0.20, 0.58)           # analytic junction L window
F_P4_TAU_PS = (2.5, 9.9)           # sqrt(L*C) + 1.30 ps per-fixture syst.
C_P_CLASS_FF = (71.0, 128.0)       # report-only orientation class
F_A1_MAX4, F_A1_RMS4 = 0.017, 0.008    # loss term repaired (0.008 -> 0.001)
F_A2_MAX4, F_A2_RMS4 = 0.052, 0.026
F_D2_REDUCTION = 0.5 * RAW_WORST_MEASURED

# Reproduction gates (pre-declaration section 5, data-reuse statement):
# attempt-3 committed deterministic values; drift beyond float class = STOP.
REPRO_I3 = (0.4976, 7.3302)        # (L nH, tau ps)
REPRO_I2 = (0.5089, 6.7028)
REPRO_DL_NH, REPRO_DTAU_PS = 0.001, 0.01
REPRO_ZC = {"thru p1": (48.662, 48.737), "thru p2": (48.598, 48.726),
            "single-post": (48.596, 48.651)}
REPRO_ZC_TOL = 0.05                # ~0.1 %

# Prior-provenance classes (pre-declaration section 2).
ZC_CLASS = (47.9, 48.6)            # #313 mid-line measured class [ohm]
N_CLASS = (1.048, 1.062)           # #313 beta-factor dx-stable class
MU0_2PI = 2e-7
DX = 0.5e-3
H_POST = 1.0e-3
R0 = 0.135 * DX
R_NY = 0.23 * DX                   # Noda-Yokoyama equivalent-radius class
L_OV = 0.5e-3
W_TR, T_TR = 5.0e-3, 0.5e-3
C_JUNC_FF = (0.0, 20.0)            # bare-junction shunt-C prior class


def _gp_via(h: float, r: float) -> float:
    """Goldfarb-Pucel via/post partial self-inductance [H]."""
    s = np.sqrt(h * h + r * r)
    return MU0_2PI * (h * np.log((h + s) / r) + 1.5 * (r - s))


def derive_windows(verbose: bool = True):
    """Recompute the pre-declared analytic windows term by term (the
    adversary-verification recipe, executable)."""
    lp_lo = ZC_CLASS[0] * N_CLASS[0] / C0      # L' per-length class
    lp_hi = ZC_CLASS[1] * N_CLASS[1] / C0
    cp_lo = N_CLASS[0] / (ZC_CLASS[1] * C0)    # C' per-length class
    cp_hi = N_CLASS[1] / (ZC_CLASS[0] * C0)

    t_l1 = _gp_via(H_POST, R0)
    t_l2 = (0.0, lp_hi * L_OV)
    t_l3 = (-lp_hi * DX / 2, 0.0)
    d_rad = _gp_via(H_POST, R_NY) - t_l1        # one-sided (negative)
    d_25 = 0.25 * t_l1
    t_l4 = (-max(abs(d_rad), d_25), +d_25)
    l_lo_a = t_l1 + t_l2[0] + t_l3[0] + t_l4[0]
    l_hi_a = t_l1 + t_l2[1] + t_l3[1] + t_l4[1]

    t_c1 = (cp_lo * L_OV, cp_hi * L_OV)
    dl_end = 0.412 * H_POST * ((1.0 + 0.3) * (W_TR / H_POST + 0.264)) / (
        (1.0 - 0.258) * (W_TR / H_POST + 0.8))
    c_end = (cp_lo * dl_end, cp_hi * dl_end)
    c_face = 8.854e-12 * W_TR * T_TR / H_POST
    t_c2 = (0.75 * c_end[0], c_end[1] + c_face)
    c_lo = t_c1[0] + t_c2[0] + C_JUNC_FF[0] * 1e-15
    c_hi = t_c1[1] + t_c2[1] + C_JUNC_FF[1] * 1e-15

    tau_geo = (np.sqrt(F_P4_L_NH[0] * 1e-9 * C_P_CLASS_FF[0] * 1e-15),
               np.sqrt(F_P4_L_NH[1] * 1e-9 * C_P_CLASS_FF[1] * 1e-15))
    tau_syst = 1.30e-12

    if verbose:
        nh = 1e9
        print("=== ATTEMPT-4 ANALYTIC WINDOWS (pre-declaration sec. 3-4) ===")
        print(f"L' class = [{lp_lo*nh:.2f}, {lp_hi*nh:.2f}] nH/m ; "
              f"C' class = [{cp_lo*1e12:.2f}, {cp_hi*1e12:.2f}] pF/m")
        print(f"T-L1 Goldfarb-Pucel post (h=1mm, r0=0.135dx) = "
              f"{t_l1*nh:.4f} nH")
        print(f"T-L2 overhang/launch-plane           in [0, "
              f"{t_l2[1]*nh:.4f}] nH")
        print(f"T-L3 junction fringe/mutual (sign -) in "
              f"[{t_l3[0]*nh:.4f}, 0] nH")
        print(f"T-L4 discretization envelope: r_eff->0.23dx gives "
              f"{d_rad*nh:+.4f}; +-25% class {d_25*nh:.4f} -> "
              f"[{t_l4[0]*nh:.4f}, {t_l4[1]*nh:+.4f}] nH")
        print(f"L analytic sum = [{l_lo_a*nh:.4f}, {l_hi_a*nh:.4f}] nH ; "
              f"lower edge unioned with prior #318 witness edge 0.20 ; "
              f"FROZEN [{F_P4_L_NH[0]}, {F_P4_L_NH[1]}] nH")
        print(f"T-C1 overhang = [{t_c1[0]*1e15:.1f}, {t_c1[1]*1e15:.1f}] fF")
        print(f"T-C2 open-end (Hammerstad dl = {dl_end*1e3:.4f} mm; "
              f"-25% / +face {c_face*1e15:.1f} fF) = "
              f"[{t_c2[0]*1e15:.1f}, {t_c2[1]*1e15:.1f}] fF")
        print(f"T-C3 junction prior class = [{C_JUNC_FF[0]}, "
              f"{C_JUNC_FF[1]}] fF")
        print(f"C_p sum = [{c_lo*1e15:.1f}, {c_hi*1e15:.1f}] fF ; FROZEN "
              f"class (report-only) [{C_P_CLASS_FF[0]:.0f}, "
              f"{C_P_CLASS_FF[1]:.0f}] fF")
        print(f"tau geometric sqrt(L*C) = [{tau_geo[0]*1e12:.2f}, "
              f"{tau_geo[1]*1e12:.2f}] ps ; +- per-fixture systematic "
              f"{tau_syst*1e12:.2f} ps -> FROZEN [{F_P4_TAU_PS[0]}, "
              f"{F_P4_TAU_PS[1]}] ps")
        lam = C0 / 2.6e9
        rrad = 40 * np.pi ** 2 * (H_POST / lam) ** 2
        print(f"F-A loss repair: R_rad(2.6 GHz) = {rrad:.4f} ohm -> "
              f"|dS| = {rrad/(2*Z0):.2e}/radiator ; x4 bound 0.001 ; "
              f"F-A1 = 0.005+0.006+0.005+0.001 = {F_A1_MAX4} "
              f"(rms {F_A1_RMS4}) ; F-A2 = 0.025+0.010+0.017 = "
              f"{F_A2_MAX4} (rms {F_A2_RMS4})")
    # Frozen-window consistency asserts (rounding must be INWARD only).
    assert F_P4_L_NH[0] >= min(l_lo_a * 1e9, 0.20) - 1e-12
    assert F_P4_L_NH[1] <= l_hi_a * 1e9 + 1e-12
    assert C_P_CLASS_FF[0] >= c_lo * 1e15 - 0.5
    assert C_P_CLASS_FF[1] <= c_hi * 1e15 + 0.5
    assert F_P4_TAU_PS[0] >= (tau_geo[0] - tau_syst) * 1e12 - 1e-9
    assert F_P4_TAU_PS[1] <= (tau_geo[1] + tau_syst) * 1e12 + 1e-9


def budget_b4(l_h, tau_s, sig_l_h, sig_tau_s, dl_cf_h, dtau_cf_s):
    """Attempt-4 band budget (pre-declaration section 6). Returns
    (formula value, effective gate min(B, 0.13)) and the deltas."""
    w7 = 2 * np.pi * 7e9
    c_p = tau_s ** 2 / l_h
    d_l = 0.035e-9 + 3 * sig_l_h + dl_cf_h
    d_tau = 0.74e-12 + 3 * sig_tau_s + dtau_cf_s
    d_c = (2 * d_tau / tau_s + d_l / l_h) * c_p
    b = 0.0430 + d_l * w7 / Z0 + w7 * d_c * Z0 + 0.012 + 0.005
    return b, min(b, 0.13), d_l, d_tau, d_c


# ------------------------------------------------------------- FDTD arms --
def arm_extract():
    """Re-run the deterministic attempt-3 extraction; evaluate the
    ATTEMPT-4 verdicts (windows from the attempt-4 pre-declaration)."""
    print("=== ATTEMPT-4 EXTRACTION ARM (re-run of the attempt-3 "
          "deterministic fixtures; pre-declaration sec. 5) ===")
    f = EXTRACT_FREQS
    pulse = GaussianPulse(**EXTRACT_PULSE)

    zc_thru, beta_thru = {}, {}
    for j in (0, 1):
        sim = build_thru(pulse, reference_plane_cells=N_REFPLANE)
        raw, dt = raw_drive(sim, f, EXTRACT_N_STEPS, j, THRU_CODES)
        z_in, _, _ = wholeport_channels(raw, j)
        check_re_positive(z_in, f"thru-insitu drive port{j+1}")   # F-X5
        ch = plane_channels(raw, dt, f, j)
        zc_thru[j], beta_thru[j] = ch["zc"], ch["beta"]
    zc_mean = 0.5 * (zc_thru[0].real + zc_thru[1].real)
    beta_mean = 0.5 * (beta_thru[0] + beta_thru[1])

    sp = build_singlepost(pulse, reference_plane_cells=N_REFPLANE)
    raw_sp, dt_sp = raw_drive(sp, f, EXTRACT_N_STEPS, 0, SINGLEPOST_CODES)
    z_in_sp, s11_sp, a_sp = wholeport_channels(raw_sp, 0)
    check_re_positive(z_in_sp, "single-post port1")               # F-X5
    ch_sp = plane_channels(raw_sp, dt_sp, f, 0)
    g_max = float(np.max(np.abs(ch_sp["gamma_top"])))
    assert g_max <= 0.5, (
        f"single-post fixture invalid: max|Gamma_top| = {g_max:.4f} > 0.5")
    t_sp = (ch_sp["out_top"] / np.sqrt(ch_sp["zc"].real)) / a_sp

    S = run_thru(f, EXTRACT_N_STEPS, EXTRACT_PULSE)
    for port in (0, 1):
        check_re_positive(z_in_from_diag(S[port, port]),
                          f"thru port{port+1}")                   # F-X5

    fitT = fit_thru(S, f, zc_mean, beta_mean)
    fitS = fit_singlepost(s11_sp, t_sp, ch_sp["zc"].real,
                          ch_sp["gamma_top"], f)
    for tag, fit in (("I3 thru", fitT), ("I2 single-post", fitS)):
        print(f"  {tag}: L = {fit['l_nh']:.4f} +- {fit['sig_l_nh']:.4f} nH,"
              f" tau = {fit['tau_ps']:.4f} +- {fit['sig_tau_ps']:.4f} ps"
              f" (Zp = {fit['l_nh']/fit['tau_ps']*1e3:.2f} ohm, C_p = "
              f"{fit['tau_ps']**2/fit['l_nh']:.2f} fF); corr "
              f"{fit['corr']:.4f}, cond {fit['cond']:.2f}, resid max/rms "
              f"{fit['resid_max']:.5f}/{fit['resid_rms']:.5f}, basin "
              f"{fit['n_same_basin']}/{fit['n_starts']}")

    # --- reproduction gates (deterministic-fixture data-reuse statement)
    print("--- reproduction vs attempt-3 committed values ---")
    repro = []
    for tag, fit, exp in (("I3", fitT, REPRO_I3), ("I2", fitS, REPRO_I2)):
        repro.append((f"repro {tag} |dL| [nH]",
                      round(abs(fit["l_nh"] - exp[0]), 5), REPRO_DL_NH,
                      abs(fit["l_nh"] - exp[0]) <= REPRO_DL_NH))
        repro.append((f"repro {tag} |dtau| [ps]",
                      round(abs(fit["tau_ps"] - exp[1]), 5), REPRO_DTAU_PS,
                      abs(fit["tau_ps"] - exp[1]) <= REPRO_DTAU_PS))
    for lab, zc in (("thru p1", zc_thru[0]), ("thru p2", zc_thru[1]),
                    ("single-post", ch_sp["zc"])):
        lo, hi = REPRO_ZC[lab]
        dev = max(abs(float(zc.real.min()) - lo),
                  abs(float(zc.real.max()) - hi))
        repro.append((f"repro Zc range ({lab}) [ohm]", round(dev, 4),
                      REPRO_ZC_TOL, dev <= REPRO_ZC_TOL))
    repro_ok = _verdict_table(repro)
    assert repro_ok, ("APPARATUS DRIFT: deterministic fixtures failed to "
                      "reproduce the attempt-3 committed values — STOP")

    # --- attempt-4 verdicts
    print("--- attempt-4 extraction verdicts ---")
    v = []
    v += instrument_verdicts(
        [zc_thru[0], zc_thru[1], ch_sp["zc"]],
        [beta_thru[0], beta_thru[1], ch_sp["beta"]],
        ["thru p1", "thru p2", "single-post"])
    dzc = float(np.max(np.abs(zc_thru[0].real - zc_thru[1].real)))
    dbt = float(np.max(np.abs(beta_thru[0] / beta_thru[1] - 1)))
    v.append(("F-I2 |Zc_p1 - Zc_p2| [ohm]", round(dzc, 4), 1.2, dzc <= 1.2))
    v.append(("F-I2 |beta ratio - 1|", round(dbt, 5), 0.02, dbt <= 0.02))
    dzc4 = float(np.max(np.abs(zc_mean - ch_sp["zc"].real)))
    dbt4 = float(np.max(np.abs(beta_mean / ch_sp["beta"] - 1)))
    v.append(("F-I4 |Zc_thru - Zc_sp| [ohm]", round(dzc4, 4), 1.2,
              dzc4 <= 1.2))
    v.append(("F-I4 |beta ratio - 1|", round(dbt4, 5), 0.02, dbt4 <= 0.02))
    # NOTE (pre-declaration section 9): the imported attempt-3 helper
    # hardcodes the SUPERSEDED attempt-3 F-P windows into its rows; those
    # are not attempt-4 falsifiers (the attempt-4 F-P4 rows below are the
    # binding ones) and are filtered here.
    v += [row for row in fit_verdicts(
        fitT, "I3 thru", F_V1_CORR, F_V1_COND, F_V1_SIG_L,
        F_V1_SIG_TAU, F_A1_MAX4, F_A1_RMS4) if not row[0].startswith("F-P ")]
    v += [row for row in fit_verdicts(
        fitS, "I2 single-post", F_V2_CORR, F_V2_COND,
        F_V2_SIG_L, F_V2_SIG_TAU, F_A2_MAX4, F_A2_RMS4)
        if not row[0].startswith("F-P ")]
    for tag, fit in (("I3", fitT), ("I2", fitS)):
        v.append((f"F-P4 L_p ({tag}) [nH]", round(fit["l_nh"], 4),
                  F_P4_L_NH,
                  F_P4_L_NH[0] <= fit["l_nh"] <= F_P4_L_NH[1]))
        v.append((f"F-P4 tau_p ({tag}) [ps]", round(fit["tau_ps"], 4),
                  F_P4_TAU_PS,
                  F_P4_TAU_PS[0] <= fit["tau_ps"] <= F_P4_TAU_PS[1]))
        c_p = fit["tau_ps"] ** 2 / fit["l_nh"]
        print(f"  [report-only] C_p ({tag}) = {c_p:.2f} fF "
              f"(orientation class {C_P_CLASS_FF}), Zp = "
              f"{fit['l_nh']/fit['tau_ps']*1e3:.2f} ohm")
    dl_c = abs(fitT["l_nh"] - fitS["l_nh"])
    dt_c = abs(fitT["tau_ps"] - fitS["tau_ps"])
    v.append(("F-C |L_thru - L_sp| [nH]", round(dl_c, 4), F_C_L_NH,
              dl_c <= F_C_L_NH))
    v.append(("F-C |tau_thru - tau_sp| [ps]", round(dt_c, 4), F_C_TAU_PS,
              dt_c <= F_C_TAU_PS))
    ok = _verdict_table(v)
    print("ATTEMPT-4 EXTRACTION ARM: "
          + ("ALL PASS" if ok else "STOP (falsifier fired)"))
    if not ok:
        return None
    lstar, taustar = fitT["l_nh"] * 1e-9, fitT["tau_ps"] * 1e-12
    b, b_eff, d_l, d_tau, d_c = budget_b4(
        lstar, taustar, fitT["sig_l_nh"] * 1e-9, fitT["sig_tau_ps"] * 1e-12,
        dl_c * 1e-9, dt_c * 1e-12)
    print(f"ADOPTED (L*, tau*) = ({fitT['l_nh']:.4f} nH, "
          f"{fitT['tau_ps']:.4f} ps) from I3, per predeclaration")
    print(f"FROZEN attempt-4 budget: delta_L = {d_l*1e9:.4f} nH, "
          f"delta_tau = {d_tau*1e12:.4f} ps, delta_C = {d_c*1e15:.1f} fF; "
          f"B_formula = {b:.4f} -> B_eff = min(B, 0.13) = {b_eff:.4f}")
    return dict(lstar=lstar, taustar=taustar, b_eff=b_eff, b_formula=b)


def arm_band(adopted: dict):
    print("=== ATTEMPT-4 BAND ARM (pre-declaration sec. 6) ===")
    lstar, taustar = adopted["lstar"], adopted["taustar"]
    b_eff = adopted["b_eff"]
    zp = lstar / taustar
    print(f"L* = {lstar*1e9:.4f} nH, tau* = {taustar*1e12:.4f} ps "
          f"(Zp* = {zp:.2f} ohm), frozen B_eff = {b_eff:.4f}")
    S = run_thru(BAND_FREQS, BAND_N_STEPS, BAND_PULSE)
    for port in (0, 1):
        check_re_positive(z_in_from_diag(S[port, port]), f"raw port{port+1}")
    Sd = deembed_line_segment(S, BAND_FREQS, [(zp, taustar)] * 2, z0=Z0)
    raw11, raw22 = np.abs(S[0, 0]), np.abs(S[1, 1])
    d11, d22 = np.abs(Sd[0, 0]), np.abs(Sd[1, 1])
    with np.printoptions(precision=4):
        print(f"  raw   |S11| = {raw11}\n  raw   |S22| = {raw22}")
        print(f"  deemb |S11| = {d11}\n  deemb |S22| = {d22}")
        print(f"  raw   |S21| = {np.abs(S[1, 0])}")
        print(f"  deemb |S21| = {np.abs(Sd[1, 0])}")
    raw_worst = max(raw11.max(), raw22.max())
    worst = float(max(d11.max(), d22.max()))
    sv = np.array([np.linalg.svd(Sd[:, :, k], compute_uv=False).max()
                   for k in range(Sd.shape[2])])
    recip = float(np.max(np.abs(Sd[1, 0] - Sd[0, 1])))
    s21d = np.abs(Sd[1, 0])
    dev = np.angle(Sd[1, 0] * np.exp(1j * 2 * np.pi * BAND_FREQS
                                     * LINE_L / C0))
    with np.printoptions(precision=4):
        print(f"  [report-only] deemb S21 phase dev vs c-line delay: {dev}")
        print(f"  deemb per-bin sv_max = {sv}")
    print(f"  raw worst diagonal = {raw_worst:.4f} "
          f"(held-gate provenance {RAW_WORST_MEASURED})")
    print(f"  de-embedded worst diagonal = {worst:.4f}")
    verdicts = [
        ("F-D1 floor < B_eff", round(worst, 4), round(b_eff, 4),
         worst < b_eff),
        ("F-D2 reduction < 0.1455", round(worst, 4), F_D2_REDUCTION,
         worst < F_D2_REDUCTION),
        ("F-X1 sv_max <= 1.01", round(float(sv.max()), 5), F_X1_SV_MAX,
         sv.max() <= F_X1_SV_MAX),
        ("F-X2 |S21d-S12d| <= 1e-3", f"{recip:.2e}", F_X2_RECIP,
         recip <= F_X2_RECIP),
        ("F-X3 |S21d| in [0.93, 1.005]",
         (round(float(s21d.min()), 4), round(float(s21d.max()), 4)),
         F_X3_S21_BAND,
         s21d.min() >= F_X3_S21_BAND[0] and s21d.max() <= F_X3_S21_BAND[1]),
    ]
    print("--- attempt-4 band verdicts ---")
    ok = _verdict_table(verdicts)
    gate = float(np.ceil(worst * 1.25 * 1000) / 1000)
    placeable = gate <= b_eff
    print(f"  candidate pin: worst * 1.25 rounded up = {gate:.3f} "
          f"(placeable <= B_eff: {placeable})")
    print("ATTEMPT-4 BAND ARM: "
          + ("ALL PASS" if ok and placeable else "STOP (falsifier fired)"
             if not ok else "STOP (pin not placeable within B_eff)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--band", action="store_true")
    args = ap.parse_args()
    derive_windows(verbose=args.windows or not (args.extract or args.band))
    adopted = None
    if args.extract:
        adopted = arm_extract()
    if args.band:
        assert adopted is not None, (
            "--band requires --extract ALL PASS in the same invocation "
            "(the budget freezes on the adopted parameters)")
        arm_band(adopted)
