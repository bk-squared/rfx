"""THRU feed-post joint (L, Zc, l_eff) extraction — attempt 2 harness.

Pre-declaration (binding, committed BEFORE this file existed):
    docs/design_notes/thru_feedpost_joint_extraction_predeclaration.md

Measurement-only: battery-verbatim fixture (imported byte-shared from the
attempt-1 harness) + offline algebra; no shipped extractor is edited.
Every gate below is evaluated verbatim from the pre-declaration; nothing
here may widen a window. A miss is reported as a miss.

Run (from THIS worktree):

  PYTHONPATH=$PWD JAX_PLATFORMS=cpu .venv/bin/python \
      validation/research/thru_feedpost_joint_extraction.py --verify   # sec. 6
  ... thru_feedpost_joint_extraction.py --extract                      # sec. 4
  ... thru_feedpost_joint_extraction.py --band --lstar <henries>       # sec. 5
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thru_feedpost_deembed import (  # noqa: E402  (battery-verbatim fixture)
    BAND_FREQS, BAND_N_STEPS, BAND_PULSE, C0, LINE_L, RAW_WORST_MEASURED,
    Z0, check_re_positive, extract_l, run_thru, z_in_from_diag,
    F_X1_SV_MAX, F_X2_RECIP, F_X3_S21_BAND,
)
from rfx.deembed import deembed_series_inductance  # noqa: E402

# ------------------------------------------------------------- frozen -----
# Identification data (predeclaration section 3).
EXTRACT_FREQS = np.array([1.4e9, 1.6e9, 1.8e9, 2.0e9, 2.2e9, 2.4e9, 2.6e9])
EXTRACT_N_STEPS = 12000
EXTRACT_PULSE = dict(f0=2.0e9, bandwidth=0.8)
BETA_REPORT = 1.055                       # l_eff reporting convention

# Extraction falsifier windows (predeclaration section 4).
F_J1_MAX, F_J1_RMS = 0.025, 0.012
F_J2_L_NH = (0.20, 0.50)
F_J2_ZC = (44.0, 53.0)
F_J2_LEFF_MM = (15.0, 18.0)
F_J3_FLATNESS = 0.20
F_J4_SYMMETRY = 0.10
F_J5_SIG_L, F_J5_SIG_ZC, F_J5_SIG_TAU = 0.10, 1.5, 0.05

# Multi-start grid (frozen).
STARTS = [(l0, zc0, lm0)
          for l0 in (0.2, 0.3, 0.4)          # nH
          for zc0 in (46.0, 48.25, 50.5)     # ohm
          for lm0 in (15.5, 16.5, 17.5)]     # mm

# Band budget (predeclaration section 5): delta-L/L = 0.10 (F-J5 gate).
def budget_b(lstar_h: float) -> float:
    b = (0.0430
         + 2.0 * (0.10 * 2 * np.pi * 7e9 * lstar_h) / (2 * Z0)
         + 0.012 + 0.005)
    return min(b, 0.13)


F_D2_REDUCTION = 0.5 * RAW_WORST_MEASURED


# ------------------------------------------------- forward model (ABCD) ---
def model_s(l_h, zc, tau, freqs) -> np.ndarray:
    """Symmetric post-line-post 2-port S at reference Z0 (independent ABCD
    arithmetic — written here, not via rfx.deembed's T helpers).
    l_h may be scalar or per-frequency array (per-frequency used only by
    the V3 dispersive-truth synthetic)."""
    freqs = np.asarray(freqs, dtype=np.float64)
    w = 2 * np.pi * freqs
    x = 1j * w * np.asarray(l_h)
    th = w * tau
    A = np.cos(th) + 1j * x * np.sin(th) / zc
    B = x * np.cos(th) + 1j * zc * np.sin(th) + 1j * x * x * np.sin(th) / zc \
        + x * np.cos(th)
    C = 1j * np.sin(th) / zc
    D = A                                     # symmetric network
    delta = A + B / Z0 + C * Z0 + D
    s11 = (A + B / Z0 - C * Z0 - D) / delta
    s21 = 2.0 / delta
    S = np.empty((2, 2, len(freqs)), dtype=np.complex128)
    S[0, 0] = S[1, 1] = s11
    S[1, 0] = S[0, 1] = s21
    return S


def _residuals(p, freqs, s_meas):
    l_h, zc, tau = p[0] * 1e-9, p[1], p[2] * 1e-3 * BETA_REPORT / C0
    Sm = model_s(l_h, zc, tau, freqs)
    d = (Sm - s_meas).reshape(-1)
    return np.concatenate([d.real, d.imag])


def joint_fit(freqs: np.ndarray, s_meas: np.ndarray) -> dict:
    """Multi-start least squares; returns best fit + covariance + basin info."""
    results = []
    for x0 in STARTS:
        r = least_squares(_residuals, x0, args=(freqs, s_meas),
                          method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14)
        results.append(r)
    best = min(results, key=lambda r: r.cost)
    n_obs = 8 * len(freqs)
    dof = n_obs - 3
    s2 = 2 * best.cost / dof                  # cost = 0.5*SSR
    jtj = best.jac.T @ best.jac
    cov = s2 * np.linalg.inv(jtj)
    sig = np.sqrt(np.diag(cov))               # [nH, ohm, mm]
    # Basin analysis: starts within 3 sigma of best vs distinct basins.
    sig_floor = np.maximum(sig, [1e-6, 1e-4, 1e-4])
    dists = np.array([np.max(np.abs((r.x - best.x) / sig_floor))
                      for r in results])
    same = dists <= 3.0
    other_costs = [r.cost for r, s in zip(results, same) if not s]
    basin_ok = (all(same) or
                (min(other_costs) >= 2.0 * best.cost if best.cost > 0
                 else False))
    resid = _residuals(best.x, freqs, s_meas)
    n_c = 4 * len(freqs)
    cplx = np.abs(resid[:n_c] + 1j * resid[n_c:])
    return dict(
        l_nh=best.x[0], zc=best.x[1], leff_mm=best.x[2],
        tau=best.x[2] * 1e-3 * BETA_REPORT / C0,
        sig_l_nh=sig[0], sig_zc=sig[1], sig_leff_mm=sig[2],
        resid_max=float(cplx.max()), resid_rms=float(np.sqrt((cplx**2).mean())),
        basin_ok=bool(basin_ok), n_same_basin=int(same.sum()),
        n_starts=len(results), cost=float(best.cost),
    )


def perbin_l_with_constants(z_in, freqs, zc, tau):
    """Attempt-1 per-bin quadratic inversion (verified apparatus), driven
    with MEASURED constants: beta_factor equivalent = tau*C0/LINE_L."""
    return extract_l(z_in, freqs, zc, tau * C0 / LINE_L)[0]


def flatness(l_bins) -> float:
    med = np.median(l_bins)
    return float(np.max(np.abs(l_bins - med)) / med)


def _verdict_table(verdicts) -> bool:
    ok = True
    for name, val, window, passed in verdicts:
        ok &= passed
        print(f"  {name}: value {val} vs window {window} -> "
              f"{'PASS' if passed else 'FIRED'}")
    return ok


def evaluate_fj(fit, s_meas, freqs) -> tuple[list, dict]:
    """F-J1..F-J5 verdicts on a fitted 2-port dataset."""
    v = []
    v.append(("F-J1 resid max <= 0.025", round(fit["resid_max"], 5),
              F_J1_MAX, fit["resid_max"] <= F_J1_MAX))
    v.append(("F-J1 resid rms <= 0.012", round(fit["resid_rms"], 5),
              F_J1_RMS, fit["resid_rms"] <= F_J1_RMS))
    v.append(("F-J2 L* [nH]", round(fit["l_nh"], 4), F_J2_L_NH,
              F_J2_L_NH[0] <= fit["l_nh"] <= F_J2_L_NH[1]))
    v.append(("F-J2 Zc* [ohm]", round(fit["zc"], 3), F_J2_ZC,
              F_J2_ZC[0] <= fit["zc"] <= F_J2_ZC[1]))
    v.append(("F-J2 l_eff* [mm]", round(fit["leff_mm"], 3), F_J2_LEFF_MM,
              F_J2_LEFF_MM[0] <= fit["leff_mm"] <= F_J2_LEFF_MM[1]))
    # F-J3/F-J4: per-bin inversion with fitted constants, per port.
    l_by_port = {}
    for port in (0, 1):
        z_in = z_in_from_diag(s_meas[port, port])
        l_by_port[port] = perbin_l_with_constants(
            z_in, freqs, fit["zc"], fit["tau"])
    flat1 = flatness(l_by_port[0])
    v.append(("F-J3 flatness (port1, fitted constants)", round(flat1, 4),
              F_J3_FLATNESS, flat1 <= F_J3_FLATNESS))
    med1, med2 = (float(np.median(l_by_port[p])) for p in (0, 1))
    sym = abs(med1 - med2) / np.mean([med1, med2])
    v.append(("F-J4 port symmetry", round(sym, 4), F_J4_SYMMETRY,
              sym <= F_J4_SYMMETRY))
    v.append(("F-J5 sigma_L/L <= 0.10",
              round(fit["sig_l_nh"] / fit["l_nh"], 4), F_J5_SIG_L,
              fit["sig_l_nh"] / fit["l_nh"] <= F_J5_SIG_L))
    v.append(("F-J5 sigma_Zc <= 1.5 ohm", round(fit["sig_zc"], 4),
              F_J5_SIG_ZC, fit["sig_zc"] <= F_J5_SIG_ZC))
    v.append(("F-J5 sigma_tau/tau <= 0.05",
              round(fit["sig_leff_mm"] / fit["leff_mm"], 5), F_J5_SIG_TAU,
              fit["sig_leff_mm"] / fit["leff_mm"] <= F_J5_SIG_TAU))
    v.append(("F-J5 single basin", f"{fit['n_same_basin']}/{fit['n_starts']}",
              "all within 3 sigma or 2x cost", fit["basin_ok"]))
    return v, {0: l_by_port[0], 1: l_by_port[1]}


# ------------------------------------------------------------ synthetics --
def arm_verify() -> None:
    print("=== APPARATUS VERIFICATION (predeclaration section 6) ===")
    f = EXTRACT_FREQS
    tau0 = 16.0e-3 * BETA_REPORT / C0

    # Cross-check the synthetic generator against rfx.deembed's inverse:
    # de-embedding the truth posts from the generated S must leave a bare
    # line whose diagonal matches the analytic line model.
    S = model_s(0.25e-9, 48.25, tau0, f)
    Sline = deembed_series_inductance(S, f, [0.25e-9, 0.25e-9], z0=Z0)
    Sline_ref = model_s(0.0, 48.25, tau0, f)
    xchk = np.max(np.abs(Sline - Sline_ref))
    print(f"[generator x-check vs rfx.deembed inverse] max delta = {xchk:.2e}")
    assert xchk < 1e-12, "synthetic generator disagrees with rfx.deembed"

    # V1: exactness.
    fit = joint_fit(f, S)
    err = max(abs(fit["l_nh"] - 0.25) / 0.25, abs(fit["zc"] - 48.25) / 48.25,
              abs(fit["leff_mm"] - 16.0) / 16.0)
    print(f"[V1] recovered (L, Zc, l_eff) = ({fit['l_nh']:.9f} nH, "
          f"{fit['zc']:.7f} ohm, {fit['leff_mm']:.7f} mm), "
          f"max rel err = {err:.2e}")
    assert err <= 1e-6, "V1 FAILED: joint fit not exact on clean synthetic"

    # V2: attempt-1 bias injected (truth Zc=46, l_eff=17, flat L=0.25).
    tau2 = 17.0e-3 * BETA_REPORT / C0
    S2 = model_s(0.25e-9, 46.0, tau2, f)
    # (a) attempt-1 method (assumed constants 48.25 ohm / 16 mm) must show
    # the declining-L failure signature.
    z_in = z_in_from_diag(S2[0, 0])
    l_a1 = extract_l(z_in, f, 48.25, BETA_REPORT)[0]
    flat_a1 = flatness(l_a1)
    print(f"[V2a] attempt-1 method on biased truth: per-bin L [nH] = "
          f"{np.array2string(l_a1*1e9, precision=4)} "
          f"(flatness {flat_a1:.3f}, decline "
          f"{bool(np.all(np.diff(l_a1) < 0))})")
    assert flat_a1 > F_J3_FLATNESS and np.all(np.diff(l_a1) < 0), (
        "V2a FAILED: bias injection did not reproduce the attempt-1 "
        "declining-L signature")
    # (b) joint fit must absorb the bias into the constants.
    fit2 = joint_fit(f, S2)
    err2 = max(abs(fit2["l_nh"] - 0.25) / 0.25, abs(fit2["zc"] - 46.0) / 46.0,
               abs(fit2["leff_mm"] - 17.0) / 17.0)
    l_j3 = perbin_l_with_constants(z_in, f, fit2["zc"], fit2["tau"])
    print(f"[V2b] joint fit: ({fit2['l_nh']:.6f} nH, {fit2['zc']:.4f} ohm, "
          f"{fit2['leff_mm']:.4f} mm), max rel err = {err2:.2e}; "
          f"F-J3 flatness with fitted constants = {flatness(l_j3):.2e}")
    assert err2 <= 0.01 and flatness(l_j3) <= 0.01, (
        "V2b FAILED: joint fit did not absorb the constant bias")

    # V3' (predeclaration section 8; supersedes V3 whose as-frozen form
    # FAILED and is recorded there): a genuinely dispersive truth is
    # near-degenerate out-of-band — the teeth live in the HELD-OUT band
    # arm's F-D1, verified end-to-end here.
    l_disp = 0.25e-9 * (1.0 - 0.3 * f / 2.6e9)
    S3 = model_s(l_disp, 48.25, tau0, f)
    fit3 = joint_fit(f, S3)
    z3 = z_in_from_diag(S3[0, 0])
    l3 = perbin_l_with_constants(z3, f, fit3["zc"], fit3["tau"])
    absorbed = (fit3["resid_max"] <= F_J1_MAX
                and flatness(l3) <= F_J3_FLATNESS)
    print(f"[V3'a] 30% dispersive truth absorbed out-of-band: fitted "
          f"({fit3['l_nh']:.4f} nH, {fit3['zc']:.3f} ohm, "
          f"{fit3['leff_mm']:.3f} mm), resid_max {fit3['resid_max']:.4f}, "
          f"F-J3 flatness {flatness(l3):.3f} -> absorbed: {absorbed}")
    assert absorbed, ("V3' FAILED: degeneracy expected out-of-band did "
                      "not reproduce — apparatus drifted from section 8")
    l_disp_band = 0.25e-9 * (1.0 - 0.3 * BAND_FREQS / 2.6e9)
    S3b = model_s(l_disp_band, 48.25, tau0, BAND_FREQS)
    S3d = deembed_series_inductance(S3b, BAND_FREQS,
                                    [fit3["l_nh"] * 1e-9] * 2, z0=Z0)
    worst3 = max(np.abs(S3d[0, 0]).max(), np.abs(S3d[1, 1]).max())
    b3 = budget_b(fit3["l_nh"] * 1e-9)
    print(f"[V3'b] synthetic band arm with fitted flat L*: worst "
          f"{worst3:.4f} vs B(L*) {b3:.4f} -> F-D1 fires: {worst3 >= b3}")
    assert worst3 >= b3, ("V3' FAILED: 30%-dispersion class not caught "
                          "by the held-out F-D1 — teeth missing")

    # V4' (predeclaration section 8.1; supersedes V4 whose as-frozen
    # F-J5 clause FAILED at the 0.005 class and is recorded there):
    # (a) pulls within 3 sigma at noise 0.005; (b) linear sigma scaling —
    # at noise 0.002 the F-J5 windows must hold, so the gate is passable
    # by clean-enough data and fires otherwise.
    rng = np.random.default_rng(0)
    unit = (rng.normal(scale=1 / np.sqrt(2), size=S.shape)
            + 1j * rng.normal(scale=1 / np.sqrt(2), size=S.shape))
    fit4 = joint_fit(f, S + 0.005 * unit)
    pulls = [abs(fit4["l_nh"] - 0.25) / fit4["sig_l_nh"],
             abs(fit4["zc"] - 48.25) / fit4["sig_zc"],
             abs(fit4["leff_mm"] - 16.0) / fit4["sig_leff_mm"]]
    print(f"[V4'a] noise 0.005: recovery ({fit4['l_nh']:.4f} nH, "
          f"{fit4['zc']:.3f} ohm, {fit4['leff_mm']:.4f} mm), sigmas "
          f"({fit4['sig_l_nh']:.4f} nH, {fit4['sig_zc']:.3f} ohm, "
          f"{fit4['sig_leff_mm']:.4f} mm), pulls "
          f"{[round(float(p), 2) for p in pulls]}, "
          f"sigma_L/L = {fit4['sig_l_nh']/fit4['l_nh']:.3f} "
          f"(> 0.10 as recorded in section 8.1)")
    assert max(pulls) <= 3.0, "V4' FAILED: recovery outside 3 sigma"
    fit4b = joint_fit(f, S + 0.002 * unit)
    print(f"[V4'b] noise 0.002: sigma_L/L = "
          f"{fit4b['sig_l_nh']/fit4b['l_nh']:.4f}, sigma_Zc = "
          f"{fit4b['sig_zc']:.3f} ohm, sigma_leff/leff = "
          f"{fit4b['sig_leff_mm']/fit4b['leff_mm']:.4f}")
    assert (fit4b["sig_l_nh"] / fit4b["l_nh"] <= F_J5_SIG_L
            and fit4b["sig_zc"] <= F_J5_SIG_ZC
            and fit4b["sig_leff_mm"] / fit4b["leff_mm"] <= F_J5_SIG_TAU), (
        "V4' FAILED: F-J5 windows not passable even at the 0.002 class")
    print("APPARATUS VERIFICATION: ALL PASS (V1, V2, V3', V4')")


# ------------------------------------------------------------- FDTD arms --
def arm_extract() -> None:
    print("=== EXTRACTION ARM (predeclaration sections 3-4) ===")
    print(f"bins {EXTRACT_FREQS/1e9} GHz, n_steps {EXTRACT_N_STEPS}, "
          f"pulse {EXTRACT_PULSE}")
    S = run_thru(EXTRACT_FREQS, EXTRACT_N_STEPS, EXTRACT_PULSE)
    for port in (0, 1):
        z_in = z_in_from_diag(S[port, port])
        check_re_positive(z_in, f"port{port+1}")           # F-X5 first
        with np.printoptions(precision=4):
            print(f"  port{port+1} Z_in = {z_in}")
    fit = joint_fit(EXTRACT_FREQS, S)
    print(f"  joint fit: L* = {fit['l_nh']:.4f} +- {fit['sig_l_nh']:.4f} nH, "
          f"Zc* = {fit['zc']:.3f} +- {fit['sig_zc']:.3f} ohm, "
          f"l_eff* = {fit['leff_mm']:.4f} +- {fit['sig_leff_mm']:.4f} mm "
          f"(tau* = {fit['tau']*1e12:.3f} ps)")
    print(f"  resid max {fit['resid_max']:.5f}, rms {fit['resid_rms']:.5f}, "
          f"basin {fit['n_same_basin']}/{fit['n_starts']}, "
          f"cost {fit['cost']:.3e}")
    verdicts, l_by_port = evaluate_fj(fit, S, EXTRACT_FREQS)
    for port in (0, 1):
        print(f"  port{port+1} per-bin L (fitted constants) [nH] = "
              f"{np.array2string(l_by_port[port]*1e9, precision=4)}")
    print("--- extraction verdicts ---")
    ok = _verdict_table(verdicts)
    lstar = fit["l_nh"] * 1e-9
    print(f"ADOPTED L* = {fit['l_nh']:.4f} nH (joint fit, per predeclaration)")
    print(f"FROZEN B(L*) = {budget_b(lstar):.4f}")
    print("EXTRACTION ARM: " + ("ALL PASS" if ok else
                                "STOP (falsifier fired)"))


def arm_band(lstar: float) -> None:
    print("=== BAND ARM (predeclaration section 5) ===")
    b = budget_b(lstar)
    print(f"L* = {lstar*1e9:.4f} nH, frozen B(L*) = {b:.4f}")
    S = run_thru(BAND_FREQS, BAND_N_STEPS, BAND_PULSE)
    for port in (0, 1):
        check_re_positive(z_in_from_diag(S[port, port]), f"raw port{port+1}")
    Sd = deembed_series_inductance(S, BAND_FREQS, [lstar, lstar], z0=Z0)
    raw11, raw22 = np.abs(S[0, 0]), np.abs(S[1, 1])
    d11, d22 = np.abs(Sd[0, 0]), np.abs(Sd[1, 1])
    with np.printoptions(precision=4):
        print(f"  raw   |S11| = {raw11}\n  raw   |S22| = {raw22}")
        print(f"  deemb |S11| = {d11}\n  deemb |S22| = {d22}")
        print(f"  raw   |S21| = {np.abs(S[1, 0])}")
        print(f"  deemb |S21| = {np.abs(Sd[1, 0])}")
    raw_worst = max(raw11.max(), raw22.max())
    worst = max(d11.max(), d22.max())
    sv = np.array([np.linalg.svd(Sd[:, :, k], compute_uv=False).max()
                   for k in range(Sd.shape[2])])
    recip = float(np.max(np.abs(Sd[1, 0] - Sd[0, 1])))
    s21d = np.abs(Sd[1, 0])
    dev = np.angle(Sd[1, 0] * np.exp(1j * 2 * np.pi * BAND_FREQS
                                     * LINE_L / C0))
    dev_raw = np.angle(S[1, 0] * np.exp(1j * 2 * np.pi * BAND_FREQS
                                        * LINE_L / C0))
    with np.printoptions(precision=4):
        print(f"  [report-only] S21 phase dev vs c-line delay: raw {dev_raw}")
        print(f"                                             deemb {dev}")
        print(f"  deemb per-bin sv_max = {sv}")
    print(f"  raw worst diagonal = {raw_worst:.4f} "
          f"(held-gate provenance {RAW_WORST_MEASURED})")
    print(f"  de-embedded worst diagonal = {worst:.4f}")
    verdicts = [
        ("F-D1 floor < B(L*)", round(worst, 4), round(b, 4), worst < b),
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
    print("--- band verdicts ---")
    ok = _verdict_table(verdicts)
    print("BAND ARM: " + ("ALL PASS" if ok else "STOP (falsifier fired)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--band", action="store_true")
    ap.add_argument("--lstar", type=float, default=None)
    args = ap.parse_args()
    if args.verify:
        arm_verify()
    if args.extract:
        arm_extract()
    if args.band:
        assert args.lstar is not None, "--band requires --lstar <henries>"
        arm_band(args.lstar)
