"""THRU feed-post two-segment extraction — attempt 3 harness.

Pre-declaration (binding, committed BEFORE this file existed):
    docs/design_notes/thru_feedpost_twoseg_predeclaration.md

Measurement-only: battery-verbatim thru fixture (imported byte-shared
from the attempt-1 harness) + a NEW single-post 1-port fixture + offline
algebra; no shipped extractor is edited. Every gate below is evaluated
verbatim from the pre-declaration; nothing here may widen a window. A
miss is reported as a miss.

Run (from THIS worktree):

  PYTHONPATH=$PWD JAX_PLATFORMS=cpu .venv/bin/python \
      validation/research/thru_feedpost_twoseg_extraction.py --verify   # sec. 9
  ... thru_feedpost_twoseg_extraction.py --extract                      # sec. 3-4
  ... thru_feedpost_twoseg_extraction.py --band \
        --lstar <henries> --taustar <seconds>                           # sec. 5
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
    BAND_FREQS, BAND_N_STEPS, BAND_PULSE, C0, CPML_LAYERS, DOMAIN, DX,
    FREQ_MAX, H, LINE_L, RAW_WORST_MEASURED, W, X1, Y_MID, Z0,
    F_X1_SV_MAX, F_X2_RECIP, F_X3_S21_BAND,
    build_thru, check_re_positive, run_thru, z_in_from_diag,
)
from thru_feedpost_joint_extraction import (  # noqa: E402  (attempt-2 appar.)
    EXTRACT_FREQS, EXTRACT_N_STEPS, EXTRACT_PULSE, joint_fit,
)
from rfx import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.deembed import deembed_line_segment  # noqa: E402
from rfx.sources.sources import GaussianPulse  # noqa: E402
from rfx.probes.refplane import (  # noqa: E402
    refplane_beta, refplane_centered_current, refplane_split,
    refplane_zc_two_plane,
)
from rfx import Box  # noqa: E402

# ------------------------------------------------------------- frozen -----
N_REFPLANE = 10                    # clean-plane class (>= 10 cells; #313)

# Instrument gates (predeclaration section 4).
F_I1_IM_RE = 0.03
F_I2_ZC_OHM, F_I2_BETA = 1.2, 0.02
F_I3_ZC = (44.0, 53.0)
F_I3_BETA_FAC = (1.00, 1.10)
F_I4_ZC_OHM, F_I4_BETA = 1.2, 0.02

# Fit adequacy + physicality + identifiability (section 4).
F_A1_MAX, F_A1_RMS = 0.025, 0.012
F_A2_MAX, F_A2_RMS = 0.06, 0.03
F_P_L_NH = (0.20, 0.50)
F_P_TAU_PS = (1.67, 8.34)
F_V1_CORR, F_V1_COND, F_V1_SIG_L, F_V1_SIG_TAU = 0.90, 60.0, 0.10, 0.15
F_V2_CORR, F_V2_COND, F_V2_SIG_L, F_V2_SIG_TAU = 0.90, 15.0, 0.10, 0.20
F_C_L_NH, F_C_TAU_PS = 0.11, 2.6

# Multi-start grid + bounds (frozen; bounds kill the exact
# (-Zp, -tau) mirror of the segment model).
STARTS = [(l0, t0) for l0 in (0.25, 0.35, 0.45) for t0 in (2.5, 4.0, 6.0)]
BOUNDS = ([0.05, 0.5], [1.0, 12.0])          # L [nH], tau [ps]

# Band budget (predeclaration section 5).
F_D2_REDUCTION = 0.5 * RAW_WORST_MEASURED


def budget_b(lstar_h: float, taustar_s: float) -> float:
    w7 = 2 * np.pi * 7e9
    c_p = taustar_s ** 2 / lstar_h
    b = (0.0430
         + 2.0 * (0.13 * lstar_h * w7) / (2 * Z0)
         + 2.0 * (w7 * 0.60 * c_p * Z0 / 2.0)
         + 0.012 + 0.005)
    return min(b, 0.13)


# ------------------------------------------------- forward models ---------
def seg_junction_s(zp, tau, w, z0a, z0b):
    """Power-wave S of the (zp, tau) line segment between reference
    impedances z0a (port side) and z0b (line side). Independent ABCD
    arithmetic — written here, not via rfx.deembed's T helpers."""
    th = w * tau
    a = np.cos(th)
    b = 1j * zp * np.sin(th)
    c = 1j * np.sin(th) / zp
    den = a * z0b + b + c * z0a * z0b + a * z0a
    s11 = (a * z0b + b - c * z0a * z0b - a * z0a) / den
    s21 = 2.0 * np.sqrt(z0a * z0b) / den            # det(ABCD) = 1
    s22 = (-a * z0b + b - c * z0a * z0b + a * z0a) / den
    return s11, s21, s22


def thru_model_s(l_h, tau, zc, beta, freqs) -> np.ndarray:
    """post-seg . line(zc(f), beta(f)*LINE_L) . post-seg, reference Z0.
    l_h may be per-frequency (V3 dispersive synthetic only)."""
    freqs = np.asarray(freqs, dtype=np.float64)
    l_arr = np.broadcast_to(np.asarray(l_h, dtype=np.float64), freqs.shape)
    zc = np.broadcast_to(np.asarray(zc, dtype=np.float64), freqs.shape)
    beta = np.broadcast_to(np.asarray(beta, dtype=np.float64), freqs.shape)
    w = 2 * np.pi * freqs
    S = np.empty((2, 2, len(freqs)), dtype=np.complex128)
    for k in range(len(freqs)):
        zp = l_arr[k] / tau
        th = w[k] * tau
        seg = np.array([[np.cos(th), 1j * zp * np.sin(th)],
                        [1j * np.sin(th) / zp, np.cos(th)]])
        tl = beta[k] * LINE_L
        ln = np.array([[np.cos(tl), 1j * zc[k] * np.sin(tl)],
                       [1j * np.sin(tl) / zc[k], np.cos(tl)]])
        m = seg @ ln @ seg
        a, b, c, d = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
        delta = a + b / Z0 + c * Z0 + d
        S[0, 0, k] = (a + b / Z0 - c * Z0 - d) / delta
        S[1, 1, k] = (-a + b / Z0 - c * Z0 + d) / delta
        S[1, 0, k] = 2.0 / delta
        S[0, 1, k] = 2.0 * (a * d - b * c) / delta
    return S


def singlepost_model_obs(l_h, tau, zc_sp, gamma_top, freqs):
    """(S11_wp, T) of the single-post fixture: segment loaded by the
    MEASURED downstream reflection gamma_top(f) at the post top."""
    freqs = np.asarray(freqs, dtype=np.float64)
    w = 2 * np.pi * freqs
    zp = l_h / tau
    s11p, s21p, s22p = seg_junction_s(zp, tau, w, Z0, zc_sp)
    denom = 1.0 - s22p * gamma_top
    s11 = s11p + s21p * s21p * gamma_top / denom
    t = s21p / denom
    return s11, t


# ------------------------------------------------- fitting machinery ------
def _cov_report(res, n_obs):
    dof = n_obs - 2
    s2 = 2 * res.cost / dof
    jtj = res.jac.T @ res.jac
    cov = s2 * np.linalg.inv(jtj)
    sig = np.sqrt(np.diag(cov))               # [nH, ps]
    corr = float(cov[0, 1] / (sig[0] * sig[1]))
    js = res.jac * res.x[None, :]             # value-scaled Jacobian
    sv = np.linalg.svd(js, compute_uv=False)
    cond = float(sv[0] / sv[-1])
    return sig, corr, cond


def _multistart(resid_fn, n_obs):
    results = []
    for x0 in STARTS:
        r = least_squares(resid_fn, x0, bounds=BOUNDS, method="trf",
                          xtol=1e-14, ftol=1e-14, gtol=1e-14)
        results.append(r)
    best = min(results, key=lambda r: r.cost)
    sig, corr, cond = _cov_report(best, n_obs)
    sig_floor = np.maximum(sig, [1e-6, 1e-6])
    dists = np.array([np.max(np.abs((r.x - best.x) / sig_floor))
                      for r in results])
    same = dists <= 3.0
    other = [r.cost for r, s in zip(results, same) if not s]
    basin_ok = (all(same) or
                (min(other) >= 2.0 * best.cost if best.cost > 0 else False))
    resid = resid_fn(best.x)
    n_c = len(resid) // 2
    cplx = np.abs(resid[:n_c] + 1j * resid[n_c:])
    return dict(
        l_nh=float(best.x[0]), tau_ps=float(best.x[1]),
        sig_l_nh=float(sig[0]), sig_tau_ps=float(sig[1]),
        corr=corr, cond=cond,
        resid_max=float(cplx.max()),
        resid_rms=float(np.sqrt((cplx ** 2).mean())),
        basin_ok=bool(basin_ok), n_same_basin=int(same.sum()),
        n_starts=len(results), cost=float(best.cost),
    )


def fit_thru(s_meas, freqs, zc_arr, beta_arr):
    """I3: 2-parameter (L, tau) fit on the full complex 2x2 S with the
    line constants FIXED at the measured per-bin values."""
    def resid(p):
        S = thru_model_s(p[0] * 1e-9, p[1] * 1e-12, zc_arr, beta_arr, freqs)
        d = (S - s_meas).reshape(-1)
        return np.concatenate([d.real, d.imag])
    return _multistart(resid, 8 * len(freqs))


def fit_singlepost(s11_meas, t_meas, zc_sp, gamma_top, freqs):
    """I2: 2-parameter (L, tau) fit on (S11_wp, T) with the measured
    per-bin load gamma_top and line impedance zc_sp."""
    def resid(p):
        s11, t = singlepost_model_obs(p[0] * 1e-9, p[1] * 1e-12,
                                      zc_sp, gamma_top, freqs)
        d = np.concatenate([s11 - s11_meas, t - t_meas])
        return np.concatenate([d.real, d.imag])
    return _multistart(resid, 4 * len(freqs))


# ------------------------------------------------- fixtures + raw drives --
X_FAR_SP = 0.028                   # single-post far termination column


def build_singlepost(pulse: GaussianPulse,
                     reference_plane_cells: int | None = None) -> Simulation:
    """Single-post fixture (predeclaration I2 + section-11 repaired
    realization): battery-verbatim port + post + trace cross-section and
    port-side overhang on a 20 mm line, terminated at x = 28 mm in the
    VALIDATED passive matched wire-port class (a PEC trace cannot
    continue into the CPML pad — pec_mask is never pad-extended; the
    section-11 apparatus finding). The far termination's own post sits
    inside the MEASURED Gamma_top load and never enters the model.
    Returns the Simulation (no solve call)."""
    sim = Simulation(
        freq_max=FREQ_MAX, domain=DOMAIN, dx=DX,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        cpml_layers=CPML_LAYERS,
    )
    sim.add(
        Box((X1 - DX, Y_MID - W / 2, H),
            (X_FAR_SP + DX, Y_MID + W / 2, H + DX)),
        material="pec",
    )
    kw = ({} if reference_plane_cells is None
          else {"reference_plane_cells": reference_plane_cells})
    sim.add_port(position=(X1, Y_MID, 0.0), component="ez", impedance=Z0,
                 extent=H, waveform=pulse, direction="-x", **kw)
    sim.add_port(position=(X_FAR_SP, Y_MID, 0.0), component="ez",
                 impedance=Z0, extent=H, excite=False, direction="+x")
    return sim


def raw_drive(sim, freqs, n_steps, drive_idx, expected_codes):
    """One driven production-scan pass returning the RAW port + plane
    accumulators (the refplane plumbing-test hook) and the grid dt."""
    report = sim.preflight()
    for msg in report:
        print(f"[preflight verbatim] {msg}")
    codes = sorted(getattr(i, "code", None) for i in report)
    assert codes == expected_codes, (
        f"fixture preflight drifted: {codes} vs pinned {expected_codes}")
    grid = sim._build_grid()
    mats, dsp, lsp, pm, _, _, _ = sim._assemble_materials(grid)
    raw = sim._forward_from_materials(
        grid, mats, dsp, lsp, n_steps=n_steps, checkpoint=False,
        pec_mask=pm, port_s11_freqs=np.asarray(freqs, dtype=np.float64),
        _sparam_drive_idx=drive_idx, _return_raw_port_sparams=True)
    return raw, float(grid.dt)


THRU_CODES = ["pec_faces_finite_pec", "wire_port_dead_extent_cells",
              "wire_port_dead_extent_cells"]
# refplane_partial_optin is the DELIBERATE design: only the driven port
# opts into the plane instrument (the passive far termination stays on
# the default path; the harness consumes raw plane accumulators only).
SINGLEPOST_CODES = ["pec_faces_finite_pec", "refplane_partial_optin",
                    "wire_port_dead_extent_cells",
                    "wire_port_dead_extent_cells"]


def wholeport_channels(raw, port_idx):
    """(z_in, s11, a) of the DRIVEN port from the raw wire accumulators —
    the exact #764/#770 whole-port conventions (v_port, i as accumulated)."""
    spec, vi = raw["wire"][port_idx]
    i_dft = np.asarray(vi[1], dtype=np.complex128)
    v_port = np.asarray(vi[3], dtype=np.complex128)
    z_in = v_port / i_dft
    s11 = (v_port - Z0 * i_dft) / (v_port + Z0 * i_dft)
    a = (v_port + Z0 * i_dft) / (2.0 * np.sqrt(Z0))
    return z_in, s11, a


def plane_channels(raw, dt, freqs, port_idx):
    """Measured (zc, beta, gamma_top, out_top) from the two raw planes of
    ``port_idx`` — the refplane module's own public extraction math."""
    planes = {int(s.plane_slot): (s, accs) for s, accs in raw["wire_refplane"]
              if int(s.port_index) == port_idx}
    assert set(planes) == {0, 1}, f"expected 2 planes for port {port_idx}"
    spec0, acc0 = planes[0]
    spec1, acc1 = planes[1]
    v0 = np.asarray(acc0[0], dtype=np.complex128)
    v1 = np.asarray(acc1[0], dtype=np.complex128)
    i0 = refplane_centered_current(acc0[1], acc0[2], freqs, dt)
    i1 = refplane_centered_current(acc1[1], acc1[2], freqs, dt)
    zc = refplane_zc_two_plane(v0, i0, v1, i1)
    sign = int(spec0.outboard_sign)
    out0, in0 = refplane_split(v0, i0, zc, sign)
    out1, _ = refplane_split(v1, i1, zc, sign)
    d_sep = int(spec0.n_cells_outboard) * DX
    beta = refplane_beta(out0, out1, d_sep)
    # wrap guard (module constants): beta positive, |beta|*N*dx < 0.9*pi
    assert np.all(beta > 0.0) and np.all(np.abs(beta) * d_sep < 0.9 * np.pi), (
        f"beta wrap guard tripped on port {port_idx}: {beta}")
    gamma_top = (in0 / out0) * np.exp(-2j * beta * d_sep)
    out_top = out0 * np.exp(+1j * beta * d_sep)
    return dict(zc=zc, beta=beta, gamma_top=gamma_top, out_top=out_top)


# ------------------------------------------------------------ verdicts ----
def _verdict_table(verdicts) -> bool:
    ok = True
    for name, val, window, passed in verdicts:
        ok &= passed
        print(f"  {name}: value {val} vs window {window} -> "
              f"{'PASS' if passed else 'FIRED'}")
    return ok


def instrument_verdicts(zc_sets, beta_sets, labels):
    """F-I1/F-I3 on each measured (zc, beta) set."""
    v = []
    for lab, zc, beta in zip(labels, zc_sets, beta_sets):
        im_re = float(np.max(np.abs(zc.imag) / np.abs(zc.real)))
        v.append((f"F-I1 |Im/Re Zc| ({lab})", round(im_re, 4), F_I1_IM_RE,
                  im_re <= F_I1_IM_RE))
        re_lo, re_hi = float(zc.real.min()), float(zc.real.max())
        v.append((f"F-I3 Re(Zc) ({lab})", (round(re_lo, 3), round(re_hi, 3)),
                  F_I3_ZC, F_I3_ZC[0] <= re_lo and re_hi <= F_I3_ZC[1]))
        bf = beta * C0 / (2 * np.pi * EXTRACT_FREQS)
        v.append((f"F-I3 beta/(w/c) ({lab})",
                  (round(float(bf.min()), 4), round(float(bf.max()), 4)),
                  F_I3_BETA_FAC,
                  F_I3_BETA_FAC[0] <= bf.min() and bf.max() <= F_I3_BETA_FAC[1]))
    return v


def fit_verdicts(fit, tag, corr_w, cond_w, sig_l_w, sig_tau_w,
                 res_max_w, res_rms_w):
    v = []
    v.append((f"F-A resid max ({tag})", round(fit["resid_max"], 5),
              res_max_w, fit["resid_max"] <= res_max_w))
    v.append((f"F-A resid rms ({tag})", round(fit["resid_rms"], 5),
              res_rms_w, fit["resid_rms"] <= res_rms_w))
    v.append((f"F-P L* [nH] ({tag})", round(fit["l_nh"], 4), F_P_L_NH,
              F_P_L_NH[0] <= fit["l_nh"] <= F_P_L_NH[1]))
    v.append((f"F-P tau* [ps] ({tag})", round(fit["tau_ps"], 4), F_P_TAU_PS,
              F_P_TAU_PS[0] <= fit["tau_ps"] <= F_P_TAU_PS[1]))
    v.append((f"F-V |corr(L,tau)| ({tag})", round(abs(fit["corr"]), 4),
              corr_w, abs(fit["corr"]) <= corr_w))
    v.append((f"F-V cond(scaled J) ({tag})", round(fit["cond"], 2),
              cond_w, fit["cond"] <= cond_w))
    v.append((f"F-V sigma_L/L ({tag})",
              round(fit["sig_l_nh"] / fit["l_nh"], 4), sig_l_w,
              fit["sig_l_nh"] / fit["l_nh"] <= sig_l_w))
    v.append((f"F-V sigma_tau/tau ({tag})",
              round(fit["sig_tau_ps"] / fit["tau_ps"], 4), sig_tau_w,
              fit["sig_tau_ps"] / fit["tau_ps"] <= sig_tau_w))
    v.append((f"F-V single basin ({tag})",
              f"{fit['n_same_basin']}/{fit['n_starts']}",
              "all within 3 sigma or 2x cost", fit["basin_ok"]))
    return v


# ------------------------------------------------------------ synthetics --
def _synth_plane_channels(freqs, zc_true, beta_true, dt, n, f0, g0):
    """Synthetic raw plane phasors for a two-wave uniform line (the
    refplane test pattern), returned in the raw-accumulator convention:
    V at the plane, UNcorrected loop currents at plane -/+ dx/2."""
    d = n * DX
    w = 2 * np.pi * freqs
    hcorr = np.exp(+1j * w * dt / 2)

    def field(x):
        fwd = f0 * np.exp(-1j * beta_true * x)
        bwd = g0 * np.exp(+1j * beta_true * x)
        return fwd + bwd, (fwd - bwd) / zc_true

    def raw_i(i_exact):
        return i_exact / hcorr, i_exact / hcorr
    out = {}
    for slot, x in ((0, d), (1, 2 * d)):
        v, i = field(x)
        im, ip = raw_i(i)
        out[slot] = (v, im, ip)
    return out


def arm_verify() -> None:
    print("=== APPARATUS VERIFICATION (predeclaration section 9) ===")
    f = EXTRACT_FREQS
    w = 2 * np.pi * f
    zc_t, bfac = 47.3, 1.055
    beta_t = bfac * w / C0
    l_t, tau_t = 0.38e-9, 4.0e-12

    # generator x-check vs rfx.deembed inverse
    S = thru_model_s(l_t, tau_t, zc_t, beta_t, f)
    zp_t = l_t / tau_t
    Sline = deembed_line_segment(S, f, [(zp_t, tau_t)] * 2, z0=Z0)
    Sref = np.empty_like(S)
    for k in range(len(f)):
        tl = beta_t[k] * LINE_L
        a, b = np.cos(tl), 1j * zc_t * np.sin(tl)
        c = 1j * np.sin(tl) / zc_t
        delta = a + b / Z0 + c * Z0 + a
        Sref[0, 0, k] = Sref[1, 1, k] = (a + b / Z0 - c * Z0 - a) / delta
        Sref[1, 0, k] = Sref[0, 1, k] = 2.0 / delta
    xchk = np.max(np.abs(Sline - Sref))
    print(f"[generator x-check vs rfx.deembed inverse] max delta = {xchk:.2e}")
    assert xchk < 1e-12

    # V0: plane-channel math on synthetic two-wave phasors
    dt = 9.6e-13
    gamma_true = 0.02 * np.exp(1j * 1.0)
    f0w = (1.0 + 0.3j) * np.ones(len(f), dtype=np.complex128)
    g0w = gamma_true * f0w                     # gamma at x = 0 (post top)
    acc = _synth_plane_channels(f, zc_t, beta_t, dt, N_REFPLANE, f0w, g0w)

    class _FakeSpec:
        port_index, plane_slot = 0, 0
        outboard_sign, n_cells_outboard = +1, N_REFPLANE
    raw = {"wire_refplane": [
        (type("S0", (), dict(port_index=0, plane_slot=0, outboard_sign=1,
                             n_cells_outboard=N_REFPLANE))(), acc[0]),
        (type("S1", (), dict(port_index=0, plane_slot=1, outboard_sign=1,
                             n_cells_outboard=2 * N_REFPLANE))(), acc[1]),
    ]}
    ch = plane_channels(raw, dt, f, 0)
    e_zc = np.max(np.abs(ch["zc"] - zc_t))
    e_bt = np.max(np.abs(ch["beta"] / beta_t - 1))
    e_g = np.max(np.abs(ch["gamma_top"] - gamma_true))
    e_o = np.max(np.abs(ch["out_top"] - f0w))
    print(f"[V0] plane math: |dZc| {e_zc:.2e}, |dbeta/beta| {e_bt:.2e}, "
          f"|dGamma_top| {e_g:.2e}, |d out_top| {e_o:.2e}")
    assert max(e_zc, e_bt, e_g, e_o) < 1e-9, "V0 FAILED"

    # V1: exactness of both fits
    fitT = fit_thru(S, f, np.full(len(f), zc_t), beta_t)
    errT = max(abs(fitT["l_nh"] - 0.38) / 0.38,
               abs(fitT["tau_ps"] - 4.0) / 4.0)
    s11_t, t_t = singlepost_model_obs(l_t, tau_t, zc_t,
                                      gamma_true * np.ones(len(f)), f)
    fitS = fit_singlepost(s11_t, t_t, np.full(len(f), zc_t),
                          gamma_true * np.ones(len(f)), f)
    errS = max(abs(fitS["l_nh"] - 0.38) / 0.38,
               abs(fitS["tau_ps"] - 4.0) / 4.0)
    print(f"[V1] thru fit ({fitT['l_nh']:.8f} nH, {fitT['tau_ps']:.8f} ps) "
          f"err {errT:.2e}; single-post fit ({fitS['l_nh']:.8f} nH, "
          f"{fitS['tau_ps']:.8f} ps) err {errS:.2e}")
    assert errT <= 1e-6 and errS <= 1e-6, "V1 FAILED"

    # V2: attempt-2-style l_eff absorption REPRODUCED (a) and RESOLVED (b)
    fit2 = joint_fit(f, S)                     # attempt-2 apparatus verbatim
    dl_eff = fit2["leff_mm"] - 16.0
    l_bias = (fit2["l_nh"] - 0.38) / 0.38
    print(f"[V2a] attempt-2 3-param flat-L fit on segment truth: "
          f"L = {fit2['l_nh']:.4f} nH ({l_bias*100:+.1f}%), "
          f"Zc = {fit2['zc']:.3f}, l_eff = {fit2['leff_mm']:.4f} mm "
          f"(excess {dl_eff:+.3f} mm), resid_max = {fit2['resid_max']:.4f}")
    assert abs(dl_eff) >= 0.5 and l_bias <= -0.10, (
        "V2a FAILED: absorption signature not reproduced")
    print(f"[V2b] attempt-3 pipeline recovers truth: err {errT:.2e} <= 1%")
    assert errT <= 0.01, "V2b FAILED"

    # V3: held-out teeth — dispersive truth absorbed, band F-D1 fires
    l_disp = 0.38e-9 * (1.0 - 0.3 * f / 2.6e9)
    S3 = thru_model_s(l_disp, tau_t, zc_t, beta_t, f)
    fit3 = fit_thru(S3, f, np.full(len(f), zc_t), beta_t)
    absorbed = fit3["resid_max"] <= F_A1_MAX
    print(f"[V3a] 30% dispersive truth: fitted ({fit3['l_nh']:.4f} nH, "
          f"{fit3['tau_ps']:.4f} ps), resid_max {fit3['resid_max']:.4f} "
          f"-> absorbed: {absorbed}")
    assert absorbed, "V3a FAILED: expected out-of-band degeneracy missing"
    beta_b = bfac * 2 * np.pi * BAND_FREQS / C0
    l_disp_b = 0.38e-9 * (1.0 - 0.3 * BAND_FREQS / 2.6e9)
    S3b = thru_model_s(l_disp_b, tau_t, zc_t, beta_b, BAND_FREQS)
    lh, th = fit3["l_nh"] * 1e-9, fit3["tau_ps"] * 1e-12
    S3d = deembed_line_segment(S3b, BAND_FREQS, [(lh / th, th)] * 2, z0=Z0)
    worst3 = max(np.abs(S3d[0, 0]).max(), np.abs(S3d[1, 1]).max())
    b3 = budget_b(lh, th)
    print(f"[V3b] synthetic band arm: worst {worst3:.4f} vs B {b3:.4f} "
          f"-> F-D1 fires: {worst3 >= b3}")
    assert worst3 >= b3, "V3b FAILED: held-out teeth missing"

    # V4: noise pulls + Fisher bar at the 0.005 class
    rng = np.random.default_rng(0)
    noiseS = (rng.normal(scale=1 / np.sqrt(2), size=S.shape)
              + 1j * rng.normal(scale=1 / np.sqrt(2), size=S.shape))
    fit4 = fit_thru(S + 0.005 * noiseS, f, np.full(len(f), zc_t), beta_t)
    pullsT = [abs(fit4["l_nh"] - 0.38) / fit4["sig_l_nh"],
              abs(fit4["tau_ps"] - 4.0) / fit4["sig_tau_ps"]]
    n1 = (rng.normal(scale=1 / np.sqrt(2), size=(2, len(f)))
          + 1j * rng.normal(scale=1 / np.sqrt(2), size=(2, len(f))))
    fit4s = fit_singlepost(s11_t + 0.005 * n1[0], t_t + 0.005 * n1[1],
                           np.full(len(f), zc_t),
                           gamma_true * np.ones(len(f)), f)
    pullsS = [abs(fit4s["l_nh"] - 0.38) / fit4s["sig_l_nh"],
              abs(fit4s["tau_ps"] - 4.0) / fit4s["sig_tau_ps"]]
    print(f"[V4] thru: sigma_L/L {fit4['sig_l_nh']/fit4['l_nh']:.4f}, "
          f"sigma_tau/tau {fit4['sig_tau_ps']/fit4['tau_ps']:.4f}, "
          f"corr {fit4['corr']:.3f}, cond {fit4['cond']:.2f}, pulls "
          f"{[round(float(p), 2) for p in pullsT]}")
    print(f"[V4] single-post: sigma_L/L "
          f"{fit4s['sig_l_nh']/fit4s['l_nh']:.4f}, sigma_tau/tau "
          f"{fit4s['sig_tau_ps']/fit4s['tau_ps']:.4f}, corr "
          f"{fit4s['corr']:.3f}, cond {fit4s['cond']:.2f}, pulls "
          f"{[round(float(p), 2) for p in pullsS]}")
    assert max(pullsT + pullsS) <= 3.0, "V4 FAILED: pulls outside 3 sigma"
    assert (fit4["sig_l_nh"] / fit4["l_nh"] <= F_V1_SIG_L
            and fit4["sig_tau_ps"] / fit4["tau_ps"] <= F_V1_SIG_TAU
            and abs(fit4["corr"]) <= F_V1_CORR
            and fit4["cond"] <= F_V1_COND), "V4 FAILED: F-V1 not passable"
    assert (fit4s["sig_l_nh"] / fit4s["l_nh"] <= F_V2_SIG_L
            and fit4s["sig_tau_ps"] / fit4s["tau_ps"] <= F_V2_SIG_TAU
            and abs(fit4s["corr"]) <= F_V2_CORR
            and fit4s["cond"] <= F_V2_COND), "V4 FAILED: F-V2 not passable"

    # V5: reproduce the design-phase systematic-injection table (the F-C
    # window's provenance) to 10%.
    table = [  # (amp, phase[rad], dL[nH], dtau[ps]) — design run 2026-08-29
        (1.02, 0.0, -0.0039, +0.058),
        (0.98, 0.0, +0.0041, -0.063),
        (1.00, +0.015, -0.0554, -1.547),
        (1.00, -0.015, +0.0554, +1.348),
    ]
    for amp, ph, dl_ref, dt_ref in table:
        fitb = fit_singlepost(s11_t, t_t * amp * np.exp(1j * ph),
                              np.full(len(f), zc_t),
                              gamma_true * np.ones(len(f)), f)
        dl = fitb["l_nh"] - 0.38
        dtau = fitb["tau_ps"] - 4.0
        ok = (abs(dl - dl_ref) <= 0.1 * max(abs(dl_ref), 1e-3)
              and abs(dtau - dt_ref) <= 0.1 * max(abs(dt_ref), 1e-2))
        print(f"[V5] amp {amp} ph {ph:+.3f}: dL {dl:+.4f} nH "
              f"(ref {dl_ref:+.4f}), dtau {dtau:+.3f} ps (ref {dt_ref:+.3f})"
              f" -> {'ok' if ok else 'MISMATCH'}")
        assert ok, "V5 FAILED: F-C derivation not reproduced"
    print("APPARATUS VERIFICATION: ALL PASS (V0, V1, V2, V3, V4, V5)")


# ------------------------------------------------------------- FDTD arms --
def arm_extract() -> None:
    print("=== EXTRACTION ARM (predeclaration sections 3-4) ===")
    f = EXTRACT_FREQS
    pulse = GaussianPulse(**EXTRACT_PULSE)

    # --- I1: in-situ two-plane on the thru (raw drives, refplane opted)
    print(f"--- I1: in-situ two-plane (thru, N = {N_REFPLANE}) ---")
    zc_thru, beta_thru = {}, {}
    for j in (0, 1):
        sim = build_thru(pulse, reference_plane_cells=N_REFPLANE)
        raw, dt = raw_drive(sim, f, EXTRACT_N_STEPS, j, THRU_CODES)
        z_in, _, _ = wholeport_channels(raw, j)
        check_re_positive(z_in, f"thru-insitu drive port{j+1}")   # F-X5
        ch = plane_channels(raw, dt, f, j)
        zc_thru[j], beta_thru[j] = ch["zc"], ch["beta"]
        with np.printoptions(precision=4):
            print(f"  port{j+1} Zc(f) = {ch['zc']}")
            print(f"  port{j+1} beta/(w/c) = "
                  f"{ch['beta'] * C0 / (2 * np.pi * f)}")
    zc_mean = 0.5 * (zc_thru[0].real + zc_thru[1].real)
    beta_mean = 0.5 * (beta_thru[0] + beta_thru[1])

    # --- I2: single-post fixture (one raw drive, refplane opted)
    print("--- I2: single-post 1-port fixture ---")
    sp = build_singlepost(pulse, reference_plane_cells=N_REFPLANE)
    raw_sp, dt_sp = raw_drive(sp, f, EXTRACT_N_STEPS, 0, SINGLEPOST_CODES)
    z_in_sp, s11_sp, a_sp = wholeport_channels(raw_sp, 0)
    check_re_positive(z_in_sp, "single-post port1")               # F-X5
    ch_sp = plane_channels(raw_sp, dt_sp, f, 0)
    # Section-11 apparatus validity PRECONDITION (assert, not a
    # falsifier): the realized fixture must present a measurable,
    # non-resonant load or the harness stops loudly.
    g_max = float(np.max(np.abs(ch_sp["gamma_top"])))
    assert g_max <= 0.5, (
        f"single-post fixture invalid: max|Gamma_top| = {g_max:.4f} > 0.5 "
        "— the line termination is again not measurable-load class")
    # Section-12 corrected observable: a is ALREADY power-normalized —
    # T = (out_top/sqrt(Re Zc)) / a (the declared /sqrt(Z0) double-divide
    # is the recorded apparatus algebra bug).
    t_sp = (ch_sp["out_top"] / np.sqrt(ch_sp["zc"].real)) / a_sp
    with np.printoptions(precision=4):
        print(f"  Z_in = {z_in_sp}")
        print(f"  Zc_sp(f) = {ch_sp['zc']}")
        print(f"  beta_sp/(w/c) = {ch_sp['beta'] * C0 / (2 * np.pi * f)}")
        print(f"  |Gamma_top| = {np.abs(ch_sp['gamma_top'])}")
        print(f"  |T| = {np.abs(t_sp)}, arg T = {np.angle(t_sp)}")

    # --- I3: thru physical S (battery-verbatim default path)
    print("--- I3: thru physical 2x2 S (default battery path) ---")
    S = run_thru(f, EXTRACT_N_STEPS, EXTRACT_PULSE)
    for port in (0, 1):
        z_in = z_in_from_diag(S[port, port])
        check_re_positive(z_in, f"thru port{port+1}")             # F-X5
        with np.printoptions(precision=4):
            print(f"  port{port+1} Z_in = {z_in}")

    # --- fits
    fitT = fit_thru(S, f, zc_mean, beta_mean)
    print(f"  I3 thru fit: L* = {fitT['l_nh']:.4f} +- {fitT['sig_l_nh']:.4f}"
          f" nH, tau* = {fitT['tau_ps']:.4f} +- {fitT['sig_tau_ps']:.4f} ps"
          f" (Zp* = {fitT['l_nh']/fitT['tau_ps']*1e3:.2f} ohm, "
          f"C_p* = {fitT['tau_ps']**2/fitT['l_nh']:.2f} fF)")
    print(f"    corr {fitT['corr']:.4f}, cond {fitT['cond']:.2f}, resid "
          f"max/rms {fitT['resid_max']:.5f}/{fitT['resid_rms']:.5f}, basin "
          f"{fitT['n_same_basin']}/{fitT['n_starts']}")
    fitS = fit_singlepost(s11_sp, t_sp, ch_sp["zc"].real,
                          ch_sp["gamma_top"], f)
    print(f"  I2 single-post fit: L = {fitS['l_nh']:.4f} +- "
          f"{fitS['sig_l_nh']:.4f} nH, tau = {fitS['tau_ps']:.4f} +- "
          f"{fitS['sig_tau_ps']:.4f} ps")
    print(f"    corr {fitS['corr']:.4f}, cond {fitS['cond']:.2f}, resid "
          f"max/rms {fitS['resid_max']:.5f}/{fitS['resid_rms']:.5f}, basin "
          f"{fitS['n_same_basin']}/{fitS['n_starts']}")

    # --- verdicts (windows verbatim from the predeclaration)
    print("--- extraction verdicts ---")
    v = []
    v += instrument_verdicts(
        [zc_thru[0], zc_thru[1], ch_sp["zc"]],
        [beta_thru[0], beta_thru[1], ch_sp["beta"]],
        ["thru p1", "thru p2", "single-post"])
    dzc = float(np.max(np.abs(zc_thru[0].real - zc_thru[1].real)))
    dbt = float(np.max(np.abs(beta_thru[0] / beta_thru[1] - 1)))
    v.append(("F-I2 |Zc_p1 - Zc_p2| [ohm]", round(dzc, 4), F_I2_ZC_OHM,
              dzc <= F_I2_ZC_OHM))
    v.append(("F-I2 |beta ratio - 1|", round(dbt, 5), F_I2_BETA,
              dbt <= F_I2_BETA))
    dzc4 = float(np.max(np.abs(zc_mean - ch_sp["zc"].real)))
    dbt4 = float(np.max(np.abs(beta_mean / ch_sp["beta"] - 1)))
    v.append(("F-I4 |Zc_thru - Zc_sp| [ohm]", round(dzc4, 4), F_I4_ZC_OHM,
              dzc4 <= F_I4_ZC_OHM))
    v.append(("F-I4 |beta ratio - 1|", round(dbt4, 5), F_I4_BETA,
              dbt4 <= F_I4_BETA))
    v += fit_verdicts(fitT, "I3 thru", F_V1_CORR, F_V1_COND, F_V1_SIG_L,
                      F_V1_SIG_TAU, F_A1_MAX, F_A1_RMS)
    v += fit_verdicts(fitS, "I2 single-post", F_V2_CORR, F_V2_COND,
                      F_V2_SIG_L, F_V2_SIG_TAU, F_A2_MAX, F_A2_RMS)
    dl_c = abs(fitT["l_nh"] - fitS["l_nh"])
    dt_c = abs(fitT["tau_ps"] - fitS["tau_ps"])
    v.append(("F-C |L_thru - L_sp| [nH]", round(dl_c, 4), F_C_L_NH,
              dl_c <= F_C_L_NH))
    v.append(("F-C |tau_thru - tau_sp| [ps]", round(dt_c, 4), F_C_TAU_PS,
              dt_c <= F_C_TAU_PS))
    ok = _verdict_table(v)
    lstar = fitT["l_nh"] * 1e-9
    taustar = fitT["tau_ps"] * 1e-12
    print(f"ADOPTED (L*, tau*) = ({fitT['l_nh']:.4f} nH, "
          f"{fitT['tau_ps']:.4f} ps) from I3, per predeclaration")
    print(f"FROZEN B(L*, tau*) = {budget_b(lstar, taustar):.4f}")
    print("EXTRACTION ARM: " + ("ALL PASS" if ok else
                                "STOP (falsifier fired)"))


def arm_band(lstar: float, taustar: float) -> None:
    print("=== BAND ARM (predeclaration section 5) ===")
    b = budget_b(lstar, taustar)
    zp = lstar / taustar
    print(f"L* = {lstar*1e9:.4f} nH, tau* = {taustar*1e12:.4f} ps "
          f"(Zp* = {zp:.2f} ohm), frozen B = {b:.4f}")
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
        ("F-D1 floor < B(L*, tau*)", round(worst, 4), round(b, 4),
         worst < b),
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
    ap.add_argument("--taustar", type=float, default=None)
    args = ap.parse_args()
    if args.verify:
        arm_verify()
    if args.extract:
        arm_extract()
    if args.band:
        assert args.lstar is not None and args.taustar is not None, (
            "--band requires --lstar <henries> --taustar <seconds>")
        arm_band(args.lstar, args.taustar)
