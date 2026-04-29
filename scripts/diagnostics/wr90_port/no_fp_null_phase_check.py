"""WR-90 phase-tracking validation on geometries WITHOUT Fabry-Perot nulls.

The 2026-04-29 slab-phase diagnostic (`slab_phase_unwrap_diag.py`)
showed that the 143° gate failure on the single-slab Airy comparison
is dominated by FP nulls (|S11|<0.1 frequencies in the 10-11 GHz
region). To test whether rfx phase tracking is genuinely reliable
where |S| is well-defined, this script compares rfx S-parameter
phases against analytic closed-form references on two geometries
that have NO FP nulls in the band:

  1. **Empty WR-90 guide**, S21 phase. Reference = exp(−j·β_v·L)
     where L = port-to-port distance (100 mm) and β_v(f) is the
     analytic empty-guide TE10 propagation constant. Tests pure
     forward propagation through `_shift_modal_waves`.

  2. **PEC-short termination**, S11 phase. Reference = exp(−j·2·β_v·d)
     where d = distance from port reference plane to the PEC short
     face. |S11|≈1 across the entire band, so phase is well-defined
     at every frequency. Tests round-trip phase shift through both
     wave decomposition AND `_shift_modal_waves`.

If both pass to within a few degrees → rfx phase tracking is
reliably accurate, and the slab-phase 143° gate failure is purely
gate-definition (FP nulls) and per-solver convention asymmetry,
NOT an rfx bug.

If either fails by tens of degrees → rfx has a real systematic
phase bias. The size of the bias localizes the source: if the
empty-guide one-way phase is off, `_shift_modal_waves` has a small
β-vs-shift mismatch; if only the PEC-short round-trip is off, wave
decomposition or _co_located_current_spectrum sign needs review.

Outputs: per-frequency phase residual table + overlay plot + JSON.
"""
from __future__ import annotations
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "out_no_fp_null_phase"
OUT.mkdir(parents=True, exist_ok=True)

C0 = 2.998e8


def _load_cv11():
    spec = importlib.util.spec_from_file_location(
        "cv11", REPO / "examples" / "crossval" / "11_waveguide_port_wr90.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _wrap(deg):
    return ((deg + 180.0) % 360.0) - 180.0


def main():
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    cv = _load_cv11()

    omega_factor = 2.0 * np.pi
    kc = omega_factor * cv.F_CUTOFF_TE10 / C0

    # ---- Run 1: empty guide ---------------------------------------------
    print("=== rfx empty WR-90 guide: S21 phase vs analytic propagation ===")
    f_hz, s11_e, s21_e = cv.run_rfx_empty()

    # rfx reports at port reference planes 50 and 150 mm: 100 mm of empty
    # propagation between them.
    L_empty = 0.150 - 0.050   # 100 mm

    omega = omega_factor * f_hz
    beta_v = np.sqrt(np.maximum((omega / C0) ** 2 - kc ** 2, 0.0))
    s21_ref = np.exp(-1j * beta_v * L_empty)

    print(f"L_empty = {L_empty*1000:.3f} mm  |  beta_v range "
          f"{beta_v.min():.2f} ... {beta_v.max():.2f} rad/m")
    print(f"\n{'f[GHz]':>7s} | {'|S21_rfx|':>9s} | {'∠S21_rfx':>9s} | "
          f"{'∠S21_ref':>9s} | {'residual':>9s}")
    print("-" * 62)
    res_s21 = []
    for i, f in enumerate(f_hz):
        rfx_deg = np.degrees(np.angle(s21_e[i]))
        ref_deg = np.degrees(np.angle(s21_ref[i]))
        resid = _wrap(rfx_deg - ref_deg)
        res_s21.append(resid)
        print(f"{f/1e9:7.2f} | {abs(s21_e[i]):9.4f} | "
              f"{rfx_deg:+9.2f} | {ref_deg:+9.2f} | {resid:+9.2f}")
    res_s21 = np.array(res_s21)
    print(f"\nempty guide S21 phase residual: "
          f"mean {np.mean(res_s21):+.2f}°  std {np.std(res_s21):.2f}°  "
          f"max|·| {np.max(np.abs(res_s21)):.2f}°")

    # ---- Run 2: PEC short -----------------------------------------------
    print("\n=== rfx PEC-short: S11 phase vs ideal round-trip ===")
    f_hz_p, s11_p, _ = cv.run_rfx_pec_short()

    # crossval/11 places the PEC short face at PEC_SHORT_X = 145 mm.
    # rfx port 1 reference plane is at 50 mm. Round-trip empty distance
    # = 2·(145 - 50) = 190 mm of empty guide.
    d_pec = cv.PEC_SHORT_X - 0.050   # 95 mm
    s11_ref_pec = -np.exp(-1j * beta_v * 2.0 * d_pec)
    # Sign: at a PEC short the boundary condition is E_t = 0, which gives
    # a reflection coefficient of -1 at the face itself. Round-trip
    # propagation phase added on top of that.

    print(f"d_pec = {d_pec*1000:.3f} mm  |  round-trip 2d = {2*d_pec*1000:.3f} mm")
    print(f"\n{'f[GHz]':>7s} | {'|S11_rfx|':>9s} | {'∠S11_rfx':>9s} | "
          f"{'∠S11_ref':>9s} | {'residual':>9s}")
    print("-" * 62)
    res_s11 = []
    for i, f in enumerate(f_hz_p):
        rfx_deg = np.degrees(np.angle(s11_p[i]))
        ref_deg = np.degrees(np.angle(s11_ref_pec[i]))
        resid = _wrap(rfx_deg - ref_deg)
        res_s11.append(resid)
        print(f"{f/1e9:7.2f} | {abs(s11_p[i]):9.4f} | "
              f"{rfx_deg:+9.2f} | {ref_deg:+9.2f} | {resid:+9.2f}")
    res_s11 = np.array(res_s11)
    print(f"\npec-short S11 phase residual: "
          f"mean {np.mean(res_s11):+.2f}°  std {np.std(res_s11):.2f}°  "
          f"max|·| {np.max(np.abs(res_s11)):.2f}°")

    # ---- Verdicts -------------------------------------------------------
    print("\n=== verdict ===")
    gate = 5.0
    s21_ok = np.max(np.abs(res_s21)) < gate
    s11_ok = np.max(np.abs(res_s11)) < gate
    if s21_ok and s11_ok:
        verdict = (f"rfx phase tracking is accurate to <{gate}° on both "
                   "no-FP-null geometries. The slab-phase 143° gate failure is "
                   "purely gate-definition + FP-null phase noise, not an rfx bug.")
    elif np.max(np.abs(res_s21)) < gate and np.max(np.abs(res_s11)) > gate:
        verdict = ("Forward propagation is clean but round-trip is biased — "
                   "wave decomposition / _co_located_current_spectrum sign-side "
                   "review needed.")
    elif np.max(np.abs(res_s11)) < gate and np.max(np.abs(res_s21)) > gate:
        verdict = ("Round-trip is clean but one-way is biased — _shift_modal_waves "
                   "needs review.")
    else:
        bias21 = np.mean(res_s21)
        bias11 = np.mean(res_s11)
        verdict = (f"Both geometries show systematic bias — empty mean "
                   f"{bias21:+.1f}°, pec-short mean {bias11:+.1f}°. The slab "
                   "residual ~15° is consistent with this and the bias should "
                   "be located before any cross-tool work.")
    print(verdict)

    # ---- Plot -----------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    f_ghz = f_hz / 1e9
    ax = axes[0]
    ax.axhline(0.0, color="k", lw=1, ls="--")
    ax.axhspan(-gate, +gate, color="green", alpha=0.10,
               label=f"±{gate}° gate")
    ax.plot(f_ghz, res_s21, "o-", color="tab:orange",
            label=f"empty guide S21 (max|·| {np.max(np.abs(res_s21)):.2f}°)")
    ax.set_ylabel("∠S21_rfx − ∠S21_analytic [deg]")
    ax.set_title("rfx phase tracking on no-FP-null geometries")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.axhline(0.0, color="k", lw=1, ls="--")
    ax.axhspan(-gate, +gate, color="green", alpha=0.10,
               label=f"±{gate}° gate")
    ax.plot(f_ghz, res_s11, "o-", color="tab:blue",
            label=f"PEC-short S11 (max|·| {np.max(np.abs(res_s11)):.2f}°)")
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel("∠S11_rfx − ∠S11_analytic [deg]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_png = OUT / "no_fp_null_phase_check.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\n[plot] {out_png}")

    out_json = OUT / "no_fp_null_phase_check.json"
    out_json.write_text(json.dumps({
        "freqs_hz": f_hz.tolist(),
        "L_empty_m": L_empty,
        "d_pec_m": d_pec,
        "beta_v": beta_v.tolist(),
        "empty_S21_residual_deg": res_s21.tolist(),
        "pec_short_S11_residual_deg": res_s11.tolist(),
        "verdict": verdict,
    }, indent=2))
    print(f"[json] {out_json}")


if __name__ == "__main__":
    main()
