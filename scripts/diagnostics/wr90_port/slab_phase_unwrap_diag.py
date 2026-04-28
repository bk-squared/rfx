"""WR-90 single-slab analytic-Airy phase mismatch — root cause localization.

Goal: separate which of the following is the source of the 143° phase
gate failure on `examples/crossval/11_waveguide_port_wr90.py`'s
single-slab Airy comparison:

  (a) the convention shift `exp(-j·β_v·2d)` applied to the analytic
      reference has the wrong sign;
  (b) the convention shift has wrong magnitude (e.g. wrong d or
      missing factor of 2);
  (c) `analytic_slab_s` itself produces a phase that doesn't match
      Meep/OpenEMS at the slab edge (a closed-form bug);
  (d) rfx's S11 phase at the port plane is wrong (real bug in
      `_shift_modal_waves` / `_extract_global_waves` / β handling).

Strategy:
  1. Run rfx slab; record S11(f) complex.
  2. Compute the analytic Airy reference at the slab edges (no shift)
     and at the rfx port plane (with the shift).
  3. Compute phase(rfx) - phase(airy_with_shift) per frequency. If
     the residual is small and frequency-INDEPENDENT, the shift is
     correct and the gate failure is convention-only (constant
     offset). If frequency-DEPENDENT, the shift sign/magnitude or β
     handling is wrong.
  4. Compute phase(rfx) - phase(airy_edge) per frequency. The
     residual should equal -2·β_v·d if the shift is correct. Check
     that empirically.

This script does NOT try to fix the bug; it reports the diagnostic
table so the next session can decide between hypotheses (a)-(d).
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
OUT = Path(__file__).parent / "out_slab_phase_diag"
OUT.mkdir(parents=True, exist_ok=True)

C0 = 2.998e8


def _load_cv11():
    spec = importlib.util.spec_from_file_location(
        "cv11", REPO / "examples" / "crossval" / "11_waveguide_port_wr90.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    cv = _load_cv11()
    eps_r = 2.0
    slab_L = 0.010

    print(f"=== rfx slab vs analytic Airy at eps_r={eps_r}, L={slab_L*1000:.0f} mm ===\n")

    f_hz, s11_rfx, s21_rfx = cv.run_rfx_slab(eps_r, slab_L)
    s11_edge, s21_edge = cv.analytic_slab_s(f_hz, eps_r, slab_L)

    omega = 2.0 * np.pi * f_hz
    kc = 2.0 * np.pi * cv.F_CUTOFF_TE10 / C0
    beta_v = np.sqrt(np.maximum((omega / C0) ** 2 - kc ** 2, 0.0))
    slab_center = 0.5 * (cv.PORT_LEFT_X + cv.PORT_RIGHT_X)
    d_left = slab_center - 0.5 * slab_L - 0.050   # 45 mm
    print(f"d_left = {d_left*1000:.3f} mm  |  beta_v range = "
          f"{beta_v.min():.2f} ... {beta_v.max():.2f} rad/m\n")

    # As-is in crossval/11 today
    s11_shifted_neg = s11_edge * np.exp(-1j * beta_v * 2.0 * d_left)
    # Hypothesis (a): opposite sign
    s11_shifted_pos = s11_edge * np.exp(+1j * beta_v * 2.0 * d_left)
    # Hypothesis (b): half magnitude (single-trip not round-trip)
    s11_shifted_half = s11_edge * np.exp(-1j * beta_v * 1.0 * d_left)

    def deg_diff(a, b):
        d = np.angle(a / np.where(np.abs(b) > 1e-30, b, 1.0))
        return np.degrees(np.unwrap(d))

    print("Per-freq phase comparison (deg):\n")
    print(f"{'f[GHz]':>7s} | {'∠S_rfx':>8s} | {'∠S_edge':>8s} | "
          f"{'rfx-edge':>9s} | {'2β_v·d':>9s} | "
          f"{'shift=−':>8s} | {'shift=+':>8s} | {'shift=−/2':>9s}")
    print("-" * 100)
    for i, f in enumerate(f_hz):
        rfx_deg = np.degrees(np.angle(s11_rfx[i]))
        edge_deg = np.degrees(np.angle(s11_edge[i]))
        rfx_minus_edge = ((rfx_deg - edge_deg + 180) % 360) - 180
        beta2d_deg = np.degrees(2.0 * beta_v[i] * d_left)
        beta2d_mod = ((beta2d_deg + 180) % 360) - 180
        d_neg = ((np.degrees(np.angle(s11_rfx[i] / s11_shifted_neg[i])) + 180) % 360) - 180
        d_pos = ((np.degrees(np.angle(s11_rfx[i] / s11_shifted_pos[i])) + 180) % 360) - 180
        d_half = ((np.degrees(np.angle(s11_rfx[i] / s11_shifted_half[i])) + 180) % 360) - 180
        print(f"{f/1e9:7.2f} | {rfx_deg:+8.2f} | {edge_deg:+8.2f} | "
              f"{rfx_minus_edge:+9.2f} | {beta2d_mod:+9.2f} | "
              f"{d_neg:+8.2f} | {d_pos:+8.2f} | {d_half:+9.2f}")

    # Verdict: which shift makes (rfx - shifted_edge) closest to 0?
    err_neg = np.std([
        ((np.degrees(np.angle(s11_rfx[i] / s11_shifted_neg[i])) + 180) % 360) - 180
        for i in range(len(f_hz))
    ])
    err_pos = np.std([
        ((np.degrees(np.angle(s11_rfx[i] / s11_shifted_pos[i])) + 180) % 360) - 180
        for i in range(len(f_hz))
    ])
    err_half = np.std([
        ((np.degrees(np.angle(s11_rfx[i] / s11_shifted_half[i])) + 180) % 360) - 180
        for i in range(len(f_hz))
    ])
    print(f"\nphase residual std (deg):  shift=−2βd: {err_neg:.2f}  "
          f"shift=+2βd: {err_pos:.2f}  shift=−βd: {err_half:.2f}")

    # Also compare ∠S_rfx − ∠S_edge with -2β_v·d to test whether the 'rfx
    # vs edge' diff is exactly the propagation through 2d of empty guide.
    rfx_minus_edge_arr = np.array([
        ((np.degrees(np.angle(s11_rfx[i] / s11_edge[i])) + 180) % 360) - 180
        for i in range(len(f_hz))
    ])
    expected_minus_2bd = np.array([
        ((-np.degrees(2.0 * beta_v[i] * d_left) + 180) % 360) - 180
        for i in range(len(f_hz))
    ])
    expected_plus_2bd = np.array([
        ((+np.degrees(2.0 * beta_v[i] * d_left) + 180) % 360) - 180
        for i in range(len(f_hz))
    ])
    diff_minus = np.std((rfx_minus_edge_arr - expected_minus_2bd + 180) % 360 - 180)
    diff_plus = np.std((rfx_minus_edge_arr - expected_plus_2bd + 180) % 360 - 180)
    print(f"std(∠(rfx/edge) − (−2βd)) = {diff_minus:.2f}°  "
          f"std(∠(rfx/edge) − (+2βd)) = {diff_plus:.2f}°")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    f_ghz = f_hz / 1e9
    ax = axes[0]
    ax.plot(f_ghz, np.degrees(np.angle(s11_rfx)), "o-", label="rfx (port plane)")
    ax.plot(f_ghz, np.degrees(np.angle(s11_edge)), "s-", label="analytic edge")
    ax.plot(f_ghz, np.degrees(np.angle(s11_shifted_neg)),
            "^--", label="edge · exp(−j·2βd)  (current crossval)")
    ax.plot(f_ghz, np.degrees(np.angle(s11_shifted_pos)),
            "v--", label="edge · exp(+j·2βd)")
    ax.set_ylabel("∠S11 [deg]")
    ax.set_title("WR-90 slab S11 phase: rfx vs analytic Airy under three shift conventions")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.axhline(0.0, color="k", lw=1, ls="--")
    diffs_neg = np.array([
        ((np.degrees(np.angle(s11_rfx[i] / s11_shifted_neg[i])) + 180) % 360) - 180
        for i in range(len(f_hz))
    ])
    diffs_pos = np.array([
        ((np.degrees(np.angle(s11_rfx[i] / s11_shifted_pos[i])) + 180) % 360) - 180
        for i in range(len(f_hz))
    ])
    ax.plot(f_ghz, diffs_neg, "^-", label="rfx − edge·exp(−j·2βd)")
    ax.plot(f_ghz, diffs_pos, "v-", label="rfx − edge·exp(+j·2βd)")
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel("phase residual [deg]")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_png = OUT / "slab_phase_unwrap_diag.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\n[plot] {out_png}")

    out_json = OUT / "slab_phase_unwrap_diag.json"
    out_json.write_text(json.dumps({
        "freqs_hz": f_hz.tolist(),
        "s11_rfx_real": s11_rfx.real.tolist(),
        "s11_rfx_imag": s11_rfx.imag.tolist(),
        "s11_edge_real": s11_edge.real.tolist(),
        "s11_edge_imag": s11_edge.imag.tolist(),
        "beta_v": beta_v.tolist(),
        "d_left_m": d_left,
        "phase_residual_std_deg": {
            "shift_minus_2bd": err_neg,
            "shift_plus_2bd": err_pos,
            "shift_minus_bd": err_half,
        },
    }, indent=2))
    print(f"[json] {out_json}")


if __name__ == "__main__":
    main()
