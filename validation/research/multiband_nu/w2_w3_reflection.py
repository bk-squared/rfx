"""W2 (per-transition reflection, F-S2) and W3 (symmetric traversal
amplitude, F-S3) via 2-run differencing + geometric time gating.

Method pre-declared in docs/design_notes/20260829_spec01_multiband_predeclaration.md §3.

Usage:
    PYTHONPATH=. python -m validation.research.multiband_nu.w2_w3_reflection
"""

from __future__ import annotations

import json
import time

import numpy as np
import jax.numpy as jnp

from rfx.nonuniform import run_nonuniform

from . import fixtures as fx
from .chain_model import bloch_kz, power_rt
from .harness import build_pec_fixture

F0 = 10e9
SIGMA_T = 64e-12
T0 = 5 * SIGMA_T
R_VALUES = (1.1, 1.2, 1.4, 1.5, 2.0)
FS2_FLOOR = 3e-5
FS3_FLOOR = 3e-4


def gaussian_sine(n_steps: int, dt: float, sigma_t: float = SIGMA_T,
                  t0: float | None = None) -> np.ndarray:
    t = np.arange(n_steps) * dt
    t0 = 5 * sigma_t if t0 is None else t0
    return (np.exp(-((t - t0) / sigma_t) ** 2 / 2.0)
            * np.sin(2 * np.pi * F0 * (t - t0))).astype(np.float32)


def te10_sources(grid, k_src: int, waveform: np.ndarray):
    """Soft Ex sources over the transverse plane with the exact discrete
    TE10 sine profile (x-invariant: all physical i)."""
    ny = grid.ny
    dy = float(grid.dy)
    b = (ny - 1) * dy
    srcs = []
    for i in range(grid.nx - 1):
        for j in range(1, ny - 1):
            amp = float(np.sin(np.pi * (j * dy) / b))
            srcs.append((i, j, k_src, "ex", waveform * np.float32(amp)))
    return srcs


def dft_at(x: np.ndarray, dt: float, f: float, n0: int, n1: int) -> complex:
    n = np.arange(n0, n1)
    return complex(np.sum(x[n0:n1] * np.exp(-2j * np.pi * f * n * dt)))


def vg_of(dcell: float, dt: float, dy: float, b: float) -> float:
    df = F0 * 1e-6
    kp = bloch_kz(F0 + df, dt, dy, b, dcell)
    km = bloch_kz(F0 - df, dt, dy, b, dcell)
    return 2 * np.pi * 2 * df / (kp - km)


def run_probe(profile: np.ndarray, k_prb: int, n_steps: int, k_src: int,
              sigma_t: float = SIGMA_T, t0: float | None = None,
              b_y: float = fx.B_Y):
    grid, mats = build_pec_fixture(profile, (fx.A_X, b_y), fx.DXY)
    wf = gaussian_sine(n_steps, float(grid.dt), sigma_t, t0)
    srcs = te10_sources(grid, k_src, wf)
    out = run_nonuniform(grid, mats, n_steps, sources=srcs,
                         probes=[(1, grid.ny // 2, k_prb, "ex")])
    return grid, np.asarray(out["time_series"][:, 0], dtype=np.float64)


def group_delay(profile: np.ndarray, k_from: int, k_to: int,
                dt: float, dy: float, b: float) -> float:
    """Sum of per-cell group delays between two node indices."""
    seg = profile[k_from:k_to]
    sizes = {}
    tot = 0.0
    for d in seg:
        key = round(float(d), 15)
        if key not in sizes:
            sizes[key] = vg_of(float(d), dt, dy, b)
        tot += float(d) / sizes[key]
    return tot


def w2_arm(r: float, variant: str, trace_b: np.ndarray, grid_b) -> dict:
    prof_a = fx.single_transition_profile(r, variant)
    n_steps = len(trace_b)
    grid_a, trace_a = run_probe(prof_a, fx.K_PRB, n_steps, fx.K_SRC)
    dt = float(grid_a.dt)
    assert abs(dt - float(grid_b.dt)) < 1e-20
    dy, b = float(grid_a.dy), fx.B_Y
    vg_f = vg_of(fx.DZ_FINE, dt, dy, b)

    z_src = fx.K_SRC * fx.DZ_FINE
    z_prb = fx.K_PRB * fx.DZ_FINE
    # transition start = end of the fine runway (ramp counts as transition)
    z_tr = fx.N_FINE_RUNWAY * fx.DZ_FINE
    t_r = T0 + (2 * z_tr - z_src - z_prb) / vg_f
    t_s = T0 + (z_src + 2 * z_tr - z_prb) / vg_f
    # far-wall return in A (through the coarse tail and back)
    l_coarse = float(prof_a.sum()) - z_tr
    vg_c = vg_of(float(prof_a[-1]), dt, dy, b)
    t_f = T0 + (z_tr - z_src) / vg_f + 2 * l_coarse / vg_c + (z_tr - z_prb) / vg_f
    gate_end = min(t_s, t_f) - 4 * SIGMA_T
    assert t_r + 4 * SIGMA_T < gate_end, (t_r, t_s, t_f)

    diff = trace_a - trace_b
    n_gate = int(gate_end / dt)
    refl = dft_at(diff, dt, F0, 0, n_gate)
    t_echo_prb = T0 + (z_src + z_prb) / vg_f     # source wall echo at probe
    t_inc_end = min(T0 + (z_prb - z_src) / vg_f + 8 * SIGMA_T,
                    t_echo_prb - 4 * SIGMA_T)
    inc = dft_at(trace_b, dt, F0, 0, int(t_inc_end / dt))
    r_meas = abs(refl) / abs(inc)
    r_model, t_model, _ = power_rt(prof_a, 20, 20, F0, dt, dy, b)
    window = max(3.0 * r_model, FS2_FLOOR)
    return {
        "r": r, "variant": variant,
        "R_meas": r_meas, "R_meas_db": 20 * np.log10(max(r_meas, 1e-300)),
        "R_model": float(r_model), "window": float(window),
        "fs2_fired": bool(r_meas > window) if r <= 1.4 else None,
        "in_envelope_claim": r <= 1.4,
        "gates_ns": {"t_r": t_r * 1e9, "t_s": t_s * 1e9, "t_f": t_f * 1e9,
                     "gate_end": gate_end * 1e9},
    }


# --- W3 instrumentation (corrected 2026-08-29, note correction C1) ------
# The transition structure under test is unchanged (coarse 30 | fine 40 |
# coarse 30 with abrupt/smooth transitions); the RUNWAYS, source position,
# source bandwidth and analysis window are instrumentation, re-sized so the
# gating requirement declared in note 3.1 ("closing before ... any other
# arrival") is actually satisfiable:
#   lead/tail 240 fine cells, source k=200 (echo lag 2*200mm/vg = 1.54 ns),
#   out-probe 140 cells before the far wall (return lag 1.08 ns),
#   sigma_t = 100 ps (5-GHz-cutoff stragglers 3.2 sigma out of band),
#   Gaussian analysis window sigma_w = 200 ps centred on the group-delay
#   arrival (spectral leakage from |df|>=3 GHz suppressed by e^-7).
W3_LEAD = 240
W3_KSRC = 200
W3_TAIL_GUARD = 200
W3_SIGMA_T = 100e-12
W3_T0 = 5 * W3_SIGMA_T
W3_SIGMA_W = 200e-12
# Correction C2 (2026-08-29): the W3 transverse instrument moves the TE10
# cutoff far below band (b: 30mm -> 90mm, fc 5 -> 1.67 GHz). At fc/f0=0.5
# the waveguide GVD (~0.13 ps/mm/GHz) biased the windowed amplitude by
# ~3.5e-3 for the A/B distance mismatch — the null control caught it.
# At fc/f0=1/6 the same mismatch sits below the 3e-4 floor; additionally
# the B reference length is group-delay-matched to the arm family mean.
W3_BY = 90e-3
W3_B_LEN = 624            # B: k_src=200, out at 424 (224 fine cells), 200 guard


def w3_profile(r: float, variant: str) -> np.ndarray:
    if r == 1.0:  # null-control arm: same cell count, no transitions
        n = W3_LEAD + fx.N_COARSE + fx.N_FINE + fx.N_COARSE + W3_LEAD
        return np.full(n, fx.DZ_FINE, dtype=np.float64)
    c = [r * fx.DZ_FINE] * fx.N_COARSE
    up = fx._ramp(fx.DZ_FINE, r * fx.DZ_FINE) if variant == "smooth" else []
    dn = fx._ramp(r * fx.DZ_FINE, fx.DZ_FINE) if variant == "smooth" else []
    lead = [fx.DZ_FINE] * W3_LEAD
    mid = [fx.DZ_FINE] * fx.N_FINE
    return np.asarray(lead + up + c + dn + mid + up + c + dn + lead,
                      np.float64)


def windowed_dft_at(x: np.ndarray, dt: float, f: float, t_center: float,
                    sigma_w: float) -> complex:
    n = np.arange(len(x))
    t = n * dt
    win = np.exp(-((t - t_center) / sigma_w) ** 2 / 2.0)
    return complex(np.sum(x * win * np.exp(-2j * np.pi * f * t)))


def main():
    t_start = time.time()
    results = {"w2": [], "w3": []}

    # ---- W2 ----
    n_steps_w2 = 800
    prof_b2 = fx.uniform_reference_profile()
    grid_b2, trace_b2 = run_probe(prof_b2, fx.K_PRB, n_steps_w2, fx.K_SRC)
    for r in R_VALUES:
        for variant in ("abrupt", "smooth"):
            if variant == "smooth" and r <= fx.SMOOTH_STEP:
                continue  # ramp empty -> identical to abrupt
            res = w2_arm(r, variant, trace_b2, grid_b2)
            print(f"W2 r={r} {variant}: R_meas={res['R_meas']:.3e} "
                  f"({res['R_meas_db']:.1f} dB) model={res['R_model']:.3e} "
                  f"window={res['window']:.3e} fired={res['fs2_fired']}",
                  flush=True)
            results["w2"].append(res)

    # ---- W3 (corrected instrumentation, note C1 + C2) ----
    n_steps_w3 = 2000
    prof_b3 = np.full(W3_B_LEN, fx.DZ_FINE)
    k_out_b = W3_B_LEN - W3_TAIL_GUARD
    grid_b3, trace_b3 = run_probe(prof_b3, k_out_b, n_steps_w3, W3_KSRC,
                                  W3_SIGMA_T, W3_T0, W3_BY)
    dt = float(grid_b3.dt)
    dy, b = float(grid_b3.dy), W3_BY
    vg_f = vg_of(fx.DZ_FINE, dt, dy, b)
    t_pass_b = W3_T0 + (k_out_b - W3_KSRC) * fx.DZ_FINE / vg_f
    # window-support safety: echo lag and wall-return lag vs 3.5 sigma_w
    echo_lag = 2 * W3_KSRC * fx.DZ_FINE / vg_f
    wall_lag = 2 * W3_TAIL_GUARD * fx.DZ_FINE / vg_f
    assert echo_lag > 5 * W3_SIGMA_W and wall_lag > 5 * W3_SIGMA_W, \
        (echo_lag, wall_lag)
    out_b = windowed_dft_at(trace_b3, dt, F0, t_pass_b, W3_SIGMA_W)

    for r in (1.0,) + R_VALUES:
        for variant in ("abrupt", "smooth"):
            if r == 1.0 and variant == "smooth":
                continue  # null control has no transitions
            if variant == "smooth" and r <= fx.SMOOTH_STEP:
                continue
            prof = w3_profile(r, variant)
            k_out = len(prof) - W3_TAIL_GUARD
            grid_a, trace_a = run_probe(prof, k_out, n_steps_w3, W3_KSRC,
                                        W3_SIGMA_T, W3_T0, W3_BY)
            t_pass = W3_T0 + group_delay(prof, W3_KSRC, k_out, dt, dy, b)
            assert (t_pass + 5 * W3_SIGMA_W) < n_steps_w3 * dt
            out_a = windowed_dft_at(trace_a, dt, F0, t_pass, W3_SIGMA_W)
            t_meas = abs(out_a) / abs(out_b)
            if r == 1.0:
                t_model = 1.0
            else:
                _, t_model, _ = power_rt(prof, 40, 40, F0, dt, dy, b)
            halfwidth = max(FS3_FLOOR, 0.5 * abs(1 - t_model))
            dev = abs(t_meas - abs(t_model))
            res = {
                "r": r, "variant": variant if r != 1.0 else "null-control",
                "T_meas": t_meas, "T_model": float(abs(t_model)),
                "deviation": dev, "window_halfwidth": float(halfwidth),
                "fs3_fired": bool(dev > halfwidth) if r <= 1.4 else None,
                "in_envelope_claim": r <= 1.4,
                "gates_ns": {"t_pass": t_pass * 1e9,
                             "sigma_w_ps": W3_SIGMA_W * 1e12},
            }
            print(f"W3 r={r} {res['variant']}: T_meas={t_meas:.6f} "
                  f"T_model={abs(t_model):.6f} dev={dev:.2e} "
                  f"win={halfwidth:.2e} fired={res['fs3_fired']}", flush=True)
            results["w3"].append(res)

    results["wallclock_s"] = time.time() - t_start
    out = "validation/research/multiband_nu/results/w2_w3.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
