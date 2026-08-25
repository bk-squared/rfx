"""Phase-1 — free-form metal topology optimization A/B/C on the notch geometry.

Probe #1 (NOTE_probe01_verdict.md) measured WHY density-based metal TO
converges to weak optima: on the production Kottke path the gray stub is a
lossless high-eps resonator that sweeps a spurious dip/barrier through the
band during continuation; on the legacy damping path the gray stub is nearly
invisible for 90% of the traversal. This experiment tests the redirected
remedy — conductivity-aware gray ("RAMP-damped gray"): intermediate density
gets a finite RAMP-scheduled conductivity so the forming resonator is damped,
while rho -> 1 still folds to true PEC via Kottke.

Arms (same geometry, objective, init, budget):
  A  kottke-linear   occ = rho_projected, RFX_PEC_OCC_KOTTKE=1   (current best)
  B  kottke+ramp     A + sigma_override = SIGMA0 * rho/(1+QRAMP*(1-rho)) in the
                     design region (concave RAMP; damps gray, irrelevant at PEC)
  C  legacy-linear   occ = rho_projected, RFX_PEC_OCC_KOTTKE=0   (known-bad ref)

Problem: free-form per-cell metal density on the trace layer over a
3 mm x 12 mm region sprouting from the through-line (the lambda/4 stub of the
validated notch example is one realizable design inside this region, so a
strong notch at f_t = 6 GHz is known to be reachable). Objective: minimize
J = |S21(f_t)|^2. Pipeline: latent theta -> sigmoid -> cone filter (r=2 cells)
-> Heaviside projection with beta-continuation (8 -> 16 -> 32) -> occupancy.

Gates:
  1. AD-vs-FD gradient check at init (arms A and B), 2 probe cells.
  2. Sanity: arm B's sigma_override must actually change the objective
     (fail loud if silently ignored).
  3. Final verdict metric: BINARIZED (rho > 0.5) hard occupancy re-evaluated
     through the Kottke PEC limit with a 3x longer window over the 9-freq
     band — identical evaluator for all arms. Success (paper-worthy signal):
     J_hard(B) clearly better than J_hard(A) at equal budget.

Run (GPU):
  python research/metal_to/phase1_freeform_ab.py --arm A
  python research/metal_to/phase1_freeform_ab.py --arm B
  python research/metal_to/phase1_freeform_ab.py --arm C
SMOKE=1 shrinks iters/periods for a CPU API check.
Outputs: research/metal_to/out/phase1_<arm>.{json,png,npz}.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "validation" / "tmtt_paper"))

import msl_stub_notch_tuning as notch  # noqa: E402
from rfx.probes.msl_wave_decomp import (  # noqa: E402
    _i_from_plane,
    _v_from_plane,
    extract_msl_nprobe,
)
from rfx.topology import apply_density_filter, apply_projection  # noqa: E402

C0 = 2.998e8
OUT = Path(os.environ.get("OUTPUT_DIR", Path(__file__).resolve().parent / "out"))
OUT.mkdir(parents=True, exist_ok=True)

SMOKE = os.environ.get("SMOKE", "0") == "1"
F_TARGET = notch.F_TARGET
FREQS_BAND = np.linspace(4.5e9, 8.5e9, 9)

# phase1b: optimize notch SELECTIVITY, not raw |S21(f_t)| — phase1a's arm C
# exploited the single-frequency objective with a broadband metal brick.
# Passband transmissions are normalized per-frequency by the EMPTY-LINE
# reference so the uncalibrated extractor scale cancels.
TAG = "phase1c"
FREQS_OPT = np.array([4.5e9, 5.25e9, 6.0e9, 6.75e9, 7.5e9])
IDX_NOTCH = 2
W_PB = float(os.environ.get("W_PB", 1.0))

# Design region: sprouts from the trace top edge, spans the stub footprint
# generously (the analytic 7.37 mm stub fits well inside 12 mm).
REGION_WX = 3.0e-3
REGION_LY = 12.0e-3
FILTER_R_CELLS = 2.5
BETA_STAGES = ((0, 8.0), (20, 32.0), (40, 128.0))
N_ITERS = 4 if SMOKE else 60
LR = 0.2
NUM_PERIODS = 4.0 if SMOKE else 10.0
NUM_PERIODS_EVAL = 6.0 if SMOKE else 30.0
SEED = 0


def build_setup(freqs_hz: np.ndarray):
    freqs = jnp.asarray(freqs_hz, dtype=jnp.float32)
    sim, y_trace, trace_y_hi, d_set, p_set = notch.build_sim(freqs)
    sim.add_dft_plane_probe(
        axis="z",
        coordinate=notch.H_SUB + 0.5 * notch.DX,
        component="ez",
        freqs=jnp.asarray([F_TARGET], dtype=jnp.float32),
        name="ez_surface",
    )
    grid = sim._build_grid()
    sim.preflight()
    return sim, grid, trace_y_hi, d_set, p_set


def region_indices(grid, trace_y_hi: float):
    nx, ny, nz = grid.shape
    pad_x, pad_y, pad_z = grid.axis_pads
    xc = (np.arange(nx) - pad_x + 0.5) * notch.DX
    yc = (np.arange(ny) - pad_y + 0.5) * notch.DX
    zc = (np.arange(nz) - pad_z + 0.5) * notch.DX
    x_mid = notch.LX / 2.0
    ix = np.where((xc >= x_mid - REGION_WX / 2) & (xc <= x_mid + REGION_WX / 2))[0]
    iy = np.where((yc >= trace_y_hi) & (yc <= trace_y_hi + REGION_LY))[0]
    z_patch = notch.H_SUB + 0.5 * notch.DX
    iz = int(np.argmin(np.abs(zc - z_patch)))
    return ix, iy, iz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("A", "B", "C"), required=True)
    ap.add_argument("--sigma0", type=float, default=5.0)
    ap.add_argument("--qramp", type=float, default=3.0)
    ap.add_argument("--iters", type=int, default=N_ITERS)
    ap.add_argument("--init", choices=("uniform", "low", "stub"), default="uniform",
                    help="theta init: uniform rho=0.5 | low rho=0.25 | analytic-stub seed")
    args = ap.parse_args()
    arm = args.arm
    n_iters = args.iters
    beta_stages = ((0, 8.0), (n_iters // 3, 32.0), (2 * n_iters // 3, 128.0))
    global TAG
    TAG = f"{TAG}_{args.init}_i{n_iters}"

    os.environ["RFX_PEC_OCC_KOTTKE"] = "0" if arm == "C" else "1"

    f_opt = FREQS_OPT
    sim, grid, trace_y_hi, d_set, p_set = build_setup(f_opt)
    ix, iy, iz = region_indices(grid, trace_y_hi)
    ndx, ndy = len(ix), len(iy)
    shape3 = grid.shape

    period = 1.0 / float(sim._freq_max)
    n_raw = int(math.ceil(NUM_PERIODS * period / float(grid.dt)))
    k_seg = max(8, int(math.isqrt(n_raw)))
    n_steps = ((n_raw + k_seg - 1) // k_seg) * k_seg

    freqs_j = jnp.asarray(f_opt, dtype=jnp.float32)
    beta0 = (
        2.0 * jnp.pi * freqs_j * jnp.sqrt(jnp.asarray(notch.EPS_EFF, dtype=jnp.float32))
        / jnp.asarray(C0, dtype=jnp.float32)
    )
    x_probes = jnp.array([0.0, d_set.delta, 2.0 * d_set.delta], dtype=jnp.float32)

    print(
        f"[phase1b:{arm}] grid={shape3} region={ndx}x{ndy}={ndx*ndy} cells "
        f"iz={iz} n_steps={n_steps} k={k_seg} iters={n_iters} init={args.init} smoke={SMOKE} "
        f"kottke={os.environ['RFX_PEC_OCC_KOTTKE']} sigma0={args.sigma0} q={args.qramp}"
    )

    ix_lo, ix_hi = int(ix[0]), int(ix[-1]) + 1
    iy_lo, iy_hi = int(iy[0]), int(iy[-1]) + 1

    def fields_from_theta(theta, beta):
        rho = jax.nn.sigmoid(theta)
        rho_f = apply_density_filter(rho, FILTER_R_CELLS)
        rho_p = apply_projection(rho_f, beta)
        occ = jnp.zeros(shape3, dtype=jnp.float32)
        occ = occ.at[ix_lo:ix_hi, iy_lo:iy_hi, iz].set(rho_p)
        sigma = None
        if arm == "B":
            sig_cell = args.sigma0 * rho_p / (1.0 + args.qramp * (1.0 - rho_p))
            sigma = jnp.zeros(shape3, dtype=jnp.float32)
            sigma = sigma.at[ix_lo:ix_hi, iy_lo:iy_hi, iz].set(sig_cell)
        return occ, sigma, rho_p

    def s21_ft(occ, sigma, steps, freqs_arr, b0, dset, pset):
        kw = dict(
            pec_occupancy_override=occ,
            n_steps=steps,
            checkpoint_segments=k_seg,
            skip_preflight=True,
        )
        if sigma is not None:
            kw["sigma_override"] = sigma
        fr = sim.forward(**kw)
        v_d = jnp.stack(
            [_v_from_plane(fr, dset.ez1_name, dset),
             _v_from_plane(fr, dset.ez2_name, dset),
             _v_from_plane(fr, dset.ez3_name, dset)], axis=-1)
        v_p = jnp.stack(
            [_v_from_plane(fr, pset.ez1_name, pset),
             _v_from_plane(fr, pset.ez2_name, pset),
             _v_from_plane(fr, pset.ez3_name, pset)], axis=-1)
        res_d = extract_msl_nprobe(v_d, x_probes, _i_from_plane(fr, dset.hy_name, dset), b0)
        res_p = extract_msl_nprobe(v_p, x_probes, _i_from_plane(fr, pset.hy_name, pset), b0)
        s21 = res_p["alpha"] / (res_d["alpha"] + 1e-30)
        return s21, fr

    # Empty-line reference (occ = 0 everywhere): per-frequency normalization
    # so the diverged extractor's absolute scale cancels out of the objective.
    occ_empty = jnp.zeros(shape3, dtype=jnp.float32)
    s21_ref, _ = s21_ft(occ_empty, None, n_steps, freqs_j, beta0, d_set, p_set)
    s21_ref_mag = jnp.abs(s21_ref) + 1e-30
    pb_idx = jnp.asarray([i for i in range(len(FREQS_OPT)) if i != IDX_NOTCH])
    print(f"[phase1b:{arm}] empty-line |S21| ref = "
          f"{[round(float(x),4) for x in s21_ref_mag]}")

    def loss(theta, beta):
        occ, sigma, _ = fields_from_theta(theta, beta)
        s21, _ = s21_ft(occ, sigma, n_steps, freqs_j, beta0, d_set, p_set)
        t = jnp.abs(s21) / s21_ref_mag           # normalized transmission
        j_notch = t[IDX_NOTCH] ** 2               # suppress at f_t
        j_pass = jnp.mean((t[pb_idx] - 1.0) ** 2)  # preserve passband
        return j_notch + W_PB * j_pass

    key = jax.random.PRNGKey(SEED)
    noise = 0.01 * jax.random.normal(key, (ndx, ndy), dtype=jnp.float32)
    if args.init == "uniform":
        theta0 = 0.0 + noise                       # rho ~ 0.5
    elif args.init == "low":
        theta0 = -1.1 + noise                      # rho ~ 0.25
    else:  # "stub": seed with the analytic lambda/4 stub, softly
        occ_seed3 = notch.build_stub_occ(grid, trace_y_hi,
                                         jnp.asarray(float(notch.L_TARGET_AN)))
        seed2d = np.asarray(occ_seed3)[ix_lo:ix_hi, iy_lo:iy_hi, iz]
        theta0 = jnp.asarray(np.where(seed2d > 0.5, 2.0, -2.0),
                             dtype=jnp.float32) + noise

    # ---- gate 2: arm B's sigma must matter (fail loud on silent ignore) ----
    if arm == "B":
        occ0, sig0, _ = fields_from_theta(theta0, beta_stages[0][1])
        s_a, _ = s21_ft(occ0, None, n_steps, freqs_j, beta0, d_set, p_set)
        big = jnp.zeros(shape3, dtype=jnp.float32).at[ix_lo:ix_hi, iy_lo:iy_hi, iz].set(1.0e4)
        s_b, _ = s21_ft(occ0, big, n_steps, freqs_j, beta0, d_set, p_set)
        d_rel = float(jnp.abs(jnp.abs(s_a[0]) - jnp.abs(s_b[0])) / (jnp.abs(s_a[0]) + 1e-30))
        print(f"[phase1:B] sigma-effect gate: rel change {d_rel:.3e}")
        if d_rel < 1e-3:
            raise SystemExit("sigma_override appears to be silently ignored alongside "
                             "pec_occupancy_override — STOP (needs core support)")

    # ---- gate 1: AD-vs-FD at init (A and B) ----
    # NOTE: no jax.jit wrapper — under jit even constants become abstract and
    # rfx.topology.apply_density_filter's `int(jnp.ceil(radius))` kernel-size
    # computation raises ConcretizationTypeError. Plain value_and_grad traces
    # with concrete constants (the notch example's pattern); forward() jits
    # the scan internally, so the per-iteration overhead is negligible.
    grad_fn = jax.value_and_grad(loss)
    fd_report = []
    if arm in ("A", "B") and not SMOKE:
        beta_g = beta_stages[0][1]
        _, g = grad_fn(theta0, beta_g)
        rng = np.random.default_rng(1)
        cells = [(int(rng.integers(0, ndx)), int(rng.integers(0, ndy))) for _ in range(2)]
        for (ci, cj) in cells:
            for eps_fd in (3e-2, 1e-2):  # two-step sweep: FD noise vs truncation
                tp = theta0.at[ci, cj].add(eps_fd)
                tm = theta0.at[ci, cj].add(-eps_fd)
                jp = float(loss(tp, beta_g))
                jm = float(loss(tm, beta_g))
                fd = (jp - jm) / (2 * eps_fd)
                ad = float(g[ci, cj])
                rel = abs(ad - fd) / (abs(fd) + 1e-12)
                fd_report.append(dict(cell=[ci, cj], eps=eps_fd, ad=ad, fd=fd, rel_err=rel))
                print(f"[phase1b:{arm}] gradcheck cell({ci},{cj}) eps={eps_fd:.0e} "
                      f"AD={ad:+.4e} FD={fd:+.4e} rel={rel:.3f}")

    # ---- Adam with beta continuation ----
    import optax
    opt = optax.adam(LR)
    theta = theta0
    state = opt.init(theta)
    hist = []
    t0 = time.time()
    for it in range(n_iters):
        beta_now = max(b for s_, b in beta_stages if it >= s_)
        (j_val, g) = grad_fn(theta, beta_now)
        upd, state = opt.update(g, state)
        theta = optax.apply_updates(theta, upd)
        hist.append(dict(iter=it, J=float(j_val), beta=beta_now))
        if it % 5 == 0 or it == n_iters - 1:
            print(f"[phase1b:{arm}] it={it:3d} beta={beta_now:4.0f} "
                  f"J={float(j_val):.5f} ({time.time()-t0:.0f}s)")
            np.savez(OUT / f"{TAG}_{arm}_theta.npz", theta=np.asarray(theta), it=it)

    # ---- final: binarized hard re-evaluation, long window, full band ----
    _, _, rho_final = fields_from_theta(theta, beta_stages[-1][1])
    hard = (np.asarray(rho_final) > 0.5).astype(np.float32)
    fill = float(hard.mean())

    os.environ["RFX_PEC_OCC_KOTTKE"] = "1"  # identical hard evaluator for all arms
    sim2, grid2, trace_y_hi2, d2, p2 = build_setup(FREQS_BAND)
    n_raw2 = int(math.ceil(NUM_PERIODS_EVAL * (1.0 / float(sim2._freq_max)) / float(grid2.dt)))
    k2 = max(8, int(math.isqrt(n_raw2)))
    n_eval = ((n_raw2 + k2 - 1) // k2) * k2
    freqs_b = jnp.asarray(FREQS_BAND, dtype=jnp.float32)
    beta0_b = (2.0 * jnp.pi * freqs_b * jnp.sqrt(jnp.asarray(notch.EPS_EFF, dtype=jnp.float32))
               / jnp.asarray(C0, dtype=jnp.float32))
    occ_hard = jnp.zeros(shape3, dtype=jnp.float32).at[ix_lo:ix_hi, iy_lo:iy_hi, iz].set(
        jnp.asarray(hard))

    def hard_eval(occ):
        fr = sim2.forward(pec_occupancy_override=occ, n_steps=n_eval,
                          checkpoint_segments=k2, skip_preflight=True)
        v_d = jnp.stack([_v_from_plane(fr, d2.ez1_name, d2),
                         _v_from_plane(fr, d2.ez2_name, d2),
                         _v_from_plane(fr, d2.ez3_name, d2)], axis=-1)
        v_p = jnp.stack([_v_from_plane(fr, p2.ez1_name, p2),
                         _v_from_plane(fr, p2.ez2_name, p2),
                         _v_from_plane(fr, p2.ez3_name, p2)], axis=-1)
        rd = extract_msl_nprobe(v_d, x_probes, _i_from_plane(fr, d2.hy_name, d2), beta0_b)
        rp = extract_msl_nprobe(v_p, x_probes, _i_from_plane(fr, p2.hy_name, p2), beta0_b)
        s21 = np.asarray(rp["alpha"] / (rd["alpha"] + 1e-30))
        ez = np.abs(np.asarray(fr.dft_planes["ez_surface"].accumulator))[0]
        return s21, ez

    s21_hard, ez_hard = hard_eval(occ_hard)
    s21_hard_ref, _ = hard_eval(jnp.zeros(shape3, dtype=jnp.float32))
    it_t = int(np.argmin(np.abs(FREQS_BAND - F_TARGET)))
    j_hard = float(np.abs(s21_hard[it_t]) ** 2)
    db_hard = 10 * np.log10(j_hard + 1e-15)
    # selectivity metrics on the hard design, empty-line-normalized
    t_hard = np.abs(s21_hard) / (np.abs(s21_hard_ref) + 1e-30)
    pb_mask = np.abs(FREQS_BAND - F_TARGET) > 0.9e9
    t_notch = float(t_hard[it_t])
    t_pb = float(np.mean(t_hard[pb_mask]))
    contrast_db = 20 * np.log10((t_pb + 1e-12) / (t_notch + 1e-12))

    # oracle reference: the analytic lambda/4 stub, binarized, same evaluator
    occ_stub = notch.build_stub_occ(grid2, trace_y_hi2, jnp.asarray(float(notch.L_TARGET_AN)))
    occ_stub = (np.asarray(occ_stub) > 0.5).astype(np.float32)
    s21_stub, _ = hard_eval(jnp.asarray(occ_stub))
    j_stub = float(np.abs(s21_stub[it_t]) ** 2)
    t_stub = np.abs(s21_stub) / (np.abs(s21_hard_ref) + 1e-30)
    stub_contrast_db = 20 * np.log10(
        (float(np.mean(t_stub[pb_mask])) + 1e-12) / (float(t_stub[it_t]) + 1e-12))

    print(f"[phase1b:{arm}] FINAL J_soft={hist[-1]['J']:.5f} fill={fill:.2f} "
          f"t_notch={t_notch:.4f} t_pb={t_pb:.4f} CONTRAST={contrast_db:.1f} dB "
          f"(oracle stub contrast {stub_contrast_db:.1f} dB) "
          f"J_hard(f_t)={j_hard:.5f} ({db_hard:.1f} dB)")

    out = dict(tag=TAG, arm=arm, smoke=SMOKE, n_iters=n_iters, init=args.init, n_steps=n_steps,
               n_eval=n_eval, region=[ndx, ndy], sigma0=args.sigma0,
               qramp=args.qramp, w_pb=W_PB, fill_hard=fill, J_hard_ft=j_hard,
               J_hard_ft_db=db_hard, t_notch=t_notch, t_pb=t_pb,
               contrast_db=contrast_db, J_oracle_stub=j_stub,
               oracle_contrast_db=stub_contrast_db,
               s21_hard_band_db=[float(20 * np.log10(abs(x) + 1e-12)) for x in s21_hard],
               t_hard_band=[float(x) for x in t_hard],
               freqs_GHz=[f / 1e9 for f in FREQS_BAND],
               fd_report=fd_report, history=hist)
    (OUT / f"{TAG}_{arm}.json").write_text(json.dumps(out, indent=2))
    np.savez(OUT / f"{TAG}_{arm}_final.npz", theta=np.asarray(theta),
             rho=np.asarray(rho_final), hard=hard, ez=ez_hard)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    axes[0].plot([h["iter"] for h in hist], [h["J"] for h in hist])
    axes[0].set_yscale("log"); axes[0].set_title(f"J descent (arm {arm})")
    im = axes[1].imshow(np.asarray(rho_final).T, origin="lower", cmap="gray_r",
                        vmin=0, vmax=1, aspect="auto")
    axes[1].set_title("final density"); fig.colorbar(im, ax=axes[1], shrink=0.8)
    im = axes[2].imshow(hard.T, origin="lower", cmap="gray_r", vmin=0, vmax=1, aspect="auto")
    axes[2].set_title(f"binarized (fill {fill:.2f})"); fig.colorbar(im, ax=axes[2], shrink=0.8)
    axes[3].plot(out["freqs_GHz"], 20 * np.log10(np.asarray(out["t_hard_band"]) + 1e-12), "o-")
    axes[3].axvline(F_TARGET / 1e9, ls="--", c="r")
    axes[3].set_ylabel("normalized |S21| (dB)")
    axes[3].set_title(f"hard t(f), contrast={contrast_db:.1f} dB")
    fig.suptitle(f"{TAG} arm {arm}: selective notch TO, {ndx}x{ndy} cells, w_pb={W_PB}")
    fig.tight_layout()
    fig.savefig(OUT / f"{TAG}_{arm}.png", dpi=110)
    print(f"[phase1b:{arm}] wrote {OUT}/{TAG}_{arm}.json/.png/.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
