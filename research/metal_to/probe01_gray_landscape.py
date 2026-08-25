"""Phase-0 probe #1 — the gray-metal continuation landscape, measured.

Question
--------
Free-form binary metal topology optimization is the accepted T-MTT paper's
principal open limitation: preliminary density runs converged to weaker
optima, attributed to the "gray" (0 < occ < 1) metal that every
beta-continuation path must traverse. Before building any remedy (RAMP-style
interpolation shaping), this probe MEASURES what the traversal actually looks
like, on both solver paths that consume ``pec_occupancy_override``:

  legacy  ``apply_pec_occupancy``   E-tangential damping, E *= (1 - occ)
                                    per step -> expected: gray cells act as a
                                    strong NUMERICAL ABSORBER (A(occ) peaks
                                    mid-range).
  kottke  ``RFX_PEC_OCC_KOTTKE=1``  Kottke inv-eps PEC limit -> gray cells
                                    are a LOSSLESS high-eps anisotropic
                                    mixture -> expected: no absorption, but
                                    strong resonance DETUNING along the path.

Geometry: the validated MSL stub-notch setup (validation/tmtt_paper/
msl_stub_notch_tuning.py), stub FIXED at the analytic lambda/4 length for the
6 GHz notch. The 1-D family occ(level) = level * stub_mask is exactly the
uniform continuation path from "no metal" (level 0) to "full stub" (level 1).

Measured per level: S11 = gamma_d/alpha_d, S21 = alpha_p/alpha_d at f_target
and across a small band; R = |S11|^2, T = |S21|^2, A = 1 - R - T; plus an Ez
DFT plane map at the substrate surface for three representative levels.

Pre-agreed falsifier (STOP rule)
--------------------------------
The gray-traversal hypothesis predicts a NON-BENIGN landscape between the
endpoints on the production (kottke) path: a barrier or inversion in
J(level) = |S21(f_t)|^2, or an absorption bump (legacy). If instead J(level)
descends monotonically to its level=1 value on the kottke path, the gray
traversal is NOT the blocker and interpolation shaping is NOT the first fix
-> STOP, report, rethink (spatial multimodality / filter-projection interplay
become the suspects).

Run
---
  RFX_PEC_OCC_KOTTKE=0 python research/metal_to/probe01_gray_landscape.py --path legacy
  RFX_PEC_OCC_KOTTKE=1 python research/metal_to/probe01_gray_landscape.py --path kottke

(CPU, ~15-25 min per path at 11 levels; JAX_PLATFORMS=cpu is fine.)
Outputs land in research/metal_to/out/probe01_<path>.{json,png}.
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

import msl_stub_notch_tuning as notch  # noqa: E402  (validated template)
from rfx.probes.msl_wave_decomp import (  # noqa: E402
    _i_from_plane,
    _v_from_plane,
    extract_msl_nprobe,
)

C0 = 2.998e8
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

# Small band around the 6 GHz target so the notch's detuning is visible.
FREQS_HZ = np.linspace(4.5e9, 8.5e9, 9)
F_TARGET = notch.F_TARGET  # 6.0 GHz
LEVELS = np.round(np.linspace(0.0, 1.0, 11), 3)
FIELD_LEVELS = (0.3, 0.7, 1.0)  # Ez maps captured at these levels
NUM_PERIODS = float(os.environ.get("RFX_Y2B_PERIODS", 10.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", choices=("legacy", "kottke"), required=True)
    args = ap.parse_args()

    want = "1" if args.path == "kottke" else "0"
    os.environ["RFX_PEC_OCC_KOTTKE"] = want

    freqs = jnp.asarray(FREQS_HZ, dtype=jnp.float32)
    sim, y_trace, trace_y_hi, d_set, p_set = notch.build_sim(freqs)

    # Ez map on the substrate surface (z = h_sub + dx/2), at f_target only.
    sim.add_dft_plane_probe(
        axis="z",
        coordinate=notch.H_SUB + 0.5 * notch.DX,
        component="ez",
        freqs=jnp.asarray([F_TARGET], dtype=jnp.float32),
        name="ez_surface",
    )

    grid = sim._build_grid()
    pre = sim.preflight()
    for m in pre:
        print(f"  preflight: {m}")

    # Fixed stub at the analytic lambda/4 length; the sweep scales its level.
    L_FIX = float(notch.L_TARGET_AN)
    occ_base = notch.build_stub_occ(grid, trace_y_hi, jnp.asarray(L_FIX))

    period = 1.0 / float(sim._freq_max)
    n_raw = int(math.ceil(NUM_PERIODS * period / float(grid.dt)))
    k_seg = max(8, int(math.isqrt(n_raw)))
    n_steps = ((n_raw + k_seg - 1) // k_seg) * k_seg

    beta0 = (
        2.0 * jnp.pi * freqs * jnp.sqrt(jnp.asarray(notch.EPS_EFF, dtype=jnp.float32))
        / jnp.asarray(C0, dtype=jnp.float32)
    )
    x_probes = jnp.array([0.0, d_set.delta, 2.0 * d_set.delta], dtype=jnp.float32)

    print(
        f"[probe01] path={args.path} grid={grid.shape} n_steps={n_steps} "
        f"L_fix={L_FIX*1e3:.3f} mm levels={list(LEVELS)}"
    )

    def solve(level: float):
        occ = jnp.asarray(level, dtype=jnp.float32) * occ_base
        fr = sim.forward(
            pec_occupancy_override=occ,
            n_steps=n_steps,
            checkpoint_segments=k_seg,
            skip_preflight=True,
        )
        v_d = jnp.stack(
            [
                _v_from_plane(fr, d_set.ez1_name, d_set),
                _v_from_plane(fr, d_set.ez2_name, d_set),
                _v_from_plane(fr, d_set.ez3_name, d_set),
            ],
            axis=-1,
        )
        v_p = jnp.stack(
            [
                _v_from_plane(fr, p_set.ez1_name, p_set),
                _v_from_plane(fr, p_set.ez2_name, p_set),
                _v_from_plane(fr, p_set.ez3_name, p_set),
            ],
            axis=-1,
        )
        res_d = extract_msl_nprobe(v_d, x_probes, _i_from_plane(fr, d_set.hy_name, d_set), beta0)
        res_p = extract_msl_nprobe(v_p, x_probes, _i_from_plane(fr, p_set.hy_name, p_set), beta0)
        s11 = res_d["gamma"] / (res_d["alpha"] + 1e-30)
        s21 = res_p["alpha"] / (res_d["alpha"] + 1e-30)
        ez_map = np.abs(np.asarray(fr.dft_planes["ez_surface"]))[0]
        return np.asarray(s11), np.asarray(s21), ez_map

    rows = []
    ez_maps = {}
    for lv in LEVELS:
        t0 = time.time()
        s11, s21, ez_map = solve(float(lv))
        it = int(np.argmin(np.abs(FREQS_HZ - F_TARGET)))
        R = float(np.abs(s11[it]) ** 2)
        T = float(np.abs(s21[it]) ** 2)
        row = dict(
            level=float(lv),
            s11_ft=float(np.abs(s11[it])),
            s21_ft=float(np.abs(s21[it])),
            R=R,
            T=T,
            A=float(1.0 - R - T),
            s21_db_band=[float(20 * np.log10(abs(x) + 1e-12)) for x in s21],
            wall_s=round(time.time() - t0, 1),
        )
        rows.append(row)
        if any(abs(lv - fl) < 1e-9 for fl in FIELD_LEVELS):
            ez_maps[f"{lv:.1f}"] = ez_map
        print(
            f"  level={lv:4.1f}  |S21(f_t)|={row['s21_ft']:.4f} "
            f"R={R:.3f} T={T:.3f} A={row['A']:+.3f}  ({row['wall_s']}s)"
        )

    out = dict(
        path=args.path,
        kottke=want == "1",
        L_fix_mm=L_FIX * 1e3,
        f_target_GHz=F_TARGET / 1e9,
        freqs_GHz=[f / 1e9 for f in FREQS_HZ],
        n_steps=n_steps,
        num_periods=NUM_PERIODS,
        grid_shape=list(grid.shape),
        rows=rows,
    )
    (OUT / f"probe01_{args.path}.json").write_text(json.dumps(out, indent=2))

    # ---- figure: landscape + power balance + Ez maps ----
    lv = [r["level"] for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0, 0]
    ax.plot(lv, [r["s21_ft"] ** 2 for r in rows], "o-")
    ax.set_xlabel("occupancy level")
    ax.set_ylabel(r"$J=|S_{21}(f_t)|^2$")
    ax.set_title(f"objective landscape ({args.path})")
    ax = axes[0, 1]
    ax.plot(lv, [r["R"] for r in rows], "o-", label="R")
    ax.plot(lv, [r["T"] for r in rows], "s-", label="T")
    ax.plot(lv, [r["A"] for r in rows], "^-", label="A=1-R-T")
    ax.set_xlabel("occupancy level")
    ax.set_title("power balance at $f_t$")
    ax.legend()
    ax = axes[0, 2]
    for r in rows[:: max(1, len(rows) // 5)]:
        ax.plot(out["freqs_GHz"], r["s21_db_band"], label=f"lv {r['level']:.1f}")
    ax.set_xlabel("f (GHz)")
    ax.set_ylabel(r"$|S_{21}|$ (dB)")
    ax.set_title("band response vs level")
    ax.legend(fontsize=7)
    for j, (k, m) in enumerate(sorted(ez_maps.items())[:3]):
        ax = axes[1, j]
        im = ax.imshow(m.T, origin="lower", aspect="auto", cmap="inferno")
        ax.set_title(f"|Ez| surface DFT @ $f_t$, level {k}")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(
        f"probe01 gray-metal continuation landscape — {args.path} path, "
        f"stub fixed at {L_FIX*1e3:.2f} mm"
    )
    fig.tight_layout()
    fig.savefig(OUT / f"probe01_{args.path}.png", dpi=110)
    print(f"[probe01] wrote {OUT}/probe01_{args.path}.json + .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
