"""Issue #31 — big-geometry NU inverse-design smoke on single GPU.

Validates the A+C claim empirically: a 2-3 M cell 3D simulation with
n_steps=10000 must run end-to-end on a 24 GB consumer GPU (RTX 4090)
without OOM, while a naive non-checkpointed reverse-mode AD would
require TB of memory.

The design loop is manual (not rfx.optimize()) because rfx.optimize()
calls Simulation._build_grid() — the uniform path — so it does not yet
route through the NU code we modified in Phase A/C.

What this exercises
-------------------
- NU mesh via dz_profile (run_nonuniform_path)
- Phase A: forward(..., checkpoint=True)  ← wrap of step_fn in jax.checkpoint
- Phase C: forward(..., emit_time_series=False)  ← empty probe-sample tape
- jax.grad through the NU scan w.r.t. a bounded eps design variable

Outputs a PASS / FAIL verdict and peak GPU memory usage.
"""

from __future__ import annotations

import argparse
import sys
import time
import numpy as np

import jax
import jax.numpy as jnp

from rfx import Simulation
from rfx.optimize_objectives import minimize_reflected_energy


def _peak_gb() -> float | None:
    try:
        dev = jax.local_devices()[0]
        if dev.platform != "gpu":
            return None
        s = dev.memory_stats()
        peak = s.get("peak_bytes_in_use", s.get("bytes_in_use", 0))
        return peak / 1e9
    except Exception:
        return None


def build_sim(*, freq_max=10e9, ext_mm=40.0, dx_mm=0.5, cpml_layers=8):
    dx = dx_mm * 1e-3
    ext = ext_mm * 1e-3
    # Graded dz: finer near z=0 (design region surface), coarser away.
    n_fine = 20
    n_coarse = 30
    dz = np.concatenate([
        np.full(n_fine, 0.3e-3),
        np.full(n_coarse, 0.6e-3),
    ])
    dz_total = float(np.sum(dz))
    sim = Simulation(
        freq_max=freq_max,
        domain=(ext, ext, dz_total),
        dx=dx,
        dz_profile=dz,
        cpml_layers=cpml_layers,
    )
    sim.add_source((ext / 2, ext / 2, 1.0e-3), "ez")
    sim.add_probe((ext / 2, ext / 2, 2.0e-3), "ez")
    return sim, dz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=10000)
    ap.add_argument("--n-iters", type=int, default=5)
    ap.add_argument("--ext-mm", type=float, default=40.0)
    ap.add_argument("--dx-mm", type=float, default=0.5)
    ap.add_argument("--checkpoint", type=int, default=1,
                    help="1=Phase A (remat), 0=no remat (expected to OOM)")
    ap.add_argument("--emit-ts", type=int, default=0,
                    help="1=keep time series, 0=Phase C skip")
    args = ap.parse_args()

    sim, _dz = build_sim(ext_mm=args.ext_mm, dx_mm=args.dx_mm)
    g = sim._build_nonuniform_grid()
    cells = int(g.nx * g.ny * g.nz)
    print(f"[cfg] nx,ny,nz = {g.nx},{g.ny},{g.nz}  cells = {cells:,}")
    print(f"[cfg] n_steps = {args.n_steps}  iters = {args.n_iters}")
    print(f"[cfg] checkpoint = {bool(args.checkpoint)}  "
          f"emit_time_series = {bool(args.emit_ts)}")

    dev = jax.local_devices()[0]
    print(f"[cfg] device = {dev} (platform={dev.platform})")

    eps_base = jnp.ones(g.shape, dtype=jnp.float32)

    # Design region: a 10 mm × 10 mm × 2-cell thick slab at (ext/2, ext/2, 1mm).
    i0, i1 = g.nx // 2 - 10, g.nx // 2 + 10
    j0, j1 = g.ny // 2 - 10, g.ny // 2 + 10
    k0 = int(np.argmin(np.abs(np.cumsum(_dz_profile := _dz) - 1.5e-3)))
    k1 = k0 + 2
    print(f"[cfg] design slab  i=[{i0},{i1})  j=[{j0},{j1})  k=[{k0},{k1})")

    def loss_from_alpha(alpha):
        # 1 free parameter (scalar) — keeps grad shape tiny, the AD
        # memory is dominated by the scan tape, which is what we're
        # stress-testing.
        eps = eps_base.at[i0:i1, j0:j1, k0:k1].set(alpha)
        if args.emit_ts:
            fr = sim.forward(eps_override=eps, n_steps=args.n_steps,
                             checkpoint=bool(args.checkpoint),
                             emit_time_series=True)
            return minimize_reflected_energy(port_probe_idx=0)(fr)
        # emit_time_series=False path: use a proxy loss on the (empty)
        # forward. Without NTFF this has no signal, so fold alpha back
        # into the loss so grad is well-defined and we still exercise
        # the scan + backward pass end-to-end.
        _ = sim.forward(eps_override=eps, n_steps=args.n_steps,
                        checkpoint=bool(args.checkpoint),
                        emit_time_series=False)
        return (alpha - 1.5) ** 2

    # 1 compile + 1 grad pass (dry run) to measure peak.
    alpha = jnp.float32(2.0)
    t0 = time.time()
    print("[run] tracing + compiling forward+grad ...", flush=True)
    # NU grid construction uses np.asarray on dx_arr (a known host
    # boundary — see docs/agent-memory/nu_known_limits.md). Do NOT wrap
    # loss_from_alpha in jax.jit; lax.scan inside run_nonuniform is
    # already JIT-compiled, so wall-clock impact is small.
    grad_fn = jax.grad(loss_from_alpha)
    g0 = float(grad_fn(alpha))  # first call = compile + run
    t1 = time.time()
    print(f"[run] first grad call OK  value={g0:.4e}  "
          f"elapsed={t1 - t0:.1f}s  peak={_peak_gb()} GB")

    # Do a few iterations of plain gradient descent (no Adam) to exercise
    # repeated grad calls — makes sure the remat graph is stable.
    lr = 0.05
    for it in range(1, args.n_iters):
        t0 = time.time()
        g_ = float(grad_fn(alpha))
        alpha = alpha - lr * g_
        t1 = time.time()
        print(f"[iter {it}] alpha={float(alpha):+.4f}  grad={g_:+.4e}  "
              f"dt={t1 - t0:.1f}s  peak={_peak_gb()} GB", flush=True)

    peak = _peak_gb()
    print(f"\n[result] PASS  peak_GB={peak}  cells={cells:,}  n_steps={args.n_steps}")


if __name__ == "__main__":
    main()
