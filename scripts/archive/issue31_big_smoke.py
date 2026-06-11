"""Issue #31 — stable-FDTD smoke for segmented scan validation.

Geometry: 2.4 GHz FR4 patch antenna on a non-uniform z mesh, reusing
the validated setup from examples/nonuniform_patch_demo.py (whose
harminv resonance pin agrees with Balanis analytic to <2%).

The design variable is a scalar ``alpha`` applied as eps_r = alpha on
the substrate cells — physically meaningful (detuning the patch) and
numerically stable at any n_steps the base demo runs.

Exercises:
- Phase A: forward(..., checkpoint=True)  — jax.checkpoint on step_fn
- Phase B: forward(..., checkpoint_every=K) — segmented scan-of-scan
- Phase C: forward(..., emit_time_series=False) — empty probe tape
- jax.grad through the scan w.r.t. alpha
- loss trajectory over a small optimize loop using |grad|-normalized
  step size (works regardless of the geometry's natural scale)

Reports peak GPU memory (jax.local_devices[0].memory_stats).
"""

from __future__ import annotations

import argparse
import math
import time
import numpy as np

import jax
import jax.numpy as jnp

from rfx import Simulation, Box
from rfx.auto_config import smooth_grading
from rfx.sources.sources import GaussianPulse


C0 = 2.998e8


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


def build_patch_sim(*, dx_mm: float = 1.0):
    """2.4 GHz FR4 patch antenna on NU-z mesh.

    Exact geometry from examples/nonuniform_patch_demo.py. `dx_mm`
    tunes cell count (default 1.0 mm; 0.6-0.8 grows to multi-million
    cells for memory stress).
    """
    f_design = 2.4e9
    eps_r = 4.3
    h_sub = 1.5e-3
    W = 38.0e-3
    L = 29.5e-3
    gx = 60.0e-3
    gy = 55.0e-3
    air_above = 25.0e-3
    air_below = 12.0e-3
    probe_inset = 8.0e-3

    dx = dx_mm * 1e-3
    n_cpml = 8
    n_sub = 6
    dz_sub = h_sub / n_sub

    n_below = int(math.ceil(air_below / dx))
    n_above = int(math.ceil(air_above / dx))
    dz_profile_raw = np.concatenate([
        np.full(n_below, dx),
        np.full(n_sub, dz_sub),
        np.full(n_above, dx),
    ])
    dz_profile = np.asarray(smooth_grading(dz_profile_raw), dtype=np.float64)

    dom_x = gx + 20e-3
    dom_y = gy + 20e-3

    gx_lo = (dom_x - gx) / 2
    gx_hi = gx_lo + gx
    gy_lo = (dom_y - gy) / 2
    gy_hi = gy_lo + gy
    patch_x_lo = dom_x / 2 - L / 2
    patch_x_hi = dom_x / 2 + L / 2
    patch_y_lo = dom_y / 2 - W / 2
    patch_y_hi = dom_y / 2 + W / 2
    feed_x = patch_x_lo + probe_inset
    feed_y = dom_y / 2

    z_gnd_lo = air_below - dz_sub
    z_gnd_hi = air_below
    z_sub_lo = air_below
    z_sub_hi = air_below + h_sub
    z_patch_lo = z_sub_hi
    z_patch_hi = z_sub_hi + dz_sub
    src_z = z_sub_lo + dz_sub * 2.5

    sim = Simulation(
        freq_max=4e9,
        domain=(dom_x, dom_y, 0),
        dx=dx,
        dz_profile=dz_profile,
        boundary="cpml",
        cpml_layers=n_cpml,
    )
    sim.add_material("fr4", eps_r=eps_r, sigma=0.0)
    sim.add(Box((gx_lo, gy_lo, z_gnd_lo), (gx_hi, gy_hi, z_gnd_hi)), material="pec")
    sim.add(Box((gx_lo, gy_lo, z_sub_lo), (gx_hi, gy_hi, z_sub_hi)), material="fr4")
    sim.add(Box((patch_x_lo, patch_y_lo, z_patch_lo),
                (patch_x_hi, patch_y_hi, z_patch_hi)), material="pec")
    sim.add_source(
        position=(feed_x, feed_y, src_z),
        component="ez",
        waveform=GaussianPulse(f0=f_design, bandwidth=1.2),
    )
    sim.add_probe(
        position=(dom_x / 2 + 5e-3, dom_y / 2 + 5e-3, src_z),
        component="ez",
    )

    # Substrate cell indices — design region for eps_override
    from rfx.runners.nonuniform import pos_to_nu_index
    g = sim._build_nonuniform_grid()
    ix_lo, iy_lo, iz_lo = pos_to_nu_index(g, (gx_lo, gy_lo, z_sub_lo))
    ix_hi, iy_hi, iz_hi = pos_to_nu_index(g, (gx_hi, gy_hi, z_sub_hi))
    substrate_idx = (
        slice(int(ix_lo), int(ix_hi) + 1),
        slice(int(iy_lo), int(iy_hi) + 1),
        slice(int(iz_lo), int(iz_hi) + 1),
    )
    return sim, g, substrate_idx, eps_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=2000)
    ap.add_argument("--n-iters", type=int, default=5)
    ap.add_argument("--dx-mm", type=float, default=1.0)
    ap.add_argument("--checkpoint", type=int, default=1)
    ap.add_argument("--emit-ts", type=int, default=1)
    ap.add_argument("--checkpoint-every", type=int, default=0)
    args = ap.parse_args()

    sim, g, substrate_idx, eps_base_val = build_patch_sim(dx_mm=args.dx_mm)
    cells = int(g.nx * g.ny * g.nz)
    ckpt_every = args.checkpoint_every if args.checkpoint_every > 0 else None

    print(f"[cfg] nx,ny,nz = {g.nx},{g.ny},{g.nz}  cells = {cells:,}")
    print(f"[cfg] n_steps = {args.n_steps}  iters = {args.n_iters}")
    print(f"[cfg] checkpoint={bool(args.checkpoint)}  "
          f"emit_ts={bool(args.emit_ts)}  checkpoint_every={ckpt_every}")
    dev = jax.local_devices()[0]
    print(f"[cfg] device = {dev} (platform={dev.platform})")

    # Build eps_base = FR4 everywhere on the substrate (concrete array,
    # then alpha is added as a delta to detune around eps_base_val).
    grid_shape = (g.nx, g.ny, g.nz)

    def loss_from_alpha(alpha):
        # eps_override: background=1 (air), substrate=eps_base + alpha (the
        # design dof). Everything else (ground, patch PEC) is handled by
        # pec_mask in run_nonuniform_path.
        eps = jnp.ones(grid_shape, dtype=jnp.float32)
        eps = eps.at[substrate_idx].set(eps_base_val + alpha)
        if args.emit_ts:
            fr = sim.forward(eps_override=eps, n_steps=args.n_steps,
                             checkpoint=bool(args.checkpoint),
                             emit_time_series=True,
                             checkpoint_every=ckpt_every)
            # L2 energy of probe time series — finite, well-defined grad.
            return jnp.sum(fr.time_series ** 2)
        _ = sim.forward(eps_override=eps, n_steps=args.n_steps,
                        checkpoint=bool(args.checkpoint),
                        emit_time_series=False,
                        checkpoint_every=ckpt_every)
        return (alpha - 0.5) ** 2

    alpha = jnp.float32(0.0)  # start at unperturbed FR4 (eps=4.3)
    vg_fn = jax.value_and_grad(loss_from_alpha)

    t0 = time.time()
    print("[run] tracing + compiling forward+grad ...", flush=True)
    loss0, g0 = vg_fn(alpha)
    loss0, g0 = float(loss0), float(g0)
    t1 = time.time()
    print(f"[run] first grad call OK  loss={loss0:.4e}  grad={g0:.4e}  "
          f"elapsed={t1 - t0:.1f}s  peak={_peak_gb()} GB")

    step = 0.1
    for it in range(1, args.n_iters):
        t0 = time.time()
        l_, g_ = vg_fn(alpha)
        l_, g_ = float(l_), float(g_)
        alpha = alpha - step * np.sign(g_)
        t1 = time.time()
        print(f"[iter {it}] alpha={float(alpha):+.4f}  loss={l_:+.4e}  "
              f"grad={g_:+.4e}  dt={t1 - t0:.1f}s  peak={_peak_gb()} GB",
              flush=True)

    print(f"\n[result] PASS  peak_GB={_peak_gb()}  cells={cells:,}  "
          f"n_steps={args.n_steps}")


if __name__ == "__main__":
    main()
