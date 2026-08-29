"""W5 — AD consistency on a multiband dz_profile (F-S5).

Primary gate (pre-declared, existing NU convention): jax.grad vs central
FD < 15 % relative on dominant cells (|g| > 5 % of max), f32 path,
h = 1e-3 relative. The x64-context arm is measured and reported as
knowledge output (the NU profile path hard-casts float32 — note §5).

Usage:
    PYTHONPATH=. python -m validation.research.multiband_nu.w5_ad_consistency
"""

from __future__ import annotations

import json

import numpy as np
import jax
import jax.numpy as jnp

from rfx.core.yee import MaterialArrays
from rfx.nonuniform import make_nonuniform_grid, run_nonuniform

from . import fixtures as fx

N_STEPS = 60
FD_REL_H = 1e-3
DOMINANT_FRAC = 0.05
TOL = 0.15


def loss_from_dz(dz):
    grid = make_nonuniform_grid(
        domain_xy=(0.005, 0.005), dz_profile=dz, dx=0.5e-3, cpml_layers=4)
    nx, ny, nz = grid.shape
    shape = (nx, ny, nz)
    materials = MaterialArrays(
        eps_r=jnp.ones(shape, dtype=jnp.float32),
        mu_r=jnp.ones(shape, dtype=jnp.float32),
        sigma=jnp.zeros(shape, dtype=jnp.float32),
    )
    t = jnp.arange(N_STEPS, dtype=jnp.float32) * jnp.asarray(grid.dt, jnp.float32)
    t0 = 15.0 * jnp.asarray(grid.dt, jnp.float32)
    width = 5.0 * jnp.asarray(grid.dt, jnp.float32)
    wf = jnp.exp(-(((t - t0) / width) ** 2)).astype(jnp.float32)
    out = run_nonuniform(
        grid, materials, N_STEPS,
        sources=[(nx // 2, ny // 2, nz // 2, "ez", wf)],
        probes=[(nx // 2, ny // 2, nz // 2 - 2, "ez")])
    return jnp.sum(out["time_series"] ** 2)


def measure(label: str) -> dict:
    dz0 = fx.w5_profile()
    g_ad = np.asarray(jax.grad(loss_from_dz)(jnp.asarray(dz0)), dtype=np.float64)
    g_fd = np.zeros_like(dz0)
    for k in range(len(dz0)):
        h = FD_REL_H * dz0[k]
        dzp = dz0.copy(); dzp[k] += h
        dzm = dz0.copy(); dzm[k] -= h
        g_fd[k] = (float(loss_from_dz(jnp.asarray(dzp)))
                   - float(loss_from_dz(jnp.asarray(dzm)))) / (2 * h)
    gmax = np.abs(g_fd).max()
    dominant = np.abs(g_fd) > DOMINANT_FRAC * gmax
    rel = np.abs(g_ad - g_fd) / np.maximum(np.abs(g_fd), 1e-300)
    worst = float(rel[dominant].max())
    return {
        "label": label,
        "n_cells": len(dz0), "n_dominant": int(dominant.sum()),
        "worst_dominant_rel_err": worst,
        "median_dominant_rel_err": float(np.median(rel[dominant])),
        "fs5_fired": bool(worst > TOL),
        "g_ad_dominant": g_ad[dominant].tolist(),
        "g_fd_dominant": g_fd[dominant].tolist(),
    }


def main():
    out = {}
    res32 = measure("f32-default")
    print(f"W5 f32: worst dominant rel err {res32['worst_dominant_rel_err']:.3e} "
          f"({res32['n_dominant']} dominant cells) fired={res32['fs5_fired']}",
          flush=True)
    out["f32"] = res32

    # x64-context arm (reported, not a gate — note §5 premise finding)
    try:
        from tests._x64_compat import enable_x64
        with enable_x64():
            jax.clear_caches()
            res64 = measure("x64-context")
        jax.clear_caches()
        print(f"W5 x64-context: worst dominant rel err "
              f"{res64['worst_dominant_rel_err']:.3e}", flush=True)
        out["x64_context"] = res64
    except Exception as exc:  # noqa: BLE001 — report, don't gate
        out["x64_context"] = {"error": repr(exc)}
        print("W5 x64-context arm failed:", repr(exc), flush=True)

    with open("validation/research/multiband_nu/results/w5_ad.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote results/w5_ad.json")


if __name__ == "__main__":
    main()
