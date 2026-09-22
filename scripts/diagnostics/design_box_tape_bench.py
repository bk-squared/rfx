"""Peak device memory and value_and_grad time: `eps_override` vs `design_box` (#1179).

One formulation per process, so `memory_stats()['peak_bytes_in_use']` belongs to
that formulation alone. The driver YAML loops over the arms.

    python scripts/diagnostics/design_box_tape_bench.py --arm box --cells 60 --steps 1000 --out DIR

Arms: `override` (eps_override, checkpoint=False), `override_remat` (eps_override +
checkpoint=True, checkpoint_segments=K), `box` (design_box, checkpoint=False).
Writes DIR/<arm>_<cells>_<steps>.json. An arm that runs out of device memory
records that as its result instead of dying without a record.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

from rfx import GaussianPulse, Simulation

F0 = 8e9
DX = 1e-3
CPML = 8


def build(cells: int):
    """Cube of `cells` interior cells per axis (2/3 on z), a box of cells//5 per axis at the centre."""
    n_xy, n_z = cells, max(8, (2 * cells) // 3)
    domain = (n_xy * DX, n_xy * DX, n_z * DX)
    sim = Simulation(freq_max=2 * F0, domain=domain, dx=DX, boundary="cpml", cpml_layers=CPML)
    c = lambda n: (n // 2 + 0.5) * DX
    sim.add_source((c(n_xy // 4), c(n_xy), c(n_z)), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    sim.add_probe((c(3 * n_xy // 4), c(n_xy), c(n_z)), "ez")
    b = max(2, cells // 5)
    lo = (c(n_xy // 2 - b // 2), c(n_xy // 2 - b // 2), c(n_z // 2 - b // 4))
    hi = (c(n_xy // 2 - b // 2 + b - 1), c(n_xy // 2 - b // 2 + b - 1), c(n_z // 2 - b // 4 + max(1, b // 2) - 1))
    return sim, (lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["override", "override_remat", "box"])
    ap.add_argument("--cells", type=int, default=60)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--segments", type=int, default=0, help="remat segments; 0 = sqrt(steps)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"arm": a.arm, "cells": a.cells, "steps": a.steps, "backend": jax.default_backend(),
           "device": str(jax.devices()[0]), "jax": jax.__version__}

    sim, (lo, hi) = build(a.cells)
    grid = sim._build_grid()
    ilo, ihi = grid.position_to_index(lo), grid.position_to_index(hi)
    sl = tuple(slice(ilo[d], ihi[d] + 1) for d in range(3))
    shape = tuple(ihi[d] - ilo[d] + 1 for d in range(3))
    rec["grid_shape"] = [int(v) for v in grid.shape]
    rec["grid_cells"] = int(np.prod(grid.shape))
    rec["box_shape"] = list(shape)
    eps_design = jnp.asarray(2.0 + 2.0 * np.random.default_rng(1).random(shape), jnp.float32)
    seg = a.segments or max(1, int(math.isqrt(a.steps)))
    while a.steps % seg:
        seg -= 1
    rec["segments"] = seg if a.arm == "override_remat" else None

    if a.arm == "box":
        def loss(e):
            r = sim.forward(design_box=(lo, hi), design_eps_override=e, n_steps=a.steps,
                            checkpoint=False, skip_preflight=True)
            return jnp.sum(r.time_series ** 2)
    else:
        base = sim._assemble_materials(grid)[0].eps_r
        kw = {"checkpoint": False} if a.arm == "override" else {"checkpoint": True, "checkpoint_segments": seg}

        def loss(e):
            full = base.at[sl].set(e)
            r = sim.forward(eps_override=full, n_steps=a.steps, skip_preflight=True, **kw)
            return jnp.sum(r.time_series ** 2)

    vg = jax.jit(jax.value_and_grad(loss))
    try:
        t0 = time.time()
        v, g = vg(eps_design)
        v.block_until_ready()
        rec["first_call_s"] = time.time() - t0
        ts = []
        for _ in range(3):
            t0 = time.time()
            v, g = vg(eps_design)
            v.block_until_ready()
            ts.append(time.time() - t0)
        rec["value_and_grad_s"] = min(ts)
        rec["value"] = float(v)
        rec["grad_norm"] = float(jnp.linalg.norm(g))
        rec["grad_sum"] = float(jnp.sum(g))
        ms = jax.devices()[0].memory_stats() or {}
        rec["peak_bytes_in_use"] = int(ms.get("peak_bytes_in_use", -1))
        rec["peak_GB"] = rec["peak_bytes_in_use"] / 1e9
        rec["status"] = "ok"
    except Exception as ex:  # an OOM is a result, not a crash
        rec["status"] = "error"
        rec["error"] = f"{type(ex).__name__}: {str(ex)[:600]}"
    fn = out / f"{a.arm}_{a.cells}_{a.steps}.json"
    fn.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "error"}), flush=True)
    if rec["status"] != "ok":
        print("ERROR:", rec["error"], flush=True)


if __name__ == "__main__":
    main()
