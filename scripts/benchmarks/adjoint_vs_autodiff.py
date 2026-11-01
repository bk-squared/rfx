"""Wall time and peak memory of a design-permittivity gradient: autodiff vs adjoint (#1424).

One process, one size. Prints one JSON line per mode. Usage:

    python scripts/benchmarks/adjoint_vs_autodiff.py --cells 40 --steps 1200

A dielectric slab is the design region inside a CPML box. A soft Ez pulse drives it
and the objective is the power on an E DFT plane at two bins. Both modes run under
``jax.jit(jax.value_and_grad(...))``. The adjoint returns the settled-spectrum
gradient, so the two gradients agree only when the record has settled; the
``adjoint_settling`` witness is printed with it.
"""
import argparse
import json
import time

import jax
import jax.numpy as jnp

from rfx import Box, GaussianPulse, Simulation


def scene(cells, steps, dtype):
    mm = 1e-3
    side = 30 * mm
    sim = Simulation(freq_max=12e9, domain=(side, side, side), dx=side / cells,
                     boundary="cpml", cpml_layers=8, precision=dtype)
    lo, hi = (10 * mm, 10 * mm, 10 * mm), (20 * mm, 20 * mm, 14 * mm)
    sim.add_material("slab", eps_r=3.0)
    sim.add(Box(lo, hi), material="slab")
    sim.add_source((15 * mm, 15 * mm, 6 * mm), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=6e9, bandwidth=0.8, cutoff=6))
    grid = sim._build_grid()
    # Interior pixels only: the adjoint refuses monitors on CPML cells.
    m0, m1 = grid.position_to_index((10 * mm, 10 * mm, 22 * mm)), grid.position_to_index((20 * mm, 20 * mm, 22 * mm))
    sim.add_dft_plane_probe(axis="z", coordinate=22 * mm, component="ez",
                            freqs=jnp.asarray([5e9, 7e9]), name="out",
                            region=(m0[0], m1[0], m0[1], m1[1]))
    il, ih = grid.position_to_index(lo), grid.position_to_index(hi)
    eps = jnp.full(tuple(int(b - a + 1) for a, b in zip(il, ih)), 3.0, dtype=dtype)

    def loss(e, mode):
        r = sim.forward(design_box=(lo, hi), design_eps_override=e, n_steps=steps,
                        checkpoint_segments=20, gradient=mode, skip_preflight=True)
        settling = getattr(r, "adjoint_settling", None)
        return (jnp.sum(jnp.abs(r.dft_planes["out"].accumulator / r.dt) ** 2),
                jnp.nan if settling is None else settling)
    return loss, eps, int(grid.shape[0] * grid.shape[1] * grid.shape[2])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", type=int, default=40)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--dtype", default="float32")
    a = p.parse_args()
    loss, eps, n_cells = scene(a.cells, a.steps, a.dtype)
    grads = {}
    for mode in ("autodiff", "adjoint"):
        fn = jax.jit(jax.value_and_grad(lambda e: loss(e, mode), has_aux=True))
        t0 = time.perf_counter()
        (value, settling), g = jax.block_until_ready(fn(eps))
        first = time.perf_counter() - t0
        t0 = time.perf_counter()
        jax.block_until_ready(fn(eps))
        warm = time.perf_counter() - t0
        grads[mode] = g
        stats = jax.devices()[0].memory_stats() or {}
        print(json.dumps(dict(mode=mode, cells=n_cells, steps=a.steps, dtype=a.dtype,
                              first_call_s=first, warm_s=warm, objective=float(value),
                              peak_device_bytes=stats.get("peak_bytes_in_use"),
                              adjoint_settling=float(settling))))
    diff = jnp.max(jnp.abs(grads["adjoint"] - grads["autodiff"])) / jnp.max(jnp.abs(grads["autodiff"]))
    print(json.dumps(dict(gradient_difference_over_peak=float(diff))))


if __name__ == "__main__":
    main()
