"""GPU Benchmark: CPU vs GPU speedup for rfx FDTD.

Measures simulation time and gradient computation time across
multiple grid sizes on CPU and GPU (if available).

Run on VESSL cluster with GPU preset for meaningful results.
"""

import time
import json
import numpy as np
import jax
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import MaterialArrays
from rfx.simulation import run, make_source, make_probe, ProbeSpec
from rfx.sources.sources import GaussianPulse


def benchmark_forward(domain_m, dx, n_steps):
    """Time a forward FDTD simulation."""
    grid = Grid(freq_max=10e9, domain=(domain_m,)*3, dx=dx, cpml_layers=6)
    pulse = GaussianPulse(f0=5e9, bandwidth=0.5)
    src = make_source(grid, tuple(d/2 for d in (domain_m,)*3), "ez", pulse, n_steps)
    probe = ProbeSpec(i=grid.nx//3, j=grid.ny//3, k=grid.nz//3, component="ez")

    eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
    sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
    mu_r = jnp.ones(grid.shape, dtype=jnp.float32)
    mats = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)

    # Warmup (JIT compile) — must use same n_steps as timed run
    _ = run(grid, mats, n_steps, sources=[src], probes=[probe], boundary="pec")
    jax.block_until_ready(_)

    # Timed run
    t0 = time.perf_counter()
    result = run(grid, mats, n_steps, sources=[src], probes=[probe], boundary="pec")
    jax.block_until_ready(result.time_series)
    t1 = time.perf_counter()

    return t1 - t0, grid.shape


def benchmark_gradient(domain_m, dx, n_steps):
    """Time a gradient computation through FDTD."""
    grid = Grid(freq_max=10e9, domain=(domain_m,)*3, dx=dx, cpml_layers=6)
    pulse = GaussianPulse(f0=5e9, bandwidth=0.5)
    src = make_source(grid, tuple(d/2 for d in (domain_m,)*3), "ez", pulse, n_steps)
    probe = ProbeSpec(i=grid.nx//3, j=grid.ny//3, k=grid.nz//3, component="ez")

    sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
    mu_r = jnp.ones(grid.shape, dtype=jnp.float32)

    def objective(eps_r):
        mats = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
        result = run(grid, mats, n_steps, sources=[src], probes=[probe],
                     boundary="pec", checkpoint=True)
        return jnp.sum(result.time_series ** 2)

    eps_r = jnp.ones(grid.shape, dtype=jnp.float32)

    # Warmup
    _ = jax.grad(objective)(eps_r)
    jax.block_until_ready(_)

    # Timed
    t0 = time.perf_counter()
    grad = jax.grad(objective)(eps_r)
    jax.block_until_ready(grad)
    t1 = time.perf_counter()

    return t1 - t0, grid.shape


def main():
    backend = jax.default_backend()
    devices = jax.devices()
    print(f"JAX backend: {backend}")
    print(f"Devices: {devices}")
    print()

    configs = [
        (0.01, 0.001, 200),   # ~22³ grid, 200 steps
        (0.02, 0.001, 300),   # ~32³ grid, 300 steps
        (0.03, 0.001, 400),   # ~42³ grid, 400 steps
        (0.05, 0.001, 500),   # ~62³ grid, 500 steps
    ]

    results = {"backend": backend, "device": str(devices[0]), "benchmarks": []}

    print("=" * 60)
    print("Forward simulation benchmark")
    print("=" * 60)
    for domain_m, dx, n_steps in configs:
        t_fwd, shape = benchmark_forward(domain_m, dx, n_steps)
        cells = shape[0] * shape[1] * shape[2]
        mcells_per_sec = cells * n_steps / t_fwd / 1e6
        print(f"  Grid {shape[0]:3d}³  {n_steps} steps: {t_fwd:.3f}s  "
              f"({mcells_per_sec:.1f} Mcells/s)")
        results["benchmarks"].append({
            "type": "forward",
            "shape": list(shape),
            "n_steps": n_steps,
            "time_s": round(t_fwd, 4),
            "mcells_per_s": round(mcells_per_sec, 1),
        })

    print()
    print("=" * 60)
    print("Gradient computation benchmark")
    print("=" * 60)
    # Use smaller configs for gradient (more memory intensive)
    grad_configs = configs[:3]
    for domain_m, dx, n_steps in grad_configs:
        t_grad, shape = benchmark_gradient(domain_m, dx, n_steps)
        cells = shape[0] * shape[1] * shape[2]
        print(f"  Grid {shape[0]:3d}³  {n_steps} steps: {t_grad:.3f}s")
        results["benchmarks"].append({
            "type": "gradient",
            "shape": list(shape),
            "n_steps": n_steps,
            "time_s": round(t_grad, 4),
        })

    # Save results
    out_path = "examples/gpu_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
