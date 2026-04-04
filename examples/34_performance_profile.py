"""Profile rfx core FDTD performance — throughput, compile time, memory.

Measures Mcells/s at various grid sizes for:
1. Low-level per-step Python loop (update_h + update_e)
2. Compiled scan runner (sim_run with PEC)
3. GPU fast path (update_he_fast with pre-baked PEC coefficients)

The GPU fast path bakes PEC boundary enforcement into the update
coefficients, eliminating 12 per-step scatter operations.  This is
always beneficial on GPU where scatter ops launch separate kernels.
On CPU, XLA fuses scatter ops efficiently so the standard path is used.
"""

import time
import sys
import os

# Ensure rfx is importable from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np

from rfx.grid import Grid
from rfx.core.yee import (
    FDTDState, MaterialArrays, init_state, init_materials,
    update_h, update_e, EPS_0, MU_0,
    UpdateCoeffs, precompute_coeffs, update_he_fast,
)
from rfx.boundaries.pec import apply_pec
from rfx.simulation import run as sim_run


def _bench_scan(fn, state0, n_steps, cells, warmup=3, trials=5):
    """Benchmark a scan function, return median Mcells/s."""
    for _ in range(warmup):
        f, _ = jax.lax.scan(fn, state0, None, length=n_steps)
        f.ex.block_until_ready()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        f, _ = jax.lax.scan(fn, state0, None, length=n_steps)
        f.ex.block_until_ready()
        times.append(time.perf_counter() - t0)
    return cells * n_steps / 1e6 / np.median(times)


def profile_compile_time():
    """Measure JIT compilation time for the scan-based runner."""
    print("=== Compile Time Profile ===")
    sizes = [(20, 20, 20), (40, 40, 40), (60, 60, 60)]
    for nx, ny, nz in sizes:
        grid = Grid(freq_max=5e9, domain=(nx * 1e-3, ny * 1e-3, nz * 1e-3),
                    dx=1e-3, cpml_layers=0)
        mats = init_materials(grid.shape)
        n_steps = 10

        t0 = time.perf_counter()
        result = sim_run(grid, mats, n_steps, boundary="pec")
        result.state.ex.block_until_ready()
        compile_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        result2 = sim_run(grid, mats, n_steps, boundary="pec")
        result2.state.ex.block_until_ready()
        cached_time = time.perf_counter() - t1

        cells = grid.nx * grid.ny * grid.nz
        print(f"  {grid.nx}x{grid.ny}x{grid.nz} ({cells:>8,} cells): "
              f"compile={compile_time:.2f}s, cached={cached_time:.4f}s")


def profile_throughput_lowlevel():
    """Measure raw Yee update throughput (Python per-step loop)."""
    print("\n=== Low-Level Yee Update Throughput (Python loop) ===")
    sizes = [(20, 20, 20), (40, 40, 40), (60, 60, 60), (80, 80, 80),
             (100, 100, 100)]

    for nx, ny, nz in sizes:
        shape = (nx, ny, nz)
        state = init_state(shape)
        mats = init_materials(shape)
        dt = 1e-3 / (3e8 * np.sqrt(3.0)) * 0.99
        dx = 1e-3

        # Warmup
        state = update_h(state, mats, dt, dx)
        state = update_e(state, mats, dt, dx)
        state.ex.block_until_ready()

        n_iters = 200
        t0 = time.perf_counter()
        for _ in range(n_iters):
            state = update_h(state, mats, dt, dx)
            state = update_e(state, mats, dt, dx)
        state.ex.block_until_ready()
        elapsed = time.perf_counter() - t0

        cells = nx * ny * nz
        mcells = cells * n_iters / 1e6
        print(f"  {nx}x{ny}x{nz} ({cells:>8,} cells): "
              f"{mcells / elapsed:>8.1f} Mcells/s, {elapsed:.3f}s for {n_iters} steps")


def profile_throughput_scan():
    """Measure throughput via the compiled scan runner (sim_run)."""
    print("\n=== Scan Runner Throughput (sim_run, PEC) ===")
    sizes = [(20, 20, 20), (40, 40, 40), (60, 60, 60), (80, 80, 80),
             (100, 100, 100)]

    for nx, ny, nz in sizes:
        grid = Grid(freq_max=5e9, domain=(nx * 1e-3, ny * 1e-3, nz * 1e-3),
                    dx=1e-3, cpml_layers=0)
        mats = init_materials(grid.shape)

        # Warmup (includes compile)
        result = sim_run(grid, mats, 10, boundary="pec")
        result.state.ex.block_until_ready()

        # Benchmark
        n_steps = 500
        t0 = time.perf_counter()
        result = sim_run(grid, mats, n_steps, boundary="pec")
        result.state.ex.block_until_ready()
        elapsed = time.perf_counter() - t0

        cells = grid.nx * grid.ny * grid.nz
        mcells = cells * n_steps / 1e6
        print(f"  {grid.nx}x{grid.ny}x{grid.nz} ({cells:>8,} cells): "
              f"{mcells / elapsed:>8.1f} Mcells/s, {elapsed:.2f}s for {n_steps} steps")


def profile_fast_path():
    """Benchmark the GPU fast path (pre-baked PEC coefficients).

    Compares the standard update_h+update_e+apply_pec pipeline with
    the update_he_fast path that folds PEC into coefficients.

    On GPU, this eliminates 12 scatter-update kernel launches per step.
    On CPU, results are size-dependent (beneficial only for large grids).
    """
    print("\n=== Fast Path: Baked PEC Coefficients ===")
    print("  (update_he_fast eliminates 12 scatter ops / step)")
    sizes = [(60, 60, 60), (80, 80, 80), (100, 100, 100)]
    n_steps = 500

    for nx, ny, nz in sizes:
        grid = Grid(freq_max=5e9, domain=(nx * 1e-3, ny * 1e-3, nz * 1e-3),
                    dx=1e-3, cpml_layers=0)
        mats = init_materials(grid.shape)
        dt, dx = grid.dt, grid.dx
        state0 = init_state(grid.shape)
        cells = grid.nx * grid.ny * grid.nz

        # Standard path
        periodic = (False, False, False)

        def step_standard(st, _):
            st = update_h(st, mats, dt, dx, periodic=periodic)
            st = update_e(st, mats, dt, dx, periodic=periodic)
            st = apply_pec(st, axes="xyz")
            return st, None

        std = _bench_scan(step_standard, state0, n_steps, cells)

        # Fast path with baked PEC
        coeffs = precompute_coeffs(mats, dt, dx, pec_axes="xyz")

        def step_fast(st, _):
            return update_he_fast(st, coeffs), None

        fast = _bench_scan(step_fast, state0, n_steps, cells)

        ratio = fast / std
        tag = "faster" if ratio > 1.05 else ("slower" if ratio < 0.95 else "same")
        print(f"  {grid.nx}x{grid.ny}x{grid.nz} ({cells:>8,} cells): "
              f"std={std:>6.0f}  fast={fast:>6.0f} Mcells/s  "
              f"ratio={ratio:.2f}x ({tag})")


if __name__ == "__main__":
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")
    print()

    profile_compile_time()
    profile_throughput_lowlevel()
    profile_throughput_scan()
    profile_fast_path()
