"""Example 41: Comprehensive GPU Performance Benchmark.

Measures single-GPU throughput (Mcells/s), peak memory usage, gradient
computation overhead, and multi-GPU scaling efficiency.

Sections
--------
1. Single GPU throughput at 7 grid sizes (27K .. 8M cells)
2. Gradient computation throughput (forward vs reverse-mode)
3. Memory usage tracking per grid size
4. Multi-GPU scaling (if multiple GPUs available)

Output: benchmark_results.json + console summary table.

Run on the VESSL cluster for meaningful GPU numbers:
  - RTX 4090 (24 GB): vessl_benchmark_rtx4090.yaml
  - A6000 (48 GB):    vessl_benchmark_a6000.yaml
"""

import time
import json
import platform

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Simulation
from rfx.gpu import device_info


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FREQ_MAX = 10e9
DX = 0.5e-3  # 0.5 mm cell size

SINGLE_GPU_SIZES = [
    (30, 30, 30),     # ~27K cells
    (50, 50, 50),     # ~125K cells
    (80, 80, 80),     # ~512K cells
    (100, 100, 100),  # ~1M cells
    (120, 120, 120),  # ~1.7M cells
    (150, 150, 150),  # ~3.4M cells
    (200, 200, 200),  # ~8M cells
]

GRAD_SIZES = [
    (30, 30, 30),
    (50, 50, 50),
    (80, 80, 80),
    (100, 100, 100),
]

N_STEPS_FORWARD = 500
N_STEPS_GRADIENT = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sim(nx, ny, nz, boundary="pec"):
    """Build a minimal vacuum simulation for benchmarking."""
    domain = (nx * DX, ny * DX, nz * DX)
    sim = Simulation(freq_max=FREQ_MAX, domain=domain, dx=DX, boundary=boundary)
    center = (domain[0] / 2, domain[1] / 2, domain[2] / 2)
    sim.add_source(center, component="ez")
    sim.add_probe(center, component="ez")
    return sim


def _gpu_memory_mb():
    """Return allocated GPU memory in MB (returns None on CPU)."""
    try:
        dev = jax.devices()[0]
        stats = dev.memory_stats()
        if stats is not None:
            return stats.get("peak_bytes_in_use", 0) / 1e6
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# 1. Single GPU throughput
# ---------------------------------------------------------------------------

def benchmark_single_gpu():
    """Throughput (Mcells/s) at increasing grid sizes."""
    print("=" * 64)
    print("  1. Single GPU Throughput")
    print("=" * 64)
    print(f"  {'Grid':>12s}  {'Cells':>10s}  {'Mcells/s':>10s}  "
          f"{'Time (s)':>9s}  {'ms/step':>9s}")
    print("-" * 64)

    results = []
    for nx, ny, nz in SINGLE_GPU_SIZES:
        cells = nx * ny * nz
        try:
            sim = _make_sim(nx, ny, nz)

            # Warmup (JIT compile)
            warmup = sim.run(n_steps=10)
            warmup.time_series.block_until_ready()

            # Timed run
            t0 = time.perf_counter()
            result = sim.run(n_steps=N_STEPS_FORWARD)
            result.time_series.block_until_ready()
            elapsed = time.perf_counter() - t0

            mcells_s = cells * N_STEPS_FORWARD / elapsed / 1e6
            ms_step = elapsed / N_STEPS_FORWARD * 1000
            peak_mem = _gpu_memory_mb()

            results.append({
                "grid": f"{nx}x{ny}x{nz}",
                "cells": cells,
                "n_steps": N_STEPS_FORWARD,
                "mcells_per_s": round(mcells_s, 1),
                "elapsed_s": round(elapsed, 4),
                "ms_per_step": round(ms_step, 3),
                "peak_mem_mb": round(peak_mem, 1) if peak_mem else None,
            })
            mem_str = f"  {peak_mem:.0f}MB" if peak_mem else ""
            print(f"  {nx:>3d}x{ny:>3d}x{nz:>3d}  {cells:>10,}  "
                  f"{mcells_s:>10.1f}  {elapsed:>9.3f}  {ms_step:>9.3f}{mem_str}")

        except Exception as e:
            print(f"  {nx:>3d}x{ny:>3d}x{nz:>3d}  {cells:>10,}  "
                  f"FAILED ({type(e).__name__}: {e})")
            break  # Likely OOM -- skip larger sizes

    return results


# ---------------------------------------------------------------------------
# 2. Gradient computation
# ---------------------------------------------------------------------------

def benchmark_gradient():
    """Compare forward vs gradient (reverse-mode AD) throughput."""
    print()
    print("=" * 64)
    print("  2. Gradient Computation Overhead")
    print("=" * 64)
    print(f"  {'Grid':>12s}  {'Fwd (s)':>9s}  {'Grad (s)':>9s}  {'Overhead':>9s}")
    print("-" * 64)

    results = []
    for nx, ny, nz in GRAD_SIZES:
        cells = nx * ny * nz
        try:
            sim = _make_sim(nx, ny, nz)

            # Forward warmup + timed
            sim.run(n_steps=10).time_series.block_until_ready()
            t0 = time.perf_counter()
            sim.run(n_steps=N_STEPS_GRADIENT).time_series.block_until_ready()
            t_fwd = time.perf_counter() - t0

            # Gradient via jax.grad on the Simulation low-level path
            from rfx.grid import Grid
            from rfx.core.yee import MaterialArrays, init_materials
            from rfx.simulation import run as sim_run, make_source, make_probe
            from rfx.sources.sources import GaussianPulse

            domain = (nx * DX, ny * DX, nz * DX)
            grid = Grid(freq_max=FREQ_MAX, domain=domain, dx=DX, cpml_layers=0)
            pulse = GaussianPulse(f0=5e9, bandwidth=0.5)
            center = (domain[0] / 2, domain[1] / 2, domain[2] / 2)
            src = make_source(grid, center, "ez", pulse, N_STEPS_GRADIENT)
            prb = make_probe(grid, center, "ez")
            sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
            mu_r = jnp.ones(grid.shape, dtype=jnp.float32)

            def objective(eps_r):
                mats = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
                res = sim_run(grid, mats, N_STEPS_GRADIENT,
                              sources=[src], probes=[prb],
                              boundary="pec", checkpoint=True)
                return jnp.sum(res.time_series ** 2)

            eps_r = jnp.ones(grid.shape, dtype=jnp.float32)

            # Gradient warmup
            _ = jax.grad(objective)(eps_r)
            jax.block_until_ready(_)

            # Timed gradient
            t0 = time.perf_counter()
            grad = jax.grad(objective)(eps_r)
            jax.block_until_ready(grad)
            t_grad = time.perf_counter() - t0

            overhead = t_grad / t_fwd
            results.append({
                "grid": f"{nx}x{ny}x{nz}",
                "cells": cells,
                "fwd_s": round(t_fwd, 4),
                "grad_s": round(t_grad, 4),
                "overhead_x": round(overhead, 2),
            })
            print(f"  {nx:>3d}x{ny:>3d}x{nz:>3d}  {t_fwd:>9.3f}  "
                  f"{t_grad:>9.3f}  {overhead:>8.1f}x")

        except Exception as e:
            print(f"  {nx:>3d}x{ny:>3d}x{nz:>3d}  FAILED ({type(e).__name__}: {e})")
            break

    return results


# ---------------------------------------------------------------------------
# 3. Multi-GPU scaling
# ---------------------------------------------------------------------------

def benchmark_multi_gpu():
    """Multi-GPU scaling via 1D slab decomposition."""
    all_devices = jax.devices()
    n_available = len(all_devices)

    print()
    print("=" * 64)
    print("  3. Multi-GPU Scaling")
    print("=" * 64)

    if n_available < 2:
        print("  Single GPU detected -- skipping multi-GPU benchmark")
        return []

    print(f"  Available: {n_available} x {all_devices[0].device_kind}")
    print()

    # Use nx divisible by 1,2,3 (for up to 3xA6000)
    nx = 120  # divisible by 1,2,3,4,6
    ny, nz = 60, 60
    cells = nx * ny * nz  # 432,000 cells
    n_steps = 300

    results = []
    device_counts = [n for n in [1, 2, 3] if n <= n_available]

    print(f"  Grid: {nx}x{ny}x{nz} ({cells:,} cells), {n_steps} steps")
    print(f"  {'Devices':>8s}  {'Mcells/s':>10s}  {'Time (s)':>9s}  "
          f"{'Speedup':>8s}  {'Efficiency':>10s}")
    print("-" * 64)

    for n_dev in device_counts:
        sim = Simulation(
            freq_max=FREQ_MAX,
            domain=(nx * DX, ny * DX, nz * DX),
            dx=DX,
            boundary="pec",
        )
        sim.add_source((nx * DX / 2, ny * DX / 2, nz * DX / 2), component="ez")
        sim.add_probe((nx * DX / 2, ny * DX / 2, nz * DX / 2), component="ez")

        devs = all_devices[:n_dev] if n_dev > 1 else None

        # Warmup
        warmup = sim.run(n_steps=5, devices=devs)
        warmup.time_series.block_until_ready()

        # Timed
        t0 = time.perf_counter()
        result = sim.run(n_steps=n_steps, devices=devs)
        result.time_series.block_until_ready()
        elapsed = time.perf_counter() - t0

        mcells_s = cells * n_steps / elapsed / 1e6
        results.append({
            "n_devices": n_dev,
            "mcells_per_s": round(mcells_s, 1),
            "elapsed_s": round(elapsed, 4),
        })

    # Compute speedup and efficiency
    base_elapsed = results[0]["elapsed_s"]
    for r in results:
        speedup = base_elapsed / r["elapsed_s"]
        efficiency = speedup / r["n_devices"] * 100
        r["speedup"] = round(speedup, 2)
        r["efficiency_pct"] = round(efficiency, 1)
        n = r["n_devices"]
        print(f"  {n:>7d}   {r['mcells_per_s']:>10.1f}  {r['elapsed_s']:>9.3f}  "
              f"{speedup:>7.2f}x  {efficiency:>9.1f}%")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    info = device_info()
    print("rfx GPU Performance Benchmark")
    print(f"  Backend   : {info.backend}")
    print(f"  Devices   : {info.devices}")
    print(f"  Host      : {platform.node()}")
    print()

    # Print GPU memory limit if available
    try:
        dev = jax.devices()[0]
        stats = dev.memory_stats()
        if stats and "bytes_limit" in stats:
            print(f"  GPU Memory: {stats['bytes_limit'] / 1e9:.1f} GB")
            print()
    except Exception:
        pass

    single = benchmark_single_gpu()
    gradient = benchmark_gradient()
    multi = benchmark_multi_gpu()

    # Peak throughput
    if single:
        peak = max(single, key=lambda r: r["mcells_per_s"])
        print()
        print(f"  Peak throughput: {peak['mcells_per_s']:.1f} Mcells/s "
              f"at {peak['grid']}")

    # Save results
    output = {
        "backend": info.backend,
        "devices": info.devices,
        "host": platform.node(),
        "single_gpu": single,
        "gradient": gradient,
        "multi_gpu": multi,
    }
    out_path = "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
