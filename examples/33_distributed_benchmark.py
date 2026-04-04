"""Multi-GPU distributed FDTD performance benchmark.

Measures scaling efficiency of 1D slab decomposition across multiple
devices.  Uses virtual CPU devices for local testing --- real GPU
numbers should be collected on the VESSL cluster.

Run
---
XLA_FLAGS="--xla_force_host_platform_device_count=8" python examples/33_distributed_benchmark.py
"""

import os
os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=8"
)

import time

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Simulation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FREQ_MAX = 5e9
# Domain chosen so that nx=80 (divisible by 1,2,4,8), ny=40, nz=40
# with dx=1 mm  =>  128,000 cells total
DOMAIN = (0.079, 0.039, 0.039)
DX = 0.001  # 1 mm
N_STEPS = 300
CENTER = tuple(d / 2 for d in DOMAIN)


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def benchmark(label, sim, n_steps, devices=None):
    """Run simulation, measure wall time after JIT warmup."""

    # Warmup (triggers JIT / pmap compilation)
    warmup = sim.run(n_steps=5, devices=devices)
    warmup.time_series.block_until_ready()

    # Timed run
    t0 = time.perf_counter()
    result = sim.run(n_steps=n_steps, devices=devices)
    result.time_series.block_until_ready()
    elapsed = time.perf_counter() - t0

    grid = sim._build_grid()
    total_cells = grid.shape[0] * grid.shape[1] * grid.shape[2]
    mcells = total_cells * n_steps / 1e6

    print(
        f"  {label:30s}: {elapsed:6.2f}s  "
        f"{mcells / elapsed:8.1f} Mcells/s  "
        f"({total_cells:,} cells, {grid.shape})"
    )
    return elapsed, mcells / elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_devices = jax.devices()
    n_devices_list = [1, 2, 4, 8]

    print("=" * 64)
    print("  Distributed FDTD Benchmark (1D slab decomposition)")
    print("=" * 64)
    print(f"  Domain     : {DOMAIN} m")
    print(f"  dx         : {DX * 1e3:.1f} mm")
    print(f"  n_steps    : {N_STEPS}")
    print(f"  Devices    : {len(all_devices)} x {all_devices[0].platform}")
    print()

    results = {}
    for n in n_devices_list:
        if n > len(all_devices):
            print(f"  Skipping {n} devices (only {len(all_devices)} available)")
            continue

        sim = Simulation(
            freq_max=FREQ_MAX, domain=DOMAIN, boundary="pec", dx=DX,
        )
        sim.add_source(CENTER, component="ez")
        sim.add_probe(CENTER, component="ez")

        devs = all_devices[:n] if n > 1 else None
        label = f"{n} device{'s' if n > 1 else ' '}"
        elapsed, perf = benchmark(label, sim, N_STEPS, devices=devs)
        results[n] = {"elapsed": elapsed, "mcells_s": perf}

    # ------------------------------------------------------------------
    # Scaling efficiency table
    # ------------------------------------------------------------------
    if 1 not in results:
        print("\n  (no single-device baseline -- cannot compute scaling)")
        return results

    print()
    print("-" * 64)
    print(f"  {'Devices':>7s}  {'Time (s)':>9s}  {'Mcells/s':>10s}  "
          f"{'Speedup':>8s}  {'Efficiency':>10s}")
    print("-" * 64)

    base = results[1]["elapsed"]
    for n, r in sorted(results.items()):
        speedup = base / r["elapsed"]
        efficiency = speedup / n * 100
        print(
            f"  {n:>7d}  {r['elapsed']:>9.2f}  {r['mcells_s']:>10.1f}  "
            f"{speedup:>7.2f}x  {efficiency:>9.1f}%"
        )

    print("-" * 64)
    print()
    return results


if __name__ == "__main__":
    main()
