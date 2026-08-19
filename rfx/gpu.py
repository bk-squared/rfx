"""GPU acceleration utilities.

JAX runs on a supported GPU transparently when the corresponding backend
plugin is installed.  This module provides device information, benchmarking,
and helpers for batch parameter sweeps via ``jax.vmap``.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import jax

from rfx.backends import (
    current_backend,
    is_accelerator_platform,
    is_gpu_platform,
    normalize_platform,
)
from rfx.grid import Grid
from rfx.core.yee import init_materials
from rfx.sources.sources import GaussianPulse
from rfx.simulation import make_source, make_probe, run


# ---------------------------------------------------------------------------
# Device info
# ---------------------------------------------------------------------------

class DeviceInfo(NamedTuple):
    """JAX device summary."""
    backend: str
    gpu_available: bool
    devices: list[dict]

    @property
    def metal_available(self) -> bool:
        """Whether an Apple Metal device is present.

        This is a property rather than a NamedTuple field so the historical
        three-item tuple contract remains unchanged.
        """
        return self.backend == "metal" or any(
            normalize_platform(device.get("platform")) == "metal"
            for device in self.devices
        )

    @property
    def accelerator_available(self) -> bool:
        """Whether any known GPU or TPU accelerator is present."""
        return is_accelerator_platform(self.backend) or any(
            is_accelerator_platform(device.get("platform"))
            for device in self.devices
        )


def device_info() -> DeviceInfo:
    """Report available JAX compute devices."""
    devices = jax.devices()
    backend = current_backend()
    return DeviceInfo(
        backend=backend,
        gpu_available=(
            is_gpu_platform(backend)
            or any(is_gpu_platform(d) for d in devices)
        ),
        devices=[
            {
                "id": d.id,
                "kind": d.device_kind,
                "platform": normalize_platform(d),
            }
            for d in devices
        ],
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class BenchResult(NamedTuple):
    """Single benchmark measurement."""
    shape: tuple[int, int, int]
    n_steps: int
    time_s: float
    mcells_per_s: float
    backend: str


def benchmark(
    grid_sizes: list[tuple[int, int, int]] | None = None,
    n_steps: int = 100,
) -> list[BenchResult]:
    """Benchmark FDTD throughput across grid sizes.

    Returns a list of BenchResult sorted by grid size.
    """
    if grid_sizes is None:
        grid_sizes = [(20, 20, 20), (40, 40, 40), (80, 80, 80)]

    backend = current_backend()
    results = []

    for shape in grid_sizes:
        dx = 0.001
        domain = tuple(s * dx for s in shape)
        grid = Grid(freq_max=10e9, domain=domain, dx=dx, cpml_layers=0)
        materials = init_materials(grid.shape)

        pulse = GaussianPulse(f0=5e9)
        center = tuple(s * dx / 2 for s in shape)
        # issue #571: benchmark drive is explicitly a raw field add —
        # 'field' is make_source's native kind (bit-identical no-op), and
        # being explicit insulates the benchmark from the deprecation-window
        # default changes.
        src = make_source(grid, center, "ez", pulse, n_steps,
                          materials=materials, amplitude_kind="field")
        prb = make_probe(grid, center, "ez")

        # Warm up JIT (separate short source to match step count)
        warmup_src = make_source(grid, center, "ez", pulse, 10,
                                 materials=materials, amplitude_kind="field")
        _ = run(grid, materials, 10, sources=[warmup_src], probes=[prb])

        # Timed run
        t0 = time.perf_counter()
        result = run(grid, materials, n_steps, sources=[src], probes=[prb])
        result.time_series.block_until_ready()
        elapsed = time.perf_counter() - t0

        total_cells = grid.shape[0] * grid.shape[1] * grid.shape[2] * n_steps
        results.append(BenchResult(
            shape=grid.shape,
            n_steps=n_steps,
            time_s=elapsed,
            mcells_per_s=total_cells / elapsed / 1e6,
            backend=backend,
        ))

    return results


def print_benchmark(results: list[BenchResult]) -> None:
    """Pretty-print benchmark results."""
    print(f"\nrfx FDTD Benchmark ({results[0].backend})")
    print(f"{'Grid':>20s}  {'Steps':>6s}  {'Time (s)':>10s}  {'Mcells/s':>10s}")
    print("-" * 55)
    for r in results:
        print(f"{str(r.shape):>20s}  {r.n_steps:>6d}  {r.time_s:>10.3f}  {r.mcells_per_s:>10.1f}")
