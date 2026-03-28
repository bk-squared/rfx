"""Tests for GPU utilities.

Validates that:
1. device_info() returns valid structure
2. benchmark() runs and returns timing data
"""

from rfx.gpu import device_info, benchmark, BenchResult, DeviceInfo


def test_device_info():
    """device_info() returns valid DeviceInfo."""
    info = device_info()
    assert isinstance(info, DeviceInfo)
    assert info.backend in ("cpu", "gpu", "tpu")
    assert len(info.devices) > 0
    assert "id" in info.devices[0]
    assert "platform" in info.devices[0]
    print(f"\nBackend: {info.backend}, GPU: {info.gpu_available}, "
          f"Devices: {len(info.devices)}")


def test_benchmark_runs():
    """benchmark() completes and returns valid results."""
    results = benchmark(grid_sizes=[(10, 10, 10)], n_steps=20)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, BenchResult)
    assert r.time_s > 0
    assert r.mcells_per_s > 0
    print(f"\nBenchmark: {r.shape} x {r.n_steps} steps = "
          f"{r.mcells_per_s:.1f} Mcells/s on {r.backend}")
