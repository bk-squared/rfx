"""Tests for multi-GPU distributed FDTD runner.

Uses XLA_FLAGS to simulate 4 CPU devices for testing.
Must be set BEFORE importing JAX.
"""

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation, GaussianPulse
from rfx.runners.distributed import (
    split_array_x,
    gather_array_x,
    _split_state,
    _gather_state,
    _split_materials,
)
from rfx.core.yee import init_state, init_materials


class TestSplitGather:
    """Test domain splitting and gathering operations."""

    def test_split_and_gather_identity(self):
        """Split then gather should recover original fields (interior)."""
        nx, ny, nz = 20, 8, 8
        arr = jnp.arange(nx * ny * nz, dtype=jnp.float32).reshape(nx, ny, nz)
        n_devices = 4

        slabs = split_array_x(arr, n_devices, ghost=1)
        # Each slab should have nx_per + 2*ghost = 5+2 = 7 cells in x
        assert slabs.shape == (4, 7, 8, 8), f"Got shape {slabs.shape}"

        recovered = gather_array_x(slabs, ghost=1)
        assert recovered.shape == (nx, ny, nz)
        np.testing.assert_allclose(recovered, arr, atol=1e-6)

    def test_split_preserves_ghost_content(self):
        """Ghost cells should contain neighbor data."""
        nx, ny, nz = 12, 4, 4
        arr = jnp.arange(nx * ny * nz, dtype=jnp.float32).reshape(nx, ny, nz)
        n_devices = 4

        slabs = split_array_x(arr, n_devices, ghost=1)

        # Device 1's left ghost should be device 0's last real cell
        # Device 1 owns x=[3,4,5], ghost_lo=2, ghost_hi=6
        # slab[1][0] should be arr[2] (the cell just before device 1's region)
        np.testing.assert_allclose(slabs[1, 0, :, :], arr[2, :, :])

        # Device 1's right ghost should be arr[6]
        np.testing.assert_allclose(slabs[1, -1, :, :], arr[6, :, :])

    def test_split_boundary_ghost_zero_padded(self):
        """Device 0's left ghost and device N-1's right ghost should be zero."""
        nx, ny, nz = 8, 4, 4
        arr = jnp.ones((nx, ny, nz), dtype=jnp.float32)
        n_devices = 4

        slabs = split_array_x(arr, n_devices, ghost=1)

        # Device 0's left ghost should be zero (physical boundary)
        np.testing.assert_allclose(slabs[0, 0, :, :], 0.0)

        # Device 3's right ghost should be zero (physical boundary)
        np.testing.assert_allclose(slabs[3, -1, :, :], 0.0)

    def test_state_split_gather_roundtrip(self):
        """FDTDState split->gather roundtrip preserves fields."""
        shape = (16, 6, 6)
        state = init_state(shape)
        # Put non-trivial data in fields
        state = state._replace(
            ex=jnp.ones(shape) * 1.0,
            hy=jnp.ones(shape) * 2.0,
            ez=jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape),
        )

        slabs = _split_state(state, 4, ghost=1)
        recovered = _gather_state(slabs, ghost=1)

        np.testing.assert_allclose(recovered.ex, state.ex, atol=1e-6)
        np.testing.assert_allclose(recovered.hy, state.hy, atol=1e-6)
        np.testing.assert_allclose(recovered.ez, state.ez, atol=1e-6)


class TestDistributedRunner:
    """Integration tests for the distributed FDTD runner."""

    def test_distributed_matches_single_pec(self):
        """Distributed 4-device PEC result should match single-device."""
        # Lx=0.05 -> nx=12, divisible by 4
        sim = Simulation(
            freq_max=3e9,
            domain=(0.05, 0.02, 0.02),
            boundary="pec",
        )
        sim.add_source(position=(0.025, 0.01, 0.01), component="ez")
        sim.add_probe(position=(0.015, 0.01, 0.01), component="ez")

        n_steps = 100

        # Single device
        result_single = sim.run(n_steps=n_steps)

        # Multi device (4 virtual CPUs)
        devices = jax.devices()[:4]
        assert len(devices) == 4, f"Expected 4 devices, got {len(devices)}"
        result_multi = sim.run(n_steps=n_steps, devices=devices)

        ts_single = np.array(result_single.time_series)
        ts_multi = np.array(result_multi.time_series)

        assert ts_single.shape == ts_multi.shape, (
            f"Shape mismatch: single={ts_single.shape}, multi={ts_multi.shape}"
        )

        peak = np.max(np.abs(ts_single)) + 1e-30
        rel_err = np.max(np.abs(ts_single - ts_multi)) / peak
        assert rel_err < 1e-4, f"Distributed vs single relative error {rel_err:.2e}"

    def test_distributed_source_at_center(self):
        """Source at domain center should produce non-zero probe signal."""
        # Lx=0.03 -> nx=8, divisible by 4
        sim = Simulation(
            freq_max=3e9,
            domain=(0.03, 0.02, 0.02),
            boundary="pec",
        )
        sim.add_source(position=(0.015, 0.01, 0.01), component="ez")
        sim.add_probe(position=(0.015, 0.01, 0.01), component="ez")

        devices = jax.devices()[:4]
        result = sim.run(n_steps=50, devices=devices)
        ts = np.array(result.time_series).ravel()

        assert np.max(np.abs(ts)) > 0, "Probe signal is zero — source not injected"

    def test_distributed_pec_symmetry(self):
        """Symmetric PEC cavity with center source should produce
        symmetric probe readings at equidistant points."""
        # Lx=0.05 -> nx=12, divisible by 4
        sim = Simulation(
            freq_max=3e9,
            domain=(0.05, 0.02, 0.02),
            boundary="pec",
        )
        # Source at center
        sim.add_source(position=(0.025, 0.01, 0.01), component="ez")
        # Two symmetric probes along x
        sim.add_probe(position=(0.01, 0.01, 0.01), component="ez")
        sim.add_probe(position=(0.04, 0.01, 0.01), component="ez")

        devices = jax.devices()[:4]
        result = sim.run(n_steps=100, devices=devices)

        ts = np.array(result.time_series)
        probe_left = ts[:, 0]
        probe_right = ts[:, 1]

        # Due to PEC symmetry, probes at equal distances from center
        # should see the same signal.  Allow ~1% tolerance for
        # floating-point ordering differences from ghost exchange.
        rel_diff = np.max(np.abs(probe_left - probe_right)) / (
            np.max(np.abs(probe_left)) + 1e-30
        )
        assert rel_diff < 0.01, f"Symmetry broken: rel_diff={rel_diff:.2e}"

    def test_distributed_2_devices(self):
        """Should also work with 2 devices."""
        # Lx=0.05 -> nx=12, divisible by 2
        sim = Simulation(
            freq_max=3e9,
            domain=(0.05, 0.02, 0.02),
            boundary="pec",
        )
        sim.add_source(position=(0.025, 0.01, 0.01), component="ez")
        sim.add_probe(position=(0.025, 0.01, 0.01), component="ez")

        n_steps = 50

        result_single = sim.run(n_steps=n_steps)
        devices = jax.devices()[:2]
        result_multi = sim.run(n_steps=n_steps, devices=devices)

        ts_single = np.array(result_single.time_series)
        ts_multi = np.array(result_multi.time_series)

        peak = np.max(np.abs(ts_single)) + 1e-30
        rel_err = np.max(np.abs(ts_single - ts_multi)) / peak
        assert rel_err < 1e-4, f"2-device vs single error {rel_err:.2e}"

    def test_distributed_no_probe(self):
        """Distributed run with no probes should not crash."""
        # Lx=0.03 -> nx=8, divisible by 4
        sim = Simulation(
            freq_max=3e9,
            domain=(0.03, 0.02, 0.02),
            boundary="pec",
        )
        sim.add_source(position=(0.015, 0.01, 0.01), component="ez")

        devices = jax.devices()[:4]
        result = sim.run(n_steps=20, devices=devices)
        assert result.time_series.shape[0] == 20

    def test_nx_not_divisible_raises(self):
        """Should raise ValueError when nx is not divisible by n_devices."""
        # Lx=0.04 -> nx=10, not divisible by 4
        sim = Simulation(
            freq_max=3e9,
            domain=(0.04, 0.02, 0.02),
            boundary="pec",
        )
        sim.add_source(position=(0.02, 0.01, 0.01), component="ez")

        grid = sim._build_grid()
        assert grid.shape[0] % 4 != 0, "Expected nx not divisible by 4"
        devices = jax.devices()[:4]
        with pytest.raises(ValueError, match="not evenly divisible"):
            sim.run(n_steps=10, devices=devices)


class TestDistributedCPML:
    """Tests for CPML boundary support in the distributed runner."""

    def test_distributed_cpml_matches_single(self):
        """Distributed CPML should match single-device CPML."""
        # Lx=0.13 -> nx=48 with cpml_layers=10, divisible by 4
        sim = Simulation(
            freq_max=3e9,
            domain=(0.13, 0.04, 0.04),
            boundary="cpml",
        )
        sim.add_source(
            position=(0.065, 0.02, 0.02),
            component="ez",
            waveform=GaussianPulse(f0=1.5e9, bandwidth=1.5e9),
        )
        sim.add_probe(position=(0.065, 0.02, 0.02), component="ez")

        n_steps = 200

        result_single = sim.run(n_steps=n_steps)

        devices = jax.devices()[:4]
        assert len(devices) == 4, f"Expected 4 devices, got {len(devices)}"
        result_multi = sim.run(n_steps=n_steps, devices=devices)

        ts_s = np.array(result_single.time_series)
        ts_m = np.array(result_multi.time_series)

        assert ts_s.shape == ts_m.shape, (
            f"Shape mismatch: single={ts_s.shape}, multi={ts_m.shape}"
        )

        peak = np.max(np.abs(ts_s)) + 1e-30
        rel_err = np.max(np.abs(ts_s - ts_m)) / peak
        assert rel_err < 1e-3, f"CPML distributed error {rel_err:.2e}"

    def test_distributed_cpml_absorbs(self):
        """CPML in distributed mode should absorb outgoing waves."""
        sim = Simulation(
            freq_max=3e9,
            domain=(0.13, 0.04, 0.04),
            boundary="cpml",
        )
        sim.add_source(
            position=(0.065, 0.02, 0.02),
            component="ez",
            waveform=GaussianPulse(f0=1.5e9, bandwidth=1.5e9),
        )
        sim.add_probe(position=(0.065, 0.02, 0.02), component="ez")

        devices = jax.devices()[:4]
        result = sim.run(n_steps=500, devices=devices)
        ts = np.array(result.time_series).ravel()

        # Late-time fields should be small (absorbed by CPML)
        peak = np.max(np.abs(ts[:len(ts) // 2]))
        tail = np.max(np.abs(ts[int(0.8 * len(ts)):]))
        assert tail < 0.1 * peak, f"CPML not absorbing: tail/peak = {tail / peak:.3f}"

    def test_distributed_cpml_2_devices(self):
        """CPML should also work with 2 devices."""
        # nx=48 is also divisible by 2
        sim = Simulation(
            freq_max=3e9,
            domain=(0.13, 0.04, 0.04),
            boundary="cpml",
        )
        sim.add_source(
            position=(0.065, 0.02, 0.02),
            component="ez",
            waveform=GaussianPulse(f0=1.5e9, bandwidth=1.5e9),
        )
        sim.add_probe(position=(0.065, 0.02, 0.02), component="ez")

        n_steps = 100

        result_single = sim.run(n_steps=n_steps)
        devices = jax.devices()[:2]
        result_multi = sim.run(n_steps=n_steps, devices=devices)

        ts_s = np.array(result_single.time_series)
        ts_m = np.array(result_multi.time_series)

        peak = np.max(np.abs(ts_s)) + 1e-30
        rel_err = np.max(np.abs(ts_s - ts_m)) / peak
        assert rel_err < 1e-3, f"2-device CPML error {rel_err:.2e}"

    def test_distributed_cpml_off_center_probe(self):
        """Probe at a different position should still produce valid signal."""
        sim = Simulation(
            freq_max=3e9,
            domain=(0.13, 0.04, 0.04),
            boundary="cpml",
        )
        sim.add_source(
            position=(0.065, 0.02, 0.02),
            component="ez",
            waveform=GaussianPulse(f0=1.5e9, bandwidth=1.5e9),
        )
        # Probe away from source
        sim.add_probe(position=(0.09, 0.02, 0.02), component="ez")

        n_steps = 150

        result_single = sim.run(n_steps=n_steps)
        devices = jax.devices()[:4]
        result_multi = sim.run(n_steps=n_steps, devices=devices)

        ts_s = np.array(result_single.time_series)
        ts_m = np.array(result_multi.time_series)

        peak = np.max(np.abs(ts_s)) + 1e-30
        rel_err = np.max(np.abs(ts_s - ts_m)) / peak
        assert rel_err < 1e-3, f"Off-center probe CPML error {rel_err:.2e}"
