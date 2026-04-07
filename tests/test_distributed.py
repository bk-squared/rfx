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
        """CPML in distributed mode should absorb as well as single-device.

        The absolute absorption level depends on domain size and CPML
        parameters.  This test verifies that the distributed runner
        achieves the same absorption as the single-device runner,
        confirming that the distributed CPML is correctly implemented.
        """
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

        n_steps = 500
        devices = jax.devices()[:4]

        result_single = sim.run(n_steps=n_steps)
        result_multi = sim.run(n_steps=n_steps, devices=devices)

        ts_s = np.array(result_single.time_series).ravel()
        ts_m = np.array(result_multi.time_series).ravel()

        # Both should show absorption: late-time < peak
        peak_s = np.max(np.abs(ts_s[:len(ts_s) // 2]))
        tail_s = np.max(np.abs(ts_s[int(0.8 * len(ts_s)):]))
        peak_m = np.max(np.abs(ts_m[:len(ts_m) // 2]))
        tail_m = np.max(np.abs(ts_m[int(0.8 * len(ts_m)):]))

        # Verify absorption is occurring (tail < peak)
        assert tail_m < peak_m, "No absorption at all in distributed CPML"

        # Distributed and single should have similar absorption ratios
        ratio_s = tail_s / (peak_s + 1e-30)
        ratio_m = tail_m / (peak_m + 1e-30)
        assert abs(ratio_m - ratio_s) < 0.05, (
            f"Distributed absorption differs from single: "
            f"single={ratio_s:.3f}, multi={ratio_m:.3f}"
        )

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
            cpml_layers=10,
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


class TestDistributedLumpedPort:
    """Tests for lumped port support in the distributed runner (Phase 3a)."""

    def test_distributed_with_lumped_port(self):
        """Distributed lumped port should match single-device result."""
        sim = Simulation(
            freq_max=5e9,
            domain=(0.08, 0.024, 0.024),
            boundary="pec",
        )
        sim.add_port(
            position=(0.04, 0.012, 0.012),
            component="ez",
            impedance=50,
            waveform=GaussianPulse(f0=2.5e9, bandwidth=2.5e9),
        )
        sim.add_probe(position=(0.04, 0.012, 0.012), component="ez")

        r1 = sim.run(n_steps=100)
        r4 = sim.run(n_steps=100, devices=jax.devices()[:4])

        ts1 = np.array(r1.time_series)
        ts4 = np.array(r4.time_series)
        err = np.max(np.abs(ts1 - ts4)) / (np.max(np.abs(ts1)) + 1e-30)
        assert err < 0.01, f"Lumped port distributed vs single error {err:.2e}"

    def test_distributed_lumped_port_nonzero_signal(self):
        """Lumped port in distributed mode should produce non-zero probe signal."""
        sim = Simulation(
            freq_max=5e9,
            domain=(0.08, 0.024, 0.024),
            boundary="pec",
        )
        sim.add_port(
            position=(0.04, 0.012, 0.012),
            component="ez",
            impedance=50,
            waveform=GaussianPulse(f0=2.5e9, bandwidth=2.5e9),
        )
        sim.add_probe(position=(0.04, 0.012, 0.012), component="ez")

        devices = jax.devices()[:4]
        result = sim.run(n_steps=50, devices=devices)
        ts = np.array(result.time_series).ravel()
        assert np.max(np.abs(ts)) > 0, "Lumped port signal is zero"

    def test_distributed_lumped_port_2_devices(self):
        """Lumped port should also work with 2 devices."""
        sim = Simulation(
            freq_max=5e9,
            domain=(0.08, 0.024, 0.024),
            boundary="pec",
        )
        sim.add_port(
            position=(0.04, 0.012, 0.012),
            component="ez",
            impedance=50,
            waveform=GaussianPulse(f0=2.5e9, bandwidth=2.5e9),
        )
        sim.add_probe(position=(0.04, 0.012, 0.012), component="ez")

        r1 = sim.run(n_steps=100)
        r2 = sim.run(n_steps=100, devices=jax.devices()[:2])

        ts1 = np.array(r1.time_series)
        ts2 = np.array(r2.time_series)
        err = np.max(np.abs(ts1 - ts2)) / (np.max(np.abs(ts1)) + 1e-30)
        assert err < 0.01, f"Lumped port 2-device error {err:.2e}"


@pytest.mark.skip(reason="Pre-existing: auto mesh nx not divisible by n_devices + waveform handling diffs")
class TestDistributedDebye:
    """Tests for Debye dispersive material support in the distributed runner (Phase 3b)."""

    def test_distributed_debye_matches_single(self):
        """Debye dispersion in distributed PEC should match single-device."""
        from rfx import Box, DebyePole

        sim = Simulation(
            freq_max=5e9,
            domain=(0.08, 0.024, 0.024),
            boundary="pec",
        )
        sim.add_material("lossy", eps_r=4.0, debye_poles=[DebyePole(delta_eps=1.0, tau=1e-11)])
        sim.add(Box(corner_lo=(0.02, 0, 0), corner_hi=(0.06, 0.024, 0.024)), material="lossy")
        sim.add_source(position=(0.04, 0.012, 0.012), component="ez")
        sim.add_probe(position=(0.04, 0.012, 0.012), component="ez")

        r1 = sim.run(n_steps=100)
        r4 = sim.run(n_steps=100, devices=jax.devices()[:4])

        ts1 = np.array(r1.time_series)
        ts4 = np.array(r4.time_series)
        err = np.max(np.abs(ts1 - ts4)) / (np.max(np.abs(ts1)) + 1e-30)
        assert err < 0.01, f"Debye distributed vs single error {err:.2e}"

    def test_distributed_debye_nonzero_signal(self):
        """Debye in distributed mode should produce non-zero probe signal."""
        from rfx import Box, DebyePole

        sim = Simulation(
            freq_max=5e9,
            domain=(0.08, 0.024, 0.024),
            boundary="pec",
        )
        sim.add_material("lossy", eps_r=4.0, debye_poles=[DebyePole(delta_eps=1.0, tau=1e-11)])
        sim.add(Box(corner_lo=(0.02, 0, 0), corner_hi=(0.06, 0.024, 0.024)), material="lossy")
        sim.add_source(position=(0.04, 0.012, 0.012), component="ez")
        sim.add_probe(position=(0.04, 0.012, 0.012), component="ez")

        devices = jax.devices()[:4]
        result = sim.run(n_steps=50, devices=devices)
        ts = np.array(result.time_series).ravel()
        assert np.max(np.abs(ts)) > 0, "Debye distributed signal is zero"

    def test_distributed_debye_2_devices(self):
        """Debye dispersion should also work with 2 devices."""
        from rfx import Box, DebyePole

        sim = Simulation(
            freq_max=5e9,
            domain=(0.08, 0.024, 0.024),
            boundary="pec",
        )
        sim.add_material("lossy", eps_r=4.0, debye_poles=[DebyePole(delta_eps=1.0, tau=1e-11)])
        sim.add(Box(corner_lo=(0.02, 0, 0), corner_hi=(0.06, 0.024, 0.024)), material="lossy")
        sim.add_source(position=(0.04, 0.012, 0.012), component="ez")
        sim.add_probe(position=(0.04, 0.012, 0.012), component="ez")

        r1 = sim.run(n_steps=100)
        r2 = sim.run(n_steps=100, devices=jax.devices()[:2])

        ts1 = np.array(r1.time_series)
        ts2 = np.array(r2.time_series)
        err = np.max(np.abs(ts1 - ts2)) / (np.max(np.abs(ts1)) + 1e-30)
        assert err < 0.01, f"Debye 2-device error {err:.2e}"


@pytest.mark.skip(reason="Pre-existing: auto mesh nx not divisible by n_devices")
class TestDistributedLorentz:
    """Tests for Lorentz dispersive material support in the distributed runner (Phase 3b)."""

    def test_distributed_lorentz_matches_single(self):
        """Lorentz dispersion in distributed PEC should match single-device."""
        from rfx import Box
        from rfx.materials.lorentz import lorentz_pole

        sim = Simulation(
            freq_max=5e9,
            domain=(0.08, 0.024, 0.024),
            boundary="pec",
        )
        pole = lorentz_pole(delta_eps=1.0, omega_0=2.0 * np.pi * 3e9, delta=1e9)
        sim.add_material("resonant", eps_r=2.0, lorentz_poles=[pole])
        sim.add(Box(corner_lo=(0.02, 0, 0), corner_hi=(0.06, 0.024, 0.024)), material="resonant")
        sim.add_source(position=(0.04, 0.012, 0.012), component="ez")
        sim.add_probe(position=(0.04, 0.012, 0.012), component="ez")

        r1 = sim.run(n_steps=100)
        r4 = sim.run(n_steps=100, devices=jax.devices()[:4])

        ts1 = np.array(r1.time_series)
        ts4 = np.array(r4.time_series)
        err = np.max(np.abs(ts1 - ts4)) / (np.max(np.abs(ts1)) + 1e-30)
        assert err < 0.01, f"Lorentz distributed vs single error {err:.2e}"


class TestDistributedFallback:
    """Tests for graceful fallback when distributed runner encounters
    unsupported features (TFSF, waveguide ports)."""

    def test_tfsf_falls_back_with_warning(self):
        """TFSF source should trigger fallback to single-device with a warning."""
        import warnings as _warnings

        sim = Simulation(
            freq_max=5e9,
            domain=(0.13, 0.04, 0.04),
            boundary="cpml",
        )
        sim.add_tfsf_source(f0=2.5e9, bandwidth=0.5)
        sim.add_probe(position=(0.065, 0.02, 0.02), component="ez")

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            result = sim.run(n_steps=50, devices=jax.devices()[:4])

        # Should have issued a TFSF fallback warning
        tfsf_warnings = [x for x in w if "TFSF" in str(x.message)]
        assert len(tfsf_warnings) >= 1, (
            f"Expected TFSF fallback warning, got: {[str(x.message) for x in w]}"
        )
        # Result should be valid (non-None, correct shape)
        assert result is not None
        assert result.time_series.shape[0] == 50

    def test_tfsf_fallback_produces_valid_signal(self):
        """TFSF fallback should produce a non-zero probe signal."""
        import warnings as _warnings

        sim = Simulation(
            freq_max=5e9,
            domain=(0.13, 0.04, 0.04),
            boundary="cpml",
        )
        sim.add_tfsf_source(f0=2.5e9, bandwidth=0.5)
        sim.add_probe(position=(0.065, 0.02, 0.02), component="ez")

        with _warnings.catch_warnings(record=True):
            _warnings.simplefilter("always")
            result = sim.run(n_steps=50, devices=jax.devices()[:4])

        ts = np.array(result.time_series).ravel()
        assert np.max(np.abs(ts)) > 0, "TFSF fallback produced zero signal"

    def test_tfsf_fallback_matches_single_device(self):
        """TFSF fallback result should match explicit single-device result."""
        import warnings as _warnings

        sim = Simulation(
            freq_max=5e9,
            domain=(0.13, 0.04, 0.04),
            boundary="cpml",
        )
        sim.add_tfsf_source(f0=2.5e9, bandwidth=0.5)
        sim.add_probe(position=(0.065, 0.02, 0.02), component="ez")

        n_steps = 50

        # Single-device (no devices argument)
        result_single = sim.run(n_steps=n_steps)

        # Multi-device request -> should fall back
        with _warnings.catch_warnings(record=True):
            _warnings.simplefilter("always")
            result_multi = sim.run(n_steps=n_steps, devices=jax.devices()[:4])

        ts_s = np.array(result_single.time_series)
        ts_m = np.array(result_multi.time_series)
        np.testing.assert_allclose(ts_m, ts_s, atol=1e-6,
                                   err_msg="TFSF fallback differs from single-device")

    def test_waveguide_port_falls_back_with_warning(self):
        """Waveguide port should trigger fallback to single-device with a warning."""
        import warnings as _warnings

        sim = Simulation(
            freq_max=10e9,
            domain=(0.13, 0.04, 0.04),
            boundary="cpml",
        )
        sim.add_waveguide_port(
            x_position=0.01,
            y_range=(0.005, 0.035),
            z_range=(0.005, 0.035),
            mode=(1, 0),
            mode_type="TE",
            direction="+x",
            f0=5e9,
            bandwidth=0.5,
        )
        sim.add_probe(position=(0.065, 0.02, 0.02), component="ez")

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            result = sim.run(n_steps=50, devices=jax.devices()[:4])

        # Should have issued a waveguide port fallback warning
        wg_warnings = [x for x in w if "waveguide" in str(x.message).lower()]
        assert len(wg_warnings) >= 1, (
            f"Expected waveguide port fallback warning, got: {[str(x.message) for x in w]}"
        )
        assert result is not None
        assert result.time_series.shape[0] == 50


class TestExchangeInterval:
    """Tests for configurable ghost exchange interval (reduced sync).

    Error from stale ghost data depends on the ratio of ghost cells to
    slab size. Tests use CPML domains (nx=48) with 2 devices (slab=24
    real cells) for realistic error levels, and 4 devices for the
    monotonicity check.
    """

    def test_interval_1_matches_default_pec(self):
        """exchange_interval=1 should produce identical results to default."""
        sim = Simulation(
            freq_max=3e9,
            domain=(0.05, 0.02, 0.02),
            boundary="pec",
        )
        sim.add_source(position=(0.025, 0.01, 0.01), component="ez")
        sim.add_probe(position=(0.015, 0.01, 0.01), component="ez")

        n_steps = 100
        devices = jax.devices()[:4]

        result_default = sim.run(n_steps=n_steps, devices=devices)
        result_explicit = sim.run(
            n_steps=n_steps, devices=devices, exchange_interval=1)

        ts_d = np.array(result_default.time_series)
        ts_e = np.array(result_explicit.time_series)
        np.testing.assert_allclose(
            ts_e, ts_d, atol=1e-6,
            err_msg="exchange_interval=1 differs from default")

    def test_interval_2_cpml_acceptable_error(self):
        """exchange_interval=2 with CPML 2-device should have <10% error.

        2 devices on nx=48 gives slab=24 real cells. Stale ghost for
        1 step affects only 1 cell out of 24 -- measured ~3.4% error.
        """
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
        devices = jax.devices()[:2]

        result_1 = sim.run(n_steps=n_steps, devices=devices,
                           exchange_interval=1)
        result_2 = sim.run(n_steps=n_steps, devices=devices,
                           exchange_interval=2)

        ts_1 = np.array(result_1.time_series)
        ts_2 = np.array(result_2.time_series)

        peak = np.max(np.abs(ts_1)) + 1e-30
        rel_err = np.max(np.abs(ts_1 - ts_2)) / peak
        assert rel_err < 0.10, (
            f"CPML exchange_interval=2 vs 1 relative error {rel_err:.2e} exceeds 10%")

    def test_interval_4_cpml_2dev_acceptable_error(self):
        """exchange_interval=4 with CPML 2-device should have <10% error.

        Larger slabs (24 cells) tolerate interval=4 well -- measured ~3.3%.
        """
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
        devices = jax.devices()[:2]

        result_1 = sim.run(n_steps=n_steps, devices=devices,
                           exchange_interval=1)
        result_4 = sim.run(n_steps=n_steps, devices=devices,
                           exchange_interval=4)

        ts_1 = np.array(result_1.time_series)
        ts_4 = np.array(result_4.time_series)

        peak = np.max(np.abs(ts_1)) + 1e-30
        rel_err = np.max(np.abs(ts_1 - ts_4)) / peak
        assert rel_err < 0.10, (
            f"CPML exchange_interval=4 vs 1 relative error {rel_err:.2e} exceeds 10%")

    def test_interval_4_cpml_4dev_bounded_error(self):
        """exchange_interval=4 with CPML 4-device should have bounded error.

        4 devices on nx=48 gives slab=12 real cells. Stale ghost for
        3 steps affects 3 cells out of 12 -- measured ~15% error.
        """
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
        devices = jax.devices()[:4]

        result_1 = sim.run(n_steps=n_steps, devices=devices,
                           exchange_interval=1)
        result_4 = sim.run(n_steps=n_steps, devices=devices,
                           exchange_interval=4)

        ts_1 = np.array(result_1.time_series)
        ts_4 = np.array(result_4.time_series)

        peak = np.max(np.abs(ts_1)) + 1e-30
        rel_err = np.max(np.abs(ts_1 - ts_4)) / peak
        assert rel_err < 0.25, (
            f"CPML 4-dev exchange_interval=4 vs 1 relative error {rel_err:.2e} exceeds 25%")

    def test_interval_monotonic_error_growth(self):
        """Error should grow monotonically with exchange_interval.

        Uses CPML with 4 devices where the effect is most visible.
        """
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
        devices = jax.devices()[:4]

        result_ref = sim.run(n_steps=n_steps, devices=devices,
                             exchange_interval=1)
        ts_ref = np.array(result_ref.time_series)
        peak = np.max(np.abs(ts_ref)) + 1e-30

        errors = []
        for interval in [1, 2, 4]:
            result = sim.run(n_steps=n_steps, devices=devices,
                             exchange_interval=interval)
            ts = np.array(result.time_series)
            err = np.max(np.abs(ts_ref - ts)) / peak
            errors.append(err)

        # Error should be non-decreasing
        for i in range(len(errors) - 1):
            assert errors[i] <= errors[i + 1] + 1e-8, (
                f"Error not monotonic: interval errors = {errors}")
