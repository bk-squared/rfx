"""Tests for mixed precision (float16 fields, float32 accumulation) support."""

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.core.yee import init_state, FDTDState, MaterialArrays, init_materials
from rfx.api import Simulation
from rfx.sources.sources import GaussianPulse

pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------------------
# Unit tests for core init_state
# ---------------------------------------------------------------------------

class TestInitState:
    def test_default_dtype_is_float32(self):
        state = init_state((5, 5, 5))
        assert state.ex.dtype == jnp.float32
        assert state.hx.dtype == jnp.float32

    def test_float16_fields(self):
        state = init_state((5, 5, 5), field_dtype=jnp.float16)
        assert state.ex.dtype == jnp.float16
        assert state.ey.dtype == jnp.float16
        assert state.ez.dtype == jnp.float16
        assert state.hx.dtype == jnp.float16
        assert state.hy.dtype == jnp.float16
        assert state.hz.dtype == jnp.float16
        # step counter stays int32 regardless
        assert state.step.dtype == jnp.int32

    def test_float16_memory_half_of_float32(self):
        shape = (50, 50, 50)
        s32 = init_state(shape, field_dtype=jnp.float32)
        s16 = init_state(shape, field_dtype=jnp.float16)
        bytes_32 = sum(f.nbytes for f in [s32.ex, s32.ey, s32.ez,
                                           s32.hx, s32.hy, s32.hz])
        bytes_16 = sum(f.nbytes for f in [s16.ex, s16.ey, s16.ez,
                                           s16.hx, s16.hy, s16.hz])
        assert bytes_16 == bytes_32 // 2


# ---------------------------------------------------------------------------
# Yee update tests with float16 state
# ---------------------------------------------------------------------------

class TestYeeUpdateMixedPrecision:
    def test_update_h_preserves_dtype(self):
        from rfx.core.yee import update_h
        state = init_state((10, 10, 10), field_dtype=jnp.float16)
        materials = init_materials((10, 10, 10))
        # Set a nonzero E field to get nonzero H update
        state = state._replace(ez=state.ez.at[5, 5, 5].set(jnp.float16(1.0)))
        new_state = update_h(state, materials, 1e-12, 1e-3)
        assert new_state.hx.dtype == jnp.float16
        assert new_state.hy.dtype == jnp.float16
        assert new_state.hz.dtype == jnp.float16

    def test_update_e_preserves_dtype(self):
        from rfx.core.yee import update_e
        state = init_state((10, 10, 10), field_dtype=jnp.float16)
        materials = init_materials((10, 10, 10))
        state = state._replace(hz=state.hz.at[5, 5, 5].set(jnp.float16(1.0)))
        new_state = update_e(state, materials, 1e-12, 1e-3)
        assert new_state.ex.dtype == jnp.float16
        assert new_state.ey.dtype == jnp.float16
        assert new_state.ez.dtype == jnp.float16

    def test_update_he_fast_preserves_dtype(self):
        from rfx.core.yee import update_he_fast, precompute_coeffs
        state = init_state((10, 10, 10), field_dtype=jnp.float16)
        materials = init_materials((10, 10, 10))
        coeffs = precompute_coeffs(materials, 1e-12, 1e-3, pec_axes="xyz")
        state = state._replace(ez=state.ez.at[5, 5, 5].set(jnp.float16(1.0)))
        new_state = update_he_fast(state, coeffs)
        assert new_state.ex.dtype == jnp.float16
        assert new_state.hx.dtype == jnp.float16


# ---------------------------------------------------------------------------
# API-level integration tests
# ---------------------------------------------------------------------------

class TestSimulationMixedPrecision:
    def test_precision_parameter_validation(self):
        with pytest.raises(ValueError, match="precision"):
            Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.02),
                       boundary="pec", precision="float64")

    def test_mixed_precision_runs(self):
        """Mixed precision simulation completes without NaN."""
        sim = Simulation(
            freq_max=5e9,
            domain=(0.02, 0.02, 0.02),
            boundary="pec",
            precision="mixed",
        )
        sim.add_source(position=(0.01, 0.01, 0.01), component="ez")
        result = sim.run(n_steps=100)
        assert result is not None
        # Fields should be float16
        assert result.state.ex.dtype == jnp.float16
        assert result.state.hx.dtype == jnp.float16
        # No NaN in final state
        assert not jnp.any(jnp.isnan(result.state.ez))
        assert not jnp.any(jnp.isnan(result.state.hz))

    def test_mixed_precision_accuracy(self):
        """Mixed precision should match float32 within reasonable tolerance.

        float16 has ~3.3 decimal digits of precision, so we compare
        the overall field pattern rather than requiring exact match.
        For a short simulation (50 steps), the relative error in the
        L2 norm of the fields should be small.
        """
        kwargs = dict(
            freq_max=5e9,
            domain=(0.02, 0.02, 0.02),
            boundary="pec",
        )
        n_steps = 50

        sim32 = Simulation(**kwargs, precision="float32")
        sim32.add_source(position=(0.01, 0.01, 0.01), component="ez")
        r32 = sim32.run(n_steps=n_steps)

        sim16 = Simulation(**kwargs, precision="mixed")
        sim16.add_source(position=(0.01, 0.01, 0.01), component="ez")
        r16 = sim16.run(n_steps=n_steps)

        # Compare Ez field L2 norms
        ez32 = np.array(r32.state.ez, dtype=np.float32)
        ez16 = np.array(r16.state.ez, dtype=np.float32)

        norm32 = np.linalg.norm(ez32)
        if norm32 > 0:
            rel_error = np.linalg.norm(ez32 - ez16) / norm32
            # Allow up to 5% relative error for 50 steps with float16
            assert rel_error < 0.05, f"Relative error {rel_error:.4f} exceeds 5%"

    def test_mixed_precision_with_probe(self):
        """Probes should work in mixed precision mode."""
        sim = Simulation(
            freq_max=5e9,
            domain=(0.02, 0.02, 0.02),
            boundary="pec",
            precision="mixed",
        )
        sim.add_source(position=(0.01, 0.01, 0.01), component="ez")
        sim.add_probe(position=(0.015, 0.01, 0.01), component="ez")
        result = sim.run(n_steps=50)
        ts = np.array(result.time_series).ravel()
        assert len(ts) == 50
        # Probe should record nonzero values (source is active)
        assert np.any(ts != 0)
        assert not np.any(np.isnan(ts))

    def test_mixed_precision_memory_reduction(self):
        """Field arrays in mixed mode should be half the size of float32."""
        sim = Simulation(
            freq_max=5e9,
            domain=(0.02, 0.02, 0.02),
            boundary="pec",
            precision="mixed",
        )
        sim.add_source(position=(0.01, 0.01, 0.01), component="ez")
        result = sim.run(n_steps=10)
        st = result.state
        # Each field component should be float16 (2 bytes per element)
        assert st.ex.dtype == jnp.float16
        field_bytes = st.ex.nbytes
        expected_f32_bytes = st.ex.size * 4  # float32 would be 4 bytes
        assert field_bytes == expected_f32_bytes // 2

    def test_float32_precision_default(self):
        """Default precision='float32' should produce float32 fields."""
        sim = Simulation(
            freq_max=5e9,
            domain=(0.02, 0.02, 0.02),
            boundary="pec",
        )
        sim.add_source(position=(0.01, 0.01, 0.01), component="ez")
        result = sim.run(n_steps=10)
        assert result.state.ex.dtype == jnp.float32

    def test_repr_includes_precision(self):
        sim = Simulation(
            freq_max=5e9,
            domain=(0.02, 0.02, 0.02),
            boundary="pec",
            precision="mixed",
        )
        r = repr(sim)
        assert "precision='mixed'" in r
