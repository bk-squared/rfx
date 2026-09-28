"""A guided TMz pulse measures the optional ADI absorber at a small timestep."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.adi import apply_adi_cpml_2d, init_adi_cpml_2d
from tests._absorber_witness import adi_cpml_reflection


@pytest.mark.parametrize("axis", ("x", "y"))
@pytest.mark.parametrize("n_layers", (8, 16))
def test_adi_cpml_return_at_small_timestep(n_layers, axis):
    with jax.default_device(jax.devices("cpu")[0]):
        reflected_db = adi_cpml_reflection(n_layers, axis, np.array([2e9, 10e9, 30e9]))
    jax.clear_caches()
    # The matched-absorber goal is -60 dB. The 8/16-layer, per-face records
    # and PEC calibration resolve that goal at CFL factor 0.01 only.
    assert np.all(np.isfinite(reflected_db)), f"{axis}: nonfinite return"
    assert np.all(reflected_db < -60), f"{axis} low/high R at 2/10/30 GHz: {reflected_db} dB"


@pytest.mark.parametrize("n_layers", (8, 465, 466, 500))
def test_adi_cpml_unexcited_vacuum_at_large_depth(n_layers):
    """Unexcited vacuum must retain exactly 0 V/m and 0 A/m at any depth."""
    with jax.default_device(jax.devices("cpu")[0]):
        shape = (2 * n_layers + 35, 2 * n_layers + 37)
        dt, dx, dy = 1e-14, 1e-3, 1.3e-3
        zero = jnp.zeros(shape, dtype=jnp.float32)
        params, memory = init_adi_cpml_2d(n_layers, dt, dx, dy, *shape)
        ez, hx, hy, memory = apply_adi_cpml_2d(
            zero, zero, zero, params, memory, jnp.ones_like(zero), dt, dx, dy)
        # With no excitation every derivative and ADE memory is exactly zero;
        # no rounding tolerance can permit NaN fields or a generated wave.
        coefficients_finite = all(np.all(np.isfinite(a)) for a in jax.tree.leaves(params))
        fields_zero = all(np.all(np.asarray(a) == 0) for a in (ez, hx, hy))
        memory_zero = all(np.all(np.asarray(a) == 0) for a in memory)
    jax.clear_caches()
    assert coefficients_finite, f"{n_layers} layers: nonfinite CPML coefficients"
    assert fields_zero, f"{n_layers} layers: unexcited vacuum generated nonzero/nonfinite fields"
    assert memory_zero, f"{n_layers} layers: nonzero/nonfinite CPML memory"
