"""A dz design variable must still span its declared domain."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.nonuniform import make_nonuniform_grid


def _span(delta):
    grid = make_nonuniform_grid(
        (.008, .008), jnp.full(8, .001) + delta, .001, 0,
        domain=(.008, .008, .008))
    return jnp.sum(grid.dz[:-1])


@pytest.mark.parametrize("route", ["jvp", "grad", "jit", "vmap"])
def test_traced_dz_sum_matches_declared_extent(route):
    def call(delta):
        if route == "jvp":
            return jax.jvp(_span, (jnp.float32(delta),), (jnp.float32(1.),))[0]
        if route == "grad":
            return jax.grad(_span)(jnp.float32(delta))
        if route == "jit":
            return jax.jit(_span)(jnp.float32(delta))
        return jax.vmap(_span)(jnp.full(2, delta))

    assert np.isfinite(np.asarray(call(0.))).all()
    # Compiled callbacks wrap the ValueError in a backend runtime error.
    error = ValueError if route in ("jvp", "grad", "vmap") else jax.errors.JaxRuntimeError
    with pytest.raises(error, match="dz_profile sums to.*declared z extent"):
        np.asarray(call(.0001))
