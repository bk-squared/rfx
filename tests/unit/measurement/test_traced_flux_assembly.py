"""Monitor assembly refuses every traced quadrature axis before stepping."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.measurement.flux_weights import grid_flux_monitor
from rfx.nonuniform import make_nonuniform_grid
from rfx._traced_refusal import TracedStructureRefusal


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("route", ["jvp", "grad", "jit", "vmap"])
def test_flux_monitor_traced_cells_refuse(axis, route):
    def build(delta):
        dz = jnp.full(8, .001).at[3].add(delta).at[4].add(-delta)
        grid = make_nonuniform_grid((.008, .008), dz, .001, 0)
        mon = grid_flux_monitor(grid, axis=axis, index=2, freqs=np.array([1e9]))
        return jnp.sum(mon.e1_dft).real

    with pytest.raises(TracedStructureRefusal, match="flux monitor.*traced cell widths.*concrete mesh"):
        if route == "jvp":
            jax.jvp(build, (jnp.float32(0.),), (jnp.float32(1.),))
        elif route == "grad":
            jax.grad(build)(jnp.float32(0.))
        elif route == "jit":
            jax.jit(build)(jnp.float32(0.))
        else:
            jax.vmap(build)(jnp.zeros(2))
