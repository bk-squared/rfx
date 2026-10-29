"""Distributed CPML must consume the same physical H profiles as native."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.boundaries.cpml import apply_cpml_h, init_cpml
from types import SimpleNamespace
from rfx.core.yee import init_state
from rfx.grid import Grid
from rfx.runners._distributed_common import init_cpml_slabs
from rfx.runners.distributed_nu import init_cpml_for_sharded_nu


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("nu", [False, True])
def test_distributed_cpml_uses_native_staggered_h_profiles(axis: int, nu: bool, record_property) -> None:
    h = 1 / 512
    grid = Grid(freq_max=5e9, domain=(17 * h,) * 3, dx=h, cpml_layers=4)
    values = np.zeros(grid.shape, np.float32)
    low: list[int | slice] = [slice(None)] * 3
    high: list[int | slice] = [slice(None)] * 3
    low[axis], high[axis] = 2, grid.shape[axis] - 3
    values[tuple(low)] = values[tuple(high)] = 1
    name = "e" + "xyz"[(axis + 2) % 3]
    native = init_state(grid.shape)._replace(**{name: jnp.asarray(values)})
    native_params, native_aux = init_cpml(grid)
    expected, _ = apply_cpml_h(native, native_params, native_aux, grid)
    slab = jax.tree_util.tree_map(
        lambda value: jnp.pad(value, ((1, 1), (0, 0), (0, 0))) if value.ndim == 3 else value,
        native)
    if nu:
        metadata = SimpleNamespace(global_grid=grid, nx_local=grid.nx + 2, cpml_layers=4)
        params, auxiliary = init_cpml_for_sharded_nu(metadata, 1)
    else:
        params, auxiliary = init_cpml_slabs(grid, grid.nx + 2, 1)
    mapped = jax.pmap(
        lambda state, aux: apply_cpml_h(
            state, params, aux, grid,
            frame={"x": (1, grid.nx + 1)},
            owned={"x_lo": jnp.asarray(True), "x_hi": jnp.asarray(True)}),
        axis_name="x", devices=jax.devices()[:1])
    actual, _ = mapped(jax.tree_util.tree_map(lambda value: value[None], slab), auxiliary)
    component = "h" + "xyz"[(axis + 1) % 3]
    observed = np.asarray(getattr(actual, component))[0, 1:-1]
    reference = np.asarray(getattr(expected, component))
    assert np.any(reference != 0)
    np.testing.assert_allclose(observed, reference, rtol=1e-6, atol=0)
    record_property("bitwise", bool(np.array_equal(observed.view(np.uint32), reference.view(np.uint32))))
