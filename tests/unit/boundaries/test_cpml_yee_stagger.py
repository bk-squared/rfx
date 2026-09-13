"""Actual CPML operator mirror symmetry on staggered Yee coordinates."""

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.boundaries.cpml import apply_cpml_e, apply_cpml_h, init_cpml
from rfx.core.yee import init_state
from rfx.grid import Grid


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("magnetic", [False, True])
def test_cpml_correction_respects_physical_mirror(axis: int, magnetic: bool) -> None:
    """E reflects about (N-1)/2, transverse H about (N-2)/2."""
    h = 1 / 512
    grid = Grid(freq_max=5e9, domain=(23 * h,) * 3, dx=h, cpml_layers=12)
    params, auxiliary = init_cpml(grid)
    state = init_state(grid.shape)
    values = np.zeros(grid.shape, np.float32)
    low: list[int | slice] = [slice(None)] * 3
    high: list[int | slice] = [slice(None)] * 3
    count = grid.shape[axis]
    if magnetic:
        low[axis], high[axis] = 6, count - 7
        values[tuple(low)] = values[tuple(high)] = 1
        source_component = "e" + "xyz"[(axis + 2) % 3]
        result_component = "h" + "xyz"[(axis + 1) % 3]
        state = state._replace(**{source_component: jnp.asarray(values)})
        result, _ = apply_cpml_h(state, params, auxiliary, grid, axes="xyz"[axis])
        indices = np.arange(count - 1)
        actual = np.take(np.asarray(getattr(result, result_component)), indices, axis=axis)
        expected = -np.flip(actual, axis=axis)
    else:
        low[axis], high[axis] = 5, count - 7
        values[tuple(low)] = 1
        values[tuple(high)] = -1
        source_component = "h" + "xyz"[(axis + 1) % 3]
        result_component = "e" + "xyz"[(axis + 2) % 3]
        state = state._replace(**{source_component: jnp.asarray(values)})
        result, _ = apply_cpml_e(state, params, auxiliary, grid, axes="xyz"[axis])
        actual = np.asarray(getattr(result, result_component))
        expected = np.flip(actual, axis=axis)
    assert np.any(actual != 0)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize("budget", [1, 4])
def test_one_sample_faces_retain_legacy_coefficient_convention(budget: int) -> None:
    """One sample has no resolved grading interval; do not invent a new origin."""
    h = 1 / 512
    grid = Grid(freq_max=5e9, domain=(17 * h,) * 3, dx=h, cpml_layers=budget,
                face_layers={axis + side: 1 for axis in "xyz" for side in ("_lo", "_hi")})
    params, _ = init_cpml(grid)
    assert params.magnetic is not None
    for face in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"):
        electric = getattr(params, face)
        magnetic = getattr(params.magnetic, face)
        for name in ("sigma", "kappa", "alpha", "b", "c"):
            np.testing.assert_array_equal(getattr(electric, name), getattr(magnetic, name))
