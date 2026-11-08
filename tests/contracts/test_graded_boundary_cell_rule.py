"""T1–T7: graded end cells respect precision and traced nominal declarations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation


def _grid(**profiles):
    return Simulation(
        freq_max=20e9, domain=(20e-3, 20e-3, 10e-3), dx=1e-3,
        boundary="cpml", cpml_layers=4, **profiles,
    )._build_nonuniform_grid()


def _x_cells(profile):
    return _grid(dx_profile=profile).dx_arr


def _y_cells(profile):
    return _grid(dy_profile=profile).dy_arr


def _jvp_cells(build, profile):
    primal, _ = jax.jvp(build, (profile,), (jnp.zeros_like(profile),))
    return jax.block_until_ready(primal)


def test_t1_concrete_float32_boundary_cells_build():
    grid = _grid(dx_profile=jnp.full(20, 1e-3, dtype=jnp.float32))
    assert grid.dx == 0.001
    assert (grid.pad_x_lo, grid.pad_x_hi) == (4, 4)
    np.testing.assert_array_equal(
        np.asarray(grid.dx_arr)[4:24], np.full(20, 0.001, dtype=np.float32),
    )


def test_t2_jvp_float32_cells_are_bit_identical_to_concrete():
    profile = jnp.full(20, 1e-3, dtype=jnp.float32)
    concrete = np.asarray(_x_cells(profile))
    traced = np.asarray(_jvp_cells(_x_cells, profile))
    assert concrete.shape == traced.shape == (29,)
    assert concrete.dtype == traced.dtype == np.dtype("float32")
    np.testing.assert_array_equal(traced.view(np.uint32), concrete.view(np.uint32))
    np.testing.assert_array_equal(traced, np.full(29, 0.001, dtype=np.float32))


@pytest.mark.parametrize("traced", [False, True], ids=["concrete", "jvp"])
def test_t3_first_x_boundary_cell_refused(traced):
    profile = jnp.full(20, 1e-3, dtype=jnp.float32).at[0].set(1.001e-3)
    with pytest.raises(Exception if traced else ValueError,
                       match=r"dx_profile\[0\].*must equal boundary dx"):
        _jvp_cells(_x_cells, profile) if traced else _x_cells(profile)


@pytest.mark.parametrize("traced", [False, True], ids=["concrete", "jvp"])
def test_t4_last_x_boundary_cell_refused(traced):
    profile = jnp.full(20, 1e-3, dtype=jnp.float32).at[-1].set(1.001e-3)
    with pytest.raises(Exception if traced else ValueError,
                       match=r"dx_profile\[-1\].*must equal boundary dx"):
        _jvp_cells(_x_cells, profile) if traced else _x_cells(profile)


def test_t5_float64_boundary_tolerance_is_not_loosened():
    profile = np.full(20, 1e-3, dtype=np.float64)
    profile[0] = 0.00100000001
    with pytest.raises(ValueError, match=r"dx_profile\[0\].*tolerance=1e-12 m"):
        _x_cells(profile)


def test_t6_traced_y_boundary_cells_checked():
    profile = jnp.full(20, 1e-3, dtype=jnp.float32)
    np.testing.assert_array_equal(
        _jvp_cells(_y_cells, profile), np.full(29, 0.001, dtype=np.float32),
    )
    bad = profile.at[-1].set(1.001e-3)
    with pytest.raises(Exception, match=r"dy_profile\[-1\].*must equal boundary dx"):
        _jvp_cells(_y_cells, bad)


def test_t7_traced_nominal_x_boundary_can_differ_from_dx():
    profile = jnp.full(20, 1e-3, dtype=jnp.float32).at[0].set(2e-3)
    nominal = np.asarray(profile, dtype=np.float64)

    def build(cells):
        # The nominal sum must match the declared extent, including f32 rounding.
        grid = Simulation(
            freq_max=20e9, domain=(float(nominal.sum()), 20e-3, 10e-3), dx=1e-3,
            boundary="cpml", cpml_layers=4, dx_profile=cells,
            nominal_profiles={"x": nominal},
        )._build_nonuniform_grid()
        assert grid.dx == 0.001
        assert grid.nominal_cells["x"].dtype == np.dtype("float64")
        assert grid.traced_cells is None
        return grid.dx_arr[grid.pad_x_lo]

    first = np.asarray(_jvp_cells(build, profile))
    assert first.dtype == np.dtype("float32")
    assert first == np.float32(2e-3)
