"""Raw CPML entry points respect conductivity and explicit curl ownership."""
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.boundaries.cpml import apply_cpml_e, init_cpml
from rfx.core.yee import EPS_0, init_materials, init_state
from rfx.grid import Grid


@pytest.mark.parametrize('override', [False, True])
def test_raw_entry_uses_loss_or_explicit_curl(override):
    grid = Grid(freq_max=20e9, domain=(0.01, 0.01, 0.01), dx=1e-3, cpml_layers=2)
    state = init_state(grid.shape)
    mats = init_materials(grid.shape)
    sigma = np.float32(4 * EPS_0 / float(grid.dt))
    mats = mats._replace(sigma=jnp.full(grid.shape, sigma))
    params, psi = init_cpml(grid)
    # A stored psi with zero H isolates the coefficient actually multiplied.
    psi = psi._replace(psi_ez_xlo=jnp.ones_like(psi.psi_ez_xlo))
    kwargs = {}
    if override:
        kwargs = dict(e_loss=(jnp.full(grid.shape, 99.),) * 3,
                      e_curl_coeff=(jnp.full(grid.shape, 0.125),) * 3)
    out, new_psi = apply_cpml_e(state, params, psi, grid, materials=mats,
                               faces=('x_lo',), **kwargs)
    expected = (0.125 if override else float(grid.dt) / (EPS_0 + float(sigma) * float(grid.dt) / 2))
    np.testing.assert_array_max_ulp(np.asarray(out.ez[:2]),
        np.asarray(expected * np.asarray(new_psi.psi_ez_xlo), np.float32), maxulp=2)


def test_raw_inverse_entry_uses_cell_sigma_with_update_epsilon():
    grid = Grid(freq_max=20e9, domain=(0.01,) * 3, dx=1e-3, cpml_layers=2)
    state = init_state(grid.shape)
    mats = init_materials(grid.shape)._replace(
        eps_r=jnp.full(grid.shape, 4.), sigma=jnp.full(grid.shape, 10.))
    inv = (jnp.full(grid.shape, 0.5),) * 3
    params, psi = init_cpml(grid)
    psi = psi._replace(psi_ez_xlo=jnp.ones_like(psi.psi_ez_xlo))
    out, new_psi = apply_cpml_e(state, params, psi, grid, materials=mats,
                               inv_eps_r_update=inv, faces=('x_lo',))
    expected = float(grid.dt) / (2 * EPS_0 + 10. * float(grid.dt) / 2)
    np.testing.assert_array_max_ulp(np.asarray(out.ez[:2]),
        np.asarray(expected * np.asarray(new_psi.psi_ez_xlo), np.float32), maxulp=2)


def test_zero_loss_retains_inverse_material_precision():
    from tests._x64_compat import enable_x64
    with enable_x64():
        grid = Grid(freq_max=20e9, domain=(0.01,) * 3, dx=1e-3, cpml_layers=2)
        state = init_state(grid.shape)
        mats = init_materials(grid.shape)
        inv = (jnp.full(grid.shape, 1 / 3.25, dtype=jnp.float64),) * 3
        params, psi = init_cpml(grid)
        psi = psi._replace(psi_ez_xlo=jnp.full_like(psi.psi_ez_xlo, 1.2345))
        baseline, _ = apply_cpml_e(state, params, psi, grid, materials=mats,
                                   inv_eps_r_update=inv)
        explicit, _ = apply_cpml_e(state, params, psi, grid, materials=mats,
                                   inv_eps_r_update=inv,
                                   e_loss=(jnp.zeros(grid.shape, dtype=jnp.float64),) * 3)
        np.testing.assert_array_equal(explicit.ez, baseline.ez)
