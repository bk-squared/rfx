"""Tests for Kerr nonlinear material."""

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.yee import init_state, init_materials
from rfx.materials.nonlinear import KerrMaterial, apply_kerr_update


def test_kerr_modifies_eps_r():
    """eps_r should change when E-field is non-zero."""
    shape = (10, 10, 10)
    state = init_state(shape)
    materials = init_materials(shape)
    state = state._replace(ez=state.ez.at[5, 5, 5].set(1e6))  # strong field

    mask = jnp.zeros(shape, dtype=bool).at[4:7, 4:7, 4:7].set(True)
    kerr = KerrMaterial(eps_r_linear=2.0, chi3=1e-18)

    updated = apply_kerr_update(materials, state, [(mask, kerr)])
    assert float(updated.eps_r[5, 5, 5]) > 2.0, "Kerr should increase eps_r"
    assert float(updated.eps_r[0, 0, 0]) == 1.0, "Outside mask unchanged"


def test_kerr_zero_field_unchanged():
    """eps_r should equal linear value when E=0."""
    shape = (10, 10, 10)
    state = init_state(shape)
    materials = init_materials(shape)
    mask = jnp.ones(shape, dtype=bool)
    kerr = KerrMaterial(eps_r_linear=3.0, chi3=1e-18)

    updated = apply_kerr_update(materials, state, [(mask, kerr)])
    np.testing.assert_allclose(float(updated.eps_r[5, 5, 5]), 3.0)


def test_kerr_differentiable():
    """jax.grad should flow through Kerr update."""
    shape = (6, 6, 6)
    mask = jnp.ones(shape, dtype=bool)
    kerr = KerrMaterial(eps_r_linear=2.0, chi3=1e-12)

    def objective(ez_val):
        state = init_state(shape)
        state = state._replace(ez=state.ez.at[3, 3, 3].set(ez_val))
        materials = init_materials(shape)
        updated = apply_kerr_update(materials, state, [(mask, kerr)])
        return jnp.sum(updated.eps_r)

    grad = jax.grad(objective)(1.0)
    assert float(jnp.abs(grad)) > 0, "Gradient should be non-zero"


def test_kerr_intensity_dependent():
    """Stronger E-field should produce larger eps_r change."""
    shape = (6, 6, 6)
    mask = jnp.ones(shape, dtype=bool)
    kerr = KerrMaterial(eps_r_linear=2.0, chi3=1e-12)

    eps_vals = []
    for e_amp in [1e3, 1e4, 1e5]:
        state = init_state(shape)
        state = state._replace(ez=state.ez.at[3, 3, 3].set(e_amp))
        materials = init_materials(shape)
        updated = apply_kerr_update(materials, state, [(mask, kerr)])
        eps_vals.append(float(updated.eps_r[3, 3, 3]))

    assert eps_vals[1] > eps_vals[0], "Higher field → higher eps"
    assert eps_vals[2] > eps_vals[1], "Even higher field → even higher eps"
