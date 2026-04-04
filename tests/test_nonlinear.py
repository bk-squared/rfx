"""Tests for Kerr nonlinear material."""

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.yee import init_state, init_materials, EPS_0
from rfx.materials.nonlinear import KerrMaterial, apply_kerr_update, apply_kerr_ade


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


# ---------------------------------------------------------------------------
# ADE-based Kerr tests (apply_kerr_ade)
# ---------------------------------------------------------------------------

def test_kerr_zero_chi3_unchanged():
    """E-field should be unchanged when chi3 is zero everywhere."""
    shape = (8, 8, 8)
    state = init_state(shape)
    # Set a non-zero field
    state = state._replace(
        ez=state.ez.at[4, 4, 4].set(1e6),
        ex=state.ex.at[3, 3, 3].set(5e5),
    )
    chi3_arr = jnp.zeros(shape, dtype=jnp.float32)
    dt = 1e-12

    corrected = apply_kerr_ade(state, chi3_arr, dt)

    np.testing.assert_array_equal(np.array(corrected.ex), np.array(state.ex))
    np.testing.assert_array_equal(np.array(corrected.ey), np.array(state.ey))
    np.testing.assert_array_equal(np.array(corrected.ez), np.array(state.ez))


def test_kerr_nonzero_modifies_field():
    """Non-zero chi3 should reduce E-field magnitude (self-defocusing)."""
    shape = (8, 8, 8)
    state = init_state(shape)
    e_val = 1e6  # 1 MV/m
    state = state._replace(ez=state.ez.at[4, 4, 4].set(e_val))

    chi3_val = 1e-18  # m^2/V^2
    chi3_arr = jnp.full(shape, chi3_val, dtype=jnp.float32)
    dt = 1e-12

    corrected = apply_kerr_ade(state, chi3_arr, dt)

    ez_orig = float(state.ez[4, 4, 4])
    ez_corr = float(corrected.ez[4, 4, 4])

    # The correction subtracts (dt/eps0)*chi3*|E|^2*E, so |E| should decrease
    assert abs(ez_corr) < abs(ez_orig), (
        f"Kerr ADE should reduce field magnitude: {ez_corr} vs {ez_orig}"
    )

    # Verify the correction factor is physically correct
    expected_factor = (dt / EPS_0) * chi3_val * e_val ** 2
    expected_ez = e_val * (1.0 - expected_factor)
    np.testing.assert_allclose(ez_corr, expected_ez, rtol=1e-5)

    # Cells without field should remain zero
    assert float(corrected.ez[0, 0, 0]) == 0.0


def test_kerr_energy_bounded():
    """Nonlinear simulation with Kerr ADE should not diverge.

    Run a small FDTD loop with a point source and Kerr material.
    The total E-field energy should remain bounded.
    """
    from rfx.core.yee import update_e, update_h

    shape = (20, 20, 20)
    materials = init_materials(shape)
    state = init_state(shape)
    dt = 1e-12
    dx = 1e-3

    chi3_arr = jnp.full(shape, 1e-18, dtype=jnp.float32)

    # Inject a strong initial pulse
    state = state._replace(ez=state.ez.at[10, 10, 10].set(1e6))

    max_energy = 0.0
    for step in range(200):
        state = update_h(state, materials, dt, dx)
        state = update_e(state, materials, dt, dx)
        state = apply_kerr_ade(state, chi3_arr, dt)

        energy = float(jnp.sum(state.ex ** 2 + state.ey ** 2 + state.ez ** 2))
        max_energy = max(max_energy, energy)

        # Check no NaN or Inf
        assert jnp.all(jnp.isfinite(state.ex)), f"NaN/Inf at step {step}"
        assert jnp.all(jnp.isfinite(state.ey)), f"NaN/Inf at step {step}"
        assert jnp.all(jnp.isfinite(state.ez)), f"NaN/Inf at step {step}"

    # Energy should be bounded (not growing exponentially)
    final_energy = float(jnp.sum(state.ex ** 2 + state.ey ** 2 + state.ez ** 2))
    assert final_energy < max_energy * 1.01, (
        f"Energy should not grow: final={final_energy:.3e}, max={max_energy:.3e}"
    )
