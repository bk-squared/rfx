"""Tests for Kerr nonlinear material."""

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.yee import init_state, init_materials, EPS_0
from rfx.materials.nonlinear import KerrMaterial, apply_kerr_update, apply_kerr_ade

pytestmark = pytest.mark.gpu


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
# Reactive Kerr tests (apply_kerr_ade — #437 increment-scaling ε_eff form)
#
# apply_kerr_ade now implements the REACTIVE index change ε_eff = ε_r + χ³|E|²
# (lossless) by scaling the E-UPDATE INCREMENT, not the whole field:
#   E^{n+1} = E^n + (E_lin - E^n) / (1 + χ³·|E^n|²/ε_r)
# The pre-#437 form E_lin/(1+factor) was a nonlinear ABSORBER (reduced |E|, zero
# phase shift) — see docs/research_notes/2026-07-22_kerr_operator_defect.md.
# ---------------------------------------------------------------------------

def test_kerr_zero_chi3_unchanged():
    """chi3=0 => increment unscaled => the post-update field is returned unchanged."""
    shape = (8, 8, 8)
    e_prev = (jnp.zeros(shape, jnp.float32).at[4, 4, 4].set(3e5),
              jnp.zeros(shape, jnp.float32), jnp.zeros(shape, jnp.float32))
    state = init_state(shape)._replace(
        ez=jnp.zeros(shape, jnp.float32).at[4, 4, 4].set(1e6),
        ex=jnp.zeros(shape, jnp.float32).at[3, 3, 3].set(5e5),
    )
    chi3_arr = jnp.zeros(shape, dtype=jnp.float32)
    eps_r_arr = jnp.full(shape, 2.0, dtype=jnp.float32)

    corrected = apply_kerr_ade(state, e_prev, chi3_arr, eps_r_arr)

    np.testing.assert_array_equal(np.array(corrected.ex), np.array(state.ex))
    np.testing.assert_array_equal(np.array(corrected.ez), np.array(state.ez))


def test_kerr_reactive_selfconsistent_solve():
    """Non-zero chi3 solves the constitutive relation ε_r·E + χ³·|E|²·E = D_target
    self-consistently (the #446 D-based reactive Kerr), preserving the equilibrium E^n.

      (a) zero increment (E_lin == E^n): the field is UNCHANGED even for chi3>0 (E^n already
          satisfies the constitutive relation ⇒ D_target=ε_eff·E^n ⇒ E=E^n). A dissipative
          whole-field absorber would still shrink it.
      (b) nonzero increment: E^{n+1} solves the cubic and lies strictly between E^n and E_lin.
    """
    shape = (6, 6, 6)
    E0, E_lin, eps_r, chi3_val = 4e5, 7e5, 2.0, 1e-12
    e_prev = (jnp.full(shape, E0, jnp.float32),
              jnp.zeros(shape, jnp.float32), jnp.zeros(shape, jnp.float32))
    chi3_arr = jnp.full(shape, chi3_val, jnp.float32)
    eps_r_arr = jnp.full(shape, eps_r, jnp.float32)

    # (a) zero increment => the equilibrium field is preserved
    state_eq = init_state(shape)._replace(ex=jnp.full(shape, E0, jnp.float32))
    out_eq = apply_kerr_ade(state_eq, e_prev, chi3_arr, eps_r_arr)
    np.testing.assert_allclose(np.array(out_eq.ex), E0, rtol=1e-5)

    # (b) nonzero increment => the self-consistent constitutive solve (same fixed-point)
    state = init_state(shape)._replace(ex=jnp.full(shape, E_lin, jnp.float32))
    out = apply_kerr_ade(state, e_prev, chi3_arr, eps_r_arr)
    D_target = eps_r * E_lin + chi3_val * E0 ** 3          # ε_r·E_lin + χ³·|E^n|²·E^n
    E = E_lin
    for _ in range(4):
        E = D_target / (eps_r + chi3_val * E ** 2)
    np.testing.assert_allclose(float(out.ex[3, 3, 3]), E, rtol=1e-3)
    assert E0 < float(out.ex[3, 3, 3]) < E_lin            # between equilibrium and linear


def test_kerr_ade_differentiable():
    """jax.grad flows through the reactive Kerr correction (differentiable design surface)."""
    shape = (6, 6, 6)
    e_prev = (jnp.zeros(shape, jnp.float32).at[3, 3, 3].set(2e5),
              jnp.zeros(shape, jnp.float32), jnp.zeros(shape, jnp.float32))
    chi3_arr = jnp.full(shape, 1e-12, jnp.float32)
    eps_r_arr = jnp.full(shape, 2.0, jnp.float32)

    def objective(e_lin_val):
        state = init_state(shape)._replace(
            ex=jnp.zeros(shape, jnp.float32).at[3, 3, 3].set(e_lin_val))
        return jnp.sum(apply_kerr_ade(state, e_prev, chi3_arr, eps_r_arr).ex)

    g = jax.grad(objective)(5e5)
    assert float(jnp.abs(g)) > 0, "gradient should be non-zero"


def test_kerr_reactive_conserves_energy():
    """A reactive-Kerr FDTD run stays finite and LOSSLESS — the total energy does not
    systematically drain (the pre-#437 dissipative operator would bleed it toward zero)."""
    from rfx.core.yee import update_e, update_h

    shape = (20, 20, 20)
    materials = init_materials(shape)          # vacuum, lossless (sigma=0)
    eps_r_arr = materials.eps_r
    dt, dx = 1e-12, 1e-3
    chi3_arr = jnp.full(shape, 0.1, jnp.float32)   # moderate: factor ~ 0.1 at |E|~1

    state = init_state(shape)._replace(ez=jnp.zeros(shape, jnp.float32).at[10, 10, 10].set(1.0))

    energies = []
    for step in range(200):
        e_prev = (state.ex, state.ey, state.ez)
        state = update_h(state, materials, dt, dx)
        state = update_e(state, materials, dt, dx)
        state = apply_kerr_ade(state, e_prev, chi3_arr, eps_r_arr)
        energies.append(float(jnp.sum(state.ex ** 2 + state.ey ** 2 + state.ez ** 2)))
        assert jnp.all(jnp.isfinite(state.ez)), f"NaN/Inf at step {step}"

    energies = np.array(energies)
    peak = np.max(energies)
    # lossless: the field bounces in the PEC-like box without a systematic energy drain
    assert energies[-1] > 0.5 * peak, (
        f"reactive Kerr must not dissipate energy: final={energies[-1]:.3e}, peak={peak:.3e}")
