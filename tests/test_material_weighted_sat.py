"""Algebra gates for material-weighted z-interface SAT."""

from __future__ import annotations

import jax.numpy as jnp
import jax
import numpy as np

from rfx.subgridding.material_sat import (
    interface_pair_deltas,
    material_impedance,
    projection_adjoint_residual,
    upwind_common_state,
    upwind_energy_rate,
)


def test_mean_repeat_projection_is_adjoint_in_face_norm():
    coarse = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) - 2.0
    fine = jnp.arange(108, dtype=jnp.float32).reshape(9, 12) / 17.0

    residual = projection_adjoint_residual(
        coarse,
        fine,
        ratio=3,
        coarse_face_area=0.25,
    )

    assert abs(float(residual)) < 1e-5


def test_material_impedance_upwind_state_preserves_identical_traces():
    u = jnp.array([[1.0, -2.0], [0.5, 0.25]], dtype=jnp.float32)
    v = jnp.array([[0.1, -0.3], [0.7, -0.9]], dtype=jnp.float32)
    z_lower = material_impedance(jnp.asarray(2.0), jnp.asarray(8.0))
    z_upper = material_impedance(jnp.asarray(5.0), jnp.asarray(20.0))

    trace = upwind_common_state(u, v, z_lower, u, v, z_upper)
    rate = upwind_energy_rate(u, v, u, v, z_lower, z_upper)

    np.testing.assert_allclose(trace.u_star, u, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(trace.v_star, v, rtol=1e-6, atol=1e-6)
    assert abs(float(rate)) < 1e-7


def test_equal_impedance_energy_rate_matches_closed_form_and_is_nonpositive():
    u_l = jnp.array([[1.0, 0.0], [-0.5, 2.0]], dtype=jnp.float32)
    v_l = jnp.array([[0.25, -0.75], [1.5, 0.5]], dtype=jnp.float32)
    u_r = jnp.array([[0.2, -0.4], [-0.1, 1.25]], dtype=jnp.float32)
    v_r = jnp.array([[0.0, -0.25], [1.0, -0.5]], dtype=jnp.float32)
    weights = jnp.array([[0.5, 0.5], [0.25, 0.25]], dtype=jnp.float32)
    z = jnp.asarray(3.0, dtype=jnp.float32)

    rate = upwind_energy_rate(u_l, v_l, u_r, v_r, z, z, weights)
    expected = -jnp.sum(weights * ((u_l - u_r) ** 2 / (2.0 * z) + 0.5 * z * (v_l - v_r) ** 2))

    np.testing.assert_allclose(rate, expected, rtol=1e-6, atol=1e-6)
    assert float(rate) <= 0.0


def test_interface_pair_deltas_use_material_mass_inverse_coefficients():
    eps_l = jnp.asarray(2.0)
    mu_l = jnp.asarray(8.0)
    eps_r = jnp.asarray(3.0)
    mu_r = jnp.asarray(12.0)
    out = interface_pair_deltas(
        jnp.asarray(1.0),
        jnp.asarray(0.25),
        jnp.asarray(-0.5),
        jnp.asarray(-0.75),
        epsilon_lower=eps_l,
        mu_lower=mu_l,
        epsilon_upper=eps_r,
        mu_upper=mu_r,
        h_lower=0.004,
        h_upper=0.002,
        dt=1e-12,
    )

    np.testing.assert_allclose(
        out.du_lower,
        -(1e-12 / (eps_l * 0.004)) * (out.trace.v_star - 0.25),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        out.dv_upper,
        +(1e-12 / (mu_r * 0.002)) * (out.trace.u_star + 0.5),
        rtol=1e-6,
    )


def test_pair_b_v_equals_negative_hx_sign_conversion():
    out = interface_pair_deltas(
        jnp.asarray(0.2),   # E_y lower
        jnp.asarray(-0.4),  # V_B lower = -H_x lower
        jnp.asarray(1.1),   # E_y upper
        jnp.asarray(0.3),   # V_B upper = -H_x upper
        epsilon_lower=jnp.asarray(1.0),
        mu_lower=jnp.asarray(1.0),
        epsilon_upper=jnp.asarray(1.0),
        mu_upper=jnp.asarray(1.0),
        h_lower=1.0,
        h_upper=1.0,
        dt=0.1,
    )

    d_hx_lower = -out.dv_lower
    d_hx_upper = -out.dv_upper

    np.testing.assert_allclose(d_hx_lower, +(0.1) * (out.trace.u_star - 0.2), rtol=1e-6)
    np.testing.assert_allclose(d_hx_upper, -(0.1) * (out.trace.u_star - 1.1), rtol=1e-6)


def test_material_sat_gradient_matches_finite_difference():
    """JAX gradient through material impedance agrees with finite difference."""

    def objective(eps_lower):
        out = interface_pair_deltas(
            jnp.asarray(0.7),
            jnp.asarray(-0.2),
            jnp.asarray(-0.1),
            jnp.asarray(0.5),
            epsilon_lower=eps_lower,
            mu_lower=jnp.asarray(1.3),
            epsilon_upper=jnp.asarray(2.1),
            mu_upper=jnp.asarray(1.7),
            h_lower=0.004,
            h_upper=0.002,
            dt=1.0e-12,
        )
        return out.du_lower + 0.25 * out.dv_lower

    eps0 = jnp.asarray(1.8)
    grad_ad = float(jax.grad(objective)(eps0))
    h = 1.0e-3
    grad_fd = float((objective(eps0 + h) - objective(eps0 - h)) / (2.0 * h))

    np.testing.assert_allclose(grad_ad, grad_fd, rtol=2e-3, atol=1e-12)
