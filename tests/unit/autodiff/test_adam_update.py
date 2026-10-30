"""Bit-exact arithmetic contract for the shared Adam update."""

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.optimize import _adam_update


@pytest.mark.parametrize("step_clamp", [None, 0.01, 10.0])
@pytest.mark.parametrize("kwargs", [{}, {"beta1": 0.8, "beta2": 0.95, "eps": 0.03}])
def test_adam_update_matches_inline_formula(step_clamp, kwargs):
    m = jnp.array([0.2, -0.3, 0.04], dtype=jnp.float32)
    v = jnp.array([0.01, 0.2, 0.03], dtype=jnp.float32)
    grad = jnp.array([-0.4, 0.6, 0.1], dtype=jnp.float32)
    it, lr = 4, 0.07
    beta1 = kwargs.get("beta1", 0.9)
    beta2 = kwargs.get("beta2", 0.999)
    eps = kwargs.get("eps", 1e-8)

    expected_m = beta1 * m + (1 - beta1) * grad
    expected_v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = expected_m / (1 - beta1 ** (it + 1))
    v_hat = expected_v / (1 - beta2 ** (it + 1))
    expected_step = lr * m_hat / (jnp.sqrt(v_hat) + eps)
    if step_clamp is not None:
        snorm = jnp.linalg.norm(expected_step)
        expected_step = jnp.where(
            snorm > step_clamp, expected_step * (step_clamp / snorm), expected_step,
        )

    step, new_m, new_v = _adam_update(
        m, v, grad, it, lr, step_clamp=step_clamp, **kwargs,
    )
    assert np.array_equal(new_m, expected_m)
    assert np.array_equal(new_v, expected_v)
    assert np.array_equal(step, expected_step)
