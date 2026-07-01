import jax
import jax.numpy as jnp
import pytest

import rfx
from rfx.jax_checks import (
    check_bounds,
    check_courant_number,
    check_finite,
    check_positive,
    checkify_invariants,
)


def test_checkify_invariants_catches_user_checks_under_jit():
    def loss(eps_r):
        check_finite(eps_r, name="eps_r")
        check_bounds(eps_r, lower=1.0, upper=12.0, name="eps_r")
        return jnp.sum(eps_r)

    checked = jax.jit(checkify_invariants(loss))

    err, value = checked(jnp.array([1.0, 2.0, 12.0]))
    assert err.get() is None
    assert value == pytest.approx(15.0)

    err, _ = checked(jnp.array([1.0, 13.0]))
    assert "eps_r must be <= 12.0" in str(err.get())

    err, _ = checked(jnp.array([1.0, jnp.nan]))
    assert "eps_r must be finite" in str(err.get())


def test_checkify_invariants_can_enable_automatic_float_checks():
    def reciprocal(x):
        return 1.0 / x

    checked = jax.jit(checkify_invariants(reciprocal, float_checks=True))
    err, value = checked(jnp.array([1.0, 0.0]))

    assert jnp.isinf(value[1])
    assert "division by zero" in str(err.get()).lower()

def test_checkify_invariants_can_disable_float_checks_and_enable_index_checks():
    def reciprocal(x):
        return 1.0 / x

    no_float_checks = jax.jit(
        checkify_invariants(reciprocal, float_checks=False)
    )
    err, value = no_float_checks(jnp.array([1.0, 0.0]))
    assert err.get() is None
    assert jnp.isinf(value[1])

    def out_of_bounds(x):
        return x[jnp.array([2])][0]

    checked_index = jax.jit(
        checkify_invariants(out_of_bounds, float_checks=False, index_checks=True)
    )
    err, _ = checked_index(jnp.array([1.0, 2.0]))
    assert "out-of-bounds" in str(err.get())



def test_check_helpers_work_with_vmap():
    def validate_courant(courant):
        check_courant_number(courant, limit=0.99, name="cfl")
        return courant

    checked = checkify_invariants(jax.vmap(validate_courant))

    err, value = checked(jnp.array([0.1, 0.5, 0.99]))
    assert err.get() is None
    assert value.tolist() == pytest.approx([0.1, 0.5, 0.99])

    err, _ = checked(jnp.array([0.1, 1.01]))
    assert "cfl must be <= 0.99" in str(err.get())

def test_check_helpers_work_through_lax_scan():
    def body(carry, x):
        check_positive(x, name="scan_weight")
        return carry + x, x

    def run_scan(xs):
        total, _ = jax.lax.scan(body, 0.0, xs)
        return total

    checked = checkify_invariants(run_scan)
    err, value = checked(jnp.array([1.0, 2.0, 3.0]))
    assert err.get() is None
    assert value == pytest.approx(6.0)

    err, _ = checked(jnp.array([1.0, -1.0, 3.0]))
    assert "scan_weight must be positive" in str(err.get())


def test_positive_and_bounds_validate_python_side_configuration():
    with pytest.raises(ValueError, match="at least one"):
        check_bounds(jnp.ones((2,)))
    with pytest.raises(ValueError, match="limit"):
        check_courant_number(jnp.array(0.5), limit=0.0)

    def nonnegative(x):
        check_positive(x, name="loss_weight", allow_zero=True)
        return jnp.sum(x)

    checked = checkify_invariants(nonnegative)
    err, _ = checked(jnp.array([0.0, -1.0]))
    assert "loss_weight must be non-negative" in str(err.get())


def test_jax_checks_public_exports():
    assert rfx.checkify_invariants is checkify_invariants
    assert rfx.check_finite is check_finite
    assert rfx.check_positive is check_positive
    assert rfx.check_bounds is check_bounds
    assert rfx.check_courant_number is check_courant_number
