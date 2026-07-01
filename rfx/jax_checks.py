"""JAX checkify helpers for compiled rfx invariants.

The functions here are small wrappers around ``jax.experimental.checkify``. They
let callers put RF/design invariants inside JAX-traced code so failures survive
``jit``, ``grad``, ``vmap``, and ``lax.scan`` transformations as checkify errors.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import jax.numpy as jnp
from jax.experimental import checkify

F = TypeVar("F", bound=Callable[..., Any])


def checkify_invariants(
    fun: F,
    *,
    float_checks: bool = True,
    index_checks: bool = False,
) -> Callable[..., tuple[checkify.Error, Any]]:
    """Return a checkified wrapper for user and optional automatic checks.

    User checks are always enabled because the helpers below emit
    ``checkify.check`` predicates. ``float_checks`` additionally catches NaN and
    divide-by-zero errors inserted by JAX checkify. ``index_checks`` is opt-in
    because it can be noisier for code that intentionally uses masked indexing.
    """
    errors = checkify.user_checks
    if float_checks:
        errors = errors | checkify.float_checks
    if index_checks:
        errors = errors | checkify.index_checks
    return checkify.checkify(fun, errors=errors)


def check_finite(value: Any, name: str = "value"):
    """Assert that all entries of ``value`` are finite inside JAX tracing."""
    arr = jnp.asarray(value)
    checkify.check(jnp.all(jnp.isfinite(arr)), f"{name} must be finite")
    return value


def check_positive(value: Any, name: str = "value", *, allow_zero: bool = False):
    """Assert that all entries of ``value`` are positive or non-negative."""
    arr = jnp.asarray(value)
    if allow_zero:
        checkify.check(jnp.all(arr >= 0), f"{name} must be non-negative")
    else:
        checkify.check(jnp.all(arr > 0), f"{name} must be positive")
    return value


def check_bounds(
    value: Any,
    *,
    lower: float | None = None,
    upper: float | None = None,
    name: str = "value",
):
    """Assert lower/upper bounds for an array-like value inside JAX tracing."""
    if lower is None and upper is None:
        raise ValueError("at least one of lower or upper must be provided")
    arr = jnp.asarray(value)
    if lower is not None:
        checkify.check(jnp.all(arr >= lower), f"{name} must be >= {lower}")
    if upper is not None:
        checkify.check(jnp.all(arr <= upper), f"{name} must be <= {upper}")
    return value


def check_courant_number(
    courant: Any,
    *,
    limit: float = 1.0,
    name: str = "courant",
):
    """Assert a finite positive Courant-like number below ``limit``."""
    if not limit > 0:
        raise ValueError("limit must be positive")
    check_finite(courant, name=name)
    check_positive(courant, name=name)
    arr = jnp.asarray(courant)
    checkify.check(jnp.all(arr <= limit), f"{name} must be <= {limit}")
    return courant


__all__ = [
    "check_bounds",
    "check_courant_number",
    "check_finite",
    "check_positive",
    "checkify_invariants",
]
