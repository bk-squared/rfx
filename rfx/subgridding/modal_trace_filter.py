"""Low-order modal filters of a z-face SAT trace delta (subgrid diagnostics)."""

from __future__ import annotations

import jax.numpy as jnp


def _modal_axis_basis(n: int, kind: str, dtype):
    """Return a normalized low-order modal basis vector for one face axis."""
    constant = jnp.ones((n,), dtype=dtype) / jnp.sqrt(jnp.asarray(n, dtype=dtype))
    if kind == "constant":
        return constant
    coord = jnp.linspace(-1.0, 1.0, n, dtype=dtype)
    linear = coord - jnp.sum(coord * constant) * constant
    linear = linear / jnp.maximum(
        jnp.sqrt(jnp.sum(linear * linear)),
        jnp.asarray(1e-30, dtype=dtype),
    )
    if kind == "linear":
        return linear
    if kind == "quadratic":
        quadratic = coord * coord
        quadratic = quadratic - jnp.sum(quadratic * constant) * constant
        quadratic = quadratic - jnp.sum(quadratic * linear) * linear
        return quadratic / jnp.maximum(
            jnp.sqrt(jnp.sum(quadratic * quadratic)),
            jnp.asarray(1e-30, dtype=dtype),
        )
    raise ValueError(f"unknown modal axis basis kind={kind!r}")


def _modal_face_component(face_delta, y_kind: str, x_kind: str):
    """Project a 2-D face delta onto one separable low-order modal component."""
    y_basis = _modal_axis_basis(face_delta.shape[0], y_kind, face_delta.dtype)
    x_basis = _modal_axis_basis(face_delta.shape[1], x_kind, face_delta.dtype)
    basis = y_basis[:, None] * x_basis[None, :]
    return jnp.sum(face_delta * basis) * basis


def _modal_filter_face_delta(face_delta, mode: str):
    """Return a low-order modal subset of a z-face trace delta.

    This diagnostic filter is based only on the current simulated pre/post
    trace delta.  It does not use reference fields or probe-error artifacts.
    """
    specs_by_mode = {
        "modal_linear_delta": (
            ("constant", "linear"),
            ("linear", "constant"),
            ("linear", "linear"),
        ),
        "modal_xy_linear_delta": (("linear", "linear"),),
        "modal_mixed_quadratic_delta": (
            ("linear", "quadratic"),
            ("quadratic", "linear"),
        ),
        "modal_xy_plus_mixed_delta": (
            ("linear", "linear"),
            ("linear", "quadratic"),
            ("quadratic", "linear"),
        ),
        "modal_low2_delta": (
            ("constant", "linear"),
            ("linear", "constant"),
            ("linear", "linear"),
            ("constant", "quadratic"),
            ("quadratic", "constant"),
            ("linear", "quadratic"),
            ("quadratic", "linear"),
            ("quadratic", "quadratic"),
        ),
    }
    if mode not in specs_by_mode:
        raise ValueError(f"unknown modal trace delta filter={mode!r}")
    filtered = jnp.zeros_like(face_delta)
    for y_kind, x_kind in specs_by_mode[mode]:
        filtered = filtered + _modal_face_component(face_delta, y_kind, x_kind)
    return filtered
