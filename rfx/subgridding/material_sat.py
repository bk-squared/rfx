"""Material-weighted z-interface SAT algebra helpers.

This module is the Stage-3 D1 algebra surface for production material
subgridding.  It is intentionally small and independent of the JIT runner so
the energy/impedance formulas can be tested before any production validation is
expanded.

The helpers operate on one z-normal tangential pair:

* pair A: ``U = E_x``, ``V = H_y``
* pair B: ``U = E_y``, ``V = -H_x``

For both pairs the local 1-D Maxwell subsystem is
``U_t + (1/eps) V_n = 0`` and ``V_t + (1/mu) U_n = 0``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class UpwindTrace(NamedTuple):
    """Common interface state for one material-weighted tangential pair."""

    u_star: jnp.ndarray
    v_star: jnp.ndarray


class InterfaceDeltas(NamedTuple):
    """Lower/upper explicit SAT corrections for one tangential pair."""

    du_lower: jnp.ndarray
    dv_lower: jnp.ndarray
    du_upper: jnp.ndarray
    dv_upper: jnp.ndarray
    trace: UpwindTrace


def material_impedance(epsilon: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
    """Return the local wave impedance ``sqrt(mu / epsilon)``."""
    return jnp.sqrt(mu / epsilon)


def upwind_common_state(
    u_lower: jnp.ndarray,
    v_lower: jnp.ndarray,
    z_lower: jnp.ndarray,
    u_upper: jnp.ndarray,
    v_upper: jnp.ndarray,
    z_upper: jnp.ndarray,
) -> UpwindTrace:
    """Return the impedance upwind common state for one interface pair.

    The geometrically lower side has normal ``+z`` and the geometrically upper
    side has normal ``-z``.  ``z_lower`` and ``z_upper`` may be scalars or
    face-shaped arrays.
    """
    denom = z_lower + z_upper
    v_star = (u_lower - u_upper + z_lower * v_lower + z_upper * v_upper) / denom
    u_star = (
        z_upper * u_lower
        + z_lower * u_upper
        + z_lower * z_upper * (v_lower - v_upper)
    ) / denom
    return UpwindTrace(u_star=u_star, v_star=v_star)


def interface_pair_deltas(
    u_lower: jnp.ndarray,
    v_lower: jnp.ndarray,
    u_upper: jnp.ndarray,
    v_upper: jnp.ndarray,
    *,
    epsilon_lower: jnp.ndarray,
    mu_lower: jnp.ndarray,
    epsilon_upper: jnp.ndarray,
    mu_upper: jnp.ndarray,
    h_lower: float,
    h_upper: float,
    dt: float,
) -> InterfaceDeltas:
    """Return explicit SAT corrections for one z-normal pair.

    The returned ``dv_*`` correction is for ``V``.  For pair B
    (``V = -H_x``), callers must apply ``dH_x = -dV``.
    """
    z_lower = material_impedance(epsilon_lower, mu_lower)
    z_upper = material_impedance(epsilon_upper, mu_upper)
    trace = upwind_common_state(
        u_lower,
        v_lower,
        z_lower,
        u_upper,
        v_upper,
        z_upper,
    )
    du_lower = -(dt / (epsilon_lower * h_lower)) * (trace.v_star - v_lower)
    dv_lower = -(dt / (mu_lower * h_lower)) * (trace.u_star - u_lower)
    du_upper = +(dt / (epsilon_upper * h_upper)) * (trace.v_star - v_upper)
    dv_upper = +(dt / (mu_upper * h_upper)) * (trace.u_star - u_upper)
    return InterfaceDeltas(
        du_lower=du_lower,
        dv_lower=dv_lower,
        du_upper=du_upper,
        dv_upper=dv_upper,
        trace=trace,
    )


def upwind_energy_rate(
    u_lower: jnp.ndarray,
    v_lower: jnp.ndarray,
    u_upper: jnp.ndarray,
    v_upper: jnp.ndarray,
    z_lower: jnp.ndarray,
    z_upper: jnp.ndarray,
    face_weights: jnp.ndarray | float = 1.0,
) -> jnp.ndarray:
    """Return the non-positive interface energy rate from upwind jumps."""
    du = u_lower - u_upper
    dv = v_lower - v_upper
    denom = z_lower + z_upper
    density = -(du * du / denom) - ((z_lower * z_upper) / denom) * (dv * dv)
    return jnp.sum(face_weights * density)


def restrict_face_mean(fine_face: jnp.ndarray, coarse_shape: tuple[int, int], ratio: int) -> jnp.ndarray:
    """Restrict a fine face to a coarse face by block averaging."""
    if fine_face.ndim != 2:
        raise ValueError(f"fine_face must be 2-D, got shape {fine_face.shape}")
    if ratio < 1:
        raise ValueError("ratio must be positive")
    ni, nj = coarse_shape
    trimmed = fine_face[: ni * ratio, : nj * ratio]
    if trimmed.shape != (ni * ratio, nj * ratio):
        raise ValueError(
            f"fine face shape {fine_face.shape} is too small for "
            f"coarse_shape={coarse_shape} and ratio={ratio}"
        )
    return jnp.mean(trimmed.reshape(ni, ratio, nj, ratio), axis=(1, 3))


def prolong_face_repeat(coarse_face: jnp.ndarray, fine_shape: tuple[int, int], ratio: int) -> jnp.ndarray:
    """Prolong a coarse face to a fine face by constant repeat."""
    if coarse_face.ndim != 2:
        raise ValueError(f"coarse_face must be 2-D, got shape {coarse_face.shape}")
    if ratio < 1:
        raise ValueError("ratio must be positive")
    return jnp.repeat(jnp.repeat(coarse_face, ratio, axis=0), ratio, axis=1)[: fine_shape[0], : fine_shape[1]]


def projection_adjoint_residual(
    coarse_face: jnp.ndarray,
    fine_face: jnp.ndarray,
    *,
    ratio: int,
    coarse_face_area: float = 1.0,
) -> jnp.ndarray:
    """Return ``<c,Rf>_Bc - <Pc,f>_Bf`` for the mean/repeat pair."""
    restricted = restrict_face_mean(fine_face, coarse_face.shape, ratio)
    prolonged = prolong_face_repeat(coarse_face, fine_face.shape, ratio)
    fine_face_area = coarse_face_area / float(ratio * ratio)
    lhs = coarse_face_area * jnp.sum(coarse_face * restricted)
    rhs = fine_face_area * jnp.sum(prolonged * fine_face)
    return lhs - rhs
