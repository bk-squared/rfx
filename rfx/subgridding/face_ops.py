"""Face operators for the Phase-1 z-slab SBP-SAT lane.

The canonical Phase-1 contract uses explicit face norms together with
cell-centered linear prolongation.  The implementation stores separable
axis operators ``P_i``/``P_j`` and ``R_i = P_i^T / ratio`` /
``R_j = P_j^T / ratio``.  The full 2-D face restriction is therefore
``R_face = R_i @ face @ R_j.T`` (equivalent to ``P_face.T / ratio**2``
for uniform face cells), not a one-axis ``P^T / ratio`` shortcut.
These explicit operators are the single source of truth for the current
uniform-in-face z-slab lane, and tests should lock both their
interpolation structure and their norm-compatibility identity.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class ZFaceOps(NamedTuple):
    """Explicit z-face operators for coarse/fine coupling."""

    coarse_shape: tuple[int, int]
    fine_shape: tuple[int, int]
    ratio: int
    coarse_area: float
    fine_area: float
    coarse_norm: jnp.ndarray
    fine_norm: jnp.ndarray
    prolong_i: jnp.ndarray
    prolong_j: jnp.ndarray
    restrict_i: jnp.ndarray
    restrict_j: jnp.ndarray


def _build_linear_prolongation_1d(n_coarse: int, ratio: int) -> jnp.ndarray:
    """Cell-centered linear prolongation matrix for one face axis."""

    n_fine = n_coarse * ratio
    mat = np.zeros((n_fine, n_coarse), dtype=np.float32)
    for i_f in range(n_fine):
        x_c = (i_f + 0.5) / ratio - 0.5
        left = int(np.floor(x_c))
        alpha = x_c - left
        if left < 0:
            mat[i_f, 0] = 1.0
        elif left >= n_coarse - 1:
            mat[i_f, n_coarse - 1] = 1.0
        else:
            mat[i_f, left] = 1.0 - alpha
            mat[i_f, left + 1] = alpha
    return jnp.asarray(mat)


def build_zface_norms(
    coarse_shape: tuple[int, int],
    ratio: int,
    dx_c: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return explicit diagonal norms for one z face."""

    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    ni_c, nj_c = coarse_shape
    if ni_c <= 0 or nj_c <= 0:
        raise ValueError(f"coarse_shape must be positive, got {coarse_shape}")

    dx_f = dx_c / ratio
    coarse_area = dx_c * dx_c
    fine_area = dx_f * dx_f

    coarse_norm = jnp.full((ni_c, nj_c), coarse_area, dtype=jnp.float32)
    fine_norm = jnp.full((ni_c * ratio, nj_c * ratio), fine_area, dtype=jnp.float32)
    return coarse_norm, fine_norm


def build_zface_ops(
    coarse_shape: tuple[int, int],
    ratio: int,
    dx_c: float,
) -> ZFaceOps:
    """Build the Phase-1 operator bundle for one z face."""

    coarse_norm, fine_norm = build_zface_norms(coarse_shape, ratio, dx_c)
    ni_c, nj_c = coarse_shape
    prolong_i = _build_linear_prolongation_1d(ni_c, ratio)
    prolong_j = _build_linear_prolongation_1d(nj_c, ratio)
    return ZFaceOps(
        coarse_shape=coarse_shape,
        fine_shape=(ni_c * ratio, nj_c * ratio),
        ratio=ratio,
        coarse_area=float(dx_c * dx_c),
        fine_area=float((dx_c / ratio) ** 2),
        coarse_norm=coarse_norm,
        fine_norm=fine_norm,
        prolong_i=prolong_i,
        prolong_j=prolong_j,
        restrict_i=prolong_i.T / ratio,
        restrict_j=prolong_j.T / ratio,
    )


def build_zface_prolongation(
    coarse_shape: tuple[int, int],
    ratio: int,
    dx_c: float,
) -> ZFaceOps:
    """Return the operator bundle used for z-face prolongation."""

    return build_zface_ops(coarse_shape, ratio, dx_c)


def build_zface_restriction(
    coarse_shape: tuple[int, int],
    ratio: int,
    dx_c: float,
) -> ZFaceOps:
    """Return the operator bundle used for z-face restriction."""

    return build_zface_ops(coarse_shape, ratio, dx_c)


def prolong_zface(coarse_face: jnp.ndarray, ops: ZFaceOps) -> jnp.ndarray:
    """Apply the cell-centered linear prolongation operator ``P``."""

    coarse_face = jnp.asarray(coarse_face, dtype=jnp.float32)
    if coarse_face.shape != ops.coarse_shape:
        raise ValueError(
            f"coarse_face shape {coarse_face.shape} does not match {ops.coarse_shape}"
        )
    return ops.prolong_i @ coarse_face @ ops.prolong_j.T


def restrict_zface(fine_face: jnp.ndarray, ops: ZFaceOps) -> jnp.ndarray:
    """Apply the separable norm-derived full-face restriction."""

    fine_face = jnp.asarray(fine_face, dtype=jnp.float32)
    if fine_face.shape != ops.fine_shape:
        raise ValueError(
            f"fine_face shape {fine_face.shape} does not match {ops.fine_shape}"
        )

    return ops.restrict_i @ fine_face @ ops.restrict_j.T


def check_norm_compatibility(
    ops: ZFaceOps,
    coarse_face: jnp.ndarray | None = None,
    fine_face: jnp.ndarray | None = None,
    *,
    atol: float = 1e-6,
) -> dict[str, float | bool]:
    """Evaluate the norm-compatibility identity for one operator bundle."""

    if coarse_face is None:
        coarse_face = jnp.asarray(
            np.arange(np.prod(ops.coarse_shape), dtype=np.float32).reshape(ops.coarse_shape)
        )
    else:
        coarse_face = jnp.asarray(coarse_face, dtype=jnp.float32)

    if fine_face is None:
        fine_face = jnp.asarray(
            np.arange(np.prod(ops.fine_shape), dtype=np.float32).reshape(ops.fine_shape)
        )
    else:
        fine_face = jnp.asarray(fine_face, dtype=jnp.float32)

    restricted = restrict_zface(fine_face, ops)
    prolonged = prolong_zface(coarse_face, ops)

    lhs = float(jnp.sum(coarse_face * ops.coarse_norm * restricted))
    rhs = float(jnp.sum(prolonged * ops.fine_norm * fine_face))
    abs_error = abs(lhs - rhs)
    return {
        "lhs": lhs,
        "rhs": rhs,
        "abs_error": abs_error,
        "passes": abs_error <= atol,
    }
