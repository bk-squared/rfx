"""Private SBP derivative and mortar operators for SBP-SAT diagnostics.

This module is intentionally internal to the experimental SBP-SAT lane.  It
contains reusable algebraic contracts for derivative, boundary, and mortar
operators before any production solver hunk is admitted.  The helpers avoid
runtime configuration switches and public observable surfaces; tests consume
these contracts as private gate evidence.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0


class SBPDerivative1D(NamedTuple):
    """First-derivative SBP operator with a diagonal norm."""

    size: int
    dx: float
    norm: jnp.ndarray
    derivative: jnp.ndarray
    left_boundary: jnp.ndarray
    right_boundary: jnp.ndarray
    grid_role: str


class YeeStaggeredDerivativePair1D(NamedTuple):
    """Compatible primal/dual derivative pair for a reduced Yee line."""

    primal_size: int
    dual_size: int
    dx: float
    primal_norm: jnp.ndarray
    dual_norm: jnp.ndarray
    primal_to_dual: jnp.ndarray
    dual_to_primal: jnp.ndarray
    primal_left_boundary: jnp.ndarray
    primal_right_boundary: jnp.ndarray
    dual_left_boundary: jnp.ndarray
    dual_right_boundary: jnp.ndarray


class MortarProjection1D(NamedTuple):
    """Norm-compatible 1-D coarse/fine mortar projection."""

    coarse_size: int
    fine_size: int
    ratio: int
    dx_c: float
    dx_f: float
    coarse_norm: jnp.ndarray
    fine_norm: jnp.ndarray
    prolong: jnp.ndarray
    restrict: jnp.ndarray


class TensorFaceMortar(NamedTuple):
    """Tensor-product face mortar assembled from two 1-D mortars."""

    coarse_shape: tuple[int, int]
    fine_shape: tuple[int, int]
    ratio: int
    i: MortarProjection1D
    j: MortarProjection1D
    coarse_norm: jnp.ndarray
    fine_norm: jnp.ndarray


class FaceFluxSpec(NamedTuple):
    """Shape and normal sign for a private all-face flux identity probe."""

    face: str
    coarse_shape: tuple[int, int]
    normal_sign: int


def _as_matrix(diagonal: jnp.ndarray) -> jnp.ndarray:
    return jnp.diag(jnp.asarray(diagonal, dtype=jnp.float32))


def build_sbp_first_derivative_1d(
    size: int,
    dx: float,
    *,
    grid_role: str = "collocated",
) -> SBPDerivative1D:
    """Build the first-order diagonal-norm SBP first derivative.

    The returned operator satisfies ``H D + D.T H = e_R e_R.T - e_L e_L.T``.
    It is deliberately simple and explicit so the private architecture gate can
    audit signs, norms, and boundary extraction before production integration.
    """

    if size < 2:
        raise ValueError(f"size must be at least 2, got {size}")
    if dx <= 0:
        raise ValueError(f"dx must be positive, got {dx}")

    norm = np.full((size,), dx, dtype=np.float32)
    norm[0] *= 0.5
    norm[-1] *= 0.5

    derivative = np.zeros((size, size), dtype=np.float32)
    derivative[0, 0] = -1.0 / dx
    derivative[0, 1] = 1.0 / dx
    derivative[-1, -2] = -1.0 / dx
    derivative[-1, -1] = 1.0 / dx
    for index in range(1, size - 1):
        derivative[index, index - 1] = -0.5 / dx
        derivative[index, index + 1] = 0.5 / dx

    left = np.zeros((size,), dtype=np.float32)
    right = np.zeros((size,), dtype=np.float32)
    left[0] = 1.0
    right[-1] = 1.0
    return SBPDerivative1D(
        size=size,
        dx=float(dx),
        norm=jnp.asarray(norm),
        derivative=jnp.asarray(derivative),
        left_boundary=jnp.asarray(left),
        right_boundary=jnp.asarray(right),
        grid_role=grid_role,
    )


def sbp_boundary_matrix(operator: SBPDerivative1D) -> jnp.ndarray:
    """Return ``B = e_R e_R.T - e_L e_L.T`` for an SBP derivative."""

    return jnp.outer(operator.right_boundary, operator.right_boundary) - jnp.outer(
        operator.left_boundary,
        operator.left_boundary,
    )


def sbp_identity_residual(operator: SBPDerivative1D) -> jnp.ndarray:
    """Return the matrix defect in ``H D + D.T H = B``."""

    norm_matrix = _as_matrix(operator.norm)
    return (
        norm_matrix @ operator.derivative
        + operator.derivative.T @ norm_matrix
        - sbp_boundary_matrix(operator)
    )


def build_yee_staggered_derivative_pair_1d(
    primal_size: int,
    dx: float,
) -> YeeStaggeredDerivativePair1D:
    """Build a reduced primal/dual pair satisfying a discrete Green identity.

    ``primal_to_dual`` is the usual adjacent difference. ``dual_to_primal`` is
    then derived from the norm-weighted adjoint relation
    ``H_p D_hp + D_ph.T H_h = R_p.T R_h - L_p.T L_h``.  This gives the private
    gate an explicit Yee-staggered dual-operator identity instead of relying on
    a collocated scalar SBP check alone.
    """

    if primal_size < 3:
        raise ValueError(f"primal_size must be at least 3, got {primal_size}")
    if dx <= 0:
        raise ValueError(f"dx must be positive, got {dx}")

    dual_size = primal_size - 1
    primal_norm = np.full((primal_size,), dx, dtype=np.float32)
    primal_norm[0] *= 0.5
    primal_norm[-1] *= 0.5
    dual_norm = np.full((dual_size,), dx, dtype=np.float32)

    primal_to_dual = np.zeros((dual_size, primal_size), dtype=np.float32)
    for index in range(dual_size):
        primal_to_dual[index, index] = -1.0 / dx
        primal_to_dual[index, index + 1] = 1.0 / dx

    primal_left = np.zeros((primal_size,), dtype=np.float32)
    primal_right = np.zeros((primal_size,), dtype=np.float32)
    primal_left[0] = 1.0
    primal_right[-1] = 1.0
    dual_left = np.zeros((dual_size,), dtype=np.float32)
    dual_right = np.zeros((dual_size,), dtype=np.float32)
    dual_left[0] = 1.0
    dual_right[-1] = 1.0
    boundary = np.outer(primal_right, dual_right) - np.outer(primal_left, dual_left)
    dual_to_primal = (boundary - primal_to_dual.T * dual_norm[None, :]) / primal_norm[
        :, None
    ]

    return YeeStaggeredDerivativePair1D(
        primal_size=primal_size,
        dual_size=dual_size,
        dx=float(dx),
        primal_norm=jnp.asarray(primal_norm),
        dual_norm=jnp.asarray(dual_norm),
        primal_to_dual=jnp.asarray(primal_to_dual),
        dual_to_primal=jnp.asarray(dual_to_primal),
        primal_left_boundary=jnp.asarray(primal_left),
        primal_right_boundary=jnp.asarray(primal_right),
        dual_left_boundary=jnp.asarray(dual_left),
        dual_right_boundary=jnp.asarray(dual_right),
    )


def yee_staggered_identity_residual(pair: YeeStaggeredDerivativePair1D) -> jnp.ndarray:
    """Return the primal/dual Green-identity matrix defect."""

    boundary = jnp.outer(pair.primal_right_boundary, pair.dual_right_boundary) - jnp.outer(
        pair.primal_left_boundary,
        pair.dual_left_boundary,
    )
    return (
        _as_matrix(pair.primal_norm) @ pair.dual_to_primal
        + pair.primal_to_dual.T @ _as_matrix(pair.dual_norm)
        - boundary
    )


def build_norm_compatible_mortar_1d(
    coarse_size: int,
    ratio: int,
    dx_c: float,
) -> MortarProjection1D:
    """Build a norm-compatible piecewise-constant coarse/fine mortar.

    The projection is conservative and intentionally low order: each coarse
    value owns exactly ``ratio`` fine cells.  Therefore ``R P = I`` for all
    coarse vectors while ``H_c R = P.T H_f`` remains exact.  Higher-order
    interpolation stays outside this private gate until its projection/noop and
    energy identities can be proven.
    """

    if coarse_size <= 0:
        raise ValueError(f"coarse_size must be positive, got {coarse_size}")
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    if dx_c <= 0:
        raise ValueError(f"dx_c must be positive, got {dx_c}")

    fine_size = coarse_size * ratio
    dx_f = dx_c / ratio
    prolong = np.zeros((fine_size, coarse_size), dtype=np.float32)
    for coarse_index in range(coarse_size):
        start = coarse_index * ratio
        prolong[start : start + ratio, coarse_index] = 1.0
    restrict = prolong.T / ratio
    return MortarProjection1D(
        coarse_size=coarse_size,
        fine_size=fine_size,
        ratio=ratio,
        dx_c=float(dx_c),
        dx_f=float(dx_f),
        coarse_norm=jnp.full((coarse_size,), dx_c, dtype=jnp.float32),
        fine_norm=jnp.full((fine_size,), dx_f, dtype=jnp.float32),
        prolong=jnp.asarray(prolong),
        restrict=jnp.asarray(restrict),
    )


def mortar_adjoint_residual(mortar: MortarProjection1D) -> jnp.ndarray:
    """Return the matrix defect in ``H_c R = P.T H_f``."""

    return (
        _as_matrix(mortar.coarse_norm) @ mortar.restrict
        - mortar.prolong.T @ _as_matrix(mortar.fine_norm)
    )


def mortar_noop_residual(mortar: MortarProjection1D) -> jnp.ndarray:
    """Return ``R P - I`` for a 1-D mortar."""

    return mortar.restrict @ mortar.prolong - jnp.eye(
        mortar.coarse_size,
        dtype=jnp.float32,
    )


def build_tensor_face_mortar(
    coarse_shape: tuple[int, int],
    ratio: int,
    dx_c: float,
) -> TensorFaceMortar:
    """Build a tensor-product mortar for an oriented face."""

    if coarse_shape[0] <= 0 or coarse_shape[1] <= 0:
        raise ValueError(f"coarse_shape must be positive, got {coarse_shape}")
    i = build_norm_compatible_mortar_1d(coarse_shape[0], ratio, dx_c)
    j = build_norm_compatible_mortar_1d(coarse_shape[1], ratio, dx_c)
    return TensorFaceMortar(
        coarse_shape=coarse_shape,
        fine_shape=(coarse_shape[0] * ratio, coarse_shape[1] * ratio),
        ratio=ratio,
        i=i,
        j=j,
        coarse_norm=jnp.outer(i.coarse_norm, j.coarse_norm),
        fine_norm=jnp.outer(i.fine_norm, j.fine_norm),
    )


def prolong_face_mortar(coarse_face: jnp.ndarray, mortar: TensorFaceMortar) -> jnp.ndarray:
    """Apply tensor-product face prolongation."""

    coarse_face = jnp.asarray(coarse_face, dtype=jnp.float32)
    if coarse_face.shape != mortar.coarse_shape:
        raise ValueError(
            f"coarse_face shape {coarse_face.shape} does not match "
            f"{mortar.coarse_shape}"
        )
    return mortar.i.prolong @ coarse_face @ mortar.j.prolong.T


def restrict_face_mortar(fine_face: jnp.ndarray, mortar: TensorFaceMortar) -> jnp.ndarray:
    """Apply tensor-product face restriction."""

    fine_face = jnp.asarray(fine_face, dtype=jnp.float32)
    if fine_face.shape != mortar.fine_shape:
        raise ValueError(
            f"fine_face shape {fine_face.shape} does not match {mortar.fine_shape}"
        )
    return mortar.i.restrict @ fine_face @ mortar.j.restrict.T


def operator_projected_sat_pair_face(
    coarse_face: jnp.ndarray,
    fine_face: jnp.ndarray,
    mortar: TensorFaceMortar,
    alpha_c: float,
    alpha_f: float,
    coarse_mask: jnp.ndarray,
    fine_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply the private operator-projected face SAT update.

    This adapter is production-shaped but remains solver-independent: it uses
    only the tensor mortar identities already proved by the private A1-A4
    diagnostics and has no runner, hook, config, result, or observable surface.
    The solver-integration lane can compare this update against the current
    ``sbp_sat_3d.py`` face SAT path before deciding whether any solver hunk is
    admissible.
    """

    coarse_face = jnp.asarray(coarse_face, dtype=jnp.float32)
    fine_face = jnp.asarray(fine_face, dtype=jnp.float32)
    coarse_mask = jnp.asarray(coarse_mask, dtype=jnp.float32)
    fine_mask = jnp.asarray(fine_mask, dtype=jnp.float32)
    if coarse_mask.shape != mortar.coarse_shape:
        raise ValueError(
            f"coarse_mask shape {coarse_mask.shape} does not match {mortar.coarse_shape}"
        )
    if fine_mask.shape != mortar.fine_shape:
        raise ValueError(
            f"fine_mask shape {fine_mask.shape} does not match {mortar.fine_shape}"
        )

    coarse_mismatch = restrict_face_mortar(fine_face, mortar) - coarse_face
    fine_mismatch = prolong_face_mortar(coarse_face, mortar) - fine_face
    return (
        coarse_face + alpha_c * coarse_mismatch * coarse_mask,
        fine_face + alpha_f * fine_mismatch * fine_mask,
    )


def operator_projected_skew_eh_sat_face(
    *,
    ex_c: jnp.ndarray,
    ey_c: jnp.ndarray,
    hx_c: jnp.ndarray,
    hy_c: jnp.ndarray,
    ex_f: jnp.ndarray,
    ey_f: jnp.ndarray,
    hx_f: jnp.ndarray,
    hy_f: jnp.ndarray,
    mortar: TensorFaceMortar,
    alpha_c: float,
    alpha_f: float,
    coarse_mask: jnp.ndarray,
    fine_mask: jnp.ndarray,
    normal_sign: int = 1,
) -> tuple[jnp.ndarray, ...]:
    """Apply a private operator-projected skew E/H face-work candidate.

    The helper is intentionally solver-independent.  It combines a bounded
    ratio-derived scalar projection with an impedance-weighted tangential
    skew-pair transfer.  Inputs are interpreted in a face-local tangential
    basis whose orientation already follows the outward face convention; in that
    local basis electric updates use projected magnetic jumps through the fixed
    tangential rotation and magnetic updates use projected electric jumps
    through the same rotation.  The weights are derived only from the mortar
    ratio and vacuum impedance, so the candidate stays usable as private
    energy-transfer evidence without adding public APIs, hooks, runtime
    switches, or solver wiring.
    """

    if normal_sign not in (-1, 1):
        raise ValueError(f"normal_sign must be -1 or 1, got {normal_sign}")

    eta0 = float(np.sqrt(MU_0 / EPS_0))
    ratio_weight = 1.0 / float(mortar.ratio)
    skew_weight = 1.0 + ratio_weight

    scalar_pairs = [
        operator_projected_sat_pair_face(
            coarse,
            fine,
            mortar,
            alpha_c,
            alpha_f,
            coarse_mask,
            fine_mask,
        )
        for coarse, fine in (
            (ex_c, ex_f),
            (ey_c, ey_f),
            (hx_c, hx_f),
            (hy_c, hy_f),
        )
    ]

    def _jump_pair(coarse: jnp.ndarray, fine: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return restrict_face_mortar(fine, mortar) - coarse, prolong_face_mortar(
            coarse,
            mortar,
        ) - fine

    jex_c, jex_f = _jump_pair(ex_c, ex_f)
    jey_c, jey_f = _jump_pair(ey_c, ey_f)
    jhx_c, jhx_f = _jump_pair(hx_c, hx_f)
    jhy_c, jhy_f = _jump_pair(hy_c, hy_f)

    scalar_deltas = (
        scalar_pairs[0][0] - ex_c,
        scalar_pairs[1][0] - ey_c,
        scalar_pairs[2][0] - hx_c,
        scalar_pairs[3][0] - hy_c,
        scalar_pairs[0][1] - ex_f,
        scalar_pairs[1][1] - ey_f,
        scalar_pairs[2][1] - hx_f,
        scalar_pairs[3][1] - hy_f,
    )
    skew_deltas = (
        alpha_c * skew_weight * eta0 * (-jhy_c) * coarse_mask,
        alpha_c * skew_weight * eta0 * jhx_c * coarse_mask,
        alpha_c * skew_weight * (-jey_c) / eta0 * coarse_mask,
        alpha_c * skew_weight * jex_c / eta0 * coarse_mask,
        alpha_f * skew_weight * eta0 * (-jhy_f) * fine_mask,
        alpha_f * skew_weight * eta0 * jhx_f * fine_mask,
        alpha_f * skew_weight * (-jey_f) / eta0 * fine_mask,
        alpha_f * skew_weight * jex_f / eta0 * fine_mask,
    )
    before = (ex_c, ey_c, hx_c, hy_c, ex_f, ey_f, hx_f, hy_f)
    return tuple(
        component + ratio_weight * scalar_delta + skew_delta
        for component, scalar_delta, skew_delta in zip(
            before,
            scalar_deltas,
            skew_deltas,
            strict=True,
        )
    )


def face_mortar_adjoint_report(mortar: TensorFaceMortar) -> dict[str, float | bool]:
    """Return adjoint/noop defects for a tensor-product mortar."""

    i_adjoint = float(jnp.max(jnp.abs(mortar_adjoint_residual(mortar.i))))
    j_adjoint = float(jnp.max(jnp.abs(mortar_adjoint_residual(mortar.j))))
    i_noop = float(jnp.max(jnp.abs(mortar_noop_residual(mortar.i))))
    j_noop = float(jnp.max(jnp.abs(mortar_noop_residual(mortar.j))))
    max_defect = max(i_adjoint, j_adjoint, i_noop, j_noop)
    return {
        "i_adjoint_defect": i_adjoint,
        "j_adjoint_defect": j_adjoint,
        "i_noop_defect": i_noop,
        "j_noop_defect": j_noop,
        "max_defect": max_defect,
        "passes": max_defect <= 1.0e-6,
    }


def face_mortar_reproduction_report(mortar: TensorFaceMortar) -> dict[str, float | bool]:
    """Check constant and linear coarse-face reproduction under ``R P``."""

    i = jnp.arange(mortar.coarse_shape[0], dtype=jnp.float32)[:, None]
    j = jnp.arange(mortar.coarse_shape[1], dtype=jnp.float32)[None, :]
    probes = {
        "constant": jnp.ones(mortar.coarse_shape, dtype=jnp.float32),
        "linear_i": i + jnp.zeros(mortar.coarse_shape, dtype=jnp.float32),
        "linear_j": j + jnp.zeros(mortar.coarse_shape, dtype=jnp.float32),
    }
    defects = {}
    for name, coarse in probes.items():
        roundtrip = restrict_face_mortar(prolong_face_mortar(coarse, mortar), mortar)
        defects[f"{name}_max_error"] = float(jnp.max(jnp.abs(roundtrip - coarse)))
    max_error = max(defects.values())
    return {**defects, "max_error": max_error, "passes": max_error <= 1.0e-6}


def weighted_em_flux_residual(
    mortar: TensorFaceMortar,
    *,
    ex_c: jnp.ndarray,
    ey_c: jnp.ndarray,
    hx_c: jnp.ndarray,
    hy_c: jnp.ndarray,
    normal_sign: int,
    coarse_metric_weight: jnp.ndarray | None = None,
) -> float:
    """Return coarse-minus-fine weighted tangential EM flux residual.

    Fine traces and metric/material weights are generated by the same mortar so
    this is a reduced identity check for production-independent operator
    compatibility, not an empirical residual fit.
    """

    if normal_sign not in (-1, 1):
        raise ValueError(f"normal_sign must be -1 or 1, got {normal_sign}")
    if coarse_metric_weight is None:
        coarse_metric_weight = jnp.ones(mortar.coarse_shape, dtype=jnp.float32)
    else:
        coarse_metric_weight = jnp.asarray(coarse_metric_weight, dtype=jnp.float32)
        if coarse_metric_weight.shape != mortar.coarse_shape:
            raise ValueError(
                "coarse_metric_weight shape "
                f"{coarse_metric_weight.shape} does not match {mortar.coarse_shape}"
            )

    ex_c = jnp.asarray(ex_c, dtype=jnp.float32)
    ey_c = jnp.asarray(ey_c, dtype=jnp.float32)
    hx_c = jnp.asarray(hx_c, dtype=jnp.float32)
    hy_c = jnp.asarray(hy_c, dtype=jnp.float32)
    for name, arr in {
        "ex_c": ex_c,
        "ey_c": ey_c,
        "hx_c": hx_c,
        "hy_c": hy_c,
    }.items():
        if arr.shape != mortar.coarse_shape:
            raise ValueError(f"{name} shape {arr.shape} does not match {mortar.coarse_shape}")

    ex_f = prolong_face_mortar(ex_c, mortar)
    ey_f = prolong_face_mortar(ey_c, mortar)
    hx_f = prolong_face_mortar(hx_c, mortar)
    hy_f = prolong_face_mortar(hy_c, mortar)
    fine_metric_weight = prolong_face_mortar(coarse_metric_weight, mortar)
    coarse_flux = normal_sign * coarse_metric_weight * (ex_c * hy_c - ey_c * hx_c)
    fine_flux = normal_sign * fine_metric_weight * (ex_f * hy_f - ey_f * hx_f)
    residual = jnp.sum(
        (coarse_flux - restrict_face_mortar(fine_flux, mortar)) * mortar.coarse_norm
    )
    return float(residual)


def box_surface_partition_report(
    box_shape: tuple[int, int, int],
) -> dict[str, int | bool | str]:
    """Return a disjoint face/edge/corner accounting report for a 3-D box.

    The counts classify each boundary cell exactly once: cells with one
    boundary coordinate are face interiors, cells with two are edge interiors,
    and cells with three are corners.  This is private diagnostic evidence for
    all-six-face operator admission; it does not alter solver behavior.
    """

    if len(box_shape) != 3:
        raise ValueError(f"box_shape must have length 3, got {box_shape}")
    nx, ny, nz = box_shape
    if nx < 3 or ny < 3 or nz < 3:
        raise ValueError(f"box_shape must be at least 3 cells per axis, got {box_shape}")

    face_interior_cells = 2 * (
        (ny - 2) * (nz - 2)
        + (nx - 2) * (nz - 2)
        + (nx - 2) * (ny - 2)
    )
    edge_interior_cells = 4 * ((nx - 2) + (ny - 2) + (nz - 2))
    corner_cells = 8
    surface_cells = nx * ny * nz - (nx - 2) * (ny - 2) * (nz - 2)
    counted_surface_cells = face_interior_cells + edge_interior_cells + corner_cells
    partition_closes = counted_surface_cells == surface_cells
    return {
        "status": (
            "all_face_edge_corner_accounting_closed"
            if partition_closes
            else "all_face_edge_corner_accounting_mismatch"
        ),
        "active_faces": 6,
        "active_edges": 12,
        "active_corners": 8,
        "face_interior_cells": face_interior_cells,
        "edge_interior_cells": edge_interior_cells,
        "corner_cells": corner_cells,
        "surface_cells": surface_cells,
        "counted_surface_cells": counted_surface_cells,
        "partition_closes": partition_closes,
    }


def all_face_weighted_flux_report(
    face_specs: tuple[FaceFluxSpec, ...],
    *,
    ratio: int,
    dx_c: float,
) -> dict[str, object]:
    """Return material/metric weighted flux defects for every supplied face.

    Each face gets the same manufactured tangential-field family, with shape and
    normal sign supplied by the caller.  The fine-side fields and metric weights
    are generated through the tensor mortar, so a zero residual is an operator
    identity check rather than residual fitting.
    """

    if not face_specs:
        raise ValueError("face_specs must not be empty")

    residuals: list[dict[str, float | str | int | tuple[int, int]]] = []
    for spec in face_specs:
        mortar = build_tensor_face_mortar(spec.coarse_shape, ratio=ratio, dx_c=dx_c)
        i = jnp.arange(spec.coarse_shape[0], dtype=jnp.float32)[:, None]
        j = jnp.arange(spec.coarse_shape[1], dtype=jnp.float32)[None, :]
        zeros = jnp.zeros(spec.coarse_shape, dtype=jnp.float32)
        residual = weighted_em_flux_residual(
            mortar,
            ex_c=1.0 + 0.1 * i + 0.2 * j,
            ey_c=-0.5 + 0.05 * i - 0.1 * j,
            hx_c=2.0e-6 + 0.1e-6 * i + zeros,
            hy_c=-1.0e-6 + 0.2e-6 * j + zeros,
            normal_sign=spec.normal_sign,
            coarse_metric_weight=1.0 + 0.01 * i + 0.02 * j,
        )
        residuals.append(
            {
                "face": spec.face,
                "coarse_shape": spec.coarse_shape,
                "normal_sign": spec.normal_sign,
                "weighted_flux_residual": residual,
            }
        )

    max_abs_residual = max(
        abs(float(item["weighted_flux_residual"])) for item in residuals
    )
    return {
        "faces_tested": tuple(item["face"] for item in residuals),
        "face_count": len(residuals),
        "residuals": tuple(residuals),
        "max_abs_residual": max_abs_residual,
        "passes": max_abs_residual <= 1.0e-12,
    }
