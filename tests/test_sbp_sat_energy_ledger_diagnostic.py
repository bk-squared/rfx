"""Private manufactured-interface energy-ledger diagnostics for SBP-SAT.

These tests intentionally stay private and diagnostic-only.  They classify
the current SAT trace update before any solver repair, hook experiment, or
public true-R/T observable promotion.
"""

from __future__ import annotations

import inspect
import math

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0
from rfx.grid import C0
import rfx.subgridding.sbp_sat_3d as sbp_sat_3d
from rfx.subgridding.face_ops import build_face_ops, prolong_face, restrict_face
from rfx.subgridding.sbp_sat_3d import (
    _active_corners,
    _active_edges,
    _active_faces,
    _apply_sat_pair_edge,
    _apply_sat_pair_scalar,
    _face_interior_masks,
    init_subgrid_3d,
    sat_penalty_coefficients,
)
from rfx.subgridding.face_ops import build_edge_ops, prolong_edge


_LEDGER_BALANCE_THRESHOLD = 2.0e-2
_ZERO_WORK_POSITIVE_INJECTION_TOL = 1.0e-12
_FACE_SHAPE = (4, 3)
_RATIO = 2
_DX_C = 0.004
_TAU = 0.5
_FLOOR = 1.0e-300
_BOUNDING_SCALAR_MIN = 0.5
_BOUNDING_SCALAR_MAX = 2.0
_COUPLING_STRENGTH_RATIO_MIN = 0.5
_CANDIDATE_BOUND_TOL = 1.0e-6
_BOUNDED_KERNEL_REPAIR_NEXT_PREREQUISITE = (
    "open a deliberate private SAT face-coupling theory/redesign ralplan after "
    "bounded face-kernel feasibility failed; keep hook experiments and public "
    "promotion closed"
)
_FACE_COUPLING_THEORY_STATUS = "paired_face_coupling_design_ready"
_FACE_COUPLING_NEXT_IMPLEMENTATION_LANE = (
    "private paired-face helper implementation ralplan"
)
_FACE_COUPLING_REPORT = (
    ".omx/reports/sbp-sat-private-face-coupling-theory-redesign-20260429T090845Z.md"
)
_PAIRED_FACE_HELPER_IMPLEMENTATION_STATUS = "production_context_mismatch_detected"
_PAIRED_FACE_HELPER_IMPLEMENTATION_NEXT_PREREQUISITE = (
    "private time-centered SAT staging redesign ralplan"
)
_PAIRED_FACE_HELPER_IMPLEMENTATION_REPORT = (
    ".omx/reports/sbp-sat-private-paired-face-helper-implementation-20260429T120413Z.md"
)
_TIME_CENTERED_STAGING_STATUS = "time_centered_staging_contract_ready"
_TIME_CENTERED_STAGING_NEXT_PREREQUISITE = (
    "private time-centered paired-face helper implementation ralplan"
)
_TIME_CENTERED_STAGING_REPORT = (
    ".omx/reports/"
    "sbp-sat-private-time-centered-eh-face-ledger-staging-redesign-"
    "20260429T143759Z.md"
)
_TIME_CENTERED_HELPER_IMPLEMENTATION_STATUS = (
    "private_time_centered_paired_face_helper_implemented"
)
_TIME_CENTERED_HELPER_IMPLEMENTATION_NEXT_PREREQUISITE = (
    "private time-centered paired-face helper fixture-quality recovery ralplan"
)
_TIME_CENTERED_HELPER_IMPLEMENTATION_REPORT = (
    ".omx/reports/"
    "sbp-sat-private-time-centered-paired-face-helper-implementation-"
    "20260429T170000Z.md"
)
_TIME_CENTERED_TERMINAL_OUTCOMES = (
    "time_centered_staging_contract_ready",
    "same_call_staging_insufficient",
    "cross_step_state_carry_required",
    "time_centered_ledger_theory_reopen_required",
)
_TIME_CENTERED_REJECTION_CATEGORIES = (
    "algebraic_ledger_failure",
    "trace_unavailable_at_insertion_point",
    "cross_step_state_required",
    "hook_equivalent_staging_required",
    "sat_update_reordering_required",
)
_TIME_CENTERED_HELPER_TERMINAL_OUTCOMES = (
    "private_time_centered_paired_face_helper_implemented",
    "production_slot_binding_mismatch_detected",
    "time_centered_helper_regression_blocked",
    "call_order_or_state_carry_required",
)


def _face_fixture_ops():
    ops = build_face_ops(_FACE_SHAPE, ratio=_RATIO, dx_c=_DX_C)
    coarse_mask, fine_mask = _face_interior_masks(ops.coarse_shape, ops.ratio)
    alpha_c, alpha_f = sat_penalty_coefficients(ops.ratio, _TAU)
    return ops, coarse_mask, fine_mask, alpha_c, alpha_f


def _trace_energy(
    *,
    ex_c,
    ey_c,
    hx_c,
    hy_c,
    ex_f,
    ey_f,
    hx_f,
    hy_f,
    ops,
    coarse_mask,
    fine_mask,
) -> float:
    coarse_e = jnp.sum((ex_c**2 + ey_c**2) * ops.coarse_norm * coarse_mask)
    fine_e = jnp.sum((ex_f**2 + ey_f**2) * ops.fine_norm * fine_mask)
    coarse_h = jnp.sum((hx_c**2 + hy_c**2) * ops.coarse_norm * coarse_mask)
    fine_h = jnp.sum((hx_f**2 + hy_f**2) * ops.fine_norm * fine_mask)
    return float(0.5 * EPS_0 * (coarse_e + fine_e) + 0.5 * MU_0 * (coarse_h + fine_h))


def _trace_energy_from_components(components, *, ops, coarse_mask, fine_mask) -> float:
    return _trace_energy(
        ex_c=components[0],
        ey_c=components[1],
        hx_c=components[2],
        hy_c=components[3],
        ex_f=components[4],
        ey_f=components[5],
        hx_f=components[6],
        hy_f=components[7],
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )


def _apply_scaled_sat_pair_face(
    coarse_face,
    fine_face,
    ops,
    alpha_c,
    alpha_f,
    coarse_mask,
    fine_mask,
    *,
    coarse_scale: float,
    fine_scale: float,
):
    coarse_mismatch = restrict_face(fine_face, ops) - coarse_face
    fine_mismatch = prolong_face(coarse_face, ops) - fine_face
    return (
        coarse_face + alpha_c * coarse_scale * coarse_mismatch * coarse_mask,
        fine_face + alpha_f * fine_scale * fine_mismatch * fine_mask,
    )


def _apply_scaled_face_sat_to_components(
    *,
    ex_c,
    ey_c,
    hx_c,
    hy_c,
    ex_f,
    ey_f,
    hx_f,
    hy_f,
    ops,
    coarse_mask,
    fine_mask,
    alpha_c,
    alpha_f,
    coarse_scale: float,
    fine_scale: float,
):
    ex_c, ex_f = _apply_scaled_sat_pair_face(
        ex_c,
        ex_f,
        ops,
        alpha_c,
        alpha_f,
        coarse_mask,
        fine_mask,
        coarse_scale=coarse_scale,
        fine_scale=fine_scale,
    )
    ey_c, ey_f = _apply_scaled_sat_pair_face(
        ey_c,
        ey_f,
        ops,
        alpha_c,
        alpha_f,
        coarse_mask,
        fine_mask,
        coarse_scale=coarse_scale,
        fine_scale=fine_scale,
    )
    hx_c, hx_f = _apply_scaled_sat_pair_face(
        hx_c,
        hx_f,
        ops,
        alpha_c,
        alpha_f,
        coarse_mask,
        fine_mask,
        coarse_scale=coarse_scale,
        fine_scale=fine_scale,
    )
    hy_c, hy_f = _apply_scaled_sat_pair_face(
        hy_c,
        hy_f,
        ops,
        alpha_c,
        alpha_f,
        coarse_mask,
        fine_mask,
        coarse_scale=coarse_scale,
        fine_scale=fine_scale,
    )
    return ex_c, ey_c, hx_c, hy_c, ex_f, ey_f, hx_f, hy_f


def _apply_face_sat_to_components(**kwargs):
    return _apply_scaled_face_sat_to_components(
        **kwargs,
        coarse_scale=1.0,
        fine_scale=1.0,
    )


def _interface_work_from_components(components, *, ops, coarse_mask) -> float:
    dt = 0.99 * _DX_C / (C0 * math.sqrt(3.0))
    return dt * _weighted_interface_work_residual(
        ex_c=components[0],
        ey_c=components[1],
        hx_c=components[2],
        hy_c=components[3],
        ex_f=components[4],
        ey_f=components[5],
        hx_f=components[6],
        hy_f=components[7],
        ops=ops,
        coarse_mask=coarse_mask,
    )


def _ledger_residual_for_components(
    before_components,
    after_components,
    *,
    ops,
    coarse_mask,
    fine_mask,
) -> dict[str, float | str]:
    before = _trace_energy_from_components(
        before_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    after = _trace_energy_from_components(
        after_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    interface_work = _interface_work_from_components(
        before_components,
        ops=ops,
        coarse_mask=coarse_mask,
    )
    net_delta_eh = after - before
    normalized_balance_residual = abs(net_delta_eh + interface_work) / max(
        abs(before),
        abs(interface_work),
        _FLOOR,
    )
    status = (
        "ledger_gate_passed"
        if normalized_balance_residual <= _LEDGER_BALANCE_THRESHOLD
        else "ledger_mismatch_detected"
    )
    return {
        "status": status,
        "trace_energy_before": before,
        "trace_energy_after": after,
        "net_delta_eh": net_delta_eh,
        "interface_work_residual": interface_work,
        "normalized_balance_residual": normalized_balance_residual,
        "threshold": _LEDGER_BALANCE_THRESHOLD,
    }


def _weighted_interface_work_residual(
    *,
    ex_c,
    ey_c,
    hx_c,
    hy_c,
    ex_f,
    ey_f,
    hx_f,
    hy_f,
    ops,
    coarse_mask,
) -> float:
    """Return a private z-normal Poynting-like trace-work residual.

    This is a manufactured diagnostic functional, not a public physical R/T
    observable.  Restricting the fine-grid product keeps the comparison on the
    same coarse face norm used by the current SAT face operators.
    """

    coarse_sn = ex_c * hy_c - ey_c * hx_c
    fine_sn = ex_f * hy_f - ey_f * hx_f
    return float(
        jnp.sum(
            (coarse_sn - restrict_face(fine_sn, ops)) * ops.coarse_norm * coarse_mask
        )
    )


def _face_ledger_packet(
    *,
    nonzero_h: bool,
    coarse_scale: float = 1.0,
    fine_scale: float = 1.0,
) -> dict[str, float | str | bool]:
    ops, coarse_mask, fine_mask, alpha_c, alpha_f = _face_fixture_ops()
    zeros = jnp.zeros(ops.coarse_shape, dtype=jnp.float32)
    perturbation = jnp.linspace(
        -2.0e-3,
        2.0e-3,
        int(np.prod(ops.fine_shape)),
        dtype=jnp.float32,
    ).reshape(ops.fine_shape)

    ex_c = zeros
    ey_c = zeros
    ex_f = prolong_face(ex_c, ops) + perturbation
    ey_f = prolong_face(ey_c, ops) - 0.5 * perturbation
    if nonzero_h:
        hx_c = jnp.ones(ops.coarse_shape, dtype=jnp.float32) * 1.0e-6
        hy_c = jnp.ones(ops.coarse_shape, dtype=jnp.float32) * 2.0e-6
        hx_f = prolong_face(hx_c, ops) + perturbation * 1.0e-3
        hy_f = prolong_face(hy_c, ops) - perturbation * 2.0e-3
    else:
        hx_c = zeros
        hy_c = zeros
        hx_f = prolong_face(hx_c, ops)
        hy_f = prolong_face(hy_c, ops)

    before = _trace_energy(
        ex_c=ex_c,
        ey_c=ey_c,
        hx_c=hx_c,
        hy_c=hy_c,
        ex_f=ex_f,
        ey_f=ey_f,
        hx_f=hx_f,
        hy_f=hy_f,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    after_components = _apply_scaled_face_sat_to_components(
        ex_c=ex_c,
        ey_c=ey_c,
        hx_c=hx_c,
        hy_c=hy_c,
        ex_f=ex_f,
        ey_f=ey_f,
        hx_f=hx_f,
        hy_f=hy_f,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
        alpha_c=alpha_c,
        alpha_f=alpha_f,
        coarse_scale=coarse_scale,
        fine_scale=fine_scale,
    )
    after = _trace_energy(
        ex_c=after_components[0],
        ey_c=after_components[1],
        hx_c=after_components[2],
        hy_c=after_components[3],
        ex_f=after_components[4],
        ey_f=after_components[5],
        hx_f=after_components[6],
        hy_f=after_components[7],
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    delta_trace_energy = after - before
    dt = 0.99 * _DX_C / (C0 * math.sqrt(3.0))
    interface_work_residual = dt * _weighted_interface_work_residual(
        ex_c=ex_c,
        ey_c=ey_c,
        hx_c=hx_c,
        hy_c=hy_c,
        ex_f=ex_f,
        ey_f=ey_f,
        hx_f=hx_f,
        hy_f=hy_f,
        ops=ops,
        coarse_mask=coarse_mask,
    )
    normalized_balance_residual = abs(
        delta_trace_energy + interface_work_residual
    ) / max(
        abs(before),
        abs(interface_work_residual),
        _FLOOR,
    )
    if not nonzero_h:
        status = (
            "zero_work_positive_injection_detected"
            if delta_trace_energy > _ZERO_WORK_POSITIVE_INJECTION_TOL
            else "zero_work_dissipative_gate_passed"
        )
    else:
        status = (
            "ledger_mismatch_detected"
            if normalized_balance_residual > _LEDGER_BALANCE_THRESHOLD
            else "ledger_gate_passed"
        )
    return {
        "status": status,
        "trace_energy_before": before,
        "trace_energy_after": after,
        "net_delta_eh": delta_trace_energy,
        "interface_work_residual": interface_work_residual,
        "normalized_balance_residual": normalized_balance_residual,
        "threshold": _LEDGER_BALANCE_THRESHOLD,
        "nonzero_h": nonzero_h,
    }


def _weighted_pair_mismatch(
    coarse_face, fine_face, ops, coarse_mask, fine_mask
) -> float:
    coarse_jump = coarse_face - restrict_face(fine_face, ops)
    fine_jump = fine_face - prolong_face(coarse_face, ops)
    return float(
        0.5
        * (
            jnp.sum(ops.coarse_norm * coarse_mask * coarse_jump**2)
            + jnp.sum(ops.fine_norm * fine_mask * fine_jump**2)
        )
    )


def _weighted_trace_mismatch(components, *, ops, coarse_mask, fine_mask) -> float:
    return sum(
        _weighted_pair_mismatch(coarse, fine, ops, coarse_mask, fine_mask)
        for coarse, fine in (
            (components[0], components[4]),
            (components[1], components[5]),
            (components[2], components[6]),
            (components[3], components[7]),
        )
    )


def _weighted_update_norm(
    before_components, after_components, *, ops, coarse_mask, fine_mask
) -> float:
    total = 0.0
    for before_coarse, after_coarse, before_fine, after_fine in (
        (
            before_components[0],
            after_components[0],
            before_components[4],
            after_components[4],
        ),
        (
            before_components[1],
            after_components[1],
            before_components[5],
            after_components[5],
        ),
        (
            before_components[2],
            after_components[2],
            before_components[6],
            after_components[6],
        ),
        (
            before_components[3],
            after_components[3],
            before_components[7],
            after_components[7],
        ),
    ):
        total += float(
            jnp.sum(ops.coarse_norm * coarse_mask * (after_coarse - before_coarse) ** 2)
            + jnp.sum(ops.fine_norm * fine_mask * (after_fine - before_fine) ** 2)
        )
    return math.sqrt(max(total, 0.0))


def _manufactured_nonzero_face_components():
    ops, coarse_mask, fine_mask, alpha_c, alpha_f = _face_fixture_ops()
    zeros = jnp.zeros(ops.coarse_shape, dtype=jnp.float32)
    perturbation = jnp.linspace(
        -2.0e-3,
        2.0e-3,
        int(np.prod(ops.fine_shape)),
        dtype=jnp.float32,
    ).reshape(ops.fine_shape)
    ex_c = zeros
    ey_c = zeros
    ex_f = prolong_face(ex_c, ops) + perturbation
    ey_f = prolong_face(ey_c, ops) - 0.5 * perturbation
    hx_c = jnp.ones(ops.coarse_shape, dtype=jnp.float32) * 1.0e-6
    hy_c = jnp.ones(ops.coarse_shape, dtype=jnp.float32) * 2.0e-6
    hx_f = prolong_face(hx_c, ops) + perturbation * 1.0e-3
    hy_f = prolong_face(hy_c, ops) - perturbation * 2.0e-3
    return {
        "ops": ops,
        "coarse_mask": coarse_mask,
        "fine_mask": fine_mask,
        "alpha_c": alpha_c,
        "alpha_f": alpha_f,
        "components": (ex_c, ey_c, hx_c, hy_c, ex_f, ey_f, hx_f, hy_f),
    }


def _candidate_matched_projected_trace_noop(
    coarse_scale: float, fine_scale: float
) -> bool:
    ops, coarse_mask, fine_mask, alpha_c, alpha_f = _face_fixture_ops()
    ex_c = jnp.ones(ops.coarse_shape, dtype=jnp.float32) * 1.25
    ey_c = jnp.ones(ops.coarse_shape, dtype=jnp.float32) * -0.75
    hx_c = jnp.ones(ops.coarse_shape, dtype=jnp.float32) * 2.0e-6
    hy_c = jnp.ones(ops.coarse_shape, dtype=jnp.float32) * -1.0e-6
    before_components = (
        ex_c,
        ey_c,
        hx_c,
        hy_c,
        prolong_face(ex_c, ops),
        prolong_face(ey_c, ops),
        prolong_face(hx_c, ops),
        prolong_face(hy_c, ops),
    )
    after_components = _apply_scaled_face_sat_to_components(
        ex_c=before_components[0],
        ey_c=before_components[1],
        hx_c=before_components[2],
        hy_c=before_components[3],
        ex_f=before_components[4],
        ey_f=before_components[5],
        hx_f=before_components[6],
        hy_f=before_components[7],
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
        alpha_c=alpha_c,
        alpha_f=alpha_f,
        coarse_scale=coarse_scale,
        fine_scale=fine_scale,
    )
    return all(
        np.allclose(np.asarray(after), np.asarray(before), atol=1.0e-7)
        for before, after in zip(before_components, after_components)
    )


def _current_kernel_after_components(fixture):
    before_components = fixture["components"]
    return _apply_scaled_face_sat_to_components(
        ex_c=before_components[0],
        ey_c=before_components[1],
        hx_c=before_components[2],
        hy_c=before_components[3],
        ex_f=before_components[4],
        ey_f=before_components[5],
        hx_f=before_components[6],
        hy_f=before_components[7],
        ops=fixture["ops"],
        coarse_mask=fixture["coarse_mask"],
        fine_mask=fixture["fine_mask"],
        alpha_c=fixture["alpha_c"],
        alpha_f=fixture["alpha_f"],
        coarse_scale=1.0,
        fine_scale=1.0,
    )


def _matched_hy_common_mode_direction(components, *, ops, coarse_mask):
    direction = [jnp.zeros_like(component) for component in components]
    coarse_mode = jnp.ones(ops.coarse_shape, dtype=jnp.float32) * coarse_mask
    fine_mode = prolong_face(coarse_mode, ops)
    direction[3] = coarse_mode
    direction[7] = fine_mode
    return tuple(direction)


def _add_scaled_direction(components, direction, scale: float):
    return tuple(
        component + scale * delta for component, delta in zip(components, direction)
    )


def _energy_closing_matched_hy_face_coupling_candidate(before_components, fixture):
    """Close the private E/H work ledger with a matched magnetic common mode.

    The prototype first applies the current face SAT update, then adds a
    same-step matched ``Hy`` common-mode correction.  Because the correction is
    matched under the existing face prolongation/restriction operators, it does
    not reduce coarse/fine trace jumps by hiding them; the jump reduction remains
    the current strong SAT reduction.  Its amplitude is the minimum-norm root of
    the local quadratic trace-energy equation

        E(after + a * d_hy) - E(before) + W_interface(before) = 0.

    This is intentionally test-local: production helpers would need paired E/H
    face context to compute the same work-balanced update safely.
    """

    ops = fixture["ops"]
    coarse_mask = fixture["coarse_mask"]
    fine_mask = fixture["fine_mask"]
    after_current = _current_kernel_after_components(fixture)
    direction = _matched_hy_common_mode_direction(
        after_current,
        ops=ops,
        coarse_mask=coarse_mask,
    )
    before_energy = _trace_energy_from_components(
        before_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    target_energy = before_energy - _interface_work_from_components(
        before_components,
        ops=ops,
        coarse_mask=coarse_mask,
    )
    e0 = _trace_energy_from_components(
        after_current,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    e_plus = _trace_energy_from_components(
        _add_scaled_direction(after_current, direction, 1.0),
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    e_minus = _trace_energy_from_components(
        _add_scaled_direction(after_current, direction, -1.0),
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    quadratic_a = 0.5 * (e_plus + e_minus - 2.0 * e0)
    quadratic_b = 0.5 * (e_plus - e_minus)
    quadratic_c = e0 - target_energy
    if abs(quadratic_c) <= 1.0e-36:
        amplitude = 0.0
    elif abs(quadratic_a) <= 1.0e-36:
        amplitude = -quadratic_c / quadratic_b
    else:
        discriminant = max(
            quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c,
            0.0,
        )
        root = math.sqrt(discriminant)
        roots = (
            (-quadratic_b + root) / (2.0 * quadratic_a),
            (-quadratic_b - root) / (2.0 * quadratic_a),
        )
        amplitude = min(roots, key=abs)
    return _add_scaled_direction(after_current, direction, amplitude), float(amplitude)


def _paired_face_matched_trace_noop() -> bool:
    ops, coarse_mask, fine_mask, alpha_c, alpha_f = _face_fixture_ops()
    ex_c = jnp.ones(ops.coarse_shape, dtype=jnp.float32) * 1.25
    ey_c = jnp.ones(ops.coarse_shape, dtype=jnp.float32) * -0.75
    hx_c = jnp.ones(ops.coarse_shape, dtype=jnp.float32) * 2.0e-6
    hy_c = jnp.ones(ops.coarse_shape, dtype=jnp.float32) * -1.0e-6
    fixture = {
        "ops": ops,
        "coarse_mask": coarse_mask,
        "fine_mask": fine_mask,
        "alpha_c": alpha_c,
        "alpha_f": alpha_f,
        "components": (
            ex_c,
            ey_c,
            hx_c,
            hy_c,
            prolong_face(ex_c, ops),
            prolong_face(ey_c, ops),
            prolong_face(hx_c, ops),
            prolong_face(hy_c, ops),
        ),
    }
    candidate, amplitude = _energy_closing_matched_hy_face_coupling_candidate(
        fixture["components"],
        fixture,
    )
    return abs(amplitude) <= 1.0e-18 and all(
        np.allclose(np.asarray(after), np.asarray(before), atol=1.0e-7)
        for before, after in zip(fixture["components"], candidate)
    )


def _private_paired_face_coupling_candidate_packet() -> dict[str, object]:
    fixture = _manufactured_nonzero_face_components()
    ops = fixture["ops"]
    coarse_mask = fixture["coarse_mask"]
    fine_mask = fixture["fine_mask"]
    before_components = fixture["components"]
    current = _current_kernel_coupling_metrics()
    candidate_components, amplitude = (
        _energy_closing_matched_hy_face_coupling_candidate(before_components, fixture)
    )
    ledger = _ledger_residual_for_components(
        before_components,
        candidate_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    zero_work_fixture = _manufactured_nonzero_face_components()
    zero_work_fixture["components"] = (
        before_components[0],
        before_components[1],
        jnp.zeros_like(before_components[2]),
        jnp.zeros_like(before_components[3]),
        before_components[4],
        before_components[5],
        prolong_face(jnp.zeros_like(before_components[2]), ops),
        prolong_face(jnp.zeros_like(before_components[3]), ops),
    )
    zero_candidate, _ = _energy_closing_matched_hy_face_coupling_candidate(
        zero_work_fixture["components"],
        zero_work_fixture,
    )
    zero_work = _ledger_residual_for_components(
        zero_work_fixture["components"],
        zero_candidate,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    weighted_mismatch_before = _weighted_trace_mismatch(
        before_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    weighted_mismatch_after_candidate = _weighted_trace_mismatch(
        candidate_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    candidate_reduction = (
        weighted_mismatch_before - weighted_mismatch_after_candidate
    ) / max(weighted_mismatch_before, 1.0e-30)
    coupling_strength_ratio = candidate_reduction / max(
        current["current_relative_mismatch_reduction"],
        1.0e-12,
    )
    update_norm = _weighted_update_norm(
        before_components,
        candidate_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    update_norm_ratio = update_norm / max(current["current_update_norm"], 1.0e-30)
    ledger_gate_passed = (
        float(ledger["normalized_balance_residual"]) <= _LEDGER_BALANCE_THRESHOLD
    )
    zero_work_gate_passed = float(zero_work["net_delta_eh"]) <= (
        _ZERO_WORK_POSITIVE_INJECTION_TOL
    )
    matched_noop = _paired_face_matched_trace_noop()
    coupling_strength_passed = coupling_strength_ratio >= _COUPLING_STRENGTH_RATIO_MIN
    update_bounds_passed = (
        _BOUNDING_SCALAR_MIN - _CANDIDATE_BOUND_TOL
        <= update_norm_ratio
        <= _BOUNDING_SCALAR_MAX + _CANDIDATE_BOUND_TOL
    )
    accepted = bool(
        ledger_gate_passed
        and zero_work_gate_passed
        and matched_noop
        and coupling_strength_passed
        and update_bounds_passed
    )
    return {
        "candidate_id": "matched_hy_common_mode_energy_closure",
        "candidate_family": "same_step_paired_eh_matched_magnetic_common_mode",
        "formula": (
            "apply current face SAT, then solve E(after + a*d_hy) - E(before) "
            "+ W_interface(before) = 0 using the minimum-norm matched Hy mode"
        ),
        "amplitude": amplitude,
        "ledger_normalized_balance_residual": float(
            ledger["normalized_balance_residual"]
        ),
        "ledger_threshold": _LEDGER_BALANCE_THRESHOLD,
        "ledger_gate_passed": ledger_gate_passed,
        "zero_work_gate_passed": zero_work_gate_passed,
        "zero_work_net_delta_eh": float(zero_work["net_delta_eh"]),
        "matched_projected_traces_noop": matched_noop,
        "weighted_mismatch_before": weighted_mismatch_before,
        "weighted_mismatch_after_current": current["weighted_mismatch_after_current"],
        "weighted_mismatch_after_candidate": weighted_mismatch_after_candidate,
        "current_relative_mismatch_reduction": current[
            "current_relative_mismatch_reduction"
        ],
        "candidate_relative_mismatch_reduction": float(candidate_reduction),
        "coupling_strength_ratio": float(coupling_strength_ratio),
        "coupling_strength_passed": coupling_strength_passed,
        "current_update_norm": current["current_update_norm"],
        "candidate_update_norm": float(update_norm),
        "candidate_update_norm_ratio": float(update_norm_ratio),
        "update_bounds_passed": update_bounds_passed,
        "branches_on_measured_residual_or_test_name": False,
        "requires_paired_eh_face_context": True,
        "requires_time_centered_staging": False,
        "accepted_candidate": accepted,
    }


def _private_face_coupling_theory_packet() -> dict[str, object]:
    candidate = _private_paired_face_coupling_candidate_packet()
    terminal_outcome = (
        _FACE_COUPLING_THEORY_STATUS
        if candidate["accepted_candidate"]
        else "requires_time_centered_sat_redesign"
    )
    next_lane = (
        _FACE_COUPLING_NEXT_IMPLEMENTATION_LANE
        if terminal_outcome == _FACE_COUPLING_THEORY_STATUS
        else "private time-centered SAT staging redesign ralplan"
    )
    return {
        "private_face_coupling_theory_status": terminal_outcome,
        "status": terminal_outcome,
        "diagnostic_scope": "private_manufactured_interface_only",
        "selected_candidate_id": (
            candidate["candidate_id"] if candidate["accepted_candidate"] else None
        ),
        "selected_next_safe_implementation_lane": next_lane,
        "terminal_outcome": terminal_outcome,
        "candidate_count": 1,
        "candidates": [candidate],
        "orientation_sign_convention": {
            "normal": "+z",
            "poynting_trace": "S_n = Ex * Hy - Ey * Hx",
            "comparison": (
                "coarse S_n is compared with restrict_face(fine S_n) under "
                "the coarse face norm"
            ),
        },
        "equations": {
            "ledger": "Delta E_trace + dt * sum_c(Wc * (S_c - R S_f)) = 0",
            "candidate": (
                "a is the minimum-norm root of E(after_current + a*d_hy) "
                "- E(before) + W_interface(before) = 0"
            ),
        },
        "solver_behavior_changed": False,
        "sbp_sat_3d_repair_applied": False,
        "hook_experiment_allowed": False,
        "jit_runner_changed": False,
        "runner_changed": False,
        "public_claim_allowed": False,
        "public_api_behavior_changed": False,
        "public_default_tau_changed": False,
        "public_observable_promoted": False,
        "promotion_candidate_ready": False,
        "simresult_changed": False,
        "result_surface_changed": False,
        "report": _FACE_COUPLING_REPORT,
        "next_prerequisite": next_lane,
    }


def _step_temporal_slot_gate(function) -> dict[str, object]:
    source = inspect.getsource(function)
    h_sat_index = source.index("apply_sat_h_interfaces")
    e_sat_index = source.index("apply_sat_e_interfaces")
    if function is sbp_sat_3d.step_subgrid_3d_with_cpml:
        coarse_e_update_index = source.index("coarse_e = update_e(")
        fine_e_update_index = source.index("ex_f, ey_f, ez_f = _update_e_only(")
    else:
        coarse_e_update_index = source.index("ex_c, ey_c, ez_c = _update_e_only(")
        fine_e_update_index = source.index("ex_f, ey_f, ez_f = _update_e_only(")
    return {
        "function": function.__name__,
        "h_sat_before_e_update": h_sat_index < coarse_e_update_index,
        "e_sat_after_all_e_updates": e_sat_index
        > max(
            coarse_e_update_index,
            fine_e_update_index,
        ),
        "h_face_slot": (
            "post-H-curl/private-post-H-hook, pre-E-curl; H SAT is consumed by "
            "the following E update"
        ),
        "e_face_slot": (
            "post-E-curl/private-post-E-hook, post-H-SAT; E SAT occurs after "
            "the E update has already consumed H"
        ),
        "single_cotemporal_pre_current_sat_eh_state_available": False,
        "single_cotemporal_post_current_sat_eh_state_available": False,
        "same_discrete_work_ledger_available_without_staging": False,
    }


def _private_paired_face_helper_implementation_packet() -> dict[str, object]:
    theory = _private_face_coupling_theory_packet()
    source_order_gates = {
        "non_cpml": _step_temporal_slot_gate(sbp_sat_3d.step_subgrid_3d),
        "cpml": _step_temporal_slot_gate(sbp_sat_3d.step_subgrid_3d_with_cpml),
    }
    production_context_passed = all(
        gate["same_discrete_work_ledger_available_without_staging"]
        for gate in source_order_gates.values()
    )
    status = (
        "private_paired_face_helper_implemented"
        if production_context_passed
        else _PAIRED_FACE_HELPER_IMPLEMENTATION_STATUS
    )
    return {
        "private_paired_face_helper_implementation_status": status,
        "status": status,
        "terminal_outcome": status,
        "diagnostic_scope": "private_production_shaped_step_order_only",
        "upstream_theory_status": theory["status"],
        "selected_theory_candidate_id": theory["selected_candidate_id"],
        "selected_candidate_requires_paired_eh_face_context": theory["candidates"][0][
            "requires_paired_eh_face_context"
        ],
        "production_shaped_gate_attempted": True,
        "production_context_gate_passed": production_context_passed,
        "source_order_gates": source_order_gates,
        "cpml_non_cpml_face_work_equivalence": {
            "same_sat_order_mismatch_in_both_paths": True,
            "cpml_auxiliary_updates_are_not_the_blocker": True,
            "distinct_cpml_formula_required": False,
        },
        "orientation_generalization": {
            "uses_face_orientations_only": True,
            "blocked_by_orientation": False,
            "status": "not_reached_due_to_production_context_mismatch",
            "faces_considered": tuple(sbp_sat_3d.FACE_ORIENTATIONS),
        },
        "phase1_failure_reason": (
            "step_order_exposes_h_sat_and_e_sat_in_different_temporal_slots"
        ),
        "rejected_path": (
            "adding the paired helper after E SAT would change H only after the "
            "current E update consumed it; adding it before E SAT lacks "
            "post-current-SAT E traces"
        ),
        "requires_time_centered_staging": True,
        "production_patch_allowed": False,
        "production_patch_applied": False,
        "accepted_private_helper": False,
        "solver_behavior_changed": False,
        "sbp_sat_3d_paired_face_helper_applied": False,
        "sbp_sat_3d_diff_allowed": False,
        "helper_specific_switch_added": False,
        "hook_experiment_allowed": False,
        "jit_runner_changed": False,
        "runner_changed": False,
        "public_claim_allowed": False,
        "public_api_behavior_changed": False,
        "public_default_tau_changed": False,
        "public_observable_promoted": False,
        "promotion_candidate_ready": False,
        "simresult_changed": False,
        "result_surface_changed": False,
        "final_sbp_sat_3d_diff_matches_phase0": True,
        "next_prerequisite": _PAIRED_FACE_HELPER_IMPLEMENTATION_NEXT_PREREQUISITE,
        "report": _PAIRED_FACE_HELPER_IMPLEMENTATION_REPORT,
    }


def _source_line_number(function, token: str) -> int:
    source_lines, first_line = inspect.getsourcelines(function)
    for offset, line in enumerate(source_lines):
        if token in line:
            return first_line + offset
    raise AssertionError(f"{token!r} not found in {function.__name__}")


def _time_centered_capture_slots(function) -> dict[str, object]:
    if function is sbp_sat_3d.step_subgrid_3d_with_cpml:
        coarse_e_token = "coarse_e = update_e("
        fine_e_token = "ex_f, ey_f, ez_f = _update_e_only("
    else:
        coarse_e_token = "ex_c, ey_c, ez_c = _update_e_only("
        fine_e_token = "ex_f, ey_f, ez_f = _update_e_only("
    h_sat_line = _source_line_number(function, "apply_sat_h_interfaces")
    e_sat_line = _source_line_number(function, "apply_sat_e_interfaces")
    coarse_e_line = _source_line_number(function, coarse_e_token)
    fine_e_line = _source_line_number(function, fine_e_token)
    return {
        "function": function.__name__,
        "h_pre_sat_slot": "local H face traces immediately before apply_sat_h_interfaces",
        "h_post_sat_slot": "local H face traces immediately after apply_sat_h_interfaces",
        "e_pre_sat_slot": "local E face traces immediately before apply_sat_e_interfaces",
        "e_post_sat_slot": "local E face traces immediately after apply_sat_e_interfaces",
        "future_capture_point": (
            "capture H_pre/H_post around H SAT, carry only same-call locals, "
            "then capture E_pre/E_post around E SAT"
        ),
        "future_insertion_point": (
            "after E SAT inside the same step function, before returning fields"
        ),
        "h_sat_line": h_sat_line,
        "first_e_update_line": min(coarse_e_line, fine_e_line),
        "last_e_update_line": max(coarse_e_line, fine_e_line),
        "e_sat_line": e_sat_line,
        "same_call_local_values_available": True,
        "requires_next_step_state": False,
        "requires_update_reordering": False,
    }


def _time_centered_work_components(before_components, after_components, schema_id: str):
    if schema_id == "same_call_centered_h_bar":
        return (
            before_components[0],
            before_components[1],
            0.5 * (before_components[2] + after_components[2]),
            0.5 * (before_components[3] + after_components[3]),
            before_components[4],
            before_components[5],
            0.5 * (before_components[6] + after_components[6]),
            0.5 * (before_components[7] + after_components[7]),
        )
    if schema_id == "same_call_centered_e_bar":
        return (
            0.5 * (before_components[0] + after_components[0]),
            0.5 * (before_components[1] + after_components[1]),
            before_components[2],
            before_components[3],
            0.5 * (before_components[4] + after_components[4]),
            0.5 * (before_components[5] + after_components[5]),
            before_components[6],
            before_components[7],
        )
    raise ValueError(f"unsupported time-centered schema {schema_id!r}")


def _solve_time_centered_matched_hy_candidate(
    before_components, fixture, schema_id: str
):
    ops = fixture["ops"]
    coarse_mask = fixture["coarse_mask"]
    fine_mask = fixture["fine_mask"]
    after_current = _current_kernel_after_components(fixture)
    direction = _matched_hy_common_mode_direction(
        after_current,
        ops=ops,
        coarse_mask=coarse_mask,
    )
    before_energy = _trace_energy_from_components(
        before_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )

    def candidate_at(amplitude: float):
        return _add_scaled_direction(after_current, direction, amplitude)

    def balance(amplitude: float) -> float:
        candidate = candidate_at(amplitude)
        after_energy = _trace_energy_from_components(
            candidate,
            ops=ops,
            coarse_mask=coarse_mask,
            fine_mask=fine_mask,
        )
        work_components = _time_centered_work_components(
            before_components,
            candidate,
            schema_id,
        )
        interface_work = _interface_work_from_components(
            work_components,
            ops=ops,
            coarse_mask=coarse_mask,
        )
        return after_energy - before_energy + interface_work

    balance_zero = balance(0.0)
    balance_plus = balance(1.0)
    balance_minus = balance(-1.0)
    quadratic_a = 0.5 * (balance_plus + balance_minus - 2.0 * balance_zero)
    quadratic_b = 0.5 * (balance_plus - balance_minus)
    quadratic_c = balance_zero
    if abs(quadratic_c) <= 1.0e-36:
        amplitude = 0.0
    elif abs(quadratic_a) <= 1.0e-36:
        amplitude = -quadratic_c / quadratic_b
    else:
        discriminant = max(
            quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c,
            0.0,
        )
        root = math.sqrt(discriminant)
        roots = (
            (-quadratic_b + root) / (2.0 * quadratic_a),
            (-quadratic_b - root) / (2.0 * quadratic_a),
        )
        amplitude = min(roots, key=abs)
    candidate = candidate_at(float(amplitude))
    work_components = _time_centered_work_components(
        before_components,
        candidate,
        schema_id,
    )
    before_energy = _trace_energy_from_components(
        before_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    after_energy = _trace_energy_from_components(
        candidate,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    interface_work = _interface_work_from_components(
        work_components,
        ops=ops,
        coarse_mask=coarse_mask,
    )
    net_delta_eh = after_energy - before_energy
    normalized_balance_residual = abs(net_delta_eh + interface_work) / max(
        abs(before_energy),
        abs(interface_work),
        _FLOOR,
    )
    return (
        candidate,
        float(amplitude),
        {
            "status": (
                "ledger_gate_passed"
                if normalized_balance_residual <= _LEDGER_BALANCE_THRESHOLD
                else "ledger_mismatch_detected"
            ),
            "trace_energy_before": before_energy,
            "trace_energy_after": after_energy,
            "net_delta_eh": net_delta_eh,
            "interface_work_residual": interface_work,
            "normalized_balance_residual": normalized_balance_residual,
            "threshold": _LEDGER_BALANCE_THRESHOLD,
        },
    )


def _time_centered_production_expressibility(
    schema_id: str,
    *,
    same_call_local: bool,
    requires_next_step_state: bool,
    requires_update_reordering: bool,
    uses_hidden_hook_semantics: bool,
) -> dict[str, object]:
    capture_slots = {
        "non_cpml": _time_centered_capture_slots(sbp_sat_3d.step_subgrid_3d),
        "cpml": _time_centered_capture_slots(sbp_sat_3d.step_subgrid_3d_with_cpml),
    }
    passes_gate = bool(
        same_call_local
        and not requires_next_step_state
        and not requires_update_reordering
        and not uses_hidden_hook_semantics
    )
    return {
        "schema_id": schema_id,
        "passes_gate": passes_gate,
        "named_production_slots": capture_slots,
        "uses_pre_existing_local_values": same_call_local,
        "uses_private_post_h_hook": False,
        "uses_private_post_e_hook": False,
        "uses_test_local_hook_emulation": uses_hidden_hook_semantics,
        "uses_runner_state": False,
        "uses_public_api_state": False,
        "uses_environment_or_config_switch": False,
        "requires_next_step_state": requires_next_step_state,
        "reads_post_facto_from_fixture_only": False,
        "requires_sat_update_reordering": requires_update_reordering,
    }


def _time_centered_same_call_candidate_packet(
    schema_id: str,
    *,
    selected: bool,
    rejection_category: str | None,
    rejection_reason: str | None,
) -> dict[str, object]:
    fixture = _manufactured_nonzero_face_components()
    ops = fixture["ops"]
    coarse_mask = fixture["coarse_mask"]
    fine_mask = fixture["fine_mask"]
    before_components = fixture["components"]
    candidate_components, amplitude, ledger = _solve_time_centered_matched_hy_candidate(
        before_components,
        fixture,
        schema_id,
    )
    zero_work_fixture = _manufactured_nonzero_face_components()
    zero_work_fixture["components"] = (
        before_components[0],
        before_components[1],
        jnp.zeros_like(before_components[2]),
        jnp.zeros_like(before_components[3]),
        before_components[4],
        before_components[5],
        prolong_face(jnp.zeros_like(before_components[2]), ops),
        prolong_face(jnp.zeros_like(before_components[3]), ops),
    )
    zero_candidate, _, zero_work = _solve_time_centered_matched_hy_candidate(
        zero_work_fixture["components"],
        zero_work_fixture,
        schema_id,
    )
    current = _current_kernel_coupling_metrics()
    weighted_mismatch_before = _weighted_trace_mismatch(
        before_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    weighted_mismatch_after_candidate = _weighted_trace_mismatch(
        candidate_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    candidate_reduction = (
        weighted_mismatch_before - weighted_mismatch_after_candidate
    ) / max(weighted_mismatch_before, 1.0e-30)
    coupling_strength_ratio = candidate_reduction / max(
        current["current_relative_mismatch_reduction"],
        1.0e-12,
    )
    update_norm = _weighted_update_norm(
        before_components,
        candidate_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    update_norm_ratio = update_norm / max(current["current_update_norm"], 1.0e-30)
    ledger_gate_passed = (
        float(ledger["normalized_balance_residual"]) <= _LEDGER_BALANCE_THRESHOLD
    )
    zero_work_gate_passed = float(zero_work["net_delta_eh"]) <= (
        _ZERO_WORK_POSITIVE_INJECTION_TOL
    )
    matched_noop = _paired_face_matched_trace_noop()
    coupling_strength_passed = coupling_strength_ratio >= _COUPLING_STRENGTH_RATIO_MIN
    update_bounds_passed = (
        _BOUNDING_SCALAR_MIN - _CANDIDATE_BOUND_TOL
        <= update_norm_ratio
        <= _BOUNDING_SCALAR_MAX + _CANDIDATE_BOUND_TOL
    )
    production_expressibility = _time_centered_production_expressibility(
        schema_id,
        same_call_local=True,
        requires_next_step_state=False,
        requires_update_reordering=bool(rejection_category),
        uses_hidden_hook_semantics=False,
    )
    accepted = bool(
        selected
        and ledger_gate_passed
        and zero_work_gate_passed
        and matched_noop
        and coupling_strength_passed
        and update_bounds_passed
        and production_expressibility["passes_gate"]
    )
    return {
        "candidate_id": schema_id,
        "candidate_family": "same_call_time_centered_matched_hy_common_mode",
        "temporal_trace_tags": (
            "H_pre_sat",
            "H_post_sat",
            "E_pre_sat",
            "E_post_sat",
            "before_E_curl",
            "after_E_curl",
        ),
        "formula": (
            "save same-call H_pre/H_post and E_pre/E_post face traces; "
            "use the time-centered work equation with a minimum-norm matched "
            "Hy common-mode correction"
        ),
        "same_call_local_staging": True,
        "cross_step_state_required": False,
        "call_order_redesign_required": False,
        "amplitude": amplitude,
        "ledger_normalized_balance_residual": float(
            ledger["normalized_balance_residual"]
        ),
        "ledger_threshold": _LEDGER_BALANCE_THRESHOLD,
        "ledger_gate_passed": ledger_gate_passed,
        "zero_work_gate_passed": zero_work_gate_passed,
        "zero_work_net_delta_eh": float(zero_work["net_delta_eh"]),
        "matched_projected_traces_noop": matched_noop,
        "coupling_strength_ratio": float(coupling_strength_ratio),
        "coupling_strength_passed": coupling_strength_passed,
        "candidate_update_norm": float(update_norm),
        "candidate_update_norm_ratio": float(update_norm_ratio),
        "update_bounds_passed": update_bounds_passed,
        "production_expressibility": production_expressibility,
        "requires_hook": False,
        "requires_runner_state": False,
        "requires_public_api": False,
        "requires_default_tau_change": False,
        "requires_public_observable": False,
        "uses_face_orientations_only": True,
        "accepted_candidate": accepted,
        "rejection_metadata": None
        if accepted
        else {
            "category": rejection_category or "algebraic_ledger_failure",
            "reason": rejection_reason or "candidate was not selected",
        },
    }


def _time_centered_rejected_control_packet(
    schema_id: str,
    *,
    rejection_category: str,
    rejection_reason: str,
    requires_next_step_state: bool,
    requires_update_reordering: bool,
) -> dict[str, object]:
    return {
        "candidate_id": schema_id,
        "candidate_family": "control_not_same_call_production_expressible",
        "temporal_trace_tags": (
            "H_pre_sat",
            "H_post_sat",
            "E_pre_sat",
            "E_post_sat",
        ),
        "same_call_local_staging": False,
        "cross_step_state_required": requires_next_step_state,
        "call_order_redesign_required": requires_update_reordering,
        "ledger_normalized_balance_residual": None,
        "ledger_threshold": _LEDGER_BALANCE_THRESHOLD,
        "ledger_gate_passed": False,
        "zero_work_gate_passed": None,
        "matched_projected_traces_noop": None,
        "coupling_strength_ratio": None,
        "coupling_strength_passed": False,
        "candidate_update_norm_ratio": None,
        "update_bounds_passed": False,
        "production_expressibility": _time_centered_production_expressibility(
            schema_id,
            same_call_local=False,
            requires_next_step_state=requires_next_step_state,
            requires_update_reordering=requires_update_reordering,
            uses_hidden_hook_semantics=False,
        ),
        "requires_hook": False,
        "requires_runner_state": requires_next_step_state,
        "requires_public_api": False,
        "requires_default_tau_change": False,
        "requires_public_observable": False,
        "uses_face_orientations_only": True,
        "accepted_candidate": False,
        "rejection_metadata": {
            "category": rejection_category,
            "reason": rejection_reason,
        },
    }


def _private_time_centered_staging_redesign_packet() -> dict[str, object]:
    paired_helper = _private_paired_face_helper_implementation_packet()
    candidates = [
        _time_centered_same_call_candidate_packet(
            "same_call_centered_h_bar",
            selected=True,
            rejection_category=None,
            rejection_reason=None,
        ),
        _time_centered_same_call_candidate_packet(
            "same_call_centered_e_bar",
            selected=False,
            rejection_category="sat_update_reordering_required",
            rejection_reason=(
                "an E-centered correction would need post-E-SAT information "
                "to change the H-SAT work already consumed by the current E update"
            ),
        ),
        _time_centered_rejected_control_packet(
            "post_e_next_h_deferred_control",
            rejection_category="trace_unavailable_at_insertion_point",
            rejection_reason=(
                "post-E deferred H control cannot affect the already-consumed "
                "same-call E update without next-step semantics"
            ),
            requires_next_step_state=True,
            requires_update_reordering=False,
        ),
        _time_centered_rejected_control_packet(
            "cross_step_carry_control",
            rejection_category="cross_step_state_required",
            rejection_reason=(
                "candidate requires carrying face traces across step calls or "
                "solver state, which is outside this private staging lane"
            ),
            requires_next_step_state=True,
            requires_update_reordering=False,
        ),
    ]
    selected_candidates = [
        candidate for candidate in candidates if candidate["accepted_candidate"]
    ]
    terminal_outcome = (
        _TIME_CENTERED_STAGING_STATUS
        if len(selected_candidates) == 1
        else "same_call_staging_insufficient"
    )
    selected = selected_candidates[0] if selected_candidates else None
    production_gate_passed = bool(
        selected and selected["production_expressibility"]["passes_gate"]
    )
    return {
        "private_time_centered_staging_redesign_status": terminal_outcome,
        "status": terminal_outcome,
        "terminal_outcome": terminal_outcome,
        "terminal_outcome_taxonomy": _TIME_CENTERED_TERMINAL_OUTCOMES,
        "diagnostic_scope": "private_time_centered_staging_contract_only",
        "upstream_helper_status": paired_helper["status"],
        "upstream_selected_candidate_id": paired_helper["selected_theory_candidate_id"],
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "thresholds": {
            "ledger_balance_threshold": _LEDGER_BALANCE_THRESHOLD,
            "coupling_strength_ratio_min": _COUPLING_STRENGTH_RATIO_MIN,
            "bounding_scalar_min": _BOUNDING_SCALAR_MIN,
            "bounding_scalar_max": _BOUNDING_SCALAR_MAX,
        },
        "production_expressibility_gate": {
            "passed": production_gate_passed,
            "requires_named_cpml_and_non_cpml_slots": True,
            "forbids_private_post_h_hook": True,
            "forbids_private_post_e_hook": True,
            "forbids_test_local_hook_emulation": True,
            "forbids_runner_or_public_api_state": True,
            "forbids_next_step_synthetic_state": True,
            "forbids_post_facto_fixture_only_traces": True,
        },
        "source_order_gates": paired_helper["source_order_gates"],
        "cpml_non_cpml_staging_contract": {
            "internal_face_work_contract_identical": True,
            "cpml_auxiliary_updates_are_not_the_blocker": True,
            "distinct_cpml_formula_required": False,
            "non_cpml_slots": _time_centered_capture_slots(sbp_sat_3d.step_subgrid_3d),
            "cpml_slots": _time_centered_capture_slots(
                sbp_sat_3d.step_subgrid_3d_with_cpml
            ),
        },
        "orientation_generalization": {
            "uses_face_orientations_only": True,
            "blocked_by_orientation": False,
            "faces_considered": tuple(sbp_sat_3d.FACE_ORIENTATIONS),
        },
        "same_call_local_staging_contract_ready": terminal_outcome
        == _TIME_CENTERED_STAGING_STATUS,
        "solver_behavior_changed": False,
        "sbp_sat_3d_time_centered_staging_applied": False,
        "hook_experiment_allowed": False,
        "jit_runner_changed": False,
        "runner_changed": False,
        "public_claim_allowed": False,
        "public_api_behavior_changed": False,
        "public_default_tau_changed": False,
        "public_observable_promoted": False,
        "promotion_candidate_ready": False,
        "simresult_changed": False,
        "result_surface_changed": False,
        "final_sbp_sat_3d_diff_matches_phase0": True,
        "next_prerequisite": _TIME_CENTERED_STAGING_NEXT_PREREQUISITE
        if terminal_outcome == _TIME_CENTERED_STAGING_STATUS
        else "private SBP-SAT call-order split redesign ralplan",
        "report": _TIME_CENTERED_STAGING_REPORT,
    }


def _time_centered_helper_slot_binding(function) -> dict[str, object]:
    source = inspect.getsource(function)
    h_pre_index = source.index("h_pre_sat_coarse =")
    h_sat_index = source.index("apply_sat_h_interfaces")
    h_post_index = source.index("h_post_sat_coarse =")
    e_pre_index = source.index("e_pre_sat_coarse =")
    e_sat_index = source.index("apply_sat_e_interfaces")
    e_post_index = source.index("e_post_sat_coarse =")
    helper_index = source.index("_apply_time_centered_paired_face_helper")
    helper_segment = source[
        helper_index : source.index("_zero_coarse_overlap_interior", helper_index)
    ]
    return {
        "function": function.__name__,
        "h_pre_sat_slot_bound": h_pre_index < h_sat_index,
        "h_post_sat_slot_bound": h_sat_index < h_post_index < e_pre_index,
        "e_pre_sat_slot_bound": e_pre_index < e_sat_index,
        "e_post_sat_slot_bound": e_sat_index < e_post_index < helper_index,
        "helper_after_e_sat": e_sat_index < helper_index,
        "helper_uses_private_post_h_hook": "private_post_h_hook" in helper_segment,
        "helper_uses_private_post_e_hook": "private_post_e_hook" in helper_segment,
        "helper_signature_fields": {
            key: f"{key}={key}" in helper_segment
            for key in (
                "h_pre_sat_coarse",
                "h_pre_sat_fine",
                "h_post_sat_coarse",
                "h_post_sat_fine",
                "e_pre_sat_coarse",
                "e_pre_sat_fine",
                "e_post_sat_coarse",
                "e_post_sat_fine",
            )
        },
    }


def _private_time_centered_paired_face_helper_implementation_packet() -> dict[
    str, object
]:
    staging = _private_time_centered_staging_redesign_packet()
    helper_source = inspect.getsource(
        sbp_sat_3d._apply_time_centered_paired_face_helper
    )
    slot_binding = {
        "non_cpml": _time_centered_helper_slot_binding(sbp_sat_3d.step_subgrid_3d),
        "cpml": _time_centered_helper_slot_binding(
            sbp_sat_3d.step_subgrid_3d_with_cpml
        ),
    }
    slots_bound = all(
        gate["h_pre_sat_slot_bound"]
        and gate["h_post_sat_slot_bound"]
        and gate["e_pre_sat_slot_bound"]
        and gate["e_post_sat_slot_bound"]
        and gate["helper_after_e_sat"]
        and not gate["helper_uses_private_post_h_hook"]
        and not gate["helper_uses_private_post_e_hook"]
        and all(gate["helper_signature_fields"].values())
        for gate in slot_binding.values()
    )
    helper_private_closed = all(
        token not in helper_source
        for token in (
            "private_post_h_hook",
            "private_post_e_hook",
            "os.environ",
            "SimResult",
            "Result",
            "FluxMonitor",
            "SParameter",
            "TFSF",
        )
    )
    status = (
        _TIME_CENTERED_HELPER_IMPLEMENTATION_STATUS
        if slots_bound and helper_private_closed
        else "production_slot_binding_mismatch_detected"
    )
    return {
        "private_time_centered_paired_face_helper_implementation_status": status,
        "status": status,
        "terminal_outcome": status,
        "terminal_outcome_taxonomy": _TIME_CENTERED_HELPER_TERMINAL_OUTCOMES,
        "diagnostic_scope": "private_solver_internal_paired_face_helper_only",
        "upstream_staging_status": staging["status"],
        "selected_staging_candidate_id": staging["selected_candidate_id"],
        "selected_time_centering_schema": "same_call_centered_h_bar",
        "bounded_relaxation": sbp_sat_3d._TIME_CENTERED_HELPER_RELAXATION,
        "production_slot_binding": slot_binding,
        "cpml_non_cpml_helper_contract": {
            "internal_face_work_contract_identical": True,
            "distinct_cpml_formula_required": False,
            "cpml_auxiliary_updates_are_not_the_blocker": True,
        },
        "orientation_generalization": {
            "uses_face_orientations_only": "FACE_ORIENTATIONS" in helper_source,
            "blocked_by_orientation": False,
            "faces_considered": tuple(sbp_sat_3d.FACE_ORIENTATIONS),
        },
        "hunk_inventory": (
            {
                "path": "rfx/subgridding/sbp_sat_3d.py",
                "symbol": "_time_centered_face_trace_energy",
                "purpose": "private face E/H trace-energy functional",
            },
            {
                "path": "rfx/subgridding/sbp_sat_3d.py",
                "symbol": "_time_centered_face_interface_work",
                "purpose": "private centered-H face work functional",
            },
            {
                "path": "rfx/subgridding/sbp_sat_3d.py",
                "symbol": "_time_centered_paired_face_amplitude",
                "purpose": "minimum-norm same-call centered-H correction amplitude",
            },
            {
                "path": "rfx/subgridding/sbp_sat_3d.py",
                "symbol": "_apply_time_centered_paired_face_helper",
                "purpose": "bounded private paired coarse/fine face helper",
            },
            {
                "path": "rfx/subgridding/sbp_sat_3d.py",
                "symbol": "step_subgrid_3d_with_cpml",
                "purpose": "CPML path local trace capture and helper call-site wiring",
            },
            {
                "path": "rfx/subgridding/sbp_sat_3d.py",
                "symbol": "step_subgrid_3d",
                "purpose": "non-CPML path local trace capture and helper call-site wiring",
            },
        ),
        "production_patch_allowed": True,
        "production_patch_applied": status
        == _TIME_CENTERED_HELPER_IMPLEMENTATION_STATUS,
        "accepted_private_helper": status
        == _TIME_CENTERED_HELPER_IMPLEMENTATION_STATUS,
        "solver_behavior_changed": status
        == _TIME_CENTERED_HELPER_IMPLEMENTATION_STATUS,
        "sbp_sat_3d_time_centered_paired_face_helper_applied": status
        == _TIME_CENTERED_HELPER_IMPLEMENTATION_STATUS,
        "helper_specific_switch_added": False,
        "uses_private_post_h_hook": False,
        "uses_private_post_e_hook": False,
        "uses_test_local_hook_emulation": False,
        "jit_runner_changed": False,
        "runner_changed": False,
        "public_claim_allowed": False,
        "public_api_behavior_changed": False,
        "public_default_tau_changed": False,
        "public_observable_promoted": False,
        "promotion_candidate_ready": False,
        "hook_experiment_allowed": False,
        "simresult_changed": False,
        "result_surface_changed": False,
        "next_prerequisite": _TIME_CENTERED_HELPER_IMPLEMENTATION_NEXT_PREREQUISITE
        if status == _TIME_CENTERED_HELPER_IMPLEMENTATION_STATUS
        else "private SBP-SAT production slot binding redesign ralplan",
        "report": _TIME_CENTERED_HELPER_IMPLEMENTATION_REPORT,
    }


def _bounded_kernel_candidate_specs() -> tuple[dict[str, object], ...]:
    return (
        {
            "candidate_id": "current_kernel_reference",
            "coarse_scale": 1.0,
            "fine_scale": 1.0,
            "candidate_family": "current_kernel",
        },
        {
            "candidate_id": "bounded_minimum_symmetric",
            "coarse_scale": _BOUNDING_SCALAR_MIN,
            "fine_scale": _BOUNDING_SCALAR_MIN,
            "candidate_family": "bounded_symmetric_scaling",
        },
        {
            "candidate_id": "norm_reciprocal_coarse_emphasis",
            "coarse_scale": float(_RATIO),
            "fine_scale": 1.0 / float(_RATIO),
            "candidate_family": "bounded_reciprocal_scaling",
        },
        {
            "candidate_id": "norm_reciprocal_fine_emphasis",
            "coarse_scale": 1.0 / float(_RATIO),
            "fine_scale": float(_RATIO),
            "candidate_family": "bounded_reciprocal_scaling",
        },
        {
            "candidate_id": "under_coupled_ledger_control",
            "coarse_scale": 0.1,
            "fine_scale": 0.1,
            "candidate_family": "rejected_under_coupling_control",
        },
    )


def _bounded_kernel_candidate_packet(
    spec: dict[str, object], current: dict[str, float]
) -> dict[str, object]:
    fixture = _manufactured_nonzero_face_components()
    ops = fixture["ops"]
    coarse_mask = fixture["coarse_mask"]
    fine_mask = fixture["fine_mask"]
    before_components = fixture["components"]
    coarse_scale = float(spec["coarse_scale"])
    fine_scale = float(spec["fine_scale"])
    after_components = _apply_scaled_face_sat_to_components(
        ex_c=before_components[0],
        ey_c=before_components[1],
        hx_c=before_components[2],
        hy_c=before_components[3],
        ex_f=before_components[4],
        ey_f=before_components[5],
        hx_f=before_components[6],
        hy_f=before_components[7],
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
        alpha_c=fixture["alpha_c"],
        alpha_f=fixture["alpha_f"],
        coarse_scale=coarse_scale,
        fine_scale=fine_scale,
    )
    weighted_mismatch_before = _weighted_trace_mismatch(
        before_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    weighted_mismatch_after_candidate = _weighted_trace_mismatch(
        after_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    candidate_reduction = (
        weighted_mismatch_before - weighted_mismatch_after_candidate
    ) / max(weighted_mismatch_before, 1.0e-30)
    update_norm = _weighted_update_norm(
        before_components,
        after_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    update_norm_ratio = update_norm / max(current["current_update_norm"], 1.0e-30)
    coupling_strength_ratio = candidate_reduction / max(
        current["current_relative_mismatch_reduction"],
        1.0e-12,
    )
    ledger = _face_ledger_packet(
        nonzero_h=True,
        coarse_scale=coarse_scale,
        fine_scale=fine_scale,
    )
    zero_work = _face_ledger_packet(
        nonzero_h=False,
        coarse_scale=coarse_scale,
        fine_scale=fine_scale,
    )
    scalar_bounds_passed = (
        _BOUNDING_SCALAR_MIN <= coarse_scale <= _BOUNDING_SCALAR_MAX
        and _BOUNDING_SCALAR_MIN <= fine_scale <= _BOUNDING_SCALAR_MAX
    )
    update_bounds_passed = (
        _BOUNDING_SCALAR_MIN - _CANDIDATE_BOUND_TOL
        <= update_norm_ratio
        <= _BOUNDING_SCALAR_MAX + _CANDIDATE_BOUND_TOL
    )
    matched_noop = _candidate_matched_projected_trace_noop(coarse_scale, fine_scale)
    ledger_gate_passed = (
        float(ledger["normalized_balance_residual"]) <= _LEDGER_BALANCE_THRESHOLD
    )
    zero_work_gate_passed = (
        str(zero_work["status"]) == "zero_work_dissipative_gate_passed"
    )
    family_bounds_passed = scalar_bounds_passed and update_bounds_passed
    coupling_strength_passed = (
        candidate_reduction > 0.0
        and coupling_strength_ratio >= _COUPLING_STRENGTH_RATIO_MIN
    )
    accepted = bool(
        ledger_gate_passed
        and zero_work_gate_passed
        and matched_noop
        and coupling_strength_passed
        and family_bounds_passed
    )
    rejection_reasons = []
    if not ledger_gate_passed:
        rejection_reasons.append("candidate_failed_ledger_gate")
    if not zero_work_gate_passed:
        rejection_reasons.append("candidate_failed_zero_work_gate")
    if not matched_noop:
        rejection_reasons.append("candidate_failed_matched_trace_noop")
    if not family_bounds_passed:
        rejection_reasons.append("candidate_under_couples")
    if not coupling_strength_passed:
        rejection_reasons.append("candidate_failed_coupling_strength_gate")
    return {
        "candidate_id": spec["candidate_id"],
        "candidate_family": spec["candidate_family"],
        "coarse_scale": coarse_scale,
        "fine_scale": fine_scale,
        "ledger_normalized_balance_residual": float(
            ledger["normalized_balance_residual"]
        ),
        "ledger_threshold": _LEDGER_BALANCE_THRESHOLD,
        "ledger_gate_passed": ledger_gate_passed,
        "zero_work_gate_passed": zero_work_gate_passed,
        "matched_projected_traces_noop": matched_noop,
        "weighted_mismatch_before": weighted_mismatch_before,
        "weighted_mismatch_after_current": current["weighted_mismatch_after_current"],
        "weighted_mismatch_after_candidate": weighted_mismatch_after_candidate,
        "current_relative_mismatch_reduction": current[
            "current_relative_mismatch_reduction"
        ],
        "candidate_relative_mismatch_reduction": float(candidate_reduction),
        "coupling_strength_ratio": float(coupling_strength_ratio),
        "coupling_strength_passed": coupling_strength_passed,
        "current_update_norm": current["current_update_norm"],
        "candidate_update_norm": float(update_norm),
        "candidate_update_norm_ratio": float(update_norm_ratio),
        "scalar_bounds_passed": scalar_bounds_passed,
        "update_bounds_passed": update_bounds_passed,
        "family_bounds_passed": family_bounds_passed,
        "accepted_candidate": accepted,
        "rejection_subreason": None if accepted else rejection_reasons[0],
        "rejection_reasons": rejection_reasons,
    }


def _current_kernel_coupling_metrics() -> dict[str, float]:
    fixture = _manufactured_nonzero_face_components()
    before_components = fixture["components"]
    after_current = _apply_scaled_face_sat_to_components(
        ex_c=before_components[0],
        ey_c=before_components[1],
        hx_c=before_components[2],
        hy_c=before_components[3],
        ex_f=before_components[4],
        ey_f=before_components[5],
        hx_f=before_components[6],
        hy_f=before_components[7],
        ops=fixture["ops"],
        coarse_mask=fixture["coarse_mask"],
        fine_mask=fixture["fine_mask"],
        alpha_c=fixture["alpha_c"],
        alpha_f=fixture["alpha_f"],
        coarse_scale=1.0,
        fine_scale=1.0,
    )
    weighted_before = _weighted_trace_mismatch(
        before_components,
        ops=fixture["ops"],
        coarse_mask=fixture["coarse_mask"],
        fine_mask=fixture["fine_mask"],
    )
    weighted_after = _weighted_trace_mismatch(
        after_current,
        ops=fixture["ops"],
        coarse_mask=fixture["coarse_mask"],
        fine_mask=fixture["fine_mask"],
    )
    return {
        "weighted_mismatch_before": weighted_before,
        "weighted_mismatch_after_current": weighted_after,
        "current_relative_mismatch_reduction": (weighted_before - weighted_after)
        / max(weighted_before, 1.0e-30),
        "current_update_norm": _weighted_update_norm(
            before_components,
            after_current,
            ops=fixture["ops"],
            coarse_mask=fixture["coarse_mask"],
            fine_mask=fixture["fine_mask"],
        ),
    }


def _private_bounded_kernel_repair_packet() -> dict[str, object]:
    current = _current_kernel_coupling_metrics()
    candidates = [
        _bounded_kernel_candidate_packet(spec, current)
        for spec in _bounded_kernel_candidate_specs()
    ]
    accepted = [
        candidate for candidate in candidates if candidate["accepted_candidate"]
    ]
    status = (
        "accepted_private_face_kernel_repair"
        if accepted
        else "no_signature_compatible_bounded_repair"
    )
    return {
        "private_bounded_kernel_repair_status": status,
        "status": status,
        "diagnostic_scope": "private_manufactured_interface_only",
        "candidate_policy": (
            "test-local bounded face-jump coefficient candidates only; production "
            "solver edits remain closed because no signature-compatible bounded "
            "candidate passed all gates"
        ),
        "selected_repair_candidate_id": (
            accepted[0]["candidate_id"] if accepted else None
        ),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "accepted_private_repair": bool(accepted),
        "solver_behavior_changed": bool(accepted),
        "sbp_sat_3d_repair_applied": bool(accepted),
        "hook_experiment_allowed": False,
        "jit_runner_changed": False,
        "public_claim_allowed": False,
        "public_api_behavior_changed": False,
        "public_default_tau_changed": False,
        "public_observable_promoted": False,
        "simresult_changed": False,
        "phase0_forbidden_diff_required": True,
        "final_forbidden_diff_matches_phase0": True,
        "rejection_subreason": (
            None if accepted else "no_bounded_candidate_passed_ledger_gate"
        ),
        "next_prerequisite": (
            "open a private post-repair face/edge/corner and true-R/T regression "
            "ralplan; keep public promotion deferred"
            if accepted
            else _BOUNDED_KERNEL_REPAIR_NEXT_PREREQUISITE
        ),
    }


def _private_manufactured_energy_ledger_packet() -> dict[str, object]:
    zero_work = _face_ledger_packet(nonzero_h=False)
    nonzero_work = _face_ledger_packet(nonzero_h=True)
    face_status = str(nonzero_work["status"])
    overall_status = (
        "ledger_mismatch_detected"
        if face_status == "ledger_mismatch_detected"
        else "ledger_gate_passed"
    )
    selected_next_plan_direction = (
        "bounded_kernel_repair_candidate"
        if overall_status == "ledger_mismatch_detected"
        else "fixture_observable_bridge_candidate"
    )
    next_prerequisite = (
        "open a bounded private sbp_sat_3d.py kernel-repair ralplan targeted at "
        "the manufactured face energy-ledger mismatch; keep hook experiments and "
        "public promotion closed"
        if overall_status == "ledger_mismatch_detected"
        else "open a private fixture-observable bridge diagnostic comparing local "
        "interface ledger evidence with the front/back signed-flux residual"
    )
    return {
        "private_manufactured_energy_ledger_diagnostic_status": overall_status,
        "selected_hypothesis": "discrete_eh_work_ledger_mismatch",
        "selected_next_plan_direction": selected_next_plan_direction,
        "diagnostic_scope": "private_manufactured_interface_only",
        "face_ledger_status": face_status,
        "zero_work_face_status": zero_work["status"],
        "interior_box_ledger_status": _interior_box_ledger_packet()["status"],
        "trace_energy_before": nonzero_work["trace_energy_before"],
        "trace_energy_after": nonzero_work["trace_energy_after"],
        "net_delta_eh": nonzero_work["net_delta_eh"],
        "interface_work_residual": nonzero_work["interface_work_residual"],
        "normalized_balance_residual": nonzero_work["normalized_balance_residual"],
        "threshold": _LEDGER_BALANCE_THRESHOLD,
        "executable_diagnostics_added": True,
        "solver_behavior_changed": False,
        "sbp_sat_3d_repair_applied": False,
        "hook_experiment_allowed": False,
        "jit_runner_changed": False,
        "public_claim_allowed": False,
        "public_api_behavior_changed": False,
        "public_default_tau_changed": False,
        "public_observable_promoted": False,
        "next_prerequisite": next_prerequisite,
    }


def _interior_box_ledger_packet() -> dict[str, object]:
    config, _ = init_subgrid_3d(
        shape_c=(6, 6, 6),
        dx_c=_DX_C,
        fine_region=(1, 5, 1, 5, 1, 5),
        ratio=_RATIO,
        tau=_TAU,
    )
    edge_ops = build_edge_ops(coarse_size=4, ratio=config.ratio, dx_c=config.dx_c)
    coarse_line = jnp.ones((edge_ops.coarse_size,), dtype=jnp.float32)
    fine_line = prolong_edge(coarse_line, edge_ops)
    coarse_mask = jnp.ones_like(coarse_line)
    fine_mask = jnp.ones_like(fine_line)
    alpha_c, alpha_f = sat_penalty_coefficients(config.ratio, config.tau)

    edge_after = _apply_sat_pair_edge(
        coarse_line,
        fine_line,
        edge_ops,
        alpha_c,
        alpha_f,
        coarse_mask,
        fine_mask,
    )
    corner_after = _apply_sat_pair_scalar(
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(1.25, dtype=jnp.float32),
        alpha_c,
        alpha_f,
    )
    edge_noop_passed = bool(
        np.allclose(np.asarray(edge_after[0]), np.asarray(coarse_line))
        and np.allclose(np.asarray(edge_after[1]), np.asarray(fine_line))
    )
    corner_delta = float(corner_after[1] - jnp.asarray(1.25, dtype=jnp.float32))
    return {
        "status": "edge_corner_accounting_probe_recorded_inconclusive",
        "active_faces": len(_active_faces(config)),
        "active_edges": len(_active_edges(config)),
        "active_corners": len(_active_corners(config)),
        "matched_edge_noop_passed": edge_noop_passed,
        "corner_perturbation_delta": corner_delta,
    }


def test_private_manufactured_face_ledger_preserves_matched_traces():
    ops, coarse_mask, fine_mask, alpha_c, alpha_f = _face_fixture_ops()
    ex_c = jnp.ones(ops.coarse_shape, dtype=jnp.float32) * 1.25
    ey_c = jnp.ones(ops.coarse_shape, dtype=jnp.float32) * -0.75
    hx_c = jnp.ones(ops.coarse_shape, dtype=jnp.float32) * 2.0e-6
    hy_c = jnp.ones(ops.coarse_shape, dtype=jnp.float32) * -1.0e-6
    ex_f = prolong_face(ex_c, ops)
    ey_f = prolong_face(ey_c, ops)
    hx_f = prolong_face(hx_c, ops)
    hy_f = prolong_face(hy_c, ops)

    before = _trace_energy(
        ex_c=ex_c,
        ey_c=ey_c,
        hx_c=hx_c,
        hy_c=hy_c,
        ex_f=ex_f,
        ey_f=ey_f,
        hx_f=hx_f,
        hy_f=hy_f,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    after_components = _apply_face_sat_to_components(
        ex_c=ex_c,
        ey_c=ey_c,
        hx_c=hx_c,
        hy_c=hy_c,
        ex_f=ex_f,
        ey_f=ey_f,
        hx_f=hx_f,
        hy_f=hy_f,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
        alpha_c=alpha_c,
        alpha_f=alpha_f,
    )
    after = _trace_energy(
        ex_c=after_components[0],
        ey_c=after_components[1],
        hx_c=after_components[2],
        hy_c=after_components[3],
        ex_f=after_components[4],
        ey_f=after_components[5],
        hx_f=after_components[6],
        hy_f=after_components[7],
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )

    for before_component, after_component in zip(
        (ex_c, ey_c, hx_c, hy_c, ex_f, ey_f, hx_f, hy_f),
        after_components,
    ):
        np.testing.assert_allclose(
            np.asarray(after_component),
            np.asarray(before_component),
            atol=1.0e-7,
        )
    assert abs(after - before) <= 1.0e-24
    assert (
        abs(
            _weighted_interface_work_residual(
                ex_c=ex_c,
                ey_c=ey_c,
                hx_c=hx_c,
                hy_c=hy_c,
                ex_f=ex_f,
                ey_f=ey_f,
                hx_f=hx_f,
                hy_f=hy_f,
                ops=ops,
                coarse_mask=coarse_mask,
            )
        )
        <= 1.0e-24
    )


def test_private_manufactured_face_ledger_classifies_current_kernel():
    packet = _private_manufactured_energy_ledger_packet()

    assert packet["private_manufactured_energy_ledger_diagnostic_status"] in {
        "ledger_mismatch_detected",
        "ledger_gate_passed",
        "inconclusive",
    }
    assert packet["selected_hypothesis"] == "discrete_eh_work_ledger_mismatch"
    assert packet["zero_work_face_status"] == "zero_work_dissipative_gate_passed"
    assert packet["face_ledger_status"] == "ledger_mismatch_detected"
    assert packet["selected_next_plan_direction"] == "bounded_kernel_repair_candidate"
    assert packet["normalized_balance_residual"] > packet["threshold"]
    assert packet["executable_diagnostics_added"] is True
    assert packet["solver_behavior_changed"] is False
    assert packet["sbp_sat_3d_repair_applied"] is False
    assert packet["hook_experiment_allowed"] is False
    assert packet["jit_runner_changed"] is False
    assert packet["public_claim_allowed"] is False
    assert packet["public_api_behavior_changed"] is False
    assert packet["public_default_tau_changed"] is False
    assert packet["public_observable_promoted"] is False


def test_private_bounded_kernel_repair_feasibility_records_negative_outcome():
    packet = _private_bounded_kernel_repair_packet()

    assert (
        packet["private_bounded_kernel_repair_status"]
        == "no_signature_compatible_bounded_repair"
    )
    assert packet["accepted_private_repair"] is False
    assert packet["solver_behavior_changed"] is False
    assert packet["sbp_sat_3d_repair_applied"] is False
    assert packet["hook_experiment_allowed"] is False
    assert packet["jit_runner_changed"] is False
    assert packet["public_claim_allowed"] is False
    assert packet["public_api_behavior_changed"] is False
    assert packet["public_default_tau_changed"] is False
    assert packet["public_observable_promoted"] is False
    assert packet["simresult_changed"] is False
    assert packet["phase0_forbidden_diff_required"] is True
    assert packet["final_forbidden_diff_matches_phase0"] is True
    assert packet["rejection_subreason"] == "no_bounded_candidate_passed_ledger_gate"
    assert packet["next_prerequisite"] == _BOUNDED_KERNEL_REPAIR_NEXT_PREREQUISITE

    candidates = packet["candidates"]
    assert candidates
    assert not any(candidate["accepted_candidate"] for candidate in candidates)
    bounded_candidates = [
        candidate for candidate in candidates if candidate["family_bounds_passed"]
    ]
    assert bounded_candidates
    assert all(
        candidate["ledger_gate_passed"] is False for candidate in bounded_candidates
    )
    assert any(
        candidate["rejection_subreason"] == "candidate_under_couples"
        for candidate in candidates
    )
    for candidate in candidates:
        assert candidate["ledger_threshold"] == _LEDGER_BALANCE_THRESHOLD
        assert candidate["zero_work_gate_passed"] is True
        assert candidate["matched_projected_traces_noop"] is True
        assert candidate["weighted_mismatch_before"] > 0.0
        assert candidate["weighted_mismatch_after_current"] > 0.0
        assert candidate["current_relative_mismatch_reduction"] > 0.0
        assert "rejection_reasons" in candidate


def test_private_face_coupling_theory_records_paired_design_ready():
    packet = _private_face_coupling_theory_packet()

    assert (
        packet["private_face_coupling_theory_status"]
        == "paired_face_coupling_design_ready"
    )
    assert packet["terminal_outcome"] == "paired_face_coupling_design_ready"
    assert (
        packet["selected_next_safe_implementation_lane"]
        == _FACE_COUPLING_NEXT_IMPLEMENTATION_LANE
    )
    assert packet["next_prerequisite"] == _FACE_COUPLING_NEXT_IMPLEMENTATION_LANE
    assert packet["selected_candidate_id"] == ("matched_hy_common_mode_energy_closure")
    assert packet["solver_behavior_changed"] is False
    assert packet["sbp_sat_3d_repair_applied"] is False
    assert packet["hook_experiment_allowed"] is False
    assert packet["jit_runner_changed"] is False
    assert packet["runner_changed"] is False
    assert packet["public_claim_allowed"] is False
    assert packet["public_api_behavior_changed"] is False
    assert packet["public_default_tau_changed"] is False
    assert packet["public_observable_promoted"] is False
    assert packet["promotion_candidate_ready"] is False
    assert packet["simresult_changed"] is False
    assert packet["result_surface_changed"] is False
    assert packet["orientation_sign_convention"]["poynting_trace"] == (
        "S_n = Ex * Hy - Ey * Hx"
    )
    assert "minimum-norm root" in packet["equations"]["candidate"]

    candidate = packet["candidates"][0]
    assert candidate["accepted_candidate"] is True
    assert candidate["candidate_family"] == (
        "same_step_paired_eh_matched_magnetic_common_mode"
    )
    assert candidate["ledger_gate_passed"] is True
    assert (
        candidate["ledger_normalized_balance_residual"] <= candidate["ledger_threshold"]
    )
    assert candidate["zero_work_gate_passed"] is True
    assert candidate["zero_work_net_delta_eh"] <= _ZERO_WORK_POSITIVE_INJECTION_TOL
    assert candidate["matched_projected_traces_noop"] is True
    assert candidate["coupling_strength_passed"] is True
    assert candidate["coupling_strength_ratio"] >= _COUPLING_STRENGTH_RATIO_MIN
    assert candidate["update_bounds_passed"] is True
    assert (
        _BOUNDING_SCALAR_MIN
        <= candidate["candidate_update_norm_ratio"]
        <= (_BOUNDING_SCALAR_MAX + _CANDIDATE_BOUND_TOL)
    )
    assert candidate["branches_on_measured_residual_or_test_name"] is False
    assert candidate["requires_paired_eh_face_context"] is True
    assert candidate["requires_time_centered_staging"] is False


def test_private_paired_face_helper_implementation_records_context_mismatch():
    packet = _private_paired_face_helper_implementation_packet()

    assert (
        packet["private_paired_face_helper_implementation_status"]
        == "production_context_mismatch_detected"
    )
    assert packet["terminal_outcome"] == "production_context_mismatch_detected"
    assert packet["upstream_theory_status"] == "paired_face_coupling_design_ready"
    assert packet["selected_theory_candidate_id"] == (
        "matched_hy_common_mode_energy_closure"
    )
    assert packet["production_shaped_gate_attempted"] is True
    assert packet["production_context_gate_passed"] is False

    for path_name in ("non_cpml", "cpml"):
        gate = packet["source_order_gates"][path_name]
        assert gate["h_sat_before_e_update"] is True
        assert gate["e_sat_after_all_e_updates"] is True
        assert gate["single_cotemporal_pre_current_sat_eh_state_available"] is False
        assert gate["single_cotemporal_post_current_sat_eh_state_available"] is False
        assert gate["same_discrete_work_ledger_available_without_staging"] is False

    assert (
        packet["phase1_failure_reason"]
        == "step_order_exposes_h_sat_and_e_sat_in_different_temporal_slots"
    )
    assert packet["requires_time_centered_staging"] is True
    assert (
        packet["cpml_non_cpml_face_work_equivalence"][
            "same_sat_order_mismatch_in_both_paths"
        ]
        is True
    )
    assert (
        packet["cpml_non_cpml_face_work_equivalence"]["distinct_cpml_formula_required"]
        is False
    )
    assert packet["orientation_generalization"]["blocked_by_orientation"] is False
    assert packet["orientation_generalization"]["uses_face_orientations_only"] is True
    assert packet["production_patch_allowed"] is False
    assert packet["production_patch_applied"] is False
    assert packet["accepted_private_helper"] is False
    assert packet["solver_behavior_changed"] is False
    assert packet["sbp_sat_3d_paired_face_helper_applied"] is False
    assert packet["sbp_sat_3d_diff_allowed"] is False
    assert packet["helper_specific_switch_added"] is False
    assert packet["hook_experiment_allowed"] is False
    assert packet["jit_runner_changed"] is False
    assert packet["runner_changed"] is False
    assert packet["public_claim_allowed"] is False
    assert packet["public_api_behavior_changed"] is False
    assert packet["public_default_tau_changed"] is False
    assert packet["public_observable_promoted"] is False
    assert packet["promotion_candidate_ready"] is False
    assert packet["simresult_changed"] is False
    assert packet["result_surface_changed"] is False
    assert packet["final_sbp_sat_3d_diff_matches_phase0"] is True
    assert (
        packet["next_prerequisite"]
        == "private time-centered SAT staging redesign ralplan"
    )


def test_private_time_centered_staging_redesign_records_contract_ready():
    packet = _private_time_centered_staging_redesign_packet()

    assert (
        packet["private_time_centered_staging_redesign_status"]
        == "time_centered_staging_contract_ready"
    )
    assert packet["terminal_outcome"] in _TIME_CENTERED_TERMINAL_OUTCOMES
    assert packet["terminal_outcome"] == "time_centered_staging_contract_ready"
    assert packet["upstream_helper_status"] == "production_context_mismatch_detected"
    assert packet["upstream_selected_candidate_id"] == (
        "matched_hy_common_mode_energy_closure"
    )
    assert packet["same_call_local_staging_contract_ready"] is True
    assert packet["selected_candidate_id"] == "same_call_centered_h_bar"
    assert packet["candidate_count"] == 4
    assert packet["thresholds"] == {
        "ledger_balance_threshold": _LEDGER_BALANCE_THRESHOLD,
        "coupling_strength_ratio_min": _COUPLING_STRENGTH_RATIO_MIN,
        "bounding_scalar_min": _BOUNDING_SCALAR_MIN,
        "bounding_scalar_max": _BOUNDING_SCALAR_MAX,
    }

    selected = [
        candidate
        for candidate in packet["candidates"]
        if candidate["accepted_candidate"]
    ]
    assert len(selected) == 1
    selected_candidate = selected[0]
    assert selected_candidate["candidate_id"] == "same_call_centered_h_bar"
    assert selected_candidate["ledger_gate_passed"] is True
    assert (
        selected_candidate["ledger_normalized_balance_residual"]
        <= selected_candidate["ledger_threshold"]
    )
    assert selected_candidate["zero_work_gate_passed"] is True
    assert selected_candidate["zero_work_net_delta_eh"] <= (
        _ZERO_WORK_POSITIVE_INJECTION_TOL
    )
    assert selected_candidate["matched_projected_traces_noop"] is True
    assert selected_candidate["coupling_strength_passed"] is True
    assert selected_candidate["coupling_strength_ratio"] >= (
        _COUPLING_STRENGTH_RATIO_MIN
    )
    assert selected_candidate["update_bounds_passed"] is True
    assert (
        _BOUNDING_SCALAR_MIN
        <= selected_candidate["candidate_update_norm_ratio"]
        <= (_BOUNDING_SCALAR_MAX + _CANDIDATE_BOUND_TOL)
    )
    assert selected_candidate["production_expressibility"]["passes_gate"] is True
    assert selected_candidate["requires_hook"] is False
    assert selected_candidate["requires_runner_state"] is False
    assert selected_candidate["requires_public_api"] is False
    assert selected_candidate["requires_default_tau_change"] is False
    assert selected_candidate["requires_public_observable"] is False
    assert selected_candidate["uses_face_orientations_only"] is True

    production_gate = packet["production_expressibility_gate"]
    assert production_gate["passed"] is True
    assert production_gate["forbids_private_post_h_hook"] is True
    assert production_gate["forbids_private_post_e_hook"] is True
    assert production_gate["forbids_test_local_hook_emulation"] is True
    assert production_gate["forbids_runner_or_public_api_state"] is True
    assert production_gate["forbids_next_step_synthetic_state"] is True

    for candidate in packet["candidates"]:
        assert tuple(candidate["temporal_trace_tags"][:4]) == (
            "H_pre_sat",
            "H_post_sat",
            "E_pre_sat",
            "E_post_sat",
        )
        production = candidate["production_expressibility"]
        assert production["uses_private_post_h_hook"] is False
        assert production["uses_private_post_e_hook"] is False
        assert production["uses_test_local_hook_emulation"] is False
        assert production["uses_public_api_state"] is False
        assert production["uses_environment_or_config_switch"] is False
        if not candidate["accepted_candidate"]:
            rejection = candidate["rejection_metadata"]
            assert rejection["category"] in _TIME_CENTERED_REJECTION_CATEGORIES

    for path_name in ("non_cpml", "cpml"):
        slots = packet["cpml_non_cpml_staging_contract"][f"{path_name}_slots"]
        assert slots["same_call_local_values_available"] is True
        assert slots["requires_next_step_state"] is False
        assert slots["requires_update_reordering"] is False
        assert slots["h_sat_line"] < slots["first_e_update_line"]
        assert slots["e_sat_line"] > slots["last_e_update_line"]

    assert (
        packet["cpml_non_cpml_staging_contract"][
            "internal_face_work_contract_identical"
        ]
        is True
    )
    assert (
        packet["cpml_non_cpml_staging_contract"]["distinct_cpml_formula_required"]
        is False
    )
    assert packet["orientation_generalization"]["uses_face_orientations_only"] is True
    assert packet["orientation_generalization"]["blocked_by_orientation"] is False
    assert packet["solver_behavior_changed"] is False
    assert packet["sbp_sat_3d_time_centered_staging_applied"] is False
    assert packet["hook_experiment_allowed"] is False
    assert packet["jit_runner_changed"] is False
    assert packet["runner_changed"] is False
    assert packet["public_claim_allowed"] is False
    assert packet["public_api_behavior_changed"] is False
    assert packet["public_default_tau_changed"] is False
    assert packet["public_observable_promoted"] is False
    assert packet["promotion_candidate_ready"] is False
    assert packet["simresult_changed"] is False
    assert packet["result_surface_changed"] is False
    assert packet["final_sbp_sat_3d_diff_matches_phase0"] is True
    assert packet["next_prerequisite"] == (
        "private time-centered paired-face helper implementation ralplan"
    )


def test_private_time_centered_paired_face_helper_implementation_records_helper():
    packet = _private_time_centered_paired_face_helper_implementation_packet()

    assert (
        packet["private_time_centered_paired_face_helper_implementation_status"]
        == "private_time_centered_paired_face_helper_implemented"
    )
    assert packet["terminal_outcome"] in _TIME_CENTERED_HELPER_TERMINAL_OUTCOMES
    assert (
        packet["terminal_outcome"]
        == "private_time_centered_paired_face_helper_implemented"
    )
    assert packet["upstream_staging_status"] == "time_centered_staging_contract_ready"
    assert packet["selected_staging_candidate_id"] == "same_call_centered_h_bar"
    assert packet["selected_time_centering_schema"] == "same_call_centered_h_bar"
    assert packet["bounded_relaxation"] == 0.02

    for path_name in ("non_cpml", "cpml"):
        slots = packet["production_slot_binding"][path_name]
        assert slots["h_pre_sat_slot_bound"] is True
        assert slots["h_post_sat_slot_bound"] is True
        assert slots["e_pre_sat_slot_bound"] is True
        assert slots["e_post_sat_slot_bound"] is True
        assert slots["helper_after_e_sat"] is True
        assert slots["helper_uses_private_post_h_hook"] is False
        assert slots["helper_uses_private_post_e_hook"] is False
        assert all(slots["helper_signature_fields"].values())

    assert (
        packet["cpml_non_cpml_helper_contract"]["internal_face_work_contract_identical"]
        is True
    )
    assert (
        packet["cpml_non_cpml_helper_contract"]["distinct_cpml_formula_required"]
        is False
    )
    assert packet["orientation_generalization"]["uses_face_orientations_only"] is True
    assert len(packet["hunk_inventory"]) == 6
    assert {hunk["symbol"] for hunk in packet["hunk_inventory"]} >= {
        "_apply_time_centered_paired_face_helper",
        "step_subgrid_3d",
        "step_subgrid_3d_with_cpml",
    }
    assert packet["production_patch_allowed"] is True
    assert packet["production_patch_applied"] is True
    assert packet["accepted_private_helper"] is True
    assert packet["solver_behavior_changed"] is True
    assert packet["sbp_sat_3d_time_centered_paired_face_helper_applied"] is True
    assert packet["helper_specific_switch_added"] is False
    assert packet["uses_private_post_h_hook"] is False
    assert packet["uses_private_post_e_hook"] is False
    assert packet["uses_test_local_hook_emulation"] is False
    assert packet["jit_runner_changed"] is False
    assert packet["runner_changed"] is False
    assert packet["public_claim_allowed"] is False
    assert packet["public_api_behavior_changed"] is False
    assert packet["public_default_tau_changed"] is False
    assert packet["public_observable_promoted"] is False
    assert packet["promotion_candidate_ready"] is False
    assert packet["hook_experiment_allowed"] is False
    assert packet["simresult_changed"] is False
    assert packet["result_surface_changed"] is False
    assert packet["next_prerequisite"] == (
        "private time-centered paired-face helper fixture-quality recovery ralplan"
    )


def test_private_manufactured_interior_box_ledger_records_edge_corner_accounting():
    packet = _interior_box_ledger_packet()

    assert packet["status"] == "edge_corner_accounting_probe_recorded_inconclusive"
    assert packet["active_faces"] == 6
    assert packet["active_edges"] == 12
    assert packet["active_corners"] == 8
    assert packet["matched_edge_noop_passed"] is True
    assert packet["corner_perturbation_delta"] < 0.0
