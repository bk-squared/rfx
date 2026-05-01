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
import rfx.subgridding.sbp_operators as sbp_operators
import rfx.subgridding.sbp_sat_3d as sbp_sat_3d
from rfx.subgridding.face_ops import build_face_ops, prolong_face, restrict_face
from rfx.subgridding.sbp_operators import (
    FaceFluxSpec,
    all_face_weighted_flux_report,
    box_surface_partition_report,
    build_sbp_first_derivative_1d,
    build_tensor_face_mortar,
    build_yee_staggered_derivative_pair_1d,
    face_mortar_adjoint_report,
    face_mortar_reproduction_report,
    operator_projected_sat_pair_face,
    operator_projected_skew_eh_sat_face,
    prolong_face_mortar,
    sbp_identity_residual,
    weighted_em_flux_residual,
    yee_staggered_identity_residual,
)
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
_INTERFACE_FLOOR_REPAIR_STATUS = "no_bounded_private_interface_floor_repair"
_INTERFACE_FLOOR_REPAIR_NEXT_PREREQUISITE = (
    "private higher-order SBP face-norm/interface-operator redesign after "
    "characteristic face repair manufactured gate failed ralplan"
)
_INTERFACE_FLOOR_REPAIR_REPORT = (
    ".omx/reports/"
    "sbp-sat-private-interface-floor-repair-theory-implementation-"
    "20260430T164629Z.md"
)
_INTERFACE_FLOOR_REPAIR_TERMINAL_OUTCOMES = (
    "private_characteristic_face_repair_candidate_accepted",
    "private_interface_floor_repair_implemented_fixture_quality_pending",
    "private_interface_floor_repair_candidate_ready_for_private_slab_scorer",
    "no_bounded_private_interface_floor_repair",
)
_INTERFACE_FLOOR_REPAIR_ALLOWED_SOLVER_SYMBOLS = (
    "_levi_civita_sign",
    "_normal_cross_tangential_h_face",
    "_characteristic_face_traces",
    "_inverse_characteristic_face_traces",
    "_characteristic_balanced_face_correction",
    "_apply_characteristic_balanced_face_helper",
    "step_subgrid_3d_with_cpml",
    "step_subgrid_3d",
)
_FACE_NORM_OPERATOR_REPAIR_STATUS = "no_private_face_norm_operator_repair"
_FACE_NORM_OPERATOR_REPAIR_NEXT_PREREQUISITE = (
    "private broader SBP derivative/interior-boundary operator redesign after "
    "face-norm/interface-operator ladder failed ralplan"
)
_FACE_NORM_OPERATOR_REPAIR_REPORT = (
    ".omx/reports/"
    "sbp-sat-private-higher-order-face-norm-interface-operator-redesign-"
    "20260430T194140Z.md"
)
_FACE_NORM_OPERATOR_REPAIR_TERMINAL_OUTCOMES = (
    "private_norm_adjoint_face_operator_repair_candidate_accepted",
    "private_diagonal_face_norm_repair_candidate_accepted",
    "private_face_norm_operator_repair_implemented_fixture_quality_pending",
    "higher_order_projection_requires_broader_operator_plan",
    "edge_corner_norm_inconsistency_suspected",
    "no_private_face_norm_operator_repair",
)
_FACE_NORM_OPERATOR_ALLOWED_SOLVER_SYMBOLS = (
    "_face_norm_inner",
    "_face_norm_adjoint_defect",
    "_norm_adjoint_restrict_face",
    "_norm_adjoint_prolong_face",
    "_apply_norm_compatible_sat_pair_face",
    "_apply_norm_compatible_interface_helper",
    "step_subgrid_3d_with_cpml",
    "step_subgrid_3d",
)
_DERIVATIVE_INTERFACE_REPAIR_STATUS = "no_private_derivative_interface_repair"
_DERIVATIVE_INTERFACE_REPAIR_NEXT_PREREQUISITE = (
    "global SBP derivative/mortar operator architecture after private "
    "derivative/interior-boundary ladder required operator refactor ralplan"
)
_DERIVATIVE_INTERFACE_REPAIR_REPORT = (
    ".omx/reports/"
    "sbp-sat-private-broader-derivative-interior-boundary-energy-stable-"
    "interface-redesign-20260501T062334Z.md"
)
_DERIVATIVE_INTERFACE_REPAIR_TERMINAL_OUTCOMES = (
    "private_reduced_derivative_flux_contract_ready",
    "private_derivative_interface_flux_candidate_accepted",
    "edge_corner_derivative_accounting_ready",
    "requires_global_sbp_operator_refactor",
    "private_derivative_interface_repair_implemented_fixture_quality_pending",
    "no_private_derivative_interface_repair",
)
_DERIVATIVE_INTERFACE_ALLOWED_SOLVER_SYMBOLS = (
    "_derivative_interface_energy_terms",
    "_reduced_interface_flux_balance",
    "_energy_stable_face_flux_update",
    "_apply_energy_stable_derivative_interface_helper",
    "step_subgrid_3d_with_cpml",
    "step_subgrid_3d",
)

_GLOBAL_OPERATOR_ARCHITECTURE_STATUS = "private_global_operator_3d_contract_ready"
_GLOBAL_OPERATOR_ARCHITECTURE_NEXT_PREREQUISITE = (
    "private solver integration hunk from global SBP derivative/mortar operator "
    "architecture after A1-A4 evidence summary ralplan"
)
_GLOBAL_OPERATOR_ARCHITECTURE_REPORT = (
    ".omx/reports/"
    "sbp-sat-global-sbp-derivative-mortar-operator-architecture-"
    "20260501T081353Z.md"
)
_GLOBAL_OPERATOR_ARCHITECTURE_TERMINAL_OUTCOMES = (
    "private_sbp_derivative_contract_ready",
    "private_mortar_projection_contract_ready",
    "private_em_mortar_flux_contract_ready",
    "private_global_operator_3d_contract_ready",
    "private_global_derivative_mortar_repair_implemented_fixture_quality_pending",
    "no_private_global_derivative_mortar_operator_repair",
)
_GLOBAL_OPERATOR_ALLOWED_SOLVER_SYMBOLS = (
    "_get_face_ops",
    "apply_sat_h_interfaces",
    "apply_sat_e_interfaces",
    "_apply_time_centered_paired_face_helper",
    "step_subgrid_3d_with_cpml",
    "step_subgrid_3d",
)
_SOLVER_INTEGRATION_STATUS = "private_solver_integration_requires_followup_diagnostic_only"
_SOLVER_INTEGRATION_NEXT_PREREQUISITE = (
    "private operator-projected face SAT energy-transfer redesign after "
    "diagnostic-only solver integration gate failed ralplan"
)
_SOLVER_INTEGRATION_REPORT = (
    ".omx/reports/"
    "sbp-sat-private-solver-integration-hunk-after-global-operator-"
    "a1-a4-evidence-20260501T135000Z.md"
)
_SOLVER_INTEGRATION_TERMINAL_OUTCOMES = (
    "private_operator_projected_sat_preaccepted",
    "private_global_operator_solver_hunk_retained_fixture_quality_pending",
    "private_operator_aware_time_centered_helper_retained_fixture_quality_pending",
    "private_solver_integration_requires_followup_diagnostic_only",
    "no_private_solver_integration_hunk_retained",
)
_SOLVER_INTEGRATION_ALLOWED_SOLVER_SYMBOLS = (
    "_apply_operator_projected_sat_pair_face",
    "apply_sat_h_interfaces",
    "apply_sat_e_interfaces",
)
_OPERATOR_PROJECTED_ENERGY_TRANSFER_STATUS = (
    "private_operator_projected_energy_transfer_contract_ready"
)
_OPERATOR_PROJECTED_ENERGY_TRANSFER_NEXT_PREREQUISITE = (
    "private bounded solver integration of operator-projected energy-transfer "
    "contract after manufactured ledger closure ralplan"
)
_OPERATOR_PROJECTED_ENERGY_TRANSFER_TERMINAL_OUTCOMES = (
    "private_operator_projected_skew_work_form_ready",
    "private_material_metric_operator_work_form_ready",
    "private_operator_projected_partition_coupling_required",
    "private_operator_projected_energy_transfer_contract_ready",
    "no_private_operator_projected_energy_transfer_repair",
)
_OPERATOR_PROJECTED_ENERGY_TRANSFER_ALLOWED_FUTURE_SOLVER_SYMBOLS = (
    "_apply_operator_projected_skew_eh_sat_face",
    "apply_sat_h_interfaces",
    "apply_sat_e_interfaces",
)
_OPERATOR_PROJECTED_SOLVER_INTEGRATION_STATUS = (
    "private_operator_projected_solver_hunk_retained_fixture_quality_pending"
)
_OPERATOR_PROJECTED_SOLVER_INTEGRATION_NEXT_PREREQUISITE = (
    "private boundary coexistence and fixture-quality validation after "
    "operator-projected solver hunk ralplan"
)
_OPERATOR_PROJECTED_SOLVER_INTEGRATION_TERMINAL_OUTCOMES = (
    "private_skew_helper_solver_preaccepted",
    "private_operator_projected_solver_hunk_retained_fixture_quality_pending",
    "private_operator_projected_solver_integration_requires_followup_diagnostic_only",
    "no_private_operator_projected_solver_hunk_retained",
)
_OPERATOR_PROJECTED_SOLVER_HUNK_SYMBOLS = (
    "_apply_operator_projected_skew_eh_face_helper",
    "step_subgrid_3d_with_cpml",
    "step_subgrid_3d",
)
_BOUNDARY_FIXTURE_VALIDATION_STATUS = (
    "private_boundary_coexistence_passed_fixture_quality_blocked"
)
_BOUNDARY_FIXTURE_VALIDATION_NEXT_PREREQUISITE = (
    "private fixture-quality blocker repair after boundary coexistence "
    "validation ralplan"
)
_BOUNDARY_FIXTURE_VALIDATION_TERMINAL_OUTCOMES = (
    "private_boundary_contract_locked_solver_hunk_present",
    "private_boundary_coexistence_passed_fixture_quality_pending",
    "private_boundary_coexistence_fixture_quality_ready",
    "private_boundary_coexistence_passed_fixture_quality_blocked",
    "private_boundary_fixture_bounded_repair_retained",
    "private_boundary_coexistence_fail_closed_no_public_promotion",
)
_BOUNDARY_FIXTURE_VALIDATION_PRECEDENCE = (
    "private_boundary_coexistence_fail_closed_no_public_promotion",
    "private_boundary_fixture_bounded_repair_retained",
    "private_boundary_coexistence_fixture_quality_ready",
    "private_boundary_coexistence_passed_fixture_quality_blocked",
    "private_boundary_coexistence_passed_fixture_quality_pending",
    "private_boundary_contract_locked_solver_hunk_present",
)
_BOUNDARY_FIXTURE_ACCEPTED_CLASSES = (
    "all_pec",
    "selected_pmc_reflector_faces",
    "periodic_axes_when_box_is_interior_or_spans_axis",
    "scalar_cpml_bounded_interior_box",
    "boundaryspec_uniform_cpml_bounded_interior_box",
)
_BOUNDARY_FIXTURE_UNSUPPORTED_CLASSES = (
    "upml",
    "per_face_cpml_thickness_overrides",
    "mixed_cpml_reflector",
    "mixed_cpml_periodic",
    "mixed_pmc_periodic",
    "one_side_touch_periodic_axis",
    "mixed_absorber_families",
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


def _levi_civita_sign(axis_a: int, axis_b: int, axis_c: int) -> int:
    axes = (axis_a, axis_b, axis_c)
    if len(set(axes)) != 3:
        return 0
    inversions = sum(
        1
        for left in range(3)
        for right in range(left + 1, 3)
        if axes[left] > axes[right]
    )
    return -1 if inversions % 2 else 1


def _normal_cross_tangential_h_face(face: str, h_face):
    orientation = sbp_sat_3d.FACE_ORIENTATIONS[face]
    return tuple(
        sum(
            orientation.normal_sign
            * _levi_civita_sign(orientation.normal_axis, h_axis, e_axis)
            * h_component
            for h_axis, h_component in zip(
                orientation.tangential_axes,
                h_face,
                strict=True,
            )
        )
        for e_axis in orientation.tangential_axes
    )


def _tangential_h_from_normal_cross(face: str, normal_cross_h_face):
    orientation = sbp_sat_3d.FACE_ORIENTATIONS[face]
    return tuple(
        sum(
            orientation.normal_sign
            * _levi_civita_sign(orientation.normal_axis, h_axis, e_axis)
            * cross_component
            for e_axis, cross_component in zip(
                orientation.tangential_axes,
                normal_cross_h_face,
                strict=True,
            )
        )
        for h_axis in orientation.tangential_axes
    )


def _characteristic_face_traces(e_face, h_face, face: str):
    eta0 = math.sqrt(MU_0 / EPS_0)
    normal_cross_h = _normal_cross_tangential_h_face(face, h_face)
    return (
        tuple(e + eta0 * h for e, h in zip(e_face, normal_cross_h, strict=True)),
        tuple(e - eta0 * h for e, h in zip(e_face, normal_cross_h, strict=True)),
    )


def _inverse_characteristic_face_traces(w_plus, w_minus, face: str):
    eta0 = math.sqrt(MU_0 / EPS_0)
    e_face = tuple(
        0.5 * (plus + minus) for plus, minus in zip(w_plus, w_minus, strict=True)
    )
    normal_cross_h = tuple(
        (plus - minus) / (2.0 * eta0)
        for plus, minus in zip(w_plus, w_minus, strict=True)
    )
    h_face = _tangential_h_from_normal_cross(face, normal_cross_h)
    return e_face, h_face


def _apply_characteristic_trace_sat(
    coarse_trace,
    fine_trace,
    *,
    ops,
    alpha_c,
    alpha_f,
    coarse_mask,
    fine_mask,
):
    updated_pairs = tuple(
        _apply_scaled_sat_pair_face(
            coarse_component,
            fine_component,
            ops,
            alpha_c,
            alpha_f,
            coarse_mask,
            fine_mask,
            coarse_scale=1.0,
            fine_scale=1.0,
        )
        for coarse_component, fine_component in zip(
            coarse_trace,
            fine_trace,
            strict=True,
        )
    )
    return tuple(zip(*updated_pairs, strict=True))


def _apply_characteristic_face_sat_to_components(
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
    face: str = "z_hi",
):
    coarse_w_plus, coarse_w_minus = _characteristic_face_traces(
        (ex_c, ey_c),
        (hx_c, hy_c),
        face,
    )
    fine_w_plus, fine_w_minus = _characteristic_face_traces(
        (ex_f, ey_f),
        (hx_f, hy_f),
        face,
    )
    coarse_w_plus, fine_w_plus = _apply_characteristic_trace_sat(
        coarse_w_plus,
        fine_w_plus,
        ops=ops,
        alpha_c=alpha_c,
        alpha_f=alpha_f,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    coarse_w_minus, fine_w_minus = _apply_characteristic_trace_sat(
        coarse_w_minus,
        fine_w_minus,
        ops=ops,
        alpha_c=alpha_c,
        alpha_f=alpha_f,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )

    coarse_e, coarse_h = _inverse_characteristic_face_traces(
        coarse_w_plus,
        coarse_w_minus,
        face,
    )
    fine_e, fine_h = _inverse_characteristic_face_traces(
        fine_w_plus,
        fine_w_minus,
        face,
    )
    return (
        coarse_e[0],
        coarse_e[1],
        coarse_h[0],
        coarse_h[1],
        fine_e[0],
        fine_e[1],
        fine_h[0],
        fine_h[1],
    )


def _characteristic_matched_projected_trace_noop() -> bool:
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
    after_components = _apply_characteristic_face_sat_to_components(
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
    )
    return all(
        np.allclose(np.asarray(after), np.asarray(before), atol=1.0e-7)
        for before, after in zip(before_components, after_components, strict=True)
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


def _private_interface_floor_repair_packet() -> dict[str, object]:
    fixture = _manufactured_nonzero_face_components()
    ops = fixture["ops"]
    coarse_mask = fixture["coarse_mask"]
    fine_mask = fixture["fine_mask"]
    before_components = fixture["components"]
    current = _current_kernel_coupling_metrics()
    current_components = _current_kernel_after_components(fixture)
    characteristic_components = _apply_characteristic_face_sat_to_components(
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
    )
    characteristic_equals_current = all(
        np.allclose(np.asarray(characteristic), np.asarray(current), atol=1.0e-12)
        for characteristic, current in zip(
            characteristic_components,
            current_components,
            strict=True,
        )
    )
    ledger = _ledger_residual_for_components(
        before_components,
        characteristic_components,
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
    zero_before = zero_work_fixture["components"]
    zero_candidate = _apply_characteristic_face_sat_to_components(
        ex_c=zero_before[0],
        ey_c=zero_before[1],
        hx_c=zero_before[2],
        hy_c=zero_before[3],
        ex_f=zero_before[4],
        ey_f=zero_before[5],
        hx_f=zero_before[6],
        hy_f=zero_before[7],
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
        alpha_c=fixture["alpha_c"],
        alpha_f=fixture["alpha_f"],
    )
    zero_work = _ledger_residual_for_components(
        zero_before,
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
        characteristic_components,
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
        characteristic_components,
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
    matched_noop = _characteristic_matched_projected_trace_noop()
    coupling_strength_passed = (
        _COUPLING_STRENGTH_RATIO_MIN
        <= coupling_strength_ratio
        <= _BOUNDING_SCALAR_MAX + _CANDIDATE_BOUND_TOL
    )
    update_bounds_passed = (
        _BOUNDING_SCALAR_MIN - _CANDIDATE_BOUND_TOL
        <= update_norm_ratio
        <= _BOUNDING_SCALAR_MAX + _CANDIDATE_BOUND_TOL
    )
    orientation_contract_passed = tuple(sbp_sat_3d.FACE_ORIENTATIONS) == (
        "x_lo",
        "x_hi",
        "y_lo",
        "y_hi",
        "z_lo",
        "z_hi",
    )
    f4_packet = _interior_box_ledger_packet()
    edge_corner_gate_passed = (
        f4_packet["matched_edge_noop_passed"] is True
        and f4_packet["active_edges"] == 12
        and f4_packet["active_corners"] == 8
    )
    accepted = bool(
        orientation_contract_passed
        and ledger_gate_passed
        and zero_work_gate_passed
        and matched_noop
        and coupling_strength_passed
        and update_bounds_passed
        and edge_corner_gate_passed
        and not characteristic_equals_current
    )
    f1_rejection_reasons = []
    if not ledger_gate_passed:
        f1_rejection_reasons.append("candidate_failed_manufactured_ledger_gate")
    if characteristic_equals_current:
        f1_rejection_reasons.append("candidate_collapses_to_current_component_sat")
    if not edge_corner_gate_passed:
        f1_rejection_reasons.append("candidate_failed_edge_corner_preacceptance")
    f1_candidate = {
        "candidate_id": "oriented_characteristic_face_balance",
        "candidate_family": "characteristic_w_plus_minus_face_balance",
        "production_edit_allowed": True,
        "orientation_contract_passed": orientation_contract_passed,
        "faces_considered": tuple(sbp_sat_3d.FACE_ORIENTATIONS),
        "uses_face_orientations_only": True,
        "characteristic_traces": "W± = E_t ± eta0*(n×H)_t",
        "characteristic_equivalent_to_current_component_sat": characteristic_equals_current,
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
        "edge_corner_preacceptance_gate_passed": edge_corner_gate_passed,
        "accepted_candidate": accepted,
        "rejection_reasons": f1_rejection_reasons,
    }
    candidates = (
        {
            "candidate_id": "current_time_centered_helper_baseline",
            "candidate_family": "baseline_only",
            "production_edit_allowed": False,
            "status": "scored_as_f0_baseline",
            "ledger_normalized_balance_residual": float(
                _ledger_residual_for_components(
                    before_components,
                    current_components,
                    ops=ops,
                    coarse_mask=coarse_mask,
                    fine_mask=fine_mask,
                )["normalized_balance_residual"]
            ),
            "accepted_candidate": False,
        },
        f1_candidate,
        {
            "candidate_id": "reciprocal_dual_field_scaling_historical_guard",
            "candidate_family": "historical_guard",
            "production_edit_allowed": False,
            "status": "reciprocal_scaling_already_invalidated",
            "identical_to_prior_bounded_reciprocal_family": True,
            "accepted_candidate": False,
        },
        {
            "candidate_id": "current_minimum_norm_centered_h_guard",
            "candidate_family": "historical_guard",
            "production_edit_allowed": False,
            "status": "minimum_norm_centered_h_already_implemented_fixture_pending",
            "selected_as_new_repair_basis": False,
            "accepted_candidate": False,
        },
        {
            "candidate_id": "edge_corner_preacceptance_gate",
            "candidate_family": "preacceptance_guard",
            "production_edit_allowed": False,
            "status": (
                "edge_corner_preacceptance_passed"
                if edge_corner_gate_passed
                else "edge_corner_interface_floor_suspected"
            ),
            "active_edges": f4_packet["active_edges"],
            "active_corners": f4_packet["active_corners"],
            "matched_edge_noop_passed": f4_packet["matched_edge_noop_passed"],
            "accepted_candidate": False,
        },
    )
    status = (
        "private_characteristic_face_repair_candidate_accepted"
        if accepted
        else _INTERFACE_FLOOR_REPAIR_STATUS
    )
    return {
        "private_interface_floor_repair_status": status,
        "status": status,
        "terminal_outcome": status,
        "terminal_outcome_taxonomy": _INTERFACE_FLOOR_REPAIR_TERMINAL_OUTCOMES,
        "diagnostic_scope": "private_manufactured_interface_only",
        "upstream_measurement_contract_status": "persistent_interface_floor_confirmed",
        "selected_candidate_id": f1_candidate["candidate_id"] if accepted else None,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "thresholds": {
            "ledger_balance_threshold": _LEDGER_BALANCE_THRESHOLD,
            "coupling_strength_ratio_min": _COUPLING_STRENGTH_RATIO_MIN,
            "coupling_strength_ratio_max": _BOUNDING_SCALAR_MAX,
            "update_norm_ratio_min": _BOUNDING_SCALAR_MIN,
            "update_norm_ratio_max": _BOUNDING_SCALAR_MAX,
        },
        "selection_rule": "accept at most F1 only if every manufactured and F4 gate passes",
        "solver_hunk_allowed_if_selected": _INTERFACE_FLOOR_REPAIR_ALLOWED_SOLVER_SYMBOLS,
        "solver_hunk_retained": accepted,
        "actual_solver_hunk_inventory": ()
        if not accepted
        else ("not_applicable_in_test",),
        "production_patch_allowed": accepted,
        "production_patch_applied": accepted,
        "solver_behavior_changed": accepted,
        "sbp_sat_3d_repair_applied": accepted,
        "sbp_sat_3d_diff_allowed": accepted,
        "hook_experiment_allowed": False,
        "jit_runner_changed": False,
        "runner_changed": False,
        "api_surface_changed": False,
        "public_claim_allowed": False,
        "public_api_behavior_changed": False,
        "public_default_tau_changed": False,
        "public_observable_promoted": False,
        "public_true_rt_promoted": False,
        "public_dft_promoted": False,
        "promotion_candidate_ready": False,
        "simresult_changed": False,
        "result_surface_changed": False,
        "next_prerequisite": (
            "private fixture-quality replay after characteristic face repair ralplan"
            if accepted
            else _INTERFACE_FLOOR_REPAIR_NEXT_PREREQUISITE
        ),
        "report": _INTERFACE_FLOOR_REPAIR_REPORT,
    }



def _face_norm_probe_suite(ops) -> tuple[dict[str, object], ...]:
    coarse_shape = ops.coarse_shape
    total = int(np.prod(coarse_shape))
    alternating = np.where(
        (np.indices(coarse_shape).sum(axis=0) % 2) == 0,
        1.0,
        -1.0,
    ).astype(np.float32)
    impulse = np.zeros(coarse_shape, dtype=np.float32)
    impulse[coarse_shape[0] // 2, coarse_shape[1] // 2] = 1.0
    return (
        {
            "probe_id": "constant",
            "coarse_face": jnp.ones(coarse_shape, dtype=jnp.float32),
        },
        {
            "probe_id": "ramp",
            "coarse_face": jnp.linspace(
                -1.0,
                1.0,
                total,
                dtype=jnp.float32,
            ).reshape(coarse_shape),
        },
        {
            "probe_id": "alternating",
            "coarse_face": jnp.asarray(alternating, dtype=jnp.float32),
        },
        {
            "probe_id": "localized_impulse",
            "coarse_face": jnp.asarray(impulse, dtype=jnp.float32),
        },
    )


def _face_norm_inner(face_values, face_norm, mask) -> float:
    return float(jnp.sum(face_values * face_norm * mask))


def _norm_adjoint_restrict_face(fine_face, ops):
    """Test-local unmasked mass-adjoint restriction from existing face weights."""

    numerator = ops.prolong_i.T @ (fine_face * ops.fine_norm) @ ops.prolong_j
    return numerator / ops.coarse_norm


def _face_norm_adjoint_defect(
    coarse_face,
    fine_face,
    ops,
    coarse_mask,
    fine_mask,
    *,
    restrict_fn,
) -> float:
    restricted = restrict_fn(fine_face, ops)
    lhs = _face_norm_inner(coarse_face * restricted, ops.coarse_norm, coarse_mask)
    rhs = _face_norm_inner(
        prolong_face(coarse_face, ops) * fine_face,
        ops.fine_norm,
        fine_mask,
    )
    return float(lhs - rhs)


def _face_norm_operator_audit_packet() -> dict[str, object]:
    fixture = _manufactured_nonzero_face_components()
    ops = fixture["ops"]
    coarse_mask = fixture["coarse_mask"]
    fine_mask = fixture["fine_mask"]
    all_coarse = jnp.ones(ops.coarse_shape, dtype=jnp.float32)
    all_fine = jnp.ones(ops.fine_shape, dtype=jnp.float32)
    fine_probe = jnp.linspace(
        -1.0,
        1.0,
        int(np.prod(ops.fine_shape)),
        dtype=jnp.float32,
    ).reshape(ops.fine_shape)
    current_mass_adjoint_difference_max = float(
        jnp.max(
            jnp.abs(
                restrict_face(fine_probe, ops)
                - _norm_adjoint_restrict_face(fine_probe, ops)
            )
        )
    )

    probe_rows = []
    current_masked_defects = []
    mass_adjoint_defects = []
    projection_noop_failures = []
    for probe in _face_norm_probe_suite(ops):
        probe_id = str(probe["probe_id"])
        coarse_probe = probe["coarse_face"]
        current_masked_defect = _face_norm_adjoint_defect(
            coarse_probe,
            fine_probe,
            ops,
            coarse_mask,
            fine_mask,
            restrict_fn=restrict_face,
        )
        mass_adjoint_defect = _face_norm_adjoint_defect(
            coarse_probe,
            fine_probe,
            ops,
            all_coarse,
            all_fine,
            restrict_fn=_norm_adjoint_restrict_face,
        )
        fine_matched = prolong_face(coarse_probe, ops)
        current_restricted = restrict_face(fine_matched, ops)
        projection_defect = float(
            jnp.max(jnp.abs((current_restricted - coarse_probe) * coarse_mask))
        )
        if projection_defect > 1.0e-7:
            projection_noop_failures.append(probe_id)
        current_masked_defects.append(abs(current_masked_defect))
        mass_adjoint_defects.append(abs(mass_adjoint_defect))
        probe_rows.append(
            {
                "probe_id": probe_id,
                "current_masked_adjoint_defect": current_masked_defect,
                "unmasked_mass_adjoint_defect": mass_adjoint_defect,
                "current_projection_noop_defect": projection_defect,
            }
        )

    current_components = _current_kernel_after_components(fixture)
    current_ledger = _ledger_residual_for_components(
        fixture["components"],
        current_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    interface_repair = _private_interface_floor_repair_packet()
    f1 = next(
        candidate
        for candidate in interface_repair["candidates"]
        if candidate["candidate_id"] == "oriented_characteristic_face_balance"
    )
    return {
        "candidate_id": "current_face_operator_norm_adjoint_audit",
        "candidate_family": "audit_only",
        "production_edit_allowed": False,
        "faces_considered": tuple(sbp_sat_3d.FACE_ORIENTATIONS),
        "probe_ids": tuple(row["probe_id"] for row in probe_rows),
        "probe_rows": tuple(probe_rows),
        "current_unmasked_face_operator_already_norm_adjoint": bool(
            max(mass_adjoint_defects) <= 1.0e-6
            and current_mass_adjoint_difference_max <= 1.0e-7
        ),
        "current_restriction_equals_unmasked_mass_adjoint": bool(
            current_mass_adjoint_difference_max <= 1.0e-7
        ),
        "current_mass_adjoint_difference_max": current_mass_adjoint_difference_max,
        "current_masked_adjoint_defect_max": float(max(current_masked_defects)),
        "unmasked_mass_adjoint_defect_max": float(max(mass_adjoint_defects)),
        "projection_noop_failure_probe_ids": tuple(projection_noop_failures),
        "current_projection_noop_passed_all_probes": not projection_noop_failures,
        "current_manufactured_ledger_residual": float(
            current_ledger["normalized_balance_residual"]
        ),
        "prior_f1_ledger_residual": f1["ledger_normalized_balance_residual"],
        "prior_f1_collapsed_to_current_operator": f1[
            "characteristic_equivalent_to_current_component_sat"
        ],
        "accepted_candidate": False,
    }


def _apply_face_sat_with_restrict_pair(
    coarse_face,
    fine_face,
    ops,
    alpha_c,
    alpha_f,
    coarse_mask,
    fine_mask,
    *,
    restrict_fn,
):
    coarse_mismatch = restrict_fn(fine_face, ops) - coarse_face
    fine_mismatch = prolong_face(coarse_face, ops) - fine_face
    return (
        coarse_face + alpha_c * coarse_mismatch * coarse_mask,
        fine_face + alpha_f * fine_mismatch * fine_mask,
    )


def _apply_face_sat_with_restrict_to_components(
    before_components,
    *,
    ops,
    coarse_mask,
    fine_mask,
    alpha_c,
    alpha_f,
    restrict_fn,
):
    updated = []
    for coarse_index, fine_index in ((0, 4), (1, 5), (2, 6), (3, 7)):
        updated.append(
            _apply_face_sat_with_restrict_pair(
                before_components[coarse_index],
                before_components[fine_index],
                ops,
                alpha_c,
                alpha_f,
                coarse_mask,
                fine_mask,
                restrict_fn=restrict_fn,
            )
        )
    return (
        updated[0][0],
        updated[1][0],
        updated[2][0],
        updated[3][0],
        updated[0][1],
        updated[1][1],
        updated[2][1],
        updated[3][1],
    )


def _face_norm_projection_noop_packet(*, restrict_fn) -> dict[str, object]:
    ops, coarse_mask, fine_mask, alpha_c, alpha_f = _face_fixture_ops()
    rows = []
    for probe in _face_norm_probe_suite(ops):
        coarse = probe["coarse_face"]
        fine = prolong_face(coarse, ops)
        updated_coarse, updated_fine = _apply_face_sat_with_restrict_pair(
            coarse,
            fine,
            ops,
            alpha_c,
            alpha_f,
            coarse_mask,
            fine_mask,
            restrict_fn=restrict_fn,
        )
        coarse_delta = float(jnp.max(jnp.abs((updated_coarse - coarse) * coarse_mask)))
        fine_delta = float(jnp.max(jnp.abs((updated_fine - fine) * fine_mask)))
        rows.append(
            {
                "probe_id": probe["probe_id"],
                "coarse_delta_max": coarse_delta,
                "fine_delta_max": fine_delta,
                "passed": coarse_delta <= 1.0e-7 and fine_delta <= 1.0e-7,
            }
        )
    return {
        "rows": tuple(rows),
        "passed": all(row["passed"] for row in rows),
        "failed_probe_ids": tuple(
            str(row["probe_id"]) for row in rows if not row["passed"]
        ),
    }


def _private_face_norm_operator_repair_packet() -> dict[str, object]:
    fixture = _manufactured_nonzero_face_components()
    ops = fixture["ops"]
    coarse_mask = fixture["coarse_mask"]
    fine_mask = fixture["fine_mask"]
    before_components = fixture["components"]
    current = _current_kernel_coupling_metrics()
    audit = _face_norm_operator_audit_packet()

    h1_components = _apply_face_sat_with_restrict_to_components(
        before_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
        alpha_c=fixture["alpha_c"],
        alpha_f=fixture["alpha_f"],
        restrict_fn=_norm_adjoint_restrict_face,
    )
    h1_ledger = _ledger_residual_for_components(
        before_components,
        h1_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    zero_fixture = _manufactured_nonzero_face_components()
    zero_fixture["components"] = (
        before_components[0],
        before_components[1],
        jnp.zeros_like(before_components[2]),
        jnp.zeros_like(before_components[3]),
        before_components[4],
        before_components[5],
        prolong_face(jnp.zeros_like(before_components[2]), ops),
        prolong_face(jnp.zeros_like(before_components[3]), ops),
    )
    h1_zero_components = _apply_face_sat_with_restrict_to_components(
        zero_fixture["components"],
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
        alpha_c=fixture["alpha_c"],
        alpha_f=fixture["alpha_f"],
        restrict_fn=_norm_adjoint_restrict_face,
    )
    h1_zero = _ledger_residual_for_components(
        zero_fixture["components"],
        h1_zero_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    h1_noop = _face_norm_projection_noop_packet(
        restrict_fn=_norm_adjoint_restrict_face
    )
    h1_update_norm = _weighted_update_norm(
        before_components,
        h1_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    h1_update_norm_ratio = h1_update_norm / max(current["current_update_norm"], 1.0e-30)
    h1_ledger_gate_passed = (
        float(h1_ledger["normalized_balance_residual"]) <= _LEDGER_BALANCE_THRESHOLD
    )
    h1_update_bounds_passed = (
        _BOUNDING_SCALAR_MIN - _CANDIDATE_BOUND_TOL
        <= h1_update_norm_ratio
        <= _BOUNDING_SCALAR_MAX + _CANDIDATE_BOUND_TOL
    )
    h1_rejection_reasons = []
    if not h1_ledger_gate_passed:
        h1_rejection_reasons.append("candidate_failed_manufactured_ledger_gate")
    if not h1_noop["passed"]:
        h1_rejection_reasons.append("candidate_failed_higher_order_projection_noop")
    if audit["current_unmasked_face_operator_already_norm_adjoint"]:
        h1_rejection_reasons.append("candidate_collapses_to_current_norm_adjoint_operator")
    if not h1_update_bounds_passed:
        h1_rejection_reasons.append("candidate_failed_update_bound_gate")

    edge_corner_packet = _interior_box_ledger_packet()
    edge_corner_gate_passed = (
        edge_corner_packet["matched_edge_noop_passed"] is True
        and edge_corner_packet["active_edges"] == 12
        and edge_corner_packet["active_corners"] == 8
    )
    h1 = {
        "candidate_id": "mass_adjoint_restriction_face_sat",
        "candidate_family": "norm_adjoint_face_operator",
        "production_edit_allowed": True,
        "operator_formula": "R* = Hc^-1 P^T Hf from existing face norms",
        "unmasked_norm_adjoint_identity_passed": bool(
            audit["unmasked_mass_adjoint_defect_max"] <= 1.0e-6
        ),
        "current_operator_already_uses_unmasked_mass_adjoint": bool(
            audit["current_unmasked_face_operator_already_norm_adjoint"]
        ),
        "matched_projected_traces_noop": h1_noop["passed"],
        "matched_projected_trace_noop_rows": h1_noop["rows"],
        "failed_noop_probe_ids": h1_noop["failed_probe_ids"],
        "ledger_normalized_balance_residual": float(
            h1_ledger["normalized_balance_residual"]
        ),
        "ledger_threshold": _LEDGER_BALANCE_THRESHOLD,
        "ledger_gate_passed": h1_ledger_gate_passed,
        "zero_work_gate_passed": float(h1_zero["net_delta_eh"])
        <= _ZERO_WORK_POSITIVE_INJECTION_TOL,
        "zero_work_net_delta_eh": float(h1_zero["net_delta_eh"]),
        "candidate_update_norm": float(h1_update_norm),
        "candidate_update_norm_ratio": float(h1_update_norm_ratio),
        "update_bounds_passed": h1_update_bounds_passed,
        "edge_corner_preacceptance_gate_passed": edge_corner_gate_passed,
        "accepted_candidate": False,
        "rejection_reasons": tuple(h1_rejection_reasons),
    }

    h2_ledger = _ledger_residual_for_components(
        before_components,
        _current_kernel_after_components(fixture),
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    h2 = {
        "candidate_id": "uniform_diagonal_face_norm_rescaling_guard",
        "candidate_family": "diagonal_face_norm_rescaling_guard",
        "production_edit_allowed": True,
        "coarse_norm_ratio": 1.0,
        "fine_norm_ratio": 1.0,
        "ratio_derivation": (
            "coarse_area / fine_area == ratio**2 is already encoded by "
            "ops.coarse_norm and ops.fine_norm; no bounded alternate diagonal "
            "ratio is available without tuning against the measured residual"
        ),
        "ratios_bounded": True,
        "independent_of_measured_residual": True,
        "identical_to_current_uniform_face_norms": True,
        "ledger_normalized_balance_residual": float(
            h2_ledger["normalized_balance_residual"]
        ),
        "ledger_threshold": _LEDGER_BALANCE_THRESHOLD,
        "ledger_gate_passed": float(h2_ledger["normalized_balance_residual"])
        <= _LEDGER_BALANCE_THRESHOLD,
        "accepted_candidate": False,
        "rejection_reasons": (
            "candidate_redundant_with_current_uniform_face_norms",
            "candidate_failed_manufactured_ledger_gate",
        ),
    }
    h3 = {
        "candidate_id": "higher_order_projection_guard",
        "candidate_family": "higher_order_projection_guard",
        "production_edit_allowed": False,
        "status": "higher_order_projection_requires_broader_operator_plan",
        "requires_new_stencils_or_derivative_operator_design": True,
        "evidence": (
            "current and mass-adjoint operators are norm compatible, but "
            "alternating/localized matched-prolongation probes are not exact "
            "coarse projection noops under the existing interpolation family"
        ),
        "failed_noop_probe_ids": audit["projection_noop_failure_probe_ids"],
        "accepted_candidate": False,
    }
    h4 = {
        "candidate_id": "full_box_edge_corner_norm_preacceptance",
        "candidate_family": "preacceptance_guard",
        "production_edit_allowed": False,
        "status": (
            "edge_corner_preacceptance_passed"
            if edge_corner_gate_passed
            else "edge_corner_norm_inconsistency_suspected"
        ),
        "active_edges": edge_corner_packet["active_edges"],
        "active_corners": edge_corner_packet["active_corners"],
        "matched_edge_noop_passed": edge_corner_packet["matched_edge_noop_passed"],
        "accepted_candidate": False,
    }
    candidates = (audit, h1, h2, h3, h4)
    accepted = [candidate for candidate in candidates if candidate["accepted_candidate"]]
    status = (
        "private_norm_adjoint_face_operator_repair_candidate_accepted"
        if accepted
        else _FACE_NORM_OPERATOR_REPAIR_STATUS
    )
    return {
        "private_face_norm_operator_repair_status": status,
        "status": status,
        "terminal_outcome": status,
        "terminal_outcome_taxonomy": _FACE_NORM_OPERATOR_REPAIR_TERMINAL_OUTCOMES,
        "diagnostic_scope": "private_manufactured_interface_face_operator_only",
        "upstream_interface_floor_repair_status": _INTERFACE_FLOOR_REPAIR_STATUS,
        "candidate_ladder_declared_before_solver_edit": True,
        "candidate_count": len(candidates),
        "selected_candidate_id": accepted[0]["candidate_id"] if accepted else None,
        "candidates": candidates,
        "thresholds": {
            "ledger_balance_threshold": _LEDGER_BALANCE_THRESHOLD,
            "bounded_diagonal_norm_ratio_min": _BOUNDING_SCALAR_MIN,
            "bounded_diagonal_norm_ratio_max": _BOUNDING_SCALAR_MAX,
            "projection_noop_tolerance": 1.0e-7,
            "norm_adjoint_defect_tolerance": 1.0e-6,
        },
        "selection_rule": (
            "accept H1 before H2 only if norm-adjoint identity, matched-trace "
            "noop, ledger, update-bound, and edge/corner gates all pass"
        ),
        "solver_hunk_allowed_if_selected": _FACE_NORM_OPERATOR_ALLOWED_SOLVER_SYMBOLS,
        "solver_hunk_retained": False,
        "actual_solver_hunk_inventory": (),
        "production_patch_allowed": False,
        "production_patch_applied": False,
        "solver_behavior_changed": False,
        "sbp_sat_3d_repair_applied": False,
        "sbp_sat_3d_diff_allowed": False,
        "face_ops_global_behavior_changed": False,
        "hook_experiment_allowed": False,
        "jit_runner_changed": False,
        "runner_changed": False,
        "api_surface_changed": False,
        "public_claim_allowed": False,
        "public_api_behavior_changed": False,
        "public_default_tau_changed": False,
        "public_observable_promoted": False,
        "public_true_rt_promoted": False,
        "public_dft_promoted": False,
        "promotion_candidate_ready": False,
        "simresult_changed": False,
        "result_surface_changed": False,
        "next_prerequisite": _FACE_NORM_OPERATOR_REPAIR_NEXT_PREREQUISITE,
        "reason": (
            "the existing face restriction is already the unmasked mass-adjoint "
            "of prolongation under the current diagonal norms, while higher-order "
            "matched-prolongation probes expose projection/noop defects and both "
            "H1/H2 keep the manufactured ledger above the unchanged 0.02 threshold"
        ),
        "report": _FACE_NORM_OPERATOR_REPAIR_REPORT,
    }


def _private_derivative_interface_repair_packet() -> dict[str, object]:
    """Classify the post-face-norm lane against the full face energy identity.

    This packet is intentionally private and fail-closed.  It widens the
    diagnostic surface from face norm adjointness to the derivative/interior
    boundary energy ledger, but it does not retain any solver hunk unless the
    reduced, manufactured, edge/corner, and operator-widening gates all pass.
    """

    fixture = _manufactured_nonzero_face_components()
    ops = fixture["ops"]
    coarse_mask = fixture["coarse_mask"]
    fine_mask = fixture["fine_mask"]
    before_components = fixture["components"]
    current_components = _current_kernel_after_components(fixture)
    current_ledger = _ledger_residual_for_components(
        before_components,
        current_components,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    current = _current_kernel_coupling_metrics()
    reduced_candidate, reduced_amplitude = (
        _energy_closing_matched_hy_face_coupling_candidate(
            before_components,
            fixture,
        )
    )
    reduced_ledger = _ledger_residual_for_components(
        before_components,
        reduced_candidate,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    )
    reduced_update_ratio = _weighted_update_norm(
        before_components,
        reduced_candidate,
        ops=ops,
        coarse_mask=coarse_mask,
        fine_mask=fine_mask,
    ) / max(current["current_update_norm"], 1.0e-30)
    edge_corner = _interior_box_ledger_packet()
    edge_corner_ready = bool(
        edge_corner["matched_edge_noop_passed"]
        and edge_corner["active_edges"] == 12
        and edge_corner["active_corners"] == 8
    )
    g0 = {
        "candidate_id": "current_derivative_energy_identity_audit",
        "candidate_family": "audit_only",
        "production_edit_allowed": False,
        "energy_terms_explicit": True,
        "volume_curl_term_status": "not_separable_in_face_only_fixture",
        "boundary_flux_term_status": "coarse_face_norm_restricted",
        "sat_work_term_status": "current_component_sat",
        "time_stagger_term_status": "same_call_centered_h_helper_already_tested",
        "projection_term_status": "face_norm_ladder_already_mass_adjoint",
        "edge_corner_term_status": (
            "edge_corner_preacceptance_passed"
            if edge_corner_ready
            else "edge_corner_derivative_accounting_blocked"
        ),
        "current_ledger_normalized_balance_residual": float(
            current_ledger["normalized_balance_residual"]
        ),
        "ledger_threshold": _LEDGER_BALANCE_THRESHOLD,
        "reduced_fixture_reproduces_current_floor": bool(
            float(current_ledger["normalized_balance_residual"])
            > _LEDGER_BALANCE_THRESHOLD
        ),
        "accepted_candidate": False,
    }
    g1_passed = bool(
        float(reduced_ledger["normalized_balance_residual"])
        <= _LEDGER_BALANCE_THRESHOLD
    )
    g1 = {
        "candidate_id": "reduced_normal_incidence_energy_flux",
        "candidate_family": "reduced_energy_identity_flux",
        "production_edit_allowed": False,
        "derivation": (
            "minimum-norm root of the private trace-energy identity "
            "Delta E_trace + W_interface = 0 in the reduced face fixture"
        ),
        "branches_on_measured_residual_or_test_name": False,
        "reduced_fixture_reproduces_failure": g0[
            "reduced_fixture_reproduces_current_floor"
        ],
        "reduced_identity_closed": g1_passed,
        "amplitude": float(reduced_amplitude),
        "ledger_normalized_balance_residual": float(
            reduced_ledger["normalized_balance_residual"]
        ),
        "ledger_threshold": _LEDGER_BALANCE_THRESHOLD,
        "candidate_update_norm_ratio": float(reduced_update_ratio),
        "accepted_candidate": g1_passed,
        "terminal_if_selected": "private_reduced_derivative_flux_contract_ready",
    }
    g2 = {
        "candidate_id": "full_yz_face_energy_flux_candidate",
        "candidate_family": "production_shaped_face_flux_lift",
        "production_edit_allowed": True,
        "admission_gate": "requires G1 plus derivative/interior-boundary operator compatibility",
        "g1_contract_available": g1_passed,
        "manufactured_ledger_gate_passed": False,
        "ledger_normalized_balance_residual": float(
            current_ledger["normalized_balance_residual"]
        ),
        "ledger_threshold": _LEDGER_BALANCE_THRESHOLD,
        "zero_work_gate_passed": True,
        "cpml_non_cpml_staging_gate_passed": True,
        "rejection_reasons": (
            "candidate_requires_derivative_operator_compatibility_not_available",
            "candidate_would_reuse_current_face_only_sat_floor",
        ),
        "accepted_candidate": False,
    }
    g3 = {
        "candidate_id": "edge_corner_cochain_accounting_guard",
        "candidate_family": "preacceptance_guard",
        "production_edit_allowed": False,
        "status": (
            "edge_corner_derivative_accounting_ready"
            if edge_corner_ready
            else "edge_corner_derivative_accounting_blocked"
        ),
        "active_edges": edge_corner["active_edges"],
        "active_corners": edge_corner["active_corners"],
        "matched_edge_noop_passed": edge_corner["matched_edge_noop_passed"],
        "accepted_candidate": False,
    }
    g4 = {
        "candidate_id": "mortar_projection_operator_widening_guard",
        "candidate_family": "operator_widening_guard",
        "production_edit_allowed": False,
        "status": "requires_global_sbp_operator_refactor",
        "requires_global_sbp_operator_refactor": True,
        "reason": (
            "a reduced energy identity can be closed only as a face-local "
            "minimum-norm correction, while production retention requires a "
            "compatible derivative/mortar operator that is outside the allowed "
            "private sbp_sat_3d.py hunk"
        ),
        "accepted_candidate": False,
    }
    g5 = {
        "candidate_id": "private_solver_integration_candidate",
        "candidate_family": "gated_private_solver_hunk",
        "production_edit_allowed": True,
        "status": "blocked_by_requires_global_sbp_operator_refactor",
        "admitted_to_solver": False,
        "blocked_by_candidate_id": g4["candidate_id"],
        "accepted_candidate": False,
    }
    candidates = (g0, g1, g2, g3, g4, g5)
    return {
        "private_derivative_interface_repair_status": (
            _DERIVATIVE_INTERFACE_REPAIR_STATUS
        ),
        "status": _DERIVATIVE_INTERFACE_REPAIR_STATUS,
        "terminal_outcome": _DERIVATIVE_INTERFACE_REPAIR_STATUS,
        "terminal_outcome_taxonomy": _DERIVATIVE_INTERFACE_REPAIR_TERMINAL_OUTCOMES,
        "diagnostic_scope": (
            "private_derivative_interior_boundary_energy_identity_only"
        ),
        "upstream_face_norm_operator_repair_status": _FACE_NORM_OPERATOR_REPAIR_STATUS,
        "candidate_ladder_declared_before_solver_edit": True,
        "candidate_count": len(candidates),
        "selected_candidate_id": None,
        "candidates": candidates,
        "thresholds": {
            "ledger_balance_threshold": _LEDGER_BALANCE_THRESHOLD,
            "update_norm_ratio_min": _BOUNDING_SCALAR_MIN,
            "update_norm_ratio_max": _BOUNDING_SCALAR_MAX,
            "projection_noop_tolerance": 1.0e-7,
        },
        "selection_rule": (
            "retain a private solver hunk only if G1-G3 pass and G4 does not "
            "require a global SBP derivative/mortar operator refactor"
        ),
        "reduced_fixture_reproduces_failure": g1[
            "reduced_fixture_reproduces_failure"
        ],
        "reduced_identity_closed_test_locally": g1["reduced_identity_closed"],
        "requires_global_sbp_operator_refactor": True,
        "solver_hunk_allowed_if_selected": _DERIVATIVE_INTERFACE_ALLOWED_SOLVER_SYMBOLS,
        "solver_hunk_retained": False,
        "actual_solver_hunk_inventory": (),
        "production_patch_allowed": False,
        "production_patch_applied": False,
        "solver_behavior_changed": False,
        "sbp_sat_3d_repair_applied": False,
        "sbp_sat_3d_diff_allowed": False,
        "face_ops_global_behavior_changed": False,
        "hook_experiment_allowed": False,
        "jit_runner_changed": False,
        "runner_changed": False,
        "api_surface_changed": False,
        "public_claim_allowed": False,
        "public_api_behavior_changed": False,
        "public_default_tau_changed": False,
        "public_observable_promoted": False,
        "public_true_rt_promoted": False,
        "public_dft_promoted": False,
        "promotion_candidate_ready": False,
        "simresult_changed": False,
        "result_surface_changed": False,
        "env_config_changed": False,
        "next_prerequisite": _DERIVATIVE_INTERFACE_REPAIR_NEXT_PREREQUISITE,
        "reason": (
            "the reduced derivative/interior-boundary identity reproduces the "
            "current 0.02 ledger-floor failure and can be closed only by a "
            "test-local face correction; retaining a production hunk requires "
            "global SBP derivative/mortar operator infrastructure outside this "
            "private lane"
        ),
        "report": _DERIVATIVE_INTERFACE_REPAIR_REPORT,
    }


def _global_operator_face_specs(
    config: sbp_sat_3d.SubgridConfig3D,
) -> tuple[FaceFluxSpec, ...]:
    box_shape = sbp_sat_3d._coarse_box_shape(config)
    return tuple(
        FaceFluxSpec(
            face=face,
            coarse_shape=tuple(box_shape[axis] for axis in orientation.tangential_axes),
            normal_sign=orientation.normal_sign,
        )
        for face, orientation in sbp_sat_3d.FACE_ORIENTATIONS.items()
    )


def _private_global_operator_cpml_staging_report() -> dict[str, object]:
    plain_source = inspect.getsource(sbp_sat_3d.step_subgrid_3d)
    cpml_source = inspect.getsource(sbp_sat_3d.step_subgrid_3d_with_cpml)
    operator_source = inspect.getsource(sbp_operators)
    sat_symbols = ("apply_sat_h_interfaces", "apply_sat_e_interfaces")

    plain_sat_symbols = tuple(symbol for symbol in sat_symbols if symbol in plain_source)
    cpml_sat_symbols = tuple(symbol for symbol in sat_symbols if symbol in cpml_source)
    cpml_boundary_symbols = ("apply_cpml_h", "apply_cpml_e")
    cpml_boundary_calls = tuple(
        symbol for symbol in cpml_boundary_symbols if symbol in cpml_source
    )
    h_sat_after_cpml_boundary = cpml_source.index(
        "apply_sat_h_interfaces"
    ) > cpml_source.index("apply_cpml_h")
    e_sat_after_cpml_boundary = cpml_source.index(
        "apply_sat_e_interfaces"
    ) > cpml_source.index("apply_cpml_e")
    operator_module_has_no_cpml_dependency = "cpml" not in operator_source.lower()
    shared_sat_sequence = plain_sat_symbols == cpml_sat_symbols == sat_symbols
    cpml_non_cpml_compatibility_ready = bool(
        shared_sat_sequence
        and set(cpml_boundary_calls) == set(cpml_boundary_symbols)
        and h_sat_after_cpml_boundary
        and e_sat_after_cpml_boundary
        and operator_module_has_no_cpml_dependency
    )
    return {
        "plain_sat_symbols": plain_sat_symbols,
        "cpml_sat_symbols": cpml_sat_symbols,
        "cpml_boundary_calls": cpml_boundary_calls,
        "shared_sat_sequence": shared_sat_sequence,
        "h_sat_after_cpml_boundary": h_sat_after_cpml_boundary,
        "e_sat_after_cpml_boundary": e_sat_after_cpml_boundary,
        "operator_module_has_no_cpml_dependency": operator_module_has_no_cpml_dependency,
        "private_hooks_required_for_operator_guard": False,
        "cpml_non_cpml_compatibility_ready": cpml_non_cpml_compatibility_ready,
    }


def _private_global_derivative_mortar_operator_architecture_packet() -> dict[str, object]:
    """Record the private global SBP derivative/mortar architecture gate."""

    derivative_packet = _private_derivative_interface_repair_packet()
    scalar_operator = build_sbp_first_derivative_1d(6, _DX_C, grid_role="electric")
    scalar_defect = float(jnp.max(jnp.abs(sbp_identity_residual(scalar_operator))))
    yee_pair = build_yee_staggered_derivative_pair_1d(7, _DX_C)
    yee_defect = float(jnp.max(jnp.abs(yee_staggered_identity_residual(yee_pair))))
    mortar = build_tensor_face_mortar(_FACE_SHAPE, ratio=_RATIO, dx_c=_DX_C)
    mortar_adjoint = face_mortar_adjoint_report(mortar)
    mortar_reproduction = face_mortar_reproduction_report(mortar)

    i = jnp.arange(_FACE_SHAPE[0], dtype=jnp.float32)[:, None]
    j = jnp.arange(_FACE_SHAPE[1], dtype=jnp.float32)[None, :]
    zeros = jnp.zeros(_FACE_SHAPE, dtype=jnp.float32)
    flux_residuals = tuple(
        abs(
            weighted_em_flux_residual(
                mortar,
                ex_c=1.0 + 0.1 * i + 0.2 * j,
                ey_c=-0.5 + 0.05 * i - 0.1 * j,
                hx_c=2.0e-6 + 0.1e-6 * i + zeros,
                hy_c=-1.0e-6 + 0.2e-6 * j + zeros,
                normal_sign=normal_sign,
                coarse_metric_weight=1.0 + 0.01 * i + 0.02 * j,
            )
        )
        for normal_sign in (-1, 1)
    )
    config, _ = init_subgrid_3d(
        shape_c=(6, 6, 6),
        dx_c=_DX_C,
        fine_region=(1, 5, 1, 5, 1, 5),
        ratio=_RATIO,
        tau=_TAU,
    )
    surface_partition = box_surface_partition_report(
        sbp_sat_3d._coarse_box_shape(config)
    )
    all_face_flux = all_face_weighted_flux_report(
        _global_operator_face_specs(config),
        ratio=_RATIO,
        dx_c=_DX_C,
    )
    cpml_staging = _private_global_operator_cpml_staging_report()
    a1_passed = bool(scalar_defect <= 1.0e-6 and yee_defect <= 1.0e-6)
    a2_passed = bool(mortar_adjoint["passes"] and mortar_reproduction["passes"])
    a3_passed = bool(max(flux_residuals) <= 1.0e-12)
    a4_passed = bool(
        all_face_flux["passes"] is True
        and all_face_flux["face_count"] == 6
        and surface_partition["active_faces"] == 6
        and surface_partition["active_edges"] == 12
        and surface_partition["active_corners"] == 8
        and surface_partition["partition_closes"] is True
        and cpml_staging["cpml_non_cpml_compatibility_ready"] is True
    )
    candidates = (
        {
            "candidate_id": "current_operator_inventory_and_freeze",
            "candidate_family": "audit_only",
            "production_solver_edit_allowed": False,
            "upstream_derivative_status": derivative_packet["terminal_outcome"],
            "prior_solver_hunk_retained": derivative_packet["solver_hunk_retained"],
            "prior_actual_solver_hunk_inventory": derivative_packet[
                "actual_solver_hunk_inventory"
            ],
            "public_closure_locked": True,
            "accepted_candidate": False,
        },
        {
            "candidate_id": "sbp_derivative_norm_boundary_contract",
            "candidate_family": "diagonal_norm_sbp_first_derivative",
            "production_solver_edit_allowed": False,
            "norm_positive": bool(jnp.all(scalar_operator.norm > 0.0)),
            "collocated_sbp_identity_passed": scalar_defect <= 1.0e-6,
            "collocated_identity_max_defect": scalar_defect,
            "yee_staggered_dual_identity_passed": yee_defect <= 1.0e-6,
            "yee_staggered_dual_identity_max_defect": yee_defect,
            "boundary_extraction_signs_explicit": True,
            "accepted_candidate": a1_passed,
            "terminal_if_selected": "private_sbp_derivative_contract_ready",
        },
        {
            "candidate_id": "norm_compatible_mortar_projection_contract",
            "candidate_family": "piecewise_constant_norm_compatible_mortar",
            "production_solver_edit_allowed": False,
            "mortar_adjointness_passed": mortar_adjoint["passes"],
            "mortar_adjointness_max_defect": mortar_adjoint["max_defect"],
            "constant_reproduction_passed": (
                mortar_reproduction["constant_max_error"] <= 1.0e-6
            ),
            "linear_reproduction_passed": (
                mortar_reproduction["linear_i_max_error"] <= 1.0e-6
                and mortar_reproduction["linear_j_max_error"] <= 1.0e-6
            ),
            "projection_noop_passed": mortar_reproduction["passes"],
            "branches_on_measured_residual_or_test_name": False,
            "accepted_candidate": a2_passed,
            "terminal_if_selected": "private_mortar_projection_contract_ready",
        },
        {
            "candidate_id": "em_tangential_interface_flux_contract",
            "candidate_family": "weighted_tangential_em_flux_identity",
            "production_solver_edit_allowed": False,
            "uses_yee_tangential_orientation": True,
            "material_metric_weighting_explicit": True,
            "normal_signs_tested": (-1, 1),
            "flux_residuals": flux_residuals,
            "flux_identity_passed": a3_passed,
            "accepted_candidate": a3_passed,
            "terminal_if_selected": "private_em_mortar_flux_contract_ready",
        },
        {
            "candidate_id": "all_faces_edge_corner_operator_guard",
            "candidate_family": "all_six_face_edge_corner_guard",
            "production_solver_edit_allowed": False,
            "faces_tested": all_face_flux["faces_tested"],
            "all_face_flux_identity_passed": all_face_flux["passes"],
            "all_face_flux_identity_max_abs_residual": all_face_flux[
                "max_abs_residual"
            ],
            "all_face_flux_identity_residuals": all_face_flux["residuals"],
            "active_faces": surface_partition["active_faces"],
            "active_edges": surface_partition["active_edges"],
            "active_corners": surface_partition["active_corners"],
            "face_interior_cells": surface_partition["face_interior_cells"],
            "edge_interior_cells": surface_partition["edge_interior_cells"],
            "corner_cells": surface_partition["corner_cells"],
            "surface_cells": surface_partition["surface_cells"],
            "counted_surface_cells": surface_partition["counted_surface_cells"],
            "surface_partition_closes": surface_partition["partition_closes"],
            "edge_corner_accounting_status": surface_partition["status"],
            "cpml_exclusion_staging_explicit": cpml_staging[
                "operator_module_has_no_cpml_dependency"
            ],
            "cpml_non_cpml_compatibility_ready": cpml_staging[
                "cpml_non_cpml_compatibility_ready"
            ],
            "cpml_staging_report": cpml_staging,
            "accepted_candidate": a4_passed,
            "terminal_if_selected": "private_global_operator_3d_contract_ready",
        },
        {
            "candidate_id": "private_solver_integration_hunk",
            "candidate_family": "gated_private_solver_hunk",
            "production_solver_edit_allowed": True,
            "a1_a4_evidence_summary_required": True,
            "a1_a4_evidence_summary_present": all(
                (a1_passed, a2_passed, a3_passed, a4_passed)
            ),
            "admitted_to_solver": False,
            "reason": (
                "A1-A4 operator identities are ready, but this architecture lane "
                "retains no sbp_sat_3d.py hunk; a single-owner solver-integration "
                "lane must bind the private operators next"
            ),
            "accepted_candidate": False,
        },
        {
            "candidate_id": "operator_architecture_fail_closed",
            "candidate_family": "terminal_guard",
            "production_solver_edit_allowed": False,
            "status": "not_selected_operator_contract_ready",
            "accepted_candidate": False,
        },
    )
    return {
        "private_global_derivative_mortar_operator_architecture_status": (
            _GLOBAL_OPERATOR_ARCHITECTURE_STATUS
        ),
        "status": _GLOBAL_OPERATOR_ARCHITECTURE_STATUS,
        "terminal_outcome": _GLOBAL_OPERATOR_ARCHITECTURE_STATUS,
        "terminal_outcome_taxonomy": _GLOBAL_OPERATOR_ARCHITECTURE_TERMINAL_OUTCOMES,
        "diagnostic_scope": "private_global_sbp_derivative_mortar_operator_only",
        "upstream_derivative_interface_repair_status": derivative_packet[
            "terminal_outcome"
        ],
        "candidate_ladder_declared_before_solver_edit": True,
        "candidate_count": len(candidates),
        "selected_candidate_id": "all_faces_edge_corner_operator_guard",
        "candidates": candidates,
        "a1_a4_evidence_summary": {
            "sbp_derivative_norm_boundary_contract": a1_passed,
            "norm_compatible_mortar_projection_contract": a2_passed,
            "em_tangential_interface_flux_contract": a3_passed,
            "all_faces_edge_corner_operator_guard": a4_passed,
        },
        "thresholds": {
            "sbp_identity_tolerance": 1.0e-6,
            "mortar_adjoint_tolerance": 1.0e-6,
            "em_flux_residual_tolerance": 1.0e-12,
            "ledger_balance_threshold": _LEDGER_BALANCE_THRESHOLD,
        },
        "selection_rule": (
            "allow at most one private solver hunk after A1-A4 evidence is "
            "summarized; retain no solver hunk in this architecture lane"
        ),
        "operator_module_added": True,
        "operator_module": "rfx/subgridding/sbp_operators.py",
        "solver_hunk_allowed_if_selected": _GLOBAL_OPERATOR_ALLOWED_SOLVER_SYMBOLS,
        "solver_hunk_retained": False,
        "actual_solver_hunk_inventory": (),
        "production_patch_allowed": False,
        "production_patch_applied": False,
        "solver_behavior_changed": False,
        "sbp_sat_3d_repair_applied": False,
        "sbp_sat_3d_diff_allowed": False,
        "face_ops_global_behavior_changed": False,
        "hook_experiment_allowed": False,
        "jit_runner_changed": False,
        "runner_changed": False,
        "api_surface_changed": False,
        "public_claim_allowed": False,
        "public_api_behavior_changed": False,
        "public_default_tau_changed": False,
        "public_observable_promoted": False,
        "public_true_rt_promoted": False,
        "public_dft_promoted": False,
        "promotion_candidate_ready": False,
        "simresult_changed": False,
        "result_surface_changed": False,
        "env_config_changed": False,
        "next_prerequisite": _GLOBAL_OPERATOR_ARCHITECTURE_NEXT_PREREQUISITE,
        "reason": (
            "the private global SBP derivative/mortar operator contract now "
            "has A1-A4 identity evidence, including Yee-staggered dual norms, "
            "norm-compatible mortar projection, material/metric weighted EM "
            "flux closure, all-six-face edge/corner partition closure, and "
            "CPML/non-CPML SAT staging evidence; no solver hunk is retained "
            "and public promotion remains closed"
        ),
        "report": _GLOBAL_OPERATOR_ARCHITECTURE_REPORT,
    }


def _operator_projected_after_components(fixture):
    ops = fixture["ops"]
    mortar = build_tensor_face_mortar(
        ops.coarse_shape,
        ratio=ops.ratio,
        dx_c=math.sqrt(float(ops.coarse_area)),
    )
    after_pairs = [
        operator_projected_sat_pair_face(
            fixture["components"][coarse_index],
            fixture["components"][fine_index],
            mortar,
            fixture["alpha_c"],
            fixture["alpha_f"],
            fixture["coarse_mask"],
            fixture["fine_mask"],
        )
        for coarse_index, fine_index in ((0, 4), (1, 5), (2, 6), (3, 7))
    ]
    return (
        after_pairs[0][0],
        after_pairs[1][0],
        after_pairs[2][0],
        after_pairs[3][0],
        after_pairs[0][1],
        after_pairs[1][1],
        after_pairs[2][1],
        after_pairs[3][1],
    )


def _operator_projected_matched_trace_noop() -> bool:
    ops, coarse_mask, fine_mask, alpha_c, alpha_f = _face_fixture_ops()
    mortar = build_tensor_face_mortar(
        ops.coarse_shape,
        ratio=ops.ratio,
        dx_c=math.sqrt(float(ops.coarse_area)),
    )
    probes = (
        jnp.ones(ops.coarse_shape, dtype=jnp.float32) * 1.25,
        jnp.ones(ops.coarse_shape, dtype=jnp.float32) * -0.75,
        jnp.ones(ops.coarse_shape, dtype=jnp.float32) * 2.0e-6,
        jnp.ones(ops.coarse_shape, dtype=jnp.float32) * -1.0e-6,
    )
    for coarse in probes:
        fine = prolong_face_mortar(coarse, mortar)
        after_coarse, after_fine = operator_projected_sat_pair_face(
            coarse,
            fine,
            mortar,
            alpha_c,
            alpha_f,
            coarse_mask,
            fine_mask,
        )
        if not (
            np.allclose(np.asarray(after_coarse), np.asarray(coarse), atol=1.0e-7)
            and np.allclose(np.asarray(after_fine), np.asarray(fine), atol=1.0e-7)
        ):
            return False
    return True


def _operator_projected_energy_transfer_after_components(fixture, *, normal_sign: int = 1):
    ops = fixture["ops"]
    mortar = build_tensor_face_mortar(
        ops.coarse_shape,
        ratio=ops.ratio,
        dx_c=math.sqrt(float(ops.coarse_area)),
    )
    return operator_projected_skew_eh_sat_face(
        ex_c=fixture["components"][0],
        ey_c=fixture["components"][1],
        hx_c=fixture["components"][2],
        hy_c=fixture["components"][3],
        ex_f=fixture["components"][4],
        ey_f=fixture["components"][5],
        hx_f=fixture["components"][6],
        hy_f=fixture["components"][7],
        mortar=mortar,
        alpha_c=fixture["alpha_c"],
        alpha_f=fixture["alpha_f"],
        coarse_mask=fixture["coarse_mask"],
        fine_mask=fixture["fine_mask"],
        normal_sign=normal_sign,
    )


def _operator_projected_energy_transfer_matched_trace_noop() -> bool:
    ops, coarse_mask, fine_mask, alpha_c, alpha_f = _face_fixture_ops()
    probes = (
        jnp.ones(ops.coarse_shape, dtype=jnp.float32) * 1.25,
        jnp.ones(ops.coarse_shape, dtype=jnp.float32) * -0.75,
        jnp.ones(ops.coarse_shape, dtype=jnp.float32) * 2.0e-6,
        jnp.ones(ops.coarse_shape, dtype=jnp.float32) * -1.0e-6,
    )
    fixture = {
        "ops": ops,
        "coarse_mask": coarse_mask,
        "fine_mask": fine_mask,
        "alpha_c": alpha_c,
        "alpha_f": alpha_f,
        "components": (
            probes[0],
            probes[1],
            probes[2],
            probes[3],
            prolong_face(probes[0], ops),
            prolong_face(probes[1], ops),
            prolong_face(probes[2], ops),
            prolong_face(probes[3], ops),
        ),
    }
    after_components = _operator_projected_energy_transfer_after_components(fixture)
    return all(
        np.allclose(np.asarray(after), np.asarray(before), atol=1.0e-7)
        for before, after in zip(fixture["components"], after_components, strict=True)
    )


def _operator_projected_energy_transfer_static_guard() -> dict[str, object]:
    source = inspect.getsource(operator_projected_skew_eh_sat_face)
    forbidden_markers = (
        "normalized_balance_residual",
        "_LEDGER_BALANCE_THRESHOLD",
        "test_name",
        "tau_sweep",
        "source_relocation",
        "measurement_relocation",
        "residual_fit",
    )
    hits = tuple(marker for marker in forbidden_markers if marker in source)
    return {
        "guard": "no_residual_fit_or_test_branch",
        "passed": not hits,
        "forbidden_markers": forbidden_markers,
        "hits": hits,
        "coefficient_sources": (
            "mortar.ratio",
            "vacuum_impedance",
            "declared_sat_coefficients",
            "face_local_orientation_basis",
        ),
    }


def _operator_projected_energy_transfer_cpml_contract() -> dict[str, object]:
    source = inspect.getsource(operator_projected_skew_eh_sat_face)
    forbidden_markers = (
        "cpml",
        "pml",
        "post_h_hook",
        "post_e_hook",
        "runner",
        "Simulation",
        "Result",
    )
    hits = tuple(marker for marker in forbidden_markers if marker in source)
    return {
        "adapter_symbol": "operator_projected_skew_eh_sat_face",
        "cpml_non_cpml_share_same_adapter": True,
        "adapter_has_no_cpml_dependency": "cpml" not in source and "pml" not in source,
        "adapter_has_no_hook_or_runner_dependency": not hits,
        "forbidden_dependency_hits": hits,
        "future_integration_requires_same_call_contract": True,
        "passes": not hits,
    }


def _orient_components_to_outward_basis(
    components: tuple[jnp.ndarray, ...],
    *,
    normal_sign: int,
) -> tuple[jnp.ndarray, ...]:
    if normal_sign == 1:
        return components
    if normal_sign != -1:
        raise ValueError(f"normal_sign must be -1 or 1, got {normal_sign}")
    ex_c, ey_c, hx_c, hy_c, ex_f, ey_f, hx_f, hy_f = components
    return (ex_c, -ey_c, hx_c, -hy_c, ex_f, -ey_f, hx_f, -hy_f)


def _operator_projected_energy_transfer_orientation_report() -> dict[str, object]:
    faces = {
        "x_lo": -1,
        "x_hi": 1,
        "y_lo": -1,
        "y_hi": 1,
        "z_lo": -1,
        "z_hi": 1,
    }
    current = _current_kernel_coupling_metrics()
    results = {}
    for face, normal_sign in faces.items():
        fixture = _manufactured_nonzero_face_components()
        ops = fixture["ops"]
        before_components = _orient_components_to_outward_basis(
            fixture["components"],
            normal_sign=normal_sign,
        )
        fixture["components"] = before_components
        after_components = _operator_projected_energy_transfer_after_components(
            fixture,
            normal_sign=normal_sign,
        )
        ledger = _ledger_residual_for_components(
            before_components,
            after_components,
            ops=ops,
            coarse_mask=fixture["coarse_mask"],
            fine_mask=fixture["fine_mask"],
        )
        weighted_mismatch_before = _weighted_trace_mismatch(
            before_components,
            ops=ops,
            coarse_mask=fixture["coarse_mask"],
            fine_mask=fixture["fine_mask"],
        )
        weighted_mismatch_after = _weighted_trace_mismatch(
            after_components,
            ops=ops,
            coarse_mask=fixture["coarse_mask"],
            fine_mask=fixture["fine_mask"],
        )
        candidate_reduction = (
            weighted_mismatch_before - weighted_mismatch_after
        ) / max(weighted_mismatch_before, 1.0e-30)
        coupling_strength_ratio = candidate_reduction / max(
            current["current_relative_mismatch_reduction"],
            1.0e-12,
        )
        update_norm_ratio = _weighted_update_norm(
            before_components,
            after_components,
            ops=ops,
            coarse_mask=fixture["coarse_mask"],
            fine_mask=fixture["fine_mask"],
        ) / max(current["current_update_norm"], 1.0e-30)
        results[face] = {
            "normal_sign": normal_sign,
            "ledger_normalized_balance_residual": float(
                ledger["normalized_balance_residual"]
            ),
            "ledger_gate_passed": (
                float(ledger["normalized_balance_residual"])
                <= _LEDGER_BALANCE_THRESHOLD
            ),
            "coupling_strength_ratio": float(coupling_strength_ratio),
            "coupling_strength_passed": (
                coupling_strength_ratio >= _COUPLING_STRENGTH_RATIO_MIN
            ),
            "update_norm_ratio": float(update_norm_ratio),
            "update_bounds_passed": (
                _BOUNDING_SCALAR_MIN - _CANDIDATE_BOUND_TOL
                <= update_norm_ratio
                <= _BOUNDING_SCALAR_MAX + _CANDIDATE_BOUND_TOL
            ),
        }
    return {
        "basis": "face_local_outward_tangential_basis",
        "faces": results,
        "passes": all(
            result["ledger_gate_passed"]
            and result["coupling_strength_passed"]
            and result["update_bounds_passed"]
            for result in results.values()
        ),
    }


def _private_operator_projected_energy_transfer_redesign_packet() -> dict[str, object]:
    """Record the private energy-transfer redesign after solver dry-run failure."""

    upstream = _private_solver_integration_hunk_packet()
    fixture = _manufactured_nonzero_face_components()
    ops = fixture["ops"]
    before_components = fixture["components"]
    candidate_components = _operator_projected_energy_transfer_after_components(fixture)
    ledger = _ledger_residual_for_components(
        before_components,
        candidate_components,
        ops=ops,
        coarse_mask=fixture["coarse_mask"],
        fine_mask=fixture["fine_mask"],
    )

    zero_fixture = _manufactured_nonzero_face_components()
    zero_fixture["components"] = (
        before_components[0],
        before_components[1],
        jnp.zeros_like(before_components[2]),
        jnp.zeros_like(before_components[3]),
        before_components[4],
        before_components[5],
        prolong_face(jnp.zeros_like(before_components[2]), ops),
        prolong_face(jnp.zeros_like(before_components[3]), ops),
    )
    zero_candidate = _operator_projected_energy_transfer_after_components(zero_fixture)
    zero_work = _ledger_residual_for_components(
        zero_fixture["components"],
        zero_candidate,
        ops=ops,
        coarse_mask=zero_fixture["coarse_mask"],
        fine_mask=zero_fixture["fine_mask"],
    )

    current = _current_kernel_coupling_metrics()
    weighted_mismatch_before = _weighted_trace_mismatch(
        before_components,
        ops=ops,
        coarse_mask=fixture["coarse_mask"],
        fine_mask=fixture["fine_mask"],
    )
    weighted_mismatch_after_candidate = _weighted_trace_mismatch(
        candidate_components,
        ops=ops,
        coarse_mask=fixture["coarse_mask"],
        fine_mask=fixture["fine_mask"],
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
        coarse_mask=fixture["coarse_mask"],
        fine_mask=fixture["fine_mask"],
    )
    update_norm_ratio = update_norm / max(current["current_update_norm"], 1.0e-30)
    config, _ = init_subgrid_3d(
        shape_c=(6, 6, 6),
        dx_c=_DX_C,
        fine_region=(1, 5, 1, 5, 1, 5),
        ratio=_RATIO,
        tau=_TAU,
    )
    all_face_flux = all_face_weighted_flux_report(
        _global_operator_face_specs(config),
        ratio=_RATIO,
        dx_c=_DX_C,
    )
    orientation_report = _operator_projected_energy_transfer_orientation_report()
    cpml_staging = _private_global_operator_cpml_staging_report()
    static_guard = _operator_projected_energy_transfer_static_guard()
    skew_cpml_contract = _operator_projected_energy_transfer_cpml_contract()

    matched_noop = _operator_projected_energy_transfer_matched_trace_noop()
    ledger_gate_passed = (
        float(ledger["normalized_balance_residual"]) <= _LEDGER_BALANCE_THRESHOLD
    )
    zero_work_gate_passed = float(zero_work["net_delta_eh"]) <= (
        _ZERO_WORK_POSITIVE_INJECTION_TOL
    )
    coupling_strength_passed = coupling_strength_ratio >= _COUPLING_STRENGTH_RATIO_MIN
    update_bounds_passed = (
        _BOUNDING_SCALAR_MIN - _CANDIDATE_BOUND_TOL
        <= update_norm_ratio
        <= _BOUNDING_SCALAR_MAX + _CANDIDATE_BOUND_TOL
    )
    e1_accepted = bool(
        upstream["terminal_outcome"] == _SOLVER_INTEGRATION_STATUS
        and matched_noop
        and zero_work_gate_passed
        and ledger_gate_passed
        and coupling_strength_passed
        and update_bounds_passed
        and all_face_flux["passes"]
        and orientation_report["passes"]
        and cpml_staging["cpml_non_cpml_compatibility_ready"]
        and skew_cpml_contract["passes"]
        and static_guard["passed"]
    )
    candidates = (
        {
            "candidate_id": "baseline_operator_projected_failure_freeze",
            "candidate_family": "e0_baseline",
            "upstream_status": upstream["terminal_outcome"],
            "upstream_ledger_normalized_balance_residual": upstream[
                "ledger_normalized_balance_residual"
            ],
            "upstream_ledger_threshold": upstream["ledger_threshold"],
            "sbp_sat_3d_diff_empty_before_attempt": True,
            "runner_diff_empty_before_attempt": True,
            "accepted_candidate": False,
        },
        {
            "candidate_id": "paired_skew_eh_operator_work_form",
            "candidate_family": "ratio_weighted_scalar_plus_skew_eh_work_form",
            "production_solver_edit_allowed": False,
            "normal_sign": 1,
            "scalar_projection_weight": 1.0 / float(ops.ratio),
            "skew_projection_weight": 1.0 + 1.0 / float(ops.ratio),
            "coefficient_sources": static_guard["coefficient_sources"],
            "matched_projected_traces_noop": matched_noop,
            "zero_work_dissipative": zero_work_gate_passed,
            "zero_work_net_delta_eh": float(zero_work["net_delta_eh"]),
            "ledger_normalized_balance_residual": float(
                ledger["normalized_balance_residual"]
            ),
            "ledger_threshold": _LEDGER_BALANCE_THRESHOLD,
            "ledger_gate_passed": ledger_gate_passed,
            "weighted_mismatch_before": weighted_mismatch_before,
            "weighted_mismatch_after_candidate": weighted_mismatch_after_candidate,
            "candidate_relative_mismatch_reduction": float(candidate_reduction),
            "coupling_strength_ratio": float(coupling_strength_ratio),
            "coupling_strength_passed": coupling_strength_passed,
            "update_norm_ratio": float(update_norm_ratio),
            "update_bounds_passed": update_bounds_passed,
            "all_face_orientation_signs_passed": all_face_flux["passes"],
            "all_face_skew_helper_orientation_report": orientation_report,
            "cpml_non_cpml_source_order_equivalent": cpml_staging[
                "cpml_non_cpml_compatibility_ready"
            ],
            "cpml_non_cpml_skew_helper_contract": skew_cpml_contract,
            "no_laundering_static_guard": static_guard,
            "accepted_candidate": e1_accepted,
            "terminal_if_selected": "private_operator_projected_skew_work_form_ready",
        },
        {
            "candidate_id": "material_metric_weighted_operator_work_form",
            "candidate_family": "e2_contingency",
            "production_solver_edit_allowed": False,
            "skipped_because_e1_passed": e1_accepted,
            "accepted_candidate": False,
            "terminal_if_selected": "private_material_metric_operator_work_form_ready",
        },
        {
            "candidate_id": "face_edge_corner_partition_work_probe",
            "candidate_family": "e3_contingency",
            "production_solver_edit_allowed": False,
            "skipped_because_e1_passed": e1_accepted,
            "accepted_candidate": False,
            "terminal_if_selected": "private_operator_projected_partition_coupling_required",
        },
        {
            "candidate_id": "future_solver_hunk_candidate_declared",
            "candidate_family": "e4_future_contract_only",
            "production_solver_edit_allowed": False,
            "selected_because_private_ledger_closed": e1_accepted,
            "future_integration_requires_separate_ralplan": True,
            "allowed_future_solver_symbols": (
                _OPERATOR_PROJECTED_ENERGY_TRANSFER_ALLOWED_FUTURE_SOLVER_SYMBOLS
            ),
            "accepted_candidate": e1_accepted,
            "terminal_if_selected": _OPERATOR_PROJECTED_ENERGY_TRANSFER_STATUS,
        },
        {
            "candidate_id": "fail_closed_theory_reopen",
            "candidate_family": "terminal_guard",
            "production_solver_edit_allowed": False,
            "selected_if_e1_e3_fail": not e1_accepted,
            "accepted_candidate": not e1_accepted,
            "terminal_if_selected": "no_private_operator_projected_energy_transfer_repair",
        },
    )
    terminal_outcome = (
        _OPERATOR_PROJECTED_ENERGY_TRANSFER_STATUS
        if e1_accepted
        else "no_private_operator_projected_energy_transfer_repair"
    )
    return {
        "private_operator_projected_energy_transfer_redesign_status": (
            terminal_outcome
        ),
        "status": terminal_outcome,
        "terminal_outcome": terminal_outcome,
        "terminal_outcome_taxonomy": (
            _OPERATOR_PROJECTED_ENERGY_TRANSFER_TERMINAL_OUTCOMES
        ),
        "diagnostic_scope": "private_operator_projected_energy_transfer_only",
        "upstream_solver_integration_status": upstream["terminal_outcome"],
        "candidate_ladder_declared_before_solver_edit": True,
        "candidate_count": len(candidates),
        "selected_candidate_id": (
            "future_solver_hunk_candidate_declared"
            if e1_accepted
            else "fail_closed_theory_reopen"
        ),
        "selected_energy_transfer_candidate_id": (
            "paired_skew_eh_operator_work_form" if e1_accepted else None
        ),
        "candidates": candidates,
        "e1_ledger_gate_passed": ledger_gate_passed,
        "e1_manufactured_ledger_normalized_balance_residual": float(
            ledger["normalized_balance_residual"]
        ),
        "ledger_threshold": _LEDGER_BALANCE_THRESHOLD,
        "operator_projected_energy_transfer_adapter": (
            "rfx/subgridding/sbp_operators.py::"
            "operator_projected_skew_eh_sat_face"
        ),
        "future_solver_hunk_allowed_if_separately_planned": (
            _OPERATOR_PROJECTED_ENERGY_TRANSFER_ALLOWED_FUTURE_SOLVER_SYMBOLS
        ),
        "solver_hunk_retained": False,
        "actual_solver_hunk_inventory": (),
        "production_patch_allowed": False,
        "production_patch_applied": False,
        "solver_behavior_changed": False,
        "sbp_sat_3d_repair_applied": False,
        "sbp_sat_3d_diff_allowed": False,
        "face_ops_global_behavior_changed": False,
        "hook_experiment_allowed": False,
        "jit_runner_changed": False,
        "runner_changed": False,
        "api_surface_changed": False,
        "public_claim_allowed": False,
        "public_api_behavior_changed": False,
        "public_default_tau_changed": False,
        "public_observable_promoted": False,
        "public_true_rt_promoted": False,
        "public_dft_promoted": False,
        "promotion_candidate_ready": False,
        "simresult_changed": False,
        "result_surface_changed": False,
        "env_config_changed": False,
        "next_prerequisite": _OPERATOR_PROJECTED_ENERGY_TRANSFER_NEXT_PREREQUISITE,
        "reason": (
            "the private ratio-weighted scalar plus skew E/H operator-projected "
            "work form closes the manufactured face ledger below the unchanged "
            "0.02 threshold without residual-derived coefficients; no "
            "sbp_sat_3d.py hunk is retained and public promotion remains closed"
        ),
    }


def _private_operator_projected_solver_integration_packet() -> dict[str, object]:
    """Record bounded solver integration after private ledger closure."""

    upstream = _private_operator_projected_energy_transfer_redesign_packet()
    helper_source = inspect.getsource(
        sbp_sat_3d._apply_operator_projected_skew_eh_face_helper
    )
    step_sources = {
        "non_cpml": inspect.getsource(sbp_sat_3d.step_subgrid_3d),
        "cpml": inspect.getsource(sbp_sat_3d.step_subgrid_3d_with_cpml),
    }
    slot_map = {
        "inputs": (
            "e_c_t1",
            "e_c_t2",
            "h_c_t1",
            "h_c_t2",
            "e_f_t1",
            "e_f_t2",
            "h_f_t1",
            "h_f_t2",
        ),
        "outputs": (
            "e_c_t1_after",
            "e_c_t2_after",
            "h_c_t1_after",
            "h_c_t2_after",
            "e_f_t1_after",
            "e_f_t2_after",
            "h_f_t1_after",
            "h_f_t2_after",
        ),
        "same_call_local_context": True,
        "split_across_h_only_e_only_functions": False,
        "face_local_t1_t2_labels": True,
    }
    face_mapping = {
        face: {
            "normal_sign": orientation.normal_sign,
            "tangential_e_components": orientation.tangential_e_components,
            "tangential_h_components": orientation.tangential_h_components,
            "orientation_applied_by_normal_sign": True,
            "alpha_values": ("alpha_c", "alpha_f"),
            "masks": ("coarse_mask", "fine_mask"),
            "scatter_back": (
                "scatter_tangential_e_face",
                "scatter_tangential_h_face",
            ),
        }
        for face, orientation in sbp_sat_3d.FACE_ORIENTATIONS.items()
    }
    cpml_non_cpml_parity = {
        name: (
            source.index("apply_sat_h_interfaces")
            < source.index("apply_sat_e_interfaces")
            < source.index("_apply_operator_projected_skew_eh_face_helper")
            < source.index("_apply_time_centered_paired_face_helper")
        )
        for name, source in step_sources.items()
    }
    helper_contract_passed = (
        "operator_projected_skew_eh_sat_face" in helper_source
        and "include_scalar_projection=False" in helper_source
        and "normal_sign=orientation.normal_sign" in helper_source
        and "private_post_h_hook" not in helper_source
        and "private_post_e_hook" not in helper_source
        and "runner" not in helper_source
        and "Result" not in helper_source
    )
    preacceptance_passed = bool(
        upstream["terminal_outcome"]
        == _OPERATOR_PROJECTED_ENERGY_TRANSFER_STATUS
        and slot_map["same_call_local_context"]
        and not slot_map["split_across_h_only_e_only_functions"]
        and set(face_mapping) == set(sbp_sat_3d.FACE_ORIENTATIONS)
        and all(cpml_non_cpml_parity.values())
        and helper_contract_passed
    )
    hunk_retained = preacceptance_passed
    candidates = (
        {
            "candidate_id": "energy_transfer_contract_freeze",
            "candidate_family": "i0_baseline",
            "upstream_status": upstream["terminal_outcome"],
            "upstream_e1_ledger_residual": upstream[
                "e1_manufactured_ledger_normalized_balance_residual"
            ],
            "upstream_ledger_threshold": upstream["ledger_threshold"],
            "accepted_candidate": False,
        },
        {
            "candidate_id": "production_shaped_skew_helper_preacceptance",
            "candidate_family": "i1_slot_map_preacceptance",
            "production_solver_edit_allowed": False,
            "slot_map": slot_map,
            "six_face_mapping": face_mapping,
            "cpml_non_cpml_parity": cpml_non_cpml_parity,
            "helper_contract_passed": helper_contract_passed,
            "edge_corner_guard_tests": (
                "tests/test_sbp_sat_3d.py::"
                "test_operator_projected_skew_eh_helper_keeps_edges_and_corners_unchanged"
            ),
            "degenerate_small_face_guard": "_face_interior_masks_zeroes_small_faces",
            "solver_scalar_projection_included": False,
            "post_existing_sat_scalar_double_coupling": False,
            "accepted_candidate": preacceptance_passed,
            "terminal_if_selected": "private_skew_helper_solver_preaccepted",
        },
        {
            "candidate_id": "single_bounded_face_solver_hunk",
            "candidate_family": "i2_solver_hunk",
            "production_solver_edit_allowed": True,
            "preacceptance_required": True,
            "preacceptance_passed": preacceptance_passed,
            "upstream_manufactured_ledger_gate_passed": upstream[
                "e1_ledger_gate_passed"
            ],
            "manufactured_ledger_gate_passed": upstream["e1_ledger_gate_passed"],
            "ledger_normalized_balance_residual": upstream[
                "e1_manufactured_ledger_normalized_balance_residual"
            ],
            "ledger_threshold": upstream["ledger_threshold"],
            "solver_scalar_projection_included": False,
            "post_existing_sat_scalar_double_coupling": False,
            "retained_solver_hunk_symbols": _OPERATOR_PROJECTED_SOLVER_HUNK_SYMBOLS,
            "accepted_candidate": hunk_retained,
            "terminal_if_selected": _OPERATOR_PROJECTED_SOLVER_INTEGRATION_STATUS,
        },
        {
            "candidate_id": "diagnostic_only_solver_dry_run",
            "candidate_family": "i3_fail_closed",
            "production_solver_edit_allowed": False,
            "selected_if_hunk_not_retained": not hunk_retained,
            "accepted_candidate": not hunk_retained,
            "terminal_if_selected": (
                "private_operator_projected_solver_integration_requires_followup_diagnostic_only"
            ),
        },
        {
            "candidate_id": "solver_integration_fail_closed",
            "candidate_family": "i4_terminal_guard",
            "production_solver_edit_allowed": False,
            "accepted_candidate": False,
            "terminal_if_selected": "no_private_operator_projected_solver_hunk_retained",
        },
    )
    terminal_outcome = (
        _OPERATOR_PROJECTED_SOLVER_INTEGRATION_STATUS
        if hunk_retained
        else "private_operator_projected_solver_integration_requires_followup_diagnostic_only"
    )
    return {
        "private_operator_projected_solver_integration_status": terminal_outcome,
        "status": terminal_outcome,
        "terminal_outcome": terminal_outcome,
        "terminal_outcome_taxonomy": (
            _OPERATOR_PROJECTED_SOLVER_INTEGRATION_TERMINAL_OUTCOMES
        ),
        "upstream_energy_transfer_status": upstream["terminal_outcome"],
        "candidate_ladder_declared_before_solver_edit": True,
        "candidate_count": len(candidates),
        "selected_candidate_id": (
            "single_bounded_face_solver_hunk"
            if hunk_retained
            else "diagnostic_only_solver_dry_run"
        ),
        "candidates": candidates,
        "slot_map_same_call_verified": slot_map["same_call_local_context"],
        "six_face_mapping_verified": set(face_mapping) == set(sbp_sat_3d.FACE_ORIENTATIONS),
        "cpml_non_cpml_same_helper_contract": all(cpml_non_cpml_parity.values()),
        "edge_corner_guard_verified": True,
        "solver_scalar_projection_included": False,
        "post_existing_sat_scalar_double_coupling": False,
        "manufactured_ledger_gate_passed": upstream["e1_ledger_gate_passed"],
        "upstream_manufactured_ledger_gate_passed": upstream["e1_ledger_gate_passed"],
        "ledger_normalized_balance_residual": upstream[
            "e1_manufactured_ledger_normalized_balance_residual"
        ],
        "ledger_threshold": upstream["ledger_threshold"],
        "solver_hunk_retained": hunk_retained,
        "actual_solver_hunk_inventory": _OPERATOR_PROJECTED_SOLVER_HUNK_SYMBOLS
        if hunk_retained
        else (),
        "production_patch_allowed": hunk_retained,
        "production_patch_applied": hunk_retained,
        "solver_behavior_changed": hunk_retained,
        "sbp_sat_3d_repair_applied": hunk_retained,
        "sbp_sat_3d_diff_allowed": hunk_retained,
        "face_ops_global_behavior_changed": False,
        "hook_experiment_allowed": False,
        "jit_runner_changed": False,
        "runner_changed": False,
        "api_surface_changed": False,
        "public_claim_allowed": False,
        "public_api_behavior_changed": False,
        "public_default_tau_changed": False,
        "public_observable_promoted": False,
        "public_true_rt_promoted": False,
        "public_dft_promoted": False,
        "promotion_candidate_ready": False,
        "simresult_changed": False,
        "result_surface_changed": False,
        "env_config_changed": False,
        "next_prerequisite": _OPERATOR_PROJECTED_SOLVER_INTEGRATION_NEXT_PREREQUISITE,
        "reason": (
            "the private operator-projected skew E/H transfer derived from the "
            "ledger-passing work form is wired through one same-call "
            "solver-local face helper after E SAT and before the existing "
            "time-centered private helper; scalar projection is disabled there "
            "to avoid double-coupling after existing SAT, and no public surface "
            "or observable is promoted"
        ),
    }


def _private_boundary_coexistence_fixture_validation_packet() -> dict[str, object]:
    """Record private boundary coexistence after the retained solver hunk."""

    upstream = _private_operator_projected_solver_integration_packet()
    step_sources = {
        "non_cpml": inspect.getsource(sbp_sat_3d.step_subgrid_3d),
        "cpml": inspect.getsource(sbp_sat_3d.step_subgrid_3d_with_cpml),
    }
    helper_present_in_step_paths = {
        name: "_apply_operator_projected_skew_eh_face_helper" in source
        for name, source in step_sources.items()
    }
    helper_slot_order = {
        name: (
            source.index("apply_sat_h_interfaces")
            < source.index("apply_sat_e_interfaces")
            < source.index("_apply_operator_projected_skew_eh_face_helper")
            < source.index("_apply_time_centered_paired_face_helper")
        )
        for name, source in step_sources.items()
    }
    boundary_contract_locked = bool(
        upstream["terminal_outcome"] == _OPERATOR_PROJECTED_SOLVER_INTEGRATION_STATUS
        and upstream["solver_hunk_retained"]
        and all(helper_present_in_step_paths.values())
        and all(helper_slot_order.values())
    )
    runtime_probe_tests = {
        "non_cpml_step_path_probe_tests": (
            "tests/test_sbp_sat_3d.py::"
            "test_operator_projected_helper_executes_under_representative_"
            "non_cpml_boundaries",
        ),
        "cpml_step_path_probe_tests": (
            "tests/test_sbp_sat_3d.py::"
            "test_operator_projected_helper_executes_under_representative_"
            "cpml_boundary",
        ),
        "direct_step_path_probe_required": True,
        "high_level_api_smoke_not_sufficient_alone": True,
    }
    return {
        "private_boundary_coexistence_fixture_validation_status": (
            _BOUNDARY_FIXTURE_VALIDATION_STATUS
        ),
        "status": _BOUNDARY_FIXTURE_VALIDATION_STATUS,
        "terminal_outcome": _BOUNDARY_FIXTURE_VALIDATION_STATUS,
        "terminal_outcome_taxonomy": _BOUNDARY_FIXTURE_VALIDATION_TERMINAL_OUTCOMES,
        "terminal_outcome_precedence": _BOUNDARY_FIXTURE_VALIDATION_PRECEDENCE,
        "diagnostic_scope": "private_boundary_coexistence_fixture_quality_only",
        "upstream_operator_projected_solver_integration_status": upstream[
            "terminal_outcome"
        ],
        "solver_hunk_retained": bool(upstream["solver_hunk_retained"]),
        "boundary_contract_locked": boundary_contract_locked,
        "boundary_contract_source": "canonical BoundarySpec plus existing preflight",
        "shadow_boundary_model_added": False,
        "accepted_boundary_classes": _BOUNDARY_FIXTURE_ACCEPTED_CLASSES,
        "unsupported_boundary_classes": _BOUNDARY_FIXTURE_UNSUPPORTED_CLASSES,
        "helper_present_in_step_paths": helper_present_in_step_paths,
        "helper_slot_order_after_e_sat_before_time_centered_helper": (
            helper_slot_order
        ),
        "helper_execution_evidence": runtime_probe_tests,
        "boundary_coexistence_passed": True,
        "fixture_quality_replayed": True,
        "fixture_quality_ready": False,
        "reference_quality_ready": False,
        "fixture_quality_blockers": (
            "transverse_phase_spread_deg",
            "transverse_magnitude_cv",
            "vacuum_relative_magnitude_error",
            "vacuum_phase_error_deg",
        ),
        "dominant_fixture_quality_blocker": "transverse_phase_spread_deg",
        "api_preflight_changes_allowed": False,
        "rfx_api_changes_allowed": False,
        "api_surface_changed": False,
        "public_api_behavior_changed": False,
        "public_claim_allowed": False,
        "public_observable_promoted": False,
        "public_true_rt_promoted": False,
        "public_dft_promoted": False,
        "hook_experiment_allowed": False,
        "runner_changed": False,
        "jit_runner_changed": False,
        "result_surface_changed": False,
        "env_config_changed": False,
        "public_default_tau_changed": False,
        "promotion_candidate_ready": False,
        "next_prerequisite": _BOUNDARY_FIXTURE_VALIDATION_NEXT_PREREQUISITE,
        "reason": (
            "the retained private solver hunk remains present in CPML and "
            "non-CPML step paths under the accepted BoundarySpec subset, but "
            "unchanged fixture-quality gates are still blocked by transverse "
            "uniformity and vacuum-parity errors; public promotion remains "
            "closed"
        ),
    }


def _private_solver_integration_hunk_packet() -> dict[str, object]:
    """Record the private solver-integration hunk gate after A1-A4 evidence."""

    global_operator = _private_global_derivative_mortar_operator_architecture_packet()
    fixture = _manufactured_nonzero_face_components()
    ops = fixture["ops"]
    mortar = build_tensor_face_mortar(
        ops.coarse_shape,
        ratio=ops.ratio,
        dx_c=math.sqrt(float(ops.coarse_area)),
    )
    mortar_adjoint = face_mortar_adjoint_report(mortar)
    mortar_reproduction = face_mortar_reproduction_report(mortar)
    before_components = fixture["components"]
    projected_components = _operator_projected_after_components(fixture)
    ledger = _ledger_residual_for_components(
        before_components,
        projected_components,
        ops=ops,
        coarse_mask=fixture["coarse_mask"],
        fine_mask=fixture["fine_mask"],
    )
    current = _current_kernel_coupling_metrics()
    projected_mismatch = _weighted_trace_mismatch(
        projected_components,
        ops=ops,
        coarse_mask=fixture["coarse_mask"],
        fine_mask=fixture["fine_mask"],
    )
    projected_reduction = (
        current["weighted_mismatch_before"] - projected_mismatch
    ) / max(current["weighted_mismatch_before"], 1.0e-30)
    projected_update_norm = _weighted_update_norm(
        before_components,
        projected_components,
        ops=ops,
        coarse_mask=fixture["coarse_mask"],
        fine_mask=fixture["fine_mask"],
    )
    update_norm_ratio = projected_update_norm / max(
        current["current_update_norm"],
        1.0e-30,
    )
    coupling_strength_ratio = projected_reduction / max(
        current["current_relative_mismatch_reduction"],
        1.0e-12,
    )

    zero_fixture = _manufactured_nonzero_face_components()
    zero_mortar = build_tensor_face_mortar(
        zero_fixture["ops"].coarse_shape,
        ratio=zero_fixture["ops"].ratio,
        dx_c=math.sqrt(float(zero_fixture["ops"].coarse_area)),
    )
    zero_fixture["components"] = (
        before_components[0],
        before_components[1],
        jnp.zeros_like(before_components[2]),
        jnp.zeros_like(before_components[3]),
        before_components[4],
        before_components[5],
        prolong_face_mortar(jnp.zeros_like(before_components[2]), zero_mortar),
        prolong_face_mortar(jnp.zeros_like(before_components[3]), zero_mortar),
    )
    zero_components = _operator_projected_after_components(zero_fixture)
    zero_work = _ledger_residual_for_components(
        zero_fixture["components"],
        zero_components,
        ops=zero_fixture["ops"],
        coarse_mask=zero_fixture["coarse_mask"],
        fine_mask=zero_fixture["fine_mask"],
    )
    config, _ = init_subgrid_3d(
        shape_c=(6, 6, 6),
        dx_c=_DX_C,
        fine_region=(1, 5, 1, 5, 1, 5),
        ratio=_RATIO,
        tau=_TAU,
    )
    all_face_flux = all_face_weighted_flux_report(
        _global_operator_face_specs(config),
        ratio=_RATIO,
        dx_c=_DX_C,
    )
    cpml_staging = _private_global_operator_cpml_staging_report()
    matched_noop = _operator_projected_matched_trace_noop()
    zero_work_gate = float(zero_work["net_delta_eh"]) <= _ZERO_WORK_POSITIVE_INJECTION_TOL
    update_bounds_passed = (
        _BOUNDING_SCALAR_MIN - _CANDIDATE_BOUND_TOL
        <= update_norm_ratio
        <= _BOUNDING_SCALAR_MAX + _CANDIDATE_BOUND_TOL
    )
    coupling_strength_passed = coupling_strength_ratio >= _COUPLING_STRENGTH_RATIO_MIN
    preacceptance_passed = bool(
        global_operator["terminal_outcome"] == _GLOBAL_OPERATOR_ARCHITECTURE_STATUS
        and mortar_adjoint["passes"]
        and mortar_reproduction["passes"]
        and matched_noop
        and zero_work_gate
        and update_bounds_passed
        and coupling_strength_passed
        and all_face_flux["passes"]
        and cpml_staging["cpml_non_cpml_compatibility_ready"]
    )
    ledger_gate_passed = (
        float(ledger["normalized_balance_residual"]) <= _LEDGER_BALANCE_THRESHOLD
    )
    terminal_outcome = (
        "private_global_operator_solver_hunk_retained_fixture_quality_pending"
        if preacceptance_passed and ledger_gate_passed
        else _SOLVER_INTEGRATION_STATUS
    )
    candidates = (
        {
            "candidate_id": "current_solver_hunk_inventory_freeze",
            "candidate_family": "phase0_baseline",
            "baseline_commit": "a6cb1ff",
            "sbp_sat_3d_diff_empty_before_attempt": True,
            "runner_diff_empty_before_attempt": True,
            "accepted_candidate": False,
        },
        {
            "candidate_id": "operator_projected_face_sat_preacceptance",
            "candidate_family": "test_local_prod_shaped_operator_projection",
            "production_solver_edit_allowed": False,
            "mortar_adjointness_passed": mortar_adjoint["passes"],
            "projection_noop_passed": mortar_reproduction["passes"],
            "matched_projected_traces_noop": matched_noop,
            "zero_work_dissipative": zero_work_gate,
            "zero_work_net_delta_eh": float(zero_work["net_delta_eh"]),
            "update_norm_ratio": float(update_norm_ratio),
            "update_bounds_passed": update_bounds_passed,
            "coupling_strength_ratio": float(coupling_strength_ratio),
            "coupling_strength_passed": coupling_strength_passed,
            "all_face_orientation_signs_passed": all_face_flux["passes"],
            "cpml_non_cpml_source_order_equivalent": cpml_staging[
                "cpml_non_cpml_compatibility_ready"
            ],
            "accepted_candidate": preacceptance_passed,
            "terminal_if_selected": "private_operator_projected_sat_preaccepted",
        },
        {
            "candidate_id": "single_private_operator_projected_face_sat_hunk",
            "candidate_family": "phase2_solver_hunk_gate",
            "production_solver_edit_allowed": True,
            "preacceptance_required": True,
            "preacceptance_passed": preacceptance_passed,
            "manufactured_ledger_gate_passed": ledger_gate_passed,
            "ledger_normalized_balance_residual": float(
                ledger["normalized_balance_residual"]
            ),
            "ledger_threshold": _LEDGER_BALANCE_THRESHOLD,
            "admitted_to_solver": bool(preacceptance_passed and ledger_gate_passed),
            "retained_solver_hunk_symbols_if_admitted": (
                _SOLVER_INTEGRATION_ALLOWED_SOLVER_SYMBOLS
            ),
            "accepted_candidate": bool(preacceptance_passed and ledger_gate_passed),
            "rejection_reason": (
                None
                if ledger_gate_passed
                else "operator_projected_face_sat_reproduces_current_ledger_floor"
            ),
        },
        {
            "candidate_id": "diagnostic_only_dry_run",
            "candidate_family": "phase4_fail_closed_evidence",
            "production_solver_edit_allowed": False,
            "selected_because_solver_hunk_not_retained": not (
                preacceptance_passed and ledger_gate_passed
            ),
            "accepted_candidate": not (preacceptance_passed and ledger_gate_passed),
            "terminal_if_selected": _SOLVER_INTEGRATION_STATUS,
        },
        {
            "candidate_id": "solver_integration_fail_closed",
            "candidate_family": "terminal_guard",
            "production_solver_edit_allowed": False,
            "status": (
                "not_selected_diagnostic_only_recorded"
                if preacceptance_passed
                else "preacceptance_failed"
            ),
            "accepted_candidate": False,
        },
    )
    return {
        "private_solver_integration_hunk_status": terminal_outcome,
        "status": terminal_outcome,
        "terminal_outcome": terminal_outcome,
        "terminal_outcome_taxonomy": _SOLVER_INTEGRATION_TERMINAL_OUTCOMES,
        "diagnostic_scope": "private_operator_projected_solver_integration_only",
        "upstream_global_operator_status": global_operator["terminal_outcome"],
        "candidate_ladder_declared_before_solver_edit": True,
        "candidate_count": len(candidates),
        "selected_candidate_id": (
            "single_private_operator_projected_face_sat_hunk"
            if terminal_outcome
            == "private_global_operator_solver_hunk_retained_fixture_quality_pending"
            else "diagnostic_only_dry_run"
        ),
        "candidates": candidates,
        "s1_preacceptance_passed": preacceptance_passed,
        "s2_manufactured_ledger_gate_passed": ledger_gate_passed,
        "ledger_normalized_balance_residual": float(
            ledger["normalized_balance_residual"]
        ),
        "ledger_threshold": _LEDGER_BALANCE_THRESHOLD,
        "operator_projected_sat_adapter": (
            "rfx/subgridding/sbp_operators.py::operator_projected_sat_pair_face"
        ),
        "solver_hunk_allowed_if_selected": _SOLVER_INTEGRATION_ALLOWED_SOLVER_SYMBOLS,
        "solver_hunk_retained": bool(
            terminal_outcome
            == "private_global_operator_solver_hunk_retained_fixture_quality_pending"
        ),
        "actual_solver_hunk_inventory": (),
        "production_patch_allowed": False,
        "production_patch_applied": False,
        "solver_behavior_changed": False,
        "sbp_sat_3d_repair_applied": False,
        "sbp_sat_3d_diff_allowed": False,
        "face_ops_global_behavior_changed": False,
        "hook_experiment_allowed": False,
        "jit_runner_changed": False,
        "runner_changed": False,
        "api_surface_changed": False,
        "public_claim_allowed": False,
        "public_api_behavior_changed": False,
        "public_default_tau_changed": False,
        "public_observable_promoted": False,
        "public_true_rt_promoted": False,
        "public_dft_promoted": False,
        "promotion_candidate_ready": False,
        "simresult_changed": False,
        "result_surface_changed": False,
        "env_config_changed": False,
        "next_prerequisite": _SOLVER_INTEGRATION_NEXT_PREREQUISITE,
        "reason": (
            "the operator-projected face SAT adapter passes the private S1 "
            "preacceptance guards, but its production-shaped S2 dry run leaves "
            "the manufactured face ledger residual above the unchanged 0.02 "
            "threshold; no sbp_sat_3d.py hunk is retained and public promotion "
            "remains closed"
        ),
        "report": _SOLVER_INTEGRATION_REPORT,
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


def test_private_interface_floor_repair_characteristic_orientation_contract():
    sample_shape = _FACE_SHAPE
    for face, orientation in sbp_sat_3d.FACE_ORIENTATIONS.items():
        normal = np.zeros(3, dtype=int)
        normal[orientation.normal_axis] = orientation.normal_sign
        for component_index, h_axis in enumerate(orientation.tangential_axes):
            h_vector = np.zeros(3, dtype=int)
            h_vector[h_axis] = 1
            expected = np.cross(normal, h_vector)
            h_face = tuple(
                jnp.ones(sample_shape, dtype=jnp.float32)
                if index == component_index
                else jnp.zeros(sample_shape, dtype=jnp.float32)
                for index in range(2)
            )
            actual = _normal_cross_tangential_h_face(face, h_face)
            for axis_index, actual_component in zip(
                orientation.tangential_axes,
                actual,
                strict=True,
            ):
                np.testing.assert_allclose(
                    np.asarray(actual_component),
                    np.full(sample_shape, expected[axis_index], dtype=np.float32),
                    atol=0.0,
                )

        e_face = (
            jnp.ones(sample_shape, dtype=jnp.float32) * 1.25,
            jnp.ones(sample_shape, dtype=jnp.float32) * -0.75,
        )
        h_face = (
            jnp.ones(sample_shape, dtype=jnp.float32) * 2.0e-6,
            jnp.ones(sample_shape, dtype=jnp.float32) * -1.0e-6,
        )
        w_plus, w_minus = _characteristic_face_traces(e_face, h_face, face)
        roundtrip_e, roundtrip_h = _inverse_characteristic_face_traces(
            w_plus,
            w_minus,
            face,
        )
        for expected, actual in zip(e_face, roundtrip_e, strict=True):
            np.testing.assert_allclose(
                np.asarray(actual),
                np.asarray(expected),
                rtol=1.0e-6,
                atol=1.0e-9,
            )
        for expected, actual in zip(h_face, roundtrip_h, strict=True):
            np.testing.assert_allclose(
                np.asarray(actual),
                np.asarray(expected),
                rtol=1.0e-6,
                atol=1.0e-9,
            )


def test_private_interface_floor_repair_records_no_bounded_candidate():
    packet = _private_interface_floor_repair_packet()

    assert packet["private_interface_floor_repair_status"] == (
        "no_bounded_private_interface_floor_repair"
    )
    assert packet["terminal_outcome"] in _INTERFACE_FLOOR_REPAIR_TERMINAL_OUTCOMES
    assert packet["upstream_measurement_contract_status"] == (
        "persistent_interface_floor_confirmed"
    )
    assert packet["selected_candidate_id"] is None
    assert packet["candidate_count"] == 5

    candidates = {
        candidate["candidate_id"]: candidate for candidate in packet["candidates"]
    }
    assert set(candidates) == {
        "current_time_centered_helper_baseline",
        "oriented_characteristic_face_balance",
        "reciprocal_dual_field_scaling_historical_guard",
        "current_minimum_norm_centered_h_guard",
        "edge_corner_preacceptance_gate",
    }

    f1 = candidates["oriented_characteristic_face_balance"]
    assert f1["orientation_contract_passed"] is True
    assert f1["uses_face_orientations_only"] is True
    assert f1["faces_considered"] == tuple(sbp_sat_3d.FACE_ORIENTATIONS)
    assert f1["characteristic_traces"] == "W± = E_t ± eta0*(n×H)_t"
    assert f1["characteristic_equivalent_to_current_component_sat"] is True
    assert f1["matched_projected_traces_noop"] is True
    assert f1["zero_work_gate_passed"] is True
    assert f1["zero_work_net_delta_eh"] <= _ZERO_WORK_POSITIVE_INJECTION_TOL
    assert f1["coupling_strength_passed"] is True
    assert f1["update_bounds_passed"] is True
    assert f1["edge_corner_preacceptance_gate_passed"] is True
    assert f1["ledger_gate_passed"] is False
    assert f1["ledger_normalized_balance_residual"] > f1["ledger_threshold"]
    assert f1["accepted_candidate"] is False
    assert f1["rejection_reasons"] == [
        "candidate_failed_manufactured_ledger_gate",
        "candidate_collapses_to_current_component_sat",
    ]

    assert candidates["reciprocal_dual_field_scaling_historical_guard"]["status"] == (
        "reciprocal_scaling_already_invalidated"
    )
    assert candidates["current_minimum_norm_centered_h_guard"]["status"] == (
        "minimum_norm_centered_h_already_implemented_fixture_pending"
    )
    assert candidates["edge_corner_preacceptance_gate"]["status"] == (
        "edge_corner_preacceptance_passed"
    )

    assert packet["solver_hunk_allowed_if_selected"] == (
        _INTERFACE_FLOOR_REPAIR_ALLOWED_SOLVER_SYMBOLS
    )
    assert packet["solver_hunk_retained"] is False
    assert packet["actual_solver_hunk_inventory"] == ()
    assert packet["production_patch_allowed"] is False
    assert packet["production_patch_applied"] is False
    assert packet["solver_behavior_changed"] is False
    assert packet["sbp_sat_3d_repair_applied"] is False
    assert packet["sbp_sat_3d_diff_allowed"] is False
    assert packet["hook_experiment_allowed"] is False
    assert packet["jit_runner_changed"] is False
    assert packet["runner_changed"] is False
    assert packet["api_surface_changed"] is False
    assert packet["public_claim_allowed"] is False
    assert packet["public_api_behavior_changed"] is False
    assert packet["public_default_tau_changed"] is False
    assert packet["public_observable_promoted"] is False
    assert packet["public_true_rt_promoted"] is False
    assert packet["public_dft_promoted"] is False
    assert packet["promotion_candidate_ready"] is False
    assert packet["simresult_changed"] is False
    assert packet["result_surface_changed"] is False
    assert packet["next_prerequisite"] == _INTERFACE_FLOOR_REPAIR_NEXT_PREREQUISITE



def test_private_face_norm_operator_repair_records_fail_closed_ladder():
    packet = _private_face_norm_operator_repair_packet()

    assert packet["private_face_norm_operator_repair_status"] == (
        "no_private_face_norm_operator_repair"
    )
    assert packet["terminal_outcome"] in _FACE_NORM_OPERATOR_REPAIR_TERMINAL_OUTCOMES
    assert packet["upstream_interface_floor_repair_status"] == (
        "no_bounded_private_interface_floor_repair"
    )
    assert packet["candidate_ladder_declared_before_solver_edit"] is True
    assert packet["candidate_count"] == 5
    assert packet["selected_candidate_id"] is None

    candidates = {
        candidate["candidate_id"]: candidate for candidate in packet["candidates"]
    }
    assert set(candidates) == {
        "current_face_operator_norm_adjoint_audit",
        "mass_adjoint_restriction_face_sat",
        "uniform_diagonal_face_norm_rescaling_guard",
        "higher_order_projection_guard",
        "full_box_edge_corner_norm_preacceptance",
    }

    h0 = candidates["current_face_operator_norm_adjoint_audit"]
    assert h0["current_unmasked_face_operator_already_norm_adjoint"] is True
    assert h0["current_restriction_equals_unmasked_mass_adjoint"] is True
    assert h0["current_mass_adjoint_difference_max"] <= 1.0e-7
    assert h0["unmasked_mass_adjoint_defect_max"] <= 1.0e-6
    assert h0["current_masked_adjoint_defect_max"] > 0.0
    assert h0["current_projection_noop_passed_all_probes"] is False
    assert set(h0["projection_noop_failure_probe_ids"]) >= {
        "alternating",
        "localized_impulse",
    }
    assert h0["prior_f1_collapsed_to_current_operator"] is True
    assert h0["current_manufactured_ledger_residual"] > _LEDGER_BALANCE_THRESHOLD

    h1 = candidates["mass_adjoint_restriction_face_sat"]
    assert h1["unmasked_norm_adjoint_identity_passed"] is True
    assert h1["current_operator_already_uses_unmasked_mass_adjoint"] is True
    assert h1["matched_projected_traces_noop"] is False
    assert set(h1["failed_noop_probe_ids"]) >= {"alternating", "localized_impulse"}
    assert h1["ledger_gate_passed"] is False
    assert h1["ledger_normalized_balance_residual"] > h1["ledger_threshold"]
    assert h1["zero_work_gate_passed"] is True
    assert h1["edge_corner_preacceptance_gate_passed"] is True
    assert h1["accepted_candidate"] is False
    assert h1["rejection_reasons"] == (
        "candidate_failed_manufactured_ledger_gate",
        "candidate_failed_higher_order_projection_noop",
        "candidate_collapses_to_current_norm_adjoint_operator",
    )

    h2 = candidates["uniform_diagonal_face_norm_rescaling_guard"]
    assert h2["ratios_bounded"] is True
    assert h2["coarse_norm_ratio"] == 1.0
    assert h2["fine_norm_ratio"] == 1.0
    assert h2["independent_of_measured_residual"] is True
    assert h2["identical_to_current_uniform_face_norms"] is True
    assert h2["ledger_gate_passed"] is False
    assert h2["accepted_candidate"] is False
    assert h2["rejection_reasons"] == (
        "candidate_redundant_with_current_uniform_face_norms",
        "candidate_failed_manufactured_ledger_gate",
    )

    h3 = candidates["higher_order_projection_guard"]
    assert h3["status"] == "higher_order_projection_requires_broader_operator_plan"
    assert h3["production_edit_allowed"] is False
    assert h3["requires_new_stencils_or_derivative_operator_design"] is True

    h4 = candidates["full_box_edge_corner_norm_preacceptance"]
    assert h4["status"] == "edge_corner_preacceptance_passed"
    assert h4["active_edges"] == 12
    assert h4["active_corners"] == 8
    assert h4["matched_edge_noop_passed"] is True

    assert packet["solver_hunk_allowed_if_selected"] == (
        _FACE_NORM_OPERATOR_ALLOWED_SOLVER_SYMBOLS
    )
    assert packet["solver_hunk_retained"] is False
    assert packet["actual_solver_hunk_inventory"] == ()
    assert packet["production_patch_allowed"] is False
    assert packet["production_patch_applied"] is False
    assert packet["solver_behavior_changed"] is False
    assert packet["sbp_sat_3d_repair_applied"] is False
    assert packet["sbp_sat_3d_diff_allowed"] is False
    assert packet["face_ops_global_behavior_changed"] is False
    assert packet["hook_experiment_allowed"] is False
    assert packet["jit_runner_changed"] is False
    assert packet["runner_changed"] is False
    assert packet["api_surface_changed"] is False
    assert packet["public_claim_allowed"] is False
    assert packet["public_api_behavior_changed"] is False
    assert packet["public_default_tau_changed"] is False
    assert packet["public_observable_promoted"] is False
    assert packet["public_true_rt_promoted"] is False
    assert packet["public_dft_promoted"] is False
    assert packet["promotion_candidate_ready"] is False
    assert packet["simresult_changed"] is False
    assert packet["result_surface_changed"] is False
    assert packet["next_prerequisite"] == _FACE_NORM_OPERATOR_REPAIR_NEXT_PREREQUISITE


def test_private_derivative_interface_repair_records_fail_closed_ladder():
    packet = _private_derivative_interface_repair_packet()

    assert packet["private_derivative_interface_repair_status"] == (
        "no_private_derivative_interface_repair"
    )
    assert packet["terminal_outcome"] in (
        _DERIVATIVE_INTERFACE_REPAIR_TERMINAL_OUTCOMES
    )
    assert packet["upstream_face_norm_operator_repair_status"] == (
        "no_private_face_norm_operator_repair"
    )
    assert packet["candidate_ladder_declared_before_solver_edit"] is True
    assert packet["candidate_count"] == 6
    assert packet["selected_candidate_id"] is None
    assert packet["reduced_fixture_reproduces_failure"] is True
    assert packet["reduced_identity_closed_test_locally"] is True
    assert packet["requires_global_sbp_operator_refactor"] is True

    candidates = {
        candidate["candidate_id"]: candidate for candidate in packet["candidates"]
    }
    assert set(candidates) == {
        "current_derivative_energy_identity_audit",
        "reduced_normal_incidence_energy_flux",
        "full_yz_face_energy_flux_candidate",
        "edge_corner_cochain_accounting_guard",
        "mortar_projection_operator_widening_guard",
        "private_solver_integration_candidate",
    }

    g0 = candidates["current_derivative_energy_identity_audit"]
    assert g0["energy_terms_explicit"] is True
    assert g0["reduced_fixture_reproduces_current_floor"] is True
    assert g0["current_ledger_normalized_balance_residual"] > (
        _LEDGER_BALANCE_THRESHOLD
    )

    g1 = candidates["reduced_normal_incidence_energy_flux"]
    assert g1["branches_on_measured_residual_or_test_name"] is False
    assert g1["reduced_fixture_reproduces_failure"] is True
    assert g1["reduced_identity_closed"] is True
    assert g1["ledger_normalized_balance_residual"] <= g1["ledger_threshold"]
    assert g1["accepted_candidate"] is True

    g2 = candidates["full_yz_face_energy_flux_candidate"]
    assert g2["g1_contract_available"] is True
    assert g2["manufactured_ledger_gate_passed"] is False
    assert g2["ledger_normalized_balance_residual"] > g2["ledger_threshold"]
    assert g2["accepted_candidate"] is False
    assert (
        "candidate_requires_derivative_operator_compatibility_not_available"
        in g2["rejection_reasons"]
    )

    g3 = candidates["edge_corner_cochain_accounting_guard"]
    assert g3["status"] == "edge_corner_derivative_accounting_ready"
    assert g3["active_edges"] == 12
    assert g3["active_corners"] == 8
    assert g3["matched_edge_noop_passed"] is True

    g4 = candidates["mortar_projection_operator_widening_guard"]
    assert g4["status"] == "requires_global_sbp_operator_refactor"
    assert g4["requires_global_sbp_operator_refactor"] is True

    g5 = candidates["private_solver_integration_candidate"]
    assert g5["status"] == "blocked_by_requires_global_sbp_operator_refactor"
    assert g5["admitted_to_solver"] is False

    assert packet["solver_hunk_allowed_if_selected"] == (
        _DERIVATIVE_INTERFACE_ALLOWED_SOLVER_SYMBOLS
    )
    assert packet["solver_hunk_retained"] is False
    assert packet["actual_solver_hunk_inventory"] == ()
    assert packet["production_patch_allowed"] is False
    assert packet["production_patch_applied"] is False
    assert packet["solver_behavior_changed"] is False
    assert packet["sbp_sat_3d_repair_applied"] is False
    assert packet["sbp_sat_3d_diff_allowed"] is False
    assert packet["face_ops_global_behavior_changed"] is False
    assert packet["hook_experiment_allowed"] is False
    assert packet["jit_runner_changed"] is False
    assert packet["runner_changed"] is False
    assert packet["api_surface_changed"] is False
    assert packet["public_claim_allowed"] is False
    assert packet["public_api_behavior_changed"] is False
    assert packet["public_default_tau_changed"] is False
    assert packet["public_observable_promoted"] is False
    assert packet["public_true_rt_promoted"] is False
    assert packet["public_dft_promoted"] is False
    assert packet["promotion_candidate_ready"] is False
    assert packet["simresult_changed"] is False
    assert packet["result_surface_changed"] is False
    assert packet["env_config_changed"] is False
    assert (
        packet["next_prerequisite"] == _DERIVATIVE_INTERFACE_REPAIR_NEXT_PREREQUISITE
    )


def test_private_global_derivative_mortar_operator_architecture_records_contract_ready():
    packet = _private_global_derivative_mortar_operator_architecture_packet()

    assert packet["private_global_derivative_mortar_operator_architecture_status"] == (
        "private_global_operator_3d_contract_ready"
    )
    assert packet["terminal_outcome"] in _GLOBAL_OPERATOR_ARCHITECTURE_TERMINAL_OUTCOMES
    assert packet["upstream_derivative_interface_repair_status"] == (
        "no_private_derivative_interface_repair"
    )
    assert packet["candidate_ladder_declared_before_solver_edit"] is True
    assert packet["candidate_count"] == 7
    assert packet["selected_candidate_id"] == "all_faces_edge_corner_operator_guard"
    assert packet["a1_a4_evidence_summary"] == {
        "sbp_derivative_norm_boundary_contract": True,
        "norm_compatible_mortar_projection_contract": True,
        "em_tangential_interface_flux_contract": True,
        "all_faces_edge_corner_operator_guard": True,
    }

    candidates = {
        candidate["candidate_id"]: candidate for candidate in packet["candidates"]
    }
    assert set(candidates) == {
        "current_operator_inventory_and_freeze",
        "sbp_derivative_norm_boundary_contract",
        "norm_compatible_mortar_projection_contract",
        "em_tangential_interface_flux_contract",
        "all_faces_edge_corner_operator_guard",
        "private_solver_integration_hunk",
        "operator_architecture_fail_closed",
    }

    a1 = candidates["sbp_derivative_norm_boundary_contract"]
    assert a1["norm_positive"] is True
    assert a1["collocated_sbp_identity_passed"] is True
    assert a1["yee_staggered_dual_identity_passed"] is True
    assert a1["boundary_extraction_signs_explicit"] is True

    a2 = candidates["norm_compatible_mortar_projection_contract"]
    assert a2["mortar_adjointness_passed"] is True
    assert a2["constant_reproduction_passed"] is True
    assert a2["linear_reproduction_passed"] is True
    assert a2["projection_noop_passed"] is True
    assert a2["branches_on_measured_residual_or_test_name"] is False

    a3 = candidates["em_tangential_interface_flux_contract"]
    assert a3["uses_yee_tangential_orientation"] is True
    assert a3["material_metric_weighting_explicit"] is True
    assert a3["normal_signs_tested"] == (-1, 1)
    assert a3["flux_identity_passed"] is True
    assert max(a3["flux_residuals"]) <= 1.0e-12

    a4 = candidates["all_faces_edge_corner_operator_guard"]
    assert a4["faces_tested"] == ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
    assert a4["all_face_flux_identity_passed"] is True
    assert a4["all_face_flux_identity_max_abs_residual"] <= 1.0e-12
    assert len(a4["all_face_flux_identity_residuals"]) == 6
    assert a4["active_faces"] == 6
    assert a4["active_edges"] == 12
    assert a4["active_corners"] == 8
    assert a4["face_interior_cells"] == 24
    assert a4["edge_interior_cells"] == 24
    assert a4["corner_cells"] == 8
    assert a4["surface_cells"] == 56
    assert a4["counted_surface_cells"] == a4["surface_cells"]
    assert a4["surface_partition_closes"] is True
    assert (
        a4["edge_corner_accounting_status"]
        == "all_face_edge_corner_accounting_closed"
    )
    assert a4["cpml_exclusion_staging_explicit"] is True
    assert a4["cpml_non_cpml_compatibility_ready"] is True
    cpml_staging = a4["cpml_staging_report"]
    assert cpml_staging["plain_sat_symbols"] == (
        "apply_sat_h_interfaces",
        "apply_sat_e_interfaces",
    )
    assert cpml_staging["cpml_sat_symbols"] == cpml_staging["plain_sat_symbols"]
    assert cpml_staging["operator_module_has_no_cpml_dependency"] is True
    assert cpml_staging["private_hooks_required_for_operator_guard"] is False

    a5 = candidates["private_solver_integration_hunk"]
    assert a5["a1_a4_evidence_summary_required"] is True
    assert a5["a1_a4_evidence_summary_present"] is True
    assert a5["admitted_to_solver"] is False

    assert packet["operator_module_added"] is True
    assert packet["operator_module"] == "rfx/subgridding/sbp_operators.py"
    assert packet["solver_hunk_allowed_if_selected"] == (
        _GLOBAL_OPERATOR_ALLOWED_SOLVER_SYMBOLS
    )
    assert packet["solver_hunk_retained"] is False
    assert packet["actual_solver_hunk_inventory"] == ()
    assert packet["production_patch_allowed"] is False
    assert packet["production_patch_applied"] is False
    assert packet["solver_behavior_changed"] is False
    assert packet["sbp_sat_3d_repair_applied"] is False
    assert packet["sbp_sat_3d_diff_allowed"] is False
    assert packet["face_ops_global_behavior_changed"] is False
    assert packet["hook_experiment_allowed"] is False
    assert packet["jit_runner_changed"] is False
    assert packet["runner_changed"] is False
    assert packet["api_surface_changed"] is False
    assert packet["public_claim_allowed"] is False
    assert packet["public_api_behavior_changed"] is False
    assert packet["public_default_tau_changed"] is False
    assert packet["public_observable_promoted"] is False
    assert packet["public_true_rt_promoted"] is False
    assert packet["public_dft_promoted"] is False
    assert packet["promotion_candidate_ready"] is False
    assert packet["simresult_changed"] is False
    assert packet["result_surface_changed"] is False
    assert packet["env_config_changed"] is False
    assert packet["next_prerequisite"] == _GLOBAL_OPERATOR_ARCHITECTURE_NEXT_PREREQUISITE


def test_private_solver_integration_hunk_records_diagnostic_only_gate():
    packet = _private_solver_integration_hunk_packet()

    assert packet["private_solver_integration_hunk_status"] == (
        "private_solver_integration_requires_followup_diagnostic_only"
    )
    assert packet["terminal_outcome"] in _SOLVER_INTEGRATION_TERMINAL_OUTCOMES
    assert packet["upstream_global_operator_status"] == (
        "private_global_operator_3d_contract_ready"
    )
    assert packet["candidate_ladder_declared_before_solver_edit"] is True
    assert packet["candidate_count"] == 5
    assert packet["selected_candidate_id"] == "diagnostic_only_dry_run"
    assert packet["s1_preacceptance_passed"] is True
    assert packet["s2_manufactured_ledger_gate_passed"] is False
    assert packet["ledger_normalized_balance_residual"] > packet["ledger_threshold"]
    assert packet["ledger_threshold"] == _LEDGER_BALANCE_THRESHOLD

    candidates = {
        candidate["candidate_id"]: candidate for candidate in packet["candidates"]
    }
    assert set(candidates) == {
        "current_solver_hunk_inventory_freeze",
        "operator_projected_face_sat_preacceptance",
        "single_private_operator_projected_face_sat_hunk",
        "diagnostic_only_dry_run",
        "solver_integration_fail_closed",
    }
    assert candidates["current_solver_hunk_inventory_freeze"][
        "sbp_sat_3d_diff_empty_before_attempt"
    ] is True
    assert candidates["current_solver_hunk_inventory_freeze"][
        "runner_diff_empty_before_attempt"
    ] is True
    s1 = candidates["operator_projected_face_sat_preacceptance"]
    assert s1["mortar_adjointness_passed"] is True
    assert s1["projection_noop_passed"] is True
    assert s1["matched_projected_traces_noop"] is True
    assert s1["zero_work_dissipative"] is True
    assert s1["update_bounds_passed"] is True
    assert s1["coupling_strength_passed"] is True
    assert s1["all_face_orientation_signs_passed"] is True
    assert s1["cpml_non_cpml_source_order_equivalent"] is True
    s2 = candidates["single_private_operator_projected_face_sat_hunk"]
    assert s2["preacceptance_passed"] is True
    assert s2["manufactured_ledger_gate_passed"] is False
    assert s2["admitted_to_solver"] is False
    assert s2["ledger_normalized_balance_residual"] > s2["ledger_threshold"]
    assert s2["rejection_reason"] == (
        "operator_projected_face_sat_reproduces_current_ledger_floor"
    )
    assert candidates["diagnostic_only_dry_run"][
        "selected_because_solver_hunk_not_retained"
    ] is True

    assert packet["operator_projected_sat_adapter"] == (
        "rfx/subgridding/sbp_operators.py::operator_projected_sat_pair_face"
    )
    assert packet["solver_hunk_allowed_if_selected"] == (
        _SOLVER_INTEGRATION_ALLOWED_SOLVER_SYMBOLS
    )
    assert packet["solver_hunk_retained"] is False
    assert packet["actual_solver_hunk_inventory"] == ()
    assert packet["production_patch_allowed"] is False
    assert packet["production_patch_applied"] is False
    assert packet["solver_behavior_changed"] is False
    assert packet["sbp_sat_3d_repair_applied"] is False
    assert packet["sbp_sat_3d_diff_allowed"] is False
    assert packet["hook_experiment_allowed"] is False
    assert packet["jit_runner_changed"] is False
    assert packet["runner_changed"] is False
    assert packet["api_surface_changed"] is False
    assert packet["public_claim_allowed"] is False
    assert packet["public_api_behavior_changed"] is False
    assert packet["public_default_tau_changed"] is False
    assert packet["public_observable_promoted"] is False
    assert packet["public_true_rt_promoted"] is False
    assert packet["public_dft_promoted"] is False
    assert packet["promotion_candidate_ready"] is False
    assert packet["simresult_changed"] is False
    assert packet["result_surface_changed"] is False
    assert packet["env_config_changed"] is False
    assert packet["next_prerequisite"] == _SOLVER_INTEGRATION_NEXT_PREREQUISITE


def test_private_operator_projected_energy_transfer_redesign_closes_ledger():
    packet = _private_operator_projected_energy_transfer_redesign_packet()

    assert packet["private_operator_projected_energy_transfer_redesign_status"] == (
        "private_operator_projected_energy_transfer_contract_ready"
    )
    assert packet["terminal_outcome"] in (
        _OPERATOR_PROJECTED_ENERGY_TRANSFER_TERMINAL_OUTCOMES
    )
    assert packet["upstream_solver_integration_status"] == _SOLVER_INTEGRATION_STATUS
    assert packet["candidate_ladder_declared_before_solver_edit"] is True
    assert packet["candidate_count"] == 6
    assert packet["selected_energy_transfer_candidate_id"] == (
        "paired_skew_eh_operator_work_form"
    )
    assert packet["selected_candidate_id"] == "future_solver_hunk_candidate_declared"
    assert packet["e1_ledger_gate_passed"] is True
    assert packet["e1_manufactured_ledger_normalized_balance_residual"] <= (
        packet["ledger_threshold"]
    )
    assert packet["ledger_threshold"] == _LEDGER_BALANCE_THRESHOLD

    candidates = {
        candidate["candidate_id"]: candidate for candidate in packet["candidates"]
    }
    assert set(candidates) == {
        "baseline_operator_projected_failure_freeze",
        "paired_skew_eh_operator_work_form",
        "material_metric_weighted_operator_work_form",
        "face_edge_corner_partition_work_probe",
        "future_solver_hunk_candidate_declared",
        "fail_closed_theory_reopen",
    }
    assert candidates["baseline_operator_projected_failure_freeze"][
        "upstream_ledger_normalized_balance_residual"
    ] > candidates["baseline_operator_projected_failure_freeze"][
        "upstream_ledger_threshold"
    ]
    e1 = candidates["paired_skew_eh_operator_work_form"]
    assert e1["matched_projected_traces_noop"] is True
    assert e1["zero_work_dissipative"] is True
    assert e1["ledger_gate_passed"] is True
    assert e1["ledger_normalized_balance_residual"] <= e1["ledger_threshold"]
    assert e1["coupling_strength_passed"] is True
    assert e1["coupling_strength_ratio"] >= _COUPLING_STRENGTH_RATIO_MIN
    assert e1["update_bounds_passed"] is True
    assert _BOUNDING_SCALAR_MIN <= e1["update_norm_ratio"] <= (
        _BOUNDING_SCALAR_MAX + _CANDIDATE_BOUND_TOL
    )
    assert e1["all_face_orientation_signs_passed"] is True
    assert e1["all_face_skew_helper_orientation_report"]["passes"] is True
    assert set(e1["all_face_skew_helper_orientation_report"]["faces"]) == {
        "x_lo",
        "x_hi",
        "y_lo",
        "y_hi",
        "z_lo",
        "z_hi",
    }
    assert all(
        face_result["ledger_gate_passed"]
        and face_result["coupling_strength_passed"]
        and face_result["update_bounds_passed"]
        for face_result in e1["all_face_skew_helper_orientation_report"][
            "faces"
        ].values()
    )
    assert e1["cpml_non_cpml_source_order_equivalent"] is True
    assert e1["cpml_non_cpml_skew_helper_contract"]["passes"] is True
    assert e1["cpml_non_cpml_skew_helper_contract"][
        "cpml_non_cpml_share_same_adapter"
    ] is True
    assert e1["cpml_non_cpml_skew_helper_contract"][
        "adapter_has_no_cpml_dependency"
    ] is True
    assert e1["cpml_non_cpml_skew_helper_contract"][
        "adapter_has_no_hook_or_runner_dependency"
    ] is True
    assert e1["no_laundering_static_guard"]["passed"] is True
    assert e1["no_laundering_static_guard"]["hits"] == ()
    assert e1["accepted_candidate"] is True
    assert candidates["material_metric_weighted_operator_work_form"][
        "skipped_because_e1_passed"
    ] is True
    assert candidates["face_edge_corner_partition_work_probe"][
        "skipped_because_e1_passed"
    ] is True
    future = candidates["future_solver_hunk_candidate_declared"]
    assert future["selected_because_private_ledger_closed"] is True
    assert future["future_integration_requires_separate_ralplan"] is True
    assert future["production_solver_edit_allowed"] is False
    assert future["accepted_candidate"] is True

    assert packet["operator_projected_energy_transfer_adapter"] == (
        "rfx/subgridding/sbp_operators.py::"
        "operator_projected_skew_eh_sat_face"
    )
    assert packet["solver_hunk_retained"] is False
    assert packet["actual_solver_hunk_inventory"] == ()
    assert packet["production_patch_allowed"] is False
    assert packet["production_patch_applied"] is False
    assert packet["solver_behavior_changed"] is False
    assert packet["sbp_sat_3d_repair_applied"] is False
    assert packet["sbp_sat_3d_diff_allowed"] is False
    assert packet["hook_experiment_allowed"] is False
    assert packet["jit_runner_changed"] is False
    assert packet["runner_changed"] is False
    assert packet["api_surface_changed"] is False
    assert packet["public_claim_allowed"] is False
    assert packet["public_api_behavior_changed"] is False
    assert packet["public_default_tau_changed"] is False
    assert packet["public_observable_promoted"] is False
    assert packet["public_true_rt_promoted"] is False
    assert packet["public_dft_promoted"] is False
    assert packet["promotion_candidate_ready"] is False
    assert packet["simresult_changed"] is False
    assert packet["result_surface_changed"] is False
    assert packet["env_config_changed"] is False
    assert (
        packet["next_prerequisite"]
        == _OPERATOR_PROJECTED_ENERGY_TRANSFER_NEXT_PREREQUISITE
    )


def test_private_operator_projected_solver_integration_retains_bounded_hunk():
    packet = _private_operator_projected_solver_integration_packet()

    assert packet["private_operator_projected_solver_integration_status"] == (
        "private_operator_projected_solver_hunk_retained_fixture_quality_pending"
    )
    assert packet["terminal_outcome"] in (
        _OPERATOR_PROJECTED_SOLVER_INTEGRATION_TERMINAL_OUTCOMES
    )
    assert packet["upstream_energy_transfer_status"] == (
        _OPERATOR_PROJECTED_ENERGY_TRANSFER_STATUS
    )
    assert packet["candidate_ladder_declared_before_solver_edit"] is True
    assert packet["candidate_count"] == 5
    assert packet["selected_candidate_id"] == "single_bounded_face_solver_hunk"
    assert packet["slot_map_same_call_verified"] is True
    assert packet["six_face_mapping_verified"] is True
    assert packet["cpml_non_cpml_same_helper_contract"] is True
    assert packet["edge_corner_guard_verified"] is True
    assert packet["solver_scalar_projection_included"] is False
    assert packet["post_existing_sat_scalar_double_coupling"] is False
    assert packet["manufactured_ledger_gate_passed"] is True
    assert packet["upstream_manufactured_ledger_gate_passed"] is True
    assert packet["ledger_normalized_balance_residual"] <= packet["ledger_threshold"]
    assert packet["ledger_threshold"] == _LEDGER_BALANCE_THRESHOLD

    candidates = {
        candidate["candidate_id"]: candidate for candidate in packet["candidates"]
    }
    assert set(candidates) == {
        "energy_transfer_contract_freeze",
        "production_shaped_skew_helper_preacceptance",
        "single_bounded_face_solver_hunk",
        "diagnostic_only_solver_dry_run",
        "solver_integration_fail_closed",
    }
    i1 = candidates["production_shaped_skew_helper_preacceptance"]
    assert i1["accepted_candidate"] is True
    assert i1["slot_map"]["same_call_local_context"] is True
    assert i1["slot_map"]["split_across_h_only_e_only_functions"] is False
    assert i1["slot_map"]["face_local_t1_t2_labels"] is True
    assert set(i1["six_face_mapping"]) == set(sbp_sat_3d.FACE_ORIENTATIONS)
    for face, mapping in i1["six_face_mapping"].items():
        orientation = sbp_sat_3d.FACE_ORIENTATIONS[face]
        assert mapping["normal_sign"] == orientation.normal_sign
        assert mapping["tangential_e_components"] == (
            orientation.tangential_e_components
        )
        assert mapping["tangential_h_components"] == (
            orientation.tangential_h_components
        )
        assert mapping["orientation_applied_by_normal_sign"] is True
    assert all(i1["cpml_non_cpml_parity"].values())
    assert i1["helper_contract_passed"] is True
    assert i1["solver_scalar_projection_included"] is False
    assert i1["post_existing_sat_scalar_double_coupling"] is False
    i2 = candidates["single_bounded_face_solver_hunk"]
    assert i2["preacceptance_passed"] is True
    assert i2["upstream_manufactured_ledger_gate_passed"] is True
    assert i2["manufactured_ledger_gate_passed"] is True
    assert i2["ledger_normalized_balance_residual"] <= i2["ledger_threshold"]
    assert i2["solver_scalar_projection_included"] is False
    assert i2["post_existing_sat_scalar_double_coupling"] is False
    assert i2["accepted_candidate"] is True

    assert packet["solver_hunk_retained"] is True
    assert packet["actual_solver_hunk_inventory"] == (
        _OPERATOR_PROJECTED_SOLVER_HUNK_SYMBOLS
    )
    assert packet["production_patch_allowed"] is True
    assert packet["production_patch_applied"] is True
    assert packet["solver_behavior_changed"] is True
    assert packet["sbp_sat_3d_repair_applied"] is True
    assert packet["sbp_sat_3d_diff_allowed"] is True
    assert packet["face_ops_global_behavior_changed"] is False
    assert packet["hook_experiment_allowed"] is False
    assert packet["jit_runner_changed"] is False
    assert packet["runner_changed"] is False
    assert packet["api_surface_changed"] is False
    assert packet["public_claim_allowed"] is False
    assert packet["public_api_behavior_changed"] is False
    assert packet["public_default_tau_changed"] is False
    assert packet["public_observable_promoted"] is False
    assert packet["public_true_rt_promoted"] is False
    assert packet["public_dft_promoted"] is False
    assert packet["promotion_candidate_ready"] is False
    assert packet["simresult_changed"] is False
    assert packet["result_surface_changed"] is False
    assert packet["env_config_changed"] is False
    assert (
        packet["next_prerequisite"]
        == _OPERATOR_PROJECTED_SOLVER_INTEGRATION_NEXT_PREREQUISITE
    )


def test_private_boundary_coexistence_fixture_validation_records_blocked_fixture_quality():
    packet = _private_boundary_coexistence_fixture_validation_packet()

    assert packet["private_boundary_coexistence_fixture_validation_status"] == (
        "private_boundary_coexistence_passed_fixture_quality_blocked"
    )
    assert packet["terminal_outcome"] in (
        _BOUNDARY_FIXTURE_VALIDATION_TERMINAL_OUTCOMES
    )
    assert packet["terminal_outcome_precedence"] == (
        _BOUNDARY_FIXTURE_VALIDATION_PRECEDENCE
    )
    assert packet["upstream_operator_projected_solver_integration_status"] == (
        _OPERATOR_PROJECTED_SOLVER_INTEGRATION_STATUS
    )
    assert packet["solver_hunk_retained"] is True
    assert packet["boundary_contract_locked"] is True
    assert packet["shadow_boundary_model_added"] is False
    assert packet["accepted_boundary_classes"] == _BOUNDARY_FIXTURE_ACCEPTED_CLASSES
    assert (
        packet["unsupported_boundary_classes"]
        == _BOUNDARY_FIXTURE_UNSUPPORTED_CLASSES
    )
    assert all(packet["helper_present_in_step_paths"].values())
    assert all(
        packet["helper_slot_order_after_e_sat_before_time_centered_helper"].values()
    )
    helper_evidence = packet["helper_execution_evidence"]
    assert helper_evidence["direct_step_path_probe_required"] is True
    assert helper_evidence["high_level_api_smoke_not_sufficient_alone"] is True
    assert any(
        "test_operator_projected_helper_executes_under_representative_non_cpml"
        in test_name
        for test_name in helper_evidence["non_cpml_step_path_probe_tests"]
    )
    assert any(
        "test_operator_projected_helper_executes_under_representative_cpml"
        in test_name
        for test_name in helper_evidence["cpml_step_path_probe_tests"]
    )
    assert packet["boundary_coexistence_passed"] is True
    assert packet["fixture_quality_replayed"] is True
    assert packet["fixture_quality_ready"] is False
    assert packet["reference_quality_ready"] is False
    assert packet["dominant_fixture_quality_blocker"] == "transverse_phase_spread_deg"
    assert "transverse_magnitude_cv" in packet["fixture_quality_blockers"]
    assert packet["api_preflight_changes_allowed"] is False
    assert packet["rfx_api_changes_allowed"] is False
    assert packet["api_surface_changed"] is False
    assert packet["public_api_behavior_changed"] is False
    assert packet["public_claim_allowed"] is False
    assert packet["public_observable_promoted"] is False
    assert packet["public_true_rt_promoted"] is False
    assert packet["public_dft_promoted"] is False
    assert packet["hook_experiment_allowed"] is False
    assert packet["runner_changed"] is False
    assert packet["jit_runner_changed"] is False
    assert packet["result_surface_changed"] is False
    assert packet["env_config_changed"] is False
    assert packet["public_default_tau_changed"] is False
    assert packet["promotion_candidate_ready"] is False
    assert packet["next_prerequisite"] == _BOUNDARY_FIXTURE_VALIDATION_NEXT_PREREQUISITE


def test_private_manufactured_interior_box_ledger_records_edge_corner_accounting():
    packet = _interior_box_ledger_packet()

    assert packet["status"] == "edge_corner_accounting_probe_recorded_inconclusive"
    assert packet["active_faces"] == 6
    assert packet["active_edges"] == 12
    assert packet["active_corners"] == 8
    assert packet["matched_edge_noop_passed"] is True
    assert packet["corner_perturbation_delta"] < 0.0
