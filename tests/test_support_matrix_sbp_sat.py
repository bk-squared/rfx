"""Regression locks for SBP-SAT benchmark claim level."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPPORT_MATRIX = ROOT / "docs/guides/support_matrix.json"
TRUE_RT_SPEC = ROOT / "docs/guides/sbp_sat_zslab_true_rt_benchmark_spec.md"


def _sbp_lane() -> dict:
    data = json.loads(SUPPORT_MATRIX.read_text())
    return data["lanes"]["sbp_sat_subgridding"]


def test_sbp_sat_support_matrix_records_experimental_proxy_evidence():
    lane = _sbp_lane()

    assert lane["status"] == "experimental"
    assert lane["supported_subset"]["boundary"] == [
        "all_pec",
        "mixed_pec_pmc_reflector_faces",
        "periodic_axes_when_box_is_interior_or_spans_axis",
        "bounded_cpml_absorbing_faces_when_box_is_outside_absorber_guard",
        "not_mixed_pmc_periodic",
        "not_mixed_reflector_or_periodic_with_cpml",
    ]
    assert (
        lane["supported_subset"]["geometry"]
        == "axis_aligned_arbitrary_box_with_cpml_guard_for_absorbing_faces"
    )
    assert lane["supported_subset"]["sources"] == ["soft_point_source"]
    assert lane["supported_subset"]["observables"] == ["point_probe"]

    proxy = lane["benchmark_evidence"]["proxy_crossval"]
    assert proxy["status"] == "implemented"
    assert proxy["test_file"] == "tests/test_subgrid_crossval.py"
    assert proxy["claim_level"] == "proxy_only_not_physical_rt"
    assert proxy["tolerance"] == {
        "relative_amplitude_error_max": 0.05,
        "phase_error_deg_max": 5.0,
    }

    box_proxy = lane["benchmark_evidence"]["box_proxy_crossval"]
    assert box_proxy["status"] == "implemented"
    assert box_proxy["test_file"] == "tests/test_sbp_sat_box_crossval.py"
    assert box_proxy["fixtures"] == [
        "x_face_proxy",
        "y_face_proxy",
        "edge_proxy",
        "corner_proxy",
    ]

    boundary_proxy = lane["benchmark_evidence"]["boundary_proxy_crossval"]
    assert boundary_proxy["status"] == "implemented"
    assert boundary_proxy["test_file"] == "tests/test_sbp_sat_boundary_crossval.py"
    assert boundary_proxy["fixtures"] == [
        "reflector_only_pmc_proxy",
        "periodic_axis_proxy_full_span",
        "periodic_axis_proxy_interior",
    ]

    absorbing_proxy = lane["benchmark_evidence"]["absorbing_proxy_crossval"]
    assert absorbing_proxy["status"] == "implemented"
    assert absorbing_proxy["test_file"] == "tests/test_sbp_sat_absorbing_crossval.py"
    assert absorbing_proxy["fixtures"] == [
        "interior_box_cpml_decay_proxy",
        "cpml_late_tail_vs_pec_cavity_tail",
    ]


def test_sbp_sat_true_rt_benchmark_is_explicitly_deferred():
    true_rt = _sbp_lane()["benchmark_evidence"]["true_reflection_transmission"]

    assert true_rt["status"] == "deferred"
    assert true_rt["spec"] == "docs/guides/sbp_sat_zslab_true_rt_benchmark_spec.md"
    assert true_rt["deferred_issue"].endswith("#deferred-issue-record")
    assert true_rt["public_claim_allowed"] is False
    feasibility_gate = true_rt["feasibility_probe_gate"]
    assert feasibility_gate["status"] == "inconclusive"
    assert feasibility_gate["test_file"] == "tests/test_sbp_sat_true_rt_feasibility.py"
    assert (
        feasibility_gate["claim_level"]
        == "internal_feasibility_only_not_public_rt_or_sparameters"
    )
    assert (
        feasibility_gate["fallback"]
        == "minimal_benchmark_only_flux_or_dft_observable_plan"
    )
    benchmark_gate = true_rt["benchmark_flux_dft_gate"]
    assert benchmark_gate["status"] == "inconclusive"
    assert (
        benchmark_gate["test_file"]
        == "tests/test_sbp_sat_true_rt_flux_dft_benchmark.py"
    )
    assert (
        benchmark_gate["claim_level"]
        == "internal_benchmark_only_not_public_rt_or_sparameters"
    )
    assert benchmark_gate["public_claim_allowed"] is False
    assert benchmark_gate["fixture"] == (
        "boundary_expanded_private_tfsf_style_incident_flux_plane_vacuum"
    )
    assert benchmark_gate["fixture_name"] == "boundary_expanded"
    assert benchmark_gate["source_contract"] == "private_tfsf_style_incident"
    assert (
        benchmark_gate["normalization"]
        == "same_contract_private_reference_vacuum_gated"
    )
    assert benchmark_gate["reference_missing"] is False
    assert benchmark_gate["reference_quality_ready"] is False
    assert (
        benchmark_gate["reference_contract"] == "private_tfsf_style_uniform_reference"
    )
    assert benchmark_gate["slab_rt_scored"] is False
    assert benchmark_gate["fixture_quality_ready"] is False
    assert benchmark_gate["fixture_quality_gates"]["same_contract_reference"] is True
    assert benchmark_gate["fixture_quality_gates"]["vacuum_stability"] is False
    assert (
        benchmark_gate["dominant_reference_quality_blocker"]
        == "transverse_phase_spread_deg"
    )
    blocker_names = [
        blocker["name"] if isinstance(blocker, dict) else blocker
        for blocker in benchmark_gate["reference_quality_blockers"][:3]
    ]
    assert blocker_names[0] == "transverse_phase_spread_deg"
    assert "transverse_magnitude_cv" in blocker_names
    assert "vacuum_relative_magnitude_error" in blocker_names
    assert "predeclared" in benchmark_gate["predeclared_candidate_policy"]
    assert benchmark_gate["causal_ladder_status"] == "row2_causal_classified"
    assert benchmark_gate["causal_class"] == "sbp_sat_interface_floor"
    assert benchmark_gate["causal_class"] != "public_claim_ready"
    assert {
        "dominant_improvement_min": 0.5,
        "paired_improvement_min": 0.25,
        "new_blocker_regression_max": 0.25,
        "source_eta0_relative_error_threshold": 0.02,
        "thresholds_checksum": (
            "d288ae050423c6c2078c3b696da1cbcc05e5095a0ce727b4665b9ecfdb881f9a"
        ),
    }.items() <= benchmark_gate["material_improvement_rule"].items()
    rung5 = benchmark_gate["causal_ladder_rungs"]["rung5_interface_floor"]
    assert (rung5["status"] if isinstance(rung5, dict) else rung5) == "implicated"
    assert (
        benchmark_gate["causal_ladder_candidates"][2]["candidate_id"]
        == "rung4_central_core_aperture"
    )
    decision = benchmark_gate["causal_ladder_candidates"][2][
        "classification_decision"
    ]
    assert (
        decision["classification_decision"] if isinstance(decision, dict) else decision
    ) == "inconclusive"
    assert benchmark_gate["interface_floor_investigation_status"] == "complete"
    assert (
        benchmark_gate["interface_floor_subclass"]
        == "coarse_fine_energy_transfer_mismatch"
    )
    assert benchmark_gate["interface_distance_sensitivity"] == "persistent"
    assert {
        candidate["candidate_id"]
        for candidate in benchmark_gate["interface_distance_candidates"]
    } >= {"baseline_boundary_expanded", "nearer_current_bounded_control"}
    assert (
        benchmark_gate["interface_energy_transfer_diagnostics"][
            "front_back_ratio_formula"
        ]
        == "signed_back / max(abs(signed_front), floor)"
    )
    assert (
        benchmark_gate["interface_energy_transfer_diagnostics"][
            "interface_residual_stable"
        ]
        is True
    )
    assert (
        benchmark_gate["interface_energy_transfer_diagnostics"][
            "uniform_reference_below_threshold"
        ]
        is True
    )
    assert benchmark_gate["cpml_proximity_controls"]
    assert all(test["passed"] for test in benchmark_gate["direct_invariant_tests"])
    repair = benchmark_gate["private_energy_transfer_repair"]
    assert benchmark_gate["private_energy_transfer_repair_status"] == (
        "no_material_repair"
    )
    assert repair["status"] == "no_material_repair"
    assert repair["candidate_policy"].startswith("tau sensitivity")
    assert repair["tau_candidates"] == [0.25, 0.5, 0.75, 1.0]
    assert repair["baseline_max_ratio_error"] > repair["candidate_max_ratio_error"]
    assert repair["selected_repair_candidate_id"] == "tau_sensitivity_1p0"
    assert repair["accepted_private_repair"] is False
    assert repair["public_claim_allowed"] is False
    assert repair["promotion_candidate_ready"] is False
    assert repair["public_default_tau_changed"] is False
    assert repair["public_api_behavior_changed"] is False
    assert repair["kernel_edit_applied"] is False
    assert repair["baseline_artifact_required"] is True
    assert repair["pre_post_evidence_required"] is True
    repair_regression_metrics = {
        regression["metric"] for regression in repair["paired_metric_regressions"]
    }
    assert repair_regression_metrics & {
        "vacuum_phase_error_deg",
        "transverse_phase_spread_deg",
        "transverse_magnitude_cv",
    }
    assert all(
        candidate["material_improvement_passed"] is False
        for candidate in repair["candidates"]
    )
    theory = benchmark_gate["private_energy_transfer_theory_review"]
    assert benchmark_gate["private_energy_transfer_theory_review_status"] == "complete"
    assert theory["status"] == "complete"
    assert theory["selected_next_plan_direction"] == "kernel_repair_candidate"
    assert theory["selected_hypothesis"] == "discrete_eh_work_ledger_mismatch"
    assert theory["executable_diagnostics_added"] is False
    assert theory["solver_behavior_changed"] is False
    assert theory["sbp_sat_3d_diff_allowed"] is False
    assert theory["hook_experiment_allowed"] is False
    assert theory["jit_runner_changed"] is False
    assert theory["public_claim_allowed"] is False
    assert theory["public_api_behavior_changed"] is False
    assert theory["public_default_tau_changed"] is False
    assert theory["evidence_basis"]["prior_repair_status"] == "no_material_repair"
    assert (
        theory["evidence_basis"]["relative_improvement"]
        < (theory["evidence_basis"]["material_improvement_required"])
    )
    theory_hypotheses = {
        candidate["hypothesis"] for candidate in theory["candidate_hypotheses"]
    }
    assert theory_hypotheses >= {
        "discrete_eh_work_ledger_mismatch",
        "private_hook_or_stagger_mismatch",
    }
    ledger = benchmark_gate["private_manufactured_energy_ledger_diagnostic"]
    assert (
        benchmark_gate["private_manufactured_energy_ledger_diagnostic_status"]
        == "ledger_mismatch_detected"
    )
    assert ledger["status"] == "ledger_mismatch_detected"
    assert ledger["selected_hypothesis"] == "discrete_eh_work_ledger_mismatch"
    assert ledger["selected_next_plan_direction"] == "bounded_kernel_repair_candidate"
    assert ledger["diagnostic_scope"] == "private_manufactured_interface_only"
    assert ledger["executable_diagnostics_added"] is True
    assert ledger["face_ledger_status"] == "ledger_mismatch_detected"
    assert ledger["zero_work_face_status"] == "zero_work_dissipative_gate_passed"
    assert (
        ledger["interior_box_ledger_status"]
        == "edge_corner_accounting_probe_recorded_inconclusive"
    )
    assert ledger["normalized_balance_residual"] > ledger["threshold"]
    assert ledger["threshold"] == 0.02
    assert ledger["solver_behavior_changed"] is False
    assert ledger["sbp_sat_3d_repair_applied"] is False
    assert ledger["hook_experiment_allowed"] is False
    assert ledger["jit_runner_changed"] is False
    assert ledger["public_claim_allowed"] is False
    assert ledger["public_api_behavior_changed"] is False
    assert ledger["public_default_tau_changed"] is False
    assert ledger["public_observable_promoted"] is False
    bounded_repair = benchmark_gate["private_bounded_kernel_repair"]
    assert benchmark_gate["private_bounded_kernel_repair_status"] == (
        "no_signature_compatible_bounded_repair"
    )
    assert bounded_repair["status"] == "no_signature_compatible_bounded_repair"
    assert bounded_repair["selected_repair_candidate_id"] is None
    assert bounded_repair["accepted_private_repair"] is False
    assert bounded_repair["solver_behavior_changed"] is False
    assert bounded_repair["sbp_sat_3d_repair_applied"] is False
    assert bounded_repair["hook_experiment_allowed"] is False
    assert bounded_repair["jit_runner_changed"] is False
    assert bounded_repair["public_claim_allowed"] is False
    assert bounded_repair["public_api_behavior_changed"] is False
    assert bounded_repair["public_default_tau_changed"] is False
    assert bounded_repair["public_observable_promoted"] is False
    assert bounded_repair["simresult_changed"] is False
    assert bounded_repair["phase0_forbidden_diff_required"] is True
    assert bounded_repair["final_forbidden_diff_matches_phase0"] is True
    assert bounded_repair["rejection_subreason"] == (
        "no_bounded_candidate_passed_ledger_gate"
    )
    assert bounded_repair["candidate_count"] == len(bounded_repair["candidates"])
    assert not any(
        candidate["accepted_candidate"] for candidate in bounded_repair["candidates"]
    )
    assert any(
        candidate["candidate_id"] == "norm_reciprocal_coarse_emphasis"
        and candidate["ledger_normalized_balance_residual"]
        > candidate["ledger_threshold"]
        for candidate in bounded_repair["candidates"]
    )
    assert any(
        candidate["candidate_id"] == "under_coupled_ledger_control"
        and candidate["ledger_gate_passed"] is True
        and candidate["rejection_subreason"] == "candidate_under_couples"
        for candidate in bounded_repair["candidates"]
    )
    assert (
        benchmark_gate["private_bounded_kernel_repair_next_prerequisite"]
        == (bounded_repair["next_prerequisite"])
    )
    assert benchmark_gate["hook_contingency_justification"]["eligible"] is False
    assert benchmark_gate["same_run_repair_allowed"] is False
    assert benchmark_gate["usable_passband_threshold"]["min_bins"] == 2
    assert benchmark_gate["transverse_uniformity_threshold"] == {
        "magnitude_cv_max": 0.01,
        "phase_spread_deg_max": 1.0,
    }
    assert benchmark_gate["vacuum_stability_threshold"] == {
        "relative_magnitude_error_max": 0.02,
        "phase_error_deg_max": 2.0,
    }
    assert benchmark_gate["no_go_reason"].startswith(
        "private TFSF-style incident fixture has a same-contract reference"
    )
    assert (
        "same-contract private reference helper is present"
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        "coarse_fine_energy_transfer_mismatch" in benchmark_gate["blocking_diagnostic"]
    )
    assert "causal ladder" in benchmark_gate["blocking_diagnostic"]
    assert "discrete_eh_work_ledger_mismatch" in benchmark_gate["blocking_diagnostic"]
    assert "ledger_mismatch_detected" in benchmark_gate["blocking_diagnostic"]
    assert (
        "private SAT face-coupling theory/redesign"
        in benchmark_gate["private_bounded_kernel_repair_next_prerequisite"]
    )
    assert (
        "no_signature_compatible_bounded_repair"
        in benchmark_gate["blocking_diagnostic"]
    )
    face_coupling = benchmark_gate["private_face_coupling_theory"]
    assert benchmark_gate["private_face_coupling_theory_status"] == (
        "paired_face_coupling_design_ready"
    )
    assert face_coupling["status"] == "paired_face_coupling_design_ready"
    assert face_coupling["terminal_outcome"] == "paired_face_coupling_design_ready"
    assert face_coupling["selected_candidate_id"] == (
        "matched_hy_common_mode_energy_closure"
    )
    assert face_coupling["selected_next_safe_implementation_lane"] == (
        "private paired-face helper implementation ralplan"
    )
    assert face_coupling["next_prerequisite"] == (
        "private paired-face helper implementation ralplan"
    )
    assert (
        benchmark_gate["private_face_coupling_theory_next_prerequisite"]
        == (face_coupling["next_prerequisite"])
    )
    assert face_coupling["solver_behavior_changed"] is False
    assert face_coupling["sbp_sat_3d_repair_applied"] is False
    assert face_coupling["hook_experiment_allowed"] is False
    assert face_coupling["jit_runner_changed"] is False
    assert face_coupling["runner_changed"] is False
    assert face_coupling["public_claim_allowed"] is False
    assert face_coupling["public_api_behavior_changed"] is False
    assert face_coupling["public_default_tau_changed"] is False
    assert face_coupling["public_observable_promoted"] is False
    assert face_coupling["promotion_candidate_ready"] is False
    assert face_coupling["simresult_changed"] is False
    assert face_coupling["result_surface_changed"] is False
    assert (
        face_coupling["orientation_sign_convention"]["poynting_trace"]
        == "S_n = Ex * Hy - Ey * Hx"
    )
    assert "minimum-norm root" in face_coupling["equations"]["candidate"]
    face_candidate = face_coupling["candidates"][0]
    assert face_candidate["accepted_candidate"] is True
    assert face_candidate["candidate_family"] == (
        "same_step_paired_eh_matched_magnetic_common_mode"
    )
    assert face_candidate["ledger_gate_passed"] is True
    assert (
        face_candidate["ledger_normalized_balance_residual"]
        <= face_candidate["ledger_threshold"]
    )
    assert face_candidate["zero_work_gate_passed"] is True
    assert face_candidate["matched_projected_traces_noop"] is True
    assert face_candidate["coupling_strength_passed"] is True
    assert face_candidate["coupling_strength_ratio"] >= 0.5
    assert face_candidate["update_bounds_passed"] is True
    assert face_candidate["branches_on_measured_residual_or_test_name"] is False
    assert face_candidate["requires_paired_eh_face_context"] is True
    assert face_candidate["requires_time_centered_staging"] is False
    paired_helper = benchmark_gate["private_paired_face_helper_implementation"]
    assert benchmark_gate["private_paired_face_helper_implementation_status"] == (
        "production_context_mismatch_detected"
    )
    assert paired_helper["status"] == "production_context_mismatch_detected"
    assert paired_helper["terminal_outcome"] == "production_context_mismatch_detected"
    assert paired_helper["upstream_theory_status"] == (
        "paired_face_coupling_design_ready"
    )
    assert paired_helper["selected_theory_candidate_id"] == (
        "matched_hy_common_mode_energy_closure"
    )
    assert paired_helper["production_shaped_gate_attempted"] is True
    assert paired_helper["production_context_gate_passed"] is False
    assert paired_helper["phase1_failure_reason"] == (
        "step_order_exposes_h_sat_and_e_sat_in_different_temporal_slots"
    )
    for path_name in ("non_cpml", "cpml"):
        order_gate = paired_helper["source_order_gates"][path_name]
        assert order_gate["h_sat_before_e_update"] is True
        assert order_gate["e_sat_after_all_e_updates"] is True
        assert (
            order_gate["same_discrete_work_ledger_available_without_staging"] is False
        )
    assert (
        paired_helper["cpml_non_cpml_face_work_equivalence"][
            "same_sat_order_mismatch_in_both_paths"
        ]
        is True
    )
    assert paired_helper["orientation_generalization"]["blocked_by_orientation"] is (
        False
    )
    assert paired_helper["requires_time_centered_staging"] is True
    assert paired_helper["production_patch_allowed"] is False
    assert paired_helper["production_patch_applied"] is False
    assert paired_helper["accepted_private_helper"] is False
    assert paired_helper["solver_behavior_changed"] is False
    assert paired_helper["sbp_sat_3d_paired_face_helper_applied"] is False
    assert paired_helper["sbp_sat_3d_diff_allowed"] is False
    assert paired_helper["helper_specific_switch_added"] is False
    assert paired_helper["hook_experiment_allowed"] is False
    assert paired_helper["jit_runner_changed"] is False
    assert paired_helper["runner_changed"] is False
    assert paired_helper["public_claim_allowed"] is False
    assert paired_helper["public_api_behavior_changed"] is False
    assert paired_helper["public_default_tau_changed"] is False
    assert paired_helper["public_observable_promoted"] is False
    assert paired_helper["promotion_candidate_ready"] is False
    assert paired_helper["simresult_changed"] is False
    assert paired_helper["result_surface_changed"] is False
    assert paired_helper["final_sbp_sat_3d_diff_matches_phase0"] is True
    assert (
        benchmark_gate["private_paired_face_helper_implementation_next_prerequisite"]
        == (paired_helper["next_prerequisite"])
    )
    repair = benchmark_gate["private_interface_floor_repair"]
    assert benchmark_gate["private_interface_floor_repair_status"] == (
        "no_bounded_private_interface_floor_repair"
    )
    assert (
        repair["terminal_outcome"]
        == benchmark_gate["private_interface_floor_repair_status"]
    )
    assert repair["candidate_ladder_declared_before_solver_edit"] is True
    assert repair["candidate_count"] == 5
    repair_candidates = {
        candidate["candidate_id"]: candidate for candidate in repair["candidates"]
    }
    assert (
        repair_candidates["oriented_characteristic_face_balance"]["ledger_gate_passed"]
        is False
    )
    assert (
        repair_candidates["oriented_characteristic_face_balance"][
            "characteristic_equivalent_to_current_component_sat"
        ]
        is True
    )
    assert repair["solver_hunk_retained"] is False
    assert repair["actual_solver_hunk_inventory"] == []
    assert repair["production_patch_allowed"] is False
    assert repair["production_patch_applied"] is False
    assert repair["public_claim_allowed"] is False
    assert repair["public_observable_promoted"] is False
    assert (
        benchmark_gate["private_interface_floor_repair_next_prerequisite"]
        == repair["next_prerequisite"]
    )
    face_norm = benchmark_gate["private_face_norm_operator_repair"]
    assert benchmark_gate["private_face_norm_operator_repair_status"] == (
        "no_private_face_norm_operator_repair"
    )
    assert (
        face_norm["terminal_outcome"]
        == benchmark_gate["private_face_norm_operator_repair_status"]
    )
    assert face_norm["candidate_ladder_declared_before_solver_edit"] is True
    assert face_norm["candidate_count"] == 5
    face_norm_candidates = {
        candidate["candidate_id"]: candidate for candidate in face_norm["candidates"]
    }
    assert set(face_norm_candidates) == {
        "current_face_operator_norm_adjoint_audit",
        "mass_adjoint_restriction_face_sat",
        "uniform_diagonal_face_norm_rescaling_guard",
        "higher_order_projection_guard",
        "full_box_edge_corner_norm_preacceptance",
    }
    assert (
        face_norm_candidates["mass_adjoint_restriction_face_sat"][
            "ledger_gate_passed"
        ]
        is False
    )
    assert (
        face_norm_candidates["mass_adjoint_restriction_face_sat"][
            "matched_projected_traces_noop"
        ]
        is False
    )
    assert (
        face_norm_candidates["uniform_diagonal_face_norm_rescaling_guard"][
            "identical_to_current_uniform_face_norms"
        ]
        is True
    )
    assert (
        face_norm_candidates["higher_order_projection_guard"]["status"]
        == "higher_order_projection_requires_broader_operator_plan"
    )
    assert face_norm["solver_hunk_retained"] is False
    assert face_norm["production_patch_allowed"] is False
    assert face_norm["production_patch_applied"] is False
    assert face_norm["public_claim_allowed"] is False
    assert face_norm["public_observable_promoted"] is False
    assert (
        benchmark_gate["private_face_norm_operator_repair_next_prerequisite"]
        == face_norm["next_prerequisite"]
    )
    derivative = benchmark_gate["private_derivative_interface_repair"]
    assert benchmark_gate["private_derivative_interface_repair_status"] == (
        "no_private_derivative_interface_repair"
    )
    assert (
        derivative["terminal_outcome"]
        == benchmark_gate["private_derivative_interface_repair_status"]
    )
    assert derivative["upstream_face_norm_operator_repair_status"] == (
        benchmark_gate["private_face_norm_operator_repair_status"]
    )
    assert derivative["candidate_ladder_declared_before_solver_edit"] is True
    assert derivative["candidate_count"] == 6
    assert derivative["selected_candidate_id"] is None
    assert derivative["reduced_fixture_reproduces_failure"] is True
    assert derivative["reduced_identity_closed_test_locally"] is True
    assert derivative["requires_global_sbp_operator_refactor"] is True
    derivative_candidates = {
        candidate["candidate_id"]: candidate for candidate in derivative["candidates"]
    }
    assert set(derivative_candidates) == {
        "current_derivative_energy_identity_audit",
        "reduced_normal_incidence_energy_flux",
        "full_yz_face_energy_flux_candidate",
        "edge_corner_cochain_accounting_guard",
        "mortar_projection_operator_widening_guard",
        "private_solver_integration_candidate",
    }
    assert (
        derivative_candidates["full_yz_face_energy_flux_candidate"][
            "manufactured_ledger_gate_passed"
        ]
        is False
    )
    assert (
        derivative_candidates["mortar_projection_operator_widening_guard"]["status"]
        == "requires_global_sbp_operator_refactor"
    )
    assert derivative["solver_hunk_retained"] is False
    assert derivative["actual_solver_hunk_inventory"] == []
    assert derivative["production_patch_allowed"] is False
    assert derivative["production_patch_applied"] is False
    assert derivative["solver_behavior_changed"] is False
    assert derivative["sbp_sat_3d_repair_applied"] is False
    assert derivative["public_claim_allowed"] is False
    assert derivative["public_observable_promoted"] is False
    assert derivative["hook_experiment_allowed"] is False
    assert derivative["api_surface_changed"] is False
    assert derivative["result_surface_changed"] is False
    assert derivative["runner_surface_changed"] is False
    assert derivative["env_config_changed"] is False
    assert (
        benchmark_gate["private_derivative_interface_repair_next_prerequisite"]
        == derivative["next_prerequisite"]
    )
    global_operator = benchmark_gate[
        "private_global_derivative_mortar_operator_architecture"
    ]
    assert (
        benchmark_gate[
            "private_global_derivative_mortar_operator_architecture_status"
        ]
        == "private_global_operator_3d_contract_ready"
    )
    assert (
        global_operator["terminal_outcome"]
        == benchmark_gate[
            "private_global_derivative_mortar_operator_architecture_status"
        ]
    )
    assert global_operator["upstream_derivative_interface_repair_status"] == (
        benchmark_gate["private_derivative_interface_repair_status"]
    )
    assert global_operator["candidate_ladder_declared_before_solver_edit"] is True
    assert global_operator["candidate_count"] == 7
    assert global_operator["selected_candidate_id"] == (
        "all_faces_edge_corner_operator_guard"
    )
    assert global_operator["a1_a4_evidence_summary"] == {
        "sbp_derivative_norm_boundary_contract": True,
        "norm_compatible_mortar_projection_contract": True,
        "em_tangential_interface_flux_contract": True,
        "all_faces_edge_corner_operator_guard": True,
    }
    global_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in global_operator["candidates"]
    }
    assert set(global_candidates) == {
        "current_operator_inventory_and_freeze",
        "sbp_derivative_norm_boundary_contract",
        "norm_compatible_mortar_projection_contract",
        "em_tangential_interface_flux_contract",
        "all_faces_edge_corner_operator_guard",
        "private_solver_integration_hunk",
        "operator_architecture_fail_closed",
    }
    assert global_candidates["sbp_derivative_norm_boundary_contract"][
        "yee_staggered_dual_identity_passed"
    ] is True
    assert global_candidates["norm_compatible_mortar_projection_contract"][
        "mortar_adjointness_passed"
    ] is True
    assert global_candidates["norm_compatible_mortar_projection_contract"][
        "linear_reproduction_passed"
    ] is True
    assert global_candidates["em_tangential_interface_flux_contract"][
        "material_metric_weighting_explicit"
    ] is True
    assert global_candidates["em_tangential_interface_flux_contract"][
        "flux_identity_passed"
    ] is True
    assert global_candidates["all_faces_edge_corner_operator_guard"][
        "active_faces"
    ] == 6
    assert global_candidates["all_faces_edge_corner_operator_guard"][
        "active_edges"
    ] == 12
    assert global_candidates["all_faces_edge_corner_operator_guard"][
        "active_corners"
    ] == 8
    assert global_candidates["all_faces_edge_corner_operator_guard"][
        "all_face_flux_identity_passed"
    ] is True
    assert global_candidates["all_faces_edge_corner_operator_guard"][
        "all_face_flux_identity_max_abs_residual"
    ] <= 1.0e-12
    assert global_candidates["all_faces_edge_corner_operator_guard"][
        "surface_partition_closes"
    ] is True
    assert global_candidates["all_faces_edge_corner_operator_guard"][
        "edge_corner_accounting_status"
    ] == "all_face_edge_corner_accounting_closed"
    assert global_candidates["all_faces_edge_corner_operator_guard"][
        "cpml_staging_report"
    ]["operator_module_has_no_cpml_dependency"] is True
    assert global_candidates["private_solver_integration_hunk"][
        "a1_a4_evidence_summary_present"
    ] is True
    assert global_candidates["private_solver_integration_hunk"][
        "admitted_to_solver"
    ] is False
    assert global_operator["operator_module_added"] is True
    assert global_operator["operator_module"] == "rfx/subgridding/sbp_operators.py"
    assert global_operator["solver_hunk_retained"] is False
    assert global_operator["actual_solver_hunk_inventory"] == []
    assert global_operator["production_patch_allowed"] is False
    assert global_operator["production_patch_applied"] is False
    assert global_operator["solver_behavior_changed"] is False
    assert global_operator["sbp_sat_3d_repair_applied"] is False
    assert global_operator["public_claim_allowed"] is False
    assert global_operator["public_observable_promoted"] is False
    assert global_operator["hook_experiment_allowed"] is False
    assert (
        benchmark_gate[
            "private_global_derivative_mortar_operator_architecture_next_prerequisite"
        ]
        == global_operator["next_prerequisite"]
    )
    solver_integration = benchmark_gate["private_solver_integration_hunk"]
    assert benchmark_gate["private_solver_integration_hunk_status"] == (
        "private_solver_integration_requires_followup_diagnostic_only"
    )
    assert solver_integration["terminal_outcome"] == (
        benchmark_gate["private_solver_integration_hunk_status"]
    )
    assert solver_integration["upstream_global_operator_status"] == (
        benchmark_gate["private_global_derivative_mortar_operator_architecture_status"]
    )
    assert solver_integration["selected_candidate_id"] == "diagnostic_only_dry_run"
    assert solver_integration["s1_preacceptance_passed"] is True
    assert solver_integration["s2_manufactured_ledger_gate_passed"] is False
    assert solver_integration["ledger_normalized_balance_residual"] > (
        solver_integration["ledger_threshold"]
    )
    assert solver_integration["solver_hunk_retained"] is False
    assert solver_integration["actual_solver_hunk_inventory"] == []
    assert solver_integration["production_patch_applied"] is False
    assert solver_integration["sbp_sat_3d_repair_applied"] is False
    assert solver_integration["public_claim_allowed"] is False
    assert solver_integration["public_observable_promoted"] is False
    assert solver_integration["hook_experiment_allowed"] is False
    energy_transfer = benchmark_gate[
        "private_operator_projected_energy_transfer_redesign"
    ]
    assert benchmark_gate[
        "private_operator_projected_energy_transfer_redesign_status"
    ] == "private_operator_projected_energy_transfer_contract_ready"
    assert energy_transfer["upstream_solver_integration_status"] == (
        benchmark_gate["private_solver_integration_hunk_status"]
    )
    assert energy_transfer["selected_energy_transfer_candidate_id"] == (
        "paired_skew_eh_operator_work_form"
    )
    assert energy_transfer["selected_candidate_id"] == (
        "future_solver_hunk_candidate_declared"
    )
    assert energy_transfer["e1_ledger_gate_passed"] is True
    assert energy_transfer["e1_manufactured_ledger_normalized_balance_residual"] <= (
        energy_transfer["ledger_threshold"]
    )
    energy_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in energy_transfer["candidates"]
    }
    e1_energy = energy_candidates["paired_skew_eh_operator_work_form"]
    assert e1_energy["all_face_skew_helper_orientation_report"]["passes"] is True
    assert e1_energy["cpml_non_cpml_skew_helper_contract"]["passes"] is True
    assert energy_transfer["solver_hunk_retained"] is False
    assert energy_transfer["actual_solver_hunk_inventory"] == []
    assert energy_transfer["production_patch_applied"] is False
    assert energy_transfer["sbp_sat_3d_repair_applied"] is False
    assert energy_transfer["public_claim_allowed"] is False
    assert energy_transfer["public_observable_promoted"] is False
    assert energy_transfer["hook_experiment_allowed"] is False
    assert (
        benchmark_gate[
            "private_operator_projected_energy_transfer_redesign_next_prerequisite"
        ]
        == energy_transfer["next_prerequisite"]
    )
    operator_solver = benchmark_gate["private_operator_projected_solver_integration"]
    assert benchmark_gate["private_operator_projected_solver_integration_status"] == (
        "private_operator_projected_solver_hunk_retained_fixture_quality_pending"
    )
    assert operator_solver["upstream_energy_transfer_status"] == (
        benchmark_gate["private_operator_projected_energy_transfer_redesign_status"]
    )
    assert operator_solver["selected_candidate_id"] == "single_bounded_face_solver_hunk"
    assert operator_solver["slot_map_same_call_verified"] is True
    assert operator_solver["six_face_mapping_verified"] is True
    assert operator_solver["cpml_non_cpml_same_helper_contract"] is True
    assert operator_solver["edge_corner_guard_verified"] is True
    assert operator_solver["normal_sign_orientation_verified"] is True
    assert operator_solver["solver_scalar_projection_included"] is False
    assert operator_solver["post_existing_sat_scalar_double_coupling"] is False
    assert operator_solver["upstream_manufactured_ledger_gate_passed"] is True
    assert operator_solver["manufactured_ledger_gate_passed"] is True
    assert operator_solver["ledger_normalized_balance_residual"] <= (
        operator_solver["ledger_threshold"]
    )
    assert operator_solver["solver_hunk_retained"] is True
    assert operator_solver["production_patch_applied"] is True
    assert operator_solver["sbp_sat_3d_repair_applied"] is True
    assert operator_solver["public_claim_allowed"] is False
    assert operator_solver["public_observable_promoted"] is False
    assert operator_solver["hook_experiment_allowed"] is False
    assert (
        benchmark_gate[
            "private_operator_projected_solver_integration_next_prerequisite"
        ]
        == operator_solver["next_prerequisite"]
    )
    boundary_fixture = benchmark_gate[
        "private_boundary_coexistence_fixture_validation"
    ]
    assert benchmark_gate[
        "private_boundary_coexistence_fixture_validation_status"
    ] == "private_boundary_coexistence_passed_fixture_quality_blocked"
    assert boundary_fixture[
        "upstream_operator_projected_solver_integration_status"
    ] == benchmark_gate["private_operator_projected_solver_integration_status"]
    assert boundary_fixture["solver_hunk_retained"] is True
    assert boundary_fixture["boundary_contract_locked"] is True
    assert boundary_fixture["shadow_boundary_model_added"] is False
    assert boundary_fixture["accepted_boundary_classes"] == [
        "all_pec",
        "selected_pmc_reflector_faces",
        "periodic_axes_when_box_is_interior_or_spans_axis",
        "scalar_cpml_bounded_interior_box",
        "boundaryspec_uniform_cpml_bounded_interior_box",
    ]
    assert boundary_fixture["unsupported_boundary_classes"] == [
        "upml",
        "per_face_cpml_thickness_overrides",
        "mixed_cpml_reflector",
        "mixed_cpml_periodic",
        "mixed_pmc_periodic",
        "one_side_touch_periodic_axis",
        "mixed_absorber_families",
    ]
    assert boundary_fixture["boundary_coexistence_passed"] is True
    assert boundary_fixture["fixture_quality_replayed"] is True
    assert boundary_fixture["fixture_quality_ready"] is False
    assert boundary_fixture["reference_quality_ready"] is False
    assert (
        boundary_fixture["dominant_fixture_quality_blocker"]
        == "transverse_phase_spread_deg"
    )
    assert (
        boundary_fixture["helper_execution_evidence"][
            "direct_step_path_probe_required"
        ]
        is True
    )
    assert boundary_fixture["api_preflight_changes_allowed"] is False
    assert boundary_fixture["rfx_api_changes_allowed"] is False
    assert boundary_fixture["api_surface_changed"] is False
    assert boundary_fixture["public_api_behavior_changed"] is False
    assert boundary_fixture["public_claim_allowed"] is False
    assert boundary_fixture["public_observable_promoted"] is False
    assert (
        benchmark_gate[
            "private_boundary_coexistence_fixture_validation_next_prerequisite"
        ]
        == boundary_fixture["next_prerequisite"]
    )
    fixture_repair = benchmark_gate["private_fixture_quality_blocker_repair"]
    assert benchmark_gate["private_fixture_quality_blocker_repair_status"] == (
        "private_fixture_quality_blocker_persists_no_public_promotion"
    )
    assert fixture_repair["terminal_outcome"] == (
        "private_fixture_quality_blocker_persists_no_public_promotion"
    )
    assert fixture_repair[
        "upstream_boundary_coexistence_fixture_validation_status"
    ] == benchmark_gate["private_boundary_coexistence_fixture_validation_status"]
    assert fixture_repair["candidate_ladder_declared_before_slow_scoring"] is True
    assert fixture_repair["candidate_count"] == 5
    assert (
        fixture_repair["thresholds_checksum"]
        == benchmark_gate["material_improvement_rule"]["thresholds_checksum"]
    )
    assert fixture_repair["baseline_failure_retained"] is True
    assert fixture_repair["fixture_quality_ready"] is False
    assert fixture_repair["reference_quality_ready"] is False
    assert (
        fixture_repair["selected_candidate_id"]
        == "F4_fail_closed_fixture_blocker_persists"
    )
    assert fixture_repair["candidate_ladder"][-1]["accepted_candidate"] is True
    assert fixture_repair["measurement_controls_can_replace_original_fixture"] is False
    assert fixture_repair["solver_hunk_retained"] is False
    assert fixture_repair["solver_behavior_changed"] is False
    assert fixture_repair["production_patch_applied"] is False
    assert fixture_repair["sbp_sat_3d_repair_applied"] is False
    assert fixture_repair["api_preflight_changes_allowed"] is False
    assert fixture_repair["rfx_api_changes_allowed"] is False
    assert fixture_repair["public_claim_allowed"] is False
    assert fixture_repair["public_observable_promoted"] is False
    assert fixture_repair["true_rt_public_observable_promoted"] is False
    assert fixture_repair["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate["private_fixture_quality_blocker_repair_next_prerequisite"]
        == fixture_repair["next_prerequisite"]
    )
    fixture_candidates = {
        candidate["source_candidate_id"]: candidate
        for candidate in fixture_repair["private_fixture_candidates"]
    }
    assert "C0_current_helper_original_fixture" in fixture_candidates
    assert "C1_center_core_measurement_control" in fixture_candidates
    assert fixture_candidates["C1_center_core_measurement_control"][
        "measurement_control_only"
    ] is True
    assert not any(
        candidate["accepted_candidate"]
        for candidate in fixture_repair["private_fixture_candidates"]
    )
    source_reference = benchmark_gate[
        "private_source_reference_phase_front_fixture_contract"
    ]
    assert benchmark_gate[
        "private_source_reference_phase_front_fixture_contract_status"
    ] == "private_source_phase_front_self_oracle_failed"
    assert source_reference["terminal_outcome"] == (
        "private_source_phase_front_self_oracle_failed"
    )
    assert source_reference[
        "upstream_fixture_quality_blocker_repair_status"
    ] == benchmark_gate["private_fixture_quality_blocker_repair_status"]
    assert source_reference[
        "upstream_boundary_coexistence_fixture_validation_status"
    ] == benchmark_gate["private_boundary_coexistence_fixture_validation_status"]
    assert (
        source_reference["candidate_ladder_declared_before_slow_scoring"] is True
    )
    assert source_reference["candidate_count"] == 6
    assert (
        source_reference["thresholds_checksum"]
        == benchmark_gate["material_improvement_rule"]["thresholds_checksum"]
    )
    assert source_reference["selected_candidate_id"] == (
        "P1_phase_front_self_oracle"
    )
    assert source_reference["source_phase_front_self_oracle_failed"] is True
    assert source_reference["source_phase_front_self_oracle_ready"] is False
    assert source_reference["reference_normalization_contract_ready"] is False
    assert source_reference["private_fixture_contract_ready"] is False
    assert source_reference["solver_interface_floor_reconfirmed"] is False
    assert (
        source_reference[
            "source_reference_self_oracle_separated_from_subgrid_parity"
        ]
        is True
    )
    assert source_reference["subgrid_vacuum_parity_used_for_p1_selection"] is False
    phase_front_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in source_reference["candidate_ladder"]
    }
    p1 = phase_front_candidates["P1_phase_front_self_oracle"]
    assert p1["self_oracle_uses_uniform_reference_only"] is True
    assert p1["subgrid_vacuum_parity_used_for_self_oracle"] is False
    assert p1["uniform_reference_ready"] is False
    assert p1["metrics"]["max_uniform_center_referenced_phase_spread_deg"] > 1.0
    assert p1["metrics"]["max_uniform_modal_magnitude_cv"] > 0.01
    p2 = phase_front_candidates[
        "P2_same_contract_reference_normalization_redesign"
    ]
    assert p2["accepted_candidate"] is False
    assert p2["d3_normalization_contract_ready"] is False
    assert p2["mask_provenance_ready"] is True
    p3 = phase_front_candidates["P3_finite_fixture_contract_candidates"]
    assert p3["old_c0_failure_retained"] is True
    assert p3["measurement_controls_can_replace_original_fixture"] is False
    assert p3["accepted_candidate"] is False
    assert source_reference["solver_hunk_retained"] is False
    assert source_reference["solver_behavior_changed"] is False
    assert source_reference["production_patch_applied"] is False
    assert source_reference["sbp_sat_3d_repair_applied"] is False
    assert source_reference["api_preflight_changes_allowed"] is False
    assert source_reference["rfx_api_changes_allowed"] is False
    assert source_reference["public_claim_allowed"] is False
    assert source_reference["public_observable_promoted"] is False
    assert source_reference["true_rt_public_observable_promoted"] is False
    assert source_reference["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_source_reference_phase_front_fixture_contract_next_prerequisite"
        ]
        == source_reference["next_prerequisite"]
    )
    analytic_source = benchmark_gate[
        "private_analytic_source_phase_front_self_oracle"
    ]
    assert benchmark_gate[
        "private_analytic_source_phase_front_self_oracle_status"
    ] == "private_analytic_source_phase_front_self_oracle_blocked_no_public_promotion"
    assert analytic_source["terminal_outcome"] == (
        "private_analytic_source_phase_front_self_oracle_blocked_no_public_promotion"
    )
    assert analytic_source[
        "upstream_source_reference_phase_front_status"
    ] == benchmark_gate["private_source_reference_phase_front_fixture_contract_status"]
    assert analytic_source["candidate_ladder_declared_before_slow_scoring"] is True
    assert analytic_source["candidate_count"] == 6
    assert (
        analytic_source["thresholds_checksum"]
        == benchmark_gate["material_improvement_rule"]["thresholds_checksum"]
    )
    assert analytic_source["selected_candidate_id"] == (
        "A5_fail_closed_analytic_source_self_oracle_blocked"
    )
    assert analytic_source["source_self_oracle_separated_from_subgrid_parity"] is True
    assert analytic_source["subgrid_vacuum_parity_used_for_selection"] is False
    assert analytic_source["source_phase_front_self_oracle_ready"] is False
    assert analytic_source["source_phase_front_self_oracle_blocked"] is True
    assert analytic_source["private_fixture_contract_ready"] is False
    analytic_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in analytic_source["candidate_ladder"]
    }
    a1 = analytic_candidates["A1_temporal_phase_waveform_self_oracle"]
    assert a1["global_time_phase_rotation_invariant"] is True
    assert a1["changes_center_referenced_phase_spread"] is False
    assert a1["accepted_candidate"] is False
    a3 = analytic_candidates["A3_aperture_edge_taper_or_guard_contract"]
    assert a3["uses_existing_center_core_proxy"] is True
    assert a3["proxy_not_authoritative_source_self_oracle"] is True
    assert a3["metrics"]["transverse_phase_spread_deg"] > 1.0
    a4 = analytic_candidates["A4_uniform_reference_observable_contract"]
    assert a4["single_cell_or_center_only_mask_rejected"] is True
    assert a4["threshold_laundering_rejected"] is True
    assert a4["accepted_candidate"] is False
    assert (
        analytic_candidates["A5_fail_closed_analytic_source_self_oracle_blocked"][
            "accepted_candidate"
        ]
        is True
    )
    assert analytic_source["solver_hunk_retained"] is False
    assert analytic_source["solver_behavior_changed"] is False
    assert analytic_source["production_patch_applied"] is False
    assert analytic_source["sbp_sat_3d_repair_applied"] is False
    assert analytic_source["api_preflight_changes_allowed"] is False
    assert analytic_source["rfx_api_changes_allowed"] is False
    assert analytic_source["public_claim_allowed"] is False
    assert analytic_source["public_observable_promoted"] is False
    assert analytic_source["true_rt_public_observable_promoted"] is False
    assert analytic_source["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_analytic_source_phase_front_self_oracle_next_prerequisite"
        ]
        == analytic_source["next_prerequisite"]
    )
    plane_wave_source = benchmark_gate[
        "private_plane_wave_source_implementation_redesign"
    ]
    assert benchmark_gate[
        "private_plane_wave_source_implementation_redesign_status"
    ] == "private_uniform_plane_wave_source_self_oracle_ready"
    assert plane_wave_source["terminal_outcome"] == (
        "private_uniform_plane_wave_source_self_oracle_ready"
    )
    assert plane_wave_source[
        "upstream_analytic_source_phase_front_status"
    ] == benchmark_gate["private_analytic_source_phase_front_self_oracle_status"]
    assert plane_wave_source["candidate_ladder_declared_before_slow_scoring"] is True
    assert plane_wave_source["candidate_count"] == 5
    assert (
        plane_wave_source["thresholds_checksum"]
        == benchmark_gate["material_improvement_rule"]["thresholds_checksum"]
    )
    assert plane_wave_source["selected_candidate_id"] == (
        "W1_private_uniform_plane_wave_volume_source"
    )
    assert plane_wave_source["uniform_plane_wave_source_self_oracle_ready"] is True
    assert plane_wave_source["private_plane_wave_source_prototype_ready"] is True
    assert plane_wave_source["prototype_not_runtime_fixture_recovery"] is True
    assert plane_wave_source["private_fixture_contract_ready"] is False
    assert plane_wave_source["source_self_oracle_separated_from_subgrid_parity"] is True
    assert plane_wave_source["subgrid_vacuum_parity_used_for_selection"] is False
    wave_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in plane_wave_source["candidate_ladder"]
    }
    w1 = wave_candidates["W1_private_uniform_plane_wave_volume_source"]
    assert w1["accepted_candidate"] is True
    assert w1["prototype_only"] is True
    assert w1["runtime_public_surface_added"] is False
    assert w1["uses_public_tfsf_api"] is False
    assert w1["uses_public_flux_or_dft_monitor"] is False
    assert w1["metrics"]["max_uniform_center_referenced_phase_spread_deg"] <= 1.0
    assert w1["metrics"]["max_uniform_modal_magnitude_cv"] <= 0.01
    assert w1["admission_gate"]["passed"] is True
    assert wave_candidates["W2_private_huygens_pair_plane_source"][
        "deferred_after_w1_preacceptance"
    ] is True
    assert wave_candidates["W3_private_periodic_phase_front_fixture"][
        "periodic_boundary_public_claim_added"
    ] is False
    assert plane_wave_source["solver_hunk_retained"] is False
    assert plane_wave_source["solver_behavior_changed"] is False
    assert plane_wave_source["production_patch_applied"] is False
    assert plane_wave_source["sbp_sat_3d_repair_applied"] is False
    assert plane_wave_source["api_preflight_changes_allowed"] is False
    assert plane_wave_source["rfx_api_changes_allowed"] is False
    assert plane_wave_source["public_claim_allowed"] is False
    assert plane_wave_source["public_observable_promoted"] is False
    assert plane_wave_source["true_rt_public_observable_promoted"] is False
    assert plane_wave_source["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_source_implementation_redesign_next_prerequisite"
        ]
        == plane_wave_source["next_prerequisite"]
    )
    plane_wave_fixture = benchmark_gate["private_plane_wave_fixture_contract_recovery"]
    assert benchmark_gate["private_plane_wave_fixture_contract_recovery_status"] == (
        "private_uniform_plane_wave_reference_contract_ready"
    )
    assert plane_wave_fixture["terminal_outcome"] == (
        "private_uniform_plane_wave_reference_contract_ready"
    )
    assert plane_wave_fixture[
        "upstream_plane_wave_source_status"
    ] == benchmark_gate["private_plane_wave_source_implementation_redesign_status"]
    assert plane_wave_fixture["candidate_ladder_declared_before_slow_scoring"] is True
    assert plane_wave_fixture["candidate_count"] == 4
    assert (
        plane_wave_fixture["thresholds_checksum"]
        == benchmark_gate["material_improvement_rule"]["thresholds_checksum"]
    )
    assert plane_wave_fixture["selected_candidate_id"] == (
        "R1_uniform_reference_plane_wave_fixture_contract"
    )
    assert plane_wave_fixture["uniform_reference_plane_wave_contract_ready"] is True
    assert plane_wave_fixture["subgrid_vacuum_plane_wave_contract_ready"] is False
    assert plane_wave_fixture["fixture_quality_ready"] is False
    assert plane_wave_fixture["reference_quality_ready"] is True
    assert plane_wave_fixture["true_rt_readiness_unlocked"] is False
    assert plane_wave_fixture["plane_wave_self_oracle_visible"] is True
    assert (
        plane_wave_fixture[
            "plane_wave_self_oracle_distinct_from_fixture_recovery"
        ]
        is True
    )
    assert plane_wave_fixture["source_self_oracle_separated_from_subgrid_parity"] is True
    assert plane_wave_fixture["subgrid_vacuum_parity_used_for_r1_selection"] is False
    recovery_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in plane_wave_fixture["candidate_ladder"]
    }
    r1 = recovery_candidates["R1_uniform_reference_plane_wave_fixture_contract"]
    assert r1["accepted_candidate"] is True
    assert r1["source_phase_front_gate_passed"] is True
    assert r1["normalization_gate_passed"] is True
    assert r1["uniform_reference_only"] is True
    assert r1["subgrid_vacuum_parity_scored"] is False
    assert r1["metrics"]["max_uniform_center_referenced_phase_spread_deg"] <= 1.0
    assert r1["metrics"]["max_local_vacuum_relative_magnitude_error"] <= 0.02
    r2 = recovery_candidates["R2_subgrid_vacuum_plane_wave_fixture_contract"]
    assert r2["accepted_candidate"] is False
    assert r2["source_self_oracle_ready"] is True
    assert r2["subgrid_vacuum_parity_scored"] is False
    assert r2["true_rt_readiness_unlocked"] is False
    assert plane_wave_fixture["solver_hunk_retained"] is False
    assert plane_wave_fixture["solver_behavior_changed"] is False
    assert plane_wave_fixture["production_patch_applied"] is False
    assert plane_wave_fixture["sbp_sat_3d_repair_applied"] is False
    assert plane_wave_fixture["api_preflight_changes_allowed"] is False
    assert plane_wave_fixture["rfx_api_changes_allowed"] is False
    assert plane_wave_fixture["public_claim_allowed"] is False
    assert plane_wave_fixture["public_observable_promoted"] is False
    assert plane_wave_fixture["true_rt_public_observable_promoted"] is False
    assert plane_wave_fixture["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate["private_plane_wave_fixture_contract_recovery_next_prerequisite"]
        == plane_wave_fixture["next_prerequisite"]
    )
    subgrid_vacuum_fixture = benchmark_gate[
        "private_subgrid_vacuum_plane_wave_fixture_contract"
    ]
    assert benchmark_gate[
        "private_subgrid_vacuum_plane_wave_fixture_contract_status"
    ] == "private_plane_wave_subgrid_vacuum_fixture_blocked_no_public_promotion"
    assert subgrid_vacuum_fixture["terminal_outcome"] == (
        "private_plane_wave_subgrid_vacuum_fixture_blocked_no_public_promotion"
    )
    assert subgrid_vacuum_fixture[
        "upstream_plane_wave_fixture_status"
    ] == benchmark_gate["private_plane_wave_fixture_contract_recovery_status"]
    assert (
        subgrid_vacuum_fixture["candidate_ladder_declared_before_slow_scoring"]
        is True
    )
    assert subgrid_vacuum_fixture["candidate_count"] == 3
    assert (
        subgrid_vacuum_fixture["thresholds_checksum"]
        == benchmark_gate["material_improvement_rule"]["thresholds_checksum"]
    )
    assert subgrid_vacuum_fixture["selected_candidate_id"] == (
        "V2_subgrid_plane_wave_fixture_blocker_classified"
    )
    assert subgrid_vacuum_fixture["plane_wave_source_self_oracle_ready"] is True
    assert subgrid_vacuum_fixture["same_contract_reference_ready"] is True
    assert subgrid_vacuum_fixture["uniform_reference_plane_wave_contract_ready"] is True
    assert subgrid_vacuum_fixture["plane_wave_fixture_path_wired"] is False
    assert subgrid_vacuum_fixture["subgrid_vacuum_parity_scored"] is False
    assert subgrid_vacuum_fixture["fixture_quality_ready"] is False
    assert subgrid_vacuum_fixture["reference_quality_ready"] is True
    assert subgrid_vacuum_fixture["true_rt_readiness_unlocked"] is False
    subgrid_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in subgrid_vacuum_fixture["candidate_ladder"]
    }
    v1 = subgrid_candidates["V1_private_subgrid_plane_wave_vacuum_parity_probe"]
    assert v1["source_self_oracle_ready"] is True
    assert v1["same_contract_reference_ready"] is True
    assert v1["subgrid_vacuum_parity_scored"] is False
    assert v1["admission_gate"]["passed"] is False
    v2 = subgrid_candidates["V2_subgrid_plane_wave_fixture_blocker_classified"]
    assert v2["accepted_candidate"] is True
    assert v2["fixture_quality_ready"] is False
    assert subgrid_vacuum_fixture["public_claim_allowed"] is False
    assert subgrid_vacuum_fixture["public_observable_promoted"] is False
    assert subgrid_vacuum_fixture["true_rt_public_observable_promoted"] is False
    assert (
        subgrid_vacuum_fixture["dft_flux_tfsf_port_sparameter_promoted"] is False
    )
    assert (
        benchmark_gate[
            "private_subgrid_vacuum_plane_wave_fixture_contract_next_prerequisite"
        ]
        == subgrid_vacuum_fixture["next_prerequisite"]
    )
    plane_wave_wiring = benchmark_gate[
        "private_plane_wave_source_fixture_path_wiring"
    ]
    assert benchmark_gate[
        "private_plane_wave_source_fixture_path_wiring_status"
    ] == "private_plane_wave_fixture_path_wiring_blocked_no_public_promotion"
    assert plane_wave_wiring["terminal_outcome"] == (
        "private_plane_wave_fixture_path_wiring_blocked_no_public_promotion"
    )
    assert plane_wave_wiring[
        "upstream_subgrid_vacuum_fixture_status"
    ] == benchmark_gate["private_subgrid_vacuum_plane_wave_fixture_contract_status"]
    assert plane_wave_wiring["candidate_ladder_declared_before_slow_scoring"] is True
    assert plane_wave_wiring["candidate_count"] == 4
    assert (
        plane_wave_wiring["thresholds_checksum"]
        == benchmark_gate["material_improvement_rule"]["thresholds_checksum"]
    )
    assert plane_wave_wiring["selected_candidate_id"] == (
        "WIRE3_fixture_path_wiring_blocker_classified"
    )
    assert plane_wave_wiring["source_self_oracle_ready"] is True
    assert plane_wave_wiring["same_contract_reference_ready"] is True
    assert plane_wave_wiring["plane_wave_fixture_path_wired"] is False
    assert plane_wave_wiring["adapter_implementation_surface_available"] is False
    assert plane_wave_wiring["subgrid_vacuum_parity_scored"] is False
    assert plane_wave_wiring["fixture_quality_ready"] is False
    assert plane_wave_wiring["reference_quality_ready"] is True
    assert plane_wave_wiring["true_rt_readiness_unlocked"] is False
    wire_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in plane_wave_wiring["candidate_ladder"]
    }
    wire1 = wire_candidates["WIRE1_private_plane_wave_source_fixture_path_adapter"]
    assert wire1["accepted_candidate"] is False
    assert wire1["w1_contract_runtime_represented"] is False
    assert wire1["existing_private_tfsf_hook_reusable_as_w1"] is False
    assert wire1["rfx_runners_change_allowed_this_lane"] is False
    wire2 = wire_candidates["WIRE2_private_subgrid_vacuum_parity_score"]
    assert wire2["subgrid_vacuum_parity_scored"] is False
    assert wire2["admission_gate"]["passed"] is False
    wire3 = wire_candidates["WIRE3_fixture_path_wiring_blocker_classified"]
    assert wire3["accepted_candidate"] is True
    assert plane_wave_wiring["public_claim_allowed"] is False
    assert plane_wave_wiring["public_observable_promoted"] is False
    assert plane_wave_wiring["true_rt_public_observable_promoted"] is False
    assert (
        plane_wave_wiring["dft_flux_tfsf_port_sparameter_promoted"] is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_source_fixture_path_wiring_next_prerequisite"
        ]
        == plane_wave_wiring["next_prerequisite"]
    )
    adapter_design = benchmark_gate["private_plane_wave_source_adapter_design"]
    assert benchmark_gate["private_plane_wave_source_adapter_design_status"] == (
        "private_runner_plane_wave_adapter_design_ready"
    )
    assert adapter_design["terminal_outcome"] == (
        "private_runner_plane_wave_adapter_design_ready"
    )
    assert adapter_design[
        "upstream_fixture_path_wiring_status"
    ] == benchmark_gate["private_plane_wave_source_fixture_path_wiring_status"]
    assert adapter_design["candidate_ladder_declared_before_implementation"] is True
    assert adapter_design["candidate_ladder_declared_before_slow_scoring"] is True
    assert adapter_design["candidate_count"] == 4
    assert (
        adapter_design["thresholds_checksum"]
        == benchmark_gate["material_improvement_rule"]["thresholds_checksum"]
    )
    assert adapter_design["selected_candidate_id"] == (
        "AD2_private_runner_request_spec_adapter_design"
    )
    assert adapter_design["design_ready"] is True
    assert adapter_design["selected_design_requires_implementation"] is True
    assert adapter_design["adapter_implementation_ready"] is False
    assert adapter_design["subgrid_vacuum_parity_scored"] is False
    assert adapter_design["fixture_quality_ready"] is False
    assert "rfx/runners/subgridded.py" in adapter_design["allowed_write_surface"]
    assert "rfx/subgridding/jit_runner.py" in adapter_design["allowed_write_surface"]
    assert "rfx/api.py" in adapter_design["forbidden_public_surfaces"]
    ad_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in adapter_design["candidate_ladder"]
    }
    assert ad_candidates["AD1_jit_runner_internal_plane_wave_spec_design"][
        "accepted_candidate"
    ] is False
    ad2 = ad_candidates["AD2_private_runner_request_spec_adapter_design"]
    assert ad2["accepted_candidate"] is True
    assert ad2["reuses_existing_simulation_lowering"] is True
    assert ad2["uses_private_request_object"] is True
    assert ad2["uses_private_jit_spec"] is True
    assert ad2["public_simulation_api_changed"] is False
    assert ad2["public_result_surface_changed"] is False
    assert ad2["implementation_intent"]["private_request"] == (
        "_PrivatePlaneWaveSourceRequest"
    )
    assert adapter_design["public_claim_allowed"] is False
    assert adapter_design["public_observable_promoted"] is False
    assert adapter_design["true_rt_public_observable_promoted"] is False
    assert adapter_design["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate["private_plane_wave_source_adapter_design_next_prerequisite"]
        == adapter_design["next_prerequisite"]
    )
    adapter_implementation = benchmark_gate[
        "private_plane_wave_source_adapter_implementation"
    ]
    assert (
        benchmark_gate["private_plane_wave_source_adapter_implementation_status"]
        == "private_plane_wave_adapter_implemented_parity_pending"
    )
    assert adapter_implementation["terminal_outcome"] == (
        "private_plane_wave_adapter_implemented_parity_pending"
    )
    assert adapter_implementation["upstream_adapter_design_status"] == (
        benchmark_gate["private_plane_wave_source_adapter_design_status"]
    )
    assert (
        adapter_implementation["candidate_ladder_declared_before_implementation"]
        is True
    )
    assert adapter_implementation["candidate_count"] == 4
    assert adapter_implementation["selected_candidate_id"] == (
        "IMPL2_private_plane_wave_jit_spec_and_injection"
    )
    assert adapter_implementation["request_builder_ready"] is True
    assert adapter_implementation["adapter_implementation_ready"] is True
    assert adapter_implementation["plane_wave_fixture_path_wired"] is True
    assert adapter_implementation["w1_contract_runtime_represented"] is True
    assert adapter_implementation["subgrid_vacuum_parity_scored"] is False
    assert adapter_implementation["fixture_quality_ready"] is False
    assert adapter_implementation["public_claim_allowed"] is False
    assert adapter_implementation["public_observable_promoted"] is False
    impl_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in adapter_implementation["candidate_ladder"]
    }
    assert impl_candidates["IMPL2_private_plane_wave_jit_spec_and_injection"][
        "private_spec"
    ] == "_PrivatePlaneWaveSourceSpec"
    assert (
        benchmark_gate[
            "private_plane_wave_source_adapter_implementation_next_prerequisite"
        ]
        == adapter_implementation["next_prerequisite"]
    )
    parity_scoring = benchmark_gate[
        "private_subgrid_vacuum_plane_wave_parity_scoring"
    ]
    assert (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
        == "private_subgrid_vacuum_plane_wave_parity_failed_no_public_promotion"
    )
    assert parity_scoring["terminal_outcome"] == (
        "private_subgrid_vacuum_plane_wave_parity_failed_no_public_promotion"
    )
    assert parity_scoring["upstream_adapter_implementation_status"] == (
        benchmark_gate["private_plane_wave_source_adapter_implementation_status"]
    )
    assert parity_scoring["candidate_ladder_declared_before_slow_scoring"] is True
    assert parity_scoring["candidate_count"] == 3
    assert parity_scoring["selected_candidate_id"] == (
        "P1_private_subgrid_vacuum_plane_wave_parity_score"
    )
    assert parity_scoring["uses_private_plane_wave_request"] is True
    assert parity_scoring["uses_private_plane_wave_spec"] is True
    assert parity_scoring["existing_private_tfsf_hook_reused_as_w1"] is False
    assert parity_scoring["same_contract_reference_ready"] is True
    assert parity_scoring["plane_wave_fixture_path_wired"] is True
    assert parity_scoring["subgrid_vacuum_parity_scored"] is True
    assert parity_scoring["subgrid_vacuum_parity_passed"] is False
    assert parity_scoring["fixture_quality_ready"] is False
    assert parity_scoring["true_rt_readiness_unlocked"] is False
    assert parity_scoring["dominant_parity_blocker"] == (
        "transverse_phase_spread_deg"
    )
    assert parity_scoring["metrics"]["transverse_phase_spread_deg"] > 1.0
    assert parity_scoring["metrics"]["vacuum_relative_magnitude_error"] > 0.02
    assert parity_scoring["public_claim_allowed"] is False
    assert parity_scoring["public_observable_promoted"] is False
    parity_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in parity_scoring["candidate_ladder"]
    }
    assert parity_candidates["P1_private_subgrid_vacuum_plane_wave_parity_score"][
        "private_request"
    ] == "_PrivatePlaneWaveSourceRequest"
    assert (
        benchmark_gate[
            "private_subgrid_vacuum_plane_wave_parity_scoring_next_prerequisite"
        ]
        == parity_scoring["next_prerequisite"]
    )
    parity_repair = benchmark_gate[
        "private_plane_wave_parity_blocker_repair_design"
    ]
    assert benchmark_gate[
        "private_plane_wave_parity_blocker_repair_design_status"
    ] == "private_plane_wave_interface_floor_repair_design_required"
    assert parity_repair["terminal_outcome"] == (
        "private_plane_wave_interface_floor_repair_design_required"
    )
    assert parity_repair["upstream_parity_scoring_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert parity_repair["candidate_ladder_declared_before_slow_scoring"] is True
    assert parity_repair["candidate_count"] == 5
    assert parity_repair["selected_candidate_id"] == (
        "B3_interface_floor_repair_reentry_design"
    )
    assert parity_repair["baseline_metrics"] == parity_scoring["metrics"]
    assert parity_repair["baseline_metrics_preserved"] is True
    assert parity_repair["thresholds_unchanged"] is True
    assert parity_repair["dominant_parity_blocker"] == (
        parity_scoring["dominant_parity_blocker"]
    )
    assert parity_repair["phase_front_repair_candidate_ready"] is False
    assert parity_repair["measurement_contract_repair_candidate_ready"] is False
    assert parity_repair["interface_floor_repair_design_required"] is True
    assert parity_repair["production_scope_required"] is True
    assert parity_repair["no_production_patch_in_this_lane"] is True
    assert parity_repair["true_rt_readiness_unlocked"] is False
    assert parity_repair["public_claim_allowed"] is False
    repair_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in parity_repair["candidate_ladder"]
    }
    assert (
        repair_candidates["B3_interface_floor_repair_reentry_design"][
            "accepted_candidate"
        ]
        is True
    )
    assert (
        benchmark_gate[
            "private_plane_wave_parity_blocker_repair_design_next_prerequisite"
        ]
        == parity_repair["next_prerequisite"]
    )
    interface_impl = benchmark_gate[
        "private_plane_wave_interface_floor_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_interface_floor_implementation_status"
    ] == "no_private_plane_wave_interface_floor_repair"
    assert interface_impl["terminal_outcome"] == (
        "no_private_plane_wave_interface_floor_repair"
    )
    assert interface_impl["upstream_repair_design_status"] == (
        benchmark_gate["private_plane_wave_parity_blocker_repair_design_status"]
    )
    assert interface_impl["upstream_parity_scoring_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert interface_impl["candidate_ladder_declared_before_slow_scoring"] is True
    assert interface_impl["candidate_count"] == 4
    assert interface_impl["selected_candidate_id"] == (
        "I3_no_bounded_interface_floor_implementation"
    )
    assert interface_impl["baseline_metrics"] == parity_repair["baseline_metrics"]
    assert interface_impl["accepted_private_repair"] is False
    assert interface_impl["material_improvement_demonstrated"] is False
    assert interface_impl["production_scope_was_opened"] is True
    assert interface_impl["production_patch_applied"] is False
    assert interface_impl["requires_architecture_root_cause_redesign"] is True
    assert interface_impl["true_rt_readiness_unlocked"] is False
    assert interface_impl["public_claim_allowed"] is False
    implementation_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in interface_impl["candidate_ladder"]
    }
    assert (
        implementation_candidates["I3_no_bounded_interface_floor_implementation"][
            "accepted_candidate"
        ]
        is True
    )
    assert (
        benchmark_gate[
            "private_plane_wave_interface_floor_implementation_next_prerequisite"
        ]
        == interface_impl["next_prerequisite"]
    )
    root_cause = benchmark_gate["private_plane_wave_root_cause_redesign"]
    assert benchmark_gate["private_plane_wave_root_cause_redesign_status"] == (
        "private_plane_wave_interface_energy_form_root_cause_identified"
    )
    assert root_cause["terminal_outcome"] == (
        "private_plane_wave_interface_energy_form_root_cause_identified"
    )
    assert root_cause["upstream_implementation_status"] == (
        benchmark_gate["private_plane_wave_interface_floor_implementation_status"]
    )
    assert root_cause["upstream_parity_scoring_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert root_cause["candidate_ladder_declared_before_slow_scoring"] is True
    assert root_cause["candidate_count"] == 6
    assert root_cause["selected_candidate_id"] == (
        "R2_interface_sat_energy_form_root_cause"
    )
    assert root_cause["baseline_metrics"] == interface_impl["baseline_metrics"]
    assert root_cause["root_cause_family"] == "interface_sat_energy_form"
    assert root_cause["root_cause_identified"] is True
    assert root_cause["source_stagger_root_cause_identified"] is False
    assert root_cause["mortar_metric_root_cause_identified"] is False
    assert root_cause["fixture_geometry_root_cause_identified"] is False
    assert root_cause["next_lane_requires_design_before_solver_edit"] is True
    assert root_cause["production_patch_applied"] is False
    assert root_cause["true_rt_readiness_unlocked"] is False
    assert root_cause["public_claim_allowed"] is False
    root_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in root_cause["candidate_ladder"]
    }
    assert (
        root_candidates["R2_interface_sat_energy_form_root_cause"][
            "accepted_candidate"
        ]
        is True
    )
    assert (
        root_candidates["R4_benchmark_fixture_geometry_root_cause"][
            "threshold_laundering_rejected"
        ]
        is True
    )
    assert (
        benchmark_gate["private_plane_wave_root_cause_redesign_next_prerequisite"]
        == root_cause["next_prerequisite"]
    )
    energy_form_design = benchmark_gate[
        "private_plane_wave_interface_energy_form_design"
    ]
    assert benchmark_gate[
        "private_plane_wave_interface_energy_form_design_status"
    ] == "private_plane_wave_interface_energy_form_implementation_contract_ready"
    assert energy_form_design["terminal_outcome"] == (
        "private_plane_wave_interface_energy_form_implementation_contract_ready"
    )
    assert energy_form_design["upstream_root_cause_status"] == (
        benchmark_gate["private_plane_wave_root_cause_redesign_status"]
    )
    assert energy_form_design["upstream_implementation_status"] == (
        benchmark_gate["private_plane_wave_interface_floor_implementation_status"]
    )
    assert energy_form_design["candidate_ladder_declared_before_slow_scoring"] is True
    assert energy_form_design["candidate_count"] == 5
    assert energy_form_design["selected_candidate_id"] == (
        "E3_boundary_coexistence_preserving_implementation_contract"
    )
    assert energy_form_design["baseline_metrics"] == root_cause["baseline_metrics"]
    assert energy_form_design["energy_potential_design_ready"] is True
    assert energy_form_design["time_centered_work_form_design_ready"] is True
    assert energy_form_design["implementation_contract_ready"] is True
    assert energy_form_design["next_lane_requires_implementation_plan"] is True
    assert energy_form_design["production_patch_applied"] is False
    assert energy_form_design["true_rt_readiness_unlocked"] is False
    assert "rfx/subgridding/sbp_sat_3d.py" in energy_form_design[
        "future_write_surface"
    ]
    assert energy_form_design["implementation_gates"][
        "forbidden_public_surface_diff_required"
    ] is True
    energy_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in energy_form_design["candidate_ladder"]
    }
    assert (
        energy_candidates["E1_discrete_interface_energy_potential_design"][
            "design_component_ready"
        ]
        is True
    )
    assert (
        energy_candidates["E2_time_centered_poynting_work_form_design"][
            "design_component_ready"
        ]
        is True
    )
    assert (
        energy_candidates[
            "E3_boundary_coexistence_preserving_implementation_contract"
        ]["accepted_candidate"]
        is True
    )
    assert energy_form_design["public_claim_allowed"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_interface_energy_form_design_next_prerequisite"
        ]
        == energy_form_design["next_prerequisite"]
    )
    energy_form_implementation = benchmark_gate[
        "private_plane_wave_interface_energy_form_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_interface_energy_form_implementation_status"
    ] == "no_private_plane_wave_energy_form_implementation"
    assert energy_form_implementation["terminal_outcome"] == (
        "no_private_plane_wave_energy_form_implementation"
    )
    assert energy_form_implementation["upstream_design_status"] == (
        benchmark_gate["private_plane_wave_interface_energy_form_design_status"]
    )
    assert energy_form_implementation["upstream_root_cause_status"] == (
        benchmark_gate["private_plane_wave_root_cause_redesign_status"]
    )
    assert energy_form_implementation["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert (
        energy_form_implementation["candidate_ladder_declared_before_slow_scoring"]
        is True
    )
    assert energy_form_implementation["candidate_count"] == 5
    assert energy_form_implementation["selected_candidate_id"] == (
        "F3_energy_form_implementation_blocked"
    )
    assert (
        energy_form_implementation["baseline_metrics"]
        == energy_form_design["baseline_metrics"]
    )
    assert energy_form_implementation["dominant_metric"] == (
        "transverse_phase_spread_deg"
    )
    assert energy_form_implementation["dominant_metric_improved"] is False
    assert energy_form_implementation["implementation_lane_executed"] is True
    assert energy_form_implementation["production_patch_applied"] is False
    assert energy_form_implementation["solver_behavior_changed"] is False
    assert energy_form_implementation["no_bounded_hunk_accepted"] is True
    assert energy_form_implementation["true_rt_readiness_unlocked"] is False
    energy_form_impl_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in energy_form_implementation["candidate_ladder"]
    }
    assert (
        energy_form_impl_candidates[
            "F1_centered_interface_energy_potential_update"
        ]["accepted_candidate"]
        is False
    )
    assert (
        energy_form_impl_candidates[
            "F2_operator_backed_centered_work_helper"
        ]["accepted_candidate"]
        is False
    )
    assert (
        energy_form_impl_candidates[
            "F3_energy_form_implementation_blocked"
        ]["accepted_candidate"]
        is True
    )
    assert energy_form_implementation["public_claim_allowed"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_interface_energy_form_implementation_next_prerequisite"
        ]
        == energy_form_implementation["next_prerequisite"]
    )
    failure_theory = benchmark_gate[
        "private_plane_wave_interface_energy_form_failure_theory"
    ]
    assert benchmark_gate[
        "private_plane_wave_interface_energy_form_failure_theory_status"
    ] == "private_plane_wave_energy_form_redesign_contract_ready"
    assert failure_theory["terminal_outcome"] == (
        "private_plane_wave_energy_form_redesign_contract_ready"
    )
    assert failure_theory["upstream_implementation_status"] == (
        benchmark_gate[
            "private_plane_wave_interface_energy_form_implementation_status"
        ]
    )
    assert failure_theory["upstream_design_status"] == (
        benchmark_gate["private_plane_wave_interface_energy_form_design_status"]
    )
    assert failure_theory["upstream_root_cause_status"] == (
        benchmark_gate["private_plane_wave_root_cause_redesign_status"]
    )
    assert failure_theory["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert failure_theory["candidate_ladder_declared_before_slow_scoring"] is True
    assert failure_theory["candidate_count"] == 5
    assert failure_theory["selected_candidate_id"] == (
        "G3_operator_mortar_time_centering_redesign_contract"
    )
    assert (
        failure_theory["baseline_metrics"]
        == energy_form_implementation["baseline_metrics"]
    )
    assert failure_theory["energy_state_design_ready"] is True
    assert failure_theory["source_interface_coupling_design_ready"] is True
    assert failure_theory["operator_mortar_time_centering_design_ready"] is True
    assert failure_theory["redesign_contract_ready"] is True
    assert failure_theory["explains_local_hunk_failure"] is True
    assert failure_theory["bounded_follow_up_implementation_surface"] is True
    assert failure_theory["implementation_prerequisites"][
        "bind_time_centered_eh_state_before_local_sat_helper_stack"
    ] is True
    assert failure_theory["production_patch_applied"] is False
    assert failure_theory["solver_behavior_changed"] is False
    assert failure_theory["new_solver_hunk_retained"] is False
    assert failure_theory["true_rt_readiness_unlocked"] is False
    failure_theory_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in failure_theory["candidate_ladder"]
    }
    assert (
        failure_theory_candidates[
            "G1_missing_interface_energy_state_invariant"
        ]["design_component_ready"]
        is True
    )
    assert (
        failure_theory_candidates[
            "G2_source_interface_coupled_energy_balance_design"
        ]["design_component_ready"]
        is True
    )
    assert (
        failure_theory_candidates[
            "G3_operator_mortar_time_centering_redesign_contract"
        ]["accepted_candidate"]
        is True
    )
    assert failure_theory["public_claim_allowed"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_interface_energy_form_failure_theory_next_prerequisite"
        ]
        == failure_theory["next_prerequisite"]
    )
    operator_mortar_impl = benchmark_gate[
        "private_plane_wave_operator_mortar_energy_form_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_operator_mortar_energy_form_implementation_status"
    ] == "no_private_plane_wave_operator_mortar_energy_form_implementation"
    assert operator_mortar_impl["terminal_outcome"] == (
        "no_private_plane_wave_operator_mortar_energy_form_implementation"
    )
    assert operator_mortar_impl["upstream_failure_theory_status"] == (
        benchmark_gate["private_plane_wave_interface_energy_form_failure_theory_status"]
    )
    assert operator_mortar_impl["upstream_implementation_status"] == (
        benchmark_gate[
            "private_plane_wave_interface_energy_form_implementation_status"
        ]
    )
    assert operator_mortar_impl["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert operator_mortar_impl["candidate_ladder_declared_before_slow_scoring"] is True
    assert operator_mortar_impl["candidate_count"] == 5
    assert operator_mortar_impl["selected_candidate_id"] == (
        "H3_operator_mortar_implementation_blocked"
    )
    assert operator_mortar_impl["baseline_metrics"] == failure_theory["baseline_metrics"]
    assert operator_mortar_impl["dominant_metric"] == "transverse_phase_spread_deg"
    assert operator_mortar_impl["dominant_metric_improved"] is False
    assert operator_mortar_impl["implementation_lane_executed"] is True
    assert operator_mortar_impl["operator_mortar_contract_ready"] is True
    assert operator_mortar_impl["operator_mortar_energy_state_hunk_retained"] is False
    assert operator_mortar_impl["time_centered_operator_mortar_hunk_retained"] is False
    assert operator_mortar_impl["production_patch_applied"] is False
    assert operator_mortar_impl["solver_behavior_changed"] is False
    assert operator_mortar_impl["new_solver_hunk_retained"] is False
    assert operator_mortar_impl["no_bounded_hunk_accepted"] is True
    assert operator_mortar_impl["true_rt_readiness_unlocked"] is False
    operator_mortar_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in operator_mortar_impl["candidate_ladder"]
    }
    assert (
        operator_mortar_candidates[
            "H1_operator_owned_interface_energy_state_helper"
        ]["accepted_candidate"]
        is False
    )
    assert (
        operator_mortar_candidates[
            "H2_time_centered_eh_projection_before_helper_stack"
        ]["accepted_candidate"]
        is False
    )
    assert (
        operator_mortar_candidates[
            "H3_operator_mortar_implementation_blocked"
        ]["accepted_candidate"]
        is True
    )
    assert operator_mortar_impl["public_claim_allowed"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_operator_mortar_energy_form_implementation_next_prerequisite"
        ]
        == operator_mortar_impl["next_prerequisite"]
    )
    phase_architecture = benchmark_gate[
        "private_plane_wave_transverse_phase_coherence_architecture"
    ]
    assert benchmark_gate[
        "private_plane_wave_transverse_phase_coherence_architecture_status"
    ] == "private_plane_wave_phase_coherence_staging_contract_ready"
    assert phase_architecture["terminal_outcome"] == (
        "private_plane_wave_phase_coherence_staging_contract_ready"
    )
    assert phase_architecture["upstream_operator_mortar_status"] == (
        benchmark_gate[
            "private_plane_wave_operator_mortar_energy_form_implementation_status"
        ]
    )
    assert phase_architecture["upstream_failure_theory_status"] == (
        benchmark_gate["private_plane_wave_interface_energy_form_failure_theory_status"]
    )
    assert phase_architecture["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert phase_architecture["candidate_ladder_declared_before_slow_scoring"] is True
    assert phase_architecture["candidate_count"] == 5
    assert phase_architecture["selected_candidate_id"] == (
        "J3_combined_phase_coherence_staging_contract"
    )
    assert phase_architecture["baseline_metrics"] == operator_mortar_impl[
        "baseline_metrics"
    ]
    assert phase_architecture["interface_state_ownership_design_ready"] is True
    assert phase_architecture["transverse_phase_coherence_design_ready"] is True
    assert phase_architecture["phase_coherence_staging_contract_ready"] is True
    assert phase_architecture["explains_operator_mortar_blocker"] is True
    assert phase_architecture["bounded_follow_up_implementation_surface"] is True
    assert phase_architecture["staging_contract"][
        "single_interface_state_owner_required"
    ] is True
    assert phase_architecture["staging_contract"]["no_threshold_laundering"] is True
    assert phase_architecture["production_patch_applied"] is False
    assert phase_architecture["solver_behavior_changed"] is False
    assert phase_architecture["new_solver_hunk_retained"] is False
    assert phase_architecture["true_rt_readiness_unlocked"] is False
    phase_architecture_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in phase_architecture["candidate_ladder"]
    }
    assert (
        phase_architecture_candidates[
            "J1_interface_energy_state_ownership_design"
        ]["design_component_ready"]
        is True
    )
    assert (
        phase_architecture_candidates[
            "J2_transverse_phase_coherence_score_coupling_design"
        ]["threshold_laundering_rejected"]
        is True
    )
    assert (
        phase_architecture_candidates[
            "J3_combined_phase_coherence_staging_contract"
        ]["accepted_candidate"]
        is True
    )
    assert phase_architecture["public_claim_allowed"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_transverse_phase_coherence_architecture_next_prerequisite"
        ]
        == phase_architecture["next_prerequisite"]
    )
    phase_staging_impl = benchmark_gate[
        "private_plane_wave_phase_coherence_staging_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_phase_coherence_staging_implementation_status"
    ] == "no_private_plane_wave_phase_coherence_staging_implementation"
    assert phase_staging_impl["terminal_outcome"] == (
        "no_private_plane_wave_phase_coherence_staging_implementation"
    )
    assert phase_staging_impl["upstream_phase_architecture_status"] == (
        benchmark_gate[
            "private_plane_wave_transverse_phase_coherence_architecture_status"
        ]
    )
    assert phase_staging_impl["upstream_operator_mortar_status"] == (
        benchmark_gate[
            "private_plane_wave_operator_mortar_energy_form_implementation_status"
        ]
    )
    assert phase_staging_impl["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert phase_staging_impl["candidate_ladder_declared_before_slow_scoring"] is True
    assert phase_staging_impl["candidate_count"] == 5
    assert phase_staging_impl["selected_candidate_id"] == (
        "K3_phase_coherence_staging_implementation_blocked"
    )
    assert phase_staging_impl["baseline_metrics"] == phase_architecture[
        "baseline_metrics"
    ]
    assert phase_staging_impl["baseline_metrics_preserved"] is True
    assert phase_staging_impl["thresholds_unchanged"] is True
    assert phase_staging_impl["dominant_metric"] == "transverse_phase_spread_deg"
    assert phase_staging_impl["paired_metric"] == "transverse_magnitude_cv"
    assert phase_staging_impl["joint_phase_magnitude_improved"] is False
    assert phase_staging_impl["phase_coherence_contract_ready"] is True
    assert phase_staging_impl["implementation_lane_executed"] is True
    assert phase_staging_impl["phase_coherence_state_owner_hunk_retained"] is False
    assert phase_staging_impl["phase_coherence_joint_score_hunk_retained"] is False
    assert phase_staging_impl["production_patch_applied"] is False
    assert phase_staging_impl["solver_behavior_changed"] is False
    assert phase_staging_impl["new_solver_hunk_retained"] is False
    assert phase_staging_impl["no_bounded_hunk_accepted"] is True
    assert phase_staging_impl["true_rt_readiness_unlocked"] is False
    assert (
        phase_staging_impl["next_lane_requires_solver_state_architecture_redesign"]
        is True
    )
    phase_staging_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in phase_staging_impl["candidate_ladder"]
    }
    assert (
        phase_staging_candidates[
            "K1_private_interface_state_owner_solver_staging"
        ]["accepted_candidate"]
        is False
    )
    assert (
        phase_staging_candidates[
            "K2_phase_coherence_joint_score_guard"
        ]["accepted_candidate"]
        is False
    )
    assert (
        phase_staging_candidates[
            "K3_phase_coherence_staging_implementation_blocked"
        ]["accepted_candidate"]
        is True
    )
    assert phase_staging_impl["public_claim_allowed"] is False
    assert phase_staging_impl["public_observable_promoted"] is False
    assert phase_staging_impl["true_rt_public_observable_promoted"] is False
    assert phase_staging_impl["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_phase_coherence_staging_implementation_next_prerequisite"
        ]
        == phase_staging_impl["next_prerequisite"]
    )
    solver_wide_owner_architecture = benchmark_gate[
        "private_plane_wave_solver_wide_interface_state_owner_architecture"
    ]
    assert benchmark_gate[
        "private_plane_wave_solver_wide_interface_state_owner_architecture_status"
    ] == "private_plane_wave_solver_wide_interface_state_owner_contract_ready"
    assert solver_wide_owner_architecture["terminal_outcome"] == (
        "private_plane_wave_solver_wide_interface_state_owner_contract_ready"
    )
    assert solver_wide_owner_architecture["upstream_phase_staging_status"] == (
        benchmark_gate[
            "private_plane_wave_phase_coherence_staging_implementation_status"
        ]
    )
    assert solver_wide_owner_architecture["upstream_phase_architecture_status"] == (
        benchmark_gate[
            "private_plane_wave_transverse_phase_coherence_architecture_status"
        ]
    )
    assert solver_wide_owner_architecture["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert (
        solver_wide_owner_architecture[
            "candidate_ladder_declared_before_slow_scoring"
        ]
        is True
    )
    assert solver_wide_owner_architecture["candidate_count"] == 5
    assert solver_wide_owner_architecture["selected_candidate_id"] == (
        "L3_combined_interface_state_owner_implementation_contract"
    )
    assert solver_wide_owner_architecture["baseline_metrics"] == (
        phase_staging_impl["baseline_metrics"]
    )
    assert solver_wide_owner_architecture["baseline_metrics_preserved"] is True
    assert solver_wide_owner_architecture["thresholds_unchanged"] is True
    assert (
        solver_wide_owner_architecture[
            "solver_wide_interface_state_owner_design_ready"
        ]
        is True
    )
    assert (
        solver_wide_owner_architecture["scan_staging_state_shape_contract_ready"]
        is True
    )
    assert (
        solver_wide_owner_architecture[
            "solver_wide_interface_state_owner_contract_ready"
        ]
        is True
    )
    assert solver_wide_owner_architecture["explains_k1_k2_blocker"] is True
    assert (
        solver_wide_owner_architecture["bounded_follow_up_implementation_surface"]
        is True
    )
    assert (
        solver_wide_owner_architecture["owner_contract"]["single_owner_per_interface"]
        is True
    )
    assert (
        solver_wide_owner_architecture["scan_staging_contract"][
            "state_shape_change_required"
        ]
        is True
    )
    assert (
        solver_wide_owner_architecture["scan_staging_contract"][
            "cpml_non_cpml_slots_identical"
        ]
        is True
    )
    assert (
        solver_wide_owner_architecture["implementation_contract"][
            "joint_phase_magnitude_required"
        ]
        is True
    )
    assert solver_wide_owner_architecture["production_patch_applied"] is False
    assert solver_wide_owner_architecture["solver_behavior_changed"] is False
    assert solver_wide_owner_architecture["new_solver_hunk_retained"] is False
    solver_wide_owner_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in solver_wide_owner_architecture["candidate_ladder"]
    }
    assert (
        solver_wide_owner_candidates[
            "L1_solver_wide_interface_state_owner_architecture"
        ]["design_component_ready"]
        is True
    )
    assert (
        solver_wide_owner_candidates[
            "L2_scan_staging_state_shape_contract"
        ]["design_component_ready"]
        is True
    )
    assert (
        solver_wide_owner_candidates[
            "L3_combined_interface_state_owner_implementation_contract"
        ]["accepted_candidate"]
        is True
    )
    assert solver_wide_owner_architecture["public_claim_allowed"] is False
    assert solver_wide_owner_architecture["public_observable_promoted"] is False
    assert (
        solver_wide_owner_architecture["true_rt_public_observable_promoted"]
        is False
    )
    assert (
        solver_wide_owner_architecture["dft_flux_tfsf_port_sparameter_promoted"]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_solver_wide_interface_state_owner_architecture_next_prerequisite"
        ]
        == solver_wide_owner_architecture["next_prerequisite"]
    )
    solver_wide_owner_implementation = benchmark_gate[
        "private_plane_wave_solver_wide_interface_state_owner_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_solver_wide_interface_state_owner_implementation_status"
    ] == "no_private_plane_wave_solver_wide_interface_state_owner_implementation"
    assert solver_wide_owner_implementation["terminal_outcome"] == (
        "no_private_plane_wave_solver_wide_interface_state_owner_implementation"
    )
    assert solver_wide_owner_implementation["upstream_owner_architecture_status"] == (
        benchmark_gate[
            "private_plane_wave_solver_wide_interface_state_owner_architecture_status"
        ]
    )
    assert solver_wide_owner_implementation["upstream_phase_staging_status"] == (
        benchmark_gate[
            "private_plane_wave_phase_coherence_staging_implementation_status"
        ]
    )
    assert solver_wide_owner_implementation["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert (
        solver_wide_owner_implementation[
            "candidate_ladder_declared_before_slow_scoring"
        ]
        is True
    )
    assert solver_wide_owner_implementation["candidate_count"] == 5
    assert solver_wide_owner_implementation["selected_candidate_id"] == (
        "M4_solver_wide_owner_implementation_blocked"
    )
    assert solver_wide_owner_implementation["baseline_metrics"] == (
        solver_wide_owner_architecture["baseline_metrics"]
    )
    assert solver_wide_owner_implementation["baseline_metrics_preserved"] is True
    assert solver_wide_owner_implementation["thresholds_unchanged"] is True
    assert (
        solver_wide_owner_implementation[
            "solver_wide_interface_state_owner_contract_ready"
        ]
        is True
    )
    assert solver_wide_owner_implementation["implementation_lane_executed"] is True
    assert solver_wide_owner_implementation["owner_state_shape_hunk_retained"] is False
    assert solver_wide_owner_implementation["scan_staging_hunk_retained"] is False
    assert (
        solver_wide_owner_implementation["joint_score_with_owner_hunk_retained"]
        is False
    )
    assert solver_wide_owner_implementation["joint_phase_magnitude_improved"] is False
    assert solver_wide_owner_implementation["production_patch_applied"] is False
    assert solver_wide_owner_implementation["solver_behavior_changed"] is False
    assert solver_wide_owner_implementation["new_solver_hunk_retained"] is False
    assert solver_wide_owner_implementation["no_bounded_hunk_accepted"] is True
    assert solver_wide_owner_implementation["true_rt_readiness_unlocked"] is False
    assert (
        solver_wide_owner_implementation[
            "next_lane_requires_state_pytree_runner_boundary_design"
        ]
        is True
    )
    solver_wide_implementation_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in solver_wide_owner_implementation["candidate_ladder"]
    }
    assert (
        solver_wide_implementation_candidates[
            "M1_private_owner_state_shape_hunk"
        ]["accepted_candidate"]
        is False
    )
    assert (
        solver_wide_implementation_candidates[
            "M1_private_owner_state_shape_hunk"
        ]["requires_runner_or_jit_state_boundary"]
        is True
    )
    assert (
        solver_wide_implementation_candidates[
            "M2_same_step_scan_staging_owner_wiring"
        ]["accepted_candidate"]
        is False
    )
    assert (
        solver_wide_implementation_candidates[
            "M3_joint_phase_cv_parity_scoring_with_owner"
        ]["accepted_candidate"]
        is False
    )
    assert (
        solver_wide_implementation_candidates[
            "M4_solver_wide_owner_implementation_blocked"
        ]["accepted_candidate"]
        is True
    )
    assert solver_wide_owner_implementation["public_claim_allowed"] is False
    assert solver_wide_owner_implementation["public_observable_promoted"] is False
    assert (
        solver_wide_owner_implementation["true_rt_public_observable_promoted"]
        is False
    )
    assert (
        solver_wide_owner_implementation["dft_flux_tfsf_port_sparameter_promoted"]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_solver_wide_interface_state_owner_implementation_next_prerequisite"
        ]
        == solver_wide_owner_implementation["next_prerequisite"]
    )
    solver_state_owner_propagation = benchmark_gate[
        "private_plane_wave_solver_state_owner_propagation_boundary"
    ]
    assert benchmark_gate[
        "private_plane_wave_solver_state_owner_propagation_boundary_status"
    ] == "private_plane_wave_solver_state_owner_propagation_contract_ready"
    assert solver_state_owner_propagation["terminal_outcome"] == (
        "private_plane_wave_solver_state_owner_propagation_contract_ready"
    )
    assert solver_state_owner_propagation["upstream_owner_implementation_status"] == (
        benchmark_gate[
            "private_plane_wave_solver_wide_interface_state_owner_implementation_status"
        ]
    )
    assert solver_state_owner_propagation["upstream_owner_architecture_status"] == (
        benchmark_gate[
            "private_plane_wave_solver_wide_interface_state_owner_architecture_status"
        ]
    )
    assert solver_state_owner_propagation["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert (
        solver_state_owner_propagation[
            "candidate_ladder_declared_before_slow_scoring"
        ]
        is True
    )
    assert solver_state_owner_propagation["candidate_count"] == 5
    assert solver_state_owner_propagation["selected_candidate_id"] == (
        "N3_combined_owner_propagation_contract"
    )
    assert solver_state_owner_propagation["baseline_metrics"] == (
        solver_wide_owner_implementation["baseline_metrics"]
    )
    assert solver_state_owner_propagation["baseline_metrics_preserved"] is True
    assert solver_state_owner_propagation["thresholds_unchanged"] is True
    assert solver_state_owner_propagation["state_pytree_boundary_ready"] is True
    assert (
        solver_state_owner_propagation[
            "runner_jit_initialization_boundary_ready"
        ]
        is True
    )
    assert (
        solver_state_owner_propagation[
            "solver_state_owner_propagation_contract_ready"
        ]
        is True
    )
    assert (
        solver_state_owner_propagation["explains_owner_implementation_blocker"]
        is True
    )
    assert solver_state_owner_propagation["explains_m1_m2_m3_blocker"] is True
    assert (
        solver_state_owner_propagation["bounded_follow_up_implementation_surface"]
        is True
    )
    assert (
        solver_state_owner_propagation["state_pytree_boundary"][
            "jax_pytree_boundary_defined"
        ]
        is True
    )
    assert (
        solver_state_owner_propagation["runner_jit_boundary"][
            "jit_runner_initializes_owner_state"
        ]
        is True
    )
    assert (
        solver_state_owner_propagation["runner_jit_boundary"][
            "subgridded_runner_initializes_owner_state"
        ]
        is True
    )
    assert (
        solver_state_owner_propagation["runner_jit_boundary"]["result_surface_unchanged"]
        is True
    )
    assert (
        solver_state_owner_propagation["propagation_contract"][
            "requires_cpml_non_cpml_step_identity"
        ]
        is True
    )
    assert solver_state_owner_propagation["production_patch_applied"] is False
    assert solver_state_owner_propagation["solver_behavior_changed"] is False
    assert solver_state_owner_propagation["runner_behavior_changed"] is False
    assert solver_state_owner_propagation["new_solver_hunk_retained"] is False
    assert solver_state_owner_propagation["true_rt_readiness_unlocked"] is False
    propagation_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in solver_state_owner_propagation["candidate_ladder"]
    }
    assert (
        propagation_candidates["N1_subgrid_state_owner_pytree_boundary"][
            "design_component_ready"
        ]
        is True
    )
    assert (
        propagation_candidates["N2_runner_jit_owner_initialization_boundary"][
            "design_component_ready"
        ]
        is True
    )
    assert (
        propagation_candidates["N3_combined_owner_propagation_contract"][
            "accepted_candidate"
        ]
        is True
    )
    assert solver_state_owner_propagation["public_claim_allowed"] is False
    assert solver_state_owner_propagation["public_observable_promoted"] is False
    assert (
        solver_state_owner_propagation["true_rt_public_observable_promoted"]
        is False
    )
    assert (
        solver_state_owner_propagation["dft_flux_tfsf_port_sparameter_promoted"]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_solver_state_owner_propagation_boundary_next_prerequisite"
        ]
        == solver_state_owner_propagation["next_prerequisite"]
    )
    solver_state_owner_implementation = benchmark_gate[
        "private_plane_wave_solver_state_owner_propagation_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_solver_state_owner_propagation_implementation_status"
    ] == (
        "private_plane_wave_runner_jit_owner_propagation_hunk_retained_fixture_quality_pending"
    )
    assert solver_state_owner_implementation["terminal_outcome"] == (
        "private_plane_wave_runner_jit_owner_propagation_hunk_retained_fixture_quality_pending"
    )
    assert solver_state_owner_implementation[
        "upstream_propagation_boundary_status"
    ] == benchmark_gate[
        "private_plane_wave_solver_state_owner_propagation_boundary_status"
    ]
    assert solver_state_owner_implementation["upstream_owner_implementation_status"] == (
        benchmark_gate[
            "private_plane_wave_solver_wide_interface_state_owner_implementation_status"
        ]
    )
    assert solver_state_owner_implementation["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert solver_state_owner_implementation["candidate_count"] == 5
    assert solver_state_owner_implementation["selected_candidate_id"] == (
        "P2_runner_jit_owner_propagation_hunk"
    )
    assert solver_state_owner_implementation["baseline_metrics"] == (
        solver_state_owner_propagation["baseline_metrics"]
    )
    assert solver_state_owner_implementation["baseline_metrics_preserved"] is True
    assert solver_state_owner_implementation["thresholds_unchanged"] is True
    assert solver_state_owner_implementation["state_pytree_hunk_retained"] is True
    assert solver_state_owner_implementation["owner_state_shape_hunk_retained"] is True
    assert (
        solver_state_owner_implementation[
            "runner_jit_owner_propagation_hunk_retained"
        ]
        is True
    )
    assert solver_state_owner_implementation["scan_staging_hunk_retained"] is False
    assert (
        solver_state_owner_implementation["joint_score_with_owner_hunk_retained"]
        is False
    )
    assert solver_state_owner_implementation["joint_phase_magnitude_improved"] is False
    assert solver_state_owner_implementation["production_patch_applied"] is True
    assert solver_state_owner_implementation["solver_behavior_changed"] is True
    assert solver_state_owner_implementation["field_update_behavior_changed"] is False
    assert solver_state_owner_implementation["runner_behavior_changed"] is True
    assert solver_state_owner_implementation["new_solver_hunk_retained"] is True
    assert solver_state_owner_implementation["true_rt_readiness_unlocked"] is False
    assert solver_state_owner_implementation["fixture_quality_pending"] is True
    assert (
        solver_state_owner_implementation["next_lane_requires_owner_scan_wiring"]
        is True
    )
    implementation_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in solver_state_owner_implementation["candidate_ladder"]
    }
    assert (
        implementation_candidates["P1_private_owner_state_shape_hunk"][
            "production_hunk_retained"
        ]
        is True
    )
    assert (
        implementation_candidates["P2_runner_jit_owner_propagation_hunk"][
            "accepted_candidate"
        ]
        is True
    )
    assert (
        implementation_candidates["P3_owner_scan_wiring_joint_parity_score"][
            "attempted_in_this_lane"
        ]
        is False
    )
    assert (
        implementation_candidates["P4_solver_state_owner_propagation_blocked"][
            "accepted_candidate"
        ]
        is False
    )
    assert solver_state_owner_implementation["public_claim_allowed"] is False
    assert solver_state_owner_implementation["public_observable_promoted"] is False
    assert (
        solver_state_owner_implementation["true_rt_public_observable_promoted"]
        is False
    )
    assert (
        solver_state_owner_implementation["dft_flux_tfsf_port_sparameter_promoted"]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_solver_state_owner_propagation_implementation_next_prerequisite"
        ]
        == solver_state_owner_implementation["next_prerequisite"]
    )
    owner_scan_wiring = benchmark_gate[
        "private_plane_wave_owner_scan_wiring_joint_scoring"
    ]
    assert benchmark_gate[
        "private_plane_wave_owner_scan_wiring_joint_scoring_status"
    ] == (
        "private_plane_wave_owner_joint_parity_scoring_hunk_retained_fixture_quality_pending"
    )
    assert owner_scan_wiring["terminal_outcome"] == (
        "private_plane_wave_owner_joint_parity_scoring_hunk_retained_fixture_quality_pending"
    )
    assert owner_scan_wiring["upstream_propagation_implementation_status"] == (
        benchmark_gate[
            "private_plane_wave_solver_state_owner_propagation_implementation_status"
        ]
    )
    assert owner_scan_wiring["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert owner_scan_wiring["candidate_count"] == 5
    assert owner_scan_wiring["selected_candidate_id"] == (
        "Q2_owner_joint_phase_cv_scoring_guard"
    )
    assert owner_scan_wiring["baseline_metrics"] == (
        solver_state_owner_implementation["baseline_metrics"]
    )
    assert owner_scan_wiring["baseline_metrics_preserved"] is True
    assert owner_scan_wiring["thresholds_unchanged"] is True
    assert owner_scan_wiring["owner_scan_wiring_hunk_retained"] is True
    assert owner_scan_wiring["cpml_owner_scan_wiring_retained"] is True
    assert owner_scan_wiring["non_cpml_owner_scan_wiring_retained"] is True
    assert owner_scan_wiring["same_step_h_e_scan_visibility"] is True
    assert owner_scan_wiring["private_owner_reference_phase_recorded"] is True
    assert owner_scan_wiring["private_owner_reference_magnitude_recorded"] is True
    assert owner_scan_wiring["owner_joint_parity_scoring_hunk_retained"] is True
    assert owner_scan_wiring["joint_phase_cv_guard_retained"] is True
    assert owner_scan_wiring["joint_phase_magnitude_improved"] is False
    assert owner_scan_wiring["production_patch_applied"] is True
    assert owner_scan_wiring["solver_behavior_changed"] is True
    assert owner_scan_wiring["field_update_behavior_changed"] is False
    assert owner_scan_wiring["runner_behavior_changed"] is False
    assert owner_scan_wiring["new_solver_hunk_retained"] is True
    assert owner_scan_wiring["subgrid_vacuum_parity_passed"] is False
    assert owner_scan_wiring["fixture_quality_pending"] is True
    assert owner_scan_wiring["true_rt_readiness_unlocked"] is False
    assert owner_scan_wiring["next_lane_requires_physical_phase_cv_correction"] is True
    owner_scan_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in owner_scan_wiring["candidate_ladder"]
    }
    assert (
        owner_scan_candidates["Q1_same_step_owner_scan_wiring_hunk"][
            "production_hunk_retained"
        ]
        is True
    )
    assert (
        owner_scan_candidates["Q2_owner_joint_phase_cv_scoring_guard"][
            "accepted_candidate"
        ]
        is True
    )
    assert (
        owner_scan_candidates["Q3_private_parity_passes_true_rt_pending"][
            "accepted_candidate"
        ]
        is False
    )
    assert (
        owner_scan_candidates["Q4_owner_scan_wiring_joint_scoring_blocked"][
            "accepted_candidate"
        ]
        is False
    )
    assert owner_scan_wiring["public_claim_allowed"] is False
    assert owner_scan_wiring["public_observable_promoted"] is False
    assert owner_scan_wiring["true_rt_public_observable_promoted"] is False
    assert owner_scan_wiring["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_owner_scan_wiring_joint_scoring_next_prerequisite"
        ]
        == owner_scan_wiring["next_prerequisite"]
    )
    physical_correction = benchmark_gate[
        "private_plane_wave_owner_backed_physical_phase_cv_correction"
    ]
    assert benchmark_gate[
        "private_plane_wave_owner_backed_physical_phase_cv_correction_status"
    ] == "no_private_plane_wave_owner_backed_physical_phase_cv_correction"
    assert physical_correction["terminal_outcome"] == (
        "no_private_plane_wave_owner_backed_physical_phase_cv_correction"
    )
    assert physical_correction["upstream_owner_scan_wiring_status"] == (
        benchmark_gate["private_plane_wave_owner_scan_wiring_joint_scoring_status"]
    )
    assert physical_correction["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert physical_correction["candidate_count"] == 5
    assert physical_correction["selected_candidate_id"] == (
        "R4_owner_backed_physical_correction_blocked"
    )
    assert physical_correction["baseline_metrics"] == owner_scan_wiring["baseline_metrics"]
    assert physical_correction["baseline_metrics_preserved"] is True
    assert physical_correction["thresholds_unchanged"] is True
    assert physical_correction["owner_scan_wiring_hunk_retained"] is True
    assert physical_correction["owner_joint_parity_scoring_hunk_retained"] is True
    assert physical_correction["phase_correction_hunk_retained"] is False
    assert physical_correction["magnitude_correction_hunk_retained"] is False
    assert physical_correction["combined_phase_cv_correction_hunk_retained"] is False
    assert physical_correction["face_local_modal_operator_ready"] is False
    assert physical_correction["joint_phase_magnitude_improved"] is False
    assert physical_correction["material_improvement_demonstrated"] is False
    assert physical_correction["production_patch_applied"] is False
    assert physical_correction["solver_behavior_changed"] is False
    assert physical_correction["field_update_behavior_changed"] is False
    assert physical_correction["runner_behavior_changed"] is False
    assert physical_correction["new_solver_hunk_retained"] is False
    assert physical_correction["no_bounded_hunk_accepted"] is True
    assert physical_correction["subgrid_vacuum_parity_passed"] is False
    assert physical_correction["fixture_quality_pending"] is True
    assert physical_correction["true_rt_readiness_unlocked"] is False
    assert (
        physical_correction[
            "next_lane_requires_face_local_modal_correction_architecture"
        ]
        is True
    )
    physical_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in physical_correction["candidate_ladder"]
    }
    assert (
        physical_candidates["R1_owner_phase_reference_correction"][
            "accepted_candidate"
        ]
        is False
    )
    assert (
        physical_candidates["R1_owner_phase_reference_correction"][
            "requires_face_local_modal_operator"
        ]
        is True
    )
    assert (
        physical_candidates["R2_owner_magnitude_balancing_correction"][
            "accepted_candidate"
        ]
        is False
    )
    assert (
        physical_candidates["R2_owner_magnitude_balancing_correction"][
            "requires_face_local_modal_operator"
        ]
        is True
    )
    assert (
        physical_candidates["R4_owner_backed_physical_correction_blocked"][
            "accepted_candidate"
        ]
        is True
    )
    assert physical_correction["public_claim_allowed"] is False
    assert physical_correction["public_observable_promoted"] is False
    assert physical_correction["true_rt_public_observable_promoted"] is False
    assert physical_correction["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_owner_backed_physical_phase_cv_correction_next_prerequisite"
        ]
        == physical_correction["next_prerequisite"]
    )
    modal_architecture = benchmark_gate[
        "private_plane_wave_face_local_modal_correction_architecture"
    ]
    assert benchmark_gate[
        "private_plane_wave_face_local_modal_correction_architecture_status"
    ] == "private_plane_wave_face_local_modal_correction_contract_ready"
    assert modal_architecture["terminal_outcome"] == (
        "private_plane_wave_face_local_modal_correction_contract_ready"
    )
    assert modal_architecture["upstream_physical_correction_status"] == (
        benchmark_gate[
            "private_plane_wave_owner_backed_physical_phase_cv_correction_status"
        ]
    )
    assert modal_architecture["upstream_owner_scan_wiring_status"] == (
        benchmark_gate["private_plane_wave_owner_scan_wiring_joint_scoring_status"]
    )
    assert modal_architecture["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert modal_architecture[
        "candidate_ladder_declared_before_slow_scoring"
    ] is True
    assert modal_architecture["candidate_count"] == 5
    assert modal_architecture["selected_candidate_id"] == (
        "S3_combined_phase_cv_modal_correction_contract"
    )
    assert modal_architecture["baseline_metrics"] == (
        physical_correction["baseline_metrics"]
    )
    assert modal_architecture["baseline_metrics_preserved"] is True
    assert modal_architecture["thresholds_unchanged"] is True
    assert modal_architecture["face_local_phase_modal_contract_ready"] is True
    assert modal_architecture["face_local_magnitude_modal_contract_ready"] is True
    assert modal_architecture["face_local_modal_correction_contract_ready"] is True
    assert modal_architecture["implementation_contract_ready"] is True
    assert modal_architecture["bounded_follow_up_implementation_surface"] is True
    phase_modal_contract = modal_architecture["phase_modal_contract"]
    assert phase_modal_contract["owner_phase_reference_input"] is True
    assert phase_modal_contract["face_local_tangential_modes_required"] is True
    assert (
        phase_modal_contract["modal_distribution"]
        == "interior_masked_tangential_characteristic_mode"
    )
    assert phase_modal_contract["paired_magnitude_cv_guard_required"] is True
    assert phase_modal_contract["vacuum_regression_guard_required"] is True
    assert phase_modal_contract["cpml_non_cpml_symmetry_required"] is True
    assert phase_modal_contract["no_public_observable"] is True
    magnitude_modal_contract = modal_architecture["magnitude_modal_contract"]
    assert magnitude_modal_contract["owner_magnitude_reference_input"] is True
    assert magnitude_modal_contract["face_local_tangential_modes_required"] is True
    assert (
        magnitude_modal_contract["modal_distribution"]
        == "interior_masked_energy_weighted_tangential_mode"
    )
    assert magnitude_modal_contract["paired_phase_spread_guard_required"] is True
    assert magnitude_modal_contract["vacuum_regression_guard_required"] is True
    assert magnitude_modal_contract["cpml_non_cpml_symmetry_required"] is True
    assert magnitude_modal_contract["no_public_observable"] is True
    combined_modal_contract = modal_architecture["combined_contract"]
    assert combined_modal_contract["requires_phase_modal_operator"] is True
    assert combined_modal_contract["requires_magnitude_modal_operator"] is True
    assert combined_modal_contract["requires_owner_scan_scorer"] is True
    assert combined_modal_contract["requires_paired_phase_cv_gate"] is True
    assert combined_modal_contract["requires_vacuum_magnitude_phase_gate"] is True
    assert (
        combined_modal_contract["field_update_behavior_change_requires_slow_gate"]
        is True
    )
    assert combined_modal_contract["public_promotion_allowed"] is False
    assert modal_architecture["production_patch_applied"] is False
    assert modal_architecture["solver_behavior_changed"] is False
    assert modal_architecture["field_update_behavior_changed"] is False
    assert modal_architecture["runner_behavior_changed"] is False
    assert modal_architecture["new_solver_hunk_retained"] is False
    assert modal_architecture["true_rt_readiness_unlocked"] is False
    modal_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in modal_architecture["candidate_ladder"]
    }
    assert modal_candidates[
        "S1_face_local_phase_modal_operator_design"
    ]["design_component_ready"] is True
    assert modal_candidates[
        "S1_face_local_phase_modal_operator_design"
    ]["superseded_by"] == "S3_combined_phase_cv_modal_correction_contract"
    assert modal_candidates[
        "S2_face_local_magnitude_modal_operator_design"
    ]["design_component_ready"] is True
    assert modal_candidates[
        "S2_face_local_magnitude_modal_operator_design"
    ]["superseded_by"] == "S3_combined_phase_cv_modal_correction_contract"
    assert modal_candidates[
        "S3_combined_phase_cv_modal_correction_contract"
    ]["accepted_candidate"] is True
    assert modal_candidates[
        "S4_modal_correction_architecture_blocked"
    ]["accepted_candidate"] is False
    assert modal_architecture["public_claim_allowed"] is False
    assert modal_architecture["public_observable_promoted"] is False
    assert modal_architecture["true_rt_public_observable_promoted"] is False
    assert modal_architecture["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_face_local_modal_correction_architecture_next_prerequisite"
        ]
        == modal_architecture["next_prerequisite"]
    )
    modal_implementation = benchmark_gate[
        "private_plane_wave_face_local_modal_correction_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_face_local_modal_correction_implementation_status"
    ] == "no_private_plane_wave_face_local_modal_correction_implementation"
    assert modal_implementation["terminal_outcome"] == (
        "no_private_plane_wave_face_local_modal_correction_implementation"
    )
    assert modal_implementation["upstream_architecture_status"] == (
        benchmark_gate[
            "private_plane_wave_face_local_modal_correction_architecture_status"
        ]
    )
    assert modal_implementation["upstream_physical_correction_status"] == (
        benchmark_gate[
            "private_plane_wave_owner_backed_physical_phase_cv_correction_status"
        ]
    )
    assert modal_implementation["upstream_owner_scan_wiring_status"] == (
        benchmark_gate["private_plane_wave_owner_scan_wiring_joint_scoring_status"]
    )
    assert modal_implementation["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert modal_implementation[
        "candidate_ladder_declared_before_slow_scoring"
    ] is True
    assert modal_implementation["candidate_count"] == 5
    assert modal_implementation["selected_candidate_id"] == (
        "T4_face_local_modal_implementation_blocked"
    )
    assert modal_implementation["baseline_metrics"] == (
        modal_architecture["baseline_metrics"]
    )
    assert modal_implementation["baseline_metrics_preserved"] is True
    assert modal_implementation["thresholds_unchanged"] is True
    assert modal_implementation["architecture_contract_ready"] is True
    assert modal_implementation["face_local_phase_modal_contract_ready"] is True
    assert modal_implementation["face_local_magnitude_modal_contract_ready"] is True
    assert modal_implementation["face_local_modal_correction_contract_ready"] is True
    assert modal_implementation["implementation_lane_executed"] is True
    assert modal_implementation["phase_modal_hunk_retained"] is False
    assert modal_implementation["magnitude_modal_hunk_retained"] is False
    assert modal_implementation["combined_modal_hunk_retained"] is False
    assert modal_implementation["production_patch_applied"] is False
    assert modal_implementation["solver_behavior_changed"] is False
    assert modal_implementation["field_update_behavior_changed"] is False
    assert modal_implementation["runner_behavior_changed"] is False
    assert modal_implementation["new_solver_hunk_retained"] is False
    assert modal_implementation["no_bounded_hunk_accepted"] is True
    assert modal_implementation["true_rt_readiness_unlocked"] is False
    implementation_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in modal_implementation["candidate_ladder"]
    }
    assert implementation_candidates[
        "T1_face_local_phase_modal_helper"
    ]["accepted_candidate"] is False
    assert implementation_candidates[
        "T1_face_local_phase_modal_helper"
    ]["attempted_in_this_lane"] is True
    assert implementation_candidates[
        "T2_face_local_magnitude_modal_helper"
    ]["accepted_candidate"] is False
    assert implementation_candidates[
        "T2_face_local_magnitude_modal_helper"
    ]["attempted_in_this_lane"] is True
    assert implementation_candidates[
        "T3_combined_phase_cv_modal_private_parity_pass"
    ]["accepted_candidate"] is False
    assert implementation_candidates[
        "T4_face_local_modal_implementation_blocked"
    ]["accepted_candidate"] is True
    assert modal_implementation["public_claim_allowed"] is False
    assert modal_implementation["public_observable_promoted"] is False
    assert modal_implementation["true_rt_public_observable_promoted"] is False
    assert modal_implementation["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_face_local_modal_correction_implementation_next_prerequisite"
        ]
        == modal_implementation["next_prerequisite"]
    )
    modal_failure_theory = benchmark_gate[
        "private_plane_wave_face_local_modal_failure_theory"
    ]
    assert benchmark_gate[
        "private_plane_wave_face_local_modal_failure_theory_status"
    ] == "private_plane_wave_face_local_modal_failure_theory_contract_ready"
    assert modal_failure_theory["terminal_outcome"] == (
        "private_plane_wave_face_local_modal_failure_theory_contract_ready"
    )
    assert modal_failure_theory["upstream_implementation_status"] == (
        benchmark_gate[
            "private_plane_wave_face_local_modal_correction_implementation_status"
        ]
    )
    assert modal_failure_theory["upstream_architecture_status"] == (
        benchmark_gate[
            "private_plane_wave_face_local_modal_correction_architecture_status"
        ]
    )
    assert modal_failure_theory["upstream_owner_scan_wiring_status"] == (
        benchmark_gate["private_plane_wave_owner_scan_wiring_joint_scoring_status"]
    )
    assert modal_failure_theory["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert modal_failure_theory["candidate_ladder_declared_before_solver_edit"] is True
    assert modal_failure_theory["candidate_ladder_declared_before_slow_scoring"] is True
    assert modal_failure_theory["candidate_count"] == 5
    assert modal_failure_theory["selected_candidate_id"] == (
        "U3_failure_theory_retry_contract"
    )
    assert modal_failure_theory["baseline_metrics"] == (
        modal_implementation["baseline_metrics"]
    )
    assert modal_failure_theory["baseline_metrics_preserved"] is True
    assert modal_failure_theory["thresholds_unchanged"] is True
    assert modal_failure_theory["implementation_failure_packet_frozen"] is True
    assert modal_failure_theory["rejected_probe_recorded"] is True
    assert modal_failure_theory["owner_timing_theory_ready"] is True
    assert modal_failure_theory["observable_basis_theory_ready"] is True
    assert modal_failure_theory["failure_theory_contract_ready"] is True
    assert modal_failure_theory["bounded_follow_up_implementation_surface"] is True
    owner_timing_theory = modal_failure_theory["owner_timing_theory"]
    assert owner_timing_theory["same_step_owner_reference_is_diagnostic"] is True
    assert owner_timing_theory["field_update_feedback_can_chase_post_sat_state"] is True
    assert owner_timing_theory["requires_lagged_or_predictor_owner_reference"] is True
    observable_basis_theory = modal_failure_theory["observable_basis_theory"]
    assert observable_basis_theory["owner_reference_basis"] == (
        "face_scalar_characteristic_mean"
    )
    assert observable_basis_theory["benchmark_observable_basis"] == (
        "transverse_plane_dft_distribution"
    )
    assert (
        observable_basis_theory["basis_mismatch_explains_vacuum_phase_regression"]
        is True
    )
    retry_contract = modal_failure_theory["retry_contract"]
    assert retry_contract["requires_lagged_owner_reference"] is True
    assert retry_contract["requires_observable_aligned_modal_basis"] is True
    assert retry_contract["requires_owner_score_precheck"] is True
    assert retry_contract["requires_slow_plane_wave_parity_gate"] is True
    assert retry_contract["requires_cpml_non_cpml_symmetry_tests"] is True
    assert retry_contract["requires_vacuum_magnitude_phase_gate"] is True
    assert retry_contract["forbids_public_promotion"] is True
    assert modal_failure_theory["production_patch_applied"] is False
    assert modal_failure_theory["solver_behavior_changed"] is False
    assert modal_failure_theory["field_update_behavior_changed"] is False
    assert modal_failure_theory["runner_behavior_changed"] is False
    assert modal_failure_theory["new_solver_hunk_retained"] is False
    assert modal_failure_theory["true_rt_readiness_unlocked"] is False
    failure_theory_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in modal_failure_theory["candidate_ladder"]
    }
    assert failure_theory_candidates[
        "U1_owner_reference_timing_mismatch_theory"
    ]["theory_component_ready"] is True
    assert failure_theory_candidates[
        "U2_characteristic_observable_modal_basis_theory"
    ]["theory_component_ready"] is True
    assert failure_theory_candidates[
        "U3_failure_theory_retry_contract"
    ]["accepted_candidate"] is True
    assert failure_theory_candidates[
        "U4_failure_theory_redesign_blocked"
    ]["accepted_candidate"] is False
    assert modal_failure_theory["public_claim_allowed"] is False
    assert modal_failure_theory["public_observable_promoted"] is False
    assert modal_failure_theory["true_rt_public_observable_promoted"] is False
    assert modal_failure_theory["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_face_local_modal_failure_theory_next_prerequisite"
        ]
        == modal_failure_theory["next_prerequisite"]
    )
    modal_retry_implementation = benchmark_gate[
        "private_plane_wave_face_local_modal_retry_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_face_local_modal_retry_implementation_status"
    ] == "no_private_plane_wave_face_local_modal_retry_implementation"
    assert modal_retry_implementation["terminal_outcome"] == (
        "no_private_plane_wave_face_local_modal_retry_implementation"
    )
    assert modal_retry_implementation["upstream_failure_theory_status"] == (
        benchmark_gate["private_plane_wave_face_local_modal_failure_theory_status"]
    )
    assert modal_retry_implementation["upstream_implementation_status"] == (
        benchmark_gate[
            "private_plane_wave_face_local_modal_correction_implementation_status"
        ]
    )
    assert modal_retry_implementation["upstream_architecture_status"] == (
        benchmark_gate[
            "private_plane_wave_face_local_modal_correction_architecture_status"
        ]
    )
    assert modal_retry_implementation["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert modal_retry_implementation[
        "candidate_ladder_declared_before_slow_scoring"
    ] is True
    assert modal_retry_implementation["candidate_count"] == 5
    assert modal_retry_implementation["selected_candidate_id"] == (
        "V4_retry_implementation_blocked"
    )
    assert modal_retry_implementation["baseline_metrics"] == (
        modal_failure_theory["baseline_metrics"]
    )
    assert modal_retry_implementation["thresholds"] == (
        modal_failure_theory["thresholds"]
    )
    assert modal_retry_implementation["baseline_metrics_preserved"] is True
    assert modal_retry_implementation["thresholds_unchanged"] is True
    assert modal_retry_implementation["failure_theory_contract_ready"] is True
    assert modal_retry_implementation["implementation_retry_lane_executed"] is True
    assert modal_retry_implementation["lagged_owner_reference_required"] is True
    assert (
        modal_retry_implementation["observable_aligned_modal_basis_required"] is True
    )
    assert modal_retry_implementation["lagged_owner_modal_hunk_retained"] is False
    assert (
        modal_retry_implementation["observable_aligned_modal_hunk_retained"] is False
    )
    assert modal_retry_implementation["combined_retry_hunk_retained"] is False
    assert modal_retry_implementation["production_patch_applied"] is False
    assert modal_retry_implementation["solver_behavior_changed"] is False
    assert modal_retry_implementation["field_update_behavior_changed"] is False
    assert modal_retry_implementation["runner_behavior_changed"] is False
    assert modal_retry_implementation["new_solver_hunk_retained"] is False
    assert modal_retry_implementation["no_bounded_hunk_accepted"] is True
    assert modal_retry_implementation["subgrid_vacuum_parity_scored"] is True
    assert modal_retry_implementation["subgrid_vacuum_parity_passed"] is False
    assert modal_retry_implementation["true_rt_readiness_unlocked"] is False
    assert (
        modal_retry_implementation["next_lane_requires_observable_proxy_architecture"]
        is True
    )
    retry_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in modal_retry_implementation["candidate_ladder"]
    }
    assert retry_candidates[
        "V1_lagged_owner_reference_modal_helper"
    ]["attempted_in_this_lane"] is True
    assert retry_candidates[
        "V1_lagged_owner_reference_modal_helper"
    ]["accepted_candidate"] is False
    assert retry_candidates[
        "V1_lagged_owner_reference_modal_helper"
    ]["lagged_owner_modal_hunk_retained"] is False
    assert retry_candidates[
        "V2_observable_aligned_tangential_modal_basis"
    ]["attempted_in_this_lane"] is True
    assert retry_candidates[
        "V2_observable_aligned_tangential_modal_basis"
    ]["accepted_candidate"] is False
    assert retry_candidates[
        "V2_observable_aligned_tangential_modal_basis"
    ]["observable_aligned_modal_hunk_retained"] is False
    assert retry_candidates[
        "V3_combined_lagged_observable_modal_private_parity_pass"
    ]["accepted_candidate"] is False
    assert retry_candidates[
        "V4_retry_implementation_blocked"
    ]["accepted_candidate"] is True
    assert modal_retry_implementation["public_claim_allowed"] is False
    assert modal_retry_implementation["public_observable_promoted"] is False
    assert modal_retry_implementation["true_rt_public_observable_promoted"] is False
    assert (
        modal_retry_implementation["dft_flux_tfsf_port_sparameter_promoted"] is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_face_local_modal_retry_implementation_next_prerequisite"
        ]
        == modal_retry_implementation["next_prerequisite"]
    )
    proxy_architecture = benchmark_gate[
        "private_plane_wave_observable_proxy_modal_architecture"
    ]
    assert benchmark_gate[
        "private_plane_wave_observable_proxy_modal_architecture_status"
    ] == "private_plane_wave_observable_proxy_modal_retry_contract_ready"
    assert proxy_architecture["terminal_outcome"] == (
        "private_plane_wave_observable_proxy_modal_retry_contract_ready"
    )
    assert proxy_architecture["upstream_retry_implementation_status"] == (
        benchmark_gate["private_plane_wave_face_local_modal_retry_implementation_status"]
    )
    assert proxy_architecture["upstream_failure_theory_status"] == (
        benchmark_gate["private_plane_wave_face_local_modal_failure_theory_status"]
    )
    assert proxy_architecture["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert proxy_architecture["candidate_ladder_declared_before_solver_edit"] is True
    assert proxy_architecture[
        "candidate_ladder_declared_before_slow_scoring"
    ] is True
    assert proxy_architecture["candidate_count"] == 5
    assert proxy_architecture["selected_candidate_id"] == (
        "W3_observable_proxy_modal_retry_contract"
    )
    assert proxy_architecture["baseline_metrics"] == (
        modal_retry_implementation["baseline_metrics"]
    )
    assert proxy_architecture["thresholds"] == (
        modal_retry_implementation["thresholds"]
    )
    assert proxy_architecture["baseline_metrics_preserved"] is True
    assert proxy_architecture["thresholds_unchanged"] is True
    assert proxy_architecture["retry_fail_closed_packet_frozen"] is True
    assert proxy_architecture["solver_local_observable_proxy_basis_ready"] is True
    assert (
        proxy_architecture["lagged_owner_observable_proxy_state_contract_ready"]
        is True
    )
    assert proxy_architecture["observable_proxy_modal_retry_contract_ready"] is True
    assert proxy_architecture["implementation_contract_ready"] is True
    assert proxy_architecture["bounded_follow_up_implementation_surface"] is True
    proxy_basis_contract = proxy_architecture["proxy_basis_contract"]
    assert proxy_basis_contract["observable_proxy_basis"] == (
        "solver_local_transverse_face_energy_phase_proxy"
    )
    assert proxy_basis_contract["benchmark_observable_basis"] == (
        "transverse_plane_dft_distribution"
    )
    assert proxy_basis_contract["benchmark_plane_dft_observable_imported"] is False
    assert proxy_basis_contract["private_face_samples_only"] is True
    assert proxy_basis_contract["uses_energy_weighted_phase_proxy"] is True
    assert proxy_basis_contract["uses_plane_dft_monitor"] is False
    assert proxy_basis_contract["requires_paired_phase_cv_guard"] is True
    assert proxy_basis_contract["requires_vacuum_magnitude_phase_guard"] is True
    assert proxy_basis_contract["cpml_non_cpml_symmetry_required"] is True
    lagged_proxy_state = proxy_architecture["lagged_owner_state_contract"]
    assert lagged_proxy_state["owner_reference_timing"] == (
        "previous_step_or_predictor_owner_reference"
    )
    assert lagged_proxy_state["same_step_diagnostic_feedback_forbidden"] is True
    assert lagged_proxy_state["jax_pytree_shape_contract_required"] is True
    assert lagged_proxy_state["runner_jit_initialization_contract_required"] is True
    implementation_contract = proxy_architecture["implementation_contract"]
    assert implementation_contract["requires_proxy_basis_contract"] is True
    assert implementation_contract["requires_lagged_owner_state_contract"] is True
    assert implementation_contract["requires_owner_score_precheck"] is True
    assert implementation_contract["requires_slow_plane_wave_parity_gate"] is True
    assert implementation_contract["requires_paired_phase_cv_gate"] is True
    assert implementation_contract["requires_vacuum_magnitude_phase_gate"] is True
    assert (
        implementation_contract["requires_cpml_non_cpml_symmetry_tests"] is True
    )
    assert implementation_contract["forbids_benchmark_dft_observable_import"] is True
    assert implementation_contract["forbids_public_promotion"] is True
    assert proxy_architecture["benchmark_plane_dft_observable_imported"] is False
    assert proxy_architecture["solver_local_proxy_uses_plane_dft_monitor"] is False
    assert proxy_architecture["uses_private_face_samples"] is True
    assert proxy_architecture["uses_energy_weighted_phase_proxy"] is True
    assert proxy_architecture["uses_transverse_face_distribution_proxy"] is True
    assert proxy_architecture["production_patch_applied"] is False
    assert proxy_architecture["solver_behavior_changed"] is False
    assert proxy_architecture["field_update_behavior_changed"] is False
    assert proxy_architecture["runner_behavior_changed"] is False
    assert proxy_architecture["new_solver_hunk_retained"] is False
    assert proxy_architecture["true_rt_readiness_unlocked"] is False
    assert (
        proxy_architecture["next_lane_requires_observable_proxy_implementation"]
        is True
    )
    proxy_architecture_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in proxy_architecture["candidate_ladder"]
    }
    assert proxy_architecture_candidates[
        "W1_solver_local_observable_proxy_modal_basis"
    ]["design_component_ready"] is True
    assert proxy_architecture_candidates[
        "W2_lagged_owner_observable_proxy_state_shape"
    ]["design_component_ready"] is True
    assert proxy_architecture_candidates[
        "W3_observable_proxy_modal_retry_contract"
    ]["accepted_candidate"] is True
    assert proxy_architecture_candidates[
        "W4_observable_proxy_architecture_blocked"
    ]["accepted_candidate"] is False
    assert proxy_architecture["public_claim_allowed"] is False
    assert proxy_architecture["public_observable_promoted"] is False
    assert proxy_architecture["true_rt_public_observable_promoted"] is False
    assert proxy_architecture["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_observable_proxy_modal_architecture_next_prerequisite"
        ]
        == proxy_architecture["next_prerequisite"]
    )
    proxy_implementation = benchmark_gate[
        "private_plane_wave_observable_proxy_modal_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_observable_proxy_modal_implementation_status"
    ] == "no_private_plane_wave_observable_proxy_modal_retry_implementation"
    assert proxy_implementation["terminal_outcome"] == (
        "no_private_plane_wave_observable_proxy_modal_retry_implementation"
    )
    assert proxy_implementation["upstream_architecture_status"] == (
        benchmark_gate["private_plane_wave_observable_proxy_modal_architecture_status"]
    )
    assert proxy_implementation["upstream_retry_implementation_status"] == (
        benchmark_gate["private_plane_wave_face_local_modal_retry_implementation_status"]
    )
    assert proxy_implementation["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert proxy_implementation["candidate_ladder_declared_before_solver_edit"] is True
    assert proxy_implementation[
        "candidate_ladder_declared_before_slow_scoring"
    ] is True
    assert proxy_implementation["candidate_count"] == 5
    assert proxy_implementation["selected_candidate_id"] == (
        "X4_observable_proxy_modal_retry_implementation_blocked"
    )
    assert proxy_implementation["baseline_metrics"] == (
        proxy_architecture["baseline_metrics"]
    )
    assert proxy_implementation["thresholds"] == proxy_architecture["thresholds"]
    assert proxy_implementation["baseline_metrics_preserved"] is True
    assert proxy_implementation["thresholds_unchanged"] is True
    assert proxy_implementation["architecture_contract_ready"] is True
    assert proxy_implementation["implementation_lane_executed"] is True
    assert proxy_implementation["packed_face_proxy_state_required"] is True
    assert proxy_implementation["lagged_proxy_state_hunk_retained"] is False
    assert proxy_implementation["observable_proxy_modal_hunk_retained"] is False
    assert proxy_implementation["combined_proxy_retry_hunk_retained"] is False
    assert proxy_implementation["benchmark_plane_dft_observable_imported"] is False
    assert proxy_implementation["solver_local_proxy_uses_plane_dft_monitor"] is False
    assert proxy_implementation["production_patch_applied"] is False
    assert proxy_implementation["solver_behavior_changed"] is False
    assert proxy_implementation["field_update_behavior_changed"] is False
    assert proxy_implementation["runner_behavior_changed"] is False
    assert proxy_implementation["new_solver_hunk_retained"] is False
    assert proxy_implementation["no_bounded_hunk_accepted"] is True
    assert proxy_implementation["true_rt_readiness_unlocked"] is False
    assert (
        proxy_implementation["next_lane_requires_face_packet_state_shape_design"]
        is True
    )
    proxy_implementation_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in proxy_implementation["candidate_ladder"]
    }
    assert proxy_implementation_candidates[
        "X1_lagged_owner_proxy_state_plumbing"
    ]["attempted_in_this_lane"] is True
    assert proxy_implementation_candidates[
        "X1_lagged_owner_proxy_state_plumbing"
    ]["accepted_candidate"] is False
    assert proxy_implementation_candidates[
        "X1_lagged_owner_proxy_state_plumbing"
    ]["required_owner_state_shape"] == (
        "packed_face_local_proxy_distribution_with_face_offsets_and_masks"
    )
    assert proxy_implementation_candidates[
        "X2_solver_local_observable_proxy_modal_correction"
    ]["requires_packed_face_proxy_state"] is True
    assert proxy_implementation_candidates[
        "X3_combined_observable_proxy_modal_private_parity_pass"
    ]["accepted_candidate"] is False
    assert proxy_implementation_candidates[
        "X4_observable_proxy_modal_retry_implementation_blocked"
    ]["accepted_candidate"] is True
    assert proxy_implementation["public_claim_allowed"] is False
    assert proxy_implementation["public_observable_promoted"] is False
    assert proxy_implementation["true_rt_public_observable_promoted"] is False
    assert proxy_implementation["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_observable_proxy_modal_implementation_next_prerequisite"
        ]
        == proxy_implementation["next_prerequisite"]
    )
    face_packet_shape = benchmark_gate[
        "private_plane_wave_observable_proxy_face_packet_state_shape"
    ]
    assert benchmark_gate[
        "private_plane_wave_observable_proxy_face_packet_state_shape_status"
    ] == "private_plane_wave_proxy_face_packet_state_contract_ready"
    assert face_packet_shape["terminal_outcome"] == (
        "private_plane_wave_proxy_face_packet_state_contract_ready"
    )
    assert face_packet_shape["upstream_implementation_status"] == (
        benchmark_gate["private_plane_wave_observable_proxy_modal_implementation_status"]
    )
    assert face_packet_shape["upstream_architecture_status"] == (
        benchmark_gate["private_plane_wave_observable_proxy_modal_architecture_status"]
    )
    assert face_packet_shape["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert face_packet_shape["candidate_ladder_declared_before_solver_edit"] is True
    assert face_packet_shape["candidate_ladder_declared_before_slow_scoring"] is True
    assert face_packet_shape["candidate_count"] == 5
    assert face_packet_shape["selected_candidate_id"] == (
        "Y3_proxy_face_packet_state_contract"
    )
    assert face_packet_shape["baseline_metrics"] == (
        proxy_implementation["baseline_metrics"]
    )
    assert face_packet_shape["thresholds"] == proxy_implementation["thresholds"]
    assert face_packet_shape["baseline_metrics_preserved"] is True
    assert face_packet_shape["thresholds_unchanged"] is True
    assert face_packet_shape["implementation_fail_closed_packet_frozen"] is True
    assert face_packet_shape["packed_face_proxy_buffer_contract_ready"] is True
    assert face_packet_shape["face_orientation_index_contract_ready"] is True
    assert face_packet_shape["cpml_non_cpml_initialization_contract_ready"] is True
    assert face_packet_shape["proxy_face_packet_state_contract_ready"] is True
    assert face_packet_shape["implementation_contract_ready"] is True
    packed_buffer_contract = face_packet_shape["packed_buffer_contract"]
    assert packed_buffer_contract["state_extension_target"] == (
        "_PrivateInterfaceOwnerState"
    )
    assert packed_buffer_contract["buffer_layout"] == (
        "static_packed_face_local_proxy_distribution"
    )
    assert (
        packed_buffer_contract[
            "heterogeneous_faces_supported_by_offsets_and_masks"
        ]
        is True
    )
    assert packed_buffer_contract["fixed_jax_pytree_shapes"] is True
    face_index_contract = face_packet_shape["face_index_contract"]
    assert face_index_contract["face_order_source"] == "FACE_ORIENTATIONS"
    assert face_index_contract["active_face_order_uses_existing_active_faces"] is True
    assert face_index_contract["x_y_z_face_shapes_supported"] is True
    assert face_index_contract["requires_public_config"] is False
    assert face_index_contract["imports_benchmark_fixture_metadata"] is False
    initialization_contract = face_packet_shape["initialization_contract"]
    assert initialization_contract["non_cpml_initialization_boundary"] == (
        "_init_private_interface_owner_state"
    )
    assert initialization_contract["cpml_initialization_boundary"] == (
        "_init_private_interface_owner_state"
    )
    assert initialization_contract["jit_runner_initialization_boundary"] == (
        "_init_private_interface_owner_state"
    )
    assert initialization_contract["cpml_non_cpml_shape_symmetry_required"] is True
    assert initialization_contract["benchmark_plane_dft_observable_imported"] is False
    assert face_packet_shape["packed_face_packet_state_required"] is True
    assert (
        face_packet_shape["heterogeneous_faces_supported_by_offsets_and_masks"] is True
    )
    assert face_packet_shape["fixed_jax_pytree_shapes"] is True
    assert face_packet_shape["derives_from_face_orientations"] is True
    assert face_packet_shape["production_patch_applied"] is False
    assert face_packet_shape["solver_behavior_changed"] is False
    assert face_packet_shape["field_update_behavior_changed"] is False
    assert face_packet_shape["runner_behavior_changed"] is False
    assert face_packet_shape["new_solver_hunk_retained"] is False
    assert face_packet_shape["true_rt_readiness_unlocked"] is False
    assert (
        face_packet_shape[
            "next_lane_requires_face_packet_state_shape_implementation"
        ]
        is True
    )
    face_packet_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in face_packet_shape["candidate_ladder"]
    }
    assert face_packet_candidates[
        "Y1_packed_per_face_proxy_buffer_design"
    ]["design_component_ready"] is True
    assert face_packet_candidates[
        "Y2_face_orientation_index_metadata_design"
    ]["design_component_ready"] is True
    assert face_packet_candidates[
        "Y3_proxy_face_packet_state_contract"
    ]["accepted_candidate"] is True
    assert face_packet_candidates[
        "Y4_face_packet_state_shape_design_blocked"
    ]["accepted_candidate"] is False
    assert face_packet_shape["public_claim_allowed"] is False
    assert face_packet_shape["public_observable_promoted"] is False
    assert face_packet_shape["true_rt_public_observable_promoted"] is False
    assert face_packet_shape["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_observable_proxy_face_packet_state_shape_next_prerequisite"
        ]
        == face_packet_shape["next_prerequisite"]
    )
    face_packet_implementation = benchmark_gate[
        "private_plane_wave_observable_proxy_face_packet_state_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_observable_proxy_face_packet_state_implementation_status"
    ] == "private_plane_wave_proxy_face_packet_capture_hunk_retained_fixture_quality_pending"
    assert face_packet_implementation["terminal_outcome"] == (
        "private_plane_wave_proxy_face_packet_capture_hunk_retained_fixture_quality_pending"
    )
    assert face_packet_implementation["upstream_state_shape_status"] == (
        benchmark_gate[
            "private_plane_wave_observable_proxy_face_packet_state_shape_status"
        ]
    )
    assert face_packet_implementation["upstream_modal_implementation_status"] == (
        benchmark_gate["private_plane_wave_observable_proxy_modal_implementation_status"]
    )
    assert face_packet_implementation["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert (
        face_packet_implementation["candidate_ladder_declared_before_solver_edit"]
        is True
    )
    assert (
        face_packet_implementation["candidate_ladder_declared_before_slow_scoring"]
        is True
    )
    assert face_packet_implementation["candidate_count"] == 5
    assert face_packet_implementation["selected_candidate_id"] == (
        "Z3_face_packet_proxy_capture_hunk"
    )
    assert face_packet_implementation["baseline_metrics"] == (
        face_packet_shape["baseline_metrics"]
    )
    assert face_packet_implementation["thresholds"] == face_packet_shape["thresholds"]
    assert face_packet_implementation["baseline_metrics_preserved"] is True
    assert face_packet_implementation["thresholds_unchanged"] is True
    assert face_packet_implementation["state_shape_contract_ready"] is True
    assert face_packet_implementation["implementation_lane_executed"] is True
    assert face_packet_implementation["packed_owner_state_buffers_hunk_retained"] is True
    assert face_packet_implementation["face_proxy_reference_buffers_retained"] is True
    assert face_packet_implementation["face_proxy_weight_mask_buffers_retained"] is True
    assert (
        face_packet_implementation["face_packet_offset_length_buffers_retained"]
        is True
    )
    assert face_packet_implementation["face_packet_orientation_buffers_retained"] is True
    assert face_packet_implementation["cpml_initialization_hunk_retained"] is True
    assert face_packet_implementation["non_cpml_initialization_hunk_retained"] is True
    assert face_packet_implementation["jit_initialization_hunk_retained"] is True
    assert face_packet_implementation["cpml_non_cpml_shape_symmetry_retained"] is True
    assert face_packet_implementation["face_packet_proxy_capture_hunk_retained"] is True
    assert face_packet_implementation["capture_uses_private_face_samples_only"] is True
    assert face_packet_implementation["benchmark_plane_dft_observable_imported"] is False
    assert face_packet_implementation["solver_local_proxy_uses_plane_dft_monitor"] is False
    assert face_packet_implementation["production_patch_applied"] is True
    assert face_packet_implementation["solver_behavior_changed"] is True
    assert face_packet_implementation["field_update_behavior_changed"] is False
    assert face_packet_implementation["runner_behavior_changed"] is True
    assert face_packet_implementation["new_solver_hunk_retained"] is True
    assert face_packet_implementation["joint_phase_magnitude_improved"] is False
    assert face_packet_implementation["material_improvement_demonstrated"] is False
    assert face_packet_implementation["subgrid_vacuum_parity_scored"] is True
    assert face_packet_implementation["subgrid_vacuum_parity_passed"] is False
    assert face_packet_implementation["true_rt_readiness_unlocked"] is False
    assert (
        face_packet_implementation["next_lane_requires_modal_retry_with_packed_state"]
        is True
    )
    face_packet_implementation_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in face_packet_implementation["candidate_ladder"]
    }
    assert face_packet_implementation_candidates[
        "Z1_packed_owner_state_buffers_hunk"
    ]["packed_owner_state_buffers_hunk_retained"] is True
    assert face_packet_implementation_candidates[
        "Z1_packed_owner_state_buffers_hunk"
    ]["superseded_by"] == "Z3_face_packet_proxy_capture_hunk"
    assert face_packet_implementation_candidates[
        "Z2_cpml_non_cpml_jit_initialization_hunk"
    ]["cpml_non_cpml_shape_symmetry_retained"] is True
    assert face_packet_implementation_candidates[
        "Z2_cpml_non_cpml_jit_initialization_hunk"
    ]["superseded_by"] == "Z3_face_packet_proxy_capture_hunk"
    assert face_packet_implementation_candidates[
        "Z3_face_packet_proxy_capture_hunk"
    ]["accepted_candidate"] is True
    assert face_packet_implementation_candidates[
        "Z4_face_packet_state_shape_implementation_blocked"
    ]["accepted_candidate"] is False
    assert face_packet_implementation["public_claim_allowed"] is False
    assert face_packet_implementation["public_observable_promoted"] is False
    assert face_packet_implementation["true_rt_public_observable_promoted"] is False
    assert (
        face_packet_implementation["dft_flux_tfsf_port_sparameter_promoted"] is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_observable_proxy_face_packet_state_implementation_next_prerequisite"
        ]
        == face_packet_implementation["next_prerequisite"]
    )
    modal_retry_after_packet = benchmark_gate[
        "private_plane_wave_observable_proxy_modal_retry_after_face_packet"
    ]
    assert benchmark_gate[
        "private_plane_wave_observable_proxy_modal_retry_after_face_packet_status"
    ] == "private_plane_wave_observable_proxy_modal_retry_hunk_retained_fixture_quality_pending"
    assert modal_retry_after_packet["terminal_outcome"] == (
        "private_plane_wave_observable_proxy_modal_retry_hunk_retained_fixture_quality_pending"
    )
    assert modal_retry_after_packet[
        "upstream_face_packet_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_observable_proxy_face_packet_state_implementation_status"
    ]
    assert modal_retry_after_packet["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert (
        modal_retry_after_packet["candidate_ladder_declared_before_solver_edit"]
        is True
    )
    assert (
        modal_retry_after_packet["candidate_ladder_declared_before_slow_scoring"]
        is True
    )
    assert modal_retry_after_packet["candidate_count"] == 4
    assert modal_retry_after_packet["selected_candidate_id"] == (
        "A1_packed_state_modal_retry_hunk"
    )
    assert modal_retry_after_packet["thresholds"] == (
        face_packet_implementation["thresholds"]
    )
    assert modal_retry_after_packet["baseline_metrics_preserved"] is True
    assert modal_retry_after_packet["thresholds_unchanged"] is True
    assert modal_retry_after_packet["face_packet_state_hunk_retained"] is True
    assert modal_retry_after_packet["modal_retry_uses_packed_face_packet_state"] is True
    assert modal_retry_after_packet["lagged_owner_packet_required"] is True
    assert modal_retry_after_packet["packed_state_hunk_retained"] is True
    assert modal_retry_after_packet["modal_retry_hunk_retained"] is True
    assert modal_retry_after_packet["field_update_behavior_changed"] is True
    assert modal_retry_after_packet["uses_private_face_samples_only"] is True
    assert modal_retry_after_packet["benchmark_plane_dft_observable_imported"] is False
    assert modal_retry_after_packet["solver_local_proxy_uses_plane_dft_monitor"] is (
        False
    )
    assert modal_retry_after_packet["update_bound_guard_retained"] is True
    assert modal_retry_after_packet["coupling_bound_guard_retained"] is True
    assert modal_retry_after_packet["boundary_coexistence_guard_retained"] is True
    assert modal_retry_after_packet["cpml_non_cpml_shape_symmetry_retained"] is True
    assert modal_retry_after_packet["production_patch_applied"] is True
    assert modal_retry_after_packet["solver_behavior_changed"] is True
    assert modal_retry_after_packet["runner_behavior_changed"] is True
    assert modal_retry_after_packet["new_solver_hunk_retained"] is True
    assert modal_retry_after_packet["subgrid_vacuum_parity_scored"] is True
    assert modal_retry_after_packet["subgrid_vacuum_parity_passed"] is False
    assert modal_retry_after_packet["material_improvement_demonstrated"] is False
    assert modal_retry_after_packet["fixture_quality_pending"] is True
    assert modal_retry_after_packet["true_rt_readiness_unlocked"] is False
    assert (
        modal_retry_after_packet["next_lane_requires_modal_retry_parity_scoring"]
        is True
    )
    modal_retry_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in modal_retry_after_packet["candidate_ladder"]
    }
    assert modal_retry_candidates[
        "A1_packed_state_modal_retry_hunk"
    ]["accepted_candidate"] is True
    assert modal_retry_candidates[
        "A1_packed_state_modal_retry_hunk"
    ]["modal_retry_hunk_retained"] is True
    assert modal_retry_candidates[
        "A2_packed_state_modal_retry_joint_parity_score"
    ]["accepted_candidate"] is False
    assert modal_retry_candidates[
        "A3_modal_retry_after_face_packet_blocked"
    ]["accepted_candidate"] is False
    assert modal_retry_after_packet["public_claim_allowed"] is False
    assert modal_retry_after_packet["public_observable_promoted"] is False
    assert modal_retry_after_packet["true_rt_public_observable_promoted"] is False
    assert (
        modal_retry_after_packet["dft_flux_tfsf_port_sparameter_promoted"] is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_observable_proxy_modal_retry_after_face_packet_next_prerequisite"
        ]
        == modal_retry_after_packet["next_prerequisite"]
    )
    modal_retry_parity_scoring = benchmark_gate[
        "private_plane_wave_observable_proxy_modal_retry_parity_scoring"
    ]
    assert benchmark_gate[
        "private_plane_wave_observable_proxy_modal_retry_parity_scoring_status"
    ] == (
        "private_plane_wave_observable_proxy_modal_retry_hunk_insufficient_fixture_quality_pending"
    )
    assert modal_retry_parity_scoring["terminal_outcome"] == (
        "private_plane_wave_observable_proxy_modal_retry_hunk_insufficient_fixture_quality_pending"
    )
    assert modal_retry_parity_scoring["upstream_modal_retry_status"] == (
        benchmark_gate[
            "private_plane_wave_observable_proxy_modal_retry_after_face_packet_status"
        ]
    )
    assert modal_retry_parity_scoring["upstream_parity_status"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
    )
    assert (
        modal_retry_parity_scoring["candidate_ladder_declared_before_slow_scoring"]
        is True
    )
    assert modal_retry_parity_scoring["candidate_count"] == 5
    assert modal_retry_parity_scoring["selected_candidate_id"] == (
        "B4_modal_retry_hunk_insufficient_fixture_quality_pending"
    )
    assert modal_retry_parity_scoring["baseline_metrics"] == {
        "transverse_phase_spread_deg": 114.98593935742637,
        "transverse_magnitude_cv": 0.44908395350634805,
        "vacuum_relative_magnitude_error": 0.9010421890370277,
        "vacuum_phase_error_deg": 42.91350635096882,
        "usable_bins": 3,
    }
    assert modal_retry_parity_scoring["metrics"] == (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring"]["metrics"]
    )
    assert modal_retry_parity_scoring["baseline_metrics_preserved"] is True
    assert modal_retry_parity_scoring["thresholds_unchanged"] is True
    assert modal_retry_parity_scoring["modal_retry_hunk_retained"] is True
    assert modal_retry_parity_scoring["packed_state_hunk_retained"] is True
    assert modal_retry_parity_scoring["private_plane_wave_parity_scored"] is True
    assert modal_retry_parity_scoring["finite_reproducible_score"] is True
    assert modal_retry_parity_scoring["dominant_parity_blocker"] == (
        "transverse_phase_spread_deg"
    )
    assert (
        modal_retry_parity_scoring["material_improvement_decision"]["passed"]
        is False
    )
    assert modal_retry_parity_scoring["dominant_relative_improvement"] < 0.5
    assert modal_retry_parity_scoring["material_improvement_demonstrated"] is False
    assert modal_retry_parity_scoring["subgrid_vacuum_parity_scored"] is True
    assert modal_retry_parity_scoring["subgrid_vacuum_parity_passed"] is False
    assert modal_retry_parity_scoring["fixture_quality_pending"] is True
    assert modal_retry_parity_scoring["true_rt_readiness_unlocked"] is False
    assert (
        modal_retry_parity_scoring["next_lane_requires_failure_theory_redesign"]
        is True
    )
    assert modal_retry_parity_scoring["benchmark_plane_dft_observable_imported"] is (
        False
    )
    assert modal_retry_parity_scoring["solver_local_proxy_uses_plane_dft_monitor"] is (
        False
    )
    modal_retry_parity_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in modal_retry_parity_scoring["candidate_ladder"]
    }
    assert modal_retry_parity_candidates[
        "B1_private_plane_wave_parity_score_after_hunk"
    ]["finite_reproducible_score"] is True
    assert modal_retry_parity_candidates[
        "B1_private_plane_wave_parity_score_after_hunk"
    ]["superseded_by"] == "B4_modal_retry_hunk_insufficient_fixture_quality_pending"
    assert modal_retry_parity_candidates[
        "B2_material_improvement_acceptance"
    ]["material_improvement_demonstrated"] is False
    assert modal_retry_parity_candidates[
        "B3_parity_readiness_pass"
    ]["accepted_candidate"] is False
    assert modal_retry_parity_candidates[
        "B4_modal_retry_hunk_insufficient_fixture_quality_pending"
    ]["accepted_candidate"] is True
    assert modal_retry_parity_scoring["public_claim_allowed"] is False
    assert modal_retry_parity_scoring["public_observable_promoted"] is False
    assert modal_retry_parity_scoring["true_rt_public_observable_promoted"] is False
    assert (
        modal_retry_parity_scoring["dft_flux_tfsf_port_sparameter_promoted"] is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_observable_proxy_modal_retry_parity_scoring_next_prerequisite"
        ]
        == modal_retry_parity_scoring["next_prerequisite"]
    )
    modal_retry_failure_theory = benchmark_gate[
        "private_plane_wave_observable_proxy_modal_retry_failure_theory"
    ]
    assert benchmark_gate[
        "private_plane_wave_observable_proxy_modal_retry_failure_theory_status"
    ] == "private_plane_wave_modal_retry_failure_theory_redesign_contract_ready"
    assert modal_retry_failure_theory["terminal_outcome"] == (
        "private_plane_wave_modal_retry_failure_theory_redesign_contract_ready"
    )
    assert modal_retry_failure_theory["upstream_modal_retry_parity_status"] == (
        benchmark_gate[
            "private_plane_wave_observable_proxy_modal_retry_parity_scoring_status"
        ]
    )
    assert modal_retry_failure_theory["upstream_modal_retry_status"] == (
        benchmark_gate[
            "private_plane_wave_observable_proxy_modal_retry_after_face_packet_status"
        ]
    )
    assert (
        modal_retry_failure_theory["candidate_ladder_declared_before_solver_edit"]
        is True
    )
    assert (
        modal_retry_failure_theory["candidate_ladder_declared_before_slow_scoring"]
        is True
    )
    assert modal_retry_failure_theory["candidate_count"] == 5
    assert modal_retry_failure_theory["selected_candidate_id"] == (
        "C3_source_interface_ownership_redesign_contract"
    )
    assert modal_retry_failure_theory["baseline_metrics"] == (
        modal_retry_parity_scoring["baseline_metrics"]
    )
    assert modal_retry_failure_theory["metrics"] == (
        modal_retry_parity_scoring["metrics"]
    )
    assert modal_retry_failure_theory["thresholds"] == (
        modal_retry_parity_scoring["thresholds"]
    )
    assert modal_retry_failure_theory["baseline_metrics_preserved"] is True
    assert modal_retry_failure_theory["thresholds_unchanged"] is True
    assert (
        modal_retry_failure_theory["insufficient_packed_state_parity_packet_frozen"]
        is True
    )
    assert modal_retry_failure_theory["modal_retry_hunk_retained"] is True
    assert modal_retry_failure_theory["packed_state_hunk_retained"] is True
    assert modal_retry_failure_theory["phase_model_mismatch_theory_ready"] is True
    assert modal_retry_failure_theory["observable_basis_mismatch_theory_ready"] is True
    assert modal_retry_failure_theory["failure_theory_contract_ready"] is True
    assert (
        modal_retry_failure_theory[
            "source_interface_ownership_redesign_contract_ready"
        ]
        is True
    )
    assert modal_retry_failure_theory["production_patch_applied"] is False
    assert modal_retry_failure_theory["solver_behavior_changed"] is False
    assert modal_retry_failure_theory["field_update_behavior_changed"] is False
    assert modal_retry_failure_theory["new_solver_hunk_retained"] is False
    assert modal_retry_failure_theory["subgrid_vacuum_parity_scored"] is True
    assert modal_retry_failure_theory["subgrid_vacuum_parity_passed"] is False
    assert modal_retry_failure_theory["material_improvement_demonstrated"] is False
    assert modal_retry_failure_theory["fixture_quality_pending"] is True
    assert modal_retry_failure_theory["true_rt_readiness_unlocked"] is False
    assert (
        modal_retry_failure_theory["next_lane_requires_implementation_plan"] is True
    )
    modal_retry_failure_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in modal_retry_failure_theory["candidate_ladder"]
    }
    assert modal_retry_failure_candidates[
        "C1_lagged_packet_phase_model_mismatch_theory"
    ]["design_component_ready"] is True
    assert modal_retry_failure_candidates[
        "C1_lagged_packet_phase_model_mismatch_theory"
    ]["superseded_by"] == "C3_source_interface_ownership_redesign_contract"
    assert modal_retry_failure_candidates[
        "C2_observable_proxy_basis_mismatch_theory"
    ]["design_component_ready"] is True
    assert modal_retry_failure_candidates[
        "C2_observable_proxy_basis_mismatch_theory"
    ]["superseded_by"] == "C3_source_interface_ownership_redesign_contract"
    assert modal_retry_failure_candidates[
        "C3_source_interface_ownership_redesign_contract"
    ]["accepted_candidate"] is True
    assert modal_retry_failure_candidates[
        "C4_failure_theory_blocked_no_public_promotion"
    ]["accepted_candidate"] is False
    assert modal_retry_failure_theory["public_claim_allowed"] is False
    assert modal_retry_failure_theory["public_observable_promoted"] is False
    assert modal_retry_failure_theory["true_rt_public_observable_promoted"] is False
    assert (
        modal_retry_failure_theory["dft_flux_tfsf_port_sparameter_promoted"] is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_observable_proxy_modal_retry_failure_theory_next_prerequisite"
        ]
        == modal_retry_failure_theory["next_prerequisite"]
    )
    modal_retry_redesign_implementation = benchmark_gate[
        "private_plane_wave_observable_proxy_modal_retry_redesign_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_observable_proxy_modal_retry_redesign_implementation_status"
    ] == "no_private_plane_wave_observable_proxy_modal_retry_redesign_implementation"
    assert modal_retry_redesign_implementation["terminal_outcome"] == (
        "no_private_plane_wave_observable_proxy_modal_retry_redesign_implementation"
    )
    assert modal_retry_redesign_implementation["upstream_failure_theory_status"] == (
        benchmark_gate[
            "private_plane_wave_observable_proxy_modal_retry_failure_theory_status"
        ]
    )
    assert modal_retry_redesign_implementation[
        "upstream_modal_retry_parity_status"
    ] == benchmark_gate[
        "private_plane_wave_observable_proxy_modal_retry_parity_scoring_status"
    ]
    assert (
        modal_retry_redesign_implementation[
            "candidate_ladder_declared_before_solver_edit"
        ]
        is True
    )
    assert (
        modal_retry_redesign_implementation[
            "candidate_ladder_declared_before_slow_scoring"
        ]
        is True
    )
    assert modal_retry_redesign_implementation["candidate_count"] == 5
    assert modal_retry_redesign_implementation["selected_candidate_id"] == (
        "D4_redesign_implementation_blocked"
    )
    assert modal_retry_redesign_implementation["baseline_metrics"] == (
        modal_retry_failure_theory["baseline_metrics"]
    )
    assert modal_retry_redesign_implementation["metrics"] == (
        modal_retry_failure_theory["metrics"]
    )
    assert modal_retry_redesign_implementation["thresholds"] == (
        modal_retry_failure_theory["thresholds"]
    )
    assert modal_retry_redesign_implementation["baseline_metrics_preserved"] is True
    assert modal_retry_redesign_implementation["thresholds_unchanged"] is True
    assert modal_retry_redesign_implementation["failure_theory_contract_ready"] is True
    assert modal_retry_redesign_implementation["implementation_lane_executed"] is True
    assert (
        modal_retry_redesign_implementation[
            "source_interface_ownership_state_required"
        ]
        is True
    )
    assert (
        modal_retry_redesign_implementation[
            "incident_normalized_modal_basis_required"
        ]
        is True
    )
    assert (
        modal_retry_redesign_implementation[
            "propagation_aware_modal_basis_required"
        ]
        is True
    )
    assert modal_retry_redesign_implementation[
        "source_interface_state_gap"
    ]["incident_source_packet_available"] is False
    assert modal_retry_redesign_implementation[
        "source_interface_state_gap"
    ]["bounded_solver_hunk_requires_prior_state_shape_design"] is True
    assert (
        modal_retry_redesign_implementation["source_interface_ownership_hunk_retained"]
        is False
    )
    assert (
        modal_retry_redesign_implementation[
            "incident_normalized_modal_basis_hunk_retained"
        ]
        is False
    )
    assert (
        modal_retry_redesign_implementation[
            "propagation_aware_modal_basis_hunk_retained"
        ]
        is False
    )
    assert modal_retry_redesign_implementation["modal_retry_hunk_retained"] is True
    assert modal_retry_redesign_implementation["packed_state_hunk_retained"] is True
    assert modal_retry_redesign_implementation["no_bounded_hunk_accepted"] is True
    assert modal_retry_redesign_implementation["production_patch_applied"] is False
    assert modal_retry_redesign_implementation["solver_behavior_changed"] is False
    assert modal_retry_redesign_implementation["field_update_behavior_changed"] is False
    assert modal_retry_redesign_implementation["new_solver_hunk_retained"] is False
    assert (
        modal_retry_redesign_implementation[
            "benchmark_plane_dft_observable_imported"
        ]
        is False
    )
    assert (
        modal_retry_redesign_implementation[
            "solver_local_proxy_uses_plane_dft_monitor"
        ]
        is False
    )
    assert modal_retry_redesign_implementation["subgrid_vacuum_parity_scored"] is True
    assert modal_retry_redesign_implementation[
        "subgrid_vacuum_parity_passed"
    ] is False
    assert modal_retry_redesign_implementation[
        "material_improvement_demonstrated"
    ] is False
    assert modal_retry_redesign_implementation["fixture_quality_pending"] is True
    assert modal_retry_redesign_implementation["true_rt_readiness_unlocked"] is False
    assert (
        modal_retry_redesign_implementation[
            "next_lane_requires_source_interface_owner_state_shape_design"
        ]
        is True
    )
    modal_retry_redesign_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in modal_retry_redesign_implementation["candidate_ladder"]
    }
    assert modal_retry_redesign_candidates[
        "D1_propagation_aware_incident_normalized_modal_basis_owner_hunk"
    ]["attempted_in_this_lane"] is True
    assert modal_retry_redesign_candidates[
        "D1_propagation_aware_incident_normalized_modal_basis_owner_hunk"
    ]["propagation_aware_modal_basis_hunk_retained"] is False
    assert modal_retry_redesign_candidates[
        "D2_private_parity_material_improvement_score"
    ]["finite_reproducible_score"] is True
    assert modal_retry_redesign_candidates[
        "D3_private_true_rt_readiness_pass"
    ]["accepted_candidate"] is False
    assert modal_retry_redesign_candidates[
        "D4_redesign_implementation_blocked"
    ]["accepted_candidate"] is True
    assert modal_retry_redesign_implementation["public_claim_allowed"] is False
    assert modal_retry_redesign_implementation["public_observable_promoted"] is False
    assert (
        modal_retry_redesign_implementation["true_rt_public_observable_promoted"]
        is False
    )
    assert (
        modal_retry_redesign_implementation[
            "dft_flux_tfsf_port_sparameter_promoted"
        ]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_observable_proxy_modal_retry_redesign_implementation_next_prerequisite"
        ]
        == modal_retry_redesign_implementation["next_prerequisite"]
    )
    source_interface_shape = benchmark_gate[
        "private_plane_wave_source_interface_ownership_state_shape"
    ]
    assert benchmark_gate[
        "private_plane_wave_source_interface_ownership_state_shape_status"
    ] == "private_plane_wave_source_interface_owner_state_shape_contract_ready"
    assert source_interface_shape["terminal_outcome"] == (
        "private_plane_wave_source_interface_owner_state_shape_contract_ready"
    )
    assert source_interface_shape["upstream_redesign_implementation_status"] == (
        benchmark_gate[
            "private_plane_wave_observable_proxy_modal_retry_redesign_implementation_status"
        ]
    )
    assert source_interface_shape["upstream_failure_theory_status"] == (
        benchmark_gate[
            "private_plane_wave_observable_proxy_modal_retry_failure_theory_status"
        ]
    )
    assert (
        source_interface_shape["candidate_ladder_declared_before_solver_edit"]
        is True
    )
    assert (
        source_interface_shape["candidate_ladder_declared_before_slow_scoring"]
        is True
    )
    assert source_interface_shape["candidate_count"] == 5
    assert source_interface_shape["selected_candidate_id"] == (
        "E3_combined_source_interface_state_shape_contract"
    )
    assert source_interface_shape["baseline_metrics"] == (
        modal_retry_redesign_implementation["baseline_metrics"]
    )
    assert source_interface_shape["metrics"] == (
        modal_retry_redesign_implementation["metrics"]
    )
    assert source_interface_shape["thresholds"] == (
        modal_retry_redesign_implementation["thresholds"]
    )
    assert source_interface_shape["baseline_metrics_preserved"] is True
    assert source_interface_shape["thresholds_unchanged"] is True
    assert source_interface_shape["redesign_implementation_blocker_preserved"] is True
    assert source_interface_shape["source_owner_packet_state_contract_ready"] is True
    assert (
        source_interface_shape["interface_owner_packet_separation_contract_ready"]
        is True
    )
    assert source_interface_shape[
        "source_interface_owner_state_shape_contract_ready"
    ] is True
    assert source_interface_shape["incident_normalizer_state_contract_ready"] is True
    assert source_interface_shape["fixed_jax_pytree_shapes"] is True
    assert (
        source_interface_shape["cpml_non_cpml_initialization_contract_ready"] is True
    )
    assert source_interface_shape["jit_runner_initialization_contract_ready"] is True
    assert source_interface_shape["private_state_only"] is True
    assert source_interface_shape["source_owner_packet_contract"][
        "public_tfsf_required"
    ] is False
    assert source_interface_shape["source_owner_packet_contract"][
        "benchmark_dft_required"
    ] is False
    assert source_interface_shape["interface_owner_separation_contract"][
        "source_packet_must_not_alias_interface_packet"
    ] is True
    assert source_interface_shape["combined_state_shape_contract"][
        "bounded_follow_up_implementation_surface"
    ] is True
    assert source_interface_shape["production_patch_applied"] is False
    assert source_interface_shape["solver_behavior_changed"] is False
    assert source_interface_shape["field_update_behavior_changed"] is False
    assert source_interface_shape["runner_behavior_changed"] is False
    assert source_interface_shape["new_solver_hunk_retained"] is False
    assert source_interface_shape["subgrid_vacuum_parity_scored"] is True
    assert source_interface_shape["subgrid_vacuum_parity_passed"] is False
    assert source_interface_shape["material_improvement_demonstrated"] is False
    assert source_interface_shape["fixture_quality_pending"] is True
    assert source_interface_shape["true_rt_readiness_unlocked"] is False
    assert (
        source_interface_shape[
            "next_lane_requires_source_interface_owner_state_shape_implementation"
        ]
        is True
    )
    source_interface_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in source_interface_shape["candidate_ladder"]
    }
    assert source_interface_candidates[
        "E1_source_owner_packet_state_contract"
    ]["design_component_ready"] is True
    assert source_interface_candidates[
        "E1_source_owner_packet_state_contract"
    ]["superseded_by"] == "E3_combined_source_interface_state_shape_contract"
    assert source_interface_candidates[
        "E2_interface_owner_packet_separation_contract"
    ]["design_component_ready"] is True
    assert source_interface_candidates[
        "E2_interface_owner_packet_separation_contract"
    ]["superseded_by"] == "E3_combined_source_interface_state_shape_contract"
    assert source_interface_candidates[
        "E3_combined_source_interface_state_shape_contract"
    ]["accepted_candidate"] is True
    assert source_interface_candidates[
        "E4_source_interface_state_shape_design_blocked"
    ]["accepted_candidate"] is False
    assert source_interface_shape["public_claim_allowed"] is False
    assert source_interface_shape["public_observable_promoted"] is False
    assert source_interface_shape["true_rt_public_observable_promoted"] is False
    assert source_interface_shape["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_source_interface_ownership_state_shape_next_prerequisite"
        ]
        == source_interface_shape["next_prerequisite"]
    )
    source_interface_impl = benchmark_gate[
        "private_plane_wave_source_interface_ownership_state_shape_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_source_interface_ownership_state_shape_implementation_status"
    ] == "private_plane_wave_source_interface_state_shape_hunk_retained_fixture_quality_pending"
    assert source_interface_impl["terminal_outcome"] == (
        "private_plane_wave_source_interface_state_shape_hunk_retained_fixture_quality_pending"
    )
    assert source_interface_impl["upstream_state_shape_status"] == (
        benchmark_gate["private_plane_wave_source_interface_ownership_state_shape_status"]
    )
    assert source_interface_impl["upstream_redesign_implementation_status"] == (
        benchmark_gate[
            "private_plane_wave_observable_proxy_modal_retry_redesign_implementation_status"
        ]
    )
    assert source_interface_impl["candidate_ladder_declared_before_solver_edit"] is True
    assert (
        source_interface_impl["candidate_ladder_declared_before_slow_scoring"] is True
    )
    assert source_interface_impl["candidate_count"] == 5
    assert source_interface_impl["selected_candidate_id"] == (
        "F3_combined_source_interface_state_shape_hunk"
    )
    assert source_interface_impl["baseline_metrics"] == (
        source_interface_shape["baseline_metrics"]
    )
    assert source_interface_impl["metrics"] == source_interface_shape["metrics"]
    assert source_interface_impl["thresholds"] == source_interface_shape["thresholds"]
    assert source_interface_impl["baseline_metrics_preserved"] is True
    assert source_interface_impl["thresholds_unchanged"] is True
    assert source_interface_impl["state_shape_contract_ready"] is True
    assert source_interface_impl["implementation_lane_executed"] is True
    assert source_interface_impl["source_owner_buffers_hunk_retained"] is True
    assert source_interface_impl["source_owner_reference_buffers_retained"] is True
    assert source_interface_impl["source_owner_weight_mask_buffers_retained"] is True
    assert source_interface_impl[
        "source_incident_normalizer_buffers_retained"
    ] is True
    assert source_interface_impl[
        "source_packet_offset_length_buffers_retained"
    ] is True
    assert source_interface_impl["source_packet_orientation_buffers_retained"] is True
    assert source_interface_impl["interface_owner_packet_preserved"] is True
    assert source_interface_impl["source_interface_buffers_do_not_alias"] is True
    assert source_interface_impl[
        "source_buffers_not_updated_by_interface_scan"
    ] is True
    assert source_interface_impl["source_interface_state_shape_hunk_retained"] is True
    assert source_interface_impl["cpml_initialization_hunk_retained"] is True
    assert source_interface_impl["non_cpml_initialization_hunk_retained"] is True
    assert source_interface_impl["jit_initialization_hunk_retained"] is True
    assert source_interface_impl["cpml_non_cpml_shape_symmetry_retained"] is True
    assert source_interface_impl["production_patch_applied"] is True
    assert source_interface_impl["solver_behavior_changed"] is True
    assert source_interface_impl["field_update_behavior_changed"] is False
    assert source_interface_impl["runner_behavior_changed"] is True
    assert source_interface_impl["new_solver_hunk_retained"] is True
    assert source_interface_impl["benchmark_plane_dft_observable_imported"] is False
    assert source_interface_impl["solver_local_proxy_uses_plane_dft_monitor"] is False
    assert source_interface_impl["subgrid_vacuum_parity_scored"] is True
    assert source_interface_impl["subgrid_vacuum_parity_passed"] is False
    assert source_interface_impl["material_improvement_demonstrated"] is False
    assert source_interface_impl["fixture_quality_pending"] is True
    assert source_interface_impl["true_rt_readiness_unlocked"] is False
    assert (
        source_interface_impl[
            "next_lane_requires_propagation_aware_modal_retry_implementation"
        ]
        is True
    )
    source_interface_impl_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in source_interface_impl["candidate_ladder"]
    }
    assert source_interface_impl_candidates[
        "F1_source_owner_buffers_hunk"
    ]["source_owner_buffers_hunk_retained"] is True
    assert source_interface_impl_candidates[
        "F1_source_owner_buffers_hunk"
    ]["superseded_by"] == "F3_combined_source_interface_state_shape_hunk"
    assert source_interface_impl_candidates[
        "F2_interface_owner_separation_hunk"
    ]["source_interface_buffers_do_not_alias"] is True
    assert source_interface_impl_candidates[
        "F2_interface_owner_separation_hunk"
    ]["superseded_by"] == "F3_combined_source_interface_state_shape_hunk"
    assert source_interface_impl_candidates[
        "F3_combined_source_interface_state_shape_hunk"
    ]["accepted_candidate"] is True
    assert source_interface_impl_candidates[
        "F4_source_interface_state_shape_implementation_blocked"
    ]["accepted_candidate"] is False
    assert source_interface_impl["public_claim_allowed"] is False
    assert source_interface_impl["public_observable_promoted"] is False
    assert source_interface_impl["true_rt_public_observable_promoted"] is False
    assert source_interface_impl["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_source_interface_ownership_state_shape_implementation_next_prerequisite"
        ]
        == source_interface_impl["next_prerequisite"]
    )
    propagation_retry_impl = benchmark_gate[
        "private_plane_wave_propagation_aware_modal_retry_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_propagation_aware_modal_retry_implementation_status"
    ] == (
        "private_plane_wave_propagation_aware_modal_basis_hunk_retained_fixture_quality_pending"
    )
    assert propagation_retry_impl["terminal_outcome"] == (
        "private_plane_wave_propagation_aware_modal_basis_hunk_retained_fixture_quality_pending"
    )
    assert propagation_retry_impl["upstream_state_shape_implementation_status"] == (
        benchmark_gate[
            "private_plane_wave_source_interface_ownership_state_shape_implementation_status"
        ]
    )
    assert propagation_retry_impl["candidate_ladder_declared_before_solver_edit"] is True
    assert (
        propagation_retry_impl["candidate_ladder_declared_before_slow_scoring"]
        is True
    )
    assert propagation_retry_impl["candidate_count"] == 5
    assert propagation_retry_impl["selected_candidate_id"] == (
        "G1_source_normalized_modal_basis_hunk"
    )
    assert propagation_retry_impl["baseline_metrics"] == (
        source_interface_impl["baseline_metrics"]
    )
    assert propagation_retry_impl["metrics"] == source_interface_impl["metrics"]
    assert propagation_retry_impl["thresholds"] == source_interface_impl["thresholds"]
    assert propagation_retry_impl["baseline_metrics_preserved"] is True
    assert propagation_retry_impl["thresholds_unchanged"] is True
    assert propagation_retry_impl["state_shape_implementation_ready"] is True
    assert propagation_retry_impl["implementation_lane_executed"] is True
    assert propagation_retry_impl["source_interface_state_shape_hunk_retained"] is True
    assert propagation_retry_impl["source_owner_packet_used"] is True
    assert propagation_retry_impl["interface_owner_packet_used"] is True
    assert propagation_retry_impl["incident_normalizer_packet_used"] is True
    assert propagation_retry_impl["source_interface_buffers_do_not_alias"] is True
    assert (
        propagation_retry_impl["propagation_aware_modal_basis_hunk_retained"]
        is True
    )
    assert propagation_retry_impl["bounded_field_update_hunk_retained"] is True
    assert propagation_retry_impl["no_op_without_source_packet"] is True
    assert propagation_retry_impl["requires_source_packet_population"] is True
    assert propagation_retry_impl["production_patch_applied"] is True
    assert propagation_retry_impl["solver_behavior_changed"] is True
    assert propagation_retry_impl["field_update_behavior_changed"] is True
    assert propagation_retry_impl["runner_behavior_changed"] is False
    assert propagation_retry_impl["new_solver_hunk_retained"] is True
    assert propagation_retry_impl["benchmark_plane_dft_observable_imported"] is False
    assert propagation_retry_impl["solver_local_proxy_uses_plane_dft_monitor"] is False
    assert propagation_retry_impl["subgrid_vacuum_parity_scored"] is True
    assert propagation_retry_impl["subgrid_vacuum_parity_passed"] is False
    assert propagation_retry_impl["material_improvement_demonstrated"] is False
    assert propagation_retry_impl["fixture_quality_pending"] is True
    assert propagation_retry_impl["true_rt_readiness_unlocked"] is False
    assert (
        propagation_retry_impl[
            "next_lane_requires_propagation_aware_modal_retry_parity_scoring"
        ]
        is True
    )
    propagation_retry_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in propagation_retry_impl["candidate_ladder"]
    }
    assert propagation_retry_candidates[
        "G1_source_normalized_modal_basis_hunk"
    ]["accepted_candidate"] is True
    assert propagation_retry_candidates[
        "G1_source_normalized_modal_basis_hunk"
    ]["bounded_field_update_hunk_retained"] is True
    assert propagation_retry_candidates[
        "G2_private_parity_material_improvement_score"
    ]["material_improvement_demonstrated"] is False
    assert propagation_retry_candidates[
        "G4_propagation_aware_modal_retry_implementation_blocked"
    ]["accepted_candidate"] is False
    assert propagation_retry_impl["public_claim_allowed"] is False
    assert propagation_retry_impl["public_observable_promoted"] is False
    assert propagation_retry_impl["true_rt_public_observable_promoted"] is False
    assert propagation_retry_impl["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_propagation_aware_modal_retry_implementation_next_prerequisite"
        ]
        == propagation_retry_impl["next_prerequisite"]
    )
    propagation_retry_parity = benchmark_gate[
        "private_plane_wave_propagation_aware_modal_retry_parity_scoring"
    ]
    assert benchmark_gate[
        "private_plane_wave_propagation_aware_modal_retry_parity_scoring_status"
    ] == (
        "private_plane_wave_propagation_aware_modal_retry_parity_scored_fixture_quality_pending"
    )
    assert propagation_retry_parity["terminal_outcome"] == (
        "private_plane_wave_propagation_aware_modal_retry_parity_scored_fixture_quality_pending"
    )
    assert propagation_retry_parity[
        "upstream_propagation_aware_modal_retry_status"
    ] == benchmark_gate[
        "private_plane_wave_propagation_aware_modal_retry_implementation_status"
    ]
    assert propagation_retry_parity["candidate_ladder_declared_before_solver_edit"] is True
    assert (
        propagation_retry_parity["candidate_ladder_declared_before_slow_scoring"]
        is True
    )
    assert propagation_retry_parity["candidate_count"] == 5
    assert propagation_retry_parity["selected_candidate_id"] == (
        "H1_finite_private_parity_score"
    )
    assert propagation_retry_parity["baseline_metrics"] == (
        propagation_retry_impl["baseline_metrics"]
    )
    assert propagation_retry_parity["metrics"] == propagation_retry_impl["metrics"]
    assert propagation_retry_parity["thresholds"] == propagation_retry_impl["thresholds"]
    assert propagation_retry_parity["baseline_metrics_preserved"] is True
    assert propagation_retry_parity["thresholds_unchanged"] is True
    assert propagation_retry_parity["source_normalized_modal_basis_hunk_retained"] is True
    assert propagation_retry_parity["bounded_field_update_hunk_retained"] is True
    assert propagation_retry_parity["finite_reproducible_score"] is True
    assert propagation_retry_parity["parity_scoring_lane_executed"] is True
    assert propagation_retry_parity["material_improvement_demonstrated"] is False
    assert (
        propagation_retry_parity["material_improvement_decision"][
            "classification_decision"
        ]
        == "inconclusive"
    )
    assert propagation_retry_parity["source_owner_incident_packet_populated"] is False
    assert propagation_retry_parity["source_packet_population_required"] is True
    assert propagation_retry_parity["fixture_quality_pending"] is True
    assert propagation_retry_parity["subgrid_vacuum_parity_scored"] is True
    assert propagation_retry_parity["subgrid_vacuum_parity_passed"] is False
    assert propagation_retry_parity["true_rt_readiness_unlocked"] is False
    assert propagation_retry_parity["production_patch_applied"] is False
    assert propagation_retry_parity["solver_behavior_changed"] is False
    assert propagation_retry_parity["field_update_behavior_changed"] is False
    assert propagation_retry_parity["new_solver_hunk_retained"] is False
    assert propagation_retry_parity["benchmark_plane_dft_observable_imported"] is False
    assert propagation_retry_parity["solver_local_proxy_uses_plane_dft_monitor"] is False
    assert (
        propagation_retry_parity[
            "next_lane_requires_source_owner_incident_packet_population"
        ]
        is True
    )
    propagation_retry_parity_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in propagation_retry_parity["candidate_ladder"]
    }
    assert propagation_retry_parity_candidates[
        "H1_finite_private_parity_score"
    ]["accepted_candidate"] is True
    assert propagation_retry_parity_candidates[
        "H2_material_improvement_gate"
    ]["material_improvement_demonstrated"] is False
    assert propagation_retry_parity_candidates[
        "H4_parity_scoring_blocked"
    ]["accepted_candidate"] is False
    assert propagation_retry_parity["public_claim_allowed"] is False
    assert propagation_retry_parity["public_observable_promoted"] is False
    assert propagation_retry_parity["true_rt_public_observable_promoted"] is False
    assert propagation_retry_parity["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_propagation_aware_modal_retry_parity_scoring_next_prerequisite"
        ]
        == propagation_retry_parity["next_prerequisite"]
    )
    source_owner_population_design = benchmark_gate[
        "private_plane_wave_source_owner_incident_packet_population_design"
    ]
    assert benchmark_gate[
        "private_plane_wave_source_owner_incident_packet_population_design_status"
    ] == "private_plane_wave_source_owner_incident_packet_population_contract_ready"
    assert source_owner_population_design["terminal_outcome"] == (
        "private_plane_wave_source_owner_incident_packet_population_contract_ready"
    )
    assert source_owner_population_design["upstream_parity_scoring_status"] == (
        benchmark_gate[
            "private_plane_wave_propagation_aware_modal_retry_parity_scoring_status"
        ]
    )
    assert (
        source_owner_population_design["candidate_ladder_declared_before_implementation"]
        is True
    )
    assert (
        source_owner_population_design["candidate_ladder_declared_before_slow_scoring"]
        is True
    )
    assert source_owner_population_design["candidate_count"] == 5
    assert source_owner_population_design["selected_candidate_id"] == (
        "I3_combined_source_owner_incident_packet_population_contract"
    )
    assert source_owner_population_design["baseline_metrics"] == (
        propagation_retry_parity["baseline_metrics"]
    )
    assert source_owner_population_design["metrics"] == propagation_retry_parity["metrics"]
    assert source_owner_population_design["thresholds"] == (
        propagation_retry_parity["thresholds"]
    )
    assert source_owner_population_design["baseline_metrics_preserved"] is True
    assert source_owner_population_design["thresholds_unchanged"] is True
    assert (
        source_owner_population_design[
            "source_owner_incident_packet_population_contract_ready"
        ]
        is True
    )
    assert source_owner_population_design["source_owner_incident_packet_populated"] is False
    assert (
        source_owner_population_design[
            "source_owner_incident_packet_population_implemented"
        ]
        is False
    )
    assert source_owner_population_design["source_packet_population_private_only"] is True
    assert source_owner_population_design["source_interface_buffers_do_not_alias"] is True
    assert (
        source_owner_population_design["source_incident_normalizer_contract_ready"]
        is True
    )
    assert (
        source_owner_population_design["source_packet_offset_length_contract_ready"]
        is True
    )
    assert (
        source_owner_population_design["source_packet_orientation_contract_ready"]
        is True
    )
    assert source_owner_population_design["cpml_non_cpml_timing_contract_ready"] is True
    assert (
        source_owner_population_design["jit_runner_initialization_contract_ready"]
        is True
    )
    packet_contract = source_owner_population_design["packet_contract"]
    assert packet_contract["source_packet_must_not_alias_interface_packet"] is True
    assert packet_contract["packet_shape_matches_interface_owner_packet"] is True
    assert packet_contract["public_tfsf_required"] is False
    timing_contract = source_owner_population_design["timing_contract"]
    assert (
        "_apply_propagation_aware_modal_retry_face_helper"
        in timing_contract["population_slot"]
    )
    assert timing_contract["modal_retry_consumes_populated_source_packet"] is True
    assert timing_contract["requires_private_post_h_hook"] is False
    assert timing_contract["requires_private_post_e_hook"] is False
    assert source_owner_population_design["implementation_lane_executed"] is False
    assert source_owner_population_design["production_patch_applied"] is False
    assert source_owner_population_design["solver_behavior_changed"] is False
    assert source_owner_population_design["field_update_behavior_changed"] is False
    assert source_owner_population_design["new_solver_hunk_retained"] is False
    assert (
        source_owner_population_design["benchmark_plane_dft_observable_imported"]
        is False
    )
    assert source_owner_population_design["solver_local_proxy_uses_plane_dft_monitor"] is False
    assert source_owner_population_design["fixture_quality_pending"] is True
    assert source_owner_population_design["true_rt_readiness_unlocked"] is False
    assert (
        source_owner_population_design[
            "next_lane_requires_source_owner_incident_packet_population_implementation"
        ]
        is True
    )
    source_owner_design_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in source_owner_population_design["candidate_ladder"]
    }
    assert source_owner_design_candidates[
        "I1_private_source_owner_incident_packet_contract"
    ]["design_component_ready"] is True
    assert source_owner_design_candidates[
        "I1_private_source_owner_incident_packet_contract"
    ]["superseded_by"] == (
        "I3_combined_source_owner_incident_packet_population_contract"
    )
    assert source_owner_design_candidates[
        "I2_private_source_owner_timing_contract"
    ]["design_component_ready"] is True
    assert source_owner_design_candidates[
        "I3_combined_source_owner_incident_packet_population_contract"
    ]["accepted_candidate"] is True
    assert source_owner_design_candidates[
        "I4_source_owner_incident_packet_population_design_blocked"
    ]["accepted_candidate"] is False
    assert source_owner_population_design["public_claim_allowed"] is False
    assert source_owner_population_design["public_observable_promoted"] is False
    assert source_owner_population_design["true_rt_public_observable_promoted"] is False
    assert (
        source_owner_population_design["dft_flux_tfsf_port_sparameter_promoted"]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_source_owner_incident_packet_population_design_next_prerequisite"
        ]
        == source_owner_population_design["next_prerequisite"]
    )
    source_owner_population_impl = benchmark_gate[
        "private_plane_wave_source_owner_incident_packet_population_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_source_owner_incident_packet_population_implementation_status"
    ] == (
        "private_plane_wave_source_owner_incident_packet_population_hunk_retained_fixture_quality_pending"
    )
    assert source_owner_population_impl["terminal_outcome"] == (
        "private_plane_wave_source_owner_incident_packet_population_hunk_retained_fixture_quality_pending"
    )
    assert source_owner_population_impl["upstream_population_design_status"] == (
        benchmark_gate[
            "private_plane_wave_source_owner_incident_packet_population_design_status"
        ]
    )
    assert source_owner_population_impl["candidate_ladder_declared_before_solver_edit"]
    assert source_owner_population_impl["candidate_ladder_declared_before_slow_scoring"]
    assert source_owner_population_impl["candidate_count"] == 5
    assert source_owner_population_impl["selected_candidate_id"] == (
        "J2_cpml_non_cpml_source_owner_population_wiring"
    )
    assert source_owner_population_impl["baseline_metrics"] == (
        source_owner_population_design["baseline_metrics"]
    )
    assert source_owner_population_impl["metrics"] == (
        source_owner_population_design["metrics"]
    )
    assert source_owner_population_impl["thresholds"] == (
        source_owner_population_design["thresholds"]
    )
    assert source_owner_population_impl["baseline_metrics_preserved"]
    assert source_owner_population_impl["thresholds_unchanged"]
    assert source_owner_population_impl["population_design_contract_ready"]
    assert source_owner_population_impl["implementation_lane_executed"]
    assert source_owner_population_impl[
        "source_owner_incident_packet_population_hunk_retained"
    ]
    assert source_owner_population_impl["source_owner_incident_packet_populated"]
    assert source_owner_population_impl[
        "source_owner_incident_packet_population_implemented"
    ]
    assert source_owner_population_impl["source_packet_updates_only_source_owner_fields"]
    assert source_owner_population_impl["interface_owner_packet_preserved"]
    assert source_owner_population_impl["source_interface_buffers_do_not_alias"]
    assert source_owner_population_impl[
        "source_incident_normalizer_real_set_to_mask"
    ]
    assert source_owner_population_impl["source_incident_normalizer_imag_zero"]
    assert source_owner_population_impl["cpml_source_population_wiring_retained"]
    assert source_owner_population_impl["non_cpml_source_population_wiring_retained"]
    assert source_owner_population_impl["cpml_non_cpml_wiring_symmetry_retained"]
    assert source_owner_population_impl[
        "source_population_before_propagation_aware_modal_retry"
    ]
    assert source_owner_population_impl["source_population_before_interface_owner_scan"]
    impl_contract = source_owner_population_impl["implementation_contract"]
    assert impl_contract["solver_helper"] == "_update_private_source_owner_state_from_scan"
    assert "_apply_propagation_aware_modal_retry_face_helper" in impl_contract[
        "population_slot"
    ]
    assert impl_contract["source_packet_updates_only_source_owner_fields"]
    assert source_owner_population_impl["production_patch_applied"]
    assert source_owner_population_impl["solver_behavior_changed"]
    assert source_owner_population_impl["field_update_behavior_changed"] is False
    assert source_owner_population_impl["runner_behavior_changed"] is False
    assert source_owner_population_impl["new_solver_hunk_retained"]
    assert source_owner_population_impl["benchmark_plane_dft_observable_imported"] is False
    assert source_owner_population_impl["solver_local_proxy_uses_plane_dft_monitor"] is False
    assert source_owner_population_impl["fixture_quality_pending"]
    assert source_owner_population_impl["subgrid_vacuum_parity_passed"] is False
    assert source_owner_population_impl["true_rt_readiness_unlocked"] is False
    assert source_owner_population_impl[
        "next_lane_requires_source_populated_modal_retry_parity_scoring"
    ]
    source_owner_impl_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in source_owner_population_impl["candidate_ladder"]
    }
    assert source_owner_impl_candidates[
        "J1_source_owner_packet_population_helper"
    ]["source_owner_incident_packet_population_hunk_retained"]
    assert source_owner_impl_candidates[
        "J1_source_owner_packet_population_helper"
    ]["superseded_by"] == "J2_cpml_non_cpml_source_owner_population_wiring"
    assert source_owner_impl_candidates[
        "J2_cpml_non_cpml_source_owner_population_wiring"
    ]["accepted_candidate"]
    assert source_owner_impl_candidates[
        "J4_source_owner_population_implementation_blocked"
    ]["accepted_candidate"] is False
    assert source_owner_population_impl["public_claim_allowed"] is False
    assert source_owner_population_impl["public_observable_promoted"] is False
    assert source_owner_population_impl["true_rt_public_observable_promoted"] is False
    assert (
        source_owner_population_impl["dft_flux_tfsf_port_sparameter_promoted"]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_source_owner_incident_packet_population_implementation_next_prerequisite"
        ]
        == source_owner_population_impl["next_prerequisite"]
    )
    source_populated_parity = benchmark_gate[
        "private_plane_wave_source_populated_propagation_aware_modal_retry_parity_scoring"
    ]
    assert benchmark_gate[
        "private_plane_wave_source_populated_propagation_aware_modal_retry_parity_scoring_status"
    ] == (
        "private_plane_wave_source_populated_propagation_aware_modal_retry_hunk_insufficient_fixture_quality_pending"
    )
    assert source_populated_parity["terminal_outcome"] == (
        "private_plane_wave_source_populated_propagation_aware_modal_retry_hunk_insufficient_fixture_quality_pending"
    )
    assert source_populated_parity["upstream_source_owner_population_status"] == (
        benchmark_gate[
            "private_plane_wave_source_owner_incident_packet_population_implementation_status"
        ]
    )
    assert source_populated_parity["candidate_ladder_declared_before_solver_edit"]
    assert source_populated_parity["candidate_ladder_declared_before_slow_scoring"]
    assert source_populated_parity["candidate_count"] == 5
    assert source_populated_parity["selected_candidate_id"] == (
        "K1_finite_source_populated_private_parity_score"
    )
    assert source_populated_parity["baseline_metrics"] == (
        source_owner_population_impl["baseline_metrics"]
    )
    assert source_populated_parity["metrics"] == source_owner_population_impl["metrics"]
    assert source_populated_parity["thresholds"] == (
        source_owner_population_impl["thresholds"]
    )
    assert source_populated_parity["baseline_metrics_preserved"]
    assert source_populated_parity["thresholds_unchanged"]
    assert source_populated_parity["source_owner_population_hunk_retained"]
    assert source_populated_parity["source_owner_incident_packet_populated"]
    assert source_populated_parity["source_packet_consumed_by_modal_retry"]
    consumption_contract = source_populated_parity[
        "source_packet_consumption_contract"
    ]
    assert consumption_contract["population_helper"] == (
        "_update_private_source_owner_state_from_scan"
    )
    assert consumption_contract["consumer_helper"] == (
        "_apply_propagation_aware_modal_retry_face_helper"
    )
    assert consumption_contract["source_packet_populated_before_consumer"]
    assert consumption_contract["source_packet_consumed_by_modal_retry"]
    assert consumption_contract["public_observable_required"] is False
    assert source_populated_parity[
        "source_population_before_propagation_aware_modal_retry"
    ]
    assert source_populated_parity["source_population_before_interface_owner_scan"]
    assert source_populated_parity["interface_owner_packet_preserved"]
    assert source_populated_parity["source_interface_buffers_do_not_alias"]
    assert source_populated_parity["parity_scoring_lane_executed"]
    assert source_populated_parity["finite_reproducible_score"]
    assert source_populated_parity["material_improvement_demonstrated"] is False
    assert source_populated_parity["paired_passed"] is False
    assert source_populated_parity["fixture_quality_pending"]
    assert source_populated_parity["subgrid_vacuum_parity_scored"]
    assert source_populated_parity["subgrid_vacuum_parity_passed"] is False
    assert source_populated_parity["true_rt_readiness_unlocked"] is False
    assert source_populated_parity["production_patch_applied"] is False
    assert source_populated_parity["solver_behavior_changed"] is False
    assert source_populated_parity["field_update_behavior_changed"] is False
    assert source_populated_parity["new_solver_hunk_retained"] is False
    assert source_populated_parity["benchmark_plane_dft_observable_imported"] is False
    assert source_populated_parity["solver_local_proxy_uses_plane_dft_monitor"] is False
    assert source_populated_parity[
        "next_lane_requires_source_populated_failure_theory_redesign"
    ]
    source_populated_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in source_populated_parity["candidate_ladder"]
    }
    assert source_populated_candidates[
        "K1_finite_source_populated_private_parity_score"
    ]["accepted_candidate"]
    assert source_populated_candidates[
        "K1_finite_source_populated_private_parity_score"
    ]["source_packet_consumed_by_modal_retry"]
    assert source_populated_candidates[
        "K2_material_improvement_gate"
    ]["material_improvement_demonstrated"] is False
    assert source_populated_candidates[
        "K4_source_populated_parity_scoring_blocked"
    ]["accepted_candidate"] is False
    assert source_populated_parity["public_claim_allowed"] is False
    assert source_populated_parity["public_observable_promoted"] is False
    assert source_populated_parity["true_rt_public_observable_promoted"] is False
    assert (
        source_populated_parity["dft_flux_tfsf_port_sparameter_promoted"]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_source_populated_propagation_aware_modal_retry_parity_scoring_next_prerequisite"
        ]
        == source_populated_parity["next_prerequisite"]
    )
    source_populated_failure_theory = benchmark_gate[
        "private_plane_wave_source_populated_propagation_aware_modal_retry_failure_theory"
    ]
    assert benchmark_gate[
        "private_plane_wave_source_populated_propagation_aware_modal_retry_failure_theory_status"
    ] == (
        "private_plane_wave_source_populated_modal_retry_time_alignment_theory_contract_ready"
    )
    assert source_populated_failure_theory["terminal_outcome"] == (
        "private_plane_wave_source_populated_modal_retry_time_alignment_theory_contract_ready"
    )
    assert source_populated_failure_theory[
        "upstream_source_populated_parity_status"
    ] == benchmark_gate[
        "private_plane_wave_source_populated_propagation_aware_modal_retry_parity_scoring_status"
    ]
    assert source_populated_failure_theory[
        "upstream_source_owner_population_status"
    ] == benchmark_gate[
        "private_plane_wave_source_owner_incident_packet_population_implementation_status"
    ]
    assert source_populated_failure_theory[
        "candidate_ladder_declared_before_implementation"
    ]
    assert source_populated_failure_theory[
        "candidate_ladder_declared_before_slow_scoring"
    ]
    assert source_populated_failure_theory["candidate_count"] == 5
    assert source_populated_failure_theory["selected_candidate_id"] == (
        "L1_lagged_interface_current_source_timing_diagnosis"
    )
    assert source_populated_failure_theory["baseline_metrics"] == (
        source_populated_parity["baseline_metrics"]
    )
    assert source_populated_failure_theory["metrics"] == source_populated_parity[
        "metrics"
    ]
    assert source_populated_failure_theory["thresholds"] == source_populated_parity[
        "thresholds"
    ]
    assert source_populated_failure_theory["baseline_metrics_preserved"]
    assert source_populated_failure_theory["thresholds_unchanged"]
    assert source_populated_failure_theory["source_owner_incident_packet_populated"]
    assert source_populated_failure_theory["source_packet_consumed_by_modal_retry"]
    assert source_populated_failure_theory["source_populated_parity_insufficient"]
    assert source_populated_failure_theory["material_improvement_demonstrated"] is False
    assert source_populated_failure_theory["true_rt_readiness_unlocked"] is False
    assert source_populated_failure_theory["failure_theory_lane_executed"]
    assert source_populated_failure_theory[
        "stale_interface_source_timing_theory_selected"
    ]
    assert source_populated_failure_theory[
        "lagged_interface_packet_current_source_packet_mismatch"
    ]
    assert source_populated_failure_theory[
        "modal_retry_subtracts_different_time_levels"
    ]
    assert source_populated_failure_theory["time_aligned_packet_staging_required"]
    timing_contract = source_populated_failure_theory["timing_contract"]
    assert timing_contract["consumer_helper"] == (
        "_apply_propagation_aware_modal_retry_face_helper"
    )
    assert timing_contract["source_population_helper"] == (
        "_update_private_source_owner_state_from_scan"
    )
    assert timing_contract["time_alignment_required"]
    assert timing_contract["public_observable_required"] is False
    assert source_populated_failure_theory["projection_normalizer_theory_deferred"]
    assert source_populated_failure_theory["transverse_phase_floor_theory_deferred"]
    assert source_populated_failure_theory["solver_behavior_changed"] is False
    assert source_populated_failure_theory["field_update_behavior_changed"] is False
    assert source_populated_failure_theory["new_solver_hunk_retained"] is False
    assert (
        source_populated_failure_theory["benchmark_plane_dft_observable_imported"]
        is False
    )
    assert source_populated_failure_theory[
        "next_lane_requires_time_aligned_packet_staging_design"
    ]
    source_populated_theory_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in source_populated_failure_theory["candidate_ladder"]
    }
    assert source_populated_theory_candidates[
        "L1_lagged_interface_current_source_timing_diagnosis"
    ]["accepted_candidate"]
    assert source_populated_theory_candidates[
        "L1_lagged_interface_current_source_timing_diagnosis"
    ]["time_aligned_packet_staging_required"]
    assert source_populated_theory_candidates[
        "L2_modal_projection_normalizer_diagnosis"
    ]["accepted_candidate"] is False
    assert source_populated_theory_candidates[
        "L3_transverse_phase_coherence_floor_diagnosis"
    ]["accepted_candidate"] is False
    assert source_populated_theory_candidates[
        "L4_source_populated_failure_theory_blocked"
    ]["accepted_candidate"] is False
    assert source_populated_failure_theory["public_claim_allowed"] is False
    assert source_populated_failure_theory["public_observable_promoted"] is False
    assert (
        source_populated_failure_theory["true_rt_public_observable_promoted"]
        is False
    )
    assert (
        source_populated_failure_theory["dft_flux_tfsf_port_sparameter_promoted"]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_source_populated_propagation_aware_modal_retry_failure_theory_next_prerequisite"
        ]
        == source_populated_failure_theory["next_prerequisite"]
    )
    time_aligned_design = benchmark_gate[
        "private_plane_wave_source_interface_time_aligned_packet_staging_design"
    ]
    assert benchmark_gate[
        "private_plane_wave_source_interface_time_aligned_packet_staging_design_status"
    ] == "private_plane_wave_source_interface_time_aligned_packet_staging_contract_ready"
    assert time_aligned_design["terminal_outcome"] == (
        "private_plane_wave_source_interface_time_aligned_packet_staging_contract_ready"
    )
    assert time_aligned_design["upstream_failure_theory_status"] == benchmark_gate[
        "private_plane_wave_source_populated_propagation_aware_modal_retry_failure_theory_status"
    ]
    assert time_aligned_design["candidate_ladder_declared_before_implementation"]
    assert time_aligned_design["candidate_ladder_declared_before_slow_scoring"]
    assert time_aligned_design["candidate_count"] == 5
    assert time_aligned_design["selected_candidate_id"] == (
        "M3_combined_time_aligned_packet_staging_contract"
    )
    assert time_aligned_design["baseline_metrics"] == (
        source_populated_failure_theory["baseline_metrics"]
    )
    assert time_aligned_design["metrics"] == source_populated_failure_theory[
        "metrics"
    ]
    assert time_aligned_design["thresholds"] == source_populated_failure_theory[
        "thresholds"
    ]
    assert time_aligned_design["baseline_metrics_preserved"]
    assert time_aligned_design["thresholds_unchanged"]
    assert time_aligned_design["time_alignment_theory_contract_ready"]
    assert time_aligned_design["source_owner_incident_packet_populated"]
    assert time_aligned_design["source_packet_consumed_by_modal_retry"]
    assert time_aligned_design["source_populated_parity_insufficient"]
    staged_schema = time_aligned_design["staged_packet_schema"]
    assert staged_schema["state_extension_target"] == "_PrivateInterfaceOwnerState"
    assert staged_schema["packet_shape_matches_existing_owner_packet"]
    assert staged_schema["staged_packets_do_not_alias_current_packets"]
    assert staged_schema["private_state_only"]
    initialization_contract = time_aligned_design["initialization_contract"]
    assert initialization_contract["cpml_initializes_staged_packets"]
    assert initialization_contract["non_cpml_initializes_staged_packets"]
    assert initialization_contract["jit_runner_initializes_staged_packets"]
    consumer_timing_contract = time_aligned_design["consumer_timing_contract"]
    assert consumer_timing_contract["consumer_helper"] == (
        "_apply_propagation_aware_modal_retry_face_helper"
    )
    assert consumer_timing_contract["population_helper"] == (
        "_update_private_source_owner_state_from_scan"
    )
    assert consumer_timing_contract["modal_retry_consumes_time_aligned_packet_pair"]
    assert consumer_timing_contract["requires_private_post_h_hook"] is False
    assert consumer_timing_contract["requires_public_observable"] is False
    assert time_aligned_design["staged_packet_fields_private_only"]
    assert time_aligned_design["staged_packets_fixed_shape"]
    assert time_aligned_design["staged_packets_do_not_alias_current_packets"]
    assert time_aligned_design["cpml_non_cpml_staging_initialization_contract_ready"]
    assert time_aligned_design["jit_runner_staging_initialization_contract_ready"]
    assert time_aligned_design["modal_retry_time_aligned_consumer_contract_ready"]
    assert time_aligned_design["implementation_lane_executed"] is False
    assert time_aligned_design["production_patch_applied"] is False
    assert time_aligned_design["solver_behavior_changed"] is False
    assert time_aligned_design["field_update_behavior_changed"] is False
    assert time_aligned_design["new_solver_hunk_retained"] is False
    assert time_aligned_design["benchmark_plane_dft_observable_imported"] is False
    assert time_aligned_design["true_rt_readiness_unlocked"] is False
    assert time_aligned_design[
        "next_lane_requires_time_aligned_packet_staging_implementation"
    ]
    time_aligned_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in time_aligned_design["candidate_ladder"]
    }
    assert time_aligned_candidates[
        "M1_source_interface_staged_packet_schema"
    ]["schema_component_ready"]
    assert time_aligned_candidates[
        "M2_cpml_non_cpml_jit_staging_initialization"
    ]["initialization_component_ready"]
    assert time_aligned_candidates[
        "M3_combined_time_aligned_packet_staging_contract"
    ]["accepted_candidate"]
    assert time_aligned_candidates[
        "M4_time_aligned_packet_staging_design_blocked"
    ]["accepted_candidate"] is False
    assert time_aligned_design["public_claim_allowed"] is False
    assert time_aligned_design["public_observable_promoted"] is False
    assert time_aligned_design["true_rt_public_observable_promoted"] is False
    assert time_aligned_design["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_source_interface_time_aligned_packet_staging_design_next_prerequisite"
        ]
        == time_aligned_design["next_prerequisite"]
    )
    time_aligned_implementation = benchmark_gate[
        "private_plane_wave_source_interface_time_aligned_packet_staging_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_source_interface_time_aligned_packet_staging_implementation_status"
    ] == (
        "private_plane_wave_source_interface_time_aligned_packet_staging_hunk_retained_fixture_quality_pending"
    )
    assert time_aligned_implementation["terminal_outcome"] == (
        "private_plane_wave_source_interface_time_aligned_packet_staging_hunk_retained_fixture_quality_pending"
    )
    assert time_aligned_implementation["upstream_design_status"] == benchmark_gate[
        "private_plane_wave_source_interface_time_aligned_packet_staging_design_status"
    ]
    assert time_aligned_implementation["upstream_failure_theory_status"] == (
        benchmark_gate[
            "private_plane_wave_source_populated_propagation_aware_modal_retry_failure_theory_status"
        ]
    )
    assert time_aligned_implementation["candidate_ladder_declared_before_solver_edit"]
    assert time_aligned_implementation[
        "candidate_ladder_declared_before_slow_scoring"
    ]
    assert time_aligned_implementation["candidate_count"] == 5
    assert time_aligned_implementation["selected_candidate_id"] == (
        "N3_modal_retry_time_aligned_consumer_hunk"
    )
    assert time_aligned_implementation["baseline_metrics"] == (
        time_aligned_design["baseline_metrics"]
    )
    assert time_aligned_implementation["metrics"] == time_aligned_design["metrics"]
    assert time_aligned_implementation["thresholds"] == (
        time_aligned_design["thresholds"]
    )
    assert time_aligned_implementation["baseline_metrics_preserved"]
    assert time_aligned_implementation["thresholds_unchanged"]
    assert time_aligned_implementation["time_aligned_packet_staging_design_ready"]
    assert time_aligned_implementation["staged_packet_fields_retained"]
    assert time_aligned_implementation["staged_packets_initialized_to_zero"]
    assert time_aligned_implementation["staged_packets_fixed_shape"]
    assert time_aligned_implementation[
        "staged_packets_do_not_alias_current_packets"
    ]
    assert time_aligned_implementation["previous_source_packet_fields_retained"]
    assert time_aligned_implementation["previous_interface_packet_fields_retained"]
    assert time_aligned_implementation["stage_helper_retained"]
    assert time_aligned_implementation["stage_helper_runs_after_observable_proxy"]
    assert time_aligned_implementation["stage_helper_runs_before_source_overwrite"]
    assert time_aligned_implementation["source_overwrite_occurs_after_staging"]
    assert time_aligned_implementation["interface_scan_occurs_after_modal_retry"]
    assert time_aligned_implementation[
        "cpml_non_cpml_staging_initialization_retained"
    ]
    assert time_aligned_implementation["cpml_non_cpml_wiring_symmetric"]
    assert time_aligned_implementation["jit_runner_staging_initialization_preserved"]
    assert time_aligned_implementation[
        "modal_retry_consumes_time_aligned_packet_pair"
    ]
    assert time_aligned_implementation["modal_retry_reads_previous_source_packet"]
    assert time_aligned_implementation["modal_retry_reads_previous_interface_packet"]
    implementation_timing = time_aligned_implementation["consumer_timing_contract"]
    assert implementation_timing["staging_helper"] == (
        "_stage_private_time_aligned_owner_packets"
    )
    assert implementation_timing["stage_helper_runs_before_source_overwrite"]
    assert time_aligned_implementation["implementation_lane_executed"]
    assert time_aligned_implementation["production_patch_applied"]
    assert time_aligned_implementation["solver_behavior_changed"]
    assert time_aligned_implementation["field_update_behavior_changed"]
    assert time_aligned_implementation["new_solver_hunk_retained"]
    assert time_aligned_implementation["benchmark_plane_dft_observable_imported"] is False
    assert time_aligned_implementation["true_rt_readiness_unlocked"] is False
    assert time_aligned_implementation[
        "next_lane_requires_time_aligned_modal_retry_parity_scoring"
    ]
    time_aligned_implementation_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in time_aligned_implementation["candidate_ladder"]
    }
    assert time_aligned_implementation_candidates[
        "N1_private_staged_packet_fields_and_initialization"
    ]["staged_packet_fields_retained"]
    assert time_aligned_implementation_candidates[
        "N2_cpml_non_cpml_jit_safe_staging_wiring"
    ]["cpml_non_cpml_wiring_symmetric"]
    assert time_aligned_implementation_candidates[
        "N3_modal_retry_time_aligned_consumer_hunk"
    ]["accepted_candidate"]
    assert time_aligned_implementation_candidates[
        "N4_time_aligned_packet_staging_implementation_blocked"
    ]["accepted_candidate"] is False
    assert time_aligned_implementation["public_claim_allowed"] is False
    assert time_aligned_implementation["public_observable_promoted"] is False
    assert time_aligned_implementation["true_rt_public_observable_promoted"] is False
    assert (
        time_aligned_implementation["dft_flux_tfsf_port_sparameter_promoted"]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_source_interface_time_aligned_packet_staging_implementation_next_prerequisite"
        ]
        == time_aligned_implementation["next_prerequisite"]
    )
    time_aligned_scoring = benchmark_gate[
        "private_plane_wave_time_aligned_modal_retry_parity_scoring"
    ]
    assert benchmark_gate[
        "private_plane_wave_time_aligned_modal_retry_parity_scoring_status"
    ] == (
        "private_plane_wave_time_aligned_modal_retry_hunk_insufficient_fixture_quality_pending"
    )
    assert time_aligned_scoring["terminal_outcome"] == (
        "private_plane_wave_time_aligned_modal_retry_hunk_insufficient_fixture_quality_pending"
    )
    assert time_aligned_scoring["upstream_staging_implementation_status"] == (
        benchmark_gate[
            "private_plane_wave_source_interface_time_aligned_packet_staging_implementation_status"
        ]
    )
    assert time_aligned_scoring["upstream_source_populated_parity_status"] == (
        benchmark_gate[
            "private_plane_wave_source_populated_propagation_aware_modal_retry_parity_scoring_status"
        ]
    )
    assert time_aligned_scoring["candidate_ladder_declared_before_solver_edit"]
    assert time_aligned_scoring["candidate_ladder_declared_before_slow_scoring"]
    assert time_aligned_scoring["candidate_count"] == 5
    assert time_aligned_scoring["selected_candidate_id"] == (
        "O1_finite_time_aligned_modal_retry_private_parity_score"
    )
    assert time_aligned_scoring["baseline_metrics"] == source_populated_parity[
        "metrics"
    ]
    assert time_aligned_scoring["metrics"] == time_aligned_implementation["metrics"]
    assert time_aligned_scoring["thresholds"] == time_aligned_implementation[
        "thresholds"
    ]
    assert time_aligned_scoring["baseline_metrics_preserved"]
    assert time_aligned_scoring["thresholds_unchanged"]
    assert time_aligned_scoring["time_aligned_packet_staging_hunk_retained"]
    assert time_aligned_scoring["staged_packet_hunk_retained"]
    assert time_aligned_scoring["previous_source_packet_fields_retained"]
    assert time_aligned_scoring["previous_interface_packet_fields_retained"]
    assert time_aligned_scoring["stage_helper_retained"]
    assert time_aligned_scoring["modal_retry_consumes_time_aligned_packet_pair"]
    assert time_aligned_scoring["modal_retry_reads_previous_source_packet"]
    assert time_aligned_scoring["modal_retry_reads_previous_interface_packet"]
    assert time_aligned_scoring["parity_scoring_lane_executed"]
    assert time_aligned_scoring["finite_reproducible_score"]
    assert time_aligned_scoring["material_improvement_demonstrated"] is False
    assert time_aligned_scoring["paired_passed"] is False
    assert time_aligned_scoring["fixture_quality_ready"] is False
    assert time_aligned_scoring["fixture_quality_pending"]
    assert time_aligned_scoring["subgrid_vacuum_parity_scored"]
    assert time_aligned_scoring["subgrid_vacuum_parity_passed"] is False
    assert time_aligned_scoring["true_rt_readiness_unlocked"] is False
    scoring_threshold_results = time_aligned_scoring["threshold_results"]
    assert scoring_threshold_results["usable_passband"]
    assert scoring_threshold_results["transverse_phase_spread_deg"] is False
    assert scoring_threshold_results["transverse_magnitude_cv"] is False
    assert scoring_threshold_results["vacuum_relative_magnitude_error"] is False
    assert scoring_threshold_results["vacuum_phase_error_deg"] is False
    assert time_aligned_scoring["production_patch_applied"] is False
    assert time_aligned_scoring["solver_behavior_changed"] is False
    assert time_aligned_scoring["field_update_behavior_changed"] is False
    assert time_aligned_scoring["new_solver_hunk_retained"] is False
    assert time_aligned_scoring["benchmark_plane_dft_observable_imported"] is False
    assert time_aligned_scoring[
        "next_lane_requires_time_aligned_modal_retry_failure_theory"
    ]
    time_aligned_scoring_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in time_aligned_scoring["candidate_ladder"]
    }
    assert time_aligned_scoring_candidates[
        "O1_finite_time_aligned_modal_retry_private_parity_score"
    ]["accepted_candidate"]
    assert time_aligned_scoring_candidates[
        "O2_paired_material_improvement_gate"
    ]["accepted_candidate"] is False
    assert time_aligned_scoring_candidates[
        "O3_fixture_quality_true_rt_readiness_preflight"
    ]["accepted_candidate"] is False
    assert time_aligned_scoring_candidates[
        "O4_time_aligned_modal_retry_parity_scoring_blocked"
    ]["accepted_candidate"] is False
    assert time_aligned_scoring["public_claim_allowed"] is False
    assert time_aligned_scoring["public_observable_promoted"] is False
    assert time_aligned_scoring["true_rt_public_observable_promoted"] is False
    assert time_aligned_scoring["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_time_aligned_modal_retry_parity_scoring_next_prerequisite"
        ]
        == time_aligned_scoring["next_prerequisite"]
    )
    time_aligned_failure_theory = benchmark_gate[
        "private_plane_wave_time_aligned_modal_retry_failure_theory"
    ]
    assert benchmark_gate[
        "private_plane_wave_time_aligned_modal_retry_failure_theory_status"
    ] == (
        "private_plane_wave_time_aligned_modal_retry_modal_projection_normalizer_theory_contract_ready"
    )
    assert time_aligned_failure_theory["terminal_outcome"] == (
        "private_plane_wave_time_aligned_modal_retry_modal_projection_normalizer_theory_contract_ready"
    )
    assert time_aligned_failure_theory["upstream_time_aligned_parity_status"] == (
        benchmark_gate[
            "private_plane_wave_time_aligned_modal_retry_parity_scoring_status"
        ]
    )
    assert time_aligned_failure_theory[
        "upstream_staging_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_source_interface_time_aligned_packet_staging_implementation_status"
    ]
    assert time_aligned_failure_theory["candidate_ladder_declared_before_implementation"]
    assert time_aligned_failure_theory["candidate_ladder_declared_before_solver_edit"]
    assert time_aligned_failure_theory["candidate_ladder_declared_before_slow_scoring"]
    assert time_aligned_failure_theory["candidate_count"] == 5
    assert time_aligned_failure_theory["selected_candidate_id"] == (
        "P1_modal_projection_normalizer_floor_theory"
    )
    assert time_aligned_failure_theory["baseline_metrics"] == (
        time_aligned_scoring["baseline_metrics"]
    )
    assert time_aligned_failure_theory["metrics"] == time_aligned_scoring["metrics"]
    assert time_aligned_failure_theory["thresholds"] == time_aligned_scoring[
        "thresholds"
    ]
    assert time_aligned_failure_theory["threshold_results"] == (
        time_aligned_scoring["threshold_results"]
    )
    assert time_aligned_failure_theory["baseline_metrics_preserved"]
    assert time_aligned_failure_theory["thresholds_unchanged"]
    assert time_aligned_failure_theory["staged_packet_hunk_retained"]
    assert time_aligned_failure_theory["time_aligned_packet_staging_hunk_retained"]
    assert time_aligned_failure_theory["time_aligned_parity_insufficient"]
    assert time_aligned_failure_theory["time_alignment_no_material_delta"]
    assert time_aligned_failure_theory[
        "metrics_identical_to_source_populated_baseline"
    ]
    assert all(
        abs(value) <= 1.0e-12
        for value in time_aligned_failure_theory["score_delta"].values()
    )
    assert time_aligned_failure_theory["finite_reproducible_score"]
    assert time_aligned_failure_theory["material_improvement_demonstrated"] is False
    assert time_aligned_failure_theory["paired_passed"] is False
    assert time_aligned_failure_theory["fixture_quality_ready"] is False
    assert time_aligned_failure_theory["fixture_quality_pending"]
    assert time_aligned_failure_theory["transverse_phase_floor_persists"]
    assert time_aligned_failure_theory["transverse_magnitude_floor_persists"]
    assert time_aligned_failure_theory["vacuum_stability_floor_persists"]
    assert time_aligned_failure_theory[
        "modal_projection_or_normalizer_floor_selected"
    ]
    projection_contract = time_aligned_failure_theory[
        "projection_normalizer_contract"
    ]
    assert projection_contract["consumer_helper"] == (
        "_apply_propagation_aware_modal_retry_face_helper"
    )
    assert projection_contract["requires_new_public_observable"] is False
    assert projection_contract["requires_benchmark_dft_or_flux_publication"] is False
    assert projection_contract["requires_hook"] is False
    assert projection_contract["requires_solver_hunk_in_this_lane"] is False
    assert time_aligned_failure_theory["failure_theory_lane_executed"]
    assert time_aligned_failure_theory["true_rt_readiness_unlocked"] is False
    assert time_aligned_failure_theory["production_patch_applied"] is False
    assert time_aligned_failure_theory["solver_behavior_changed"] is False
    assert time_aligned_failure_theory["field_update_behavior_changed"] is False
    assert time_aligned_failure_theory["new_solver_hunk_retained"] is False
    assert time_aligned_failure_theory["benchmark_plane_dft_observable_imported"] is False
    assert time_aligned_failure_theory[
        "next_lane_requires_modal_projection_normalizer_contract_design"
    ]
    time_aligned_failure_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in time_aligned_failure_theory["candidate_ladder"]
    }
    assert time_aligned_failure_candidates[
        "P1_modal_projection_normalizer_floor_theory"
    ]["accepted_candidate"]
    assert time_aligned_failure_candidates[
        "P2_transverse_phase_coherence_residual_theory"
    ]["accepted_candidate"] is False
    assert time_aligned_failure_candidates[
        "P3_fixture_source_contract_residual_theory"
    ]["accepted_candidate"] is False
    assert time_aligned_failure_candidates[
        "P4_time_aligned_modal_retry_failure_theory_blocked"
    ]["accepted_candidate"] is False
    assert time_aligned_failure_theory["public_claim_allowed"] is False
    assert time_aligned_failure_theory["public_observable_promoted"] is False
    assert time_aligned_failure_theory["true_rt_public_observable_promoted"] is False
    assert time_aligned_failure_theory["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_time_aligned_modal_retry_failure_theory_next_prerequisite"
        ]
        == time_aligned_failure_theory["next_prerequisite"]
    )
    modal_contract_design = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_contract_design"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_contract_design_status"
    ] == "private_plane_wave_modal_projection_normalizer_contract_design_ready"
    assert modal_contract_design["terminal_outcome"] == (
        "private_plane_wave_modal_projection_normalizer_contract_design_ready"
    )
    assert modal_contract_design["upstream_failure_theory_status"] == benchmark_gate[
        "private_plane_wave_time_aligned_modal_retry_failure_theory_status"
    ]
    assert modal_contract_design["candidate_ladder_declared_before_implementation"]
    assert modal_contract_design["candidate_ladder_declared_before_solver_edit"]
    assert modal_contract_design["candidate_ladder_declared_before_slow_scoring"]
    assert modal_contract_design["candidate_count"] == 5
    assert modal_contract_design["selected_candidate_id"] == (
        "Q3_combined_projection_normalizer_mask_contract"
    )
    assert modal_contract_design["baseline_metrics"] == (
        time_aligned_failure_theory["baseline_metrics"]
    )
    assert modal_contract_design["metrics"] == time_aligned_failure_theory["metrics"]
    assert modal_contract_design["thresholds"] == time_aligned_failure_theory[
        "thresholds"
    ]
    assert modal_contract_design["threshold_results"] == (
        time_aligned_failure_theory["threshold_results"]
    )
    assert modal_contract_design["baseline_metrics_preserved"]
    assert modal_contract_design["thresholds_unchanged"]
    assert modal_contract_design["time_alignment_no_material_delta"]
    assert modal_contract_design["modal_projection_or_normalizer_floor_selected"]
    assert modal_contract_design["basis_contract_component_ready"]
    assert modal_contract_design["normalizer_contract_component_ready"]
    assert modal_contract_design["mask_weighting_contract_component_ready"]
    assert modal_contract_design["combined_contract_ready"]
    projection_basis = modal_contract_design["projection_basis_contract"]
    assert projection_basis["state_owner"] == "_PrivateInterfaceOwnerState"
    assert projection_basis["consumer_helper"] == (
        "_apply_propagation_aware_modal_retry_face_helper"
    )
    assert projection_basis["requires_public_observable"] is False
    assert projection_basis["requires_hook"] is False
    normalizer_contract = modal_contract_design["normalizer_contract"]
    assert normalizer_contract["normalized_packet_helper"] == (
        "_incident_normalized_source_packet"
    )
    assert normalizer_contract["normalization_equivalence_required"]
    assert normalizer_contract["requires_public_config"] is False
    mask_contract = modal_contract_design["face_mask_weighting_contract"]
    assert mask_contract["combined_packet_mask"] == (
        "face_proxy_mask * source_owner_mask"
    )
    assert mask_contract["weighting_equivalence_required"]
    assert mask_contract["cpml_non_cpml_contract_identical"]
    implementation_contract = modal_contract_design["implementation_contract"]
    assert implementation_contract["thresholds_frozen"]
    assert implementation_contract["material_improvement_gate_required_after_hunk"]
    assert modal_contract_design["private_design_lane_executed"]
    assert modal_contract_design["fixture_quality_ready"] is False
    assert modal_contract_design["fixture_quality_pending"]
    assert modal_contract_design["true_rt_readiness_unlocked"] is False
    assert modal_contract_design["production_patch_applied"] is False
    assert modal_contract_design["solver_behavior_changed"] is False
    assert modal_contract_design["field_update_behavior_changed"] is False
    assert modal_contract_design["new_solver_hunk_retained"] is False
    assert modal_contract_design["benchmark_plane_dft_observable_imported"] is False
    assert modal_contract_design[
        "next_lane_requires_modal_projection_normalizer_contract_implementation"
    ]
    modal_design_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in modal_contract_design["candidate_ladder"]
    }
    assert modal_design_candidates[
        "Q1_shared_modal_projection_basis_contract"
    ]["basis_contract_component_ready"]
    assert modal_design_candidates[
        "Q2_packet_normalizer_contract"
    ]["normalizer_contract_component_ready"]
    assert modal_design_candidates[
        "Q3_combined_projection_normalizer_mask_contract"
    ]["accepted_candidate"]
    assert modal_design_candidates[
        "Q4_modal_projection_normalizer_contract_design_blocked"
    ]["accepted_candidate"] is False
    assert modal_contract_design["public_claim_allowed"] is False
    assert modal_contract_design["public_observable_promoted"] is False
    assert modal_contract_design["true_rt_public_observable_promoted"] is False
    assert modal_contract_design["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_contract_design_next_prerequisite"
        ]
        == modal_contract_design["next_prerequisite"]
    )
    modal_contract_implementation = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_contract_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_contract_implementation_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_contract_hunk_retained_fixture_quality_pending"
    )
    assert modal_contract_implementation["terminal_outcome"] == (
        "private_plane_wave_modal_projection_normalizer_contract_hunk_retained_fixture_quality_pending"
    )
    assert modal_contract_implementation["upstream_contract_design_status"] == (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_contract_design_status"
        ]
    )
    assert modal_contract_implementation["candidate_ladder_declared_before_implementation"]
    assert modal_contract_implementation["candidate_ladder_declared_before_solver_edit"]
    assert modal_contract_implementation["candidate_ladder_declared_before_slow_scoring"]
    assert modal_contract_implementation["candidate_count"] == 5
    assert modal_contract_implementation["selected_candidate_id"] == (
        "R3_combined_mask_weighted_modal_contract_hunk"
    )
    assert modal_contract_implementation["baseline_metrics"] == (
        modal_contract_design["baseline_metrics"]
    )
    assert modal_contract_implementation["metrics"] == modal_contract_design["metrics"]
    assert modal_contract_implementation["thresholds"] == modal_contract_design[
        "thresholds"
    ]
    assert modal_contract_implementation["threshold_results"] == (
        modal_contract_design["threshold_results"]
    )
    assert modal_contract_implementation["baseline_metrics_preserved"]
    assert modal_contract_implementation["thresholds_unchanged"]
    assert modal_contract_implementation["contract_design_ready"]
    assert modal_contract_implementation["projection_basis_contract_consumed"]
    assert modal_contract_implementation["normalizer_contract_consumed"]
    assert modal_contract_implementation["mask_weighting_contract_consumed"]
    assert modal_contract_implementation["contract_gate_helper_retained"]
    assert modal_contract_implementation["layout_contract_enforced"]
    assert modal_contract_implementation["orientation_contract_enforced"]
    assert modal_contract_implementation["mask_contract_enforced"]
    assert modal_contract_implementation["weight_contract_enforced"]
    assert modal_contract_implementation["normalizer_contract_enforced"]
    assert modal_contract_implementation["contract_gate_fail_closed"]
    assert modal_contract_implementation["no_op_on_contract_mismatch"]
    assert modal_contract_implementation["cpml_non_cpml_wiring_inherited"]
    assert modal_contract_implementation["jit_safe_scalar_gate"]
    implementation_gate_contract = modal_contract_implementation[
        "implementation_contract"
    ]
    assert implementation_gate_contract["contract_gate_helper"] == (
        "_private_modal_projection_normalizer_contract_gate"
    )
    assert implementation_gate_contract["fail_closed_on_contract_mismatch"]
    assert implementation_gate_contract["jit_safe_scalar_gate"]
    assert modal_contract_implementation["material_improvement_demonstrated"] is False
    assert modal_contract_implementation["paired_passed"] is False
    assert modal_contract_implementation["fixture_quality_ready"] is False
    assert modal_contract_implementation["fixture_quality_pending"]
    assert modal_contract_implementation["true_rt_readiness_unlocked"] is False
    assert modal_contract_implementation["production_patch_applied"]
    assert modal_contract_implementation["solver_behavior_changed"]
    assert modal_contract_implementation["field_update_behavior_changed"]
    assert modal_contract_implementation["new_solver_hunk_retained"]
    assert modal_contract_implementation["benchmark_plane_dft_observable_imported"] is False
    assert modal_contract_implementation[
        "next_lane_requires_modal_projection_normalizer_contract_parity_scoring"
    ]
    modal_implementation_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in modal_contract_implementation["candidate_ladder"]
    }
    assert modal_implementation_candidates[
        "R1_projection_basis_provenance_gate"
    ]["layout_contract_enforced"]
    assert modal_implementation_candidates[
        "R2_incident_normalizer_weight_gate"
    ]["normalizer_contract_enforced"]
    assert modal_implementation_candidates[
        "R3_combined_mask_weighted_modal_contract_hunk"
    ]["accepted_candidate"]
    assert modal_implementation_candidates[
        "R4_modal_projection_normalizer_implementation_blocked"
    ]["accepted_candidate"] is False
    assert modal_contract_implementation["public_claim_allowed"] is False
    assert modal_contract_implementation["public_observable_promoted"] is False
    assert modal_contract_implementation["true_rt_public_observable_promoted"] is False
    assert modal_contract_implementation["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_contract_implementation_next_prerequisite"
        ]
        == modal_contract_implementation["next_prerequisite"]
    )
    modal_contract_parity_scoring = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_contract_parity_scoring"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_contract_parity_scoring_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_contract_hunk_insufficient_fixture_quality_pending"
    )
    assert modal_contract_parity_scoring["terminal_outcome"] == (
        "private_plane_wave_modal_projection_normalizer_contract_hunk_insufficient_fixture_quality_pending"
    )
    assert modal_contract_parity_scoring[
        "upstream_contract_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_contract_implementation_status"
    ]
    assert modal_contract_parity_scoring["candidate_ladder_declared_before_solver_edit"]
    assert modal_contract_parity_scoring["candidate_ladder_declared_before_slow_scoring"]
    assert modal_contract_parity_scoring["candidate_count"] == 5
    assert modal_contract_parity_scoring["selected_candidate_id"] == (
        "S1_finite_modal_contract_private_parity_score"
    )
    assert modal_contract_parity_scoring["baseline_metrics"] == (
        modal_contract_implementation["baseline_metrics"]
    )
    assert modal_contract_parity_scoring["metrics"] == (
        modal_contract_implementation["metrics"]
    )
    assert modal_contract_parity_scoring["thresholds"] == (
        modal_contract_implementation["thresholds"]
    )
    assert modal_contract_parity_scoring["threshold_results"] == (
        modal_contract_implementation["threshold_results"]
    )
    assert modal_contract_parity_scoring["baseline_metrics_preserved"]
    assert modal_contract_parity_scoring["thresholds_unchanged"]
    assert modal_contract_parity_scoring["contract_gate_hunk_retained"]
    assert modal_contract_parity_scoring["contract_gate_helper_retained"]
    assert modal_contract_parity_scoring["contract_gate_fail_closed"]
    assert modal_contract_parity_scoring["layout_contract_enforced"]
    assert modal_contract_parity_scoring["orientation_contract_enforced"]
    assert modal_contract_parity_scoring["mask_contract_enforced"]
    assert modal_contract_parity_scoring["weight_contract_enforced"]
    assert modal_contract_parity_scoring["normalizer_contract_enforced"]
    assert modal_contract_parity_scoring["parity_scoring_lane_executed"]
    assert modal_contract_parity_scoring["finite_reproducible_score"]
    assert modal_contract_parity_scoring["score_uses_retained_implementation_metrics"]
    assert modal_contract_parity_scoring["material_improvement_demonstrated"] is False
    assert modal_contract_parity_scoring["paired_passed"] is False
    assert modal_contract_parity_scoring["fixture_quality_ready"] is False
    assert modal_contract_parity_scoring["fixture_quality_pending"]
    assert modal_contract_parity_scoring["subgrid_vacuum_parity_scored"]
    assert modal_contract_parity_scoring["subgrid_vacuum_parity_passed"] is False
    assert modal_contract_parity_scoring["true_rt_readiness_unlocked"] is False
    assert modal_contract_parity_scoring["production_patch_applied"] is False
    assert modal_contract_parity_scoring["solver_behavior_changed"] is False
    assert modal_contract_parity_scoring["field_update_behavior_changed"] is False
    assert modal_contract_parity_scoring["new_solver_hunk_retained"] is False
    assert modal_contract_parity_scoring["benchmark_plane_dft_observable_imported"] is False
    assert modal_contract_parity_scoring[
        "next_lane_requires_modal_projection_normalizer_contract_failure_theory"
    ]
    modal_parity_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in modal_contract_parity_scoring["candidate_ladder"]
    }
    assert modal_parity_candidates[
        "S1_finite_modal_contract_private_parity_score"
    ]["accepted_candidate"]
    assert modal_parity_candidates[
        "S2_material_improvement_gate"
    ]["accepted_candidate"] is False
    assert modal_parity_candidates[
        "S3_fixture_quality_true_rt_readiness_preflight"
    ]["accepted_candidate"] is False
    assert modal_parity_candidates[
        "S4_modal_contract_parity_scoring_blocked"
    ]["accepted_candidate"] is False
    assert modal_contract_parity_scoring["public_claim_allowed"] is False
    assert modal_contract_parity_scoring["public_observable_promoted"] is False
    assert modal_contract_parity_scoring["true_rt_public_observable_promoted"] is False
    assert (
        modal_contract_parity_scoring["dft_flux_tfsf_port_sparameter_promoted"]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_contract_parity_scoring_next_prerequisite"
        ]
        == modal_contract_parity_scoring["next_prerequisite"]
    )
    modal_contract_failure_theory = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_contract_failure_theory"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_contract_failure_theory_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_basis_redesign_contract_ready"
    )
    assert modal_contract_failure_theory["terminal_outcome"] == (
        "private_plane_wave_modal_projection_normalizer_projected_basis_redesign_contract_ready"
    )
    assert modal_contract_failure_theory[
        "upstream_contract_parity_scoring_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_contract_parity_scoring_status"
    ]
    assert modal_contract_failure_theory[
        "upstream_contract_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_contract_implementation_status"
    ]
    assert modal_contract_failure_theory[
        "candidate_ladder_declared_before_implementation"
    ]
    assert modal_contract_failure_theory["candidate_ladder_declared_before_solver_edit"]
    assert modal_contract_failure_theory["candidate_ladder_declared_before_slow_scoring"]
    assert modal_contract_failure_theory["candidate_count"] == 5
    assert modal_contract_failure_theory["selected_candidate_id"] == (
        "T1_projection_basis_floor_theory"
    )
    assert modal_contract_failure_theory["baseline_metrics"] == (
        modal_contract_parity_scoring["baseline_metrics"]
    )
    assert modal_contract_failure_theory["metrics"] == (
        modal_contract_parity_scoring["metrics"]
    )
    assert modal_contract_failure_theory["thresholds"] == (
        modal_contract_parity_scoring["thresholds"]
    )
    assert modal_contract_failure_theory["threshold_results"] == (
        modal_contract_parity_scoring["threshold_results"]
    )
    assert modal_contract_failure_theory["baseline_metrics_preserved"]
    assert modal_contract_failure_theory["thresholds_unchanged"]
    assert modal_contract_failure_theory["contract_parity_scoring_insufficient"]
    assert modal_contract_failure_theory["contract_gate_hunk_retained"]
    assert modal_contract_failure_theory["contract_gate_fail_closed"]
    assert modal_contract_failure_theory["finite_reproducible_score"]
    assert modal_contract_failure_theory["metrics_identical_to_contract_baseline"]
    assert all(
        abs(value) <= 1.0e-12
        for value in modal_contract_failure_theory["score_delta"].values()
    )
    assert modal_contract_failure_theory["material_improvement_demonstrated"] is False
    assert modal_contract_failure_theory["paired_passed"] is False
    assert modal_contract_failure_theory["fixture_quality_ready"] is False
    assert modal_contract_failure_theory["fixture_quality_pending"]
    assert modal_contract_failure_theory["transverse_phase_floor_persists"]
    assert modal_contract_failure_theory["transverse_magnitude_floor_persists"]
    assert modal_contract_failure_theory["vacuum_stability_floor_persists"]
    assert modal_contract_failure_theory["projection_basis_floor_selected"]
    assert modal_contract_failure_theory["projected_basis_redesign_contract_ready"]
    projected_basis_contract = modal_contract_failure_theory[
        "projected_basis_contract"
    ]
    assert projected_basis_contract["consumer_helper"] == (
        "_apply_propagation_aware_modal_retry_face_helper"
    )
    assert projected_basis_contract["contract_gate_helper"] == (
        "_private_modal_projection_normalizer_contract_gate"
    )
    assert projected_basis_contract["requires_public_observable"] is False
    assert projected_basis_contract["requires_hook"] is False
    assert projected_basis_contract["requires_solver_hunk_in_this_lane"] is False
    assert modal_contract_failure_theory["contract_gate_only_no_projection_transform"]
    assert modal_contract_failure_theory["normalizer_weight_floor_deferred"]
    assert modal_contract_failure_theory["source_fixture_phase_floor_deferred"]
    assert modal_contract_failure_theory["failure_theory_lane_executed"]
    assert modal_contract_failure_theory["true_rt_readiness_unlocked"] is False
    assert modal_contract_failure_theory["production_patch_applied"] is False
    assert modal_contract_failure_theory["solver_behavior_changed"] is False
    assert modal_contract_failure_theory["field_update_behavior_changed"] is False
    assert modal_contract_failure_theory["new_solver_hunk_retained"] is False
    assert modal_contract_failure_theory[
        "next_lane_requires_projected_basis_redesign_contract"
    ]
    modal_failure_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in modal_contract_failure_theory["candidate_ladder"]
    }
    assert modal_failure_candidates[
        "T1_projection_basis_floor_theory"
    ]["accepted_candidate"]
    assert modal_failure_candidates[
        "T2_normalizer_weight_floor_theory"
    ]["accepted_candidate"] is False
    assert modal_failure_candidates[
        "T3_source_fixture_phase_floor_theory"
    ]["accepted_candidate"] is False
    assert modal_failure_candidates[
        "T4_failure_theory_blocked_no_public_promotion"
    ]["accepted_candidate"] is False
    assert modal_contract_failure_theory["public_claim_allowed"] is False
    assert modal_contract_failure_theory["public_observable_promoted"] is False
    assert modal_contract_failure_theory["true_rt_public_observable_promoted"] is False
    assert (
        modal_contract_failure_theory["dft_flux_tfsf_port_sparameter_promoted"]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_contract_failure_theory_next_prerequisite"
        ]
        == modal_contract_failure_theory["next_prerequisite"]
    )
    projected_basis_design = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_basis_design"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_basis_design_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_basis_contract_design_ready"
    )
    assert projected_basis_design["terminal_outcome"] == (
        "private_plane_wave_modal_projection_normalizer_projected_basis_contract_design_ready"
    )
    assert projected_basis_design["upstream_failure_theory_status"] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_contract_failure_theory_status"
    ]
    assert projected_basis_design["candidate_ladder_declared_before_implementation"]
    assert projected_basis_design["candidate_ladder_declared_before_solver_edit"]
    assert projected_basis_design["candidate_ladder_declared_before_slow_scoring"]
    assert projected_basis_design["candidate_count"] == 5
    assert projected_basis_design["selected_candidate_id"] == (
        "U3_bounded_projected_basis_implementation_contract"
    )
    assert projected_basis_design["baseline_metrics"] == (
        modal_contract_failure_theory["baseline_metrics"]
    )
    assert projected_basis_design["metrics"] == (
        modal_contract_failure_theory["metrics"]
    )
    assert projected_basis_design["thresholds"] == (
        modal_contract_failure_theory["thresholds"]
    )
    assert projected_basis_design["threshold_results"] == (
        modal_contract_failure_theory["threshold_results"]
    )
    assert projected_basis_design["baseline_metrics_preserved"]
    assert projected_basis_design["thresholds_unchanged"]
    assert projected_basis_design["projected_basis_floor_selected"]
    assert projected_basis_design["basis_schema_component_ready"]
    assert projected_basis_design["power_normalizer_component_ready"]
    assert projected_basis_design["mask_weighting_contract_ready"]
    assert projected_basis_design["bounded_private_implementation_contract_ready"]
    projected_basis_schema = projected_basis_design["projected_basis_schema"]
    assert projected_basis_schema["state_owner"] == "_PrivateInterfaceOwnerState"
    assert projected_basis_schema["consumer_helper"] == (
        "_apply_propagation_aware_modal_retry_face_helper"
    )
    assert projected_basis_schema["fixed_shape"]
    assert projected_basis_schema["private_state_only"]
    assert projected_basis_schema["requires_public_observable"] is False
    projected_power_normalizer = projected_basis_design[
        "projected_power_normalizer"
    ]
    assert projected_power_normalizer["mask_weighting_required"]
    assert projected_power_normalizer["cpml_non_cpml_contract_identical"]
    assert projected_power_normalizer["jit_safe_reduction"]
    projected_implementation_contract = projected_basis_design[
        "implementation_contract"
    ]
    assert projected_implementation_contract["target_helper"] == (
        "_project_private_modal_basis_packets"
    )
    assert projected_implementation_contract[
        "fail_closed_if_projection_energy_missing"
    ]
    assert projected_implementation_contract[
        "fail_closed_if_packet_contract_mismatch"
    ]
    assert projected_implementation_contract["no_new_public_switch"]
    assert projected_implementation_contract["thresholds_frozen"]
    assert projected_basis_design["private_design_lane_executed"]
    assert projected_basis_design["material_improvement_demonstrated"] is False
    assert projected_basis_design["paired_passed"] is False
    assert projected_basis_design["fixture_quality_ready"] is False
    assert projected_basis_design["fixture_quality_pending"]
    assert projected_basis_design["true_rt_readiness_unlocked"] is False
    assert projected_basis_design["production_patch_applied"] is False
    assert projected_basis_design["solver_behavior_changed"] is False
    assert projected_basis_design["field_update_behavior_changed"] is False
    assert projected_basis_design["new_solver_hunk_retained"] is False
    assert projected_basis_design["benchmark_plane_dft_observable_imported"] is False
    assert projected_basis_design["next_lane_requires_projected_basis_implementation"]
    projected_basis_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in projected_basis_design["candidate_ladder"]
    }
    assert projected_basis_candidates[
        "U1_shared_projected_basis_schema"
    ]["basis_schema_component_ready"]
    assert projected_basis_candidates[
        "U2_projected_power_normalizer_contract"
    ]["power_normalizer_component_ready"]
    assert projected_basis_candidates[
        "U3_bounded_projected_basis_implementation_contract"
    ]["accepted_candidate"]
    assert projected_basis_candidates[
        "U4_projected_basis_design_blocked"
    ]["accepted_candidate"] is False
    assert projected_basis_design["public_claim_allowed"] is False
    assert projected_basis_design["public_observable_promoted"] is False
    assert projected_basis_design["true_rt_public_observable_promoted"] is False
    assert projected_basis_design["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_basis_design_next_prerequisite"
        ]
        == projected_basis_design["next_prerequisite"]
    )
    projected_basis_implementation = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_basis_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_basis_implementation_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_basis_hunk_retained_fixture_quality_pending"
    )
    assert projected_basis_implementation["terminal_outcome"] == (
        "private_plane_wave_modal_projection_normalizer_projected_basis_hunk_retained_fixture_quality_pending"
    )
    assert projected_basis_implementation[
        "upstream_projected_basis_design_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_basis_design_status"
    ]
    assert projected_basis_implementation["candidate_ladder_declared_before_implementation"]
    assert projected_basis_implementation["candidate_ladder_declared_before_solver_edit"]
    assert projected_basis_implementation["candidate_ladder_declared_before_slow_scoring"]
    assert projected_basis_implementation["candidate_count"] == 5
    assert projected_basis_implementation["selected_candidate_id"] == (
        "V3_combined_projected_basis_implementation_hunk"
    )
    assert projected_basis_implementation["baseline_metrics"] == (
        projected_basis_design["baseline_metrics"]
    )
    assert projected_basis_implementation["metrics"] == (
        projected_basis_design["metrics"]
    )
    assert projected_basis_implementation["thresholds"] == (
        projected_basis_design["thresholds"]
    )
    assert projected_basis_implementation["threshold_results"] == (
        projected_basis_design["threshold_results"]
    )
    assert projected_basis_implementation["baseline_metrics_preserved"]
    assert projected_basis_implementation["thresholds_unchanged"]
    assert projected_basis_implementation["projected_basis_design_ready"]
    assert projected_basis_implementation["projected_basis_schema_consumed"]
    assert projected_basis_implementation["projected_power_normalizer_consumed"]
    assert projected_basis_implementation["mask_weighting_contract_consumed"]
    assert projected_basis_implementation["projected_basis_packet_helper_retained"]
    assert projected_basis_implementation["projected_basis_helper"] == (
        "_project_private_modal_basis_packets"
    )
    assert projected_basis_implementation["modal_retry_consumer_wiring_retained"]
    assert projected_basis_implementation["contract_gate_helper_retained"]
    assert projected_basis_implementation["contract_gate_fail_closed"]
    assert projected_basis_implementation["projection_gate_fail_closed"]
    assert projected_basis_implementation["fail_closed_if_projection_energy_missing"]
    assert projected_basis_implementation["fail_closed_if_packet_contract_mismatch"]
    assert projected_basis_implementation["reuse_existing_owner_packet_shapes"]
    assert projected_basis_implementation["fixed_shape_private_coefficients"]
    assert projected_basis_implementation["cpml_non_cpml_wiring_inherited"]
    assert projected_basis_implementation["jit_safe_reduction"]
    projected_basis_implementation_contract = projected_basis_implementation[
        "implementation_contract"
    ]
    assert projected_basis_implementation_contract["projected_basis_helper"] == (
        "_project_private_modal_basis_packets"
    )
    assert projected_basis_implementation_contract["fixed_shape_reductions"]
    assert projected_basis_implementation_contract[
        "fail_closed_on_projection_energy_missing"
    ]
    assert projected_basis_implementation["material_improvement_demonstrated"] is False
    assert projected_basis_implementation["paired_passed"] is False
    assert projected_basis_implementation["fixture_quality_ready"] is False
    assert projected_basis_implementation["fixture_quality_pending"]
    assert projected_basis_implementation["true_rt_readiness_unlocked"] is False
    assert projected_basis_implementation["production_patch_applied"]
    assert projected_basis_implementation["solver_behavior_changed"]
    assert projected_basis_implementation["field_update_behavior_changed"]
    assert projected_basis_implementation["new_solver_hunk_retained"]
    assert projected_basis_implementation["next_lane_requires_projected_basis_parity_scoring"]
    projected_basis_implementation_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in projected_basis_implementation["candidate_ladder"]
    }
    assert projected_basis_implementation_candidates[
        "V1_projected_basis_packet_helper"
    ]["projected_basis_helper_retained"]
    assert projected_basis_implementation_candidates[
        "V2_modal_retry_consumer_wiring"
    ]["helper_consumed_before_modal_subtraction"]
    assert projected_basis_implementation_candidates[
        "V3_combined_projected_basis_implementation_hunk"
    ]["accepted_candidate"]
    assert projected_basis_implementation_candidates[
        "V4_projected_basis_implementation_blocked"
    ]["accepted_candidate"] is False
    assert projected_basis_implementation["public_claim_allowed"] is False
    assert projected_basis_implementation["public_observable_promoted"] is False
    assert projected_basis_implementation["true_rt_public_observable_promoted"] is False
    assert (
        projected_basis_implementation["dft_flux_tfsf_port_sparameter_promoted"]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_basis_implementation_next_prerequisite"
        ]
        == projected_basis_implementation["next_prerequisite"]
    )
    projected_basis_parity_scoring = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_basis_parity_scoring"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_basis_parity_scoring_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_basis_hunk_insufficient_fixture_quality_pending"
    )
    assert projected_basis_parity_scoring["terminal_outcome"] == (
        "private_plane_wave_modal_projection_normalizer_projected_basis_hunk_insufficient_fixture_quality_pending"
    )
    assert projected_basis_parity_scoring[
        "upstream_projected_basis_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_basis_implementation_status"
    ]
    assert projected_basis_parity_scoring["candidate_ladder_declared_before_solver_edit"]
    assert projected_basis_parity_scoring["candidate_ladder_declared_before_slow_scoring"]
    assert projected_basis_parity_scoring["candidate_count"] == 5
    assert projected_basis_parity_scoring["selected_candidate_id"] == (
        "W1_finite_projected_basis_private_parity_score"
    )
    assert projected_basis_parity_scoring["baseline_metrics"] == (
        projected_basis_implementation["baseline_metrics"]
    )
    assert projected_basis_parity_scoring["metrics"] == (
        projected_basis_implementation["metrics"]
    )
    assert projected_basis_parity_scoring["thresholds"] == (
        projected_basis_implementation["thresholds"]
    )
    assert projected_basis_parity_scoring["threshold_results"] == (
        projected_basis_implementation["threshold_results"]
    )
    assert projected_basis_parity_scoring["baseline_metrics_preserved"]
    assert projected_basis_parity_scoring["thresholds_unchanged"]
    assert projected_basis_parity_scoring["projected_basis_hunk_retained"]
    assert projected_basis_parity_scoring["projected_basis_helper_retained"]
    assert projected_basis_parity_scoring["projected_basis_helper"] == (
        "_project_private_modal_basis_packets"
    )
    assert projected_basis_parity_scoring["projection_gate_fail_closed"]
    assert projected_basis_parity_scoring["contract_gate_fail_closed"]
    assert projected_basis_parity_scoring["parity_scoring_lane_executed"]
    assert projected_basis_parity_scoring["finite_reproducible_score"]
    assert projected_basis_parity_scoring["score_uses_retained_implementation_metrics"]
    assert projected_basis_parity_scoring["material_improvement_demonstrated"] is False
    assert projected_basis_parity_scoring["paired_passed"] is False
    assert projected_basis_parity_scoring["usable_bins_passed"]
    assert projected_basis_parity_scoring["fixture_quality_ready"] is False
    assert projected_basis_parity_scoring["fixture_quality_pending"]
    assert projected_basis_parity_scoring["subgrid_vacuum_parity_scored"]
    assert projected_basis_parity_scoring["subgrid_vacuum_parity_passed"] is False
    assert projected_basis_parity_scoring["true_rt_readiness_unlocked"] is False
    assert projected_basis_parity_scoring["production_patch_applied"] is False
    assert projected_basis_parity_scoring["solver_behavior_changed"] is False
    assert projected_basis_parity_scoring["new_solver_hunk_retained"] is False
    assert projected_basis_parity_scoring["next_lane_requires_projected_basis_failure_theory"]
    projected_basis_parity_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in projected_basis_parity_scoring["candidate_ladder"]
    }
    assert projected_basis_parity_candidates[
        "W1_finite_projected_basis_private_parity_score"
    ]["accepted_candidate"]
    assert projected_basis_parity_candidates[
        "W2_projected_basis_material_improvement_gate"
    ]["accepted_candidate"] is False
    assert projected_basis_parity_candidates[
        "W3_projected_basis_fixture_quality_true_rt_readiness"
    ]["accepted_candidate"] is False
    assert projected_basis_parity_candidates[
        "W4_projected_basis_parity_scoring_blocked"
    ]["accepted_candidate"] is False
    assert projected_basis_parity_scoring["public_claim_allowed"] is False
    assert projected_basis_parity_scoring["public_observable_promoted"] is False
    assert projected_basis_parity_scoring["true_rt_public_observable_promoted"] is False
    assert (
        projected_basis_parity_scoring["dft_flux_tfsf_port_sparameter_promoted"]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_basis_parity_scoring_next_prerequisite"
        ]
        == projected_basis_parity_scoring["next_prerequisite"]
    )
    projected_basis_failure_theory = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_basis_failure_theory"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_basis_failure_theory_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_basis_redesign_contract_ready"
    )
    assert projected_basis_failure_theory["terminal_outcome"] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_basis_redesign_contract_ready"
    )
    assert projected_basis_failure_theory[
        "upstream_projected_basis_parity_scoring_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_basis_parity_scoring_status"
    ]
    assert projected_basis_failure_theory[
        "upstream_projected_basis_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_basis_implementation_status"
    ]
    assert projected_basis_failure_theory["candidate_ladder_declared_before_implementation"]
    assert projected_basis_failure_theory["candidate_ladder_declared_before_solver_edit"]
    assert projected_basis_failure_theory["candidate_ladder_declared_before_slow_scoring"]
    assert projected_basis_failure_theory["candidate_count"] == 5
    assert projected_basis_failure_theory["selected_candidate_id"] == (
        "X1_projected_target_basis_mismatch_theory"
    )
    assert projected_basis_failure_theory["baseline_metrics"] == (
        projected_basis_parity_scoring["baseline_metrics"]
    )
    assert projected_basis_failure_theory["metrics"] == (
        projected_basis_parity_scoring["metrics"]
    )
    assert projected_basis_failure_theory["thresholds"] == (
        projected_basis_parity_scoring["thresholds"]
    )
    assert projected_basis_failure_theory["threshold_results"] == (
        projected_basis_parity_scoring["threshold_results"]
    )
    assert projected_basis_failure_theory["baseline_metrics_preserved"]
    assert projected_basis_failure_theory["thresholds_unchanged"]
    assert projected_basis_failure_theory["projected_basis_parity_scoring_insufficient"]
    assert projected_basis_failure_theory["projected_basis_hunk_retained"]
    assert projected_basis_failure_theory["projection_gate_fail_closed"]
    assert projected_basis_failure_theory["finite_reproducible_score"]
    assert projected_basis_failure_theory["metrics_identical_to_projected_basis_baseline"]
    assert projected_basis_failure_theory["material_improvement_demonstrated"] is False
    assert projected_basis_failure_theory["paired_passed"] is False
    assert projected_basis_failure_theory["fixture_quality_ready"] is False
    assert projected_basis_failure_theory["fixture_quality_pending"]
    assert projected_basis_failure_theory["transverse_phase_floor_persists"]
    assert projected_basis_failure_theory["transverse_magnitude_floor_persists"]
    assert projected_basis_failure_theory["vacuum_stability_floor_persists"]
    assert projected_basis_failure_theory["projected_target_basis_floor_selected"]
    assert projected_basis_failure_theory[
        "projected_target_basis_redesign_contract_ready"
    ]
    projected_target_basis_contract = projected_basis_failure_theory[
        "projected_target_basis_contract"
    ]
    assert projected_target_basis_contract["projection_helper"] == (
        "_project_private_modal_basis_packets"
    )
    assert "unprojected interface packet" in projected_target_basis_contract[
        "current_hunk_behavior"
    ]
    assert projected_target_basis_contract["requires_public_observable"] is False
    assert projected_target_basis_contract["requires_solver_hunk_in_this_lane"] is False
    assert projected_basis_failure_theory["source_projected_interface_unprojected"]
    assert projected_basis_failure_theory["normalizer_weight_floor_deferred"]
    assert projected_basis_failure_theory["temporal_packet_phase_floor_deferred"]
    assert projected_basis_failure_theory["failure_theory_lane_executed"]
    assert projected_basis_failure_theory["true_rt_readiness_unlocked"] is False
    assert projected_basis_failure_theory["production_patch_applied"] is False
    assert projected_basis_failure_theory["solver_behavior_changed"] is False
    assert projected_basis_failure_theory["new_solver_hunk_retained"] is False
    assert projected_basis_failure_theory[
        "next_lane_requires_projected_target_basis_implementation"
    ]
    projected_basis_failure_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in projected_basis_failure_theory["candidate_ladder"]
    }
    assert projected_basis_failure_candidates[
        "X1_projected_target_basis_mismatch_theory"
    ]["accepted_candidate"]
    assert projected_basis_failure_candidates[
        "X2_projected_normalizer_weight_floor_theory"
    ]["accepted_candidate"] is False
    assert projected_basis_failure_candidates[
        "X3_temporal_packet_phase_floor_theory"
    ]["accepted_candidate"] is False
    assert projected_basis_failure_candidates[
        "X4_projected_basis_failure_theory_blocked"
    ]["accepted_candidate"] is False
    assert projected_basis_failure_theory["public_claim_allowed"] is False
    assert projected_basis_failure_theory["public_observable_promoted"] is False
    assert projected_basis_failure_theory["true_rt_public_observable_promoted"] is False
    assert (
        projected_basis_failure_theory["dft_flux_tfsf_port_sparameter_promoted"]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_basis_failure_theory_next_prerequisite"
        ]
        == projected_basis_failure_theory["next_prerequisite"]
    )
    projected_target_implementation = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_implementation_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_basis_hunk_retained_fixture_quality_pending"
    )
    assert projected_target_implementation["terminal_outcome"] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_basis_hunk_retained_fixture_quality_pending"
    )
    assert projected_target_implementation[
        "upstream_projected_basis_failure_theory_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_basis_failure_theory_status"
    ]
    assert (
        projected_target_implementation["candidate_ladder_declared_before_implementation"]
    )
    assert projected_target_implementation["candidate_ladder_declared_before_solver_edit"]
    assert projected_target_implementation["candidate_ladder_declared_before_slow_scoring"]
    assert projected_target_implementation["candidate_count"] == 5
    assert projected_target_implementation["selected_candidate_id"] == (
        "Y3_combined_projected_target_basis_implementation_hunk"
    )
    assert projected_target_implementation["baseline_metrics"] == (
        projected_basis_failure_theory["baseline_metrics"]
    )
    assert projected_target_implementation["metrics"] == (
        projected_basis_failure_theory["metrics"]
    )
    assert projected_target_implementation["thresholds"] == (
        projected_basis_failure_theory["thresholds"]
    )
    assert projected_target_implementation["threshold_results"] == (
        projected_basis_failure_theory["threshold_results"]
    )
    assert projected_target_implementation["baseline_metrics_preserved"]
    assert projected_target_implementation["thresholds_unchanged"]
    assert projected_target_implementation["projected_target_basis_design_ready"]
    assert projected_target_implementation["projected_target_basis_contract_consumed"]
    assert projected_target_implementation["projected_target_basis_helper_retained"]
    assert projected_target_implementation["projected_basis_helper"] == (
        "_project_private_modal_basis_packets"
    )
    assert projected_target_implementation["source_packet_projected"]
    assert projected_target_implementation["interface_packet_projected"]
    assert projected_target_implementation["subtraction_uses_projected_packets_only"]
    assert not projected_target_implementation["source_projected_interface_unprojected"]
    assert projected_target_implementation["modal_retry_consumer_wiring_retained"]
    assert projected_target_implementation["projection_gate_fail_closed"]
    assert projected_target_implementation["contract_gate_fail_closed"]
    assert projected_target_implementation["fail_closed_if_projection_energy_missing"]
    assert projected_target_implementation["fail_closed_if_packet_contract_mismatch"]
    assert projected_target_implementation["reuse_existing_owner_packet_shapes"]
    assert projected_target_implementation["fixed_shape_private_coefficients"]
    assert projected_target_implementation["cpml_non_cpml_wiring_inherited"]
    assert projected_target_implementation["jit_safe_reduction"]
    projected_target_contract = projected_target_implementation[
        "implementation_contract"
    ]
    assert projected_target_contract["projection_helper"] == (
        "_project_private_modal_basis_packets"
    )
    assert "project source and interface packets" in projected_target_contract[
        "shared_target_basis_behavior"
    ]
    assert projected_target_contract["target_packet_formula"] == (
        "projected_interface_packet - projected_source_packet"
    )
    assert projected_target_contract["fixed_shape_reductions"]
    assert projected_target_contract["fail_closed_on_projection_energy_missing"]
    assert projected_target_implementation["material_improvement_demonstrated"] is False
    assert projected_target_implementation["paired_passed"] is False
    assert projected_target_implementation["fixture_quality_ready"] is False
    assert projected_target_implementation["fixture_quality_pending"]
    assert projected_target_implementation["true_rt_readiness_unlocked"] is False
    assert projected_target_implementation["production_patch_applied"]
    assert projected_target_implementation["solver_behavior_changed"]
    assert projected_target_implementation["field_update_behavior_changed"]
    assert projected_target_implementation["new_solver_hunk_retained"]
    assert projected_target_implementation[
        "next_lane_requires_projected_target_basis_parity_scoring"
    ]
    projected_target_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in projected_target_implementation["candidate_ladder"]
    }
    assert projected_target_candidates[
        "Y1_project_interface_packet_into_shared_basis"
    ]["interface_packet_projected"]
    assert projected_target_candidates[
        "Y2_source_interface_target_basis_subtraction"
    ]["subtraction_uses_projected_packets_only"]
    assert projected_target_candidates[
        "Y3_combined_projected_target_basis_implementation_hunk"
    ]["accepted_candidate"]
    assert (
        projected_target_candidates[
            "Y4_projected_target_basis_implementation_blocked"
        ]["accepted_candidate"]
        is False
    )
    assert projected_target_implementation["public_claim_allowed"] is False
    assert projected_target_implementation["public_observable_promoted"] is False
    assert (
        projected_target_implementation["true_rt_public_observable_promoted"] is False
    )
    assert (
        projected_target_implementation["dft_flux_tfsf_port_sparameter_promoted"]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_implementation_next_prerequisite"
        ]
        == projected_target_implementation["next_prerequisite"]
    )
    projected_target_parity_scoring = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_parity_scoring"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_parity_scoring_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_basis_hunk_insufficient_fixture_quality_pending"
    )
    assert projected_target_parity_scoring["terminal_outcome"] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_basis_hunk_insufficient_fixture_quality_pending"
    )
    assert projected_target_parity_scoring[
        "upstream_projected_target_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_implementation_status"
    ]
    assert projected_target_parity_scoring["candidate_ladder_declared_before_solver_edit"]
    assert projected_target_parity_scoring["candidate_ladder_declared_before_slow_scoring"]
    assert projected_target_parity_scoring["candidate_count"] == 5
    assert projected_target_parity_scoring["selected_candidate_id"] == (
        "Z1_finite_projected_target_private_parity_score"
    )
    assert projected_target_parity_scoring["baseline_metrics"] == (
        projected_target_implementation["baseline_metrics"]
    )
    assert projected_target_parity_scoring["metrics"] == (
        projected_target_implementation["metrics"]
    )
    assert projected_target_parity_scoring["thresholds"] == (
        projected_target_implementation["thresholds"]
    )
    assert projected_target_parity_scoring["threshold_results"] == (
        projected_target_implementation["threshold_results"]
    )
    assert projected_target_parity_scoring["baseline_metrics_preserved"]
    assert projected_target_parity_scoring["thresholds_unchanged"]
    assert projected_target_parity_scoring["projected_target_basis_hunk_retained"]
    assert projected_target_parity_scoring["projected_target_basis_helper_retained"]
    assert projected_target_parity_scoring["projected_basis_helper"] == (
        "_project_private_modal_basis_packets"
    )
    assert projected_target_parity_scoring["source_packet_projected"]
    assert projected_target_parity_scoring["interface_packet_projected"]
    assert projected_target_parity_scoring["subtraction_uses_projected_packets_only"]
    assert not projected_target_parity_scoring["source_projected_interface_unprojected"]
    assert projected_target_parity_scoring["projection_gate_fail_closed"]
    assert projected_target_parity_scoring["contract_gate_fail_closed"]
    assert projected_target_parity_scoring["parity_scoring_lane_executed"]
    assert projected_target_parity_scoring["finite_reproducible_score"]
    assert projected_target_parity_scoring["score_uses_retained_implementation_metrics"]
    assert projected_target_parity_scoring["material_improvement_demonstrated"] is False
    assert projected_target_parity_scoring["paired_passed"] is False
    assert projected_target_parity_scoring["usable_bins_passed"]
    assert projected_target_parity_scoring["fixture_quality_ready"] is False
    assert projected_target_parity_scoring["fixture_quality_pending"]
    assert projected_target_parity_scoring["subgrid_vacuum_parity_scored"]
    assert projected_target_parity_scoring["subgrid_vacuum_parity_passed"] is False
    assert projected_target_parity_scoring["true_rt_readiness_unlocked"] is False
    assert projected_target_parity_scoring["production_patch_applied"] is False
    assert projected_target_parity_scoring["solver_behavior_changed"] is False
    assert projected_target_parity_scoring["new_solver_hunk_retained"] is False
    assert projected_target_parity_scoring[
        "next_lane_requires_projected_target_basis_failure_theory"
    ]
    projected_target_parity_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in projected_target_parity_scoring["candidate_ladder"]
    }
    assert projected_target_parity_candidates[
        "Z1_finite_projected_target_private_parity_score"
    ]["accepted_candidate"]
    assert (
        projected_target_parity_candidates[
            "Z2_projected_target_material_improvement_gate"
        ]["accepted_candidate"]
        is False
    )
    assert (
        projected_target_parity_candidates[
            "Z3_projected_target_fixture_quality_true_rt_readiness"
        ]["accepted_candidate"]
        is False
    )
    assert (
        projected_target_parity_candidates[
            "Z4_projected_target_parity_scoring_blocked"
        ]["accepted_candidate"]
        is False
    )
    assert projected_target_parity_scoring["public_claim_allowed"] is False
    assert projected_target_parity_scoring["public_observable_promoted"] is False
    assert (
        projected_target_parity_scoring["true_rt_public_observable_promoted"] is False
    )
    assert (
        projected_target_parity_scoring["dft_flux_tfsf_port_sparameter_promoted"]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_parity_scoring_next_prerequisite"
        ]
        == projected_target_parity_scoring["next_prerequisite"]
    )
    projected_target_failure_theory = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_failure_theory"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_failure_theory_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_redesign_contract_ready"
    )
    assert projected_target_failure_theory["terminal_outcome"] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_redesign_contract_ready"
    )
    assert projected_target_failure_theory[
        "upstream_projected_target_parity_scoring_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_parity_scoring_status"
    ]
    assert projected_target_failure_theory[
        "upstream_projected_target_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_implementation_status"
    ]
    assert projected_target_failure_theory["candidate_ladder_declared_before_implementation"]
    assert projected_target_failure_theory["candidate_ladder_declared_before_solver_edit"]
    assert projected_target_failure_theory["candidate_ladder_declared_before_slow_scoring"]
    assert projected_target_failure_theory["candidate_count"] == 5
    assert projected_target_failure_theory["selected_candidate_id"] == (
        "AA1_residual_basis_mismatch_theory"
    )
    assert projected_target_failure_theory["baseline_metrics"] == (
        projected_target_parity_scoring["baseline_metrics"]
    )
    assert projected_target_failure_theory["metrics"] == (
        projected_target_parity_scoring["metrics"]
    )
    assert projected_target_failure_theory["thresholds"] == (
        projected_target_parity_scoring["thresholds"]
    )
    assert projected_target_failure_theory["threshold_results"] == (
        projected_target_parity_scoring["threshold_results"]
    )
    assert projected_target_failure_theory["baseline_metrics_preserved"]
    assert projected_target_failure_theory["thresholds_unchanged"]
    assert projected_target_failure_theory["projected_target_parity_scoring_insufficient"]
    assert projected_target_failure_theory["projected_target_basis_hunk_retained"]
    assert projected_target_failure_theory["source_packet_projected"]
    assert projected_target_failure_theory["interface_packet_projected"]
    assert projected_target_failure_theory["subtraction_uses_projected_packets_only"]
    assert projected_target_failure_theory["single_incident_basis_only"]
    assert projected_target_failure_theory["projection_gate_fail_closed"]
    assert projected_target_failure_theory["finite_reproducible_score"]
    assert projected_target_failure_theory[
        "metrics_identical_to_projected_target_baseline"
    ]
    assert projected_target_failure_theory["material_improvement_demonstrated"] is False
    assert projected_target_failure_theory["paired_passed"] is False
    assert projected_target_failure_theory["fixture_quality_ready"] is False
    assert projected_target_failure_theory["fixture_quality_pending"]
    assert projected_target_failure_theory["transverse_phase_floor_persists"]
    assert projected_target_failure_theory["transverse_magnitude_floor_persists"]
    assert projected_target_failure_theory["vacuum_stability_floor_persists"]
    assert projected_target_failure_theory["residual_basis_floor_selected"]
    assert projected_target_failure_theory[
        "projected_target_residual_basis_redesign_contract_ready"
    ]
    residual_basis_contract = projected_target_failure_theory[
        "residual_basis_contract"
    ]
    assert residual_basis_contract["projection_helper"] == (
        "_project_private_modal_basis_packets"
    )
    assert "single incident-normalizer basis" in residual_basis_contract[
        "current_hunk_behavior"
    ]
    assert "residual/reflected/transverse" in residual_basis_contract[
        "missing_contract"
    ]
    assert residual_basis_contract["requires_public_observable"] is False
    assert residual_basis_contract["requires_solver_hunk_in_this_lane"] is False
    assert projected_target_failure_theory["normalizer_weight_floor_deferred"]
    assert projected_target_failure_theory["temporal_packet_phase_floor_deferred"]
    assert projected_target_failure_theory["failure_theory_lane_executed"]
    assert projected_target_failure_theory["true_rt_readiness_unlocked"] is False
    assert projected_target_failure_theory["production_patch_applied"] is False
    assert projected_target_failure_theory["solver_behavior_changed"] is False
    assert projected_target_failure_theory["new_solver_hunk_retained"] is False
    assert projected_target_failure_theory[
        "next_lane_requires_projected_target_residual_basis_design"
    ]
    projected_target_failure_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in projected_target_failure_theory["candidate_ladder"]
    }
    assert projected_target_failure_candidates[
        "AA1_residual_basis_mismatch_theory"
    ]["accepted_candidate"]
    assert (
        projected_target_failure_candidates[
            "AA2_projected_target_normalizer_weight_floor_theory"
        ]["accepted_candidate"]
        is False
    )
    assert (
        projected_target_failure_candidates[
            "AA3_projected_target_temporal_packet_phase_floor_theory"
        ]["accepted_candidate"]
        is False
    )
    assert (
        projected_target_failure_candidates[
            "AA4_projected_target_failure_theory_blocked"
        ]["accepted_candidate"]
        is False
    )
    assert projected_target_failure_theory["public_claim_allowed"] is False
    assert projected_target_failure_theory["public_observable_promoted"] is False
    assert (
        projected_target_failure_theory["true_rt_public_observable_promoted"] is False
    )
    assert (
        projected_target_failure_theory["dft_flux_tfsf_port_sparameter_promoted"]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_failure_theory_next_prerequisite"
        ]
        == projected_target_failure_theory["next_prerequisite"]
    )
    residual_basis_design = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_design"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_design_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_contract_design_ready"
    )
    assert residual_basis_design["terminal_outcome"] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_contract_design_ready"
    )
    assert residual_basis_design[
        "upstream_projected_target_failure_theory_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_failure_theory_status"
    ]
    assert residual_basis_design["candidate_ladder_declared_before_implementation"]
    assert residual_basis_design["candidate_ladder_declared_before_solver_edit"]
    assert residual_basis_design["candidate_ladder_declared_before_slow_scoring"]
    assert residual_basis_design["candidate_count"] == 5
    assert residual_basis_design["selected_candidate_id"] == (
        "AB3_combined_residual_basis_design_contract"
    )
    assert residual_basis_design["baseline_metrics"] == (
        projected_target_failure_theory["baseline_metrics"]
    )
    assert residual_basis_design["metrics"] == (
        projected_target_failure_theory["metrics"]
    )
    assert residual_basis_design["thresholds"] == (
        projected_target_failure_theory["thresholds"]
    )
    assert residual_basis_design["threshold_results"] == (
        projected_target_failure_theory["threshold_results"]
    )
    assert residual_basis_design["baseline_metrics_preserved"]
    assert residual_basis_design["thresholds_unchanged"]
    assert residual_basis_design["failure_theory_contract_consumed"]
    assert residual_basis_design["projected_target_residual_basis_design_ready"]
    assert residual_basis_design["residual_basis_schema_ready"]
    assert residual_basis_design["residual_coefficient_contract_ready"]
    assert residual_basis_design["normalizer_contract_ready"]
    assert residual_basis_design["residual_basis_contract_ready"]
    assert residual_basis_design["bounded_private_implementation_contract_ready"]
    assert residual_basis_design["basis_vectors"] == [
        "incident_normal_mode",
        "reflected_normal_mode",
        "transverse_residual_mode",
    ]
    assert residual_basis_design["source_packet_projected"]
    assert residual_basis_design["interface_packet_projected"]
    assert residual_basis_design["subtraction_uses_projected_packets_only"]
    assert residual_basis_design["single_incident_basis_replaced"]
    assert residual_basis_design["single_incident_basis_only"] is False
    assert residual_basis_design["projection_gate_fail_closed"]
    assert residual_basis_design["contract_gate_fail_closed"]
    assert residual_basis_design["fail_closed_if_projection_energy_missing"]
    assert residual_basis_design["reuse_existing_owner_packet_shapes"]
    assert residual_basis_design["fixed_shape_private_coefficients"]
    assert residual_basis_design["fixed_shape_reductions"]
    assert residual_basis_design["jit_safe_reduction"]
    assert residual_basis_design["no_threshold_laundering"]
    design_contract = residual_basis_design["residual_basis_contract"]
    assert design_contract["projection_helper"] == "_project_private_modal_basis_packets"
    assert "incident, reflected" in design_contract["target_basis_behavior"]
    assert design_contract["single_incident_basis_replaced"]
    assert design_contract["fixed_shape_reductions"]
    assert design_contract["no_threshold_laundering"]
    design_schema = residual_basis_design["residual_basis_schema"]
    assert design_schema["basis_vectors"] == residual_basis_design["basis_vectors"]
    assert design_schema["fixed_shape_reductions"]
    assert design_schema["jit_safe_reduction"]
    assert design_schema["requires_public_observable"] is False
    assert residual_basis_design["material_improvement_demonstrated"] is False
    assert residual_basis_design["paired_passed"] is False
    assert residual_basis_design["fixture_quality_ready"] is False
    assert residual_basis_design["fixture_quality_pending"]
    assert residual_basis_design["true_rt_readiness_unlocked"] is False
    assert residual_basis_design["production_patch_applied"] is False
    assert residual_basis_design["solver_behavior_changed"] is False
    assert residual_basis_design["new_solver_hunk_retained"] is False
    assert residual_basis_design[
        "next_lane_requires_projected_target_residual_basis_implementation"
    ]
    residual_basis_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_basis_design["candidate_ladder"]
    }
    assert residual_basis_candidates[
        "AB1_residual_reflected_transverse_basis_schema"
    ]["residual_basis_schema_ready"]
    assert residual_basis_candidates[
        "AB2_target_residual_coefficient_normalizer_contract"
    ]["residual_coefficient_contract_ready"]
    assert residual_basis_candidates[
        "AB3_combined_residual_basis_design_contract"
    ]["accepted_candidate"]
    assert (
        residual_basis_candidates[
            "AB4_residual_basis_design_blocked"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_design["public_claim_allowed"] is False
    assert residual_basis_design["public_observable_promoted"] is False
    assert residual_basis_design["true_rt_public_observable_promoted"] is False
    assert residual_basis_design["dft_flux_tfsf_port_sparameter_promoted"] is False
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_design_next_prerequisite"
        ]
        == residual_basis_design["next_prerequisite"]
    )
    residual_basis_implementation = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_implementation_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_hunk_retained_fixture_quality_pending"
    )
    assert residual_basis_implementation["terminal_outcome"] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_hunk_retained_fixture_quality_pending"
    )
    assert residual_basis_implementation["upstream_residual_basis_design_status"] == (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_design_status"
        ]
    )
    assert residual_basis_implementation["candidate_ladder_declared_before_implementation"]
    assert residual_basis_implementation["candidate_ladder_declared_before_solver_edit"]
    assert residual_basis_implementation["candidate_ladder_declared_before_slow_scoring"]
    assert residual_basis_implementation["candidate_count"] == 5
    assert residual_basis_implementation["selected_candidate_id"] == (
        "AC3_combined_residual_basis_implementation_hunk"
    )
    assert residual_basis_implementation["baseline_metrics"] == (
        residual_basis_design["baseline_metrics"]
    )
    assert residual_basis_implementation["metrics"] == residual_basis_design["metrics"]
    assert residual_basis_implementation["thresholds"] == (
        residual_basis_design["thresholds"]
    )
    assert residual_basis_implementation["threshold_results"] == (
        residual_basis_design["threshold_results"]
    )
    assert residual_basis_implementation["residual_basis_design_ready"]
    assert residual_basis_implementation["residual_basis_contract_consumed"]
    assert residual_basis_implementation["residual_basis_helper_retained"]
    assert residual_basis_implementation["projection_helper"] == (
        "_project_private_modal_basis_packets"
    )
    assert residual_basis_implementation["consumer_helper"] == (
        "_apply_propagation_aware_modal_retry_face_helper"
    )
    assert residual_basis_implementation["basis_vectors"] == [
        "incident_normal_mode",
        "reflected_normal_mode",
        "transverse_residual_mode",
    ]
    assert residual_basis_implementation["residual_basis_modes_projected"]
    assert residual_basis_implementation["source_packet_projected"]
    assert residual_basis_implementation["interface_packet_projected"]
    assert residual_basis_implementation["subtraction_uses_projected_packets_only"]
    assert residual_basis_implementation["single_incident_basis_replaced"]
    assert residual_basis_implementation["single_incident_basis_only"] is False
    assert residual_basis_implementation["projection_gate_fail_closed"]
    assert residual_basis_implementation["contract_gate_fail_closed"]
    assert residual_basis_implementation["fail_closed_if_projection_energy_missing"]
    assert residual_basis_implementation["reuse_existing_owner_packet_shapes"]
    assert residual_basis_implementation["fixed_shape_private_coefficients"]
    assert residual_basis_implementation["fixed_shape_reductions"]
    assert residual_basis_implementation["cpml_non_cpml_wiring_inherited"]
    assert residual_basis_implementation["jit_safe_reduction"]
    assert residual_basis_implementation["no_threshold_laundering"]
    assert residual_basis_implementation["production_patch_applied"]
    assert residual_basis_implementation["solver_behavior_changed"]
    assert residual_basis_implementation["field_update_behavior_changed"]
    assert residual_basis_implementation["new_solver_hunk_retained"]
    assert residual_basis_implementation["solver_hunk_retained"]
    assert residual_basis_implementation["true_rt_readiness_unlocked"] is False
    assert residual_basis_implementation[
        "next_lane_requires_projected_target_residual_basis_parity_scoring"
    ]
    residual_basis_implementation_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_basis_implementation["candidate_ladder"]
    }
    assert residual_basis_implementation_candidates[
        "AC1_residual_reflected_transverse_projection_helper"
    ]["fixed_shape_reductions"]
    assert residual_basis_implementation_candidates[
        "AC2_target_residual_coefficient_subtraction"
    ]["subtraction_uses_projected_packets_only"]
    assert residual_basis_implementation_candidates[
        "AC3_combined_residual_basis_implementation_hunk"
    ]["accepted_candidate"]
    assert (
        residual_basis_implementation_candidates[
            "AC4_residual_basis_implementation_blocked"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_implementation["public_claim_allowed"] is False
    assert residual_basis_implementation["public_observable_promoted"] is False
    assert (
        residual_basis_implementation["true_rt_public_observable_promoted"] is False
    )
    assert (
        residual_basis_implementation["dft_flux_tfsf_port_sparameter_promoted"]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_implementation_next_prerequisite"
        ]
        == residual_basis_implementation["next_prerequisite"]
    )
    residual_basis_parity_scoring = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_parity_scoring"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_parity_scoring_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_hunk_insufficient_fixture_quality_pending"
    )
    assert residual_basis_parity_scoring["terminal_outcome"] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_hunk_insufficient_fixture_quality_pending"
    )
    assert residual_basis_parity_scoring[
        "upstream_residual_basis_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_implementation_status"
    ]
    assert residual_basis_parity_scoring["candidate_ladder_declared_before_implementation"]
    assert residual_basis_parity_scoring["candidate_ladder_declared_before_solver_edit"]
    assert residual_basis_parity_scoring["candidate_ladder_declared_before_slow_scoring"]
    assert residual_basis_parity_scoring["candidate_count"] == 5
    assert residual_basis_parity_scoring["selected_candidate_id"] == (
        "AD1_finite_residual_basis_private_parity_score"
    )
    assert residual_basis_parity_scoring["baseline_metrics"] == (
        residual_basis_implementation["baseline_metrics"]
    )
    assert residual_basis_parity_scoring["metrics"] == (
        residual_basis_implementation["metrics"]
    )
    assert residual_basis_parity_scoring["thresholds"] == (
        residual_basis_implementation["thresholds"]
    )
    assert residual_basis_parity_scoring["threshold_results"] == (
        residual_basis_implementation["threshold_results"]
    )
    assert residual_basis_parity_scoring["baseline_metrics_preserved"]
    assert residual_basis_parity_scoring["thresholds_unchanged"]
    assert residual_basis_parity_scoring["residual_basis_hunk_retained"]
    assert residual_basis_parity_scoring["residual_basis_helper_retained"]
    assert residual_basis_parity_scoring["projection_helper"] == (
        "_project_private_modal_basis_packets"
    )
    assert residual_basis_parity_scoring["consumer_helper"] == (
        "_apply_propagation_aware_modal_retry_face_helper"
    )
    assert residual_basis_parity_scoring["basis_vectors"] == [
        "incident_normal_mode",
        "reflected_normal_mode",
        "transverse_residual_mode",
    ]
    assert residual_basis_parity_scoring["residual_basis_modes_projected"]
    assert residual_basis_parity_scoring["source_packet_projected"]
    assert residual_basis_parity_scoring["interface_packet_projected"]
    assert residual_basis_parity_scoring["subtraction_uses_projected_packets_only"]
    assert residual_basis_parity_scoring["single_incident_basis_replaced"]
    assert residual_basis_parity_scoring["single_incident_basis_only"] is False
    assert residual_basis_parity_scoring["projection_gate_fail_closed"]
    assert residual_basis_parity_scoring["contract_gate_fail_closed"]
    assert residual_basis_parity_scoring["fail_closed_if_projection_energy_missing"]
    assert residual_basis_parity_scoring["parity_scoring_lane_executed"]
    assert residual_basis_parity_scoring["finite_reproducible_score"]
    assert residual_basis_parity_scoring["score_uses_retained_implementation_metrics"]
    assert residual_basis_parity_scoring["material_improvement_demonstrated"] is False
    assert residual_basis_parity_scoring["paired_passed"] is False
    assert residual_basis_parity_scoring["fixture_quality_ready"] is False
    assert residual_basis_parity_scoring["fixture_quality_pending"]
    assert residual_basis_parity_scoring["subgrid_vacuum_parity_scored"]
    assert residual_basis_parity_scoring["subgrid_vacuum_parity_passed"] is False
    assert residual_basis_parity_scoring["true_rt_readiness_unlocked"] is False
    assert residual_basis_parity_scoring["slab_rt_scored"] is False
    assert residual_basis_parity_scoring["production_patch_applied"] is False
    assert residual_basis_parity_scoring["solver_behavior_changed"] is False
    assert residual_basis_parity_scoring["new_solver_hunk_retained"] is False
    assert residual_basis_parity_scoring["retained_residual_basis_solver_hunk"]
    assert residual_basis_parity_scoring[
        "next_lane_requires_projected_target_residual_basis_failure_theory"
    ]
    residual_basis_parity_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_basis_parity_scoring["candidate_ladder"]
    }
    assert residual_basis_parity_candidates[
        "AD1_finite_residual_basis_private_parity_score"
    ]["accepted_candidate"]
    assert residual_basis_parity_candidates[
        "AD2_residual_basis_material_improvement_gate"
    ]["paired_passed"] is False
    assert residual_basis_parity_candidates[
        "AD3_residual_basis_fixture_quality_true_rt_readiness"
    ]["true_rt_readiness_unlocked"] is False
    assert (
        residual_basis_parity_candidates[
            "AD4_residual_basis_parity_scoring_insufficient"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_parity_scoring["public_claim_allowed"] is False
    assert residual_basis_parity_scoring["public_observable_promoted"] is False
    assert (
        residual_basis_parity_scoring["true_rt_public_observable_promoted"] is False
    )
    assert (
        residual_basis_parity_scoring["dft_flux_tfsf_port_sparameter_promoted"]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_parity_scoring_next_prerequisite"
        ]
        == residual_basis_parity_scoring["next_prerequisite"]
    )
    residual_basis_failure_theory = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_failure_theory"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_failure_theory_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_modal_orthogonality_floor_theory_ready"
    )
    assert residual_basis_failure_theory["terminal_outcome"] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_modal_orthogonality_floor_theory_ready"
    )
    assert residual_basis_failure_theory[
        "upstream_residual_basis_parity_scoring_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_parity_scoring_status"
    ]
    assert residual_basis_failure_theory[
        "upstream_residual_basis_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_implementation_status"
    ]
    assert residual_basis_failure_theory["candidate_ladder_declared_before_implementation"]
    assert residual_basis_failure_theory["candidate_ladder_declared_before_solver_edit"]
    assert residual_basis_failure_theory["candidate_ladder_declared_before_slow_scoring"]
    assert residual_basis_failure_theory["candidate_count"] == 5
    assert residual_basis_failure_theory["selected_candidate_id"] == (
        "AE1_modal_orthogonality_energy_inner_product_floor"
    )
    assert residual_basis_failure_theory["baseline_metrics"] == (
        residual_basis_parity_scoring["baseline_metrics"]
    )
    assert residual_basis_failure_theory["metrics"] == (
        residual_basis_parity_scoring["metrics"]
    )
    assert residual_basis_failure_theory["thresholds"] == (
        residual_basis_parity_scoring["thresholds"]
    )
    assert residual_basis_failure_theory["threshold_results"] == (
        residual_basis_parity_scoring["threshold_results"]
    )
    assert residual_basis_failure_theory["baseline_metrics_preserved"]
    assert residual_basis_failure_theory["thresholds_unchanged"]
    assert residual_basis_failure_theory["residual_basis_parity_scoring_insufficient"]
    assert residual_basis_failure_theory["residual_basis_hunk_retained"]
    assert residual_basis_failure_theory["residual_basis_modes_projected"]
    assert residual_basis_failure_theory["source_packet_projected"]
    assert residual_basis_failure_theory["interface_packet_projected"]
    assert residual_basis_failure_theory["subtraction_uses_projected_packets_only"]
    assert residual_basis_failure_theory["single_incident_basis_replaced"]
    assert residual_basis_failure_theory["single_incident_basis_only"] is False
    assert residual_basis_failure_theory["projection_gate_fail_closed"]
    assert residual_basis_failure_theory["contract_gate_fail_closed"]
    assert residual_basis_failure_theory["finite_reproducible_score"]
    assert residual_basis_failure_theory["scalar_l2_orthogonality_only"]
    assert residual_basis_failure_theory["energy_biorthogonal_basis_missing"]
    assert residual_basis_failure_theory["characteristic_impedance_weighting_missing"]
    assert residual_basis_failure_theory["source_interface_packet_timing_floor_deferred"]
    assert residual_basis_failure_theory["normalizer_weight_mask_floor_deferred"]
    assert residual_basis_failure_theory["failure_theory_lane_executed"]
    assert residual_basis_failure_theory["material_improvement_demonstrated"] is False
    assert residual_basis_failure_theory["paired_passed"] is False
    assert residual_basis_failure_theory["fixture_quality_ready"] is False
    assert residual_basis_failure_theory["fixture_quality_pending"]
    assert residual_basis_failure_theory["subgrid_vacuum_parity_scored"]
    assert residual_basis_failure_theory["subgrid_vacuum_parity_passed"] is False
    assert residual_basis_failure_theory["true_rt_readiness_unlocked"] is False
    assert residual_basis_failure_theory["slab_rt_scored"] is False
    assert residual_basis_failure_theory["production_patch_applied"] is False
    assert residual_basis_failure_theory["solver_behavior_changed"] is False
    assert residual_basis_failure_theory["new_solver_hunk_retained"] is False
    assert residual_basis_failure_theory[
        "next_lane_requires_energy_biorthogonal_residual_basis_design"
    ]
    floor_contract = residual_basis_failure_theory["modal_floor_contract"]
    assert floor_contract["projection_helper"] == "_project_private_modal_basis_packets"
    assert "scalar weighted complex L2" in floor_contract["current_hunk_behavior"]
    assert "energy-biorthogonal" in floor_contract["remaining_floor"]
    assert floor_contract["requires_public_observable"] is False
    assert floor_contract["requires_solver_hunk_in_this_lane"] is False
    residual_basis_failure_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_basis_failure_theory["candidate_ladder"]
    }
    assert residual_basis_failure_candidates[
        "AE1_modal_orthogonality_energy_inner_product_floor"
    ]["accepted_candidate"]
    assert (
        residual_basis_failure_candidates[
            "AE2_source_interface_packet_timing_floor"
        ]["accepted_candidate"]
        is False
    )
    assert (
        residual_basis_failure_candidates[
            "AE3_normalizer_weight_mask_floor"
        ]["accepted_candidate"]
        is False
    )
    assert (
        residual_basis_failure_candidates[
            "AE4_residual_basis_failure_theory_blocked"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_failure_theory["public_claim_allowed"] is False
    assert residual_basis_failure_theory["public_observable_promoted"] is False
    assert (
        residual_basis_failure_theory["true_rt_public_observable_promoted"] is False
    )
    assert (
        residual_basis_failure_theory["dft_flux_tfsf_port_sparameter_promoted"]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_failure_theory_next_prerequisite"
        ]
        == residual_basis_failure_theory["next_prerequisite"]
    )
    residual_basis_energy_biorthogonal_design = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_design"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_design_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_contract_design_ready"
    )
    assert residual_basis_energy_biorthogonal_design["terminal_outcome"] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_contract_design_ready"
    )
    assert residual_basis_energy_biorthogonal_design[
        "upstream_residual_basis_failure_theory_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_failure_theory_status"
    ]
    assert residual_basis_energy_biorthogonal_design[
        "candidate_ladder_declared_before_implementation"
    ]
    assert residual_basis_energy_biorthogonal_design[
        "candidate_ladder_declared_before_solver_edit"
    ]
    assert residual_basis_energy_biorthogonal_design[
        "candidate_ladder_declared_before_slow_scoring"
    ]
    assert residual_basis_energy_biorthogonal_design["candidate_count"] == 5
    assert residual_basis_energy_biorthogonal_design["selected_candidate_id"] == (
        "AF3_combined_energy_biorthogonal_design_contract"
    )
    assert residual_basis_energy_biorthogonal_design["baseline_metrics"] == (
        residual_basis_failure_theory["baseline_metrics"]
    )
    assert residual_basis_energy_biorthogonal_design["metrics"] == (
        residual_basis_failure_theory["metrics"]
    )
    assert residual_basis_energy_biorthogonal_design["thresholds"] == (
        residual_basis_failure_theory["thresholds"]
    )
    assert residual_basis_energy_biorthogonal_design["threshold_results"] == (
        residual_basis_failure_theory["threshold_results"]
    )
    assert residual_basis_energy_biorthogonal_design["baseline_metrics_preserved"]
    assert residual_basis_energy_biorthogonal_design["thresholds_unchanged"]
    assert residual_basis_energy_biorthogonal_design[
        "failure_theory_contract_consumed"
    ]
    assert residual_basis_energy_biorthogonal_design["energy_biorthogonal_design_ready"]
    assert residual_basis_energy_biorthogonal_design["energy_biorthogonal_schema_ready"]
    assert residual_basis_energy_biorthogonal_design[
        "characteristic_impedance_weighting_contract_ready"
    ]
    assert residual_basis_energy_biorthogonal_design[
        "energy_inner_product_contract_ready"
    ]
    assert residual_basis_energy_biorthogonal_design[
        "bounded_private_implementation_contract_ready"
    ]
    assert residual_basis_energy_biorthogonal_design["modal_floor_contract"] == (
        floor_contract
    )
    energy_schema = residual_basis_energy_biorthogonal_design[
        "energy_biorthogonal_schema"
    ]
    assert energy_schema["state_owner"] == "_PrivateInterfaceOwnerState"
    assert (
        energy_schema["consumer_helper"]
        == "_apply_propagation_aware_modal_retry_face_helper"
    )
    assert energy_schema["projection_helper"] == "_project_private_modal_basis_packets"
    assert energy_schema["basis_vectors"] == [
        "incident_characteristic_power_mode",
        "reflected_characteristic_power_mode",
        "transverse_residual_power_mode",
    ]
    assert "tangential E/H power form" in energy_schema["inner_product"]
    assert energy_schema["fixed_shape_reductions"]
    assert energy_schema["jit_safe_reduction"]
    assert energy_schema["requires_public_observable"] is False
    energy_contract = residual_basis_energy_biorthogonal_design[
        "energy_biorthogonal_contract"
    ]
    assert energy_contract["upstream_modal_floor_contract"] == floor_contract
    assert energy_contract["basis_schema"] == energy_schema
    assert energy_contract["scalar_l2_orthogonality_replaced"]
    assert energy_contract["characteristic_impedance_weighting_contract_ready"]
    assert energy_contract["energy_inner_product_contract_ready"]
    assert energy_contract["fail_closed_on_projection_energy_missing"]
    assert energy_contract["reuse_existing_owner_packet_shapes"]
    assert energy_contract["fixed_shape_reductions"]
    assert energy_contract["no_threshold_laundering"]
    assert residual_basis_energy_biorthogonal_design["basis_vectors"] == (
        energy_schema["basis_vectors"]
    )
    assert residual_basis_energy_biorthogonal_design[
        "scalar_l2_orthogonality_replaced"
    ]
    assert residual_basis_energy_biorthogonal_design["source_packet_projected"]
    assert residual_basis_energy_biorthogonal_design["interface_packet_projected"]
    assert residual_basis_energy_biorthogonal_design[
        "subtraction_uses_projected_packets_only"
    ]
    assert residual_basis_energy_biorthogonal_design[
        "single_incident_basis_replaced"
    ]
    assert (
        residual_basis_energy_biorthogonal_design["single_incident_basis_only"]
        is False
    )
    assert residual_basis_energy_biorthogonal_design["projection_gate_fail_closed"]
    assert residual_basis_energy_biorthogonal_design["contract_gate_fail_closed"]
    assert residual_basis_energy_biorthogonal_design[
        "fail_closed_if_projection_energy_missing"
    ]
    assert residual_basis_energy_biorthogonal_design[
        "reuse_existing_owner_packet_shapes"
    ]
    assert residual_basis_energy_biorthogonal_design[
        "fixed_shape_private_coefficients"
    ]
    assert residual_basis_energy_biorthogonal_design["fixed_shape_reductions"]
    assert residual_basis_energy_biorthogonal_design[
        "cpml_non_cpml_wiring_inherited"
    ]
    assert residual_basis_energy_biorthogonal_design["jit_safe_reduction"]
    assert residual_basis_energy_biorthogonal_design["no_threshold_laundering"]
    assert (
        residual_basis_energy_biorthogonal_design["material_improvement_demonstrated"]
        is False
    )
    assert residual_basis_energy_biorthogonal_design["paired_passed"] is False
    assert (
        residual_basis_energy_biorthogonal_design["fixture_quality_ready"] is False
    )
    assert residual_basis_energy_biorthogonal_design["fixture_quality_pending"]
    assert (
        residual_basis_energy_biorthogonal_design["true_rt_readiness_unlocked"]
        is False
    )
    assert residual_basis_energy_biorthogonal_design["slab_rt_scored"] is False
    assert (
        residual_basis_energy_biorthogonal_design["production_patch_applied"]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_design["solver_behavior_changed"]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_design["field_update_behavior_changed"]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_design["runner_behavior_changed"]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_design["new_solver_hunk_retained"]
        is False
    )
    residual_basis_energy_design_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_basis_energy_biorthogonal_design["candidate_ladder"]
    }
    assert (
        residual_basis_energy_design_candidates["AF0_modal_floor_theory_freeze"][
            "accepted_candidate"
        ]
        is False
    )
    assert residual_basis_energy_design_candidates[
        "AF1_energy_biorthogonal_eh_inner_product_schema"
    ]["energy_biorthogonal_schema_ready"]
    assert residual_basis_energy_design_candidates[
        "AF2_characteristic_impedance_normalizer_weighting_contract"
    ]["characteristic_impedance_weighting_contract_ready"]
    assert residual_basis_energy_design_candidates[
        "AF3_combined_energy_biorthogonal_design_contract"
    ]["accepted_candidate"]
    assert (
        residual_basis_energy_design_candidates[
            "AF4_energy_biorthogonal_design_blocked"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_design["public_claim_allowed"] is False
    assert (
        residual_basis_energy_biorthogonal_design["public_observable_promoted"]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_design[
            "true_rt_public_observable_promoted"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_design[
            "dft_flux_tfsf_port_sparameter_promoted"
        ]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_design_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_design["next_prerequisite"]
    )
    residual_basis_energy_biorthogonal_implementation = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_implementation_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_hunk_retained_fixture_quality_pending"
    )
    assert residual_basis_energy_biorthogonal_implementation[
        "terminal_outcome"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_hunk_retained_fixture_quality_pending"
    )
    assert residual_basis_energy_biorthogonal_implementation[
        "upstream_energy_biorthogonal_design_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_design_status"
    ]
    assert residual_basis_energy_biorthogonal_implementation[
        "candidate_ladder_declared_before_implementation"
    ]
    assert residual_basis_energy_biorthogonal_implementation[
        "candidate_ladder_declared_before_solver_edit"
    ]
    assert residual_basis_energy_biorthogonal_implementation[
        "candidate_ladder_declared_before_slow_scoring"
    ]
    assert residual_basis_energy_biorthogonal_implementation["candidate_count"] == 5
    assert residual_basis_energy_biorthogonal_implementation[
        "selected_candidate_id"
    ] == "AG3_combined_energy_biorthogonal_residual_basis_helper_hunk"
    assert residual_basis_energy_biorthogonal_implementation[
        "baseline_metrics"
    ] == residual_basis_energy_biorthogonal_design["baseline_metrics"]
    assert residual_basis_energy_biorthogonal_implementation["metrics"] == (
        residual_basis_energy_biorthogonal_design["metrics"]
    )
    assert residual_basis_energy_biorthogonal_implementation["thresholds"] == (
        residual_basis_energy_biorthogonal_design["thresholds"]
    )
    assert residual_basis_energy_biorthogonal_implementation[
        "threshold_results"
    ] == residual_basis_energy_biorthogonal_design["threshold_results"]
    assert residual_basis_energy_biorthogonal_implementation[
        "energy_biorthogonal_design_ready"
    ]
    assert residual_basis_energy_biorthogonal_implementation[
        "energy_biorthogonal_contract_consumed"
    ]
    assert residual_basis_energy_biorthogonal_implementation[
        "energy_biorthogonal_helper_retained"
    ]
    assert residual_basis_energy_biorthogonal_implementation[
        "energy_biorthogonal_contract"
    ] == residual_basis_energy_biorthogonal_design["energy_biorthogonal_contract"]
    assert (
        residual_basis_energy_biorthogonal_implementation["projection_helper"]
        == "_project_private_modal_basis_packets"
    )
    assert (
        residual_basis_energy_biorthogonal_implementation["consumer_helper"]
        == "_apply_propagation_aware_modal_retry_face_helper"
    )
    assert residual_basis_energy_biorthogonal_implementation["basis_vectors"] == (
        residual_basis_energy_biorthogonal_design["basis_vectors"]
    )
    assert residual_basis_energy_biorthogonal_implementation[
        "characteristic_admittance_weighted"
    ]
    assert residual_basis_energy_biorthogonal_implementation[
        "energy_density_shape_weighted"
    ]
    assert residual_basis_energy_biorthogonal_implementation[
        "energy_biorthogonal_gram_projection"
    ]
    assert residual_basis_energy_biorthogonal_implementation[
        "scalar_l2_orthogonality_replaced"
    ]
    assert residual_basis_energy_biorthogonal_implementation[
        "subtraction_uses_projected_packets_only"
    ]
    assert residual_basis_energy_biorthogonal_implementation[
        "single_incident_basis_replaced"
    ]
    assert residual_basis_energy_biorthogonal_implementation[
        "projection_gate_fail_closed"
    ]
    assert residual_basis_energy_biorthogonal_implementation[
        "fail_closed_if_projection_energy_missing"
    ]
    assert residual_basis_energy_biorthogonal_implementation[
        "reuse_existing_owner_packet_shapes"
    ]
    assert residual_basis_energy_biorthogonal_implementation["fixed_shape_reductions"]
    assert residual_basis_energy_biorthogonal_implementation["jit_safe_reduction"]
    assert residual_basis_energy_biorthogonal_implementation["no_threshold_laundering"]
    assert (
        residual_basis_energy_biorthogonal_implementation[
            "material_improvement_demonstrated"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_implementation[
            "subgrid_vacuum_parity_scored"
        ]
        is False
    )
    assert residual_basis_energy_biorthogonal_implementation[
        "fixture_quality_pending"
    ]
    assert residual_basis_energy_biorthogonal_implementation["production_patch_applied"]
    assert residual_basis_energy_biorthogonal_implementation["solver_behavior_changed"]
    assert residual_basis_energy_biorthogonal_implementation[
        "field_update_behavior_changed"
    ]
    assert residual_basis_energy_biorthogonal_implementation["new_solver_hunk_retained"]
    energy_impl_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_basis_energy_biorthogonal_implementation[
            "candidate_ladder"
        ]
    }
    assert energy_impl_candidates[
        "AG1_characteristic_eh_energy_weight_helper"
    ]["characteristic_admittance_weighted"]
    assert energy_impl_candidates[
        "AG2_energy_biorthogonal_packet_projection"
    ]["energy_biorthogonal_gram_projection"]
    assert energy_impl_candidates[
        "AG3_combined_energy_biorthogonal_residual_basis_helper_hunk"
    ]["accepted_candidate"]
    assert (
        energy_impl_candidates["AG4_energy_biorthogonal_implementation_blocked"][
            "accepted_candidate"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_implementation["public_claim_allowed"]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_implementation[
            "public_observable_promoted"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_implementation[
            "true_rt_public_observable_promoted"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_implementation[
            "dft_flux_tfsf_port_sparameter_promoted"
        ]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_implementation_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_implementation["next_prerequisite"]
    )
    residual_basis_energy_biorthogonal_parity_scoring = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_parity_scoring"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_parity_scoring_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_hunk_insufficient_fixture_quality_pending"
    )
    assert residual_basis_energy_biorthogonal_parity_scoring[
        "terminal_outcome"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_hunk_insufficient_fixture_quality_pending"
    )
    assert residual_basis_energy_biorthogonal_parity_scoring[
        "upstream_energy_biorthogonal_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_implementation_status"
    ]
    assert residual_basis_energy_biorthogonal_parity_scoring[
        "candidate_ladder_declared_before_implementation"
    ]
    assert residual_basis_energy_biorthogonal_parity_scoring[
        "candidate_ladder_declared_before_solver_edit"
    ]
    assert residual_basis_energy_biorthogonal_parity_scoring[
        "candidate_ladder_declared_before_slow_scoring"
    ]
    assert residual_basis_energy_biorthogonal_parity_scoring["candidate_count"] == 5
    assert residual_basis_energy_biorthogonal_parity_scoring[
        "selected_candidate_id"
    ] == "AH1_finite_energy_biorthogonal_private_parity_score"
    assert residual_basis_energy_biorthogonal_parity_scoring[
        "baseline_metrics"
    ] == residual_basis_energy_biorthogonal_implementation["baseline_metrics"]
    assert residual_basis_energy_biorthogonal_parity_scoring["metrics"] == (
        residual_basis_energy_biorthogonal_implementation["metrics"]
    )
    assert residual_basis_energy_biorthogonal_parity_scoring["thresholds"] == (
        residual_basis_energy_biorthogonal_implementation["thresholds"]
    )
    assert residual_basis_energy_biorthogonal_parity_scoring[
        "threshold_results"
    ] == residual_basis_energy_biorthogonal_implementation["threshold_results"]
    assert residual_basis_energy_biorthogonal_parity_scoring[
        "energy_biorthogonal_hunk_retained"
    ]
    assert residual_basis_energy_biorthogonal_parity_scoring[
        "energy_biorthogonal_helper_retained"
    ]
    assert (
        residual_basis_energy_biorthogonal_parity_scoring["projection_helper"]
        == "_project_private_modal_basis_packets"
    )
    assert (
        residual_basis_energy_biorthogonal_parity_scoring["consumer_helper"]
        == "_apply_propagation_aware_modal_retry_face_helper"
    )
    assert residual_basis_energy_biorthogonal_parity_scoring["basis_vectors"] == (
        residual_basis_energy_biorthogonal_implementation["basis_vectors"]
    )
    assert residual_basis_energy_biorthogonal_parity_scoring[
        "characteristic_admittance_weighted"
    ]
    assert residual_basis_energy_biorthogonal_parity_scoring[
        "energy_biorthogonal_gram_projection"
    ]
    assert residual_basis_energy_biorthogonal_parity_scoring[
        "scalar_l2_orthogonality_replaced"
    ]
    assert residual_basis_energy_biorthogonal_parity_scoring[
        "parity_scoring_lane_executed"
    ]
    assert residual_basis_energy_biorthogonal_parity_scoring[
        "finite_reproducible_score"
    ]
    assert residual_basis_energy_biorthogonal_parity_scoring[
        "score_uses_retained_implementation_metrics"
    ]
    assert (
        residual_basis_energy_biorthogonal_parity_scoring[
            "material_improvement_demonstrated"
        ]
        is False
    )
    assert residual_basis_energy_biorthogonal_parity_scoring["paired_passed"] is False
    assert (
        residual_basis_energy_biorthogonal_parity_scoring["fixture_quality_ready"]
        is False
    )
    assert residual_basis_energy_biorthogonal_parity_scoring[
        "fixture_quality_pending"
    ]
    assert residual_basis_energy_biorthogonal_parity_scoring[
        "subgrid_vacuum_parity_scored"
    ]
    assert (
        residual_basis_energy_biorthogonal_parity_scoring[
            "subgrid_vacuum_parity_passed"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_parity_scoring[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    assert residual_basis_energy_biorthogonal_parity_scoring[
        "retained_energy_biorthogonal_solver_hunk"
    ]
    assert (
        residual_basis_energy_biorthogonal_parity_scoring["new_solver_hunk_retained"]
        is False
    )
    energy_parity_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_basis_energy_biorthogonal_parity_scoring[
            "candidate_ladder"
        ]
    }
    assert energy_parity_candidates[
        "AH1_finite_energy_biorthogonal_private_parity_score"
    ]["accepted_candidate"]
    assert (
        energy_parity_candidates[
            "AH2_energy_biorthogonal_material_improvement_gate"
        ]["accepted_candidate"]
        is False
    )
    assert (
        energy_parity_candidates[
            "AH3_energy_biorthogonal_fixture_quality_true_rt_readiness"
        ]["accepted_candidate"]
        is False
    )
    assert (
        energy_parity_candidates[
            "AH4_energy_biorthogonal_parity_scoring_insufficient"
        ]["accepted_candidate"]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_parity_scoring["public_claim_allowed"]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_parity_scoring[
            "public_observable_promoted"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_parity_scoring[
            "true_rt_public_observable_promoted"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_parity_scoring[
            "dft_flux_tfsf_port_sparameter_promoted"
        ]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_parity_scoring_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_parity_scoring["next_prerequisite"]
    )
    residual_basis_energy_biorthogonal_failure_theory = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_failure_theory"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_failure_theory_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_floor_theory_ready"
    )
    assert residual_basis_energy_biorthogonal_failure_theory[
        "terminal_outcome"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_floor_theory_ready"
    )
    assert residual_basis_energy_biorthogonal_failure_theory[
        "upstream_energy_biorthogonal_parity_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_parity_scoring_status"
    ]
    assert residual_basis_energy_biorthogonal_failure_theory[
        "upstream_energy_biorthogonal_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_implementation_status"
    ]
    assert residual_basis_energy_biorthogonal_failure_theory[
        "candidate_ladder_declared_before_implementation"
    ]
    assert residual_basis_energy_biorthogonal_failure_theory[
        "candidate_ladder_declared_before_solver_edit"
    ]
    assert residual_basis_energy_biorthogonal_failure_theory[
        "candidate_ladder_declared_before_slow_scoring"
    ]
    assert residual_basis_energy_biorthogonal_failure_theory["candidate_count"] == 5
    assert residual_basis_energy_biorthogonal_failure_theory[
        "selected_candidate_id"
    ] == "AI1_energy_metric_shape_floor"
    assert residual_basis_energy_biorthogonal_failure_theory[
        "baseline_metrics"
    ] == residual_basis_energy_biorthogonal_parity_scoring["baseline_metrics"]
    assert residual_basis_energy_biorthogonal_failure_theory["metrics"] == (
        residual_basis_energy_biorthogonal_parity_scoring["metrics"]
    )
    assert residual_basis_energy_biorthogonal_failure_theory["thresholds"] == (
        residual_basis_energy_biorthogonal_parity_scoring["thresholds"]
    )
    assert residual_basis_energy_biorthogonal_failure_theory[
        "threshold_results"
    ] == residual_basis_energy_biorthogonal_parity_scoring["threshold_results"]
    assert residual_basis_energy_biorthogonal_failure_theory[
        "energy_biorthogonal_parity_scoring_insufficient"
    ]
    assert residual_basis_energy_biorthogonal_failure_theory[
        "energy_biorthogonal_hunk_retained"
    ]
    assert residual_basis_energy_biorthogonal_failure_theory[
        "finite_reproducible_score"
    ]
    metric_shape_floor_contract = residual_basis_energy_biorthogonal_failure_theory[
        "metric_shape_floor_contract"
    ]
    assert (
        metric_shape_floor_contract["projection_helper"]
        == "_project_private_modal_basis_packets"
    )
    assert "packet-local" in metric_shape_floor_contract["remaining_floor"]
    assert "metric-shape calibration" in metric_shape_floor_contract[
        "required_next_contract"
    ]
    assert metric_shape_floor_contract["requires_public_observable"] is False
    assert metric_shape_floor_contract["requires_solver_hunk_in_this_lane"] is False
    assert residual_basis_energy_biorthogonal_failure_theory[
        "energy_metric_shape_floor_detected"
    ]
    assert residual_basis_energy_biorthogonal_failure_theory[
        "characteristic_mode_basis_floor_deferred"
    ]
    assert residual_basis_energy_biorthogonal_failure_theory[
        "time_centered_power_pairing_floor_deferred"
    ]
    assert residual_basis_energy_biorthogonal_failure_theory[
        "failure_theory_lane_executed"
    ]
    assert (
        residual_basis_energy_biorthogonal_failure_theory[
            "material_improvement_demonstrated"
        ]
        is False
    )
    assert residual_basis_energy_biorthogonal_failure_theory[
        "subgrid_vacuum_parity_scored"
    ]
    assert (
        residual_basis_energy_biorthogonal_failure_theory[
            "subgrid_vacuum_parity_passed"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_failure_theory[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_failure_theory["new_solver_hunk_retained"]
        is False
    )
    assert residual_basis_energy_biorthogonal_failure_theory[
        "next_lane_requires_energy_metric_shape_calibration_design"
    ]
    energy_failure_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_basis_energy_biorthogonal_failure_theory[
            "candidate_ladder"
        ]
    }
    assert energy_failure_candidates["AI1_energy_metric_shape_floor"][
        "accepted_candidate"
    ]
    assert (
        energy_failure_candidates["AI2_characteristic_mode_basis_floor"][
            "accepted_candidate"
        ]
        is False
    )
    assert (
        energy_failure_candidates["AI3_time_centered_power_pairing_floor"][
            "accepted_candidate"
        ]
        is False
    )
    assert (
        energy_failure_candidates[
            "AI4_energy_biorthogonal_failure_theory_blocked"
        ]["accepted_candidate"]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_failure_theory["public_claim_allowed"]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_failure_theory[
            "public_observable_promoted"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_failure_theory[
            "true_rt_public_observable_promoted"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_failure_theory[
            "dft_flux_tfsf_port_sparameter_promoted"
        ]
        is False
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_failure_theory_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_failure_theory["next_prerequisite"]
    )
    residual_basis_energy_biorthogonal_metric_shape_calibration_design = (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_design"
        ]
    )
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_design_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_contract_design_ready"
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_design[
        "terminal_outcome"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_contract_design_ready"
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_design[
        "upstream_energy_biorthogonal_failure_theory_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_failure_theory_status"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_design[
        "candidate_ladder_declared_before_implementation"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_design[
        "candidate_ladder_declared_before_solver_edit"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_design[
        "candidate_ladder_declared_before_slow_scoring"
    ]
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_design[
            "candidate_count"
        ]
        == 5
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_design[
        "selected_candidate_id"
    ] == "AJ3_combined_metric_shape_calibration_design_contract"
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_design[
        "baseline_metrics"
    ] == residual_basis_energy_biorthogonal_failure_theory["baseline_metrics"]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_design[
        "metrics"
    ] == residual_basis_energy_biorthogonal_failure_theory["metrics"]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_design[
        "thresholds"
    ] == residual_basis_energy_biorthogonal_failure_theory["thresholds"]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_design[
        "threshold_results"
    ] == residual_basis_energy_biorthogonal_failure_theory["threshold_results"]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_design[
        "failure_theory_contract_consumed"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_design[
        "metric_shape_calibration_design_ready"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_design[
        "face_local_sbp_mortar_metric_schema_ready"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_design[
        "metric_normalization_contract_ready"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_design[
        "bounded_private_implementation_contract_ready"
    ]
    metric_shape_calibration_schema = (
        residual_basis_energy_biorthogonal_metric_shape_calibration_design[
            "metric_shape_calibration_schema"
        ]
    )
    assert (
        metric_shape_calibration_schema["state_owner"]
        == "_PrivateInterfaceOwnerState"
    )
    assert (
        metric_shape_calibration_schema["consumer_helper"]
        == "_apply_propagation_aware_modal_retry_face_helper"
    )
    assert (
        metric_shape_calibration_schema["projection_helper"]
        == "_project_private_modal_basis_packets"
    )
    assert (
        metric_shape_calibration_schema["calibration_surface"]
        == "private_face_local_sbp_mortar_power_metric"
    )
    assert metric_shape_calibration_schema["metric_components"] == [
        "face_proxy_weight",
        "source_owner_weight",
        "face_proxy_mask",
        "source_owner_mask",
        "face_normal_sign",
        "source_normal_sign",
    ]
    assert metric_shape_calibration_schema["basis_vectors"] == (
        residual_basis_energy_biorthogonal_failure_theory["basis_vectors"]
    )
    assert metric_shape_calibration_schema["fixed_shape_reductions"]
    assert metric_shape_calibration_schema["jit_safe_reduction"]
    assert metric_shape_calibration_schema["cpml_non_cpml_wiring_inherited"]
    assert metric_shape_calibration_schema["requires_public_observable"] is False
    metric_shape_calibration_contract = (
        residual_basis_energy_biorthogonal_metric_shape_calibration_design[
            "metric_shape_calibration_contract"
        ]
    )
    assert (
        metric_shape_calibration_contract["upstream_metric_shape_floor_contract"]
        == metric_shape_floor_contract
    )
    assert (
        metric_shape_calibration_contract["metric_shape_calibration_schema"]
        == metric_shape_calibration_schema
    )
    assert metric_shape_calibration_contract[
        "face_local_sbp_mortar_metric_contract_ready"
    ]
    assert metric_shape_calibration_contract["metric_normalization_contract_ready"]
    assert metric_shape_calibration_contract["fail_closed_on_metric_shape_missing"]
    assert metric_shape_calibration_contract["reuse_existing_owner_packet_shapes"]
    assert metric_shape_calibration_contract["fixed_shape_reductions"]
    assert metric_shape_calibration_contract["jit_safe_reduction"]
    assert metric_shape_calibration_contract["cpml_non_cpml_wiring_inherited"]
    assert metric_shape_calibration_contract["no_threshold_laundering"]
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_design[
            "material_improvement_demonstrated"
        ]
        is False
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_design[
        "fixture_quality_pending"
    ]
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_design[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_design[
            "slab_rt_scored"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_design[
            "new_solver_hunk_retained"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_design[
            "public_observable_promoted"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_design[
            "true_rt_public_observable_promoted"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_design[
            "dft_flux_tfsf_port_sparameter_promoted"
        ]
        is False
    )
    energy_metric_shape_design_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in (
            residual_basis_energy_biorthogonal_metric_shape_calibration_design[
                "candidate_ladder"
            ]
        )
    }
    assert energy_metric_shape_design_candidates[
        "AJ1_face_local_sbp_mortar_metric_schema"
    ]["face_local_sbp_mortar_metric_schema_ready"]
    assert energy_metric_shape_design_candidates[
        "AJ2_bounded_metric_normalization_contract"
    ]["metric_normalization_contract_ready"]
    assert energy_metric_shape_design_candidates[
        "AJ3_combined_metric_shape_calibration_design_contract"
    ]["accepted_candidate"]
    assert (
        energy_metric_shape_design_candidates[
            "AJ4_metric_shape_calibration_design_blocked"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_design[
        "next_lane_requires_metric_shape_calibration_implementation"
    ]
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_design_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_metric_shape_calibration_design[
            "next_prerequisite"
        ]
    )
    residual_basis_energy_biorthogonal_metric_shape_calibration_implementation = (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_implementation"
        ]
    )
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_implementation_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_hunk_retained_fixture_quality_pending"
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "terminal_outcome"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_hunk_retained_fixture_quality_pending"
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "upstream_metric_shape_calibration_design_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_design_status"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "candidate_ladder_declared_before_implementation"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "candidate_ladder_declared_before_solver_edit"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "candidate_ladder_declared_before_slow_scoring"
    ]
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
            "candidate_count"
        ]
        == 5
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "selected_candidate_id"
    ] == "AK3_combined_metric_shape_calibration_hunk"
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "baseline_metrics"
    ] == residual_basis_energy_biorthogonal_metric_shape_calibration_design[
        "baseline_metrics"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "metrics"
    ] == residual_basis_energy_biorthogonal_metric_shape_calibration_design["metrics"]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "thresholds"
    ] == residual_basis_energy_biorthogonal_metric_shape_calibration_design[
        "thresholds"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "threshold_results"
    ] == residual_basis_energy_biorthogonal_metric_shape_calibration_design[
        "threshold_results"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "metric_shape_calibration_contract_consumed"
    ]
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
            "metric_shape_calibration_schema"
        ]
        == metric_shape_calibration_schema
    )
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
            "metric_shape_calibration_contract"
        ]
        == metric_shape_calibration_contract
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "face_local_sbp_mortar_metric_applied"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "metric_normalized_projection"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "packet_local_energy_density_shape_replaced"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "normal_sign_contract_fail_closed"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "finite_reproducible_score"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "production_patch_applied"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "solver_behavior_changed"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "field_update_behavior_changed"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "new_solver_hunk_retained"
    ]
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
            "runner_behavior_changed"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
            "public_observable_promoted"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
            "true_rt_public_observable_promoted"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
            "dft_flux_tfsf_port_sparameter_promoted"
        ]
        is False
    )
    energy_metric_shape_impl_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in (
            residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
                "candidate_ladder"
            ]
        )
    }
    assert energy_metric_shape_impl_candidates[
        "AK1_face_local_sbp_mortar_metric_helper"
    ]["face_local_sbp_mortar_metric_applied"]
    assert energy_metric_shape_impl_candidates[
        "AK2_metric_normalized_energy_gram_projection"
    ]["metric_normalized_projection"]
    assert energy_metric_shape_impl_candidates[
        "AK3_combined_metric_shape_calibration_hunk"
    ]["accepted_candidate"]
    assert (
        energy_metric_shape_impl_candidates[
            "AK4_metric_shape_calibration_implementation_blocked"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "next_lane_requires_metric_shape_calibration_parity_scoring"
    ]
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_implementation_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
            "next_prerequisite"
        ]
    )
    residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring = (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring"
        ]
    )
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_hunk_insufficient_fixture_quality_pending"
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
        "terminal_outcome"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_hunk_insufficient_fixture_quality_pending"
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
        "upstream_metric_shape_calibration_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_implementation_status"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
        "candidate_ladder_declared_before_implementation"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
        "candidate_ladder_declared_before_solver_edit"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
        "candidate_ladder_declared_before_slow_scoring"
    ]
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
            "candidate_count"
        ]
        == 5
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
        "selected_candidate_id"
    ] == "AL1_finite_metric_shape_calibration_private_parity_score"
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
        "baseline_metrics"
    ] == residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "baseline_metrics"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
        "metrics"
    ] == residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "metrics"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
        "thresholds"
    ] == residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "thresholds"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
        "threshold_results"
    ] == residual_basis_energy_biorthogonal_metric_shape_calibration_implementation[
        "threshold_results"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
        "metric_shape_calibration_hunk_retained"
    ]
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
            "metric_shape_calibration_contract"
        ]
        == metric_shape_calibration_contract
    )
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
            "metric_shape_calibration_schema"
        ]
        == metric_shape_calibration_schema
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
        "finite_reproducible_score"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
        "score_uses_retained_implementation_metrics"
    ]
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
            "material_improvement_demonstrated"
        ]
        is False
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
        "subgrid_vacuum_parity_scored"
    ]
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
            "subgrid_vacuum_parity_passed"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
            "new_solver_hunk_retained"
        ]
        is False
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
        "retained_metric_shape_calibration_solver_hunk"
    ]
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
            "public_observable_promoted"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
            "true_rt_public_observable_promoted"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
            "dft_flux_tfsf_port_sparameter_promoted"
        ]
        is False
    )
    energy_metric_shape_parity_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in (
            residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
                "candidate_ladder"
            ]
        )
    }
    assert energy_metric_shape_parity_candidates[
        "AL1_finite_metric_shape_calibration_private_parity_score"
    ]["accepted_candidate"]
    assert (
        energy_metric_shape_parity_candidates[
            "AL2_metric_shape_calibration_material_improvement_gate"
        ]["accepted_candidate"]
        is False
    )
    assert (
        energy_metric_shape_parity_candidates[
            "AL3_metric_shape_calibration_fixture_quality_true_rt_readiness"
        ]["accepted_candidate"]
        is False
    )
    assert (
        energy_metric_shape_parity_candidates[
            "AL4_metric_shape_calibration_parity_scoring_insufficient"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
        "next_lane_requires_metric_shape_calibration_failure_theory"
    ]
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
            "next_prerequisite"
        ]
    )
    residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory = (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory"
        ]
    )
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_transverse_modal_coupling_floor_theory_ready"
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "terminal_outcome"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory_status"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "upstream_metric_shape_calibration_parity_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring_status"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "upstream_metric_shape_calibration_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_implementation_status"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "candidate_ladder_declared_before_implementation"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "candidate_ladder_declared_before_solver_edit"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "candidate_ladder_declared_before_slow_scoring"
    ]
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
            "candidate_count"
        ]
        == 5
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "selected_candidate_id"
    ] == "AM2_transverse_modal_basis_coupling_floor"
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "baseline_metrics"
    ] == residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
        "baseline_metrics"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "metrics"
    ] == residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
        "metrics"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "thresholds"
    ] == residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
        "thresholds"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "threshold_results"
    ] == residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring[
        "threshold_results"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "metric_shape_calibration_parity_scoring_insufficient"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "metric_shape_calibration_hunk_retained"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "finite_reproducible_score"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "score_delta_all_zero"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "face_metric_sign_orientation_contract_present"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "face_metric_orientation_floor_rejected"
    ]
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "transverse_modal_coupling_floor_detected"
    ]
    transverse_modal_contract = (
        residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
            "transverse_modal_coupling_contract"
        ]
    )
    assert "basis-to-basis tangential coupling" in transverse_modal_contract[
        "remaining_floor"
    ]
    assert transverse_modal_contract["requires_public_observable"] is False
    assert transverse_modal_contract["requires_solver_hunk_in_this_lane"] is False
    assert transverse_modal_contract["requires_threshold_change"] is False
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
            "material_improvement_demonstrated"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
            "subgrid_vacuum_parity_passed"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
            "new_solver_hunk_retained"
        ]
        is False
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "retained_metric_shape_calibration_solver_hunk"
    ]
    assert (
        residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
            "public_observable_promoted"
        ]
        is False
    )
    energy_metric_shape_failure_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in (
            residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
                "candidate_ladder"
            ]
        )
    }
    assert (
        energy_metric_shape_failure_candidates[
            "AM1_face_metric_sign_orientation_floor"
        ]["accepted_candidate"]
        is False
    )
    assert energy_metric_shape_failure_candidates[
        "AM2_transverse_modal_basis_coupling_floor"
    ]["accepted_candidate"]
    assert (
        energy_metric_shape_failure_candidates[
            "AM3_time_centered_power_pairing_floor"
        ]["accepted_candidate"]
        is False
    )
    assert (
        energy_metric_shape_failure_candidates[
            "AM4_metric_shape_calibration_failure_theory_blocked"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "next_lane_requires_transverse_modal_coupling_metric_design"
    ]
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
            "next_prerequisite"
        ]
    )
    residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design = (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design"
        ]
    )
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_contract_design_ready"
    )
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
        "terminal_outcome"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design_status"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
        "upstream_metric_shape_calibration_failure_theory_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory_status"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
        "candidate_ladder_declared_before_implementation"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
        "candidate_ladder_declared_before_solver_edit"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
        "candidate_ladder_declared_before_slow_scoring"
    ]
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
            "candidate_count"
        ]
        == 5
    )
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
        "selected_candidate_id"
    ] == "AN2_transverse_3x3_modal_coupling_metric_contract"
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
        "baseline_metrics"
    ] == residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "baseline_metrics"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
        "metrics"
    ] == residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "metrics"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
        "thresholds"
    ] == residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "thresholds"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
        "threshold_results"
    ] == residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory[
        "threshold_results"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
        "metric_shape_calibration_failure_theory_consumed"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
        "transverse_modal_coupling_metric_design_ready"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
        "scalar_face_metric_extension_rejected"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
        "time_centered_cross_coupling_deferred"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
        "fixed_shape_jit_safe_contract"
    ]
    transverse_modal_schema = (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
            "transverse_modal_coupling_schema"
        ]
    )
    assert transverse_modal_schema["matrix_shape"] == [3, 3]
    assert transverse_modal_schema["requires_public_observable"] is False
    assert transverse_modal_schema["requires_threshold_change"] is False
    assert transverse_modal_schema["requires_new_solver_state_in_design_lane"] is False
    transverse_modal_design_contract = (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
            "transverse_modal_coupling_contract"
        ]
    )
    assert "fixed-shape private 3x3 modal" in transverse_modal_design_contract[
        "target_basis_behavior"
    ]
    assert transverse_modal_design_contract["no_threshold_laundering"] is True
    assert transverse_modal_design_contract["fail_closed_on_shape_mismatch"] is True
    assert transverse_modal_design_contract["fail_closed_on_nonfinite_coupling"] is True
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
            "material_improvement_demonstrated"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
            "subgrid_vacuum_parity_passed"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
            "new_solver_hunk_retained"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
            "public_observable_promoted"
        ]
        is False
    )
    transverse_modal_design_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in (
            residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
                "candidate_ladder"
            ]
        )
    }
    assert (
        transverse_modal_design_candidates[
            "AN1_scalar_face_metric_extension_rejected"
        ]["accepted_candidate"]
        is False
    )
    assert transverse_modal_design_candidates[
        "AN2_transverse_3x3_modal_coupling_metric_contract"
    ]["accepted_candidate"]
    assert (
        transverse_modal_design_candidates[
            "AN3_time_centered_cross_coupling_contract_deferred"
        ]["accepted_candidate"]
        is False
    )
    assert (
        transverse_modal_design_candidates[
            "AN4_transverse_modal_coupling_design_blocked"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
        "next_lane_requires_transverse_modal_coupling_metric_implementation"
    ]
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
            "next_prerequisite"
        ]
    )
    residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation = (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation"
        ]
    )
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_hunk_retained_fixture_quality_pending"
    )
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "terminal_outcome"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation_status"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "upstream_transverse_modal_coupling_design_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design_status"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "candidate_ladder_declared_before_implementation"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "candidate_ladder_declared_before_solver_edit"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "candidate_ladder_declared_before_slow_scoring"
    ]
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
            "candidate_count"
        ]
        == 5
    )
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "selected_candidate_id"
    ] == "AO1_private_3x3_modal_coupling_matrix_helper"
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "baseline_metrics"
    ] == residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
        "baseline_metrics"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "metrics"
    ] == residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
        "metrics"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "thresholds"
    ] == residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
        "thresholds"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "threshold_results"
    ] == residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design[
        "threshold_results"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "transverse_modal_coupling_design_consumed"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "transverse_modal_coupling_metric_hunk_retained"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "coupling_helper"
    ] == "_private_transverse_modal_coupling_metric"
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "matrix_shape"
    ] == [3, 3]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "fixed_shape_jit_safe_contract"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "modal_coupling_bound"
    ] == 0.35
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "fail_closed_on_nonfinite_coupling"
    ]
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
            "material_improvement_demonstrated"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
            "subgrid_vacuum_parity_scored"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "new_solver_hunk_retained"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "solver_behavior_changed"
    ]
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
            "public_observable_promoted"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
            "true_rt_public_observable_promoted"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
            "dft_flux_tfsf_port_sparameter_promoted"
        ]
        is False
    )
    transverse_modal_impl_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in (
            residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
                "candidate_ladder"
            ]
        )
    }
    assert transverse_modal_impl_candidates[
        "AO1_private_3x3_modal_coupling_matrix_helper"
    ]["accepted_candidate"]
    assert (
        transverse_modal_impl_candidates[
            "AO2_private_modal_coupling_normalization_guard"
        ]["accepted_candidate"]
        is False
    )
    assert (
        transverse_modal_impl_candidates[
            "AO3_implementation_blocked_by_state_shape"
        ]["accepted_candidate"]
        is False
    )
    assert (
        transverse_modal_impl_candidates[
            "AO4_implementation_blocked_by_public_surface_or_threshold"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "next_lane_requires_transverse_modal_coupling_metric_parity_scoring"
    ]
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
            "next_prerequisite"
        ]
    )
    residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring = (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring"
        ]
    )
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_hunk_insufficient_fixture_quality_pending"
    )
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
        "terminal_outcome"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring_status"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
        "upstream_transverse_modal_coupling_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation_status"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
        "candidate_ladder_declared_before_slow_scoring"
    ]
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
            "candidate_count"
        ]
        == 5
    )
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
        "selected_candidate_id"
    ] == "AP1_finite_transverse_modal_coupling_private_parity_score"
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
        "baseline_metrics"
    ] == residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "baseline_metrics"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
        "metrics"
    ] == residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "metrics"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
        "thresholds"
    ] == residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "thresholds"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
        "threshold_results"
    ] == residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation[
        "threshold_results"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
        "transverse_modal_coupling_metric_hunk_retained"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
        "coupling_helper"
    ] == "_private_transverse_modal_coupling_metric"
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
        "matrix_shape"
    ] == [3, 3]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
        "parity_scoring_lane_executed"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
        "finite_reproducible_score"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
        "score_uses_retained_implementation_metrics"
    ]
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
            "material_improvement_demonstrated"
        ]
        is False
    )
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
        "subgrid_vacuum_parity_scored"
    ]
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
            "subgrid_vacuum_parity_passed"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
            "solver_behavior_changed"
        ]
        is False
    )
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
        "retained_transverse_modal_coupling_solver_hunk"
    ]
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
            "public_observable_promoted"
        ]
        is False
    )
    transverse_modal_parity_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in (
            residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
                "candidate_ladder"
            ]
        )
    }
    assert transverse_modal_parity_candidates[
        "AP1_finite_transverse_modal_coupling_private_parity_score"
    ]["accepted_candidate"]
    assert (
        transverse_modal_parity_candidates[
            "AP2_private_transverse_modal_coupling_normalization_only_fix"
        ]["accepted_candidate"]
        is False
    )
    assert (
        transverse_modal_parity_candidates[
            "AP3_transverse_modal_coupling_parity_blocked_by_state_shape"
        ]["accepted_candidate"]
        is False
    )
    assert (
        transverse_modal_parity_candidates[
            "AP4_transverse_modal_coupling_parity_scoring_insufficient"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
        "next_lane_requires_transverse_modal_coupling_metric_failure_theory"
    ]
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
            "next_prerequisite"
        ]
    )
    residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory = (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory"
        ]
    )
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_source_interface_transfer_floor_theory_ready"
    )
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
        "terminal_outcome"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory_status"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
        "upstream_transverse_modal_coupling_parity_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring_status"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
        "candidate_ladder_declared_before_implementation"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
        "candidate_ladder_declared_before_solver_edit"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
        "candidate_ladder_declared_before_slow_scoring"
    ]
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
            "candidate_count"
        ]
        == 5
    )
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
        "selected_candidate_id"
    ] == "AQ1_source_interface_transverse_modal_transfer_floor"
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
        "metrics"
    ] == residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring[
        "metrics"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
        "transverse_modal_coupling_parity_scoring_consumed"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
        "finite_parity_score_consumed"
    ]
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
        "selected_next_private_design_target"
    ] == "source_interface_transverse_modal_transfer_map"
    transfer_contract = (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
            "source_interface_transfer_map_contract"
        ]
    )
    assert transfer_contract["shape_contract"] == (
        "private fixed-shape 3x3 source/interface transfer map"
    )
    assert transfer_contract["requires_public_observable"] is False
    assert transfer_contract["requires_threshold_change"] is False
    assert transfer_contract["requires_runner_state"] is False
    assert transfer_contract["implementation_deferred"] is True
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
            "material_improvement_demonstrated"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
            "subgrid_vacuum_parity_passed"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
            "solver_behavior_changed"
        ]
        is False
    )
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
        "implementation_deferred"
    ]
    assert (
        residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
            "public_observable_promoted"
        ]
        is False
    )
    transverse_modal_failure_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in (
            residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
                "candidate_ladder"
            ]
        )
    }
    assert transverse_modal_failure_candidates[
        "AQ1_source_interface_transverse_modal_transfer_floor"
    ]["accepted_candidate"]
    assert (
        transverse_modal_failure_candidates[
            "AQ2_metric_sign_phase_orientation_floor"
        ]["accepted_candidate"]
        is False
    )
    assert (
        transverse_modal_failure_candidates[
            "AQ3_packet_timing_ownership_mismatch_persists"
        ]["accepted_candidate"]
        is False
    )
    assert (
        transverse_modal_failure_candidates[
            "AQ4_failure_theory_blocked_by_public_surface_or_threshold"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
        "next_lane_requires_source_interface_transfer_map_implementation"
    ]
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
            "next_prerequisite"
        ]
    )
    residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation = (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_implementation"
        ]
    )
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_implementation_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_hunk_retained_fixture_quality_pending"
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
        "terminal_outcome"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_implementation_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
        "upstream_failure_theory_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
        "selected_candidate_id"
    ] == "AR1_private_source_interface_transfer_map_helper"
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
            "candidate_count"
        ]
        == 5
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
        "metrics"
    ] == residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory[
        "metrics"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
        "failure_theory_consumed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
        "source_interface_transfer_map_contract"
    ] == transfer_contract
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
        "transfer_map_helper"
    ] == "_private_source_interface_transverse_modal_transfer_map"
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
        "matrix_shape"
    ] == [3, 3]
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
            "modal_transfer_bound"
        ]
        == 0.35
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
        "source_interface_transfer_map_hunk_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
        "bounded_transfer_map_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
        "production_patch_applied"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
        "solver_behavior_changed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
        "field_update_behavior_changed"
    ]
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
            "subgrid_vacuum_parity_scored"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
        "fixture_quality_pending"
    ]
    transfer_map_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
            "candidate_ladder"
        ]
    }
    assert transfer_map_candidates[
        "AR1_private_source_interface_transfer_map_helper"
    ]["accepted_candidate"]
    assert (
        transfer_map_candidates[
            "AR4_implementation_blocked_by_public_surface_or_threshold"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
        "next_lane_requires_source_interface_transfer_map_parity_scoring"
    ]
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_implementation_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
            "next_prerequisite"
        ]
    )
    residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring = (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_parity_scoring"
        ]
    )
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_parity_scoring_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_hunk_insufficient_fixture_quality_pending"
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
        "terminal_outcome"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_parity_scoring_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
        "upstream_source_interface_transfer_map_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_implementation_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
        "selected_candidate_id"
    ] == "AS1_finite_source_interface_transfer_map_private_parity_score"
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
            "candidate_count"
        ]
        == 5
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
        "metrics"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
        "metrics"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
        "thresholds"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_implementation[
        "thresholds"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
        "source_interface_transfer_map_implementation_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
        "source_interface_transfer_map_hunk_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
        "bounded_transfer_map_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
        "parity_scoring_lane_executed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
        "finite_reproducible_score"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
        "score_uses_retained_implementation_metrics"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
        "matrix_shape"
    ] == [3, 3]
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
            "modal_transfer_bound"
        ]
        == 0.35
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
            "material_improvement_demonstrated"
        ]
        is False
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
        "subgrid_vacuum_parity_scored"
    ]
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
            "subgrid_vacuum_parity_passed"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
            "solver_behavior_changed"
        ]
        is False
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
        "retained_source_interface_transfer_map_solver_hunk"
    ]
    transfer_map_parity_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
            "candidate_ladder"
        ]
    }
    assert transfer_map_parity_candidates[
        "AS1_finite_source_interface_transfer_map_private_parity_score"
    ]["accepted_candidate"]
    assert (
        transfer_map_parity_candidates[
            "AS4_source_interface_transfer_map_parity_scoring_insufficient"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
        "next_lane_requires_source_interface_transfer_map_failure_theory"
    ]
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_parity_scoring_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
            "next_prerequisite"
        ]
    )
    residual_basis_energy_biorthogonal_source_interface_transfer_map_failure_theory = (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_failure_theory"
        ]
    )
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_failure_theory_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_floor_theory_ready"
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_failure_theory[
        "terminal_outcome"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_failure_theory_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_failure_theory[
        "upstream_source_interface_transfer_map_parity_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_parity_scoring_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_failure_theory[
        "selected_candidate_id"
    ] == "AT1_source_interface_transfer_target_basis_orientation_floor"
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_failure_theory[
            "candidate_count"
        ]
        == 5
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_failure_theory[
        "metrics"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
        "metrics"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_failure_theory[
        "thresholds"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_parity_scoring[
        "thresholds"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_failure_theory[
        "source_interface_transfer_map_parity_scoring_consumed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_failure_theory[
        "finite_parity_score_consumed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_failure_theory[
        "selected_next_private_design_target"
    ] == "source_interface_transfer_target_basis_orientation"
    target_basis_contract = (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_failure_theory[
            "target_basis_orientation_contract"
        ]
    )
    assert target_basis_contract["shape_contract"] == (
        "private fixed-shape 3x3 target-basis-oriented transfer map"
    )
    assert target_basis_contract["requires_public_observable"] is False
    assert target_basis_contract["requires_threshold_change"] is False
    assert target_basis_contract["requires_runner_state"] is False
    assert target_basis_contract["implementation_deferred"] is True
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_failure_theory[
        "implementation_deferred"
    ]
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_failure_theory[
            "solver_behavior_changed"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_failure_theory[
            "production_patch_applied"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_failure_theory[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    transfer_map_failure_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_basis_energy_biorthogonal_source_interface_transfer_map_failure_theory[
            "candidate_ladder"
        ]
    }
    assert transfer_map_failure_candidates[
        "AT1_source_interface_transfer_target_basis_orientation_floor"
    ]["accepted_candidate"]
    assert (
        transfer_map_failure_candidates[
            "AT4_failure_theory_blocked_by_public_surface_or_threshold"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_failure_theory[
        "next_lane_requires_target_basis_orientation_implementation"
    ]
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_failure_theory_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_source_interface_transfer_map_failure_theory[
            "next_prerequisite"
        ]
    )
    residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation = (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_implementation"
        ]
    )
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_implementation_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_hunk_retained_fixture_quality_pending"
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
        "terminal_outcome"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_implementation_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
        "upstream_failure_theory_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_failure_theory_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
        "selected_candidate_id"
    ] == "AU1_private_target_basis_oriented_transfer_map_helper"
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
            "candidate_count"
        ]
        == 5
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
        "metrics"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_failure_theory[
        "metrics"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
        "thresholds"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_failure_theory[
        "thresholds"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
        "failure_theory_consumed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
        "target_basis_orientation_contract_consumed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
        "target_basis_orientation_hunk_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
        "target_basis_oriented_transfer_map_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
        "transfer_map_helper"
    ] == "_private_target_basis_oriented_source_interface_transverse_modal_transfer_map"
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
        "legacy_transfer_map_helper_retained"
    ] == "_private_source_interface_transverse_modal_transfer_map"
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
        "matrix_shape"
    ] == [3, 3]
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
            "modal_transfer_bound"
        ]
        == 0.35
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
        "target_basis_overlap_gate_fail_closed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
        "production_patch_applied"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
        "solver_behavior_changed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
        "field_update_behavior_changed"
    ]
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    target_basis_orientation_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
            "candidate_ladder"
        ]
    }
    assert target_basis_orientation_candidates[
        "AU1_private_target_basis_oriented_transfer_map_helper"
    ]["accepted_candidate"]
    assert (
        target_basis_orientation_candidates[
            "AU4_implementation_blocked_by_public_surface_or_threshold"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
        "next_lane_requires_target_basis_orientation_parity_scoring"
    ]
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_implementation_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
            "next_prerequisite"
        ]
    )
    residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring = (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_parity_scoring"
        ]
    )
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_parity_scoring_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_hunk_insufficient_fixture_quality_pending"
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
        "terminal_outcome"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_parity_scoring_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
        "upstream_target_basis_orientation_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_implementation_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
        "selected_candidate_id"
    ] == "AV1_finite_target_basis_orientation_private_parity_score"
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
            "candidate_count"
        ]
        == 5
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
        "metrics"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
        "metrics"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
        "thresholds"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_implementation[
        "thresholds"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
        "target_basis_orientation_implementation_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
        "target_basis_orientation_hunk_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
        "parity_scoring_lane_executed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
        "finite_reproducible_score"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
        "score_uses_retained_implementation_metrics"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
        "matrix_shape"
    ] == [3, 3]
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
            "modal_transfer_bound"
        ]
        == 0.35
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
            "material_improvement_demonstrated"
        ]
        is False
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
        "subgrid_vacuum_parity_scored"
    ]
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
            "subgrid_vacuum_parity_passed"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
            "solver_behavior_changed"
        ]
        is False
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
        "retained_target_basis_orientation_solver_hunk"
    ]
    target_basis_orientation_parity_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
            "candidate_ladder"
        ]
    }
    assert target_basis_orientation_parity_candidates[
        "AV1_finite_target_basis_orientation_private_parity_score"
    ]["accepted_candidate"]
    assert (
        target_basis_orientation_parity_candidates[
            "AV4_target_basis_orientation_parity_scoring_insufficient"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
        "next_lane_requires_target_basis_orientation_failure_theory"
    ]
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_parity_scoring_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
            "next_prerequisite"
        ]
    )
    residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory = (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_failure_theory"
        ]
    )
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_failure_theory_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_floor_theory_ready"
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
        "terminal_outcome"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_failure_theory_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
        "upstream_target_basis_orientation_parity_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_parity_scoring_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
        "upstream_target_basis_orientation_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_implementation_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
        "selected_candidate_id"
    ] == "AW1_target_basis_residual_phase_sign_floor"
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
            "candidate_count"
        ]
        == 5
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
        "metrics"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
        "metrics"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
        "thresholds"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_parity_scoring[
        "thresholds"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
        "target_basis_orientation_parity_scoring_consumed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
        "finite_parity_score_consumed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
        "selected_next_private_design_target"
    ] == "target_basis_orientation_residual_phase_sign"
    residual_phase_sign_contract = (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
            "residual_phase_sign_contract"
        ]
    )
    assert residual_phase_sign_contract["shape_contract"] == (
        "private fixed-shape 3x3 target-basis residual phase/sign correction map"
    )
    assert residual_phase_sign_contract["consumes_target_basis_orientation_contract"]
    assert residual_phase_sign_contract["requires_public_observable"] is False
    assert residual_phase_sign_contract["requires_threshold_change"] is False
    assert residual_phase_sign_contract["requires_runner_state"] is False
    assert residual_phase_sign_contract["implementation_deferred"]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
        "implementation_deferred"
    ]
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
            "solver_behavior_changed"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
            "production_patch_applied"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
            "public_observable_promoted"
        ]
        is False
    )
    target_basis_orientation_failure_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
            "candidate_ladder"
        ]
    }
    assert target_basis_orientation_failure_candidates[
        "AW1_target_basis_residual_phase_sign_floor"
    ]["accepted_candidate"]
    assert (
        target_basis_orientation_failure_candidates[
            "AW4_failure_theory_blocked_by_public_surface_or_threshold"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
        "next_lane_requires_target_basis_orientation_residual_phase_sign_implementation"
    ]
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_failure_theory_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
            "next_prerequisite"
        ]
    )
    residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation = (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_implementation"
        ]
    )
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_implementation_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_hunk_retained_fixture_quality_pending"
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
        "terminal_outcome"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_implementation_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
        "upstream_failure_theory_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_failure_theory_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
        "selected_candidate_id"
    ] == "AX1_private_residual_phase_sign_transfer_map_helper"
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
            "candidate_count"
        ]
        == 5
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
        "metrics"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
        "metrics"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
        "thresholds"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_failure_theory[
        "thresholds"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
        "failure_theory_consumed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
        "residual_phase_sign_contract_consumed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
        "residual_phase_sign_hunk_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
        "residual_phase_sign_transfer_map_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
        "transfer_map_helper"
    ] == "_private_target_basis_residual_phase_sign_source_interface_transverse_modal_transfer_map"
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
        "target_basis_orientation_helper_retained"
    ] == "_private_target_basis_oriented_source_interface_transverse_modal_transfer_map"
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
        "matrix_shape"
    ] == [3, 3]
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
            "modal_transfer_bound"
        ]
        == 0.35
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
        "residual_phase_sign_gate_fail_closed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
        "production_patch_applied"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
        "solver_behavior_changed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
        "field_update_behavior_changed"
    ]
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    residual_phase_sign_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
            "candidate_ladder"
        ]
    }
    assert residual_phase_sign_candidates[
        "AX1_private_residual_phase_sign_transfer_map_helper"
    ]["accepted_candidate"]
    assert (
        residual_phase_sign_candidates[
            "AX4_implementation_blocked_by_public_surface_or_threshold"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
        "next_lane_requires_target_basis_orientation_residual_phase_sign_parity_scoring"
    ]
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_implementation_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
            "next_prerequisite"
        ]
    )
    residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring = (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring"
        ]
    )
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_hunk_insufficient_fixture_quality_pending"
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
        "terminal_outcome"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
        "upstream_residual_phase_sign_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_implementation_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
        "selected_candidate_id"
    ] == "AY1_finite_residual_phase_sign_private_parity_score"
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
            "candidate_count"
        ]
        == 5
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
        "metrics"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
        "metrics"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
        "thresholds"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_implementation[
        "thresholds"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
        "residual_phase_sign_implementation_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
        "residual_phase_sign_hunk_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
        "residual_phase_sign_transfer_map_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
        "parity_scoring_lane_executed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
        "finite_reproducible_score"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
        "score_uses_retained_implementation_metrics"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
        "matrix_shape"
    ] == [3, 3]
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
            "modal_transfer_bound"
        ]
        == 0.35
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
            "material_improvement_demonstrated"
        ]
        is False
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
        "subgrid_vacuum_parity_scored"
    ]
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
            "subgrid_vacuum_parity_passed"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
            "solver_behavior_changed"
        ]
        is False
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
        "retained_residual_phase_sign_solver_hunk"
    ]
    residual_phase_sign_parity_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
            "candidate_ladder"
        ]
    }
    assert residual_phase_sign_parity_candidates[
        "AY1_finite_residual_phase_sign_private_parity_score"
    ]["accepted_candidate"]
    assert (
        residual_phase_sign_parity_candidates[
            "AY4_residual_phase_sign_parity_scoring_insufficient"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
        "next_lane_requires_target_basis_orientation_residual_phase_sign_failure_theory"
    ]
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
            "next_prerequisite"
        ]
    )
    residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory = (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory"
        ]
    )
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_phase_magnitude_imbalance_floor_theory_ready"
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
        "terminal_outcome"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
        "upstream_residual_phase_sign_parity_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
        "upstream_residual_phase_sign_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_implementation_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
        "selected_candidate_id"
    ] == "AZ1_residual_phase_magnitude_imbalance_floor"
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
            "candidate_count"
        ]
        == 5
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
        "metrics"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
        "metrics"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
        "thresholds"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring[
        "thresholds"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
        "residual_phase_sign_parity_scoring_consumed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
        "finite_parity_score_consumed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
        "residual_phase_sign_hunk_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
        "selected_floor"
    ] == "residual_phase_magnitude_imbalance_floor"
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
        "selected_next_private_design_target"
    ] == "target_basis_orientation_residual_phase_magnitude_balance"
    phase_magnitude_balance_contract = (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
            "phase_magnitude_balance_contract"
        ]
    )
    assert phase_magnitude_balance_contract["shape_contract"] == (
        "private fixed-shape 3x3 residual phase/magnitude balance map"
    )
    assert phase_magnitude_balance_contract[
        "consumes_residual_phase_sign_contract"
    ]
    assert phase_magnitude_balance_contract["requires_public_observable"] is False
    assert phase_magnitude_balance_contract["requires_threshold_change"] is False
    assert phase_magnitude_balance_contract["requires_runner_state"] is False
    assert phase_magnitude_balance_contract["implementation_deferred"]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
        "failure_theory_lane_executed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
        "implementation_deferred"
    ]
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
            "solver_behavior_changed"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
            "production_patch_applied"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    residual_phase_sign_failure_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
            "candidate_ladder"
        ]
    }
    assert residual_phase_sign_failure_candidates[
        "AZ1_residual_phase_magnitude_imbalance_floor"
    ]["accepted_candidate"]
    assert (
        residual_phase_sign_failure_candidates[
            "AZ4_failure_theory_blocked_by_public_surface_or_threshold"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
        "next_lane_requires_target_basis_orientation_residual_phase_magnitude_balance_implementation"
    ]
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
            "next_prerequisite"
        ]
    )
    residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation = (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation"
        ]
    )
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_hunk_retained_fixture_quality_pending"
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
        "terminal_outcome"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
        "upstream_failure_theory_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
        "selected_candidate_id"
    ] == "BA1_private_phase_magnitude_balance_transfer_map_helper"
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
            "candidate_count"
        ]
        == 5
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
        "metrics"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
        "metrics"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
        "thresholds"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory[
        "thresholds"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
        "failure_theory_consumed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
        "phase_magnitude_balance_contract_consumed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
        "phase_magnitude_balance_hunk_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
        "phase_magnitude_balance_transfer_map_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
        "transfer_map_helper"
    ] == "_private_target_basis_residual_phase_magnitude_balance_source_interface_transverse_modal_transfer_map"
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
        "residual_phase_sign_helper_retained"
    ] == "_private_target_basis_residual_phase_sign_source_interface_transverse_modal_transfer_map"
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
        "matrix_shape"
    ] == [3, 3]
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
            "residual_balance_bound"
        ]
        == 0.35
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
        "phase_magnitude_balance_gate_fail_closed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
        "production_patch_applied"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
        "solver_behavior_changed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
        "field_update_behavior_changed"
    ]
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
            "runner_behavior_changed"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
            "public_observable_promoted"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
            "dft_flux_tfsf_port_sparameter_promoted"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    phase_magnitude_balance_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
            "candidate_ladder"
        ]
    }
    assert phase_magnitude_balance_candidates[
        "BA1_private_phase_magnitude_balance_transfer_map_helper"
    ]["accepted_candidate"]
    assert (
        phase_magnitude_balance_candidates[
            "BA4_implementation_blocked_by_public_surface_or_threshold"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
        "next_lane_requires_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring"
    ]
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
            "next_prerequisite"
        ]
    )
    residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring = (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring"
        ]
    )
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_hunk_insufficient_fixture_quality_pending"
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
        "terminal_outcome"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
        "upstream_residual_phase_magnitude_balance_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
        "selected_candidate_id"
    ] == "BB1_finite_residual_phase_magnitude_balance_private_parity_score"
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
            "candidate_count"
        ]
        == 5
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
        "metrics"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
        "metrics"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
        "thresholds"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation[
        "thresholds"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
        "phase_magnitude_balance_implementation_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
        "phase_magnitude_balance_hunk_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
        "phase_magnitude_balance_transfer_map_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
        "parity_scoring_lane_executed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
        "finite_reproducible_score"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
        "score_uses_retained_implementation_metrics"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
        "transfer_map_helper"
    ] == "_private_target_basis_residual_phase_magnitude_balance_source_interface_transverse_modal_transfer_map"
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
        "matrix_shape"
    ] == [3, 3]
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
            "residual_balance_bound"
        ]
        == 0.35
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
        "subgrid_vacuum_parity_scored"
    ]
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
            "subgrid_vacuum_parity_passed"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
            "material_improvement_demonstrated"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
            "production_patch_applied"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
            "public_observable_promoted"
        ]
        is False
    )
    phase_magnitude_balance_parity_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
            "candidate_ladder"
        ]
    }
    assert phase_magnitude_balance_parity_candidates[
        "BB1_finite_residual_phase_magnitude_balance_private_parity_score"
    ]["accepted_candidate"]
    assert (
        phase_magnitude_balance_parity_candidates[
            "BB4_residual_phase_magnitude_balance_parity_scoring_insufficient"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
        "next_lane_requires_target_basis_orientation_residual_phase_magnitude_balance_failure_theory"
    ]
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
            "next_prerequisite"
        ]
    )
    residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory = (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory"
        ]
    )
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_floor_theory_ready"
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory[
        "terminal_outcome"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory[
        "upstream_residual_phase_magnitude_balance_parity_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring_status"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory[
        "selected_candidate_id"
    ] == "BC1_residual_modal_coupling_floor"
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory[
            "candidate_count"
        ]
        == 5
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory[
        "metrics"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
        "metrics"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory[
        "thresholds"
    ] == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring[
        "thresholds"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory[
        "residual_phase_magnitude_balance_parity_scoring_consumed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory[
        "finite_parity_score_consumed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory[
        "phase_magnitude_balance_hunk_retained"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory[
        "selected_floor"
    ] == "residual_modal_coupling_floor"
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory[
        "selected_next_private_design_target"
    ] == "target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling"
    residual_modal_coupling_contract = residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory[
        "residual_modal_coupling_contract"
    ]
    assert residual_modal_coupling_contract["shape_contract"] == (
        "private fixed-shape 3x3 residual modal-coupling correction map"
    )
    assert residual_modal_coupling_contract[
        "consumes_phase_magnitude_balance_contract"
    ]
    assert residual_modal_coupling_contract["requires_public_observable"] is False
    assert residual_modal_coupling_contract["requires_threshold_change"] is False
    assert residual_modal_coupling_contract["requires_runner_state"] is False
    assert residual_modal_coupling_contract["implementation_deferred"]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory[
        "theory_lane_executed"
    ]
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory[
        "implementation_deferred"
    ]
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory[
            "production_patch_applied"
        ]
        is False
    )
    assert (
        residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory[
            "true_rt_readiness_unlocked"
        ]
        is False
    )
    phase_magnitude_balance_failure_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory[
            "candidate_ladder"
        ]
    }
    assert phase_magnitude_balance_failure_candidates[
        "BC1_residual_modal_coupling_floor"
    ]["accepted_candidate"]
    assert (
        phase_magnitude_balance_failure_candidates[
            "BC4_failure_theory_blocked_by_public_surface_or_threshold"
        ]["accepted_candidate"]
        is False
    )
    assert residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory[
        "next_lane_requires_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_implementation"
    ]
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory_next_prerequisite"
        ]
        == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory[
            "next_prerequisite"
        ]
    )
    residual_modal_coupling_implementation = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_implementation_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_hunk_retained_fixture_quality_pending"
    )
    assert residual_modal_coupling_implementation["terminal_outcome"] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_implementation_status"
    ]
    assert residual_modal_coupling_implementation["upstream_failure_theory_status"] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory_status"
    ]
    assert (
        residual_modal_coupling_implementation["selected_candidate_id"]
        == "BD1_private_residual_modal_coupling_transfer_map_helper"
    )
    assert residual_modal_coupling_implementation["candidate_count"] == 5
    assert residual_modal_coupling_implementation["metrics"] == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory["metrics"]
    assert residual_modal_coupling_implementation["thresholds"] == residual_basis_energy_biorthogonal_source_interface_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory["thresholds"]
    assert residual_modal_coupling_implementation["failure_theory_consumed"]
    assert residual_modal_coupling_implementation["residual_modal_coupling_contract_consumed"]
    implemented_residual_modal_coupling_contract = residual_modal_coupling_implementation[
        "residual_modal_coupling_contract"
    ]
    assert implemented_residual_modal_coupling_contract[
        "implementation_deferred"
    ] is False
    assert implemented_residual_modal_coupling_contract[
        "implementation_helper"
    ] == "_private_target_basis_residual_modal_coupling_source_interface_transverse_modal_transfer_map"
    assert implemented_residual_modal_coupling_contract[
        "consumes_failure_theory_contract"
    ]
    assert residual_modal_coupling_implementation["residual_modal_coupling_hunk_retained"]
    assert residual_modal_coupling_implementation["residual_modal_coupling_transfer_map_retained"]
    assert residual_modal_coupling_implementation["transfer_helper"] == (
        "_private_target_basis_residual_modal_coupling_source_interface_transverse_modal_transfer_map"
    )
    assert residual_modal_coupling_implementation["phase_magnitude_balance_helper_retained"] == (
        "_private_target_basis_residual_phase_magnitude_balance_source_interface_transverse_modal_transfer_map"
    )
    assert residual_modal_coupling_implementation["residual_phase_sign_helper_retained"] == (
        "_private_target_basis_residual_phase_sign_source_interface_transverse_modal_transfer_map"
    )
    assert residual_modal_coupling_implementation["matrix_shape"] == [3, 3]
    assert residual_modal_coupling_implementation["modal_coupling_bound"] == 0.35
    assert residual_modal_coupling_implementation["residual_modal_coupling_gate_fail_closed"]
    assert residual_modal_coupling_implementation["production_patch_applied"]
    assert residual_modal_coupling_implementation["solver_behavior_changed"]
    assert residual_modal_coupling_implementation["field_update_behavior_changed"]
    assert residual_modal_coupling_implementation["runner_behavior_changed"] is False
    assert residual_modal_coupling_implementation["new_solver_hunk_retained"]
    assert residual_modal_coupling_implementation["solver_hunk_retained"]
    assert residual_modal_coupling_implementation["subgrid_vacuum_parity_scored"] is False
    assert residual_modal_coupling_implementation["true_rt_readiness_unlocked"] is False
    assert residual_modal_coupling_implementation["public_observable_promoted"] is False
    residual_modal_coupling_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_modal_coupling_implementation["candidate_ladder"]
    }
    assert residual_modal_coupling_candidates[
        "BD1_private_residual_modal_coupling_transfer_map_helper"
    ]["accepted_candidate"]
    assert (
        residual_modal_coupling_candidates[
            "BD4_implementation_blocked_by_public_surface_or_threshold"
        ]["accepted_candidate"]
        is False
    )
    assert residual_modal_coupling_implementation[
        "next_lane_requires_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_parity_scoring"
    ]
    assert (
        benchmark_gate["private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_implementation_next_prerequisite"]
        == residual_modal_coupling_implementation["next_prerequisite"]
    )
    residual_modal_coupling_parity_scoring = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_parity_scoring"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_parity_scoring_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_hunk_insufficient_fixture_quality_pending"
    )
    assert residual_modal_coupling_parity_scoring["terminal_outcome"] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_parity_scoring_status"
    ]
    assert residual_modal_coupling_parity_scoring[
        "upstream_residual_modal_coupling_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_implementation_status"
    ]
    assert (
        residual_modal_coupling_parity_scoring["selected_candidate_id"]
        == "BE1_finite_residual_modal_coupling_private_parity_score"
    )
    assert residual_modal_coupling_parity_scoring["candidate_count"] == 5
    assert residual_modal_coupling_parity_scoring["metrics"] == residual_modal_coupling_implementation[
        "metrics"
    ]
    assert residual_modal_coupling_parity_scoring["thresholds"] == residual_modal_coupling_implementation[
        "thresholds"
    ]
    assert residual_modal_coupling_parity_scoring[
        "residual_modal_coupling_implementation_retained"
    ]
    assert residual_modal_coupling_parity_scoring[
        "residual_modal_coupling_hunk_retained"
    ]
    assert residual_modal_coupling_parity_scoring[
        "residual_modal_coupling_transfer_map_retained"
    ]
    assert residual_modal_coupling_parity_scoring[
        "parity_scoring_lane_executed"
    ]
    assert residual_modal_coupling_parity_scoring[
        "finite_reproducible_score"
    ]
    assert residual_modal_coupling_parity_scoring[
        "score_uses_retained_implementation_metrics"
    ]
    assert residual_modal_coupling_parity_scoring["transfer_map_helper"] == (
        "_private_target_basis_residual_modal_coupling_source_interface_transverse_modal_transfer_map"
    )
    assert residual_modal_coupling_parity_scoring[
        "phase_magnitude_balance_helper_retained"
    ] == (
        "_private_target_basis_residual_phase_magnitude_balance_source_interface_transverse_modal_transfer_map"
    )
    assert residual_modal_coupling_parity_scoring[
        "residual_phase_sign_helper_retained"
    ] == (
        "_private_target_basis_residual_phase_sign_source_interface_transverse_modal_transfer_map"
    )
    assert residual_modal_coupling_parity_scoring["matrix_shape"] == [3, 3]
    assert residual_modal_coupling_parity_scoring["modal_coupling_bound"] == 0.35
    assert residual_modal_coupling_parity_scoring["subgrid_vacuum_parity_scored"]
    assert residual_modal_coupling_parity_scoring[
        "subgrid_vacuum_parity_passed"
    ] is False
    assert residual_modal_coupling_parity_scoring[
        "material_improvement_demonstrated"
    ] is False
    assert residual_modal_coupling_parity_scoring[
        "true_rt_readiness_unlocked"
    ] is False
    assert residual_modal_coupling_parity_scoring[
        "production_patch_applied"
    ] is False
    assert residual_modal_coupling_parity_scoring[
        "solver_behavior_changed"
    ] is False
    assert residual_modal_coupling_parity_scoring[
        "runner_behavior_changed"
    ] is False
    assert residual_modal_coupling_parity_scoring[
        "retained_residual_modal_coupling_solver_hunk"
    ]
    assert residual_modal_coupling_parity_scoring[
        "public_observable_promoted"
    ] is False
    residual_modal_coupling_parity_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_modal_coupling_parity_scoring["candidate_ladder"]
    }
    assert residual_modal_coupling_parity_candidates[
        "BE1_finite_residual_modal_coupling_private_parity_score"
    ]["accepted_candidate"]
    assert (
        residual_modal_coupling_parity_candidates[
            "BE4_residual_modal_coupling_parity_scoring_insufficient"
        ]["accepted_candidate"]
        is False
    )
    assert residual_modal_coupling_parity_scoring[
        "next_lane_requires_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_failure_theory"
    ]
    assert (
        benchmark_gate["private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_parity_scoring_next_prerequisite"]
        == residual_modal_coupling_parity_scoring["next_prerequisite"]
    )
    residual_modal_coupling_failure_theory = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_failure_theory"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_failure_theory_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_floor_theory_ready"
    )
    assert residual_modal_coupling_failure_theory["terminal_outcome"] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_failure_theory_status"
    ]
    assert residual_modal_coupling_failure_theory[
        "upstream_residual_modal_coupling_parity_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_parity_scoring_status"
    ]
    assert residual_modal_coupling_failure_theory[
        "upstream_residual_modal_coupling_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_implementation_status"
    ]
    assert (
        residual_modal_coupling_failure_theory["selected_candidate_id"]
        == "BF1_packet_basis_mismatch_floor"
    )
    assert residual_modal_coupling_failure_theory["candidate_count"] == 5
    assert residual_modal_coupling_failure_theory[
        "metrics"
    ] == residual_modal_coupling_parity_scoring["metrics"]
    assert residual_modal_coupling_failure_theory[
        "thresholds"
    ] == residual_modal_coupling_parity_scoring["thresholds"]
    assert residual_modal_coupling_failure_theory[
        "residual_modal_coupling_parity_scoring_consumed"
    ]
    assert residual_modal_coupling_failure_theory["finite_parity_score_consumed"]
    assert residual_modal_coupling_failure_theory[
        "retained_residual_modal_coupling_solver_hunk"
    ]
    assert (
        residual_modal_coupling_failure_theory["selected_floor"]
        == "packet_basis_mismatch_floor"
    )
    assert residual_modal_coupling_failure_theory[
        "selected_next_private_design_target"
    ] == (
        "target_basis_orientation_residual_phase_magnitude_balance_"
        "residual_modal_coupling_packet_basis_mismatch"
    )
    packet_basis_mismatch_contract = residual_modal_coupling_failure_theory[
        "packet_basis_mismatch_contract"
    ]
    assert packet_basis_mismatch_contract["shape_contract"] == (
        "private fixed-shape source/interface packet-basis mismatch "
        "correction inside existing modal packet contract"
    )
    assert packet_basis_mismatch_contract[
        "consumes_residual_modal_coupling_contract"
    ]
    assert packet_basis_mismatch_contract["requires_public_observable"] is False
    assert packet_basis_mismatch_contract["requires_threshold_change"] is False
    assert packet_basis_mismatch_contract["requires_runner_state"] is False
    assert packet_basis_mismatch_contract["implementation_deferred"]
    assert residual_modal_coupling_failure_theory["theory_lane_executed"]
    assert residual_modal_coupling_failure_theory["implementation_deferred"]
    assert residual_modal_coupling_failure_theory["production_patch_applied"] is False
    assert (
        residual_modal_coupling_failure_theory["true_rt_readiness_unlocked"]
        is False
    )
    residual_modal_coupling_failure_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in residual_modal_coupling_failure_theory["candidate_ladder"]
    }
    assert residual_modal_coupling_failure_candidates[
        "BF1_packet_basis_mismatch_floor"
    ]["accepted_candidate"]
    assert (
        residual_modal_coupling_failure_candidates[
            "BF4_failure_theory_blocked_by_public_surface_or_threshold"
        ]["accepted_candidate"]
        is False
    )
    assert residual_modal_coupling_failure_theory[
        "next_lane_requires_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_implementation"
    ]
    assert (
        benchmark_gate["private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_failure_theory_next_prerequisite"]
        == residual_modal_coupling_failure_theory["next_prerequisite"]
    )
    packet_basis_mismatch_implementation = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_implementation"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_implementation_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_hunk_retained_fixture_quality_pending"
    )
    assert packet_basis_mismatch_implementation["terminal_outcome"] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_implementation_status"
    ]
    assert packet_basis_mismatch_implementation[
        "upstream_failure_theory_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_failure_theory_status"
    ]
    assert (
        packet_basis_mismatch_implementation["selected_candidate_id"]
        == "BG1_private_packet_basis_mismatch_transfer_map_helper"
    )
    assert packet_basis_mismatch_implementation["candidate_count"] == 5
    assert packet_basis_mismatch_implementation[
        "metrics"
    ] == residual_modal_coupling_failure_theory["metrics"]
    assert packet_basis_mismatch_implementation[
        "thresholds"
    ] == residual_modal_coupling_failure_theory["thresholds"]
    assert packet_basis_mismatch_implementation["failure_theory_consumed"]
    assert packet_basis_mismatch_implementation[
        "packet_basis_mismatch_contract_consumed"
    ]
    implemented_packet_basis_contract = packet_basis_mismatch_implementation[
        "packet_basis_mismatch_contract"
    ]
    assert implemented_packet_basis_contract["implementation_deferred"] is False
    assert implemented_packet_basis_contract["implementation_helper"] == "_private_target_basis_residual_modal_coupling_packet_basis_mismatch_source_interface_transverse_modal_transfer_map"
    assert implemented_packet_basis_contract["consumes_failure_theory_contract"]
    assert packet_basis_mismatch_implementation[
        "packet_basis_mismatch_hunk_retained"
    ]
    assert packet_basis_mismatch_implementation[
        "packet_basis_mismatch_transfer_map_retained"
    ]
    assert packet_basis_mismatch_implementation["transfer_helper"] == "_private_target_basis_residual_modal_coupling_packet_basis_mismatch_source_interface_transverse_modal_transfer_map"
    assert packet_basis_mismatch_implementation[
        "residual_modal_coupling_helper_retained"
    ] == (
        "_private_target_basis_residual_modal_coupling_source_interface_transverse_modal_transfer_map"
    )
    assert packet_basis_mismatch_implementation["matrix_shape"] == [3, 3]
    assert packet_basis_mismatch_implementation[
        "packet_basis_mismatch_bound"
    ] == 0.35
    assert packet_basis_mismatch_implementation[
        "packet_basis_mismatch_gate_fail_closed"
    ]
    assert packet_basis_mismatch_implementation["production_patch_applied"]
    assert packet_basis_mismatch_implementation["solver_behavior_changed"]
    assert packet_basis_mismatch_implementation["field_update_behavior_changed"]
    assert packet_basis_mismatch_implementation["runner_behavior_changed"] is False
    assert packet_basis_mismatch_implementation["new_solver_hunk_retained"]
    assert packet_basis_mismatch_implementation["solver_hunk_retained"]
    assert packet_basis_mismatch_implementation[
        "subgrid_vacuum_parity_scored"
    ] is False
    assert packet_basis_mismatch_implementation[
        "true_rt_readiness_unlocked"
    ] is False
    assert packet_basis_mismatch_implementation["public_observable_promoted"] is False
    packet_basis_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in packet_basis_mismatch_implementation["candidate_ladder"]
    }
    assert packet_basis_candidates[
        "BG1_private_packet_basis_mismatch_transfer_map_helper"
    ]["accepted_candidate"]
    assert (
        packet_basis_candidates[
            "BG4_implementation_blocked_by_public_surface_or_threshold"
        ]["accepted_candidate"]
        is False
    )
    assert packet_basis_mismatch_implementation[
        "next_lane_requires_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_parity_scoring"
    ]
    assert (
        benchmark_gate["private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_implementation_next_prerequisite"]
        == packet_basis_mismatch_implementation["next_prerequisite"]
    )
    packet_basis_mismatch_parity_scoring = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_parity_scoring"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_parity_scoring_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_hunk_insufficient_fixture_quality_pending"
    )
    assert packet_basis_mismatch_parity_scoring["terminal_outcome"] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_parity_scoring_status"
    ]
    assert packet_basis_mismatch_parity_scoring[
        "upstream_packet_basis_mismatch_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_implementation_status"
    ]
    assert (
        packet_basis_mismatch_parity_scoring["selected_candidate_id"]
        == "BH1_finite_packet_basis_mismatch_private_parity_score"
    )
    assert packet_basis_mismatch_parity_scoring["candidate_count"] == 5
    assert packet_basis_mismatch_parity_scoring[
        "metrics"
    ] == packet_basis_mismatch_implementation["metrics"]
    assert packet_basis_mismatch_parity_scoring[
        "thresholds"
    ] == packet_basis_mismatch_implementation["thresholds"]
    assert packet_basis_mismatch_parity_scoring[
        "packet_basis_mismatch_implementation_retained"
    ]
    assert packet_basis_mismatch_parity_scoring[
        "packet_basis_mismatch_hunk_retained"
    ]
    assert packet_basis_mismatch_parity_scoring[
        "packet_basis_mismatch_transfer_map_retained"
    ]
    assert packet_basis_mismatch_parity_scoring["parity_scoring_lane_executed"]
    assert packet_basis_mismatch_parity_scoring["finite_reproducible_score"]
    assert packet_basis_mismatch_parity_scoring[
        "score_uses_retained_implementation_metrics"
    ]
    assert packet_basis_mismatch_parity_scoring["transfer_helper"] == "_private_target_basis_residual_modal_coupling_packet_basis_mismatch_source_interface_transverse_modal_transfer_map"
    assert packet_basis_mismatch_parity_scoring[
        "residual_modal_coupling_helper_retained"
    ] == (
        "_private_target_basis_residual_modal_coupling_source_interface_transverse_modal_transfer_map"
    )
    assert packet_basis_mismatch_parity_scoring["matrix_shape"] == [3, 3]
    assert packet_basis_mismatch_parity_scoring[
        "packet_basis_mismatch_bound"
    ] == 0.35
    assert packet_basis_mismatch_parity_scoring["subgrid_vacuum_parity_scored"]
    assert packet_basis_mismatch_parity_scoring[
        "subgrid_vacuum_parity_passed"
    ] is False
    assert packet_basis_mismatch_parity_scoring[
        "material_improvement_demonstrated"
    ] is False
    assert packet_basis_mismatch_parity_scoring[
        "true_rt_readiness_unlocked"
    ] is False
    assert packet_basis_mismatch_parity_scoring["production_patch_applied"] is False
    assert packet_basis_mismatch_parity_scoring["solver_behavior_changed"] is False
    assert packet_basis_mismatch_parity_scoring["runner_behavior_changed"] is False
    assert packet_basis_mismatch_parity_scoring[
        "retained_packet_basis_mismatch_solver_hunk"
    ]
    assert packet_basis_mismatch_parity_scoring[
        "public_observable_promoted"
    ] is False
    packet_basis_parity_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in packet_basis_mismatch_parity_scoring["candidate_ladder"]
    }
    assert packet_basis_parity_candidates[
        "BH1_finite_packet_basis_mismatch_private_parity_score"
    ]["accepted_candidate"]
    assert (
        packet_basis_parity_candidates[
            "BH4_packet_basis_mismatch_parity_scoring_insufficient"
        ]["accepted_candidate"]
        is False
    )
    assert packet_basis_mismatch_parity_scoring[
        "next_lane_requires_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_failure_theory"
    ]
    assert (
        benchmark_gate["private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_parity_scoring_next_prerequisite"]
        == packet_basis_mismatch_parity_scoring["next_prerequisite"]
    )
    packet_basis_mismatch_failure_theory = benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_failure_theory"
    ]
    assert benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_failure_theory_status"
    ] == (
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_floor_theory_ready"
    )
    assert packet_basis_mismatch_failure_theory["terminal_outcome"] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_failure_theory_status"
    ]
    assert packet_basis_mismatch_failure_theory[
        "upstream_packet_basis_mismatch_parity_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_parity_scoring_status"
    ]
    assert packet_basis_mismatch_failure_theory[
        "upstream_packet_basis_mismatch_implementation_status"
    ] == benchmark_gate[
        "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_implementation_status"
    ]
    assert (
        packet_basis_mismatch_failure_theory["selected_candidate_id"]
        == "BI2_owner_packet_weighting_floor"
    )
    assert packet_basis_mismatch_failure_theory["candidate_count"] == 5
    assert (
        packet_basis_mismatch_failure_theory["metrics"]
        == packet_basis_mismatch_parity_scoring["metrics"]
    )
    assert (
        packet_basis_mismatch_failure_theory["thresholds"]
        == packet_basis_mismatch_parity_scoring["thresholds"]
    )
    assert packet_basis_mismatch_failure_theory[
        "packet_basis_mismatch_parity_scoring_consumed"
    ]
    assert packet_basis_mismatch_failure_theory["finite_parity_score_consumed"]
    assert packet_basis_mismatch_failure_theory[
        "retained_packet_basis_mismatch_solver_hunk"
    ]
    assert (
        packet_basis_mismatch_failure_theory["selected_floor"]
        == "owner_packet_weighting_floor"
    )
    assert packet_basis_mismatch_failure_theory[
        "selected_next_private_design_target"
    ] == (
        "target_basis_orientation_residual_phase_magnitude_balance_"
        "residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting"
    )
    owner_packet_weighting_contract = packet_basis_mismatch_failure_theory[
        "owner_packet_weighting_contract"
    ]
    assert owner_packet_weighting_contract["shape_contract"] == (
        "private fixed-shape source/interface owner-packet weighting "
        "correction inside existing modal packet contract"
    )
    assert owner_packet_weighting_contract[
        "consumes_packet_basis_mismatch_contract"
    ]
    assert owner_packet_weighting_contract[
        "consumes_packet_basis_mismatch_parity_score"
    ]
    assert owner_packet_weighting_contract["requires_public_observable"] is False
    assert owner_packet_weighting_contract["requires_threshold_change"] is False
    assert owner_packet_weighting_contract["requires_runner_state"] is False
    assert owner_packet_weighting_contract["implementation_deferred"]
    assert packet_basis_mismatch_failure_theory["theory_lane_executed"]
    assert packet_basis_mismatch_failure_theory["failure_theory_lane_executed"]
    assert packet_basis_mismatch_failure_theory["implementation_deferred"]
    assert packet_basis_mismatch_failure_theory["production_patch_applied"] is False
    assert packet_basis_mismatch_failure_theory["solver_behavior_changed"] is False
    assert packet_basis_mismatch_failure_theory["runner_behavior_changed"] is False
    assert (
        packet_basis_mismatch_failure_theory["true_rt_readiness_unlocked"]
        is False
    )
    assert packet_basis_mismatch_failure_theory[
        "public_observable_promoted"
    ] is False
    packet_basis_failure_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in packet_basis_mismatch_failure_theory["candidate_ladder"]
    }
    assert packet_basis_failure_candidates[
        "BI2_owner_packet_weighting_floor"
    ]["accepted_candidate"]
    assert (
        packet_basis_failure_candidates[
            "BI4_failure_theory_blocked_by_public_surface_or_threshold"
        ]["accepted_candidate"]
        is False
    )
    assert packet_basis_mismatch_failure_theory[
        "next_lane_requires_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_owner_packet_weighting_implementation"
    ]
    assert (
        benchmark_gate["private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_failure_theory_next_prerequisite"]
        == packet_basis_mismatch_failure_theory["next_prerequisite"]
    )
    assert benchmark_gate["next_prerequisite"] == (
        "private plane-wave modal projection/normalizer projected target residual-basis "
        "energy-biorthogonal source-interface transverse modal transfer-map target-"
        "basis orientation residual phase/magnitude balance residual modal-coupling "
        "packet-basis mismatch owner-packet weighting implementation after "
        "failure-theory contract ready ralplan"
    )
    assert benchmark_gate["follow_up_recommendation"] == (
        "private plane-wave modal projection/normalizer projected target residual-basis "
        "energy-biorthogonal source-interface transverse modal transfer-map target-"
        "basis orientation residual phase/magnitude balance residual modal-coupling "
        "packet-basis mismatch owner-packet weighting implementation after "
        "failure-theory contract ready ralplan"
    )
    assert "paired_face_coupling_design_ready" in benchmark_gate["blocking_diagnostic"]
    assert (
        "production_context_mismatch_detected" in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate["private_subgrid_vacuum_plane_wave_fixture_contract_status"]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate["private_plane_wave_source_fixture_path_wiring_status"]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate["private_plane_wave_source_adapter_design_status"]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate["private_plane_wave_parity_blocker_repair_design_status"]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate["private_plane_wave_interface_floor_implementation_status"]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate["private_plane_wave_root_cause_redesign_status"]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate["private_plane_wave_interface_energy_form_design_status"]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_interface_energy_form_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_interface_energy_form_failure_theory_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_operator_mortar_energy_form_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_transverse_phase_coherence_architecture_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_phase_coherence_staging_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_solver_wide_interface_state_owner_architecture_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_solver_wide_interface_state_owner_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_solver_state_owner_propagation_boundary_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_solver_state_owner_propagation_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate["private_plane_wave_owner_scan_wiring_joint_scoring_status"]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_owner_backed_physical_phase_cv_correction_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_face_local_modal_correction_architecture_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_face_local_modal_correction_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate["private_plane_wave_face_local_modal_failure_theory_status"]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_face_local_modal_retry_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate["private_plane_wave_observable_proxy_modal_architecture_status"]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_observable_proxy_modal_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_observable_proxy_face_packet_state_shape_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_observable_proxy_face_packet_state_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_observable_proxy_modal_retry_after_face_packet_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_observable_proxy_modal_retry_parity_scoring_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_observable_proxy_modal_retry_failure_theory_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_observable_proxy_modal_retry_redesign_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_source_interface_ownership_state_shape_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_source_interface_ownership_state_shape_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_propagation_aware_modal_retry_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_propagation_aware_modal_retry_parity_scoring_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_source_owner_incident_packet_population_design_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_source_owner_incident_packet_population_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_source_populated_propagation_aware_modal_retry_parity_scoring_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_source_populated_propagation_aware_modal_retry_failure_theory_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_source_interface_time_aligned_packet_staging_design_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_source_interface_time_aligned_packet_staging_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_time_aligned_modal_retry_parity_scoring_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_time_aligned_modal_retry_failure_theory_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_contract_design_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_contract_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_contract_parity_scoring_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_contract_failure_theory_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_basis_design_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_basis_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_basis_parity_scoring_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_basis_failure_theory_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_parity_scoring_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_failure_theory_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_design_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_parity_scoring_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_failure_theory_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_design_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_parity_scoring_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_failure_theory_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_design_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_parity_scoring_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_metric_shape_calibration_failure_theory_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_design_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_parity_scoring_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_transverse_modal_coupling_metric_failure_theory_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_parity_scoring_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_failure_theory_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_parity_scoring_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_failure_theory_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_parity_scoring_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_sign_failure_theory_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_parity_scoring_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_failure_theory_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_parity_scoring_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_failure_theory_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_implementation_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_parity_scoring_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate[
            "private_plane_wave_modal_projection_normalizer_projected_target_residual_basis_energy_biorthogonal_source_interface_transverse_modal_transfer_map_target_basis_orientation_residual_phase_magnitude_balance_residual_modal_coupling_packet_basis_mismatch_failure_theory_status"
        ]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate["private_plane_wave_source_adapter_implementation_status"]
        in benchmark_gate["blocking_diagnostic"]
    )
    assert (
        benchmark_gate["private_subgrid_vacuum_plane_wave_parity_scoring_status"]
        in benchmark_gate["blocking_diagnostic"]
    )
    staging = benchmark_gate["private_time_centered_staging_redesign"]
    assert benchmark_gate["private_time_centered_staging_redesign_status"] == (
        "time_centered_staging_contract_ready"
    )
    assert staging["status"] == "time_centered_staging_contract_ready"
    assert staging["terminal_outcome"] == "time_centered_staging_contract_ready"
    assert staging["upstream_helper_status"] == "production_context_mismatch_detected"
    assert staging["selected_candidate_id"] == "same_call_centered_h_bar"
    assert staging["same_call_local_staging_contract_ready"] is True
    assert staging["production_expressibility_gate"]["passed"] is True
    assert (
        staging["production_expressibility_gate"]["forbids_private_post_h_hook"] is True
    )
    assert (
        staging["production_expressibility_gate"]["forbids_private_post_e_hook"] is True
    )
    assert (
        staging["production_expressibility_gate"]["forbids_test_local_hook_emulation"]
        is True
    )
    assert staging["thresholds"]["ledger_balance_threshold"] == 0.02
    assert staging["thresholds"]["coupling_strength_ratio_min"] == 0.5
    assert staging["thresholds"]["bounding_scalar_min"] == 0.5
    assert staging["thresholds"]["bounding_scalar_max"] == 2.0
    selected_staging = [
        candidate
        for candidate in staging["candidates"]
        if candidate["accepted_candidate"]
    ]
    assert len(selected_staging) == 1
    selected_staging = selected_staging[0]
    assert selected_staging["candidate_id"] == "same_call_centered_h_bar"
    assert selected_staging["ledger_gate_passed"] is True
    assert (
        selected_staging["ledger_normalized_balance_residual"]
        <= selected_staging["ledger_threshold"]
    )
    assert selected_staging["production_expressibility"]["passes_gate"] is True
    assert selected_staging["requires_hook"] is False
    assert selected_staging["requires_runner_state"] is False
    assert selected_staging["requires_public_api"] is False
    assert selected_staging["requires_public_observable"] is False
    rejection_categories = {
        candidate["rejection_metadata"]["category"]
        for candidate in staging["candidates"]
        if not candidate["accepted_candidate"]
    }
    assert "sat_update_reordering_required" in rejection_categories
    assert "trace_unavailable_at_insertion_point" in rejection_categories
    assert "cross_step_state_required" in rejection_categories
    for path_name in ("non_cpml", "cpml"):
        slots = staging["cpml_non_cpml_staging_contract"][f"{path_name}_slots"]
        assert slots["same_call_local_values_available"] is True
        assert slots["requires_next_step_state"] is False
        assert slots["requires_update_reordering"] is False
        assert slots["h_sat_line"] < slots["first_e_update_line"]
        assert slots["e_sat_line"] > slots["last_e_update_line"]
    assert (
        staging["cpml_non_cpml_staging_contract"][
            "internal_face_work_contract_identical"
        ]
        is True
    )
    assert staging["orientation_generalization"]["uses_face_orientations_only"] is True
    assert staging["solver_behavior_changed"] is False
    assert staging["sbp_sat_3d_time_centered_staging_applied"] is False
    assert staging["hook_experiment_allowed"] is False
    assert staging["jit_runner_changed"] is False
    assert staging["runner_changed"] is False
    assert staging["public_claim_allowed"] is False
    assert staging["public_api_behavior_changed"] is False
    assert staging["public_default_tau_changed"] is False
    assert staging["public_observable_promoted"] is False
    assert staging["promotion_candidate_ready"] is False
    assert staging["simresult_changed"] is False
    assert staging["result_surface_changed"] is False
    assert (
        benchmark_gate["private_time_centered_staging_redesign_next_prerequisite"]
        == staging["next_prerequisite"]
    )
    assert staging["next_prerequisite"] == (
        "private time-centered paired-face helper implementation ralplan"
    )
    helper = benchmark_gate["private_time_centered_paired_face_helper_implementation"]
    assert (
        benchmark_gate["private_time_centered_paired_face_helper_implementation_status"]
        == "private_time_centered_paired_face_helper_implemented"
    )
    assert helper["status"] == "private_time_centered_paired_face_helper_implemented"
    assert (
        helper["terminal_outcome"]
        == "private_time_centered_paired_face_helper_implemented"
    )
    assert helper["upstream_staging_status"] == "time_centered_staging_contract_ready"
    assert helper["selected_time_centering_schema"] == "same_call_centered_h_bar"
    assert helper["bounded_relaxation"] == 0.02
    for path_name in ("non_cpml", "cpml"):
        slots = helper["production_slot_binding"][path_name]
        assert slots["h_pre_sat_slot_bound"] is True
        assert slots["h_post_sat_slot_bound"] is True
        assert slots["e_pre_sat_slot_bound"] is True
        assert slots["e_post_sat_slot_bound"] is True
        assert slots["helper_after_e_sat"] is True
        assert slots["helper_uses_private_post_h_hook"] is False
        assert slots["helper_uses_private_post_e_hook"] is False
        assert all(slots["helper_signature_fields"].values())
    assert (
        helper["cpml_non_cpml_helper_contract"]["internal_face_work_contract_identical"]
        is True
    )
    assert helper["orientation_generalization"]["uses_face_orientations_only"] is True
    assert len(helper["hunk_inventory"]) == 6
    assert helper["production_patch_applied"] is True
    assert helper["accepted_private_helper"] is True
    assert helper["solver_behavior_changed"] is True
    assert helper["sbp_sat_3d_time_centered_paired_face_helper_applied"] is True
    assert helper["helper_specific_switch_added"] is False
    assert helper["uses_private_post_h_hook"] is False
    assert helper["uses_private_post_e_hook"] is False
    assert helper["uses_test_local_hook_emulation"] is False
    assert helper["hook_experiment_allowed"] is False
    assert helper["jit_runner_changed"] is False
    assert helper["runner_changed"] is False
    assert helper["public_claim_allowed"] is False
    assert helper["public_api_behavior_changed"] is False
    assert helper["public_default_tau_changed"] is False
    assert helper["public_observable_promoted"] is False
    assert helper["promotion_candidate_ready"] is False
    assert helper["simresult_changed"] is False
    assert helper["result_surface_changed"] is False
    assert (
        benchmark_gate[
            "private_time_centered_paired_face_helper_implementation_next_prerequisite"
        ]
        == helper["next_prerequisite"]
    )
    assert helper["next_prerequisite"] == (
        "private time-centered paired-face helper fixture-quality recovery ralplan"
    )
    recovery = benchmark_gate["private_time_centered_helper_fixture_quality_recovery"]
    assert (
        benchmark_gate["private_time_centered_helper_fixture_quality_recovery_status"]
        == "measurement_contract_or_interface_floor_persists"
    )
    assert recovery["terminal_outcome"] == (
        "measurement_contract_or_interface_floor_persists"
    )
    assert recovery["candidate_ladder_declared_before_slow_scoring"] is True
    assert recovery["candidate_count"] == 4
    assert recovery["selected_candidate_id"] == "C0_current_helper_original_fixture"
    assert recovery["solver_hunk_touched"] is True
    assert recovery["solver_hunk_retained"] is False
    assert recovery["current_fixture_metrics_retained"] is True
    assert recovery["slab_rt_private_only"] is True
    assert recovery["fixture_quality_ready"] is False
    assert recovery["reference_quality_ready"] is False
    assert recovery["public_claim_allowed"] is False
    assert recovery["public_observable_promoted"] is False
    assert recovery["hook_experiment_allowed"] is False
    assert recovery["slab_rt_public_claim_allowed"] is False
    candidate_by_id = {
        candidate["candidate_id"]: candidate for candidate in recovery["candidates"]
    }
    assert set(candidate_by_id) == {
        "C0_current_helper_original_fixture",
        "C1_center_core_measurement_control",
        "C2_one_cell_downstream_plane_control",
        "C3_helper_relaxation_0p05_original_fixture",
    }
    assert (
        candidate_by_id["C1_center_core_measurement_control"][
            "can_claim_original_fixture_recovery"
        ]
        is False
    )
    assert (
        candidate_by_id["C2_one_cell_downstream_plane_control"][
            "can_claim_original_fixture_recovery"
        ]
        is False
    )
    c3 = candidate_by_id["C3_helper_relaxation_0p05_original_fixture"]
    assert c3["solver_touch"] is True
    assert c3["rollback_required"] is True
    assert c3["rollback_verified"] is True
    assert c3["retained_solver_relaxation"] == 0.02
    assert (
        benchmark_gate[
            "private_time_centered_helper_fixture_quality_recovery_next_prerequisite"
        ]
        == recovery["next_prerequisite"]
    )
    redesign = benchmark_gate["private_measurement_contract_interface_floor_redesign"]
    assert (
        benchmark_gate["private_measurement_contract_interface_floor_redesign_status"]
        == "persistent_interface_floor_confirmed"
    )
    assert (
        redesign["terminal_outcome"]
        == (
            benchmark_gate[
                "private_measurement_contract_interface_floor_redesign_status"
            ]
        )
    )
    assert redesign["diagnostic_ladder_declared_before_scoring"] is True
    assert redesign["diagnostic_count"] == 5
    assert redesign["diagnostic_ids"] == [
        "D0_current_integrated_flux_contract",
        "D1_prior_measurement_controls_summary",
        "D2_phase_referenced_modal_coherence_projection",
        "D3_local_eh_impedance_poynting_projection",
        "D4_interface_ledger_correlation",
    ]
    assert redesign["d2_ready"] is False
    assert redesign["d3_ready"] is False
    assert redesign["d4_positive"] is True
    assert redesign["solver_hunk_touched"] is False
    assert redesign["public_claim_allowed"] is False
    assert redesign["public_observable_promoted"] is False
    assert redesign["hook_experiment_allowed"] is False
    assert redesign["api_surface_changed"] is False
    assert redesign["result_surface_changed"] is False
    assert redesign["runner_surface_changed"] is False
    assert redesign["env_config_changed"] is False
    diagnostic_by_id = {
        diagnostic["diagnostic_id"]: diagnostic
        for diagnostic in redesign["diagnostics"]
    }
    assert set(diagnostic_by_id) == set(redesign["diagnostic_ids"])
    d2 = diagnostic_by_id["D2_phase_referenced_modal_coherence_projection"]
    assert d2["field_array_inputs"] == ["e1_dft", "e2_dft", "h1_dft", "h2_dft"]
    assert d2["fixture_quality_gate_replacement"] is False
    assert d2["d2_ready"] is False
    assert d2["uniform_reference_ready"] is False
    d3 = diagnostic_by_id["D3_local_eh_impedance_poynting_projection"]
    assert d3["fixture_quality_gate_replacement"] is False
    assert d3["mask_provenance_ready"] is True
    assert d3["d3_ready"] is False
    assert d3["metrics"]["mask_provenance_mismatch_count"] == 0
    d4 = diagnostic_by_id["D4_interface_ledger_correlation"]
    assert d4["d4_positive"] is True
    assert d4["provenance"]["interface_energy_transfer_diagnostics"] == (
        "current_helper_state_recomputed"
    )
    assert d4["provenance"]["manufactured_face_ledger_evidence"] == (
        "prior_committed_evidence"
    )
    assert d4["manufactured_face_ledger_evidence"]["context_only"] is True
    assert (
        benchmark_gate[
            "private_measurement_contract_interface_floor_redesign_next_prerequisite"
        ]
        == redesign["next_prerequisite"]
    )
    assert benchmark_gate["next_prerequisite"] == (
        "private plane-wave modal projection/normalizer projected target residual-basis "
        "energy-biorthogonal source-interface transverse modal transfer-map target-"
        "basis orientation residual phase/magnitude balance residual modal-coupling "
        "packet-basis mismatch owner-packet weighting implementation after "
        "failure-theory contract ready ralplan"
    )
    assert (
        "time_centered_staging_contract_ready"
        in (benchmark_gate["blocking_diagnostic"])
    )
    assert (
        "private_time_centered_paired_face_helper_implemented"
        in benchmark_gate["blocking_diagnostic"]
    )
    assert "no_material_repair" in benchmark_gate["blocking_diagnostic"]
    assert "not public TFSF" in benchmark_gate["diagnostic_basis"]
    assert _sbp_lane()["supported_subset"]["observables"] == ["point_probe"]
    assert TRUE_RT_SPEC.exists()

    spec_text = TRUE_RT_SPEC.read_text()
    assert "proxy numerical-equivalence benchmark" in spec_text
    assert "does **not** establish calibrated physical reflection" in spec_text
    assert "true R/T benchmark: deferred" in spec_text
    assert "bounded-CPML feasibility result" in spec_text
    assert "inconclusive" in spec_text
    assert "private analytic sheet/source" in spec_text
    assert "private TFSF-style incident field" in spec_text
    assert "same-contract private reference" in spec_text
    assert "support matrix continues to mark true R/T as deferred" in spec_text
    assert "no_signature_compatible_bounded_repair" in spec_text
    assert "paired_face_coupling_design_ready" in spec_text
    assert "production_context_mismatch_detected" in spec_text
    assert "time_centered_staging_contract_ready" in spec_text
    assert "private_time_centered_paired_face_helper_implemented" in spec_text
    assert (
        "private time-centered paired-face helper fixture-quality recovery" in spec_text
    )
    assert "measurement_contract_or_interface_floor_persists" in spec_text
    assert (
        "private measurement-contract/interface-floor redesign after helper recovery failed"
        in spec_text
    )
    assert "persistent_interface_floor_confirmed" in spec_text
    assert "private interface-floor repair theory/implementation" in spec_text
    assert "no_bounded_private_interface_floor_repair" in spec_text
    assert "oriented_characteristic_face_balance" in spec_text
    assert "higher-order SBP face-norm/interface-operator redesign" in spec_text
    assert "no_private_face_norm_operator_repair" in spec_text
    assert "private broader SBP derivative/interior-boundary operator" in spec_text
    assert "no_private_derivative_interface_repair" in spec_text
    assert "requires_global_sbp_operator_refactor" in spec_text
    assert "global SBP derivative/mortar operator architecture" in spec_text
    assert "private_global_operator_3d_contract_ready" in spec_text
    assert "A1-A4 identity evidence" in spec_text
    assert "all-six-face edge/corner partition closure" in spec_text
    assert "CPML/non-CPML" in spec_text
    assert "private solver integration hunk" in spec_text
    assert "from global SBP derivative/mortar operator architecture" in spec_text
    assert (
        "private_subgrid_vacuum_plane_wave_parity_failed_no_public_promotion"
        in spec_text
    )
    assert (
        "private_plane_wave_interface_floor_repair_design_required" in spec_text
    )
    assert "no_private_plane_wave_interface_floor_repair" in spec_text
    assert (
        "private_plane_wave_interface_energy_form_root_cause_identified"
        in spec_text
    )
    assert (
        "private_plane_wave_interface_energy_form_implementation_contract_ready"
        in spec_text
    )
    assert "## Deferred issue record" in spec_text


def test_sbp_sat_unsupported_surface_remains_hard_fail_in_support_matrix():
    unsupported = _sbp_lane()["unsupported_combinations"]

    for key in (
        "upml_boundary",
        "cpml_box_inside_absorber_guard",
        "per_face_cpml_thickness_override",
        "mixed_reflector_cpml_boundaryspec",
        "mixed_periodic_cpml_boundaryspec",
        "mixed_pmc_periodic_boundaryspec",
        "partial_touch_periodic_axis",
        "xy_margin_geometry_autobox",
        "ntff",
        "dft_plane",
        "flux_monitor",
        "tfsf",
        "waveguide_port",
        "floquet_port",
        "lumped_rlc",
        "coaxial_port",
        "impedance_or_wire_port",
    ):
        assert unsupported[key] == "hard_fail"
