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
    assert benchmark_gate["next_prerequisite"] == (
        "private plane-wave interface-floor repair implementation before true R/T "
        "readiness ralplan"
    )
    assert benchmark_gate["follow_up_recommendation"] == (
        "private plane-wave interface-floor repair implementation before true R/T "
        "readiness ralplan"
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
        "private plane-wave interface-floor repair implementation before true R/T "
        "readiness ralplan"
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
