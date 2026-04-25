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
        "boundary_expanded_private_analytic_sheet_flux_plane_vacuum_slab"
    )
    assert benchmark_gate["fixture_name"] == "boundary_expanded"
    assert benchmark_gate["source_contract"] == "private_analytic_sheet_source"
    assert benchmark_gate["normalization"] == (
        "vacuum_device_two_run_incident_normalized"
    )
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
        "private analytic-sheet bounded-CPML fixture"
    )
    sweep = benchmark_gate["boundary_expansion_sweep"]
    assert sweep["status"] == "inconclusive"
    assert sweep["candidate_count"] == 2
    assert sweep["selected_fixture"] == "boundary_expanded"
    assert sweep["selected_usable_bins"] == 3
    assert sweep["materially_improved_vs_baseline"] is True
    assert "non-public" in benchmark_gate["blocking_diagnostic"]
    assert "private TFSF-style" in benchmark_gate["next_prerequisite"]
    assert "normalization-repair" in benchmark_gate["next_prerequisite"]
    assert "analytic-sheet injection" in benchmark_gate["diagnostic_basis"]
    assert (
        "boundary-expanded fixture-quality sweep" in benchmark_gate["diagnostic_basis"]
    )
    assert _sbp_lane()["supported_subset"]["observables"] == ["point_probe"]
    assert TRUE_RT_SPEC.exists()

    spec_text = TRUE_RT_SPEC.read_text()
    assert "proxy numerical-equivalence benchmark" in spec_text
    assert "does **not** establish calibrated physical reflection" in spec_text
    assert "true R/T benchmark: deferred" in spec_text
    assert "bounded-CPML feasibility result" in spec_text
    assert "inconclusive" in spec_text
    assert "private analytic sheet/source" in spec_text
    assert "boundary-expanded analytic-sheet sweep" in spec_text
    assert "support matrix continues to mark true R/T as deferred" in spec_text
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
