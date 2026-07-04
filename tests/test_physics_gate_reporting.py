from __future__ import annotations

import importlib.util
import json
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_DIR = REPO_ROOT / "scripts" / "diagnostics"


def _load_module(name: str, path: Path):
    sys.path.insert(0, str(DIAGNOSTICS_DIR))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(DIAGNOSTICS_DIR))
        except ValueError:
            pass


run_gate = _load_module(
    "run_physics_gate", DIAGNOSTICS_DIR / "run_physics_gate.py"
)
aggregate_gate = _load_module(
    "aggregate_physics_gate", DIAGNOSTICS_DIR / "aggregate_physics_gate.py"
)
msl_openems_compare = _load_module(
    "compare_msl_thru_openems_reference",
    DIAGNOSTICS_DIR / "compare_msl_thru_openems_reference.py",
)
waveguide_report = _load_module(
    "report_waveguide_envelope",
    DIAGNOSTICS_DIR / "report_waveguide_envelope.py",
)
msl_report = _load_module(
    "report_msl_envelope",
    DIAGNOSTICS_DIR / "report_msl_envelope.py",
)
coaxial_tem_report = _load_module(
    "report_coaxial_tem_oracles",
    DIAGNOSTICS_DIR / "report_coaxial_tem_oracles.py",
)
sparameter_claim_audit = _load_module(
    "audit_sparameter_claims",
    DIAGNOSTICS_DIR / "audit_sparameter_claims.py",
)
rf_infra_goal_audit = _load_module(
    "audit_rf_infra_e5_goal",
    DIAGNOSTICS_DIR / "audit_rf_infra_e5_goal.py",
)
floquet_modal_report = _load_module(
    "report_floquet_modal_oracles",
    DIAGNOSTICS_DIR / "report_floquet_modal_oracles.py",
)
vessl_shard_check = _load_module(
    "check_vessl_physics_shards",
    DIAGNOSTICS_DIR / "check_vessl_physics_shards.py",
)
floquet_modal_replay = _load_module(
    "replay_floquet_modal_field_dump",
    DIAGNOSTICS_DIR / "replay_floquet_modal_field_dump.py",
)
port_external_reference_check = _load_module(
    "check_port_external_references",
    DIAGNOSTICS_DIR / "check_port_external_references.py",
)
port_external_reference_shard = _load_module(
    "report_port_external_reference_shard",
    DIAGNOSTICS_DIR / "report_port_external_reference_shard.py",
)
port_external_shard_result_check = _load_module(
    "check_port_external_shard_results",
    DIAGNOSTICS_DIR / "check_port_external_shard_results.py",
)
port_external_shard_execution_manifest = _load_module(
    "build_port_external_shard_execution_manifest",
    DIAGNOSTICS_DIR / "build_port_external_shard_execution_manifest.py",
)
sparameter_reference_compare = _load_module(
    "compare_sparameter_reference",
    DIAGNOSTICS_DIR / "compare_sparameter_reference.py",
)
msl_openems_generic_compare = _load_module(
    "build_msl_openems_sparameter_comparison",
    DIAGNOSTICS_DIR / "build_msl_openems_sparameter_comparison.py",
)
lumped_openems_generic_compare = _load_module(
    "build_lumped_openems_sparameter_comparison",
    DIAGNOSTICS_DIR / "build_lumped_openems_sparameter_comparison.py",
)
lumped_openems_sweep_compare = _load_module(
    "build_lumped_openems_sweep_comparison",
    DIAGNOSTICS_DIR / "build_lumped_openems_sweep_comparison.py",
)
lumped_openems_parallel_plan = _load_module(
    "build_lumped_openems_parallel_plan",
    DIAGNOSTICS_DIR / "build_lumped_openems_parallel_plan.py",
)
coaxial_gap_openems_generic_compare = _load_module(
    "build_coaxial_gap_openems_sparameter_comparison",
    DIAGNOSTICS_DIR / "build_coaxial_gap_openems_sparameter_comparison.py",
)
coaxial_reference_plane_dft_dump = _load_module(
    "generate_coaxial_reference_plane_dft_dump",
    DIAGNOSTICS_DIR / "generate_coaxial_reference_plane_dft_dump.py",
)
coaxial_reference_plane_sweep = _load_module(
    "sweep_coaxial_reference_plane_dft",
    DIAGNOSTICS_DIR / "sweep_coaxial_reference_plane_dft.py",
)
coaxial_tem_signal_path_audit = _load_module(
    "audit_coaxial_tem_signal_path",
    DIAGNOSTICS_DIR / "audit_coaxial_tem_signal_path.py",
)
coaxial_tem_plane_source_prototype = _load_module(
    "prototype_coaxial_tem_plane_source",
    DIAGNOSTICS_DIR / "prototype_coaxial_tem_plane_source.py",
)
floquet_empty_space_analytic_compare = _load_module(
    "build_floquet_empty_space_analytic_comparison",
    DIAGNOSTICS_DIR / "build_floquet_empty_space_analytic_comparison.py",
)
floquet_slab_analytic_compare = _load_module(
    "build_floquet_slab_analytic_comparison",
    DIAGNOSTICS_DIR / "build_floquet_slab_analytic_comparison.py",
)
floquet_periodic_slab_report = _load_module(
    "report_floquet_periodic_slab_oracles",
    DIAGNOSTICS_DIR / "report_floquet_periodic_slab_oracles.py",
)
generalized_planar_quasitem_report = _load_module(
    "report_generalized_planar_quasitem_oracles",
    DIAGNOSTICS_DIR / "report_generalized_planar_quasitem_oracles.py",
)
rf_e5_blocker_ladder_report = _load_module(
    "report_rf_e5_blocker_ladder",
    DIAGNOSTICS_DIR / "report_rf_e5_blocker_ladder.py",
)
waveguide_wr90_generic_compare = _load_module(
    "build_waveguide_wr90_external_sparameter_comparison",
    DIAGNOSTICS_DIR / "build_waveguide_wr90_external_sparameter_comparison.py",
)
patch_wire_openems_generic_compare = _load_module(
    "build_patch_openems_wire_sparameter_comparison",
    DIAGNOSTICS_DIR / "build_patch_openems_wire_sparameter_comparison.py",
)
external_solver_dependency_check = _load_module(
    "check_external_solver_dependencies",
    DIAGNOSTICS_DIR / "check_external_solver_dependencies.py",
)


def _dataset_from_arrays(module, freqs_hz, s_params):
    """Build an in-memory SParameterDataset (no file round-trip)."""
    return module.SParameterDataset(
        path=Path("<memory>"),
        freqs_hz=module.np.asarray(freqs_hz, dtype=float),
        s_params=module.np.asarray(s_params, dtype=module.np.complex128),
        source_format="memory",
    )


def test_required_external_solver_skip_blocks_full_coverage():
    stdout = """
=========================== short test summary info ============================
SKIPPED [1] tests/test_meep_crossval.py:142: Meep is required.
SKIPPED [1] tests/test_meep_crossval.py:174: Meep is required.
SKIPPED [1] tests/test_openems_crossval.py:161: CSXCAD/openEMS is required.
3 skipped in 0.93s
"""
    group = run_gate.gate_groups_by_id()["slow_external_crossval"]

    metadata = run_gate.coverage_metadata(
        group,
        execution_status="passed",
        stdout=stdout,
        vessl_run_ids=["369367237813"],
    )

    assert metadata["coverage_status"] == "blocked"
    assert metadata["required_skip_count"] == 3
    assert metadata["optional_skip_count"] == 0
    assert metadata["strict_xfail_count"] == 0
    assert metadata["vessl_run_ids"] == ["369367237813"]
    assert any(
        claim["evidence_level"] == "E4" and "Meep" in claim["reason"]
        for claim in metadata["blocked_claims"]
    )


def test_strict_xfail_is_visible_as_passed_with_xfails():
    stdout = """
=========================== short test summary info ============================
XFAIL tests/test_msl_port_integration.py::test_msl_thru_line_eigenmode_gate - mode='eigenmode' is planned.
1 passed, 1 xfailed, 18 warnings in 15.22s
"""
    group = run_gate.gate_groups_by_id()["slow_msl"]

    metadata = run_gate.coverage_metadata(
        group,
        execution_status="passed",
        stdout=stdout,
    )

    assert metadata["coverage_status"] == "passed_with_xfails"
    assert metadata["strict_xfail_count"] == 1
    assert any(
        "test_msl_thru_line_eigenmode_gate" in claim["claim"]
        for claim in metadata["blocked_claims"]
    )


def test_aggregate_refuses_required_skip_as_full_claim(tmp_path: Path):
    stdout_path = tmp_path / "slow_external_crossval.stdout.txt"
    stdout_path.write_text(
        "\n".join(
            [
                "=========================== short test summary info ============================",
                "SKIPPED [1] tests/test_meep_crossval.py:142: Meep is required.",
                "SKIPPED [1] tests/test_meep_crossval.py:174: Meep is required.",
                "SKIPPED [1] tests/test_openems_crossval.py:161: CSXCAD/openEMS is required.",
                "3 skipped in 0.93s",
            ]
        ),
        encoding="utf-8",
    )
    stderr_path = tmp_path / "slow_external_crossval.stderr.txt"
    stderr_path.write_text("", encoding="utf-8")
    result_json = tmp_path / "physics_gate_results.json"
    result_json.write_text(
        json.dumps(
            {
                "status": "passed",
                "results": [
                    {
                        "group_id": "slow_external_crossval",
                        "description": "external solver gate",
                        "tests": ["tests/test_meep_crossval.py"],
                        "command": ["python", "-m", "pytest"],
                        "status": "passed",
                        "returncode": 0,
                        "duration_s": 1.0,
                        "stdout_path": str(stdout_path),
                        "stderr_path": str(stderr_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rc = aggregate_gate.main(
        [
            "--result-json",
            str(result_json),
            "--output-dir",
            str(tmp_path / "aggregate"),
            "--require-full-coverage",
        ]
    )

    assert rc == 2
    payload = json.loads(
        (tmp_path / "aggregate" / "physics_gate_aggregate.json").read_text()
    )
    assert payload["status"] == "passed"
    assert payload["coverage_status"] == "blocked"
    assert payload["required_skip_count"] == 3


def test_msl_openems_magnitude_comparison_metrics_pass_for_close_curves():
    """Matched thru-line: |S11| is intrinsically small (~0.12), so the S11 channel is
    NON-discriminating (a degenerate rfx S11 would 'pass' the 0.15 tol). The honest
    status is transmission-only; S11 is informational, not a gate."""
    np = msl_openems_compare.np
    freqs = np.linspace(3.0e9, 4.5e9, 5)
    ref_s11 = np.full(freqs.shape, 0.12 + 0.01j, dtype=complex)
    ref_s21 = np.full(freqs.shape, 0.98 - 0.02j, dtype=complex)
    rfx_s11 = np.full(freqs.shape, 0.14 + 0.01j, dtype=complex)
    rfx_s21 = np.full(freqs.shape, 0.95 - 0.02j, dtype=complex)

    metrics = msl_openems_compare.compare_magnitudes(
        freqs,
        rfx_s11,
        rfx_s21,
        freqs,
        ref_s11,
        ref_s21,
        f_lo_hz=3.0e9,
        f_hi_hz=4.5e9,
    )

    # ref |S11| ~0.12 < 2x tol (0.30) -> S11 channel cannot discriminate -> informational.
    assert metrics["s11_gate_discriminating"] is False
    assert metrics["pass_s11_mean_abs_diff_le_0p15"] is None
    assert metrics["status"] == "passed_transmission_only_s11_nondiscriminating"
    assert metrics["s11_mean_abs_diff"] < 0.15
    assert metrics["s21_mean_abs_diff"] < 0.15


def test_msl_openems_comparison_s11_channel_is_not_vacuous():
    """Validation-honesty lock (2026-06-17): the comparator must NOT report a bare
    'passed' for a degenerate rfx output on a matched line (the bug the reference-
    verification found: |S11| tol 0.15 exceeded the reference |S11| ~0.12, so a
    degenerate rfx S11=0 'passed'). It must flag the S11 channel non-discriminating.
    A genuinely strong-reflector reference (|S11| >= 2x tol) must still gate on S11."""
    np = msl_openems_compare.np
    freqs = np.linspace(3.0e9, 4.5e9, 5)
    ref_s11 = np.full(freqs.shape, 0.12 + 0.01j, dtype=complex)
    ref_s21 = np.full(freqs.shape, 0.98 - 0.02j, dtype=complex)

    # Degenerate rfx (zero reflection physics, ideal transmission): must NOT bare-pass.
    degenerate = msl_openems_compare.compare_magnitudes(
        freqs, np.zeros(freqs.shape, complex), np.ones(freqs.shape, complex),
        freqs, ref_s11, ref_s21, f_lo_hz=3.0e9, f_hi_hz=4.5e9,
    )
    assert degenerate["status"] != "passed"
    assert degenerate["s11_gate_discriminating"] is False

    # Strong reflector (ref |S11| = 0.5 >= 2x tol): the S11 channel IS a real gate,
    # and a mismatched rfx S11 (=0) must FAIL it.
    strong_ref = np.full(freqs.shape, 0.5 + 0.0j, dtype=complex)
    strong_s21 = np.full(freqs.shape, 0.86 + 0.0j, dtype=complex)
    far = msl_openems_compare.compare_magnitudes(
        freqs, np.zeros(freqs.shape, complex), strong_s21,
        freqs, strong_ref, strong_s21, f_lo_hz=3.0e9, f_hi_hz=4.5e9,
    )
    assert far["s11_gate_discriminating"] is True
    assert far["pass_s11_mean_abs_diff_le_0p15"] is False
    assert far["status"] == "failed"


def test_msl_openems_comparison_pass_status_maps_to_success_exit_code():
    """Exit-code regression lock: the transmission-only PASS must map to success.

    The honesty fix added a new success status; an exact ``== "passed"`` exit-code
    check would turn a GREEN M3 VESSL lane RED on a matched line — the exact
    harness false-negative the fix must NOT introduce. ``main()`` keys its exit
    code off ``is_pass_status``.
    """
    assert msl_openems_compare.is_pass_status("passed") is True
    assert (
        msl_openems_compare.is_pass_status(
            "passed_transmission_only_s11_nondiscriminating"
        )
        is True
    )
    assert msl_openems_compare.is_pass_status("failed") is False


def test_waveguide_report_parser_captures_cv11_gates_and_refs():
    parsed = waveguide_report.parse_cv11_stdout(
        "\n".join(
            [
                "[meep-ref] loaded MEEP reference with geometries: ['r4']",
                "[openems-ref] loaded (r4) with geometries: ['empty']",
                "[palace-ref] loaded (r_h2) with geometries: ['empty']",
                "[slab S21] |S|: max_diff=0.0074 mean_diff=0.0030 (gate 0.070)",
                "[slab S21] ∠S: max_diff=0.90° mean_diff=0.36° (gate 60.0°)",
                "[slab S21] |S_rfx−S_ref|: max=0.0156 mean=0.0076 (gate 0.300)",
                "CROSSVAL-11 PASS — all geometries within accept gate.",
            ]
        )
    )

    assert parsed["status"] == "passed"
    assert parsed["loaded_external_references"] == ["MEEP", "OpenEMS", "Palace"]
    assert parsed["metrics"]["slab S21"]["magnitude_mean"] == 0.003
    assert parsed["metrics"]["slab S21"]["phase_gate"] == 60.0
    assert parsed["metrics"]["slab S21"]["complex_max"] == 0.0156


def test_msl_report_parser_captures_notch_demo_gates():
    parsed = msl_report.parse_cv06b_stdout(
        "\n".join(
            [
                "Notch frequency error = 6.2%",
                "Notch depth |S21| = -17.4 dB",
                "Re(Z0) median = 51.8 Ω",
            ]
        ),
        rc=0,
    )

    assert parsed["status"] == "passed"
    assert parsed["metrics"]["notch_frequency_error_pct"] == 6.2
    assert parsed["metrics"]["notch_depth_db"] == -17.4
    assert parsed["metrics"]["z0_median_ohm"] == 51.8


def test_msl_report_infers_legacy_xfail_count_from_stdout(tmp_path: Path):
    stdout = tmp_path / "slow_msl.stdout.txt"
    stdout.write_text(
        "\n".join(
            [
                "=========================== short test summary info ============================",
                "XFAIL tests/test_msl_port_integration.py::test_msl_thru_line_eigenmode_gate - planned",
                "1 passed, 1 xfailed, 18 warnings in 15.22s",
            ]
        ),
        encoding="utf-8",
    )
    result_json = tmp_path / "physics_gate_results.json"

    assert (
        msl_report.infer_strict_xfail_count(
            {"stdout_path": str(stdout), "status": "passed"},
            base_json=result_json,
        )
        == 1
    )


def test_sparameter_claim_audit_expected_family_levels_are_current():
    assert (
        sparameter_claim_audit.EXPECTED_FAMILY_LEVELS["lumped_port"]
        == "E2/E3/E4-partial"
    )
    assert (
        sparameter_claim_audit.EXPECTED_FAMILY_LEVELS["microstrip_line_port"]
        == "E5-narrow/eigenmode-blocked"
    )
    assert (
        sparameter_claim_audit.EXPECTED_FAMILY_LEVELS[
            "rectangular_waveguide_port"
        ]
        == "E5-narrow"
    )
    assert (
        sparameter_claim_audit.EXPECTED_FAMILY_LEVELS["floquet_port"]
        == "E2/E3-modal/slab-analytic/no-promoted-api"
    )
    assert (
        sparameter_claim_audit.EXPECTED_FAMILY_LEVELS["coaxial_port"]
        == "E2-sweep/E3/E4-gap/no-sparam"
    )


def test_floquet_modal_oracle_report_passes_synthetic_te_checks():
    report = floquet_modal_report.evaluate_floquet_modal_oracles()

    assert report["status"] == "passed"
    assert report["claim_scope"].startswith("Floquet/Bloch specular TE")
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["s11_matches_synthetic_gamma"]["status"] == "passed"
    assert checks["s21_matches_synthetic_tau"]["status"] == "passed"
    assert checks["lossless_synthetic_power_balance"]["status"] == "passed"


def test_floquet_periodic_slab_oracle_report_covers_nonempty_slab():
    report = floquet_periodic_slab_report.evaluate_floquet_periodic_slab_oracles()

    assert report["status"] == "passed"
    assert report["evidence_level"] == "E2-analytic"
    assert "no RCWA/external" in report["claim_scope"]
    assert report["case_count"] >= 8
    assert report["row_count"] == report["case_count"] * len(report["freqs_hz"])
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["lossless_power_balance"]["status"] == "passed"
    assert checks["fabry_perot_matches_airy_reflectance"]["status"] == "passed"
    assert checks["air_or_zero_thickness_has_zero_reflection"]["status"] == "passed"
    assert checks["design_frequency_halfwave_has_zero_reflection"]["status"] == "passed"
    assert any(row["eps_r"] > 1.0 and row["thickness_m"] > 0.0 for row in report["rows"])


def test_coaxial_tem_oracle_report_covers_geometry_sweep():
    report = coaxial_tem_report.evaluate_coaxial_tem_oracles()

    assert report["status"] == "passed"
    assert report["evidence_level"] == "E2-calibration-oracle"
    assert report["geometry_sweep_case_count"] >= 4
    assert report["max_sweep_z0_abs_diff"] < 5e-8
    assert report["max_sweep_beta_rel_error"] < 5e-7
    assert report["reference_plane_extractor"]["status"] == "passed"
    assert report["reference_plane_extractor"]["max_s11_abs_diff"] < 3e-7
    assert report["cartesian_plane_adapter"]["status"] == "passed"
    assert report["cartesian_plane_adapter"]["max_s11_abs_diff"] < 2e-3
    case_names = {case["name"] for case in report["geometry_sweep"]}
    assert {"sma_ptfe_default", "compact_high_eps"} <= case_names


def test_coaxial_reference_plane_dft_replay_classifier_blocks_weak_tem_signal():
    payload = coaxial_reference_plane_dft_dump.classify_coaxial_reference_plane_replay(
        plane_s11=np.asarray([4.0 + 0.0j]),
        gap_s11=np.asarray([0.8 + 0.0j]),
        max_abs_voltage=1e-15,
        max_abs_current=1e-16,
    )

    assert payload["status"] == "blocked"
    assert payload["evidence_level"] == "E3-diagnostic-blocked"
    assert "not broad E5" in payload["claim_scope"]
    assert any("voltage below signal floor" in item for item in payload["blockers"])
    assert any("current below signal floor" in item for item in payload["blockers"])


def test_coaxial_reference_plane_dft_replay_classifier_is_not_e5_when_consistent():
    payload = coaxial_reference_plane_dft_dump.classify_coaxial_reference_plane_replay(
        plane_s11=np.asarray([0.8 + 0.01j]),
        gap_s11=np.asarray([0.81 + 0.0j]),
        max_abs_voltage=1e-4,
        max_abs_current=1e-5,
    )

    assert payload["status"] == "passed"
    assert payload["evidence_level"] == "E3-diagnostic-internal-consistency"
    assert not payload["evidence_level"].startswith("E4")
    assert not payload["evidence_level"].startswith("E5")
    assert "not external full-wave evidence" in payload["claim_scope"]


def test_coaxial_reference_plane_sweep_summary_blocks_when_no_plane_passes():
    rows = [
        {
            "plane_index": 20,
            "signal_score": 1e-16,
            "classification": {
                "status": "blocked",
                "max_abs_voltage": 1e-15,
                "max_abs_current": 1e-16,
                "max_plane_gap_s11_abs_diff": 0.6,
            },
        },
        {
            "plane_index": 21,
            "signal_score": 2e-16,
            "classification": {
                "status": "blocked",
                "max_abs_voltage": 2e-15,
                "max_abs_current": 2e-16,
                "max_plane_gap_s11_abs_diff": 0.7,
            },
        },
    ]

    payload = coaxial_reference_plane_sweep.summarize_reference_plane_sweep(rows)

    assert payload["status"] == "blocked"
    assert payload["evidence_level"] == "E3-diagnostic-blocked"
    assert payload["passed_plane_count"] == 0
    assert payload["best_signal_plane_index"] == 21
    assert payload["best_s11_plane_index"] == 20
    assert "not broad E5" in payload["claim_scope"]


def test_coaxial_tem_signal_path_audit_identifies_structural_blockers():
    payload = coaxial_tem_signal_path_audit.audit_default_coaxial_tem_signal_path()

    assert payload["status"] == "blocked"
    assert payload["evidence_level"] == "structural-diagnostic-blocked"
    checks = {check["name"]: check for check in payload["checks"]}
    assert (
        checks["source_component_is_transverse_to_tem_reference_plane"]["status"]
        == "failed"
    )
    assert (
        checks["source_component_is_transverse_to_tem_reference_plane"][
            "source_component"
        ]
        == checks["source_component_is_transverse_to_tem_reference_plane"][
            "normal_component"
        ]
    )
    assert checks["outer_conductor_shell_has_pec_cells"]["status"] == "passed"
    assert checks["outer_conductor_shell_has_pec_cells"]["shell_pec_cell_count"] > 0
    assert checks["dielectric_annulus_is_present"]["status"] == "passed"
    assert "not broad E5" in payload["claim_scope"]


def test_coaxial_tem_plane_source_prototype_classifier_is_not_e5():
    payload = (
        coaxial_tem_plane_source_prototype.classify_tem_plane_source_prototype(
            s11=np.asarray([0.1 + 0.0j, -0.2 + 0.05j]),
            max_abs_voltage=1e-6,
            max_abs_current=1e-8,
        )
    )

    assert payload["status"] == "passed"
    assert payload["evidence_level"] == "E3-diagnostic-prototype"
    assert not payload["evidence_level"].startswith("E4")
    assert not payload["evidence_level"].startswith("E5")
    assert "not the public add_coaxial_port API" in payload["claim_scope"]
    assert "not broad E5" in payload["claim_scope"]


def test_coaxial_tem_plane_source_prototype_classifier_blocks_weak_signal():
    payload = (
        coaxial_tem_plane_source_prototype.classify_tem_plane_source_prototype(
            s11=np.asarray([0.1 + 0.0j]),
            max_abs_voltage=1e-15,
            max_abs_current=1e-16,
        )
    )

    assert payload["status"] == "blocked"
    assert payload["evidence_level"] == "E3-diagnostic-prototype-blocked"
    assert any("voltage below signal floor" in item for item in payload["blockers"])
    assert any("current below signal floor" in item for item in payload["blockers"])


def test_vessl_shard_checker_can_require_full_coverage(tmp_path: Path):
    yaml_path = tmp_path / "slow_external_crossval.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "run: |-",
                "  python scripts/diagnostics/run_physics_gate.py",
                "  --group slow_external_crossval",
                "  RFX_PHYSICS_GATE_OUTPUT_ROOT=.omx/physics-gate/tmp",
                "  --vessl-run-id slow_external_crossval=unit-test",
            ]
        ),
        encoding="utf-8",
    )
    stdout_path = tmp_path / "slow_external_crossval.stdout.txt"
    stdout_path.write_text(
        "\n".join(
            [
                "=========================== short test summary info ============================",
                "SKIPPED [1] tests/test_meep_crossval.py:142: Meep is required for this external cross-validation gate.",
                "1 skipped in 0.93s",
            ]
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "physics_gate_results.json"
    result_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "group_id": "slow_external_crossval",
                        "status": "passed",
                        "stdout_path": str(stdout_path),
                        "stderr_path": str(tmp_path / "stderr.txt"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "shards": [
                    {
                        "group_id": "slow_external_crossval",
                        "yaml_path": str(yaml_path),
                        "result_json": str(result_path),
                        "last_verified_run_id": "unit-test",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rc = vessl_shard_check.main(
        [
            "--manifest",
            str(manifest_path),
            "--verify-results",
            "--require-full-coverage",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert rc == 2
    payload = json.loads(
        (tmp_path / "out" / "vessl_physics_shard_check.json").read_text()
    )
    assert payload["status"] == "passed"
    assert payload["coverage_status"] == "blocked"
    assert payload["require_full_coverage"] is True


def test_port_external_reference_audit_blocks_until_every_family_has_broad_e5(tmp_path: Path):
    audit = port_external_reference_check.build_external_reference_audit(
        REPO_ROOT / "scripts" / "diagnostics" / "port_external_reference_requirements.json"
    )

    # schema_status is "failed" on any clean checkout because it requires
    # missing_artifact_count == 0, and every referenced comparison/envelope
    # artifact lives under .omx/physics-gate/ — VESSL job outputs that are
    # gitignored (.gitignore `.omx/`) and never committed. The audit logic
    # in check_port_external_references.py is byte-identical to its c7071d7
    # introduction; only the committed requirements JSON has advanced. The
    # surface and vessl-yaml contracts (which depend on committed files) do
    # pass, and the overall gate stays correctly "blocked".
    assert audit["schema_status"] == "failed"
    assert audit["surface_coverage_status"] == "passed"
    assert audit["vessl_yaml_contract_status"] == "passed"
    assert audit["vessl_yaml_contract_launchable_family_count"] == 7
    assert audit["vessl_yaml_contract_diagnostic_command_family_count"] == 7
    assert audit["comparison_artifact_coverage_status"] == "blocked"
    # broad-E5 envelope coverage stays "blocked" on a clean checkout: although
    # rectangular_waveguide_port's envelopes are now committed (see below), the
    # other required families still lack a committed broad-E5 envelope.
    assert audit["broad_e5_envelope_artifact_coverage_status"] == "blocked"
    assert audit["required_surface_family_count"] == 7
    assert audit["missing_manifest_family_count"] == 0
    # TWO families now have a committed passing external comparison on a clean
    # checkout: rectangular_waveguide_port (rfx-vs-Palace FEM, PR #181) and
    # coaxial_port (rfx-vs-Meep power-flux broad-E4, VESSL 369367245835 recommit
    # 2026-07-04, tests/fixtures/coax_broad_e4/). So 2 families have a passed
    # comparison + passed broad-E4 artifact; the other 5 required families are
    # still missing a passed comparison artifact on a clean (no-.omx/) checkout.
    # coaxial_port still stays `incomplete` below — its ad_fd_test is null (no
    # AD-traceable extractor), so committing the E4 comparison does not flip it.
    assert audit["missing_passed_comparison_artifact_count"] == 5
    assert audit["passed_comparison_artifact_count"] == 2
    assert audit["passed_broad_e4_comparison_artifact_count"] == 2
    # No family is now MISSING a broad-E4/E5 artifact. rectangular_waveguide_port
    # is the only current_status=broad_e5_passed family and its broad-E4 + broad-E5
    # artifacts ARE committed (tests/fixtures/, PR #181). coaxial_port was
    # downgraded from broad_e5_passed -> broad_e5_demonstrated_evidence_uncommitted
    # (validation-framework honesty pass): the clean-checkout audit reports it
    # BLOCKED, so it no longer CLAIMS broad-E5 and therefore no longer "owes" a
    # broad-E4/E5 artifact. Its broad-E5 envelope (PR #256) AND broad-E4 Meep
    # comparison (PR #259) are now committed, so coaxial_port has DROPPED OUT of
    # the missing-passed-comparison set (count 5 above) — yet it stays
    # `incomplete` below on the null ad_fd_test alone.
    assert audit["missing_broad_e4_comparison_artifact_count"] == 0
    assert sorted(audit["missing_broad_e4_comparison_artifact_families"]) == []
    assert audit["missing_broad_e5_envelope_artifact_count"] == 0
    # 6 = the 5 rectangular-waveguide band envelopes (PR #181) + the coaxial
    # line envelope committed to tests/fixtures/coax_broad_e5/ (PR #256 —
    # regenerated vs analytic TL with a machine-readable envelope_summary).
    # coaxial_port still stays `incomplete` below: with both its broad-E5
    # envelope AND broad-E4 Meep comparison now committed, the SOLE remaining
    # blocker is its null ad_fd_test (no AD-traceable reflection extractor).
    assert audit["passed_broad_e5_envelope_artifact_count"] == 6
    assert audit["status"] == "blocked"
    incomplete = {item["family"]: item for item in audit["incomplete"]}
    assert "lumped_port" in incomplete
    assert "wire_port" in incomplete
    assert "coaxial_port" in incomplete
    assert "floquet_port" in incomplete
    assert "update_goal" in audit["completion_decision"]

    rc = port_external_reference_check.main(
        [
            "--output-dir",
            str(tmp_path / "external-reference-audit"),
            "--require-complete",
        ]
    )
    # main() returns 1 (schema invalid) rather than 2 (complete-gate failure)
    # on a clean checkout, because schema_status is "failed" — the referenced
    # .omx/ artifacts are gitignored and absent. Both are non-zero "gate is not
    # green" exit codes; the schema-invalid check fires first.
    assert rc == 1


def test_port_external_reference_audit_requires_support_matrix_coverage(tmp_path: Path):
    support_matrix = tmp_path / "sparameter_support_matrix.json"
    support_matrix.write_text(
        json.dumps(
            {
                "port_families": [
                    {
                        "family": "lumped_port",
                        "primitive": "add_port(extent=None)",
                        "is_port": True,
                        "evidence_level": "E2/E3-limited",
                    },
                    {
                        "family": "soft_current_source",
                        "primitive": "add_source(...)",
                        "is_port": False,
                        "evidence_level": "not_a_port",
                    },
                ],
                "future_port_families": [
                    {
                        "family": "generalized_planar_ports",
                        "status": "planned_not_promoted",
                        "planned_primitives": ["stripline"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "port_external_reference_requirements.json"
    manifest.write_text(
        json.dumps(
            {
                "purpose": "unit test",
                "completion_rule": "all required families must be tracked",
                "requirements": [
                    {
                        "family": "lumped_port",
                        "required_for_e5": True,
                        "required_scope": "broad_e5",
                        "current_status": "blocked_no_external_reference",
                        "missing_evidence": ["external reference missing"],
                        "existing_artifacts": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    audit = port_external_reference_check.build_external_reference_audit(
        manifest,
        support_matrix,
    )

    assert audit["surface_coverage_status"] == "failed"
    assert audit["schema_status"] == "failed"
    assert audit["missing_manifest_family_count"] == 1
    assert audit["missing_manifest_families"] == ["generalized_planar_ports"]


def test_port_external_reference_audit_requires_passed_comparison_for_broad_e5(
    tmp_path: Path,
):
    support_matrix = tmp_path / "sparameter_support_matrix.json"
    support_matrix.write_text(
        json.dumps(
            {
                "port_families": [
                    {
                        "family": "lumped_port",
                        "primitive": "add_port(extent=None)",
                        "is_port": True,
                        "evidence_level": "E5",
                    }
                ],
                "future_port_families": [],
            }
        ),
        encoding="utf-8",
    )
    comparison = tmp_path / "lumped_external_comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "status": "passed",
                "evidence_level": "E4-enabling",
                "claim_scope": "narrow external S-parameter comparison",
            }
        ),
        encoding="utf-8",
    )
    broad_comparison = tmp_path / "lumped_broad_external_comparison.json"
    broad_comparison.write_text(
        json.dumps(
            {
                "status": "passed",
                "evidence_level": "E4",
                "claim_scope": "broad external S-parameter comparison",
                "summary": {
                    "geometry_count": 3,
                    "pair_count": 5,
                    "passed_pair_count": 5,
                    "failed_pair_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    narrow_envelope = tmp_path / "lumped_narrow_envelope.json"
    narrow_envelope.write_text(
        json.dumps(
            {
                "status": "passed",
                "evidence_level": "E5-narrow",
                "claim_scope": "narrow fixture only",
            }
        ),
        encoding="utf-8",
    )
    broad_envelope = tmp_path / "lumped_broad_envelope.json"
    broad_envelope.write_text(
        json.dumps(
            {
                "status": "passed",
                "evidence_level": "E5",
                "required_scope": "broad_e5",
                "claim_scope": "broad mesh/frequency/geometry envelope",
                "max_mag_abs_tol": 0.05,
                "envelope_summary": {
                    "case_count": 4,
                    "passed_case_count": 4,
                    "dx_values_m": [5e-5, 1e-4],
                    "eps_r_values": [2.0, 4.0],
                    "geometries": ["slab"],
                    "freq_range_hz": [1.0e10, 1.5e10],
                    "max_mag_abs_diff_across_cases": 0.02,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "port_external_reference_requirements.json"

    base_entry = {
        "family": "lumped_port",
        "required_for_e5": True,
        "required_scope": "broad_e5",
        "current_status": "broad_e5_passed",
        "ad_fd_test": "tests/test_waveguide_flux_ad.py::test_flux_smatrix_grad_finite_and_fd_consistent",
        "missing_evidence": [],
        "existing_artifacts": [],
        "existing_vessl_yamls": [],
    }
    manifest.write_text(
        json.dumps({"requirements": [{**base_entry, "external_comparison_artifacts": []}]}),
        encoding="utf-8",
    )
    blocked_audit = port_external_reference_check.build_external_reference_audit(
        manifest,
        support_matrix,
    )
    assert blocked_audit["status"] == "blocked"
    assert blocked_audit["missing_passed_comparison_artifact_count"] == 1
    assert blocked_audit["missing_broad_e5_envelope_artifact_count"] == 1

    yaml_path = tmp_path / "physics_gate_port_external_lumped.yaml"
    yaml_path.write_text("name: unit\n", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "requirements": [
                    {
                        **base_entry,
                        "existing_vessl_yamls": [str(yaml_path)],
                        "external_comparison_artifacts": [str(comparison)],
                        "broad_e5_envelope_artifacts": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    comparison_only_audit = port_external_reference_check.build_external_reference_audit(
        manifest,
        support_matrix,
    )
    assert comparison_only_audit["status"] == "blocked"
    assert comparison_only_audit["missing_passed_comparison_artifact_count"] == 0
    assert comparison_only_audit["missing_broad_e4_comparison_artifact_count"] == 1
    assert comparison_only_audit["missing_broad_e5_envelope_artifact_count"] == 1

    manifest.write_text(
        json.dumps(
            {
                "requirements": [
                    {
                        **base_entry,
                        "existing_vessl_yamls": [str(yaml_path)],
                        "external_comparison_artifacts": [str(comparison)],
                        "broad_e5_envelope_artifacts": [str(narrow_envelope)],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    narrow_audit = port_external_reference_check.build_external_reference_audit(
        manifest,
        support_matrix,
    )
    assert narrow_audit["status"] == "blocked"
    assert narrow_audit["missing_broad_e4_comparison_artifact_count"] == 1
    assert narrow_audit["missing_broad_e5_envelope_artifact_count"] == 1
    assert narrow_audit["failed_broad_e5_envelope_artifact_count"] == 1

    manifest.write_text(
        json.dumps(
            {
                "requirements": [
                    {
                        **base_entry,
                        "existing_vessl_yamls": [str(yaml_path)],
                        "external_comparison_artifacts": [str(comparison)],
                        "broad_e5_envelope_artifacts": [str(broad_envelope)],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    passed_audit = port_external_reference_check.build_external_reference_audit(
        manifest,
        support_matrix,
    )
    assert passed_audit["status"] == "blocked"
    assert passed_audit["passed_comparison_artifact_count"] == 1
    assert passed_audit["passed_broad_e4_comparison_artifact_count"] == 0
    assert passed_audit["missing_broad_e4_comparison_artifact_count"] == 1
    assert passed_audit["missing_broad_e5_envelope_artifact_count"] == 0

    manifest.write_text(
        json.dumps(
            {
                "requirements": [
                    {
                        **base_entry,
                        "existing_vessl_yamls": [str(yaml_path)],
                        "external_comparison_artifacts": [str(comparison), str(broad_comparison)],
                        "broad_e5_envelope_artifacts": [str(broad_envelope)],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    passed_audit = port_external_reference_check.build_external_reference_audit(
        manifest,
        support_matrix,
    )
    assert passed_audit["status"] == "passed"
    assert passed_audit["comparison_artifact_coverage_status"] == "passed"
    assert passed_audit["broad_e5_envelope_artifact_coverage_status"] == "passed"
    assert passed_audit["passed_comparison_artifact_count"] == 2
    assert passed_audit["passed_broad_e4_comparison_artifact_count"] == 1
    assert passed_audit["missing_broad_e4_comparison_artifact_count"] == 0
    assert passed_audit["missing_broad_e5_envelope_artifact_count"] == 0
    assert passed_audit["passed_broad_e5_envelope_artifact_count"] == 1


def test_port_external_reference_shard_requires_comparison_and_envelope(
    tmp_path: Path,
):
    support_matrix = tmp_path / "sparameter_support_matrix.json"
    support_matrix.write_text(
        json.dumps(
            {
                "port_families": [
                    {
                        "family": "lumped_port",
                        "primitive": "add_port(extent=None)",
                        "is_port": True,
                        "evidence_level": "E5",
                    }
                ],
                "future_port_families": [],
            }
        ),
        encoding="utf-8",
    )
    comparison = tmp_path / "lumped_external_comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "status": "passed",
                "evidence_level": "E4",
                "claim_scope": "broad external S-parameter comparison",
                "summary": {
                    "geometry_count": 3,
                    "pair_count": 5,
                    "passed_pair_count": 5,
                    "failed_pair_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    envelope = tmp_path / "lumped_broad_envelope.json"
    envelope.write_text(
        json.dumps(
            {
                "status": "passed",
                "evidence_level": "E5",
                "required_scope": "broad_e5",
                "claim_scope": "broad mesh/frequency/geometry envelope",
                "max_mag_abs_tol": 0.05,
                "envelope_summary": {
                    "case_count": 4,
                    "passed_case_count": 4,
                    "dx_values_m": [5e-5, 1e-4],
                    "eps_r_values": [2.0, 4.0],
                    "geometries": ["slab"],
                    "freq_range_hz": [1.0e10, 1.5e10],
                    "max_mag_abs_diff_across_cases": 0.02,
                },
            }
        ),
        encoding="utf-8",
    )
    yaml_path = tmp_path / "physics_gate_port_external_lumped.yaml"
    yaml_path.write_text("name: unit\n", encoding="utf-8")
    manifest = tmp_path / "port_external_reference_requirements.json"
    base_entry = {
        "family": "lumped_port",
        "required_for_e5": True,
        "required_scope": "broad_e5",
        "current_status": "broad_e5_passed",
        "ad_fd_test": "tests/test_waveguide_flux_ad.py::test_flux_smatrix_grad_finite_and_fd_consistent",
        "missing_evidence": [],
        "existing_artifacts": [],
        "existing_vessl_yamls": [str(yaml_path)],
        "external_comparison_artifacts": [str(comparison)],
    }
    manifest.write_text(
        json.dumps({"requirements": [{**base_entry, "broad_e5_envelope_artifacts": []}]}),
        encoding="utf-8",
    )

    blocked = port_external_reference_shard.build_family_reference_shard(
        "lumped_port",
        manifest,
        support_matrix,
    )
    assert blocked["status"] == "blocked"
    assert blocked["passed_comparison_artifact_count"] == 1
    assert blocked["passed_broad_e4_comparison_artifact_count"] == 1
    assert blocked["passed_broad_e5_envelope_artifact_count"] == 0

    manifest.write_text(
        json.dumps(
            {
                "requirements": [
                    {
                        **base_entry,
                        "broad_e5_envelope_artifacts": [str(envelope)],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    passed = port_external_reference_shard.build_family_reference_shard(
        "lumped_port",
        manifest,
        support_matrix,
    )
    assert passed["status"] == "passed"
    assert passed["passed_comparison_artifact_count"] == 1
    assert passed["passed_broad_e4_comparison_artifact_count"] == 1
    assert passed["passed_broad_e5_envelope_artifact_count"] == 1


def test_port_external_reference_shard_report_blocks_incomplete_family(
    tmp_path: Path,
):
    # lumped_port is the genuinely-incomplete example (switched from coaxial_port,
    # which PR #130 promoted to broad_e5_passed). lumped_port is still
    # narrow_external_reference_broad_blocked — a clean "incomplete family blocks"
    # case independent of the clean-checkout .omx contract.
    payload = port_external_reference_shard.build_family_reference_shard(
        "lumped_port",
        REPO_ROOT / "scripts" / "diagnostics" / "port_external_reference_requirements.json",
        REPO_ROOT / "docs" / "guides" / "sparameter_support_matrix.json",
    )

    assert payload["family"] == "lumped_port"
    assert payload["status"] == "blocked"
    assert payload["current_status"] == "narrow_external_reference_broad_blocked"
    assert "broad E5" in payload["completion_decision"]

    rc = port_external_reference_shard.main(
        [
            "--family",
            "lumped_port",
            "--output-dir",
            str(tmp_path / "lumped"),
            "--require-complete",
        ]
    )
    assert rc == 2
    assert (
        tmp_path
        / "lumped"
        / "lumped_port_external_reference_shard.json"
    ).exists()


def test_port_external_shard_result_audit_requires_present_passed_results(
    tmp_path: Path,
):
    support_matrix = tmp_path / "sparameter_support_matrix.json"
    support_matrix.write_text(
        json.dumps(
            {
                "port_families": [
                    {
                        "family": "lumped_port",
                        "primitive": "add_port(extent=None)",
                        "is_port": True,
                        "evidence_level": "E2/E3-limited",
                    }
                ],
                "future_port_families": [],
            }
        ),
        encoding="utf-8",
    )
    yaml_path = tmp_path / "physics_gate_port_external_lumped.yaml"
    yaml_path.write_text("name: unit\nrun: echo unit\n", encoding="utf-8")
    manifest = tmp_path / "port_external_reference_requirements.json"
    manifest.write_text(
        json.dumps(
            {
                "purpose": "unit test",
                "completion_rule": "unit test",
                "requirements": [
                    {
                        "family": "lumped_port",
                        "primitive": "add_port(extent=None)",
                        "required_for_e5": True,
                        "required_scope": "broad_e5",
                        "current_status": "blocked_no_external_reference",
                        "recommended_vessl_shard_id": "port_external_lumped",
                        "recommended_reference": "unit external solver",
                        "existing_artifacts": [],
                        "existing_vessl_yamls": [str(yaml_path)],
                        "missing_evidence": ["external reference missing"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result_root = tmp_path / "results"

    missing_audit = port_external_shard_result_check.build_shard_result_audit(
        manifest,
        support_matrix,
        result_root,
        "shard_id",
    )

    assert missing_audit["result_coverage_status"] == "failed"
    assert missing_audit["missing_result_count"] == 1

    result_dir = result_root / "port_external_lumped"
    result_dir.mkdir(parents=True)
    (result_dir / "lumped_port_external_reference_shard.json").write_text(
        json.dumps(
            {
                "family": "lumped_port",
                "recommended_vessl_shard_id": "port_external_lumped",
                "status": "blocked",
                "blockers": ["external reference missing"],
            }
        ),
        encoding="utf-8",
    )

    blocked_audit = port_external_shard_result_check.build_shard_result_audit(
        manifest,
        support_matrix,
        result_root,
        "shard_id",
    )

    assert blocked_audit["result_coverage_status"] == "passed"
    assert blocked_audit["status"] == "blocked"
    assert blocked_audit["blocked_family_count"] == 1
    assert "update_goal" in blocked_audit["completion_decision"]

    rc = port_external_shard_result_check.main(
        [
            "--manifest",
            str(manifest),
            "--support-matrix",
            str(support_matrix),
            "--result-root",
            str(result_root),
            "--output-dir",
            str(tmp_path / "out"),
            "--require-complete",
        ]
    )
    assert rc == 2


def test_port_external_shard_execution_manifest_covers_all_required_families():
    manifest = port_external_shard_execution_manifest.build_execution_manifest(
        REPO_ROOT / "scripts" / "diagnostics" / "port_external_reference_requirements.json"
    )

    assert manifest["status"] == "passed"
    assert manifest["required_family_count"] == 7
    assert manifest["launchable_family_count"] == 7
    assert manifest["diagnostic_command_family_count"] == 7
    assert manifest["missing_diagnostic_command_families"] == []
    for row in manifest["shards"]:
        assert row["has_launchable_yaml"] is True
        assert row["expected_result_json"].endswith(
            f"/{row['recommended_vessl_shard_id']}/{row['family']}_external_reference_shard.json"
        )
        for yaml_check in row["yaml_checks"]:
            # Some listed yaml_checks reference scripts/vessl_*.yaml, which are
            # gitignored by policy (.gitignore `**/vessl*.yaml`) and therefore
            # absent on any clean checkout. The manifest already records them as
            # status="missing" and does not count them toward has_launchable_yaml,
            # so skip the on-disk content checks for files that are not present.
            yaml_path = REPO_ROOT / yaml_check["yaml_path"]
            if not yaml_path.exists():
                assert yaml_check["status"] == "missing"
                continue
            yaml_text = yaml_path.read_text(encoding="utf-8")
            assert "pip install -q -e" not in yaml_text
            assert "RFX_REPO_ROOT" in yaml_text
            run_body = port_external_shard_execution_manifest._load_yaml(
                yaml_path
            )["run"]
            subprocess.run(["bash", "-n"], input=run_body, text=True, check=True)
            if "check_external_solver_dependencies.py" in run_body:
                assert "dependency_audit_rc=$?" in run_body
                assert "external_solver_dependency_audit_rc=" in run_body
            if row["family"] in {"lumped_port", "wire_port", "coaxial_port"}:
                assert "apt-get install -y -qq openems python3-openems" in yaml_text
                assert "RFX_SYSTEM_SITE_DIR" in yaml_text


def test_generic_sparameter_reference_comparator_passes_npz_fixture(tmp_path: Path):
    np = sparameter_reference_compare.np
    freqs_hz = np.asarray([1.0e9, 2.0e9, 3.0e9])
    reference_s = np.zeros((2, 2, 3), dtype=np.complex128)
    reference_s[0, 0, :] = [0.1 + 0.02j, 0.12 + 0.01j, 0.13 + 0.0j]
    reference_s[1, 0, :] = [0.8 - 0.1j, 0.82 - 0.08j, 0.85 - 0.05j]
    reference_s[0, 1, :] = reference_s[1, 0, :]
    reference_s[1, 1, :] = reference_s[0, 0, :]
    candidate_s = reference_s.copy()
    candidate_s[1, 0, :] += 1.0e-3 + 2.0e-3j

    reference_path = tmp_path / "reference.npz"
    candidate_path = tmp_path / "candidate.npz"
    np.savez(reference_path, freqs_hz=freqs_hz, s_params=reference_s)
    np.savez(candidate_path, freqs_hz=freqs_hz, s_params=candidate_s)

    candidate = sparameter_reference_compare.load_sparameter_dataset(candidate_path)
    reference = sparameter_reference_compare.load_sparameter_dataset(reference_path)
    payload = sparameter_reference_compare.compare_sparameter_datasets(
        candidate,
        reference,
        terms="S11,S21",
        max_abs_tol=5e-3,
        mean_abs_tol=5e-3,
        max_mag_abs_tol=5e-3,
        mean_mag_abs_tol=5e-3,
    )

    assert payload["status"] == "passed"
    assert payload["evidence_level"] == "E4-enabling"
    assert payload["comparison_mode"] == "complex"
    assert payload["terms"] == ["S11", "S21"]
    assert payload["summary"]["max_abs_diff"] < 3e-3

    rc = sparameter_reference_compare.main(
        [
            "--candidate",
            str(candidate_path),
            "--reference",
            str(reference_path),
            "--output-json",
            str(tmp_path / "comparison.json"),
            "--terms",
            "S11,S21",
            "--max-abs-tol",
            "0.005",
            "--mean-abs-tol",
            "0.005",
            "--max-mag-abs-tol",
            "0.005",
            "--mean-mag-abs-tol",
            "0.005",
        ]
    )
    assert rc == 0
    saved = json.loads((tmp_path / "comparison.json").read_text())
    assert saved["status"] == "passed"


def test_phase_gate_catches_phase_only_error_that_magnitude_mode_passes():
    """T2.1: opt-in phase gate closes the audit's 'phase computed but not gated' gap.

    A candidate with IDENTICAL magnitudes but a rotated phase passes magnitude
    mode (legacy) and stays passing when phase is ungated — but fails once a
    tight `max_phase_abs_tol_rad` is supplied.
    """
    np = sparameter_reference_compare.np
    freqs_hz = np.asarray([3.0e9, 4.0e9, 5.0e9])
    ref_s = np.zeros((2, 2, 3), dtype=np.complex128)
    ref_s[0, 0, :] = 0.5 + 0.0j
    ref_s[1, 0, :] = 0.5 + 0.0j
    ref_s[0, 1, :] = ref_s[1, 0, :]
    ref_s[1, 1, :] = ref_s[0, 0, :]
    # Candidate: same |S| but rotated 90° -> phase diff = pi/2 ~ 1.571 rad.
    cand_s = ref_s * np.exp(1j * (np.pi / 2))

    ref = _dataset_from_arrays(sparameter_reference_compare, freqs_hz, ref_s)
    cand = _dataset_from_arrays(sparameter_reference_compare, freqs_hz, cand_s)

    # Magnitude mode, phase UNGATED (legacy): magnitudes match -> passes.
    legacy = sparameter_reference_compare.compare_sparameter_datasets(
        cand, ref, terms="S11", comparison_mode="magnitude",
    )
    assert legacy["status"] == "passed"
    # The phase diff was computed all along (the audit's point) ...
    assert legacy["metrics_by_term"][0]["max_phase_abs_diff_rad"] > 1.5
    assert legacy["metrics_by_term"][0]["phase_passed"] is True  # ungated

    # Same comparison, phase GATED tight -> the rotation now fails.
    gated = sparameter_reference_compare.compare_sparameter_datasets(
        cand, ref, terms="S11", comparison_mode="magnitude",
        max_phase_abs_tol_rad=0.1,
    )
    assert gated["status"] == "failed"
    assert gated["metrics_by_term"][0]["phase_passed"] is False
    assert gated["tolerances"]["max_phase_abs_tol_rad"] == 0.1


def test_phase_gate_fails_closed_when_no_bin_clears_mask():
    """T2.1: requesting a phase gate that cannot be evaluated must FAIL, not pass.

    When every |S| bin is below `min_phase_mag` the phase diff is unverifiable
    (max_phase_abs_diff_rad is None); fail-closed mirrors the T1 numeric-breadth
    philosophy (absent evidence != pass).
    """
    np = sparameter_reference_compare.np
    freqs_hz = np.asarray([3.0e9, 4.0e9])
    ref_s = np.full((2, 2, 2), 1.0e-9, dtype=np.complex128)
    cand_s = ref_s.copy()
    ref = _dataset_from_arrays(sparameter_reference_compare, freqs_hz, ref_s)
    cand = _dataset_from_arrays(sparameter_reference_compare, freqs_hz, cand_s)

    payload = sparameter_reference_compare.compare_sparameter_datasets(
        cand, ref, terms="S11", comparison_mode="magnitude",
        min_phase_mag=1e-6, max_phase_abs_tol_rad=1.0,
    )
    assert payload["metrics_by_term"][0]["max_phase_abs_diff_rad"] is None
    assert payload["metrics_by_term"][0]["phase_passed"] is False
    assert payload["status"] == "failed"


def test_msl_openems_generic_builder_uses_magnitude_mode(tmp_path: Path):
    reference = tmp_path / "msl_openems_reference.json"
    reference.write_text(
        json.dumps(
            {
                "freqs_ghz": [3.0, 4.0],
                "s11": [[0.1, 0.0], [0.12, 0.0]],
                "s21": [[0.8, 0.0], [0.82, 0.0]],
            }
        ),
        encoding="utf-8",
    )
    rfx_comparison = tmp_path / "msl_rfx_comparison.json"
    rfx_comparison.write_text(
        json.dumps(
            {
                "rfx_freqs_hz": [3.0e9, 4.0e9],
                # Same magnitudes, deliberately different phase.
                "rfx_s11": [[0.0, 0.1], [0.0, 0.12]],
                "rfx_s21": [[0.0, 0.8], [0.0, 0.82]],
            }
        ),
        encoding="utf-8",
    )

    payload = msl_openems_generic_compare.build_msl_openems_generic_comparison(
        reference,
        rfx_comparison,
        tmp_path / "out",
    )

    assert payload["status"] == "passed"
    assert payload["comparison_mode"] == "magnitude"
    assert payload["summary"]["max_mag_abs_diff"] == 0.0


def test_lumped_openems_generic_builder_uses_magnitude_mode(tmp_path: Path):
    freqs = np.asarray([0.8e9, 1.0e9, 1.2e9], dtype=float)
    rfx_s = np.zeros((2, 2, freqs.size), dtype=np.complex128)
    ref_s = np.zeros_like(rfx_s)
    # Same magnitudes, deliberately different phase.
    rfx_s[0, 0, :] = [0.0 + 0.9j, 0.0 + 0.8j, 0.0 + 0.7j]
    rfx_s[1, 0, :] = [0.0 + 0.02j, 0.0 + 0.03j, 0.0 + 0.04j]
    ref_s[0, 0, :] = [0.9 + 0.0j, 0.8 + 0.0j, 0.7 + 0.0j]
    ref_s[1, 0, :] = [0.02 + 0.0j, 0.03 + 0.0j, 0.04 + 0.0j]

    payload = lumped_openems_generic_compare.build_lumped_openems_comparison_from_sparams(
        freqs_hz=freqs,
        rfx_sparams=rfx_s,
        openems_sparams=ref_s,
        output_dir=tmp_path / "out",
    )

    assert payload["status"] == "passed"
    assert payload["comparison_mode"] == "magnitude"
    assert payload["claim_scope"].startswith("narrow two-port PEC-cavity")
    assert payload["summary"]["max_mag_abs_diff"] == 0.0


def test_lumped_openems_sweep_summary_requires_all_cases_pass():
    passed_case = {
        "status": "passed",
        "summary": {"max_mag_abs_diff": 0.10, "mean_mag_abs_diff": 0.04},
    }
    failed_case = {
        "status": "failed",
        "summary": {"max_mag_abs_diff": 0.20, "mean_mag_abs_diff": 0.08},
    }

    passed = lumped_openems_sweep_compare.summarize_lumped_openems_sweep(
        [passed_case, {**passed_case, "summary": {"max_mag_abs_diff": 0.11, "mean_mag_abs_diff": 0.05}}],
    )
    blocked = lumped_openems_sweep_compare.summarize_lumped_openems_sweep(
        [passed_case, failed_case],
    )

    assert passed["status"] == "passed"
    assert passed["evidence_level"] == "E4-enabling"
    assert passed["passed_case_count"] == 2
    assert passed["max_case_max_mag_abs_diff"] == 0.11
    assert "not broad calibrated lumped-port E5" in passed["claim_scope"]
    assert blocked["status"] == "failed"
    assert blocked["passed_case_count"] == 1


def test_lumped_openems_sweep_can_aggregate_parallel_case_artifacts(tmp_path: Path):
    cases = (
        {
            "case_name": "case_a",
            "port1_pos_m": "0.010,0.010,0.005",
            "port2_pos_m": "0.020,0.010,0.005",
        },
        {
            "case_name": "case_b",
            "port1_pos_m": "0.010,0.005,0.005",
            "port2_pos_m": "0.020,0.015,0.005",
        },
    )
    root = tmp_path / "case-artifacts"
    for idx, case in enumerate(cases):
        case_dir = root / case["case_name"]
        case_dir.mkdir(parents=True)
        (case_dir / "lumped_openems_generic_sparameter_comparison.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "summary": {
                        "max_mag_abs_diff": 0.10 + 0.01 * idx,
                        "mean_mag_abs_diff": 0.04 + 0.01 * idx,
                    },
                }
            ),
            encoding="utf-8",
        )

    payload = (
        lumped_openems_sweep_compare
        .build_lumped_openems_sweep_comparison_from_artifacts(
            case_artifact_root=root,
            output_dir=tmp_path / "out",
            cases=cases,
        )
    )

    assert payload["status"] == "passed"
    assert payload["execution_mode"] == "parallel_artifact_aggregation"
    assert payload["passed_case_count"] == 2
    assert payload["max_case_max_mag_abs_diff"] == 0.11
    assert payload["case_payloads"][0]["parallel_artifact_replay"] is True


def test_lumped_openems_parallel_plan_writes_case_yamls(tmp_path: Path):
    payload = lumped_openems_parallel_plan.build_lumped_openems_parallel_plan(
        output_dir=tmp_path / "plan",
        output_root=".omx/physics-gate/unit-lumped-openems-parallel",
    )

    assert payload["status"] == "passed"
    assert payload["evidence_level"] == "orchestration-only"
    assert payload["case_count"] == len(
        lumped_openems_sweep_compare.DEFAULT_SWEEP_CASES
    )
    assert "build_lumped_openems_sweep_comparison.py" in payload["aggregation_command"]
    for row in payload["cases"]:
        yaml_path = REPO_ROOT / row["yaml_path"]
        assert yaml_path.exists()
        text = yaml_path.read_text(encoding="utf-8")
        assert row["case_name"] in text
        assert "apt-get install -y -qq openems python3-openems" in text
        assert "pip install -q -e" not in text
        assert "RFX_SITECUSTOMIZE_DIR" in text
        assert "sys.path.append(system_site)" in text
        assert "RFX_REPO_ROOT" in text
        assert "check_external_solver_dependencies.py" in text
        assert "build_lumped_openems_sparameter_comparison.py" in text


def test_coaxial_gap_openems_generic_builder_uses_magnitude_mode(tmp_path: Path):
    freqs = np.asarray([3.0e9, 5.0e9, 7.0e9], dtype=float)
    # Same magnitudes, deliberately different phase.
    rfx_s11 = np.asarray([0.0 + 0.94j, 0.0 + 0.95j, 0.0 + 0.96j])
    openems_s11 = np.asarray([0.94 + 0.0j, 0.95 + 0.0j, 0.96 + 0.0j])

    payload = (
        coaxial_gap_openems_generic_compare
        .build_coaxial_gap_openems_comparison_from_s11(
            freqs_hz=freqs,
            rfx_s11=rfx_s11,
            openems_s11=openems_s11,
            output_dir=tmp_path / "out",
        )
    )

    assert payload["status"] == "passed"
    assert payload["comparison_mode"] == "magnitude"
    assert payload["claim_scope"].startswith("narrow single-cell coaxial gap")
    assert payload["summary"]["max_mag_abs_diff"] == 0.0


def test_floquet_empty_space_analytic_builder_uses_zero_reference(tmp_path: Path):
    freqs = np.asarray([3.0e9, 4.0e9, 5.0e9], dtype=float)
    candidate = np.asarray([0.01 + 0.02j, 0.02 + 0.01j, 0.03 + 0.0j])

    payload = (
        floquet_empty_space_analytic_compare
        .build_floquet_empty_space_comparison_from_s11(
            freqs_hz=freqs,
            candidate_s11=candidate,
            output_dir=tmp_path / "out",
        )
    )

    assert payload["status"] == "passed"
    assert payload["evidence_level"] == "E2/E3-enabling"
    assert payload["claim_scope"].startswith("narrow empty-space Floquet modal")
    assert payload["summary"]["max_abs_diff"] == np.max(np.abs(candidate))


def test_floquet_slab_analytic_builder_uses_magnitude_mode(tmp_path: Path):
    freqs = np.asarray([3.0e9, 4.0e9, 5.0e9], dtype=float)
    # Same magnitudes, deliberately different phase.
    candidate = np.asarray([0.0 + 0.12j, 0.0 + 0.18j, 0.0 + 0.22j])
    reference = np.asarray([0.12 + 0.0j, 0.18 + 0.0j, 0.22 + 0.0j])

    payload = (
        floquet_slab_analytic_compare
        .build_floquet_slab_comparison_from_s11(
            freqs_hz=freqs,
            candidate_s11=candidate,
            reference_s11=reference,
            output_dir=tmp_path / "out",
        )
    )

    assert payload["status"] == "passed"
    assert payload["evidence_level"] == "E2/E3-enabling"
    assert payload["comparison_mode"] == "magnitude"
    assert payload["claim_scope"].startswith(
        "narrow broadside homogeneous-slab"
    )
    assert payload["summary"]["max_mag_abs_diff"] == 0.0


def test_generalized_planar_quasitem_oracle_report_is_planning_only():
    report = generalized_planar_quasitem_report.evaluate_generalized_planar_quasitem_oracles()

    assert report["status"] == "passed"
    assert report["evidence_level"] == "E2-planning"
    assert "no implemented generalized planar-port API" in report["claim_scope"]
    assert report["case_count"] == 4
    assert report["sweep_count"] == 8
    assert report["sweep_row_count"] == 24
    assert report["min_z0_ohm"] > 10.0
    assert report["max_z0_ohm"] < 150.0
    for row in report["rows"]:
        assert row["status"] == "passed"
        assert all(row["checks"].values())
    assert report["sweeps"]["status"] == "passed"
    assert report["sweeps"]["evidence_level"] == "E2-planning-envelope-template"
    for row in report["sweeps"]["rows"]:
        assert row["status"] == "passed"
        assert all(row["checks"].values())


def test_rf_e5_blocker_ladder_keeps_broad_goal_blocked():
    report = rf_e5_blocker_ladder_report.build_blocker_ladder(
        manifest_path=(
            REPO_ROOT
            / "scripts"
            / "diagnostics"
            / "port_external_reference_requirements.json"
        ),
        support_matrix_path=(
            REPO_ROOT / "docs" / "guides" / "sparameter_support_matrix.json"
        ),
    )

    assert report["status"] == "blocked"
    assert report["evidence_level"] == "planning-only"
    assert report["required_family_count"] == 7
    assert set(report["families_without_external_comparison_artifacts"]) == {
        "floquet_port",
        "generalized_planar_ports",
    }
    # Down from all 7 to 4: the waveguide and coaxial families closed their
    # broad-E5 envelopes via committed PRs (rectangular_waveguide_port by the
    # flux-extractor / NU-flux work in 48a2ce6 + dc3be92, coaxial_port by the
    # M71 gap envelope in 1c06643). wire_port also gained a broad-E5 envelope
    # entry (0817bf3). The remaining 4 still have an empty
    # broad_e5_envelope_artifacts list in the committed requirements JSON.
    assert set(report["families_without_broad_e5_envelope_artifacts"]) == {
        "floquet_port",
        "generalized_planar_ports",
        "lumped_port",
        "microstrip_line_port",
    }
    assert "e5_envelope" in report["stage_counts"]
    by_family = {row["family"]: row for row in report["family_ladders"]}
    assert by_family["generalized_planar_ports"]["first_blocking_stage"] == "api_surface"
    assert by_family["floquet_port"]["blocked_dependency_count"] >= 1
    assert "Do not call update_goal" in report["completion_decision"]


def test_waveguide_wr90_generic_builder_parses_4way_stdout(tmp_path: Path):
    stdout = tmp_path / "cv11.stdout.txt"
    stdout.write_text(
        "\n".join(
            [
                "[4way slab S11] f_GHz |    rfx     |   MEEP_r4  | OpenEMS_r4 | Palace_r_h2 | |S|_diff(rfx-Palace) | phase_diff_deg(rfx-Palace)",
                "-------------------------------------------------------------------------------------------------------------------------------",
                "   8.20 | 0.5000@  +0.00d | 0.5100@  +0.00d | 0.5100@  +0.00d | 0.5200@ +90.00d |       -0.0200        |    -90.00",
                "   8.40 | 0.4000@ +10.00d | 0.4100@ +10.00d | 0.4100@ +10.00d | 0.4200@+100.00d |       -0.0200        |    -90.00",
                "[summary slab S11 vs Palace_r_h2] |S|_diff: max=0.0200 mean=0.0200 | phase: max|d|=90.00d mean|d|=90.00d",
                "",
                "[4way slab S21] f_GHz |    rfx     |   MEEP_r4  | OpenEMS_r4 | Palace_r_h2 | |S|_diff(rfx-Palace) | phase_diff_deg(rfx-Palace)",
                "-------------------------------------------------------------------------------------------------------------------------------",
                "   8.20 | 0.9000@  +0.00d | 0.9100@  +0.00d | 0.9100@  +0.00d | 0.9100@ +90.00d |       -0.0100        |    -90.00",
                "   8.40 | 0.9200@ +10.00d | 0.9300@ +10.00d | 0.9300@ +10.00d | 0.9300@+100.00d |       -0.0100        |    -90.00",
                "[summary slab S21 vs Palace_r_h2] |S|_diff: max=0.0100 mean=0.0100 | phase: max|d|=90.00d mean|d|=90.00d",
            ]
        ),
        encoding="utf-8",
    )

    payload = waveguide_wr90_generic_compare.build_waveguide_wr90_generic_comparison(
        stdout,
        tmp_path / "out",
    )

    assert payload["status"] == "passed"
    assert payload["comparison_mode"] == "magnitude"
    assert payload["external_reference_column"] == "Palace_r_h2"
    assert payload["summary"]["max_mag_abs_diff"] < 0.021


def test_patch_wire_openems_generic_builder_uses_magnitude_mode(tmp_path: Path):
    crossval = tmp_path / "crossval05_patch_openems_rfx.json"
    crossval.write_text(
        json.dumps(
            {
                "rfx_freqs_hz": [1.5e9, 2.0e9, 3.0e9, 3.4e9],
                "openems_freqs_hz": [1.5e9, 2.0e9, 3.0e9, 3.4e9],
                # Same magnitudes, deliberately different phase/reference convention.
                "rfx_s11": [[0.0, 0.4], [0.0, 0.3], [0.0, 0.2], [0.0, 0.1]],
                "openems_s11": [[0.4, 0.0], [0.3, 0.0], [0.2, 0.0], [0.1, 0.0]],
                "rfx_vs_openems_harminv_pct": 3.8,
                "rfx_internal_pct": 3.0,
                "rfx_vs_analytic_pct": 6.5,
                "rfx_s11_passive": True,
                "rfx_s11_max_abs": 0.4,
            }
        ),
        encoding="utf-8",
    )

    payload = patch_wire_openems_generic_compare.build_patch_wire_openems_generic_comparison(
        crossval,
        tmp_path / "out",
    )

    assert payload["status"] == "passed"
    assert payload["comparison_mode"] == "magnitude"
    assert payload["claim_scope"].startswith("narrow crossval05")
    assert payload["summary"]["max_mag_abs_diff"] == 0.0


def test_external_solver_dependency_audit_is_not_physics_evidence():
    audit = external_solver_dependency_check.build_dependency_audit()

    assert "not E4/E5 physics evidence" in audit["claim_scope"]
    assert "meep_crossval" in audit["capabilities"]
    assert "openems_crossval" in audit["capabilities"]
    assert "rcwa_floquet" in audit["capabilities"]
    for item in audit["module_results"].values():
        assert "find_spec_available" in item
        assert "import_checked" in item
        assert "import_error" in item
    for item in audit["capabilities"].values():
        assert item["status"] in {"available", "blocked"}


def test_external_solver_dependency_probe_handles_missing_parent_package(monkeypatch):
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "openEMS.openEMS":
            raise ModuleNotFoundError("No module named 'openEMS'")
        return original_find_spec(name)

    monkeypatch.setattr(external_solver_dependency_check.importlib.util, "find_spec", fake_find_spec)

    audit = external_solver_dependency_check.build_dependency_audit()

    assert audit["module_results"]["openEMS.openEMS"]["available"] is False
    assert audit["module_results"]["openEMS.openEMS"]["import_checked"] is False
    assert "ModuleNotFoundError" in audit["module_results"]["openEMS.openEMS"]["import_error"]
    assert audit["capabilities"]["openems_crossval"]["status"] == "blocked"


def test_floquet_modal_field_dump_replay_matches_helper_arrays(tmp_path: Path):
    np = floquet_modal_replay.np
    theta_deg = 30.0
    eta_te = (floquet_modal_replay.MU_0 / floquet_modal_replay.EPS_0) ** 0.5
    eta_te /= np.cos(np.deg2rad(theta_deg))
    forward = np.asarray([1.0 + 0.2j, 0.8 - 0.1j], dtype=np.complex128)
    gamma = 0.25 + 0.1j
    backward = gamma * forward
    ex = forward + backward
    hy = (forward - backward) / eta_te
    ex_dft = np.broadcast_to(ex[:, None, None], (2, 3, 4)).copy()
    hy_dft = np.broadcast_to(hy[:, None, None], (2, 3, 4)).copy()
    helper_s = backward / forward
    dump_path = tmp_path / "floquet_dump.npz"
    np.savez_compressed(
        dump_path,
        ex_dft=ex_dft,
        hy_dft=hy_dft,
        freqs_hz=np.asarray([3.0e9, 4.0e9]),
        theta_deg=np.asarray(theta_deg),
        helper_s=helper_s,
        helper_forward=forward,
        helper_backward=backward,
    )

    replay = floquet_modal_replay.replay_floquet_modal_dump(dump_path)

    assert replay["status"] == "passed"
    assert replay["max_s_abs_diff"] < 1e-12
    assert replay["max_forward_abs_diff"] < 1e-12
    assert replay["max_backward_abs_diff"] < 1e-12


def test_rf_infra_e5_goal_audit_stays_blocked_until_all_port_families_are_e5():
    audit = rf_infra_goal_audit.build_goal_audit(
        REPO_ROOT / "docs" / "guides" / "sparameter_support_matrix.json"
    )

    assert audit["status"] == "blocked"
    assert audit["summary"]["future_family_count"] == 1
    assert audit["prompt_to_artifact_checklist"]
    checklist = {
        item["requirement_id"]: item
        for item in audit["prompt_to_artifact_checklist"]
    }
    assert (
        checklist["all_current_and_planned_port_families_e5"]["status"]
        == "blocked"
    )
    assert (
        checklist["broad_e5_external_reference_manifest"]["status"]
        == "blocked"
    )
    assert audit["summary"]["external_reference_status"] == "blocked"
    incomplete = {item["family"]: item for item in audit["incomplete"]}
    assert "lumped_port" in incomplete
    assert "wire_port" in incomplete
    assert "microstrip_line_port" in incomplete
    assert "coaxial_port" in incomplete
    assert "floquet_port" in incomplete
    assert "update_goal" in audit["completion_decision"]


def test_rf_infra_e5_goal_audit_requires_external_manifest_envelope(
    tmp_path: Path,
):
    evidence = tmp_path / "lumped_broad_evidence.json"
    evidence.write_text(
        json.dumps({"status": "passed", "evidence_level": "E5"}),
        encoding="utf-8",
    )
    support_matrix = tmp_path / "sparameter_support_matrix.json"
    support_matrix.write_text(
        json.dumps(
            {
                "port_families": [
                    {
                        "family": "lumped_port",
                        "primitive": "add_port(extent=None)",
                        "is_port": True,
                        "evidence_level": "E5",
                        "evidence_artifacts": [str(evidence)],
                        "known_limits": [],
                        "caveats": [],
                        "promotion_requirements": [],
                    }
                ],
                "future_port_families": [],
            }
        ),
        encoding="utf-8",
    )
    comparison = tmp_path / "lumped_external_comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "status": "passed",
                "evidence_level": "E4",
                "claim_scope": "broad external S-parameter comparison",
                "summary": {
                    "geometry_count": 3,
                    "pair_count": 5,
                    "passed_pair_count": 5,
                    "failed_pair_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    envelope = tmp_path / "lumped_broad_envelope.json"
    envelope.write_text(
        json.dumps(
            {
                "status": "passed",
                "evidence_level": "E5",
                "required_scope": "broad_e5",
                "claim_scope": "broad mesh/frequency/geometry envelope",
                "max_mag_abs_tol": 0.05,
                "envelope_summary": {
                    "case_count": 4,
                    "passed_case_count": 4,
                    "dx_values_m": [5e-5, 1e-4],
                    "eps_r_values": [2.0, 4.0],
                    "geometries": ["slab"],
                    "freq_range_hz": [1.0e10, 1.5e10],
                    "max_mag_abs_diff_across_cases": 0.02,
                },
            }
        ),
        encoding="utf-8",
    )
    yaml_path = tmp_path / "physics_gate_port_external_lumped.yaml"
    yaml_path.write_text("name: unit\n", encoding="utf-8")
    manifest = tmp_path / "port_external_reference_requirements.json"
    base_entry = {
        "family": "lumped_port",
        "required_for_e5": True,
        "required_scope": "broad_e5",
        "current_status": "broad_e5_passed",
        "ad_fd_test": "tests/test_waveguide_flux_ad.py::test_flux_smatrix_grad_finite_and_fd_consistent",
        "missing_evidence": [],
        "existing_artifacts": [],
        "existing_vessl_yamls": [str(yaml_path)],
        "external_comparison_artifacts": [str(comparison)],
    }
    manifest.write_text(
        json.dumps({"requirements": [{**base_entry, "broad_e5_envelope_artifacts": []}]}),
        encoding="utf-8",
    )

    blocked = rf_infra_goal_audit.build_goal_audit(support_matrix, manifest)

    assert blocked["family_results"][0]["status"] == "passed"
    assert blocked["status"] == "blocked"
    assert blocked["summary"]["external_reference_status"] == "blocked"
    checklist = {
        item["requirement_id"]: item
        for item in blocked["prompt_to_artifact_checklist"]
    }
    assert checklist["broad_e5_external_reference_manifest"]["status"] == "blocked"

    manifest.write_text(
        json.dumps(
            {
                "requirements": [
                    {
                        **base_entry,
                        "broad_e5_envelope_artifacts": [str(envelope)],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    passed = rf_infra_goal_audit.build_goal_audit(support_matrix, manifest)

    assert passed["status"] == "complete"
    assert passed["summary"]["external_reference_status"] == "passed"
    checklist = {
        item["requirement_id"]: item
        for item in passed["prompt_to_artifact_checklist"]
    }
    assert checklist["broad_e5_external_reference_manifest"]["status"] == "passed"


def test_rf_infra_e5_goal_audit_rejects_failed_json_evidence_artifact(
    tmp_path: Path,
):
    evidence = tmp_path / "lumped_broad_evidence.json"
    evidence.write_text(
        json.dumps({"status": "failed", "evidence_level": "E5"}),
        encoding="utf-8",
    )
    support_matrix = tmp_path / "sparameter_support_matrix.json"
    support_matrix.write_text(
        json.dumps(
            {
                "port_families": [
                    {
                        "family": "lumped_port",
                        "primitive": "add_port(extent=None)",
                        "is_port": True,
                        "evidence_level": "E5",
                        "evidence_artifacts": [str(evidence)],
                        "known_limits": [],
                        "caveats": [],
                        "promotion_requirements": [],
                    }
                ],
                "future_port_families": [],
            }
        ),
        encoding="utf-8",
    )
    comparison = tmp_path / "lumped_external_comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "status": "passed",
                "evidence_level": "E4",
                "claim_scope": "broad external S-parameter comparison",
                "summary": {
                    "geometry_count": 3,
                    "pair_count": 5,
                    "passed_pair_count": 5,
                    "failed_pair_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    envelope = tmp_path / "lumped_broad_envelope.json"
    envelope.write_text(
        json.dumps(
            {
                "status": "passed",
                "evidence_level": "E5",
                "required_scope": "broad_e5",
                "claim_scope": "broad mesh/frequency/geometry envelope",
                "max_mag_abs_tol": 0.05,
                "envelope_summary": {
                    "case_count": 4,
                    "passed_case_count": 4,
                    "dx_values_m": [5e-5, 1e-4],
                    "eps_r_values": [2.0, 4.0],
                    "geometries": ["slab"],
                    "freq_range_hz": [1.0e10, 1.5e10],
                    "max_mag_abs_diff_across_cases": 0.02,
                },
            }
        ),
        encoding="utf-8",
    )
    yaml_path = tmp_path / "physics_gate_port_external_lumped.yaml"
    yaml_path.write_text("name: unit\n", encoding="utf-8")
    manifest = tmp_path / "port_external_reference_requirements.json"
    manifest.write_text(
        json.dumps(
            {
                "requirements": [
                    {
                        "family": "lumped_port",
                        "required_for_e5": True,
                        "required_scope": "broad_e5",
                        "current_status": "broad_e5_passed",
                        "ad_fd_test": "tests/test_waveguide_flux_ad.py::test_flux_smatrix_grad_finite_and_fd_consistent",
                        "missing_evidence": [],
                        "existing_artifacts": [],
                        "existing_vessl_yamls": [str(yaml_path)],
                        "external_comparison_artifacts": [str(comparison)],
                        "broad_e5_envelope_artifacts": [str(envelope)],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    audit = rf_infra_goal_audit.build_goal_audit(support_matrix, manifest)

    assert audit["summary"]["external_reference_status"] == "passed"
    assert audit["status"] == "blocked"
    family = audit["family_results"][0]
    assert family["status"] == "partial"
    assert "status=passed" in "; ".join(family["blockers"])
    assert family["artifact_checks"][0]["reported_status"] == "failed"


@pytest.mark.parametrize(
    "ad_fd_test,expect_passed",
    [
        # A real, existing, non-xfail AD-vs-FD test -> the AD gate is satisfied.
        ("tests/test_waveguide_flux_ad.py::test_flux_smatrix_grad_finite_and_fd_consistent", True),
        # No AD test declared -> the differentiability moat is absent -> blocked.
        (None, False),
        # Declared but the file/test does not exist -> dangling pointer -> blocked.
        ("tests/test_does_not_exist.py::test_nope", False),
    ],
)
def test_ad_fd_gate_blocks_broad_e5_without_valid_ad_test(
    tmp_path: Path, ad_fd_test, expect_passed
):
    """T2.2 falsifier: broad_e5_passed requires a valid named AD-vs-FD test.

    Everything else (broad E4 comparison + broad E5 envelope + YAML) passes; the
    ONLY differentiator is `ad_fd_test`. Removing/dangling it must flip the
    family GREEN->blocked — wiring the differentiability moat into the verdict
    (framework audit #6). The companion collection-time check (a declared test
    that is xfail/skip cannot satisfy the gate) lives in
    tests/test_ad_surface_contract.py::test_ad_fd_gate_tests_are_collected_and_not_xfail_skip.
    """
    support_matrix = tmp_path / "sparameter_support_matrix.json"
    support_matrix.write_text(
        json.dumps(
            {
                "port_families": [
                    {
                        "family": "lumped_port",
                        "primitive": "add_port(extent=None)",
                        "is_port": True,
                        "evidence_level": "E5",
                    }
                ],
                "future_port_families": [],
            }
        ),
        encoding="utf-8",
    )
    comparison = tmp_path / "broad_comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "status": "passed",
                "evidence_level": "E4",
                "claim_scope": "broad external S-parameter comparison",
                "summary": {
                    "geometry_count": 3,
                    "pair_count": 5,
                    "passed_pair_count": 5,
                    "failed_pair_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    envelope = tmp_path / "broad_envelope.json"
    envelope.write_text(
        json.dumps(
            {
                "status": "passed",
                "evidence_level": "E5",
                "required_scope": "broad_e5",
                "claim_scope": "broad mesh/frequency/geometry envelope",
                "max_mag_abs_tol": 0.05,
                "envelope_summary": {
                    "case_count": 4,
                    "passed_case_count": 4,
                    "dx_values_m": [5e-5, 1e-4],
                    "eps_r_values": [2.0, 4.0],
                    "geometries": ["slab", "notch"],
                    "freq_range_hz": [1.0e10, 1.5e10],
                    "max_mag_abs_diff_across_cases": 0.02,
                },
            }
        ),
        encoding="utf-8",
    )
    yaml_path = tmp_path / "physics_gate_port_external_lumped.yaml"
    yaml_path.write_text("name: unit\n", encoding="utf-8")
    entry = {
        "family": "lumped_port",
        "required_for_e5": True,
        "required_scope": "broad_e5",
        "current_status": "broad_e5_passed",
        "missing_evidence": [],
        "existing_artifacts": [],
        "existing_vessl_yamls": [str(yaml_path)],
        "external_comparison_artifacts": [str(comparison)],
        "broad_e5_envelope_artifacts": [str(envelope)],
    }
    if ad_fd_test is not None:
        entry["ad_fd_test"] = ad_fd_test
    manifest = tmp_path / "port_external_reference_requirements.json"
    manifest.write_text(json.dumps({"requirements": [entry]}), encoding="utf-8")

    audit = port_external_reference_check.build_external_reference_audit(
        manifest, support_matrix
    )
    family = audit["requirements"][0]
    if expect_passed:
        assert family["status"] == "passed", family["blockers"]
        assert family["ad_gate_ok"] is True
    else:
        assert family["status"] == "blocked"
        assert family["ad_gate_ok"] is False
        assert any("ad_fd_test" in b for b in family["blockers"]), family["blockers"]


def test_every_family_declares_target_ceiling_and_usage_rule():
    """Per-port formalization (2026-06-17): broad-E5 is NOT a universal goal.

    Every E5-required family must declare a physics-set `target_ceiling` from the
    controlled vocab + a `usage_rule` ("use this port when ..."). This locks the
    rule that single-cell ports top out at E4 (validated-to-ceiling, not blocked),
    MSL is matched-regime-only, coax needs a differentiable API, floquet is
    broadside-only, generalized-planar is unimplemented — so a `blocked` broad_e5
    verdict is contextualized, not read as failure.
    """
    manifest = json.loads(
        (REPO_ROOT / "scripts/diagnostics/port_external_reference_requirements.json").read_text()
    )
    vocab = port_external_reference_check.VALID_TARGET_CEILINGS
    for e in manifest["requirements"]:
        fam = e["family"]
        assert e.get("target_ceiling") in vocab, (
            f"{fam}: target_ceiling {e.get('target_ceiling')!r} not in {sorted(vocab)}"
        )
        assert e.get("usage_rule", "").strip(), f"{fam}: missing usage_rule"
    # The auditor must surface the formalized fields per family.
    audit = port_external_reference_check.build_external_reference_audit(
        REPO_ROOT / "scripts/diagnostics/port_external_reference_requirements.json",
        REPO_ROOT / "docs/guides/sparameter_support_matrix.json",
    )
    for r in audit["requirements"]:
        assert r["target_ceiling"] in vocab
        assert r["usage_rule"]
        # broad_e5_is_the_target_ceiling is a PURE LABEL restatement, not an
        # achieved-vs-ceiling check: it must equal (ceiling == "broad-E5") and
        # never read evidence/status. Locks the 2026-06-17 rename away from the
        # earlier misnamed `at_or_below_target_ceiling` (which a reader could
        # mistake for a real "validated to its ceiling" signal).
        assert r["broad_e5_is_the_target_ceiling"] == (
            r["target_ceiling"] == "broad-E5"
        ), f"{r['family']}: broad_e5_is_the_target_ceiling must restate the label only"
    by = {e["family"]: e["target_ceiling"] for e in manifest["requirements"]}
    assert by["rectangular_waveguide_port"] == "broad-E5"
    assert by["lumped_port"] == "E4-natural-ceiling"
    assert by["wire_port"] == "E4-natural-ceiling"
    # Concrete value witnesses: only the broad-E5-target family reads True.
    flag = {r["family"]: r["broad_e5_is_the_target_ceiling"] for r in audit["requirements"]}
    assert flag["rectangular_waveguide_port"] is True
    assert flag["lumped_port"] is False
    assert flag["wire_port"] is False
    assert flag["coaxial_port"] is False


def test_real_manifest_broad_e5_family_survives_require_committed():
    """T2.5 'on in CI': the real broad_e5_passed family (waveguide) survives the
    committed-artifact check — its gating evidence is GENUINELY git-tracked, not
    riding on gitignored .omx. This gates every PR in the fast suite (which has
    git). Skips only where git is unavailable (the check degrades to empty-set;
    e.g. the git-less VESSL image), so it never false-fails there.
    """
    if not port_external_reference_check._git_committed_paths():
        pytest.skip("git unavailable; require_committed cannot be verified")
    audit = port_external_reference_check.build_external_reference_audit(
        REPO_ROOT / "scripts/diagnostics/port_external_reference_requirements.json",
        REPO_ROOT / "docs/guides/sparameter_support_matrix.json",
        require_committed=True,
    )
    wg = [r for r in audit["requirements"]
          if r["family"] == "rectangular_waveguide_port"][0]
    assert wg["status"] == "passed", wg["blockers"]
    assert not wg["uncommitted_gating_artifacts"], wg["uncommitted_gating_artifacts"]


def test_is_committed_distinguishes_tracked_from_present(tmp_path: Path):
    """T2.5: _is_committed uses git-tracked membership, not path.exists()."""
    # A real tracked repo file is committed.
    assert port_external_reference_check._is_committed(
        "scripts/diagnostics/port_external_reference_requirements.json"
    )
    # A present-on-disk-but-untracked file (the .omx-overclaim shape) is NOT.
    untracked = tmp_path / "present_but_untracked.json"
    untracked.write_text("{}", encoding="utf-8")
    assert untracked.exists()  # path.exists() would have passed it (the old bug)
    assert not port_external_reference_check._is_committed(str(untracked))


def test_require_committed_blocks_uncommitted_gating_artifacts(tmp_path: Path):
    """T2.5 falsifier: under require_committed, present-but-untracked gating
    artifacts (gitignored .omx) do NOT count -> broad_e5_passed is blocked.

    Everything else passes (broad-E4 comparison + broad-E5 envelope content +
    YAML + a real AD test); the artifacts live in tmp_path, so they exist on disk
    but are not git-tracked — exactly the coaxial-overclaim shape (audit #2).
    """
    support_matrix = tmp_path / "sparameter_support_matrix.json"
    support_matrix.write_text(
        json.dumps({"port_families": [{"family": "lumped_port",
                                       "primitive": "add_port(extent=None)",
                                       "is_port": True, "evidence_level": "E5"}],
                    "future_port_families": []}),
        encoding="utf-8",
    )
    comparison = tmp_path / "broad_comparison.json"
    comparison.write_text(
        json.dumps({"status": "passed", "evidence_level": "E4",
                    "claim_scope": "broad external S-parameter comparison",
                    "summary": {"geometry_count": 3, "pair_count": 5,
                                "passed_pair_count": 5, "failed_pair_count": 0}}),
        encoding="utf-8",
    )
    envelope = tmp_path / "broad_envelope.json"
    envelope.write_text(
        json.dumps({"status": "passed", "evidence_level": "E5",
                    "required_scope": "broad_e5",
                    "claim_scope": "broad mesh/frequency/geometry envelope",
                    "max_mag_abs_tol": 0.05,
                    "envelope_summary": {"case_count": 4, "passed_case_count": 4,
                                         "dx_values_m": [5e-5, 1e-4],
                                         "eps_r_values": [2.0, 4.0],
                                         "geometries": ["slab", "notch"],
                                         "freq_range_hz": [1.0e10, 1.5e10],
                                         "max_mag_abs_diff_across_cases": 0.02}}),
        encoding="utf-8",
    )
    yaml_path = tmp_path / "physics_gate_port_external_lumped.yaml"
    yaml_path.write_text("name: unit\n", encoding="utf-8")
    manifest = tmp_path / "port_external_reference_requirements.json"
    manifest.write_text(
        json.dumps({"requirements": [{
            "family": "lumped_port", "required_for_e5": True,
            "required_scope": "broad_e5", "current_status": "broad_e5_passed",
            "ad_fd_test": "tests/test_waveguide_flux_ad.py::test_flux_smatrix_grad_finite_and_fd_consistent",
            "missing_evidence": [], "existing_artifacts": [],
            "existing_vessl_yamls": [str(yaml_path)],
            "external_comparison_artifacts": [str(comparison)],
            "broad_e5_envelope_artifacts": [str(envelope)],
        }]}),
        encoding="utf-8",
    )

    # This is the EXACT audit-#2 shape: the gating artifacts are PRESENT on disk
    # (so the old path.exists() would pass them) but NOT committed to HEAD.
    assert comparison.exists() and envelope.exists()

    # Default (require_committed=False): present + content-pass -> passed. This is
    # precisely the overclaim the old path.exists()-based check let through.
    default = port_external_reference_check.build_external_reference_audit(
        manifest, support_matrix
    )
    assert default["requirements"][0]["status"] == "passed", (
        default["requirements"][0]["blockers"]
    )

    # require_committed=True: present-but-uncommitted -> blocked, with the gating
    # artifacts surfaced as uncommitted and marked git_tracked=False (the unique
    # present-but-untracked path, exercised here in CI — not just the missing-file
    # path that a clean checkout would hit).
    committed = port_external_reference_check.build_external_reference_audit(
        manifest, support_matrix, require_committed=True
    )
    fam = committed["requirements"][0]
    assert fam["status"] == "blocked"
    assert fam["uncommitted_gating_artifacts"], "uncommitted gating artifacts not surfaced"
    assert any("committed to HEAD" in b for b in fam["blockers"]), fam["blockers"]
    assert all(
        c["git_tracked"] is False
        for c in fam["broad_e5_envelope_artifact_checks"]
    ), "present-but-untracked envelope artifacts not flagged git_tracked=False"


def test_broad_envelope_rejected_without_numeric_breadth(tmp_path: Path):
    """Single-case prose-'broad' artifact must be rejected (T1 gameability fix)."""
    # Case 1: single-case artifact — fails case_count < 4.
    single_case = tmp_path / "single_case_envelope.json"
    single_case.write_text(
        json.dumps(
            {
                "status": "passed",
                "evidence_level": "E5-broad-totally-legit",
                "claim_scope": "broad mesh frequency geometry coverage",
                "envelope_summary": {
                    "case_count": 1,
                    "passed_case_count": 1,
                    "dx_values_m": [1e-4],
                    "eps_r_values": [2.0],
                    "geometries": ["slab"],
                    "freq_range_hz": [1e9, 1e9],
                },
                "max_mag_abs_tol": 0.05,
            }
        ),
        encoding="utf-8",
    )
    # Pass absolute paths — _repo_path() returns them unchanged when is_absolute().
    result_single = port_external_reference_check._broad_e5_envelope_artifact_check(
        str(single_case)
    )
    assert result_single["is_passed_broad_e5_envelope"] is False

    # Case 2: artifact with NO envelope_summary key at all.
    no_summary = tmp_path / "no_summary_envelope.json"
    no_summary.write_text(
        json.dumps(
            {
                "status": "passed",
                "evidence_level": "E5-broad-totally-legit",
                "claim_scope": "broad mesh frequency geometry coverage",
                "max_mag_abs_tol": 0.05,
            }
        ),
        encoding="utf-8",
    )
    result_no_summary = port_external_reference_check._broad_e5_envelope_artifact_check(
        str(no_summary)
    )
    assert result_no_summary["is_passed_broad_e5_envelope"] is False
