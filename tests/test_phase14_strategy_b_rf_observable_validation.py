"""Phase XIV RF-observable validation tests."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import phase14_strategy_b_rf_observable_validation as phase14

ROOT = Path(__file__).resolve().parents[1]


def _green_phase13(tmp_path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "promotion_contract": "phase_x_strategy_b_production_promotion",
        "summary": {
            "all_eligible_promoted": True,
            "blocked_families": [],
            "promoted_families": [
                "source_probe",
                "cpml_topology",
                "pec_topology",
                "port_proxy",
            ],
        },
    }
    path = tmp_path / "phase13_all_promotion.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _base_artifact(tmp_path: Path) -> dict:
    return phase14.build_phase14_artifact(
        phase13_baseline=_green_phase13(tmp_path),
        n_steps=64,
    )


def test_extract_resonance_frequency_accepts_finite_ringdown():
    dt = 1e-12
    t = np.arange(2048) * dt
    frequency = 5e9
    series = np.exp(-t / 2e-10) * np.sin(2 * np.pi * frequency * t)

    result = phase14.extract_resonance_frequency(series, dt)

    assert result["method"] == "fft_peak"
    assert math.isfinite(result["frequency_hz"])
    assert result["frequency_hz"] > 0.0
    assert abs(result["frequency_hz"] - frequency) / frequency < 0.1
    assert result["finite"] is True


@pytest.mark.parametrize(
    "series, reason",
    [
        ([], "insufficient_time_series_samples"),
        ([0.0] * 32, "no_resolvable_peak"),
        ([0.0] * 31 + [math.nan], "nonfinite_time_series_or_dt"),
    ],
)
def test_extract_resonance_frequency_rejects_invalid_series(series, reason):
    with pytest.raises(ValueError, match=reason):
        phase14.extract_resonance_frequency(series, 1e-12)


def test_gate_policy_rejects_missing_observable_source(tmp_path):
    artifact = _base_artifact(tmp_path)
    del artifact["fixtures"][0]["observables"]["resonance"]["observable_source"]

    result = phase14.evaluate_artifact_gates(artifact)

    assert result["overall_status"] == "rf_observable_blocked"
    assert "observable_source_missing_or_ambiguous" in result["failed_gates"]


def test_gate_policy_rejects_noncanonical_native_sparam_claim(tmp_path):
    artifact = _base_artifact(tmp_path)
    artifact["fixtures"][0]["observables"]["sparams_reference"][
        "native_strategy_b_sparams"
    ] = True

    result = phase14.evaluate_artifact_gates(artifact)

    assert result["gates"]["native_sparams_truthfulness_valid"] is False
    assert "forbidden_native_sparam_artifact_key" in result["failed_gates"]


def test_gate_policy_rejects_native_sparam_true_with_null_arrays(tmp_path):
    artifact = _base_artifact(tmp_path)
    seam = artifact["strategy_b_seam"]
    seam["native_s_params_supported"] = True
    seam["phase1_forward_result_s_params"] = None
    seam["phase1_forward_result_freqs"] = None
    seam["phase1_forward_result_s_params_shape"] = None
    seam["phase1_forward_result_freqs_shape"] = None
    seam["observable_source"] = "strategy_b_native_sparams"
    seam["finite"] = False

    result = phase14.evaluate_artifact_gates(artifact)

    assert result["gates"]["native_sparams_truthfulness_valid"] is False
    assert "native_strategy_b_sparams_claimed_without_arrays" in result["failed_gates"]
    assert "native_strategy_b_sparams_claimed_without_freqs" in result["failed_gates"]
    assert "native_strategy_b_sparams_shape_invalid" in result["failed_gates"]


def test_passivity_check_accepts_bounded_reference_sparams_and_rejects_gain():
    bounded = phase14.passivity_check(np.array([0.1 + 0.1j, 0.8 + 0.0j]))
    gained = phase14.passivity_check(np.array([0.1 + 0.0j, 1.2 + 0.0j]))

    assert bounded["passed"] is True
    assert gained["passed"] is False
    assert gained["failure_reason"] == "passivity_threshold_exceeded"


def test_input_impedance_not_applicable_requires_reason():
    missing_reason = phase14.input_impedance_check(None, applicable=False)
    with_reason = phase14.input_impedance_check(
        None,
        applicable=False,
        not_applicable_reason="no port observable in this fixture",
    )

    assert missing_reason["passed"] is False
    assert missing_reason["failure_reason"] == "missing_not_applicable_reason"
    assert with_reason["passed"] is True


def test_optional_solver_wrapper_records_absent_optional_skip():
    record = phase14._run_optional_solver_correlation(
        "meep",
        required=False,
        fixture={"strategy_b_frequency_hz": 1.0e9},
        availability_checker=lambda _solver: False,
    )

    assert record["available"] is False
    assert record["executed"] is False
    assert record["required"] is False
    assert record["skipped_reason"] == "not_installed"
    assert record["passed"] is None


def test_optional_solver_wrapper_fails_closed_when_required_absent():
    record = phase14._run_optional_solver_correlation(
        "openems",
        required=True,
        fixture={"strategy_b_frequency_hz": 1.0e9},
        availability_checker=lambda _solver: False,
    )

    assert record["available"] is False
    assert record["executed"] is False
    assert record["required"] is True
    assert record["passed"] is False
    assert record["failure_reason"] == "required_solver_not_installed"


def test_subprocess_meep_runner_accepts_isolated_python(monkeypatch):
    monkeypatch.setattr(
        phase14,
        "_standalone_meep_cavity_code",
        lambda: "import json; print(json.dumps({'frequency_hz': 123.0}))",
    )

    frequency = phase14._run_subprocess_solver(
        "meep",
        solver_python=sys.executable,
        timeout_s=10.0,
    )

    assert frequency == 123.0


def test_optional_solver_wrapper_records_executed_pass_and_tolerance_fail():
    passed = phase14._run_optional_solver_correlation(
        "meep",
        required=True,
        fixture={"strategy_b_frequency_hz": 1.0e9},
        runner=lambda _fixture: 1.005e9,
        availability_checker=lambda _solver: True,
        tolerance=0.02,
    )
    failed = phase14._run_optional_solver_correlation(
        "meep",
        required=True,
        fixture={"strategy_b_frequency_hz": 1.0e9},
        runner=lambda _fixture: 1.2e9,
        availability_checker=lambda _solver: True,
        tolerance=0.02,
    )

    assert passed["executed"] is True
    assert passed["passed"] is True
    assert failed["executed"] is True
    assert failed["passed"] is False
    assert failed["failure_reason"] == "solver_correlation_tolerance_exceeded"


def test_optional_solver_wrapper_restores_cwd_when_solver_changes_directory(tmp_path):
    original = Path.cwd()
    solver_cwd = tmp_path / "solver-cwd"
    solver_cwd.mkdir()

    def runner(_fixture):
        os.chdir(solver_cwd)
        return 1.0e9

    record = phase14._run_optional_solver_correlation(
        "meep",
        required=False,
        fixture={"strategy_b_frequency_hz": 1.0e9},
        runner=runner,
        availability_checker=lambda _solver: True,
    )

    assert record["passed"] is True
    assert Path.cwd() == original


def test_phase13_baseline_policy_accepts_green_and_rejects_blocked(tmp_path):
    green = phase14.phase13_baseline_record(_green_phase13(tmp_path))
    blocked_path = tmp_path / "blocked.json"
    blocked_path.write_text(
        json.dumps(
            {"summary": {"all_eligible_promoted": False, "blocked_families": ["pec"]}}
        ),
        encoding="utf-8",
    )
    blocked = phase14.phase13_baseline_record(blocked_path)

    assert green["valid"] is True
    assert blocked["valid"] is False
    assert blocked["failure_reason"] == "phase13_not_all_eligible_promoted"


def test_aligned_cavity_fixture_matches_analytic_reference_for_solver_correlation():
    fixture = phase14._run_aligned_cavity_strategy_b_fixture(n_steps=512)
    resonance = fixture["observables"]["resonance"]

    assert fixture["fixture_id"] == "aligned_pec_cavity_tm110_strategy_b_time_series"
    assert fixture["grid_shape"] == [101, 101, 51]
    assert resonance["reference_source"] == "analytic_reference"
    assert resonance["passed"] is True
    assert resonance["relative_error"] <= phase14.DEFAULT_RESONANCE_TOLERANCE


def test_optional_solver_correlation_passes_when_aligned_solver_matches():
    fixture = phase14._run_aligned_cavity_strategy_b_fixture(n_steps=512)
    record = phase14._run_optional_solver_correlation(
        "openems",
        required=True,
        fixture=fixture,
        runner=lambda _fixture: phase14.ALIGNED_CAVITY_FREQ_HZ,
        availability_checker=lambda _solver: True,
    )

    assert record["fixture_id"] == fixture["fixture_id"]
    assert record["passed"] is True
    assert record["relative_error"] <= phase14.DEFAULT_RESONANCE_TOLERANCE


def test_default_harness_emits_complete_source_labelled_artifact(tmp_path):
    output = tmp_path / "phase14_rf_observable_validation.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase14_strategy_b_rf_observable_validation.py",
            "--phase13-baseline",
            str(_green_phase13(tmp_path)),
            "--output",
            str(output),
            "--n-steps",
            "64",
            "--indent",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == phase14.SCHEMA_VERSION
    assert artifact["overall_status"] == "rf_observable_validated_limited"
    assert artifact["failed_gates"] == []
    seam = artifact["strategy_b_seam"]
    assert seam["native_s_params_supported"] is True
    assert seam["observable_source"] == "strategy_b_native_sparams"
    assert seam["phase1_forward_result_s_params_shape"] == [1, 1, 2]
    assert seam["phase1_forward_result_freqs_shape"] == [2]
    assert seam["phase1_forward_result_s_params"] is not None
    assert seam["phase1_forward_result_freqs"] is not None
    assert seam["finite"] is True
    resonance = artifact["fixtures"][0]["observables"]["resonance"]
    assert resonance["observable_source"] == "strategy_b_time_series"
    assert resonance["reference_source"] == "standard_rfx_time_series_reference"
    assert resonance["frequency_hz"] > 0.0
    impedance = artifact["fixtures"][0]["observables"]["input_impedance"]
    assert impedance["applicable"] is False
    assert impedance["not_applicable_reason"]
    assert all(record["skipped_reason"] for record in artifact["optional_solver_correlation"])
