"""Phase VIII Strategy B physical-validation contract tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import phase8_strategy_b_physical_validation as phase8

ROOT = Path(__file__).resolve().parents[1]


def _assert_verified_post_source_window(row):
    window = row["physical_metrics"]["post_source_window"]
    assert window["source_off_window_verified"] is True
    assert window["graded_window_start_index"] > window["source_active_last_index"]
    assert window["graded_sample_count"] < window["full_sample_count"]


@pytest.fixture(scope="module")
def quick_report():
    return phase8.build_phase8_report(mode="quick")


def test_family_classifier_keeps_physical_statuses_distinct():
    pass_row = {"row_state": "pass"}
    warn_row = {"row_state": "warn"}
    fail_row = {"row_state": "fail"}

    assert phase8.classify_family([pass_row])["status"] == "physics_validated_limited"
    assert phase8.classify_family([warn_row])["status"] == "physics_experimental"
    assert phase8.classify_family([fail_row])["status"] == "physics_blocked"
    assert phase8.classify_family([])["status"] == "not_evaluated"
    assert phase8.classify_family([pass_row], fail_closed_pass=False)["status"] == "physics_blocked"

    overall = phase8.classify_overall(
        {
            "source_probe": {"status": "physics_validated_limited"},
            "cpml_topology": {"status": "physics_experimental"},
        },
        fail_closed_pass=True,
    )
    assert overall["status"] == "physics_experimental"
    assert "production_ready_limited" not in json.dumps(overall)


def test_quick_report_schema_and_phase_vii_integration(quick_report):
    assert quick_report["schema_version"] == 1
    assert quick_report["benchmark_contract"] == phase8.BENCHMARK_CONTRACT
    assert quick_report["contract_status"] == phase8.CONTRACT_STATUS
    assert quick_report["preserves_phase_vii_readiness_contract"] is True
    assert quick_report["readiness_integration"]["does_not_set_production_ready_limited"] is True
    assert quick_report["fail_closed_audit"]["row_state"] == "pass"
    assert quick_report["summary"]["rows_with_required_fields"] == len(phase8.FAMILIES)
    assert quick_report["summary"]["overall_status"] == "physics_experimental"

    serialized = json.dumps(quick_report)
    assert "phase_viii_strategy_b_physical_validation" in serialized
    statuses = [payload["status"] for payload in quick_report["summary"]["family_statuses"].values()]
    assert "production_ready_limited" not in statuses
    assert quick_report["summary"]["overall_status"] != "production_ready_limited"


def test_quick_report_has_all_mandatory_physical_oracles(quick_report):
    rows_by_family = {row["family"]: row for row in quick_report["rows"]}
    assert set(rows_by_family) == set(phase8.FAMILIES)

    for family, row in rows_by_family.items():
        assert all(field in row for field in phase8.REQUIRED_ROW_FIELDS), family
        assert row["strategy_b_executed"] is True
        assert row["row_state"] in {"pass", "warn"}
        assert row["measured_value"] is not None
        assert row["physical_metrics"]["finite"] is True
        assert row["oracle_type"]
        assert row["tolerance"]

    source = rows_by_family["source_probe"]
    assert source["metric_name"] == "causal_margin_steps"
    assert source["row_state"] == "pass"
    assert source["physical_metrics"]["source_probe_distance_m"] > 0.0
    assert source["physical_metrics"]["c0_travel_time_s"] > 0.0
    assert source["physical_metrics"]["causal_margin_steps"] >= 0.0

    cpml = rows_by_family["cpml_topology"]
    assert cpml["metric_name"] == "tail_to_previous_quarter_ratio"
    assert cpml["row_state"] == "pass"
    assert cpml["physical_metrics"]["tail_to_previous_quarter_ratio"] <= quick_report["thresholds"]["cpml_tail_growth_limit"]
    _assert_verified_post_source_window(cpml)
    assert cpml["physical_metrics"]["executed_topology_optimize_strategy_b"] is True

    pec = rows_by_family["pec_topology"]
    assert pec["metric_name"] == "tail_to_previous_quarter_ratio"
    assert pec["row_state"] == "pass"
    assert pec["physical_metrics"]["tail_to_previous_quarter_ratio"] <= quick_report["thresholds"]["pec_energy_growth_limit"]
    _assert_verified_post_source_window(pec)
    assert pec["physical_metrics"]["executed_topology_optimize_strategy_b"] is True

    port = rows_by_family["port_proxy"]
    assert port["metric_name"] == "passive_to_excited_power_ratio"
    assert port["row_state"] == "warn"
    assert port["physical_metrics"]["passive_to_excited_power_ratio"] <= quick_report["thresholds"]["passive_gain_limit"]
    assert port["physical_metrics"]["executed_optimize_strategy_b"] is True
    assert port["physical_metrics"]["excited_lumped_port_count"] == 1
    assert port["physical_metrics"]["passive_lumped_port_count"] == 1


def test_post_source_window_excludes_active_source_tail():
    class _Inputs:
        raw_sources = ((0, 0, 0, "ez", np.array([0.0, 1.0, 0.1, 1e-5, 0.0, 0.0, 0.0, 0.0])),)

    series = np.arange(8, dtype=float)
    graded, metrics = phase8._post_source_window(series, _Inputs())

    assert metrics["source_active_last_index"] == 2
    assert metrics["graded_window_start_index"] == 3
    assert np.array_equal(graded, series[3:])
    assert metrics["max_source_abs_in_graded_window"] <= metrics["source_off_threshold_abs"]


def test_topology_growth_oracles_block_when_tail_grows(monkeypatch):
    class _Grid:
        dt = 1e-12

    class _Inputs:
        grid = _Grid()

    class _Sim:
        def build_hybrid_phase1_inputs(self, *, n_steps):
            return _Inputs()

    growing_series = np.concatenate([
        np.zeros(64),
        np.zeros(64),
        np.ones(64),
        np.full(64, 2.0),
    ])

    monkeypatch.setattr(phase8.phase7, "_run_topology_quick", lambda *args, **kwargs: {})
    post_source = {
        "full_sample_count": int(growing_series.size),
        "graded_sample_count": int(growing_series.size),
        "source_peak_abs": 0.0,
        "source_off_threshold_ratio": phase8.TOPOLOGY_SOURCE_OFF_THRESHOLD_RATIO,
        "source_off_threshold_abs": 0.0,
        "source_active_last_index": -1,
        "graded_window_start_index": 0,
        "max_source_abs_in_graded_window": 0.0,
        "min_post_source_samples": phase8.TOPOLOGY_MIN_POST_SOURCE_SAMPLES,
        "source_off_window_verified": True,
    }
    monkeypatch.setattr(
        phase8,
        "_topology_strategy_b_series",
        lambda boundary: (_Sim(), growing_series, 0.0, post_source),
    )

    cpml = phase8._run_cpml_topology_quick(phase8.Thresholds())
    pec = phase8._run_pec_topology_quick(phase8.Thresholds())

    assert cpml["metric_name"] == "tail_to_previous_quarter_ratio"
    assert cpml["measured_value"] > cpml["tolerance"]["cpml_tail_growth_limit"]
    assert cpml["row_state"] == "fail"

    assert pec["metric_name"] == "tail_to_previous_quarter_ratio"
    assert pec["measured_value"] > pec["tolerance"]["pec_energy_growth_limit"]
    assert pec["row_state"] == "fail"


def test_threshold_override_can_block_bad_physics_proxy():
    blocked = phase8.build_phase8_report(
        mode="quick",
        family="port_proxy",
        thresholds=phase8.Thresholds(passive_gain_limit=1e-12),
    )
    assert blocked["rows"][0]["row_state"] == "fail"
    assert blocked["summary"]["family_statuses"]["port_proxy"]["status"] == "physics_blocked"
    assert blocked["summary"]["overall_status"] == "physics_blocked"


def test_full_mode_is_metadata_only_and_not_overpromoted():
    report = phase8.build_phase8_report(mode="full", family="source_probe")
    assert report["rows"][0]["row_state"] == "not_evaluated"
    assert report["rows"][0]["strategy_b_executed"] is False
    assert report["summary"]["overall_status"] == "not_evaluated"
    assert report["fail_closed_audit"]["row_state"] == "not_evaluated"
    statuses = [payload["status"] for payload in report["summary"]["family_statuses"].values()]
    assert "production_ready_limited" not in statuses
    assert report["summary"]["overall_status"] != "production_ready_limited"


def test_cli_emits_json_for_full_metadata_mode():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/phase8_strategy_b_physical_validation.py",
            "--mode",
            "full",
            "--family",
            "source_probe",
            "--indent",
            "0",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["benchmark_contract"] == phase8.BENCHMARK_CONTRACT
    assert payload["rows"][0]["row_state"] == "not_evaluated"
