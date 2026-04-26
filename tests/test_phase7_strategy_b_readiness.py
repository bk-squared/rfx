"""Phase VII Strategy B production-readiness harness tests."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "phase7_strategy_b_readiness.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase7_strategy_b_readiness", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(*, state="pass", mode="quick", gradient_state="pass", meets_floor=False):
    return {
        "row_state": state,
        "mode": mode,
        "workload_floor": {"meets_full_floor": meets_floor},
        "required_gradient_evidence": {"required": True, "state": gradient_state},
    }


def test_phase7_classifier_keeps_quick_only_evidence_experimental():
    module = _load_module()

    status = module.classify_family([_row(state="pass", mode="quick", gradient_state="pass")])

    assert status["status"] == "experimental_limited"
    assert "missing full-mode workload-floor evidence" in status["reasons"]


def test_phase7_classifier_requires_gradient_for_limited_production_status():
    module = _load_module()

    missing_gradient = module.classify_family(
        [_row(state="pass", mode="full", gradient_state="not_evaluated", meets_floor=True)]
    )
    gradient_passed = module.classify_family(
        [_row(state="pass", mode="full", gradient_state="pass", meets_floor=True)]
    )
    failed = module.classify_family([_row(state="fail", mode="full", gradient_state="pass", meets_floor=True)])

    assert missing_gradient["status"] == "experimental_limited"
    assert "missing required gradient evidence" in missing_gradient["reasons"]
    assert gradient_passed["status"] == "production_ready_limited"
    assert failed["status"] == "blocked"


def test_phase7_classifier_warn_and_not_evaluated_contracts_do_not_promote():
    module = _load_module()

    warn_status = module.classify_family(
        [_row(state="warn", mode="full", gradient_state="pass", meets_floor=False)]
    )
    not_evaluated_status = module.classify_family(
        [_row(state="not_evaluated", mode="full", gradient_state="not_evaluated", meets_floor=True)]
    )

    assert warn_status["status"] == "experimental_limited"
    assert not_evaluated_status["status"] == "not_evaluated"


def test_phase7_quick_report_schema_runtime_rows_and_fail_closed_audit():
    module = _load_module()

    report = module.build_phase7_report(mode="quick")

    assert report["schema_version"] == 1
    assert report["benchmark_contract"] == "phase_vii_strategy_b_production_readiness"
    assert report["contract_status"] == "phase_vii_readiness_evidence"
    assert report["preserves_phase_iii_gate0_contract"] is True
    assert report["baseline_commit"] == "eeb365e"

    rows = report["rows"]
    assert {row["family"] for row in rows} == {
        "source_probe",
        "cpml_topology",
        "pec_topology",
        "port_proxy",
    }
    required = set(module.REQUIRED_ROW_FIELDS)
    assert all(required <= set(row) for row in rows)
    assert report["summary"]["rows_with_required_fields"] == len(rows)

    for row in rows:
        assert row["mode"] == "quick"
        assert row["row_state"] == "pass"
        assert row["n_steps"] == 8
        assert row["checkpoint_every"] == 3
        assert row["correctness_metric"] <= report["thresholds"]["parity"]
        assert row["required_gradient_evidence"]["state"] == "pass"
        assert row["strategy_b_estimated_memory_gb"] < row["strategy_a_estimated_memory_gb"]
        assert row["workload_floor"]["meets_full_floor"] is False

    assert report["fail_closed_audit"]["row_state"] == "pass"
    assert {row["row_state"] for row in report["fail_closed_audit"]["rows"]} == {"pass"}
    assert {row["audit_kind"] for row in report["fail_closed_audit"]["rows"]} == {
        "explicit_strategy_b_raise"
    }
    assert len(report["fail_closed_audit"]["rows"]) >= 10
    assert {
        "nonuniform_strategy_b",
        "ntff_directivity_strategy_b",
        "generic_multi_port_strategy_b",
        "excited_port_design_region_overlap_auto_strategy_b",
        "port_design_region_overlap_auto_strategy_b",
        "wire_port_strategy_b",
        "waveguide_port_strategy_b",
        "floquet_port_strategy_b",
        "debye_strategy_b",
        "lorentz_strategy_b",
        "mixed_dispersion_strategy_b",
        "topology_pec_occupancy_strategy_b",
    } <= {row["case_id"] for row in report["fail_closed_audit"]["rows"]}
    assert report["summary"]["overall_status"] == "experimental_limited"
    assert all(
        payload["status"] == "experimental_limited"
        for payload in report["summary"]["family_statuses"].values()
    )


def test_phase7_full_mode_records_locked_floor_without_promoting():
    module = _load_module()

    report = module.build_phase7_report(mode="full", family="source_probe")
    row = report["rows"][0]

    assert row["family"] == "source_probe"
    assert row["row_state"] == "not_evaluated"
    assert row["workload_floor"]["case_id"] == "source_probe_optimize_patch"
    assert row["workload_floor"]["boundary"] == "cpml"
    assert row["n_steps"] >= 10_000
    assert row["checkpoint_every"] >= 1_000
    assert row["workload_floor"]["meets_full_floor"] is True
    assert report["summary"]["overall_status"] == "not_evaluated"
    assert "command_to_evaluate" not in row
    assert "--execute-full" not in json.dumps(row)
    assert row["full_evidence_requirement"]["status"] == "requires_split_run_floor_execution"
    assert row["full_evidence_requirement"]["minimum_floor"]["case_id"] == "source_probe_optimize_patch"
    assert "no full workload-floor execution is claimed" in row["evidence"]["full_mode_note"]


def test_phase7_cli_quick_emits_machine_readable_json():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", "quick", "--indent", "0"],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)

    assert report["summary"]["overall_status"] == "experimental_limited"
    assert report["summary"]["fail_closed_audit_state"] == "pass"
    assert report["summary"]["production_ready_limited_requires_full_mode"] is True
