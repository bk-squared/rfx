"""Phase IX Strategy B full workload-floor coordinator tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


from scripts import phase9_strategy_b_full_floor as phase9

ROOT = Path(__file__).resolve().parents[1]


def _fake_fail_closed(tmp_path: Path, *, state: str = "pass") -> dict:
    payload = {
        "schema_version": 1,
        "benchmark_contract": "phase_vii_strategy_b_production_readiness",
        "generated_at": "2026-04-26T00:00:00Z",
        "fail_closed_audit": {"row_state": state, "rows": [], "runtime_s": 0.0},
    }
    path = tmp_path / "phase7_quick.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {"path": path, "payload": payload}


def test_family_run_writes_schema_distinct_artifact_topology(tmp_path):
    fail = _fake_fail_closed(tmp_path)
    result = phase9.build_family_artifacts(
        family="source_probe",
        artifact_dir=tmp_path,
        fail_closed_artifact=fail["path"],
    )

    readiness_path = Path(result["paths"]["readiness"])
    physical_path = Path(result["paths"]["physical"])
    assert readiness_path.name == "phase9_source_probe_readiness_full.json"
    assert physical_path.name == "phase9_source_probe_physical_full.json"

    readiness = json.loads(readiness_path.read_text())
    physical = json.loads(physical_path.read_text())
    assert (
        readiness["benchmark_contract"] == "phase_vii_strategy_b_production_readiness"
    )
    assert readiness["contract_status"] == "phase_ix_full_floor_readiness_evidence"
    assert readiness["preserves_phase_vii_metadata_full_mode"] is True
    assert physical["benchmark_contract"] == "phase_viii_strategy_b_physical_validation"
    assert physical["contract_status"] == "phase_ix_full_floor_physical_evidence"
    assert physical["preserves_phase_viii_metadata_full_mode"] is True
    assert (
        physical["readiness_integration"]["does_not_set_production_ready_limited"]
        is True
    )
    assert readiness["fail_closed_evidence"]["row_state"] == "pass"
    assert physical["fail_closed_evidence"]["row_state"] == "pass"


def test_summary_references_artifacts_and_has_no_merged_taxonomy(tmp_path):
    fail = _fake_fail_closed(tmp_path)
    phase9.build_family_artifacts(
        family="source_probe",
        artifact_dir=tmp_path,
        fail_closed_artifact=fail["path"],
    )
    summary = phase9.summarize_artifacts(
        artifact_dir=tmp_path, output=tmp_path / phase9.SUMMARY_FILENAME
    )

    assert summary["merged_status_taxonomy"] is False
    assert "source_probe" in summary["readiness_artifacts"]
    assert "source_probe" in summary["physical_artifacts"]
    assert summary["fail_closed_evidence"]["all_known_sources_pass"] is True
    assert summary["stale_context"] == {}
    assert (tmp_path / phase9.SUMMARY_FILENAME).exists()


def test_metadata_only_or_deferred_rows_cannot_satisfy_full_floor(tmp_path):
    fail = _fake_fail_closed(tmp_path)
    result = phase9.build_family_artifacts(
        family="cpml_topology",
        artifact_dir=tmp_path,
        fail_closed_artifact=fail["path"],
    )
    readiness = result["readiness"]

    assert readiness["rows"][0]["full_floor_execution"]["executed"] is False
    assert readiness["summary"]["metadata_only_rows_satisfy_full_floor"] is False
    assert readiness["summary"]["family_status"] == "experimental_limited"
    assert readiness["summary"]["family_status"] != "production_ready_limited"


def test_execute_workload_path_is_reachable_when_guard_allows(monkeypatch, tmp_path):
    fail = _fake_fail_closed(tmp_path)
    observed = {}

    def fake_execute(family):
        observed["family"] = family
        return phase9.simulated_pass_result(family)

    monkeypatch.setattr(phase9, "_execute_workload_floor", fake_execute)
    result = phase9.build_family_artifacts(
        family="source_probe",
        artifact_dir=tmp_path,
        fail_closed_artifact=fail["path"],
        execute_workload=True,
        max_cell_steps=10**18,
    )

    assert observed == {"family": "source_probe"}
    assert result["readiness"]["rows"][0]["full_floor_execution"]["executed"] is True
    assert result["readiness"]["summary"]["family_status"] == "production_ready_limited"


def test_non_pec_split_source_promotion_requires_executed_full_floor(tmp_path):
    fail = _fake_fail_closed(tmp_path)
    fail_closed = phase9.import_fail_closed_evidence(fail["path"])
    provenance = phase9.build_provenance(
        family="source_probe",
        mode="full",
        command=["test"],
        fail_closed=fail_closed,
        thresholds={"max_cell_steps": 1, "execute_workload": True},
    )
    readiness = phase9.build_readiness_artifact(
        family="source_probe",
        execution=phase9.simulated_pass_result("source_probe"),
        fail_closed=fail_closed,
        provenance=provenance,
        split_source_gradient_state="pass",
    )

    gradient = readiness["rows"][0]["required_gradient_evidence"]
    assert readiness["rows"][0]["full_floor_execution"]["executed"] is True
    assert gradient["split_source_allowed"] is True
    assert gradient["source_artifacts"]["readiness_quick_artifact"]["sha256"]
    assert gradient["source_artifacts"]["readiness_quick_artifact"]["source_contract"]
    assert readiness["summary"]["family_status"] == "production_ready_limited"


def test_split_source_promotion_rejects_descriptive_only_refs(tmp_path):
    fail = _fake_fail_closed(tmp_path)
    fail_closed = phase9.import_fail_closed_evidence(fail["path"])
    provenance = phase9.build_provenance(
        family="source_probe",
        mode="full",
        command=["test"],
        fail_closed=fail_closed,
        thresholds={"max_cell_steps": 1, "execute_workload": True},
    )
    readiness = phase9.build_readiness_artifact(
        family="source_probe",
        execution=phase9.simulated_pass_result("source_probe"),
        fail_closed=fail_closed,
        provenance=provenance,
        split_source_gradient_state="pass",
        split_source_refs={"note": "descriptive only"},
    )

    gradient = readiness["rows"][0]["required_gradient_evidence"]
    assert gradient["source_artifact_provenance_valid"] is False
    assert "no_source_artifacts" in gradient["source_artifact_provenance_errors"]
    assert readiness["summary"]["family_status"] == "experimental_limited"
    assert readiness["summary"]["family_status"] != "production_ready_limited"


def test_split_source_promotion_uses_referenced_artifact_content(tmp_path):
    fail = _fake_fail_closed(tmp_path)
    fail_closed = phase9.import_fail_closed_evidence(fail["path"])
    source = tmp_path / "source_with_failing_gradient.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark_contract": "phase_vii_strategy_b_production_readiness",
                "generated_at": "2026-04-26T00:00:00Z",
                "rows": [
                    {
                        "family": "source_probe",
                        "row_state": "pass",
                        "required_gradient_evidence": {
                            "required": True,
                            "state": "fail",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    provenance = phase9.build_provenance(
        family="source_probe",
        mode="full",
        command=["test"],
        fail_closed=fail_closed,
        thresholds={"max_cell_steps": 1, "execute_workload": True},
    )
    readiness = phase9.build_readiness_artifact(
        family="source_probe",
        execution=phase9.simulated_pass_result("source_probe"),
        fail_closed=fail_closed,
        provenance=provenance,
        split_source_refs={
            "readiness_quick_artifact": phase9.source_artifact_reference(source)
        },
    )

    gradient = readiness["rows"][0]["required_gradient_evidence"]
    assert gradient["state"] == "not_evaluated"
    assert (
        "readiness_quick_artifact.required_gradient_not_pass:source_probe"
        in gradient["source_artifact_provenance_errors"]
    )
    assert readiness["summary"]["family_status"] == "experimental_limited"


def test_fail_closed_rerun_failure_precedence_over_imported_pass(tmp_path):
    imported = phase9.import_fail_closed_evidence(
        _fake_fail_closed(tmp_path, state="pass")["path"]
    )
    rerun = {
        "mode": "rerun",
        "row_state": "fail",
        "id": "rerun-fail",
        "audit": {"row_state": "fail", "rows": []},
    }

    chosen = phase9.choose_fail_closed_evidence(
        source="import", imported=imported, rerun=rerun
    )
    assert chosen["mode"] == "rerun"
    assert chosen["row_state"] == "fail"
    assert phase9.fail_closed_pass(chosen) is False


def test_pec_remains_limited_without_representative_floor(tmp_path):
    fail = _fake_fail_closed(tmp_path)
    result = phase9.build_family_artifacts(
        family="pec_topology",
        artifact_dir=tmp_path,
        fail_closed_artifact=fail["path"],
    )
    readiness = result["readiness"]

    assert readiness["rows"][0]["full_floor_execution"]["status"] == "defer_limited"
    assert readiness["summary"]["family_status"] == "experimental_limited"
    assert readiness["summary"]["family_status"] != "production_ready_limited"


def test_stale_provenance_is_reported_in_summary(tmp_path):
    fail = _fake_fail_closed(tmp_path)
    result = phase9.build_family_artifacts(
        family="source_probe",
        artifact_dir=tmp_path,
        fail_closed_artifact=fail["path"],
    )
    readiness_path = Path(result["paths"]["readiness"])
    payload = json.loads(readiness_path.read_text())
    payload["provenance"]["thresholds"]["max_cell_steps"] = -1
    readiness_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = phase9.summarize_artifacts(artifact_dir=tmp_path)
    assert "source_probe:readiness" in summary["stale_context"]
    assert "provenance_hash" in summary["stale_context"]["source_probe:readiness"]


def test_stale_run_artifact_is_quarantined_before_overwrite(tmp_path):
    fail = _fake_fail_closed(tmp_path)
    result = phase9.build_family_artifacts(
        family="source_probe",
        artifact_dir=tmp_path,
        fail_closed_artifact=fail["path"],
    )
    readiness_path = Path(result["paths"]["readiness"])
    payload = json.loads(readiness_path.read_text())
    payload["provenance"]["thresholds"]["max_cell_steps"] = -1
    readiness_path.write_text(json.dumps(payload), encoding="utf-8")

    rerun = phase9.build_family_artifacts(
        family="source_probe",
        artifact_dir=tmp_path,
        fail_closed_artifact=fail["path"],
    )

    stale = rerun["stale_context"]["readiness"]
    assert "thresholds" in stale["mismatches"]
    assert Path(stale["quarantined_path"]).exists()
    assert "stale-" in Path(stale["quarantined_path"]).name
    assert (
        json.loads(readiness_path.read_text())["provenance"]["thresholds"][
            "max_cell_steps"
        ]
        == phase9.DEFAULT_MAX_CELL_STEPS
    )


def test_validate_provenance_detects_requested_threshold_mismatch(tmp_path):
    fail = _fake_fail_closed(tmp_path)
    result = phase9.build_family_artifacts(
        family="source_probe",
        artifact_dir=tmp_path,
        fail_closed_artifact=fail["path"],
    )
    artifact = json.loads(Path(result["paths"]["readiness"]).read_text())
    expected = dict(artifact["provenance"])
    expected["thresholds"] = {**expected["thresholds"], "max_cell_steps": -1}

    assert "thresholds" in phase9.validate_provenance(artifact, expected)


def test_validate_provenance_detects_floor_and_worktree_signature_mismatch(tmp_path):
    fail = _fake_fail_closed(tmp_path)
    result = phase9.build_family_artifacts(
        family="source_probe",
        artifact_dir=tmp_path,
        fail_closed_artifact=fail["path"],
    )
    artifact = json.loads(Path(result["paths"]["readiness"]).read_text())
    expected = json.loads(json.dumps(artifact["provenance"]))
    expected["domain_m"] = [9, 9, 9]
    expected["dx_m"] = -1
    expected["workload_floor"]["freq_max_hz"] = -1
    expected["worktree_signature"] = {
        **expected["worktree_signature"],
        "dirty": not expected["worktree_signature"]["dirty"],
        "status_sha256": "bad",
    }

    mismatches = phase9.validate_provenance(artifact, expected)
    assert "domain_m" in mismatches
    assert "dx_m" in mismatches
    assert "workload_floor" in mismatches
    assert "worktree_signature.dirty" in mismatches
    assert "worktree_signature.status_sha256" in mismatches


def test_port_proxy_physical_promotion_requires_passive_no_gain_metric(tmp_path):
    fail_closed = phase9.import_fail_closed_evidence(
        _fake_fail_closed(tmp_path)["path"]
    )
    provenance = phase9.build_provenance(
        family="port_proxy",
        mode="full",
        command=["test"],
        fail_closed=fail_closed,
        thresholds={"max_cell_steps": 1, "execute_workload": True},
    )
    physical = phase9.build_physical_artifact(
        family="port_proxy",
        execution=phase9.simulated_pass_result("port_proxy"),
        fail_closed=fail_closed,
        provenance=provenance,
    )

    assert physical["summary"]["family_status"] == "physics_experimental"
    assert physical["rows"][0]["row_state"] == "not_evaluated"
    assert (
        physical["rows"][0]["evidence_strength"]
        == "representative_full_missing_physical_oracle"
    )


def test_port_proxy_physical_promotion_allows_passive_no_gain_metric(tmp_path):
    fail_closed = phase9.import_fail_closed_evidence(
        _fake_fail_closed(tmp_path)["path"]
    )
    provenance = phase9.build_provenance(
        family="port_proxy",
        mode="full",
        command=["test"],
        fail_closed=fail_closed,
        thresholds={"max_cell_steps": 1, "execute_workload": True},
    )
    execution = phase9.simulated_pass_result("port_proxy")
    execution.metrics.update(
        {
            "passive_no_gain_pass": True,
            "passive_to_excited_power_ratio": 0.5,
        }
    )
    physical = phase9.build_physical_artifact(
        family="port_proxy",
        execution=execution,
        fail_closed=fail_closed,
        provenance=provenance,
    )

    assert physical["summary"]["family_status"] == "physics_validated_limited"
    assert physical["rows"][0]["row_state"] == "pass"


def test_cli_family_and_summary_commands(tmp_path):
    fail = _fake_fail_closed(tmp_path)
    subprocess.run(
        [
            sys.executable,
            "scripts/phase9_strategy_b_full_floor.py",
            "--family",
            "source_probe",
            "--mode",
            "full",
            "--artifact-dir",
            str(tmp_path),
            "--fail-closed-artifact",
            str(fail["path"]),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/phase9_strategy_b_full_floor.py",
            "--summarize",
            "--artifact-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "summary.json"),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    summary = json.loads(proc.stdout)
    assert summary["merged_status_taxonomy"] is False
    assert "source_probe" in summary["readiness_artifacts"]
    assert (tmp_path / "summary.json").exists()
