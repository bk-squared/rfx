"""Phase X Strategy B production-promotion decision tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import phase10_strategy_b_production_promotion as phase10
from scripts import phase9_strategy_b_full_floor as phase9

ROOT = Path(__file__).resolve().parents[1]


def _clean_worktree(monkeypatch):
    monkeypatch.setattr(
        phase9,
        "worktree_signature",
        lambda: {
            "head": "test-clean",
            "dirty": False,
            "status_sha256": phase9.stable_json_hash(""),
        },
    )


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


def _fake_split_source_ref(tmp_path: Path, *, family: str, state: str = "pass") -> dict:
    payload = {
        "schema_version": 1,
        "benchmark_contract": "phase_vii_strategy_b_production_readiness",
        "generated_at": "2026-04-26T00:00:00Z",
        "rows": [
            {
                "family": family,
                "row_state": "pass",
                "required_gradient_evidence": {"required": True, "state": state},
            }
        ],
    }
    path = tmp_path / f"{family}_split_source.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "path": path,
        "payload": payload,
        "ref": {"readiness_quick_artifact": phase9.source_artifact_reference(path)},
    }


def _optimized_topology_replay_metrics(
    *,
    supported: bool = True,
    finite: bool = True,
    beta_matches: bool = True,
    material_consistent: bool = True,
    oracle_source: bool = True,
) -> dict:
    return {
        "topology_replay_mode": phase9.TOPOLOGY_REPLAY_MODE,
        "topology_replay_density_source": phase9.TOPOLOGY_REPLAY_DENSITY_SOURCE,
        "topology_replay_supported": supported,
        "topology_replay_finite": finite,
        "topology_replay_beta_matches_optimizer": beta_matches,
        "topology_replay_material_consistency_passed": material_consistent,
        "topology_replay_source_is_required_physical_oracle": oracle_source,
        "topology_replay_material_consistency_tolerance": (
            phase9.TOPOLOGY_REPLAY_MATERIAL_ATOL
        ),
    }


def _execution(
    family: str, *, physical_oracle: bool = True
) -> phase9.FloorExecutionResult:
    execution = phase9.simulated_pass_result(family)
    if not physical_oracle:
        return execution
    if family == "source_probe":
        execution.metrics.update(
            {
                "source_probe_causal_full_floor_pass": True,
                "causal_margin_steps": 2.0,
                "source_probe_distance_m": 0.01,
                "dt_s": 1e-12,
            }
        )
    elif family == "cpml_topology":
        execution.metrics.update(
            {
                "cpml_topology_tail_full_floor_pass": True,
                "tail_to_previous_quarter_ratio": 0.8,
                "cpml_tail_growth_limit": 1.1,
                "post_source_window": {"source_off_window_verified": True},
                **_optimized_topology_replay_metrics(),
            }
        )
    elif family == "pec_topology":
        execution.metrics.update(
            {
                "pec_topology_bounded_full_floor_pass": True,
                "tail_to_previous_quarter_ratio": 0.9,
                "tail_to_total_energy_ratio": 0.25,
                "pec_energy_growth_limit": 1.1,
                "quarter_energy": [1.0, 1.0, 1.0, 0.9],
                "ratio_metrics_finite": True,
                "post_source_window": {"source_off_window_verified": True},
                **_optimized_topology_replay_metrics(),
            }
        )
    elif family == "port_proxy":
        execution.metrics.update(
            {
                "passive_no_gain_pass": True,
                "passive_to_excited_power_ratio": 0.6,
                "passive_gain_limit": 1.25,
            }
        )
    return execution


def _build_family(
    tmp_path: Path,
    family: str,
    *,
    physical_oracle: bool = True,
    split_source_state: str = "pass",
    execution: phase9.FloorExecutionResult | None = None,
) -> dict:
    fail = _fake_fail_closed(tmp_path)
    source = _fake_split_source_ref(tmp_path, family=family, state=split_source_state)
    result = phase9.build_family_artifacts(
        family=family,
        artifact_dir=tmp_path,
        fail_closed_artifact=fail["path"],
        execution=execution or _execution(family, physical_oracle=physical_oracle),
        execute_workload=True,
        max_cell_steps=10**18,
        split_source_refs=source["ref"],
    )
    phase9.summarize_artifacts(
        artifact_dir=tmp_path, output=tmp_path / phase9.SUMMARY_FILENAME
    )
    return {"phase9": result, "source": source, "fail": fail}


def _decision(tmp_path: Path, family: str) -> dict:
    return phase10.build_promotion_decision(artifact_dir=tmp_path, families=(family,))[
        "decisions"
    ][family]


def test_missing_phase9_artifacts_block_promotion(tmp_path):
    decision = _decision(tmp_path, "source_probe")

    assert decision["decision"] == "not_evaluated"
    assert "missing_readiness_artifact" in decision["failed_gates"]
    assert "missing_physical_artifact" in decision["failed_gates"]


def test_deferred_phase9_artifacts_do_not_promote(monkeypatch, tmp_path):
    _clean_worktree(monkeypatch)
    fail = _fake_fail_closed(tmp_path)
    phase9.build_family_artifacts(
        family="source_probe",
        artifact_dir=tmp_path,
        fail_closed_artifact=fail["path"],
        split_source_refs=_fake_split_source_ref(tmp_path, family="source_probe")[
            "ref"
        ],
    )
    phase9.summarize_artifacts(
        artifact_dir=tmp_path, output=tmp_path / phase9.SUMMARY_FILENAME
    )

    decision = _decision(tmp_path, "source_probe")

    assert decision["decision"] == "experimental_limited"
    assert "readiness_full_floor_not_executed" in decision["failed_gates"]


def test_candidate_family_stale_phase9_summary_blocks_that_family_promotion(
    monkeypatch, tmp_path
):
    _clean_worktree(monkeypatch)
    _build_family(tmp_path, "source_probe")
    readiness_path = phase9.artifact_paths(tmp_path, "source_probe")["readiness"]
    payload = json.loads(readiness_path.read_text())
    payload["provenance"]["thresholds"]["max_cell_steps"] = -1
    readiness_path.write_text(json.dumps(payload), encoding="utf-8")

    decision = _decision(tmp_path, "source_probe")

    assert decision["decision"] == "blocked"
    assert "candidate_family_stale_provenance" in decision["failed_gates"]


def test_stale_split_source_ref_blocks_candidate_family_promotion(
    monkeypatch, tmp_path
):
    _clean_worktree(monkeypatch)
    built = _build_family(tmp_path, "source_probe")
    source_payload = built["source"]["payload"]
    source_payload["rows"][0]["required_gradient_evidence"]["state"] = "fail"
    built["source"]["path"].write_text(json.dumps(source_payload), encoding="utf-8")

    decision = _decision(tmp_path, "source_probe")

    assert decision["decision"] == "blocked"
    assert "candidate_family_stale_provenance" in decision["failed_gates"]


def test_unrelated_family_stale_context_is_warning_not_candidate_blocker(
    monkeypatch, tmp_path
):
    _clean_worktree(monkeypatch)
    _build_family(tmp_path, "source_probe")
    _build_family(tmp_path, "cpml_topology")
    cpml_path = phase9.artifact_paths(tmp_path, "cpml_topology")["readiness"]
    payload = json.loads(cpml_path.read_text())
    payload["provenance"]["thresholds"]["max_cell_steps"] = -1
    cpml_path.write_text(json.dumps(payload), encoding="utf-8")
    phase9.summarize_artifacts(
        artifact_dir=tmp_path, output=tmp_path / phase9.SUMMARY_FILENAME
    )

    decision = _decision(tmp_path, "source_probe")

    assert decision["decision"] == "promoted_limited"
    assert "unrelated_family_stale_context_present" in decision["warnings"]
    assert decision["unrelated_stale_context"]


def test_phasex_consumes_phase9_debug_only_dirty_worktree_status_and_blocks_promotion(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        phase9,
        "worktree_signature",
        lambda: {"head": "test-dirty", "dirty": True, "status_sha256": "dirty"},
    )
    _build_family(tmp_path, "source_probe")

    decision = _decision(tmp_path, "source_probe")

    assert decision["decision"] == "blocked"
    assert "readiness_debug_only_dirty_worktree" in decision["failed_gates"]


def test_split_source_artifact_content_is_required_for_promotion(monkeypatch, tmp_path):
    _clean_worktree(monkeypatch)
    _build_family(tmp_path, "source_probe", split_source_state="fail")

    decision = _decision(tmp_path, "source_probe")

    assert decision["decision"] == "blocked"
    assert "split_source_evidence_not_valid" in decision["failed_gates"]


def test_source_probe_requires_full_floor_physical_oracle(monkeypatch, tmp_path):
    _clean_worktree(monkeypatch)
    _build_family(tmp_path, "source_probe", physical_oracle=False)

    decision = _decision(tmp_path, "source_probe")

    assert decision["decision"] == "blocked"
    assert "family_specific_physical_oracle_not_valid" in decision["failed_gates"]


def test_cpml_topology_requires_full_floor_physical_oracle(monkeypatch, tmp_path):
    _clean_worktree(monkeypatch)
    _build_family(tmp_path, "cpml_topology", physical_oracle=False)

    decision = _decision(tmp_path, "cpml_topology")

    assert decision["decision"] == "blocked"
    assert "family_specific_physical_oracle_not_valid" in decision["failed_gates"]


def test_port_proxy_requires_passive_no_gain_full_floor_oracle(monkeypatch, tmp_path):
    _clean_worktree(monkeypatch)
    _build_family(tmp_path, "port_proxy", physical_oracle=False)

    decision = _decision(tmp_path, "port_proxy")

    assert decision["decision"] == "blocked"
    assert "family_specific_physical_oracle_not_valid" in decision["failed_gates"]


def test_pec_topology_deferred_floor_does_not_promote(monkeypatch, tmp_path):
    _clean_worktree(monkeypatch)
    fail = _fake_fail_closed(tmp_path)
    phase9.build_family_artifacts(
        family="pec_topology",
        artifact_dir=tmp_path,
        fail_closed_artifact=fail["path"],
        split_source_refs=_fake_split_source_ref(tmp_path, family="pec_topology")[
            "ref"
        ],
    )
    phase9.summarize_artifacts(
        artifact_dir=tmp_path, output=tmp_path / phase9.SUMMARY_FILENAME
    )

    decision = _decision(tmp_path, "pec_topology")

    assert decision["decision"] == "experimental_limited"
    assert "readiness_full_floor_not_executed" in decision["failed_gates"]


def test_pec_family_subset_is_selected_eligible_after_phase_xii(monkeypatch, tmp_path):
    _clean_worktree(monkeypatch)
    fail = _fake_fail_closed(tmp_path)
    phase9.build_family_artifacts(
        family="pec_topology",
        artifact_dir=tmp_path,
        fail_closed_artifact=fail["path"],
        split_source_refs=_fake_split_source_ref(tmp_path, family="pec_topology")[
            "ref"
        ],
    )

    decision = phase10.build_promotion_decision(
        artifact_dir=tmp_path, families=("pec_topology",)
    )

    assert decision["summary"]["selected_eligible_families"] == ["pec_topology"]
    assert decision["summary"]["all_eligible_promoted"] is False


def test_pec_topology_requires_full_floor_physical_oracle(monkeypatch, tmp_path):
    _clean_worktree(monkeypatch)
    _build_family(tmp_path, "pec_topology", physical_oracle=False)

    decision = _decision(tmp_path, "pec_topology")

    assert decision["decision"] == "blocked"
    assert "family_specific_physical_oracle_not_valid" in decision["failed_gates"]


def test_topology_base_only_physical_oracle_does_not_promote(monkeypatch, tmp_path):
    _clean_worktree(monkeypatch)
    execution = phase9.simulated_pass_result("cpml_topology")
    execution.metrics.update(
        {
            "cpml_topology_tail_full_floor_pass": True,
            "tail_to_previous_quarter_ratio": 0.8,
            "cpml_tail_growth_limit": 1.1,
            "post_source_window": {"source_off_window_verified": True},
        }
    )
    _build_family(tmp_path, "cpml_topology", execution=execution)

    decision = _decision(tmp_path, "cpml_topology")

    assert decision["decision"] == "blocked"
    assert "topology_optimized_density_replay_not_valid" in decision["failed_gates"]


def test_topology_invalid_optimized_replay_policy_does_not_promote(
    monkeypatch, tmp_path
):
    _clean_worktree(monkeypatch)
    execution = phase9.simulated_pass_result("pec_topology")
    execution.metrics.update(
        {
            "pec_topology_bounded_full_floor_pass": True,
            "tail_to_previous_quarter_ratio": 0.9,
            "tail_to_total_energy_ratio": 0.25,
            "pec_energy_growth_limit": 1.1,
            "quarter_energy": [1.0, 1.0, 1.0, 0.9],
            "ratio_metrics_finite": True,
            "post_source_window": {"source_off_window_verified": True},
            **_optimized_topology_replay_metrics(
                finite=False, material_consistent=False
            ),
        }
    )
    _build_family(tmp_path, "pec_topology", execution=execution)

    decision = _decision(tmp_path, "pec_topology")

    assert decision["decision"] == "blocked"
    assert "topology_optimized_density_replay_not_valid" in decision["failed_gates"]


def test_pec_topology_nonrepresentative_guard_remains_experimental_fallback(
    monkeypatch, tmp_path
):
    _clean_worktree(monkeypatch)
    original_floor_for_family = phase9._floor_for_family

    def nonrepresentative_floor(family: str):
        floor = original_floor_for_family(family)
        if family == "pec_topology":
            return {**floor, "representative": False}
        return floor

    monkeypatch.setattr(phase9, "_floor_for_family", nonrepresentative_floor)
    fail = _fake_fail_closed(tmp_path)
    phase9.build_family_artifacts(
        family="pec_topology",
        artifact_dir=tmp_path,
        fail_closed_artifact=fail["path"],
        split_source_refs=_fake_split_source_ref(tmp_path, family="pec_topology")[
            "ref"
        ],
    )

    decision = _decision(tmp_path, "pec_topology")

    assert decision["decision"] == "experimental_limited"
    assert "representative_pec_topology_floor_missing" in decision["failed_gates"]


def test_simulated_executed_pass_promotes_all_eligible_families(monkeypatch, tmp_path):
    _clean_worktree(monkeypatch)
    for family in phase10.PROMOTION_ELIGIBLE_FAMILIES:
        _build_family(tmp_path, family)

    decision = phase10.build_promotion_decision(artifact_dir=tmp_path)

    assert set(decision["summary"]["promoted_families"]) == set(
        phase10.PROMOTION_ELIGIBLE_FAMILIES
    )
    assert decision["summary"]["all_eligible_promoted"] is True
    assert decision["decisions"]["pec_topology"]["decision"] == "promoted_limited"


def test_cli_writes_promotion_decision_artifact_without_promoting_dirty_candidate(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        phase9,
        "worktree_signature",
        lambda: {"head": "test-dirty", "dirty": True, "status_sha256": "dirty"},
    )
    _build_family(tmp_path, "source_probe")
    output = tmp_path / "phase10.json"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/phase10_strategy_b_production_promotion.py",
            "--family",
            "source_probe",
            "--artifact-dir",
            str(tmp_path),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["decisions"]["source_probe"]["decision"] == "blocked"
    assert output.exists()


def test_require_promotion_exits_nonzero_when_any_required_family_blocked(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/phase10_strategy_b_production_promotion.py",
            "--family",
            "source_probe",
            "--artifact-dir",
            str(tmp_path),
            "--require-promotion",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 1
