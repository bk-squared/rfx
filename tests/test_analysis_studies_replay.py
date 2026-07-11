from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from rfx.experiments import (
    DurableStudyService,
    ExperimentService,
    compare_sparameter_runs,
    export_replay_bundle,
    verify_replay_bundle,
)


FIXTURES = Path(__file__).parent / "fixtures" / "experiments"


def _spec(name="patch_antenna_v2.json") -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _run_revision(service, experiment_id, revision_id, key):
    linked = service.submit_revision(
        experiment_id, revision_id=revision_id, idempotency_key=key
    )
    service.start(linked.run.id)
    final = service.wait(linked.run.id, timeout=60)
    assert final.state == "succeeded", final.error
    return final


def test_comparison_cites_runs_metrics_validation_and_limitations(tmp_path):
    service = ExperimentService(tmp_path / "workspace")
    experiment, first = service.create_experiment(_spec())
    second_spec = _spec()
    second_spec["materials"][0]["relative_permittivity"] = 4.0
    second = service.application_repository.create_derived_revision(
        experiment.id,
        parent_revision_id=first.id,
        document=second_spec,
        actor="study:test",
    )
    first_run = _run_revision(service, experiment.id, first.id, "compare:first")
    second_run = _run_revision(service, experiment.id, second.id, "compare:second")

    comparison = compare_sparameter_runs(service, [first_run.id, second_run.id])

    assert comparison["input_run_ids"] == [first_run.id, second_run.id]
    assert comparison["baseline_run_id"] == first_run.id
    assert len(comparison["rows"]) == 2
    assert {item["run_id"] for item in comparison["citations"]} == {
        first_run.id,
        second_run.id,
    }
    assert comparison["metric_definition"]["max_s11_abs"]
    assert comparison["metric_definition"]["field_normalized_l2_delta_vs_baseline"]
    assert comparison["rows"][0]["field_normalized_l2_delta_vs_baseline"] == 0.0
    assert comparison["rows"][1]["field_normalized_l2_delta_vs_baseline"] >= 0.0
    assert (
        sum(
            citation["observable"] == "final-state field plane"
            for citation in comparison["citations"]
        )
        == 2
    )
    assert comparison["validation_status"] == {
        first_run.id: "validated",
        second_run.id: "validated",
    }
    assert any("not-for-quantitative" in item for item in comparison["limitations"])


def test_sweep_restart_reuses_completed_points_and_promotes_best(tmp_path):
    service = ExperimentService(tmp_path / "workspace")
    experiment, base = service.create_experiment(_spec())
    studies = DurableStudyService(service)
    study = studies.create(
        experiment.id,
        kind="sweep",
        parameter_path="/materials/0/relative_permittivity",
        values=[4.0, 4.2],
        objective="minimum_s11_db",
        direction="minimize",
        budget_points=2,
    )

    first = studies.run(study.id)
    assert first["status"] == "succeeded"
    assert first["budget_estimate"] == {
        "backend": "cpu",
        "max_concurrent_runs": 1,
        "budget_points": 2,
        "requested_points": 2,
        "remaining_points": 0,
        "estimated_grid_cells_per_run": 6783,
        "estimated_steps_per_run": 12,
        "estimated_total_cell_steps": 162792,
        "estimate_scope": "structural upper-bound before runtime/JIT overhead",
    }
    assert [point["status"] for point in first["points"]] == [
        "succeeded",
        "succeeded",
    ]
    run_ids = [point["run_id"] for point in first["points"]]
    assert len(set(run_ids)) == 2

    restarted = DurableStudyService(ExperimentService(tmp_path / "workspace"))
    second = restarted.run(study.id)
    assert second["status"] == "succeeded"
    assert second["reused_completed_points"] == 2
    assert [point["run_id"] for point in second["points"]] == run_ids

    promoted = restarted.promote_best(study.id)
    assert promoted["promoted_revision_id"] != base.id
    current = restarted.service.application_repository.get_experiment(experiment.id)
    assert current.current_revision_id == promoted["promoted_revision_id"]


def test_early_stop_and_unsupported_optimization_lane_reject_before_run(tmp_path):
    service = ExperimentService(tmp_path / "workspace")
    patch, _ = service.create_experiment(_spec())
    studies = DurableStudyService(service)
    study = studies.create(
        patch.id,
        kind="optimization",
        parameter_path="/materials/0/relative_permittivity",
        values=[4.0, 4.2, 4.4],
        objective="minimum_s11_db",
        direction="minimize",
        budget_points=3,
        early_stop_value=1e9,
    )
    result = studies.run(study.id)
    assert result["status"] == "succeeded"
    assert [point["status"] for point in result["points"]] == [
        "succeeded",
        "skipped",
        "skipped",
    ]

    wr90, _ = service.create_experiment(_spec("wr90_waveguide_v2.json"))
    run_count = len(service.repository.list_runs())
    with pytest.raises(ValueError, match="unsupported_optimization_lane"):
        studies.create(
            wr90.id,
            kind="optimization",
            parameter_path="/materials/0/relative_permittivity",
            values=[3.9, 4.1],
        )
    assert len(service.repository.list_runs()) == run_count


def test_study_cancellation_is_durable_and_creates_no_unapproved_runs(tmp_path):
    workspace = tmp_path / "workspace"
    service = ExperimentService(workspace)
    experiment, _ = service.create_experiment(_spec())
    studies = DurableStudyService(service)
    study = studies.create(
        experiment.id,
        kind="sweep",
        parameter_path="/materials/0/relative_permittivity",
        values=[4.0, 4.2],
        budget_points=2,
    )

    requested = studies.cancel(study.id)
    assert requested["status"] == "cancellation_requested"
    assert requested["budget_estimate"]["remaining_points"] == 2

    restarted = DurableStudyService(ExperimentService(workspace))
    cancelled = restarted.run(study.id)
    assert cancelled["status"] == "cancelled"
    assert [point["status"] for point in cancelled["points"]] == [
        "cancelled",
        "cancelled",
    ]
    assert cancelled["budget_estimate"]["remaining_points"] == 0
    assert restarted.service.repository.list_runs() == []


def test_bundle_checksum_and_fresh_process_tolerance_replay(tmp_path):
    service = ExperimentService(tmp_path / "source")
    experiment, revision = service.create_experiment(_spec())
    run = _run_revision(service, experiment.id, revision.id, "bundle:source")
    bundle = export_replay_bundle(
        service, run.id, destination=tmp_path / "patch-replay.zip"
    )

    manifest = verify_replay_bundle(bundle)
    assert manifest["schema_version"] == "rfx-replay-bundle/v1"
    assert manifest["metric_baseline"]
    assert manifest["replay_tolerances"]
    assert {item["path"] for item in manifest["files"]} >= {
        "experiment/spec.json",
        "experiment/generated.py",
        "experiment/scene.json",
        "run/events.json",
        "run/runtime.json",
        "analysis/metrics.json",
        "analysis/validation.json",
        "analysis/plots/s11.json",
        "environment/environment.json",
        "artifacts/index.json",
    }
    with zipfile.ZipFile(bundle) as archive:
        artifact_index = json.loads(archive.read("artifacts/index.json"))
        field_entry = next(
            item for item in artifact_index if item["kind"] == "field-slice"
        )
        field_payload = json.loads(archive.read(field_entry["logical_path"]))
        assert field_payload["schema_version"] == "rfx-field-slice-artifact/v1"
        assert field_payload["maximum_absolute"] > 0

    replay_workspace = tmp_path / "fresh-replay-workspace"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "rfx.cli",
            "experiment",
            "replay",
            str(bundle),
            "--workspace",
            str(replay_workspace),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
        env={
            **__import__("os").environ,
            "JAX_PLATFORMS": "cpu",
            "JAX_PLATFORM_NAME": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
        },
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["schema_version"] == "rfx-replay-report/v1"
    assert report["passed"] is True
    assert all(metric["passed"] for metric in report["metrics"].values())
    assert Path(report["source_bundle"]) == bundle

    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(tampered, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name == "analysis/metrics.json":
                data += b"tamper"
            target.writestr(name, data)
    with pytest.raises(ValueError, match="(size|checksum) mismatch"):
        verify_replay_bundle(tampered)

    semantic_tamper = tmp_path / "artifact-index-tampered.zip"
    with zipfile.ZipFile(bundle) as source:
        files = {name: source.read(name) for name in source.namelist()}
    artifact_index = json.loads(files["artifacts/index.json"])
    artifact_index[0]["sha256"] = "0" * 64
    files["artifacts/index.json"] = (
        json.dumps(artifact_index, sort_keys=True, indent=2) + "\n"
    ).encode()
    manifest = json.loads(files["manifest.json"])
    index_manifest = next(
        item for item in manifest["files"] if item["path"] == "artifacts/index.json"
    )
    index_manifest["size_bytes"] = len(files["artifacts/index.json"])
    index_manifest["sha256"] = hashlib.sha256(files["artifacts/index.json"]).hexdigest()
    files["manifest.json"] = (
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    ).encode()
    with zipfile.ZipFile(semantic_tamper, "w") as target:
        for name, data in files.items():
            target.writestr(name, data)
    with pytest.raises(ValueError, match="artifact index digest mismatch"):
        verify_replay_bundle(semantic_tamper)
