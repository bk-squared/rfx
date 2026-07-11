from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfx.experiments import (
    ExperimentService,
    ResourceBusyError,
    analyze_run,
    export_replay_bundle,
    replay_bundle,
    verify_field_slice_artifact,
    verify_reflection_transmission_artifact,
    verify_replay_bundle,
    verify_sparameters_artifact,
)


FIXTURE = Path(__file__).parent / "fixtures" / "experiments" / "patch_antenna_v2.json"


def _document() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _submitted(service: ExperimentService, document: dict | None = None):
    experiment, revision = service.create_experiment(document or _document())
    linked = service.submit_revision(
        experiment.id,
        revision_id=revision.id,
        idempotency_key=f"run-{revision.id}",
    )
    return experiment, revision, linked


def test_v2_revision_linked_worker_survives_service_restart(tmp_path):
    workspace = tmp_path / "workspace"
    service = ExperimentService(workspace)
    experiment, revision, linked = _submitted(service)

    service.start(linked.run.id)
    final = service.wait(linked.run.id, timeout=60)

    assert final.state == "succeeded", final.error
    reopened = ExperimentService(workspace)
    durable = reopened.application_repository.get_linked_run(final.id)
    assert durable.experiment_id == experiment.id
    assert durable.revision_id == revision.id
    assert durable.progress == 1.0
    assert (
        reopened.application_repository.get_revision(revision.id).spec_sha256
        == final.spec_sha256
    )
    artifact_kinds = {
        item.kind for item in reopened.application_repository.list_artifacts(final.id)
    }
    assert {"s11", "field-slice", "stdout-log", "stderr-log"} <= artifact_kinds
    field_record = next(
        item
        for item in reopened.application_repository.list_artifacts(final.id)
        if item.kind == "field-slice"
    )
    field = verify_field_slice_artifact(Path(field_record.path).parent)
    field_payload = json.loads(field.data_json.read_text(encoding="utf-8"))
    assert field_payload["component"] == "ez"
    assert field_payload["slice_axis"] == "z"
    assert field_payload["runtime"]["backend"] == "cpu"
    runtime = field_payload["runtime"]
    assert runtime["python_version"]
    assert runtime["platform"]
    assert runtime["packages"]["rfx-fdtd"] == "1.6.6"
    assert runtime["source"]["kind"] in {"source-checkout", "wheel"}
    assert runtime["seed"]["value"] is None
    assert "no stochastic operator" in runtime["seed"]["policy"]
    assert field_payload["maximum_absolute"] > 0
    assert len(field_payload["values"]) == field_payload["shape"][0]
    assert len(field_payload["values"][0]) == field_payload["shape"][1]
    assert reopened.repository.list_events(final.id)[-1].event_type == "run_succeeded"


def test_browser_json_integer_frequency_round_trip_runs_on_jax_cpu(tmp_path):
    document = _document()
    # JSON.stringify drops the decimal marker from whole-valued JavaScript
    # numbers.  The canonical worker must still pass frequencies to JAX as
    # floating-point values instead of overflowing its default int32 scalar.
    s11 = document["observations"][0]
    s11["start_hz"] = 1_800_000_000
    s11["stop_hz"] = 3_000_000_000
    service = ExperimentService(tmp_path / "workspace")
    _, _, linked = _submitted(service, document)

    service.start(linked.run.id)
    final = service.wait(linked.run.id, timeout=60)

    assert final.state == "succeeded", final.error
    assert any(
        artifact.kind == "s11"
        for artifact in service.application_repository.list_artifacts(final.id)
    )


@pytest.mark.parametrize(
    ("fixture_name", "primary_kind", "schema_version", "metrics"),
    [
        (
            "wr90_waveguide_v2.json",
            "sparameters",
            "rfx-sparameters-artifact/v1",
            {"max_s11_abs", "min_s21_abs"},
        ),
        (
            "multilayer_fresnel_v2.json",
            "reflection-transmission",
            "rfx-reflection-transmission-artifact/v1",
            {
                "mean_reflectance_error",
                "mean_transmittance_error",
                "mean_energy_closure_error",
            },
        ),
    ],
)
def test_non_patch_golden_workflow_executes_analyzes_and_replays(
    tmp_path, fixture_name, primary_kind, schema_version, metrics
):
    document = json.loads(FIXTURE.with_name(fixture_name).read_text(encoding="utf-8"))
    service = ExperimentService(tmp_path / "source")
    run = service.run_sync(document)
    assert run.state == "succeeded", run.error
    artifacts = service.application_repository.list_artifacts(run.id)
    assert {primary_kind, "field-slice"} <= {item.kind for item in artifacts}
    primary = next(item for item in artifacts if item.kind == primary_kind)
    if primary_kind == "sparameters":
        verified = verify_sparameters_artifact(Path(primary.path).parent)
    else:
        verified = verify_reflection_transmission_artifact(Path(primary.path).parent)
    assert (
        json.loads(verified.data_json.read_text(encoding="utf-8"))["schema_version"]
        == schema_version
    )
    analysis = analyze_run(service, run.id)
    assert metrics <= set(analysis["metrics"])
    assert analysis["citations"][0]["artifact_id"] == primary.id

    bundle = export_replay_bundle(
        service, run.id, destination=tmp_path / f"{primary_kind}.zip"
    )
    manifest = verify_replay_bundle(bundle)
    assert manifest["workflow"] == document["kind"]
    replayed = replay_bundle(bundle, tmp_path / "replayed")
    assert replayed["passed"] is True
    assert set(replayed["metrics"]) == set(analysis["metrics"])


def test_worker_digest_crash_is_failed_once_with_traceback_artifact(tmp_path):
    service = ExperimentService(tmp_path / "workspace")
    _, _, linked = _submitted(service)
    spec_path = service.runs_root / linked.run.id / "spec.json"
    document = json.loads(spec_path.read_text(encoding="utf-8"))
    document["metadata"]["title"] = "tampered after durable submission"
    spec_path.write_text(json.dumps(document), encoding="utf-8")

    service.start(linked.run.id)
    final = service.wait(linked.run.id, timeout=60)

    assert final.state == "failed"
    events = service.repository.list_events(final.id)
    terminal = [
        event for event in events if event.state in {"failed", "cancelled", "succeeded"}
    ]
    assert len(terminal) == 1
    artifacts = service.application_repository.list_artifacts(final.id)
    traceback_artifact = next(item for item in artifacts if item.kind == "traceback")
    traceback_text = Path(traceback_artifact.path).read_text(encoding="utf-8")
    assert "digest does not match" in traceback_text
    assert traceback_artifact.size_bytes <= 65_536


def test_worker_timeout_is_durable_failed_outcome(tmp_path):
    document = _document()
    document["execution"]["timeout_seconds"] = 1
    document["execution"]["n_steps"] = 500_000
    document["execution"]["s_param_n_steps"] = 500_000
    service = ExperimentService(tmp_path / "workspace")
    _, _, linked = _submitted(service, document)

    service.start(linked.run.id)
    final = service.wait(linked.run.id, timeout=60)

    assert final.state == "failed"
    assert "timeout" in final.error.lower()
    assert any(
        event.event_type == "run_timed_out"
        for event in service.repository.list_events(final.id)
    )
    assert any(
        artifact.kind == "traceback"
        for artifact in service.application_repository.list_artifacts(final.id)
    )


def test_stale_active_run_reconciliation_releases_cpu_lease(tmp_path):
    workspace = tmp_path / "workspace"
    service = ExperimentService(workspace)
    _, _, first = _submitted(service)
    service.application_repository.acquire_cpu_lease(first.run.id)
    service.repository.set_pid(first.run.id, 999_999)
    service.repository.transition(
        first.run.id,
        "preflighting",
        expected="queued",
        event_type="test_worker_claimed",
    )

    restarted = ExperimentService(workspace)
    reconciled = restarted.reconcile_stale_runs()

    assert [record.id for record in reconciled] == [first.run.id]
    assert reconciled[0].state == "failed"
    second = restarted.application_repository.create_linked_run(
        first.experiment_id, idempotency_key="replacement"
    )
    restarted.application_repository.acquire_cpu_lease(second.run.id)


def test_resource_gate_rejects_before_worker_or_lease(tmp_path):
    service = ExperimentService(tmp_path / "workspace", max_cpu_cells=100)
    _, _, linked = _submitted(service)

    with pytest.raises(RuntimeError, match="resource gate rejects"):
        service.start(linked.run.id)

    # Reaching the lease call proves the rejected start did not consume cpu:0.
    service.application_repository.acquire_cpu_lease(linked.run.id)
    with pytest.raises(ResourceBusyError):
        service.application_repository.acquire_cpu_lease(
            service.application_repository.create_linked_run(
                linked.experiment_id, idempotency_key="other"
            ).run.id
        )
