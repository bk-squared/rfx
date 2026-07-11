from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfx.experiments import (
    ResourceBusyError,
    RevisionConflictError,
    SQLiteApplicationRepository,
    apply_json_patch,
)


FIXTURE = Path(__file__).parent / "fixtures" / "experiments" / "patch_antenna_v2.json"


def _document() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_experiment_revision_and_validation_survive_repository_restart(tmp_path):
    path = tmp_path / "application.db"
    repository = SQLiteApplicationRepository(path)
    experiment, revision = repository.create_experiment(
        _document(), actor="human:owner"
    )

    reopened = SQLiteApplicationRepository(path)

    assert reopened.get_experiment(experiment.id) == experiment
    assert reopened.get_revision(revision.id) == revision
    assert revision.validation_state == "validated"
    assert revision.preflight["ok"] is True
    assert reopened.list_revisions(experiment.id) == [revision]


def test_json_patch_creates_immutable_revision_and_stale_base_conflicts(tmp_path):
    repository = SQLiteApplicationRepository(tmp_path / "application.db")
    experiment, first = repository.create_experiment(_document())
    patch = [
        {
            "op": "replace",
            "path": "/geometry/2/bounds_m/1/0",
            "value": 0.026,
        }
    ]

    second = repository.apply_patch(
        experiment.id,
        base_revision_id=first.id,
        patch=patch,
        actor="agent:test",
        message="lengthen patch",
    )

    assert second.sequence == 2
    assert second.parent_revision_id == first.id
    assert second.spec_sha256 != first.spec_sha256
    assert repository.get_revision(first.id).spec == _document()
    assert repository.get_experiment(experiment.id).current_revision_id == second.id
    with pytest.raises(RevisionConflictError) as caught:
        repository.apply_patch(
            experiment.id,
            base_revision_id=first.id,
            patch=patch,
            actor="agent:stale",
        )
    assert caught.value.current_revision_id == second.id


def test_linked_run_creation_is_idempotent_and_revision_pinned(tmp_path):
    repository = SQLiteApplicationRepository(tmp_path / "application.db")
    experiment, revision = repository.create_experiment(_document())

    first = repository.create_linked_run(
        experiment.id, idempotency_key="client-request-001"
    )
    duplicate = repository.create_linked_run(
        experiment.id, idempotency_key="client-request-001"
    )

    assert duplicate == first
    assert first.revision_id == revision.id
    assert first.run.spec_sha256 == revision.spec_sha256
    assert first.run.state == "queued"
    assert (
        repository.runs.list_events(first.run.id)[0].payload["revision_id"]
        == revision.id
    )


def test_cpu_lease_enforces_one_active_worker_and_can_be_recovered(tmp_path):
    repository = SQLiteApplicationRepository(tmp_path / "application.db")
    experiment, _ = repository.create_experiment(_document())
    first = repository.create_linked_run(experiment.id, idempotency_key="first")
    second = repository.create_linked_run(experiment.id, idempotency_key="second")

    repository.acquire_cpu_lease(first.run.id)
    with pytest.raises(ResourceBusyError, match=first.run.id):
        repository.acquire_cpu_lease(second.run.id)
    repository.release_cpu_lease(first.run.id)
    repository.acquire_cpu_lease(second.run.id)


def test_progress_and_artifact_records_are_durable(tmp_path):
    repository = SQLiteApplicationRepository(tmp_path / "application.db")
    experiment, _ = repository.create_experiment(_document())
    linked = repository.create_linked_run(experiment.id)
    payload = tmp_path / "diagnostic.txt"
    payload.write_text("bounded traceback", encoding="utf-8")

    updated = repository.heartbeat(
        linked.run.id, progress=0.4, phase="preflight-complete"
    )
    artifact = repository.register_artifact(
        linked.run.id, kind="traceback", path=payload
    )

    assert updated is not None
    assert updated.progress == 0.4
    assert repository.list_artifacts(linked.run.id) == [artifact]
    assert repository.get_artifact(artifact.id).sha256 == artifact.sha256
    assert repository.runs.list_events(linked.run.id)[-1].event_type == "progress"


def test_json_patch_rejects_schema_and_path_overreach():
    document = _document()

    with pytest.raises(ValueError, match="schema_version"):
        apply_json_patch(
            document,
            [{"op": "replace", "path": "/schema_version", "value": "evil"}],
        )
    with pytest.raises(ValueError, match="out of range"):
        apply_json_patch(
            document,
            [{"op": "replace", "path": "/geometry/99/id", "value": "evil"}],
        )
