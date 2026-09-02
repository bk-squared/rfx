from __future__ import annotations

from types import SimpleNamespace

import json
import numpy as np
import pytest

from rfx.experiments import (
    SQLiteRunRepository,
    export_s11_artifact,
    verify_s11_artifact,
)
from rfx.experiments.repository import InvalidRunTransitionError


def _create(repository: SQLiteRunRepository):
    return repository.create_run(
        spec_json='{"schema_version":"rfx-experiment/v1"}',
        spec_sha256="a" * 64,
        compiled_sha256="b" * 64,
    )


def test_repository_persists_state_machine_and_ordered_events(tmp_path):
    repository = SQLiteRunRepository(tmp_path / "runs.db")
    run = _create(repository)
    repository.transition(run.id, "preflighting", expected="queued")
    repository.append_event(run.id, "preflight_completed", payload={"ok": True})
    repository.transition(run.id, "running", expected="preflighting")
    final = repository.transition(
        run.id,
        "succeeded",
        expected="running",
        artifact_sha256="c" * 64,
        artifact_path="/immutable/c",
    )

    assert final.state == "succeeded"
    assert final.artifact_sha256 == "c" * 64
    events = repository.list_events(run.id)
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert events[0].event_type == "run_created"
    assert events[-1].state == "succeeded"

    reopened = SQLiteRunRepository(tmp_path / "runs.db")
    assert reopened.get_run(run.id) == final
    with pytest.raises(InvalidRunTransitionError):
        reopened.transition(run.id, "running", expected="succeeded")


def test_queued_cancel_is_immediate_and_terminal(tmp_path):
    repository = SQLiteRunRepository(tmp_path / "runs.db")
    run = _create(repository)

    cancelled = repository.request_cancel(run.id)

    assert cancelled.state == "cancelled"
    assert cancelled.cancel_requested is True
    assert repository.list_events(run.id)[-1].event_type == "run_cancelled"


def test_active_cancel_sets_request_for_worker_boundary(tmp_path):
    repository = SQLiteRunRepository(tmp_path / "runs.db")
    run = _create(repository)
    repository.transition(run.id, "preflighting")

    pending = repository.request_cancel(run.id)

    assert pending.state == "preflighting"
    assert pending.cancel_requested is True
    assert repository.list_events(run.id)[-1].event_type == "cancel_requested"


def test_s11_artifact_is_content_addressed_read_only_and_verifiable(tmp_path):
    result = SimpleNamespace(
        s_params=np.asarray([[[0.5 + 0.25j, 0.25 - 0.1j]]]),
        freqs=np.asarray([1.0e9, 2.0e9]),
    )
    kwargs = {
        "result": result,
        "run_id": "00000000-0000-0000-0000-000000000001",
        "spec_sha256": "a" * 64,
        "compiled_sha256": "b" * 64,
        "runtime": {"backend": "cpu", "devices": [{"platform": "cpu"}]},
    }

    artifact = export_s11_artifact(tmp_path / "artifacts", **kwargs)
    duplicate = export_s11_artifact(tmp_path / "artifacts", **kwargs)

    assert duplicate == artifact
    assert artifact.root.name == artifact.sha256
    assert verify_s11_artifact(artifact) == artifact
    payload = json.loads(artifact.data_json.read_text(encoding="utf-8"))
    assert payload["points"][0]["real"] == 0.5
    assert payload["runtime"]["backend"] == "cpu"
    assert artifact.data_json.stat().st_mode & 0o222 == 0


def test_s11_artifact_verifier_detects_tampering(tmp_path):
    result = SimpleNamespace(
        s_params=np.asarray([[[0.5 + 0.0j]]]),
        freqs=np.asarray([1.0e9]),
    )
    artifact = export_s11_artifact(
        tmp_path,
        result=result,
        run_id="00000000-0000-0000-0000-000000000001",
        spec_sha256="a" * 64,
        compiled_sha256="b" * 64,
        runtime={"backend": "cpu"},
    )
    artifact.data_json.chmod(0o644)
    artifact.data_json.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="content address"):
        verify_s11_artifact(artifact)
