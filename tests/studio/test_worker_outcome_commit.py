"""Worker outcome publication is one durable decision, without running a solver."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rfx.experiments.durable import SQLiteApplicationRepository
from rfx.experiments.repository import InvalidRunTransitionError, SQLiteRunRepository


def _run(repository, *, linked=True, phase="running"):
    run = repository.runs.create_run(
        spec_json=(
            '{"schema_version":"rfx-experiment/v2"}'
            if linked
            else '{"schema_version":"rfx-experiment/v1"}'
        ),
        spec_sha256="a" * 64,
        compiled_sha256="b" * 64,
    )
    if linked:
        # Only the FK-valid application rows are needed here. Do not compile
        # or preflight a numerical fixture to test an SQLite transaction.
        experiment, revision = "experiment-" + run.id, "revision-" + run.id
        with repository._connect() as connection:
            connection.execute(
                "INSERT INTO experiments VALUES (?, ?, ?, ?, ?)",
                (
                    experiment,
                    "worker transaction",
                    revision,
                    run.created_at,
                    run.created_at,
                ),
            )
            connection.execute(
                """INSERT INTO experiment_revisions(
                    id, experiment_id, sequence, spec_json, spec_sha256,
                    semantic_fingerprint, validation_state, preflight_json,
                    actor, created_at
                ) VALUES (?, ?, 1, '{}', ?, ?, 'validated', '{}', 'test', ?)""",
                (revision, experiment, "a" * 64, "b" * 64, run.created_at),
            )
            connection.execute(
                """INSERT INTO run_links(
                    run_id, experiment_id, revision_id, timeout_seconds,
                    heartbeat_at, progress
                ) VALUES (?, ?, ?, 1, ?, 0.0)""",
                (run.id, experiment, revision, run.created_at),
            )
    if phase != "queued":
        repository.runs.transition(run.id, "preflighting", expected="queued")
    if phase == "running":
        repository.runs.transition(run.id, "running", expected="preflighting")
    if linked:
        repository.heartbeat(run.id, progress=0.4, phase="fixture-ready")
    repository.acquire_cpu_lease(run.id)
    return repository.runs.get_run(run.id)


def _success(tmp_path):
    data = b'{"kind":"s11","points":[1,2,3]}\n'
    digest = hashlib.sha256(data).hexdigest()
    root = tmp_path / "sha256" / digest
    root.mkdir(parents=True)
    primary = root / "s11.json"
    primary.write_bytes(data)
    field = tmp_path / "field-slice.json"
    field.write_text('{"component":"ez"}\n', encoding="utf-8")
    return dict(
        state="succeeded",
        event_type="run_succeeded",
        artifact_sha256=digest,
        artifact_path=root,
        artifacts=[
            {"kind": "s11", "path": primary},
            {"kind": "field-slice", "path": field},
        ],
        payload={"phase": "complete"},
    )


def _lease_owner(repository):
    with repository._connect() as connection:
        row = connection.execute(
            "SELECT run_id FROM worker_leases WHERE resource = 'cpu:0'"
        ).fetchone()
    return None if row is None else row["run_id"]


@pytest.mark.parametrize("linked", [False, True], ids=["v1-unlinked", "v2-linked"])
def test_success_publishes_state_artifacts_and_progress_in_one_commit(
    tmp_path, monkeypatch, linked
):
    repository = SQLiteApplicationRepository(tmp_path / "runs.db")
    run = _run(repository, linked=linked)
    observer = SQLiteApplicationRepository(repository.path)
    proposal = _success(tmp_path)
    append = SQLiteRunRepository._append_event
    observed = []

    def before_commit(connection, run_id, **kwargs):
        # This is another connection while the real method holds its write
        # transaction, after its artifact/state/progress/lease SQL executes.
        assert observer.runs.get_run(run_id).state == "running"
        assert observer.list_artifacts(run_id) == []
        assert _lease_owner(observer) == run_id
        if linked:
            assert observer.get_linked_run(run_id).progress == 0.4
        observed.append(True)
        return append(connection, run_id, **kwargs)

    monkeypatch.setattr(
        SQLiteRunRepository, "_append_event", staticmethod(before_commit)
    )
    final = repository.finish_worker_run(run.id, **proposal)
    assert observed == [True]
    assert observer.runs.get_run(run.id) == final
    assert final.state == "succeeded"
    assert final.artifact_sha256 == proposal["artifact_sha256"]
    assert final.artifact_path == str(proposal["artifact_path"])
    publications = observer.list_artifacts(run.id)
    assert {a.kind for a in publications} == {"s11", "field-slice"}
    for artifact in publications:
        contents = Path(artifact.path).read_bytes()
        assert artifact.sha256 == hashlib.sha256(contents).hexdigest()
        assert artifact.size_bytes == len(contents)
    if linked:
        linked_run = observer.get_linked_run(run.id)
        assert linked_run.progress == 1.0
        assert linked_run.heartbeat_at == final.updated_at
    else:
        with pytest.raises(KeyError, match="not revision-linked"):
            observer.get_linked_run(run.id)
    assert _lease_owner(observer) is None
    events = observer.runs.list_events(run.id)
    assert events[-1].event_type == "run_succeeded"
    assert events[-1].payload["phase"] == "complete"
    assert [e.sequence for e in events] == list(range(1, len(events) + 1))
    assert sum(e.state == "succeeded" for e in events) == 1
    assert observer.runs.request_cancel(run.id) == final
    assert observer.runs.list_events(run.id) == events


def test_cancel_committed_during_file_preparation_wins_over_success(
    tmp_path, monkeypatch
):
    repository = SQLiteApplicationRepository(tmp_path / "runs.db")
    run = _run(repository)
    controller = SQLiteApplicationRepository(repository.path)
    proposal = _success(tmp_path)
    primary = proposal["artifacts"][0]["path"]
    read_bytes = Path.read_bytes
    cancelled = []

    def cancel_during_read(path):
        contents = read_bytes(path)
        if path == primary:
            # A real separately committed cancellation between the method's
            # initial read and its final BEGIN IMMEDIATE, not mirrored SQL.
            controller.runs.request_cancel(run.id)
            cancelled.append(True)
        return contents

    monkeypatch.setattr(Path, "read_bytes", cancel_during_read)
    final = repository.finish_worker_run(run.id, **proposal)
    assert cancelled == [True]
    assert final.state == "cancelled" and final.cancel_requested
    assert final.artifact_path is None and final.artifact_sha256 is None
    assert controller.list_artifacts(run.id) == []
    assert controller.get_linked_run(run.id).progress == 0.4
    assert _lease_owner(controller) is None
    terminal = controller.runs.list_events(run.id)[-1]
    assert terminal.event_type == "run_cancelled"
    assert terminal.payload["proposed_event_type"] == "run_succeeded"
    assert "artifact_path" not in terminal.payload


@pytest.mark.parametrize("cancel_first", [False, True])
def test_timeout_and_cancellation_follow_durable_commit_order(tmp_path, cancel_first):
    repository = SQLiteApplicationRepository(tmp_path / "runs.db")
    run = _run(repository)
    controller = SQLiteApplicationRepository(repository.path)
    if cancel_first:
        controller.runs.request_cancel(run.id)
    final = repository.finish_worker_run(
        run.id,
        state="failed",
        event_type="run_timed_out",
        error="timeout expired",
    )
    if not cancel_first:
        assert controller.runs.request_cancel(run.id) == final
    assert final.state == ("cancelled" if cancel_first else "failed")
    assert final.cancel_requested is cancel_first
    event = controller.runs.list_events(run.id)[-1]
    assert event.event_type == ("run_cancelled" if cancel_first else "run_timed_out")
    if cancel_first:
        assert event.payload["proposed_error"] == "timeout expired"
    else:
        assert final.error == "timeout expired"
    assert controller.list_artifacts(run.id) == []
    assert controller.get_linked_run(run.id).progress == 0.4
    assert _lease_owner(controller) is None


@pytest.mark.parametrize("state", ["failed", "cancelled"])
def test_non_success_ignores_result_publication_and_keeps_diagnostics(tmp_path, state):
    repository = SQLiteApplicationRepository(tmp_path / "runs.db")
    run = _run(repository)
    diagnostic = tmp_path / "traceback.txt"
    diagnostic.write_text("bounded diagnostic", encoding="utf-8")
    registered = repository.register_artifact(run.id, kind="traceback", path=diagnostic)
    final = repository.finish_worker_run(
        run.id,
        state=state,
        event_type="run_" + state,
        error="interrupted",
        artifact_path="/discarded",
        artifact_sha256="discarded",
        artifacts=[{"kind": "s11", "path": tmp_path / "does-not-exist.json"}],
        payload={"artifact_path": "/discarded", "phase": "interrupted"},
    )
    assert final.state == state
    assert final.artifact_sha256 is None and final.artifact_path is None
    assert repository.list_artifacts(run.id) == [registered]
    assert repository.get_linked_run(run.id).progress == 0.4
    assert "artifact_path" not in repository.runs.list_events(run.id)[-1].payload
    assert _lease_owner(repository) is None


@pytest.mark.parametrize("state", ["succeeded", "failed", "cancelled"])
def test_terminal_retry_is_idempotent_and_cannot_release_another_runs_lease(
    tmp_path, state
):
    repository = SQLiteApplicationRepository(tmp_path / "runs.db")
    run = _run(repository)
    proposal = (
        _success(tmp_path)
        if state == "succeeded"
        else dict(state=state, event_type="run_" + state)
    )
    final = repository.finish_worker_run(run.id, **proposal)
    events = repository.runs.list_events(run.id)
    artifacts = repository.list_artifacts(run.id)
    linked = repository.get_linked_run(run.id)
    for artifact in artifacts:
        Path(artifact.path).unlink()
    other = _run(repository, linked=False)

    def unreadable_proposal():
        raise AssertionError("a terminal retry consumed its obsolete artifact set")
        yield  # pragma: no cover

    retry = repository.finish_worker_run(
        run.id,
        state="succeeded",
        event_type="run_succeeded",
        artifacts=unreadable_proposal(),
        artifact_path="/gone",
        artifact_sha256="gone",
    )
    assert retry == final
    assert repository.runs.list_events(run.id) == events
    assert repository.list_artifacts(run.id) == artifacts
    assert repository.get_linked_run(run.id) == linked
    assert _lease_owner(repository) == other.id


def test_queued_cancellation_releases_its_lease_without_a_second_terminal_event(
    tmp_path,
):
    repository = SQLiteApplicationRepository(tmp_path / "runs.db")
    run = _run(repository, linked=False, phase="queued")
    cancelled = repository.runs.request_cancel(run.id)
    events = repository.runs.list_events(run.id)
    assert (
        repository.finish_worker_run(
            run.id, state="cancelled", event_type="run_cancelled"
        )
        == cancelled
    )
    assert repository.runs.list_events(run.id) == events
    assert _lease_owner(repository) is None


def test_a_generic_exception_rolls_back_artifacts_state_progress_event_and_lease(
    tmp_path, monkeypatch
):
    repository = SQLiteApplicationRepository(tmp_path / "runs.db")
    run = _run(repository)
    observer = SQLiteApplicationRepository(repository.path)
    linked = observer.get_linked_run(run.id)
    events = observer.runs.list_events(run.id)
    proposal = _success(tmp_path)
    append = SQLiteRunRepository._append_event

    def fail_event(*args, **kwargs):
        raise RuntimeError("injected event failure after publication SQL")

    monkeypatch.setattr(SQLiteRunRepository, "_append_event", staticmethod(fail_event))
    with pytest.raises(RuntimeError, match="injected event failure"):
        repository.finish_worker_run(run.id, **proposal)
    assert observer.runs.get_run(run.id) == run
    assert observer.list_artifacts(run.id) == []
    assert observer.get_linked_run(run.id) == linked
    assert observer.runs.list_events(run.id) == events
    assert _lease_owner(observer) == run.id
    monkeypatch.setattr(SQLiteRunRepository, "_append_event", staticmethod(append))
    assert repository.finish_worker_run(run.id, **proposal).state == "succeeded"


@pytest.mark.parametrize("phase", ["queued", "preflighting"])
def test_success_requires_running_even_with_valid_artifact_files(tmp_path, phase):
    repository = SQLiteApplicationRepository(tmp_path / "runs.db")
    run = _run(repository, phase=phase)
    with pytest.raises(InvalidRunTransitionError):
        repository.finish_worker_run(run.id, **_success(tmp_path))
    assert repository.runs.get_run(run.id) == run
    assert repository.list_artifacts(run.id) == []
    assert repository.get_linked_run(run.id).progress == 0.4
    assert _lease_owner(repository) == run.id


def test_primary_digest_must_match_an_actual_published_file(tmp_path):
    repository = SQLiteApplicationRepository(tmp_path / "runs.db")
    run = _run(repository)
    proposal = _success(tmp_path)
    proposal["artifact_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="primary artifact does not match"):
        repository.finish_worker_run(run.id, **proposal)
    assert repository.runs.get_run(run.id) == run
    assert repository.list_artifacts(run.id) == []
    assert _lease_owner(repository) == run.id


def test_an_existing_identical_artifact_registration_keeps_its_id(tmp_path):
    repository = SQLiteApplicationRepository(tmp_path / "runs.db")
    run = _run(repository, linked=False)
    proposal = _success(tmp_path)
    entry = proposal["artifacts"][0]
    existing = repository.register_artifact(run.id, **entry)
    proposal["artifacts"].append(entry)
    repository.finish_worker_run(run.id, **proposal)
    records = repository.list_artifacts(run.id)
    assert len(records) == 2
    assert next(r for r in records if r.kind == existing.kind) == existing
