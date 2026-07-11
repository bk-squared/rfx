from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfx.experiments import ExperimentService
from rfx.operations import (
    backup_workspace,
    migrate_workspace,
    restore_workspace,
    schema_versions,
    verify_workspace_backup,
)
from rfx.studio.api import create_app
from rfx.studio.cli import build_parser, validate_launch_security
from rfx.studio.tenancy import (
    FilesystemObjectStore,
    PerUserScheduler,
    PostgresMetadataConfig,
    QuotaPolicy,
    TenantWorkspaceAdapter,
)
from rfx.telemetry import LocalTelemetrySink, TelemetryPolicy


FIXTURE = Path(__file__).parent / "fixtures" / "experiments" / "patch_antenna_v2.json"


def _spec() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_upgrade_backup_and_rollback_preserve_experiment_run_metadata(tmp_path):
    workspace = tmp_path / "workspace"
    service = ExperimentService(workspace)
    experiment, revision = service.create_experiment(_spec())
    linked = service.submit_revision(
        experiment.id, revision_id=revision.id, idempotency_key="before-upgrade"
    )
    assert schema_versions(service.repository.path) == [1]

    backup = backup_workspace(workspace, tmp_path / "pre-upgrade.rfx-backup.zip")
    manifest = verify_workspace_backup(backup)
    assert manifest["database_schema_versions"] == [1]
    assert all("secret" not in item["path"] for item in manifest["files"])

    migrated = migrate_workspace(workspace)
    assert migrated["schema_versions"] == [1, 2, 3]
    upgraded = ExperimentService(workspace)
    extra, _ = upgraded.create_experiment(
        {**_spec(), "metadata": {**_spec()["metadata"], "title": "post-upgrade"}}
    )
    assert extra.id != experiment.id

    restored = restore_workspace(backup, workspace)
    assert restored["schema_versions"] == [1]
    rolled_back = ExperimentService(workspace)
    assert (
        rolled_back.application_repository.get_experiment(
            experiment.id
        ).current_revision_id
        == revision.id
    )
    assert (
        rolled_back.application_repository.get_linked_run(linked.run.id).revision_id
        == revision.id
    )
    with pytest.raises(KeyError):
        rolled_back.application_repository.get_experiment(extra.id)

    remigrated = migrate_workspace(workspace)
    assert remigrated["schema_versions"] == [1, 2, 3]
    assert (
        ExperimentService(workspace).repository.get_run(linked.run.id).state == "queued"
    )


def test_remote_bind_requires_token_tls_origin_and_enforces_auth(tmp_path):
    from fastapi.testclient import TestClient

    token = "rfx-lab-release-token-0123456789abcdef"
    token_file = tmp_path / "token"
    token_file.write_text(token + "\n", encoding="utf-8")
    token_file.chmod(0o600)
    parser = build_parser()
    missing = parser.parse_args(["--host", "0.0.0.0"])
    with pytest.raises(ValueError, match="auth-token-file"):
        validate_launch_security(missing)
    remote = parser.parse_args(
        [
            "--host",
            "0.0.0.0",
            "--auth-token-file",
            str(token_file),
            "--allowed-origin",
            "https://lab.example",
            "--tls-terminated",
        ]
    )
    assert validate_launch_security(remote) == token

    app = create_app(
        tmp_path / "remote-workspace",
        remote_auth_token=token,
        allowed_origins=["https://lab.example"],
        mcp_host="0.0.0.0",
    )
    with TestClient(app, base_url="https://lab.example") as client:
        health = client.get("/api/health")
        assert health.json()["authentication_required"] is True
        assert client.get("/api/experiments").status_code == 401
        assert (
            client.get(
                "/api/experiments", headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )
        authorization = {"Authorization": f"Bearer {token}"}
        assert client.get("/api/experiments", headers=authorization).status_code == 200
        assert (
            client.post(
                "/api/experiments",
                headers={**authorization, "Origin": "https://evil.invalid"},
                json=_spec(),
            ).status_code
            == 403
        )
        created = client.post(
            "/api/experiments",
            headers={**authorization, "Origin": "https://lab.example"},
            json=_spec(),
        )
        assert created.status_code == 201, created.text

        initialized = client.post(
            "/mcp/",
            headers={
                **authorization,
                "Origin": "https://lab.example",
                "Accept": "application/json, text/event-stream",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "lab-test", "version": "1"},
                },
            },
        )
        assert initialized.status_code == 200, initialized.text


def test_two_user_artifact_path_secret_and_scheduler_isolation(tmp_path):
    adapter = TenantWorkspaceAdapter(
        tmp_path / "lab", quota=QuotaPolicy(max_cpu_cells=100_000)
    )
    alice = adapter.service_for("alice@example.org")
    bob = adapter.service_for("bob@example.org")
    assert alice.workspace != bob.workspace
    alice_experiment, alice_revision = alice.create_experiment(_spec())
    bob_experiment, _ = bob.create_experiment(_spec())
    assert alice_experiment.id != bob_experiment.id
    linked = alice.submit_revision(
        alice_experiment.id,
        revision_id=alice_revision.id,
        idempotency_key="alice-only",
    )
    artifact = alice.application_repository.register_artifact(
        linked.run.id,
        kind="alice-spec",
        path=alice.runs_root / linked.run.id / "spec.json",
    )
    with pytest.raises(KeyError):
        bob.application_repository.get_artifact(artifact.id)
    with pytest.raises(PermissionError):
        adapter.assert_owned_path("bob@example.org", artifact.path)

    alice_secret = adapter.secret_directory_for("alice@example.org") / "provider.token"
    alice_secret.write_text("must-not-cross-users", encoding="utf-8")
    alice_secret.chmod(0o600)
    assert (
        adapter.assert_owned_secret_path("alice@example.org", alice_secret)
        == alice_secret
    )
    with pytest.raises(PermissionError):
        adapter.assert_owned_secret_path("bob@example.org", alice_secret)

    store = FilesystemObjectStore(tmp_path / "objects")
    alice_namespace = TenantWorkspaceAdapter._user_key("alice@example.org")
    bob_namespace = TenantWorkspaceAdapter._user_key("bob@example.org")
    store.put(alice_namespace, "result.json", b"alice")
    store.put(bob_namespace, "result.json", b"bob")
    assert store.get(alice_namespace, "result.json") == b"alice"
    assert store.get(bob_namespace, "result.json") == b"bob"
    with pytest.raises(ValueError):
        store.get(alice_namespace, "../../result.json")

    scheduler = PerUserScheduler()
    assert scheduler.resource_for("alice@example.org") != scheduler.resource_for(
        "bob@example.org"
    )
    with pytest.raises(ValueError, match="CPU-only"):
        scheduler.resource_for("alice@example.org", "gpu")

    postgres = PostgresMetadataConfig(
        "postgresql://rfx:top-secret@db.internal:5432/rfx?sslmode=require&sslpassword=query-secret"
    )
    assert "top-secret" not in postgres.redacted_dsn
    assert "query-secret" not in postgres.redacted_dsn
    assert "***" in postgres.redacted_dsn
    assert "sslmode=require" in postgres.redacted_dsn


def test_tenant_quota_blocks_new_runs_but_preserves_idempotent_retry(tmp_path):
    adapter = TenantWorkspaceAdapter(
        tmp_path / "lab",
        quota=QuotaPolicy(
            max_cpu_cells=100_000,
            max_runs_per_user=1,
            max_artifact_bytes=1024,
        ),
    )
    service = adapter.service_for("quota@example.org")
    experiment, revision = service.create_experiment(_spec())
    first = service.submit_revision(
        experiment.id,
        revision_id=revision.id,
        idempotency_key="quota-first",
    )
    retried = service.submit_revision(
        experiment.id,
        revision_id=revision.id,
        idempotency_key="quota-first",
    )
    assert retried.run.id == first.run.id
    with pytest.raises(RuntimeError, match="run quota"):
        service.submit_revision(
            experiment.id,
            revision_id=revision.id,
            idempotency_key="quota-second",
        )

    artifact_limited = TenantWorkspaceAdapter(
        tmp_path / "artifact-lab",
        quota=QuotaPolicy(
            max_cpu_cells=100_000,
            max_runs_per_user=2,
            max_artifact_bytes=4,
        ),
    ).service_for("artifact@example.org")
    other_experiment, other_revision = artifact_limited.create_experiment(_spec())
    artifact_limited.artifacts_root.joinpath("existing.bin").write_bytes(b"1234")
    with pytest.raises(RuntimeError, match="artifact quota"):
        artifact_limited.submit_revision(
            other_experiment.id,
            revision_id=other_revision.id,
            idempotency_key="artifact-over-limit",
        )


def test_telemetry_is_opt_in_and_rejects_experiment_or_secret_content(tmp_path):
    sink = LocalTelemetrySink(tmp_path / "telemetry.jsonl", TelemetryPolicy(False))
    assert sink.emit({"event": "studio_launch", "backend": "cpu"}) is False
    assert not sink.path.exists()

    enabled = LocalTelemetrySink(tmp_path / "telemetry.jsonl", TelemetryPolicy(True))
    assert (
        enabled.emit(
            {
                "event": "run_terminal",
                "backend": "cpu",
                "outcome": "succeeded",
                "duration_ms": 123,
                "workflow_kind": "patch_antenna",
            }
        )
        is True
    )
    payload = json.loads(enabled.path.read_text(encoding="utf-8"))
    assert set(payload) <= {
        "event",
        "backend",
        "outcome",
        "duration_ms",
        "workflow_kind",
    }
    for forbidden in ("spec", "prompt", "artifact_path", "api_key", "token"):
        with pytest.raises(ValueError, match="not allowlisted|content"):
            TelemetryPolicy(True).sanitize({"event": "bad", forbidden: "secret"})
