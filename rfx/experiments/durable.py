"""Durable experiment/revision/run/artifact application repository."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping
import uuid

from .canonical import CanonicalExperimentSpec, compile_canonical_experiment
from .repository import RunRecord, SQLiteRunRepository, utc_now


class ExperimentNotFoundError(KeyError):
    pass


class RevisionNotFoundError(KeyError):
    pass


class RevisionConflictError(RuntimeError):
    def __init__(self, current_revision_id: str):
        self.current_revision_id = current_revision_id
        super().__init__(
            f"stale base revision; current revision is {current_revision_id}"
        )


class ResourceBusyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExperimentRecord:
    id: str
    title: str
    current_revision_id: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class RevisionRecord:
    id: str
    experiment_id: str
    sequence: int
    parent_revision_id: str | None
    spec_json: str
    spec_sha256: str
    semantic_fingerprint: str
    validation_state: str
    preflight_json: str
    actor: str
    message: str | None
    created_at: str

    @property
    def spec(self) -> dict[str, Any]:
        return json.loads(self.spec_json)

    @property
    def preflight(self) -> dict[str, Any]:
        return json.loads(self.preflight_json)


@dataclass(frozen=True)
class LinkedRunRecord:
    run: RunRecord
    experiment_id: str
    revision_id: str
    idempotency_key: str | None
    timeout_seconds: int
    heartbeat_at: str
    progress: float


@dataclass(frozen=True)
class ArtifactRecord:
    id: str
    run_id: str
    kind: str
    sha256: str
    path: str
    size_bytes: int
    created_at: str


class SQLiteApplicationRepository:
    """Application persistence layered on the process-safe run repository."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.runs = SQLiteRunRepository(self.path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    current_revision_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS experiment_revisions (
                    id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL REFERENCES experiments(id),
                    sequence INTEGER NOT NULL,
                    parent_revision_id TEXT,
                    spec_json TEXT NOT NULL,
                    spec_sha256 TEXT NOT NULL,
                    semantic_fingerprint TEXT NOT NULL,
                    validation_state TEXT NOT NULL CHECK (validation_state IN ('validated', 'invalid')),
                    preflight_json TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    message TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE (experiment_id, sequence),
                    UNIQUE (experiment_id, spec_sha256)
                );

                CREATE TABLE IF NOT EXISTS run_links (
                    run_id TEXT PRIMARY KEY REFERENCES runs(id),
                    experiment_id TEXT NOT NULL REFERENCES experiments(id),
                    revision_id TEXT NOT NULL REFERENCES experiment_revisions(id),
                    idempotency_key TEXT UNIQUE,
                    timeout_seconds INTEGER NOT NULL,
                    heartbeat_at TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0.0 CHECK (progress >= 0.0 AND progress <= 1.0)
                );

                CREATE TABLE IF NOT EXISTS experiment_artifacts (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    kind TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (run_id, kind, sha256)
                );

                CREATE TABLE IF NOT EXISTS worker_leases (
                    resource TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE REFERENCES runs(id),
                    acquired_at TEXT NOT NULL
                );

                INSERT OR IGNORE INTO schema_migrations(version, applied_at)
                VALUES (1, 'bootstrap');
                """
            )

    def create_experiment(
        self,
        document: Mapping[str, Any],
        *,
        actor: str = "local-user",
        experiment_id: str | None = None,
        message: str | None = "initial revision",
    ) -> tuple[ExperimentRecord, RevisionRecord]:
        compiled = compile_canonical_experiment(document)
        preflight = compiled.preflight()
        validation_state = "validated" if preflight["ok"] else "invalid"
        identifier = experiment_id or str(uuid.uuid4())
        revision_id = str(uuid.uuid4())
        _uuid(identifier, "experiment_id")
        timestamp = utc_now()
        title = compiled.spec.metadata["title"]
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO experiments(id, title, current_revision_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (identifier, title, revision_id, timestamp, timestamp),
            )
            connection.execute(
                """INSERT INTO experiment_revisions(
                    id, experiment_id, sequence, parent_revision_id, spec_json,
                    spec_sha256, semantic_fingerprint, validation_state,
                    preflight_json, actor, message, created_at
                ) VALUES (?, ?, 1, NULL, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    revision_id,
                    identifier,
                    compiled.spec.canonical_json(),
                    compiled.spec.sha256,
                    compiled.semantic_fingerprint,
                    validation_state,
                    _canonical_json(preflight),
                    actor,
                    message,
                    timestamp,
                ),
            )
        return self.get_experiment(identifier), self.get_revision(revision_id)

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
        if row is None:
            raise ExperimentNotFoundError(experiment_id)
        return ExperimentRecord(**dict(row))

    def list_experiments(self) -> list[ExperimentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM experiments ORDER BY created_at, id"
            ).fetchall()
        return [ExperimentRecord(**dict(row)) for row in rows]

    def get_revision(self, revision_id: str) -> RevisionRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiment_revisions WHERE id = ?", (revision_id,)
            ).fetchone()
        if row is None:
            raise RevisionNotFoundError(revision_id)
        return RevisionRecord(**dict(row))

    def list_revisions(self, experiment_id: str) -> list[RevisionRecord]:
        self.get_experiment(experiment_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM experiment_revisions WHERE experiment_id = ? ORDER BY sequence",
                (experiment_id,),
            ).fetchall()
        return [RevisionRecord(**dict(row)) for row in rows]

    def apply_patch(
        self,
        experiment_id: str,
        *,
        base_revision_id: str,
        patch: list[dict[str, Any]],
        actor: str,
        message: str | None = None,
    ) -> RevisionRecord:
        experiment = self.get_experiment(experiment_id)
        if experiment.current_revision_id != base_revision_id:
            raise RevisionConflictError(experiment.current_revision_id)
        base = self.get_revision(base_revision_id)
        candidate = apply_json_patch(base.spec, patch)
        compiled = compile_canonical_experiment(candidate)
        preflight = compiled.preflight()
        validation_state = "validated" if preflight["ok"] else "invalid"
        timestamp = utc_now()
        revision_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT current_revision_id FROM experiments WHERE id = ?",
                (experiment_id,),
            ).fetchone()
            if current is None:
                raise ExperimentNotFoundError(experiment_id)
            if current["current_revision_id"] != base_revision_id:
                raise RevisionConflictError(current["current_revision_id"])
            sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 AS value FROM experiment_revisions WHERE experiment_id = ?",
                    (experiment_id,),
                ).fetchone()["value"]
            )
            connection.execute(
                """INSERT INTO experiment_revisions(
                    id, experiment_id, sequence, parent_revision_id, spec_json,
                    spec_sha256, semantic_fingerprint, validation_state,
                    preflight_json, actor, message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    revision_id,
                    experiment_id,
                    sequence,
                    base_revision_id,
                    compiled.spec.canonical_json(),
                    compiled.spec.sha256,
                    compiled.semantic_fingerprint,
                    validation_state,
                    _canonical_json(preflight),
                    actor,
                    message,
                    timestamp,
                ),
            )
            connection.execute(
                "UPDATE experiments SET current_revision_id = ?, title = ?, updated_at = ? WHERE id = ?",
                (
                    revision_id,
                    compiled.spec.metadata["title"],
                    timestamp,
                    experiment_id,
                ),
            )
        return self.get_revision(revision_id)

    def create_derived_revision(
        self,
        experiment_id: str,
        *,
        parent_revision_id: str,
        document: Mapping[str, Any],
        actor: str,
        message: str | None = None,
    ) -> RevisionRecord:
        """Persist a study candidate without changing the promoted revision."""

        self.get_experiment(experiment_id)
        parent = self.get_revision(parent_revision_id)
        if parent.experiment_id != experiment_id:
            raise RevisionNotFoundError(parent_revision_id)
        compiled = compile_canonical_experiment(document)
        preflight = compiled.preflight()
        validation_state = "validated" if preflight["ok"] else "invalid"
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """SELECT id FROM experiment_revisions
                   WHERE experiment_id = ? AND spec_sha256 = ?""",
                (experiment_id, compiled.spec.sha256),
            ).fetchone()
            if existing is not None:
                return self.get_revision(existing["id"])
            sequence = int(
                connection.execute(
                    """SELECT COALESCE(MAX(sequence), 0) + 1 AS value
                       FROM experiment_revisions WHERE experiment_id = ?""",
                    (experiment_id,),
                ).fetchone()["value"]
            )
            revision_id = str(uuid.uuid4())
            connection.execute(
                """INSERT INTO experiment_revisions(
                    id, experiment_id, sequence, parent_revision_id, spec_json,
                    spec_sha256, semantic_fingerprint, validation_state,
                    preflight_json, actor, message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    revision_id,
                    experiment_id,
                    sequence,
                    parent_revision_id,
                    compiled.spec.canonical_json(),
                    compiled.spec.sha256,
                    compiled.semantic_fingerprint,
                    validation_state,
                    _canonical_json(preflight),
                    actor,
                    message,
                    timestamp,
                ),
            )
        return self.get_revision(revision_id)

    def promote_revision(
        self, experiment_id: str, revision_id: str
    ) -> ExperimentRecord:
        """Promote an existing candidate revision to the experiment head."""

        revision = self.get_revision(revision_id)
        if revision.experiment_id != experiment_id:
            raise RevisionNotFoundError(revision_id)
        title = revision.spec["metadata"]["title"]
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """UPDATE experiments
                   SET current_revision_id = ?, title = ?, updated_at = ?
                   WHERE id = ?""",
                (revision_id, title, utc_now(), experiment_id),
            ).rowcount
            if updated != 1:
                raise ExperimentNotFoundError(experiment_id)
        return self.get_experiment(experiment_id)

    def create_linked_run(
        self,
        experiment_id: str,
        *,
        revision_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> LinkedRunRecord:
        experiment = self.get_experiment(experiment_id)
        revision_id = revision_id or experiment.current_revision_id
        revision = self.get_revision(revision_id)
        if revision.experiment_id != experiment_id:
            raise RevisionNotFoundError(revision_id)
        if idempotency_key:
            with self._connect() as connection:
                existing = connection.execute(
                    "SELECT run_id FROM run_links WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
            if existing is not None:
                return self.get_linked_run(existing["run_id"])
        compiled = compile_canonical_experiment(revision.spec)
        run_id = str(uuid.uuid4())
        timestamp = utc_now()
        timeout_seconds = int(revision.spec["execution"]["timeout_seconds"])
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if idempotency_key:
                existing = connection.execute(
                    "SELECT run_id FROM run_links WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if existing is not None:
                    return self.get_linked_run(existing["run_id"])
            connection.execute(
                """INSERT INTO runs(
                    id, state, spec_json, spec_sha256, compiled_sha256,
                    created_at, updated_at
                ) VALUES (?, 'queued', ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    compiled.spec.canonical_json(),
                    compiled.spec.sha256,
                    compiled.sha256,
                    timestamp,
                    timestamp,
                ),
            )
            SQLiteRunRepository._append_event(
                connection,
                run_id,
                event_type="run_created",
                state="queued",
                payload={
                    "experiment_id": experiment_id,
                    "revision_id": revision_id,
                    "spec_sha256": compiled.spec.sha256,
                    "compiled_sha256": compiled.sha256,
                },
                timestamp=timestamp,
            )
            connection.execute(
                """INSERT INTO run_links(
                    run_id, experiment_id, revision_id, idempotency_key,
                    timeout_seconds, heartbeat_at, progress
                ) VALUES (?, ?, ?, ?, ?, ?, 0.0)""",
                (
                    run_id,
                    experiment_id,
                    revision_id,
                    idempotency_key,
                    timeout_seconds,
                    timestamp,
                ),
            )
        return self.get_linked_run(run_id)

    def get_linked_run(self, run_id: str) -> LinkedRunRecord:
        run = self.runs.get_run(run_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM run_links WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"run {run_id} is not revision-linked")
        return LinkedRunRecord(
            run=run, **{key: row[key] for key in row.keys() if key != "run_id"}
        )

    def find_linked_run_by_idempotency_key(
        self, idempotency_key: str
    ) -> LinkedRunRecord | None:
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("idempotency_key must be a non-empty string")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT run_id FROM run_links WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        return self.get_linked_run(row["run_id"]) if row is not None else None

    def heartbeat(
        self,
        run_id: str,
        *,
        progress: float,
        phase: str,
    ) -> LinkedRunRecord | None:
        progress = min(1.0, max(0.0, float(progress)))
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                "UPDATE run_links SET heartbeat_at = ?, progress = ? WHERE run_id = ?",
                (timestamp, progress, run_id),
            ).rowcount
        if not updated:
            return None
        self.runs.append_event(
            run_id,
            "progress",
            payload={"phase": phase, "progress": progress},
        )
        return self.get_linked_run(run_id)

    def acquire_cpu_lease(self, run_id: str) -> None:
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT run_id FROM worker_leases WHERE resource = 'cpu:0'"
            ).fetchone()
            if existing is not None and existing["run_id"] != run_id:
                raise ResourceBusyError(f"cpu:0 is leased by run {existing['run_id']}")
            connection.execute(
                "INSERT OR REPLACE INTO worker_leases(resource, run_id, acquired_at) VALUES ('cpu:0', ?, ?)",
                (run_id, timestamp),
            )

    def release_cpu_lease(self, run_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM worker_leases WHERE resource = 'cpu:0' AND run_id = ?",
                (run_id,),
            )

    def register_artifact(
        self,
        run_id: str,
        *,
        kind: str,
        path: str | Path,
    ) -> ArtifactRecord:
        artifact_path = Path(path).expanduser().resolve()
        data = artifact_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        timestamp = utc_now()
        identifier = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM experiment_artifacts WHERE run_id = ? AND kind = ? AND sha256 = ?",
                (run_id, kind, digest),
            ).fetchone()
            if existing is not None:
                return ArtifactRecord(**dict(existing))
            connection.execute(
                "INSERT INTO experiment_artifacts(id, run_id, kind, sha256, path, size_bytes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    identifier,
                    run_id,
                    kind,
                    digest,
                    str(artifact_path),
                    len(data),
                    timestamp,
                ),
            )
        return self.get_artifact(identifier)

    def get_artifact(self, artifact_id: str) -> ArtifactRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiment_artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()
        if row is None:
            raise KeyError(artifact_id)
        return ArtifactRecord(**dict(row))

    def list_artifacts(self, run_id: str) -> list[ArtifactRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM experiment_artifacts WHERE run_id = ? ORDER BY created_at, id",
                (run_id,),
            ).fetchall()
        return [ArtifactRecord(**dict(row)) for row in rows]


def apply_json_patch(
    document: Mapping[str, Any], patch: list[dict[str, Any]]
) -> dict[str, Any]:
    if not isinstance(patch, list) or not patch or len(patch) > 100:
        raise ValueError("patch must contain 1..100 operations")
    result = json.loads(json.dumps(document, allow_nan=False))
    for index, operation in enumerate(patch):
        if not isinstance(operation, dict) or set(operation) - {"op", "path", "value"}:
            raise ValueError(f"patch[{index}] has unsupported fields")
        op = operation.get("op")
        path = operation.get("path")
        if op not in {"add", "replace", "remove"}:
            raise ValueError(f"patch[{index}].op is unsupported")
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError(f"patch[{index}].path must be a JSON Pointer")
        tokens = [_unescape_pointer(token) for token in path.lstrip("/").split("/")]
        if not tokens or tokens[0] == "schema_version":
            raise ValueError("schema_version cannot be patched")
        parent: Any = result
        for token in tokens[:-1]:
            if isinstance(parent, list):
                parent = parent[_list_index(token, len(parent), allow_end=False)]
            elif isinstance(parent, dict) and token in parent:
                parent = parent[token]
            else:
                raise ValueError(f"patch[{index}] path does not exist")
        leaf = tokens[-1]
        if isinstance(parent, list):
            target = _list_index(leaf, len(parent), allow_end=op == "add")
            if op == "add":
                parent.insert(target, operation.get("value"))
            elif op == "replace":
                parent[target] = operation.get("value")
            else:
                parent.pop(target)
        elif isinstance(parent, dict):
            if op in {"replace", "remove"} and leaf not in parent:
                raise ValueError(f"patch[{index}] path does not exist")
            if op == "remove":
                del parent[leaf]
            else:
                if "value" not in operation:
                    raise ValueError(f"patch[{index}] requires value")
                parent[leaf] = operation["value"]
        else:
            raise ValueError(f"patch[{index}] parent is not a container")
    CanonicalExperimentSpec.from_dict(result)
    return result


def _unescape_pointer(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _list_index(token: str, length: int, *, allow_end: bool) -> int:
    if token == "-" and allow_end:
        return length
    if not token.isdigit():
        raise ValueError("array JSON Pointer token must be an index")
    value = int(token)
    maximum = length if allow_end else length - 1
    if value < 0 or value > maximum:
        raise ValueError("array JSON Pointer index out of range")
    return value


def _uuid(value: str, field: str) -> None:
    try:
        uuid.UUID(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a UUID") from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
