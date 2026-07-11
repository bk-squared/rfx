"""SQLite run repository and append-only experiment event log."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable
import uuid


RUN_STATES = (
    "queued",
    "preflighting",
    "running",
    "succeeded",
    "failed",
    "cancelled",
)
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})
ALLOWED_TRANSITIONS = {
    "queued": frozenset({"preflighting", "cancelled", "failed"}),
    "preflighting": frozenset({"running", "cancelled", "failed"}),
    "running": frozenset({"succeeded", "cancelled", "failed"}),
    "succeeded": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


class RunNotFoundError(KeyError):
    """Raised when a run id is absent from the repository."""


class InvalidRunTransitionError(RuntimeError):
    """Raised when a state transition violates the run state machine."""


@dataclass(frozen=True)
class RunRecord:
    id: str
    state: str
    spec_json: str
    spec_sha256: str
    compiled_sha256: str
    created_at: str
    updated_at: str
    pid: int | None
    cancel_requested: bool
    artifact_sha256: str | None
    artifact_path: str | None
    error: str | None


@dataclass(frozen=True)
class RunEvent:
    run_id: str
    sequence: int
    event_type: str
    state: str
    payload: dict[str, Any]
    created_at: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class SQLiteRunRepository:
    """Process-safe run state and event persistence backed by SQLite."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    state TEXT NOT NULL CHECK (state IN (
                        'queued', 'preflighting', 'running',
                        'succeeded', 'failed', 'cancelled'
                    )),
                    spec_json TEXT NOT NULL,
                    spec_sha256 TEXT NOT NULL,
                    compiled_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    pid INTEGER,
                    cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0, 1)),
                    artifact_sha256 TEXT,
                    artifact_path TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS run_events (
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, sequence)
                );

                CREATE INDEX IF NOT EXISTS idx_run_events_run
                ON run_events(run_id, sequence);
                """
            )

    def create_run(
        self,
        *,
        spec_json: str,
        spec_sha256: str,
        compiled_sha256: str,
        run_id: str | None = None,
    ) -> RunRecord:
        identifier = run_id or str(uuid.uuid4())
        try:
            uuid.UUID(identifier)
        except ValueError as exc:
            raise ValueError("run_id must be a UUID") from exc
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO runs (
                    id, state, spec_json, spec_sha256, compiled_sha256,
                    created_at, updated_at
                ) VALUES (?, 'queued', ?, ?, ?, ?, ?)""",
                (
                    identifier,
                    spec_json,
                    spec_sha256,
                    compiled_sha256,
                    timestamp,
                    timestamp,
                ),
            )
            self._append_event(
                connection,
                identifier,
                event_type="run_created",
                state="queued",
                payload={
                    "spec_sha256": spec_sha256,
                    "compiled_sha256": compiled_sha256,
                },
                timestamp=timestamp,
            )
        return self.get_run(identifier)

    def get_run(self, run_id: str) -> RunRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise RunNotFoundError(run_id)
        return self._record(row)

    def list_runs(self, *, states: Iterable[str] | None = None) -> list[RunRecord]:
        with self._connect() as connection:
            if states is None:
                rows = connection.execute(
                    "SELECT * FROM runs ORDER BY created_at, id"
                ).fetchall()
            else:
                state_list = sorted(set(states))
                if not state_list:
                    return []
                if any(state not in RUN_STATES for state in state_list):
                    raise ValueError("states contains an unknown run state")
                placeholders = ",".join("?" for _ in state_list)
                rows = connection.execute(
                    f"SELECT * FROM runs WHERE state IN ({placeholders}) ORDER BY created_at, id",
                    state_list,
                ).fetchall()
        return [self._record(row) for row in rows]

    def list_events(self, run_id: str) -> list[RunEvent]:
        self.get_run(run_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM run_events WHERE run_id = ? ORDER BY sequence",
                (run_id,),
            ).fetchall()
        return [
            RunEvent(
                run_id=row["run_id"],
                sequence=int(row["sequence"]),
                event_type=row["event_type"],
                state=row["state"],
                payload=json.loads(row["payload_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def transition(
        self,
        run_id: str,
        to_state: str,
        *,
        expected: str | Iterable[str] | None = None,
        event_type: str = "state_changed",
        payload: dict[str, Any] | None = None,
        artifact_sha256: str | None = None,
        artifact_path: str | None = None,
        error: str | None = None,
    ) -> RunRecord:
        if to_state not in RUN_STATES:
            raise InvalidRunTransitionError(f"unknown run state {to_state!r}")
        expected_states = None
        if expected is not None:
            expected_states = {expected} if isinstance(expected, str) else set(expected)
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RunNotFoundError(run_id)
            current = row["state"]
            if expected_states is not None and current not in expected_states:
                raise InvalidRunTransitionError(
                    f"run {run_id} is {current!r}, expected one of {sorted(expected_states)!r}"
                )
            if to_state not in ALLOWED_TRANSITIONS[current]:
                raise InvalidRunTransitionError(
                    f"run {run_id} cannot transition from {current!r} to {to_state!r}"
                )
            connection.execute(
                """UPDATE runs SET state = ?, updated_at = ?,
                    artifact_sha256 = COALESCE(?, artifact_sha256),
                    artifact_path = COALESCE(?, artifact_path),
                    error = COALESCE(?, error)
                    WHERE id = ?""",
                (to_state, timestamp, artifact_sha256, artifact_path, error, run_id),
            )
            event_payload = dict(payload or {})
            event_payload.update({"from_state": current, "to_state": to_state})
            if artifact_sha256 is not None:
                event_payload["artifact_sha256"] = artifact_sha256
            if error is not None:
                event_payload["error"] = error
            self._append_event(
                connection,
                run_id,
                event_type=event_type,
                state=to_state,
                payload=event_payload,
                timestamp=timestamp,
            )
        return self.get_run(run_id)

    def set_pid(self, run_id: str, pid: int) -> RunRecord:
        if pid <= 0:
            raise ValueError("pid must be positive")
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RunNotFoundError(run_id)
            if row["state"] not in TERMINAL_STATES:
                connection.execute(
                    "UPDATE runs SET pid = ?, updated_at = ? WHERE id = ?",
                    (pid, timestamp, run_id),
                )
                self._append_event(
                    connection,
                    run_id,
                    event_type="worker_started",
                    state=row["state"],
                    payload={"pid": pid},
                    timestamp=timestamp,
                )
        return self.get_run(run_id)

    def request_cancel(self, run_id: str) -> RunRecord:
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RunNotFoundError(run_id)
            state = row["state"]
            if state == "queued":
                connection.execute(
                    "UPDATE runs SET state = 'cancelled', cancel_requested = 1, updated_at = ? WHERE id = ?",
                    (timestamp, run_id),
                )
                self._append_event(
                    connection,
                    run_id,
                    event_type="run_cancelled",
                    state="cancelled",
                    payload={"from_state": "queued", "to_state": "cancelled"},
                    timestamp=timestamp,
                )
            elif state not in TERMINAL_STATES:
                connection.execute(
                    "UPDATE runs SET cancel_requested = 1, updated_at = ? WHERE id = ?",
                    (timestamp, run_id),
                )
                self._append_event(
                    connection,
                    run_id,
                    event_type="cancel_requested",
                    state=state,
                    payload={},
                    timestamp=timestamp,
                )
        return self.get_run(run_id)

    def append_event(
        self,
        run_id: str,
        event_type: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> RunEvent:
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RunNotFoundError(run_id)
            sequence = self._append_event(
                connection,
                run_id,
                event_type=event_type,
                state=row["state"],
                payload=payload or {},
                timestamp=timestamp,
            )
        return self.list_events(run_id)[sequence - 1]

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        run_id: str,
        *,
        event_type: str,
        state: str,
        payload: dict[str, Any],
        timestamp: str,
    ) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence FROM run_events WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        sequence = int(row["next_sequence"])
        payload_json = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        connection.execute(
            "INSERT INTO run_events (run_id, sequence, event_type, state, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, sequence, event_type, state, payload_json, timestamp),
        )
        return sequence

    @staticmethod
    def _record(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            id=row["id"],
            state=row["state"],
            spec_json=row["spec_json"],
            spec_sha256=row["spec_sha256"],
            compiled_sha256=row["compiled_sha256"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            pid=row["pid"],
            cancel_requested=bool(row["cancel_requested"]),
            artifact_sha256=row["artifact_sha256"],
            artifact_path=row["artifact_path"],
            error=row["error"],
        )
