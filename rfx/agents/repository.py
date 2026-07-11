"""Append-only audit and exact-arguments approval persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping
import uuid


@dataclass(frozen=True)
class ApprovalRecord:
    id: str
    tool_name: str
    actor: str
    arguments_json: str
    arguments_sha256: str
    semantic_diff_json: str
    status: str
    requested_at: str
    expires_at: str
    decided_at: str | None
    decided_by: str | None
    consumed_at: str | None

    @property
    def arguments(self) -> dict[str, Any]:
        return json.loads(self.arguments_json)

    @property
    def semantic_diff(self) -> dict[str, Any]:
        return json.loads(self.semantic_diff_json)


@dataclass(frozen=True)
class AuditRecord:
    sequence: int
    id: str
    actor: str
    tool_name: str
    arguments_sha256: str
    approval_id: str | None
    approval_status: str | None
    outcome: str
    experiment_id: str | None
    revision_id: str | None
    run_id: str | None
    detail_json: str
    created_at: str

    @property
    def detail(self) -> dict[str, Any]:
        return json.loads(self.detail_json)


class SQLiteAgentRepository:
    """Approval and audit store sharing the Studio SQLite database file."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
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
                CREATE TABLE IF NOT EXISTS agent_approvals (
                    id TEXT PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    arguments_sha256 TEXT NOT NULL,
                    semantic_diff_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected', 'consumed', 'expired')),
                    requested_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    decided_at TEXT,
                    decided_by TEXT,
                    consumed_at TEXT
                );

                CREATE UNIQUE INDEX IF NOT EXISTS one_pending_agent_approval
                ON agent_approvals(tool_name, actor, arguments_sha256)
                WHERE status = 'pending';

                CREATE TABLE IF NOT EXISTS agent_audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    actor TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_sha256 TEXT NOT NULL,
                    approval_id TEXT,
                    approval_status TEXT,
                    outcome TEXT NOT NULL,
                    experiment_id TEXT,
                    revision_id TEXT,
                    run_id TEXT,
                    detail_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TRIGGER IF NOT EXISTS agent_audit_no_update
                BEFORE UPDATE ON agent_audit_events
                BEGIN SELECT RAISE(ABORT, 'agent audit is append-only'); END;

                CREATE TRIGGER IF NOT EXISTS agent_audit_no_delete
                BEFORE DELETE ON agent_audit_events
                BEGIN SELECT RAISE(ABORT, 'agent audit is append-only'); END;

                INSERT OR IGNORE INTO schema_migrations(version, applied_at)
                VALUES (2, 'agent-control-plane');
                """
            )

    def request_approval(
        self,
        *,
        tool_name: str,
        actor: str,
        arguments: Mapping[str, Any],
        semantic_diff: Mapping[str, Any],
        ttl_seconds: int = 900,
    ) -> ApprovalRecord:
        arguments_json = canonical_json(arguments)
        arguments_sha256 = sha256_json(arguments)
        now = utc_now()
        expires = (
            datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        ).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """SELECT * FROM agent_approvals
                   WHERE tool_name = ? AND actor = ? AND arguments_sha256 = ?
                     AND status = 'pending'""",
                (tool_name, actor, arguments_sha256),
            ).fetchone()
            if existing is not None:
                record = ApprovalRecord(**dict(existing))
                if _expired(record.expires_at):
                    connection.execute(
                        "UPDATE agent_approvals SET status = 'expired' WHERE id = ?",
                        (record.id,),
                    )
                else:
                    return record
            identifier = str(uuid.uuid4())
            connection.execute(
                """INSERT INTO agent_approvals(
                    id, tool_name, actor, arguments_json, arguments_sha256,
                    semantic_diff_json, status, requested_at, expires_at,
                    decided_at, decided_by, consumed_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, NULL, NULL, NULL)""",
                (
                    identifier,
                    tool_name,
                    actor,
                    arguments_json,
                    arguments_sha256,
                    canonical_json(semantic_diff),
                    now,
                    expires,
                ),
            )
        return self.get_approval(identifier)

    def get_approval(self, approval_id: str) -> ApprovalRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_approvals WHERE id = ?", (approval_id,)
            ).fetchone()
        if row is None:
            raise KeyError(approval_id)
        return ApprovalRecord(**dict(row))

    def list_approvals(self, *, status: str | None = None) -> list[ApprovalRecord]:
        with self._connect() as connection:
            if status is None:
                rows = connection.execute(
                    "SELECT * FROM agent_approvals ORDER BY requested_at DESC, id"
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT * FROM agent_approvals
                       WHERE status = ? ORDER BY requested_at DESC, id""",
                    (status,),
                ).fetchall()
        return [ApprovalRecord(**dict(row)) for row in rows]

    def decide(
        self, approval_id: str, *, approved: bool, decided_by: str
    ) -> ApprovalRecord:
        if not decided_by.startswith("human:"):
            raise ValueError("approval decision actor must use the human: prefix")
        outcome = "approved" if approved else "rejected"
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM agent_approvals WHERE id = ?", (approval_id,)
            ).fetchone()
            if row is None:
                raise KeyError(approval_id)
            record = ApprovalRecord(**dict(row))
            if record.status != "pending":
                raise RuntimeError(f"approval is already {record.status}")
            if _expired(record.expires_at):
                connection.execute(
                    "UPDATE agent_approvals SET status = 'expired' WHERE id = ?",
                    (approval_id,),
                )
                raise RuntimeError("approval request expired")
            connection.execute(
                """UPDATE agent_approvals
                   SET status = ?, decided_at = ?, decided_by = ? WHERE id = ?""",
                (outcome, utc_now(), decided_by, approval_id),
            )
        return self.get_approval(approval_id)

    def consume(
        self,
        approval_id: str,
        *,
        tool_name: str,
        actor: str,
        arguments: Mapping[str, Any],
    ) -> ApprovalRecord:
        digest = sha256_json(arguments)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM agent_approvals WHERE id = ?", (approval_id,)
            ).fetchone()
            if row is None:
                raise KeyError(approval_id)
            record = ApprovalRecord(**dict(row))
            if record.status != "approved":
                raise RuntimeError(f"approval is {record.status}, not approved")
            if _expired(record.expires_at):
                connection.execute(
                    "UPDATE agent_approvals SET status = 'expired' WHERE id = ?",
                    (approval_id,),
                )
                raise RuntimeError("approval request expired")
            if record.tool_name != tool_name or record.actor != actor:
                raise RuntimeError("approval scope does not match tool and actor")
            if record.arguments_sha256 != digest:
                raise RuntimeError("approval arguments hash does not match exact call")
            connection.execute(
                """UPDATE agent_approvals
                   SET status = 'consumed', consumed_at = ? WHERE id = ?""",
                (utc_now(), approval_id),
            )
        return self.get_approval(approval_id)

    def append_audit(
        self,
        *,
        actor: str,
        tool_name: str,
        arguments_sha256: str,
        outcome: str,
        approval_id: str | None = None,
        approval_status: str | None = None,
        experiment_id: str | None = None,
        revision_id: str | None = None,
        run_id: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> AuditRecord:
        identifier = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO agent_audit_events(
                    id, actor, tool_name, arguments_sha256, approval_id,
                    approval_status, outcome, experiment_id, revision_id,
                    run_id, detail_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    identifier,
                    actor,
                    tool_name,
                    arguments_sha256,
                    approval_id,
                    approval_status,
                    outcome,
                    experiment_id,
                    revision_id,
                    run_id,
                    canonical_json(detail or {}),
                    utc_now(),
                ),
            )
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_audit_events WHERE id = ?", (identifier,)
            ).fetchone()
        assert row is not None
        return AuditRecord(**dict(row))

    def list_audit(self, *, limit: int = 200) -> list[AuditRecord]:
        limit = max(1, min(int(limit), 1000))
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM agent_audit_events
                   ORDER BY sequence DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [AuditRecord(**dict(row)) for row in rows]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expired(value: str) -> bool:
    return datetime.fromisoformat(value) <= datetime.now(timezone.utc)
