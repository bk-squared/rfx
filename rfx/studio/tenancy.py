"""Optional lab-server user, quota, object-store, and scheduler adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from rfx.experiments import ExperimentService


_USER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}$")


@dataclass(frozen=True)
class QuotaPolicy:
    max_cpu_cells: int = 5_000_000
    max_runs_per_user: int = 1000
    max_artifact_bytes: int = 10 * 1024**3

    def __post_init__(self):
        if (
            min(
                self.max_cpu_cells,
                self.max_runs_per_user,
                self.max_artifact_bytes,
            )
            <= 0
        ):
            raise ValueError("all quota limits must be positive")


class TenantWorkspaceAdapter:
    """Map external user ids to non-enumerating, isolated workspace roots."""

    def __init__(self, root: str | Path, *, quota: QuotaPolicy | None = None):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.quota = quota or QuotaPolicy()

    def workspace_for(self, user_id: str) -> Path:
        key = self._user_key(user_id)
        workspace = (self.root / "users" / key / "workspace").resolve()
        if not workspace.is_relative_to(self.root):
            raise ValueError("tenant workspace escapes root")
        workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
        return workspace

    def secret_directory_for(self, user_id: str) -> Path:
        key = self._user_key(user_id)
        directory = (self.root / "users" / key / "secrets").resolve()
        if not directory.is_relative_to(self.root):
            raise ValueError("tenant secret directory escapes root")
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        return directory

    def service_for(self, user_id: str) -> "TenantExperimentService":
        return TenantExperimentService(self.workspace_for(user_id), quota=self.quota)

    def assert_owned_path(self, user_id: str, path: str | Path) -> Path:
        workspace = self.workspace_for(user_id)
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_relative_to(workspace):
            raise PermissionError("path is outside the authenticated user's workspace")
        return candidate

    def assert_owned_secret_path(self, user_id: str, path: str | Path) -> Path:
        secret_root = self.secret_directory_for(user_id)
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_relative_to(secret_root):
            raise PermissionError(
                "secret path is outside the authenticated user's secret root"
            )
        return candidate

    @staticmethod
    def _user_key(user_id: str) -> str:
        if not isinstance(user_id, str) or not _USER_ID.fullmatch(user_id):
            raise ValueError("invalid user id")
        return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:32]


class TenantExperimentService(ExperimentService):
    """Experiment service with per-tenant run and artifact admission quotas."""

    def __init__(self, workspace: str | Path, *, quota: QuotaPolicy):
        self.quota = quota
        super().__init__(workspace, max_cpu_cells=quota.max_cpu_cells)

    def submit(self, document):
        self._assert_admission_quota()
        return super().submit(document)

    def submit_revision(
        self,
        experiment_id: str,
        *,
        revision_id: str | None = None,
        idempotency_key: str | None = None,
    ):
        if idempotency_key:
            existing = self.application_repository.find_linked_run_by_idempotency_key(
                idempotency_key
            )
            if existing is not None:
                return existing
        self._assert_admission_quota()
        return super().submit_revision(
            experiment_id,
            revision_id=revision_id,
            idempotency_key=idempotency_key,
        )

    def _assert_admission_quota(self) -> None:
        if len(self.repository.list_runs()) >= self.quota.max_runs_per_user:
            raise RuntimeError("user run quota exceeded")
        used = sum(
            path.stat().st_size
            for path in self.artifacts_root.rglob("*")
            if path.is_file()
        )
        if used >= self.quota.max_artifact_bytes:
            raise RuntimeError("user artifact quota exceeded")


class ObjectStoreAdapter(Protocol):
    def put(self, namespace: str, object_id: str, data: bytes) -> str: ...

    def get(self, namespace: str, object_id: str) -> bytes: ...


class FilesystemObjectStore:
    """Path-confined object store used to validate the future remote adapter."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, namespace: str, object_id: str, data: bytes) -> str:
        target = self._path(namespace, object_id)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(target)
        return hashlib.sha256(data).hexdigest()

    def get(self, namespace: str, object_id: str) -> bytes:
        return self._path(namespace, object_id).read_bytes()

    def _path(self, namespace: str, object_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{16,64}", namespace):
            raise ValueError("invalid object-store namespace")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", object_id):
            raise ValueError("invalid object id")
        target = (self.root / namespace / object_id).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("object path escapes root")
        return target


@dataclass(frozen=True)
class PostgresMetadataConfig:
    """Validated configuration boundary for a future Postgres repository."""

    dsn: str
    pool_min: int = 1
    pool_max: int = 8

    def __post_init__(self):
        parsed = urlsplit(self.dsn)
        if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
            raise ValueError("Postgres DSN must use postgres:// or postgresql://")
        if not 1 <= self.pool_min <= self.pool_max <= 64:
            raise ValueError("Postgres pool bounds must satisfy 1 <= min <= max <= 64")

    @property
    def redacted_dsn(self) -> str:
        parsed = urlsplit(self.dsn)
        user = parsed.username or ""
        authentication = f"{user}:***@" if user else ""
        host = parsed.hostname or ""
        if parsed.port:
            host += f":{parsed.port}"
        query = urlencode(
            [
                (
                    key,
                    "***"
                    if any(
                        token in key.lower()
                        for token in (
                            "password",
                            "secret",
                            "token",
                            "key",
                            "credential",
                        )
                    )
                    else value,
                )
                for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            ]
        )
        return urlunsplit(
            (parsed.scheme, authentication + host, parsed.path, query, "")
        )


class SchedulerAdapter(Protocol):
    def resource_for(self, user_id: str, device: str = "cpu") -> str: ...


class PerUserScheduler:
    """Deterministic scheduler namespace; workers still honor CPU leases."""

    def resource_for(self, user_id: str, device: str = "cpu") -> str:
        if device != "cpu":
            raise ValueError("CPU-only release does not schedule accelerator devices")
        return f"tenant:{TenantWorkspaceAdapter._user_key(user_id)}:cpu:0"
