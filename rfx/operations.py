"""Workspace migrations, checksummed backup, and atomic restore."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import tempfile
from typing import Any
import uuid
import zipfile


WORKSPACE_BACKUP_VERSION = "rfx-workspace-backup/v1"
LATEST_SCHEMA_VERSION = 3


def migrate_workspace(workspace: str | Path) -> dict[str, Any]:
    """Apply every idempotent application migration to a workspace."""

    from rfx.agents import AgentControlPlane
    from rfx.experiments import DurableStudyService, ExperimentService

    service = ExperimentService(workspace)
    AgentControlPlane(service)
    DurableStudyService(service)
    versions = schema_versions(service.repository.path)
    if versions != list(range(1, LATEST_SCHEMA_VERSION + 1)):
        raise RuntimeError(f"incomplete schema migration set: {versions}")
    return {
        "workspace": str(service.workspace),
        "schema_versions": versions,
        "latest_schema_version": LATEST_SCHEMA_VERSION,
    }


def schema_versions(database: str | Path) -> list[int]:
    path = Path(database).expanduser().resolve()
    if not path.is_file():
        return []
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
    return [int(row[0]) for row in rows]


def backup_workspace(workspace: str | Path, destination: str | Path) -> Path:
    """Create a consistent SQLite/filesystem backup without secret files."""

    root = Path(workspace).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    database = root / "runs.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(database)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rfx-backup-") as temporary:
        consistent_database = Path(temporary) / "runs.sqlite3"
        with (
            sqlite3.connect(database) as source,
            sqlite3.connect(consistent_database) as target,
        ):
            source.backup(target)
        files: dict[str, bytes] = {
            "workspace/runs.sqlite3": consistent_database.read_bytes()
        }
        for directory in ("runs", "artifacts"):
            base = root / directory
            if not base.is_dir():
                continue
            for path in sorted(item for item in base.rglob("*") if item.is_file()):
                relative = path.relative_to(root).as_posix()
                if _looks_secret(relative):
                    continue
                files[f"workspace/{relative}"] = path.read_bytes()
        manifest = {
            "schema_version": WORKSPACE_BACKUP_VERSION,
            "database_schema_versions": schema_versions(database),
            "content_policy": "experiment database, run inputs/logs, and artifacts only; token/key/env files excluded",
            "files": [
                {
                    "path": name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
                for name, data in sorted(files.items())
            ],
        }
        files["manifest.json"] = _json_bytes(manifest)
        temporary_archive = destination_path.with_suffix(
            destination_path.suffix + ".tmp"
        )
        with zipfile.ZipFile(temporary_archive, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(files.items()):
                _write_deterministic(archive, name, data)
        os.replace(temporary_archive, destination_path)
    verify_workspace_backup(destination_path)
    return destination_path


def verify_workspace_backup(backup: str | Path) -> dict[str, Any]:
    path = Path(backup).expanduser().resolve()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("backup contains duplicate paths")
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                raise ValueError("backup contains unsafe path")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema_version") != WORKSPACE_BACKUP_VERSION:
            raise ValueError("unsupported workspace backup version")
        declared = {item["path"]: item for item in manifest["files"]}
        if set(declared) != set(names) - {"manifest.json"}:
            raise ValueError("backup manifest file set mismatch")
        if "workspace/runs.sqlite3" not in declared:
            raise ValueError("backup has no SQLite database")
        for name, item in declared.items():
            data = archive.read(name)
            if len(data) != item["size_bytes"]:
                raise ValueError(f"backup size mismatch: {name}")
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"backup checksum mismatch: {name}")
            if _looks_secret(name):
                raise ValueError(f"backup contains forbidden secret-like path: {name}")
    return manifest


def restore_workspace(backup: str | Path, workspace: str | Path) -> dict[str, Any]:
    """Verify and atomically restore a workspace, enabling rollback."""

    manifest = verify_workspace_backup(backup)
    archive_path = Path(backup).expanduser().resolve()
    root = Path(workspace).expanduser().resolve()
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    staged = parent / f".{root.name}.restore-{uuid.uuid4().hex}"
    previous = parent / f".{root.name}.previous-{uuid.uuid4().hex}"
    staged.mkdir(mode=0o700)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for item in manifest["files"]:
                logical = PurePosixPath(item["path"])
                relative = Path(*logical.parts[1:])
                target = (staged / relative).resolve()
                if not target.is_relative_to(staged):
                    raise ValueError("restore path escapes workspace")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(item["path"]))
        restored_versions = schema_versions(staged / "runs.sqlite3")
        if restored_versions != manifest["database_schema_versions"]:
            raise ValueError("restored database schema version mismatch")
        if root.exists():
            os.replace(root, previous)
        os.replace(staged, root)
        if previous.exists():
            shutil.rmtree(previous)
    except Exception:
        if staged.exists():
            shutil.rmtree(staged)
        if previous.exists() and not root.exists():
            os.replace(previous, root)
        raise
    return {
        "workspace": str(root),
        "schema_versions": manifest["database_schema_versions"],
        "restored_files": len(manifest["files"]),
    }


def _looks_secret(path: str) -> bool:
    lowered = "/" + path.lower().replace("\\", "/")
    name = PurePosixPath(lowered).name
    if name in {".env", ".pypirc", "credentials", "credentials.json"}:
        return True
    return any(token in name for token in ("secret", "token", "api_key", "apikey"))


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _write_deterministic(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100600 << 16
    archive.writestr(info, data)
