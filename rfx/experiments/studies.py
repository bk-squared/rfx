"""Durable bounded parameter sweeps and candidate optimization studies."""

from __future__ import annotations

import builtins
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterable
import uuid

from .analysis import analyze_sparameters
from .durable import apply_json_patch
from .repository import utc_now


SUPPORTED_PATCH_PARAMETERS = frozenset(
    {
        "/materials/0/relative_permittivity",
        "/excitations/0/position_m/0",
        "/excitations/0/position_m/1",
        "/geometry/2/bounds_m/0/0",
        "/geometry/2/bounds_m/1/0",
        "/geometry/2/bounds_m/0/1",
        "/geometry/2/bounds_m/1/1",
    }
)
SUPPORTED_OBJECTIVES = frozenset({"minimum_s11_db", "max_s11_abs"})


@dataclass(frozen=True)
class StudyRecord:
    id: str
    experiment_id: str
    base_revision_id: str
    kind: str
    parameter_path: str
    objective: str
    direction: str
    budget_points: int
    early_stop_value: float | None
    status: str
    best_point_id: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class StudyPointRecord:
    id: str
    study_id: str
    point_index: int
    value_json: str
    revision_id: str | None
    run_id: str | None
    status: str
    metric_json: str | None
    error: str | None
    created_at: str
    updated_at: str

    @property
    def value(self) -> Any:
        return json.loads(self.value_json)

    @property
    def metric(self) -> dict[str, Any] | None:
        return json.loads(self.metric_json) if self.metric_json else None


class DurableStudyService:
    """Sequential CPU study runner with restart-safe completed points."""

    def __init__(self, experiment_service):
        self.service = experiment_service
        self.path = Path(experiment_service.repository.path)
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
                CREATE TABLE IF NOT EXISTS studies (
                    id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL REFERENCES experiments(id),
                    base_revision_id TEXT NOT NULL REFERENCES experiment_revisions(id),
                    kind TEXT NOT NULL CHECK (kind IN ('sweep', 'optimization')),
                    parameter_path TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    direction TEXT NOT NULL CHECK (direction IN ('minimize', 'maximize')),
                    budget_points INTEGER NOT NULL,
                    early_stop_value REAL,
                    status TEXT NOT NULL CHECK (status IN ('created', 'running', 'cancellation_requested', 'succeeded', 'failed', 'cancelled')),
                    best_point_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS study_points (
                    id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL REFERENCES studies(id),
                    point_index INTEGER NOT NULL,
                    value_json TEXT NOT NULL,
                    revision_id TEXT REFERENCES experiment_revisions(id),
                    run_id TEXT REFERENCES runs(id),
                    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'cancelled', 'skipped')),
                    metric_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(study_id, point_index)
                );

                INSERT OR IGNORE INTO schema_migrations(version, applied_at)
                VALUES (3, 'durable-studies');
                """
            )

    def create(
        self,
        experiment_id: str,
        *,
        kind: str,
        parameter_path: str,
        values: Iterable[Any],
        objective: str = "minimum_s11_db",
        direction: str = "minimize",
        budget_points: int = 20,
        early_stop_value: float | None = None,
    ) -> StudyRecord:
        if kind not in {"sweep", "optimization"}:
            raise ValueError("study kind must be sweep or optimization")
        if objective not in SUPPORTED_OBJECTIVES:
            raise ValueError("unsupported study objective")
        if direction not in {"minimize", "maximize"}:
            raise ValueError("direction must be minimize or maximize")
        if not 1 <= int(budget_points) <= 100:
            raise ValueError("budget_points must be in 1..100")
        experiment = self.service.application_repository.get_experiment(experiment_id)
        base = self.service.application_repository.get_revision(
            experiment.current_revision_id
        )
        if base.spec["kind"] != "patch_antenna":
            raise ValueError(
                "unsupported_optimization_lane: studies currently support patch_antenna only"
            )
        if parameter_path not in SUPPORTED_PATCH_PARAMETERS:
            raise ValueError(
                f"unsupported_study_parameter: {parameter_path} is outside the patch allowlist"
            )
        candidates = list(values)
        if not candidates or len(candidates) > int(budget_points):
            raise ValueError("candidate values must contain 1..budget_points entries")
        for value in candidates:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("study candidate values must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError("study candidate values must be finite")
            apply_json_patch(
                base.spec,
                [{"op": "replace", "path": parameter_path, "value": value}],
            )

        identifier = str(uuid.uuid4())
        timestamp = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO studies(
                    id, experiment_id, base_revision_id, kind, parameter_path,
                    objective, direction, budget_points, early_stop_value,
                    status, best_point_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'created', NULL, ?, ?)""",
                (
                    identifier,
                    experiment_id,
                    base.id,
                    kind,
                    parameter_path,
                    objective,
                    direction,
                    int(budget_points),
                    early_stop_value,
                    timestamp,
                    timestamp,
                ),
            )
            for index, value in enumerate(candidates):
                connection.execute(
                    """INSERT INTO study_points(
                        id, study_id, point_index, value_json, revision_id,
                        run_id, status, metric_json, error, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, NULL, NULL, 'pending', NULL, NULL, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        identifier,
                        index,
                        _json(value),
                        timestamp,
                        timestamp,
                    ),
                )
        return self.get(identifier)

    def get(self, study_id: str) -> StudyRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM studies WHERE id = ?", (study_id,)
            ).fetchone()
        if row is None:
            raise KeyError(study_id)
        return StudyRecord(**dict(row))

    def list(self, experiment_id: str | None = None) -> builtins.list[StudyRecord]:
        with self._connect() as connection:
            if experiment_id is None:
                rows = connection.execute(
                    "SELECT * FROM studies ORDER BY created_at, id"
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT * FROM studies WHERE experiment_id = ?
                       ORDER BY created_at, id""",
                    (experiment_id,),
                ).fetchall()
        return [StudyRecord(**dict(row)) for row in rows]

    def points(self, study_id: str) -> builtins.list[StudyPointRecord]:
        self.get(study_id)
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM study_points WHERE study_id = ?
                   ORDER BY point_index""",
                (study_id,),
            ).fetchall()
        return [StudyPointRecord(**dict(row)) for row in rows]

    def run(self, study_id: str) -> dict[str, Any]:
        study = self.get(study_id)
        if study.status == "cancelled":
            return self.payload(study_id, reused_completed=0)
        if study.status == "cancellation_requested":
            self._skip_remaining(study_id, status="cancelled")
            self._update_study(study_id, status="cancelled")
            return self.payload(study_id, reused_completed=0)
        self._update_study(study_id, status="running")
        reused = 0
        failures = 0
        for point in self.points(study_id):
            current_study = self.get(study_id)
            if current_study.status == "cancellation_requested":
                self._skip_remaining(study_id, status="cancelled")
                self._update_study(study_id, status="cancelled")
                break
            if point.status == "succeeded":
                reused += 1
                continue
            if point.status in {"skipped", "cancelled"}:
                continue
            try:
                base = self.service.application_repository.get_revision(
                    study.base_revision_id
                )
                candidate = apply_json_patch(
                    base.spec,
                    [
                        {
                            "op": "replace",
                            "path": study.parameter_path,
                            "value": point.value,
                        }
                    ],
                )
                revision = self.service.application_repository.create_derived_revision(
                    study.experiment_id,
                    parent_revision_id=study.base_revision_id,
                    document=candidate,
                    actor="study:runner",
                    message=f"{study.kind} {study.parameter_path}={point.value}",
                )
                linked = self.service.submit_revision(
                    study.experiment_id,
                    revision_id=revision.id,
                    idempotency_key=f"study:{study.id}:point:{point.point_index}",
                )
                self._update_point(
                    point.id,
                    status="running",
                    revision_id=revision.id,
                    run_id=linked.run.id,
                )
                if linked.run.state == "queued":
                    self.service.start(linked.run.id)
                final = self.service.wait(linked.run.id, timeout=None)
                if (
                    final.state == "cancelled"
                    and self.get(study_id).status == "cancellation_requested"
                ):
                    self._update_point(point.id, status="cancelled")
                    self._skip_remaining(study_id, status="cancelled")
                    self._update_study(study_id, status="cancelled")
                    break
                if final.state != "succeeded":
                    raise RuntimeError(final.error or f"run ended {final.state}")
                analysis = analyze_sparameters(self.service, final.id)
                metric = {
                    "name": study.objective,
                    "value": analysis["metrics"][study.objective],
                    "analysis": analysis,
                }
                self._update_point(point.id, status="succeeded", metric=metric)
                self._refresh_best(study_id)
                if self._early_stop_met(study_id, float(metric["value"])):
                    self._skip_remaining(study_id, status="skipped")
                    break
            except Exception as exc:
                failures += 1
                self._update_point(point.id, status="failed", error=str(exc)[:4000])

        final_study = self.get(study_id)
        if final_study.status == "running":
            points = self.points(study_id)
            succeeded = sum(item.status == "succeeded" for item in points)
            terminal = "succeeded" if succeeded and failures == 0 else "failed"
            self._update_study(study_id, status=terminal)
        self._refresh_best(study_id)
        return self.payload(study_id, reused_completed=reused)

    def cancel(self, study_id: str) -> dict[str, Any]:
        study = self.get(study_id)
        if study.status in {"succeeded", "failed", "cancelled"}:
            return self.payload(study_id)
        self._update_study(study_id, status="cancellation_requested")
        active = next(
            (item for item in self.points(study_id) if item.status == "running"), None
        )
        if active and active.run_id:
            self.service.cancel(active.run_id)
        return self.payload(study_id)

    def promote_best(self, study_id: str) -> dict[str, Any]:
        study = self.get(study_id)
        if study.best_point_id is None:
            raise RuntimeError("study has no successful best point")
        point = next(
            item for item in self.points(study_id) if item.id == study.best_point_id
        )
        if point.revision_id is None:
            raise RuntimeError("best point has no revision")
        experiment = self.service.application_repository.promote_revision(
            study.experiment_id, point.revision_id
        )
        return {
            "study_id": study_id,
            "experiment_id": experiment.id,
            "promoted_revision_id": point.revision_id,
            "best_point_id": point.id,
            "metric": point.metric,
        }

    def payload(self, study_id: str, *, reused_completed: int = 0) -> dict[str, Any]:
        study = self.get(study_id)
        points = self.points(study_id)
        return {
            "schema_version": "rfx-study/v1",
            **study.__dict__,
            "reused_completed_points": reused_completed,
            "budget_estimate": self._budget_estimate(study, points),
            "points": [
                {
                    **item.__dict__,
                    "value": item.value,
                    "metric": item.metric,
                }
                for item in points
            ],
        }

    def _budget_estimate(
        self, study: StudyRecord, points: builtins.list[StudyPointRecord]
    ) -> dict[str, Any]:
        revision = self.service.application_repository.get_revision(
            study.base_revision_id
        )
        simulation = revision.spec["simulation"]
        execution = revision.spec["execution"]
        domain = simulation["domain_m"]
        cell_size = float(simulation["cell_size_m"])
        grid_cells = math.prod(
            math.ceil(float(length) / cell_size) + 1 for length in domain
        )
        steps = int(execution["n_steps"])
        requested = len(points)
        remaining = sum(item.status in {"pending", "running"} for item in points)
        return {
            "backend": "cpu",
            "max_concurrent_runs": 1,
            "budget_points": study.budget_points,
            "requested_points": requested,
            "remaining_points": remaining,
            "estimated_grid_cells_per_run": grid_cells,
            "estimated_steps_per_run": steps,
            "estimated_total_cell_steps": grid_cells * steps * requested,
            "estimate_scope": "structural upper-bound before runtime/JIT overhead",
        }

    def _refresh_best(self, study_id: str) -> None:
        study = self.get(study_id)
        candidates = [
            item
            for item in self.points(study_id)
            if item.status == "succeeded" and item.metric is not None
        ]
        if not candidates:
            return

        def metric_value(item: StudyPointRecord) -> float:
            assert item.metric is not None
            return float(item.metric["value"])

        best = (
            min(candidates, key=metric_value)
            if study.direction == "minimize"
            else max(candidates, key=metric_value)
        )
        with self._connect() as connection:
            connection.execute(
                "UPDATE studies SET best_point_id = ?, updated_at = ? WHERE id = ?",
                (best.id, utc_now(), study_id),
            )

    def _early_stop_met(self, study_id: str, value: float) -> bool:
        study = self.get(study_id)
        threshold = study.early_stop_value
        if threshold is None:
            return False
        return (
            value <= threshold if study.direction == "minimize" else value >= threshold
        )

    def _skip_remaining(self, study_id: str, *, status: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE study_points SET status = ?, updated_at = ?
                   WHERE study_id = ? AND status = 'pending'""",
                (status, utc_now(), study_id),
            )

    def _update_study(self, study_id: str, *, status: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE studies SET status = ?, updated_at = ? WHERE id = ?",
                (status, utc_now(), study_id),
            )

    def _update_point(
        self,
        point_id: str,
        *,
        status: str,
        revision_id: str | None = None,
        run_id: str | None = None,
        metric: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE study_points
                   SET status = ?, revision_id = COALESCE(?, revision_id),
                       run_id = COALESCE(?, run_id),
                       metric_json = COALESCE(?, metric_json), error = ?,
                       updated_at = ? WHERE id = ?""",
                (
                    status,
                    revision_id,
                    run_id,
                    _json(metric) if metric is not None else None,
                    error,
                    utc_now(),
                    point_id,
                ),
            )


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
