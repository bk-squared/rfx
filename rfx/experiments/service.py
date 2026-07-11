"""Application service for submitting and controlling isolated CPU runs."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any, Mapping

from .compiler import compile_experiment
from .durable import SQLiteApplicationRepository
from .repository import RunRecord, TERMINAL_STATES
from .spec import ExperimentSpec


class ExperimentService:
    """Orchestrate durable experiment inputs and isolated worker processes."""

    def __init__(self, workspace: str | Path, *, max_cpu_cells: int = 5_000_000):
        self.workspace = Path(workspace).expanduser().resolve()
        self.runs_root = self.workspace / "runs"
        self.artifacts_root = self.workspace / "artifacts"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.application_repository = SQLiteApplicationRepository(
            self.workspace / "runs.sqlite3"
        )
        self.repository = self.application_repository.runs
        self.max_cpu_cells = int(max_cpu_cells)
        if self.max_cpu_cells <= 0:
            raise ValueError("max_cpu_cells must be positive")
        self._processes: dict[str, subprocess.Popen] = {}

    def submit(self, document: ExperimentSpec | Mapping[str, Any]) -> RunRecord:
        compiled = compile_experiment(document)
        if compiled.spec.to_dict().get("schema_version") == "rfx-experiment/v2":
            experiment, revision = self.application_repository.create_experiment(
                compiled.spec.to_dict(), actor="local-user"
            )
            return self.submit_revision(experiment.id, revision_id=revision.id).run
        spec_json = compiled.spec.canonical_json()
        record = self.repository.create_run(
            spec_json=spec_json,
            spec_sha256=compiled.spec.sha256,
            compiled_sha256=compiled.sha256,
        )
        self._persist_run_inputs(record, compiled)
        return record

    def create_experiment(
        self,
        document: Mapping[str, Any],
        *,
        actor: str = "local-user",
        message: str | None = "initial revision",
    ):
        return self.application_repository.create_experiment(
            document, actor=actor, message=message
        )

    def apply_experiment_patch(
        self,
        experiment_id: str,
        *,
        base_revision_id: str,
        patch: list[dict[str, Any]],
        actor: str,
        message: str | None = None,
    ):
        return self.application_repository.apply_patch(
            experiment_id,
            base_revision_id=base_revision_id,
            patch=patch,
            actor=actor,
            message=message,
        )

    def submit_revision(
        self,
        experiment_id: str,
        *,
        revision_id: str | None = None,
        idempotency_key: str | None = None,
    ):
        linked = self.application_repository.create_linked_run(
            experiment_id,
            revision_id=revision_id,
            idempotency_key=idempotency_key,
        )
        run_dir = self._run_dir(linked.run.id)
        if not run_dir.exists():
            compiled = compile_experiment(json.loads(linked.run.spec_json))
            self._persist_run_inputs(linked.run, compiled)
        return self.application_repository.get_linked_run(linked.run.id)

    def _persist_run_inputs(self, record: RunRecord, compiled) -> None:
        run_dir = self._run_dir(record.id)
        try:
            run_dir.mkdir(mode=0o700)
            self._atomic_write(
                run_dir / "spec.json", _pretty_json(compiled.spec.to_dict())
            )
            self._atomic_write(
                run_dir / "compiled.json", _pretty_json(compiled.to_dict())
            )
            self._atomic_write(run_dir / "generated.py", compiled.generated_python)
        except Exception as exc:
            self.repository.transition(
                record.id,
                "failed",
                expected="queued",
                event_type="input_persist_failed",
                error=str(exc),
            )
            raise

    def start(self, run_id: str) -> subprocess.Popen:
        record = self.repository.get_run(run_id)
        if record.state != "queued":
            raise RuntimeError(f"run {run_id} is {record.state!r}, not 'queued'")
        estimate = self._resource_gate(record)
        self.application_repository.acquire_cpu_lease(run_id)
        run_dir = self._run_dir(run_id)
        stdout_path = run_dir / "worker.stdout.log"
        stderr_path = run_dir / "worker.stderr.log"
        env = self._cpu_environment()
        command = [
            sys.executable,
            "-m",
            "rfx.experiments.worker",
            "--database",
            str(self.repository.path),
            "--workspace",
            str(self.workspace),
            "--run-id",
            run_id,
        ]
        try:
            with (
                stdout_path.open("ab", buffering=0) as stdout,
                stderr_path.open("ab", buffering=0) as stderr,
            ):
                process = subprocess.Popen(
                    command,
                    cwd=run_dir,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
        except Exception:
            self.application_repository.release_cpu_lease(run_id)
            raise
        self._processes[run_id] = process
        self.repository.set_pid(run_id, process.pid)
        self.repository.append_event(run_id, "resource_gate_passed", payload=estimate)
        return process

    def run_sync(self, document: ExperimentSpec | Mapping[str, Any]) -> RunRecord:
        record = self.submit(document)
        process = self.start(record.id)
        return_code = process.wait()
        self._processes.pop(record.id, None)
        return self._finalize_process(record.id, return_code)

    def wait(self, run_id: str, timeout: float | None = None) -> RunRecord:
        process = self._processes.get(run_id)
        if process is None:
            return self.repository.get_run(run_id)
        return_code = process.wait(timeout=timeout)
        self._processes.pop(run_id, None)
        return self._finalize_process(run_id, return_code)

    def refresh(self, run_id: str) -> RunRecord:
        """Refresh a locally-started process and persist its terminal logs once."""

        process = self._processes.get(run_id)
        if process is None:
            return self.repository.get_run(run_id)
        return_code = process.poll()
        if return_code is None:
            return self.repository.get_run(run_id)
        self._processes.pop(run_id, None)
        return self._finalize_process(run_id, return_code)

    def export_run_snapshot(self, run_id: str) -> Path:
        """Create the canonical G5 replay bundle for a terminal run."""

        from .replay import export_replay_bundle

        return export_replay_bundle(self, run_id)

    def _finalize_process(self, run_id: str, return_code: int) -> RunRecord:
        record = self.repository.get_run(run_id)
        if return_code != 0 and record.state not in TERMINAL_STATES:
            record = self.repository.transition(
                run_id,
                "failed",
                expected={"queued", "preflighting", "running"},
                event_type="worker_process_failed",
                error=f"worker exited with status {return_code}",
            )
        run_dir = self._run_dir(run_id)
        for kind, name in (
            ("stdout-log", "worker.stdout.log"),
            ("stderr-log", "worker.stderr.log"),
        ):
            path = run_dir / name
            if path.is_file():
                self.application_repository.register_artifact(
                    run_id, kind=kind, path=path
                )
        self.application_repository.release_cpu_lease(run_id)
        return record

    def cancel(self, run_id: str) -> RunRecord:
        record = self.repository.request_cancel(run_id)
        process = self._processes.get(run_id)
        if process is not None and process.poll() is None:
            process.terminate()
        elif (
            record.pid is not None
            and record.state not in TERMINAL_STATES
            and self._pid_matches_run(record.pid, run_id)
        ):
            try:
                os.kill(record.pid, signal.SIGTERM)
            except ProcessLookupError:
                # The verified process exited in the ps/kill race. Its worker
                # transaction (or a later status call) remains authoritative.
                pass
        return record

    def get(self, run_id: str) -> RunRecord:
        return self.repository.get_run(run_id)

    def reconcile_stale_runs(self) -> list[RunRecord]:
        reconciled = []
        for record in self.repository.list_runs(
            states={"queued", "preflighting", "running"}
        ):
            if record.pid is None:
                continue
            local = self._processes.get(record.id)
            alive = (
                local is not None and local.poll() is None
            ) or self._pid_matches_run(record.pid, record.id)
            if alive:
                continue
            reconciled_record = self.repository.transition(
                record.id,
                "failed",
                expected={"queued", "preflighting", "running"},
                event_type="stale_run_reconciled",
                error="worker process is absent during startup reconciliation",
            )
            self.application_repository.release_cpu_lease(record.id)
            reconciled.append(reconciled_record)
        return reconciled

    def _resource_gate(self, record: RunRecord) -> dict[str, Any]:
        document = json.loads(record.spec_json)
        if document.get("schema_version") == "rfx-experiment/v2":
            simulation = document["simulation"]
            domain = simulation["domain_m"]
            dx = float(simulation["cell_size_m"])
        else:
            model = document["model"]
            domain = model["domain_m"]
            dx = float(model["cell_size_m"])
        shape = [max(1, int(math.ceil(float(length) / dx))) for length in domain]
        cells = math.prod(shape)
        # Conservative field/material/carry planning factor for a uniform CPU
        # smoke. It is an admission estimate, not a measured peak claim.
        estimated_bytes = cells * 6 * 4 * 12
        estimate = {
            "backend": "cpu",
            "grid_shape_estimate": shape,
            "cell_count_estimate": cells,
            "memory_bytes_estimate": estimated_bytes,
            "max_cpu_cells": self.max_cpu_cells,
        }
        if cells > self.max_cpu_cells:
            raise RuntimeError(
                f"CPU resource gate rejects {cells} cells; limit is {self.max_cpu_cells}"
            )
        return estimate

    def _run_dir(self, run_id: str) -> Path:
        # Repository ids are UUIDs; resolving and checking containment keeps a
        # future caller from turning this helper into a path traversal surface.
        candidate = (self.runs_root / run_id).resolve()
        if candidate.parent != self.runs_root:
            raise ValueError("invalid run id path")
        return candidate

    @staticmethod
    def _atomic_write(path: Path, contents: str) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _cpu_environment() -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "JAX_PLATFORMS": "cpu",
                "JAX_PLATFORM_NAME": "cpu",
                "CUDA_VISIBLE_DEVICES": "",
                "ROCR_VISIBLE_DEVICES": "",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "PYTHONUNBUFFERED": "1",
            }
        )
        # Source-tree execution (pytest/editable installs) must give the child
        # the same import roots as the parent. Installed environments already
        # work and harmlessly receive their site-packages paths again.
        inherited = [entry for entry in sys.path if entry]
        existing = env.get("PYTHONPATH")
        if existing:
            inherited.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(inherited)
        return env

    @staticmethod
    def _pid_matches_run(pid: int, run_id: str) -> bool:
        """Guard cross-process cancellation against stale/reused PIDs."""

        proc_cmdline = Path(f"/proc/{pid}/cmdline")
        if proc_cmdline.is_file():
            try:
                command = (
                    proc_cmdline.read_bytes()
                    .replace(b"\x00", b" ")
                    .decode("utf-8", errors="replace")
                )
            except OSError:
                return False
        else:
            completed = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                return False
            command = completed.stdout
        return "rfx.experiments.worker" in command and run_id in command


def _pretty_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
