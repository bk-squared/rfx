"""Isolated worker entrypoint for a single CPU experiment run."""

from __future__ import annotations

import argparse
from contextlib import closing
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Callable

from .artifacts import (
    ResultArtifact,
    S11Artifact,
    export_field_slice_artifact,
    export_reflection_transmission_artifact,
    export_s11_artifact,
    export_sparameters_artifact,
)
from .compiler import compile_experiment
from .durable import SQLiteApplicationRepository
from .repository import SQLiteRunRepository, TERMINAL_STATES


class RunCancelled(BaseException):
    """Cooperative cancellation in the execution child.

    Deliberately a ``BaseException`` subclass, not ``RuntimeError`` (its
    original base -- issue #482): no ``except Exception`` anywhere on the
    signal-to-``execute_run`` path can swallow it. See ``RunTimedOut``
    below for the incident that forced this.
    """


class RunTimedOut(BaseException):
    """Historical cooperative timeout exception; retain the #482 safeguard.

    The supervisor now enforces the deadline from a separate process. The
    old SIGALRM approach could not interrupt long native code (#790), even
    after its exception-swallowing problem was fixed.

    Keep the ``BaseException`` base: PR #555 found that an ordinary
    ``Exception`` raised by the alarm could be swallowed by a preflight
    advisory's catch block. Callers/tests retaining the cooperative signal
    path must not reintroduce that failure when changing exception handling.
    """


def _cancel_signal(signum, _frame) -> None:
    raise RunCancelled(f"worker received signal {signum}")


def _child_command(*, database: Path, workspace: Path, run_id: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("_worker_child.py")),
        str(os.getpid()),
        "--database",
        str(database),
        "--workspace",
        str(workspace),
        "--run-id",
        run_id,
    ]


def _stop_child(process: subprocess.Popen) -> None:
    # Popen retains ownership of an unreaped child, so this cannot signal a
    # reused PID. Do not kill our process group: the supervisor must survive
    # long enough to persist the outcome and release the CPU lease.
    if process.poll() is None:
        process.kill()
    process.wait()


def _cancel_requested(database: Path, run_id: str) -> bool:
    """Poll the canonical flag without initializing schema or waiting on locks.

    Windows TerminateProcess cannot notify a Python signal handler. Reading
    the durable request also supports cancellation from a reopened service.
    BUSY/LOCKED postpones this observation only; the wall deadline still runs.
    """
    try:
        with closing(
            sqlite3.connect(
                database.as_uri() + "?mode=ro",
                uri=True,
                timeout=0,
                isolation_level=None,
            )
        ) as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        return row is not None and bool(row[0])
    except sqlite3.OperationalError as exc:
        # sqlite_errorcode was added after our Python 3.10 floor.
        if str(exc) in {
            "database is locked",
            "database table is locked",
            "database schema is locked",
        }:
            return False
        raise


def _wait_for_child(
    process: subprocess.Popen,
    *,
    deadline: float,
    cancelled: Callable[[], bool],
) -> str:
    """Observe a wall deadline independently of the child's Python/GIL state.

    The cancellation callback must not wait on database locks or mutate
    state. Check the deadline first; final state/artifact writes happen only
    after stopping expired computation.
    Every return (and exception) leaves the owned child reaped.
    """
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timeout"
            if cancelled():
                return "cancelled"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timeout"
            try:
                process.wait(timeout=min(0.05, remaining))
                if cancelled():
                    return "cancelled"
                if time.monotonic() >= deadline:
                    return "timeout"
                return "exited"
            except subprocess.TimeoutExpired:
                pass
    finally:
        _stop_child(process)


def _read_outcome(path: Path, returncode: int, workspace: Path) -> dict[str, Any]:
    try:
        contents = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(
            f"executor exited with status {returncode} without a complete outcome"
        ) from exc
    outcome = json.loads(contents)
    events = {
        "succeeded": {"run_succeeded"},
        "failed": {"run_failed", "run_timed_out"},
        "cancelled": {"run_cancelled"},
    }
    if not isinstance(outcome, dict) or outcome.get("state") not in events:
        raise ValueError("executor returned an invalid outcome")
    if outcome.get("event_type") not in events[outcome["state"]]:
        raise ValueError("executor outcome event does not match its state")
    expected_code = 1 if outcome["state"] == "failed" else 0
    if returncode != expected_code:
        raise ValueError(f"executor outcome conflicts with exit status {returncode}")
    if not isinstance(outcome.get("error", ""), str) or not isinstance(
        outcome.get("traceback", ""), str
    ):
        raise ValueError("executor returned invalid diagnostics")
    if outcome["state"] == "succeeded":
        artifact_root = (workspace / "artifacts").resolve()
        paths = [outcome["artifact_path"]] + [
            item["path"] for item in outcome["artifacts"]
        ]
        if any(
            not Path(path).resolve().is_relative_to(artifact_root) for path in paths
        ):
            raise ValueError("executor artifact is outside its workspace store")
    return outcome


def execute_run(*, database: Path, workspace: Path, run_id: str) -> int:
    """Supervise one fresh child, then atomically publish its final outcome.

    The persisted budget covers child startup, compile, preflight, solve and
    export. Final durable bookkeeping follows child termination. The service
    keeps this supervisor PID across restarts and cancellation requests.
    """
    repository = SQLiteRunRepository(database)
    application = SQLiteApplicationRepository(database)
    run_dir = (workspace / "runs" / run_id).resolve()
    if run_dir.parent != (workspace / "runs").resolve():
        raise ValueError("invalid run id path")

    cancellation_signal: int | None = None

    def request_stop(signum, _frame):
        nonlocal cancellation_signal
        cancellation_signal = int(signum)

    previous_handlers = {
        sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    for sig in previous_handlers:
        signal.signal(sig, request_stop)
    process = None
    outcome: dict[str, Any]
    try:
        record = repository.get_run(run_id)
        if record.state in TERMINAL_STATES:
            return 1 if record.state == "failed" else 0
        if record.cancel_requested or cancellation_signal is not None:
            outcome = {"state": "cancelled", "event_type": "run_cancelled"}
        else:
            # The durable submitted document is authoritative for the budget;
            # a tampered spec.json must not extend the supervisor's deadline.
            document = json.loads(record.spec_json)
            timeout_seconds = int(
                document.get("execution", {}).get("timeout_seconds", 3600)
            )
            if timeout_seconds <= 0:
                raise ValueError("experiment timeout must be positive")
            outcome_path = run_dir / "executor-outcome.json"
            if outcome_path.exists():
                raise ValueError("executor outcome already exists before execution")
            deadline = time.monotonic() + timeout_seconds
            process = subprocess.Popen(
                _child_command(database=database, workspace=workspace, run_id=run_id),
                stdin=subprocess.DEVNULL,
                # Inherit the service-created session/process group and logs.
                # An outer killpg(supervisor_pid) must terminate both processes.
            )
            observation = _wait_for_child(
                process,
                deadline=deadline,
                cancelled=lambda: (
                    cancellation_signal is not None
                    or _cancel_requested(repository.path, run_id)
                ),
            )
            if observation == "timeout":
                detail = f"worker exceeded the experiment timeout ({timeout_seconds}s); executor terminated and reaped"
                outcome = {
                    "state": "failed",
                    "event_type": "run_timed_out",
                    "error": detail,
                    "traceback": f"RunTimedOut: {detail}\nThis diagnostic was generated by the supervisor enforcing the wall deadline.\n",
                }
            elif observation == "cancelled":
                outcome = {"state": "cancelled", "event_type": "run_cancelled"}
            else:
                outcome = _read_outcome(outcome_path, process.returncode, workspace)
            if cancellation_signal is not None:
                outcome = {"state": "cancelled", "event_type": "run_cancelled"}
            outcome["payload"] = {
                "executor_pid": process.pid,
                "executor_returncode": process.returncode,
                "supervisor_observation": observation,
                "timeout_seconds": timeout_seconds,
            }
        return _publish_outcome(application, run_dir, run_id, outcome)
    except Exception as exc:
        if process is not None:
            _stop_child(process)
        outcome = {
            "state": "failed",
            "event_type": "run_failed",
            "error": str(exc)[:4000],
            "traceback": traceback.format_exc(),
        }
        traceback.print_exc()
        try:
            return _publish_outcome(application, run_dir, run_id, outcome)
        except Exception:
            traceback.print_exc()
            return 1
    finally:
        # Cleanup order is load-bearing, including unexpected parent errors.
        if process is not None:
            _stop_child(process)
        try:
            application.release_cpu_lease(run_id)
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


def _publish_outcome(
    application, run_dir: Path, run_id: str, outcome: dict[str, Any]
) -> int:
    diagnostic = outcome.get("traceback", "")
    if diagnostic:
        try:
            _write_traceback_artifact(application, run_dir, run_id, diagnostic)
        except Exception:
            # A broken diagnostic path must not skip the terminal transaction.
            traceback.print_exc()
    final = application.finish_worker_run(
        run_id,
        state=outcome["state"],
        event_type=outcome["event_type"],
        error=outcome.get("error"),
        artifact_sha256=outcome.get("artifact_sha256"),
        artifact_path=outcome.get("artifact_path"),
        artifacts=outcome.get("artifacts", ()),
        payload=outcome.get("payload"),
    )
    return 1 if final.state == "failed" else 0


def _execute_child(*, database: Path, workspace: Path, run_id: str) -> int:
    repository = SQLiteRunRepository(database)
    application = SQLiteApplicationRepository(database)
    run_dir = (workspace / "runs" / run_id).resolve()
    expected_parent = (workspace / "runs").resolve()
    if run_dir.parent != expected_parent:
        raise ValueError("invalid run id path")

    signal.signal(signal.SIGTERM, _cancel_signal)
    signal.signal(signal.SIGINT, _cancel_signal)
    try:
        record = repository.get_run(run_id)
        if record.cancel_requested or record.state == "cancelled":
            raise RunCancelled("cancellation requested before execution")
        repository.transition(
            run_id,
            "preflighting",
            expected="queued",
            event_type="preflight_started",
        )
        application.heartbeat(run_id, progress=0.05, phase="compiling")

        document = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
        compiled = compile_experiment(document)
        spec = compiled.spec
        record = repository.get_run(run_id)
        if (
            spec.sha256 != record.spec_sha256
            or compiled.sha256 != record.compiled_sha256
        ):
            raise ValueError("persisted experiment digest does not match repository")

        preflight = compiled.preflight()
        _atomic_write(run_dir / "preflight.json", _pretty_json(preflight))
        repository.append_event(
            run_id,
            "preflight_completed",
            payload={
                "ok": preflight["ok"],
                "n_issues": preflight["n_issues"],
                "n_errors": preflight["n_errors"],
            },
        )
        application.heartbeat(run_id, progress=0.25, phase="preflight-complete")
        if not preflight["ok"]:
            raise ValueError(
                f"preflight found {preflight['n_errors']} blocking error(s)"
            )
        _check_cancel(repository, run_id)

        runtime = _cpu_runtime()
        _atomic_write(run_dir / "runtime.json", _pretty_json(runtime))
        repository.transition(
            run_id,
            "running",
            expected="preflighting",
            event_type="simulation_started",
            payload={"backend": "cpu", "devices": runtime["devices"]},
        )
        application.heartbeat(run_id, progress=0.35, phase="simulation-running")
        if hasattr(compiled, "execute"):
            result = compiled.execute()
        else:
            simulation = compiled.build_simulation()
            result = simulation.run(**compiled.run_kwargs())
        _check_cancel(repository, run_id)
        application.heartbeat(run_id, progress=0.85, phase="artifact-export")

        workflow = getattr(spec, "workflow", "patch_antenna")
        artifact: ResultArtifact | S11Artifact
        if workflow == "wr90_waveguide":
            artifact = export_sparameters_artifact(
                workspace / "artifacts",
                result=result,
                run_id=run_id,
                spec_sha256=spec.sha256,
                compiled_sha256=compiled.sha256,
                runtime=runtime,
            )
            artifact_kind = "sparameters"
        elif workflow == "multilayer_fresnel":
            artifact = export_reflection_transmission_artifact(
                workspace / "artifacts",
                result=result,
                run_id=run_id,
                spec_sha256=spec.sha256,
                compiled_sha256=compiled.sha256,
                runtime=runtime,
            )
            artifact_kind = "reflection-transmission"
        else:
            artifact = export_s11_artifact(
                workspace / "artifacts",
                result=result,
                run_id=run_id,
                spec_sha256=spec.sha256,
                compiled_sha256=compiled.sha256,
                runtime=runtime,
                reference_impedance_ohm=_reference_impedance(spec),
            )
            artifact_kind = "s11"
        artifacts = [{"kind": artifact_kind, "path": str(artifact.data_json)}]
        field_artifact = export_field_slice_artifact(
            workspace / "artifacts",
            result=result,
            spec_document=spec.to_dict(),
            run_id=run_id,
            spec_sha256=spec.sha256,
            compiled_sha256=compiled.sha256,
            runtime=runtime,
        )
        if field_artifact is not None:
            artifacts.append(
                {"kind": "field-slice", "path": str(field_artifact.data_json)}
            )
        _check_cancel(repository, run_id)
        outcome = {
            "state": "succeeded",
            "event_type": "run_succeeded",
            "artifact_sha256": artifact.sha256,
            "artifact_path": str(artifact.root),
            "artifacts": artifacts,
        }
    except RunCancelled as exc:
        outcome = {
            "state": "cancelled",
            "event_type": "run_cancelled",
            "error": str(exc),
        }
    except (RunTimedOut, Exception) as exc:
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        outcome = {
            "state": "failed",
            "event_type": "run_timed_out"
            if isinstance(exc, RunTimedOut)
            else "run_failed",
            "error": detail[:4000],
            "traceback": traceback.format_exc()[-65_536:],
        }
        traceback.print_exc()
    # Files may have been produced, but no terminal state or successful
    # artifact is public until the supervisor reaps us and accepts this proposal.
    _atomic_write(run_dir / "executor-outcome.json", _pretty_json(outcome))
    return 0 if outcome["state"] != "failed" else 1


def _cpu_runtime() -> dict[str, Any]:
    # Import after the parent has fixed the backend environment. This is a
    # second enforcement boundary, not merely provenance reporting.
    import jax

    devices = [
        {
            "platform": device.platform,
            "device_kind": device.device_kind,
            "id": int(device.id),
        }
        for device in jax.devices()
    ]
    if not devices or any(device["platform"] != "cpu" for device in devices):
        raise RuntimeError(f"CPU-only worker observed non-CPU devices: {devices}")
    try:
        distribution_version = importlib.metadata.version("rfx-fdtd")
    except importlib.metadata.PackageNotFoundError:
        from rfx import __version__ as source_version

        distribution_version = source_version

    return {
        "backend": "cpu",
        "devices": devices,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "rfx-fdtd": distribution_version,
            "jax": jax.__version__,
            "jaxlib": importlib.metadata.version("jaxlib"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "source": _source_provenance(),
        "environment_policy": {
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "JAX_PLATFORM_NAME": os.environ.get("JAX_PLATFORM_NAME"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "seed": {
            "value": None,
            "policy": "not applicable: supported FDTD execution lanes use no stochastic operator",
        },
    }


def _source_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return {"git_commit": None, "git_worktree_dirty": None, "kind": "wheel"}
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
        )
        return {
            "git_commit": commit,
            "git_worktree_dirty": dirty,
            "kind": "source-checkout",
        }
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_worktree_dirty": None, "kind": "unknown"}


def _reference_impedance(spec) -> float:
    document = spec.to_dict()
    if document.get("schema_version") == "rfx-experiment/v2":
        for excitation in document["excitations"]:
            if excitation["kind"] == "lumped_port":
                return float(excitation["impedance_ohm"])
        return 50.0
    return float(spec.model.feed.impedance_ohm)


def _write_traceback_artifact(
    application: SQLiteApplicationRepository,
    run_dir: Path,
    run_id: str,
    contents: str,
) -> None:
    if not contents.strip():
        return
    # Bound persisted diagnostics while keeping the exception tail.
    encoded = contents.encode("utf-8", errors="replace")[-65_536:]
    path = run_dir / "traceback.txt"
    # A terminal-transaction retry must not overwrite bytes already indexed
    # by a durable artifact hash. Preserve the first diagnostic on retries.
    if not path.exists():
        _atomic_write(path, encoded.decode("utf-8", errors="replace"))
    application.register_artifact(run_id, kind="traceback", path=path)


def _check_cancel(repository: SQLiteRunRepository, run_id: str) -> None:
    if repository.get_run(run_id).cancel_requested:
        raise RunCancelled("cancellation requested")


def _atomic_write(path: Path, contents: str) -> None:
    # A forcibly stopped child can leave a temporary file. Unique siblings
    # prevent that residue from breaking the supervisor's diagnostic write.
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pretty_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one isolated rfx experiment")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return execute_run(
        database=args.database.expanduser().resolve(),
        workspace=args.workspace.expanduser().resolve(),
        run_id=args.run_id,
    )


if __name__ == "__main__":
    sys.exit(main())
