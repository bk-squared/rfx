"""Isolated worker entrypoint for a single CPU experiment run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import traceback
from typing import Any

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
from .repository import (
    InvalidRunTransitionError,
    SQLiteRunRepository,
    TERMINAL_STATES,
)


class RunCancelled(RuntimeError):
    pass


class RunTimedOut(TimeoutError):
    pass


_signal_number: int | None = None


def _cancel_signal(signum, _frame) -> None:
    global _signal_number
    _signal_number = int(signum)
    raise RunCancelled(f"worker received signal {signum}")


def _timeout_signal(_signum, _frame) -> None:
    raise RunTimedOut("worker exceeded the experiment timeout")


def execute_run(*, database: Path, workspace: Path, run_id: str) -> int:
    repository = SQLiteRunRepository(database)
    application = SQLiteApplicationRepository(database)
    run_dir = (workspace / "runs" / run_id).resolve()
    expected_parent = (workspace / "runs").resolve()
    if run_dir.parent != expected_parent:
        raise ValueError("invalid run id path")

    signal.signal(signal.SIGTERM, _cancel_signal)
    signal.signal(signal.SIGINT, _cancel_signal)
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout_signal)
    try:
        record = repository.get_run(run_id)
        if record.cancel_requested or record.state == "cancelled":
            return 0
        repository.transition(
            run_id,
            "preflighting",
            expected="queued",
            event_type="preflight_started",
        )
        application.heartbeat(run_id, progress=0.05, phase="compiling")

        document = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
        timeout_seconds = int(
            document.get("execution", {}).get("timeout_seconds", 3600)
        )
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
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
        application.register_artifact(
            run_id, kind=artifact_kind, path=artifact.data_json
        )
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
            application.register_artifact(
                run_id, kind="field-slice", path=field_artifact.data_json
            )
        application.heartbeat(run_id, progress=1.0, phase="complete")
        repository.transition(
            run_id,
            "succeeded",
            expected="running",
            event_type="run_succeeded",
            artifact_sha256=artifact.sha256,
            artifact_path=str(artifact.root),
            payload={"artifact_path": str(artifact.root)},
        )
        return 0
    except RunCancelled as exc:
        _finish_cancelled(repository, run_id, str(exc))
        return 0
    except RunTimedOut as exc:
        _write_traceback_artifact(application, run_dir, run_id)
        current = repository.get_run(run_id)
        if current.state not in TERMINAL_STATES:
            repository.transition(
                run_id,
                "failed",
                expected={"queued", "preflighting", "running"},
                event_type="run_timed_out",
                error=str(exc),
            )
        return 1
    except Exception as exc:
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        _write_traceback_artifact(application, run_dir, run_id)
        try:
            current = repository.get_run(run_id)
            if current.state not in TERMINAL_STATES:
                repository.transition(
                    run_id,
                    "failed",
                    expected={"queued", "preflighting", "running"},
                    event_type="run_failed",
                    error=detail[:4000],
                )
        except Exception:
            traceback.print_exc()
        traceback.print_exc()
        return 1
    finally:
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0)
        application.release_cpu_lease(run_id)


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
) -> None:
    contents = traceback.format_exc()
    if not contents.strip() or contents.strip() == "NoneType: None":
        return
    # Bound persisted diagnostics while keeping the exception tail.
    encoded = contents.encode("utf-8", errors="replace")[-65_536:]
    path = run_dir / "traceback.txt"
    if not path.exists():
        _atomic_write(path, encoded.decode("utf-8", errors="replace"))
    application.register_artifact(run_id, kind="traceback", path=path)


def _check_cancel(repository: SQLiteRunRepository, run_id: str) -> None:
    if repository.get_run(run_id).cancel_requested:
        raise RunCancelled("cancellation requested")


def _finish_cancelled(
    repository: SQLiteRunRepository, run_id: str, reason: str
) -> None:
    try:
        current = repository.get_run(run_id)
        if current.state not in TERMINAL_STATES:
            repository.transition(
                run_id,
                "cancelled",
                expected={"queued", "preflighting", "running"},
                event_type="run_cancelled",
                payload={"reason": reason, "signal": _signal_number},
            )
    except InvalidRunTransitionError:
        pass


def _atomic_write(path: Path, contents: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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
