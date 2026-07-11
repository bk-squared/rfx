"""Checksummed, tolerance-replayable experiment bundles."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
from typing import Any, Mapping
import zipfile

from .analysis import analyze_run, explain_validation


REPLAY_BUNDLE_VERSION = "rfx-replay-bundle/v1"
REPLAY_REPORT_VERSION = "rfx-replay-report/v1"


def export_replay_bundle(
    service,
    run_id: str,
    *,
    destination: str | Path | None = None,
) -> Path:
    run = service.refresh(run_id)
    if run.state != "succeeded":
        raise RuntimeError("only succeeded runs can be exported as replay bundles")
    linked = service.application_repository.get_linked_run(run_id)
    revision = service.application_repository.get_revision(linked.revision_id)
    run_dir = service._run_dir(run_id)
    bundle = (
        Path(destination).expanduser().resolve()
        if destination is not None
        else run_dir / f"rfx-replay-{run_id}.zip"
    )
    if destination is not None:
        bundle.parent.mkdir(parents=True, exist_ok=True)
    analysis = analyze_run(service, run_id)
    validation = explain_validation(service, run_id)
    events = [
        {
            "sequence": item.sequence,
            "type": item.event_type,
            "state": item.state,
            "payload": item.payload,
            "created_at": item.created_at,
        }
        for item in service.repository.list_events(run_id)
    ]
    compiled_path = run_dir / "compiled.json"
    generated_path = run_dir / "generated.py"
    runtime_path = run_dir / "runtime.json"
    preflight_path = run_dir / "preflight.json"
    compiled = json.loads(compiled_path.read_text(encoding="utf-8"))
    plot_path, plot_payload = _analysis_plot(service, run_id, analysis)
    files: dict[str, bytes] = {
        "experiment/spec.json": _json_bytes(revision.spec),
        "experiment/compiled.json": _json_bytes(compiled),
        "experiment/generated.py": generated_path.read_bytes(),
        "experiment/scene.json": _json_bytes(
            __import__("rfx.experiments", fromlist=["compile_experiment"])
            .compile_experiment(revision.spec)
            .scene_preview()
        ),
        "run/events.json": _json_bytes(events),
        "run/runtime.json": runtime_path.read_bytes(),
        "run/preflight.json": preflight_path.read_bytes(),
        "analysis/metrics.json": _json_bytes(analysis),
        "analysis/validation.json": _json_bytes(validation),
        plot_path: _json_bytes(plot_payload),
        "environment/environment.json": _json_bytes(_environment()),
    }
    artifact_index = []
    for artifact in service.application_repository.list_artifacts(run_id):
        if artifact.kind in {"replay-bundle", "studio-export"}:
            continue
        path = Path(artifact.path).resolve()
        if not path.is_relative_to(service.workspace) or not path.is_file():
            raise ValueError(f"artifact escapes workspace: {artifact.id}")
        suffix = path.suffix if len(path.suffix) <= 10 else ""
        logical = f"artifacts/{_safe_name(artifact.kind)}-{artifact.id}{suffix}"
        files[logical] = path.read_bytes()
        artifact_index.append(
            {
                "artifact_id": artifact.id,
                "kind": artifact.kind,
                "logical_path": logical,
                "sha256": artifact.sha256,
            }
        )
    files["artifacts/index.json"] = _json_bytes(artifact_index)
    tolerances = {
        name: {
            "comparison": "absolute",
            "tolerance": (
                _frequency_step(plot_payload)
                if name.endswith("frequency_hz")
                else max(1e-6, abs(float(value)) * 1e-5)
            ),
        }
        for name, value in analysis["metrics"].items()
    }
    manifest_files = [
        {
            "path": name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
        for name, data in sorted(files.items())
    ]
    manifest = {
        "schema_version": REPLAY_BUNDLE_VERSION,
        "run_id": run_id,
        "experiment_id": linked.experiment_id,
        "revision_id": linked.revision_id,
        "state": run.state,
        "spec_sha256": run.spec_sha256,
        "compiled_sha256": run.compiled_sha256,
        "semantic_fingerprint": revision.semantic_fingerprint,
        "workflow": revision.spec["kind"],
        "validation_status": revision.validation_state,
        "metric_baseline": analysis["metrics"],
        "replay_tolerances": tolerances,
        "limitations": analysis["limitations"],
        "files": manifest_files,
    }
    files["manifest.json"] = _json_bytes(manifest)
    temporary = bundle.with_suffix(bundle.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            _write_deterministic(archive, name, data)
    os.replace(temporary, bundle)
    verify_replay_bundle(bundle)
    service.application_repository.register_artifact(
        run_id, kind="replay-bundle", path=bundle
    )
    return bundle


def verify_replay_bundle(bundle: str | Path) -> dict[str, Any]:
    path = Path(bundle).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("bundle contains duplicate logical paths")
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                raise ValueError("bundle contains unsafe path")
        if "manifest.json" not in names:
            raise ValueError("bundle has no manifest.json")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema_version") != REPLAY_BUNDLE_VERSION:
            raise ValueError("unsupported replay bundle version")
        declared = {item["path"]: item for item in manifest.get("files", [])}
        actual_files = set(names) - {"manifest.json"}
        if set(declared) != actual_files:
            raise ValueError("bundle manifest file set mismatch")
        for name, item in declared.items():
            data = archive.read(name)
            if len(data) != item["size_bytes"]:
                raise ValueError(f"bundle size mismatch: {name}")
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"bundle checksum mismatch: {name}")
        spec = json.loads(archive.read("experiment/spec.json"))
        canonical = json.dumps(
            spec, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != manifest["spec_sha256"]:
            raise ValueError("bundle spec digest mismatch")
        artifact_index = json.loads(archive.read("artifacts/index.json"))
        artifact_ids: set[str] = set()
        logical_paths: set[str] = set()
        for item in artifact_index:
            artifact_id = item.get("artifact_id")
            logical_path = item.get("logical_path")
            expected_sha = item.get("sha256")
            if (
                not isinstance(artifact_id, str)
                or artifact_id in artifact_ids
                or not isinstance(logical_path, str)
                or logical_path in logical_paths
                or logical_path not in declared
                or not isinstance(expected_sha, str)
                or len(expected_sha) != 64
            ):
                raise ValueError("bundle artifact index is invalid")
            artifact_ids.add(artifact_id)
            logical_paths.add(logical_path)
            if hashlib.sha256(archive.read(logical_path)).hexdigest() != expected_sha:
                raise ValueError(
                    f"bundle artifact index digest mismatch: {artifact_id}"
                )
    return manifest


def replay_bundle(bundle: str | Path, workspace: str | Path) -> dict[str, Any]:
    manifest = verify_replay_bundle(bundle)
    with zipfile.ZipFile(Path(bundle).expanduser().resolve()) as archive:
        spec = json.loads(archive.read("experiment/spec.json"))
    from .service import ExperimentService

    service = ExperimentService(workspace)
    run = service.run_sync(spec)
    if run.state != "succeeded":
        raise RuntimeError(run.error or f"replay ended {run.state}")
    analysis = analyze_run(service, run.id)
    comparisons = {}
    passed = True
    for name, baseline in manifest["metric_baseline"].items():
        replayed = analysis["metrics"][name]
        tolerance = manifest["replay_tolerances"][name]["tolerance"]
        delta = abs(float(replayed) - float(baseline))
        metric_passed = delta <= float(tolerance)
        comparisons[name] = {
            "baseline": baseline,
            "replayed": replayed,
            "absolute_delta": delta,
            "tolerance": tolerance,
            "passed": metric_passed,
        }
        passed = passed and metric_passed
    return {
        "schema_version": REPLAY_REPORT_VERSION,
        "source_bundle": str(Path(bundle).expanduser().resolve()),
        "source_run_id": manifest["run_id"],
        "replay_run_id": run.id,
        "spec_sha256": run.spec_sha256,
        "semantic_fingerprint": manifest["semantic_fingerprint"],
        "environment": _environment(),
        "metrics": comparisons,
        "passed": passed,
        "reproducibility_claim": "declared physics metrics within explicit tolerance; not bit-identical output",
    }


def _s11_plot(service, run_id: str) -> dict[str, Any]:
    artifact = next(
        item
        for item in service.application_repository.list_artifacts(run_id)
        if item.kind in {"s11", "sparameters"}
    )
    payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
    points = (
        payload["points"]
        if artifact.kind == "s11"
        else [
            {"frequency_hz": item["frequency_hz"], **item["matrix"][0][0]}
            for item in payload["points"]
        ]
    )
    return {
        "schema_version": "rfx-plot-data/v1",
        "kind": "s11-magnitude",
        "x": [item["frequency_hz"] for item in points],
        "y": [item["magnitude_db"] for item in points],
        "x_label": "Frequency (Hz)",
        "y_label": "|S11| (dB)",
        "artifact_sha256": artifact.sha256,
    }


def _analysis_plot(
    service, run_id: str, analysis: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    if analysis["analysis_kind"] != "reflection-transmission":
        return "analysis/plots/s11.json", _s11_plot(service, run_id)
    artifact = next(
        item
        for item in service.application_repository.list_artifacts(run_id)
        if item.kind == "reflection-transmission"
    )
    payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
    points = payload["points"]
    return (
        "analysis/plots/reflection-transmission.json",
        {
            "schema_version": "rfx-plot-data/v1",
            "kind": "reflection-transmission",
            "x": [item["frequency_hz"] for item in points],
            "series": {
                "reflection": [item["reflection"] for item in points],
                "transmission": [item["transmission"] for item in points],
                "analytic_reflection": [item["analytic_reflection"] for item in points],
                "analytic_transmission": [
                    item["analytic_transmission"] for item in points
                ],
            },
            "x_label": "Frequency (Hz)",
            "y_label": "Power ratio",
            "artifact_sha256": artifact.sha256,
        },
    )


def _frequency_step(plot: Mapping[str, Any]) -> float:
    values = plot["x"]
    if len(values) < 2:
        return 0.0
    return float(min(abs(right - left) for left, right in zip(values, values[1:])))


def _environment() -> dict[str, Any]:
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "rfx-fdtd": version("rfx-fdtd"),
            "jax": version("jax"),
            "jaxlib": version("jaxlib"),
            "numpy": version("numpy"),
        },
        "backend_policy": "cpu",
    }


def _safe_name(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value
    )
    return safe[:64] or "artifact"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _write_deterministic(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data)
