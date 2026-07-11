"""Content-addressed immutable S11 result artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np


S11_SCHEMA_VERSION = "rfx-s11-artifact/v1"
S11_MANIFEST_VERSION = "rfx-s11-manifest/v1"
FIELD_SLICE_SCHEMA_VERSION = "rfx-field-slice-artifact/v1"
FIELD_SLICE_MANIFEST_VERSION = "rfx-field-slice-manifest/v1"
SPARAMETERS_SCHEMA_VERSION = "rfx-sparameters-artifact/v1"
SPARAMETERS_MANIFEST_VERSION = "rfx-sparameters-manifest/v1"
REFLECTION_TRANSMISSION_SCHEMA_VERSION = "rfx-reflection-transmission-artifact/v1"
REFLECTION_TRANSMISSION_MANIFEST_VERSION = "rfx-reflection-transmission-manifest/v1"


@dataclass(frozen=True)
class S11Artifact:
    sha256: str
    root: Path
    data_json: Path
    manifest_json: Path


@dataclass(frozen=True)
class FieldSliceArtifact:
    sha256: str
    root: Path
    data_json: Path
    manifest_json: Path


@dataclass(frozen=True)
class ResultArtifact:
    sha256: str
    root: Path
    data_json: Path
    manifest_json: Path


def export_s11_artifact(
    store_root: str | Path,
    *,
    result: Any,
    run_id: str,
    spec_sha256: str,
    compiled_sha256: str,
    runtime: Mapping[str, Any],
    reference_impedance_ohm: float = 50.0,
) -> S11Artifact:
    """Export raw one-port S11 data to a content-addressed read-only folder."""

    if (
        getattr(result, "s_params", None) is None
        or getattr(result, "freqs", None) is None
    ):
        raise ValueError(
            "simulation result does not contain S-parameters and frequencies"
        )
    s_params = np.asarray(result.s_params)
    frequencies = np.asarray(result.freqs, dtype=np.float64)
    if s_params.ndim != 3 or s_params.shape[0] < 1 or s_params.shape[1] < 1:
        raise ValueError("result.s_params must have shape (ports, ports, frequencies)")
    s11 = np.asarray(s_params[0, 0, :], dtype=np.complex128)
    if frequencies.shape != s11.shape:
        raise ValueError("result frequencies do not match S11 samples")
    if not np.all(np.isfinite(frequencies)) or not np.all(np.isfinite(s11)):
        raise ValueError("S11 artifact refuses non-finite result data")

    points = []
    for frequency, value in zip(frequencies, s11):
        magnitude = float(abs(value))
        points.append(
            {
                "frequency_hz": float(frequency),
                "real": float(value.real),
                "imag": float(value.imag),
                "magnitude_db": 20.0 * math.log10(max(magnitude, 1e-15)),
            }
        )
    payload = {
        "schema_version": S11_SCHEMA_VERSION,
        "run_id": run_id,
        "spec_sha256": spec_sha256,
        "compiled_sha256": compiled_sha256,
        "reference_impedance_ohm": float(reference_impedance_ohm),
        "runtime": dict(runtime),
        "points": points,
    }
    data = _json_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    root = Path(store_root).expanduser().resolve() / "sha256"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / digest

    if destination.exists():
        artifact = _artifact_paths(destination, digest)
        verify_s11_artifact(artifact)
        if artifact.data_json.read_bytes() != data:
            raise FileExistsError(f"artifact digest collision at {destination}")
        return artifact

    temporary = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=root))
    try:
        data_path = temporary / "s11.json"
        data_path.write_bytes(data)
        manifest = {
            "schema_version": S11_MANIFEST_VERSION,
            "artifact_sha256": digest,
            "files": [{"path": "s11.json", "sha256": digest, "bytes": len(data)}],
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        os.chmod(data_path, 0o444)
        os.chmod(manifest_path, 0o444)
        try:
            os.rename(temporary, destination)
        except FileExistsError:
            shutil.rmtree(temporary)
            artifact = _artifact_paths(destination, digest)
            verify_s11_artifact(artifact)
            if artifact.data_json.read_bytes() != data:
                raise FileExistsError(f"artifact digest collision at {destination}")
            return artifact
        os.chmod(destination, 0o555)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    artifact = _artifact_paths(destination, digest)
    verify_s11_artifact(artifact)
    return artifact


def verify_s11_artifact(artifact: S11Artifact | str | Path) -> S11Artifact:
    """Verify the content address, manifest, and declared file hash."""

    if isinstance(artifact, S11Artifact):
        record = artifact
    else:
        root = Path(artifact).expanduser().resolve()
        record = _artifact_paths(root, root.name)
    if not record.data_json.is_file() or not record.manifest_json.is_file():
        raise ValueError(f"incomplete S11 artifact at {record.root}")
    data = record.data_json.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != record.sha256 or record.root.name != record.sha256:
        raise ValueError("S11 artifact content address mismatch")
    manifest = json.loads(record.manifest_json.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != S11_MANIFEST_VERSION:
        raise ValueError("unsupported S11 artifact manifest")
    if manifest.get("artifact_sha256") != actual:
        raise ValueError("S11 artifact manifest digest mismatch")
    files = manifest.get("files")
    if files != [{"bytes": len(data), "path": "s11.json", "sha256": actual}]:
        raise ValueError("S11 artifact manifest file declaration mismatch")
    payload = json.loads(data)
    if payload.get("schema_version") != S11_SCHEMA_VERSION:
        raise ValueError("unsupported S11 artifact data schema")
    return record


def export_field_slice_artifact(
    store_root: str | Path,
    *,
    result: Any,
    spec_document: Mapping[str, Any],
    run_id: str,
    spec_sha256: str,
    compiled_sha256: str,
    runtime: Mapping[str, Any],
) -> FieldSliceArtifact | None:
    """Export the requested final-state field plane as immutable plot data."""

    observation = next(
        (
            item
            for item in spec_document.get("observations", [])
            if item.get("kind") == "field_snapshot"
        ),
        None,
    )
    if observation is None:
        return None
    if "field-slice" not in spec_document.get("artifacts", {}).get("save", []):
        return None

    grid = getattr(result, "grid", None)
    state = getattr(result, "state", None)
    component = str(observation["component"]).lower()
    if grid is None or state is None or not hasattr(state, component):
        raise ValueError(
            "simulation result cannot satisfy requested field-slice artifact"
        )

    axis_name = str(observation["axis"]).lower()
    axis_index = {"x": 0, "y": 1, "z": 2}[axis_name]
    coordinate = float(observation["coordinate_m"])
    domain = tuple(float(value) for value in spec_document["simulation"]["domain_m"])
    position = [value / 2.0 for value in domain]
    position[axis_index] = coordinate
    grid_index = grid.position_to_index(tuple(position))[axis_index]
    selector = list(grid.interior)
    selector[axis_index] = grid_index
    values = np.asarray(getattr(state, component)[tuple(selector)], dtype=np.float32)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("field-slice artifact requires one finite 2-D plane")

    other_axes = [axis for axis in (0, 1, 2) if axis != axis_index]
    axis_labels = ["xyz"[axis] for axis in other_axes]
    actual_coordinate = float(
        (grid_index - grid.axis_pads[axis_index]) * float(grid.dx)
    )
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    payload = {
        "schema_version": FIELD_SLICE_SCHEMA_VERSION,
        "run_id": run_id,
        "spec_sha256": spec_sha256,
        "compiled_sha256": compiled_sha256,
        "observation_id": observation["id"],
        "component": component,
        "units": "V/m" if component.startswith("e") else "A/m",
        "slice_axis": axis_name,
        "requested_coordinate_m": coordinate,
        "actual_coordinate_m": actual_coordinate,
        "axis_labels": axis_labels,
        "extent_m": [
            [0.0, domain[other_axes[0]]],
            [0.0, domain[other_axes[1]]],
        ],
        "shape": list(values.shape),
        "value_encoding": "row-major",
        "values": values.tolist(),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "maximum_absolute": max_abs,
        "runtime": dict(runtime),
    }
    data = _json_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    root = Path(store_root).expanduser().resolve() / "sha256"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / digest

    if destination.exists():
        artifact = _field_artifact_paths(destination, digest)
        verify_field_slice_artifact(artifact)
        if artifact.data_json.read_bytes() != data:
            raise FileExistsError(f"artifact digest collision at {destination}")
        return artifact

    temporary = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=root))
    try:
        data_path = temporary / "field-slice.json"
        data_path.write_bytes(data)
        manifest = {
            "schema_version": FIELD_SLICE_MANIFEST_VERSION,
            "artifact_sha256": digest,
            "files": [
                {
                    "path": "field-slice.json",
                    "sha256": digest,
                    "bytes": len(data),
                }
            ],
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        os.chmod(data_path, 0o444)
        os.chmod(manifest_path, 0o444)
        try:
            os.rename(temporary, destination)
        except FileExistsError:
            shutil.rmtree(temporary)
            artifact = _field_artifact_paths(destination, digest)
            verify_field_slice_artifact(artifact)
            if artifact.data_json.read_bytes() != data:
                raise FileExistsError(f"artifact digest collision at {destination}")
            return artifact
        os.chmod(destination, 0o555)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    artifact = _field_artifact_paths(destination, digest)
    verify_field_slice_artifact(artifact)
    return artifact


def export_sparameters_artifact(
    store_root: str | Path,
    *,
    result: Any,
    run_id: str,
    spec_sha256: str,
    compiled_sha256: str,
    runtime: Mapping[str, Any],
) -> ResultArtifact:
    values = np.asarray(getattr(result, "s_params", None), dtype=np.complex128)
    frequencies = np.asarray(getattr(result, "freqs", None), dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != values.shape[1]:
        raise ValueError(
            "S-parameter artifact requires a square (ports, ports, frequency) matrix"
        )
    if frequencies.ndim != 1 or values.shape[2] != len(frequencies):
        raise ValueError("S-parameter matrix frequency dimension mismatch")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(frequencies)):
        raise ValueError("S-parameter artifact refuses non-finite data")
    port_names = tuple(
        getattr(result, "port_names", None)
        or tuple(f"port-{index + 1}" for index in range(values.shape[0]))
    )
    if len(port_names) != values.shape[0]:
        raise ValueError("S-parameter port name count mismatch")
    points = []
    for frequency_index, frequency in enumerate(frequencies):
        matrix = []
        for output_port in range(values.shape[0]):
            row = []
            for input_port in range(values.shape[1]):
                value = values[output_port, input_port, frequency_index]
                row.append(
                    {
                        "real": float(value.real),
                        "imag": float(value.imag),
                        "magnitude_db": 20.0
                        * math.log10(max(float(abs(value)), 1e-15)),
                    }
                )
            matrix.append(row)
        points.append({"frequency_hz": float(frequency), "matrix": matrix})
    return _export_result_artifact(
        store_root,
        filename="sparameters.json",
        manifest_version=SPARAMETERS_MANIFEST_VERSION,
        payload={
            "schema_version": SPARAMETERS_SCHEMA_VERSION,
            "run_id": run_id,
            "spec_sha256": spec_sha256,
            "compiled_sha256": compiled_sha256,
            "port_names": list(port_names),
            "runtime": dict(runtime),
            "points": points,
        },
    )


def export_reflection_transmission_artifact(
    store_root: str | Path,
    *,
    result: Any,
    run_id: str,
    spec_sha256: str,
    compiled_sha256: str,
    runtime: Mapping[str, Any],
) -> ResultArtifact:
    raw = getattr(result, "reflection_transmission", None)
    if not isinstance(raw, Mapping):
        raise ValueError("result has no reflection/transmission observable")
    keys = (
        "frequencies_hz",
        "reflection",
        "transmission",
        "analytic_reflection",
        "analytic_transmission",
        "signal_valid",
    )
    lengths = {len(raw[key]) for key in keys}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2:
        raise ValueError("reflection/transmission observable lengths do not match")
    numeric = np.asarray(
        [raw[key] for key in keys if key != "signal_valid"], dtype=np.float64
    )
    if not np.all(np.isfinite(numeric)):
        raise ValueError("reflection/transmission artifact refuses non-finite data")
    points = [
        {
            "frequency_hz": float(frequency),
            "reflection": float(reflection),
            "transmission": float(transmission),
            "analytic_reflection": float(analytic_reflection),
            "analytic_transmission": float(analytic_transmission),
            "signal_valid": bool(signal_valid),
        }
        for frequency, reflection, transmission, analytic_reflection, analytic_transmission, signal_valid in zip(
            *(raw[key] for key in keys)
        )
    ]
    return _export_result_artifact(
        store_root,
        filename="reflection-transmission.json",
        manifest_version=REFLECTION_TRANSMISSION_MANIFEST_VERSION,
        payload={
            "schema_version": REFLECTION_TRANSMISSION_SCHEMA_VERSION,
            "run_id": run_id,
            "spec_sha256": spec_sha256,
            "compiled_sha256": compiled_sha256,
            "runtime": dict(runtime),
            "points": points,
        },
    )


def verify_field_slice_artifact(
    artifact: FieldSliceArtifact | str | Path,
) -> FieldSliceArtifact:
    """Verify field-plane content address, manifest, shape, and finite values."""

    if isinstance(artifact, FieldSliceArtifact):
        record = artifact
    else:
        root = Path(artifact).expanduser().resolve()
        record = _field_artifact_paths(root, root.name)
    if not record.data_json.is_file() or not record.manifest_json.is_file():
        raise ValueError(f"incomplete field-slice artifact at {record.root}")
    data = record.data_json.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != record.sha256 or record.root.name != record.sha256:
        raise ValueError("field-slice artifact content address mismatch")
    manifest = json.loads(record.manifest_json.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FIELD_SLICE_MANIFEST_VERSION:
        raise ValueError("unsupported field-slice artifact manifest")
    if manifest.get("artifact_sha256") != actual:
        raise ValueError("field-slice artifact manifest digest mismatch")
    if manifest.get("files") != [
        {"bytes": len(data), "path": "field-slice.json", "sha256": actual}
    ]:
        raise ValueError("field-slice artifact manifest file declaration mismatch")
    payload = json.loads(data)
    if payload.get("schema_version") != FIELD_SLICE_SCHEMA_VERSION:
        raise ValueError("unsupported field-slice artifact data schema")
    values = np.asarray(payload.get("values"), dtype=np.float64)
    if list(values.shape) != payload.get("shape") or values.ndim != 2:
        raise ValueError("field-slice artifact shape mismatch")
    if not np.all(np.isfinite(values)):
        raise ValueError("field-slice artifact contains non-finite values")
    return record


def verify_sparameters_artifact(
    artifact: ResultArtifact | str | Path,
) -> ResultArtifact:
    record, payload = _verify_result_artifact(
        artifact,
        filename="sparameters.json",
        schema_version=SPARAMETERS_SCHEMA_VERSION,
        manifest_version=SPARAMETERS_MANIFEST_VERSION,
    )
    points = payload.get("points")
    port_names = payload.get("port_names")
    if not isinstance(points, list) or not points or not isinstance(port_names, list):
        raise ValueError("invalid S-parameter artifact payload")
    port_count = len(port_names)
    for point in points:
        matrix = point.get("matrix")
        if len(matrix) != port_count or any(len(row) != port_count for row in matrix):
            raise ValueError("S-parameter artifact matrix shape mismatch")
        numeric = [float(point["frequency_hz"])]
        for row in matrix:
            for value in row:
                numeric.extend(
                    [
                        float(value["real"]),
                        float(value["imag"]),
                        float(value["magnitude_db"]),
                    ]
                )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("S-parameter artifact contains non-finite data")
    return record


def verify_reflection_transmission_artifact(
    artifact: ResultArtifact | str | Path,
) -> ResultArtifact:
    record, payload = _verify_result_artifact(
        artifact,
        filename="reflection-transmission.json",
        schema_version=REFLECTION_TRANSMISSION_SCHEMA_VERSION,
        manifest_version=REFLECTION_TRANSMISSION_MANIFEST_VERSION,
    )
    points = payload.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError("invalid reflection/transmission artifact payload")
    numeric = np.asarray(
        [
            [
                point["frequency_hz"],
                point["reflection"],
                point["transmission"],
                point["analytic_reflection"],
                point["analytic_transmission"],
            ]
            for point in points
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(numeric)):
        raise ValueError("reflection/transmission artifact contains non-finite data")
    return record


def _artifact_paths(root: Path, digest: str) -> S11Artifact:
    return S11Artifact(
        sha256=digest,
        root=root,
        data_json=root / "s11.json",
        manifest_json=root / "manifest.json",
    )


def _field_artifact_paths(root: Path, digest: str) -> FieldSliceArtifact:
    return FieldSliceArtifact(
        sha256=digest,
        root=root,
        data_json=root / "field-slice.json",
        manifest_json=root / "manifest.json",
    )


def _export_result_artifact(
    store_root: str | Path,
    *,
    filename: str,
    manifest_version: str,
    payload: Mapping[str, Any],
) -> ResultArtifact:
    data = _json_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    root = Path(store_root).expanduser().resolve() / "sha256"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / digest
    if destination.exists():
        artifact = _result_artifact_paths(destination, digest, filename)
        record, _ = _verify_result_artifact(
            artifact,
            filename=filename,
            schema_version=str(payload["schema_version"]),
            manifest_version=manifest_version,
        )
        if record.data_json.read_bytes() != data:
            raise FileExistsError(f"artifact digest collision at {destination}")
        return artifact

    temporary = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=root))
    try:
        data_path = temporary / filename
        data_path.write_bytes(data)
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(
            _json_bytes(
                {
                    "schema_version": manifest_version,
                    "artifact_sha256": digest,
                    "files": [{"path": filename, "sha256": digest, "bytes": len(data)}],
                }
            )
        )
        os.chmod(data_path, 0o444)
        os.chmod(manifest_path, 0o444)
        try:
            os.rename(temporary, destination)
        except FileExistsError:
            shutil.rmtree(temporary)
        os.chmod(destination, 0o555)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    artifact = _result_artifact_paths(destination, digest, filename)
    _verify_result_artifact(
        artifact,
        filename=filename,
        schema_version=str(payload["schema_version"]),
        manifest_version=manifest_version,
    )
    return artifact


def _verify_result_artifact(
    artifact: ResultArtifact | str | Path,
    *,
    filename: str,
    schema_version: str,
    manifest_version: str,
) -> tuple[ResultArtifact, dict[str, Any]]:
    if isinstance(artifact, ResultArtifact):
        record = artifact
    else:
        root = Path(artifact).expanduser().resolve()
        record = _result_artifact_paths(root, root.name, filename)
    if not record.data_json.is_file() or not record.manifest_json.is_file():
        raise ValueError(f"incomplete result artifact at {record.root}")
    data = record.data_json.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != record.sha256 or record.root.name != record.sha256:
        raise ValueError("result artifact content address mismatch")
    manifest = json.loads(record.manifest_json.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != manifest_version:
        raise ValueError("unsupported result artifact manifest")
    if manifest.get("artifact_sha256") != actual:
        raise ValueError("result artifact manifest digest mismatch")
    if manifest.get("files") != [
        {"bytes": len(data), "path": filename, "sha256": actual}
    ]:
        raise ValueError("result artifact manifest file declaration mismatch")
    payload = json.loads(data)
    if payload.get("schema_version") != schema_version:
        raise ValueError("unsupported result artifact data schema")
    return record, payload


def _result_artifact_paths(root: Path, digest: str, filename: str) -> ResultArtifact:
    return ResultArtifact(
        sha256=digest,
        root=root,
        data_json=root / filename,
        manifest_json=root / "manifest.json",
    )


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
