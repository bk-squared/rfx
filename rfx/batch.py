"""Batch simulation and parameter sweep utilities for ML data generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import time
from typing import Callable, Any

import numpy as np


BATCH_MANIFEST_SCHEMA = "rfx-batch-manifest-v1"


@dataclass(frozen=True)
class BatchCaseResult:
    """Result record returned by :func:`run_batch_with_manifest`.

    ``result`` is ``None`` for cases skipped through resume.
    """

    case_id: str
    params: dict[str, Any]
    status: str
    metrics: dict[str, Any]
    artifacts: dict[str, str]
    result: Any = None
    error: str | None = None
    skipped: bool = False


class ParameterSweep:
    """Define a multi-dimensional parameter sweep.

    Parameters are given as keyword arguments, each mapping to an array
    of values.  The sweep iterates over the full Cartesian product.

    Example
    -------
    >>> sweep = ParameterSweep(width=[0.01, 0.02], eps_r=[2.0, 4.0])
    >>> sweep.total
    4
    >>> list(sweep.combinations())
    [{'width': 0.01, 'eps_r': 2.0}, {'width': 0.01, 'eps_r': 4.0}, ...]
    """

    def __init__(self, **params):
        self._keys = list(params.keys())
        self._values = [np.asarray(v).ravel() for v in params.values()]

    @property
    def keys(self) -> list[str]:
        return list(self._keys)

    @property
    def total(self) -> int:
        n = 1
        for v in self._values:
            n *= len(v)
        return n

    def combinations(self):
        """Iterate over all parameter combinations as dicts."""
        for combo in itertools.product(*self._values):
            yield dict(zip(self._keys, [_jsonable(c) for c in combo]))


def _jsonable(value: Any) -> Any:
    """Convert common NumPy/JAX-ish values into deterministic JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def case_id_from_params(params: dict[str, Any], *, prefix: str = "case") -> str:
    """Return a stable, compact case id for a parameter dictionary."""
    canonical = json.dumps(_jsonable(params), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _json_digest(value: Any, *, length: int = 16) -> str:
    canonical = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def _run_fingerprint(run_kwargs: dict[str, Any], user_fingerprint: str | None) -> str:
    """Return a stable cache-validity token for run settings."""
    payload = {
        "run_kwargs": _jsonable(run_kwargs),
        "user_fingerprint": user_fingerprint,
    }
    return _json_digest(payload)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": BATCH_MANIFEST_SCHEMA,
            "created_at": _utc_now(),
            "updated_at": None,
            "sweep": {},
            "cases": {},
        }
    with open(path) as f:
        manifest = json.load(f)
    if manifest.get("schema_version") != BATCH_MANIFEST_SCHEMA:
        raise ValueError(
            f"Unsupported batch manifest schema: {manifest.get('schema_version')!r}"
        )
    manifest.setdefault("cases", {})
    return manifest


def _write_manifest_atomic(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = _utc_now()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=_jsonable)
        f.write("\n")
    tmp.replace(path)


def _artifact_path(root: Path, artifact_path: str) -> Path | None:
    if not isinstance(artifact_path, str) or not artifact_path:
        return None
    path = Path(artifact_path)
    if str(path) == ".":
        return None
    candidate = path if path.is_absolute() else root / path
    if _path_relative_to(candidate, root) is None:
        return None
    return candidate


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _artifact_metadata(root: Path, artifacts: dict[str, str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name, artifact_path in artifacts.items():
        candidate = _artifact_path(root, artifact_path)
        if candidate is None or not candidate.is_file():
            raise ValueError(
                "artifact paths must point to files inside the output directory"
            )
        stat = candidate.stat()
        metadata[str(name)] = {
            "path": artifact_path,
            "size": stat.st_size,
            "sha256": _hash_file(candidate),
        }
    return metadata


def _artifact_metadata_valid(
    root: Path,
    artifacts: dict[str, str],
    artifact_metadata: Any,
) -> bool:
    if not isinstance(artifact_metadata, dict):
        return False
    try:
        current = _artifact_metadata(root, artifacts)
    except (OSError, ValueError):
        return False
    return current == artifact_metadata


def _completed_record_valid(
    root: Path,
    record: dict[str, Any],
    expected_run_fingerprint: str,
    expected_params: dict[str, Any],
    expected_params_digest: str,
) -> bool:
    metrics = record.get("metrics")
    artifacts = record.get("artifacts")
    return (
        record.get("status") == "completed"
        and record.get("run_fingerprint") == expected_run_fingerprint
        and record.get("params") == expected_params
        and record.get("params_digest") == expected_params_digest
        and isinstance(metrics, dict)
        and len(metrics) > 0
        and isinstance(artifacts, dict)
        and _artifact_metadata_valid(root, artifacts, record.get("artifact_metadata"))
    )


def _path_relative_to(path: Path, root: Path) -> str | None:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def _normalize_artifacts(
    output_root: Path,
    case_dir: Path,
    artifacts: dict[str, Any],
) -> dict[str, str]:
    """Normalize artifact paths to manifest-stable strings.

    ``artifact_fn`` is usually called with ``case_dir`` and naturally returns
    case-local paths such as ``"metrics.json"``.  The manifest, however, needs
    paths that can be validated from the output root during a later resume.
    Paths already relative to ``output_root`` are preserved; absolute paths
    inside ``output_root`` are stored relatively.  Absolute paths outside the
    output root and ``..`` escapes are rejected so manifests remain portable.
    """
    normalized: dict[str, str] = {}
    for name, value in artifacts.items():
        if isinstance(value, Path):
            artifact_path = value
        elif isinstance(value, str):
            artifact_path = Path(value)
        else:
            raise ValueError("artifact_fn must return path strings or Path values")

        if str(artifact_path) in ("", "."):
            raise ValueError("artifact paths must not be empty or '.'")

        if artifact_path.is_absolute():
            rel = _path_relative_to(artifact_path, output_root)
            if rel is None:
                raise ValueError("artifact paths must stay inside output_dir")
            normalized[str(name)] = rel
            continue

        root_candidate = output_root / artifact_path
        case_candidate = case_dir / artifact_path
        if (
            _path_relative_to(root_candidate, output_root) is None
            and _path_relative_to(case_candidate, output_root) is None
        ):
            raise ValueError("artifact paths must stay inside output_dir")
        if root_candidate.exists():
            normalized[str(name)] = artifact_path.as_posix()
        elif case_candidate.exists():
            normalized[str(name)] = case_candidate.relative_to(output_root).as_posix()
        else:
            # Preserve the caller-provided path so resume validation will fail
            # rather than silently blessing a missing artifact.
            normalized[str(name)] = artifact_path.as_posix()
    return normalized


def run_batch_with_manifest(
    sim_factory: Callable[..., Any],
    sweep: ParameterSweep,
    output_dir: str | Path,
    *,
    run_kwargs: dict | None = None,
    metric_fn: Callable[[Any], dict[str, Any]] | None = None,
    artifact_fn: Callable[
        [Path, dict[str, Any], Any], dict[str, str | Path] | None
    ] | None = None,
    resume: bool = True,
    manifest_name: str = "manifest.json",
    run_fingerprint: str | None = None,
) -> list[BatchCaseResult]:
    """Run a sequential sweep with durable per-case provenance.

    The runner records stable case ids, parameters, status, timings, metrics,
    and optional artifact paths.  ``artifact_fn`` may return paths relative to
    the case directory, paths relative to the output root, or absolute paths
    inside the output root.
    The runner automatically fingerprints ``run_kwargs`` so changing run
    settings (for example ``n_steps``) invalidates resume.  Pass
    ``run_fingerprint`` when simulation factory, metric semantics, or external
    inputs changed but ``run_kwargs`` did not.  When ``resume=True``, a case is
    skipped only if the manifest record is completed, has the expected run
    fingerprint, contains at least one metric, and all declared artifact paths
    still exist.
    """
    if run_kwargs is None:
        run_kwargs = {}
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / manifest_name
    manifest = _read_manifest(manifest_path)
    manifest["sweep"] = {"keys": sweep.keys, "total": sweep.total}
    run_kwargs_json = _jsonable(run_kwargs)
    expected_run_fingerprint = _run_fingerprint(run_kwargs_json, run_fingerprint)
    manifest["run_config"] = {
        "run_kwargs": run_kwargs_json,
        "user_fingerprint": run_fingerprint,
        "run_fingerprint": expected_run_fingerprint,
    }

    returned: list[BatchCaseResult] = []
    for params in sweep.combinations():
        params_json = _jsonable(params)
        case_id = case_id_from_params(params_json)
        params_digest = _json_digest(params_json)
        case_dir = output_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        existing = manifest["cases"].get(case_id)
        if (
            resume
            and isinstance(existing, dict)
            and _completed_record_valid(
                output_root,
                existing,
                expected_run_fingerprint,
                params_json,
                params_digest,
            )
        ):
            returned.append(BatchCaseResult(
                case_id=case_id,
                params=existing.get("params", params_json),
                status="skipped",
                metrics=existing.get("metrics", {}),
                artifacts=existing.get("artifacts", {}),
                result=None,
                skipped=True,
            ))
            continue

        start = time.time()
        record = {
            "case_id": case_id,
            "params": params_json,
            "params_digest": params_digest,
            "run_fingerprint": expected_run_fingerprint,
            "run_config": manifest["run_config"],
            "status": "running",
            "started_at": _utc_now(),
            "completed_at": None,
            "duration_s": None,
            "metrics": {},
            "artifacts": {},
            "error": None,
        }
        manifest["cases"][case_id] = record
        _write_manifest_atomic(manifest_path, manifest)

        try:
            sim = sim_factory(**params)
            result = sim.run(**run_kwargs)
            metrics = _jsonable(
                metric_fn(result) if metric_fn else {"completed": True}
            )
            if not isinstance(metrics, dict):
                raise ValueError("metric_fn must return a dict")
            artifacts = (
                artifact_fn(case_dir, params_json, result) if artifact_fn else {}
            )
            artifacts = _jsonable(artifacts or {})
            if not isinstance(artifacts, dict):
                raise ValueError("artifact_fn must return a dict of artifact paths")
            artifacts = _normalize_artifacts(output_root, case_dir, artifacts)
            artifact_metadata = _artifact_metadata(output_root, artifacts)

            record.update({
                "status": "completed",
                "completed_at": _utc_now(),
                "duration_s": time.time() - start,
                "metrics": metrics,
                "artifacts": artifacts,
                "artifact_metadata": artifact_metadata,
                "error": None,
            })
            _write_manifest_atomic(manifest_path, manifest)
            returned.append(BatchCaseResult(
                case_id=case_id,
                params=params_json,
                status="completed",
                metrics=metrics,
                artifacts=artifacts,
                result=result,
            ))
        except Exception as exc:
            record.update({
                "status": "failed",
                "completed_at": _utc_now(),
                "duration_s": time.time() - start,
                "error": f"{type(exc).__name__}: {exc}",
            })
            _write_manifest_atomic(manifest_path, manifest)
            returned.append(BatchCaseResult(
                case_id=case_id,
                params=params_json,
                status="failed",
                metrics={},
                artifacts={},
                result=None,
                error=record["error"],
            ))
            raise

    return returned


def run_batch(
    sim_factory: Callable[..., Any],
    sweep: ParameterSweep,
    *,
    run_kwargs: dict | None = None,
) -> list[tuple[dict, Any]]:
    """Run simulations for all parameter combinations.

    Parameters
    ----------
    sim_factory : callable
        Function that takes keyword args from the sweep and returns
        a configured Simulation object (or any object with a .run() method).
    sweep : ParameterSweep
        Parameter sweep definition.
    run_kwargs : dict or None
        Extra keyword arguments passed to sim.run().

    Returns
    -------
    list of (params_dict, result) tuples
    """
    if run_kwargs is None:
        run_kwargs = {}
    results = []
    for i, params in enumerate(sweep.combinations()):
        sim = sim_factory(**params)
        result = sim.run(**run_kwargs)
        results.append((params, result))
    return results


class SimulationDataset:
    """Structured dataset from batch simulation results.

    Parameters
    ----------
    inputs : ndarray, shape (n_samples, n_params)
    outputs : ndarray, shape (n_samples, n_outputs)
    input_keys : list of str
    """

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray,
                 input_keys: list[str]):
        self.inputs = inputs
        self.outputs = outputs
        self.input_keys = input_keys

    @classmethod
    def from_results(
        cls,
        results: list[tuple[dict, Any]],
        input_keys: list[str],
        output_fn: Callable,
    ) -> "SimulationDataset":
        """Build dataset from run_batch() results.

        Parameters
        ----------
        results : list of (params_dict, Result)
        input_keys : which param keys to use as inputs
        output_fn : callable(Result) -> 1D array of output values
        """
        X_rows, Y_rows = [], []
        for params, result in results:
            X_rows.append([params[k] for k in input_keys])
            Y_rows.append(np.asarray(output_fn(result)).ravel())
        return cls(
            inputs=np.array(X_rows),
            outputs=np.array(Y_rows),
            input_keys=input_keys,
        )

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, Y) arrays."""
        return self.inputs, self.outputs

    def to_hdf5(self, path: str):
        """Save to HDF5 file."""
        import h5py
        with h5py.File(path, "w") as f:
            f.create_dataset("inputs", data=self.inputs)
            f.create_dataset("outputs", data=self.outputs)
            f.attrs["input_keys"] = self.input_keys

    def to_csv(self, path: str):
        """Save to CSV file."""
        header = ",".join(self.input_keys +
                          [f"y{i}" for i in range(self.outputs.shape[1])])
        data = np.hstack([self.inputs, self.outputs])
        np.savetxt(path, data, delimiter=",", header=header, comments="")
