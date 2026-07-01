"""Private helpers for scoped JAX compiled-memory certificates.

This module intentionally handles only one evidence class: a bounded certificate
for one caller-supplied compiled JAX executable via ``Compiled.memory_analysis``.
It does not compile callables, run FDTD, profile runtime peaks, or prove source to
executable correspondence from digests.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral
from typing import Mapping, NamedTuple

import jax
import numpy as np


_REQUIRED_MEMORY_FIELDS = (
    "temp_size_in_bytes",
    "argument_size_in_bytes",
    "output_size_in_bytes",
    "alias_size_in_bytes",
)


class CompiledMemoryAnalysis(NamedTuple):
    status: str
    reason: str
    required_bytes: int | None
    fields: dict[str, int]
    analysis_type: str | None
    present_known_attrs: tuple[str, ...]


class CanonicalScope(NamedTuple):
    status: str
    reason: str
    exact_scope: dict[str, object] | None
    source_preflight: object | None
    scope_digest: str | None
    config_digest: str | None
    environment_digest: str | None


_MISSING_ATTR = object()


def _read_attr(value: object, name: str, *, path: str) -> object:
    try:
        return getattr(value, name)
    except AttributeError:
        return _MISSING_ATTR
    except Exception as exc:
        raise TypeError(
            f"{path}.{name} attribute read raised {type(exc).__name__}: {exc}"
        ) from exc


def _json_safe(value: object, *, path: str = "value") -> object:
    """Return JSON-native data or raise for unsupported/non-finite values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains non-finite JSON value")
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item(), path=path)
    if isinstance(value, np.dtype):
        return str(value)
    shape = _read_attr(value, "shape", path=path)
    dtype = _read_attr(value, "dtype", path=path)
    if shape is not _MISSING_ATTR and dtype is not _MISSING_ATTR:
        weak_type = _read_attr(value, "weak_type", path=path)
        try:
            shape_summary = [int(dim) for dim in tuple(shape)]
        except Exception as exc:
            raise TypeError(
                f"{path}.shape could not be summarized as integer dimensions"
            ) from exc
        summary: dict[str, object] = {
            "shape": shape_summary,
            "dtype": str(dtype),
        }
        if weak_type is not _MISSING_ATTR:
            summary["weak_type"] = bool(weak_type)
        return summary
    to_dict = _read_attr(value, "to_dict", path=path)
    if to_dict is not _MISSING_ATTR and callable(to_dict):
        try:
            return _json_safe(to_dict(), path=path)
        except (TypeError, ValueError):
            raise
        except Exception as exc:
            raise TypeError(f"{path}.to_dict() raised {type(exc).__name__}: {exc}") from exc
    to_json = _read_attr(value, "to_json", path=path)
    if to_json is not _MISSING_ATTR and callable(to_json):
        try:
            parsed = json.loads(to_json())
        except Exception as exc:
            raise TypeError(f"{path}.to_json() did not return JSON data") from exc
        return _json_safe(parsed, path=path)
    tolist = _read_attr(value, "tolist", path=path)
    if tolist is not _MISSING_ATTR and callable(tolist):
        try:
            return _json_safe(tolist(), path=path)
        except (TypeError, ValueError):
            raise
        except Exception as exc:
            raise TypeError(f"{path}.tolist() raised {type(exc).__name__}: {exc}") from exc
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains non-string mapping key {key!r}")
            out[key] = _json_safe(item, path=f"{path}.{key}")
        return out
    if isinstance(value, tuple):
        return [_json_safe(item, path=f"{path}[{idx}]") for idx, item in enumerate(value)]
    if isinstance(value, list):
        return [_json_safe(item, path=f"{path}[{idx}]") for idx, item in enumerate(value)]
    raise TypeError(f"{path} contains unsupported value {type(value).__name__}")


def _canonical_json(value: object) -> str:
    safe = _json_safe(value)
    return json.dumps(safe, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _stable_digest(value: object) -> str:
    """Return SHA-256 over canonical JSON. This is an audit identity only."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _required_metadata_text(name: str, raw: object) -> tuple[str | None, str | None]:
    if raw is None:
        return None, f"{name} is missing"
    text = str(raw).strip()
    if not text or text.lower() in {"unknown", "none"}:
        return None, f"{name} is missing or unknown"
    return text, None


def _environment_summary() -> dict[str, object]:
    """Return fail-closed JSON-safe JAX backend/device metadata."""
    try:
        jax_version_raw = getattr(jax, "__version__")
    except Exception as exc:
        return {
            "status": "incomplete",
            "reason": f"jax.__version__ read failed: {type(exc).__name__}: {exc}",
            "jax_version": None,
            "backend_platform": None,
            "device_platform": None,
            "device_kind": None,
            "device_count": None,
        }
    jax_version, jax_version_error = _required_metadata_text(
        "jax.__version__",
        jax_version_raw,
    )
    if jax_version_error is not None:
        return {
            "status": "incomplete",
            "reason": jax_version_error,
            "jax_version": None,
            "backend_platform": None,
            "device_platform": None,
            "device_kind": None,
            "device_count": None,
        }

    try:
        backend_platform_raw = jax.default_backend()
    except Exception as exc:
        return {
            "status": "incomplete",
            "reason": f"jax.default_backend() failed: {type(exc).__name__}: {exc}",
            "jax_version": jax_version,
            "backend_platform": None,
            "device_platform": None,
            "device_kind": None,
            "device_count": None,
        }
    backend_platform, backend_error = _required_metadata_text(
        "jax.default_backend()",
        backend_platform_raw,
    )
    if backend_error is not None:
        return {
            "status": "incomplete",
            "reason": backend_error,
            "jax_version": jax_version,
            "backend_platform": None,
            "device_platform": None,
            "device_kind": None,
            "device_count": None,
        }

    try:
        devices = tuple(jax.local_devices())
    except Exception as exc:
        return {
            "status": "incomplete",
            "reason": f"jax.local_devices() failed: {type(exc).__name__}: {exc}",
            "jax_version": jax_version,
            "backend_platform": backend_platform,
            "device_platform": None,
            "device_kind": None,
            "device_count": None,
        }
    if not devices:
        return {
            "status": "incomplete",
            "reason": "jax.local_devices() returned no devices",
            "jax_version": jax_version,
            "backend_platform": backend_platform,
            "device_platform": None,
            "device_kind": None,
            "device_count": 0,
        }

    first = devices[0]
    try:
        device_platform_raw = getattr(first, "platform")
        device_kind_raw = getattr(first, "device_kind")
    except Exception as exc:
        return {
            "status": "incomplete",
            "reason": f"JAX device metadata read failed: {type(exc).__name__}: {exc}",
            "jax_version": jax_version,
            "backend_platform": backend_platform,
            "device_platform": None,
            "device_kind": None,
            "device_count": len(devices),
        }
    device_platform, platform_error = _required_metadata_text(
        "JAX device platform",
        device_platform_raw,
    )
    device_kind, kind_error = _required_metadata_text(
        "JAX device_kind",
        device_kind_raw,
    )
    if platform_error is not None or kind_error is not None:
        return {
            "status": "incomplete",
            "reason": platform_error or kind_error,
            "jax_version": jax_version,
            "backend_platform": backend_platform,
            "device_platform": device_platform,
            "device_kind": device_kind,
            "device_count": len(devices),
        }

    return {
        "status": "complete",
        "reason": "JAX backend/device metadata collected",
        "jax_version": jax_version,
        "backend_platform": backend_platform,
        "device_platform": device_platform,
        "device_kind": device_kind,
        "device_count": int(len(devices)),
    }


def _normalize_byte_field(value: object, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be a non-negative integer byte count")
    out = int(value)
    if out < 0:
        raise ValueError(f"{field} must be non-negative")
    return out


def _normalize_compiled_memory_analysis(compiled: object) -> CompiledMemoryAnalysis:
    """Normalize ``compiled.memory_analysis()`` into required byte fields."""
    try:
        memory_analysis = getattr(compiled, "memory_analysis")
    except AttributeError:
        memory_analysis = None
    except Exception as exc:
        return CompiledMemoryAnalysis(
            status="analysis_unavailable",
            reason=f"reading memory_analysis attribute raised {type(exc).__name__}: {exc}",
            required_bytes=None,
            fields={},
            analysis_type=None,
            present_known_attrs=(),
        )
    if memory_analysis is None or not callable(memory_analysis):
        return CompiledMemoryAnalysis(
            status="analysis_unavailable",
            reason="compiled object has no callable memory_analysis()",
            required_bytes=None,
            fields={},
            analysis_type=None,
            present_known_attrs=(),
        )
    try:
        analysis = memory_analysis()
    except Exception as exc:
        return CompiledMemoryAnalysis(
            status="analysis_unavailable",
            reason=f"memory_analysis() raised {type(exc).__name__}: {exc}",
            required_bytes=None,
            fields={},
            analysis_type=None,
            present_known_attrs=(),
        )
    if analysis is None:
        return CompiledMemoryAnalysis(
            status="analysis_unavailable",
            reason="memory_analysis() returned None",
            required_bytes=None,
            fields={},
            analysis_type=None,
            present_known_attrs=(),
        )

    present: list[str] = []
    missing: list[str] = []
    raw_fields: dict[str, object] = {}
    for field in _REQUIRED_MEMORY_FIELDS:
        try:
            raw_fields[field] = getattr(analysis, field)
        except AttributeError:
            missing.append(field)
        except Exception as exc:
            return CompiledMemoryAnalysis(
                status="analysis_incomplete",
                reason=f"reading {field} raised {type(exc).__name__}: {exc}",
                required_bytes=None,
                fields={},
                analysis_type=type(analysis).__name__,
                present_known_attrs=tuple(present),
            )
        else:
            present.append(field)
    present_tuple = tuple(present)
    if missing:
        return CompiledMemoryAnalysis(
            status="analysis_incomplete",
            reason="memory_analysis() missing required byte field(s): " + ", ".join(missing),
            required_bytes=None,
            fields={},
            analysis_type=type(analysis).__name__,
            present_known_attrs=present_tuple,
        )

    fields: dict[str, int] = {}
    try:
        for field in _REQUIRED_MEMORY_FIELDS:
            fields[field] = _normalize_byte_field(raw_fields[field], field=field)
        required_bytes = (
            fields["temp_size_in_bytes"]
            + fields["argument_size_in_bytes"]
            + fields["output_size_in_bytes"]
            - fields["alias_size_in_bytes"]
        )
        if required_bytes < 0:
            raise ValueError("computed required bytes must be non-negative")
    except (TypeError, ValueError) as exc:
        return CompiledMemoryAnalysis(
            status="analysis_incomplete",
            reason=str(exc),
            required_bytes=None,
            fields={},
            analysis_type=type(analysis).__name__,
            present_known_attrs=present_tuple,
        )

    return CompiledMemoryAnalysis(
        status="complete",
        reason="memory_analysis() returned complete required byte fields",
        required_bytes=required_bytes,
        fields=fields,
        analysis_type=type(analysis).__name__,
        present_known_attrs=present_tuple,
    )


def _checkpoint_mode(
    *,
    checkpoint_every: int | None,
    checkpoint_segments: int | None,
) -> str:
    if checkpoint_every is not None:
        return "checkpoint_every"
    if checkpoint_segments is not None:
        return "checkpoint_segments"
    return "none"


def _compiled_introspection_scope(compiled: object) -> tuple[object | None, str | None]:
    for attr in ("rfx_memory_scope", "_rfx_memory_scope", "scope_metadata"):
        try:
            value = getattr(compiled, attr)
        except AttributeError:
            continue
        except Exception as exc:
            return None, (
                f"compiled object introspection attribute {attr!r} raised "
                f"{type(exc).__name__}: {exc}"
            )
        if value is not None:
            return value, None
    return None, None


def _preflight_snapshot(preflight: object | None) -> object | None:
    if preflight is None:
        return None
    return _json_safe(preflight, path="preflight")


def _float_mismatch(actual: object, expected: float, *, atol: float = 1e-12) -> bool:
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        return True
    return abs(float(actual) - float(expected)) > atol


def _preflight_mismatches(
    preflight_snapshot: object | None,
    *,
    n_steps: int,
    available_memory_gb: float,
    target_fraction: float,
    target_memory_gb: float,
    checkpoint_mode: str,
    checkpoint_every: int | None,
    checkpoint_segments: int | None,
) -> list[str]:
    if not isinstance(preflight_snapshot, Mapping):
        return []
    mismatches: list[str] = []
    if preflight_snapshot.get("n_steps") != n_steps:
        mismatches.append("preflight n_steps differs from certificate scope")
    if _float_mismatch(preflight_snapshot.get("available_memory_gb"), available_memory_gb):
        mismatches.append("preflight available_memory_gb differs from certificate scope")
    if _float_mismatch(preflight_snapshot.get("target_fraction"), target_fraction):
        mismatches.append("preflight target_fraction differs from certificate scope")
    if _float_mismatch(preflight_snapshot.get("target_memory_gb"), target_memory_gb):
        mismatches.append("preflight target_memory_gb differs from certificate scope")

    preflight_mode = preflight_snapshot.get("supported_checkpoint_mode")
    normalized_preflight_mode = "none" if preflight_mode is None else preflight_mode
    if normalized_preflight_mode != checkpoint_mode:
        mismatches.append("preflight checkpoint mode differs from certificate scope")
    if preflight_snapshot.get("checkpoint_every") != checkpoint_every:
        mismatches.append("preflight checkpoint_every differs from certificate scope")
    if preflight_snapshot.get("checkpoint_segments") != checkpoint_segments:
        mismatches.append("preflight checkpoint_segments differs from certificate scope")
    return mismatches


def _introspection_mismatches(exact_scope: Mapping[str, object], introspection: object | None) -> list[str]:
    if introspection is None:
        return []
    try:
        safe = _json_safe(introspection, path="compiled_introspection_scope")
    except (TypeError, ValueError) as exc:
        return [f"compiled object introspection scope is not JSON-safe: {exc}"]
    if not isinstance(safe, Mapping):
        return ["compiled object introspection scope is not a mapping"]
    mismatches: list[str] = []
    for key, actual in safe.items():
        key_s = str(key)
        if key_s in exact_scope and exact_scope[key_s] != actual:
            mismatches.append(f"compiled object introspection field {key_s!r} differs")
    return mismatches


def _canonicalize_exact_scope(
    *,
    compiled: object,
    n_steps: int,
    n_warmup: int,
    checkpoint_every: int | None,
    checkpoint_segments: int | None,
    available_memory_gb: float,
    target_fraction: float,
    target_memory_gb: float,
    precision: object,
    input_signature: object,
    static_signature: object,
    compiled_object_id: object,
    runner_or_objective: object,
    memory_analysis_fields: tuple[str, ...] | None,
    preflight: object | None,
    scope_context: Mapping[str, object] | None,
) -> CanonicalScope:
    """Build canonical exact scope and fail closed on incomplete/mismatched data."""
    missing: list[str] = []
    if precision is None or (isinstance(precision, str) and not precision.strip()):
        missing.append("precision")
    if input_signature is None:
        missing.append("input_signature")
    if static_signature is None:
        missing.append("static_signature")
    if compiled_object_id is None or (
        isinstance(compiled_object_id, str) and not compiled_object_id.strip()
    ):
        missing.append("compiled_object_id")
    if runner_or_objective is None or (
        isinstance(runner_or_objective, str) and not runner_or_objective.strip()
    ):
        missing.append("runner_or_objective")
    if missing:
        return CanonicalScope(
            status="scope_incomplete",
            reason="missing required exact-scope metadata: " + ", ".join(missing),
            exact_scope=None,
            source_preflight=None,
            scope_digest=None,
            config_digest=None,
            environment_digest=None,
        )

    env = _environment_summary()
    if env["status"] != "complete":
        return CanonicalScope(
            status="scope_incomplete",
            reason=str(env["reason"]),
            exact_scope=None,
            source_preflight=None,
            scope_digest=None,
            config_digest=None,
            environment_digest=None,
        )
    env_scope = {
        "jax_version": env["jax_version"],
        "backend_platform": env["backend_platform"],
        "device_platform": env["device_platform"],
        "device_kind": env["device_kind"],
        "device_count": env["device_count"],
    }
    checkpoint_mode = _checkpoint_mode(
        checkpoint_every=checkpoint_every,
        checkpoint_segments=checkpoint_segments,
    )
    try:
        input_signature_safe = _json_safe(input_signature, path="input_signature")
        static_signature_safe = _json_safe(static_signature, path="static_signature")
        scope_context_safe = (
            None if scope_context is None else _json_safe(scope_context, path="scope_context")
        )
        preflight_safe = _preflight_snapshot(preflight)
        if preflight is not None and not isinstance(preflight_safe, Mapping):
            return CanonicalScope(
                status="scope_incomplete",
                reason="preflight snapshot must be a JSON object mapping",
                exact_scope=None,
                source_preflight=preflight_safe,
                scope_digest=None,
                config_digest=None,
                environment_digest=None,
            )
        exact_scope: dict[str, object] = {
            "schema_version": 1,
            "jax_version": env_scope["jax_version"],
            "backend_platform": env_scope["backend_platform"],
            "device_platform": env_scope["device_platform"],
            "device_kind": env_scope["device_kind"],
            "device_count": env_scope["device_count"],
            "compiled_object_type": f"{type(compiled).__module__}.{type(compiled).__qualname__}",
            "compiled_object_id": _json_safe(compiled_object_id, path="compiled_object_id"),
            "n_steps": int(n_steps),
            "n_warmup": int(n_warmup),
            "checkpoint_mode": checkpoint_mode,
            "checkpoint_every": checkpoint_every,
            "checkpoint_segments": checkpoint_segments,
            "available_memory_gb": float(available_memory_gb),
            "target_fraction": float(target_fraction),
            "target_memory_gb": float(target_memory_gb),
            "precision": _json_safe(precision, path="precision"),
            "input_signature": input_signature_safe,
            "static_signature": static_signature_safe,
            "runner_or_objective": _json_safe(runner_or_objective, path="runner_or_objective"),
            "memory_analysis_fields": (
                None if memory_analysis_fields is None else list(memory_analysis_fields)
            ),
        }
        if preflight_safe is not None:
            exact_scope["source_preflight_digest"] = _stable_digest(preflight_safe)
        if scope_context_safe is not None:
            exact_scope["user_context"] = scope_context_safe
    except (TypeError, ValueError) as exc:
        return CanonicalScope(
            status="scope_incomplete",
            reason=str(exc),
            exact_scope=None,
            source_preflight=None,
            scope_digest=None,
            config_digest=None,
            environment_digest=None,
        )

    mismatches = _preflight_mismatches(
        preflight_safe,
        n_steps=n_steps,
        available_memory_gb=available_memory_gb,
        target_fraction=target_fraction,
        target_memory_gb=target_memory_gb,
        checkpoint_mode=checkpoint_mode,
        checkpoint_every=checkpoint_every,
        checkpoint_segments=checkpoint_segments,
    )
    introspection_scope, introspection_error = _compiled_introspection_scope(compiled)
    if introspection_error is not None:
        mismatches.append(introspection_error)
    else:
        mismatches.extend(_introspection_mismatches(exact_scope, introspection_scope))
    if mismatches:
        return CanonicalScope(
            status="scope_mismatch",
            reason="; ".join(mismatches),
            exact_scope=exact_scope,
            source_preflight=preflight_safe,
            scope_digest=_stable_digest(exact_scope),
            config_digest=_stable_digest(
                {
                    "n_steps": n_steps,
                    "n_warmup": n_warmup,
                    "checkpoint_mode": checkpoint_mode,
                    "checkpoint_every": checkpoint_every,
                    "checkpoint_segments": checkpoint_segments,
                    "available_memory_gb": float(available_memory_gb),
                    "target_fraction": float(target_fraction),
                    "target_memory_gb": float(target_memory_gb),
                    "precision": exact_scope["precision"],
                    "input_signature": input_signature_safe,
                    "static_signature": static_signature_safe,
                    "runner_or_objective": exact_scope["runner_or_objective"],
                }
            ),
            environment_digest=_stable_digest(env_scope),
        )

    return CanonicalScope(
        status="complete",
        reason="exact scope metadata is complete and JSON-safe",
        exact_scope=exact_scope,
        source_preflight=preflight_safe,
        scope_digest=_stable_digest(exact_scope),
        config_digest=_stable_digest(
            {
                "n_steps": n_steps,
                "n_warmup": n_warmup,
                "checkpoint_mode": checkpoint_mode,
                "checkpoint_every": checkpoint_every,
                "checkpoint_segments": checkpoint_segments,
                "available_memory_gb": float(available_memory_gb),
                "target_fraction": float(target_fraction),
                "target_memory_gb": float(target_memory_gb),
                "precision": exact_scope["precision"],
                "input_signature": input_signature_safe,
                "static_signature": static_signature_safe,
                "runner_or_objective": exact_scope["runner_or_objective"],
            }
        ),
        environment_digest=_stable_digest(env_scope),
    )


__all__ = [
    "CanonicalScope",
    "CompiledMemoryAnalysis",
    "_canonicalize_exact_scope",
    "_environment_summary",
    "_normalize_compiled_memory_analysis",
    "_stable_digest",
]
