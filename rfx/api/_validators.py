"""Shared scalar and residual-context validation for the simulation API."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np


def _require_integral_param(name: str, value: object) -> int:
    """Return ``value`` as int after rejecting bools and non-integral values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _require_positive_finite_scalar(name: str, value: object) -> float:
    """Return ``value`` as a finite positive Python float."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite real scalar, got {value!r}")
    arr = np.asarray(value)
    if arr.shape != ():
        raise TypeError(
            f"{name} must be a finite real scalar, got array shape {arr.shape}"
        )
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(
        arr.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must be a finite real scalar, got {value!r}")
    out = float(arr.item())
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


def _positive_divisors(value: int) -> list[int]:
    """Return positive divisors of ``value`` sorted ascending."""
    small: list[int] = []
    large: list[int] = []
    root = math.isqrt(value)
    for divisor in range(1, root + 1):
        if value % divisor == 0:
            small.append(divisor)
            paired = value // divisor
            if paired != divisor:
                large.append(paired)
    return small + large[::-1]


def _preflight_json_safe(value: object, *, path: str = "residual_context") -> object:
    """Return ``value`` as JSON-native data or raise a residual-context error."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains non-finite JSON value")
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _preflight_json_safe(value.to_dict(), path=path)
    if hasattr(value, "to_json") and callable(value.to_json):
        try:
            parsed = json.loads(value.to_json())
        except Exception as exc:
            raise TypeError(f"{path} to_json() did not return JSON data") from exc
        return _preflight_json_safe(parsed, path=path)
    if hasattr(value, "tolist") and callable(value.tolist):
        return _preflight_json_safe(value.tolist(), path=path)
    if isinstance(value, Mapping):
        return {
            str(key): _preflight_json_safe(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _preflight_json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains unsupported value {type(value).__name__}")


def _validate_residual_context(
    context: Mapping[str, object] | None,
    *,
    owned_fields: Mapping[str, object],
) -> dict[str, object]:
    """Merge and validate caller residual context before returning a report."""
    caller_context = (
        {}
        if context is None
        else _preflight_json_safe(context, path="residual_context")
    )
    if not isinstance(caller_context, dict):
        raise TypeError("residual_context must be a mapping")
    merged = dict(caller_context)
    merged.update(
        {
            key: _preflight_json_safe(value, path=f"residual_context.{key}")
            for key, value in owned_fields.items()
        }
    )
    try:
        json.dumps(merged, allow_nan=False)
    except ValueError as exc:
        raise ValueError("residual_context contains non-finite JSON value") from exc
    except TypeError as exc:
        raise TypeError("residual_context contains unsupported JSON value") from exc
    return merged
