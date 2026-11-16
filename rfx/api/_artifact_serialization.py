"""Serialization of nested preflight report artifacts."""

from __future__ import annotations

import json
from typing import Mapping


def _artifact_to_dict(value: object) -> object:
    """Serialize nested report artifacts without importing non-leaf rfx modules."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _artifact_to_dict(value.to_dict())
    if hasattr(value, "to_json") and callable(value.to_json):
        return _artifact_to_dict(json.loads(value.to_json()))
    if hasattr(value, "tolist") and callable(value.tolist):
        return _artifact_to_dict(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _artifact_to_dict(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_artifact_to_dict(item) for item in value]
    if isinstance(value, list):
        return [_artifact_to_dict(item) for item in value]
    raise TypeError(
        f"preflight nested artifact contains unsupported value {type(value).__name__}"
    )
