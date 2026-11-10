"""Structural comparison helpers for design round-trip checks."""

from __future__ import annotations

from typing import Any


def _equal(left: Any, right: Any) -> bool:
    """Structural equality that treats list and tuple alike.

    Numbers compare with ``==``, so an ``int`` in a hand-written document
    matches the ``float`` the exporter emits for the same value.
    """
    if isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict)):
            return False
        if set(left) != set(right):
            return False
        return all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not (
            isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
        ):
            return False
        if len(left) != len(right):
            return False
        return all(_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    return left == right


def _first_difference(left: Any, right: Any, path: str = "") -> str | None:
    """Path of the first structural difference, or None when equal."""
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            here = f"{path}.{key}" if path else key
            if key not in left:
                return f"{here} (missing from the rebuilt design)"
            if key not in right:
                return f"{here} (missing from the document)"
            found = _first_difference(left[key], right[key], here)
            if found is not None:
                return found
        return None
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return f"{path} (length {len(left)} rebuilt vs {len(right)} in document)"
        for index, (a, b) in enumerate(zip(left, right)):
            found = _first_difference(a, b, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if _equal(left, right):
        return None
    return f"{path} (rebuilt {left!r} vs document {right!r})"
