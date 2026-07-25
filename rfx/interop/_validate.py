"""Value coercions shared by the design-interop codecs.

A leaf module on purpose: ``_materials`` needed these and was importing them
from ``_shapes``, which made materials depend on geometry (and dragged
``rfx.geometry.*`` plus ``jax`` into any import of the material codec). Nothing
here imports another interop module except the error type.

Every coercion refuses rather than repairs, and names the field it is refusing,
because the alternative failure modes are all silent or late:

- a **JAX tracer** means the value is a differentiable design variable with no
  concrete number to record;
- **NaN/inf** would surface only at ``json.dump`` time as a bare
  ``ValueError: Out of range float values are not JSON compliant`` that names no
  field;
- a **``bool``** is an ``int`` in Python, so ``float(True)`` silently yields
  ``1.0`` — a JSON ``true`` would become a length;
- a **``str``** of digits is iterable and float-able per character, so
  ``"123"`` would silently become the vector ``(1.0, 2.0, 3.0)``;
- a **one-shot iterator** is consumed by the act of exporting it, so a second
  export of the same object emits an empty sequence with no error.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from rfx.interop._errors import UnsupportedDesignFeature

__all__ = [
    "check_number",
    "check_sequence",
    "check_text",
    "check_vector",
]


def _is_tracer(value: Any) -> bool:
    # Imported lazily: this leaf must stay importable without paying for jax.
    from rfx.core.jax_utils import is_tracer

    return bool(is_tracer(value))


def check_number(value: Any, *, what: str) -> float:
    """Coerce to a finite, non-boolean Python float."""
    if _is_tracer(value):
        raise UnsupportedDesignFeature(
            f"{what} is a JAX tracer, so it is a differentiable design variable "
            f"with no concrete value to record. Export the design outside the "
            f"traced/jax.grad context, or record the concrete design you want "
            f"to hand to another tool"
        )
    if isinstance(value, bool):
        raise UnsupportedDesignFeature(
            f"{what} is the boolean {value!r}; a boolean is an int in Python, "
            f"so accepting it would silently record {float(value)!r}"
        )
    if isinstance(value, (str, bytes)):
        raise UnsupportedDesignFeature(
            f"{what} is the string {value!r}, not a number. A design document "
            f"with quoted numerics is accepted silently by float() and would "
            f"record a value nobody wrote"
        )
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise UnsupportedDesignFeature(
            f"{what} must be a number, got {value!r}"
        ) from exc
    if not math.isfinite(out):
        raise UnsupportedDesignFeature(
            f"{what} is {out}, which is not a finite number; a design "
            f"description with NaN/inf geometry does not describe a structure"
        )
    return out


def check_text(value: Any, *, what: str) -> str:
    """Require an actual string.

    ``str(value)`` cannot fail, so an unguarded string field will happily record
    ``None``, a whole ``MaterialSpec`` repr, or a multi-line tracer repr as if it
    were a name.
    """
    if not isinstance(value, str):
        raise UnsupportedDesignFeature(
            f"{what} must be a string, got {type(value).__name__} "
            f"({value!r:.80}); str() would have recorded its repr as a name"
        )
    return value


def check_sequence(value: Any, *, what: str, min_length: int = 0) -> tuple:
    """Materialise a sized, non-string sequence.

    Requires ``__len__`` so a generator or other one-shot iterator is refused
    instead of being consumed: consuming it makes export non-idempotent, and a
    second export emits an empty sequence with no error at all.
    """
    if isinstance(value, (str, bytes)):
        raise UnsupportedDesignFeature(
            f"{what} must be a sequence, got the string {value!r:.60}"
        )
    if not isinstance(value, Iterable):
        raise UnsupportedDesignFeature(
            f"{what} must be a sequence, got {type(value).__name__} "
            f"({value!r:.60})"
        )
    if not hasattr(value, "__len__"):
        # Iterable but unsized: a generator or other one-shot iterator.
        raise UnsupportedDesignFeature(
            f"{what} is a {type(value).__name__}, which has no length. A "
            f"one-shot iterator is consumed by exporting it, so a second export "
            f"of the same object would silently emit an empty sequence. Store a "
            f"tuple or list instead"
        )
    items = tuple(value)
    if len(items) < min_length:
        raise UnsupportedDesignFeature(
            f"{what} needs at least {min_length} element(s), got {len(items)}; "
            f"an empty sequence does not describe a structure"
        )
    return items


def check_vector(value: Any, n: int, *, what: str) -> tuple[float, ...]:
    """Coerce to exactly ``n`` finite floats."""
    if _is_tracer(value):
        raise UnsupportedDesignFeature(
            f"{what} is a JAX tracer, so it is a differentiable design variable "
            f"with no concrete value to record. Export the design outside the "
            f"traced/jax.grad context"
        )
    components = check_sequence(value, what=what)
    if len(components) != n:
        raise UnsupportedDesignFeature(
            f"{what} must have exactly {n} components, got "
            f"{len(components)}: {value!r:.80}"
        )
    return tuple(
        check_number(v, what=f"{what}[{i}]")
        for i, v in enumerate(components)
    )
