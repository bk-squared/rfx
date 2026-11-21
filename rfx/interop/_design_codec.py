"""Shared validation and entry codecs for design documents."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.interop._errors import UnsupportedDesignFeature
from rfx.interop._validate import check_number, check_text


def _refuse(message: str) -> "UnsupportedDesignFeature":
    return UnsupportedDesignFeature(message)


# ``check_number`` / ``check_text`` / ``check_vector`` come from
# ``rfx/interop/_validate.py`` — the shared leaf the shape and material codecs
# also sit on. They refuse JAX tracers, ``bool`` (an int in Python, so
# ``float(True)`` would silently record 1.0), ``str``/``bytes`` (a quoted
# numeric in a hand-written document), one-shot iterators (consumed by the act
# of exporting, so a second export would emit an empty sequence with no error)
# and non-finite floats. The two coercions below are the ones the leaf does not
# provide; they build on it rather than re-deriving its rules.


def _integer(value: Any, *, what: str) -> int:
    """Require an exact integer.

    Not a wrapper around ``check_number``: cell counts, mode indices and
    ``n_freqs`` must not silently accept 3.5 or come back as a float, and the
    document distinguishes JSON integers from JSON numbers.
    """
    if isinstance(value, bool):
        raise _refuse(f"{what} must be an integer, got a bool ({value!r})")
    if is_tracer(value):
        raise _refuse(f"{what} is a JAX tracer; expected a concrete integer")
    if isinstance(value, (str, bytes)):
        raise _refuse(
            f"{what} is the string {value!r}, not an integer; int() would have "
            f"accepted it silently"
        )
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise _refuse(f"{what} must be an integer, got {value!r}") from exc
    if out != value:
        raise _refuse(f"{what} must be an exact integer, got {value!r}")
    return out


def _integer_vector(value: Any, n: int, *, what: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        raise _refuse(f"{what} must be a sequence of {n} integers, got {value!r}")
    if len(value) != n:
        raise _refuse(
            f"{what} must have exactly {n} components, got {len(value)}: {value!r}"
        )
    return tuple(_integer(v, what=f"{what}[{i}]") for i, v in enumerate(value))


def _require_exact_keys(payload: dict, expected: set[str], *, what: str) -> None:
    present = set(payload)
    unknown = present - expected
    missing = expected - present
    if unknown or missing:
        raise _refuse(
            f"{what} key mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


# ---------------------------------------------------------------------------
# Array codec — full values, dtype and namespace preserved
# ---------------------------------------------------------------------------
#
# Mesh profiles and frequency lists reach the builder as Python lists, numpy
# arrays or jax arrays, and the container matters: ``Simulation.__init__``
# synthesises ``domain`` from ``float(np.sum(profile))``, so a float32 profile
# reloaded as float64 shifts the domain extent in the last ULPs and breaks the
# bit-identity gates the repo relies on.  The dtype and the array namespace are
# therefore part of the record.

_ARRAY_CONTAINERS = ("list", "tuple", "numpy", "jax")


def _array_to_dict(value: Any, *, what: str) -> dict[str, Any]:
    """Record a 1-D numeric array value-by-value with its container identity."""
    if is_tracer(value):
        raise _refuse(
            f"{what} is a JAX tracer (differentiable-mesh / traced path). A "
            f"tracer has no concrete values, so exporting it would record a "
            f"placeholder rather than a mesh. Export the design outside "
            f"jit/grad with concrete profile values."
        )
    if isinstance(value, np.ndarray):
        container = "numpy"
    elif isinstance(value, jax.Array):
        container = "jax"
    elif isinstance(value, list):
        container = "list"
    elif isinstance(value, tuple):
        container = "tuple"
    else:
        raise _refuse(
            f"{what} is a {type(value).__name__}; the design document records "
            f"lists, tuples, numpy arrays and jax arrays only"
        )

    payload: dict[str, Any] = {"container": container}
    if container in ("numpy", "jax"):
        if value.ndim != 1:
            raise _refuse(
                f"{what} must be 1-D, got shape {tuple(value.shape)}"
            )
        payload["dtype"] = str(value.dtype)
    payload["values"] = [
        check_number(v, what=f"{what}[{i}]") for i, v in enumerate(np.asarray(value).tolist())
    ]
    return payload


def _array_from_dict(payload: Any, *, what: str) -> Any:
    if not isinstance(payload, dict):
        raise _refuse(f"{what} must be a mapping, got {type(payload).__name__}")
    container = payload.get("container")
    if container not in _ARRAY_CONTAINERS:
        raise _refuse(
            f"{what}.container must be one of {_ARRAY_CONTAINERS}, got "
            f"{container!r}"
        )
    expected = {"container", "values"} | ({"dtype"} if container in ("numpy", "jax") else set())
    _require_exact_keys(payload, expected, what=what)

    raw = payload["values"]
    if not isinstance(raw, list):
        raise _refuse(f"{what}.values must be a list, got {type(raw).__name__}")
    values = [check_number(v, what=f"{what}.values[{i}]") for i, v in enumerate(raw)]

    if container == "list":
        return values
    if container == "tuple":
        return tuple(values)

    dtype = check_text(payload["dtype"], what=f"{what}.dtype")
    try:
        arr = np.asarray(values, dtype=np.dtype(dtype))
    except TypeError as exc:
        raise _refuse(f"{what}.dtype {dtype!r} is not a numpy dtype") from exc
    if container == "numpy":
        return arr
    out = jnp.asarray(arr)
    if str(out.dtype) != dtype:
        raise _refuse(
            f"{what} was exported as a jax array of dtype {dtype!r} but this "
            f"process rebuilds it as {str(out.dtype)!r} — the JAX x64 setting "
            f"differs from the exporting process. Rebuilding would change the "
            f"numbers, so the import is refused (see JAX_ENABLE_X64)."
        )
    return out


class _F(NamedTuple):
    dump: Callable[[Any, str], Any]
    load: Callable[[Any, str], Any]


def _opt(field: _F) -> _F:
    return _F(
        dump=lambda v, w: None if v is None else field.dump(v, w),
        load=lambda v, w: None if v is None else field.load(v, w),
    )


_NUM = _F(dump=lambda v, w: check_number(v, what=w), load=lambda v, w: check_number(v, what=w))
_INT = _F(dump=lambda v, w: _integer(v, what=w), load=lambda v, w: _integer(v, what=w))
_STR = _F(dump=lambda v, w: check_text(v, what=w), load=lambda v, w: check_text(v, what=w))
_ARRAY = _F(
    dump=lambda v, w: _array_to_dict(v, what=w),
    load=lambda v, w: _array_from_dict(v, what=w),
)


def live_field_names(cls: type) -> tuple[str, ...]:
    """Declared field names of a dataclass or NamedTuple record class.

    Used by the interop contract test to pin the registries above against the
    live builder record classes.
    """
    if dataclasses.is_dataclass(cls):
        return tuple(f.name for f in dataclasses.fields(cls))
    fields = getattr(cls, "_fields", None)
    if fields is not None:
        return tuple(fields)
    raise TypeError(f"{cls.__name__} is neither a dataclass nor a NamedTuple")


def _dump_entry(
    entry: Any, fields: dict[str, _F], cls: type, *, what: str
) -> dict[str, Any]:
    if type(entry) is not cls:
        raise _refuse(
            f"{what} is a {type(entry).__name__}, expected {cls.__name__}; the "
            f"design document does not infer fields of unknown record types"
        )
    live = set(live_field_names(cls))
    unrecorded = live - set(fields)
    if unrecorded:
        raise _refuse(
            f"{cls.__name__} carries fields the design document does not "
            f"record: {sorted(unrecorded)}. Update rfx/interop/_design.py "
            f"rather than exporting a partial entry"
        )
    return {
        name: field.dump(getattr(entry, name), f"{what}.{name}")
        for name, field in fields.items()
    }


def _load_entry(
    payload: Any, fields: dict[str, _F], *, what: str, defaults=None
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise _refuse(f"{what} must be a mapping, got {type(payload).__name__}")
    if defaults:
        payload = {**defaults, **payload}
    _require_exact_keys(payload, set(fields), what=what)
    return {
        name: field.load(payload[name], f"{what}.{name}")
        for name, field in fields.items()
    }
