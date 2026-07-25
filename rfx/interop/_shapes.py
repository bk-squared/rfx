"""Lossless serialisation of rfx geometry primitives.

``rfx.artifacts.build_scene_artifact`` records a shape as its class name plus a
bounding box (``artifacts.py`` pops the ``shape`` field and keeps ``_bbox``).
That is adequate for a review summary but cannot distinguish a ``Cylinder``
from a ``Box`` occupying the same bounding box, so it cannot rebuild its own
input and cannot drive an external solver.

This module records the *constructor parameters* instead.  Every supported
primitive survives a JSON round trip exactly; anything not in the registry is
refused loudly (:class:`~rfx.interop._errors.UnsupportedDesignFeature`) rather
than degraded to a bounding box.

JSON canonical form uses lists for vectors; the Python canonical form restores
the tuples the primitives declare, so a round-tripped frozen dataclass compares
equal to the original.

Vocabulary: the discriminator is ``kind`` with snake_case names, matching the
two document layers that already name shapes — ``rfx/config/_shapes.py``
(``shape: "box"``) and ``rfx/experiments/canonical.py`` (``kind: "box"``) — so
the repo does not grow a third name for the same primitive.  Parameter names
are the *constructor* names, which is what makes the registry pinnable against
each live class signature; per-layer spellings of a box (``bounds``,
``bounds_m``) stay an adapter concern for those layers.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from typing import Any, Callable, NamedTuple

from rfx.core.jax_utils import is_tracer
from rfx.geometry.csg import Box, Cylinder, PolylineWire, Sphere
from rfx.geometry.curved import CurvedPatch
from rfx.geometry.via import Via
from rfx.interop._errors import UnsupportedDesignFeature

__all__ = [
    "SUPPORTED_SHAPE_KINDS",
    "shape_from_dict",
    "shape_kind_of",
    "shape_to_dict",
    "shape_field_names",
]


class _Field(NamedTuple):
    """A pair of coercions: to JSON-canonical form, and back to Python."""

    dump: Callable[[Any], Any]
    load: Callable[[Any], Any]


def _checked_scalar(value: Any, *, what: str) -> float:
    """Coerce to a finite Python float, refusing tracers and NaN/inf.

    A tracer means the parameter is a differentiable design variable with no
    concrete value to record; NaN/inf would be written by ``json.dump`` only as
    non-standard tokens (and ``allow_nan=False`` raises a bare ``ValueError``
    far from the offending field), so both are refused here where the field name
    is still known.
    """
    if is_tracer(value):
        raise UnsupportedDesignFeature(
            f"{what} is a JAX tracer, so it is a differentiable design variable "
            f"with no concrete value to record. Export the design outside the "
            f"traced/jax.grad context, or record the concrete design you want "
            f"to hand to another tool"
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


def _checked_vec(value: Any, n: int, *, what: str) -> tuple[float, ...]:
    if is_tracer(value):
        raise UnsupportedDesignFeature(
            f"{what} is a JAX tracer, so it is a differentiable design variable "
            f"with no concrete value to record. Export the design outside the "
            f"traced/jax.grad context"
        )
    try:
        components = tuple(value)
    except TypeError as exc:
        raise UnsupportedDesignFeature(
            f"{what} must be a sequence of {n} numbers, got {value!r}"
        ) from exc
    if len(components) != n:
        raise UnsupportedDesignFeature(
            f"{what} must have exactly {n} components, got {len(components)}: "
            f"{value!r}"
        )
    return tuple(
        _checked_scalar(v, what=f"{what}[{i}]")
        for i, v in enumerate(components)
    )


def _vec(n: int, *, what: str) -> _Field:
    return _Field(
        dump=lambda v: list(_checked_vec(v, n, what=what)),
        load=lambda v: _checked_vec(v, n, what=what),
    )


def _vec_seq(n: int, *, what: str, container: Callable[[Any], Any]) -> _Field:
    return _Field(
        dump=lambda v: [list(_checked_vec(p, n, what=what)) for p in v],
        load=lambda v: container(_checked_vec(p, n, what=what) for p in v),
    )


def _flt(*, what: str) -> _Field:
    return _Field(
        dump=lambda v: _checked_scalar(v, what=what),
        load=lambda v: _checked_scalar(v, what=what),
    )


_STR = _Field(dump=lambda v: str(v), load=lambda v: str(v))


class _ShapeCodec(NamedTuple):
    cls: type
    fields: dict[str, _Field]


_CODECS: dict[str, _ShapeCodec] = {
    "box": _ShapeCodec(Box, {
        "corner_lo": _vec(3, what="box.corner_lo"),
        "corner_hi": _vec(3, what="box.corner_hi"),
    }),
    "cylinder": _ShapeCodec(Cylinder, {
        "center": _vec(3, what="cylinder.center"),
        "radius": _flt(what="cylinder.radius"),
        "height": _flt(what="cylinder.height"),
        "axis": _STR,
    }),
    "sphere": _ShapeCodec(Sphere, {
        "center": _vec(3, what="sphere.center"),
        "radius": _flt(what="sphere.radius"),
    }),
    "polyline_wire": _ShapeCodec(PolylineWire, {
        "points": _vec_seq(3, what="polyline_wire.points", container=tuple),
        "radius": _flt(what="polyline_wire.radius"),
    }),
    # ``Via`` carries its own ``material`` in addition to the ``material=``
    # handed to ``Simulation.add()``.  Both are recorded; resolving the two is
    # the caller's decision, not something this codec may silently pick.
    "via": _ShapeCodec(Via, {
        "center": _vec(2, what="via.center"),
        "drill_radius": _flt(what="via.drill_radius"),
        "pad_radius": _flt(what="via.pad_radius"),
        "layers": _vec_seq(2, what="via.layers", container=list),
        "material": _STR,
    }),
    "curved_patch": _ShapeCodec(CurvedPatch, {
        "center": _vec(3, what="curved_patch.center"),
        "length": _flt(what="curved_patch.length"),
        "width": _flt(what="curved_patch.width"),
        "radius": _flt(what="curved_patch.radius"),
        "axis": _STR,
    }),
}

SUPPORTED_SHAPE_KINDS: tuple[str, ...] = tuple(sorted(_CODECS))

_KIND_BY_CLASS: dict[type, str] = {
    codec.cls: kind for kind, codec in _CODECS.items()
}


def shape_field_names(kind: str) -> tuple[str, ...]:
    """Return the recorded parameter names for a shape ``kind``."""
    codec = _codec_for_kind(kind)
    return tuple(codec.fields)


def shape_kind_of(shape: Any) -> str:
    """Return the IR ``kind`` for a shape instance.

    Raises
    ------
    UnsupportedDesignFeature
        If the shape's exact class is not registered.  Subclasses are refused
        because they may carry state the registry cannot see.
    """
    try:
        return _KIND_BY_CLASS[type(shape)]
    except KeyError:
        raise UnsupportedDesignFeature(
            f"shape class {type(shape).__name__!r} is not supported by the "
            f"design-interop layer; supported kinds are "
            f"{', '.join(SUPPORTED_SHAPE_KINDS)}"
        ) from None


def constructor_parameter_names(cls: type) -> tuple[str, ...]:
    """Return the constructor parameters of a primitive class.

    Used by the contract test that pins the registry against the live classes,
    so that adding a parameter upstream fails a test instead of silently
    vanishing from exported designs.
    """
    if dataclasses.is_dataclass(cls):
        return tuple(f.name for f in dataclasses.fields(cls))
    params = inspect.signature(cls.__init__).parameters
    return tuple(name for name in params if name != "self")


def _codec_for_kind(kind: str) -> _ShapeCodec:
    try:
        return _CODECS[kind]
    except (KeyError, TypeError):
        raise UnsupportedDesignFeature(
            f"shape kind {kind!r} is not supported by the design-interop "
            f"layer; supported kinds are {', '.join(SUPPORTED_SHAPE_KINDS)}"
        ) from None


def _instance_field_names(shape: Any) -> tuple[str, ...]:
    if dataclasses.is_dataclass(shape):
        return tuple(f.name for f in dataclasses.fields(shape))
    return tuple(vars(shape))


def shape_to_dict(shape: Any) -> dict[str, Any]:
    """Serialise a geometry primitive to its JSON-canonical parameters.

    Raises
    ------
    UnsupportedDesignFeature
        If the shape's class is not registered, or if the live class carries
        state the registry does not know about (which would otherwise be
        dropped silently).
    """
    kind = shape_kind_of(shape)
    codec = _codec_for_kind(kind)

    recorded = set(codec.fields)
    present = set(_instance_field_names(shape))
    missing = present - recorded
    if missing:
        raise UnsupportedDesignFeature(
            f"{kind} carries state the design-interop registry does not "
            f"record: {sorted(missing)}. Update rfx/interop/_shapes.py rather "
            f"than exporting a partial shape"
        )

    params: dict[str, Any] = {}
    for name, field in codec.fields.items():
        try:
            value = getattr(shape, name)
        except AttributeError as exc:
            raise UnsupportedDesignFeature(
                f"{kind} is missing the recorded parameter {name!r}"
            ) from exc
        params[name] = field.dump(value)
    return {"kind": kind, "params": params}


def shape_from_dict(payload: dict[str, Any]) -> Any:
    """Rebuild a geometry primitive from :func:`shape_to_dict` output."""
    if not isinstance(payload, dict):
        raise UnsupportedDesignFeature(
            f"shape payload must be a mapping, got {type(payload).__name__}"
        )
    try:
        kind = payload["kind"]
    except KeyError:
        raise UnsupportedDesignFeature(
            "shape payload is missing the 'kind' key"
        ) from None

    codec = _codec_for_kind(kind)
    params = payload.get("params", {})
    if not isinstance(params, dict):
        raise UnsupportedDesignFeature(
            f"{kind} params must be a mapping, got {type(params).__name__}"
        )

    unknown = set(params) - set(codec.fields)
    if unknown:
        raise UnsupportedDesignFeature(
            f"{kind} payload carries unknown parameters {sorted(unknown)}"
        )
    absent = set(codec.fields) - set(params)
    if absent:
        raise UnsupportedDesignFeature(
            f"{kind} payload is missing parameters {sorted(absent)}"
        )

    kwargs = {
        name: field.load(params[name])
        for name, field in codec.fields.items()
    }
    return codec.cls(**kwargs)
