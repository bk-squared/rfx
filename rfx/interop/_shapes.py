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
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Callable, NamedTuple

from rfx.geometry.csg import Box, Cylinder, PolylineWire, Sphere
from rfx.geometry.curved import CurvedPatch
from rfx.geometry.via import Via
from rfx.interop._errors import UnsupportedDesignFeature

__all__ = [
    "SUPPORTED_SHAPE_TYPES",
    "shape_from_dict",
    "shape_to_dict",
    "shape_field_names",
]


class _Field(NamedTuple):
    """A pair of coercions: to JSON-canonical form, and back to Python."""

    dump: Callable[[Any], Any]
    load: Callable[[Any], Any]


def _checked_vec(value: Any, n: int, *, what: str) -> tuple[float, ...]:
    try:
        seq = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise UnsupportedDesignFeature(
            f"{what} must be a sequence of {n} numbers, got {value!r}"
        ) from exc
    if len(seq) != n:
        raise UnsupportedDesignFeature(
            f"{what} must have exactly {n} components, got {len(seq)}: {value!r}"
        )
    return seq


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


_FLOAT = _Field(dump=lambda v: float(v), load=lambda v: float(v))
_STR = _Field(dump=lambda v: str(v), load=lambda v: str(v))


class _ShapeCodec(NamedTuple):
    cls: type
    fields: dict[str, _Field]


_CODECS: dict[str, _ShapeCodec] = {
    "Box": _ShapeCodec(Box, {
        "corner_lo": _vec(3, what="Box.corner_lo"),
        "corner_hi": _vec(3, what="Box.corner_hi"),
    }),
    "Cylinder": _ShapeCodec(Cylinder, {
        "center": _vec(3, what="Cylinder.center"),
        "radius": _FLOAT,
        "height": _FLOAT,
        "axis": _STR,
    }),
    "Sphere": _ShapeCodec(Sphere, {
        "center": _vec(3, what="Sphere.center"),
        "radius": _FLOAT,
    }),
    "PolylineWire": _ShapeCodec(PolylineWire, {
        "points": _vec_seq(3, what="PolylineWire.points", container=tuple),
        "radius": _FLOAT,
    }),
    # ``Via`` carries its own ``material`` in addition to the ``material=``
    # handed to ``Simulation.add()``.  Both are recorded; resolving the two is
    # the caller's decision, not something this codec may silently pick.
    "Via": _ShapeCodec(Via, {
        "center": _vec(2, what="Via.center"),
        "drill_radius": _FLOAT,
        "pad_radius": _FLOAT,
        "layers": _vec_seq(2, what="Via.layers", container=list),
        "material": _STR,
    }),
    "CurvedPatch": _ShapeCodec(CurvedPatch, {
        "center": _vec(3, what="CurvedPatch.center"),
        "length": _FLOAT,
        "width": _FLOAT,
        "radius": _FLOAT,
        "axis": _STR,
    }),
}

SUPPORTED_SHAPE_TYPES: tuple[str, ...] = tuple(sorted(_CODECS))


def shape_field_names(shape_type: str) -> tuple[str, ...]:
    """Return the recorded parameter names for ``shape_type``."""
    codec = _codec_for_type(shape_type)
    return tuple(codec.fields)


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


def _codec_for_type(shape_type: str) -> _ShapeCodec:
    try:
        return _CODECS[shape_type]
    except KeyError:
        raise UnsupportedDesignFeature(
            f"shape type {shape_type!r} is not supported by the design-interop "
            f"layer; supported types are {', '.join(SUPPORTED_SHAPE_TYPES)}"
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
    shape_type = type(shape).__name__
    codec = _codec_for_type(shape_type)
    if type(shape) is not codec.cls:
        raise UnsupportedDesignFeature(
            f"{shape_type!r} resolves to {type(shape)!r}, which is not the "
            f"registered {codec.cls!r}; subclasses may carry state the "
            f"interop layer cannot see"
        )

    recorded = set(codec.fields)
    present = set(_instance_field_names(shape))
    missing = present - recorded
    if missing:
        raise UnsupportedDesignFeature(
            f"{shape_type} carries state the design-interop registry does not "
            f"record: {sorted(missing)}. Update rfx/interop/_shapes.py rather "
            f"than exporting a partial shape"
        )

    params: dict[str, Any] = {}
    for name, field in codec.fields.items():
        try:
            value = getattr(shape, name)
        except AttributeError as exc:
            raise UnsupportedDesignFeature(
                f"{shape_type} is missing the recorded parameter {name!r}"
            ) from exc
        params[name] = field.dump(value)
    return {"type": shape_type, "params": params}


def shape_from_dict(payload: dict[str, Any]) -> Any:
    """Rebuild a geometry primitive from :func:`shape_to_dict` output."""
    if not isinstance(payload, dict):
        raise UnsupportedDesignFeature(
            f"shape payload must be a mapping, got {type(payload).__name__}"
        )
    try:
        shape_type = payload["type"]
    except KeyError:
        raise UnsupportedDesignFeature(
            "shape payload is missing the 'type' key"
        ) from None

    codec = _codec_for_type(shape_type)
    params = payload.get("params", {})
    if not isinstance(params, dict):
        raise UnsupportedDesignFeature(
            f"{shape_type} params must be a mapping, got {type(params).__name__}"
        )

    unknown = set(params) - set(codec.fields)
    if unknown:
        raise UnsupportedDesignFeature(
            f"{shape_type} payload carries unknown parameters {sorted(unknown)}"
        )
    absent = set(codec.fields) - set(params)
    if absent:
        raise UnsupportedDesignFeature(
            f"{shape_type} payload is missing parameters {sorted(absent)}"
        )

    kwargs = {
        name: field.load(params[name])
        for name, field in codec.fields.items()
    }
    return codec.cls(**kwargs)
