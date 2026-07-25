"""Lossless serialisation of rfx geometry primitives.

``rfx.artifacts.build_scene_artifact`` records a shape as its class name plus a
bounding box (``artifacts.py`` pops the ``shape`` field and keeps ``_bbox``).
That is adequate for a review summary but cannot distinguish a ``Cylinder``
from a ``Box`` occupying the same bounding box, so it cannot rebuild its own
input and cannot drive an external solver.

This module records the *constructor parameters* instead.  An unregistered shape
class, and any value the layer cannot represent exactly, are refused loudly
(:class:`~rfx.interop._errors.UnsupportedDesignFeature`) rather than degraded to
a bounding box.

JSON canonical form uses lists for vectors; the Python canonical form restores
the tuples the primitives declare.  Round-tripping is therefore **normalising,
not identity**: a shape constructed with canonical containers (the tuples the
classes declare, which is what every construction path in this repo uses)
compares equal after a round trip, but one built with lists — ``Box(corner_lo=[0,
0, 0], ...)`` is legal, since the primitives do not coerce in ``__init__`` —
comes back with tuples and compares unequal while describing identical geometry.
``Via.layers`` normalises to a ``list`` of tuples and ``PolylineWire.points`` to
a ``tuple`` of tuples, matching what those classes declare.

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
from typing import Any, Callable, NamedTuple

from rfx.geometry.csg import Box, Cylinder, PolylineWire, Sphere
from rfx.geometry.curved import CurvedPatch
from rfx.geometry.via import Via
from rfx.interop._errors import UnsupportedDesignFeature
from rfx.interop._validate import (
    check_number,
    check_sequence,
    check_text,
    check_vector,
)

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


def _vec(n: int, *, what: str) -> _Field:
    return _Field(
        dump=lambda v: list(check_vector(v, n, what=what)),
        load=lambda v: check_vector(v, n, what=what),
    )


def _vec_seq(
    n: int,
    *,
    what: str,
    container: Callable[[Any], Any],
    min_length: int = 1,
) -> _Field:
    """A sequence of fixed-width vectors.

    ``min_length`` defaults to 1: an empty point list or layer stack does not
    describe a structure, and accepting one is how a consumed iterator used to
    turn into a silently absent conductor.
    """

    def _items(value: Any) -> tuple:
        return check_sequence(value, what=what, min_length=min_length)

    return _Field(
        dump=lambda v: [
            list(check_vector(p, n, what=f"{what}[{i}]"))
            for i, p in enumerate(_items(v))
        ],
        load=lambda v: container(
            check_vector(p, n, what=f"{what}[{i}]")
            for i, p in enumerate(_items(v))
        ),
    )


def _flt(*, what: str) -> _Field:
    return _Field(
        dump=lambda v: check_number(v, what=what),
        load=lambda v: check_number(v, what=what),
    )


def _txt(*, what: str) -> _Field:
    return _Field(
        dump=lambda v: check_text(v, what=what),
        load=lambda v: check_text(v, what=what),
    )


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
        "axis": _txt(what="cylinder.axis"),
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
        "material": _txt(what="via.material"),
    }),
    "curved_patch": _ShapeCodec(CurvedPatch, {
        "center": _vec(3, what="curved_patch.center"),
        "length": _flt(what="curved_patch.length"),
        "width": _flt(what="curved_patch.width"),
        "radius": _flt(what="curved_patch.radius"),
        "axis": _txt(what="curved_patch.axis"),
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

    Reads ``inspect.signature(cls.__init__)`` even for dataclasses, because
    ``dataclasses.fields`` has blind spots in both directions: it **excludes**
    ``InitVar`` (so an upstream ``units: InitVar[str] = "mm"`` consumed in
    ``__post_init__`` would not show up, and every export would silently carry
    values under an undeclared unit convention) and it **includes**
    ``field(init=False)`` derived attributes, which are not constructor
    parameters at all and must not be passed back on import.
    """
    signature_names = tuple(
        name
        for name in inspect.signature(cls.__init__).parameters
        if name != "self"
    )
    if not dataclasses.is_dataclass(cls):
        return signature_names
    # Union both views, preserving signature order, so an InitVar shows up and a
    # non-init field does not masquerade as a constructor parameter.
    init_fields = tuple(f.name for f in dataclasses.fields(cls) if f.init)
    extra = tuple(n for n in init_fields if n not in signature_names)
    return signature_names + extra


def _codec_for_kind(kind: str) -> _ShapeCodec:
    try:
        return _CODECS[kind]
    except (KeyError, TypeError):
        raise UnsupportedDesignFeature(
            f"shape kind {kind!r} is not supported by the design-interop "
            f"layer; supported kinds are {', '.join(SUPPORTED_SHAPE_KINDS)}"
        ) from None


def _instance_state_names(shape: Any) -> tuple[str, ...]:
    """Public instance state, for the "did this class grow a field?" check.

    Leading-underscore names are excluded because they are caches and memos, not
    design state — ``MeshShape.__init__`` already stamps ``self._mask_cache``
    (``rfx/geometry/mesh_import.py:66``), and without this exclusion a shape
    would become unexportable the moment it grew one, *after* having exported
    fine a moment earlier.

    Handles ``__slots__`` / NamedTuple classes, which have no ``__dict__``;
    unreachable today because :func:`shape_kind_of` refuses unregistered classes
    first, but ``rfx/geometry/thin_wire.py`` already defines a NamedTuple shape,
    so this is a live trap for whoever registers one.
    """
    if dataclasses.is_dataclass(shape):
        names: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(shape))
    elif hasattr(shape, "_fields"):  # NamedTuple
        names = tuple(shape._fields)
    elif hasattr(shape, "__dict__"):
        names = tuple(vars(shape))
    else:  # __slots__ without __dict__
        names = tuple(
            slot
            for klass in type(shape).__mro__
            for slot in getattr(klass, "__slots__", ())
        )
    return tuple(name for name in names if not name.startswith("_"))


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
    unrecorded = set(_instance_state_names(shape)) - recorded
    if unrecorded:
        raise UnsupportedDesignFeature(
            f"{kind} carries state the design-interop registry does not "
            f"record: {sorted(unrecorded)}. Update rfx/interop/_shapes.py "
            f"rather than exporting a partial shape"
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
    try:
        return codec.cls(**kwargs)
    except UnsupportedDesignFeature:
        raise
    except (ValueError, TypeError) as exc:
        # The primitive's own invariants (e.g. Via requires pad_radius >=
        # drill_radius, rfx/geometry/via.py:49) would otherwise escape as a bare
        # ValueError naming neither the kind nor the document. Note
        # UnsupportedDesignFeature IS a ValueError, so a caller writing
        # `except UnsupportedDesignFeature` would have missed those.
        raise UnsupportedDesignFeature(
            f"{kind} payload violates the primitive's own invariants: {exc}"
        ) from exc
