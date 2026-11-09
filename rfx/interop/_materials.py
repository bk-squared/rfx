"""Lossless serialisation of rfx material definitions.

``rfx.artifacts._material_summary`` reduces ``debye_poles`` / ``lorentz_poles``
to ``{"present": bool, "count": int}``, which is enough to review a run but
discards the pole parameters that define the dispersion.  A design description
rebuilt from that summary would silently become a non-dispersive material with
the same static ``eps_r`` — a different structure wearing the same name.

This module records every ``MaterialSpec`` field, including each pole's
parameters, and refuses anything it cannot represent exactly.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from rfx.api._spec import MaterialSpec
from rfx.interop._errors import UnsupportedDesignFeature
from rfx.interop._validate import check_number, check_text
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole
from rfx.materials.relative_strength import RelativeStrength

__all__ = [
    "material_from_dict",
    "material_to_dict",
    "materials_from_dict",
    "materials_to_dict",
]

# Pole parameter names are pinned here and asserted against the live NamedTuple
# definitions by the interop contract test, so adding a pole parameter upstream
# fails a test instead of vanishing from exported designs.
_POLE_FIELDS: dict[str, tuple[type, tuple[str, ...]]] = {
    "debye_poles": (DebyePole, ("delta_eps", "tau")),
    "lorentz_poles": (LorentzPole, ("omega_0", "delta", "kappa")),
}

_SCALAR_FIELDS: tuple[str, ...] = ("eps_r", "sigma", "mu_r", "chi3")


def _relative_value(value, what):
    """Record concrete dimensionless scalars or arrays, refusing live functions."""
    from rfx.core.jax_utils import is_tracer

    if is_tracer(value) or callable(value):
        raise UnsupportedDesignFeature(f"{what}: RelativeStrength requires a concrete value for export")
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
        raise UnsupportedDesignFeature(f"{what}: RelativeStrength value must contain finite numbers")
    return array.tolist()


def _material_spec_field_names() -> tuple[str, ...]:
    return tuple(f.name for f in dataclasses.fields(MaterialSpec))


def _dump_poles(kind: str, poles: Any, *, preserve_relative=False) -> list[dict] | None:
    if poles is None:
        return None
    pole_cls, names = _POLE_FIELDS[kind]
    dumped: list[dict[str, float]] = []
    for index, pole in enumerate(poles):
        if not isinstance(pole, pole_cls):
            raise UnsupportedDesignFeature(
                f"{kind}[{index}] is {type(pole).__name__}, expected "
                f"{pole_cls.__name__}; the interop layer will not guess its "
                f"parameter meaning"
            )
        values = {}
        for name in names:
            value = getattr(pole, name)
            what = f"{kind}[{index}].{name}"
            if isinstance(value, RelativeStrength):
                if preserve_relative:
                    values[name] = {"type": "RelativeStrength", "reference": float(value.reference),
                                    "value": _relative_value(value.value, f"{what}.value")}
                    continue
                value = value.si
            values[name] = check_number(value, what=what)
        dumped.append(values)
    return dumped


def _load_poles(kind: str, payload: Any) -> list[Any] | None:
    if payload is None:
        return None
    pole_cls, names = _POLE_FIELDS[kind]
    if not isinstance(payload, list):
        raise UnsupportedDesignFeature(
            f"{kind} must be a list or null, got {type(payload).__name__}"
        )
    poles: list[Any] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise UnsupportedDesignFeature(
                f"{kind}[{index}] must be a mapping, got {type(item).__name__}"
            )
        missing = set(names) - set(item)
        unknown = set(item) - set(names)
        if missing or unknown:
            raise UnsupportedDesignFeature(
                f"{kind}[{index}] parameter mismatch for "
                f"{pole_cls.__name__}: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        values = {}
        for name in names:
            value = item[name]
            what = f"{kind}[{index}].{name}"
            if name == "kappa" and isinstance(value, dict) and value.get("type") == "RelativeStrength":
                if set(value) != {"type", "reference", "value"}:
                    raise UnsupportedDesignFeature(f"{what}: invalid RelativeStrength fields")
                relative = _relative_value(value["value"], f"{what}.value")
                value = RelativeStrength(check_number(value["reference"], what=f"{what}.reference"),
                                         np.asarray(relative) if isinstance(relative, list) else relative)
            else:
                value = check_number(value, what=what)
            values[name] = value
        poles.append(pole_cls(**values))
    return poles


def material_to_dict(material: Any, *, preserve_relative=False) -> dict[str, Any]:
    """Serialise a :class:`MaterialSpec` losslessly.

    Raises
    ------
    UnsupportedDesignFeature
        If ``material`` is not a ``MaterialSpec``, or if ``MaterialSpec`` has
        gained a field this module does not record.
    """
    if type(material) is not MaterialSpec:
        raise UnsupportedDesignFeature(
            f"expected MaterialSpec, got {type(material).__name__}; the "
            f"interop layer does not infer fields of unknown material types"
        )

    recorded = set(_SCALAR_FIELDS) | set(_POLE_FIELDS)
    live = set(_material_spec_field_names())
    unrecorded = live - recorded
    if unrecorded:
        raise UnsupportedDesignFeature(
            f"MaterialSpec carries fields the design-interop layer does not "
            f"record: {sorted(unrecorded)}. Update rfx/interop/_materials.py "
            f"rather than exporting a partial material"
        )

    out: dict[str, Any] = {
        name: check_number(getattr(material, name), what=f"material.{name}")
        for name in _SCALAR_FIELDS
    }
    for kind in _POLE_FIELDS:
        out[kind] = _dump_poles(kind, getattr(material, kind), preserve_relative=preserve_relative)
    return out


def material_from_dict(payload: dict[str, Any]) -> MaterialSpec:
    """Rebuild a :class:`MaterialSpec` from :func:`material_to_dict` output."""
    if not isinstance(payload, dict):
        raise UnsupportedDesignFeature(
            f"material payload must be a mapping, got {type(payload).__name__}"
        )

    expected = set(_SCALAR_FIELDS) | set(_POLE_FIELDS)
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown or missing:
        raise UnsupportedDesignFeature(
            f"material payload field mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )

    kwargs: dict[str, Any] = {
        name: check_number(payload[name], what=f"material.{name}") for name in _SCALAR_FIELDS
    }
    for kind in _POLE_FIELDS:
        kwargs[kind] = _load_poles(kind, payload[kind])
    return MaterialSpec(**kwargs)


def materials_to_dict(materials: dict[str, Any]) -> dict[str, Any]:
    """Serialise a name → :class:`MaterialSpec` mapping."""
    if not isinstance(materials, dict):
        raise UnsupportedDesignFeature(
            f"materials must be a mapping of name to MaterialSpec, got "
            f"{type(materials).__name__}"
        )
    return {
        check_text(name, what="material name"): material_to_dict(spec)
        for name, spec in materials.items()
    }


def materials_from_dict(payload: dict[str, Any]) -> dict[str, MaterialSpec]:
    """Rebuild a name → :class:`MaterialSpec` mapping."""
    if not isinstance(payload, dict):
        raise UnsupportedDesignFeature(
            f"materials payload must be a mapping, got {type(payload).__name__}"
        )
    # Symmetric with materials_to_dict: a non-string key is refused rather than
    # coerced, so a JSON object with numeric-looking keys cannot quietly become a
    # material named "5".
    return {
        check_text(name, what="material name"): material_from_dict(spec)
        for name, spec in payload.items()
    }
