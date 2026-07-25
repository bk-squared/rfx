"""Round-trip-complete design description of an rfx ``Simulation``.

``design_to_dict(sim)`` serialises the complete *design state* of a
:class:`~rfx.api.Simulation` — the state a user established with the
constructor and the ``add_*`` builder methods — and
``simulation_from_design(document)`` rebuilds it.  The gate is field-by-field
equality of the builder state, not visual similarity of the JSON.

Contract
--------
1. **Refuse, never approximate.**  Any state that cannot be represented
   exactly raises :class:`~rfx.interop._errors.UnsupportedDesignFeature`,
   naming the entry family and its index.  There is no ``warnings.warn``-and-
   drop path and no bounding-box fallback: a design description that quietly
   loses a shape parameter, a dispersion pole or a mesh profile is worse than
   no description at all, because the resulting comparison looks valid while
   describing a different structure.
2. **Full arrays.**  Mesh profiles and frequency lists are emitted value by
   value, with their dtype and array namespace, never summarised.  Numbers are
   plain Python floats, so ``json.dump(..., allow_nan=False)`` with no
   ``default=str`` round-trips them exactly.  Non-finite values are refused.
3. **Closed world.**  An unknown key — at the top level or in any nested
   section — is refused on import.  This mirrors ``rfx/config/loader.py``.
4. **Design state only.**  No derived or transient state: no ``Grid`` /
   ``NonUniformGrid``, no ``dt``, no grid shape, no preflight scratch, no
   results.  No run-time control either (``n_steps``, ``until_decay``,
   ``compute_s_params``, ``eps_override``, ...): those change the answer but
   are arguments to ``run()`` / ``forward()``, not properties of the design.
5. **Fences are mirrored, never widened.**  The importer drives the public
   ``add_*`` builders, so every builder fence (for example the Floquet
   ``n_modes > 1`` / ``polarization='tm'`` ``NotImplementedError``) applies to
   an imported design exactly as it applies to hand-written code.  After the
   rebuild the importer re-exports the result and refuses if it differs from
   the input document, so an inexact reconstruction fails loudly instead of
   returning a plausible-looking ``Simulation``.

When NOT to call the exporter
-----------------------------
**Do not export from inside an S-parameter driver.**  ``rfx/api/_sparams.py``
temporarily *rewrites* builder state — ``_dz_profile``, ``_msl_ports``,
``_ports``, ``_dft_planes``, ``_waveguide_ports`` — and restores it in a
``finally`` block (see ``_sparams.py`` around lines 1489, 1528-1531,
1877-1880, 2691, 2727, 2765, 2958-2959).  A document captured from a callback
that runs inside one of those drivers describes the driver's synthetic
per-run configuration, not the user's design, and there is currently **no
sentinel on ``Simulation`` to detect it** — the substitutions are
value-level (a uniform ``_dz_profile``, a one-port ``_msl_ports``) and are
indistinguishable from a legitimate design.  Export from user-level code,
before or after the driver call.

A second time dependence: with ``dx=None`` the auto-mesh runs inside ``run()``
and writes back ``_dx``, ``_dz_profile`` and ``_domain``.  Exporting before a
run therefore yields ``"dx": null`` and no ``dz_profile``, and exporting after
the same run yields the resolved mesh.  Both are faithful; they describe
different design states of the same script.

Relationship to the other rfx setup serialisers
-----------------------------------------------
rfx already has four setup serialisers and every one of them is box-level:

- ``rfx/io.py::export_geometry_json`` — class name plus bounding box;
- ``rfx/artifacts.py::build_scene_artifact`` (``rfx-scene-artifact-v1``) — a
  provenance/report summary that self-declares it is "not a CAD export";
- ``rfx/config/_shapes.py`` — the YAML config CLI, ``_SUPPORTED_SHAPES =
  ("box",)``;
- ``rfx/experiments/canonical.py`` (``rfx-experiment/v1``) — "P0 supports box
  geometry", geometry carried as ``bounds_m``.

This module is deliberately **not** a fifth independent vocabulary.  The
``geometry`` and ``materials`` sections are shaped so they can be folded into
a future ``rfx-experiment/v2`` mechanically: shapes use the ``{"kind":
"<snake_case>", "params": {...}}`` discriminated union of
``rfx/interop/_shapes.py`` — the same ``kind`` spelling the config layer and
``canonical.py`` already use for a box — and materials use the
``rfx/interop/_materials.py`` payload verbatim.  A consumer that wants
box-only geometry can filter on ``kind == "box"``; it does not need a second
shape decoder.

External-solver portability
---------------------------
This schema is rfx→rfx.  Some design state round-trips perfectly here yet has
no meaning in another solver, and the document says so explicitly in its
``non_portable`` list rather than leaving a later external emitter to
rediscover it:

- ``_coaxial_terminations`` / ``_coaxial_open_terminations`` /
  ``_coaxial_pec_end_caps`` carry **cell-relative** axial offsets;
- ``_MSLPortEntry.n_probe_offset`` / ``n_probe_spacing`` are **cell counts**
  derived from ``_dx`` at registration time;
- ``_WaveguidePortEntry.probe_offset`` / ``ref_offset`` and
  ``_PortEntry.reference_plane_cells`` are likewise cell counts;
- ``_refinement`` is the rfx SBP-SAT subgrid prototype — experimental, and
  falsified in 3D (PR #90);
- ``cpml_layers`` is a cell count against an absorber whose remaining knobs
  are hard-coded in ``rfx/boundaries/cpml.py``;
- non-uniform mesh profiles have no counterpart in a solver with a scalar
  resolution.

``non_portable`` is an annotation derived from the document's own content, so
it carries no numbers and nothing is applied from it on import.  It is,
however, **verified**: because the importer re-exports and compares, a document
whose ``non_portable`` list has been edited is refused.  A downstream tool
cannot strip the annotation to make rfx-only state look portable.

Status: **provisional**.  Round-trip fidelity is gated by
``tests/test_interop_design_document.py``; no external emitter exists yet.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rfx.api._spec import (
    MATERIAL_LIBRARY,
    _DFTPlaneEntry,
    _FloquetPortEntry,
    _FluxMonitorEntry,
    _GeometryEntry,
    _MSLPortEntry,
    _PortEntry,
    _ProbeEntry,
    _TFSFEntry,
    _WaveguidePortEntry,
)
from rfx.boundaries.spec import BoundarySpec
from rfx.core.jax_utils import is_tracer
from rfx.interop._errors import UnsupportedDesignFeature
from rfx.interop._materials import (
    material_to_dict,
    materials_from_dict,
    materials_to_dict,
)
from rfx.interop._shapes import shape_from_dict, shape_to_dict
from rfx.lumped import LumpedRLCSpec
from rfx.materials.thin_conductor import ThinConductor
from rfx.sources.coaxial_port import CoaxialPort
from rfx.sources.sources import CWSource, GaussianPulse, ModulatedGaussian

__all__ = [
    "DESIGN_SCHEMA_VERSION",
    "SUPPORTED_WAVEFORM_KINDS",
    "design_to_dict",
    "design_to_json",
    "simulation_from_design",
]

DESIGN_SCHEMA_VERSION = "rfx-design-ir/v1"


# ---------------------------------------------------------------------------
# Scalar coercions
# ---------------------------------------------------------------------------

def _refuse(message: str) -> "UnsupportedDesignFeature":
    return UnsupportedDesignFeature(message)


def _num(value: Any, *, what: str) -> float:
    """Coerce to a finite Python float or refuse."""
    if is_tracer(value):
        raise _refuse(
            f"{what} is a JAX tracer; a traced value has no concrete number "
            f"to record. Export outside jit/grad, or pass a concrete value."
        )
    if isinstance(value, bool):
        raise _refuse(f"{what} must be a number, got a bool ({value!r})")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise _refuse(f"{what} must be a number, got {value!r}") from exc
    if not math.isfinite(out):
        raise _refuse(
            f"{what} is {out!r}; the design document refuses non-finite "
            f"numbers (JSON cannot represent them portably)"
        )
    return out


def _integer(value: Any, *, what: str) -> int:
    if isinstance(value, bool):
        raise _refuse(f"{what} must be an integer, got a bool ({value!r})")
    if is_tracer(value):
        raise _refuse(f"{what} is a JAX tracer; expected a concrete integer")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise _refuse(f"{what} must be an integer, got {value!r}") from exc
    if out != value:
        raise _refuse(f"{what} must be an exact integer, got {value!r}")
    return out


def _text(value: Any, *, what: str) -> str:
    if not isinstance(value, str):
        raise _refuse(f"{what} must be a string, got {type(value).__name__}")
    return value


def _flag(value: Any, *, what: str) -> bool:
    if not isinstance(value, bool):
        raise _refuse(f"{what} must be a bool, got {type(value).__name__}")
    return value


def _vector(value: Any, n: int, *, what: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        raise _refuse(f"{what} must be a sequence of {n} numbers, got {value!r}")
    if len(value) != n:
        raise _refuse(
            f"{what} must have exactly {n} components, got {len(value)}: {value!r}"
        )
    return tuple(_num(v, what=f"{what}[{i}]") for i, v in enumerate(value))


def _integer_vector(value: Any, n: int, *, what: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        raise _refuse(f"{what} must be a sequence of {n} integers, got {value!r}")
    if len(value) != n:
        raise _refuse(
            f"{what} must have exactly {n} components, got {len(value)}: {value!r}"
        )
    return tuple(_integer(v, what=f"{what}[{i}]") for i, v in enumerate(value))


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
        _num(v, what=f"{what}[{i}]") for i, v in enumerate(np.asarray(value).tolist())
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
    values = [_num(v, what=f"{what}.values[{i}]") for i, v in enumerate(raw)]

    if container == "list":
        return values
    if container == "tuple":
        return tuple(values)

    dtype = _text(payload["dtype"], what=f"{what}.dtype")
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


# ---------------------------------------------------------------------------
# Waveform codec
# ---------------------------------------------------------------------------
#
# ``CustomWaveform`` wraps a user callable and is unrepresentable by
# construction.  Note that ``add_polarized_source`` builds a
# ``CustomWaveform`` closure for a complex Jones vector, so a circularly
# polarised source is refused here — it is already unrecoverable from
# ``_ports`` inside rfx itself.

class _WaveformCodec(NamedTuple):
    cls: type
    fields: tuple[str, ...]


_WAVEFORM_CODECS: dict[str, _WaveformCodec] = {
    "gaussian_pulse": _WaveformCodec(
        GaussianPulse, ("f0", "bandwidth", "amplitude", "cutoff")
    ),
    "modulated_gaussian": _WaveformCodec(
        ModulatedGaussian, ("f0", "bandwidth", "amplitude", "cutoff")
    ),
    "cw_source": _WaveformCodec(CWSource, ("f0", "amplitude", "ramp_steps")),
}

SUPPORTED_WAVEFORM_KINDS: tuple[str, ...] = tuple(sorted(_WAVEFORM_CODECS))

_WAVEFORM_KIND_BY_CLASS = {c.cls: k for k, c in _WAVEFORM_CODECS.items()}

_WAVEFORM_INT_FIELDS = frozenset({"ramp_steps"})


def _waveform_to_dict(waveform: Any, *, what: str) -> dict[str, Any] | None:
    if waveform is None:
        return None
    try:
        kind = _WAVEFORM_KIND_BY_CLASS[type(waveform)]
    except KeyError:
        raise _refuse(
            f"{what} is a {type(waveform).__name__}; the design document "
            f"records {', '.join(SUPPORTED_WAVEFORM_KINDS)} only. A "
            f"CustomWaveform (including the closure add_polarized_source "
            f"builds for a complex Jones vector) wraps a Python callable and "
            f"has no serialisable form."
        ) from None
    codec = _WAVEFORM_CODECS[kind]
    live = tuple(f.name for f in dataclasses.fields(codec.cls))
    if live != codec.fields:
        raise _refuse(
            f"{codec.cls.__name__} now declares fields {live} but the design "
            f"document records {codec.fields}. Update rfx/interop/_design.py "
            f"rather than exporting a partial waveform"
        )
    params = {
        name: (
            _integer(getattr(waveform, name), what=f"{what}.{name}")
            if name in _WAVEFORM_INT_FIELDS
            else _num(getattr(waveform, name), what=f"{what}.{name}")
        )
        for name in codec.fields
    }
    return {"kind": kind, "params": params}


def _waveform_from_dict(payload: Any, *, what: str) -> Any:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise _refuse(f"{what} must be a mapping or null, got {type(payload).__name__}")
    _require_exact_keys(payload, {"kind", "params"}, what=what)
    kind = _text(payload["kind"], what=f"{what}.kind")
    if kind not in _WAVEFORM_CODECS:
        raise _refuse(
            f"{what}.kind {kind!r} is not supported; supported kinds are "
            f"{', '.join(SUPPORTED_WAVEFORM_KINDS)}"
        )
    codec = _WAVEFORM_CODECS[kind]
    params = payload["params"]
    if not isinstance(params, dict):
        raise _refuse(f"{what}.params must be a mapping, got {type(params).__name__}")
    _require_exact_keys(params, set(codec.fields), what=f"{what}.params")
    kwargs = {
        name: (
            _integer(params[name], what=f"{what}.params.{name}")
            if name in _WAVEFORM_INT_FIELDS
            else _num(params[name], what=f"{what}.params.{name}")
        )
        for name in codec.fields
    }
    return codec.cls(**kwargs)


# ---------------------------------------------------------------------------
# Closed-world key checking
# ---------------------------------------------------------------------------

def _require_exact_keys(payload: dict, expected: set[str], *, what: str) -> None:
    present = set(payload)
    unknown = present - expected
    missing = expected - present
    if unknown or missing:
        raise _refuse(
            f"{what} key mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _section(payload: Any, key: str, *, what: str) -> dict:
    value = payload[key]
    if not isinstance(value, dict):
        raise _refuse(
            f"{what}.{key} must be a mapping, got {type(value).__name__}"
        )
    return value


def _entry_list(payload: Any, key: str, *, what: str) -> list:
    value = payload[key]
    if not isinstance(value, list):
        raise _refuse(f"{what}.{key} must be a list, got {type(value).__name__}")
    return value


# ---------------------------------------------------------------------------
# Per-entry field registry
# ---------------------------------------------------------------------------
#
# Every builder-owned record type has one entry here.  ``_dump_entry`` asserts
# the live class declares exactly these fields, so a field added upstream reds
# the interop contract test instead of vanishing from exported designs.

class _F(NamedTuple):
    dump: Callable[[Any, str], Any]
    load: Callable[[Any, str], Any]


def _opt(field: _F) -> _F:
    return _F(
        dump=lambda v, w: None if v is None else field.dump(v, w),
        load=lambda v, w: None if v is None else field.load(v, w),
    )


_NUM = _F(dump=lambda v, w: _num(v, what=w), load=lambda v, w: _num(v, what=w))
_INT = _F(dump=lambda v, w: _integer(v, what=w), load=lambda v, w: _integer(v, what=w))
_STR = _F(dump=lambda v, w: _text(v, what=w), load=lambda v, w: _text(v, what=w))
_BOOL = _F(dump=lambda v, w: _flag(v, what=w), load=lambda v, w: _flag(v, what=w))
_SHAPE = _F(dump=lambda v, w: shape_to_dict(v), load=lambda v, w: shape_from_dict(v))
_WAVEFORM = _F(
    dump=lambda v, w: _waveform_to_dict(v, what=w),
    load=lambda v, w: _waveform_from_dict(v, what=w),
)
_ARRAY = _F(
    dump=lambda v, w: _array_to_dict(v, what=w),
    load=lambda v, w: _array_from_dict(v, what=w),
)


def _vec(n: int) -> _F:
    return _F(
        dump=lambda v, w: list(_vector(v, n, what=w)),
        load=lambda v, w: _vector(v, n, what=w),
    )


def _ivec(n: int) -> _F:
    return _F(
        dump=lambda v, w: list(_integer_vector(v, n, what=w)),
        load=lambda v, w: _integer_vector(v, n, what=w),
    )


_GEOMETRY_FIELDS: dict[str, _F] = {
    "shape": _SHAPE,
    "material_name": _STR,
}

# ``_PortEntry`` holds two physically different objects, discriminated only by
# the ``impedance == 0`` sentinel that ``add_source`` stamps.  The document
# splits them into ``soft_sources`` and ``lumped_ports``; a soft source records
# only the three fields ``add_source`` can set.
_SOFT_SOURCE_FIELDS: dict[str, _F] = {
    "position": _vec(3),
    "component": _STR,
    "waveform": _WAVEFORM,
}

_LUMPED_PORT_FIELDS: dict[str, _F] = {
    "position": _vec(3),
    "component": _STR,
    "impedance": _NUM,
    "waveform": _opt(_WAVEFORM),
    "extent": _opt(_NUM),
    "excite": _BOOL,
    "direction": _opt(_STR),
    "reference_plane_cells": _opt(_INT),
}

_PROBE_FIELDS: dict[str, _F] = {
    "position": _vec(3),
    "component": _STR,
}

_THIN_CONDUCTOR_FIELDS: dict[str, _F] = {
    "shape": _SHAPE,
    "sigma_bulk": _NUM,
    "thickness": _NUM,
    "eps_r": _NUM,
}

_COAXIAL_PORT_FIELDS: dict[str, _F] = {
    "position": _vec(3),
    "face": _STR,
    "pin_length": _NUM,
    "pin_radius": _NUM,
    "outer_radius": _NUM,
    "impedance": _NUM,
    "excitation": _WAVEFORM,
}

_TFSF_FIELDS: dict[str, _F] = {
    "f0": _opt(_NUM),
    "bandwidth": _NUM,
    "amplitude": _NUM,
    "margin": _INT,
    "polarization": _STR,
    "direction": _STR,
    "angle_deg": _NUM,
    "waveform": _STR,
}

_DFT_PLANE_FIELDS: dict[str, _F] = {
    "name": _STR,
    "axis": _STR,
    "coordinate": _NUM,
    "component": _STR,
    "freqs": _opt(_ARRAY),
    "n_freqs": _INT,
}

_FLUX_MONITOR_FIELDS: dict[str, _F] = {
    "name": _STR,
    "axis": _STR,
    "coordinate": _NUM,
    "freqs": _opt(_ARRAY),
    "n_freqs": _INT,
    "size": _opt(_vec(2)),
    "center": _opt(_vec(2)),
    "dft_window": _STR,
    "dft_window_alpha": _NUM,
}

_WAVEGUIDE_PORT_FIELDS: dict[str, _F] = {
    "name": _STR,
    "x_position": _NUM,
    "y_range": _opt(_vec(2)),
    "z_range": _opt(_vec(2)),
    "x_range": _opt(_vec(2)),
    "mode": _ivec(2),
    "mode_type": _STR,
    "direction": _STR,
    "freqs": _opt(_ARRAY),
    "n_freqs": _INT,
    "f0": _opt(_NUM),
    "bandwidth": _NUM,
    "amplitude": _NUM,
    "probe_offset": _INT,
    "ref_offset": _INT,
    "calibration_preset": _opt(_STR),
    "reference_plane": _opt(_NUM),
    "probe_plane": _opt(_NUM),
    "n_modes": _INT,
    "waveform": _STR,
    "mode_profile": _STR,
}

_FLOQUET_PORT_FIELDS: dict[str, _F] = {
    "name": _STR,
    "position": _NUM,
    "axis": _STR,
    "scan_theta": _NUM,
    "scan_phi": _NUM,
    "polarization": _STR,
    "n_modes": _INT,
    "freqs": _opt(_ARRAY),
    "n_freqs": _INT,
    "f0": _opt(_NUM),
    "bandwidth": _NUM,
    "amplitude": _NUM,
}

_MSL_PORT_FIELDS: dict[str, _F] = {
    "name": _STR,
    "position": _vec(3),
    "width": _NUM,
    "height": _NUM,
    "direction": _STR,
    "impedance": _NUM,
    "waveform": _opt(_WAVEFORM),
    "excite": _BOOL,
    "n_probe_offset": _INT,
    "n_probe_spacing": _INT,
    "n_probes": _INT,
    "mode": _STR,
    "eps_r_sub": _opt(_NUM),
}

_LUMPED_RLC_FIELDS: dict[str, _F] = {
    "R": _NUM,
    "L": _NUM,
    "C": _NUM,
    "topology": _STR,
    "position": _vec(3),
    "component": _STR,
}

_REFINEMENT_FIELDS: dict[str, _F] = {
    "z_range": _vec(2),
    "ratio": _INT,
    "xy_margin": _opt(_NUM),
    "tau": _NUM,
    "validation": _STR,
    "topology": _STR,
}

_NTFF_FIELDS: dict[str, _F] = {
    "corner_lo": _vec(3),
    "corner_hi": _vec(3),
    "freqs": _ARRAY,
}

# Record classes pinned against their field registry.  ``_PortEntry`` is
# absent because it is split across two document sections and checked
# separately in ``_dump_ports``.
_PINNED_RECORDS: tuple[tuple[type, dict[str, _F]], ...] = (
    (_GeometryEntry, _GEOMETRY_FIELDS),
    (_ProbeEntry, _PROBE_FIELDS),
    (ThinConductor, _THIN_CONDUCTOR_FIELDS),
    (CoaxialPort, _COAXIAL_PORT_FIELDS),
    (_TFSFEntry, _TFSF_FIELDS),
    (_DFTPlaneEntry, _DFT_PLANE_FIELDS),
    (_FluxMonitorEntry, _FLUX_MONITOR_FIELDS),
    (_WaveguidePortEntry, _WAVEGUIDE_PORT_FIELDS),
    (_FloquetPortEntry, _FLOQUET_PORT_FIELDS),
    (_MSLPortEntry, _MSL_PORT_FIELDS),
    (LumpedRLCSpec, _LUMPED_RLC_FIELDS),
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
    payload: Any, fields: dict[str, _F], *, what: str
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise _refuse(f"{what} must be a mapping, got {type(payload).__name__}")
    _require_exact_keys(payload, set(fields), what=what)
    return {
        name: field.load(payload[name], f"{what}.{name}")
        for name, field in fields.items()
    }


def _dump_list(
    entries: Any, fields: dict[str, _F], cls: type, *, what: str
) -> list[dict[str, Any]]:
    return [
        _dump_entry(entry, fields, cls, what=f"{what}[{index}]")
        for index, entry in enumerate(entries)
    ]


# ---------------------------------------------------------------------------
# Exported / excluded attribute inventory
# ---------------------------------------------------------------------------
#
# Anti-drift ledger.  ``tests/test_interop_design_document.py`` asserts that
# every ``self._*`` attribute a fresh ``Simulation`` carries appears in exactly
# one of these two tuples, so a new builder field cannot slip into rfx without
# a decision being recorded here.

EXPORTED_SIMULATION_ATTRS: tuple[str, ...] = (
    "_adi_cfl_factor",
    "_boundary",
    "_boundary_spec",
    "_coaxial_open_terminations",
    "_coaxial_pec_end_caps",
    "_coaxial_ports",
    "_coaxial_terminations",
    "_cpml_kappa_max",
    "_cpml_layers",
    "_dft_planes",
    "_domain",
    "_dx",
    "_dx_profile",
    "_dy_profile",
    "_dz_profile",
    "_floquet_ports",
    "_flux_monitors",
    "_freq_max",
    "_geometry",
    "_lumped_rlc",
    "_materials",
    "_mode",
    "_msl_ports",
    "_ntff",
    "_pec_faces",
    "_periodic_axes",
    "_ports",
    "_precision",
    "_probes",
    "_refinement",
    "_solver",
    "_stencil_order",
    "_tfsf",
    "_thin_conductors",
    "_waveguide_ports",
)

#: Attributes deliberately absent from the document.  These are derived,
#: transient, or run-time state — see the module docstring, rule 4.  A
#: ``Simulation`` does not carry them at construction; they appear only after a
#: preflight/run pass, which is exactly why they are not design state.
EXCLUDED_SIMULATION_ATTRS: tuple[str, ...] = (
    "_ntff_min_steps_hint",
)


# ---------------------------------------------------------------------------
# Boundary section
# ---------------------------------------------------------------------------

def _predict_legacy_spec(
    boundary: str, pec_faces: set[str], periodic_axes: str
) -> BoundarySpec:
    """Pure mirror of ``Simulation._build_spec_from_legacy``.

    Used to decide which construction path reproduces a recorded boundary
    state; kept as a separate function so the choice is made before any
    ``Simulation`` is built (and so the mirror is testable on its own).
    """
    from rfx.boundaries.spec import Boundary

    axes = {}
    for axis in "xyz":
        if axis in periodic_axes:
            axes[axis] = Boundary(lo="periodic", hi="periodic")
        else:
            lo = "pec" if f"{axis}_lo" in pec_faces else boundary
            hi = "pec" if f"{axis}_hi" in pec_faces else boundary
            axes[axis] = Boundary(lo=lo, hi=hi)
    return BoundarySpec(x=axes["x"], y=axes["y"], z=axes["z"])


def _dump_boundary(sim: Any) -> dict[str, Any]:
    spec = sim._boundary_spec
    if not isinstance(spec, BoundarySpec):
        raise _refuse(
            f"_boundary_spec is a {type(spec).__name__}, expected BoundarySpec; "
            f"the design document treats the spec as authoritative"
        )
    return {
        # BoundarySpec.to_dict / from_dict are reused verbatim — the boundary
        # vocabulary lives in rfx/boundaries/spec.py, not here.
        "spec": spec.to_dict(),
        # Derived views, emitted explicitly rather than re-derived: _boundary
        # and _cpml_layers can disagree with the spec on the legacy
        # scalar + set_periodic_axes() path, and _periodic_axes is also
        # mutated as a side effect by add_floquet_port.
        "legacy": {
            "boundary": _text(sim._boundary, what="_boundary"),
            "cpml_layers": _integer(sim._cpml_layers, what="_cpml_layers"),
            "cpml_kappa_max": _num(sim._cpml_kappa_max, what="_cpml_kappa_max"),
            "pec_faces": sorted(_text(f, what="_pec_faces entry") for f in sim._pec_faces),
            "periodic_axes": _text(sim._periodic_axes, what="_periodic_axes"),
        },
    }


class _BoundaryPlan(NamedTuple):
    """Constructor kwargs plus an optional deprecated periodic-axes call."""

    kwargs: dict[str, Any]
    set_periodic_axes: str | None


def _plan_boundary(payload: dict, *, has_floquet: bool) -> _BoundaryPlan:
    """Choose the construction path that reproduces a recorded boundary state.

    ``Simulation`` accepts either a ``BoundarySpec`` (authoritative; the
    legacy views are then derived from it) or the legacy
    ``boundary=<scalar> + pec_faces`` triad.  The two paths do **not** produce
    the same attributes for the same physics: a legacy ``boundary="pec"`` cavity
    keeps ``_pec_faces == set()`` while the equivalent all-PEC ``BoundarySpec``
    derives ``_pec_faces`` = all six faces.  Reproducing the recorded state
    therefore means reproducing the path, which is decided from the document
    rather than guessed.
    """
    _require_exact_keys(payload, {"spec", "legacy"}, what="boundary")
    spec_payload = payload["spec"]
    if not isinstance(spec_payload, dict):
        raise _refuse(
            f"boundary.spec must be a mapping, got {type(spec_payload).__name__}"
        )
    _require_exact_keys(spec_payload, {"x", "y", "z"}, what="boundary.spec")
    try:
        spec = BoundarySpec.from_dict(spec_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise _refuse(f"boundary.spec is not a valid BoundarySpec: {exc}") from exc

    legacy = payload["legacy"]
    if not isinstance(legacy, dict):
        raise _refuse(
            f"boundary.legacy must be a mapping, got {type(legacy).__name__}"
        )
    _require_exact_keys(
        legacy,
        {"boundary", "cpml_layers", "cpml_kappa_max", "pec_faces", "periodic_axes"},
        what="boundary.legacy",
    )
    scalar = _text(legacy["boundary"], what="boundary.legacy.boundary")
    cpml_layers = _integer(legacy["cpml_layers"], what="boundary.legacy.cpml_layers")
    kappa = _num(legacy["cpml_kappa_max"], what="boundary.legacy.cpml_kappa_max")
    raw_faces = legacy["pec_faces"]
    if not isinstance(raw_faces, list):
        raise _refuse(
            f"boundary.legacy.pec_faces must be a list, got "
            f"{type(raw_faces).__name__}"
        )
    pec_faces = {
        _text(face, what=f"boundary.legacy.pec_faces[{i}]")
        for i, face in enumerate(raw_faces)
    }
    periodic = _text(legacy["periodic_axes"], what="boundary.legacy.periodic_axes")

    common = {"cpml_kappa_max": kappa}

    # Preferred path: hand the spec to the constructor and let it derive the
    # legacy views. Accept it only when the derivation reproduces every
    # recorded legacy value. ``_periodic_axes`` is allowed to be empty in the
    # spec when Floquet ports are present, because add_floquet_port fills it in
    # as a documented side effect.
    derived_scalar = spec.absorber_type or "pec"
    derived_layers = cpml_layers if derived_scalar in ("cpml", "upml") else 0
    spec_periodic = spec.periodic_axes()
    periodic_ok = spec_periodic == periodic or (has_floquet and spec_periodic == "")
    if (
        derived_scalar == scalar
        and derived_layers == cpml_layers
        and spec.pec_faces() == pec_faces
        and periodic_ok
    ):
        return _BoundaryPlan(
            kwargs={"boundary": spec, "cpml_layers": cpml_layers, **common},
            set_periodic_axes=None,
        )

    # Fallback: reproduce the legacy triad. ``_build_spec_from_legacy`` folds
    # ``_periodic_axes`` into the spec, so try both the empty value (Floquet
    # side effect, applied later by the builder) and the recorded value
    # (an explicit deprecated set_periodic_axes call).
    if not (scalar == "pec" and pec_faces):
        for axes in ("", periodic):
            if _predict_legacy_spec(scalar, pec_faces, axes) == spec:
                return _BoundaryPlan(
                    kwargs={
                        "boundary": scalar,
                        "cpml_layers": cpml_layers,
                        "pec_faces": set(pec_faces) or None,
                        **common,
                    },
                    set_periodic_axes=axes or None,
                )

    raise _refuse(
        "boundary.spec and boundary.legacy cannot both be reproduced by any "
        f"Simulation construction path: spec={spec.to_dict()!r}, "
        f"legacy boundary={scalar!r}, cpml_layers={cpml_layers}, "
        f"pec_faces={sorted(pec_faces)}, periodic_axes={periodic!r}"
    )


# ---------------------------------------------------------------------------
# Port / source split
# ---------------------------------------------------------------------------

_SOFT_SOURCE_PINNED_DEFAULTS: dict[str, Any] = {
    "extent": None,
    "excite": True,
    "direction": None,
    "reference_plane_cells": None,
}


def _dump_ports(sim: Any) -> tuple[list[dict], list[dict]]:
    """Split ``_ports`` into soft sources and lumped/wire ports.

    The two families share ``_PortEntry`` and are discriminated only by the
    ``impedance == 0`` sentinel ``add_source`` stamps (``add_port`` rejects a
    non-positive impedance).  The document names them separately so a consumer
    never has to re-derive the intent from a sentinel.
    """
    live = set(live_field_names(_PortEntry))
    unrecorded = live - set(_LUMPED_PORT_FIELDS)
    if unrecorded:
        raise _refuse(
            f"_PortEntry carries fields the design document does not record: "
            f"{sorted(unrecorded)}. Update rfx/interop/_design.py rather than "
            f"exporting a partial port"
        )

    soft: list[dict] = []
    lumped: list[dict] = []
    for index, entry in enumerate(sim._ports):
        what = f"_ports[{index}]"
        if type(entry) is not _PortEntry:
            raise _refuse(
                f"{what} is a {type(entry).__name__}, expected _PortEntry"
            )
        if _num(entry.impedance, what=f"{what}.impedance") == 0.0:
            for name, expected in _SOFT_SOURCE_PINNED_DEFAULTS.items():
                actual = getattr(entry, name)
                if actual != expected:
                    raise _refuse(
                        f"{what} is a soft source (impedance == 0) but carries "
                        f"{name}={actual!r}; add_source() cannot set that "
                        f"field, so the entry was not built through the public "
                        f"API and cannot be rebuilt through it"
                    )
            soft.append(
                {
                    name: field.dump(getattr(entry, name), f"{what}.{name}")
                    for name, field in _SOFT_SOURCE_FIELDS.items()
                }
            )
        else:
            lumped.append(
                {
                    name: field.dump(getattr(entry, name), f"{what}.{name}")
                    for name, field in _LUMPED_PORT_FIELDS.items()
                }
            )
    return soft, lumped


# ---------------------------------------------------------------------------
# Coaxial termination tuples (cell-relative)
# ---------------------------------------------------------------------------

def _dump_coaxial_matched_loads(sim: Any) -> list[dict[str, Any]]:
    out = []
    for index, item in enumerate(sim._coaxial_terminations):
        what = f"_coaxial_terminations[{index}]"
        if len(item) != 3:
            raise _refuse(f"{what} must be a 3-tuple, got {item!r}")
        port_index, impedance, offset = item
        out.append(
            {
                "port_index": _integer(port_index, what=f"{what}.port_index"),
                "target_impedance": _num(impedance, what=f"{what}.target_impedance"),
                "axial_offset_cells": _integer(
                    offset, what=f"{what}.axial_offset_cells"
                ),
            }
        )
    return out


def _dump_coaxial_pairs(
    items: Any, second_key: str, *, what: str
) -> list[dict[str, Any]]:
    out = []
    for index, item in enumerate(items):
        label = f"{what}[{index}]"
        if len(item) != 2:
            raise _refuse(f"{label} must be a 2-tuple, got {item!r}")
        port_index, value = item
        out.append(
            {
                "port_index": _integer(port_index, what=f"{label}.port_index"),
                second_key: _integer(value, what=f"{label}.{second_key}"),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Non-portability annotation
# ---------------------------------------------------------------------------

def _non_portable(document: dict[str, Any]) -> list[dict[str, str]]:
    """Annotate the sections of *document* whose semantics are rfx-only.

    Derived purely from the document's own content, so it is deterministic and
    byte-stable.  It is an annotation: the importer validates its shape and
    ignores its content.
    """
    notes: list[dict[str, str]] = []

    def note(path: str, reason: str) -> None:
        notes.append({"path": path, "reason": reason})

    mesh = document["mesh"]
    if any(mesh[k] is not None for k in ("dx_profile", "dy_profile", "dz_profile")):
        note(
            "mesh",
            "per-axis cell-size profiles describe a graded rfx mesh; openEMS "
            "needs explicit mesh lines and Meep has no non-uniform mesh at "
            "all, so a profile must be re-meshed, never averaged to a scalar "
            "resolution",
        )
    if document["boundary"]["legacy"]["cpml_layers"] > 0:
        note(
            "boundary.legacy.cpml_layers",
            "absorber depth as a cell count; the remaining CPML knobs "
            "(sigma_max, alpha_max, grading order) are hard-coded in "
            "rfx/boundaries/cpml.py, so two solvers given the same layer "
            "count build different absorbers",
        )

    solver = document["solver"]
    if (
        solver["solver"] != "yee"
        or solver["precision"] != "float32"
        or solver["stencil_order"] != 2
    ):
        note(
            "solver",
            "ADI, mixed precision and the (2,4) stencil are rfx-specific "
            "solver settings with no counterpart in openEMS or Meep",
        )

    excitations = document["excitations"]
    if excitations["msl_ports"]:
        note(
            "excitations.msl_ports",
            "n_probe_offset and n_probe_spacing are cell counts derived from "
            "dx when the port was registered; they carry no physical length "
            "and are meaningless on another mesh",
        )
    if any(p["reference_plane_cells"] is not None for p in excitations["lumped_ports"]):
        note(
            "excitations.lumped_ports",
            "reference_plane_cells is a cell count, not a physical "
            "measurement-plane distance",
        )
    if excitations["waveguide_ports"]:
        note(
            "excitations.waveguide_ports",
            "probe_offset and ref_offset are cell counts from the source "
            "plane, not physical distances",
        )
    for key, field in (
        ("coaxial_matched_loads", "axial_offset_cells"),
        ("coaxial_open_terminations", "pin_retract_cells"),
        ("coaxial_pec_end_caps", "axial_offset_cells"),
    ):
        if excitations[key]:
            note(
                f"excitations.{key}",
                f"{field} is a cell-relative offset; the termination moves "
                f"physically if dx changes",
            )
    if excitations["tfsf"] is not None:
        note(
            "excitations.tfsf",
            "the rfx TFSF injector is a narrow-scope 1-D auxiliary-grid source "
            "(x-normal, ez/ey, CPML only) and margin is in cells; Meep has no "
            "TFSF primitive",
        )
    if document["refinement"] is not None:
        note(
            "refinement",
            "SBP-SAT subgridding is an rfx research prototype, falsified in 3D "
            "(PR #90) and outside the public support scope; it must not be "
            "exported to another solver, where its presence would read as a "
            "validated capability",
        )

    notes.sort(key=lambda item: item["path"])
    return notes


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------

def _rfx_version() -> str:
    import rfx

    return str(rfx.__version__)


def _dump_materials(sim: Any) -> dict[str, Any]:
    materials = sim._materials
    if not isinstance(materials, dict):
        raise _refuse(
            f"_materials must be a dict, got {type(materials).__name__}"
        )
    return materials_to_dict(materials)


def _dump_material_library(sim: Any) -> dict[str, Any]:
    """Snapshot the library materials this design references but never registered.

    A geometry entry may name ``"pec"`` or ``"fr4"`` without ever touching
    ``_materials``; ``Simulation._resolve_material`` falls through to
    ``MATERIAL_LIBRARY``.  Those values are version-dependent (the Rogers
    conductivities are computed from Df at a fixed frequency), so the document
    records the resolved numbers and the importer verifies them against the
    live library.  The section is provenance, not applied state: importing does
    not register these materials, exactly as the original design did not.
    """
    referenced = {
        entry.material_name
        for entry in sim._geometry
        if isinstance(entry, _GeometryEntry)
    }
    out: dict[str, Any] = {}
    for name in sorted(referenced):
        if name in sim._materials:
            continue
        if name not in MATERIAL_LIBRARY:
            raise _refuse(
                f"geometry references material {name!r}, which is neither "
                f"registered with add_material() nor a library name "
                f"({sorted(MATERIAL_LIBRARY)})"
            )
        out[name] = material_to_dict(sim._resolve_material(name))
    return out


def design_to_dict(sim: Any) -> dict[str, Any]:
    """Serialise the complete design state of *sim* to a JSON-ready mapping.

    Parameters
    ----------
    sim : Simulation
        A simulation whose design has been established through the
        constructor and the ``add_*`` builders.

    Returns
    -------
    dict
        A ``rfx-design-ir/v1`` document.  Every value is a JSON scalar, list
        or mapping; ``json.dump(document, allow_nan=False)`` succeeds without
        ``default=``.

    Raises
    ------
    UnsupportedDesignFeature
        If any design state cannot be represented exactly — an unregistered
        shape class (``MeshShape``, a user-defined ``Shape``), a
        ``CustomWaveform``, a JAX-traced mesh profile, a non-finite number, or
        a record class that has grown a field this module does not record.

    Notes
    -----
    Do **not** call this from inside an S-parameter driver: those drivers
    temporarily rewrite ``_dz_profile`` / ``_msl_ports`` / ``_ports`` /
    ``_dft_planes`` / ``_waveguide_ports`` and restore them afterwards, so a
    document captured mid-driver describes the driver's synthetic
    configuration.  There is no sentinel on ``Simulation`` that lets this
    function detect the situation, so the responsibility is the caller's.  See
    the module docstring.
    """
    soft_sources, lumped_ports = _dump_ports(sim)

    ntff = None
    if sim._ntff is not None:
        if len(sim._ntff) != 3:
            raise _refuse(
                f"_ntff must be a (corner_lo, corner_hi, freqs) triple, got "
                f"{sim._ntff!r}"
            )
        corner_lo, corner_hi, freqs = sim._ntff
        ntff = {
            "corner_lo": _NTFF_FIELDS["corner_lo"].dump(corner_lo, "_ntff.corner_lo"),
            "corner_hi": _NTFF_FIELDS["corner_hi"].dump(corner_hi, "_ntff.corner_hi"),
            "freqs": _NTFF_FIELDS["freqs"].dump(freqs, "_ntff.freqs"),
        }

    refinement = None
    if sim._refinement is not None:
        if not isinstance(sim._refinement, dict):
            raise _refuse(
                f"_refinement must be a dict, got {type(sim._refinement).__name__}"
            )
        _require_exact_keys(
            sim._refinement, set(_REFINEMENT_FIELDS), what="_refinement"
        )
        refinement = {
            name: field.dump(sim._refinement[name], f"_refinement.{name}")
            for name, field in _REFINEMENT_FIELDS.items()
        }

    tfsf = None
    if sim._tfsf is not None:
        tfsf = _dump_entry(sim._tfsf, _TFSF_FIELDS, _TFSFEntry, what="_tfsf")

    document: dict[str, Any] = {
        "schema": DESIGN_SCHEMA_VERSION,
        "rfx_version": _rfx_version(),
        "domain": {
            "freq_max": _num(sim._freq_max, what="_freq_max"),
            "extent": list(_vector(sim._domain, 3, what="_domain")),
            "mode": _text(sim._mode, what="_mode"),
        },
        "mesh": {
            # dx is None on the auto-mesh path: "let rfx choose" is itself the
            # design decision, and it round-trips as null.
            "dx": None if sim._dx is None else _num(sim._dx, what="_dx"),
            "dx_profile": (
                None
                if sim._dx_profile is None
                else _array_to_dict(sim._dx_profile, what="_dx_profile")
            ),
            "dy_profile": (
                None
                if sim._dy_profile is None
                else _array_to_dict(sim._dy_profile, what="_dy_profile")
            ),
            "dz_profile": (
                None
                if sim._dz_profile is None
                else _array_to_dict(sim._dz_profile, what="_dz_profile")
            ),
        },
        "boundary": _dump_boundary(sim),
        "solver": {
            "precision": _text(sim._precision, what="_precision"),
            "solver": _text(sim._solver, what="_solver"),
            "adi_cfl_factor": _num(sim._adi_cfl_factor, what="_adi_cfl_factor"),
            "stencil_order": _integer(sim._stencil_order, what="_stencil_order"),
        },
        "materials": _dump_materials(sim),
        "material_library": _dump_material_library(sim),
        "geometry": _dump_list(
            sim._geometry, _GEOMETRY_FIELDS, _GeometryEntry, what="_geometry"
        ),
        "thin_conductors": _dump_list(
            sim._thin_conductors,
            _THIN_CONDUCTOR_FIELDS,
            ThinConductor,
            what="_thin_conductors",
        ),
        "excitations": {
            "soft_sources": soft_sources,
            "lumped_ports": lumped_ports,
            "msl_ports": _dump_list(
                sim._msl_ports, _MSL_PORT_FIELDS, _MSLPortEntry, what="_msl_ports"
            ),
            "waveguide_ports": _dump_list(
                sim._waveguide_ports,
                _WAVEGUIDE_PORT_FIELDS,
                _WaveguidePortEntry,
                what="_waveguide_ports",
            ),
            "coaxial_ports": _dump_list(
                sim._coaxial_ports,
                _COAXIAL_PORT_FIELDS,
                CoaxialPort,
                what="_coaxial_ports",
            ),
            "coaxial_matched_loads": _dump_coaxial_matched_loads(sim),
            "coaxial_open_terminations": _dump_coaxial_pairs(
                sim._coaxial_open_terminations,
                "pin_retract_cells",
                what="_coaxial_open_terminations",
            ),
            "coaxial_pec_end_caps": _dump_coaxial_pairs(
                sim._coaxial_pec_end_caps,
                "axial_offset_cells",
                what="_coaxial_pec_end_caps",
            ),
            "floquet_ports": _dump_list(
                sim._floquet_ports,
                _FLOQUET_PORT_FIELDS,
                _FloquetPortEntry,
                what="_floquet_ports",
            ),
            "tfsf": tfsf,
            "lumped_rlc": _dump_list(
                sim._lumped_rlc, _LUMPED_RLC_FIELDS, LumpedRLCSpec, what="_lumped_rlc"
            ),
        },
        "observables": {
            "probes": _dump_list(
                sim._probes, _PROBE_FIELDS, _ProbeEntry, what="_probes"
            ),
            "dft_planes": _dump_list(
                sim._dft_planes, _DFT_PLANE_FIELDS, _DFTPlaneEntry, what="_dft_planes"
            ),
            "flux_monitors": _dump_list(
                sim._flux_monitors,
                _FLUX_MONITOR_FIELDS,
                _FluxMonitorEntry,
                what="_flux_monitors",
            ),
            "ntff": ntff,
        },
        "refinement": refinement,
    }
    document["non_portable"] = _non_portable(document)
    return document


def design_to_json(sim: Any, *, indent: int | None = 2) -> str:
    """Canonical JSON text for :func:`design_to_dict`.

    Keys are sorted and NaN/Infinity are rejected, so two exports of the same
    design produce byte-identical text.  Numbers use Python's shortest
    round-trip repr, which reproduces float64 exactly.
    """
    return json.dumps(
        design_to_dict(sim), sort_keys=True, indent=indent, allow_nan=False
    )


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------

_TOP_LEVEL_KEYS = {
    "schema",
    "rfx_version",
    "domain",
    "mesh",
    "boundary",
    "solver",
    "materials",
    "material_library",
    "geometry",
    "thin_conductors",
    "excitations",
    "observables",
    "refinement",
    "non_portable",
}

_EXCITATION_KEYS = {
    "soft_sources",
    "lumped_ports",
    "msl_ports",
    "waveguide_ports",
    "coaxial_ports",
    "coaxial_matched_loads",
    "coaxial_open_terminations",
    "coaxial_pec_end_caps",
    "floquet_ports",
    "tfsf",
    "lumped_rlc",
}

_OBSERVABLE_KEYS = {"probes", "dft_planes", "flux_monitors", "ntff"}


def _resolve_library_material(name: str):
    """Resolve a ``MATERIAL_LIBRARY`` row exactly as ``_resolve_material`` does."""
    from rfx.api._spec import MaterialSpec

    row = MATERIAL_LIBRARY[name]
    return MaterialSpec(
        eps_r=row.get("eps_r", 1.0),
        sigma=row.get("sigma", 0.0),
        mu_r=row.get("mu_r", 1.0),
        debye_poles=row.get("debye_poles"),
        lorentz_poles=row.get("lorentz_poles"),
    )


def _verify_material_library(payload: Any, materials: dict) -> None:
    if not isinstance(payload, dict):
        raise _refuse(
            f"material_library must be a mapping, got {type(payload).__name__}"
        )
    for name, recorded in payload.items():
        if name in materials:
            raise _refuse(
                f"material_library[{name!r}] shadows a material registered in "
                f"the materials section; a name resolves to one or the other, "
                f"never both"
            )
        if name not in MATERIAL_LIBRARY:
            raise _refuse(
                f"material_library[{name!r}] is not a name in this rfx "
                f"version's MATERIAL_LIBRARY ({sorted(MATERIAL_LIBRARY)})"
            )
        live = material_to_dict(_resolve_library_material(name))
        if not _equal(live, recorded):
            raise _refuse(
                f"material_library[{name!r}] does not match this rfx version's "
                f"library: recorded={recorded!r}, live={live!r}. The library "
                f"values are version-dependent, so importing would silently "
                f"change the design's material."
            )


def simulation_from_design(document: Any) -> Any:
    """Rebuild a :class:`~rfx.api.Simulation` from a ``rfx-design-ir/v1`` document.

    The rebuild goes through the public ``add_*`` builders, so every builder
    fence applies unchanged — an imported design is not a way around a
    ``NotImplementedError`` in the API.  Warnings the builders emit for the
    recorded configuration (a deprecated ``pec_faces=``, a PEC-routed thin
    conductor, a single-component series RLC) are re-emitted, because the
    imported design really does have that configuration.

    After the rebuild the design is re-exported and compared with *document*;
    a mismatch raises :class:`UnsupportedDesignFeature` naming the first
    differing path, so an inexact reconstruction fails loudly instead of
    returning a plausible-looking ``Simulation``.

    Raises
    ------
    UnsupportedDesignFeature
        On an unknown or missing key, an unsupported shape/waveform kind, a
        schema mismatch, a drifted material-library value, or a design that no
        ``Simulation`` construction path reproduces exactly.
    """
    from rfx.api import Simulation

    if not isinstance(document, dict):
        raise _refuse(
            f"design document must be a mapping, got {type(document).__name__}"
        )
    _require_exact_keys(document, _TOP_LEVEL_KEYS, what="design document")

    schema = _text(document["schema"], what="schema")
    if schema != DESIGN_SCHEMA_VERSION:
        raise _refuse(
            f"schema {schema!r} is not {DESIGN_SCHEMA_VERSION!r}; this reader "
            f"does not translate between schema versions"
        )
    _text(document["rfx_version"], what="rfx_version")
    if not isinstance(document["non_portable"], list):
        raise _refuse(
            f"non_portable must be a list, got "
            f"{type(document['non_portable']).__name__}"
        )

    domain = _section(document, "domain", what="design document")
    _require_exact_keys(domain, {"freq_max", "extent", "mode"}, what="domain")

    mesh = _section(document, "mesh", what="design document")
    _require_exact_keys(
        mesh, {"dx", "dx_profile", "dy_profile", "dz_profile"}, what="mesh"
    )

    solver = _section(document, "solver", what="design document")
    _require_exact_keys(
        solver,
        {"precision", "solver", "adi_cfl_factor", "stencil_order"},
        what="solver",
    )

    excitations = _section(document, "excitations", what="design document")
    _require_exact_keys(excitations, _EXCITATION_KEYS, what="excitations")

    observables = _section(document, "observables", what="design document")
    _require_exact_keys(observables, _OBSERVABLE_KEYS, what="observables")

    materials = materials_from_dict(
        _section(document, "materials", what="design document")
    )
    _verify_material_library(document["material_library"], materials)

    floquet_payloads = _entry_list(excitations, "floquet_ports", what="excitations")
    plan = _plan_boundary(
        _section(document, "boundary", what="design document"),
        has_floquet=bool(floquet_payloads),
    )

    sim = Simulation(
        freq_max=_num(domain["freq_max"], what="domain.freq_max"),
        domain=_vector(domain["extent"], 3, what="domain.extent"),
        dx=None if mesh["dx"] is None else _num(mesh["dx"], what="mesh.dx"),
        mode=_text(domain["mode"], what="domain.mode"),
        dx_profile=(
            None
            if mesh["dx_profile"] is None
            else _array_from_dict(mesh["dx_profile"], what="mesh.dx_profile")
        ),
        dy_profile=(
            None
            if mesh["dy_profile"] is None
            else _array_from_dict(mesh["dy_profile"], what="mesh.dy_profile")
        ),
        dz_profile=(
            None
            if mesh["dz_profile"] is None
            else _array_from_dict(mesh["dz_profile"], what="mesh.dz_profile")
        ),
        precision=_text(solver["precision"], what="solver.precision"),
        solver=_text(solver["solver"], what="solver.solver"),
        adi_cfl_factor=_num(solver["adi_cfl_factor"], what="solver.adi_cfl_factor"),
        stencil_order=_integer(solver["stencil_order"], what="solver.stencil_order"),
        **plan.kwargs,
    )
    if plan.set_periodic_axes is not None:
        sim.set_periodic_axes(plan.set_periodic_axes)

    for name, spec in materials.items():
        sim.add_material(
            name,
            eps_r=spec.eps_r,
            sigma=spec.sigma,
            mu_r=spec.mu_r,
            debye_poles=spec.debye_poles,
            lorentz_poles=spec.lorentz_poles,
            chi3=spec.chi3,
        )

    refinement = document["refinement"]
    if refinement is not None:
        values = _load_entry(refinement, _REFINEMENT_FIELDS, what="refinement")
        sim.add_refinement(
            values.pop("z_range"),
            **values,
        )

    for index, payload in enumerate(
        _entry_list(document, "geometry", what="design document")
    ):
        values = _load_entry(
            payload, _GEOMETRY_FIELDS, what=f"geometry[{index}]"
        )
        sim.add(values["shape"], material=values["material_name"])

    for index, payload in enumerate(
        _entry_list(document, "thin_conductors", what="design document")
    ):
        values = _load_entry(
            payload, _THIN_CONDUCTOR_FIELDS, what=f"thin_conductors[{index}]"
        )
        sim.add_thin_conductor(values.pop("shape"), **values)

    for index, payload in enumerate(
        _entry_list(excitations, "coaxial_ports", what="excitations")
    ):
        values = _load_entry(
            payload,
            _COAXIAL_PORT_FIELDS,
            what=f"excitations.coaxial_ports[{index}]",
        )
        sim.add_coaxial_port(
            values["position"],
            values["face"],
            pin_length=values["pin_length"],
            pin_radius=values["pin_radius"],
            outer_radius=values["outer_radius"],
            impedance=values["impedance"],
            waveform=values["excitation"],
        )

    for index, payload in enumerate(
        _entry_list(excitations, "coaxial_matched_loads", what="excitations")
    ):
        what = f"excitations.coaxial_matched_loads[{index}]"
        if not isinstance(payload, dict):
            raise _refuse(f"{what} must be a mapping, got {type(payload).__name__}")
        _require_exact_keys(
            payload,
            {"port_index", "target_impedance", "axial_offset_cells"},
            what=what,
        )
        sim.add_coaxial_matched_load(
            _integer(payload["port_index"], what=f"{what}.port_index"),
            target_impedance=_num(
                payload["target_impedance"], what=f"{what}.target_impedance"
            ),
            axial_offset_cells=_integer(
                payload["axial_offset_cells"], what=f"{what}.axial_offset_cells"
            ),
        )

    for index, payload in enumerate(
        _entry_list(excitations, "coaxial_open_terminations", what="excitations")
    ):
        what = f"excitations.coaxial_open_terminations[{index}]"
        if not isinstance(payload, dict):
            raise _refuse(f"{what} must be a mapping, got {type(payload).__name__}")
        _require_exact_keys(payload, {"port_index", "pin_retract_cells"}, what=what)
        sim.add_coaxial_open_termination(
            _integer(payload["port_index"], what=f"{what}.port_index"),
            pin_retract_cells=_integer(
                payload["pin_retract_cells"], what=f"{what}.pin_retract_cells"
            ),
        )

    for index, payload in enumerate(
        _entry_list(excitations, "coaxial_pec_end_caps", what="excitations")
    ):
        what = f"excitations.coaxial_pec_end_caps[{index}]"
        if not isinstance(payload, dict):
            raise _refuse(f"{what} must be a mapping, got {type(payload).__name__}")
        _require_exact_keys(payload, {"port_index", "axial_offset_cells"}, what=what)
        sim.add_coaxial_pec_end_cap(
            _integer(payload["port_index"], what=f"{what}.port_index"),
            axial_offset_cells=_integer(
                payload["axial_offset_cells"], what=f"{what}.axial_offset_cells"
            ),
        )

    for index, payload in enumerate(
        _entry_list(excitations, "soft_sources", what="excitations")
    ):
        values = _load_entry(
            payload,
            _SOFT_SOURCE_FIELDS,
            what=f"excitations.soft_sources[{index}]",
        )
        sim.add_source(
            values["position"], values["component"], waveform=values["waveform"]
        )

    for index, payload in enumerate(
        _entry_list(excitations, "lumped_ports", what="excitations")
    ):
        values = _load_entry(
            payload,
            _LUMPED_PORT_FIELDS,
            what=f"excitations.lumped_ports[{index}]",
        )
        sim.add_port(
            values.pop("position"),
            values.pop("component"),
            **values,
        )

    for index, payload in enumerate(
        _entry_list(excitations, "msl_ports", what="excitations")
    ):
        values = _load_entry(
            payload, _MSL_PORT_FIELDS, what=f"excitations.msl_ports[{index}]"
        )
        sim.add_msl_port(values.pop("position"), **values)

    for index, payload in enumerate(
        _entry_list(excitations, "lumped_rlc", what="excitations")
    ):
        values = _load_entry(
            payload, _LUMPED_RLC_FIELDS, what=f"excitations.lumped_rlc[{index}]"
        )
        sim.add_lumped_rlc(
            values.pop("position"),
            values.pop("component"),
            **values,
        )

    tfsf = excitations["tfsf"]
    if tfsf is not None:
        sim.add_tfsf_source(
            **_load_entry(tfsf, _TFSF_FIELDS, what="excitations.tfsf")
        )

    for index, payload in enumerate(
        _entry_list(excitations, "waveguide_ports", what="excitations")
    ):
        values = _load_entry(
            payload,
            _WAVEGUIDE_PORT_FIELDS,
            what=f"excitations.waveguide_ports[{index}]",
        )
        sim.add_waveguide_port(values.pop("x_position"), **values)

    for index, payload in enumerate(floquet_payloads):
        values = _load_entry(
            payload,
            _FLOQUET_PORT_FIELDS,
            what=f"excitations.floquet_ports[{index}]",
        )
        sim.add_floquet_port(values.pop("position"), **values)

    for index, payload in enumerate(
        _entry_list(observables, "probes", what="observables")
    ):
        values = _load_entry(
            payload, _PROBE_FIELDS, what=f"observables.probes[{index}]"
        )
        sim.add_probe(values["position"], values["component"])

    for index, payload in enumerate(
        _entry_list(observables, "dft_planes", what="observables")
    ):
        values = _load_entry(
            payload, _DFT_PLANE_FIELDS, what=f"observables.dft_planes[{index}]"
        )
        sim.add_dft_plane_probe(**values)

    for index, payload in enumerate(
        _entry_list(observables, "flux_monitors", what="observables")
    ):
        values = _load_entry(
            payload,
            _FLUX_MONITOR_FIELDS,
            what=f"observables.flux_monitors[{index}]",
        )
        sim.add_flux_monitor(**values)

    ntff = observables["ntff"]
    if ntff is not None:
        values = _load_entry(ntff, _NTFF_FIELDS, what="observables.ntff")
        sim.add_ntff_box(
            values["corner_lo"], values["corner_hi"], values["freqs"]
        )

    _assert_round_trip(sim, document)
    return sim


# ---------------------------------------------------------------------------
# Round-trip self-check
# ---------------------------------------------------------------------------

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


def _assert_round_trip(sim: Any, document: dict) -> None:
    rebuilt = design_to_dict(sim)
    if _equal(rebuilt, document):
        return
    where = _first_difference(rebuilt, document) or "<unknown>"
    raise _refuse(
        f"the design could not be rebuilt exactly: {where}. The document "
        f"describes state no public builder call reproduces, so the import is "
        f"refused rather than returning an approximate Simulation."
    )
