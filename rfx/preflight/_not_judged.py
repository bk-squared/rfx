"""Name traced declarations and report advisory checks that cannot judge them.

Declaration references live only in the call scope; names are resolved on demand.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, is_dataclass
from types import SimpleNamespace
from typing import Any, Iterable
import warnings

from jax import errors

from rfx.core.jax_utils import is_tracer
from rfx.errors import RfxError


# Phase-1 Table 2, including declaration containers created after __init__.
CONTAINER_METHODS = {
    "_domain": "Simulation",
    "_nominal_profiles": "Simulation",
    "_materials": "add_material",
    "_geometry": "add",
    "_ports": "add_port",
    "_probes": "add_probe",
    "_thin_conductors": "add_thin_conductor",
    "_pinned_sheets": "add_pinned_sheet",
    "_coaxial_ports": "add_coaxial_port",
    "_ntff": "add_ntff_box",
    "_current_moments": "add_current_moment_monitor",
    "_dft_planes": "add_dft_plane_probe",
    "_flux_monitors": "add_flux_monitor",
    "_waveguide_ports": "add_waveguide_port",
    "_msl_ports": "add_msl_port",
    "_msl_auto_offset_min": "add_msl_port",
    "_msl_auto_probe_spacing": "add_msl_port",
    "_msl_auto_probe_lengths": "add_msl_port",
    "_refinement": "add_refinement",
    "_lumped_rlc": "add_lumped_rlc",
    "_floquet_ports": "add_floquet_port",
    "_tfsf": "add_tfsf_source",
    "_wall_losses": "add_wall_loss",
    "_dft_plane_regions": "add_dft_plane_probe",
    "_boundary_scalar_spec": "Simulation",  # boundary= scalar and its expanded face spec.
}
CONSTRUCTOR_PARAMETERS = {
    "_freq_max": "freq_max",
    "_boundary": "boundary",
    "_boundary_explicit": "boundary",
    "_cpml_layers": "cpml_layers",
    "_cpml_kappa_max": "cpml_kappa_max",
    "_dx": "dx",
    "_mode": "mode",
    "_dz_profile": "dz_profile",
    "_dx_profile": "dx_profile",
    "_dy_profile": "dy_profile",
    "_dt_pin": "dt",
    "_dt_min_cell": "dt_min_cell",
    "_precision": "precision",
    "_solver": "solver",
    "_stencil_order": "stencil_order",
    "_snap": "snap",
    "_interface_eps": "interface_eps",
    "_pec_occupancy_model": "pec_occupancy_model",
}
_POSITIONAL_FIELDS = {
    "_ntff": ("corner_lo", "corner_hi", "freqs"),
    "_current_moments": ("corner_lo", "corner_hi", "block_size", "freqs", "order", "extra"),
}
TRACER_ERRORS = (
    errors.ConcretizationTypeError,
    errors.TracerArrayConversionError,
    errors.TracerBoolConversionError,
    errors.TracerIntegerConversionError,
)


def _has_tracer(value, ancestors=frozenset()):
    if is_tracer(value):
        return True
    if id(value) in ancestors:
        return False
    ancestors = ancestors | {id(value)}
    children: Iterable[Any]
    if is_dataclass(value) and not isinstance(value, type):
        children = (getattr(value, f.name) for f in fields(value))
    elif isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    else:
        return False  # Arrays are leaves, never traversed or converted.
    return any(_has_tracer(child, ancestors) for child in children)


def _record_fields(record):
    if is_dataclass(record) and not isinstance(record, type):
        return ((f.name, getattr(record, f.name)) for f in fields(record))
    if isinstance(record, tuple) and hasattr(record, "_fields"):
        return zip(record._fields, record)
    if isinstance(record, dict):
        return record.items()
    return (("value", record),)


def traced_input_names(sim, entry_inputs) -> tuple[str, ...]:
    """Walk declarations once, naming first-level record fields, sorted and unique."""
    names = set()
    for attr, method in CONTAINER_METHODS.items():
        value = getattr(sim, attr, None)
        pairs: Iterable[tuple[str, Any]]
        if attr == "_domain":
            pairs = (("domain", value),)
        elif attr == "_nominal_profiles":
            pairs = (("nominal_profiles", value),)
        elif attr == "_boundary_scalar_spec":
            pairs = (("boundary", value),)
        elif attr in _POSITIONAL_FIELDS:
            pairs = zip(_POSITIONAL_FIELDS[attr], () if value is None else value)
        elif attr in ("_refinement", "_dft_plane_regions"):
            pairs = () if value is None else value.items()
        elif isinstance(value, (list, dict)):
            records = value.values() if isinstance(value, dict) else value
            pairs = (pair for record in records for pair in _record_fields(record))
        else:
            pairs = _record_fields(value)
        for field_name, child in pairs:
            if _has_tracer(child):
                names.add(f"{method}({field_name})")
    for attr, parameter in CONSTRUCTOR_PARAMETERS.items():
        if _has_tracer(getattr(sim, attr, None)):
            names.add(f"Simulation({parameter})")
    for field_name, value in entry_inputs.items():
        if _has_tracer(value):
            names.add(f"forward({field_name})")
    return tuple(sorted(names))


@dataclass
class _Scope:
    declarations: object
    entry_inputs: dict
    names: tuple[str, ...] | None = None
    seen: set = field(default_factory=set)


# Snapshot and mutable set belong to this invocation, including nested checks.
_scope: ContextVar[_Scope | None] = ContextVar("preflight_not_judged_scope", default=None)
_check = ContextVar("preflight_not_judged_check", default=None)
_report = ContextVar("preflight_not_judged_report", default=None)


@contextmanager
def preflight_scope(sim, entry_inputs=None):
    declarations = SimpleNamespace(**{
        attr: getattr(sim, attr, None)
        for attr in (*CONTAINER_METHODS, *CONSTRUCTOR_PARAMETERS)
    })
    token = _scope.set(_Scope(declarations, dict(entry_inputs or {})))
    report_token = _report.set(None)
    check_token = _check.set(None)
    try:
        yield
    finally:
        _check.reset(check_token)
        _report.reset(report_token)
        _scope.reset(token)


@contextmanager
def reporting_to(report):
    """Late subject resolution is outside warning capture; append to its report."""
    token = _report.set(report)
    try:
        yield
    finally:
        _report.reset(token)


def current_names():
    """Read already resolved names without causing a declaration scan."""
    scope = _scope.get()
    return () if scope is None or scope.names is None else scope.names


def current_check(default):
    return _check.get() or default


def is_tracer_error(exc):
    """Recognize conversion failures swallowed by existing narrow handlers."""
    return not isinstance(exc, RfxError) and isinstance(exc, TRACER_ERRORS)


def traced_skip(value, check_name):
    """Test one traced value and report the skipped check in the same call."""
    traced = is_tracer(value)
    if traced:
        not_judged(None, current_check(check_name))
    return traced


def report_tracer_error(exc, check_name):
    """Report only a tracer conversion failure from an existing handler."""
    if is_tracer_error(exc):
        not_judged(None, current_check(check_name))


def not_judged(report, check_name):
    """Deliver one warning per (check, names) in this preflight invocation."""
    scope = _scope.get()
    if scope is None:
        return
    from rfx.preflight._common import PreflightIssue, PreflightWarning

    if scope.names is None:
        scope.names = traced_input_names(scope.declarations, scope.entry_inputs)
    names = scope.names
    key = (check_name, names)
    if key in scope.seen:
        return
    scope.seen.add(key)
    message = (f"not judged: {check_name} reads a traced input ({', '.join(names)})"
               if names else f"not judged: {check_name} reads an input traced inside rfx")
    if report is None:
        report = _report.get()
    if report is not None and hasattr(report, "append"):
        report.append(PreflightIssue(message, severity="warning", code="not_judged_traced"))
    else:
        emit = warnings.warn if report is None else getattr(report, "warn", report)
        emit(PreflightWarning(message, severity="warning", code="not_judged_traced"))


def judge_or_report(check_name, call, report):
    """Convert only JAX conversion failures; named rfx refusals take precedence."""
    token = _check.set(check_name)
    try:
        try:
            return call()
        except RfxError:
            raise
        except TRACER_ERRORS:
            not_judged(report, check_name)
        return None
    finally:
        _check.reset(token)
