"""Name traced declarations and report advisory checks that cannot judge them.

Only strings and de-duplication keys live in the call scope; no tracer is retained.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
import warnings

from jax import errors

from rfx.core.jax_utils import is_tracer
from rfx.errors import RfxError


# Phase-1 Table 2, including declaration containers created after __init__.
CONTAINER_METHODS = {
    "_domain": "__init__",
    "_nominal_profiles": "__init__",
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
    "_boundary_scalar_spec": "__init__",  # boundary= scalar and its expanded face spec.
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
    attributes = dict(vars(sim))
    for attr, value in attributes.items():
        if attr not in CONTAINER_METHODS and not isinstance(value, (list, dict, tuple)):
            continue
        method = CONTAINER_METHODS.get(attr, attr)
        if attr == "_domain":
            pairs = (("domain", value),)
        elif attr == "_nominal_profiles":
            pairs = (("nominal_profiles", value),)
        elif attr in _POSITIONAL_FIELDS:
            pairs = zip(_POSITIONAL_FIELDS[attr], () if value is None else value)
        elif attr in ("_refinement", "_dft_plane_regions"):
            pairs = () if value is None else value.items()
        elif isinstance(value, (list, dict)):
            records = value.values() if isinstance(value, dict) else value
            pairs = (pair for record in records for pair in _record_fields(record))
        else:
            pairs = _record_fields(value)
        for field, child in pairs:
            if _has_tracer(child):
                names.add(f"{method}({field})")
    for field, value in entry_inputs.items():
        if _has_tracer(value):
            names.add(f"forward({field})")
    return tuple(sorted(names))


# Mutable set belongs to this invocation, including nested check calls.
_scope = ContextVar("preflight_not_judged_scope", default=None)
_check = ContextVar("preflight_not_judged_check", default=None)
_report = ContextVar("preflight_not_judged_report", default=None)


@contextmanager
def preflight_scope(names):
    token = _scope.set((names, set()))
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
    scope = _scope.get()
    return () if scope is None else scope[0]


def current_check(default):
    return _check.get() or default


def is_tracer_error(exc):
    """Recognize conversion failures swallowed by existing narrow handlers."""
    return not isinstance(exc, RfxError) and isinstance(exc, TRACER_ERRORS)


def not_judged(report, check_name, names):
    """Deliver one warning per (check, names) in this preflight invocation."""
    from rfx.preflight._common import PreflightIssue, PreflightWarning

    names = tuple(sorted(set(names)))
    key = (check_name, names)
    scope = _scope.get()
    if scope is not None:
        if key in scope[1]:
            return
        scope[1].add(key)
    message = (f"not judged: {check_name} reads a traced input ({', '.join(names)})"
               if names else f"not judged: {check_name} reads an input traced inside rfx")
    if report is None:
        report = _report.get()
    if report is not None and hasattr(report, "append"):
        report.append(PreflightIssue(message, severity="warning", code="not_judged_traced"))
    else:
        emit = warnings.warn if report is None else getattr(report, "warn", report)
        emit(PreflightWarning(message, severity="warning", code="not_judged_traced"))


def judge_or_report(check_name, call, report, names):
    """Convert only JAX conversion failures; named rfx refusals take precedence."""
    token = _check.set(check_name)
    try:
        try:
            return call()
        except RfxError:
            raise
        except TRACER_ERRORS:
            not_judged(report, check_name, names)
        return None
    finally:
        _check.reset(token)
