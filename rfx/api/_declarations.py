"""Simulation declarations shared by the preflight and geometry record caches.

Entries and material records are immutable declarations: keys retain them rather
than just their addresses. Concrete mesh profiles are small mutable arrays, so
keys snapshot their contents. Traced inputs are retained without concretization.
"""
from collections.abc import Mapping

import jax
import numpy as np


DECLARATIONS = (
    ("_geometry", "entries"),
    ("_thin_conductors", "entries"),
    ("_pinned_sheets", "entries"),
    ("_ports", "entries"),
    ("_waveguide_ports", "entries"),
    ("_msl_ports", "entries"),
    ("_coaxial_ports", "entries"),
    ("_floquet_ports", "entries"),
    ("_lumped_rlc", "entries"),
    ("_probes", "entries"),
    ("_dft_planes", "entries"),
    ("_flux_monitors", "entries"),
    ("_materials", "mapping"),
    ("_dx_profile", "array"),
    ("_dy_profile", "array"),
    ("_dz_profile", "array"),
    ("_dx", "value"),
    ("_domain", "value"),
    ("_freq_max", "value"),
    ("_boundary", "value"),
    ("_boundary_spec", "value"),
    ("_boundary_model", "value"),
    ("_boundary_explicit", "value"),
    ("_boundary_scalar_spec", "value"),
    ("_pec_faces", "value"),
    ("_periodic_axes", "value"),
    ("_cpml_layers", "value"),
    ("_cpml_kappa_max", "value"),
    ("_mode", "value"),
    ("_solver", "value"),
    ("_precision", "value"),
    ("_adi_cfl_factor", "value"),
    ("_stencil_order", "value"),
    ("_interface_eps", "value"),
    ("_snap", "value"),
    ("_dt_pin", "value"),
    ("_dt_min_cell", "value"),
    ("_nominal_profiles", "value"),
    ("_refinement", "value"),
    ("_tfsf", "value"),
    ("_ntff", "value"),
    ("_current_moments", "value"),
    ("_dft_plane_regions", "value"),
    ("_internal_probe_indices", "value"),
    ("_msl_auto_offset_min", "value"),
    ("_msl_auto_probe_spacing", "value"),
    ("_msl_auto_probe_lengths", "value"),
)

# The literal reason "cache" is also the cache inventory for declaration copies.
NOT_DECLARATIONS = (
    ("_pf_campaign_ctx", "cache"),
    ("_realized_geometry_record", "cache"),
    ("_coax_geometry", "cache"),
    ("_mesh_resolution", "cache"),
    ("_frozen_mesh", "committed mesh snapshot; resolved mesh declarations key it"),
    ("_amplitude_kind_warned", "run state: source amplitude warning already emitted"),
    ("_boundary_default_warned", "run state: implicit waveguide boundary warning already emitted"),
    ("_waveguide_call_overrides", "run state: per-call preflight material overrides"),
)


class _Same:
    """Identity equality with a strong reference, safe even for traced records."""

    __slots__ = ("obj",)

    def __init__(self, obj):
        self.obj = obj

    def __eq__(self, other):
        return isinstance(other, _Same) and self.obj is other.obj

    def __hash__(self):
        return id(self.obj)


def _array_key(value):
    if value is None:
        return None
    if isinstance(value, jax.core.Tracer):
        return _Same(value)
    array = np.asarray(value)
    return array.shape, array.dtype.str, array.tobytes()


def _value_key(value):
    """Freeze container-valued declarations into hashable scalar/tuple values."""
    if isinstance(value, jax.core.Tracer):
        return _Same(value)
    if isinstance(value, (np.ndarray, jax.Array)):
        return _array_key(value)
    if isinstance(value, Mapping):
        return tuple(sorted((name, _value_key(item)) for name, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_value_key(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_value_key(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return _Same(value)
    return value


def declaration_key(sim):
    """Snapshot every declaration, including the observer declarations."""
    key = []
    for attr, form in DECLARATIONS:
        value = getattr(sim, attr, None)
        if form == "entries":
            term = tuple(_Same(entry) for entry in (value or ()))
        elif form == "mapping":
            term = tuple(sorted((name, _Same(entry)) for name, entry in (value or {}).items()))
        elif form == "array":
            term = _array_key(value)
        else:
            term = _value_key(value)
        key.append((attr, term))
    return tuple(key)
