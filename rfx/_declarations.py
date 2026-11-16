"""Simulation declarations shared by the preflight and geometry record caches.

Entries and material records are immutable declarations: keys retain them rather
than just their addresses. Concrete mesh profiles are small mutable arrays, so
keys snapshot their contents. Traced inputs are retained without concretization.
"""
from collections.abc import Mapping
from copy import copy
import math

import jax
import numpy as np


DECLARATIONS = (
    ("_geometry", "entries"),
    ("_thin_conductors", "entries"),
    ("_pinned_sheets", "entries"),
    ("_wall_losses", "entries"),
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
    ("_boundary_explicit", "value"),
    ("_boundary_scalar_spec", "value"),
    ("_pec_faces", "value"),
    ("_periodic_axes", "value"),
    ("_cpml_layers", "value"),
    ("_cpml_kappa_max", "value"),
    ("_mode", "value"),
    ("_solver", "value"),
    ("_precision", "value"),
    ("_stencil_order", "value"),
    ("_interface_eps", "value"),
    ("_pec_occupancy_model", "value"),
    ("_snap", "value"),
    ("_dt_pin", "value"),
    ("_dt_min_cell", "value"),
    ("_nominal_profiles", "value"),
    ("_refinement", "value"),
    ("_tfsf", "value"),
    ("_ntff", "value"),
    ("_current_moments", "value"),
    ("_dft_plane_regions", "value"),
    ("_time_recorded_dft_planes", "value"),
    ("_internal_probe_indices", "value"),
    ("_msl_auto_offset_min", "value"),
    ("_msl_auto_probe_spacing", "value"),
    ("_msl_auto_probe_lengths", "value"),
)

# The literal reason "cache" is also the cache inventory for declaration copies.
NOT_DECLARATIONS = (
    ("_realization_early_outcomes", "derived: early_outcomes feeds the realization table"),
    ("_pf_campaign_ctx", "cache"),
    ("_realized_geometry_record", "cache"),
    ("_coax_geometry", "cache"),
    ("_mesh_resolution", "cache keyed on its inputs; not released"),
    ("_msl_default_memo", "cache"),
    ("_boundary_model", "derived; its inputs are keyed"),
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


def release_declaration_caches(sim):
    """Release declaration-derived products before a new declaration or solve."""
    for attr, reason in NOT_DECLARATIONS:
        if reason == "cache":
            setattr(sim, attr, None)


def declaration_root(sim):
    return getattr(sim, "_declaration_root", sim)


def declaration_variant(sim, **replacements):
    """Copy a declaration with private containers and release derived products.

    Declaration entries and array inputs remain shared; callers replacing
    mutable value-form state must supply their own container.
    """
    unknown = replacements.keys() - dict(DECLARATIONS).keys()
    if unknown:
        raise TypeError(f"Unknown DECLARATIONS replacement(s): {', '.join(sorted(unknown))}")
    variant = copy(sim)
    variant._declaration_root = declaration_root(sim)
    for attr, form in DECLARATIONS:
        if form in {"entries", "mapping"}:
            value = getattr(sim, attr, None)
            setattr(variant, attr, list(value or ()) if form == "entries" else dict(value or {}))
    for attr, value in replacements.items():
        setattr(variant, attr, value)
    release_declaration_caches(variant)
    return variant


DEVICE_DECLARATIONS = frozenset(
    name for name, form in DECLARATIONS if form in {"entries", "mapping"}
) | {
    "_refinement", "_tfsf", "_ntff", "_current_moments", "_dft_plane_regions",
    "_time_recorded_dft_planes",
    "_internal_probe_indices", "_msl_auto_offset_min", "_msl_auto_probe_spacing",
    "_msl_auto_probe_lengths",
}


def declaration_at_dx(sim, factor=1.0, *, caller, pin_resolved_mesh=False, _base_dx=None):
    """Declare the same model at ``factor`` times its built cell size.

    Profiles, mesh-deformation pins, node-index sheets/crops and frozen meshes
    cannot be rescaled: use ``convergence_study`` with a factory declaring each
    mesh. Cell counts (CPML layers, port offsets, probe spacing, current-moment
    offsets and TF/SF margins) stay counts; their size in metres scales with
    the factor. MSL auto memos are carried unchanged. Entries and mesh arrays
    remain shared, while
    declaration containers and run state belong to this new model.

    ``pin_resolved_mesh`` fixes the built dx and resolved domain/profiles before
    geometry is removed from an empty-guide reference. ``_base_dx`` lets a
    convergence study build the factor-one grid just once, on its own copy.
    """
    try:
        factor = float(factor)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{caller}: factor must be finite and positive") from exc
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError(f"{caller}: factor must be finite and positive")
    if pin_resolved_mesh and factor != 1.0:
        raise ValueError(f"{caller}: pin_resolved_mesh requires factor=1.0")

    mesh_names = {"_dx", "_domain", "_dx_profile", "_dy_profile", "_dz_profile"}
    if factor != 1.0:
        mesh = sim._resolve_mesh()
        reasons = []
        for name in ("_dx_profile", "_dy_profile", "_dz_profile"):
            if mesh[name] is not None:
                origin = "inferred by automatic meshing" if sim.__dict__.get(name) is None else "declared"
                reasons.append(f"{name[1:]}= ({origin}) fixes axis cell widths")
        for name, public in (("_nominal_profiles", "nominal_profiles="),
                             ("_dt_min_cell", "dt_min_cell=")):
            if getattr(sim, name, None) is not None:
                reasons.append(f"{public} pins a mesh-deformation family")
        if getattr(sim, "_pinned_sheets", None):
            reasons.append("add_pinned_sheet: a node-pinned sheet has size set by NODE INDEX; "
                           "at dx/2 the conductor is half as long")
        if getattr(sim, "_dft_plane_regions", None):
            reasons.append("add_dft_plane_probe(region=) fixes a node-index crop")
        if "_frozen_mesh" in sim.__dict__:
            reasons.append("freeze_mesh(): mesh is frozen. Build a new Simulation to remesh")
        if reasons:
            raise ValueError(
                f"{caller}: cannot scale dx: {'; '.join(reasons)}. "
                "Use convergence_study with a factory that builds each rung, "
                "or declare an explicit uniform dx= without mesh pins/profiles "
                "and declare conductors in metres with add_thin_conductor."
            )

    variant = declaration_variant(sim)
    variant._declaration_root = variant
    for name, form in DECLARATIONS:
        value = variant.__dict__.get(name)
        if form == "value" and name not in mesh_names and isinstance(value, (dict, set, list)):
            variant.__dict__[name] = copy(value)
    for name in ("_realization_early_outcomes", "_amplitude_kind_warned",
                 "_boundary_default_warned", "_waveguide_call_overrides"):
        variant.__dict__.pop(name, None)

    if factor != 1.0 or pin_resolved_mesh:
        # Grid-building findings belong to this model, never the user's model.
        base_dx = float(variant._build_realized_grid().dx) if _base_dx is None else _base_dx
        if pin_resolved_mesh:
            mesh = variant._resolve_mesh()
            variant.__dict__.pop("_frozen_mesh", None)
            variant._dx = base_dx
            variant._domain = tuple(mesh["_domain"])
            for name in ("_dx_profile", "_dy_profile", "_dz_profile"):
                setattr(variant, name, mesh[name])
        else:
            variant._dx = factor * base_dx
            if sim._dt_pin is not None:
                variant._dt_pin = factor * sim._dt_pin
    release_declaration_caches(variant)
    return variant


def reset_device_declarations(variant):
    """Reset device contents to a bare Simulation, retaining mesh and numerics."""
    bare = type(variant)(freq_max=variant._freq_max, domain=tuple(variant._domain))
    for name, form in DECLARATIONS:
        if name not in DEVICE_DECLARATIONS:
            continue
        if form == "entries":
            setattr(variant, name, [])
        elif name in bare.__dict__:
            setattr(variant, name, bare.__dict__[name])
        else:
            variant.__dict__.pop(name, None)
    release_declaration_caches(variant)
