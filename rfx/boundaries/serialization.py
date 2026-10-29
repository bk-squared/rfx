"""Preserve whether the caller declared a boundary when rebuilding a model."""
from rfx.boundaries.spec import Boundary, BoundarySpec


def constructor_boundary(sim):
    from rfx.boundaries.features import boundary_was_explicit
    from rfx.boundaries.model import DEFAULT_BOUNDARY, refresh_boundary_model
    refresh_boundary_model(sim)
    if sim._boundary_explicit is None:
        return resolved_guide_spec(sim)
    # A Floquet port and an open TF/SF source replace the shorthand's lateral
    # faces; an explicit spec would declare them (#1944, #1945).
    open_tfsf = sim._tfsf is not None and not sim._tfsf.closed_box
    if (sim._floquet_ports or open_tfsf) and sim._boundary_spec is scalar_spec(sim):
        return str(sim._boundary) if boundary_was_explicit(sim) else DEFAULT_BOUNDARY
    return sim._boundary_spec if boundary_was_explicit(sim) else DEFAULT_BOUNDARY


def resolved_guide_spec(sim):
    """Resolve only main's zero-pad full-guide faces, preserving port depths."""
    if sim._uses_nonuniform_mesh or not sim._waveguide_ports:
        return sim._boundary_spec
    payload = sim._boundary_spec.to_dict()
    # The admission record retains the former full-guide faces even when an
    # explicit absorber now allocates pads. Old documents lack provenance;
    # importing them must still recognize the same implicit guide default.
    for face in sim._build_grid()._waveguide_admission[0]:
        axis, side = face.split("_")
        if payload[axis][side] in ("cpml", "upml"):
            payload[axis][side] = "pec"
            payload[axis].pop(side + "_thickness", None)
    return BoundarySpec.from_dict(payload)


def export_spec(sim):
    return resolved_guide_spec(sim) if sim._boundary_explicit is False else sim._boundary_spec


def restore_legacy_boundary(sim):
    """Main's documents cannot say whether cpml was explicitly passed.

    None records that unknown provenance internally, without a format field.
    Keep the historical document on re-export, and its zero-depth walls on
    execution. Rebuilds receive the equivalent resolved PEC declaration.
    """
    if resolved_guide_spec(sim) != sim._boundary_spec:
        sim._boundary_explicit = None
    return sim


def predict_legacy_spec(boundary, pec_faces, periodic_axes):
    """Reproduce the legacy scalar/PEC/periodic constructor declaration."""
    axes = {}
    for axis in "xyz":
        if axis in periodic_axes:
            axes[axis] = Boundary(lo="periodic", hi="periodic")
        else:
            lo = "pec" if f"{axis}_lo" in pec_faces else boundary
            hi = "pec" if f"{axis}_hi" in pec_faces else boundary
            axes[axis] = Boundary(lo=lo, hi=hi)
    return BoundarySpec(x=axes["x"], y=axes["y"], z=axes["z"])


def design_constructor_boundary(spec, scalar, pec_faces, *, keeps_shorthand):
    """Preserve absorber shorthand for Floquet and open TF/SF design documents.

    Both replace the shorthand's lateral faces (Floquet registration adds
    periodic axes later). Rebuilding this scalar as an explicit spec would
    invent a declaration of absorbing lateral faces.
    """
    if (keeps_shorthand and scalar in ("cpml", "upml") and not pec_faces
            and predict_legacy_spec(scalar, pec_faces, "") == spec):
        return scalar
    return spec


def legacy_spec(sim):
    """Expand the constructor's scalar and remember its original face spec.

    ``sim._boundary`` stays a plain ``str``: writers such as h5py refuse a
    ``str`` subclass (#2214). The expansion is kept beside it, keyed to the
    scalar it came from; deepcopy keeps both and their shared spec reference.
    """
    sim._boundary = str(sim._boundary)
    spec = predict_legacy_spec(sim._boundary, sim._pec_faces, sim._periodic_axes)
    sim._boundary_scalar_spec = (sim._boundary, spec) if sim._boundary in ("cpml", "upml") else None
    return spec


def scalar_spec(sim):
    """The face spec the current absorber scalar expanded to, or None.

    None once ``sim._boundary`` no longer holds the scalar that was expanded.
    """
    entry = getattr(sim, "_boundary_scalar_spec", None)
    return entry[1] if entry is not None and entry[0] == sim._boundary else None
