"""Preserve whether the caller declared a boundary when rebuilding a model."""
from rfx.boundaries.spec import Boundary, BoundarySpec


def constructor_boundary(sim):
    from rfx.boundaries.features import boundary_was_explicit
    from rfx.boundaries.model import DEFAULT_BOUNDARY
    if sim._boundary_explicit is None:
        return resolved_guide_spec(sim)
    return sim._boundary_spec if boundary_was_explicit(sim) else DEFAULT_BOUNDARY


def resolved_guide_spec(sim):
    """Resolve only main's zero-pad full-guide faces, preserving port depths."""
    if sim._uses_nonuniform_mesh or not sim._waveguide_ports:
        return sim._boundary_spec
    from rfx.boundaries.features import guide_faces
    payload = sim._boundary_spec.to_dict()
    for face in guide_faces(sim, sim._build_grid()):
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


def legacy_spec(sim):
    """The constructor and old-design importer share one scalar face rule."""
    return predict_legacy_spec(sim._boundary, sim._pec_faces, sim._periodic_axes)
