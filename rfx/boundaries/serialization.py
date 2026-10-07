"""Preserve whether the caller declared a boundary when rebuilding a model."""
from rfx.boundaries.spec import Boundary, BoundarySpec, normalize_boundary


def constructor_boundary(sim):
    from rfx.boundaries.features import boundary_was_explicit
    from rfx.boundaries.model import DEFAULT_BOUNDARY
    return sim._boundary_spec if boundary_was_explicit(sim) else DEFAULT_BOUNDARY


def restore_constructor_default(plan, payload):
    """Old documents have no provenance: treat their recorded faces as explicit."""
    from rfx.boundaries.model import DEFAULT_BOUNDARY
    explicit = payload.get("explicit", True)
    if type(explicit) is not bool:
        raise ValueError("boundary.explicit must be a boolean")
    if explicit:
        return plan
    if normalize_boundary(plan.kwargs["boundary"]) != normalize_boundary("cpml"):
        raise ValueError("An omitted boundary must carry the default cpml declaration")
    return plan._replace(kwargs=dict(plan.kwargs, boundary=DEFAULT_BOUNDARY))


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
