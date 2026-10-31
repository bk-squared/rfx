"""Admission of declared conductors when mesh nodes cannot be read on the host."""

from rfx._traced_refusal import TracedStructureRefusal
from rfx.core.jax_utils import is_tracer
from rfx.geometry.csg import declared_bounds


def refuse_traced_conductor_at_absorber(sim, grid, shape, traced_axes):
    """Decide reach from concrete declarations and pad counts, never node values."""
    from rfx.geometry.smoothing import _reaches_pad_face

    bounds = declared_bounds(shape)
    if bounds is None:
        return
    faces = []
    for axis, name in enumerate("xyz"):
        if any(is_tracer(bound[axis]) for bound in bounds):
            continue
        for side in ("lo", "hi"):
            if getattr(grid, f"pad_{name}_{side}") <= 0:
                continue
            # No occupied lattice or rounded node face is available. Reuse only
            # the declared-domain test and its ULP slack, with nominal cell size.
            if _reaches_pad_face(bounds, axis, side, 0., sim._dx,
                                 sim._unresolved_domain, occupied_faces=()):
                faces.append(f"{name}-{side}")
    if faces:
        axes = ", ".join("xyz"[axis] for axis in traced_axes)
        profiles = ", ".join(f"d{'xyz'[axis]}_profile" for axis in traced_axes)
        raise TracedStructureRefusal(
            f"Cannot continue a conductor into absorbing face(s) {', '.join(faces)} "
            f"while mesh axis {axes} is traced: the run would solve a ground cut "
            f"at the absorber entrance. Pass that axis's profile ({profiles}) "
            "as a concrete NumPy array for this run.")
