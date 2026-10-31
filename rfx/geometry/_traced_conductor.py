"""Admission of declared conductors when mesh nodes cannot be read on the host."""

import numpy as np

from rfx._traced_refusal import TracedStructureRefusal
from rfx.core.jax_utils import is_tracer
from rfx.geometry.csg import declared_bounds


def refuse_traced_conductor_at_absorber(sim, grid, shape):
    """Decide reach from concrete declarations and pad counts, never node values."""
    from rfx.geometry.smoothing import _reaches_pad_face

    bounds = declared_bounds(shape)
    if bounds is None:
        return
    faces = []
    nominal = sim._nominal_profiles or {}
    for axis, name in enumerate("xyz"):
        if any(is_tracer(bound[axis]) for bound in bounds):
            continue
        cell = float(np.max(nominal[name])) if name in nominal else sim._dx
        for side in ("lo", "hi"):
            if getattr(grid, f"pad_{name}_{side}") <= 0:
                continue
            # Conservatively cover the occupied outermost interior layer using
            # declarations alone, including the largest nominal cell on this axis.
            within_cell = (float(bounds[0][axis]) <= cell if side == "lo" else
                           float(bounds[1][axis]) >= sim._unresolved_domain[axis] - cell)
            if within_cell or _reaches_pad_face(
                    bounds, axis, side, 0., cell, sim._unresolved_domain, occupied_faces=()):
                faces.append(f"{name}-{side}")
    if faces:
        traced = [name for name in "xyz" if is_tracer(getattr(sim, f"_d{name}_profile"))]
        axes = ", ".join(traced)
        profiles = ", ".join(f"d{name}_profile" for name in traced)
        raise TracedStructureRefusal(
            "Cannot continue a conductor that reaches or ends within one cell of "
            f"absorbing face(s) {', '.join(faces)} "
            f"while mesh axis {axes} is traced: the run would solve a ground cut "
            f"at the absorber entrance. Pass that axis's profile ({profiles}) "
            "as a concrete NumPy array for this run.")
