"""Host coordinate rules for an explicitly declared uniform periodic axis."""
from __future__ import annotations

import math


def interval_coordinates(grid, axis, start, end):
    """Keep interval endpoints unwrapped; refuse an unsupported seam crossing."""
    if "xyz"[axis] not in getattr(grid, "periodic_axes", ""):
        return start, end
    from rfx.core.jax_utils import is_tracer
    if is_tracer(start) or is_tracer(end):
        # Shape bounds may be design variables. Validate their runtime
        # values without converting a tracer or changing its derivatives.
        import jax
        def validate(a, b):
            interval_coordinates(grid, axis, float(a), float(b))
        jax.debug.callback(validate, start, end, ordered=True)
        return start, end
    length = float(grid.domain[axis])
    # Permit only expression roundoff at a declared endpoint, not a cell fraction.
    tol = 8 * math.ulp(length)
    lo, hi = sorted((float(start), float(end)))
    if not math.isfinite(lo + hi) or lo < -tol or hi > length + tol:
        raise ValueError(
            f"periodic axis {'xyz'[axis]!r}: interval [{start}, {end}] m "
            f"crosses the seam of period L={length} m; split the interval "
            "into declarations inside [0, L].")
    return tuple(0.0 if -tol <= x < 0 else length if length < x <= length + tol
                 else x for x in (start, end))


def plane_coordinate(grid, axis, coordinate):
    """Identify the seam plane before a nearest-plane distance check."""
    if "xyz"[axis] not in getattr(grid, "periodic_axes", ""):
        return coordinate
    grid.index_of(axis, coordinate)  # retain the point API's out-of-grid refusal
    length = float(grid.domain[axis])
    return coordinate - length if coordinate > length - float(grid.cells(axis)[-1]) / 2 else coordinate


def interval_indices(grid, start, end):
    """Two corner indices; constant axes are planes, nonzero spans are intervals."""
    if hasattr(grid, "interval_to_indices"):
        return grid.interval_to_indices(start, end)
    from rfx.nonuniform import position_to_index
    return position_to_index(grid, start), position_to_index(grid, end)


def periodic_shape(grid, shape):
    """Validate a volume's extent and identify a zero-thickness Box's seam plane."""
    if not getattr(grid, "periodic_axes", ""):
        return shape
    from dataclasses import replace
    from rfx.geometry.csg import Box, declared_bounds
    if isinstance(shape, Box):
        lo, hi = list(shape.corner_lo), list(shape.corner_hi)
        for a in range(3):
            if lo[a] == hi[a]:
                lo[a] = hi[a] = plane_coordinate(grid, a, lo[a])
            else:
                lo[a], hi[a] = interval_coordinates(grid, a, lo[a], hi[a])
        return replace(shape, corner_lo=tuple(lo), corner_hi=tuple(hi))
    # A filament's vertices bound its edges; its radius does not bound a volume.
    points = getattr(shape, "points", None)
    if points is not None and float(shape.radius) < min(float(grid.cells(a)[0]) for a in range(3)) / 2:
        for p, q in zip(points, points[1:]):
            for a in range(3):
                if p[a] == q[a]:
                    plane_coordinate(grid, a, p[a])
                else:
                    interval_coordinates(grid, a, p[a], q[a])
        return shape
    bounds = declared_bounds(shape)
    if bounds is not None:
        lo, hi = bounds
        for a in range(3):
            interval_coordinates(grid, a, lo[a], hi[a])
    return shape
