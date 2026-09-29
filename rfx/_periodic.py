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
    """Identify Box planes and validate directed filaments; fills may cross."""
    if not getattr(grid, "periodic_axes", ""):
        return shape
    from dataclasses import replace
    from rfx.geometry.csg import Box
    from rfx.core.jax_utils import is_tracer
    if isinstance(shape, Box):
        lo, hi = list(shape.corner_lo), list(shape.corner_hi)
        for a in range(3):
            if not (is_tracer(lo[a]) or is_tracer(hi[a])) and lo[a] == hi[a]:
                lo[a] = hi[a] = plane_coordinate(grid, a, lo[a])
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
    return shape


def _fill_image_range(grid, shape, axis, *, closed_footprint=False):
    """Static images, including the previous image of a closed endpoint L."""
    from rfx.core.jax_utils import is_tracer
    from rfx.geometry.csg import Box, declared_bounds
    bounds = declared_bounds(shape)
    if bounds is None:
        # Custom coordinate samplers without bounds cannot be validated.
        # Built-in shapes (including MeshShape) all expose bounding_box.
        return (-1, 0, 1)
    lo, hi = bounds[0][axis], bounds[1][axis]
    length = float(grid.domain[axis])
    if is_tracer(lo) or is_tracer(hi):
        import jax
        import numpy as np
        def validate(a, b):
            if isinstance(shape, Box) and a == b:
                raise ValueError(
                    f"periodic axis {'xyz'[axis]!r}: a zero-thickness Box is a "
                    f"plane and needs a concrete coordinate; bounds [{a}, {b}] m, L={length} m")
            dtype = np.result_type(a.dtype, b.dtype)
            tol = 8 * np.finfo(dtype).eps * length if np.issubdtype(dtype, np.inexact) else 0.
            if not np.isfinite(a + b) or a < -length - tol or b > 2 * length + tol:
                raise ValueError(
                    f"periodic axis {'xyz'[axis]!r}: fill bounds [{a}, {b}] m "
                    f"must lie inside [-L, 2L] for traced bounds; L={length} m")
        jax.debug.callback(validate, lo, hi, ordered=True)
        return (-1, 0, 1)
    if closed_footprint:
        # Keep the two sheet rims distinct until E edges are enumerated.
        # An image touching [0,L] at only one rim contributes that endpoint
        # through the other rim's later fold, not an extra edge into a gap.
        def period_quotient(x):
            q = float(x) / length
            nearest = round(q)
            # Preserve endpoint expression roundoff, including a bound one
            # float step outside [0,L]; it must not add an exterior image.
            return nearest if abs(q - nearest) <= 8 * math.ulp(max(1., abs(q))) else q
        return range(math.floor(period_quotient(lo)),
                     math.ceil(period_quotient(hi)))
    return range(math.floor(float(lo) / length) - 1,
                 math.floor(float(hi) / length) + 1)


def periodic_mask(grid, shape, coords, *, sample=None, axes=None,
                  closed_footprint=False):
    """Union a fill's images using its existing coordinate sampler.

    Image coordinates are evaluated together, then OR-folded. In particular,
    a thin Box's nearest-sample convention must choose on the whole image
    lattice, not snap to an unrelated endpoint once per image. Node, centre,
    half-open volume and closed sheet-footprint samplers stay with callers.
    Non-periodic coordinate arrays pass through unchanged.
    """
    from rfx.core.jax_utils import is_tracer
    import jax.numpy as jnp
    import numpy as np
    sample = shape.mask_on_coords if sample is None else sample
    periodic = getattr(grid, 'periodic_axes', '')
    if not periodic:
        return sample(*coords)
    images = []
    expanded = []
    for a, line in enumerate(coords):
        if 'xyz'[a] in periodic and (axes is None or a in axes):
            ks = _fill_image_range(grid, shape, a, closed_footprint=closed_footprint)
            xp = jnp if is_tracer(line) else np
            expanded.append(xp.concatenate([line + k * float(grid.domain[a]) for k in ks]))
            images.append(len(ks))
        else:
            expanded.append(line)
            images.append(1)
    mask = sample(*expanded)
    for a, count in enumerate(images):
        if count != 1:
            shape_out = list(mask.shape)
            shape_out[a:a + 1] = [count, len(coords[a])]
            mask = mask.reshape(shape_out).any(axis=a)
    return mask
