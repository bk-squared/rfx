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


def _fill_image_range(grid, shape, axis, *, closed_footprint=False, padding=0.):
    """Static image indices whose declared bounds meet the base period."""
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
    lo, hi = float(lo) - padding, float(hi) + padding
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
        # A footprint of zero width on a multiple of L is one rim, not none:
        # it keeps the image that puts it on node 0.
        first = math.floor(period_quotient(lo))
        return range(first, max(math.ceil(period_quotient(hi)), first + 1))
    return range(math.floor(float(lo) / length),
                 math.floor(float(hi) / length) + 1)


def periodic_mask(grid, shape, coords, *, sample=None, axes=None,
                  closed_footprint=False, axis_samplers=None):
    """Union a fill's images using its existing coordinate sampler.

    Separable axis masks are folded before their Cartesian product is built,
    retaining a thin Box's nearest sample on the complete image lattice.
    Composites retain that convention for each child. Other containment
    samplers stream and fold one image axis at a time.
    No array has the product of two image counts. Non-periodic calls retain
    their original sampler and coordinates.

    ``axis_samplers`` = three callables, one per axis, each mapping that axis's
    coordinate line to a 1-D mask, for a ``sample`` that is their outer
    product (a Box). The union over the image lattice of an outer product is
    the outer product of the per-axis unions, so the images are folded one
    axis at a time: the cost is the SUM of the image counts, not their
    product. A Box declared 2 m wide on two periodic axes of 0.75 mm took
    1602 s through the nested fold below (2667 x 2667 whole-grid samples).
    """
    from rfx.core.jax_utils import is_tracer
    import jax.numpy as jnp
    import numpy as np
    components = getattr(shape, '_mask_components', None) if sample is None else None
    separable = getattr(shape, '_axis_masks_on_coords', None) if sample is None else None
    if sample is None:
        sample = getattr(shape, 'mask_on_coords', None)
        if sample is None:
            raise NotImplementedError(
                f"periodic fill for {type(shape).__name__} requires "
                "mask_on_coords(x, y, z); implement that coordinate sampler "
                "instead of only mask(grid).")
    periodic = getattr(grid, 'periodic_axes', '')
    if not periodic:
        return sample(*coords)
    if components is not None:
        result = jnp.zeros(tuple(len(c) for c in coords), dtype=bool)
        for part in components(grid.cells(0)[0]):
            result = result | periodic_mask(
                grid, periodic_shape(grid, part), coords, axes=axes,
                closed_footprint=closed_footprint)
        return result
    images = []
    for a, line in enumerate(coords):
        if 'xyz'[a] in periodic and (axes is None or a in axes):
            ks = _fill_image_range(grid, shape, a, closed_footprint=closed_footprint)
        else:
            ks = (0,)
        images.append(ks)
    if axis_samplers is not None:
        masks = []
        for a, (line, ks) in enumerate(zip(coords, images)):
            folded = None
            for k in ks:
                part = axis_samplers[a](
                    line + k * float(grid.domain[a]) if k else line)
                folded = part if folded is None else folded | part
            masks.append(folded)
        return (masks[0][:, None, None] & masks[1][None, :, None]
                & masks[2][None, None, :])
    if separable is not None:
        expanded = []
        for a, (line, ks) in enumerate(zip(coords, images)):
            xp = jnp if is_tracer(line) else np
            expanded.append(xp.concatenate([line if k == 0 else line + k * float(grid.domain[a])
                                            for k in ks]))
        masks = [m.reshape(len(ks), len(line)).any(axis=0)
                 for m, ks, line in zip(separable(*expanded), images, coords)]
        return jnp.asarray(masks[0][:, None, None] & masks[1][None, :, None]
                           & masks[2][None, None, :])

    def fold(axis, lines):
        if axis == 3:
            return sample(*lines)
        result = None
        for k in images[axis]:
            shifted = list(lines)
            if k:
                shifted[axis] = coords[axis] + k * float(grid.domain[axis])
            part = fold(axis + 1, shifted)
            result = part if result is None else result | part
        return result
    return fold(0, coords)


def periodic_sdf(grid, shape, coords, sdf_fn, *, xp, normal_fn=None):
    """Minimum signed distance over fill images, with the winning normal.

    Images are reduced one axis at a time; only base-grid arrays are held.
    SDF weights sample a half-cell transition outside a body's bounds, so
    the shared image selector includes that support as well as its interior.
    The non-periodic call is exactly the original SDF/normal evaluation.
    """
    periodic = getattr(grid, 'periodic_axes', '')
    images = [_fill_image_range(grid, shape, a, padding=.5 * float(grid.cells(a)[0]))
              if 'xyz'[a] in periodic else (0,)
              for a in range(3)]

    def fold(axis, values):
        if axis == 3:
            distance = sdf_fn(*values, shape, xp=xp)
            normal = None if normal_fn is None else normal_fn(*values, shape, xp=xp)
            return distance, normal
        best = normal = None
        for k in images[axis]:
            shifted = list(values)
            if k:
                shifted[axis] = coords[axis] + k * float(grid.domain[axis])
            distance, candidate = fold(axis + 1, shifted)
            if best is None:
                best, normal = distance, candidate
            else:
                if normal_fn is not None:
                    normal = tuple(xp.where(distance < best, n, old)
                                   for n, old in zip(candidate, normal))
                best = xp.minimum(best, distance)
        return best, normal
    distance, normal = fold(0, coords)
    return distance if normal_fn is None else (distance, normal)
