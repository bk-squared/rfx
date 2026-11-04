"""Exact series/parallel dual-cell integrals for Box PEC normal components."""

from collections import OrderedDict
from itertools import product

import numpy as np

from rfx.geometry.csg import Box, Cylinder, Sphere, declared_bounds


def dielectric_groups(shapes):
    """The dielectric smoother's first-occurrence material precedence."""
    groups = OrderedDict()
    for shape, eps_r in shapes:
        groups.setdefault(eps_r, []).append(shape)
    return groups


def _images(grid, shape):
    from rfx._internal._periodic import _fill_image_range
    return product(*[
        _fill_image_range(grid, shape, a, padding=grid.dx / 2)
        if 'xyz'[a] in getattr(grid, 'periodic_axes', '') else (0,)
        for a in range(3)])


def _touches(xp, shape, points, half):
    """Reject non-Box material regions meeting the closed dual cell."""
    if isinstance(shape, (Sphere, Cylinder)):
        gaps = [xp.maximum(xp.abs(p - c) - half, 0.)
                for p, c in zip(points, shape.center)]
        if isinstance(shape, Sphere):
            return sum(g * g for g in gaps) <= shape.radius**2
        axis = 'xyz'.index(shape.axis)
        return ((gaps[axis] <= shape.height / 2)
                & (sum(gaps[a]**2 for a in range(3) if a != axis) <= shape.radius**2))
    bounds = declared_bounds(shape)
    if bounds is None:
        return xp.ones_like(points[0], dtype=bool)
    return ((points[0] + half >= bounds[0][0]) & (points[0] - half <= bounds[1][0])
            & (points[1] + half >= bounds[0][1]) & (points[1] - half <= bounds[1][1])
            & (points[2] + half >= bounds[0][2]) & (points[2] - half <= bounds[1][2]))


def normal_box_mask(xp, grid, pec, dielectrics, coords, f, normal):
    """Select fractional normal samples with a certified all-Box dual cell."""
    if not isinstance(pec, Box):
        return None
    mask = (xp.abs(normal) >= 1 - 1e-6) & (f > 0) & (f < 1)
    for shape, _ in dielectrics:
        if isinstance(shape, Box):
            continue
        for image in _images(grid, shape):
            points = tuple(c + k * grid.domain[a] for a, (c, k) in enumerate(zip(coords, image)))
            mask = mask & ~_touches(xp, shape, points, grid.dx / 2)
    return mask


def _box_images(xp, grid, shape):
    return [(xp.asarray(shape.corner_lo) - xp.asarray(image) * xp.asarray(grid.domain),
             xp.asarray(shape.corner_hi) - xp.asarray(image) * xp.asarray(grid.domain))
            for image in _images(grid, shape)]


def exact_box_inverse(xp, grid, pec, dielectrics, background, coords, axis):
    """Integrate constant material intervals, then invert the parallel mean.

    Interval midpoints only identify their constant material; their exact
    lengths are the integration weights. There is no quadrature sampling.
    """
    metal = _box_images(xp, grid, pec)
    groups = [(eps, [bounds for shape in shapes if isinstance(shape, Box)
                     for bounds in _box_images(xp, grid, shape)])
              for eps, shapes in dielectric_groups(dielectrics).items()]
    boxes = metal + [bounds for _, group in groups for bounds in group]
    points = xp.stack(coords, axis=-1)
    low, high = points - grid.dx / 2, points + grid.dx / 2
    edges = xp.sort(xp.stack([low, high] + [xp.clip(b, low, high)
                                          for bounds in boxes for b in bounds], axis=-2), axis=-2)
    mids = (edges[..., 1:, :] + edges[..., :-1, :]) / 2
    widths = (edges[..., 1:, :] - edges[..., :-1, :]) / grid.dx
    order = [axis] + [a for a in range(3) if a != axis]
    positions = [mids[..., :, a].reshape(mids.shape[:-2] + tuple(
        mids.shape[-2] if j == k else 1 for j in range(3))) for k, a in enumerate(order)]
    weights = [widths[..., :, a].reshape(widths.shape[:-2] + tuple(
        widths.shape[-2] if j == k else 1 for j in range(3))) for k, a in enumerate(order)]

    def inside(bounds):
        lo, hi = bounds
        return ((positions[0] >= lo[order[0]]) & (positions[0] < hi[order[0]])
                & (positions[1] >= lo[order[1]]) & (positions[1] < hi[order[1]])
                & (positions[2] >= lo[order[2]]) & (positions[2] < hi[order[2]]))

    inverse = 1. / background
    for eps, group in groups:
        for bounds in group:
            inverse = xp.where(inside(bounds), 1. / eps, inverse)
    for bounds in metal:
        inverse = xp.where(inside(bounds), 0., inverse)
    series = xp.sum(inverse * weights[0], axis=-3)
    area = (weights[1] * weights[2])[..., 0, :, :]
    short = xp.any((series == 0) & (area > 0), axis=(-2, -1))
    parallel = xp.sum(area / xp.where(series > 0, series, 1.), axis=(-2, -1))
    return xp.where(short, 0., 1. / xp.where(parallel > 0, parallel, 1.))


def replace_normal_boxes(xp, grid, pec, dielectrics, background, coords, axis, mask, legacy):
    """Bound concrete work by fractional samples, retaining traced arithmetic."""
    if mask is None:
        return legacy
    points = xp.broadcast_arrays(*coords)
    if xp is np:
        indices = np.flatnonzero(mask)
        result = legacy.copy()
        for start in range(0, len(indices), 256):
            take = indices[start:start + 256]
            values = exact_box_inverse(xp, grid, pec, dielectrics, background,
                                       tuple(p.ravel()[take] for p in points), axis)
            result.ravel()[take] = values
        return result
    values = exact_box_inverse(xp, grid, pec, dielectrics, background, points, axis)
    return xp.where(mask, values, legacy)
