"""Exact closed Box unions on the host table lattice, outside other bodies' reach."""

import itertools
import math

import numpy as np

from rfx._internal._periodic import periodic_sdf
from rfx.geometry.csg import Box


def _along(values, axis):
    shape = [1, 1, 1]
    shape[axis] = -1
    return np.asarray(values).reshape(shape)


def _images(grid, boxes, starts, stops, tol):
    images = []
    for box in boxes:
        ranges = []
        for axis in range(3):
            if 'xyz'[axis] in getattr(grid, 'periodic_axes', ''):
                period = float(grid.domain[axis])
                ranges.append(range(math.ceil((min(starts[axis]) - box.corner_hi[axis] - tol) / period),
                                    math.floor((max(stops[axis]) - box.corner_lo[axis] + tol) / period) + 1))
            else:
                ranges.append((0,))
        for shifts in itertools.product(*ranges):
            bounds = []
            for corner in (box.corner_lo, box.corner_hi):
                values = []
                for axis, value in enumerate(corner):
                    if 'xyz'[axis] in getattr(grid, 'periodic_axes', ''):
                        value += shifts[axis] * float(grid.domain[axis])
                    nodes = np.concatenate((starts[axis], stops[axis]))
                    nearest = nodes[np.argmin(np.abs(nodes - value))]
                    values.append(float(nearest if abs(nearest - value) <= tol else value))
                bounds.append(values)
            images.append(bounds)
    return images


def exact_box_overlap(grid, carried, lines, sample_lines, tol):
    """Return lengths, areas, and their replacement masks with both owner halos.

    Starts use wrapped sample coordinates, widths use consecutive unwrapped
    coordinates. Zero widths are points with closed containment. Partitioning
    at every Box boundary integrates each union tile once. A non-Box distance
    at any element node within its diagonal preserves the existing union rule.
    """
    starts = [line[:-1] for line in lines]
    widths = [np.diff(line) for line in sample_lines]
    stops = [start + width for start, width in zip(starts, widths)]
    shape = tuple(len(start) for start in starts)
    boxes = [body for body, _ in carried if isinstance(body, Box)]
    images = _images(grid, boxes, starts, stops, tol)
    parts = []
    indices = [np.arange(size) for size in shape]
    for axis in range(3):
        if np.all(widths[axis] == 0):
            parts.append(None)
            continue
        low, high = min(starts[axis]), max(stops[axis])
        cuts = sorted({low, high} | {corner[axis] for image in images for corner in image
                                    if low < corner[axis] < high})
        start_order = np.argsort(starts[axis], kind='stable')
        stop_order = np.argsort(stops[axis], kind='stable')
        tiles = []
        for lo, hi in zip(cuts, cuts[1:]):
            before = np.searchsorted(starts[axis][start_order], hi, side='left')
            after = np.searchsorted(stops[axis][stop_order], lo, side='right')
            selected = np.intersect1d(start_order[:before], stop_order[after:])
            weights = np.maximum(0., np.minimum(stops[axis][selected], hi)
                                 - np.maximum(starts[axis][selected], lo)) / widths[axis][selected]
            tiles.append(((lo + hi) / 2, selected, _along(weights, axis)))
        parts.append(tiles)

    def fraction(varying):
        varying = [axis for axis in varying if parts[axis] is not None]
        fixed = [axis for axis in range(3) if axis not in varying]
        metal = np.zeros(shape, dtype=np.float64)
        has_open = np.zeros(shape, dtype=bool)
        for tiles in itertools.product(*(parts[axis] for axis in varying)):
            selected = list(indices)
            for axis, tile in zip(varying, tiles):
                selected[axis] = tile[1]
            region = np.ix_(*selected)
            weight = math.prod(tile[2] for tile in tiles)
            occupied = False
            for lo, hi in images:
                if all(lo[axis] <= tile[0] <= hi[axis] for axis, tile in zip(varying, tiles)):
                    inside = True
                    for axis in fixed:
                        inside = inside & _along((lo[axis] <= starts[axis]) &
                                                 (starts[axis] <= hi[axis]), axis)
                    occupied = occupied | inside
            metal[region] += weight * occupied
            has_open[region] |= (weight > 0) & np.logical_not(occupied)
        return np.where(has_open, np.clip(1. - metal, 0., 1.), 0.)

    lengths = tuple(fraction((axis,)) for axis in range(3))
    areas = tuple(fraction([other for other in range(3) if other != axis]) for axis in range(3))
    edge_reach = [np.zeros(shape, bool) for _ in range(3)]
    face_reach = [np.zeros(shape, bool) for _ in range(3)]
    coords = np.meshgrid(*lines, indexing='ij', sparse=True)
    for body, fn in carried:
        if isinstance(body, Box):
            continue
        distance = np.broadcast_to(periodic_sdf(grid, body, coords, fn, xp=np),
                                   tuple(len(line) for line in lines))
        for axis in range(3):
            for varying, reaches in (((axis,), edge_reach),
                                     (tuple(a for a in range(3) if a != axis), face_reach)):
                diagonal = np.sqrt(sum(_along(widths[a], a) ** 2 for a in varying))
                for moves in itertools.product((0, 1), repeat=len(varying)):
                    corner = [slice(None, -1)] * 3
                    for a, moved in zip(varying, moves):
                        if moved:
                            corner[a] = slice(1, None)
                    reaches[axis] |= distance[tuple(corner)] <= diagonal + tol
    return lengths, areas, tuple(~reach for reach in edge_reach), tuple(~reach for reach in face_reach)
