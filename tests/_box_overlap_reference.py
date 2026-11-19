"""Hand interval and rectangle unions on the conformal sample lattice."""

import itertools

import numpy as np


def _along(values, axis):
    shape = [1, 1, 1]
    shape[axis] = -1
    return np.asarray(values).reshape(shape)


def _lattice(grid):
    starts, widths = [], []
    for axis in range(3):
        cells = np.asarray(grid.cells(axis), dtype=float)
        nodes = np.array([grid.node_of(axis, i) for i in range(grid.shape[axis])], dtype=float)
        periodic = 'xyz'[axis] in grid.periodic_axes
        end = cells[-1 if periodic or len(cells) == 1 else -2]
        line = np.r_[nodes[0] - cells[0], nodes, nodes[-1] + end, nodes[-1] + 2 * end]
        if getattr(grid, 'is_2d', False) and axis == 2:
            line[:] = nodes[0]
        elif periodic and not np.all(cells == cells[0]):
            line[0] = nodes[0] - end
            line[-1] = nodes[-1] + end + cells[0]
        widths.append(np.diff(line))
        if periodic:
            line = line % float(grid.domain[axis])
            if not np.all(cells == cells[0]):
                line[0] = line[-3]
                line[-2:] = line[1:3]
        starts.append(line[:-1])
    return starts, widths


def _hand_arrays(grid, boxes, *, lines=None, sample_lines=None, tol=None):
    """Inclusion-exclusion of clipped intervals and rectangles, independently of SDFs."""
    if lines is None:
        starts, widths = _lattice(grid)
    else:
        starts = [line[:-1] for line in lines]
        widths = [np.diff(line) for line in sample_lines]
    stops = [a + b for a, b in zip(starts, widths)]
    shape = tuple(len(a) for a in starts)
    images = []
    if tol is None:
        tol = 1e-6 * min(float(np.min(grid.cells(a))) for a in range(3))
    for box in boxes:
        for shifts in itertools.product(*(range(-2, 3) if a in grid.periodic_axes else (0,) for a in 'xyz')):
            bounds = np.array([box.corner_lo, box.corner_hi], dtype=float)
            for axis in range(3):
                if 'xyz'[axis] in grid.periodic_axes:
                    bounds[:, axis] += shifts[axis] * float(grid.domain[axis])
                nodes = np.r_[starts[axis], stops[axis]]
                for side in range(2):
                    index = np.argmin(abs(nodes - bounds[side, axis]))
                    if abs(nodes[index] - bounds[side, axis]) <= tol:
                        bounds[side, axis] = nodes[index]
            if all(bounds[1, a] >= min(starts[a]) and bounds[0, a] <= max(stops[a]) for a in range(3)):
                images.append(bounds)

    def fraction(varying):
        varying = [a for a in varying if np.any(widths[a] > 0)]
        covered = np.zeros(shape)
        for count in range(1, len(images) + 1):
            for group in itertools.combinations(images, count):
                lo = np.max([b[0] for b in group], axis=0)
                hi = np.min([b[1] for b in group], axis=0)
                if np.any(hi < lo):
                    continue
                weight = 1.
                for a in range(3):
                    if a in varying:
                        factor = np.maximum(0., np.minimum(stops[a], hi[a]) - np.maximum(starts[a], lo[a])) / widths[a]
                    else:
                        factor = (lo[a] <= starts[a]) & (starts[a] <= hi[a])
                    weight = weight * _along(factor, a)
                covered += (-1.) ** (count + 1) * weight
        return np.clip(1. - covered, 0., 1.)

    return (tuple(fraction([a]) for a in range(3)),
            tuple(fraction([b for b in range(3) if b != a]) for a in range(3)))
