"""NumPy formulas for the B-mat value rows; no rfx numerical helpers.

Signed distances and midpoint samples come from the cell-table build. Zero is
on the free side for marching squares. A sign-changing segment intersects at
s0/(s0-s1). Cyclic positive corners and intersections delimit the free polygon.
For diagonal positive corners (cases 5 and 10), a strictly positive arithmetic
mean of the four distances connects the polygon; otherwise sum the two corner
triangles, including a zero centre. Areas use the shoelace formula, normalized
by the face's two physical widths, and are clipped to [0, 1].

The cell-table boundary convention precedes that polygon rule: when all face
corners are <= tol and at least one is within tol of zero, the sampled face
midpoint decides (distance > tol). For an edge the analogous two-endpoint rule
uses its midpoint. tol is the table's 1e-6 times its least cell width. Thus zero
endpoints, coincident surfaces, and saddle ties have explicit reference rules.
The SC edge bound is twice the least incident face area; a missing low face
has area one. Raw fractions are checked separately from the binding clamp.

Carried Boxes use independent interval and rectangle unions after coordinates
within tol of a sample node are placed on that node. An element reached by
another body retains the node-distance rule. Reach uses the individual
Sphere or capped Cylinder distance at the element nodes, including periodic
images, within the element diagonal plus tol. Other carried types are refused.
"""
import itertools
import math

import numpy as np

from rfx.geometry.csg import Box, Cylinder, Sphere
from tests._box_overlap_reference import _along, _hand_arrays


def assert_spacings(actual, expected, count):
    """B4: integer float64 spacing distance, with count*2**-53 absolute floor.

    Signed error is actual minus reference; both read the same scene geometry,
    widths and declarations. These material quantities are nonnegative.
    """
    actual, expected = np.asarray(actual), np.asarray(expected, dtype=np.float64)
    assert actual.dtype == np.dtype('float64'), actual.dtype
    assert actual.shape == expected.shape
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
    assert np.all(actual >= 0) and np.all(expected >= 0)
    a, b = actual.view(np.uint64), expected.view(np.uint64)
    distance = np.maximum(a, b) - np.minimum(a, b)
    error = actual - expected
    passed = (distance <= count) | ((expected < 1) & (np.abs(error) <= count * 2.**-53))
    assert np.all(passed), (f'{np.count_nonzero(~passed)} values above {count} u; '
                            f'max spacing distance={distance.max()}, '
                            f'signed error range=[{error.min()}, {error.max()}]')


def edge_reference(cells, grid, axis):
    """Four explicit cell areas, with replicated exterior material cells."""
    cells = np.asarray(cells, dtype=np.float64)
    transverse = [a for a in range(3) if a != axis]
    axes = []
    for a in transverse:
        w = np.asarray(grid.cells(a), dtype=np.float64)
        if len(w) == cells.shape[a] - 1:
            w = np.r_[w, w[-1]]
        assert len(w) == cells.shape[a]
        axes.append(w)
    numerator = np.zeros(cells.shape, dtype=np.float64)
    denominator = np.zeros(cells.shape, dtype=np.float64)
    for i in (0, 1):
        for j in (0, 1):
            q = cells
            area = np.ones([1, 1, 1], dtype=np.float64)
            for a, w, back in zip(transverse, axes, (i, j)):
                index = np.maximum(np.arange(cells.shape[a]) - back, 0)
                q = np.take(q, index, axis=a)
                shape = [1, 1, 1]
                shape[a] = len(w)
                area = area * w[index].reshape(shape)
            numerator += area * q
            denominator += area
    return numerator / denominator


def face_reference(corners, widths):
    """Cyclic polygon integration with the saddle/zero rules above."""
    d = np.stack(corners, axis=-1)
    result = np.empty(d.shape[:-1], dtype=np.float64)
    signs = d >= 0
    codes = np.sum(signs * np.array([1, 2, 4, 8]), axis=-1)
    result[codes == 0], result[codes == 15] = 0., 1.
    u, v = widths
    xy = np.array([[0., 0.], [u, 0.], [u, v], [0., v]])

    def integral(vertices):
        x = np.stack(vertices, axis=1)
        return np.abs(np.sum(x[:, :, 0] * np.roll(x[:, :, 1], -1, axis=1)
                             - x[:, :, 1] * np.roll(x[:, :, 0], -1, axis=1), axis=1)) / (2*u*v)

    for code in range(1, 15):
        selected = codes == code
        values = d[selected]
        if not len(values):
            continue
        inside = [(code >> k) & 1 for k in range(4)]
        vertices = [np.broadcast_to(p, (len(values), 2)) for p in xy]
        crossing = {}
        boundary = []
        for k in range(4):
            nxt = (k+1) % 4
            if inside[k]:
                boundary.append(vertices[k])
            if inside[k] != inside[nxt]:
                ratio = values[:, k] / (values[:, k] - values[:, nxt])
                crossing[k] = vertices[k] + ratio[:, None] * (xy[nxt]-xy[k])
                boundary.append(crossing[k])
        connected = integral(boundary)
        if code in (5, 10):
            separate = np.zeros(len(values))
            for k in range(4):
                if inside[k]:
                    separate += integral([vertices[k], crossing[k], crossing[(k-1) % 4]])
            connected = np.where(np.sum(values, axis=1) > 0, connected, separate)
        result[selected] = connected
    return np.clip(result, 0., 1.)


def _body_distance(shape, grid, lines):
    """Own Sphere or capped Cylinder distance, including the periodic images."""
    if not isinstance(shape, (Sphere, Cylinder)):
        raise TypeError(f'conformal reference has no distance for {type(shape).__name__}')
    center = np.asarray(shape.center, dtype=np.float64)
    half = np.full(3, shape.radius, dtype=np.float64)
    if isinstance(shape, Cylinder):
        axial = 'xyz'.index(shape.axis)
        half[axial] = shape.height / 2
    padding = max(float(np.max(grid.cells(a))) for a in range(3))
    shifts = [range(math.floor((center[a] - half[a] - padding) / float(grid.domain[a])),
                    math.floor((center[a] + half[a] + padding) / float(grid.domain[a])) + 1)
              if 'xyz'[a] in grid.periodic_axes else (0,) for a in range(3)]
    distance = np.full(tuple(len(line) for line in lines), np.inf)
    for image in itertools.product(*shifts):
        delta = [_along(line + k * float(grid.domain[a]) - center[a], a)
                 for a, (line, k) in enumerate(zip(lines, image))]
        if isinstance(shape, Sphere):
            own = np.sqrt(sum(d ** 2 for d in delta)) - shape.radius
        else:
            radial = np.sqrt(sum(delta[a] ** 2 for a in range(3) if a != axial)) - shape.radius
            cap = np.abs(delta[axial]) - shape.height / 2
            own = (np.sqrt(np.maximum(radial, 0.) ** 2 + np.maximum(cap, 0.) ** 2)
                   + np.minimum(np.maximum(radial, cap), 0.))
        distance = np.minimum(distance, own)
    return distance


def _box_reference(samples, lengths, areas):
    """Exact closed Box overlap outside each other body's diagonal reach."""
    carried = samples.get('carried', ())
    boxes = [shape for shape in carried if isinstance(shape, Box)]
    others = [shape for shape in carried if not isinstance(shape, Box)]
    for shape in others:
        if not isinstance(shape, (Sphere, Cylinder)):
            raise TypeError(f'conformal reference has no distance for {type(shape).__name__}')
    if not boxes:
        return tuple(lengths), tuple(areas)
    grid, lines = samples['grid'], samples['lines']
    widths = [np.diff(line) for line in samples['sample_lines']]
    exact = _hand_arrays(grid, boxes, lines=lines, sample_lines=samples['sample_lines'],
                         tol=samples['tol'])
    distances = [_body_distance(shape, grid, lines) for shape in others]
    result = []
    for kind, old, hand in zip(('edge', 'face'), (lengths, areas), exact):
        values = []
        for axis in range(3):
            varying = (axis,) if kind == 'edge' else tuple(a for a in range(3) if a != axis)
            diagonal = np.sqrt(sum(_along(widths[a], a) ** 2 for a in varying))
            reach = np.zeros_like(old[axis], dtype=bool)
            for distance in distances:
                for moves in itertools.product((0, 1), repeat=len(varying)):
                    corner = [slice(None, -1)] * 3
                    for a, moved in zip(varying, moves):
                        if moved:
                            corner[a] = slice(1, None)
                    reach |= distance[tuple(corner)] <= diagonal + samples['tol']
            values.append(np.where(reach, old[axis], hand[axis]))
        result.append(tuple(values))
    return tuple(result)


def conformal_reference(samples):
    sdf, widths, tol = samples['sdf'], samples['widths'], samples['tol']
    base = sdf[:-1, :-1, :-1]
    lengths, areas = [], []
    for axis in range(3):
        end = [slice(None, -1)] * 3
        end[axis] = slice(1, None)
        other = sdf[tuple(end)]
        crossing = (base < 0) & (other > 0) | (other < 0) & (base > 0)
        length = np.zeros_like(base)
        np.divide(np.maximum(base, other), np.abs(other-base), out=length, where=crossing)
        length[(base >= 0) & (other >= 0)] = 1.
        touching = (base <= tol) & (other <= tol) & ((np.abs(base) <= tol) | (np.abs(other) <= tol))
        length = np.where(touching, samples['midpoints'][(axis,)] > tol, length)
        lengths.append(length)
        u, v = [a for a in range(3) if a != axis]
        corners = [base]
        for shift in ((u,), (u, v), (v,)):
            index = [slice(None, -1)] * 3
            for a in shift:
                index[a] = slice(1, None)
            corners.append(sdf[tuple(index)])
        distances = np.stack(corners)
        touching = np.all(distances <= tol, axis=0) & np.any(np.abs(distances) <= tol, axis=0)
        area = face_reference(corners, (widths[u], widths[v]))
        areas.append(np.where(touching, samples['midpoints'][(u, v)] > tol, area))
    return _box_reference(samples, lengths, areas)


def clamp_reference(lengths, areas):
    result = []
    for axis, length in enumerate(lengths):
        incident = []
        for normal in range(3):
            if normal == axis:
                continue
            direction = 3 - axis - normal
            previous = np.ones_like(length)
            dst, src = [slice(None)]*3, [slice(None)]*3
            dst[direction], src[direction] = slice(1, None), slice(None, -1)
            previous[tuple(dst)] = areas[normal][tuple(src)]
            incident.extend((areas[normal], previous))
        result.append(np.minimum(length, 2*np.minimum.reduce(incident)))
    return tuple(result)
