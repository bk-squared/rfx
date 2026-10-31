"""Host-side node lengths, marching-square areas, and the SC stability clamp."""

import numpy as np
import jax.numpy as jnp

from rfx.geometry.rasterize_grid import _axis_node_positions
from rfx.geometry.smoothing import _get_sdf_fn
from rfx._periodic import periodic_sdf
from rfx.errors import RfxUnsupported


def edge_fraction(a, b):
    """Free length from the linear zero between two node distances."""
    crossing = ((a < 0) & (b > 0)) | ((b < 0) & (a > 0))
    root = np.divide(np.maximum(a, b), np.abs(b - a),
                     out=np.zeros_like(a), where=crossing)
    return np.where((a >= 0) & (b >= 0), 1., root)


def face_fraction(corners, widths):
    """Marching squares, with the bilinear centre deciding saddle connectivity."""
    distances = np.stack(corners, axis=-1)
    code = np.sum((distances >= 0) * (1 << np.arange(4)), axis=-1)
    area = np.zeros(code.shape, dtype=np.float64)
    area[code == 15] = 1.
    u, v = widths
    vertices = np.array(((0., 0.), (u, 0.), (u, v), (0., v)))

    def polygon(points):
        points = np.stack(points, axis=-2)
        following = np.roll(points, -1, axis=-2)
        return np.abs(np.sum(points[..., 0] * following[..., 1]
                             - points[..., 1] * following[..., 0], axis=-1)) / (2 * u * v)

    for case in range(1, 15):
        selected = code == case
        d = distances[selected]
        if not d.size:
            continue
        roots = {}
        for i in range(4):
            j = (i + 1) % 4
            if bool(case & (1 << i)) != bool(case & (1 << j)):
                t = d[:, i] / (d[:, i] - d[:, j])
                roots[i] = vertices[i] + t[:, None] * (vertices[j] - vertices[i])
        points = []
        for i in range(4):
            if case & (1 << i):
                points.append(np.broadcast_to(vertices[i], (len(d), 2)))
            if i in roots:
                points.append(roots[i])
        fraction = polygon(points)
        if case in (5, 10):
            separate = sum(polygon((roots[(i - 1) % 4],
                                   np.broadcast_to(vertices[i], (len(d), 2)), roots[i]))
                           for i in range(4) if case & (1 << i))
            fraction = np.where(np.mean(d, axis=-1) > 0, fraction, separate)
        area[selected] = fraction
    return np.clip(area, 0., 1.)


def sc_clamp(lengths, areas):
    """Bound each edge by twice the least of its four incident face areas.

    Arrays start at the same node. Missing low-side faces have no constraint;
    geometry supplies a halo so every returned physical edge has four faces.
    """
    result = []
    for axis, length in enumerate(lengths):
        bound = np.ones_like(length)
        for normal in range(3):
            if normal == axis:
                continue
            transverse = 3 - axis - normal
            previous = np.roll(areas[normal], 1, axis=transverse).copy()
            low = [slice(None)] * 3
            low[transverse] = 0
            previous[tuple(low)] = 1.
            bound = np.minimum(bound, np.minimum(areas[normal], previous))
        result.append(np.minimum(length, 2 * bound))
    return tuple(result)


def _refuse_off_lattice_box(box, lattice, widths, domain, what):
    """A Box with no node inside gives the cut cells nothing; only its staircase faces exist.

    A face on or beyond a domain face is that wall's, placed by the boundary itself.
    """
    for axis, ((line, period), width) in enumerate(zip(lattice, widths)):
        if line is None:
            continue
        for value in (box.corner_lo[axis], box.corner_hi[axis]):
            value = value % period if period else value
            if not (period or 1e-6 * width < value < float(domain[axis]) - 1e-6 * width) \
                    or not line[0] < value < line[-1]:
                continue
            offset = np.min(np.abs(line - value)) / width
            if offset > 1e-6:
                raise RfxUnsupported(
                    f'conformal PEC places PEC volumes that hold a grid node; {what} has a {"xyz"[axis]} '
                    f'face {offset:.3g} cell off the node lattice and would stay at its staircase position; '
                    'move it onto node planes, refine the mesh, or use conformal_pec=False')


def compute_conformal_fractions(grid, pec_shapes, *, clamp=True):
    """Return E free lengths and H free areas on the uniform Yee lattice."""
    if not pec_shapes:
        ones = (jnp.ones(grid.shape, dtype=jnp.float32),) * 3
        return ones, ones
    lines, widths, lattice = [], [], []
    for axis in range(3):
        cells = np.asarray(grid.cells(axis), dtype=np.float64)
        if not np.all(cells == cells[0]):
            raise NotImplementedError('conformal PEC SC clamp requires uniform cells')
        width = float(cells[0])
        nodes = _axis_node_positions(cells, getattr(grid, f'pad_{"xyz"[axis]}_lo'))
        line = np.concatenate(([nodes[0] - width], nodes, [nodes[-1] + width]))
        lattice.append((line, None))
        if axis == 2 and getattr(grid, 'is_2d', False):
            line = np.full_like(line, nodes[0])  # The collapsed lane repeats its physical slice.
            lattice[-1] = (None, None)
        elif 'xyz'[axis] in getattr(grid, 'periodic_axes', ''):
            lattice[-1] = (line, float(grid.domain[axis]))
            line = np.mod(line, float(grid.domain[axis]))
        lines.append(line)
        widths.append(width)
    coords = np.meshgrid(*lines, indexing='ij', sparse=True)
    sdf = np.full(tuple(len(line) for line in lines), np.inf, dtype=np.float64)
    from rfx.geometry.csg import Box, PolylineWire
    from rfx.geometry.smoothing import _ContinuedSheetShape
    for shape in pec_shapes:
        fn = _get_sdf_fn(shape)
        if fn is None and isinstance(shape, PolylineWire) and float(shape.radius) < 0.5 * min(widths):
            continue  # A sub-cell filament reaches the step through its own wire edges.
        sheet = shape.declared if isinstance(shape, _ContinuedSheetShape) else shape
        if isinstance(sheet, Box) and min(h - lo for lo, h in zip(sheet.corner_lo, sheet.corner_hi)) <= 0:
            _refuse_off_lattice_box(sheet, lattice, widths, grid.domain, 'this zero-thickness sheet')
            continue  # A zero-thickness Box on the node lattice is exact through its own edge masks.
        if fn is None:
            raise NotImplementedError(
                f'conformal PEC requires a node signed distance for {type(shape).__name__}')
        shape_sdf = periodic_sdf(grid, shape, coords, fn, xp=np)
        if not np.any(shape_sdf < -1e-9 * min(widths)) and np.min(shape_sdf) < max(widths):
            if isinstance(sheet, Box):  # On node planes its staircase faces are exact.
                _refuse_off_lattice_box(sheet, lattice, widths, grid.domain, 'this PEC Box')
                continue
            raise NotImplementedError(
                f'conformal PEC: the {type(shape).__name__} holds no grid node, so the cut-cell '
                'operands would drop it; refine the mesh or use the staircase'
                + ('; on the 2-D lane the z = 0 slice must cross its interior, not a face'
                   if getattr(grid, 'is_2d', False) else ''))
        sdf = np.minimum(sdf, shape_sdf)
    base = sdf[:-1, :-1, :-1]
    lengths, areas = [], []
    for axis in range(3):
        end = [slice(None, -1)] * 3
        end[axis] = slice(1, None)
        lengths.append(edge_fraction(base, sdf[tuple(end)]))
        u, v = [a for a in range(3) if a != axis]
        corners = [base]
        for moved in ((u,), (u, v), (v,)):
            corner = [slice(None, -1)] * 3
            for a in moved:
                corner[a] = slice(1, None)
            corners.append(sdf[tuple(corner)])
        areas.append(face_fraction(corners, (widths[u], widths[v])))
    if clamp:
        lengths = sc_clamp(lengths, areas)
    return tuple(tuple(jnp.asarray(a[1:, 1:, 1:], dtype=jnp.float32) for a in arrays)
                 for arrays in (lengths, areas))
