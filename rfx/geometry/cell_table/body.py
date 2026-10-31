"""Body-only signed distances, edge lengths, and marching-square face areas."""

import numpy as np

from rfx._periodic import periodic_sdf
from rfx.core.jax_utils import is_tracer
from rfx.errors import RfxUnsupported
from rfx.geometry.cell_table.nodes import _axis_node_positions
from rfx.geometry.cell_table.table import CellGeometryTable


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


def _refuse_off_lattice_box(box, lattice, widths, domain, what):
    """A Box with no node inside gives the cut cells nothing; only its staircase faces exist.

    A face on or beyond a domain face is that wall's, placed by the boundary itself.
    """
    for axis, ((line, period, uniform), width) in enumerate(zip(lattice, widths)):
        if line is None:
            continue
        for value in (box.corner_lo[axis], box.corner_hi[axis]):
            value = value % period if period else value
            if not (period or 1e-6 * width < value < float(domain[axis]) - 1e-6 * width) \
                    or not line[0] < value < line[-1]:
                continue
            index = np.searchsorted(line, value) - 1
            local_width = width if uniform else line[index + 1] - line[index]
            offset = np.min(np.abs(line - value)) / local_width
            if offset > 1e-6:
                raise RfxUnsupported(
                    f'conformal PEC places PEC volumes that hold a grid node; {what} has a {"xyz"[axis]} '
                    f'face {offset:.3g} cell off the node lattice and would stay at its staircase position; '
                    'move it onto node planes, refine the mesh, or use conformal_pec=False')


def build_cell_table(grid, pec_shapes):
    """Form body fractions once on the host, including both consumer halos."""
    for axis in range(3):
        if getattr(grid, 'is_traced', lambda axis: False)(axis):
            raise RfxUnsupported(f'cell geometry table requires concrete cells on axis {"xyz"[axis]}')
    axis_cells = [grid.cells(axis) for axis in range(3)]
    for axis, cells in enumerate(axis_cells):
        if is_tracer(cells):
            raise RfxUnsupported(f'cell geometry table requires concrete cells on axis {"xyz"[axis]}')
    if not pec_shapes:
        shape = tuple(n + 2 for n in grid.shape)
        return CellGeometryTable(tuple(np.ones(shape, dtype=np.float64) for _ in range(3)),
                                 tuple(np.ones(shape, dtype=np.float64) for _ in range(3)))
    if hasattr(grid, 'domain') and grid.domain is None:
        domain = tuple(float(np.sum(axis_cells[axis])) if name in grid.periodic_axes else
                       grid.node_of(axis, grid.shape[axis] - getattr(grid, f'pad_{name}_hi') - 1)
                       for axis, name in enumerate('xyz'))
        grid = grid._replace(domain=domain)
    from rfx.geometry.smoothing import _get_sdf_fn

    lines, widths, lattice = [], [], []
    for axis, cells in enumerate(axis_cells):
        cells = np.asarray(cells, dtype=np.float64)
        uniform = bool(np.all(cells == cells[0]))
        width = float(np.min(cells))
        periodic = 'xyz'[axis] in getattr(grid, 'periodic_axes', '')
        # Non-periodic stores end in a node-provider duplicate, not a physical cell.
        end_width = float(cells[-1 if periodic or len(cells) == 1 else -2])
        nodes = _axis_node_positions(cells, getattr(grid, f'pad_{"xyz"[axis]}_lo'))
        line = np.concatenate(([nodes[0] - float(cells[0])], nodes, [nodes[-1] + end_width]))
        line = np.concatenate((line, [nodes[-1] + 2 * end_width]))
        lattice.append((line, None, uniform))
        if axis == 2 and getattr(grid, 'is_2d', False):
            line = np.full_like(line, nodes[0])  # The collapsed lane repeats its physical slice.
            lattice[-1] = (None, None, uniform)
        elif periodic:
            lattice[-1] = (line, float(grid.domain[axis]), uniform)
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
        # The node-holding verdict reads the lattice without the trailing halo node,
        # which only the floor rule's slice uses.
        held = np.broadcast_to(shape_sdf, sdf.shape)[:-1, :-1, :-1]
        if not np.any(held < -1e-9 * min(widths)) and np.min(held) < max(widths):
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
    return CellGeometryTable(tuple(lengths), tuple(areas))
