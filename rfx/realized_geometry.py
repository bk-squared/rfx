"""Host-side records of the geometry built for a simulation.

Lengths are metres. Index ranges are inclusive for nodes and half-open for
cells. Sheet bounds include the solved free-edge offset, not just node span.
Arrays are read-only numpy arrays; this API is not an autodiff interface.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AxisGeometry:
    """One entity axis, including signed (solved minus declared) residuals."""

    axis: str
    node_range: tuple[int, int]
    cell_range: tuple[int, int] | None
    declared_bounds_m: tuple[float, float] | None
    bounds_m: tuple[float, float]
    extent_m: float
    face_residual_m: tuple[float, float] | None
    cell_size_m: float


@dataclass(frozen=True)
class EntityGeometry:
    """A declared material or conductor, identified by declaration route/index."""

    label: str
    name: str
    kind: str
    axes: tuple[AxisGeometry, ...]
    n_cells: int
    plane: tuple[int, int, float] | None
    wall_planes: tuple[tuple[int, ...], ...]
    mask: np.ndarray | None
    edge_masks: tuple[np.ndarray, ...]
    error: str | None = None


@dataclass(frozen=True)
class DomainAxisGeometry:
    """Interior cell count, mesh length and absorber padding on one axis."""

    axis: str
    n_cells: int
    length_m: float
    padding: tuple[int, int]
    declared_length_m: float


@dataclass(frozen=True)
class PortGeometry:
    """Driven E edges (component and indices), or a waveguide aperture."""

    label: str
    kind: str
    component: str | None
    edges: tuple[tuple[int, int, int], ...]
    aperture: tuple[tuple[int, int], ...] | None = None
    error: str | None = None


@dataclass(frozen=True)
class RealizedGeometry:
    """One immutable host record shared by build-time readers and Result.

    ``entities`` are in geometry then thin-conductor declaration order.
    ``nodes`` include the far face of the last stored cell. ``edge_masks``
    are the assembled PEC edges before any driven port edge is cleared.
    ``ports`` identifies the driven edges separately.
    """

    entities: tuple[EntityGeometry, ...]
    ports: tuple[PortGeometry, ...]
    domain: tuple[DomainAxisGeometry, ...]
    nodes: tuple[np.ndarray, ...]
    cell_sizes: tuple[np.ndarray, ...]
    edge_masks: tuple[np.ndarray, ...]
    pec_mask: np.ndarray | None

    def wall_planes(self, axis: int, **kwargs):
        """Read tangential PEC wall planes from the assembled edge masks."""
        from rfx.boundaries.pec import realized_wall_planes
        return realized_wall_planes(self.edge_masks, axis, **kwargs)


def _readonly(value):
    a = np.asarray(value)
    # An immutable bytes backing prevents setflags(write=True), too.
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _build_record(sim, ctx):
    from rfx.fidelity import _node_arrays
    from rfx.mesh_edges import solved_sheet_span
    from rfx.preflight.realization import _shape_bounds

    assembled = ctx.realized()
    if assembled is None:
        raise ValueError(ctx.assembly_error or ctx.error or "geometry assembly unavailable")
    sizes, nodes = _node_arrays(sim, ctx.grid, ctx.lane == "nonuniform")
    conductors = {e.label: e for e in ctx.entry_realizations()}
    interior = {e.label: e for e in ctx.interior_pec_entries()}
    union = None
    for e in interior.values():
        if e.sheet is not None:
            fp = np.asarray(e.sheet.footprint, dtype=bool)
            union = fp.copy() if union is None else union | fp
    declarations = [(f"geometry[{i}]", e.material_name, e.shape)
                    for i, e in enumerate(sim._geometry)]
    declarations.extend((f"thin_conductor[{i}]", f"thin_conductor[{i}]", e.shape)
                        for i, e in enumerate(sim._thin_conductors))
    entities = []
    for index, (label, name, shape) in enumerate(declarations):
        e = interior.get(label, conductors.get(label))
        bounds = _shape_bounds(shape)
        sheet = None if e is None else e.sheet
        kind = "material" if e is None else e.kind
        if e is None:
            mask = np.asarray(assembled.geometry_masks[index], dtype=bool)
        elif sheet is not None:
            mask = np.asarray(sheet.footprint, dtype=bool)
        elif e.cells is not None:
            mask = e.cells
        elif e.kind == "wire":
            mask = np.logical_or.reduce(e.edges(ctx.periodic, ctx.grid.shape))
        else:
            mask = None
        axes = []
        if mask is not None and mask.any():
            occ = np.where(mask)
            for a in range(3):
                i0, i1 = int(occ[a].min()), int(occ[a].max())
                cell_range = None if kind in ("sheet", "wire") else (i0, i1 + 1)
                rlo = float(nodes[a][i0])
                rhi = float(nodes[a][i1 if cell_range is None else i1 + 1])
                if sheet is not None and a != int(sheet.normal_axis) and bounds is not None:
                    span = solved_sheet_span(
                        mask, a, nodes[a], float(bounds[0][a]), float(bounds[1][a]),
                        float(sim._domain[a]), union=union,
                        periodic='xyz'[a] in getattr(ctx.grid, 'periodic_axes', ''))
                    if span is not None:
                        rlo, rhi = span.lo, span.hi
                declared = None if bounds is None else (float(bounds[0][a]), float(bounds[1][a]))
                residual = None if declared is None else (rlo - declared[0], rhi - declared[1])
                axes.append(AxisGeometry(
                    'xyz'[a], (i0, i1 if cell_range is None else i1 + 1), cell_range,
                    declared, (rlo, rhi), rhi - rlo, residual,
                    float(np.mean(sizes[a][i0:i1 + 1]))))
        edges = () if e is None else tuple(_readonly(m) for m in e.edges(ctx.periodic, ctx.grid.shape))
        walls = (() if e is None else tuple(tuple(e.wall_planes(a, ctx.periodic, ctx.grid.shape))
                                           for a in range(3)))
        plane = (None if sheet is None else
                 (int(sheet.normal_axis), int(sheet.plane),
                  float(nodes[int(sheet.normal_axis)][int(sheet.plane)])))
        entities.append(EntityGeometry(
            label, name, kind, tuple(axes),
            int(mask.sum()) if mask is not None and kind in ("material", "volume") else 0,
            plane, walls, None if mask is None else _readonly(mask), edges,
            None if e is None else e.error))
    domain = []
    for a, axis in enumerate('xyz'):
        plo = int(getattr(ctx.grid, f'pad_{axis}_lo'))
        phi = int(getattr(ctx.grid, f'pad_{axis}_hi'))
        end = max(len(sizes[a]) - phi - 1, plo)
        if axis in getattr(ctx.grid, 'periodic_axes', ''):
            end += 1
        domain.append(DomainAxisGeometry(axis, end - plo,
                                        float(nodes[a][end] - nodes[a][plo]),
                                        (plo, phi), float(sim._domain[a])))
    return RealizedGeometry(tuple(entities), _ports(sim, ctx), tuple(domain),
                            tuple(_readonly(n) for n in nodes), tuple(_readonly(s) for s in sizes),
                            tuple(_readonly(m) for m in assembled.edges),
                            None if assembled.pec_mask is None else _readonly(assembled.pec_mask))


def _ports(sim, ctx):
    from rfx.sources.sources import WirePort, _wire_port_cells, wire_port_edge_span
    from rfx.nonuniform import position_to_index

    grid = ctx.grid
    locate = (lambda pos: position_to_index(grid, pos)) if ctx.lane == "nonuniform" else grid.position_to_index
    rows = []
    for i, pe in enumerate(sim._ports):
        if pe.extent is None:
            cells = (tuple(int(k) for k in locate(pe.position)),)
        else:
            a = 'xyz'.index(pe.component[1])
            end = list(pe.position)
            end[a] += pe.extent
            if ctx.lane == "uniform":
                cells = tuple(tuple(int(k) for k in cell) for cell in _wire_port_cells(
                    grid, WirePort(start=pe.position, end=tuple(end), component=pe.component,
                                   impedance=pe.impedance)))
            else:
                start_idx, end_idx = locate(pe.position), locate(tuple(end))
                lo, hi = sorted((start_idx[a], end_idx[a]))
                first, last = wire_port_edge_span(grid, a, lo, hi, pe.position[a], end[a])
                cells = tuple(tuple(k if j == a else int(start_idx[j]) for j in range(3))
                              for k in range(first, last + 1))
        rows.append(PortGeometry(f"port[{i}]", "lumped" if pe.extent is None else "wire",
                                 pe.component, cells))
    for i, pe in enumerate(sim._waveguide_ports):
        a = 'xyz'.index(pe.direction[1])
        pos = [0.0] * 3
        pos[a] = pe.x_position
        k = int(locate(tuple(pos))[a])
        aperture = []
        for b, axis in enumerate('xyz'):
            if b == a:
                aperture.append((k, k))
            else:
                sl, _ = sim._range_to_slice(getattr(pe, f'{axis}_range'), sim._domain[b],
                                            grid.dx, grid.shape[b], grid.axis_pads[b])
                aperture.append((int(sl.start), int(sl.stop - 1)))
        rows.append(PortGeometry(f"waveguide_port[{i}]", "waveguide", None, (), tuple(aperture)))
    return tuple(rows)


def realized_geometry(sim):
    """Build once per preflight context and port configuration, without stepping."""
    ctx = sim._campaign_ctx()
    key = (id(ctx), tuple(id(p) for p in sim._ports), tuple(id(p) for p in sim._waveguide_ports))
    cached = getattr(sim, '_realized_geometry_record', None)
    if cached is not None and cached[0] == key:
        return cached[1]
    record = _build_record(sim, ctx)
    sim._realized_geometry_record = key, record
    return record
