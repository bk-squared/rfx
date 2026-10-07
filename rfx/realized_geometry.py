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
    """One entity axis, including signed (solved minus declared) residuals.

    Sheet in-plane axes also expose ``comparison_bounds_m`` (the drawing
    with non-free ends replaced by their solved coordinates) and ``free_ends`` for size verdicts.
    ``declared_bounds_m`` and ``face_residual_m`` retain the original drawing.
    """

    axis: str
    node_range: tuple[int, int]
    cell_range: tuple[int, int] | None
    declared_bounds_m: tuple[float, float] | None
    bounds_m: tuple[float, float]
    extent_m: float
    face_residual_m: tuple[float, float] | None
    cell_size_m: float
    # Sheet-size verdict excludes residuals at all non-free ends.
    comparison_bounds_m: tuple[float, float] | None = None
    free_ends: tuple[bool, bool] | None = None


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
    sheet: object = None
    declared_bounds_m: tuple | None = None
    continued_faces: tuple[str, ...] = ()
    mask_error: str | None = None
    # Refused occupancy is diagnostic only; no edges were assembled.
    occupancy_role: str = "solved"
    node_count: int = 0
    edge_counts: tuple[int, ...] = ()
    edge_ranges: tuple = ()


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
    """Immutable geometry metadata with optional build-time diagnostic arrays.

    ``entities`` are in geometry then thin-conductor declaration order.
    ``nodes`` include the far face of the last stored cell. ``edge_masks``
    preview the Yee PEC edges after driven port edge clearing. Yee Result
    records are bound at the kernel; ADI/subgrid records describe pre-port
    assembly (the subgrid refinement is not represented).
    ``ports`` identifies the driven edges separately. Result records omit
    dense masks, materials and sheet/wire specs; entity edge counts/ranges
    summarize the interior solver geometry. Request diagnostic arrays through
    ``Simulation.realized_geometry()``. ``lane`` names the execution lane.
    ``snap`` records the strict or declared sheet-size acceptance policy.
    """

    entities: tuple[EntityGeometry, ...]
    ports: tuple[PortGeometry, ...]
    domain: tuple[DomainAxisGeometry, ...]
    nodes: tuple[np.ndarray, ...]
    cell_sizes: tuple[np.ndarray, ...]
    edge_masks: tuple[np.ndarray, ...]
    pec_mask: np.ndarray | None
    sheets: tuple
    wires: tuple
    materials: object
    periodic: tuple[bool, bool, bool]
    refused: tuple[tuple[int, str], ...] = ()
    refused_thin_conductors: tuple[tuple[int, str], ...] = ()
    pad_fill_findings: tuple = ()
    lane: str = "uniform"
    limitations: tuple[str, ...] = ()
    snap: str = "strict"

    @property
    def conductors(self):
        """The source object of this view (outside the public record schema)."""
        return getattr(self, '_conductors', None)

    def wall_planes(self, axis: int, **kwargs):
        """Read tangential PEC wall planes from the assembled edge masks."""
        from rfx.boundaries.pec import realized_wall_planes
        if not self.edge_masks:
            raise ValueError("Dense wall query requires Simulation.realized_geometry()")
        kwargs.setdefault("periodic", self.periodic)
        return realized_wall_planes(self.edge_masks, axis, **kwargs)


def _readonly(value):
    a = np.asarray(value)
    # An immutable bytes backing prevents setflags(write=True), too.
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _freeze(value):
    from dataclasses import fields, is_dataclass, replace
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray) or hasattr(value, "__array__"):
        return _readonly(value)
    if is_dataclass(value):
        return replace(value, **{f.name: _freeze(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_freeze(v) for v in value))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    return value


def _assembly(sim, ctx):
    """Suppress the assembly duplicate; diagnostic entry classification warns."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Conducting geometry reaches an absorbing face")
        return _assembly_impl(sim, ctx)


def _assembly_impl(sim, ctx):
    """Reuse production assembly; retain refused-model audit evidence too."""
    assembled = ctx.realized()
    if assembled is not None:
        return assembled, {}, {}, list(getattr(assembled, "pad_fill_findings", ()))
    if ctx.grid is None:
        raise ValueError(ctx.error)
    import copy
    from rfx.fidelity import _contract_refusals
    from rfx.runners.nonuniform import nu_thin_conductor_refusal
    from rfx.model.conductors import realized_conductors
    nonuniform = ctx.lane == "nonuniform"
    refused = _contract_refusals(sim, ctx.grid, nonuniform)
    refused_tc = {i: why for i, tc in enumerate(sim._thin_conductors)
                  if nonuniform and (why := nu_thin_conductor_refusal(tc)) is not None}
    audit = copy.copy(sim)
    audit._geometry = [e for i, e in enumerate(sim._geometry) if i not in refused]
    audit._thin_conductors = [tc for i, tc in enumerate(sim._thin_conductors) if i not in refused_tc]
    assembled = realized_conductors(audit, ctx.grid, nonuniform=nonuniform,
                                    periodic=ctx.periodic, mode="audit")
    return assembled, refused, refused_tc, list(assembled.pad_fill_findings)


def _build_record(sim, ctx, *, compact=False):
    from rfx.preflight.realization import _shape_bounds

    assembled, refused, refused_tc, pad_findings = _assembly(sim, ctx)
    sizes, nodes = _node_arrays(sim, ctx.grid, ctx.lane == "nonuniform")
    conductors = {e.label: e for e in ctx.entry_realizations()}
    interior = {e.label: e for e in ctx.interior_pec_entries()}
    declarations = [(f"geometry[{i}]", e.material_name, e.shape)
                    for i, e in enumerate(sim._geometry)]
    declarations.extend((f"thin_conductor[{i}]", f"thin_conductor[{i}]", e.shape)
                        for i, e in enumerate(sim._thin_conductors))
    entities = []
    f0_specs = _f0_specs_by_conductor(sim, assembled, refused_tc)
    for index, (label, name, shape) in enumerate(declarations):
        mask_error = None
        node_based = False
        f0_spec = None
        e = interior.get(label, conductors.get(label))
        bounds = _shape_bounds(shape)
        sheet = None if e is None else e.sheet
        kind = "material" if e is None else e.kind
        if e is None:
            mask = np.asarray(assembled.geometry_masks[id(sim._geometry[index])], dtype=bool)
        elif sheet is not None:
            mask = np.asarray(sheet.footprint, dtype=bool)
        elif e.cells is not None:
            mask = e.cells
        elif e.kind == "wire":
            mask = np.logical_or.reduce(e.edges(ctx.periodic, ctx.grid.shape))
        elif e.kind == "lossy" and id(sim._thin_conductors[index - len(sim._geometry)]) in assembled.geometry_masks:
            from rfx.geometry.rasterize_grid import interior_lattice_mask
            mask = interior_lattice_mask(
                assembled.geometry_masks[id(sim._thin_conductors[index - len(sim._geometry)])], ctx.grid)
            # An f0 sheet's mask is its NODE footprint (the DC fold's is cells).
            node_based = sim._thin_conductors[index - len(sim._geometry)].surface_impedance_f0 is not None
            f0_spec = f0_specs.get(index - len(sim._geometry)) if node_based else None
        else:
            # Retain the refused declaration's diagnostic occupancy once.
            # It is explicitly NOT a solved conductor and contributes no edges.
            try:
                mask = ctx.rasterize(shape)
            except Exception as exc:
                mask = None
                mask_error = f"{type(exc).__name__}: {exc}"
        axes = []
        if mask is not None and mask.any():
            widths = assembled.width_axes(sim, mask, kind=kind, sheet=sheet,
                node_based=node_based, f0_spec=f0_spec, bounds=bounds,
                sizes=sizes, nodes=nodes, interior=interior.values())
            for a, width in enumerate(widths):
                i0, i1, cell_range = width.first, width.last, width.cell_range
                rlo, rhi = width.lo, width.hi
                comparison, free_ends = width.comparison, width.free_ends
                declared = None if bounds is None else (float(bounds[0][a]), float(bounds[1][a]))
                residual = None if declared is None else (rlo - declared[0], rhi - declared[1])
                axes.append(AxisGeometry(
                    'xyz'[a], (i0, i1 if cell_range is None else i1 + 1), cell_range,
                    declared, (rlo, rhi), rhi - rlo, residual,
                    float(np.mean(sizes[a][i0:i1 + 1])), comparison, free_ends))
        raw_edges = () if e is None else e.edges(ctx.periodic, ctx.grid.shape)
        edge_counts = tuple(int(m.sum()) for m in raw_edges)
        edge_ranges = tuple(_mask_ranges(m) for m in raw_edges)
        edges = () if compact else tuple(_readonly(m) for m in raw_edges)
        walls = (() if e is None else tuple(tuple(e.wall_planes(a, ctx.periodic, ctx.grid.shape))
                                           for a in range(3)))
        plane = (None if sheet is None else
                 (int(sheet.normal_axis), int(sheet.plane),
                  float(nodes[int(sheet.normal_axis)][int(sheet.plane)])))
        continued = ()
        if e is not None:
            solved_bounds = _shape_bounds(e.solved_shape)
            if bounds is not None and solved_bounds is not None:
                continued = tuple(f"{'xyz'[a]}-{'hi' if side else 'lo'}"
                                  for a in range(3) for side in (0, 1)
                                  if bounds[side][a] != solved_bounds[side][a])
        entities.append(EntityGeometry(
            label, name, kind, tuple(axes),
            int(mask.sum()) if mask is not None and kind not in ("sheet", "wire") else 0,
            plane, walls, None if compact or mask is None else _readonly(mask), edges,
            None if e is None else e.error, None if compact else _freeze(sheet),
            None if bounds is None else tuple(tuple(float(v) for v in b) for b in bounds[:2]),
            continued, mask_error, "diagnostic" if kind == "refused" else "solved",
            int(mask.sum()) if mask is not None and kind in ("sheet", "wire") else 0,
            edge_counts, edge_ranges))
        if compact and e is not None:
            e._edges = None
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
    entities.extend(_pinned_entities(sim, ctx, assembled, nodes, sizes, compact=compact))
    record = RealizedGeometry(tuple(entities), _ports(sim, ctx, assembled), tuple(domain),
                            tuple(_readonly(n) for n in nodes), tuple(_readonly(s) for s in sizes),
                            () if compact else tuple(_readonly(m) for m in assembled.edges),
                            None if compact or assembled.pec_mask is None else _readonly(assembled.pec_mask),
                            () if compact else _freeze(assembled.sheets), () if compact else _freeze(assembled.wires),
                            None if compact else _freeze(assembled.materials), tuple(ctx.periodic),
                            tuple(refused.items()), tuple(refused_tc.items()),
                            tuple(tuple(row.items()) for row in pad_findings), ctx.lane,
                            snap=sim._snap)
    return record if compact else _conductor_view(record, assembled)


def _f0_specs_by_conductor(sim, assembled, refused_tc):
    """``thin_conductor`` index -> its ``SheetImpedanceSpec`` (assembly order)."""
    f0 = [i for i, tc in enumerate(sim._thin_conductors)
          if i not in refused_tc and getattr(tc, "surface_impedance_f0", None) is not None]
    specs = tuple(getattr(assembled, "sheet_impedance", ()) or ())
    return dict(zip(f0, specs)) if len(f0) == len(specs) else {}



def _pinned_entities(sim, ctx, assembled, nodes, sizes, *, compact=False):
    """Read the pinned specs appended by both production assemblers."""
    from rfx.preflight.realization import _realized_edges_np
    from rfx.boundaries.pec import realized_wall_planes
    declarations = getattr(sim, "_pinned_sheets", ())
    specs = assembled.sheets[len(assembled.sheets) - len(declarations):] if declarations else ()
    rows = []
    for i, (declaration, sheet) in enumerate(zip(declarations, specs, strict=True)):
        fp = np.asarray(sheet.footprint, bool)
        unwrapped = getattr(sheet, "unwrapped_footprint", None)
        occ = np.where(fp if unwrapped is None else np.asarray(unwrapped, bool))
        axes = []
        for a in range(3):
            lo, hi = int(occ[a].min()), int(occ[a].max())
            bounds = (float(nodes[a][lo]), float(nodes[a][hi]))
            axes.append(AxisGeometry('xyz'[a], (lo, hi), None, None, bounds,
                                     bounds[1] - bounds[0], None, float(sizes[a][lo])))
        a, k = int(sheet.normal_axis), int(sheet.plane)
        edges = _realized_edges_np(None, [sheet], [], ctx.periodic, ctx.grid.shape)
        walls = tuple(tuple(realized_wall_planes(edges, axis, periodic=ctx.periodic)) for axis in range(3))
        rows.append(EntityGeometry(
            f"pinned_sheet[{i}]", declaration.name or "pinned_sheet", "sheet", tuple(axes),
            0, (a, k, float(nodes[a][k])), walls, None if compact else _readonly(fp),
            () if compact else tuple(_readonly(m) for m in edges),
            sheet=None if compact else _freeze(sheet), node_count=int(fp.sum()),
            edge_counts=tuple(int(m.sum()) for m in edges),
            edge_ranges=tuple(_mask_ranges(m) for m in edges)))
    return rows


def _ports(sim, ctx, assembled):
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
        if pe.extent is not None:
            from rfx.boundaries.pec import edges_are_pec
            cells = tuple(cell for cell, dead in zip(
                cells, edges_are_pec(assembled.edges, pe.component, cells), strict=True) if not dead)
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
                if ctx.lane == "nonuniform":
                    from rfx.runners.nonuniform import _range_to_slice_nu
                    sl, _ = _range_to_slice_nu(grid, axis, getattr(pe, f'{axis}_range'))
                else:
                    sl, _ = sim._range_to_slice(getattr(pe, f'{axis}_range'), sim._domain[b],
                                                grid.boundary_cell(b, "lo"), grid.shape[b], grid.axis_pads[b])
                aperture.append((int(sl[0]), int(sl[1] - 1)))
        rows.append(PortGeometry(f"waveguide_port[{i}]", "waveguide", None, (), tuple(aperture)))
    for attr, kind in (("_msl_ports", "msl"), ("_coaxial_ports", "coaxial"),
                       ("_floquet_ports", "floquet")):
        for i, _ in enumerate(getattr(sim, attr, ())):
            rows.append(PortGeometry(f"{kind}_port[{i}]", kind, None, (),
                                     error="not represented: family-specific aperture/drive assembly"))
    return tuple(rows)


def realized_geometry(sim):
    """Build once per preflight context and port configuration, without stepping."""
    ctx = sim._campaign_ctx()
    key = (id(ctx), *(tuple(id(p) for p in getattr(sim, attr, ())) for attr in
                     ("_ports", "_waveguide_ports", "_msl_ports", "_coaxial_ports", "_floquet_ports")))
    cached = getattr(sim, '_realized_geometry_record', None)
    if cached is not None and cached[0] == key:
        return cached[1]
    from rfx.model.conductors import RealizedConductors, preview_port_stages
    assembled = ctx.realized()
    if isinstance(assembled, RealizedConductors):
        record = _record_from_conductors_impl(
            sim, preview_port_stages(sim, assembled), lane=ctx.lane, compact=False)
    else:
        record = _build_record(sim, ctx)
    sim._realized_geometry_record = key, record
    return record


def _node_arrays(sim, grid, nonuniform):
    """Per-axis ``(cell sizes, node positions)`` on the padded grid.

    Both come from the SAME exact host-float64 producers the rasterizer
    reads — ``coords_from_nonuniform_grid`` (float64 spine) on the
    non-uniform lane and ``_uniform_axis_nodes`` (closed form) on the
    uniform lane — so the report and the realized mask cannot disagree on
    where a node is. Before this the report re-derived its own node line by
    cumsumming the solver's float32 cell-size stores (NU) or a float64
    ``np.full`` (uniform); on a graded 0.3048 mm profile that line sat
    3.9e-10 m off the rasterizer's, a uniform-valued NU axis reported a
    declared 500.0 um cell as 500.00002374872565 um, and exactly-on-node
    faces carried ~1e-12 um residuals (#833 item 2, measured 2026-09-02).

    ``sizes[a]`` has one entry per node-provider cell (``grid.shape[a]``
    entries); ``nodes[a]`` has one more — the grid's own nodes plus the far
    edge of the last cell — so ``nodes[a][i + 1]`` is the far face of cell
    ``i`` for every ``i``. Node index ``pad_lo`` sits at domain 0 on every
    axis, by the producers' construction.
    """
    from rfx.geometry.rasterize_grid import (
        _uniform_axis_nodes, coords_from_nonuniform_grid)
    if nonuniform:
        c = coords_from_nonuniform_grid(grid)
        lines = tuple(np.asarray(v, dtype=np.float64) for v in (c.x, c.y, c.z))
        stores = (grid.dx_arr, grid.dy_arr, grid.dz)
        spines = (getattr(grid, "dx_arr_f64", None),
                  getattr(grid, "dy_arr_f64", None),
                  getattr(grid, "dz_f64", None))
        # Cell sizes from the same spine the node line was built from; a
        # grid without one has already been warned about by the producer.
        sizes = tuple(np.asarray(store if spine is None else spine,
                                 dtype=np.float64)
                      for store, spine in zip(stores, spines))
    else:
        n = grid.shape
        pads = grid.axis_pads
        sizes = tuple(grid.cells(a) for a in range(3))
        lines = tuple(_uniform_axis_nodes(n[a], pads[a], float(sizes[a][0]))
                      for a in range(3))
    nodes = tuple(np.concatenate([line, [line[-1] + d[-1]]])
                  for line, d in zip(lines, sizes))
    return sizes, nodes


def _mask_ranges(mask):
    indices = np.where(mask)
    return tuple((int(i.min()), int(i.max())) for i in indices) if indices[0].size else ()


def _record_from_conductors_impl(sim, conductors, *, lane, compact=True):
    """Summarize the run's classifier outputs; never consult preflight caches."""
    from dataclasses import replace
    from rfx.preflight.realization import _CampaignStaticsContext, _EntryRealization
    from rfx.core.jax_utils import is_tracer
    grid = conductors.grid
    if any(is_tracer(getattr(grid, name, None)) for name in ("dx_arr", "dy_arr", "dz")):
        return None
    if any(is_tracer(edge) for edge in conductors.pec_edges or ()):
        return None
    assembly_entries = conductors.assembly_entries
    # Legacy direct runners can receive arrays without per-entity collectors.
    # Those arrays still define the kernel, but cannot support a geometry record.
    # Do not reassemble or invent entity occupancy from the aggregate PEC mask.
    observed = set(conductors.geometry_masks) | {entry[0] for entry in assembly_entries}
    if any(id(entry) not in observed
           for entry in (*sim._geometry, *sim._thin_conductors)):
        return None
    ctx = object.__new__(_CampaignStaticsContext)
    ctx.sim, ctx.grid = sim, grid
    ctx.lane = "nonuniform" if hasattr(grid, "dx_arr") else "uniform"
    ctx.periodic = tuple(sim._periodic_flags()) if ctx.lane == "uniform" else (False, False, False)
    ctx.assembly_error = ctx._assembly_exception = None
    ctx._realized = conductors
    products = {key: (cells, sheet, wire, shape) for key, cells, sheet, wire, shape in assembly_entries}
    ctx._entries = []
    for collection, prefix in ((sim._geometry, "geometry"), (sim._thin_conductors, "thin_conductor")):
        for i, entry in enumerate(collection):
            label = f"{prefix}[{i}]"
            if id(entry) in products:
                cells, sheet, wire, shape = products[id(entry)]
                kind = "volume" if cells is not None else "sheet" if sheet is not None else "wire" if wire is not None else "lossy"
                ctx._entries.append(_EntryRealization(label=label, name=getattr(entry, "material_name", label),
                    shape=entry.shape, solved_shape=shape, kind=kind, cells=cells, sheet=sheet, wire=wire))
            elif prefix == "thin_conductor":
                ctx._entries.append(_EntryRealization(label=label, name=label, shape=entry.shape, kind="lossy"))
    record = _build_record(sim, ctx, compact=compact)
    record = replace(record, lane=lane, limitations=("refined region not represented",)
                     if lane == "run_subgridded" else ())
    return record if compact else _conductor_view(record, conductors)


def _record_from_assembly(sim, grid, materials, pec_mask, sheets, wires,
                          geometry_masks, assembly_entries, *, lane,
                          conductors=None, compact=True):
    """Compatibility hook; production callers supply their kernel object."""
    from rfx.model.conductors import realized_conductors
    if conductors is None:
        conductors = realized_conductors(sim, grid,
            nonuniform=hasattr(grid, 'dx_arr'), assembly=(materials, None, None, pec_mask),
            pec_sheets=sheets, pec_wires=wires, geometry_masks=geometry_masks,
            assembly_entries=assembly_entries)
    return _record_from_conductors_impl(sim, conductors, lane=lane, compact=compact)


def record_from_assembly(sim, grid, *args, **kwargs):
    """Skip only traced coordinates; concrete record failures propagate."""
    from rfx.core.jax_utils import is_tracer
    if any(is_tracer(getattr(grid, name, None)) for name in ("dx_arr", "dy_arr", "dz")):
        return None
    return _record_from_assembly(sim, grid, *args, **kwargs)


def record_from_conductors(sim, conductors, *, lane, compact=True):
    return record_from_assembly(sim, conductors.grid, conductors.materials,
        conductors.pec_cells, conductors.sheets, conductors.wires,
        conductors.geometry_masks.items(), conductors.assembly_entries,
        lane=lane, conductors=conductors, compact=compact)


def attach_record(result, record):
    """Attach metadata to the public Result, preserving custom runner results."""
    from rfx.api._spec import Result
    if isinstance(result, Result):
        return result._replace(realized_geometry=record)
    return result


def _conductor_view(record, conductors):
    # Keep the historical dataclass fields/serialization stable. The source
    # object includes assembly bookkeeping and lane-specific grid types.
    object.__setattr__(record, '_conductors', conductors)
    return record
