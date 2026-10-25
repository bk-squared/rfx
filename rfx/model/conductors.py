"""Realized conductor products and the stages that change their edge masks.

The classifier and Yee edge rules remain in rasterize_grid and boundaries.pec.
This module owns their products, not another rasterization convention.
"""
from __future__ import annotations

from rfx._grid_metric import field_index
from dataclasses import dataclass, field, replace

import numpy as np

from rfx.preflight._common import PreflightReport
from rfx.preflight.realization import _RealizedPEC
from rfx.boundaries.pec import SheetSpec
from rfx.model.design_admission import with_design_diagnostics



@dataclass(frozen=True)
class ConductorStage:
    entity_ids: tuple[str, ...]
    stage: str
    component: str | None = None
    edges: tuple = ()


@dataclass(frozen=True)
class RealizedSheet(SheetSpec):
    """The production footprint with its once-computed solved in-plane spans."""
    entity_id: str = ""
    solved_spans: tuple = (None, None, None)


@dataclass(frozen=True)
class ConductorWidthAxis:
    """The solved interval consumed by the geometry record, in metres."""
    first: int
    last: int
    cell_range: tuple | None
    lo: float
    hi: float
    comparison: object = None
    free_ends: object = None


def conductor_width_axes(root, sim, mask, *, kind, sheet, node_based,
                         f0_spec, bounds, sizes, nodes, interior):
    """One record product: PEC solved spans, DC cells, or weighted f0 nodes."""
    from rfx.mesh_edges import solved_sheet_span, sheet_continuation_masks
    union, volume_edges = sheet_continuation_masks(interior, root.periodic, root.grid.shape)
    fed = (port_fed_nodes(sim, root)
           if sheet is not None and not hasattr(sheet, 'solved_spans') else None)
    occupied = np.where(mask)
    result = []
    for a in range(3):
        periodic = 'xyz'[a] in getattr(root.grid, 'periodic_axes', '')
        i0, i1 = int(occupied[a].min()), int(occupied[a].max())
        cell_range = None if kind in ('sheet', 'wire') or node_based else (i0, i1 + 1)
        lo = float(nodes[a][i0])
        hi = float(nodes[a][i1 if cell_range is None else i1 + 1])
        comparison = free_ends = None
        if f0_spec is not None and a != int(f0_spec.normal_axis):
            i0, i1, lo, hi = _f0_solved_bounds(
                mask, a, nodes[a], sizes[a], f0_spec.end_rows[a],
                int(getattr(root.grid, f"pad_{'xyz'[a]}_lo")),
                int(getattr(root.grid, f"pad_{'xyz'[a]}_hi")), periodic)
        if sheet is not None and a != int(sheet.normal_axis) and bounds is not None:
            span = (sheet.solved_spans[a] if hasattr(sheet, 'solved_spans') else
                    solved_sheet_span(mask, a, nodes[a], float(bounds[0][a]),
                        float(bounds[1][a]), float(sim._domain[a]), union=union,
                        volume_edges=volume_edges, periodic=periodic, fed=fed, normal_axis=sheet.normal_axis))
            if span is not None:
                lo, hi = span.lo, span.hi
                comparison = span.comparison_bounds(bounds[0][a], bounds[1][a], sim._domain[a])
                free_ends = (span.free_lo, span.free_hi)
        result.append(ConductorWidthAxis(i0, i1, cell_range, lo, hi, comparison, free_ends))
    return tuple(result)


@dataclass(frozen=True)
class SheetConductors:
    """Sheet products for standalone geometry callers with existing collectors.

    No assembly is performed here. Production callers lend RealizedConductors;
    this smaller view preserves the geometry validator's collector API.
    """
    sheet_impedance: tuple
    periodic: tuple

    def sheet_context(self, pec_edges):
        """Build the operator for the explicitly selected conductor stage."""
        from rfx.materials.thin_conductor import build_sheet_impedance_ctx
        ctx = build_sheet_impedance_ctx(self.sheet_impedance,
                                         pec_edge_masks=pec_edges,
                                         periodic=self.periodic)
        materials = getattr(self, 'materials', None)
        if ctx is not None and materials is not None:
            from rfx.core.yee import lumped_components
            from rfx._traced_check import traced_check
            import jax.numpy as jnp
            for axis, (mask, film) in enumerate(zip(
                    (ctx.mask_ex, ctx.mask_ey, ctx.mask_ez),
                    lumped_components(materials.sigma_film))):
                if mask is None or film is None:
                    continue
                def check(overlap, axis=axis):
                    edges = np.argwhere(np.asarray(overlap))
                    if edges.size:
                        raise ValueError(
                            "an f0 sheet and a lossy film share edges "
                            f"on {'xyz'[axis]} at {edges.tolist()}")
                overlap = jnp.asarray(mask) & (film != 0)
                traced_check("thin conductors", "sheet/film overlap", check, overlap, ordered=True)
        return ctx


@dataclass(frozen=True)
class RealizedConductors(_RealizedPEC):
    lane: str
    grid: object
    materials: object
    pec_cells: object
    pec_edges: object
    sheets: tuple
    sheet_impedance: tuple
    wires: tuple
    periodic: tuple
    geometry_masks: object
    assembly_entries: tuple = ()
    provenance: tuple[ConductorStage, ...] = ()
    assembly: tuple = ()
    pad_fill_findings: tuple = ()
    sheet_operator: object = None
    mode: str = "solve"
    assembly_from_sim: bool = True
    released_port_edges: tuple = ()
    released_cells: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int32))
    _released_port_details: tuple = ()
    _waveguide_port_planes: tuple = ()
    stamped_entities: tuple = ()
    stamp_check_findings: tuple = ()
    design_findings: tuple = ()

    sheet_context = SheetConductors.sheet_context
    width_axes = conductor_width_axes

    @property
    def pec_mask(self):
        return self.pec_cells

    @property
    def sheet_specs(self):
        return self.sheet_impedance

    @property
    def edges(self):
        if self.pec_edges is None:
            return tuple(np.zeros(self.grid.shape, dtype=bool) for _ in range(3))
        return tuple(np.asarray(edge, dtype=bool) for edge in self.pec_edges)


def coax_declared_conductors(sim, grid):
    """Collect the transition's existing assembly, without a second assembly."""
    sheets, wires, masks, entries = [], [], [], []
    assembly = sim._assemble_materials(grid, pec_sheets=sheets, pec_wires=wires,
                                      geometry_masks=masks, assembly_entries=entries)
    return realized_conductors(sim, grid, assembly=assembly, pec_sheets=sheets,
        pec_wires=wires, geometry_masks=masks, assembly_entries=entries)


def coax_kernel_conductors(sim, grid, materials, cells, edges, entities, *, root=None,
                           skip_preflight=False):
    """Own the stamp and union AFTER the calculator's existing port operations.

    No classification, sampling, material update or new acceptance gate occurs.
    A concrete solve lends its last kernel product to realized_geometry(); a
    traced solve must not leave a tracer in the simulation's diagnostic cache.
    """
    if root is None:
        root = realized_conductors(sim, grid, assembly=(materials, None, None, None))
    if root.pec_edges is not None:
        edges = tuple(np.asarray(e) | np.asarray(m) for e, m in zip(edges, root.pec_edges))
    cells = cells if root.pec_cells is None else np.asarray(root.pec_cells) | cells
    root = replace(root, materials=materials, pec_cells=cells, pec_edges=edges,
        stamped_entities=tuple(entities), provenance=root.provenance + (
            ConductorStage(tuple(e.entity_id for e in entities), 'coax stamp'),))
    sim._coax_geometry = None
    sim._realized_geometry_record = None
    if not _traced_product(root):
        from rfx.model.coax import stamp_check_findings
        ctx = None
        try:
            ctx = sim._campaign_ctx()
            if not skip_preflight:
                root = replace(root, stamp_check_findings=stamp_check_findings(sim, root))
        except Exception as exc:
            root = replace(root, stamp_check_findings=root.stamp_check_findings + (
                ('read-failed', (type(exc).__name__, str(exc))),))
        if ctx is not None:
            sim._coax_geometry = (ctx, root)
    return root


def realized_conductors(sim, grid, *, nonuniform=False, assembly=None,
                        pec_sheets=(), pec_wires=(), sheet_specs=(),
                        geometry_masks=(), assembly_entries=(),
                        pad_fill_findings=(), periodic=None, mode="solve", n_devices=1):
    """Build from production collectors, assembling only when none were supplied."""
    from rfx.boundaries.pec import realized_pec_edge_masks
    if mode not in ("solve", "audit"):
        raise ValueError(f"Unknown conductor assembly mode: {mode}")
    if assembly is None:
        pec_sheets, pec_wires, sheet_specs = [], [], []
        geometry_masks, assembly_entries, pad_fill_findings = [], [], []
        kwargs = dict(pec_sheets=pec_sheets, pec_wires=pec_wires,
                      sheet_specs=sheet_specs, geometry_masks=geometry_masks,
                      assembly_entries=assembly_entries, n_devices=n_devices)
        if nonuniform:
            assembly = sim._assemble_materials_nu(grid, **kwargs)
        else:
            assembly = sim._assemble_materials(grid, _check_declared_span=False, **kwargs)
        from rfx.model.pad_fill import check_pad_fill
        check_pad_fill(sim, grid, geometry_masks,
                       record=pad_fill_findings if mode == "audit" else None)
    periodic = tuple(sim._periodic_flags() if periodic is None else periodic)
    cells = assembly[3]
    edges = None
    if cells is not None or pec_sheets or pec_wires:
        edges = realized_pec_edge_masks(cells, sheets=pec_sheets,
                                       wires=pec_wires, periodic=periodic)
    labels = tuple(f'{prefix}[{i}]' for collection, prefix in
                   ((sim._geometry, 'geometry'), (sim._thin_conductors, 'thin_conductor'),
                    (getattr(sim, '_pinned_sheets', ()), 'pinned_sheet'))
                   for i, _ in enumerate(collection))
    result = RealizedConductors(
        'nonuniform' if nonuniform else 'uniform', grid, assembly[0], cells,
        edges, tuple(pec_sheets), tuple(sheet_specs), tuple(pec_wires), periodic,
        dict(geometry_masks), tuple(assembly_entries),
        (ConductorStage(labels, 'assembly'),), tuple(assembly), tuple(pad_fill_findings), mode=mode)
    return _with_sheet_spans(sim, result)


def publish_port_release(conductors, cells, *, component, entity_id, kind):
    """Publish the incident cells once, in realized indices, for each port edge."""
    edges = list(conductors.released_port_edges)
    known = set(edges)
    details = list(conductors._released_port_details)
    released = set(map(tuple, conductors.released_cells.tolist()))
    transverse = [a for a in range(3) if a != "xyz".index(component[1])]
    shape = conductors.grid.shape
    for cell in cells:
        edge = (entity_id, component, tuple(map(int, cell)))
        if edge in known:
            continue
        incident = set()
        for da, db in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
            index = list(edge[2])
            index[transverse[0]] += da
            index[transverse[1]] += db
            for axis in range(3):
                if conductors.periodic[axis] or shape[axis] == 1:
                    index[axis] %= shape[axis]
            if all(0 <= index[a] < shape[a] for a in range(3)):
                incident.add(tuple(index))
        edges.append(edge)
        known.add(edge)
        details.append((entity_id, kind, component, edge[2], tuple(sorted(incident))))
        released.update(incident)
    return replace(conductors, released_port_edges=tuple(edges),
                   released_cells=np.asarray(sorted(released), dtype=np.int32).reshape(-1, 3),
                   _released_port_details=tuple(details))


def waveguide_port_stage(conductors, config, entity_id):
    """Publish R1's transverse E entries from a compiled waveguide config."""
    if isinstance(config, list):
        for mode_config in config:
            conductors = waveguide_port_stage(conductors, mode_config, entity_id)
        return conductors
    normal = 'xyz'.index(config.normal_axis)
    transverse = [a for a in range(3) if a != normal]
    edges = []
    for p in sorted({config.x_index, config.x_index + 1, config.ref_x, config.probe_x}):
        for u in range(config.u_lo, config.u_hi):
            for v in range(config.v_lo, config.v_hi):
                index = [0, 0, 0]
                index[normal] = p
                index[transverse[0]], index[transverse[1]] = u, v
                edges.append(tuple(index))
    for component in (config.e_u_component, config.e_v_component):
        conductors = publish_port_release(conductors, edges, component=component,
                                         entity_id=entity_id, kind='waveguide')
    planes = dict(conductors._waveguide_port_planes)
    planes[entity_id] = (config.normal_axis, config.x_index, config.ref_x, config.probe_x)
    return replace(conductors, _waveguide_port_planes=tuple(planes.items()))


def _port_release_description(conductors):
    descriptions = []
    waveguides = {}
    for entity, kind, component, edge, cells in conductors._released_port_details:
        if kind == 'waveguide':
            waveguides.setdefault(entity, set()).update(cells)
        else:
            descriptions.append(f"{entity} ({kind}, {component}, edge {edge}): cells {cells}")
    planes = dict(conductors._waveguide_port_planes)
    for entity, cells in waveguides.items():
        axis, source, reference, measurement = planes[entity]
        descriptions.append(
            f"{entity} (waveguide, normal {axis}, source {source} (also {source + 1}), "
            f"reference {reference}, measurement {measurement}): {len(cells)} cells")
    return "; ".join(descriptions)


def prepare_port_occupancy(conductors, waveguide_configs, occupancy, design, *, sim=None, sheet_operator=None):
    """Publish only for an occupancy input, then release and guard its box."""
    if occupancy is not None or design is not None:
        for i, config in enumerate(waveguide_configs):
            conductors = waveguide_port_stage(conductors, config, f'waveguide_port[{i}]')
    if sim is not None:
        from rfx.model.design_admission import publish_declarations, source_findings
        if occupancy is not None or design is not None:
            conductors = publish_declarations(conductors, sim, design, sheet_operator, occupancy=occupancy)
        conductors = replace(conductors, design_findings=source_findings(sim, conductors, occupancy, design))
    occupancy = release_port_occupancy(occupancy, conductors)
    refuse_port_design_box(design, conductors)
    return conductors, occupancy


def release_port_occupancy(occupancy, conductors):
    """Keep port edges free of design metal; warn once for this application."""
    if occupancy is None:
        return occupancy
    if np.shape(occupancy) != conductors.grid.shape:
        raise ValueError(f"Port occupancy shape {np.shape(occupancy)} differs from "
                         f"conductors grid shape {conductors.grid.shape}.")
    if not len(conductors.released_cells):
        return occupancy
    import warnings
    from rfx.core.jax_utils import is_tracer
    indices = tuple(conductors.released_cells.T)
    if is_tracer(occupancy) or np.any(np.asarray(occupancy)[indices] != 0):
        ports = _port_release_description(conductors)
        warnings.warn("Port occupancy release: " + ports
                      + ". Each driven edge is kept free of design metal.",
                      UserWarning, stacklevel=2)
    import jax.numpy as jnp
    return jnp.asarray(occupancy).at[indices].set(0.0)


def refuse_port_design_box(design, conductors):
    """The occupancy box writes after release and must not overwrite it (#1183)."""
    if design is None or not len(conductors.released_cells):
        return
    from rfx.model.design_admission import cells_in_box
    bounds = design.bounds
    hits = cells_in_box(conductors.released_cells.tolist(), bounds, conductors.grid.shape)
    if hits:
        ports = sorted({entity for entity, _, _, _, cells in conductors._released_port_details
                        if any(c in hits for c in cells)})
        sample = sorted(hits)[:6]
        remainder = f" and {len(hits) - 6} more" if len(hits) > 6 else ""
        raise ValueError(f"{', '.join(ports)}: the design occupancy box (cells {bounds}) "
                         f"covers port-cleared cell(s) {sample}{remainder}. A port forces the "
                         "occupancy to zero around its driven edges before the run. "
                         "Move the box off the port, or use pec_occupancy_override.")


def clear_conductor_edges(conductors, cells, *, component, entity_id, port_kind, clear_cells=False, release_edges=True, released_edges=None):
    """One persistent port-clearing stage; the incoming object stays unchanged."""
    from rfx.boundaries.pec import clear_edges
    cells = tuple(tuple(int(i) for i in cell) for cell in cells)
    conductors = publish_port_release(conductors,
        cells if released_edges is None else released_edges, component=component,
        entity_id=entity_id, kind=port_kind)
    if conductors.pec_edges is None:
        return conductors
    edges = (clear_edges(conductors.pec_edges, cells, component=component)
             if release_edges else conductors.pec_edges)
    pec_cells = conductors.pec_cells
    if clear_cells and pec_cells is not None:
        for cell in cells:
            pec_cells = pec_cells.at[cell].set(False)
    return replace(conductors, pec_cells=pec_cells, pec_edges=edges, provenance=conductors.provenance + (
        ConductorStage((entity_id,), 'port-edge-clearing' if release_edges else 'port-cell-clearing', component, cells),))


def lumped_port_stage(conductors, port, entity_id, *, clear_cells=False):
    """Release the single driven PEC edge after loading a lumped port."""
    if port.impedance <= 0 or port.extent is not None:
        return conductors
    grid = conductors.grid
    cell = field_index(grid, port.position, port.component)
    return clear_conductor_edges(conductors, [cell], component=port.component,
                                 entity_id=entity_id, port_kind="lumped", clear_cells=clear_cells)


def _same_grid(a, b):
    """Only reuse a production assembly on the identical realized lattice."""
    from rfx.core.jax_utils import is_tracer
    if a.shape != b.shape or a.axis_pads != b.axis_pads:
        return False
    for axis in range(3):
        x, y = a.cells(axis), b.cells(axis)
        if is_tracer(x) or is_tracer(y) or not np.array_equal(x, y):
            return False
    return True


def admit_adi_sources(sim, grid, pec_edges):
    """Cover direct ADI material entries before their existing interior guard."""
    from types import SimpleNamespace
    from rfx.model.source_admission import refuse_dead_soft_sources
    refuse_dead_soft_sources(sim, SimpleNamespace(
        grid=grid, lane='uniform', pec_edges=pec_edges,
        assembly_entries=(), sheets=()))
    sim._validate_adi_interior_pec(pec_edges)


def solve_conductors(sim, grid, *, nonuniform=False, preflight=None, n_devices=1):
    """Assemble once for execution and lend the pre-port object to its checks."""
    import jax
    from rfx.model.source_admission import refuse_dead_soft_sources
    with jax.ensure_compile_time_eval():
        root = realized_conductors(sim, grid, nonuniform=nonuniform, mode="solve", n_devices=n_devices)
        refuse_dead_soft_sources(sim, root)
    if preflight is not None:
        sim._auto_preflight(conductors=root, **preflight)
    return root


class PreparedSolve:
    """Execution-local ownership, transferred once to the selected runner.

    With preflight disabled, assembly remains lazy until after lane admission.
    The empty handoff can stay in dispatch frames without pinning dense arrays
    while a distributed runner stages and drops its full-domain products.
    """
    def __init__(self, sim):
        self.sim = sim
        self._product = None
        self._taken = False
        self.diagnostics = ()

    def realized(self, *, n_devices=1):
        if self._taken:
            raise RuntimeError("Solve assembly was already consumed")
        if self._product is None:
            nonuniform = self.sim._uses_nonuniform_mesh
            grid = (self.sim._build_nonuniform_grid() if nonuniform
                    else self.sim._build_grid())
            self._product = solve_conductors(self.sim, grid, nonuniform=nonuniform, n_devices=n_devices)
        return self._product

    def take(self, *, n_devices=1):
        product = self.realized(n_devices=n_devices)
        self._product = None
        self._taken = True
        return product


def prepare_solve(sim, *, distributed=False):
    """Keep solve ownership local to this execution, never in audit caches."""
    if distributed:
        sim._pf_campaign_ctx = None
        sim._realized_geometry_record = None
    return PreparedSolve(sim)


def assembled_materials(root, **collectors):
    """Unpack explicitly supplied solve products into legacy runner arguments."""
    if root.mode != "solve":
        raise ValueError("A runner cannot consume audit-mode conductors")
    for name, out in collectors.items():
        if out is not None:
            value = getattr(root, {'pec_sheets': 'sheets', 'pec_wires': 'wires',
                                  'sheet_specs': 'sheet_impedance'}.get(name, name))
            out.extend(value.items() if name == 'geometry_masks' else value)
    return root.assembly


def kernel_conductors(sim, grid, materials, pec_cells, sheets=(), wires=(),
                      *, periodic, sheet_specs=(), root=None):
    """Carry explicit solve products forward, retaining material/PEC overrides."""
    if root is not None and root.mode != "solve":
        raise ValueError("A runner cannot consume audit-mode conductors")
    if (root is not None and _same_grid(root.grid, grid)
            and pec_cells is root.pec_cells
            and tuple(map(id, sheets)) == tuple(map(id, root.sheets))
            and tuple(map(id, wires)) == tuple(map(id, root.wires))
            and tuple(periodic) == root.periodic):
        return replace(root, materials=materials, sheet_impedance=tuple(sheet_specs))
    # Internal override/S-parameter entry points already have assembled products.
    import jax
    from rfx.model.source_admission import refuse_dead_soft_sources
    with jax.ensure_compile_time_eval():
        obj = realized_conductors(sim, grid, nonuniform=hasattr(grid, 'dx_arr'),
            assembly=(materials, None, None, pec_cells), pec_sheets=sheets,
            pec_wires=wires, sheet_specs=sheet_specs, periodic=periodic,
            geometry_masks=() if root is None else root.geometry_masks.items(),
            assembly_entries=() if root is None else root.assembly_entries)
        refuse_dead_soft_sources(sim, obj)
    return (replace(obj, assembly_from_sim=False) if root is None else
            replace(obj, assembly=root.assembly, assembly_from_sim=root.assembly_from_sim))


def at_kernel(sim, conductors, *, lane, pec_edges, sheet_operator=None, compact=True):
    """Record/replay the exact object passed to the stepping call."""
    if conductors.mode != "solve":
        raise ValueError("A runner cannot consume audit-mode conductors")
    if pec_edges is not conductors.pec_edges:
        raise ValueError("PEC edges changed outside a conductor stage")
    if sheet_operator is not conductors.sheet_operator:
        conductors = replace(conductors, sheet_operator=sheet_operator)
    from rfx import _realized
    if _realized.ACTIVE is not None:
        edges = conductors.pec_edges
        if edges is not None:
            edges = _realized.ACTIVE.apply('conductors', 'pec_edges', edges)
            if edges is not conductors.pec_edges and conductors.sheet_operator is not None:
                conductors = replace(conductors, sheet_operator=conductors.sheet_context(edges))
            conductors = replace(conductors, pec_edges=edges)
            _realized.ACTIVE.observe('conductors', dict(pec_edges=edges))
    if _traced_product(conductors):
        return conductors, None
    from rfx.realized_geometry import record_from_conductors
    record = (record_from_conductors(sim, conductors, lane=lane, compact=compact)
              if lane.startswith('run_') else None)
    return conductors, record


def _with_sheet_spans(sim, conductors):
    """Annotate collectors through the existing interior and solved-span rules."""
    if not conductors.sheets or _traced_product(conductors):
        return conductors
    from dataclasses import fields
    from rfx.preflight.realization import _CampaignStaticsContext, _EntryRealization, _shape_bounds
    from rfx.realized_geometry import _node_arrays
    from rfx.mesh_edges import sheet_continuation_masks, solved_sheet_span
    from rfx.core.jax_utils import is_tracer
    grid = conductors.grid
    if any(is_tracer(getattr(grid, name, None)) for name in ('dx_arr', 'dy_arr', 'dz')):
        return conductors
    _, nodes = _node_arrays(sim, grid, conductors.lane == 'nonuniform')
    products = {key: (cells, sheet, wire, shape)
                for key, cells, sheet, wire, shape in conductors.assembly_entries}
    ctx = object.__new__(_CampaignStaticsContext)
    ctx.grid, ctx.periodic, ctx._entries = grid, conductors.periodic, []
    owners = {}
    for collection, prefix in ((sim._geometry, 'geometry'), (sim._thin_conductors, 'thin_conductor')):
        for i, entry in enumerate(collection):
            product = products.get(id(entry))
            if product is None:
                continue
            cells, sheet, wire, shape = product
            label = f'{prefix}[{i}]'
            kind = 'volume' if cells is not None else 'sheet' if sheet is not None else 'wire' if wire is not None else 'lossy'
            ctx._entries.append(_EntryRealization(label=label, name=label, shape=entry.shape,
                kind=kind, cells=cells, sheet=sheet, wire=wire, solved_shape=shape))
            if sheet is not None:
                owners[id(sheet)] = label
    interior = ctx.interior_pec_entries()
    union, volume_edges = sheet_continuation_masks(interior, ctx.periodic, grid.shape)
    pinned = getattr(sim, '_pinned_sheets', ())
    if pinned:
        from rfx.geometry.rasterize_grid import interior_lattice_mask
        pin_specs = conductors.sheets[-len(pinned):]
        pin_masks = [interior_lattice_mask(sheet.footprint, grid) for sheet in pin_specs]
        from rfx.boundaries.pec import _sheet_edge_masks
        report_sheets = [entry.sheet for entry in interior if entry.sheet is not None]
        report_sheets += [replace(sheet, footprint=mask)
                          for sheet, mask in zip(pin_specs, pin_masks, strict=True)]
        union = _sheet_edge_masks(report_sheets, grid.shape, ctx.periodic)
    fed = port_fed_nodes(sim, conductors)
    spans = {}
    for entry in interior:
        bounds = _shape_bounds(entry.shape)
        if entry.sheet is None or bounds is None:
            continue
        spans[entry.label] = tuple(None if a == entry.sheet.normal_axis else solved_sheet_span(
            entry.sheet.footprint, a, nodes[a], float(bounds[0][a]), float(bounds[1][a]),
            float(sim._domain[a]), union=union, volume_edges=volume_edges, fed=fed,
            normal_axis=entry.sheet.normal_axis,
            periodic='xyz'[a] in getattr(grid, 'periodic_axes', '')) for a in range(3))
    if pinned:
        for i, (sheet, footprint) in enumerate(zip(pin_specs, pin_masks, strict=True)):
            label = f'pinned_sheet[{i}]'
            owners[id(sheet)] = label
            unwrapped = sheet.unwrapped_footprint
            occupied = np.where(np.asarray(sheet.footprint if unwrapped is None else unwrapped))
            if occupied[0].size:
                spans[label] = tuple(None if a == sheet.normal_axis else solved_sheet_span(
                    footprint, a, nodes[a], float(nodes[a][occupied[a].min()]),
                    float(nodes[a][occupied[a].max()]), float(sim._domain[a]),
                    union=union, volume_edges=volume_edges, fed=fed,
                    normal_axis=sheet.normal_axis,
                    periodic='xyz'[a] in getattr(grid, 'periodic_axes', '')) for a in range(3))
    replacements = {}
    for sheet in conductors.sheets:
        label = owners.get(id(sheet), '')
        replacements[id(sheet)] = RealizedSheet(
            **{f.name: getattr(sheet, f.name) for f in fields(SheetSpec)},
            entity_id=label, solved_spans=spans.get(label, (None, None, None)))
    return replace(conductors, sheets=tuple(replacements[id(s)] for s in conductors.sheets),
        assembly_entries=tuple((key, cells, replacements.get(id(sheet), sheet), wire, shape)
            for key, cells, sheet, wire, shape in conductors.assembly_entries))


def preview_port_stages(sim, conductors):
    """Apply the default run's port stages for a standalone geometry view."""
    grid = conductors.grid
    done = {entity for stage in conductors.provenance
            if stage.stage == 'port-edge-clearing' for entity in stage.entity_ids}
    for i, port in enumerate(sim._ports):
        label = f'port[{i}]'
        if label not in done:
            if port.impedance > 0 and port.extent is not None:
                from rfx.sources.sources import wire_port_from_entry, _wire_port_live_cells
                try:
                    cells, live, _ = _wire_port_live_cells(grid, wire_port_from_entry(port), conductors.pec_edges)
                except ValueError:
                    # A geometry preview still exists when the wire cannot drive.
                    continue
                conductors = publish_port_release(conductors, [c for c, on in zip(cells, live) if on],
                    component=port.component, entity_id=label, kind="wire")
            else:
                conductors = lumped_port_stage(conductors, port, label)
    if sim._msl_ports:
        from rfx.sources.msl_port import (
            msl_port_from_entry, _msl_yz_cells, msl_normal_component)
        for i, entry in enumerate(sim._msl_ports):
            label = f'msl_port[{i}]'
            if label not in done:
                port = msl_port_from_entry(entry)
                support = list(_msl_yz_cells(grid, port))
                if conductors.lane == 'nonuniform' or getattr(entry, "mode", "laplace") != "uniform":
                    from rfx.sources.msl_port import compute_msl_mode_profile, msl_cross_section_span, msl_cell
                    span = msl_cross_section_span(grid, port)
                    centre = msl_cell(entry.direction, span["i_feed"], span["w_centre"],
                                      (span["n_lo"] + span["n_hi"]) // 2)
                    eps = entry.eps_r_sub if entry.eps_r_sub is not None else float(conductors.materials.eps_r[centre])
                    support = compute_msl_mode_profile(grid, port, eps)["cell_indices"]
                conductors = clear_conductor_edges(conductors, _msl_yz_cells(grid, port),
                    component=msl_normal_component(port), entity_id=label, port_kind="microstrip", released_edges=support)
    return conductors


def port_fed_nodes(sim, conductors):
    """Both nodes of lumped/wire/MSL feed edges, for solved sheet ends.

    Use the port stages' edge rules without computing a Laplace profile.
    MSL clearing edges already cover only the trace's width nodes. Temporary
    lumped stages never replace the solver's conductor products.
    """
    # A failed audit assembly can still report individually realized sheets.
    # There is no port-stage product to read; retain their unfed spans while
    # the existing assembly/scan diagnostics report what could not be built.
    if conductors is None:
        return None
    if not (getattr(sim, '_ports', ()) or getattr(sim, '_msl_ports', ())):
        return None
    grid = conductors.grid
    fed = np.zeros(grid.shape, dtype=bool)
    edges = []
    staged = conductors
    for i, port in enumerate(getattr(sim, '_ports', ())):
        label = f'port[{i}]'
        if port.impedance > 0 and port.extent is not None:
            from rfx.sources.sources import wire_port_from_entry, _wire_port_live_cells
            try:
                cells, live, _ = _wire_port_live_cells(
                    grid, wire_port_from_entry(port), staged.pec_edges)
            except ValueError:
                # Match preview: a wire that cannot drive contributes no nodes.
                continue
            edges.extend((port.component, cell) for cell, on in zip(cells, live) if on)
        else:
            staged = lumped_port_stage(staged, port, label)
            edges.extend((component, edge) for entity, kind, component, edge, _
                         in staged._released_port_details if entity == label and kind == 'lumped')
    if getattr(sim, '_msl_ports', ()):
        from rfx.sources.msl_port import msl_port_from_entry, _msl_yz_cells, msl_normal_component
        for entry in sim._msl_ports:
            port = msl_port_from_entry(entry)
            edges.extend((msl_normal_component(port), cell) for cell in _msl_yz_cells(grid, port))
    for component, edge in edges:
        axis = 'xyz'.index(component[1])
        for offset in (0, 1):
            node = list(edge)
            node[axis] += offset
            if conductors.periodic[axis] or grid.shape[axis] == 1:
                node[axis] %= grid.shape[axis]
            if all(0 <= node[a] < grid.shape[a] for a in range(3)):
                fed[tuple(node)] = True
    return fed


def _traced_product(conductors):
    import jax
    from rfx.core.jax_utils import is_tracer
    return any(is_tracer(leaf) for leaf in jax.tree.leaves(
        (conductors.materials, conductors.pec_edges, conductors.assembly)))

def auto_preflight(sim, *, skip=False, context="forward", check_ntff=True,
                   conductors=None, prepare=False, distributed=False, tfsf_material_overrides=()):
    """Emit a UserWarning if preflight finds issues (issue #66).

    Called automatically at the start of ``forward()``, ``optimize()``,
    and ``topology_optimize()`` so users discover physics violations
    (under-resolved mesh, geometry in CPML, probe in PEC, ...) before
    spending minutes of GPU compute. Pass ``skip_preflight=True`` at
    the call site to opt out (tests, already-validated configs).

    ``check_ntff`` gates the NTFF check family. ``run()`` passes
    ``check_ntff="advisory"`` (issue #303): the λ/4 near-field gap
    and sub-wavelength ground-plane (issue #334) advisories are
    physics-relevant to any far-field computation and
    run() is the common NTFF entry point, so they are surfaced as
    warnings — but only ``forward(port_s11_freqs=...)`` / ``optimize``
    (the inverse-design entry points) hard-fail on the PEC-overlap
    error. Previously run() skipped the family entirely and still
    printed "All checks passed", which contradicted explicit
    ``sim.preflight()`` output on the same configuration.

    ``prepare=True`` also returns an execution-local solve handoff. With
    skipped preflight it stays lazy until lane admission; otherwise checks
    read its assembled product before the selected runner takes ownership.
    """
    assembly = prepare_solve(sim, distributed=distributed) if prepare else None
    if skip:
        return assembly if prepare else PreflightReport()
    if assembly is not None:
        conductors = assembly.realized()
    # A validator bug must propagate, rather than become an advisory.
    if (getattr(sim, "_tfsf", None) is not None or sim._waveguide_ports) and any(v is not None for v in tfsf_material_overrides):
        issues = sim._preflight_impl(strict=False, check_ntff=check_ntff,
            _conductors=conductors, _tfsf_material_overrides=tfsf_material_overrides)
    else:
        issues = sim._preflight_impl(strict=False, check_ntff=check_ntff,
                                     _conductors=conductors)
    # gate -> this helper -> facade -> public entry -> user
    sim._run_preflight_gate(issues, context=context, stacklevel=5)
    if assembly is not None:
        assembly.diagnostics = tuple(getattr(issues, "diagnostics", ()))
    return assembly if prepare else issues


def bind_solve_call(assembly, call):
    """Consume a graded solve before ring-down adds internal probes."""
    from functools import partial
    root = assembly.take()
    return partial(
        call,
        conductors=root,
        diagnostics=assembly.diagnostics,
    ), root.grid


def uniform_solve_inputs(assembly):
    """Return the solve grid, material tuple and legacy sheet/wire collectors."""
    root = assembly.take()
    sheets, wires, specs = [], [], []
    materials = assembled_materials(root, pec_sheets=sheets, pec_wires=wires,
                                    sheet_specs=specs)
    return root, root.grid, materials, specs, sheets, wires


def early_run_record(sim, root, lane):
    """ADI/subgrid record timing remains at assembly, as before this move."""
    if lane == "run_uniform":
        return None
    from rfx.realized_geometry import record_from_assembly
    return record_from_assembly(sim, root.grid, root.materials, root.pec_cells,
        root.sheets, root.wires, list(root.geometry_masks.items()),
        list(root.assembly_entries), lane=lane, conductors=root)


def forward_products(sim, grid, materials, cells, sheets, wires, periodic, root,
                     sheet_operator=None):
    """Bind overrides while keeping the original substrate for MSL launches."""
    unbound = root is None
    drawn = None if unbound else root.assembly[0]
    root = kernel_conductors(sim, grid, materials, cells, sheets, wires,
        periodic=periodic, root=root,
        sheet_specs=() if root is None else root.sheet_impedance)
    return replace(root, sheet_operator=sheet_operator), drawn, root.pec_edges


def forward_lumped_port_stage(root, port, entity_id):
    """Keep forward's cell/occupancy products alongside the owned edge stage."""
    root = lumped_port_stage(root, port, entity_id, clear_cells=True)
    return root, root.pec_cells, root.pec_edges


def forward_port_stage(root, cells, component, entity_id, *,
                       port_kind, release_edges=True, skip_empty=False, released_edges=None):
    """Release only the driven component; preserve the wire's edge support."""
    if not skip_empty or cells:
        root = clear_conductor_edges(root, cells, component=component,
            entity_id=entity_id, port_kind=port_kind, clear_cells=True, release_edges=release_edges,
            released_edges=released_edges)
    return root, root.pec_cells, root.pec_edges


def drawn_launch_eps(sim, grid, materials):
    """Read the drawn substrate, including legacy direct-array entry points."""
    if materials is None:
        materials = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0]
    return materials.eps_r


def forward_kernel_inputs(sim, root, lane, edges, sheet_operator):
    root, _ = at_kernel(sim, root, lane=lane, pec_edges=edges,
                        sheet_operator=sheet_operator)
    return root, root.pec_edges, root.sheet_operator


def distributed_solve_inputs(sim, grid, assembly, skip, gather, *, n_devices=1):
    """Consume the dispatch handoff without retaining it during slab staging."""
    sim._pf_campaign_ctx = None
    sim._realized_geometry_record = None
    if assembly is None:
        root = solve_conductors(sim, grid, nonuniform=True, n_devices=n_devices)
        report = sim._auto_preflight(
            conductors=root,
            skip=skip,
            context="run" if gather else "forward",
            check_ntff="advisory" if gather else True,
        )
        diagnostics = tuple(getattr(report, "diagnostics", ()))
    else:
        diagnostics = assembly.diagnostics
        root = assembly.take(n_devices=n_devices)
    sheets, wires = [], []
    materials = assembled_materials(root, pec_sheets=sheets, pec_wires=wires)
    return root, materials, sheets, wires, diagnostics


def distributed_kernel_inputs(sim, root, grid, materials, cells, sheets, wires, gather):
    root = kernel_conductors(sim, grid, materials, cells, sheets, wires,
                            periodic=(False, False, False), root=root)
    root, record = at_kernel(sim, root,
        lane="run_distributed_nu" if gather else "fwd_distributed_nu",
        pec_edges=root.pec_edges)
    return root.pec_edges, record


def stage_distributed_occupancy(override, grid, mesh):
    """Cut a PEC occupancy override in the layout of the step ``grid`` recorded; None stays None."""
    from rfx.stepping.slab import cut
    if override is None:
        return None
    return cut(override, grid.layout, "pec_occupancy", mesh=mesh, uniform=grid.step_kind == 'wrapped')


def stage_distributed_edges(edges, grid, mesh, multiprocess, *, inputs=None):
    """Stage final edge components; traced geometry remains single-process."""
    import jax
    from rfx.stepping.slab import Slab, cut
    if edges is None:
        return None
    if multiprocess and any(isinstance(edge, jax.core.Tracer) for edge in edges):
        raise ValueError("a traced pec_mask_override is supported in one process only")
    from rfx import _realized
    if inputs is not None and _realized.ACTIVE is None and grid.step_kind != 'legacy':
        from rfx.stepping.slab_setup import stage_pec
        cells, sheets, wires = inputs
        return stage_pec(cells, sheets, wires, grid.global_grid, Slab.from_grid(grid), mesh)
    return cut(edges, Slab.from_grid(grid), "pec_edge", mesh=mesh)


def distributed_mask_spec(mask):
    """Partition either a cell-mask array or three precomputed edge arrays."""
    from jax.sharding import PartitionSpec as P
    return (P("x"),) * 3 if isinstance(mask, tuple) else P("x")


def distributed_mask_edges(mask):
    """Use staged conductor edges, or realize legacy slab-local cell masks.

    The slab's x ghosts carry the seam neighbours; the caller gates both
    ghost rows out before applying the masks. The y/z faces do not wrap.
    """
    from rfx.boundaries.pec import realized_pec_edge_masks
    return (mask if isinstance(mask, tuple) else
            realized_pec_edge_masks(mask, periodic=(True, False, False)))


def forward_sheet_context(root, cells):
    """The outer forward context precedes port clearing and includes overrides."""
    from rfx.boundaries.pec import realized_pec_edge_masks
    edges = (None if cells is None and not root.sheets and not root.wires else
             realized_pec_edge_masks(cells, sheets=root.sheets, wires=root.wires,
                                     periodic=root.periodic))
    return root.sheet_context(edges)


def _f0_solved_bounds(mask, axis, nodes, sizes, end_rows, pad_lo, pad_hi, periodic):
    """In-plane node range and ends of an f0 sheet as the sheet operator solves them.

    A tangential row at node ``i`` loads its dual cell, half of the cell on
    each side, times the row's weight on the sheet conductance: 1, or
    ``END_ROW_WEIGHT`` on a row marked as lying on a drawn face. The solved
    end is therefore ``weight * dual - inside half`` beyond the end node
    (zero at weight 0.5 between equal cells). On a non-periodic axis an end
    on the domain face has no outside cell and stays on its node. A periodic
    axis has no wall: the occupied rows are read as one run around the
    period (a strip ending on the seam ends at node N, one that starts there
    starts at node 0), and a sheet spanning the whole period has no end.
    """
    from rfx.core.jax_utils import is_tracer
    from rfx.materials.thin_conductor import END_ROW_WEIGHT
    n = len(sizes)
    rows = np.flatnonzero(mask.any(axis=tuple(b for b in range(mask.ndim) if b != axis)))
    marked = (np.zeros(n, dtype=bool) if end_rows is None or is_tracer(end_rows)
              else np.asarray(end_rows, dtype=bool))
    i0, i1 = int(rows[0]), int(rows[-1])
    free_lo, free_hi = i0 > pad_lo, i1 < n - pad_hi - 1
    if periodic:
        occupied = np.zeros(n, dtype=bool)
        occupied[rows % n] = True
        if occupied.all():
            return 0, n, float(nodes[0]), float(nodes[n])
        # Where the one occupied run around the period begins: its rising edge,
        # or node 0 when the run starts there without wrapping.
        rising = np.flatnonzero(np.diff(occupied.astype(np.int8)) == 1) + 1
        if rising.size + int(occupied[0] and not occupied[-1]) == 1:
            i0 = int(rising[0]) if rising.size else 0
            i1 = i0 + int(occupied.sum()) - 1
        free_lo = free_hi = True

    def at(i):
        return float(nodes[i % n]) + (i // n) * float(nodes[n] - nodes[0])

    lo, hi = at(i0), at(i1)
    if free_lo:
        w = END_ROW_WEIGHT if marked[i0 % n] else 1.0
        lo -= w * 0.5 * float(sizes[(i0 - 1) % n] + sizes[i0 % n]) - 0.5 * float(sizes[i0 % n])
    if free_hi:
        w = END_ROW_WEIGHT if marked[i1 % n] else 1.0
        hi += w * 0.5 * float(sizes[(i1 - 1) % n] + sizes[i1 % n]) - 0.5 * float(sizes[(i1 - 1) % n])
    return i0, i1, lo, hi


_SHEET_LAYOUT_REPORTS = (
    "sheet span report", "sheet preflight advisories", "dead soft-source check",
    "line-stub admission", "sheet-in-dielectric warning",
)


def prepare_sheet_layout(sim, layout, *, distributed):
    """Make an execution-local static declaration with addressed sheets removed.

    Static assembly and host reports see only fixed sheets. The addressed
    declarations and their traced values travel separately to the graded keep.
    The caller's declaration and its cached reports are never mutated.
    """
    if layout is None:
        return sim, None
    from copy import copy
    from rfx.runners._admission import admit_sheet_layout_lane
    from rfx.model.occupancy import admit_sheet_layout
    admit_sheet_layout_lane(sim, distributed=distributed)
    addressed = admit_sheet_layout(sim._pinned_sheets, layout)
    if not addressed:
        return sim, addressed
    fixed = copy(sim)
    indices = {index for index, _, _ in addressed}
    fixed._pinned_sheets = [sheet for index, sheet in enumerate(sim._pinned_sheets)
                            if index not in indices]
    fixed._pf_campaign_ctx = None
    return fixed, addressed


def with_sheet_layout_reports(result, layout):
    """Name each excluded declared-footprint report once in the forward result."""
    skipped = tuple((index, report) for index, _, _ in (layout or ())
                    for report in _SHEET_LAYOUT_REPORTS)
    return result._replace(sheet_reports_skipped=skipped or None)


def refuse_layout_drives(edge_sets, sources, waveguide_ports):
    """An addressed E edge cannot own a source/port, even at layout value zero."""
    from rfx.preflight._common import _H_LOOP
    driven = set()
    for i, j, k, component, *_ in sources:
        if component.startswith('e'):
            driven.add(('xyz'.index(component[1]), int(i), int(j), int(k)))
        else:
            for c, di, dj, dk in _H_LOOP[component]:
                driven.add((c, int(i) + di, int(j) + dj, int(k) + dk))
    configs = [mode for config in (waveguide_ports or ())
               for mode in (config if isinstance(config, list) else (config,))]
    for config in configs:
        normal = 'xyz'.index(config.normal_axis)
        axes = [a for a in range(3) if a != normal]
        for plane in (config.x_index, config.x_index + 1):
            for u in range(config.u_lo, config.u_hi):
                for v in range(config.v_lo, config.v_hi):
                    cell = [0, 0, 0]
                    cell[normal], cell[axes[0]], cell[axes[1]] = plane, u, v
                    for component in (config.e_u_component, config.e_v_component):
                        driven.add(('xyz'.index(component[1]), *cell))
    for index, sheet, _tangent, axis, indices, _values in edge_sets:
        edges = {(axis, *cell) for cell in zip(*(v.ravel().tolist() for v in indices))}
        overlap = sorted(edges & driven)
        if overlap:
            raise ValueError(f"pinned_sheet_layout sheet {index} ({sheet.name!r}): "
                             f"source or port drives addressed edge {overlap[0]}; "
                             "move the drive off the layout switch on the graded lane")


def _layout_edges(layout, grid):
    from rfx.model.occupancy import layout_edge_indices
    for index, sheet, _ in layout or ():
        for tangent in (0, 1):
            axis, indices = layout_edge_indices(grid, sheet, tangent)
            yield index, sheet, axis, indices


def _layout_stage_error(index, sheet, stage, axis, edge):
    raise ValueError(
        f"pinned_sheet_layout sheet {index} ({sheet.name!r}): "
        f"{stage} conflicts with addressed edge e{'xyz'[axis]}{tuple(edge)} "
        "on the graded lane; move the stage or the layout switch off that edge")


def refuse_layout_port_releases(layout, conductors):
    """Every published port release matters, including passive loads (R12.2)."""
    details = {(component, tuple(edge)): f"{kind} port-release stage ({entity})"
               for entity, kind, component, edge, _ in conductors._released_port_details}
    for entity, component, edge in conductors.released_port_edges:
        details.setdefault((component, tuple(edge)), f"port-release stage ({entity})")
    for index, sheet, axis, indices in _layout_edges(layout, conductors.grid):
        for edge in zip(*(v.ravel().tolist() for v in indices)):
            stage = details.get((f"e{'xyz'[axis]}", edge))
            if stage is not None:
                _layout_stage_error(index, sheet, stage, axis, edge)


def refuse_layout_sheet_impedance(layout, conductors):
    """Check the full operator edge set, before PEC exclusions (R12.3)."""
    if not layout or not conductors.sheet_impedance:
        return
    from rfx.materials.thin_conductor import build_sheet_impedance_ctx
    import jax
    with jax.ensure_compile_time_eval():
        ctx = build_sheet_impedance_ctx(conductors.sheet_impedance,
                                        periodic=conductors.periodic)
        for index, sheet, axis, indices in _layout_edges(layout, conductors.grid):
            mask = np.asarray(getattr(ctx, f"mask_e{'xyz'[axis]}"))
            hits = np.flatnonzero(mask[indices])
            if len(hits):
                edge = tuple(int(v.flat[hits[0]]) for v in indices)
                _layout_stage_error(index, sheet, "finite-impedance sheet operator", axis, edge)


def refuse_layout_metal_consumers(layout, sim, grid):
    """Admission for other readers of metal and post-keep writers (R12.4)."""
    if not layout:
        return
    if getattr(sim, '_tfsf', None) is not None:
        # The TF/SF invariance admission (#1816) judges the static metal only.
        index, sheet, _ = layout[0]
        raise ValueError(
            f"pinned_sheet_layout sheet {index} ({sheet.name!r}): the TF/SF invariance "
            "admission reads the static metal, so it cannot judge a switched sheet, and an "
            "open plane wave on the graded mesh admits no finite scatterer, drawn or switched; "
            "excite the switched layout with a port, or draw each layout and run it on the "
            "uniform mesh (closed_box=True)")
    from rfx._grid_metric import field_index
    from rfx.sources.msl_port import msl_cross_section_span, msl_port_from_entry
    solved = {(spec.component, tuple(field_index(grid, spec.position, spec.component)))
              for spec in sim._lumped_rlc if spec.topology == 'series' or spec.L > 0}
    spans = [msl_cross_section_span(grid, msl_port_from_entry(port))
             for port in sim._msl_ports]
    for index, sheet, axis, indices in _layout_edges(layout, grid):
        for edge in zip(*(v.ravel().tolist() for v in indices)):
            if (f"e{'xyz'[axis]}", edge) in solved:
                _layout_stage_error(index, sheet, "lumped RLC post-keep solve", axis, edge)
            if getattr(sim, '_current_moments', None) is not None:
                _layout_stage_error(index, sheet, "current-moment metal census", axis, edge)
            for span in spans:
                ip, iw = span['prop_idx'], span['width_idx']
                at_feed = edge[ip] in (span['i_feed'], (span['i_feed'] - 1) % grid.shape[ip])
                if (at_feed and edge[iw] in span['width_nodes'] and
                        span['n_lo'] <= edge[2] <= span['n_hi']):
                    _layout_stage_error(index, sheet, "MSL geometry-edge admission", axis, edge)
