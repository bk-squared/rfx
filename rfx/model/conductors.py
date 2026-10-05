"""Realized conductor products and the stages that change their edge masks.

The classifier and Yee edge rules remain in rasterize_grid and boundaries.pec.
This module owns their products, not another rasterization convention.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from rfx.preflight.realization import _RealizedPEC
from rfx.boundaries.pec import SheetSpec


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


def realized_conductors(sim, grid, *, nonuniform=False, assembly=None,
                        pec_sheets=(), pec_wires=(), sheet_specs=(),
                        geometry_masks=(), assembly_entries=(),
                        pad_fill_findings=(), periodic=None, mode="solve"):
    """Build from production collectors, assembling only when none were supplied."""
    from rfx.boundaries.pec import realized_pec_edge_masks
    if mode not in ("solve", "audit"):
        raise ValueError(f"Unknown conductor assembly mode: {mode}")
    if assembly is None:
        pec_sheets, pec_wires, sheet_specs = [], [], []
        geometry_masks, assembly_entries, pad_fill_findings = [], [], []
        kwargs = dict(pec_sheets=pec_sheets, pec_wires=pec_wires,
                      sheet_specs=sheet_specs, geometry_masks=geometry_masks,
                      assembly_entries=assembly_entries)
        if nonuniform:
            assembly = sim._assemble_materials_nu(grid, **kwargs)
        else:
            if mode == "audit":
                kwargs["pad_fill_findings"] = pad_fill_findings
            assembly = sim._assemble_materials(grid, **kwargs)
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


def clear_conductor_edges(conductors, cells, *, component, entity_id, clear_cells=False, release_edges=True):
    """One persistent port-clearing stage; the incoming object stays unchanged."""
    from rfx.boundaries.pec import clear_edges
    if conductors.pec_edges is None:
        return conductors
    cells = tuple(tuple(int(i) for i in cell) for cell in cells)
    edges = (clear_edges(conductors.pec_edges, cells, component=component)
             if release_edges else conductors.pec_edges)
    pec_cells = conductors.pec_cells
    if clear_cells and pec_cells is not None:
        for cell in cells:
            pec_cells = pec_cells.at[cell].set(False)
    return replace(conductors, pec_cells=pec_cells, pec_edges=edges, provenance=conductors.provenance + (
        ConductorStage((entity_id,), 'port-edge-clearing' if release_edges else 'port-cell-clearing', component, cells),))


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


def solve_conductors(sim, grid, *, nonuniform=False, preflight=None):
    """Assemble once for execution and lend the pre-port object to its checks."""
    import jax
    with jax.ensure_compile_time_eval():
        root = realized_conductors(sim, grid, nonuniform=nonuniform, mode="solve")
    if preflight is not None:
        sim._auto_preflight(conductors=root, **preflight)
    return root


class _PreparedSolve:
    """Execution-local ownership, transferred once to the selected runner.

    With preflight disabled, assembly remains lazy until after lane admission.
    The empty handoff can stay in dispatch frames without pinning dense arrays
    while a distributed runner stages and drops its full-domain products.
    """
    def __init__(self, sim):
        self.sim = sim
        self._product = None
        self._taken = False

    def realized(self):
        if self._taken:
            raise RuntimeError("Solve assembly was already consumed")
        if self._product is None:
            nonuniform = self.sim._uses_nonuniform_mesh
            grid = (self.sim._build_nonuniform_grid() if nonuniform
                    else self.sim._build_grid())
            self._product = solve_conductors(self.sim, grid, nonuniform=nonuniform)
        return self._product

    def take(self):
        product = self.realized()
        self._product = None
        self._taken = True
        return product


def prepare_solve(sim, *, distributed=False):
    """Keep solve ownership local to this execution, never in audit caches."""
    if distributed:
        sim._pf_campaign_ctx = None
        sim._realized_geometry_record = None
    return _PreparedSolve(sim)


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
    obj = realized_conductors(sim, grid, nonuniform=hasattr(grid, 'dx_arr'),
        assembly=(materials, None, None, pec_cells), pec_sheets=sheets,
        pec_wires=wires, sheet_specs=sheet_specs, periodic=periodic,
        geometry_masks=() if root is None else root.geometry_masks.items(),
        assembly_entries=() if root is None else root.assembly_entries)
    return obj if root is None else replace(obj, assembly=root.assembly)


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
                from rfx.materials.thin_conductor import build_sheet_impedance_ctx
                conductors = replace(conductors, sheet_operator=build_sheet_impedance_ctx(
                    conductors.sheet_impedance, pec_edge_masks=edges))
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
    spans = {}
    for entry in interior:
        bounds = _shape_bounds(entry.shape)
        if entry.sheet is None or bounds is None:
            continue
        spans[entry.label] = tuple(None if a == entry.sheet.normal_axis else solved_sheet_span(
            entry.sheet.footprint, a, nodes[a], float(bounds[0][a]), float(bounds[1][a]),
            float(sim._domain[a]), union=union, volume_edges=volume_edges,
            periodic='xyz'[a] in getattr(grid, 'periodic_axes', '')) for a in range(3))
    pinned = getattr(sim, '_pinned_sheets', ())
    if pinned:
        from rfx.geometry.rasterize_grid import interior_lattice_mask
        pin_specs = conductors.sheets[-len(pinned):]
        pin_masks = [interior_lattice_mask(sheet.footprint, grid) for sheet in pin_specs]
        pin_union = union
        for footprint in pin_masks:
            pin_union = footprint if pin_union is None else pin_union | footprint
        for i, (sheet, footprint) in enumerate(zip(pin_specs, pin_masks, strict=True)):
            label = f'pinned_sheet[{i}]'
            owners[id(sheet)] = label
            unwrapped = sheet.unwrapped_footprint
            occupied = np.where(np.asarray(sheet.footprint if unwrapped is None else unwrapped))
            if occupied[0].size:
                spans[label] = tuple(None if a == sheet.normal_axis else solved_sheet_span(
                    footprint, a, nodes[a], float(nodes[a][occupied[a].min()]),
                    float(nodes[a][occupied[a].max()]), float(sim._domain[a]),
                    union=pin_union, volume_edges=volume_edges,
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
    if conductors.lane == 'nonuniform':
        from rfx.nonuniform import position_to_index
        def locate(position):
            return position_to_index(grid, position)
    else:
        locate = grid.position_to_index
    done = {entity for stage in conductors.provenance
            if stage.stage == 'port-edge-clearing' for entity in stage.entity_ids}
    for i, port in enumerate(sim._ports):
        label = f'port[{i}]'
        if port.impedance > 0 and port.extent is None and label not in done:
            conductors = clear_conductor_edges(conductors, [locate(port.position)],
                component=port.component, entity_id=label)
    if sim._msl_ports:
        from rfx.sources.msl_port import (
            msl_port_from_entry, _msl_yz_cells, msl_normal_component)
        for i, entry in enumerate(sim._msl_ports):
            label = f'msl_port[{i}]'
            if label not in done:
                port = msl_port_from_entry(entry)
                conductors = clear_conductor_edges(conductors, _msl_yz_cells(grid, port),
                    component=msl_normal_component(port), entity_id=label)
    return conductors


def _traced_product(conductors):
    import jax
    from rfx.core.jax_utils import is_tracer
    return any(is_tracer(leaf) for leaf in jax.tree.leaves(
        (conductors.materials, conductors.pec_edges, conductors.assembly)))

def auto_preflight(sim, *, skip=False, context="forward", check_ntff=True,
                   conductors=None, prepare=False, distributed=False):
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
        return assembly
    if assembly is not None:
        conductors = assembly.realized()
    # A validator bug must propagate, rather than become an advisory.
    issues = sim._preflight_impl(strict=False, check_ntff=check_ntff,
                                 _conductors=conductors)
    # gate -> this helper -> facade -> public entry -> user
    sim._run_preflight_gate(issues, context=context, stacklevel=5)
    return assembly


def bind_solve_call(assembly, call):
    """Consume a graded solve before ring-down adds internal probes."""
    from functools import partial
    root = assembly.take()
    return partial(call, conductors=root), root.grid


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


def forward_products(sim, grid, materials, cells, sheets, wires, periodic, root):
    """Bind overrides while keeping the original substrate for MSL launches."""
    drawn = None if root is None else root.assembly[0]
    root = kernel_conductors(sim, grid, materials, cells, sheets, wires,
        periodic=periodic, root=root,
        sheet_specs=() if root is None else root.sheet_impedance)
    return root, drawn, root.pec_edges


def forward_port_stage(root, cells, component, entity_id, *,
                       release_edges=True, skip_empty=False):
    """Release only the driven component; preserve the wire's edge support."""
    if not skip_empty or cells:
        root = clear_conductor_edges(root, cells, component=component,
            entity_id=entity_id, clear_cells=True, release_edges=release_edges)
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


def distributed_solve_inputs(sim, grid, assembly, skip, gather):
    """Consume the dispatch handoff without retaining it during slab staging."""
    sim._pf_campaign_ctx = None
    sim._realized_geometry_record = None
    root = (assembly.take() if assembly is not None else
        solve_conductors(sim, grid, nonuniform=True,
            preflight=dict(skip=skip, context="run" if gather else "forward",
                           check_ntff="advisory" if gather else True)))
    sheets, wires = [], []
    materials = assembled_materials(root, pec_sheets=sheets, pec_wires=wires)
    return root, materials, sheets, wires


def distributed_kernel_inputs(sim, root, grid, materials, cells, sheets, wires, gather):
    root = kernel_conductors(sim, grid, materials, cells, sheets, wires,
                            periodic=(False, False, False), root=root)
    root, record = at_kernel(sim, root,
        lane="run_distributed" if gather else "fwd_distributed_nu",
        pec_edges=root.pec_edges)
    return root.pec_edges, record


def stage_distributed_edges(edges, grid, mesh, multiprocess):
    """Stage final components with unchanged traced/concrete ownership rules."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from rfx.runners.distributed_nu import (
        shard_pec_mask_x_slab, stage_concrete_forward_array)
    if edges is None:
        return None
    if any(isinstance(edge, jax.core.Tracer) for edge in edges):
        if multiprocess:
            raise ValueError("a traced pec_mask_override is supported in one process only")
        return tuple(jax.device_put(shard_pec_mask_x_slab(edge, grid),
                                   NamedSharding(mesh, P("x"))) for edge in edges)
    return tuple(stage_concrete_forward_array(
        edge, grid, mesh, True, ghost_value=False) for edge in edges)


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
