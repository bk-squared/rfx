"""One immutable comparison table for the products handed to a time step.

Family sampling and solved-edge rules belong to the existing geometry record.
This module compares its quantities; it does not rasterize or issue verdicts.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import weakref

import numpy as np


@dataclass(frozen=True)
class RealizationAxis:
    axis: str
    declared_bounds_m: tuple[float, float] | None
    bounds_m: tuple[float, float]
    declared_span_m: float | None
    realized_span_m: float
    cell_size_m: float
    miss_cells: float | None
    miss_percent: float | None
    centroid_shift_cells: float | None


@dataclass(frozen=True)
class RealizationRow:
    label: str
    family: str
    axes: tuple[RealizationAxis, ...] = ()
    continued_faces: tuple[str, ...] = ()
    status: str = "evaluated"
    model: str = "mask"
    declared_area_m2: float | None = None
    realized_area_m2: float | None = None
    area_equivalent_length_m: float | None = None
    area_miss_percent: float | None = None
    filled_volume_m3: float | None = None
    open_area_m2: tuple[float, ...] | None = None
    traced_inputs: tuple[str, ...] = ()
    regions: tuple = ()
    record_axes: tuple = ()
    declared_area_equivalent_length_m: float | None = None
    area_miss_cells: float | None = None
    declared_volume_m3: float | None = None
    volume_equivalent_length_m: float | None = None
    volume_miss_percent: float | None = None
    max_owner_deviation: float | None = None
    name: str | None = None
    shape_name: str | None = None
    declared_bounds_m: tuple | None = None
    occupancy_count: int = 0
    plane: tuple | None = None
    wall_planes: tuple = ()
    wall_positions_m: tuple = ()
    domain_geometry: tuple = ()
    domain_face_cells_m: tuple = ()


@dataclass(frozen=True)
class RealizationTable:
    rows: tuple[RealizationRow, ...]
    lane: str
    prepared: object = field(repr=False, compare=False)

    def row(self, label):
        return next(row for row in self.rows if row.label == label)


@dataclass(frozen=True)
class PreparedRealization:
    sim: object
    conductors: object
    lane: str
    record: object


def _axis(axis, declared, realized, cell):
    span = float(realized[1] - realized[0])
    drawn = None if declared is None else float(declared[1] - declared[0])
    miss = None if drawn is None else span - drawn
    return RealizationAxis(
        axis, declared, realized, drawn, span, cell,
        None if miss is None else miss / cell,
        None if not drawn else 100 * miss / drawn,
        None if declared is None else (sum(realized) - sum(declared)) / (2 * cell))


def realization_table(prepared) -> RealizationTable:
    """Read the prepared record once, retaining its exact assembly identity."""
    record, sim = prepared.record, prepared.sim
    if record is None:
        return _traced_table(prepared)
    rows = []
    for entity in record.entities:
        axes = tuple(_axis(a.axis, a.declared_bounds_m, a.bounds_m, a.cell_size_m)
                     for a in entity.axes)
        declared, realized = entity.declared_footprint_area_m2, entity.realized_footprint_area_m2
        model = "mask"
        shape = None
        declared_bounds = entity.declared_bounds_m
        for prefix, entries in (("geometry[", sim._geometry), ("thin_conductor[", sim._thin_conductors)):
            if entity.label.startswith(prefix):
                declaration = entries[int(entity.label[len(prefix):-1])].shape
                shape = type(declaration).__name__
                # Preserve the reporter's analytic-bounds availability contract.
                try:
                    declared_bounds = tuple(tuple(float(v) for v in side)
                                            for side in declaration.bounding_box())
                except Exception:
                    declared_bounds = None
        if entity.kind == "material":
            from rfx.model.realization_integrals import quantities
            sampler = quantities(getattr(record, "_quantity_grid", prepared.conductors.grid))
            declared_shape = sim._geometry[int(entity.label[9:-1])].shape
            model = sampler.get("material_sampling", {}).get(id(declared_shape), "material sampler")
        elif entity.kind == "wire":
            model = "lattice edge-start envelope"
        elif entity.kind == "lossy" and realized is None:
            model = "weighted f0 sheet end rows"
        if realized is not None and entity.plane is not None:
            normal, _, position = entity.plane
            axes = tuple(_axis(a.axis, a.declared_bounds_m, (position, position), a.cell_size_m)
                         if i == normal else a for i, a in enumerate(axes))
            model = "cell-centre footprint; node plane"
        if entity.kind == "sheet":
            model = "solved sheet edges"
        tangent = [a for i, a in enumerate(axes) if entity.plane is not None and i != entity.plane[0]]
        area_cell = float(np.sqrt(np.prod([a.cell_size_m for a in tangent]))) if len(tangent) == 2 else None
        rows.append(RealizationRow(
            entity.label, entity.kind, axes, entity.continued_faces,
            status=entity.error or entity.mask_error or "evaluated", model=model,
            declared_area_m2=declared, realized_area_m2=realized,
            area_equivalent_length_m=None if realized is None else float(np.sqrt(realized)),
            area_miss_percent=None if not declared or realized is None else
            100 * (float(np.sqrt(realized / declared)) - 1),
            record_axes=entity.axes,
            declared_area_equivalent_length_m=None if declared is None else float(np.sqrt(declared)),
            area_miss_cells=None if declared is None or realized is None or not area_cell else
            (float(np.sqrt(realized)) - float(np.sqrt(declared))) / area_cell,
            name=entity.name, shape_name=shape, declared_bounds_m=declared_bounds,
            occupancy_count=entity.n_cells or entity.node_count, plane=entity.plane,
            wall_planes=entity.wall_planes,
            wall_positions_m=tuple(tuple(float(record.nodes[a][k]) for k in planes)
                                   for a, planes in enumerate(entity.wall_planes))))
    original_domain = getattr(sim._unresolved_domain, "declared", sim._unresolved_domain)
    rows.append(RealizationRow("domain", "domain", tuple(
        _axis(d.axis, (0., original_domain[a] if a < len(original_domain) and
                      original_domain[a] > 0 else d.declared_length_m), (0., d.length_m),
              float(np.mean(record.cell_sizes[a][d.padding[0]:d.padding[0] + max(d.n_cells, 1)])))
        for a, d in enumerate(record.domain)), model="grid nodes",
        domain_geometry=record.domain,
        domain_face_cells_m=tuple((float(record.cell_sizes[a][d.padding[0]]),
            float(record.cell_sizes[a][max(d.padding[0] + d.n_cells - 1, d.padding[0])]))
            for a, d in enumerate(record.domain))))
    for probe in record.probes:
        if not probe.label.startswith("probe["):
            continue
        entry = sim._probes[int(probe.label[6:-1])]
        if probe.index is None:
            rows.append(RealizationRow(probe.label, "probe", status=probe.error or "not evaluated"))
            continue
        rows.append(RealizationRow(probe.label, "probe", tuple(
            _axis("xyz"[a], (entry.position[a],) * 2,
                  (float(record.nodes[a][index]),) * 2, float(record.cell_sizes[a][index]))
            for a, index in enumerate(probe.index)), model="field node"))
    from rfx.model.realization_placements import placement_rows
    rows.extend(placement_rows(sim, record, (r.label for r in rows)))
    from rfx.model.realization_integrals import read_rows
    quantity_grid = getattr(record, "_quantity_grid", prepared.conductors.grid)
    return RealizationTable(read_rows(rows, quantity_grid, sim), prepared.lane,
                            weakref.ref(prepared.conductors))


def _traced_table(prepared):
    """Keep unavailable geometry explicit without forcing a tracer to the host."""
    sim = prepared.sim
    from rfx.model.realization_reference import traced_inputs
    inputs = traced_inputs(sim, prepared.conductors.grid)
    status = "not evaluated: traced" if inputs else "not represented: no readable per-object kernel record"
    rows = [RealizationRow(f"geometry[{i}]", type(e.shape).__name__,
                          status=status, traced_inputs=inputs)
            for i, e in enumerate(sim._geometry)]
    rows.extend(RealizationRow(f"thin_conductor[{i}]", "thin conductor",
                              status=status, traced_inputs=inputs)
                for i, _ in enumerate(sim._thin_conductors))
    rows.append(RealizationRow("domain", "domain", status=status, traced_inputs=inputs))
    rows.extend(RealizationRow(f"probe[{i}]", "probe", status=status,
                              traced_inputs=inputs) for i, _ in enumerate(sim._probes))
    return RealizationTable(tuple(rows), prepared.lane, weakref.ref(prepared.conductors))


def bind_table(sim, conductors, *, lane, compact=True):
    """Bind metadata to the kernel product, including forward and traced lanes."""
    from rfx.model.conductors import _traced_product
    from rfx.realized_geometry import record_from_conductors, RealizedGeometry
    from rfx.model.realization_reference import nominal_record, traced_inputs
    if conductors.realization_table is not None:
        return conductors._realization_record
    inputs = traced_inputs(sim, conductors.grid)
    reader = sim
    if conductors.kernel_context is not None:
        import copy
        reader = copy.copy(sim)
        reader._geometry, reader._thin_conductors = [], []
    record = (None if inputs or (not compact and _traced_product(conductors)) else
              record_from_conductors(reader, conductors, lane=lane, compact=compact))
    nominal = False
    if record is None and inputs:
        record = nominal_record(sim, conductors, lane, compact=compact)
        nominal = record is not None
    table = realization_table(PreparedRealization(sim, conductors, lane, record))
    if conductors.refinements:
        table = _refined_rows(table, conductors.refinements, sim)
    if conductors.kernel_context is not None:
        from rfx.model.realization_disjoint import disjoint_rows
        table = replace(table, rows=disjoint_rows(sim, table.rows, conductors.kernel_context[1]))
        record = replace(record, limitations=("disjoint vacuum stepper does not carry material declarations",))
    if inputs:
        table = replace(table, rows=tuple(replace(row, traced_inputs=inputs,
                        status="evaluated: nominal_profiles" if nominal else row.status)
                                          for row in table.rows))
    object.__setattr__(conductors, "realization_table", table)
    conductors._realization_tables.append(table)
    if record is None and not inputs:
        return None
    if record is None:
        record = RealizedGeometry((), (), (), (), (), (), None, (), (), None,
                                  conductors.periodic, lane=lane)
    record = replace(record, realization_table=table)
    if compact:
        record = replace(record, nodes=tuple(tuple(float(x) for x in axis) for axis in record.nodes),
                         cell_sizes=tuple(tuple(float(x) for x in axis) for axis in record.cell_sizes))
    object.__setattr__(conductors, "_realization_record", record)
    if not compact:
        from rfx.realized_geometry import _conductor_view
        record = _conductor_view(record, conductors)
    return record


def _refined_rows(table, refinements, sim):
    """Read each fine record's existing interval rule, retaining both lattices."""
    rows = []
    for row in table.rows:
        regions = []
        for name, record in refinements:
            probe = next((p for p in record.probes if p.label == row.label), None)
            if probe is not None and probe.index is not None:
                entry = sim._probes[int(probe.label[6:-1])]
                axes = tuple(_axis("xyz"[a], (entry.position[a],) * 2,
                                   (float(record.nodes[a][index]),) * 2,
                                   float(record.cell_sizes[a][index]))
                             for a, index in enumerate(probe.index))
                row = replace(row, axes=axes, status="evaluated", model="fine field node")
            entity = next((e for e in record.entities if e.label == row.label), None)
            if entity is not None:
                regions.append((name, tuple(_axis(a.axis, a.declared_bounds_m, a.bounds_m,
                                                 a.cell_size_m) for a in entity.axes)))
        rows.append(replace(row, regions=tuple(regions)))
    return replace(table, rows=tuple(rows))


def finish_table(sim, conductors, *, lane):
    """Complete the existing prepared table with stage-owned rows."""
    record = bind_table(sim, conductors, lane=lane)
    if record is None:
        return None
    table = conductors.realization_table
    from rfx.model.realization_integrals import read_rows
    object.__setattr__(table, "rows", read_rows(table.rows, conductors.grid, sim))
    object.__setattr__(table, "lane", lane)
    if conductors.refinements:
        object.__setattr__(table, "rows", _refined_rows(table, conductors.refinements, sim).rows)
    record = replace(record, lane=lane, dt=float(conductors.grid.dt))
    if lane == "run_subgridded" and not conductors.refinements and conductors.kernel_context is None:
        record = replace(record, limitations=("refined region not represented",))
    if not any(row.traced_inputs for row in table.rows):
        from rfx.preflight.realization import context_from_conductors
        from rfx.realized_geometry import _ports
        from rfx.model.realization_placements import placement_rows
        ctx = context_from_conductors(sim, conductors)
        record = replace(record, ports=_ports(sim, ctx, conductors))
        rows = tuple(row for row in table.rows if not row.label.startswith(("port[", "waveguide_port[")))
        object.__setattr__(table, "rows", rows + tuple(placement_rows(sim, record, (r.label for r in rows))))
    if conductors.kernel_context is not None:
        from rfx.model.realization_disjoint import disjoint_rows
        rows = tuple(row for row in table.rows if not row.label.startswith(("geometry[", "thin_conductor[")))
        object.__setattr__(table, "rows", disjoint_rows(sim, rows, conductors.kernel_context[1]))
        record = replace(record, limitations=("disjoint vacuum stepper does not carry material declarations",))
    object.__setattr__(conductors, "_realization_record", record)
    return record
