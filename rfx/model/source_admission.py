"""Source admission against geometry conductors and declared PEC walls."""
from rfx._grid_metric import field_index
import warnings
from types import SimpleNamespace

import jax
import numpy as np

from rfx.boundaries.depths import Kind
from rfx.boundaries.pec import wall_edge_masks
from rfx.core.jax_utils import is_tracer
from rfx.preflight._common import _component_is_dead, _H_LOOP


def declared_pec_faces(sim):
    """Read declared walls, excluding absorber backings and invariant axes."""
    model = sim.boundary_model()
    invariant = {axis.name for axis in model.axes if axis.invariant}
    return tuple(f.name for f in model.faces
                 if f.kind == Kind.PEC and f.name[0] not in invariant)


def wall_source_findings(sim, grid, *, nonuniform=False):
    """Standalone preflight uses the same realized wall-edge classification."""
    root = SimpleNamespace(grid=grid, pec_edges=None,
                           lane='nonuniform' if nonuniform else 'uniform')
    return _dead_sources(sim, root)


def refuse_dead_soft_sources(sim, conductors):
    """Refuse dead soft sources and ports before stepping, even without preflight.

    Releases apply to geometry only. The returned conductor product and the
    kernels' existing wire live-cell handling are not changed by this check.
    """
    for message in _dead_sources(sim, conductors):
        raise ValueError(message)


def _dead_sources(sim, conductors):
    if not sim._ports:
        return
    faces = declared_pec_faces(sim)
    if conductors.pec_edges is None and not faces:
        return
    grid = conductors.grid
    leaves = (tuple(grid.cells(a) for a in range(3)), conductors.pec_edges,
              tuple((s.position, s.extent) for s in sim._ports))
    if any(is_tracer(leaf) for leaf in jax.tree.leaves(leaves)):
        # Preserve the existing no-conductor/traced-mesh silent return.
        # A traced declaration itself still needs the cannot-classify notice.
        if conductors.pec_edges is not None or any(
                is_tracer(v) for s in sim._ports
                for v in jax.tree.leaves((s.position, s.extent))):
            warnings.warn(
                "Soft-source PEC admission check cannot run on a traced mesh "
                "or traced conductor edges/source positions; no concrete "
                "realized edge classification is available.", UserWarning,
                stacklevel=3)
        return
    from rfx.model.conductors import lumped_port_stage
    from rfx.sources.sources import _wire_port_cells, wire_port_from_entry
    released = conductors
    for index, source in enumerate(sim._ports if conductors.pec_edges is not None else ()):
        released = lumped_port_stage(released, source, f'port[{index}]')
    walls = wall_edge_masks(grid.shape, faces)
    edges = (walls if released.pec_edges is None else
             tuple(wall | np.asarray(body, dtype=bool)
                   for wall, body in zip(walls, released.pec_edges)))
    for index, source in enumerate(sim._ports):
        component = source.component.lower()
        cells = (_wire_port_cells(grid, wire_port_from_entry(source))
                 if source.extent is not None else
                 [tuple(int(v) for v in field_index(grid, source.position, component))])
        if not cells or not all(_component_is_dead(edges, component, cell)
                                for cell in cells):
            continue
        # Attribution reuses the owner's masks; it contains no wall-edge rule.
        wall_faces = []
        offsets = ((("xyz".index(component[1]), 0, 0, 0),)
                   if component.startswith('e') else _H_LOOP[component])
        for face in faces:
            own = wall_edge_masks(grid.shape, (face,))
            if any(_component_is_dead(own, "e" + "xyz"[c],
                                      tuple(v + d for v, d in zip(cell, (di, dj, dk))))
                   for cell in cells for c, di, dj, dk in offsets):
                wall_faces.append(face)
        label = ('Wire port' if source.extent is not None else
                 'Lumped port' if source.impedance > 0 else 'Soft source')
        reason = ('every driven E edge is PEC' if source.extent is not None else
                  'its own E edge is PEC' if component.startswith('e') else
                  'all four E edges of its curl loop are PEC')
        message = (f"{label} _ports[{index}] at {tuple(source.position)} "
                   f"({component}), grid index {cells[0]}, is dead: {reason}. ")
        if wall_faces:
            outside = []
            for axis in sorted({face[0] for face in wall_faces}):
                coordinate = source.position['xyz'.index(axis)]
                length = sim._domain['xyz'.index(axis)]
                if coordinate < 0 or coordinate > length:
                    outside.append(f"Declared {axis} coordinate {coordinate} is outside "
                                   f"domain [0, {length}]. ")
            yield (message + f"Declared PEC wall face(s): {', '.join(wall_faces)}. "
                   f"Realized driven indices: {cells}. "
                   "This declaration raises because its tangential E edges in a PEC wall are shorted. "
                   + ''.join(outside)
                   + "Move it at least one cell inside the domain, off the wall.")
        else:
            if source.extent is not None:
                # Keep the geometry-only wire guard and its live-cell message.
                from rfx.sources.sources import _wire_port_live_cells
                _wire_port_live_cells(grid, wire_port_from_entry(source), released.pec_edges)
            owners = _owners(sim, conductors, component, cells[0])
            yield (message + f"Realized conductor provenance (assembly): {', '.join(owners)}. "
                   "Move the source off the conductor, or use a port.")


def _owners(sim, conductors, component, cell):
    """Attribute every occupied loop edge, including jointly frozen H loops.

    Reuse assembly collectors and the production edge rule; never classify
    declared shapes again or assemble a second material product.
    """
    from rfx.boundaries.pec import realized_pec_edge_masks
    labels = {id(entry): f'{prefix}[{i}]'
              for collection, prefix in ((sim._geometry, 'geometry'),
                                          (sim._thin_conductors, 'thin_conductor'))
              for i, entry in enumerate(collection)}
    offsets = ((("xyz".index(component[1]), 0, 0, 0),)
               if component.startswith('e') else _H_LOOP[component])
    products = [(labels[key], cells, sheet, wire)
                for key, cells, sheet, wire, _ in conductors.assembly_entries
                if key in labels]
    products.extend((sheet.entity_id, None, sheet, None)
                    for sheet in conductors.sheets
                    if getattr(sheet, 'entity_id', '').startswith('pinned_sheet['))
    owners = []
    for label, cells, sheet, wire in products:
        if cells is None and sheet is None and wire is None:
            continue
        own = realized_pec_edge_masks(cells,
            sheets=() if sheet is None else (sheet,),
            wires=() if wire is None else (wire,), periodic=conductors.periodic)
        if any(bool(own[c][tuple(v + d for v, d in zip(cell, (di, dj, dk)))])
               for c, di, dj, dk in offsets):
            owners.append(label)
    return owners or ['unattributed realized PEC edge']
