"""Soft-source admission against the assembled conductor edge set."""
import warnings

import jax
import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.preflight._common import _component_is_dead, _H_LOOP


def refuse_dead_soft_sources(sim, conductors):
    """Refuse declarations frozen by PEC, independently of advisory opt-out.

    Port-generated drives are deliberately excluded: those ports release
    their driven edges. User soft sources are zero-impedance port entries,
    including internal H-component declarations.
    """
    sources = [(i, source) for i, source in enumerate(sim._ports)
               if source.impedance == 0.0]
    if not sources or conductors.pec_edges is None:
        return
    grid = conductors.grid
    leaves = (tuple(grid.cells(a) for a in range(3)), conductors.pec_edges,
              tuple(source.position for _, source in sources))
    if any(is_tracer(leaf) for leaf in jax.tree.leaves(leaves)):
        warnings.warn(
            "Soft-source PEC admission check cannot run on a traced mesh "
            "or traced conductor edges/source positions; no concrete "
            "realized edge classification is available.", UserWarning,
            stacklevel=3)
        return
    edges = tuple(np.asarray(edge, dtype=bool) for edge in conductors.pec_edges)
    if conductors.lane == 'nonuniform':
        from rfx.nonuniform import position_to_index
        def locate(position):
            return position_to_index(grid, position)
    else:
        locate = grid.position_to_index
    for index, source in sources:
        cell = tuple(int(v) for v in locate(source.position))
        component = source.component.lower()
        if not _component_is_dead(edges, component, cell):
            continue
        owners = _owners(sim, conductors, component, cell)
        reason = ("its own E edge is PEC" if component.startswith('e') else
                  "all four E edges of its curl loop are PEC")
        raise ValueError(
            f"Soft source _ports[{index}] at {tuple(source.position)} "
            f"({component}), grid index {cell}, is dead: {reason}. "
            f"Realized conductor provenance (assembly): {', '.join(owners)}. "
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
