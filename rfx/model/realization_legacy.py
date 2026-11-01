"""Bind ADI and refined-grid metadata at their actual stepping boundaries."""
from dataclasses import replace


def legacy_kernel(sim, grid, materials, cells, edges, lane):
    from rfx.model.conductors import at_kernel, kernel_conductors
    root = getattr(grid, "_prepared_conductors", None)
    if hasattr(grid, "_prepared_conductors"):
        del grid._prepared_conductors
    if root is not None and hasattr(root.grid, "_prepared_conductors"):
        del root.grid._prepared_conductors
    root = kernel_conductors(sim, grid, materials, cells,
        periodic=tuple(sim._periodic_flags()), root=root,
        sheets=() if root is None else root.sheets,
        wires=() if root is None else root.wires)
    root = replace(root, grid=grid, materials=materials, pec_cells=cells, pec_edges=edges)
    return at_kernel(sim, root, lane=lane, pec_edges=root.pec_edges)


def refined_kernel(sim, coarse_grid, coarse_materials, coarse_cells,
                   fine_grid, fine_materials, fine_cells, masks, entries, origin, probe_indices):
    import copy
    from rfx.model.conductors import at_kernel, kernel_conductors, realized_conductors
    from rfx.realized_geometry import record_from_conductors, ProbeGeometry
    fine_grid._realization_origin = origin
    reader = copy.copy(sim)
    reader._ports, reader._probes = [], []
    reader._tfsf = None
    fine = realized_conductors(reader, fine_grid,
        assembly=(fine_materials, None, None, fine_cells), geometry_masks=masks,
        assembly_entries=entries, periodic=(False,) * 3)
    fine_record = record_from_conductors(reader, fine, lane="refinement", compact=True)
    fine_record = replace(fine_record, probes=tuple(
        ProbeGeometry(f"probe[{i}]", "point", (entry.component,), index=tuple(index))
        for i, (entry, index) in enumerate(zip(sim._probes, probe_indices))))
    root = kernel_conductors(sim, coarse_grid, coarse_materials, coarse_cells,
        periodic=tuple(sim._periodic_flags()), root=getattr(coarse_grid, "_prepared_conductors", None))
    root = replace(root, refinements=(("fine", fine_record),))
    if hasattr(coarse_grid, "_prepared_conductors"):
        del coarse_grid._prepared_conductors
    return at_kernel(sim, root, lane="run_subgridded", pec_edges=root.pec_edges)


def disjoint_kernel(sim, grid, config, contract):
    import copy
    from rfx.model.conductors import at_kernel, kernel_conductors
    root = getattr(grid, "_prepared_conductors", None)
    if root is None:
        from rfx.core.yee import init_materials
        root = kernel_conductors(sim, grid, init_materials(grid.shape), None,
                                  periodic=tuple(sim._periodic_flags()))
    else:
        del grid._prepared_conductors
    grid = copy.copy(grid)
    grid.dt = config.dt
    root = replace(root, grid=grid, kernel_context=(config, contract))
    return at_kernel(sim, root, lane="run_subgridded", pec_edges=root.pec_edges)
