"""Read prepared products on their grid or an explicit nominal grid."""
import numpy as np


def context_from_conductors(sim, conductors, *, grid=None):
    """Lend products, with nominal non-sheet occupancy for host size readers."""
    from rfx.realized_geometry import _node_arrays
    from rfx.preflight.realization import _CampaignStaticsContext, _EntryRealization, _shape_bounds
    from rfx.geometry.rasterize_grid import GridCoords
    ctx = object.__new__(_CampaignStaticsContext)
    ctx.sim, ctx.grid = sim, conductors.grid if grid is None else grid
    ctx.lane, ctx.periodic = conductors.lane, conductors.periodic
    ctx.error = ctx.assembly_error = ctx._assembly_exception = None
    ctx._realized, ctx._entries = conductors, []
    ctx.spacings, ctx.nodes = _node_arrays(sim, ctx.grid, ctx.lane == 'nonuniform')
    ctx.cell_sizes = ctx.spacings
    x, y, z = (n[:ctx.grid.shape[a]] for a, n in enumerate(ctx.nodes))
    ctx.coords = GridCoords(x, y, z, shape=ctx.grid.shape)

    def host_products(cells, sheet, wire, shape):
        if grid is not None and grid is not conductors.grid and (cells is not None or wire is not None):
            from rfx.geometry.rasterize_grid import (
                classify_pec_entry, centres_from_nonuniform_grid, pec_volume_cell_mask)
            centres = centres_from_nonuniform_grid(grid, ctx.coords)
            if cells is not None:
                cells = pec_volume_cell_mask(shape, centres, ctx.spacings, grid=grid)
            else:
                cells, sheet, wire = classify_pec_entry(
                    shape, ctx.coords, centres, ctx.spacings, grid=grid)
        return cells, sheet, wire

    products = {key: (cells, sheet, wire, shape)
                for key, cells, sheet, wire, shape in conductors.assembly_entries}
    for collection, prefix in ((sim._geometry, 'geometry'), (sim._thin_conductors, 'thin_conductor')):
        for i, entry in enumerate(collection):
            label = f'{prefix}[{i}]'
            cells, sheet, wire, shape = products.get(id(entry), (None, None, None, entry.shape))
            if id(entry) not in products and prefix == 'geometry':
                continue
            cells, sheet, wire = host_products(cells, sheet, wire, shape)
            kind = 'volume' if cells is not None else 'sheet' if sheet is not None else 'wire' if wire is not None else 'lossy'
            bounds = _shape_bounds(entry.shape)
            ctx._entries.append(_EntryRealization(label=label,
                name=getattr(entry, 'material_name', label), shape=entry.shape,
                solved_shape=shape, kind=kind, cells=cells, sheet=sheet, wire=wire,
                lo=None if bounds is None else np.asarray(bounds[0]),
                hi=None if bounds is None else np.asarray(bounds[1])))
    for entity in conductors.stamped_entities:
        if entity.kind != 'material':
            lo, hi = entity.shape.bounding_box()
            cells, sheet, wire = host_products(entity.cells, None, None, entity.shape)
            ctx._entries.append(_EntryRealization(label=entity.entity_id,
                name=entity.entity_id, shape=entity.shape, kind=entity.kind,
                cells=cells, sheet=sheet, wire=wire, lo=np.asarray(lo), hi=np.asarray(hi)))
    return ctx

