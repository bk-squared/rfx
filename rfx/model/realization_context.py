"""Read a prepared product without consulting diagnostic caches or overrides."""
import numpy as np


def context_from_conductors(sim, conductors):
    """Lend realized products to readers without reassembling declarations."""
    from rfx.realized_geometry import _node_arrays
    from rfx.preflight.realization import _CampaignStaticsContext, _EntryRealization, _shape_bounds
    from rfx.geometry.rasterize_grid import GridCoords
    ctx = object.__new__(_CampaignStaticsContext)
    ctx.sim, ctx.grid = sim, conductors.grid
    ctx.lane, ctx.periodic = conductors.lane, conductors.periodic
    ctx.error = ctx.assembly_error = ctx._assembly_exception = None
    ctx._realized, ctx._entries = conductors, []
    ctx.spacings, ctx.nodes = _node_arrays(sim, ctx.grid, ctx.lane == 'nonuniform')
    ctx.cell_sizes = ctx.spacings
    ctx.coords = GridCoords(*(n[:ctx.grid.shape[a]] for a, n in enumerate(ctx.nodes)), shape=ctx.grid.shape)
    products = {key: (cells, sheet, wire, shape)
                for key, cells, sheet, wire, shape in conductors.assembly_entries}
    for collection, prefix in ((sim._geometry, 'geometry'), (sim._thin_conductors, 'thin_conductor')):
        for i, entry in enumerate(collection):
            label = f'{prefix}[{i}]'
            cells, sheet, wire, shape = products.get(id(entry), (None, None, None, entry.shape))
            if id(entry) not in products and prefix == 'geometry':
                continue
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
            ctx._entries.append(_EntryRealization(label=entity.entity_id,
                name=entity.entity_id, shape=entity.shape, kind=entity.kind,
                cells=entity.cells, lo=np.asarray(lo), hi=np.asarray(hi)))
    return ctx

