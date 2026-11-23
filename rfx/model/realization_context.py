"""Read prepared products on their grid or an explicit nominal grid."""
import numpy as np


def context_from_conductors(sim, conductors, *, grid=None):
    """Lend products, with nominal non-sheet occupancy for host size readers."""
    from rfx.realized_geometry import _node_arrays
    from rfx.preflight.realization import _CampaignStaticsContext, _EntryRealization, _shape_bounds
    from rfx.geometry.rasterize_grid import GridCoords
    from dataclasses import fields, is_dataclass
    import jax
    from rfx.core.jax_utils import is_tracer
    ctx = object.__new__(_CampaignStaticsContext)
    ctx.sim, ctx.grid = sim, conductors.grid if grid is None else grid
    ctx.lane, ctx.periodic = conductors.lane, conductors.periodic
    ctx.error = ctx.assembly_error = ctx._assembly_exception = None
    ctx._realized, ctx._entries = conductors, []
    ctx.omitted_pec_products = []
    ctx.spacings, ctx.nodes = _node_arrays(sim, ctx.grid, ctx.lane == 'nonuniform')
    ctx.cell_sizes = ctx.spacings
    x, y, z = (n[:ctx.grid.shape[a]] for a, n in enumerate(ctx.nodes))
    coords = GridCoords(x, y, z, shape=ctx.grid.shape)
    ctx.coords = coords

    nominal = grid is not None and grid is not conductors.grid

    def traced(value):
        return any(is_tracer(leaf) or (
            is_dataclass(leaf) and any(traced(getattr(leaf, f.name)) for f in fields(leaf)))
            for leaf in jax.tree.leaves(value))

    def host_products(cells, sheet, wire, shape):
        if nominal and (cells is not None or wire is not None):
            from rfx.geometry.rasterize_grid import (
                classify_pec_entry, centres_from_nonuniform_grid, pec_volume_cell_mask)
            centres = centres_from_nonuniform_grid(grid, coords)
            try:
                if cells is not None:
                    cells = pec_volume_cell_mask(shape, centres, ctx.spacings, grid=grid)
                else:
                    cells, sheet, wire = classify_pec_entry(
                        shape, coords, centres, ctx.spacings, grid=grid)
            except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
                return None
        if nominal and traced((cells, sheet, wire, shape)):
            return None
        return cells, sheet, wire

    from rfx.geometry.cross_section import section_entries
    sections = section_entries(sim, ctx.grid)
    products = {key: (cells, sheet, wire, shape)
                for key, cells, sheet, wire, shape in conductors.assembly_entries}
    for collection, prefix in ((sim._geometry, 'geometry'), (sim._thin_conductors, 'thin_conductor')):
        for i, entry in enumerate(collection):
            label = f'{prefix}[{i}]'
            cells, sheet, wire, shape = products.get(id(entry), (None, None, None, entry.shape))
            if id(entry) not in products and prefix == 'geometry':
                continue
            products_on_host = host_products(cells, sheet, wire, shape)
            if products_on_host is None:
                ctx.omitted_pec_products.append(f'{label} ({type(entry.shape).__name__})')
                continue
            cells, sheet, wire = products_on_host
            kind = 'volume' if cells is not None else 'sheet' if sheet is not None else 'wire' if wire is not None else 'lossy'
            section = sections[i].shape if prefix == "geometry" else entry.shape
            bounds = _shape_bounds(section)
            ctx._entries.append(_EntryRealization(label=label,
                name=getattr(entry, 'material_name', label), shape=section, declared_shape=entry.shape,
                solved_shape=shape, kind=kind, cells=cells, sheet=sheet, wire=wire,
                lo=None if bounds is None else np.asarray(bounds[0]),
                hi=None if bounds is None else np.asarray(bounds[1])))
    for entity in conductors.stamped_entities:
        if entity.kind != 'material':
            products_on_host = host_products(entity.cells, None, None, entity.shape)
            if products_on_host is None:
                ctx.omitted_pec_products.append(f'{entity.entity_id} ({type(entity.shape).__name__})')
                continue
            lo, hi = entity.shape.bounding_box()
            cells, sheet, wire = products_on_host
            ctx._entries.append(_EntryRealization(label=entity.entity_id,
                name=entity.entity_id, shape=entity.shape, kind=entity.kind,
                cells=cells, sheet=sheet, wire=wire, lo=np.asarray(lo), hi=np.asarray(hi)))
    return ctx

