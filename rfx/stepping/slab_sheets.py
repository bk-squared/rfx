"""Place the one-device sheet operands on the existing slab rows."""
from dataclasses import fields, replace

import jax
import jax.numpy as jnp

from rfx.stepping.slab import cut


def cut_sheets(context, coefficients, layout, mesh, *, uniform=False):
    if context is None:
        return None, None
    values = {}
    merged = jnp.arange(layout.n_devices * layout.nx_local)
    row = merged % layout.nx_local
    real = (row >= layout.ghost_width) & (row < layout.nx_local - layout.ghost_width)
    # Global x row of each merged row; rows at or beyond the realized nx are
    # alignment rows (or the last slab's high ghost) and carry no sheet edge.
    global_row = (merged // layout.nx_local) * layout.nx_per_rank + row - layout.ghost_width
    inside = global_row < layout.nx
    for field in fields(context):
        edge = field.name != 'sigma_sheet'
        value = cut(getattr(context, field.name), layout,
                    'pec_edge' if edge else 'sigma', mesh=mesh)
        if edge and value is not None:
            keep = (real & inside) if uniform else inside
            value = value & keep[:, None, None]
        values[field.name] = value
    coefficients = jax.tree.map(lambda a: cut(a, layout, 'eps_r', mesh=mesh), coefficients)
    return replace(context, **values), coefficients
