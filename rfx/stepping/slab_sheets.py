"""Place the one-device sheet operands on the existing slab rows."""
from dataclasses import fields, replace

import jax
import jax.numpy as jnp

from rfx.stepping.slab import cut


def cut_sheets(context, coefficients, layout, mesh, *, uniform=False):
    if context is None:
        return None, None
    values = {}
    row = jnp.arange(layout.n_devices * layout.nx_local) % layout.nx_local
    real = (row >= layout.ghost_width) & (row < layout.nx_local - layout.ghost_width)
    for field in fields(context):
        edge = field.name != 'sigma_sheet'
        value = cut(getattr(context, field.name), layout,
                    'pec_edge' if edge else 'sigma', mesh=mesh)
        if uniform and edge and value is not None:
            value = value & real[:, None, None]
        values[field.name] = value
    coefficients = jax.tree.map(lambda a: cut(a, layout, 'eps_r', mesh=mesh), coefficients)
    return replace(context, **values), coefficients
