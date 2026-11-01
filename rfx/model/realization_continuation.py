"""Carry material ownership through the solver's existing absorber copies."""
import numpy as np
import jax.numpy as jnp


def owner_tags(sim, grid, geometry_masks):
    if not geometry_masks or not any(getattr(grid, f"pad_{a}_{s}") for a in "xyz" for s in ("lo", "hi")):
        return []
    masks = dict(geometry_masks)
    owner = jnp.full(grid.shape, -1, dtype=jnp.int32)
    for i, entry in enumerate(sim._geometry):
        if id(entry) in masks:
            owner = jnp.where(masks[id(entry)], i, owner)
    return [owner]


def retain_faces(grid, tags):
    from rfx.core.jax_utils import is_tracer
    if not tags or is_tracer(tags[0]):
        return
    owner = np.asarray(tags[0])
    continued = {}
    for a, axis in enumerate("xyz"):
        for side, word in enumerate(("lo", "hi")):
            depth = getattr(grid, f"pad_{axis}_{word}")
            if not depth:
                continue
            region = [slice(None)] * 3
            region[a] = slice(-depth, None) if side else slice(0, depth)
            for index in np.unique(owner[tuple(region)]):
                if index >= 0:
                    continued.setdefault(int(index), []).append(2*a+side)
    from rfx.model.realization_integrals import quantities
    quantities(grid, create=True)["continued"] = continued
