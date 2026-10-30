"""Global RLC realization and owner-only calls to the one-device element solve."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from rfx.core.yee import FDTDState
from rfx.lumped import (build_rlc_meta, init_rlc_state, refuse_stacked_solved_elements,
                        rlc_carry_dtype, rlc_element_values, setup_rlc_materials)
from rfx.stepping.rank import rank_shard_map


def fold(sim, grid, materials):
    """Fold before ports, using the same material stamp as one device."""
    if sim._lumped_rlc and sim._boundary_spec.pmc_faces():
        raise NotImplementedError('Multi-device add_lumped_rlc with PMC faces is not supported; use one device.')
    for spec in sim._lumped_rlc:
        materials = setup_rlc_materials(grid, spec, materials)
    return materials


def metadata(sim, grid, materials, *, pec_edges=None, uniform=False):
    """Read final global coefficients after every load has been folded."""
    metas = tuple(build_rlc_meta(grid, spec, materials, periodic=sim._periodic_flags())
                  for spec in sim._lumped_rlc)
    refuse_stacked_solved_elements(sim._lumped_rlc, metas)
    if uniform:
        # Uniform slabs enforce PEC after sources; one device enforces it
        # before RLC. Refuse overlapping edges instead of changing that rule.
        for meta in metas:
            cell, axis = (meta.i, meta.j, meta.k), 'xyz'.index(meta.component[1])
            wall = any(cell[a] in (0, grid.shape[a] - 1) for a in range(3) if a != axis)
            metal = pec_edges is not None and bool(np.asarray(pec_edges[axis][cell]))
            if wall or metal:
                raise NotImplementedError('Multi-device add_lumped_rlc on a PEC face or realized PEC edge requires the one-device step order; use one device.')
    return metas


def initial(metas, layout, mesh, dtype=jnp.float32):
    states = tuple(init_rlc_state(dtype=rlc_carry_dtype(metas, dtype)) for _ in metas)
    return jax.tree.map(lambda a: jax.device_put(jnp.broadcast_to(a, (layout.n_devices,)),
                                                NamedSharding(mesh, P('x'))), states)


def local_meta(meta, layout, rank):
    index = jnp.clip(meta.i - rank * layout.nx_per_rank + layout.ghost_width,
                     layout.ghost_width, layout.ghost_width + layout.nx_per_rank - 1)
    return meta._replace(i=index), rank == meta.i // layout.nx_per_rank


def read(state, metas, layout, rank):
    """Read E at frame start, before the electric update."""
    return tuple(getattr(state, m.component)[m.i, m.j, m.k]
                 for m, _ in (local_meta(meta, layout, rank) for meta in metas))


def update(state, states, metas, previous, layout, rank):
    """Discard non-owner reads, writes and ADE advances."""
    updated = []
    fields = {}
    for old, global_meta, e_prev in zip(states, metas, previous, strict=True):
        meta, owner = local_meta(global_meta, layout, rank)
        field = getattr(state, meta.component)
        cell = (meta.i, meta.j, meta.k)
        value, new = rlc_element_values(field[cell], old, meta, e_prev)
        updated.append(jax.tree.map(lambda a, b: jnp.where(owner, a, b), new, old))
        if meta.is_series or meta.has_inductor:
            target = fields.get(meta.component, field)
            fields[meta.component] = target.at[cell].set(
                jnp.where(owner, value.astype(field.dtype), target[cell]))
    return state._replace(**fields), tuple(updated)


@dataclass(frozen=True)
class SlabRLC:
    metas: tuple
    layout: object
    mesh: object

    def read(self, state, *, ranks):
        def local(state, *, rank):
            return tuple(a[None] for a in read(state, self.metas, self.layout, rank))
        spec = FDTDState(*(P('x') for _ in range(6)), step=P())
        return rank_shard_map(local, mesh=self.mesh, in_specs=(spec,),
                              out_specs=P('x'))(ranks, state)

    def update(self, state, states, previous, *, ranks):
        def local(state, states, previous, *, rank):
            state, states = update(state, jax.tree.map(lambda a: a[0], states), self.metas,
                                   jax.tree.map(lambda a: a[0], previous), self.layout, rank)
            return state, jax.tree.map(lambda a: a[None], states)
        spec = FDTDState(*(P('x') for _ in range(6)), step=P())
        return rank_shard_map(local, mesh=self.mesh, in_specs=(spec, P('x'), P('x')),
                              out_specs=(spec, P('x')))(ranks, state, states, previous)
