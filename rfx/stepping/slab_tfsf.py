"""Owned closed-box TF/SF corrections and replicated incident lines on x slabs."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from rfx.sources.tfsf_discrete_plane_wave import COMPONENTS, update_line_e, update_line_h
from rfx.stepping.rank import rank_shard_map


def split_tables(tables, shape, layout, mesh):
    """Cut rows once on the host, preserving table/row order and excluding ghosts.

    Each operand has an explicit rank axis, sharded on x. Padding targets
    are one past the local flat field; scatter-add drops them.
    """
    result = []
    local_shape = (layout.nx_local, *shape[1:])
    for table in tables:
        x, y, z = np.unravel_index(np.asarray(table.target_index), shape)
        owners = np.array([layout.owner(int(i)) for i in x])
        rows = [np.flatnonzero(owners == rank) for rank in range(layout.n_devices)]
        count = max(1, *(len(row) for row in rows))
        target = np.full((layout.n_devices, count), np.prod(local_shape), np.int32)
        source = np.zeros_like(target)
        coefficient = np.zeros(target.shape, np.float32)
        for rank, row in enumerate(rows):
            local_x = np.array([layout.local_index(int(i)) for i in x[row]], np.int32)
            target[rank, :len(row)] = np.ravel_multi_index((local_x, y[row], z[row]), local_shape)
            source[rank, :len(row)] = np.asarray(table.source_index)[row]
            coefficient[rank, :len(row)] = np.asarray(table.coefficient)[row]
        result.append(tuple(jax.device_put(a, NamedSharding(mesh, P('x')))
                            for a in (target, source, coefficient)))
    return tuple(result)


def prepare(sim, grid, n_steps, layout, mesh):
    """Build the bounded line and both sets of owned table operands."""
    if sim._tfsf is None:
        return None, None, ((), ())
    from rfx.sources.tfsf import tfsf_from_declaration
    from rfx.runners._admission import refuse_slab_tfsf
    cfg, line = tfsf_from_declaration(sim._tfsf, grid, sim._freq_max, n_steps=n_steps)
    refuse_slab_tfsf(cfg)
    tables = tuple(split_tables(ts, grid.shape, layout, mesh) for ts in (cfg.tables_h, cfg.tables_e))
    line = jax.device_put(line, NamedSharding(mesh, P()))
    return cfg, line, tables


def _apply(state, line, operands, cfg, mesh, ranks, *, magnetic):
    if cfg is None:
        return state, line
    tables = cfg.tables_h if magnetic else cfg.tables_e

    @partial(rank_shard_map, mesh=mesh, in_specs=(P('x'), P(), P('x')),
             out_specs=(P('x'), P()), check_rep=False)
    def local(fields, incident, cuts, *, rank):
        fields = list(fields)
        for table, (target, source, coefficient) in zip(tables, cuts):
            field = fields[table.target_component]
            values = getattr(incident, COMPONENTS[table.source_component])[source[0]]
            corrected = field.reshape(-1).at[target[0]].add(coefficient[0] * values, mode='drop')
            fields[table.target_component] = corrected.reshape(field.shape)
        if magnetic and mesh.size > 1:
            # The Yee curl exchanges only Hy/Hz left ghosts. NTFF also
            # reads the right Hx ghost; without the owner correction it
            # retains an uncancelled incident field at a transverse box face.
            hx = fields[3]
            received = jax.lax.ppermute(hx[1], "x", [(r, r - 1) for r in range(1, mesh.size)])
            fields[3] = hx.at[-1].set(jnp.where(rank < mesh.size - 1, received, hx[-1]))
        advance = update_line_h if magnetic else update_line_e
        incident = advance(cfg, incident, cfg.dx_1d, cfg.dt)
        return tuple(fields), incident

    fields, line = local(ranks, tuple(getattr(state, c) for c in COMPONENTS), line, operands)
    return state._replace(**dict(zip(COMPONENTS, fields))), line


def apply_h(state, line, operands, cfg, mesh, *, ranks):
    """Correct H from incident E, then advance the replicated incident H."""
    return _apply(state, line, operands, cfg, mesh, ranks, magnetic=True)


def apply_e(state, line, operands, cfg, mesh, *, ranks):
    """Correct E from incident H, then advance the replicated incident E."""
    return _apply(state, line, operands, cfg, mesh, ranks, magnetic=False)
