"""Owned closed-box TF/SF corrections and replicated incident lines on x slabs."""
import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P



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
