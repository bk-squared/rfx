"""Rank-stacked current-moment operands and their static local windows."""
from typing import NamedTuple

import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from rfx.stepping.slab import place


class MomentBounds(NamedTuple):
    i_lo: int
    i_hi: int
    j_lo: int
    j_hi: int
    k_lo: int
    k_hi: int


def snapshot_bounds(monitor):
    return MomentBounds(*(getattr(monitor, name) for name in MomentBounds._fields))


def moment_operands(parts, mesh):
    """Pad only the owner-row axis, then shard the leading rank axis."""
    result = {}
    for name in ('w_ex', 'w_ey', 'w_ez', 'inv_dx_e', 'seg'):
        arrays = [np.asarray(getattr(m, name)) for m in parts]
        if name == 'seg':
            arrays = [a.reshape(m.i_hi - m.i_lo, m.j_hi - m.j_lo)
                      for a, m in zip(arrays, parts)]
        axis = 1 if name.startswith('w_') else 0
        width = max(a.shape[axis] for a in arrays)
        padded = []
        for a in arrays:
            padding = [(0, 0)] * a.ndim
            padding[axis] = (0, width - a.shape[axis])
            padded.append(np.pad(a, padding))
        result[name] = place(np.stack(padded), NamedSharding(mesh, P('x')))
    return result


def moment_metadata(monitor):
    """Keep local bounds and the small shared parameters used by accumulation."""
    return monitor._replace(w_ex=None, w_ey=None, w_ez=None, inv_dx_e=None,
                            seg=None, centres=None, block_key=None)


def local_monitor(metadata, operands):
    """Remove padding at the original static row count before any arithmetic."""
    rows = metadata.i_hi - metadata.i_lo
    values = {name: operands[name][0, :, :rows]
              for name in ('w_ex', 'w_ey', 'w_ez')}
    values.update(inv_dx_e=operands['inv_dx_e'][0, :rows],
                  seg=operands['seg'][0, :rows].reshape(-1))
    return metadata._replace(**values)
