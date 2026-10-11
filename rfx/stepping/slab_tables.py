"""Point-sample tables cut to x slabs: placement only, no new arithmetic.

A channel is linear in the field, so its sample is the sum over slabs of each
slab's owned share. ``cut_sample_table`` moves a global
``rfx.measurement.projector.SampleTable`` to slab-local rows (a node is kept
only on the slab that owns its x row; a ghost row is never read).
``slab_project`` is ``projector.project`` on one slab's arrays and returns
that slab's partial channel values with no cross-device operation;
``reduce_channels`` sums the slabs once, after the time loop.
"""
import jax
import jax.numpy as jnp
import numpy as np

_COMPONENTS = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')


@jax.tree_util.register_pytree_node_class
class SlabSampleTable:
    """Per-slab terms stacked on a leading slab axis (or one slab's row).

    ``indices[c]`` is a local (i, j, k) triple of int32 arrays, ``weights[c]``
    the matching weights; ``order`` / ``segment_ids`` sort one slab's
    concatenated terms by channel. Padding reads owned row ``ghost_width`` at
    j = k = 0 with weight 0 into the dummy segment ``n_channels``.
    """

    def __init__(self, components, indices, weights, order, segment_ids, n_channels):
        self.components = tuple(components)
        self.indices = tuple(tuple(ijk) for ijk in indices)
        self.weights = tuple(weights)
        self.order = order
        self.segment_ids = segment_ids
        self.n_channels = int(n_channels)

    def tree_flatten(self):
        return ((self.indices, self.weights, self.order, self.segment_ids),
                (self.components, self.n_channels))

    @classmethod
    def tree_unflatten(cls, aux, children):
        indices, weights, order, segment_ids = children
        return cls(aux[0], indices, weights, order, segment_ids, aux[1])

    def row(self, r):
        """One slab's arrays (leading slab axis removed)."""
        return jax.tree.map(lambda a: a[r], self)


def cut_sample_table(table, slab):
    """Cut a global-node SampleTable to per-slab local tables.

    Within each slab and component the global table's relative order is kept;
    every slab is padded to the same length per component.
    """
    n, gw, nc = slab.n_devices, slab.ghost_width, table.n_channels
    # Vectorised Slab.owner / Slab.local_index (same arithmetic, array input).
    lows = np.asarray([lo for lo, _ in slab.owned], dtype=np.int64)
    components, indices, weights = [], [], []
    channel_ids = [[] for _ in range(n)]
    for terms in table.components:
        i, j, k = (np.asarray(a, dtype=np.int64) for a in terms.indices)
        if np.any((i < 0) | (i >= slab.nx)):
            raise ValueError(f'{terms.component}: x row outside [0, {slab.nx})')
        owner = i // slab.nx_per_rank
        traced = isinstance(terms.weights, jax.core.Tracer)
        xp = jnp if traced else np
        w = terms.weights if traced else np.asarray(terms.weights)
        keeps = []
        for r in range(n):
            local_r = i - lows[r] + gw
            keep = owner == r
            keeps.append((np.nonzero(keep)[0], local_r))
        length = max(len(sel) for sel, _ in keeps)
        rows_i, rows_j, rows_k, rows_w = [], [], [], []
        for r, (sel, local_r) in enumerate(keeps):
            pad = length - len(sel)
            rows_i.append(np.concatenate([local_r[sel], np.full(pad, gw)]))
            rows_j.append(np.concatenate([j[sel], np.zeros(pad, np.int64)]))
            rows_k.append(np.concatenate([k[sel], np.zeros(pad, np.int64)]))
            rows_w.append(xp.concatenate([w[sel], xp.zeros(pad, dtype=w.dtype)]))
            channel_ids[r].append(np.concatenate([np.asarray(terms.channel_ids)[sel],
                                                  np.full(pad, nc)]))
        components.append(terms.component)
        indices.append(tuple(np.stack(rows).astype(np.int32) for rows in (rows_i, rows_j, rows_k)))
        weights.append(xp.stack(rows_w))
    orders, segments = [], []
    for r in range(n):
        ids = np.concatenate(channel_ids[r]) if channel_ids[r] else np.empty(0, np.int64)
        order = np.argsort(ids, kind='stable')
        orders.append(order.astype(np.int32))
        segments.append(ids[order].astype(np.int32))
    return SlabSampleTable(components, indices, weights,
                           np.stack(orders), np.stack(segments), nc)


def slab_project(local_state, row):
    """One slab's partial channel values, shape (n_channels,).

    The same indexed reads and sorted segment sum as ``projector.project``;
    the dummy padding segment is dropped after the sum.
    """
    dims = jax.lax.GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0, 1, 2),
                                         start_index_map=(0, 1, 2))
    parts = [jax.lax.gather(getattr(local_state, c), jnp.stack(ijk, axis=1),
                            dims, (1, 1, 1), mode='promise_in_bounds')
             for c, ijk in zip(row.components, row.indices)]
    if not parts:
        return jnp.zeros((row.n_channels,), dtype=local_state.ex.dtype)
    samples = jnp.concatenate(parts)
    weighted = samples * jnp.concatenate([jnp.asarray(w) for w in row.weights])
    result = jax.ops.segment_sum(weighted[row.order], row.segment_ids,
                                num_segments=row.n_channels + 1, indices_are_sorted=True)
    return result[:row.n_channels]


def reduce_channels(per_slab):
    """Sum per-slab partial channel values over the leading slab axis."""
    return jnp.sum(jnp.asarray(per_slab), axis=0)
