"""Shared owner cuts and post-scan reductions for slab measurements."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from rfx.stepping.slab import place


def _place(value, mesh):
    return place(value, NamedSharding(mesh, P('x')))


def _nodes(indices, layout):
    owners = np.asarray(indices[0]) // layout.nx_per_rank
    rows = []
    for rank in range(layout.n_devices):
        keep = np.flatnonzero(owners == rank)
        rows.append(keep)
    width = max(map(len, rows), default=0)
    nodes, selections, valid = [], [], []
    for rank, keep in enumerate(rows):
        n = len(keep)
        local = np.zeros((3, width), np.int32)
        local[0] = layout.ghost_width
        for axis in range(3):
            local[axis, :n] = np.asarray(indices[axis])[keep]
        local[0, :n] += layout.ghost_width - rank * layout.nx_per_rank
        nodes.append(local)
        selections.append(np.pad(keep, (0, width-n)))
        valid.append(np.arange(width) < n)
    return np.asarray(nodes), np.asarray(selections), np.asarray(valid)


def cut_probe_table(table, layout, mesh):
    """Array payload; component names and channel count stay static in wrap."""
    terms, all_ids = [], []
    for term in table.components:
        local, ids, valid = _nodes(term.indices, layout)
        channels = np.where(valid, term.channel_ids[ids], table.n_channels)
        weights = jnp.where(valid, term.weights[ids], 0)
        terms.append((tuple(local[:, axis] for axis in range(3)), channels, weights))
        all_ids.append(channels)
    ids = np.concatenate(all_ids, axis=1) if all_ids else np.empty((layout.n_devices, 0), np.int32)
    order = np.argsort(ids, axis=1, kind='stable').astype(np.int32)
    sorted_ids = np.take_along_axis(ids, order, axis=1)
    assert np.all(np.diff(sorted_ids, axis=1) >= 0), 'slab segment ids must be sorted'
    return jax.tree.map(lambda a: _place(a, mesh), (tuple(terms), order, sorted_ids))


def sum_partials(final, mesh, ranks):
    """One post-scan psum inside shard_map; its transpose broadcasts cotangents."""
    from rfx.stepping.rank import rank_shard_map
    partials = {key: final[key] for key in ('wire_sparams', 'wire_sparam_accs', 'lumped_sparam_accs', 'sparam_time_records') if key in final}
    if not partials:
        return final
    def local(values, *, rank):
        totals = jax.tree.map(lambda a: jax.lax.psum(a[0], 'x'), values)
        if 'sparam_time_records' in values:
            records = []
            for partial, total in zip(values['sparam_time_records'], totals['sparam_time_records']):
                ref = partial[0, :, -1]
                # Only the midpoint owner contributes a reference; other
                # ranks contribute +0. Preserve its signed zero in the
                # stop-gradient time record, which ordinary psum erases.
                negative_zero = jax.lax.psum(((ref == 0) & jnp.signbit(ref)).astype(jnp.int32), 'x') > 0
                ref_total = total[:, -1]
                ref_total = jnp.where((ref_total == 0) & negative_zero,
                                      jnp.asarray(-0., dtype=ref.dtype), ref_total)
                records.append(jnp.concatenate((total[:, :-1], ref_total[:, None]), axis=-1))
            totals['sparam_time_records'] = tuple(records)
        return totals
    totals = rank_shard_map(local, mesh=mesh, in_specs=P('x'), out_specs=P(), check_rep=False)(ranks, partials)
    return dict(final, **totals)
