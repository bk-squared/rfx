"""Point sampling: one gather per component, one sorted channel reduction."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class ComponentTerms(NamedTuple):
    component: str
    indices: tuple
    channel_ids: np.ndarray
    weights: np.ndarray


class SampleTable(NamedTuple):
    """Static host terms in ex, ey, ez, hx, hy, hz order.

    Only a differentiated mesh has traced weights; its indices and ordering
    remain static. Channel names retain owner order (and can repeat).
    """
    components: tuple[ComponentTerms, ...]
    n_channels: int
    channel_names: tuple[str, ...]
    order: np.ndarray
    segment_ids: np.ndarray


def sample_table(owners, *, dtype):
    """Concatenate point channels in owner order, then each owner's order."""
    channels = tuple(c for o in owners for c in o.channels)
    terms = []
    for component in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'):
        nodes, ids = [], []
        for channel_id, channel in enumerate(channels):
            for node in channel.nodes:
                if node.component[0].upper() != channel.kind:
                    raise ValueError(f'{channel.name}: {channel.kind} channel reads {node.component}')
                if node.component == component:
                    nodes.append(node)
                    ids.append(channel_id)
        if nodes:
            weights = [n.weight for n in nodes]
            xp = jnp if any(isinstance(w, jax.core.Tracer) for w in weights) else np
            terms.append(ComponentTerms(component,
                tuple(np.asarray([getattr(n, a) for n in nodes], dtype=np.int32) for a in 'ijk'),
                np.asarray(ids, dtype=np.int32), xp.asarray(weights, dtype=dtype)))
    ids = np.concatenate([t.channel_ids for t in terms]) if terms else np.empty(0, np.int32)
    order = np.argsort(ids, kind='stable').astype(np.int32)
    return SampleTable(tuple(terms), len(channels), tuple(c.name for c in channels), order, ids[order])


def gather(state, table):
    """One indexed read per present field component; return samples[n_terms]."""
    dims = jax.lax.GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0, 1, 2),
                                         start_index_map=(0, 1, 2))
    parts = [jax.lax.gather(getattr(state, t.component),
                            (jnp if any(isinstance(i, jax.core.Tracer) for i in t.indices) else np).stack(t.indices, axis=1),
                            dims, (1, 1, 1), mode='promise_in_bounds')
             for t in table.components]
    return jnp.concatenate(parts) if parts else jnp.empty((0,), dtype=state.ex.dtype)


def combine(samples, table):
    """Reduce samples[n_terms, ...] to values[..., n_channels].

    Terms are first; optional trailing dimensions (including time) are moved
    ahead of channels in the result. Sorting is compiled on the host.
    NumPy samples with concrete weights stay in the samples' dtype on the
    host: np.add.at accumulates reordered terms sequentially by segment id.
    """
    traced = (any(isinstance(t.weights, jax.core.Tracer) for t in table.components)
              or isinstance(table.order, jax.core.Tracer)
              or isinstance(table.segment_ids, jax.core.Tracer))
    xp = jnp if traced else np
    weights = xp.concatenate([t.weights for t in table.components]) if table.components else np.empty(0)
    if isinstance(samples, np.ndarray) and not traced:
        weighted = samples * np.asarray(weights, dtype=samples.dtype).reshape(
            (-1,) + (1,) * (samples.ndim - 1))
        result = np.zeros((table.n_channels,) + samples.shape[1:], dtype=samples.dtype)
        np.add.at(result, table.segment_ids, weighted[table.order])
        return np.moveaxis(result, 0, -1)
    weighted = samples * jnp.asarray(weights).reshape((-1,) + (1,) * (samples.ndim - 1))
    result = jax.ops.segment_sum(weighted[table.order], table.segment_ids,
                                num_segments=table.n_channels, indices_are_sorted=True)
    return jnp.moveaxis(result, 0, -1)


def project(state, table):
    """Sample completed fields and evaluate all channels together."""
    return combine(gather(state, table), table)


def port_owners(grid, specs, *, nu, boundary):
    """Compile (component, live cells, kind, midpoint, port index) specs; pin record order."""
    from .ports import lumped_wire_owner
    sampled = []
    for i, (component, live, kind, mid, port_id) in enumerate(specs):
        owner = lumped_wire_owner(grid, component, live, nu=nu,
                    pmc_faces=boundary.pmc_faces, periodic=boundary.periodic,
                    port_id=port_id, kind=kind)
        channels = tuple(c for c in owner.channels if c.sample_stage == 'post-injection')
        names = tuple(c.name for c in channels)
        expected = ('V', 'I', 'V_port') if kind == 'wire' else ('V', 'I')
        if names != expected:
            raise ValueError(f'{owner.id}: channel order {names}, expected {expected}')
        nodes = channels[0].nodes
        actual = tuple((n.i, n.j, n.k) for n in nodes)
        if actual != (tuple(mid),):
            raise ValueError(f'{owner.id}: owner midpoint {actual}, runner midpoint {tuple(mid)}')
        sampled.append(owner)
    return tuple(sampled)


def port_table(grid, specs, *, nu, boundary, dtype):
    """Evaluate owners in order, retaining only completed-field channels."""
    from dataclasses import replace
    owners = port_owners(grid, specs, nu=nu, boundary=boundary)
    return sample_table(tuple(replace(o, channels=tuple(
        c for c in o.channels if c.sample_stage == 'post-injection')) for o in owners),
        dtype=dtype)
