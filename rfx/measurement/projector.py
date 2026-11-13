"""Static point and rectangular-plane sampling of completed fields."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class ComponentTerms(NamedTuple):
    component: str
    indices: tuple
    channel_ids: np.ndarray
    weights: np.ndarray


class BoxTerm(NamedTuple):
    component: str
    channel_id: int
    indices: tuple
    weights: np.ndarray


def _box_terms(nodes, component, channel_id, dtype):
    """Pack complete rectangular planes; leave all other nodes as points."""
    if not nodes:
        return (), nodes
    coords = np.asarray([(n.i, n.j, n.k) for n in nodes])
    axis = min(range(3), key=lambda a: len(np.unique(coords[:, a])))
    boxes, remaining = [], []
    for plane in sorted(set(coords[:, axis])):
        group = [n for n in nodes if getattr(n, 'ijk'[axis]) == plane]
        points = [(n.i, n.j, n.k) for n in group]
        lo, hi = np.min(points, axis=0), np.max(points, axis=0) + 1
        shape = hi - lo
        if (all(shape[a] >= 2 for a in range(3) if a != axis)
                and len(set(points)) == len(points) == int(np.prod(shape))):
            weights = np.zeros(tuple(shape), dtype=dtype)
            for n, point in zip(group, points):
                weights[tuple(np.asarray(point) - lo)] = n.weight
            boxes.append(BoxTerm(component, channel_id,
                                 tuple(slice(int(a), int(b)) for a, b in zip(lo, hi)), weights))
        else:
            remaining.extend(group)
    return tuple(boxes), remaining


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
    boxes: tuple[BoxTerm, ...] = ()


def sample_table(owners, *, dtype):
    """Compile channels in owner order, packing complete component planes."""
    channels = tuple(c for o in owners for c in o.channels)
    terms, boxes = [], []
    for component in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'):
        nodes: list = []
        ids: list = []
        for channel_id, channel in enumerate(channels):
            selected = [n for n in channel.nodes if n.component == component]
            if selected and component[0].upper() != channel.kind:
                raise ValueError(f'{channel.name}: {channel.kind} channel reads {component}')
            packed, points = _box_terms(selected, component, channel_id, dtype)
            boxes.extend(packed)
            nodes.extend(points if packed else selected)
            ids.extend([channel_id] * len(points if packed else selected))
        if nodes:
            weights = [n.weight for n in nodes]
            xp = jnp if any(isinstance(w, jax.core.Tracer) for w in weights) else np
            terms.append(ComponentTerms(component,
                tuple(np.asarray([getattr(n, a) for n in nodes], dtype=np.int32) for a in 'ijk'),
                np.asarray(ids, dtype=np.int32), xp.asarray(weights, dtype=dtype)))
    ids = np.concatenate([t.channel_ids for t in terms]) if terms else np.empty(0, np.int32)
    order = np.argsort(ids, kind='stable').astype(np.int32)
    return SampleTable(tuple(terms), len(channels), tuple(c.name for c in channels), order, ids[order], tuple(boxes))


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
    if not table.boxes:
        return combine(gather(state, table), table)
    values: list = [None] * table.n_channels
    for component in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz'):
        field = getattr(state, component)
        terms = [(b.channel_id, b.indices, b.weights) for b in table.boxes if b.component == component]
        for t in table.components:
            if t.component == component:
                terms.extend((int(c), tuple(int(i) for i in p), w)
                             for c, p, w in zip(t.channel_ids, zip(*t.indices), t.weights))
        axes = {b.channel_id: next(a for a, s in enumerate(b.indices) if s.stop-s.start == 1)
                for b in table.boxes if b.component == component}
        def plane(term):
            channel, indices, _ = term
            index = indices[axes.get(channel, 0)]
            return channel, index.start if isinstance(index, slice) else index
        for channel, indices, weights in sorted(terms, key=plane):
            value = jnp.sum(field[indices] * jnp.asarray(weights, dtype=field.dtype))
            values[channel] = value if values[channel] is None else values[channel] + value
    return jnp.stack([jnp.asarray(0, state.ex.dtype) if v is None else v for v in values])


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


def plan_port_owners(plan, specs):
    """Select the plan's port owners in spec order, by each spec's port index."""
    owners = {o.id: o for o in plan.owners if o.kind in ('wire', 'lumped')}
    selected = []
    for _, _, kind, mid, port_id in specs:
        key = f'port:{port_id}:mode:0'
        if key not in owners:
            raise ValueError(f'{key}: the measurement plan has no owner for this port')
        owner = owners[key]
        channels = tuple(c for c in owner.channels if c.sample_stage == 'post-injection')
        names = tuple(c.name for c in channels)
        expected = ('V', 'I', 'V_port') if kind == 'wire' else ('V', 'I')
        if names != expected:
            raise ValueError(f'{key}: plan channel order {names}, expected {expected}')
        actual = tuple((n.i, n.j, n.k) for n in channels[0].nodes)
        if actual != (tuple(mid),):
            raise ValueError(f'{key}: plan midpoint {actual}, runner midpoint {tuple(mid)}')
        selected.append(owner)
    return tuple(selected)


def plan_tables(plan, *, n_probes, port_specs, dtype):
    """Compile the step's probe and port tables from the attached plan's owners."""
    from dataclasses import replace
    probes = tuple(o for o in plan.owners if o.kind in ('probe', 'identification'))
    if len(probes) != n_probes:
        raise ValueError(f'the measurement plan has {len(probes)} probe owners; '
                         f'the runner records {n_probes} probes')
    owners = plan_port_owners(plan, port_specs)
    return dict(probe_table=sample_table(probes, dtype=dtype), port_owners=owners,
                port_table=sample_table(tuple(replace(o, channels=tuple(
                    c for c in o.channels if c.sample_stage == 'post-injection')) for o in owners),
                    dtype=dtype))
