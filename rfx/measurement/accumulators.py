"""Path-independent plane sampling and streaming DFT accumulation."""
from .dft import accumulate
from .plan import field_channel
from .flux_weights import normal_h_neighbor


def plane_sample(state, component, axis, index, region=None):
    field = getattr(state, component)
    axes = [a for a in range(3) if a != axis]
    lo1, hi1, lo2, hi2 = region or (0, field.shape[axes[0]], 0, field.shape[axes[1]])
    slices = [slice(None)] * 3
    slices[axis] = index
    slices[axes[0]], slices[axes[1]] = slice(lo1, hi1), slice(lo2, hi2)
    if hi1 > field.shape[axes[0]] or hi2 > field.shape[axes[1]]:
        # A finite periodic window can include the closing node of the period.
        import jax.numpy as jnp
        value = jnp.take(field, index, axis=axis)
        value = jnp.take(value, jnp.arange(lo1, hi1) % field.shape[axes[0]], axis=0)
        return jnp.take(value, jnp.arange(lo2, hi2) % field.shape[axes[1]], axis=1)
    return field[tuple(slices)]


def planes(state, accumulators, metadata, dt, step):
    updated, samples = [], []
    for acc, (channel, axis, index, freqs, region) in zip(accumulators, metadata):
        sample = plane_sample(state, channel.name, axis, index, region)
        samples.append(sample)
        updated.append(accumulate(acc, sample, step, freqs, dt, channel))
    return updated, samples


def flux_samples(state, axis, index, components, regions, normal_h_zero=False,
                 normal_h_weights=(.5, .5), normal_periodic=False):
    values = []
    for component, region in zip(components, regions):
        value = plane_sample(state, component, axis, index, region)
        if component.startswith('h') and normal_h_zero:
            value = value * 0.
        elif component.startswith('h'):
            left = plane_sample(state, component, axis, normal_h_neighbor(index, getattr(state, component).shape[axis], normal_periodic), region)
            value = ((left + value) * .5 if normal_h_weights == (.5, .5) else
                     left * normal_h_weights[0] + value * normal_h_weights[1])
        values.append(value)
    return tuple(values)


def flux(state, accumulators, metadata, dt, step, *, window_step=None):
    updated = []
    for accs, meta in zip(accumulators, metadata):
        axis, index, freqs, components, regions, normal_h_zero, normal_h_weights, normal_periodic, *window = meta
        total, name, alpha = window or (1, 'rect', .5)
        samples = flux_samples(state, axis, index, tuple(c.name for c in components), regions,
                               normal_h_zero, normal_h_weights, normal_periodic)
        updated.append(tuple(accumulate(acc, sample, step, freqs, dt, component,
                                        total_steps=total, window=name, alpha=alpha,
                                        window_step=window_step)
                             for acc, sample, component in zip(accs, samples, components)))
    return updated


def plane_metadata(probes):
    return tuple((field_channel(p.component), p.axis, p.index, p.freqs, p.region) for p in probes)


def flux_metadata(monitors, *, window):
    from rfx.probes.probes import _FLUX_COMPONENTS
    from .flux_weights import flux_regions
    return tuple((m.axis, m.index, m.freqs, tuple(field_channel(c) for c in _FLUX_COMPONENTS[m.axis]),
                  flux_regions(m), m.normal_h_zero, m.normal_h_weights, m.normal_periodic)
                 + ((m.total_steps, m.window, m.window_alpha) if window else ()) for m in monitors)


def validate_flux_prefix(sim, grid, plan, monitors):
    """Declared monitors precede the graded runner's waveguide witnesses."""
    import jax
    import numpy as np
    from rfx.model.graded_grid import NonUniformGrid
    count = len(sim._flux_monitors)
    owners = [o for o in plan.owners if o.kind == 'flux']
    if len(owners) != count or len(monitors) < count:
        raise ValueError(f'flux_plane:{min(len(owners), count)}: runner count mismatch')
    if not isinstance(grid, NonUniformGrid) or not count:
        return
    from rfx.runners.nonuniform import build_flux_monitor_cfgs_nu
    with jax.ensure_compile_time_eval():
        expected = build_flux_monitor_cfgs_nu(sim, grid, plan.time_base.record_length)
    from .flux_weights import flux_regions
    for i, (actual, ref) in enumerate(zip(monitors, expected)):
        keys = ('axis', 'index', 'normal_h_zero', 'normal_h_weights', 'normal_periodic', 'total_steps')
        if (any(getattr(actual, k) != getattr(ref, k) for k in keys)
                or flux_regions(actual) != flux_regions(ref)):
            raise ValueError(f'flux_plane:{i}: runner prefix mismatch')
        for key in ('freqs', 'window', 'window_alpha'):
            a, b = getattr(actual, key), getattr(ref, key)
            if not any(isinstance(v, jax.core.Tracer) for v in jax.tree.leaves((a, b))):
                if not np.array_equal(a, b):
                    raise ValueError(f'flux_plane:{i}: runner prefix mismatch')


def plan_accumulator_meta(plan, dtype, *, grid, dft_planes, flux_monitors, allow_flux_tail=False):
    """Compile boxes once, checking nodes and the runner-owned carry shapes."""
    from .plan import _plane_nodes
    from rfx.probes.probes import _FLUX_COMPONENTS

    def owners(kind, prefix, count, tail=False):
        selected = sorted((o for o in plan.owners if o.kind == kind),
                          key=lambda o: int(o.id.rsplit(':', 1)[1]))
        if len(selected) > count or (not tail and len(selected) != count):
            raise ValueError(f'{prefix}:{min(len(selected), count)}: runner count mismatch')
        for i, owner in enumerate(selected):
            if owner.id != f'{prefix}:{i}':
                raise ValueError(f'{owner.id}: owner order mismatch')
        return selected

    def check(owner, channel, acc):
        if channel.box is None:
            raise ValueError(f'{owner.id}: missing box')
        axis, index, region = channel.box
        axes = [a for a in range(3) if a != axis]
        lo, hi, low, high = region or (0, grid.shape[axes[0]], 0, grid.shape[axes[1]])
        if acc.shape != (len(owner.frequencies), hi-lo, high-low):
            raise ValueError(f'{owner.id}: carry shape mismatch')
        planes: tuple = ((index, 1.),)
        if owner.kind == 'flux' and channel.kind == 'H':
            zero, weights, periodic = owner.normal_h
            planes = (((index, 0.),) if zero else
                      ((normal_h_neighbor(index, grid.shape[axis], periodic), weights[0]),
                       (index, weights[1])))
        expanded = tuple((n.component, n.i, n.j, n.k, weight)
                         for plane, weight in planes
                         for n in _plane_nodes(grid, axis, plane, channel.name, region))
        actual = tuple((n.component, n.i, n.j, n.k, n.weight) for n in channel.nodes)
        if expanded != actual:
            raise ValueError(f'{owner.id}: box and nodes disagree')
        return axis, index, region

    planes, fluxes = [], []
    for owner, probe in zip(owners('dft_plane', 'dft_plane', len(dft_planes)), dft_planes):
        if len(owner.channels) != 1:
            raise ValueError(f'{owner.id}: expected one channel')
        channel = owner.channels[0]
        axis, index, region = check(owner, channel, probe.accumulator)
        planes.append((field_channel(channel.name), axis, index, owner.frequencies, region))
    selected = owners('flux', 'flux_plane', len(flux_monitors), allow_flux_tail)
    for owner, monitor in zip(selected, flux_monitors):
        if len(owner.channels) != 4 or owner.normal_h is None:
            raise ValueError(f'{owner.id}: expected four channels and normal H metadata')
        boxes = [check(owner, c, a) for c, a in zip(owner.channels, monitor[:4])]
        axis, index, _ = boxes[0]
        if (any(box[:2] != (axis, index) for box in boxes)
                or tuple(c.name for c in owner.channels) != _FLUX_COMPONENTS[axis]):
            raise ValueError(f'{owner.id}: inconsistent flux channels')
        fluxes.append((axis, index, owner.frequencies,
                       tuple(field_channel(c.name) for c in owner.channels),
                       tuple(box[2] for box in boxes), *owner.normal_h,
                       plan.time_base.record_length, owner.dft_window, owner.dft_window_alpha))
    return tuple(planes), tuple(fluxes) + flux_metadata(flux_monitors[len(selected):], window=True)
