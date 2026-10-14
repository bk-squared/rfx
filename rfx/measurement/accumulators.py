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
