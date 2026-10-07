"""Path-independent plane sampling and streaming DFT accumulation."""
from .dft import accumulate
from .plan import field_channel


def plane_sample(state, component, axis, index, region=None):
    field = getattr(state, component)
    axes = [a for a in range(3) if a != axis]
    lo1, hi1, lo2, hi2 = region or (0, field.shape[axes[0]], 0, field.shape[axes[1]])
    slices = [slice(None)] * 3
    slices[axis] = index
    slices[axes[0]], slices[axes[1]] = slice(lo1, hi1), slice(lo2, hi2)
    return field[tuple(slices)]


def planes(state, accumulators, metadata, dt, step):
    updated, samples = [], []
    for acc, (channel, axis, index, freqs, region) in zip(accumulators, metadata):
        sample = plane_sample(state, channel.name, axis, index, region)
        samples.append(sample)
        updated.append(accumulate(acc, sample, step, freqs, dt, channel))
    return updated, samples


def flux_samples(state, axis, index, components, region):
    values = []
    for component in components:
        value = plane_sample(state, component, axis, index, region)
        if component.startswith('h'):
            value = (plane_sample(state, component, axis, max(index-1, 0), region) + value) * .5
        values.append(value)
    return tuple(values)


def flux(state, accumulators, metadata, dt, step, *, window_step=None):
    updated = []
    for accs, meta in zip(accumulators, metadata):
        axis, index, freqs, components, lo1, hi1, lo2, hi2, *window = meta
        total, name, alpha = window or (1, 'rect', .5)
        samples = flux_samples(state, axis, index, tuple(c.name for c in components), (lo1, hi1, lo2, hi2))
        updated.append(tuple(accumulate(acc, sample, step, freqs, dt, component,
                                        total_steps=total, window=name, alpha=alpha,
                                        window_step=window_step)
                             for acc, sample, component in zip(accs, samples, components)))
    return updated


def plane_metadata(probes):
    return tuple((field_channel(p.component), p.axis, p.index, p.freqs, p.region) for p in probes)


def flux_metadata(monitors, *, window):
    from rfx.probes.probes import _FLUX_COMPONENTS
    return tuple((m.axis, m.index, m.freqs, tuple(field_channel(c) for c in _FLUX_COMPONENTS[m.axis]),
                  m.lo1, m.hi1, m.lo2, m.hi2)
                 + ((m.total_steps, m.window, m.window_alpha) if window else ()) for m in monitors)
