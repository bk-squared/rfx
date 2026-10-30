"""#2218: public one-device solves are the oracle for the judged classes.

Stateless S3 uses B1 and B2--B4 on CPML and PEC. Stateful S1/S2/S4 use
B1a/B1b and B2--B4 on CPML at 400/800/1600 steps, and B2--B4 only on PEC.
Closed-box stateful one-device parity is not judged: design outcome,
rfx-core-desk option (3).
"""
import json
import sys
from contextlib import contextmanager

import jax
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation, SnapshotSpec
from rfx._grid_metric import field_index


# PR and weekly jobs pin two/three devices through their host configuration.
DEVICES = min(3, len(jax.devices('cpu')))
assert DEVICES >= 2, 'set XLA_FLAGS for at least two host devices before starting Python'
E_COMPONENTS = ("ex", "ey", "ez")


def scene(mesh, boundary, name, *, elements=True):
    dx = .001
    profile = np.asarray([1., *([.9, 1.1] * 11), 1.]) * dx
    sim = Simulation(freq_max=10e9, domain=(.024, .010, .009), dx=dx,
                     boundary=boundary, cpml_layers=2,
                     **({'dx_profile': profile} if mesh == 'graded' else {}))
    grid = sim._build_nonuniform_grid() if mesh == 'graded' else sim._build_grid()
    width = (grid.shape[0] + DEVICES - 1) // DEVICES
    pads = grid.axis_pads

    def position(cell):
        return tuple(float(grid.node_of(axis, index)) for axis, index in enumerate(cell))

    definitions = {
        'S1': ((width, pads[1] + 3, pads[2] + 4), 'ez',
               dict(R=50., L=1e-9, C=1e-12, topology='series')),
        'S2': ((width - 1, pads[1] + 6, pads[2] + 3), 'ex',
               dict(R=100., L=2e-9, C=.5e-12, topology='parallel')),
        'S3': ((width - 2, pads[1] + 2, pads[2] + 6), 'ey', dict(R=10.)),
        'RL': ((width + 3, pads[1] + 5, pads[2] + 2), 'ez',
               dict(R=50., L=1e-9, topology='series')),
    }
    selected = tuple(definitions) if name == 'S4' else (name,)
    realized = []
    channels = []
    for key in selected:
        cell, component, values = definitions[key]
        point = position(cell)
        actual = tuple(map(int, field_index(grid, point, component)))
        expected_owner = 1 if key in ('S1', 'RL') else 0
        expected_local = {'S1': 1, 'S2': width, 'S3': width - 1, 'RL': 4}[key]
        assert actual == cell, ('B3', key, actual, cell)
        realized.append((key, actual, expected_owner, expected_local))
        if elements:
            sim.add_lumped_rlc(point, component, **values)
        for label, probe in (
                ('edge', cell), ('left', (width - 3, cell[1] + 1, cell[2] - 1)),
                ('right', (width + 2, cell[1] - 1, cell[2] + 1))):
            sim.add_probe(position(probe), component)
            channels.append(f'{key}/{component}/{label}')
    sim.add_source(position((width - 4, pads[1] + 4, pads[2] + 5)), 'ez',
                   waveform=GaussianPulse(f0=5e9, bandwidth=1.6), amplitude_kind='field')
    return sim, channels, realized


@contextmanager
def realized_slab(expected, label, enabled):
    """Observe the runner's actual layout at RLC carry construction, before tracing."""
    from rfx.stepping import slab_lumped

    initial = slab_lumped.initial
    seen = []

    def check(metas, layout, mesh, dtype):
        # This exact layout is also passed to SlabRLC / slab_wrap.local_context.
        # Read product ownership, including its mask, rather than repeating its cut.
        for meta, (key, cell, expected_owner, expected_local) in zip(metas, expected, strict=True):
            assert (meta.i, meta.j, meta.k) == cell, ('B3', key, meta, cell)
            owned = []
            for rank in range(layout.n_devices):
                local, owner = slab_lumped.local_meta(meta, layout, rank)
                if bool(owner):
                    owned.append((rank, int(local.i)))
            diagnostic('B3', label, element=key, nx_per_rank=layout.nx_per_rank,
                       actual=owned, expected=[(expected_owner, expected_local)])
            assert owned == [(expected_owner, expected_local)], ('B3', key, owned,
                                                                expected_owner, expected_local)
        seen.append(layout)
        return initial(metas, layout, mesh, dtype)

    with pytest.MonkeyPatch.context() as patch:
        if enabled:
            patch.setattr(slab_lumped, 'initial', check)
        yield
    if enabled:
        assert seen, ('B3', label, 'runner did not build RLC carry')


def solve(mesh, boundary, name, steps, entry='run', *, distributed,
          elements=True, snapshots=False):
    sim, channels, expected = scene(mesh, boundary, name, elements=elements)
    kwargs = dict(n_steps=steps, skip_preflight=True)
    if distributed:
        devices = jax.devices('cpu')[:DEVICES]
        assert len(devices) == DEVICES, 'set XLA_FLAGS before starting Python'
        kwargs['devices'] = devices
    if snapshots:
        assert entry == 'run'
        if mesh == 'graded' and not distributed:
            return observed_graded_run(sim, kwargs), channels
        kwargs['snapshot'] = SnapshotSpec(interval=1, components=E_COMPONENTS)
    with realized_slab(expected, f'{DEVICES}/{mesh}/{boundary}/{name}/{entry}',
                       distributed and elements):
        if entry == 'run':
            result = sim.run(**kwargs, compute_s_params=False)
        else:
            result = sim.forward(**kwargs, distributed=distributed, checkpoint=False)
    jax.block_until_ready(result.time_series)
    return result, channels


def observed_graded_run(sim, kwargs):
    """Instrument note: observe the public solve, without an alternate solver."""
    from rfx.stepping import graded
    from rfx.stepping.sequence import HookPoint

    frames = {c: [] for c in E_COMPONENTS}
    original = graded.make_graded_step

    def observe(context, **step_kwargs):
        def capture(frame):
            def save(*fields):
                for component, value in zip(E_COMPONENTS, fields, strict=True):
                    frames[component].append(np.asarray(value).copy())
            jax.debug.callback(save, *(getattr(frame.st, c) for c in E_COMPONENTS), ordered=True)
        return original(context, hooks={HookPoint.STEP_END: (capture,)}, **step_kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(graded, 'make_graded_step', observe)
        result = sim.run(**kwargs, compute_s_params=False)
        jax.block_until_ready(result.state)
        jax.effects_barrier()
    return result._replace(snapshots={c: np.stack(values) for c, values in frames.items()})


def diagnostic(bar, label, **numbers):
    print('RLC ' + json.dumps(dict(bar=bar, case=label, **numbers)), file=sys.stderr, flush=True)


def accumulated_law(one, many, steps, label):
    peaks, errors = np.zeros(steps), np.zeros(steps)
    for component in E_COMPONENTS:
        expected = np.asarray(one.snapshots[component])
        actual = np.asarray(many.snapshots[component])
        assert expected.shape == actual.shape and expected.shape[0] == steps
        assert np.isfinite(expected).all() and np.isfinite(actual).all()
        axes = tuple(range(1, expected.ndim))
        peaks = np.maximum(peaks, np.max(np.abs(expected.astype(np.float64)), axis=axes))
        errors = np.maximum(errors, np.max(np.abs(actual.astype(np.float64) - expected), axis=axes))
    allowance = 1.2e-7 * np.cumsum(peaks)
    ratios = np.divide(errors, allowance, out=np.zeros(steps), where=allowance != 0)
    ratios[(allowance == 0) & (errors != 0)] = np.inf
    worst = int(np.argmax(ratios))
    diagnostic('B1a', label, worst_ratio=float(ratios[worst]), step=worst + 1,
               error=float(errors[worst]), allowance=float(allowance[worst]))
    return [('B1a', float(ratios[worst]), 1)] if np.any(errors > allowance) else []


CASES = [(boundary, 'S3', 400) for boundary in ('cpml', 'pec')] + [
    ('cpml', name, steps) for name in ('S1', 'S2', 'S4') for steps in (400, 800, 1600)
] + [('pec', name, 1600) for name in ('S1', 'S2', 'S4')]


@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
@pytest.mark.parametrize('boundary,name,steps', CASES)
def test_multidevice_lumped_rlc(mesh, boundary, name, steps):
    """Each case covers run and forward; B4 judges the forward entry."""
    label = f'{DEVICES}/{mesh}/{boundary}/{name}/{steps}'
    diagnostic('class', label, element_class='stateless' if name == 'S3' else 'stateful',
               entries=['run', 'forward'])
    one, channels = solve(mesh, boundary, name, steps, distributed=False)
    many, _ = solve(mesh, boundary, name, steps, distributed=True)
    forward, _ = solve(mesh, boundary, name, steps, 'forward', distributed=True)
    expected, actual, other = (np.asarray(r.time_series) for r in (one, many, forward))
    assert expected.shape == actual.shape == other.shape == (steps, len(channels))
    assert all(np.isfinite(a).all() for a in (expected, actual, other))
    failures = []
    equal = np.array_equal(actual, other)
    diagnostic('B4', label, bit_equal=equal, unequal_samples=int(np.count_nonzero(actual != other)),
               max_difference=float(np.max(np.abs(actual.astype(np.float64) - other))))
    if not equal:
        failures.append(('B4', equal))
    dt = float(one.dt if one.dt is not None else one.grid.dt)
    if name != 'S3' and boundary == 'cpml':
        frames_one, _ = solve(mesh, boundary, name, steps, distributed=False, snapshots=True)
        frames_many, _ = solve(mesh, boundary, name, steps, distributed=True, snapshots=True)
        failures.extend(accumulated_law(frames_one, frames_many, steps, label))
        del frames_one, frames_many
        bins = np.linspace(2e9, 10e9, 41)
        basis = np.exp(-2j * np.pi * bins[:, None] * (np.arange(steps) * dt)[None, :])
        spectra_one, spectra_many = basis @ expected, basis @ actual
    for n, channel in enumerate(channels):
        if name == 'S3':
            peak = np.float32(np.max(np.abs(expected[:, n])))
            assert peak > 0, (label, channel, 'unexcited channel')
            error = float(np.max(np.abs(actual[:, n].astype(np.float64) - expected[:, n])))
            ulps = error / float(np.spacing(peak))
            diagnostic('B1', label, channel=channel, max_ULP=ulps, peak=float(peak))
            if ulps > 9:
                failures.append(('B1', channel, ulps, 9))
        elif boundary == 'cpml':
            peak = float(np.max(np.abs(spectra_one[:, n])))
            assert peak > 0, (label, channel, 'unexcited spectrum')
            relative = float(np.max(np.abs(spectra_many[:, n] - spectra_one[:, n]))) / peak
            diagnostic('B1b', label, channel=channel, relative_difference=relative)
            if relative > 1e-4:
                failures.append(('B1b', channel, relative, 1e-4))
    for entry, present in (('run', actual), ('forward', other)):
        absent_result, _ = solve(mesh, boundary, name, steps, entry, distributed=True, elements=False)
        absent = np.asarray(absent_result.time_series)
        assert absent.shape == actual.shape and np.isfinite(absent).all()
        for n, channel in enumerate(channels):
            if channel.endswith('/edge'):
                peak = float(np.max(np.abs(absent[:, n])))
                assert peak > 0
                percent = 100 * float(np.max(np.abs(present[:, n].astype(np.float64) - absent[:, n]))) / peak
                diagnostic('B2', label, entry=entry, channel=channel, percent=percent)
                if percent < 10:
                    failures.append(('B2', entry, channel, percent, 10))
    assert not failures, (label, failures)
