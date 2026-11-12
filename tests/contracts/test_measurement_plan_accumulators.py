"""Plan-owned plane and flux metadata, bit identity and displaced sampling."""
from dataclasses import replace
from functools import lru_cache
from unittest.mock import patch

import jax
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.measurement.accumulators import plane_metadata, flux_metadata
from rfx.stepping.sequence import HookPoint
from tests.contracts.path_equivalence.builders import build, DX, FREQS, point, waveform
from tests.contracts.test_measurement_plan_seam import flux_scene, exact

PATHS = [('uniform', 'run'), ('uniform', 'forward'), ('uniform', 'decay'),
         ('graded', 'run'), ('graded', 'forward')]
CASES = [(name, mesh, entry) for mesh, entry in PATHS
         for name in ('full', 'region', 'hann') if name != 'hann' or entry != 'decay']
CASES += [(name, 'uniform', entry) for name in ('periodic', '2d', 'floquet', 'index0', 'zero')
          for entry in ('run', 'forward')]
CASES += [('waveguide', 'graded', 's_matrix')]


def scene(name, mesh):
    graded = mesh == 'graded'
    if name in ('periodic', '2d', 'floquet'):
        sim, _ = flux_scene(name)
        sim._flux_monitors[0] = replace(sim._flux_monitors[0], dft_window='rect')
        if name == 'periodic':
            sim.add_flux_monitor(axis='y', coordinate=3*DX, freqs=FREQS,
                                 size=(2*DX, 2*DX), center=(10*DX, 3*DX))
        if name == '2d':
            sim.add_source(point(3, 3, 0), 'ez', waveform=waveform)
    elif name in ('index0', 'zero'):
        sim = Simulation(freq_max=15e9, domain=point(10, 8, 6), dx=DX,
                         boundary=BoundarySpec(x='pmc' if name == 'zero' else 'pec',
                                               y='pec', z='pec'))
        sim.add_source(point(2, 3, 3), 'ez', waveform=waveform)
        sim.add_flux_monitor(axis='x', coordinate=0., freqs=FREQS)
    else:
        if name == 'waveguide':
            from tests.contracts.test_waveguide_port_traced_mesh_refused import _section, DX as WG_DX
            sim = _section(.25*WG_DX, 'x')
            sim.add_flux_monitor(axis='x', coordinate=4*WG_DX, freqs=FREQS)
        else:
            sim = build(('_flux_monitors', 'flux'),
                        'run_nonuniform' if graded else 'run_uniform', graded=graded)
        if name == 'region':
            sim._flux_monitors[0] = replace(sim._flux_monitors[0],
                                            size=(2*DX, 2*DX), center=(3*DX, 3*DX))
        if name == 'hann':
            sim._flux_monitors[0] = replace(sim._flux_monitors[0], dft_window='hann')
    axis = 'z' if name == '2d' else 'x'
    for component in ('ez', 'hy'):
        sim.add_dft_plane_probe(axis=axis, coordinate=0. if name == '2d' else 4*DX,
                                component=component, freqs=FREQS, name=component)
    if name == 'region':
        sim._dft_plane_regions = {'ez': (2, 5, 2, 4), 'hy': (2, 5, 2, 4)}
    sim._time_recorded_dft_planes = {'ez', 'hy'}
    return sim


def invoke(sim, entry):
    if entry == 's_matrix':
        return sim.compute_waveguide_s_matrix(n_steps=12, normalize='flux', skip_preflight=True)
    opts = dict(skip_preflight=True)
    if entry == 'decay':
        return sim.run(until_decay=0, decay_max_steps=12, decay_min_steps=12,
                       decay_check_interval=4, compute_s_params=False, **opts)
    if entry == 'run':
        return sim.run(n_steps=12, compute_s_params=False, **opts)
    return sim.forward(n_steps=12, checkpoint=False, **opts)


def capture(sim, entry, edit=None, fields=False):
    from rfx.stepping import prepared, uniform, graded
    seen = dict(calls=[], contexts=[], fields=[])
    advance = prepared.advance

    def spy(product, stepper, *args, **kwargs):
        solve = product._prepared_solve
        if edit is not None:
            solve.measurement = edit(solve.measurement)
        plan = solve.measurement

        def stepped(*a, **k):
            result = stepper(*a, **k)
            jax.block_until_ready(result)
            seen['calls'].append((plan, a, k, result))
            return result
        return advance(product, stepped, *args, **kwargs)

    def factory(original):
        def make(ctx, *args, **kwargs):
            seen['contexts'].append(ctx)
            if fields:
                def keep(frame):
                    def save(n, *values):
                        seen['fields'].append((int(n), tuple(np.asarray(v) for v in values)))
                    jax.debug.callback(save, frame.step_idx,
                                       *(getattr(frame.st, c) for c in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')),
                                       ordered=True)
                kwargs['hooks'] = {HookPoint.AFTER_SOURCES: (keep,)}
            return original(ctx, *args, **kwargs)
        return make
    with patch.object(prepared, 'advance', spy), \
            patch.object(uniform, 'make_uniform_step', factory(uniform.make_uniform_step)), \
            patch.object(graded, 'make_graded_step', factory(graded.make_graded_step)):
        invoke(sim, entry)
        jax.effects_barrier()
    assert seen['calls'] and seen['contexts']
    return seen


def get(result, key):
    return result.get(key, ()) if isinstance(result, dict) else getattr(result, key)


def arrays(result):
    return ([p.accumulator for p in get(result, 'dft_planes') or ()],
            [a for m in get(result, 'flux_monitors') or () for a in m[:4]],
            list(get(result, 'dft_time_records') or ()))


def bit_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape
    for left, right in ((a.real, b.real), (a.imag, b.imag)):
        assert left.dtype == right.dtype == np.float32
        np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))


def same_meta(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            if hasattr(x, 'kind'):
                assert x == y
            elif isinstance(x, tuple) and x and hasattr(x[0], 'kind'):
                assert x == y
            elif y is None:
                assert x is None
            elif isinstance(y, str):
                assert x == y
            else:
                exact(x, y)
        if len(b) in (8, 11):
            assert isinstance(a[6], tuple)


@lru_cache(None)
def paired(name, mesh, entry):
    return (capture(scene(name, mesh), entry),
            capture(scene(name, mesh), entry, edit=lambda plan: None))


@pytest.mark.parametrize('name, mesh, entry', [
    (name, mesh, entry)
    for name,mesh,entry in CASES
    if (name, mesh, entry) not in (('index0', 'uniform', 'run'), ('zero', 'uniform', 'run'), ('hann', 'graded', 'forward'), ('hann', 'uniform', 'forward'), ('region', 'graded', 'forward'), ('region', 'uniform', 'decay'), ('region', 'uniform', 'forward'))
])
def test_b2_metadata(name, mesh, entry):
    seen, _ = paired(name, mesh, entry)
    for (_, _, kwargs, _), ctx in zip(seen['calls'], seen['contexts'], strict=True):
        same_meta(ctx.dft_meta, plane_metadata(kwargs.get('dft_planes') or ()))
        same_meta(ctx.flux_meta, flux_metadata(kwargs.get('flux_monitors') or (), window=entry != 'decay'))
        assert ctx.flux_meta
        assert ctx.dft_meta or name == 'waveguide'
        if name == 'zero':
            assert ctx.flux_meta[0][5]
        if name == 'index0':
            assert ctx.flux_meta[0][1] == 0 and not ctx.flux_meta[0][7]
        if name == 'periodic':
            assert any(r[1] > ctx.grid.shape[0] for r in ctx.flux_meta[1][4])
        if name == 'waveguide':
            assert len(ctx.flux_meta) == len(kwargs['flux_monitors'])
    if name == 'waveguide':
        assert any(len(ctx.flux_meta) == 3 for ctx in seen['contexts'])


@pytest.mark.parametrize('name, mesh, entry', [
    (name, mesh, entry)
    for name,mesh,entry in CASES
    if (name, mesh, entry) not in (('index0', 'uniform', 'run'), ('zero', 'uniform', 'run'), ('hann', 'graded', 'forward'), ('hann', 'uniform', 'forward'), ('region', 'graded', 'forward'), ('region', 'uniform', 'decay'), ('region', 'uniform', 'forward'))
])
def test_b1_bits(name, mesh, entry):
    seen, reference = paired(name, mesh, entry)
    for actual, expected in zip(seen['calls'], reference['calls'], strict=True):
        for aa, bb in zip(arrays(actual[3]), arrays(expected[3]), strict=True):
            assert len(aa) == len(bb)
            assert aa or name == 'waveguide'
            for a, b in zip(aa, bb, strict=True):
                bit_equal(a, b)


def moved(plan, kind, nodes_only=False):
    owners = list(plan.owners)
    slot = next(i for i, owner in enumerate(owners) if owner.kind == kind)
    owner = owners[slot]
    channels = []
    for c in owner.channels:
        axis, index, region = c.box
        nodes = tuple(replace(n, **{'ijk'[axis]: getattr(n, 'ijk'[axis])+1}) for n in c.nodes)
        channels.append(replace(c, nodes=nodes, box=c.box if nodes_only else (axis, index+1, region)))
    owners[slot] = replace(owner, channels=tuple(channels))
    return replace(plan, owners=tuple(owners))


def samples(fields, axis, index, region, component):
    values = np.stack([f['ex ey ez hx hy hz'.split().index(component)] for _, f in sorted(fields)])
    plane = np.take(values, index, axis=axis+1)
    if region is not None:
        lo, hi, low, high = region
        plane = plane[:, lo:hi, low:high]
    return plane


def expected_flux(fields, meta, dt, displacement):
    axis, index, freqs, channels, regions, zero, weights, periodic, *_ = meta
    result = []
    for c, region in zip(channels, regions):
        values = samples(fields, axis, index+displacement, region, c.name).astype(np.float64)
        if c.kind == 'H':
            assert not zero and not periodic
            left = samples(fields, axis, max(index+displacement-1, 0), region, c.name).astype(np.float64)
            values = left*weights[0] + values*weights[1]
        total = np.zeros((len(freqs), *values.shape[1:]), np.complex128)
        for (step, _), value in zip(sorted(fields), values, strict=True):
            time = (step+(1. if c.kind == 'E' else .5))*float(dt)
            phase = np.exp(-2j*np.pi*np.asarray(freqs, np.float64)*time)*float(dt)
            total += value[None]*phase[:, None, None]
        result.append(total)
    return result


@pytest.mark.parametrize('mesh,entry', PATHS)
@pytest.mark.parametrize('kind', ['dft_plane', 'flux'])
def test_b3_moved(mesh, entry, kind):
    seen = capture(scene('full', mesh), entry, edit=lambda p: moved(p, kind), fields=True)
    plan, _, kwargs, result = seen['calls'][0]
    owner = next(o for o in plan.owners if o.kind == kind)
    assert len(seen['fields']) == 12
    if kind == 'dft_plane':
        c = owner.channels[0]
        expected = samples(seen['fields'], *c.box, c.name)
        bit_equal(get(result, 'dft_time_records')[0], expected)
        old = samples(seen['fields'], c.box[0], c.box[1]-1, c.box[2], c.name)
        assert np.any(old != expected)
    else:
        meta = flux_metadata(kwargs['flux_monitors'], window=True)[0]
        expected = expected_flux(seen['fields'], meta, plan.time_base.dt, 1)
        old = expected_flux(seen['fields'], meta, plan.time_base.dt, 0)
        for actual, target, unmoved in zip(get(result, 'flux_monitors')[0][:4], expected, old, strict=True):
            tolerance = 1e-4*np.max(np.abs(target))
            assert tolerance > 0
            assert np.max(np.abs(target-unmoved)) >= 10*tolerance
            assert np.max(np.abs(np.asarray(actual)-target)) <= tolerance


@pytest.mark.parametrize('mesh,entry', PATHS)
@pytest.mark.parametrize('kind,prefix', [('dft_plane', 'dft_plane'), ('flux', 'flux_plane')])
def test_b3_nodes_only(mesh, entry, kind, prefix):
    with pytest.raises(ValueError, match=prefix+':0: box and nodes disagree'):
        capture(scene('full', mesh), entry, edit=lambda p: moved(p, kind, nodes_only=True))


@pytest.mark.parametrize('kind', ['dft_plane', 'flux'])
@pytest.mark.parametrize('defect', ['count', 'shape'])
def test_d5_refuses_incompatible_carry(kind, defect):
    from rfx.measurement.accumulators import plan_accumulator_meta, validate_flux_prefix
    seen, _ = paired('full', 'graded', 'run')
    plan, args, kwargs, _ = seen['calls'][0]
    owners = list(plan.owners)
    slot = next(i for i, owner in enumerate(owners) if owner.kind == kind)
    prefix = owners[slot].id.split(':')[0]
    if defect == 'count':
        owners.pop(slot)
    else:
        owner = owners[slot]
        c = owner.channels[0]
        axis, index, region = c.box
        axes = [a for a in range(3) if a != axis]
        lo, hi, low, high = region or (0, args[0].shape[axes[0]], 0, args[0].shape[axes[1]])
        owners[slot] = replace(owner, channels=(replace(c, box=(axis, index, (lo, hi-1, low, high))),
                                               *owner.channels[1:]))
    plan = replace(plan, owners=tuple(owners))
    with pytest.raises(ValueError, match=prefix+':'):
        if kind == 'flux' and defect == 'count':
            validate_flux_prefix(scene('full', 'graded'), args[0], plan, kwargs['flux_monitors'])
        else:
            plan_accumulator_meta(plan, np.float32, grid=args[0], dft_planes=kwargs['dft_planes'],
                                  flux_monitors=kwargs['flux_monitors'], allow_flux_tail=True)


def test_d8_refuses_wrong_runner_prefix():
    from rfx.measurement.accumulators import validate_flux_prefix
    seen, _ = paired('full', 'graded', 'run')
    plan, args, kwargs, _ = seen['calls'][0]
    monitors = list(kwargs['flux_monitors'])
    monitors[0] = monitors[0]._replace(index=monitors[0].index+1)
    with pytest.raises(ValueError, match='flux_plane:0: runner prefix mismatch'):
        validate_flux_prefix(scene('full', 'graded'), args[0], plan, monitors)
