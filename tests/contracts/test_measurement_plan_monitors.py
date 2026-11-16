"""Retained monitor objects, exact accumulators, and displaced plan sampling."""
from dataclasses import replace
from functools import lru_cache
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.measurement.accumulators import plan_monitor_objects
from rfx.measurement.monitors import ntff_owner, current_moment_owner
from tests.contracts.test_measurement_plan_accumulators import bit_equal, get

DX = 1e-3
FREQS = np.array([0.8e9, 1e9], np.float32)
PATHS = [(mesh, entry) for mesh in ('uniform', 'graded') for entry in ('run', 'forward', 'decay')]
CASES = [('ntff', mesh, entry, '') for mesh in ('uniform', 'graded') for entry in ('run', 'forward')]
CASES += [('ntff', 'uniform', 'run', 'periodic'),
          ('moments', 'uniform', 'run', 'conductor'),
          ('moments', 'uniform', 'forward', 'design'),
          ('moments', 'graded', 'run', 'override')]


def point(*values):
    return tuple(v*DX for v in values)


def scene(kind, mesh, variant='', shift=0):
    kwargs = {}
    if mesh == 'graded':
        for axis in 'xyz':
            profile = np.full(16, DX)
            if axis == 'z':
                profile[5:7] = (.9*DX, 1.1*DX)
            kwargs['d'+axis+'_profile'] = profile
    boundary = BoundarySpec(x='periodic', y='cpml', z='cpml') if variant == 'periodic' else 'cpml'
    sim = Simulation(freq_max=2e9, domain=point(16, 16, 16), dx=DX,
                     boundary=boundary, cpml_layers=2, **kwargs)
    sim.add_source(point(8, 7, 7), 'ez', amplitude_kind='current', waveform=lambda t: jnp.exp(-((t-8e-12)/5e-12)**2))
    if variant == 'conductor':
        sim.add(Box(point(8, 9, 8), point(9, 10, 9)), material='pec')
    lo, hi = point(3+shift, 3, 3), point(12+shift, 12, 12)
    if kind == 'ntff':
        sim.add_ntff_box(lo, hi, freqs=FREQS)
    else:
        sim.add_current_moment_monitor(lo, hi, block_size=3*DX, freqs=FREQS)
    return sim


def invoke(sim, entry, variant=''):
    opts = dict(n_steps=16, skip_preflight=True)
    if variant == 'override':
        from rfx.runners.nonuniform import run_nonuniform_path
        grid = sim._build_nonuniform_grid()
        eps = np.ones(grid.shape, np.float32)
        eps[9:11, 8:10, 8:10] = 2
        return run_nonuniform_path(sim, n_steps=16, compute_s_params=False, eps_override=jnp.asarray(eps))
    if entry == 'decay':
        return sim.run(until_decay=0, decay_max_steps=16, decay_min_steps=16,
                       decay_check_interval=4, compute_s_params=False, skip_preflight=True)
    if entry == 'run':
        return sim.run(compute_s_params=False, **opts)
    if variant == 'design':
        opts.update(design_box=(point(8, 8, 8), point(10, 10, 10)),
                    design_eps_override=jnp.full((3, 3, 3), 2., dtype=jnp.float32))
    return sim.forward(checkpoint=False, **opts)


def capture(sim, entry, variant='', edit=None):
    from rfx.stepping import prepared, uniform, graded
    from rfx import current_moments
    import rfx.runners.uniform as runner
    seen = dict(contexts=[], calls=[], monitors=[])
    advance = prepared.advance
    monitor = current_moments.monitor_for_simulation

    def build_monitor(*args, **kwargs):
        value = monitor(*args, **kwargs)
        # Retain each call's actual conductor and material override arguments.
        # The stepper argument identifies the runner's call independently.
        seen['monitors'].append((value, kwargs))
        return value

    def spy(product, stepper, *args, **kwargs):
        solve = product._prepared_solve
        if edit is not None:
            solve.measurement = edit(solve.measurement, product.grid)
        def stepped(*a, **k):
            result = stepper(*a, **k)
            seen['calls'].append((solve.measurement, a, k, result))
            return result
        return advance(product, stepped, *args, **kwargs)

    def factory(original):
        def make(ctx, *args, **kwargs):
            seen['contexts'].append(ctx)
            return original(ctx, *args, **kwargs)
        return make

    with patch.object(prepared, 'advance', spy), \
            patch.object(uniform, 'make_uniform_step', factory(uniform.make_uniform_step)), \
            patch.object(graded, 'make_graded_step', factory(graded.make_graded_step)), \
            patch.object(current_moments, 'monitor_for_simulation', build_monitor), \
            patch.object(runner, '_cm_for_sim', build_monitor):
        result = invoke(sim, entry, variant)
        jax.block_until_ready(result)
    assert len(seen['calls']) == len(seen['contexts']) == 1
    return seen


def accumulators(seen, kind):
    return get(seen['calls'][0][3], 'ntff_data' if kind == 'ntff' else 'current_moment_data')


def same_object(actual, expected):
    assert type(actual) is type(expected)
    for name in expected._fields:
        a, b = getattr(actual, name), getattr(expected, name)
        if b is None:
            assert a is None
        elif isinstance(b, (int, bool)):
            assert a == b
        else:
            a, b = np.asarray(a), np.asarray(b)
            assert a.dtype == b.dtype and a.shape == b.shape, name
            assert a.tobytes() == b.tobytes(), name


@lru_cache(None)
def paired(kind, mesh, entry, variant=''):
    return (capture(scene(kind, mesh, variant), entry, variant),
            capture(scene(kind, mesh, variant), entry, variant, edit=lambda p, g: None))


@pytest.mark.parametrize('kind,mesh,entry,variant', CASES)
def test_b2_objects(kind, mesh, entry, variant):
    from rfx.farfield import make_ntff_box
    seen, _ = paired(kind, mesh, entry, variant)
    plan, args, kwargs, _ = seen['calls'][0]
    box, monitor = plan_monitor_objects(plan, args[0])
    ctx = seen['contexts'][0]
    if kind == 'ntff':
        expected = make_ntff_box(args[0], *scene(kind, mesh, variant)._ntff)
        same_object(box, expected)
        same_object(getattr(ctx, 'ntff', getattr(ctx, 'ntff_box', None)), expected)
        assert box.freqs is next(o.source.freqs for o in plan.owners if o.kind == 'ntff')
    else:
        expected, inputs = next((value, inputs) for value, inputs in seen['monitors']
                                if value is kwargs['current_moments'])
        same_object(monitor, expected)
        same_object(ctx.current_moments, expected)
        assert inputs['conductors'] is not None
        if variant == 'design':
            assert inputs['overrides']['design_box'] is not None
        if variant == 'override':
            assert inputs['overrides']['eps_r'] is not None


@pytest.mark.parametrize('kind,mesh,entry,variant', CASES)
def test_b1_bits(kind, mesh, entry, variant):
    seen, reference = paired(kind, mesh, entry, variant)
    for a, b in zip(accumulators(seen, kind), accumulators(reference, kind), strict=True):
        bit_equal(a, b)


def replace_owner(plan, owner):
    return replace(plan, owners=tuple(owner if o.kind == owner.kind else o for o in plan.owners))


def shifted(plan, grid, kind):
    from rfx.current_moments import monitor_for_simulation
    from rfx.farfield import make_ntff_box
    mesh = 'graded' if hasattr(grid, 'dx_arr') else 'uniform'
    sim = scene(kind, mesh, shift=1)
    owner = (ntff_owner(grid, make_ntff_box(grid, *sim._ntff)) if kind == 'ntff' else
             current_moment_owner(grid, monitor_for_simulation(sim, grid, sim._periodic_flags())))
    return replace_owner(plan, owner)


def margins(seen, kind):
    from rfx._internal._grid_metric import field_index
    plan, args, kwargs, _ = seen['calls'][0]
    grid = args[0]
    obj = kwargs.get('ntff', kwargs.get('ntff_box')) if kind == 'ntff' else kwargs['current_moments']
    if plan is not None:
        obj = next(o.source for o in plan.owners if o.kind == ('ntff' if kind == 'ntff' else 'current_moment'))
    for axis, letter in enumerate('ijk'):
        low, high = getattr(obj, letter+'_lo'), getattr(obj, letter+'_hi')
        if kind == 'ntff':
            assert low >= getattr(obj, 'cpml_lo_'+'xyz'[axis]) + 2
            assert high <= grid.shape[axis]-1-getattr(obj, 'cpml_hi_'+'xyz'[axis])-2
        else:
            index = field_index(grid, point(8, 7, 7), 'ez')[axis]
            assert low+3 <= index <= high-3


@pytest.mark.parametrize('mesh,entry', PATHS)
@pytest.mark.parametrize('kind', ['ntff', 'moments'])
def test_b3_shifted(kind, mesh, entry):
    seen = capture(scene(kind, mesh), entry, edit=lambda p, g: shifted(p, g, kind))
    reference = capture(scene(kind, mesh, shift=1), entry, edit=lambda p, g: None)
    old = capture(scene(kind, mesh), entry, edit=lambda p, g: None)
    for item in (seen, reference, old):
        margins(item, kind)
    aa, bb, cc = (accumulators(item, kind) for item in (seen, reference, old))
    assert [a.shape for a in aa] == [b.shape for b in bb] == [c.shape for c in cc]
    for a, b in zip(aa, bb, strict=True):
        bit_equal(a, b)
    assert any(np.asarray(a).tobytes() != np.asarray(c).tobytes() for a, c in zip(aa, cc, strict=True))


def bad_weight(plan, grid, kind):
    owner = next(o for o in plan.owners if o.kind == kind)
    channel = owner.channels[0]
    node = channel.nodes[0]
    channel = replace(channel, nodes=(replace(node, weight=node.weight+1.), *channel.nodes[1:]))
    return replace_owner(plan, replace(owner, channels=(channel, *owner.channels[1:])))


@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
@pytest.mark.parametrize('kind,prefix', [('ntff', 'ntff_box'), ('moments', 'current_moment')])
def test_b3_weight_only(kind, prefix, mesh):
    with pytest.raises(ValueError, match=prefix+':0: source and channels disagree'):
        capture(scene(kind, mesh), 'run', edit=lambda p, g: bad_weight(p, g, 'ntff' if kind == 'ntff' else 'current_moment'))


def test_b2_ntff_traced_frequencies():
    original = plan_monitor_objects
    checked = []
    def compile_and_check(plan, grid):
        box, monitor = original(plan, grid)
        owner = next(o for o in plan.owners if o.kind == 'ntff')
        assert box.freqs is owner.source.freqs
        from rfx.farfield import make_ntff_box
        expected = make_ntff_box(grid, *active[0]._ntff)
        same_object(box._replace(freqs=FREQS), expected._replace(freqs=FREQS))
        checked.append(True)
        return box, monitor

    active = []
    def forward(freqs, fallback):
        sim = scene('ntff', 'uniform')
        sim._ntff = (*sim._ntff[:2], freqs)
        active[:] = [sim]
        seen = capture(sim, 'forward', edit=(lambda p, g: None) if fallback else None)
        return accumulators(seen, 'ntff')

    with patch('rfx.measurement.accumulators.plan_monitor_objects', compile_and_check):
        actual = jax.jit(lambda f: forward(f, False))(jnp.asarray(FREQS))
        expected = jax.jit(lambda f: forward(f, True))(jnp.asarray(FREQS))
    assert checked
    for a, b in zip(actual, expected, strict=True):
        bit_equal(a, b)


@pytest.mark.parametrize('fallback', [False, True])
def test_b2_moments_traced_frequencies_refused(fallback):
    def forward(freqs):
        sim = scene('moments', 'uniform')
        spec = list(sim._current_moments)
        spec[3] = freqs
        sim._current_moments = tuple(spec)
        return capture(sim, 'forward', edit=(lambda p, g: None) if fallback else None)
    with patch('rfx.measurement.accumulators.plan_monitor_objects', side_effect=AssertionError('compiler reached')):
        with pytest.raises(jax.errors.TracerArrayConversionError):
            jax.jit(forward)(jnp.asarray(FREQS))


@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
def test_b2_periodic_moments_refused(mesh):
    with patch('rfx.measurement.accumulators.plan_monitor_objects', side_effect=AssertionError('compiler reached')):
        with pytest.raises(NotImplementedError, match='periodic'):
            capture(scene('moments', mesh, 'periodic'), 'run')


def test_d3_traced_weights_compare_position_and_component():
    from copy import copy
    from tests.contracts.test_measurement_plan_seam import plan_for
    sim = scene('moments', 'uniform')
    grid = sim._build_grid()
    base = plan_for(sim, grid)
    owner = next(o for o in base.owners if o.kind == 'current_moment')

    def compile_dt(dt, defect):
        traced_grid = copy(grid)
        object.__setattr__(traced_grid, 'dt', dt)
        expanded = current_moment_owner(traced_grid, owner.source)
        channel = expanded.channels[0]
        node = channel.nodes[0]
        assert isinstance(node.weight, jax.core.Tracer)
        updates = {'weight': node.weight*2}
        if defect == 'position':
            updates['i'] = node.i+1
        if defect == 'component':
            updates['component'] = 'hy'
        channel = replace(channel, nodes=(replace(node, **updates), *channel.nodes[1:]))
        changed = replace(expanded, channels=(channel, *expanded.channels[1:]))
        _, monitor = plan_monitor_objects(replace_owner(base, changed), traced_grid)
        return monitor.freqs

    np.testing.assert_array_equal(jax.jit(lambda dt: compile_dt(dt, 'weight'))(grid.dt), FREQS)
    for defect in ('position', 'component'):
        with pytest.raises(ValueError, match='current_moment:0'):
            jax.jit(lambda dt: compile_dt(dt, defect))(grid.dt)


@pytest.mark.parametrize('kind,prefix', [('ntff', 'ntff_box'), ('moments', 'current_moment')])
@pytest.mark.parametrize('defect', ['missing', 'unavailable', 'shape', 'frequency', 'source'])
def test_invalid_monitor_contract(kind, prefix, defect):
    def edit(plan, grid):
        owner = next(o for o in plan.owners if o.id == prefix+':0')
        if defect == 'missing':
            return replace(plan, owners=tuple(o for o in plan.owners if o is not owner))
        if defect == 'unavailable':
            return replace_owner(plan, replace(owner, availability='unavailable: test'))
        if defect == 'source':
            return replace_owner(plan, replace(owner, source=None))
        if defect == 'frequency':
            return replace_owner(plan, replace(owner, frequencies=(np.float32(2e9),)*2))
        obj = owner.source._replace(freqs=owner.source.freqs[:1])
        expand = ntff_owner if kind == 'ntff' else current_moment_owner
        return replace_owner(plan, expand(grid, obj))
    with pytest.raises(ValueError, match=prefix+':0'):
        capture(scene(kind, 'uniform'), 'run', edit=edit)


def test_unavailable_without_runner_object_and_source_equality():
    from tests.contracts.test_measurement_plan_seam import plan_for
    from rfx.measurement.accumulators import validate_monitor_presence
    sim = scene('moments', 'uniform', 'periodic')
    grid = sim._build_grid()
    plan = plan_for(sim, grid)
    assert any(o.availability.startswith('unavailable:') for o in plan.owners)
    compiled = plan_monitor_objects(plan, grid)
    assert compiled == (None, None)
    validate_monitor_presence(compiled, (None, None))
    owner = plan.owners[-1]
    changed = replace(owner, source=object())
    assert owner == changed and hash(owner) == hash(changed) and repr(owner) == repr(changed)
