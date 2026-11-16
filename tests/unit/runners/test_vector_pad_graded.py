"""Vector padding on the graded (non-uniform) single-device step.

The graded step may allocate its arrays with padded extents, as the uniform
step does (``tests/unit/runners/test_vector_pad.py``). Its H metrics
``inv_d*_h`` are padded with 0 and its E metrics ``inv_d*`` with the last
physical value; dt comes from the physical profile. The physical cells,
probes, ports, planes and the end state must not see the pad: on the CPU each
padded run returns the unpadded run's bytes, or, for the cases listed in
``NOT_BITWISE``, stays within the bars of the uniform test. For every scene
and both targets, one padded step from an unpadded state stays within nine
last bits of the field, and every operand the padded step reads equals the
unpadded operand on the physical cells.
"""

import dataclasses
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, DebyePole, Simulation, lorentz_pole
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.stepping import vector_pad
from tests.unit.runners.test_vector_pad import (
    FIELDS, FREQS, LENGTHS, N_STEPS, ONE_STEP_BAR, PULSE, TARGETS, _arrays, _Captured, _ratio,
    _walls, assert_bars, assert_bitwise, assert_padded, bar_ratios, lumped, observed,
)

#: A z profile with four cell sizes; the PEC box has nz = 6 (aligned pads 2, odd pads 1).
DZ = 1e-3 * np.array([1.1, .8, .6, .8, 1.])


def scene(name):
    """A small graded box; x and y lengths are not whole cells."""
    common = dict(freq_max=10e9, dx=.001, dz_profile=DZ, snap='declared')
    height = float(DZ.sum())
    if name == 'vacuum':
        sim = Simulation(domain=(.0113, .0071, height), boundary=_walls(), cpml_layers=0, **common)
    elif name == 'cpml_port':
        sim = Simulation(domain=(.0103, .0071, height), boundary=BoundarySpec(**dict.fromkeys('xyz', 'cpml')),
                         cpml_layers=4, **common)
        sim.add_material('brick', eps_r=2.2)
        sim.add(Box((.004, .002, .0008), (.008, .005, .0031)), material='brick')
        sim.add_port((.003, .003, .0021), 'ez', impedance=50.)
        sim.add_dft_plane_probe(axis='z', coordinate=.0027, component='ez', freqs=FREQS, name='plane')
        sim.add_flux_monitor(axis='x', coordinate=.009, freqs=FREQS, name='flux')
    elif name == 'pec_pmc_slab':
        sim = Simulation(domain=(.0113, .0071, height), boundary=_walls(x=Boundary('pec', 'pmc')),
                         cpml_layers=0, **common)
        sim.add_material('slab', eps_r=4., sigma=.01, mu_r=1.5,
                         debye_poles=[DebyePole(delta_eps=1.0, tau=1e-11)])
        sim.add(Box((.007, .001, .0008), (.0113, .006, .0031)), material='slab')
    elif name == 'lorentz':
        sim = Simulation(domain=(.0113, .0071, height), boundary=_walls(), cpml_layers=0, **common)
        sim.add_material('disp', eps_r=2., lorentz_poles=[lorentz_pole(2.0, 2 * np.pi * 6e9, 1e9)])
        sim.add(Box((.006, .002, .0008), (.010, .006, .0031)), material='disp')
    elif name == 'mixed':
        sim = Simulation(domain=(.0113, .0071, height), boundary=_walls(), cpml_layers=0, **common)
        sim.add_material('mixed', eps_r=2., debye_poles=[DebyePole(delta_eps=1.0, tau=1e-11)],
                         lorentz_poles=[lorentz_pole(2.0, 2 * np.pi * 6e9, 1e9)])
        sim.add(Box((.006, .002, .0008), (.010, .006, .0031)), material='mixed')
    elif name == 'periodic_x':
        sim = Simulation(domain=(.010, .0071, height), boundary=_walls(x='periodic'), cpml_layers=0, **common)
        sim.add_material('slab', eps_r=3.)
        sim.add(Box((0., .001, .0008), (.010, .003, .0031)), material='slab')
    else:
        raise ValueError(name)
    if name != 'cpml_port':
        sim.add_source((.003, .004, .0021), 'ez', amplitude_kind='current', waveform=PULSE)
    sim.add_probe((.009, .005, .0015), 'ez')
    sim.add_probe((.002, .002, .0029), 'hy')
    return sim


SCENES = ('vacuum', 'cpml_port', 'pec_pmc_slab', 'lorentz', 'periodic_x', 'mixed')

#: Cases measured not bitwise on the CPU (cluster CPU image); judged by the bars at N, 2N, 4N.
#: The seeded-pad equality below holds bitwise on the same scenes.
NOT_BITWISE = {
    ('run', 'vacuum', 'aligned'), ('run', 'vacuum', 'odd'),
    ('run', 'pec_pmc_slab', 'aligned'), ('run', 'pec_pmc_slab', 'odd'),
    ('run', 'lorentz', 'aligned'), ('run', 'lorentz', 'odd'),
    ('run', 'mixed', 'aligned'), ('run', 'mixed', 'odd'),
    ('forward', 'cpml_port', 'aligned'), ('forward', 'cpml_port', 'odd'),
    ('forward', 'lorentz', 'aligned'), ('forward', 'lorentz', 'odd'),
}


def run_kwargs(sim, n_steps, **extra):
    kwargs = dict(n_steps=n_steps, skip_preflight=True)
    kwargs.update(compute_s_params=True, s_param_freqs=FREQS) if lumped(sim) else kwargs.update(
        compute_s_params=False)
    kwargs.update(extra)
    return kwargs


def run_record(name, target, n_steps=N_STEPS, **extra):
    sim = scene(name)
    with vector_pad.force_pad(target), vector_pad.record_plans() as plans:
        out = observed(sim.run(**run_kwargs(sim, n_steps, **extra)))
    return out, plans


def captured_step(name, target, n_steps):
    """The graded scan one run() of ``name`` builds, captured before it runs.

    Returns ``({'invariants', 'context', 'body', 'init', 'xs'}, plans)``.
    """
    from rfx.stepping import graded_loop
    real = graded_loop.recorded_scan
    got = {}

    def scan(body, init, xs=None, **kwargs):
        got.update(body=body, init=init, xs=xs)
        raise _Captured

    sim = scene(name)
    graded_loop.recorded_scan = scan
    try:
        with vector_pad.force_pad(target), vector_pad.record_plans() as plans:
            sim.run(**run_kwargs(sim, n_steps))
    except _Captured:
        pass
    finally:
        graded_loop.recorded_scan = real
    assert set(got) == {'body', 'init', 'xs'}, f'{name}/{target}: captured {sorted(got)}'
    got['invariants'] = got['body'].keywords['invariants']
    got['context'] = got['body'].func.context
    return got, plans


def _driven(invariants, shape):
    """{component: cells a source or port drive writes}, from the step's drive table."""
    driven = {c: np.zeros(shape, bool) for c in FIELDS}
    for c, nodes in invariants['drives'].nodes.items():
        driven[c][tuple(np.asarray(i) for i in nodes)] = True
    assert any(m.any() for m in driven.values()), 'no driven cell'
    return driven


def reference(name):
    """The unpadded step advanced to N, 2N, 4N: carries, per-step family peaks, probe peak."""
    off, _ = captured_step(name, 'off', LENGTHS[-1] + 1)
    shape = off['init']['fdtd'].ex.shape
    driven = _driven(off['invariants'], shape)
    keep = {c: jnp.asarray(~driven[c]) for c in FIELDS}
    body = off['body']

    def tracked(carry, x):
        carry, out = body(carry, x)
        fields = carry['fdtd']
        peaks = jnp.stack([jnp.max(jnp.where(keep[c], jnp.abs(getattr(fields, c)), 0.)) for c in FIELDS])
        return carry, (out, peaks)

    advance = jax.jit(lambda carry, xs: jax.lax.scan(tracked, carry, xs))
    states, probes, peaks, start, carry = {}, [], [], 0, off['init']
    for n in LENGTHS:
        carry, (p, k) = advance(carry, jax.tree_util.tree_map(lambda a: a[start:n], off['xs']))
        states[n], start = carry, n
        probes.append(np.asarray(p))
        peaks.append(np.asarray(k))
    peaks = np.concatenate(peaks)
    probes = np.abs(np.concatenate(probes))
    return dict(off=off, states=states, driven=driven,
                family=lambda n: {'e': float(peaks[:n, :3].max()), 'h': float(peaks[:n, 3:].max())},
                probes=lambda n: float(probes[:n].max()))


def probe_families(name):
    return tuple(p.component[0] for p in scene(name)._probes)


def test_profile_is_graded():
    """Assert realized, not declared: the z cells take at least three sizes."""
    grid = scene('vacuum')._build_nonuniform_grid()
    sizes = np.unique(np.round(np.asarray(grid.dz, np.float64), 12))
    assert len(sizes) >= 3, sizes
    assert grid.shape[2] % 4 and grid.shape[2] % 2 == 0, grid.shape


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('name', SCENES)
def test_graded_run_padded_is_bitwise_or_within_bars(name, target):
    label = f'graded run {name}/{target}'
    if ('run', name, target) not in NOT_BITWISE:
        ref, ref_plans = run_record(name, 'off')
        assert ref_plans and not any(p.active for p in ref_plans)
        got, plans = run_record(name, target)
        assert_padded(plans, label)
        assert_bitwise(ref, got, label)
        jax.clear_caches()
        return
    record = reference(name)
    by_length = {}
    for n in LENGTHS:
        ref, _ = run_record(name, 'off', n)
        got, plans = run_record(name, target, n)
        assert_padded(plans, f'{label} N={n}')
        by_length[n] = bar_ratios(ref, got, peaks=record['family'](n), probes=probe_families(name))
    assert_bars(label, by_length)
    jax.clear_caches()


@pytest.mark.parametrize('target', TARGETS)
def test_graded_until_decay_padded_is_bitwise(target):
    kwargs = dict(until_decay=1e-3, decay_check_interval=20, decay_min_steps=40, decay_max_steps=200)
    ref, _ = run_record('cpml_port', 'off', **kwargs)
    got, plans = run_record('cpml_port', target, **kwargs)
    assert_padded(plans, f'graded until_decay/{target}')
    assert_bitwise(ref, got, f'graded run until_decay/{target}')
    jax.clear_caches()


def _forward_objective(name, n_steps):
    sim = scene(name)
    shape = sim._build_nonuniform_grid().shape

    def objective(eps):
        result = sim.forward(n_steps=n_steps, eps_override=eps, skip_preflight=True,
                             port_s11_freqs=FREQS if lumped(sim) else None)
        value = jnp.sum(result.time_series ** 2)
        if result.s_params is not None:
            value = value + jnp.sum(jnp.abs(result.s_params) ** 2)
        return value, result

    eps = jnp.ones(shape, jnp.float32).at[4:7, 2:5, 1:4].set(2.5)
    return objective, eps


def _forward_records(name, mode, n_steps=N_STEPS):
    objective, eps = _forward_objective(name, n_steps)
    tangent = jnp.zeros_like(eps).at[5, 3, 2].set(1.)
    with vector_pad.force_pad(mode), vector_pad.record_plans() as plans:
        value, result = objective(eps)
        _, dvalue = jax.jvp(lambda e: objective(e)[0], (eps,), (tangent,))
        grad = jax.grad(lambda e: objective(e)[0])(eps)
        batch = jax.vmap(lambda e: objective(e)[0])(jnp.stack([eps, eps * 1.5]))
    out = {'time_series': np.asarray(result.time_series), 'value': np.asarray(value),
           'jvp': np.asarray(dvalue), 'grad': np.asarray(grad), 'vmap': np.asarray(batch)}
    if result.s_params is not None:
        out['s_params'] = np.asarray(result.s_params)
    return out, plans


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('name', ('cpml_port', 'lorentz'))
def test_graded_forward_value_jvp_grad_vmap_padded_bitwise_or_within_bars(name, target):
    label = f'graded forward {name}/{target}'
    lengths = LENGTHS if ('forward', name, target) in NOT_BITWISE else (N_STEPS,)
    by_length = {}
    for n in lengths:
        ref, ref_plans = _forward_records(name, 'off', n)
        got, plans = _forward_records(name, target, n)
        assert not any(p.active for p in ref_plans)
        assert_padded(plans, f'{label} N={n}')
        if lengths == (N_STEPS,):
            assert_bitwise(ref, got, label)
        else:
            by_length[n] = bar_ratios(ref, got, peaks=_forward_family_peaks(name, n),
                                      probes=probe_families(name))
    if by_length:
        assert_bars(label, by_length)
    jax.clear_caches()


def _forward_family_peaks(name, n_steps):
    """Family peaks (driven cells left out) of an unpadded forward() value call of the same model.

    forward() returns no fields: its scan is run here with the per-step peaks
    read beside it. Only the scale is taken from it.
    """
    from rfx.stepping import graded_loop
    objective, eps = _forward_objective(name, n_steps)
    step, _ = captured_step(name, 'off', 1)
    driven = _driven(step['invariants'], step['init']['fdtd'].ex.shape)
    real, captured = graded_loop.recorded_scan, []

    def scan(body, init, xs=None, **kwargs):
        keep = {c: jnp.asarray(~driven[c]) for c in FIELDS}

        def tracked(carry, x):
            carry, out = body(carry, x)
            peaks = jnp.stack([jnp.max(jnp.where(keep[c], jnp.abs(getattr(carry['fdtd'], c)), 0.))
                               for c in FIELDS])
            return carry, (out, peaks)

        final, (outs, peaks) = jax.lax.scan(tracked, init, xs)
        captured.append(np.asarray(peaks))
        return final, outs

    graded_loop.recorded_scan = scan
    try:
        with vector_pad.force_pad('off'):
            objective(eps)
    finally:
        graded_loop.recorded_scan = real
    assert captured, f'{name}: no graded scan in forward()'
    peaks = np.concatenate(captured)
    return {'e': float(peaks[:, :3].max()), 'h': float(peaks[:, 3:].max())}


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('name', ('vacuum', 'cpml_port'))
def test_graded_pad_contents_do_not_reach_physical_cells(name, target):
    """Fill every pad cell of the fields and psi with large random values."""
    ref, _ = run_record(name, target)
    with vector_pad.seed_pad(12345, magnitude=1e30):
        got, plans = run_record(name, target)
    assert_padded(plans, f'graded seeded {name}/{target}')
    assert_bitwise(ref, got, f'graded seeded {name}/{target}')
    jax.clear_caches()


def step_operands(step):
    """Every array the graded step reads besides its carry: invariants and context, by name."""
    out = {}
    for key, value in step['invariants'].items():
        out.update(_arrays(value, key))
    ctx = step['context']
    for field in dataclasses.fields(ctx):
        if field.name != 'pad_plan':
            out.update(_arrays(getattr(ctx, field.name), 'ctx.' + field.name))
    return out


def assert_operands_cropped_bitwise(off, on, label):
    """(iii): each padded operand, cropped to the unpadded extent, has the unpadded bytes."""
    ref, got = step_operands(off), step_operands(on)
    assert ref.keys() == got.keys(), f'{label}: operands {sorted(ref)} vs {sorted(got)}'
    failures, padded = [], 0
    for key, a in ref.items():
        b = got[key]
        if b.dtype != a.dtype or b.ndim != a.ndim or any(m < n for m, n in zip(b.shape, a.shape)):
            failures.append(f'{key}: {a.dtype}{a.shape} vs padded {b.dtype}{b.shape}')
            continue
        padded += b.shape != a.shape
        crop = np.ascontiguousarray(b[tuple(slice(0, n) for n in a.shape)])
        if crop.tobytes() != np.ascontiguousarray(a).tobytes():
            failures.append(f'{key}: {int(np.count_nonzero(crop != a))} of {a.size} values differ '
                            f'on the physical cells')
    assert not failures, f'{label}: realized operands not bitwise\n  ' + '\n  '.join(failures)
    assert padded, f'{label}: no operand was padded'
    return len(ref), padded


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('name', SCENES)
def test_graded_one_step_from_one_state_and_realized_operands(name, target):
    """(i) one padded step from an unpadded state; (iii) the operands it reads, cropped."""
    label = f'graded {name}/{target}'
    record = reference(name)
    off = record['off']
    on, plans = captured_step(name, target, LENGTHS[-1] + 1)
    assert_padded(plans, label)
    plan = plans[-1]
    count, padded = assert_operands_cropped_bitwise(off, on, label)
    assert_bitwise(_arrays(off['xs'], 'xs'), _arrays(on['xs'], 'xs'), f'{label} scan inputs')
    # The pad values of the metrics: H 0, E the last physical value.
    ctx = on['context']
    for axis, (e, h) in enumerate(((on['invariants']['inv_dx'], on['invariants']['inv_dx_h']),
                                   (ctx.inv_dy, ctx.inv_dy_h), (ctx.inv_dz, ctx.inv_dz_h))):
        n = plan.shape[axis]
        assert np.all(np.asarray(h)[n:] == 0), f'{label}: {"xyz"[axis]} H metric pad'
        assert np.all(np.asarray(e)[n:] == np.asarray(e)[n - 1]), f'{label}: {"xyz"[axis]} E metric pad'

    scale = record['family'](LENGTHS[-1])
    probe_scale = record['probes'](LENGTHS[-1])
    step_off, step_on = jax.jit(off['body']), jax.jit(on['body'])
    shapes = [np.shape(leaf) for leaf in jax.tree_util.tree_leaves(on['init'])]
    by_length = {}
    witness = {}
    for n in LENGTHS:
        x = jax.tree_util.tree_map(lambda a: a[n], off['xs'])
        state = vector_pad.pad_carry(record['states'][n], plan)
        assert jax.tree_util.tree_structure(state) == jax.tree_util.tree_structure(on['init'])
        assert [np.shape(leaf) for leaf in jax.tree_util.tree_leaves(state)] == shapes
        ref, ref_out = step_off(record['states'][n], x)
        got, got_out = step_on(state, x)
        ratios = {}
        for f in 'eh':
            worst = max(float(np.max(np.abs(
                np.asarray(getattr(got['fdtd'], c))[plan.physical].astype(np.float64)
                - np.asarray(getattr(ref['fdtd'], c), np.float64)))) for c in (f + 'x', f + 'y', f + 'z'))
            ratios[f.upper()] = _ratio(worst, ONE_STEP_BAR * scale[f])
        probe_diff = np.max(np.abs(np.asarray(got_out, np.float64) - np.asarray(ref_out, np.float64)))
        ratios['probes'] = _ratio(probe_diff, ONE_STEP_BAR * probe_scale)
        by_length[n] = ratios
        # Evidence only: the unpadded step evaluated op by op against its jitted form.
        with jax.disable_jit():
            eager, _ = off['body'](record['states'][n], x)
        witness[n] = {f.upper(): _ratio(max(float(np.max(np.abs(
            np.asarray(getattr(eager['fdtd'], c), np.float64) - np.asarray(getattr(ref['fdtd'], c), np.float64))))
            for c in (f + 'x', f + 'y', f + 'z')), ONE_STEP_BAR * scale[f]) for f in 'eh'}
    sys.stderr.write(
        f'OPERANDS {label}: {count} operand arrays, {padded} padded, all bitwise when cropped; '
        f'plan {plan.shape} -> {plan.padded}; scale E {scale["e"]:.4e} H {scale["h"]:.4e} '
        f'probes {probe_scale:.4e}\n')
    sys.stderr.write(f'WITNESS {label}: unpadded eager vs jit, one step, difference / bar (i) at N = '
                     f'{list(witness)}: ' + '; '.join(
                         f'{k} ' + ', '.join(f'{witness[n][k]:.3e}' for n in witness) for k in ('E', 'H'))
                     + '\n')
    assert_bars(f'graded one step {label}', by_length)
    jax.clear_caches()


def test_distributed_graded_refuses_a_forced_pad():
    """The multi-device graded runner is not padded: a forced pad is refused by name."""
    with vector_pad.force_pad('aligned'), pytest.raises(NotImplementedError, match='vector padding') as info:
        scene('vacuum').run(n_steps=8, skip_preflight=True, compute_s_params=False,
                            devices=jax.devices('cpu')[:2])
    assert 'distributed non-uniform' in str(info.value)
    jax.clear_caches()
