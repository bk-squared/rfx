"""Vector padding (internal switch) leaves every reported number unchanged.

The uniform single-device step may allocate its arrays with padded extents
(``aligned``: z to a multiple of 4; ``odd``: one cell on each even axis). The
physical cells, probes, ports, planes and snapshots must not see the pad: on
the CPU each padded run returns the unpadded run's bytes, or, for the cases
listed in ``NOT_BITWISE``, stays within the stage-move bars. Paths this stage
does not pad refuse a forced pad; ``"auto"`` pads nothing until a measured
gain is entered for the device.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, DebyePole, GaussianPulse, Simulation, SnapshotSpec, lorentz_pole
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.stepping import vector_pad

FIELDS = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')
TARGETS = ('aligned', 'odd')
N_STEPS = 60
FREQS = jnp.array([4e9, 6e9, 8e9])
PULSE = GaussianPulse(f0=6e9, bandwidth=.8)
OBSERVED = ('time_series', 's_params', 'snapshots', 'dft_planes', 'flux_monitors',
            'lumped_port_sparams', 'wire_port_sparams')


def _walls(**faces):
    return BoundarySpec(**{a: faces.get(a, 'pec') for a in 'xyz'})


def scene(name):
    """A small closed or open box; lengths give axes that are not multiples of 4."""
    if name == 'vacuum':
        sim = Simulation(freq_max=10e9, domain=(.0113, .0071, .0052), dx=.001,
                         boundary=_walls(), cpml_layers=0, snap='declared')
    elif name == 'cpml_port':
        sim = Simulation(freq_max=10e9, domain=(.0103, .0071, .0052), dx=.001,
                         boundary=BoundarySpec(**dict.fromkeys('xyz', 'cpml')), cpml_layers=4,
                         snap='declared')
        sim.add_material('brick', eps_r=2.2)
        sim.add(Box((.004, .002, .001), (.008, .005, .004)), material='brick')
        sim.add_port((.003, .003, .002), 'ez', impedance=50.)
        sim.add_dft_plane_probe(axis='z', coordinate=.003, component='ez', freqs=FREQS, name='plane')
        sim.add_flux_monitor(axis='x', coordinate=.009, freqs=FREQS, name='flux')
    elif name == 'pec_pmc_slab':
        sim = Simulation(freq_max=10e9, domain=(.0113, .0071, .0052), dx=.001,
                         boundary=_walls(x=Boundary('pec', 'pmc')), cpml_layers=0, snap='declared')
        sim.add_material('slab', eps_r=4., sigma=.01, mu_r=1.5,
                         debye_poles=[DebyePole(delta_eps=1.0, tau=1e-11)])
        sim.add(Box((.007, .001, .001), (.0113, .006, .004)), material='slab')
    elif name == 'lorentz':
        sim = Simulation(freq_max=10e9, domain=(.0113, .0071, .0052), dx=.001,
                         boundary=_walls(), cpml_layers=0, snap='declared')
        sim.add_material('disp', eps_r=2., lorentz_poles=[lorentz_pole(2.0, 2 * np.pi * 6e9, 1e9)])
        sim.add(Box((.006, .002, .001), (.010, .006, .004)), material='disp')
    elif name == 'periodic_x':
        sim = Simulation(freq_max=10e9, domain=(.010, .0071, .0052), dx=.001,
                         boundary=_walls(x='periodic'), cpml_layers=0, snap='declared')
        sim.add_material('slab', eps_r=3.)
        sim.add(Box((0., .001, .001), (.010, .003, .004)), material='slab')
    elif name == 'kerr':
        sim = Simulation(freq_max=10e9, domain=(.0113, .0071, .0052), dx=.001,
                         boundary=_walls(), cpml_layers=0, snap='declared')
        # chi3 |E|^2 reaches about 1e-2 at the field this source drives (1e8 V/m).
        sim.add_material('kerr', eps_r=2., chi3=1e-18)
        sim.add(Box((.006, .002, .001), (.008, .004, .003)), material='kerr')
    else:
        raise ValueError(name)
    if name != 'cpml_port':
        sim.add_source((.003, .004, .003), 'ez', amplitude_kind='current', waveform=PULSE)
    sim.add_probe((.009, .005, .002), 'ez')
    sim.add_probe((.002, .002, .004), 'hy')
    return sim


SCENES = ('vacuum', 'cpml_port', 'pec_pmc_slab', 'lorentz', 'periodic_x')


def observed(result):
    """Every array a user receives from ``result``, by name."""
    out = {}
    for name in OBSERVED:
        leaves = jax.tree_util.tree_leaves_with_path(getattr(result, name, None))
        for path, leaf in leaves:
            if isinstance(leaf, (np.ndarray, jax.Array)) and np.issubdtype(np.asarray(leaf).dtype, np.number):
                out[name + jax.tree_util.keystr(path)] = np.asarray(leaf)
    state = getattr(result, 'state', None)
    if state is not None:
        for c in FIELDS:
            out['state.' + c] = np.asarray(getattr(state, c))
    return out


def assert_bitwise(ref, got, label):
    assert ref.keys() == got.keys(), f'{label}: outputs {sorted(ref)} vs {sorted(got)}'
    failures = []
    for key in ref:
        a, b = ref[key], got[key]
        assert a.shape == b.shape, f'{label} {key}: shape {a.shape} vs {b.shape}'
        if a.tobytes() == b.tobytes():
            continue
        diff = np.abs(b.astype(np.complex128) - a.astype(np.complex128))
        peak = float(np.max(np.abs(a))) if a.size else 0.
        where = np.unravel_index(int(np.argmax(diff != 0)), a.shape) if a.ndim else ()
        failures.append(
            f'{key}: max|diff|={np.nanmax(diff):.3e}, peak={peak:.3e}, '
            f'ratio={np.nanmax(diff) / max(peak, 1e-300):.3e}; first at {tuple(map(int, where))}: '
            f'{a[where]!r} vs {b[where]!r}')
    if failures:
        pytest.fail(f'{label}: not bitwise\n  ' + '\n  '.join(failures))


def lumped(sim):
    return any(p.impedance for p in sim._ports)


def run_record(sim, target, **extra):
    """run() with a snapshot of every field each 5 steps; returns (outputs, plans)."""
    kwargs = dict(n_steps=N_STEPS, skip_preflight=True,
                  snapshot=SnapshotSpec(interval=5, components=FIELDS))
    kwargs.update(compute_s_params=True, s_param_freqs=FREQS) if lumped(sim) else kwargs.update(
        compute_s_params=False)
    kwargs.update(extra)
    with vector_pad.force_pad(target), vector_pad.record_plans() as plans:
        out = observed(sim.run(**kwargs))
    return out, plans


#: Cases measured not bitwise on the CPU (jax 24.10 image): each first difference is one
#: to three float32 ULP. Estimate: XLA CPU code generation changes with the array extent.
#: The measured witness that the pad does not leak is the seeded-pad equality below
#: (pad contents up to 1e30 give the zero-pad bytes). Every other case stays bitwise.
NOT_BITWISE = {
    ('run', 'vacuum', 'odd'), ('run', 'pec_pmc_slab', 'odd'), ('run', 'lorentz', 'odd'),
    ('forward', 'cpml_port', 'aligned'), ('forward', 'cpml_port', 'odd'),
    ('forward', 'lorentz', 'aligned'), ('forward', 'lorentz', 'odd'),
    ('vmap', 'vacuum', 'odd'), ('one_axis', 'vacuum', (0, 1, 0)),
}


# Bars for the cases above: the committed equivalence law (one last bit times the
# accumulated field, on one fixed scale over the record and the field family; never a
# component's or a probe's own peak), as the design's section 10 states it.
#: Per-step quantities (fields at a step, snapshots, probe series): at every recorded
#: step n, |padded - unpadded| at most this times S(n), the running sum to n of the
#: unpadded run's largest |field| over every physical cell and every component of the
#: family (E: ex, ey, ez; H: hx, hy, hz) the quantity belongs to.
STEP_BAR = 1.2e-7
#: Record quantities (DFT planes, S11, flux, the objective value, jvp, grad): max
#: |difference| at most this of the quantity's own largest magnitude.
RECORD_BAR = 1e-4
#: Both hold at N, 2N and 4N steps.
LENGTHS = (N_STEPS, 2 * N_STEPS, 4 * N_STEPS)


def _ratio(diff, bar):
    """max of difference / bar (0 where the difference is 0; inf where only the bar is)."""
    diff, bar = np.broadcast_arrays(np.asarray(diff, np.float64), np.asarray(bar, np.float64))
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(diff == 0, 0., diff / bar)
    return float(np.max(ratio)) if ratio.size else 0.


def family_sums(snapshots):
    """S(n) per family from per-step snapshots of all six components of the unpadded run."""
    def peak(c):
        record = np.asarray(snapshots[c])
        return np.abs(record).reshape(len(record), -1).max(axis=1).astype(np.float64)
    return {f: np.cumsum(np.max([peak(f + a) for a in 'xyz'], axis=0)) for f in 'eh'}


def probe_families(name):
    """The family of each probe column, in the order the scene adds its probes."""
    return tuple(p.component[0] for p in scene(name)._probes)


def bar_ratios(ref, got, *, sums=None, probes=()):
    """difference / bar for every output (> 1 fails)."""
    assert ref.keys() == got.keys(), f'outputs {sorted(ref)} vs {sorted(got)}'
    ratios = {}
    for key, a in ref.items():
        b = got[key]
        assert a.shape == b.shape, f'{key}: shape {a.shape} vs {b.shape}'
        if not a.size:
            continue
        diff = np.abs(b.astype(np.complex128) - a.astype(np.complex128))
        if key.startswith('snapshots'):
            s = sums[key[len("snapshots['")]]
            assert len(a) == len(s), f'{key}: {len(a)} frames, {len(s)} steps of S'
            ratios[key] = _ratio(diff.reshape(len(a), -1).max(axis=1), STEP_BAR * s)
        elif key.startswith('state.'):
            ratios[key] = _ratio(diff.max(), STEP_BAR * sums[key[len('state.')]][-1])
        elif key == 'time_series':
            assert a.shape[1] == len(probes), f'time_series columns {a.shape[1]}, probes {probes}'
            for column, f in enumerate(probes):
                assert len(a) == len(sums[f]), f'time_series: {len(a)} rows, {len(sums[f])} steps of S'
                ratios[f'time_series[{column}] ({f})'] = _ratio(diff[:, column], STEP_BAR * sums[f])
        else:
            ratios[key] = _ratio(diff.max(), RECORD_BAR * np.abs(a).max())
    return ratios


def assert_bars(label, by_length):
    """Every ratio at every record length at most 1; per output, the ratios at N, 2N, 4N."""
    lengths = list(by_length)
    keys = sorted(set().union(*by_length.values()))
    rows = [f'{k}: ' + ', '.join(f'{by_length[n].get(k, float("nan")):.3e}' for n in lengths)
            for k in keys]
    worst = max((max(r.values()) for r in by_length.values() if r), default=0.)
    text = f'{label}: difference / bar at N = {lengths}\n  ' + '\n  '.join(rows)
    sys.stderr.write('BAR ' + text.replace('\n', '\nBAR ') + '\n')
    assert worst <= 1., text


def run_lengths(name, target, label):
    """Bars at N, 2N, 4N for run() on ``name``, with every field snapshotted each step."""
    by_length = {}
    for n in LENGTHS:
        snap = SnapshotSpec(interval=1, components=FIELDS)
        ref, _ = run_record(scene(name), 'off', n_steps=n, snapshot=snap)
        got, plans = run_record(scene(name), target, n_steps=n, snapshot=snap)
        assert_padded(plans, f'{label} N={n}')
        sums = family_sums({c: ref[f"snapshots['{c}']"] for c in FIELDS})
        by_length[n] = bar_ratios(ref, got, sums=sums, probes=probe_families(name))
    assert_bars(label, by_length)



def assert_padded(plans, label):
    assert plans, f'{label}: no uniform run recorded a plan'
    for plan in plans:
        assert plan.active and plan.padded != plan.shape, f'{label}: plan {plan} pads nothing'


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('name', SCENES)
def test_run_padded_is_bitwise_or_within_bars(name, target):
    if ('run', name, target) in NOT_BITWISE:
        run_lengths(name, target, f'run {name}/{target}')
        jax.clear_caches()
        return
    ref, ref_plans = run_record(scene(name), 'off')
    assert ref_plans and not any(p.active for p in ref_plans)
    got, plans = run_record(scene(name), target)
    assert_padded(plans, f'{name}/{target}')
    assert_bitwise(ref, got, f'run {name}/{target}')
    jax.clear_caches()


@pytest.mark.parametrize('target', TARGETS)
def test_run_until_decay_padded_is_bitwise(target):
    kwargs = dict(until_decay=1e-3, decay_check_interval=20, decay_min_steps=40, decay_max_steps=200)
    ref, _ = run_record(scene('vacuum'), 'off', **kwargs)
    got, plans = run_record(scene('vacuum'), target, **kwargs)
    assert_padded(plans, f'until_decay/{target}')
    assert_bitwise(ref, got, f'run until_decay/{target}')
    jax.clear_caches()


def _forward_objective(name, n_steps=N_STEPS, **kw):
    sim = scene(name)
    shape = sim._build_realized_grid().shape

    def objective(eps):
        result = sim.forward(n_steps=n_steps, eps_override=eps, skip_preflight=True,
                             port_s11_freqs=FREQS if lumped(sim) else None, **kw)
        value = jnp.sum(result.time_series ** 2)
        if result.s_params is not None:
            value = value + jnp.sum(jnp.abs(result.s_params) ** 2)
        return value, result

    eps = jnp.ones(shape, jnp.float32).at[4:7, 2:5, 1:4].set(2.5)
    return objective, eps


def _forward_outputs(result):
    out = {'time_series': np.asarray(result.time_series)}
    if result.s_params is not None:
        out['s_params'] = np.asarray(result.s_params)
    return out


def _forward_records(name, mode, n_steps):
    objective, eps = _forward_objective(name, n_steps)
    tangent = jnp.zeros_like(eps).at[5, 3, 2].set(1.)
    with vector_pad.force_pad(mode), vector_pad.record_plans() as plans:
        (value, result) = objective(eps)
        _, dvalue = jax.jvp(lambda e: objective(e)[0], (eps,), (tangent,))
        grad = jax.grad(lambda e: objective(e)[0])(eps)
    return dict(_forward_outputs(result), value=np.asarray(value),
                jvp=np.asarray(dvalue), grad=np.asarray(grad)), plans


def _forward_family_sums(name, n_steps):
    """S(n) of an unpadded forward() of the same model, from per-step snapshots.

    forward() returns no fields, so a separate unpadded forward() value call is
    run with every field snapshotted each step (the time step reads
    ``rfx.simulation.run`` at call time). Only the scale is taken from it.
    """
    import rfx.simulation as simulation
    objective, eps = _forward_objective(name, n_steps)
    real, captured = simulation.run, []

    def run(*args, **kwargs):
        kwargs['snapshot'] = SnapshotSpec(interval=1, components=FIELDS)
        result = real(*args, **kwargs)
        captured.append(result.snapshots)
        return result

    simulation.run = run
    try:
        with vector_pad.force_pad('off'):
            objective(eps)
    finally:
        simulation.run = real
    assert len(captured) == 1, f'{name}: {len(captured)} uniform runs in one forward()'
    return family_sums(captured[0])


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('name', ('cpml_port', 'lorentz'))
def test_forward_value_jvp_grad_padded_bitwise_or_within_bars(name, target):
    label = f'forward {name}/{target}'
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
            by_length[n] = bar_ratios(ref, got, sums=_forward_family_sums(name, n),
                                      probes=probe_families(name))
    if by_length:
        assert_bars(label, by_length)
    jax.clear_caches()


@pytest.mark.parametrize('target', TARGETS)
def test_vmap_over_forward_padded_bitwise_or_within_bars(target):
    label = f'vmap forward vacuum/{target}'
    lengths = LENGTHS if ('vmap', 'vacuum', target) in NOT_BITWISE else (N_STEPS,)
    by_length = {}
    for n in lengths:
        objective, eps = _forward_objective('vacuum', n)
        batch = jnp.stack([eps, eps * 1.5])
        outs = {}
        for mode in ('off', target):
            with vector_pad.force_pad(mode), vector_pad.record_plans() as plans:
                outs[mode] = ({'value': np.asarray(jax.vmap(lambda e: objective(e)[0])(batch))}, plans)
        assert_padded(outs[target][1], f'{label} N={n}')
        if lengths == (N_STEPS,):
            assert_bitwise(outs['off'][0], outs[target][0], label)
        else:
            by_length[n] = bar_ratios(outs['off'][0], outs[target][0])
    if by_length:
        assert_bars(label, by_length)
    jax.clear_caches()


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('name', ('vacuum', 'cpml_port'))
def test_pad_contents_do_not_reach_physical_cells(name, target):
    """Fill every pad cell of the fields and psi with large random values."""
    # Kerr is refused under a forced pad. Were it padded, its seed would stay at 1e10:
    # Kerr squares the field in every cell, (1e30)^2 overflows float32, and the pad's
    # 0 * inf would be NaN by arithmetic, not by a leak.
    ref, _ = run_record(scene(name), target)
    with vector_pad.seed_pad(12345, magnitude=1e30):
        got, plans = run_record(scene(name), target)
    assert_padded(plans, f'seeded {name}/{target}')
    assert_bitwise(ref, got, f'seeded {name}/{target}')
    jax.clear_caches()


def test_plan_targets():
    plan = vector_pad.plan_pad((20, 17, 15), target='aligned', periodic_axes=(False,) * 3)
    assert plan.padded == (20, 17, 16)
    plan = vector_pad.plan_pad((20, 17, 14), target='odd', periodic_axes=(False,) * 3)
    assert plan.padded == (21, 17, 15)
    plan = vector_pad.plan_pad((10, 8, 6), target='odd', periodic_axes='x')
    assert plan.padded == (10, 9, 7)
    plan = vector_pad.plan_pad((10, 8, 1), target='aligned', periodic_axes='', two_d=True)
    assert not plan.active


def test_auto_pads_nothing_until_measured():
    assert vector_pad._MEASURED_GAIN == {}
    assert vector_pad.requested_target() == 'auto'
    with vector_pad.record_plans() as plans:
        scene('vacuum').run(n_steps=4, skip_preflight=True, compute_s_params=False)
    assert plans and all(p.pad == (0, 0, 0) and p.target == 'off' for p in plans)


def _graded():
    sim = Simulation(freq_max=10e9, domain=(.008, .0071, .0052), dx=.001, boundary=_walls(),
                     cpml_layers=0, dx_profile=np.full(8, .001), snap='declared')
    sim.add_source((.003, .004, .003), 'ez', amplitude_kind='current', waveform=PULSE)
    sim.add_probe((.005, .004, .003), 'ez')
    return sim


def _subgridded():
    sim = Simulation(freq_max=10e9, domain=(.012, .012, .012), dx=.001, boundary=_walls(),
                     cpml_layers=0, snap='declared')
    sim.add_refinement(z_range=(.004, .008), ratio=2, validation='research')
    sim.add_source((.006, .006, .006), 'ez', amplitude_kind='current', waveform=PULSE)
    sim.add_probe((.005, .006, .006), 'ez')
    return sim


def _fourth_order():
    sim = Simulation(freq_max=10e9, domain=(.0113, .0071, .0052), dx=.001, boundary=_walls(),
                     cpml_layers=0, stencil_order=4, snap='declared')
    sim.add_source((.003, .004, .003), 'ez', amplitude_kind='current', waveform=PULSE)
    sim.add_probe((.005, .004, .003), 'ez')
    return sim


def _adjoint():
    from tests.unit.autodiff.test_reciprocity_adjoint import BOX_HI, BOX_LO, fixture
    sim, eps = fixture('float32')
    sim.forward(design_box=(BOX_LO, BOX_HI), design_eps_override=eps, n_steps=8,
                gradient='adjoint', skip_preflight=True)


REFUSALS = {
    'graded': (lambda: _graded().run(n_steps=8, skip_preflight=True, compute_s_params=False),
               'graded'),
    'subgridded': (lambda: _subgridded().run(n_steps=8, skip_preflight=True, compute_s_params=False),
                   'subgridded'),
    'distributed': (lambda: scene('vacuum').run(n_steps=8, skip_preflight=True, compute_s_params=False,
                                                devices=jax.devices('cpu')[:2]),
                    'distributed'),
    'adjoint': (_adjoint, 'adjoint scan'),
    # Padded, Kerr was not bitwise on the CPU for either target (stage 1b ruling: refuse).
    'kerr': (lambda: scene('kerr').run(n_steps=8, skip_preflight=True, compute_s_params=False),
             'Kerr nonlinearity'),
    'stencil_order_4': (lambda: _fourth_order().run(n_steps=8, skip_preflight=True, compute_s_params=False),
                        'stencil_order=4'),
}


@pytest.mark.parametrize('path', sorted(REFUSALS))
def test_unpadded_paths_refuse_a_forced_pad(path):
    call, words = REFUSALS[path]
    with vector_pad.force_pad('aligned'), pytest.raises(NotImplementedError, match='vector padding') as info:
        call()
    assert words in str(info.value)
    jax.clear_caches()


def test_unknown_operands_have_no_pad_rule():
    plan = vector_pad.plan_pad((4, 4, 5), target='aligned', periodic_axes=(False,) * 3)
    with pytest.raises(ValueError, match='no pad rule'):
        vector_pad.pad_invariants({'something_new': jnp.zeros((4, 4, 5))}, plan)
    with pytest.raises(ValueError, match='no pad rule'):
        vector_pad.pad_carry({'fdtd_shadow': None}, plan)
    with pytest.raises(ValueError, match='not per-cell'):
        vector_pad.pad_cells(jnp.zeros((4, 4, 6)), plan, 0.)


@pytest.mark.parametrize('pad', ((1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (0, 0, 2)))
def test_one_axis_pad_is_bitwise_or_within_bars(pad, monkeypatch):
    """Each axis's pad alone, by one and two cells (the targets combine these)."""
    real = vector_pad.plan_pad

    def plan(shape, *, target, periodic_axes, two_d=False):
        base = real(shape, target='off', periodic_axes=periodic_axes, two_d=two_d)
        if target == 'off':
            return base
        return vector_pad.PadPlan(base.shape, tuple(n + p for n, p in zip(base.shape, pad)), pad, target)

    monkeypatch.setattr(vector_pad, 'plan_pad', plan)
    if ('one_axis', 'vacuum', pad) in NOT_BITWISE:
        run_lengths('vacuum', 'aligned', f'run vacuum pad {pad}')
        jax.clear_caches()
        return
    ref, _ = run_record(scene('vacuum'), 'off')
    got, plans = run_record(scene('vacuum'), 'aligned')
    assert_padded(plans, f'pad {pad}')
    assert_bitwise(ref, got, f'run vacuum pad {pad}')
    jax.clear_caches()
