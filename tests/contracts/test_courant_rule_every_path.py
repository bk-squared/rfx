"""Final material values meet the actual kernel step on every stepping lane.

Parameterized refusal pattern copied from test_admission_refusal_order.py;
public two-device entries follow test_distributed_admission_refusals.py.
Expected limits use grid-reported lengths and the stencil formula, never the
admission implementation's helpers. Growth is an observation, not a predictor.
"""
from __future__ import annotations

import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, DebyePole, Simulation
from rfx.core.yee import MaterialArrays
from rfx.sources.sources import CustomWaveform

C = 299792458.0
H = .001
LO, HI = (.002,) * 3, (.006,) * 3


def scene(path='uniform', eps=.5, mu=1., pole=None, *, impulse=False, dt=None):
    options = dict(dz_profile=np.full(8, H)) if path in ('graded', 'distributed_graded', 'legacy') else {}
    if dt is not None:
        options['dt'] = dt
    if path == 'order4':
        options['stencil_order'] = 4
    sim = Simulation(freq_max=10e9, domain=(.008,) * 3, dx=H, boundary='pec', **options)
    poles = {}
    if pole == 'debye':
        poles['debye_poles'] = [DebyePole(1., 1e-11)]
    if pole == 'lorentz':
        from rfx.materials.lorentz import LorentzPole
        poles['lorentz_poles'] = [LorentzPole(2e10, 1e9, 1e20)]
    sim.add_material('body', eps_r=eps, mu_r=mu, **poles)
    sim.add(Box(LO, HI), material='body')
    wave = CustomWaveform(lambda t: jnp.where(t == 0., 1., 0.)) if impulse else None
    sim.add_source((.003, .004, .004), 'ez', **({'waveform': wave} if wave else {}), amplitude_kind='field')
    sim.add_probe((.005, .004, .004), 'ez')
    return sim


def grid_of(sim, path):
    return sim._build_nonuniform_grid() if path in ('graded', 'distributed_graded', 'legacy') else sim._build_grid()


def invoke(sim, path, entry='run', *, n=2, **overrides):
    options = dict(n_steps=n, skip_preflight=True)
    if entry == 'run':
        options['compute_s_params'] = False
    else:
        options.update(checkpoint=False, emit_time_series=True)
    if path.startswith('distributed') or path == 'legacy':
        devices = jax.devices('cpu')
        if len(devices) < 2:
            pytest.skip('requires two virtual CPU devices')
        options['devices'] = devices[:2]
        if entry == 'forward':
            options['distributed'] = True
    return getattr(sim, entry)(**options, **overrides)


def assert_cell_message(message, grid, eps, mu=1., *, order=2, lo=LO, hi=HI,
                        slab=False, conservative=False, step=None):
    assert 'cell Courant limit' in message
    step = float(grid.dt) * (.857 if order == 4 else 1.) if step is None else step
    stable = np.sqrt(float(np.float32(eps)) * float(np.float32(mu)) /
                     (C**2 * sum(1 / np.min(grid.cells(a))**2 for a in range(3)) *
                      ((9/8 + 1/24)**2 if order == 4 else 1)))
    assert float(re.search(r'dt = ([^ ]+) s', message)[1]) == pytest.approx(step, rel=1e-7)
    assert float(re.search(r'largest stable step = ([^ ]+) s', message)[1]) == pytest.approx(stable, rel=2e-7)
    count = 1
    for axis, label in enumerate('xyz'):
        widths = np.asarray(grid.cells(axis))
        nodes = np.array([grid.node_of(axis, i) for i in range(len(widths))])
        indices = np.flatnonzero((nodes >= lo[axis] - 1e-12) & (nodes < hi[axis] - 1e-12))
        if slab and axis == 0:
            indices = indices[indices < (grid.shape[0] + 1)//2]
        count *= len(indices)
        bounds = re.search(rf'{label}=\[([^,]+), ([^\]]+)\] m', message)
        assert bounds is not None
        np.testing.assert_allclose([float(bounds[1]), float(bounds[2])],
                                   [nodes[indices[0]], nodes[indices[-1]] + widths[indices[-1]]],
                                   rtol=1e-7, atol=1e-12)
    assert f'{count} cells;' in message
    assert 'or raise eps_r / eps_inf / mu_r to 1' in message
    if conservative:
        assert 'conservative on this path' in message
        assert 'raise eps_r or mu_r to 1' in message
    elif type(grid).__name__ == 'NonUniformGrid':
        assert 'pass dt=' in message
    else:
        assert 'declare a dx_profile/dy_profile/dz_profile' in message


@pytest.mark.parametrize('path', ['uniform', 'graded', 'distributed_uniform', 'distributed_graded', 'legacy'])
@pytest.mark.parametrize('eps,mu,pole', [(.5, 1., None), (.5, 1., 'debye'), (.1, 1., 'lorentz'), (1., .5, None)],
                         ids=['plain', 'debye', 'lorentz', 'mu'])
def test_declared_cells_refused(path, eps, mu, pole, monkeypatch):
    if path == 'legacy':
        from rfx.stepping import slab_wrap
        monkeypatch.setattr(slab_wrap, 'select_step', lambda **kw: 'legacy')
    messages = []
    for entry in ('run', 'forward'):
        sim = scene(path, eps, mu, pole)
        with pytest.raises(ValueError, match='cell Courant limit') as error:
            invoke(sim, path, entry)
        message = str(error.value)
        assert_cell_message(message, grid_of(sim, path), eps, mu,
                            slab=path.startswith('distributed') or path == 'legacy')
        messages.append(message)
    assert messages[0] == messages[1]


@pytest.mark.parametrize('path,eps', [('uniform', .980), ('order4', .975)])
def test_boundary_refused(path, eps):
    messages = []
    for entry in ('run', 'forward'):
        sim = scene(path, eps)
        with pytest.raises(ValueError, match='cell Courant limit') as error:
            invoke(sim, path, entry)
        assert_cell_message(str(error.value), grid_of(sim, path), eps, order=4 if path == 'order4' else 2)
        messages.append(str(error.value))
    assert messages[0] == messages[1]


@pytest.mark.parametrize('path,kind', [('uniform', 'eps'), ('graded', 'eps'), ('uniform', 'mu'),
                                      ('uniform', 'box'), ('graded', 'box')])
def test_forward_override_refused(path, kind):
    sim = scene(path, 2.)
    grid = grid_of(sim, path)
    override = jnp.full(grid.shape, 1.)
    region = tuple(slice(grid.index_of(a, LO[a]), grid.index_of(a, HI[a])) for a in range(3))
    override = override.at[region].set(.5)
    if kind == 'box':
        # Design boxes cannot contain a source: move this source outside the box.
        sim._ports.clear()
        sim.add_source((.001, .001, .001), 'ez')
        # Design-box coordinates include both endpoint nodes.
        kw = dict(design_box=(LO, tuple(v - H for v in HI)), design_eps_override=override[region])
    else:
        kw = {('eps_override' if kind == 'eps' else 'mu_r_override'): override}
        if kind == 'mu':
            kw['eps_override'] = jnp.ones(grid.shape)
    with pytest.raises(ValueError, match='cell Courant limit') as error:
        invoke(sim, path, 'forward', **kw)
    assert_cell_message(str(error.value), grid, .5)


def test_raw_material_arrays_refused():
    import rfx
    grid = grid_of(scene(), 'uniform')
    materials = MaterialArrays(jnp.full(grid.shape, .5), jnp.zeros(grid.shape), jnp.ones(grid.shape))
    with pytest.raises(ValueError, match='cell Courant limit') as error:
        rfx.run(grid, materials, n_steps=1, boundary='pec')
    assert_cell_message(str(error.value), grid, .5, lo=(0.,)*3, hi=(.009,)*3)


def test_vmap_fast_members_refused():
    from rfx.vmap_sweep import vmap_material_sweep
    sim = scene(eps=2.)
    with pytest.raises(ValueError, match='cell Courant limit') as error:
        vmap_material_sweep(sim, 'body.eps_r', [2., .5], n_steps=2)
    assert_cell_message(str(error.value), sim._build_grid(), .5)


@pytest.mark.parametrize('transform', ['grad', 'jvp', 'vmap', 'jit'])
def test_single_device_traced_refusal(transform):
    def loss(eps):
        return jnp.sum(invoke(scene(eps=eps), 'uniform', 'forward').time_series)
    with pytest.raises(Exception, match='cell Courant limit'):
        if transform == 'grad':
            jax.grad(loss)(jnp.float32(.5))
        elif transform == 'jvp':
            jax.jvp(loss, (jnp.float32(.5),), (jnp.float32(1.),))
        elif transform == 'vmap':
            jax.vmap(loss)(jnp.array([2., .5]))
        else:
            np.asarray(jax.jit(loss)(jnp.float32(.5)))


@pytest.mark.parametrize('path', ['uniform', 'graded'])
def test_cross_trace_control(path):
    def run(eps):
        return invoke(scene(path, eps, impulse=True), path, 'forward', n=24).time_series
    concrete = np.asarray(run(jnp.float32(2.)))
    compiled = np.asarray(jax.jit(run)(jnp.float32(2.)))
    primal, _ = jax.jvp(run, (jnp.float32(2.),), (jnp.float32(1.),))
    peak = np.max(np.abs(concrete), axis=1, keepdims=True)
    for actual in (compiled, np.asarray(primal)):
        assert np.all(np.abs(actual-concrete) <= 9 * np.spacing(peak))
    def loss(eps):
        value = run(eps)
        return jnp.sum(value), value
    _, value = jax.grad(loss, has_aux=True)(jnp.float32(2.))
    assert np.max(np.abs(np.asarray(value)-concrete)) <= 1e-4 * np.max(np.abs(concrete))


@pytest.mark.parametrize('path', ['distributed_uniform', 'distributed_graded'])
def test_multidevice_jit_grad_control(path):
    def loss(eps):
        return jnp.sum(invoke(scene(path, eps), path, 'forward').time_series)
    assert np.isfinite(np.asarray(jax.jit(jax.grad(loss))(jnp.float32(2.))))


@pytest.mark.parametrize('path', ['distributed_uniform', 'distributed_graded'])
@pytest.mark.xfail(strict=True, reason='#2755: a jit-traced permittivity on several devices is not judged')
def test_multidevice_traced_gap(path):
    def loss(eps):
        return jnp.sum(invoke(scene(path, eps), path, 'forward').time_series)
    with pytest.raises(Exception, match='cell Courant limit'):
        np.asarray(jax.jit(loss)(jnp.float32(.5)))


def test_drude_mu_changes_pole_limit():
    from rfx.materials.lorentz import LorentzPole
    sim = scene(eps=2., mu=.6)
    grid = sim._build_grid()
    step = float(grid.dt)
    b = C**2 * sum(1 / np.min(grid.cells(a))**2 for a in range(3))
    wp2 = 3 / step**2
    assert (wp2 + 4*b) * step**2 <= 8
    assert (wp2 + 4*b/.6) * step**2 > 8
    # A fixed Drude pole from these numbers, not a fitted or predicted pole.
    sim.add_material('body', eps_r=2., mu_r=.6, lorentz_poles=[LorentzPole(0., 0., wp2)])
    with pytest.raises(ValueError, match='Lorentz/Drude poles') as error:
        invoke(sim, 'uniform')
    stable = np.sqrt(8 / (wp2 + 4*b/float(np.float32(.6))))
    measured = float(re.search(r'largest stable step for this set is ([^ ]+)', str(error.value))[1])
    assert measured == pytest.approx(stable, rel=2e-6)


def _observe_peaks(monkeypatch):
    """Observe full E after each real scan step without changing its arithmetic."""
    from rfx.core.yee import FDTDState
    peaks: dict[int, float] = {}
    scan = jax.lax.scan
    def save(step, peak):
        peaks[int(step)] = max(peaks.get(int(step), 0.), float(peak))
    def watched(body, init, *args, **kwargs):
        def observe(carry, item):
            updated, result = body(carry, item)
            states = jax.tree.leaves(updated, is_leaf=lambda a: isinstance(a, FDTDState))
            for state in states:
                if isinstance(state, FDTDState):
                    peak = jnp.maximum(jnp.maximum(jnp.max(jnp.abs(state.ex)), jnp.max(jnp.abs(state.ey))), jnp.max(jnp.abs(state.ez)))
                    jax.debug.callback(save, state.step, peak)
            if isinstance(updated, dict) and 'c' in updated and 'f' in updated:
                electric = (*updated['c'][:3], *updated['f'][:3])
                peak = jnp.max(jnp.stack([jnp.max(jnp.abs(a)) for a in electric]))
                jax.debug.callback(save, item[0] + 1, peak)
            return updated, result
        return scan(observe, init, *args, **kwargs)
    monkeypatch.setattr(jax.lax, 'scan', watched)
    return peaks


@pytest.mark.parametrize('case', ['uniform_0982', 'graded_pinned', 'graded_large', 'order4_0985', 'subgrid_coarse_1'])
def test_admitted_run_bar(case, monkeypatch):
    path = 'graded' if case.startswith('graded') else 'order4' if case.startswith('order4') else 'uniform'
    eps = .982 if case == 'uniform_0982' else .985 if case == 'order4_0985' else .5 if path == 'graded' else 1.
    pin = .6 * float(scene()._build_grid().dt) / .99 if case == 'graded_pinned' else None
    sim = scene(path, eps, impulse=True, dt=pin)
    if case == 'graded_large':
        profile = np.array([.001, .0014] + [.002]*8 + [.0014, .001])
        length = float(np.sum(profile))
        sim = Simulation(freq_max=10e9, domain=(length,)*3, dx=H, boundary='pec',
                         dx_profile=profile, dy_profile=profile, dz_profile=profile)
        sim.add_material('body', eps_r=.5)
        sim.add(Box((.0084,)*3, (.0124,)*3), material='body')
        sim.add_source((.0104,)*3, 'ez', waveform=CustomWaveform(lambda t: jnp.where(t == 0., 1., 0.)), amplitude_kind='field')
    if case == 'subgrid_coarse_1':
        sim.add_refinement((.003, .006), ratio=2, validation='off')
    peaks = _observe_peaks(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        invoke(sim, path, n=400)
    _assert_run_bar(peaks, case)


def _assert_run_bar(peaks, case):
    jax.effects_barrier()
    assert len(peaks) >= 400, f'not every step observed: {len(peaks)}'
    values = np.array([peaks[i] for i in range(201, 401)])
    assert np.isfinite(values).all() and np.min(values) > 0
    blocks = values.reshape(20, 10).max(axis=1)
    growth = np.exp(np.polyfit(np.arange(205, 405, 10), np.log(blocks), 1)[0])
    print(f'RUN_BAR {case} growth={growth:.12g}')
    assert growth <= 1.001


@pytest.mark.parametrize('location,eps', [('fine', .5), ('coarse', .5)])
def test_subgrid_refusal(location, eps, monkeypatch):
    from rfx.model import source_coefficients
    sim = scene(eps=1.)
    sim.add_material('body', eps_r=1.)
    sim.add_material('low', eps_r=eps)
    lo = (.002, .002, .004) if location == 'fine' else (.002, .002, .001)
    hi = (.006, .006, .005) if location == 'fine' else (.006, .006, .002)
    sim.add(Box(lo, hi), material='low')
    sim.add_refinement((.003, .006), ratio=2, validation='off')
    captured: list = []
    original = source_coefficients.subgrid_source_tables
    def capture(requests, grid, materials, *args, **kwargs):
        captured.append((grid, materials))
        return original(requests, grid, materials, *args, **kwargs)
    monkeypatch.setattr(source_coefficients, 'subgrid_source_tables', capture)
    with pytest.raises(ValueError, match='cell Courant limit') as error:
        invoke(sim, 'uniform')
    message = str(error.value)
    fine_grid = captured[0][0]
    stable = np.sqrt(float(np.float32(eps)) /
                     (C**2 * sum(1 / np.min(fine_grid.cells(a))**2 for a in range(3))))
    assert float(re.search(r'dt = ([^ ]+) s', message)[1]) == pytest.approx(fine_grid.dt)
    assert float(re.search(r'largest stable step = ([^ ]+) s', message)[1]) == pytest.approx(stable, rel=2e-7)
    for a, label in enumerate('xyz'):
        match = re.search(rf'{label}=\[([^,]+), ([^\]]+)\] m', message)
        np.testing.assert_allclose([float(match[1]), float(match[2])], [lo[a], hi[a]], atol=1e-10)
    assert 'conservative on this path' in message
    assert 'raise eps_r or mu_r to 1' in message


def test_conformal_floor_refusal():
    from rfx.grid import Grid
    sim = scene(eps=.99, dt=.5 * Grid.courant_dt(H))
    sim.add(Box((0.,)*3, (.0018, .008, .008)), material='pec')
    with pytest.raises(ValueError, match='cell Courant limit') as error:
        invoke(sim, 'uniform', conformal_pec='cut_cell_floor')
    assert_cell_message(str(error.value), sim._build_grid(), .99, conservative=True)
    with pytest.raises(NotImplementedError, match='forward.*refuses conformal_pec'):
        invoke(sim, 'uniform', 'forward', conformal_pec='cut_cell_floor')


@pytest.mark.parametrize('path', ['graded', 'distributed_uniform', 'distributed_graded'])
def test_mu_override_unreachable(path):
    sim = scene(path, eps=1.)
    with pytest.raises(NotImplementedError, match='mu_r_override'):
        invoke(sim, path, 'forward', mu_r_override=jnp.full(grid_of(sim, path).shape, .5))


@pytest.mark.parametrize('path', ['order4', 'subgrid'])
@pytest.mark.parametrize('pole', ['debye', 'lorentz'])
def test_dispersion_unreachable(path, pole):
    sim = scene('order4' if path == 'order4' else 'uniform', eps=2., pole=pole)
    if path == 'subgrid':
        sim.add_refinement((.003, .006), ratio=2, validation='off')
    with pytest.raises((ValueError, NotImplementedError), match='[Dd]ispers|[Dd]ebye|[Ll]orentz'):
        invoke(sim, 'uniform')


@pytest.mark.parametrize('compiled', [False, True], ids=['eager-members', 'staged-batch'])
def test_vmap_scan_check_scope(compiled):
    from rfx.stepping.vmap_scan import _build_vmap_scan_fn
    grid = scene()._build_grid()
    one = _build_vmap_scan_fn(grid, 1)
    def run(eps):
        mats = MaterialArrays(jnp.full(grid.shape, eps), jnp.zeros(grid.shape), jnp.ones(grid.shape))
        return one(mats, jnp.zeros((1, 0)))
    batch = jax.vmap(run)
    if compiled:
        batch = jax.jit(batch)
    with pytest.raises(Exception, match='cell Courant limit') as error:
        jax.device_get(batch(jnp.array([2., .5])))
    assert_cell_message(str(error.value), grid, .5, lo=(0.,)*3, hi=(.009,)*3)
    print(f'VMAP_SCOPE {"staged batch returns no result" if compiled else "eager failing member refused on host"}')


@pytest.mark.parametrize('wire', [False, True], ids=['lumped', 'wire'])
def test_eager_probe_builder_refused(wire):
    from rfx.core.yee import init_materials
    from rfx.probes.probes import extract_s_matrix, extract_s_matrix_wire
    from rfx.sources.sources import GaussianPulse, LumpedPort, WirePort
    grid = scene()._build_grid()
    materials = init_materials(grid.shape)._replace(eps_r=jnp.full(grid.shape, .5))
    waveform = GaussianPulse(f0=5.5e9, bandwidth=.8)
    port = (WirePort(start=(.004,)*3, end=(.004, .004, .005), component='ez',
                     impedance=50., excitation=waveform) if wire else
            LumpedPort(position=(.004,)*3, component='ez', impedance=50., excitation=waveform))
    extract = extract_s_matrix_wire if wire else extract_s_matrix
    with pytest.raises(ValueError, match='cell Courant limit') as error:
        extract(grid, materials, [port], jnp.array([5e9]), 2, boundary='pec')
    assert_cell_message(str(error.value), grid, .5, lo=(0.,)*3, hi=(.009,)*3)


def test_public_sweep_traced_values_unreachable():
    from rfx.vmap_sweep import vmap_material_sweep
    sim = scene(eps=2.)
    def sweep(values):
        return vmap_material_sweep(sim, 'body.eps_r', values, n_steps=2).time_series
    with pytest.raises(jax.errors.TracerArrayConversionError):
        jax.jit(sweep)(jnp.array([2., .5]))


def test_run_bar_detects_unstable_uniform_0975(monkeypatch):
    # Normally refused. Removing the setup rule must expose measured growth,
    # so this also checks the observer against a physically unstable control.
    import rfx
    from rfx.grid import Grid
    from rfx._internal._uniform_specs import SourceSpec
    # An even periodic lattice includes the bulk Nyquist mode exactly.
    grid = Grid(freq_max=10e9, domain=(.008,)*3, dx=H, cpml_layers=0, periodic_axes='xyz')
    materials = MaterialArrays(jnp.full(grid.shape, .975), jnp.zeros(grid.shape), jnp.ones(grid.shape))
    source = SourceSpec(3, 4, 4, 'ez', jnp.zeros(400).at[0].set(1.))
    peaks = _observe_peaks(monkeypatch)
    try:
        rfx.run(grid, materials, 400, sources=[source], periodic=(True,)*3)
    except ValueError as error:
        assert_cell_message(str(error), grid, .975, lo=(0.,)*3, hi=(.008,)*3)
        return
    _assert_run_bar(peaks, 'uniform_0975_control')
