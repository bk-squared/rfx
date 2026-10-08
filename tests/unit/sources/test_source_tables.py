"""Source tables prescribe field increments or the final edge's current update."""
from unittest.mock import patch
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation, Box, DebyePole
from rfx.model import source_coefficients as tables
from rfx.simulation import make_source, make_j_source
from rfx.nonuniform import make_current_source

EPS = 8.8541878128e-12


def _wave(t):
    return jnp.cos(2 * jnp.pi * 3e9 * t)


def _field_oracle(builder):
    sim = Simulation(freq_max=8e9, domain=(.008,) * 3, dx=.001,
                     boundary='pec', cpml_layers=0,
                     **({'dx_profile': np.full(8, .001)} if builder == 'graded' else {}))
    grid = sim._build_nonuniform_grid() if builder == 'graded' else sim._build_grid()
    from rfx.core.yee import init_materials
    mats = init_materials(grid.shape)
    def table(a):
        m = mats._replace(eps_r=mats.eps_r * a)
        if builder == 'graded':
            return make_current_source(grid, (4, 4, 4), 'ez', _wave, 32, m, 'field')[4]
        fn = make_source if builder == 'raw' else make_j_source
        return fn(grid, (.004,) * 3, 'ez', _wave, 32, m, 'field').waveform
    actual, tangent = jax.jvp(table, (jnp.float32(1.1),), (jnp.float32(1.),))
    expected = jax.vmap(_wave)(jnp.arange(32, dtype=jnp.float32) * grid.dt)
    np.testing.assert_array_equal(actual, expected)
    assert np.all(np.asarray(tangent) == 0.0)


@pytest.mark.parametrize('builder', ['raw', 'cb', 'graded'])
def test_field_table_has_exactly_zero_material_tangent(builder):
    _field_oracle(builder)


def test_field_table_never_reads_coefficient():
    def forbidden():
        pytest.fail('field table read a material coefficient')
    samples = jnp.array([0., 1., -3.], dtype=jnp.float32)
    for native in ('raw', 'cb', 'cb_over_dv'):
        assert tables.source_table(samples, 'field', native=native,
                                   coefficient=forbidden) is samples


def _sweep_model(case, eps=2.):
    from rfx.boundaries.spec import BoundarySpec
    boundary = BoundarySpec(x='periodic', y='pec', z='pec') if case == 'seam' else 'pec'
    sim = Simulation(freq_max=8e9, domain=(.008,) * 3, dx=.001,
                     cpml_layers=0, boundary=boundary)
    poles = {'debye_poles': [DebyePole(2., 1 / (2 * np.pi * 9e9))]} if case == 'debye' else {}
    sim.add_material('slab', eps_r=eps, **poles)
    sim.add(Box((.004, 0, 0), (.008,) * 3), material='slab')
    pos = (0. if case == 'seam' else .004, .004, .004)
    sim.add_source(pos, 'ez', waveform=jnp.ones_like, amplitude_kind='current')
    if case == 'later_load':
        sim.add_port(pos, 'ez', excite=False)
    sim.add_probe(pos, 'ez')
    return sim


def _sweep_oracle(case):
    from rfx.vmap_sweep import vmap_material_sweep
    sim = _sweep_model(case)
    grid = sim._build_grid()
    dt = grid.dt
    mats, debye, *_ = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])
    i, j, k = grid.position_to_index(sim._ports[0].position)
    left = (i - 1) % grid.shape[0] if case == 'seam' else max(i - 1, 0)
    cells = [(x, y, k) for x in (i, left) for y in (j, max(j - 1, 0))]
    # Build the update denominator by hand from cell arrays and declared
    # loads. No product averaging, source or coefficient helper is a reference.
    eps = sum(float(mats.eps_r[cell]) for cell in cells) / 4
    sigma = sum(float(mats.sigma[cell]) for cell in cells) / 4
    if case == 'later_load':
        sigma += 1 / (50 * grid.dx)  # the later port owns this whole Ez edge
    beta = 0.
    if debye is not None:
        poles, masks = debye
        for pole, mask in zip(poles, masks):
            fraction = 1. if mask is None else sum(float(mask[cell]) for cell in cells) / 4
            beta += EPS * pole.delta_eps * dt / (2 * pole.tau + dt) * fraction
    reference = dt / (EPS * eps + beta + sigma * dt / 2) / grid.dx**3
    # Sweep's fallback defaults to an S-matrix, which refuses a soft source
    # together with a port. Exercise its raw-record construction on both arms;
    # the public refusal is checked separately below.
    original = Simulation.run
    def raw_run(self, **kw):
        return original(self, **dict(kw, compute_s_params=False, skip_preflight=True))
    with patch.object(Simulation, 'run', raw_run):
        swept = np.asarray(vmap_material_sweep(sim, 'slab.eps_r', [2., 6.], n_steps=12).time_series[0])
        plain = np.asarray(_sweep_model(case).run(n_steps=12).time_series)
    np.testing.assert_allclose(swept[0, 0], reference, rtol=2e-7)
    np.testing.assert_array_equal(swept, plain)
    return float(swept[0, 0]), reference


@pytest.mark.parametrize('case', ['interface', 'debye', 'later_load', 'seam'])
def test_sweep_current_reads_update_coefficient(case):
    _sweep_oracle(case)


def test_sweep_later_port_public_refusal():
    from rfx.vmap_sweep import vmap_material_sweep
    with pytest.raises(NotImplementedError, match='S-matrix requests do not support plain sources'):
        vmap_material_sweep(_sweep_model('later_load'), 'slab.eps_r', [2.], n_steps=2)


def _subgrid_sample(order, kind='current', driven=False):
    import rfx.subgridding.jit_runner as runner
    sim = Simulation(freq_max=8e9, domain=(.008,) * 3, dx=.001,
                     boundary='pec', cpml_layers=0)
    sim.add_refinement((.002, .006), ratio=2, validation='off')
    sim.add_material('bulk', eps_r=2.)
    sim.add(Box((0, 0, 0), (.008,) * 3), material='bulk')
    pos = (.004,) * 3
    if order == 'after':
        sim.add_port(pos, 'ez', excite=False)
    if driven:
        sim.add_port(pos, 'ez', waveform=jnp.ones_like)
    else:
        sim.add_source(pos, 'ez', waveform=jnp.ones_like, amplitude_kind=kind)
    if order == 'before':
        sim.add_port(pos, 'ez', excite=False)
    sim.add_probe(pos, 'ez')
    saved = {}
    original = runner.run_subgridded_jit
    class Captured(Exception):
        pass
    def capture(*args, **kwargs):
        saved.update(inspect.signature(original).bind_partial(*args, **kwargs).arguments)
        raise Captured
    with patch.object(runner, 'run_subgridded_jit', capture), pytest.raises(Captured):
        sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
    mats, cfg = saved['mats_f'], saved['config']
    i, j, k, _, samples = saved['opts'].sources_f[0]
    # Read cell arrays, subtract smeared stamp records and restore the owned
    # Ez stamp, independently of the product coefficient accessor.
    eps = np.mean([float(mats.eps_r[x, y, k]) for x in (i, i-1) for y in (j, j-1)])
    bulk_sigma = np.asarray(mats.sigma).copy()
    stamps = mats.sigma_lumped or (None,) * 3
    for stamp in stamps:
        if stamp is not None:
            bulk_sigma -= np.asarray(stamp)
    sigma = np.mean([bulk_sigma[x, y, k] for x in (i, i-1) for y in (j, j-1)])
    if stamps[2] is not None:
        sigma += float(stamps[2][i, j, k])
    current = 1 / cfg.dx_f**3 if not driven else 1 / (50 * cfg.dx_f**2)
    reference = cfg.dt / (EPS * eps + sigma * cfg.dt / 2) * current
    if kind == 'field' and not driven:
        reference = 1.
    result = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
    actual = float(result.time_series[0, -1])
    assert actual == float(samples[0])
    return actual, reference


@pytest.mark.parametrize('kind,driven', [('current', False), ('field', False), ('current', True)])
def test_subgrid_table_reads_last_load(kind, driven):
    values = []
    for order in ('before', 'after'):
        value, reference = _subgrid_sample(order, kind, driven)
        np.testing.assert_allclose(value, reference, rtol=2e-7)
        values.append(value)
    assert values[0] == values[1]


def test_mutation_field_coefficient_round_trip_is_detected(monkeypatch):
    original = tables.source_table
    def mutated(samples, kind=None, **kw):
        if kind == 'field' and kw['native'] != 'raw':
            cb = kw['coefficient']()
            return (kw['volume'] / cb) * ((cb / kw['volume']) * samples)
        return original(samples, kind, **kw)
    monkeypatch.setattr(tables, 'source_table', mutated)
    with pytest.raises(AssertionError):
        _field_oracle('graded')


def test_mutation_plain_coefficient_in_sweep_is_detected(monkeypatch):
    from rfx.model import materials
    from rfx.core.yee import cell_component_e_materials, e_update_coeffs
    def plain(mats, cell, component, dt, *args, **kw):
        # Keep the table-builder call; drop ADE and final realized operands.
        eps, sigma = cell_component_e_materials(mats, cell, component)
        return e_update_coeffs(eps, sigma, dt)[1]
    monkeypatch.setattr(materials, 'e_update_coefficient_at', plain)
    with pytest.raises(AssertionError):
        _sweep_oracle('debye')
    with pytest.raises(AssertionError):
        _sweep_oracle('seam')


@pytest.mark.parametrize('lane', ['uniform-pec', 'uniform-cpml', 'graded', 'adi'])
def test_public_forward_field_table_has_zero_override_tangent(monkeypatch, lane):
    options = {'dx_profile': np.full(8, .001)} if lane == 'graded' else {}
    if lane == 'adi':
        options['solver'] = 'adi'
    sim = Simulation(freq_max=8e9, domain=(.008,) * 3, dx=.001,
                     boundary='cpml' if lane == 'uniform-cpml' else 'pec',
                     cpml_layers=2 if lane == 'uniform-cpml' else 0, **options)
    sim.add_source((.004,) * 3, 'ez', waveform=_wave, amplitude_kind='field')
    sim.add_probe((.005, .004, .004), 'ez')
    grid = sim._build_nonuniform_grid() if lane == 'graded' else sim._build_grid()
    built = []
    original = tables.source_table
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        built.append(result)
        return result
    monkeypatch.setattr(tables, 'source_table', capture)
    def table(a):
        built.clear()
        override = a if lane == 'adi' else jnp.ones(grid.shape) * a
        sim.forward(n_steps=4, eps_override=override,
                    checkpoint=False, skip_preflight=True)
        assert built
        return built[-1]
    actual, tangent = jax.jvp(table, (jnp.float32(1.1),), (jnp.float32(1.),))
    dt = grid.dt * (sim._adi_cfl_factor if lane == 'adi' else 1.)
    expected = jax.vmap(_wave)(jnp.arange(4, dtype=jnp.float32) * dt)
    np.testing.assert_array_equal(actual, expected)
    assert np.all(np.asarray(tangent) == 0.)


@pytest.mark.parametrize('kind', [None, 'field', 'current'])
@pytest.mark.parametrize('timing', ['pre_step', 'post_step'])
def test_disjoint_current_uses_vacuum_update(kind, timing, monkeypatch):
    from rfx.runners import disjoint
    sim = Simulation(freq_max=8e9, domain=(.008,) * 3, dx=.001,
                     boundary='pec', cpml_layers=0)
    sim.add_refinement((.002, .006), ratio=2, validation='off',
                       topology='stage2_disjoint_3d')
    sim._refinement['disjoint_source_timing'] = timing
    sim.add_source((.004,) * 3, 'ez', waveform=jnp.ones_like, amplitude_kind=kind)
    sim.add_probe((.004,) * 3, 'ez')
    injected = []
    original = disjoint._add_fine_source
    def capture(state, component, index, value):
        injected.append(float(value))
        return original(state, component, index, value)
    monkeypatch.setattr(disjoint, '_add_fine_source', capture)
    result = sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
    # The research stepper's material arrays are homogeneous vacuum. Its
    # source correction is dt/(eps0*dV); the public default resolves to current.
    expected = 1. if kind == 'field' else result.dt / (EPS * .0005**3)
    np.testing.assert_allclose(injected[0], expected, rtol=2e-7)
    if timing == 'post_step':
        np.testing.assert_allclose(result.time_series[0, 0], expected, rtol=2e-7)
