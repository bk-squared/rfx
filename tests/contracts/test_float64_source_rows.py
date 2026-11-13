"""B-src E1–E17 (rfx#2613); actual minus independent reference.

Identical realized inputs and selections apply to both sides. Bars are the
S1 disposition's multiples of u=2**-53. Each exp's 13 u is an estimate.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.contracts.float64_source_reference import (
    U, EPS0, MU0, assert_value, gaussian, noisy_or, profile, pulse, wire_distance,
)
from tests.contracts.float64_source_scenes import POINTS, capture, observe, scene


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64():
        yield


@pytest.fixture(scope='module')
def realized():
    cache = {}
    print(f'B_SRC_DEVICES count={len(jax.devices())} devices={jax.devices()}')

    def get(label):
        if label not in cache:
            cache[label] = capture(label)
            jax.clear_caches()
        return cache[label]
    yield get
    cache.clear()
    jax.clear_caches()


@pytest.mark.parametrize('label', ['pin', 'decay'])
def test_e1_empty_tables(realized, label):
    rows = realized(label)
    tables = [local['mag_src_waveforms'] for name in ('run', 'run_until_decay')
              for local, _ in rows[name] if 'mag_src_waveforms' in local]
    assert tables
    for table in tables:
        assert table.shape[-1] == 0
        assert table.dtype == jnp.float64
    # The two empty electric builders use the same run carrier even when no
    # source declaration calls for a column.
    from rfx.model.source_coefficients import drive_table, drive_columns
    assert drive_table([], (), 2, dtype=jnp.float64).dtype == jnp.float64
    actual = drive_columns((jnp.ones((2,)*3, jnp.float64),)*3,
        (jnp.zeros((2,)*3, jnp.float64),)*3, None, (), (),
        jnp.zeros((2, 0), jnp.float64), (), 1e-12, dtype=jnp.float64)
    assert actual.dtype == jnp.float64


def test_e1_closed_empty_electric_table(realized):
    rows = realized('closed')['run']
    tables = [local['xs'][1] for local, _ in rows if 'xs' in local]
    assert tables
    for table in tables:
        assert table.shape[-1] == 0
        assert table.dtype == jnp.float64


@pytest.mark.parametrize('label', ['U-tfsf', 'G-tfsf', 'methodB'])
def test_e2_auxiliary_profiles(realized, label):
    rows = realized(label)
    configs = rows['init_tfsf_methodB' if label == 'methodB' else 'init_tfsf']
    assert configs
    local, (cfg, _) = configs[-1]
    profiles = ([local['electric'], local['magnetic_lo'], local['magnetic_hi']]
                if label == 'methodB' else
                [local['_prof'], local['_prof_hi'], local['magnetic_lo'], local['magnetic_hi']])
    for actual in profiles:
        args = next(args for args, result in rows['_cpml_profile'] if result is actual)
        sig, kap, alpha, exponent = profile(args['n_layers'], args['dt'], args['dx'],
            args['order'], args['kappa_max'], args['R_asymptotic'], args['sample_offset'])
        assert_value(actual.b, np.exp(exponent), np.ceil(4*np.max(np.abs(exponent)))+14, scale=np.exp(exponent))
        denom = sig*kap + kap**2*alpha
        expected = np.divide(sig*(np.asarray(actual.b)-1), denom,
                             out=np.zeros_like(sig), where=denom > 1e-30)
        assert_value(actual.c, expected, 8)
    # The configuration must retain the profile's precision at its store.
    assert cfg.b_cpml.dtype == cfg.c_cpml.dtype == jnp.float64


def test_e3_dielectric_reference(realized):
    rows = realized('waveguide')['_homogeneous_reference_materials']
    assert rows
    for local, materials in rows:
        assert local['medium'].eps_r == 2.2
        assert_value(materials.eps_r, 2.2, 0)
        assert_value(materials.mu_r, local['medium'].mu_r, 0)


def test_e4_occupancy_store_and_topology():
    from rfx.model.occupancy import occupancy_from_design
    from rfx.topology import TopologyDesignRegion, topology_optimize
    from rfx import GaussianPulse, Simulation
    values = jnp.full((2,)*3, .137, dtype=jnp.float64)
    window = (slice(1, 3),)*3
    actual = occupancy_from_design((4,)*3, window, values, dtype=jnp.float64)
    expected = np.zeros((4,)*3); expected[window] = .137
    assert_value(actual, expected, 0)
    sim = Simulation(freq_max=15e9, domain=(.012,)*3, dx=.002,
                     boundary='pec', precision='float64')
    sim.add_source((.002, .004, .004), 'ez', waveform=GaussianPulse(f0=10e9, bandwidth=.8), amplitude_kind='field')
    sim.add_probe((.010, .008, .008), 'ez')
    region = TopologyDesignRegion((.004,)*3, (.008,)*3, material_bg='air', material_fg='pec')
    with observe() as rows:
        topology_optimize(sim, region, lambda r: jnp.sum(r.time_series**2),
                          n_iterations=1, jit=False, verbose=False)
    local, _ = rows['topology_optimize'][-1]
    assert local['init_density'].dtype == jnp.float64
    assert rows['occupancy_from_design']
    assert all(value.dtype == jnp.float64 for _, value in rows['occupancy_from_design'])


def test_e5_inverse_baseline(realized):
    rows = realized('occupancy')['select_occupancy_operator']
    assert rows
    for local, _ in rows:
        for actual, eps in zip(local['inv_baseline'], local['materials'].components.eps):
            assert_value(actual, 1/np.asarray(eps, dtype=np.float64), 2)


@pytest.mark.parametrize('label', ['U-stair', 'G-stair'])
def test_e6_staircase(realized, label):
    rows = [(local, value) for local, value in realized(label)['_to_jax']
            if local['host'] and not local['promoted']]
    assert rows, 'precondition: the dielectric polyline has no SDF'
    for local, arrays in rows:
        for actual, expected in zip(arrays, local['arrays']):
            expected = np.asarray(expected)
            assert np.any(expected == 3.7) and np.any(expected == 1.)
            assert_value(actual, expected, 0)


def test_e7_bulk_inverse(realized):
    rows = realized('inverse')['compute_inv_eps_tensor_diag']
    assert rows
    for local, arrays in rows:
        grid = local['grid']
        for axis, actual in enumerate(arrays):
            coords = [(np.arange(grid.shape[a])-getattr(grid, f'pad_{"xyz"[a]}_lo')
                       + (.5 if a == axis else 0.))*grid.dx for a in range(3)]
            xyz = np.meshgrid(*coords, indexing='ij')
            distance = np.sqrt(sum((x-c)**2 for x, c in zip(xyz, (.0121, .0123, .0127))))
            bulk = distance < .0061-2*grid.dx
            outside = distance > .0061+2*grid.dx
            assert bulk.any() and outside.any()
            assert_value(np.asarray(actual)[bulk], 1/np.float64(3.7), 2)
            assert_value(np.asarray(actual)[outside], 1., 0)


def test_e8_occupancy_tensor(realized):
    rows = realized('occupancy')
    eps = rows['select_occupancy_operator'][-1][0]['materials'].components.eps
    assert rows['kottke_inv_eps_from_occupancy']
    for local, arrays in rows['kottke_inv_eps_from_occupancy']:
        occupancy = np.clip(np.asarray(local['pec_occupancy'], dtype=np.float64), 0., 1.)
        occupancy = np.where(occupancy < 1e-3, 0., occupancy)
        expected = [(1-m)*(1/np.asarray(e, dtype=np.float64)) for m, e in zip(noisy_or(occupancy), eps)]
        for actual, reference in zip(arrays, expected):
            assert_value(actual, reference, 22)


def test_e9_correction_tables(realized):
    cfg, _ = realized('closed')['init_plane_wave'][-1][1]
    for magnetic, tables in ((False, cfg.tables_e), (True, cfg.tables_h)):
        assert len(tables) == 12
        for i, table in enumerate(tables):
            curl_sign = 1 if (i//2)%2 == 0 else -1
            high = bool(i%2)
            difference = (1 if high else -1) if magnetic else (-1 if high else 1)
            reference = -curl_sign*difference*cfg.dt/((MU0 if magnetic else EPS0)*cfg.dx_1d)
            assert_value(table.coefficient, reference, 4)


def test_e10_e11_delayed_window_launch(realized):
    from rfx.sources.tfsf_discrete_plane_wave import update_line_e
    cfg, state = realized('closed')['init_plane_wave'][-1][1]
    assert [len(w.nodes) for w in cfg.source_windows] == [17, 17, 16]
    step = 63
    axes = [axis for axis in range(3) if cfg.incidence["e_hat"][axis] != 0]
    assert axes
    for axis in axes:
        delays = np.asarray(cfg.source_windows[axis].delays)
        assert np.any((step-delays).astype(np.float32).astype(np.float64) != step-delays)
    state = state._replace(step=jnp.asarray(step, jnp.int32))
    actual = update_line_e(cfg, state, cfg.dx_1d, cfg.dt)
    for axis in axes:
        component = ('ex', 'ey', 'ez')[axis]
        window = cfg.source_windows[axis]
        offset = window.nodes.astype(np.float64)+cfg.origin+cfg.fractions[axis]-cfg.source_p
        weight = np.exp(-.5*(offset/cfg.window_sigma)**2); weight /= np.sum(weight)
        delay = offset/(np.linalg.norm(cfg.m)*299792458.*cfg.dt/cfg.dx_1d)
        time = (step-delay)*cfg.dt
        amplitude = cfg.src_amp*float(cfg.incidence['e_hat'][axis])
        reference = amplitude*pulse(time, cfg.src_tau, cfg.src_t0/cfg.src_tau)*weight
        assert np.max(np.abs(reference)) > 0
        assert_value(np.asarray(getattr(actual, component))[window.nodes], reference, 135)


def test_e12_methodb_interpolation(realized):
    local, (cfg, _) = realized('methodB')['init_tfsf_methodB'][-1]
    ix, iy = np.arange(cfg.x_lo, cfg.x_hi+1), np.arange(cfg.y_lo, cfg.y_hi+1)
    coordinates = [(cfg.x_lo, iy), (cfg.x_hi+1, iy), (ix, cfg.y_lo), (ix, cfg.y_hi+1),
                   (cfg.x_lo-.5, iy), (cfg.x_hi+.5, iy), (ix, cfg.y_lo-.5), (ix, cfg.y_hi+.5)]
    names = ('xlo_e', 'xhi_e', 'ylo_e', 'yhi_e', 'xlo_h', 'xhi_h', 'ylo_h', 'yhi_h')
    for i, (name, (x, y)) in enumerate(zip(names, coordinates)):
        idx = ((x*local['ct']+y*local['st'])*local['dx'])/cfg.d_aux+local['src_off']
        if i >= 4:
            idx = idx-.5
        select = np.abs(idx-np.rint(idx)) >= 1e-9
        print('E12 included', int(select.sum()), 'total', select.size)
        assert select.mean() >= .9
        _, w0, w1 = local[name]
        frac = idx-np.floor(idx)
        assert_value(np.asarray(w1)[select], frac[select], 14 if i < 4 else 16, scale=1.)
        assert_value(np.asarray(w0)[select], (1-frac)[select], 16 if i < 4 else 18, scale=1.)


@pytest.mark.parametrize('label', ['pin', 'floquet', 'port'])
def test_e14_public_time_axes(realized, label):
    rows = realized(label)
    calls = rows['port_drive_waveform' if label == 'port' else 'uniform_source_table']
    assert calls
    for local, actual in calls:
        assert type(local['grid'].dt) is float
        waveform = local['excitation' if label == 'port' else 'waveform']
        times = np.arange(local['n_steps'], dtype=np.float64)*local['grid'].dt
        assert_value(local['times'], times, 0)
        reference = gaussian(times, waveform)
        # Port rows read the samples before the already-qualified Cb scaling.
        assert_value(local['samples'] if label == 'port' else actual, reference, 307)


@pytest.mark.parametrize('kind', ['uniform', 'graded', 'inverse'])
def test_e15_traced_background(kind):
    from rfx.geometry.smoothing import compute_smoothed_eps, compute_smoothed_eps_nonuniform, compute_inv_eps_tensor_diag
    sim, _ = scene('G-stair' if kind == 'graded' else 'U-stair')
    grid = sim._build_realized_grid()
    value = 3.7 if kind == 'inverse' else 2.2
    def build(background):
        if kind == 'inverse':
            return compute_inv_eps_tensor_diag(grid, background_eps=background, dtype=jnp.float64)
        builder = compute_smoothed_eps_nonuniform if kind == 'graded' else compute_smoothed_eps
        return builder(grid, [], background_eps=background, dtype=jnp.float64)
    arrays = jax.jit(build)(jnp.asarray(value, jnp.float64))
    concrete = build(value)
    for actual, host in zip(arrays, concrete):
        assert_value(actual, 1/np.float64(value) if kind == 'inverse' else value, 2 if kind == 'inverse' else 0)
        difference = np.asarray(actual)-np.asarray(host)
        print('E15 traced-minus-concrete', kind, difference.min(), difference.max())


def test_e16_traced_wire():
    from rfx.model.wire_mask import polyline_tube_mask
    coords = (np.linspace(0, .024, 25),)*3
    radius = .003088688805814657
    exact = wire_distance(POINTS, coords)
    mutated = wire_distance(POINTS, coords, mutated=True)
    node = (19, 13, 16)
    assert (exact[node] <= radius**2) != (mutated[node] <= radius**2)
    select = np.abs(exact-radius**2) > 20*U*radius**2
    select[node] = True
    print('E16 compared', select.sum(), 'excluded', select.size-select.sum())
    assert select.mean() >= .9
    actual = jax.jit(lambda x, y, z: polyline_tube_mask(POINTS, radius, x, y, z))(
        *(jnp.asarray(v, jnp.float64) for v in coords))
    np.testing.assert_array_equal(np.asarray(actual)[select], (exact <= radius**2)[select])
