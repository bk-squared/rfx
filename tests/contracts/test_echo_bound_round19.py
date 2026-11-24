"""Realized volume, smoothing, rounding, census and sharded echo witnesses."""
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from tests.contracts.run_forward_census import BUILDERS, PATHS, equal_request_pair
from tests.contracts.test_run_forward_one_pipeline import directions
from tests.contracts.test_echo_bound_closing_rule import shorted_line


@pytest.fixture(autouse=True)
def quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield
    jax.clear_caches()


def declared_index(name):
    if name == 'P-coax':
        return np.sqrt(2.1), 1.
    if name.startswith('P-msl'):
        return np.sqrt(2.2), np.sqrt(2.2)
    if name == 'M-mu':
        return np.sqrt(8.), np.sqrt(8.)
    if name == 'M-debye':
        return np.sqrt(5.), np.sqrt(5.)
    if name == 'M-lorentz':
        w0 = 2 * np.pi * 8e9
        delta = .15 * w0
        bins = np.array([0., 20e9, np.sqrt(w0*w0 - 2*delta*delta) / (2*np.pi)])
        response = 4. + .4*w0*w0 / (w0*w0 - (2*np.pi*bins)**2 + 2j*delta*(2*np.pi*bins))
        n = max(1., float(np.max(np.sqrt(np.abs(response)))))
        return n, n
    if name.startswith('M-') or name in ('C-smooth', 'C-full-tensor'):
        return 2., 2.
    return 1., 1.


@pytest.mark.parametrize('name,path,direction', [
    pytest.param(name, path, direction or 'b', id=f'{name}-{path}-{direction or "fields"}')
    for name in BUILDERS for path in PATHS for direction in directions(name)
])
def test_census_equal_values_and_conservative_requirement(name, path, direction, capsys):
    options = ({'subpixel_smoothing': 'full_tensor'} if name == 'C-full-tensor' else
               {'subpixel_smoothing': True} if name == 'C-smooth' else {})
    pair = equal_request_pair(name, path, direction, n_steps=1, entry_options=options)
    for entry, record in pair.items():
        failure = record.get('construction', record)
        assert failure.get('type') not in ('KeyError', 'AttributeError', 'AssertionError'), (
            name, path, direction, entry, record)
    admitted = [entry for entry in ('run', 'forward') if pair[entry]['outcome'] == 'admitted']
    if len(admitted) != 2:
        refused = {entry: record for entry, record in pair.items() if entry not in admitted}
        with capsys.disabled():
            print(f'20 census {name}/{path}/{direction}: refuses {refused}', file=sys.stderr)
    if not admitted:
        return
    expected, old_host = declared_index(name)
    bounds = [pair[entry]['result'].settling_echo_bound for entry in admitted]
    for other in bounds[1:]:
        for key in bounds[0]:
            np.testing.assert_array_equal(bounds[0][key], other[key])
    actual = float(bounds[0]['n_max'])
    before = np.nextafter(np.float32(actual), np.float32(-np.inf))
    assert abs(float(before) - expected) <= np.spacing(np.float32(expected)), (name, before, expected)
    result = pair[admitted[0]]['result']
    grid = result.grid
    lengths = []
    for axis in 'xyz':
        cells = np.asarray(grid.cells(axis), np.float64)
        if axis not in getattr(grid, 'periodic_axes', ()):
            cells = cells[:-1]
        lo, hi = getattr(grid, f'pad_{axis}_lo'), getattr(grid, f'pad_{axis}_hi')
        lengths.append(np.sum(cells[lo:len(cells)-hi], dtype=np.float64))
    group = 1.
    for cfg in (result.waveguide_ports or {}).values():
        read = np.asarray(cfg.freqs, np.float64)
        propagating = read[read > float(cfg.f_cutoff)]
        if propagating.size:
            group = min(group, float(np.sqrt(1 - (float(cfg.f_cutoff) / min(propagating))**2)))
    required = 2.05 * 2 * np.linalg.norm(lengths) * expected / (299792458. * group)
    got = 2.05 * float(bounds[0]['tau_max'])
    assert required <= got <= required * (1 + 10*np.finfo(np.float32).eps), (name, got, required)
    with capsys.disabled():
        print(f'20 census {name}/{path}/{direction}: round17_host={old_host!r}; n_max={actual!r}; '
              f'host_required={required!r}; required={got!r}', file=sys.stderr, flush=True)


def test_volume_capacitor_does_not_raise_index():
    sim = shorted_line()
    baseline = sim.forward(n_steps=4, skip_preflight=True).settling_echo_bound
    sim.add_lumped_rlc((.020, .001, 0.), 'ez', C=1e-12)
    result = sim.forward(n_steps=4, skip_preflight=True)
    expected = np.nextafter(np.float32(1.), np.float32(np.inf))
    assert baseline['n_max'] == expected
    assert result.settling_echo_bound['n_max'] == expected


def test_forward_drive_witness_receives_its_realized_bound():
    """The drive row already owns its bound before the aggregate is closed."""
    from rfx.sparams._record_selection import build_settling_witness
    result = equal_request_pair('P-lumped2', 'uniform', 'a', n_steps=150)['forward']['result']
    _, witness = build_settling_witness(
        result, bins=np.asarray(result.freqs), dt=result.dt,
        freq_max=result.settling_freq_max, probe_info=result.settling_probe_info,
        source_end=result.settling_probe_source_end_index,
        port_source_end=result.settling_source_end_index, warn=False)
    row = next(row for row in witness['runs'] if row.get('route') == 's_channels')
    assert row['echo_bound'] is not None
    for key, value in result.settling_echo_bound.items():
        np.testing.assert_array_equal(row['echo_bound'][key], value)


def test_smoothing_reader_includes_subcell_shape():
    """The bound reader includes a shape even when assembly refuses its empty mask."""
    from rfx.core.yee import init_materials
    from rfx.measurement.echo_bound import runner_echo_bound
    sim = Simulation(freq_max=20e9, domain=(.012, .011, .005), dx=.001, boundary='pec')
    sim.add_material('tiny', eps_r=4.)
    sim.add(Box((.00405, .00405, .00205), (.00415, .00415, .00215)), material='tiny')
    grid = sim._build_realized_grid()
    materials = init_materials(grid.shape)
    staircase = runner_echo_bound(sim, grid, materials)
    smoothed = runner_echo_bound(sim, grid, materials, smoothing=True)
    assert staircase['n_max'] == np.nextafter(np.float32(1.), np.float32(np.inf))
    assert smoothed['n_max'] == np.nextafter(np.float32(2.), np.float32(np.inf))


@pytest.mark.parametrize('resonance', [3e9, 8e9], ids=['in-band', 'out-of-band'])
def test_traced_zero_damping_cannot_judge(resonance):
    """Zero damping under a trace has no finite susceptibility error margin."""
    from rfx.core.yee import init_materials
    from rfx.materials.lorentz import LorentzPole
    from rfx.measurement.echo_bound import realized_echo_bound
    from rfx.measurement.record_end import with_echo_bound
    from rfx.ringdown._echo import forward_closing_factor
    from types import SimpleNamespace
    grid = shorted_line()._build_realized_grid()
    materials = init_materials(grid.shape)
    w0 = 2 * np.pi * resonance

    def bound(delta):
        pole = LorentzPole(w0, delta, .4 * w0**2)
        return realized_echo_bound(grid, materials, freqs=[2e9, 4e9],
                                   lorentz=([pole], [None]))

    result = jax.jit(bound)(jnp.float32(0.))
    witness = dict(status='pass', db=-100., reason='',
                   window_starts={10**12: (5 * 10**11, 75 * 10**10)})
    db, detail = with_echo_bound(-100., witness, result, float(grid.dt))
    assert np.isnan(db) and detail['status'] == 'undetermined', detail
    assert np.isinf(result['n_max']) and np.isinf(result['tau_max'])
    assert detail['reason'].startswith('cannot judge:')
    plan = SimpleNamespace(n_steps=10**12, n_start=5 * 10**11)
    factor = jax.jit(lambda delta: forward_closing_factor(
        plan, grid.dt, bound(delta), dtype=jnp.float32))(jnp.float32(0.))
    assert np.isnan(factor)
    damped = jax.jit(bound)(jnp.float32(.15 * w0))
    assert np.isfinite(damped['n_max']) and np.isfinite(damped['tau_max'])


def test_exact_window_requirement_cannot_pass_after_device_rounding():
    """Rounding the solver step and product upward cannot admit the exact edge."""
    from types import SimpleNamespace
    from rfx.ringdown._echo import forward_closing_factor

    plan = SimpleNamespace(n_steps=6, n_start=3)
    dt = np.float64(1.) + np.float64(2.)**-24 + np.float64(2.)**-48
    exact_window = np.float64(plan.n_steps - plan.n_start) * dt
    tau_max = exact_window / np.float64(2.05)
    exact_required = np.float64(2.05) * tau_max
    assert exact_window == exact_required
    rounded_required = np.nextafter(np.float32(exact_required), np.float32(np.inf))
    rounded_window = np.float32(plan.n_steps - plan.n_start) * np.float32(dt)
    assert rounded_window >= rounded_required
    factor = jax.jit(lambda: forward_closing_factor(
        plan, dt, dict(tau_max=tau_max), dtype=jnp.float32))()
    assert np.isnan(factor), (exact_window, exact_required, rounded_window, factor)


@pytest.mark.parametrize('eps', [1., 2.2, 4., 9.1])
def test_rounding_is_one_float32_ulp(eps):
    from rfx.core.yee import init_materials
    from rfx.measurement.echo_bound import realized_echo_bound
    grid = shorted_line()._build_realized_grid()
    mats = init_materials(grid.shape)._replace(eps_r=jnp.full(grid.shape, eps))
    expected = np.nextafter(np.sqrt(np.float32(eps)), np.float32(np.inf))
    assert realized_echo_bound(grid, mats)['n_max'] == expected


@pytest.mark.parametrize('path', ['distributed_v2', 'distributed_nu', 'distributed_forward'])
def test_sharded_max_reads_the_high_index_shard(path):
    devices = jax.devices('cpu')[:2]
    assert len(devices) == 2
    options = {} if path == 'distributed_v2' else {'dx_profile': np.full(15, .001)}
    sim = Simulation(freq_max=15e9, domain=(.015, .007, .007), dx=.001,
                     boundary='pec', **options)
    sim.add_material('high', eps_r=9.)
    sim.add(Box((.011, .002, .002), (.014, .005, .005)), material='high')
    sim.add_probe((.006, .003, .003), 'ez')
    result = (sim.forward(n_steps=2, devices=devices, distributed=True, skip_preflight=True)
              if path == 'distributed_forward' else
              sim.run(n_steps=2, devices=devices, skip_preflight=True))
    assert result.settling_echo_bound['n_max'] == np.nextafter(np.float32(3.), np.float32(np.inf))


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('scene', ['dielectric', 'microstrip'])
def test_one_and_multiple_devices_return_float32_jax_bound_leaves(entry, scene):
    """One declaration carries the same four device leaves eagerly and staged."""
    devices = jax.devices('cpu')[:2]
    assert len(devices) == 2
    if scene == 'microstrip':
        from tests.unit.runners._msl_multidevice_scene import build_scene
        sim = build_scene('T1', mode='uniform').sim
    else:
        sim = Simulation(freq_max=15e9, domain=(.015, .007, .007), dx=.001,
                         boundary='pec')
        sim.add_material('dielectric', eps_r=4.)
        sim.add(Box((0., 0., 0.), (.015, .007, .007)), material='dielectric')
        sim.add_probe((.006, .003, .003), 'ez')

    def bound(count):
        options = dict(n_steps=2, skip_preflight=True)
        if entry == 'run' or count > 1:
            options['devices'] = devices[:count]
        result = (sim.forward(distributed=count > 1, **options) if entry == 'forward'
                  else sim.run(compute_s_params=False, **options))
        return result.settling_echo_bound

    reference = bound(1)
    for actual in (reference, bound(2), jax.jit(lambda: bound(1))(),
                   jax.jit(lambda: bound(2))()):
        assert set(actual) == {'D', 'g', 'n_max', 'tau_max'}
        for key, value in actual.items():
            assert isinstance(value, jax.Array), (key, type(value))
            assert value.shape == () and value.dtype == np.float32, (key, value)
            assert np.asarray(value).tobytes() == np.asarray(reference[key]).tobytes(), key


@pytest.mark.parametrize('medium', ['vacuum', 'debye', 'lorentz'])
@pytest.mark.parametrize('band', [(5e9, 6e9, 7e9), (7e9, 8e9, 9e9)])
def test_dispersive_forward_reads_traced_port_bins(medium, band):
    """Device bins retain the eager index within the declared rounding margin."""
    from rfx import DebyePole, LorentzPole

    sim = Simulation(freq_max=20e9, domain=(.008,) * 3, dx=.001, boundary='pec')
    w0 = 2 * np.pi * 8e9
    delta = .15 * w0
    strength = .4 * w0**2
    poles = ({'debye_poles': [DebyePole(delta_eps=1., tau=1e-11)]} if medium == 'debye'
             else {'lorentz_poles': [LorentzPole(w0, delta, strength)]})
    if medium != 'vacuum':
        sim.add_material('dispersive', eps_r=4., **poles)
        sim.add(Box((0., 0., 0.), (.008,) * 3), material='dispersive')
    sim.add_port((.004, .004, .003), 'ez', impedance=50.)
    bins = np.asarray(band, np.float32)

    def bound(read_bins):
        return sim.forward(n_steps=2, port_s11_freqs=read_bins,
                           skip_preflight=True).settling_echo_bound

    eager = bound(bins)
    actual = jax.jit(bound)(jnp.asarray(bins))
    samples = [0., *np.asarray(bins, np.float64)]
    peak = np.sqrt(w0**2 - 2 * delta**2) / (2 * np.pi)
    if medium == 'lorentz' and min(bins) <= peak <= max(bins):
        samples.append(peak)
    omega = 2 * np.pi * np.asarray(samples)
    response = (4. + 1. / (1 + 1j * omega * 1e-11) if medium == 'debye' else
                4. + strength / (w0**2 - omega**2 + 2j * delta * omega)
                if medium == 'lorentz' else np.ones(len(samples)))
    expected = max(1., float(np.max(np.sqrt(np.abs(response)))))
    assert abs(float(eager['n_max']) - expected) <= 2 * np.spacing(np.float32(expected))
    eps32 = np.finfo(np.float32).eps
    response_margin = eps32 * (1 + (1 + w0 / delta if medium == 'lorentz' else 0))
    margin = (1 + response_margin) * (1 + eps32)**2 - 1
    if medium == 'vacuum':
        np.testing.assert_array_equal(actual['n_max'], eager['n_max'])
    else:
        assert float(eager['n_max']) <= float(actual['n_max']) <= float(eager['n_max']) * (1 + margin), (
            medium, band, actual['n_max'], eager['n_max'], margin)
