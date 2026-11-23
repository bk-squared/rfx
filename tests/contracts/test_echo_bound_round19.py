"""Realized volume, smoothing, rounding, census and sharded echo witnesses."""
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from tests.contracts.run_forward_census import BUILDERS, equal_request_pair
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


@pytest.mark.parametrize('name', tuple(BUILDERS))
def test_census_equal_values_and_conservative_requirement(name, capsys):
    options = ({'subpixel_smoothing': 'full_tensor'} if name == 'C-full-tensor' else
               {'subpixel_smoothing': True} if name == 'C-smooth' else {})
    pair = equal_request_pair(name, 'uniform', 'b', n_steps=1, entry_options=options)
    if pair['run']['outcome'] != 'admitted' or pair['forward']['outcome'] != 'admitted':
        assert pair['run']['outcome'] == pair['forward']['outcome'], pair
        with capsys.disabled():
            print(f'19 census {name}: refuses {pair}', file=sys.stderr)
        return
    expected, old_host = declared_index(name)
    bounds = [pair[entry]['result'].settling_echo_bound for entry in ('run', 'forward')]
    for key in bounds[0]:
        np.testing.assert_array_equal(bounds[0][key], bounds[1][key])
    actual = float(bounds[0]['n_max'])
    before = np.nextafter(np.float32(actual), np.float32(-np.inf))
    assert abs(float(before) - expected) <= np.spacing(np.float32(expected)), (name, before, expected)
    result = pair['run']['result']
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
        print(f'19 census {name}: round17_host={old_host!r}; n_max={actual!r}; '
              f'host_required={required!r}; required={got!r}', file=sys.stderr, flush=True)


def test_volume_capacitor_does_not_raise_index():
    sim = shorted_line()
    baseline = sim.forward(n_steps=4, skip_preflight=True).settling_echo_bound
    sim.add_lumped_rlc((.020, .001, 0.), 'ez', C=1e-12)
    result = sim.forward(n_steps=4, skip_preflight=True)
    expected = np.nextafter(np.float32(1.), np.float32(np.inf))
    assert baseline['n_max'] == expected
    assert result.settling_echo_bound['n_max'] == expected


def test_smoothing_subcell_dielectric_raises_index():
    sim = Simulation(freq_max=20e9, domain=(.012, .011, .005), dx=.001, boundary='pec')
    sim.add_material('tiny', eps_r=4.)
    sim.add(Box((.00405, .00405, .00205), (.00415, .00415, .00215)), material='tiny')
    sim.add_probe((.008, .007, .002), 'ez')
    staircase = sim.forward(n_steps=2, skip_preflight=True)
    smoothed = sim.forward(n_steps=2, skip_preflight=True, subpixel_smoothing=True)
    assert staircase.settling_echo_bound['n_max'] == np.nextafter(np.float32(1.), np.float32(np.inf))
    assert smoothed.settling_echo_bound['n_max'] == np.nextafter(np.float32(2.), np.float32(np.inf))


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
