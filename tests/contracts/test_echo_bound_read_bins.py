"""Round 28: every returned spectral coordinate enters the realized bound."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, LorentzPole, Simulation
from rfx.measurement.read_bins import declared_read_bins
from rfx.sparams._tail_witness import result_read_bins


PATHS = ('uniform-run', 'uniform-forward', 'graded-run', 'graded-forward',
         'distributed_v2', 'distributed_nu')
S_BINS = np.array([2e9, 3e9], dtype=np.float32)


@pytest.fixture(autouse=True)
def quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield
    jax.clear_caches()


def scene(path, *, boundary='pec'):
    options = {'dx_profile': np.full(12, .001)} if (
        path.startswith('graded') or path == 'distributed_nu') else {}
    return Simulation(freq_max=20e9, domain=(.012, .011, .008), dx=.001,
                      boundary=boundary, cpml_layers=2, **options)


def invoke(sim, path, *, ports=False):
    options = dict(n_steps=2, skip_preflight=True)
    if path.startswith('distributed'):
        devices = jax.devices('cpu')[:2]
        assert len(devices) == 2
        options['devices'] = devices
    if path.endswith('forward'):
        return sim.forward(port_s11_freqs=S_BINS if ports else None, **options)
    return sim.run(compute_s_params=ports,
                   s_param_freqs=S_BINS if ports else None, **options)


def channel(sim, kind, bins):
    if kind == 'plane':
        sim.add_dft_plane_probe(axis='z', coordinate=.004, component='ez',
                                freqs=bins, name='read')
    elif kind == 'flux':
        sim.add_flux_monitor(axis='z', coordinate=.004, freqs=bins, name='read')
    elif kind == 'ntff':
        sim.add_ntff_box((.002, .002, .002), (.010, .009, .006), freqs=bins)
    else:
        sim.add_current_moment_monitor((.002, .002, .002), (.010, .009, .006),
                                       block_size=.001, freqs=bins)


@pytest.mark.parametrize('path', PATHS)
def test_declared_bins_equal_returned_channels(path):
    sim = scene(path, boundary='cpml')
    for kind, frequency in zip(('plane', 'flux', 'ntff', 'moments'),
                               (5.1e9, 6.1e9, 7.1e9, 8.1e9)):
        channel(sim, kind, np.array([frequency], dtype=np.float32))
    sim.add_port((.008, .006, .004), 'ez', impedance=50.)
    expected = np.unique(np.r_[S_BINS, np.array([5.1e9, 6.1e9, 7.1e9, 8.1e9], np.float32)])
    declared = declared_read_bins(sim, S_BINS)
    np.testing.assert_array_equal(declared, expected)
    result = invoke(sim, path, ports=True)
    assert result.lumped_port_sparams
    np.testing.assert_array_equal(result_read_bins(result), declared)


def lorentz_scene(path, kind):
    sim = scene(path, boundary='cpml' if kind == 'ntff' else 'pec')
    w0 = 2 * np.pi * 8e9
    delta, strength = .1 * w0, w0**2
    dense = np.linspace(0., 20e9, 200001)
    omega = 2 * np.pi * dense
    index = np.sqrt(np.abs(4. + strength / (w0**2 - omega**2 + 2j * delta * omega)))
    frequency = dense[np.argmax(index)]
    stored = np.float32(frequency) if kind == 'ntff' else frequency
    omega_read = 2 * np.pi * np.float64(stored)
    expected = np.sqrt(np.abs(4. + strength / (w0**2 - omega_read**2 + 2j * delta * omega_read)))
    sim.add_material('lorentz', eps_r=4.,
                     lorentz_poles=[LorentzPole(w0, delta, strength)])
    sim.add(Box((0., 0., 0.), (.012, .011, .008)), material='lorentz')
    channel(sim, kind, np.array([frequency]))
    peak = np.sqrt(w0**2 - 2 * delta**2) / (2 * np.pi)
    samples = np.array([0., stored, peak if peak == stored else 0.])
    omega = 2 * np.pi * samples
    upper = np.sqrt(4. + np.max(np.abs(strength / (w0**2 - omega**2 + 2j * delta * omega))))
    return sim, expected, upper


@pytest.mark.parametrize('path,kind', [(path, 'plane') for path in PATHS] +
                         [('uniform-run', 'flux'), ('uniform-run', 'ntff')])
def test_lorentz_read_bin_bounds_phase_index(path, kind):
    sim, expected, upper = lorentz_scene(path, kind)
    result = invoke(sim, path)
    actual = float(result.settling_echo_bound['n_max'])
    assert expected <= actual <= upper * (1 + 2**-20), (path, kind, expected, actual)


def test_traced_s_bins_include_concrete_channel_bins():
    sim, expected, upper = lorentz_scene('uniform-forward', 'plane')
    sim.add_port((.008, .006, .004), 'ez', impedance=50.)
    bins = jax.jit(lambda s: declared_read_bins(sim, s))(jnp.asarray(S_BINS))
    np.testing.assert_array_equal(np.sort(np.asarray(bins)),
                                  np.sort(np.r_[S_BINS, np.asarray(sim._dft_planes[0].freqs, np.float32)]))
    actual = jax.jit(lambda s: sim.forward(
        n_steps=2, port_s11_freqs=s, skip_preflight=True).settling_echo_bound)(jnp.asarray(S_BINS))
    eps32 = np.finfo(np.float32).eps
    margin = (1 + eps32 * (2 + 10)) * (1 + eps32)**2 - 1
    assert expected <= float(actual['n_max']) <= upper * (1 + margin)
