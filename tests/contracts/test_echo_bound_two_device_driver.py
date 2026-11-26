"""The direct distributed S calculator carries its realized settling bound."""
import warnings

import jax
import numpy as np
import pytest

from rfx import Simulation
from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan


@pytest.mark.parametrize('kind', ['lumped', 'wire'])
def test_two_device_driver_returns_same_bound_as_one_device(kind):
    devices = jax.devices('cpu')[:2]
    assert len(devices) == 2
    sim = Simulation(freq_max=10e9, domain=(.012, .011, .008), dx=.001,
                     boundary='pec')
    sim.add_port((.006, .005, .003), 'ez', impedance=50.,
                 extent=.002 if kind == 'wire' else None)
    bins = np.array([2e9, 3e9], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reference = compute_lumped_wire_s_matrix_via_scan(
            sim, bins, n_steps=320, return_settling=True, skip_preflight=True)[2]
        actual = compute_lumped_wire_s_matrix_via_scan(
            sim, bins, n_steps=320, devices=devices,
            return_settling=True, skip_preflight=True)[2]
    assert actual['status'] in ('pass', 'fail', 'undetermined')
    assert set(actual['echo_bound']) == {'D', 'g', 'n_max', 'tau_max'}
    for key in reference['echo_bound']:
        assert np.asarray(actual['echo_bound'][key], np.float32).tobytes() == np.asarray(
            reference['echo_bound'][key], np.float32).tobytes(), key
    jax.clear_caches()
