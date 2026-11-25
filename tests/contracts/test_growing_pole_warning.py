"""A uniform waveguide run reports a dielectric field growing beside CPML."""

import time
import warnings

import jax
import numpy as np

from rfx import Box, Simulation
from rfx.errors import AccuracyWarning


def test_dielectric_with_air_gap_warns_once_about_growth_on_cpu():
    width, height = .02286, .002406
    dx = width / 19
    length, slab_hi = 32 * dx, 30 * dx
    sim = Simulation(freq_max=12e9, domain=(length, width, height), dx=dx,
                     boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'}, cpml_layers=8)
    sim.add_material('slab', eps_r=10.)
    sim.add(Box((slab_hi - .002406, 0., 0.), (slab_hi, width, height)), material='slab')
    sim.add_waveguide_port(8 * dx, direction='+x', f0=9e9, bandwidth=.8,
                           freqs=np.array([4.76e9, 8e9, 10e9]),
                           probe_offset=4, ref_offset=2, name='input')
    start = time.perf_counter()
    with jax.default_device(jax.devices('cpu')[0]), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = sim.run(n_steps=4000, skip_preflight=True)
        jax.block_until_ready(result)
    elapsed = time.perf_counter() - start
    assert elapsed < 60., elapsed
    witness = result.settling_witness
    assert witness['status'] != 'pass', witness
    poles = witness['growing_poles']
    assert len(poles) == 1, poles
    assert 4.4e9 <= poles[0]['frequency_hz'] <= 4.8e9
    assert 2e-4 <= poles[0]['growth_per_step'] <= 8e-4
    assert all(pole['frequency_hz'] >= 0 for pole in poles)
    messages = [str(row.message) for row in caught
                if issubclass(row.category, AccuracyWarning)
                and 'Growing poles (largest growth first):' in str(row.message)]
    assert len(messages) == 1, messages
    for message in messages:
        assert message.count('GHz, e-folding') == 1
        assert 'term growing after the source ended' in message
        assert 'Increase n_steps' not in message
        assert 'lengthen the record' not in message
        assert 'Run a longer fixed record' not in message
