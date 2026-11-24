"""A uniform waveguide run reports a dielectric field growing beside CPML."""

import time
import warnings

import jax
import numpy as np

from rfx import Box, Simulation
from rfx.errors import AccuracyWarning


def test_dielectric_touching_cpml_warns_about_growth_on_cpu():
    dx = .001203
    length, width, height = 30 * dx, .02286, .002406
    sim = Simulation(freq_max=12e9, domain=(length, width, height), dx=dx,
                     boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'}, cpml_layers=8)
    sim.add_material('slab', eps_r=10.)
    sim.add(Box((length - .002406, 0., 0.), (length, width, height)), material='slab')
    sim.add_waveguide_port(8 * dx, direction='+x', f0=9e9, bandwidth=.8,
                           freqs=np.array([4.76e9, 8e9, 10e9]),
                           probe_offset=4, ref_offset=2, name='input')
    start = time.perf_counter()
    with jax.default_device(jax.devices('cpu')[0]), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = sim.run(n_steps=4000, skip_preflight=True)
        jax.block_until_ready(result.time_series)
    elapsed = time.perf_counter() - start
    assert elapsed < 60., elapsed
    assert result.settling_witness['growing_poles'], result.settling_witness
    messages = [str(row.message) for row in caught
                if issubclass(row.category, AccuracyWarning)
                and 'Growing poles (largest growth first):' in str(row.message)]
    assert messages
    for message in messages:
        assert 'GHz, e-folding' in message
        assert 'term growing after the source ended' in message
        assert 'Increase n_steps' not in message
