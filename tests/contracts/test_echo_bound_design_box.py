"""The uniform forward bound reads the design region stepped by the solver."""

import warnings

import jax
import numpy as np

from rfx import Box, Simulation


def test_uniform_forward_design_box_sets_largest_index():
    sim = Simulation(freq_max=20e9, domain=(.012, .011, .008), dx=.001,
                     boundary='pec')
    sim.add_material('background', eps_r=1.5, mu_r=4.)
    sim.add(Box((0., 0., 0.), (.012, .011, .008)), material='background')
    sim.add_probe((.008, .006, .004), 'ez')
    lo, hi = (.003, .003, .003), (.005, .005, .005)
    grid = sim._build_realized_grid()
    first, last = grid.position_to_index(lo), grid.position_to_index(hi)
    shape = tuple(last[d] - first[d] + 1 for d in range(3))
    design_eps = np.linspace(4., 9., np.prod(shape), dtype=np.float32).reshape(shape)
    expected = np.nextafter(
        np.float32(np.sqrt(np.max(design_eps) * np.float32(4.))), np.float32(np.inf))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = sim.forward(n_steps=2, design_box=(lo, hi),
                             design_eps_override=design_eps, skip_preflight=True)
    assert np.asarray(result.settling_echo_bound['n_max']).tobytes() == expected.tobytes()
    jax.clear_caches()
