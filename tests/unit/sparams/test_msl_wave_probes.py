"""Point-probe spacing follows the in-plane grid on both mesh lanes."""
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.probes.msl_wave_decomp import register_msl_wave_probes


@pytest.mark.parametrize('graded', [False, True])
def test_microstrip_wave_probes_keep_the_physical_x_spacing(graded):
    profiles = {'dz_profile': np.array([.00025] * 4 + [.0005] * 8)} if graded else {}
    sim = Simulation(10e9, (.016, .009, .005), dx=.0005, cpml_layers=0,
                     boundary='pec', **profiles)
    sim.add_material('laminate', eps_r=3.66)
    sim.add(Box((0, 0, 0), (.016, .009, .001)), material='laminate')
    sim.add(Box((.0011, .0032, .001), (.0147, .0043, .001)), material='pec')
    probes = register_msl_wave_probes(
        sim, feed_x=.0023, direction='+x', y_centre=.0037,
        z_ez=.00065, z_hy=.0008, n_offset_cells=5, n_spacing_cells=3)
    assert probes.delta == pytest.approx(.0015, rel=2e-14, abs=0)
    np.testing.assert_allclose(
        [p.position[0] for p in sim._probes], [.0048, .0063, .0078, .0048],
        rtol=0, atol=1e-17)
    assert sim._uses_nonuniform_mesh is graded
