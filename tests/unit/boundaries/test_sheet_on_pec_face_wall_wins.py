"""Tangential E at a PEC wall is zero even where a resistive sheet ends."""
import warnings

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation


def wall_scene(graded=False):
    mesh = {'dz_profile': np.array([.9, 1.1, 1., 1., 1., .9, 1.1, 1., 1.]) * 1e-3} if graded else {}
    sim = Simulation(30e9, (.013, .011, .009), dx=.001,
                     boundary='pec', cpml_layers=0, **mesh)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sim.add_thin_conductor(Box((0., .002, .004), (.013, .008, .004)),
                               sigma_bulk=1e4, surface_impedance_f0=12e9)
    sim.add_source((.003, .005, .003), 'ey',
                   waveform=GaussianPulse(f0=12e9), amplitude_kind='field')
    sim.add_probe((.002, .005, .004), 'ey')
    sim.add_probe((.011, .006, .004), 'ey')
    return sim


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_sheet_on_pec_face_wall_wins(graded):
    sim = wall_scene(graded)
    result = sim.run(n_steps=160, skip_preflight=True)
    ey = np.asarray(result.state.ey)
    # Both wall rows, over the entire transverse plane. The adjacent rows
    # and the two off-centre probes witness that the pulse reached the sheet.
    assert np.max(np.abs(np.asarray(result.time_series)), axis=0).min() > 0
    assert np.any(ey[1, 2:8, 4] != 0) and np.any(ey[-2, 2:8, 4] != 0)
    assert np.all(ey[[0, -1], :, :] == 0.0), 'PEC wall tangential E must be exactly zero where the sheet touches it'
