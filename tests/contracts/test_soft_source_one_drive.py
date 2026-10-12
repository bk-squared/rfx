"""A declared soft source's drive depends on its kind only (run()/forward() census row D2).

run() on a closed boundary built the 'current' drive as (Cb/dV)*w and forward() as
(1/dV)*(Cb*w): the same increment, rounded differently, and in a lossless PEC box the
1-ULP drive difference grew to hundreds of ULP of the field. The expected tables here
are formed with numpy, not with the helper.
"""
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.model.source_coefficients import source_table
from rfx.sources.port_drive import soft_source

NATIVES = ('raw', 'cb', 'cb_over_dv')


def _samples():
    t = np.arange(150, dtype=np.float32) * np.float32(9.6e-13)
    return np.asarray(GaussianPulse(f0=10e9, bandwidth=0.5)(t), dtype=np.float32)


@pytest.mark.parametrize('native', NATIVES)
def test_current_drive_is_cb_over_dv_times_w_for_every_helper(native):
    w = _samples()
    cb, dv = np.float32(0.0216839), 1e-12
    got = np.asarray(source_table(w, 'current', native=native, coefficient=cb, volume=dv))
    np.testing.assert_array_equal(got, (cb / np.float32(dv)) * w)


@pytest.mark.parametrize('native', NATIVES)
def test_field_drive_is_the_waveform_for_every_helper(native):
    w = _samples()
    got = np.asarray(source_table(w, 'field', native=native, coefficient=np.float32(3.0)))
    np.testing.assert_array_equal(got, w)


def test_soft_source_refuses_an_unresolved_kind():
    sim = Simulation(freq_max=10e9, domain=(.016, .016, .012), dx=.001, boundary='pec')
    grid = sim._build_grid()
    with pytest.raises(ValueError, match="amplitude_kind 'field' or 'current'"):
        soft_source(grid, (.008, .008, .004), 'ez', GaussianPulse(f0=5e9), 10,
                    materials=None, amplitude_kind=None)
