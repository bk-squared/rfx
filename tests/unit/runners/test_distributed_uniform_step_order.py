"""Uniform slab probes against the same plain one-device scene."""
import jax
import numpy as np
import pytest

from rfx import Simulation


def _model(boundary, position, geometry_devices):
    sim = Simulation(freq_max=15e9, domain=(.031, .015, .015), dx=.001,
                     boundary=boundary, cpml_layers=6 if boundary == 'cpml' else 0)
    grid = sim._build_grid()
    width = (grid.nx + geometry_devices - 1) // geometry_devices
    node = {'off': grid.pad_x_lo + (4 if geometry_devices == 3 else 8),
            'first': width, 'last': width - 1}[position]
    sim.add_source(((node - grid.pad_x_lo) * .001, .008, .008), 'ez',
                   amplitude_kind='field')
    for x in (.012, .020):
        sim.add_probe((x, .008, .008), 'ez')
    return sim


def _peak_ulp(reference, actual):
    """Use the whole-record reference-family peak, including every step."""
    assert actual.shape == reference.shape
    error = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    peak = np.max(np.abs(reference))
    if peak == 0:
        return np.where(error == 0, 0., np.inf)
    return error / float(np.spacing(np.float32(peak)))


@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
@pytest.mark.parametrize('geometry_devices', [2, 3], ids=['g2', 'g3'])
@pytest.mark.parametrize('devices', [2, 3], ids=['d2', 'd3'])
@pytest.mark.parametrize('position', ['off', 'first', 'last'])
@pytest.mark.parametrize('steps', [12, 60, 240])
def test_plain_one_device_probe_peak(boundary, geometry_devices, devices, position, steps):
    if jax.device_count() < devices:
        pytest.skip(f'requires {devices} devices')
    kwargs = dict(n_steps=steps, skip_preflight=True, compute_s_params=False)
    reference = _model(boundary, position, geometry_devices).run(**kwargs)
    actual = _model(boundary, position, geometry_devices).run(
        **kwargs, devices=jax.devices()[:devices])
    np.testing.assert_array_equal(actual.dt, reference.dt)
    score = _peak_ulp(np.asarray(reference.time_series), np.asarray(actual.time_series))
    assert np.isfinite(score).all()
    assert np.max(score) <= 9., f'worst peak-ULP: {np.max(score)}'
