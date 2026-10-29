"""A concrete PEC sheet remains geometry under an outer graded run trace."""
import jax
import numpy as np
import pytest

from rfx import Box, Cylinder, GaussianPulse, Simulation


def assert_trace_records(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    peak = np.max(np.abs(expected))
    assert peak > 0
    assert np.max(np.abs(actual - expected)) <= 9 * np.spacing(np.float32(peak)), 'per-step'
    assert np.max(np.abs(actual.sum(axis=0) - expected.sum(axis=0))) <= 1e-4 * peak, 'sum'


@pytest.mark.parametrize('shape', (
    Box((.003, .003, .003), (.009, .007, .003)),
    Cylinder((.006, .005, .003), radius=.002, height=0., axis='z'),
), ids=('box', 'disc'))
def test_graded_sheet_run_under_user_jit(shape):
    profile = np.array([.001] * 4 + [.0009, .0011] * 2 + [.001] * 4)
    sim = Simulation(freq_max=10e9, domain=(.012, .010, .010), dx=.001,
                     dx_profile=profile, boundary='pec')
    sim.add_thin_conductor(shape)
    sim.add_source((.005, .005, .005), 'ez', amplitude_kind='current',
                   waveform=GaussianPulse(f0=5e9, bandwidth=.8))
    sim.add_probe((.005, .005, .005), 'ez')

    def run():
        return sim.run(n_steps=24, skip_preflight=True).time_series

    # Trace first so a preceding eager call cannot supply cached geometry.
    compiled = jax.jit(run)()
    assert_trace_records(compiled, run())


def test_trace_check_rejects_changed_record():
    expected = np.ones((4, 1), dtype=np.float32)
    with pytest.raises(AssertionError, match='per-step'):
        assert_trace_records(expected * 2, expected)
