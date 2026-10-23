"""Uniform multi-device run preserves fields and probe records under a user jit."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation

pytestmark = pytest.mark.distributed

N_STEPS = 60
_PROFILE = np.array([1.0e-3] * 7 + [0.9e-3, 1.1e-3, 0.95e-3, 1.05e-3, 0.9e-3, 1.1e-3, 1.05e-3, 0.95e-3]
                    + [1.0e-3] * 9)


def scene(boundary, *, port=False, graded=False):
    kw = dict(dx_profile=_PROFILE) if graded else {}
    sim = Simulation(freq_max=10e9, domain=(24e-3, 12e-3, 12e-3), dx=1e-3, boundary=boundary,
                     cpml_layers=6 if boundary == 'cpml' else 0, **kw)
    if port:
        sim.add_port((6e-3, 6e-3, 6e-3), 'ez', impedance=50., waveform=GaussianPulse(f0=5e9, bandwidth=.8))
    else:
        sim.add_source((6e-3, 6e-3, 6e-3), 'ez', waveform=GaussianPulse(f0=5e9, bandwidth=.8),
                       amplitude_kind='current')
    sim.add_material('block', eps_r=4.0)
    sim.add(Box((-1.0, 3e-3, 3e-3), (1.0, 9e-3, 9e-3)), material='block')
    sim.add_probe((6e-3, 6e-3, 6e-3), 'ez')
    sim.add_probe((13e-3, 5e-3, 7e-3), 'ez')
    sim.add_probe((19e-3, 6e-3, 6e-3), 'hy')
    return sim


@pytest.fixture(autouse=True)
def two_cpu_devices():
    if len(jax.devices('cpu')) < 2:
        pytest.skip('requires two CPU devices')


@pytest.mark.parametrize('boundary,port', [('pec', False), ('cpml', False), ('pec', True)])
def test_uniform_user_jit_matches_eager(boundary, port):
    sim = scene(boundary, port=port)

    def run():
        return sim.run(n_steps=N_STEPS, devices=jax.devices('cpu')[:2],
                       skip_preflight=True, compute_s_params=False)

    def call():
        result = run()
        return result.time_series, result.state.ez

    eager = call()
    jitted = jax.jit(call)()
    for actual, expected in zip(jitted, eager):
        actual, expected = np.asarray(actual), np.asarray(expected)
        peak = np.max(np.abs(expected))
        assert peak > 0
        assert np.max(np.abs(actual - expected)) <= 9 * np.spacing(np.float32(peak))
    for step in (run().state.step, jax.jit(lambda: run().state.step)()):
        assert isinstance(step, jax.Array)
        assert step.shape == () and step.dtype == jnp.int32
        assert int(step) == N_STEPS


def test_graded_user_jit_default_port_s_matches_eager():
    sim = scene('pec', port=True, graded=True)

    def call():
        return sim.run(n_steps=N_STEPS, devices=jax.devices('cpu')[:2],
                       skip_preflight=True).s_params

    eager = np.asarray(call())
    jitted = np.asarray(jax.jit(call)())
    peak = np.max(np.abs(eager))
    assert peak > 0
    assert np.all(np.abs(jitted - eager) <= 1e-4 * peak)


def test_graded_user_jit_source_run_is_bitwise_equal():
    sim = scene('cpml', graded=True)

    def call():
        result = sim.run(n_steps=N_STEPS, devices=jax.devices('cpu')[:2], skip_preflight=True)
        return result.time_series, result.state.ez, result.state.step

    for actual, expected in zip(jax.jit(call)(), call()):
        np.testing.assert_array_equal(actual, expected)
