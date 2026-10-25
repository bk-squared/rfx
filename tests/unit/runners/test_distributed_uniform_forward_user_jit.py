"""Uniform multi-device forward preserves records under a user's jit."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from tests.unit.runners.test_distributed_uniform_user_jit import (
    N_STEPS, assert_record_matches, record_arrays, record_scene, scene,
)

pytestmark = pytest.mark.distributed


@pytest.fixture(autouse=True)
def two_cpu_devices():
    if len(jax.devices('cpu')) < 2:
        pytest.skip('requires two CPU devices')


def assert_probe_records(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    for probe in range(expected.shape[1]):
        peak = np.max(np.abs(expected[:, probe]))
        assert peak > 0
        error = np.max(np.abs(actual[:, probe] - expected[:, probe]))
        assert error <= 9 * np.spacing(np.float32(peak)), (probe, error, peak)


@pytest.mark.parametrize('boundary,port', [('pec', False), ('cpml', False), ('pec', True)])
def test_uniform_forward_user_jit_matches_eager(boundary, port):
    sim = scene(boundary, port=port)
    options = dict(n_steps=N_STEPS, devices=jax.devices('cpu')[:2], skip_preflight=True)

    def call():
        return sim.forward(distributed=True, **options).time_series

    eager = np.asarray(call())
    run = np.asarray(sim.run(compute_s_params=False, **options).time_series)
    assert eager.shape == run.shape and eager.dtype == run.dtype
    assert eager.tobytes() == run.tobytes()
    assert_probe_records(jax.jit(call)(), eager)


@pytest.mark.parametrize('kind', ['ntff', 'dft', 'flux', 'moments'])
def test_forward_record_user_jit_matches_eager(kind):
    sim = record_scene(kind)

    def call():
        result = sim.forward(n_steps=N_STEPS, distributed=True, devices=jax.devices('cpu')[:2],
                             skip_preflight=True)
        return record_arrays(result, kind)

    assert_record_matches(kind, call(), jax.jit(call)(), context=('forward', kind))


def fallback_scene(kind):
    """Uniform scenes from the leader's probe_fallback_user_jit.py."""
    if kind == 'waveguide':
        sim = Simulation(12e9, (.06, .02, .01), dx=.002, cpml_layers=4,
                         boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'})
        sim.add_waveguide_port(.010, direction='+x', freqs=np.array([10e9, 11e9]),
                               f0=11e9, bandwidth=.6, probe_offset=2, ref_offset=1)
        sim.add_probe((.030, .010, .006), 'ez')
    else:
        sim = Simulation(20e9, (.030, .020, .012), dx=.001, cpml_layers=4)
        sim.add_tfsf_source()
        sim.add_probe((.023, .012, .007), 'ez')
    return sim


@pytest.mark.parametrize('kind', ['waveguide', 'tfsf'])
def test_uniform_fallback_user_jit_and_path(kind):
    sim = fallback_scene(kind)
    options = dict(n_steps=40, devices=jax.devices('cpu')[:2], skip_preflight=True)
    for entry, path in [('forward', 'fwd_distributed'), ('run', 'run_distributed')]:
        def call():
            result = (sim.forward(distributed=True, **options) if entry == 'forward'
                      else sim.run(compute_s_params=False, **options))
            fallback = [d for d in result.diagnostics if d.code == 'distributed.one_device_fallback']
            assert len(fallback) == 1
            assert fallback[0].path == path
            return result.time_series

        eager = np.asarray(call())
        assert_probe_records(jax.jit(call)(), eager)
        native = np.asarray(sim.forward(n_steps=40, skip_preflight=True).time_series)
        assert eager.shape == native.shape and eager.dtype == native.dtype
        assert eager.tobytes() == native.tobytes()


def test_record_tracer_error_without_port_s_is_named_with_its_cause(monkeypatch):
    from rfx.runners._distributed_records import SlabRecords

    def broken_record_step(self, buffer, state, dt, step, **kwargs):
        raise jax.errors.TracerArrayConversionError(state.ez)

    monkeypatch.setattr(SlabRecords, 'update', broken_record_step)
    sim = record_scene('dft')
    with pytest.raises(NotImplementedError, match='read a traced value on the host') as failure:
        jax.jit(lambda: sim.forward(
            n_steps=N_STEPS, distributed=True, devices=jax.devices('cpu')[:2],
            skip_preflight=True).time_series)()
    assert 'port S-parameters' not in str(failure.value)
    assert isinstance(failure.value.__cause__, jax.errors.TracerArrayConversionError)


def test_fallback_port_s_under_jit_is_refused_by_name():
    # The one-device fallback runs before the staged port-S refusal, but forward()'s
    # S-request check refuses every fallback scene (waveguide port or TFSF) with
    # port_s11_freqs before dispatch (probe 5); a waveguide port with a lumped port here.
    sim = fallback_scene('waveguide')
    sim.add_port((.040, .010, .004), 'ez', impedance=50., waveform=GaussianPulse(f0=10e9, bandwidth=.8))
    with pytest.raises(NotImplementedError, match='waveguide ports use compute_waveguide_s_matrix'):
        jax.jit(lambda: sim.forward(
            n_steps=N_STEPS, distributed=True, devices=jax.devices('cpu')[:2], skip_preflight=True,
            port_s11_freqs=np.array([10e9, 11e9])).s_params)()


def test_forward_traced_dft_frequencies_are_refused():
    def call(scale):
        sim = record_scene('dft', freqs=jnp.array([4e9, 6e9]) * scale)
        return sim.forward(n_steps=N_STEPS, distributed=True, devices=jax.devices('cpu')[:2],
                           skip_preflight=True).dft_planes['record'].accumulator

    with pytest.raises(NotImplementedError, match='needs concrete monitor arguments'):
        jax.jit(call)(1.0)
