"""Multi-device run preserves fields and records under a user jit."""
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


def record_scene(kind, *, graded=False, freqs=None):
    kw = dict(dz_profile=np.array([1e-3] * 3 + [0.9e-3, 1.1e-3, 0.95e-3, 1.05e-3, 0.9e-3, 1.1e-3]
                                 + [1e-3] * 3)) if graded else {}
    sim = Simulation(freq_max=10e9, domain=(24e-3, 12e-3, 12e-3), dx=1e-3,
                     boundary='cpml', cpml_layers=6, **kw)
    sim.add_source((6e-3, 6e-3, 6e-3), 'ez', waveform=GaussianPulse(f0=5e9, bandwidth=.8),
                   amplitude_kind='current')
    sim.add_probe((13e-3, 5e-3, 7e-3), 'ez')
    freqs = np.array([4e9, 6e9]) if freqs is None else freqs
    if kind == 'ntff':
        sim.add_ntff_box((3e-3, 3e-3, 3e-3), (21e-3, 9e-3, 9e-3), freqs=freqs)
    elif kind == 'dft':
        sim.add_dft_plane_probe(axis='z', coordinate=6e-3, component='ez', freqs=freqs, name='record')
    elif kind == 'flux':
        sim.add_flux_monitor(axis='x', coordinate=15e-3, freqs=freqs, name='record')
    else:
        sim.add_current_moment_monitor((3e-3, 3e-3, 3e-3), (11e-3, 9e-3, 9e-3), 4e-3, freqs)
    return sim


def record_arrays(result, kind):
    if kind == 'ntff':
        accumulators = result.ntff_data._asdict()
        metadata = result.ntff_box._asdict()
    elif kind == 'moments':
        accumulators = dict(zip(('accumulator', 'kahan_compensation'), result.current_moment_data))
        metadata = result.current_moment_monitor._asdict()
    else:
        record = (result.dft_planes if kind == 'dft' else result.flux_monitors)['record']
        fields = ('accumulator',) if kind == 'dft' else ('e1_dft', 'e2_dft', 'h1_dft', 'h2_dft')
        accumulators = {name: getattr(record, name) for name in fields}
        metadata = {name: value for name, value in record._asdict().items() if name not in fields}
    numeric = {}
    for name, value in metadata.items():
        if isinstance(value, np.ndarray):
            # Preserve host float64 centres/int64 block keys without enabling x64
            # or silently narrowing them when they cross the jit return boundary.
            numeric[f'{name}/{value.dtype}/{value.shape}'] = jnp.asarray(
                np.frombuffer(value.tobytes(), dtype=np.uint8))
        elif value is not None and not isinstance(value, str):
            numeric[name] = jnp.asarray(value)
    return accumulators, numeric


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
@pytest.mark.parametrize('kind', ['ntff', 'dft', 'flux', 'moments'])
def test_record_user_jit_matches_eager(kind, graded):
    sim = record_scene(kind, graded=graded)

    def call():
        result = sim.run(n_steps=N_STEPS, devices=jax.devices('cpu')[:2],
                         skip_preflight=True, compute_s_params=False)
        return record_arrays(result, kind)

    eager_acc, eager_meta = jax.tree.map(np.asarray, call())
    actual_acc, actual_meta = jax.tree.map(np.asarray, jax.jit(call)())
    # One peak per physical quantity: the flux monitor's E and H accumulators
    # differ by about the wave impedance, so each pair gets its own peak; a
    # compensation array shares its accumulator's peak.
    group = {name: name[0] if kind == 'flux' else kind for name in eager_acc}
    peaks = {}
    for name, value in eager_acc.items():
        peaks[group[name]] = max(peaks.get(group[name], 0.0), np.max(np.abs(value)))
    for name, expected in eager_acc.items():
        peak = peaks[group[name]]
        assert peak > 0, (kind, graded, name)
        error = np.max(np.abs(actual_acc[name] - expected))
        assert error <= 1e-4 * peak, (kind, graded, name, error, peak)
    assert actual_meta.keys() == eager_meta.keys()
    for name, expected in eager_meta.items():
        actual = actual_meta[name]
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, name
        assert actual.tobytes() == expected.tobytes(), name


@pytest.mark.parametrize('kind,what', [('dft', 'a DFT plane probe'), ('flux', 'a flux monitor')])
def test_traced_monitor_frequencies_are_refused(kind, what):
    def call(scale):
        sim = record_scene(kind, freqs=jnp.array([4e9, 6e9]) * scale)
        result = sim.run(n_steps=N_STEPS, devices=jax.devices('cpu')[:2],
                         skip_preflight=True, compute_s_params=False)
        return (result.dft_planes if kind == 'dft' else result.flux_monitors)['record']

    with pytest.raises(NotImplementedError, match='needs concrete monitor arguments') as failure:
        jax.jit(call)(1.0)
    assert str(failure.value) == (
        f'{what} on the distributed multi-device path needs concrete monitor arguments '
        '(frequencies, positions) inside jax.jit: declare them outside the traced function.')
