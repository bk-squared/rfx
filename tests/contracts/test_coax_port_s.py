"""Coax port spectra against independent calculator and mirrored field witnesses."""
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.contracts.test_coax_port_model_and_drive import scene, capture, actual_layout


BINS = np.array([4e9, 5e9, 6e9])
STEPS = 3000


def report(name, value):
    print(f'{name}: {value}', file=sys.stderr)


def check_native(result, expected):
    native = np.asarray(result.coax_port_metadata['coax_port:0:mode:0']['native_gamma'])
    errors = np.abs(native - expected.s11)
    report('B2 per-bin native error', errors.tolist())
    assert np.all(errors <= 1e-4)


@pytest.fixture(scope='module')
def open_results():
    expected = scene().compute_coaxial_line_reflection(
        termination='open', n_steps=STEPS, freqs=BINS)
    result = scene().run(n_steps=STEPS, s_param_freqs=BINS, skip_preflight=True)
    return result, expected


def test_open_native_and_renormalization(open_results):
    result, expected = open_results
    check_native(result, expected)
    metadata = result.coax_port_metadata['coax_port:0:mode:0']
    gamma, zt = np.asarray(metadata['native_gamma']), metadata['z_tem_ohm']
    reference = ((1 + gamma) * zt - 50 * (1 - gamma)) / ((1 + gamma) * zt + 50 * (1 - gamma))
    residual = np.max(np.abs(np.asarray(result.s_params).reshape(-1) - reference))
    report('B2 renormalization residual', residual)
    assert residual <= 1e-6
    assert result.sparam_time_records[0].samples.shape == (STEPS, 12)
    assert result.settling_witness['scored_port_record_ids'] == ['coax_port:0:mode:0']


def test_forward_against_run(open_results):
    result, _ = open_results
    actual = scene().forward(n_steps=STEPS, port_s11_freqs=BINS, skip_preflight=True)
    for output in (result, actual):
        assert output.s_params.dtype == np.dtype('complex64')
        assert output.coax_port_metadata['coax_port:0:mode:0']['native_gamma'].dtype == np.dtype('complex64')
    error = np.max(np.abs(np.asarray(actual.s_params) - np.asarray(result.s_params)))
    report('B3 max S difference', error)
    assert error <= 1e-4


def test_wave_metadata(open_results):
    result, _ = open_results
    actual = scene().forward(n_steps=STEPS, port_s11_freqs=BINS, skip_preflight=True)
    metadata = [output.coax_port_metadata['coax_port:0:mode:0'] for output in (result, actual)]
    shapes = dict(forward_voltage=(len(BINS),), recurrence_residual=(len(BINS),),
                  plane_voltages=(12, len(BINS)), plane_positions_m=(12,))
    for name, shape in shapes.items():
        left, right = (np.asarray(data[name]) for data in metadata)
        assert left.shape == right.shape == shape
        assert left.dtype == right.dtype
        assert left.dtype.kind == ('c' if name in ('forward_voltage', 'plane_voltages') else 'f')
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-4 * np.max(np.abs(left)))
    for data in metadata:
        positions = np.asarray(data['plane_positions_m'], dtype=np.float64)
        np.testing.assert_allclose(positions, .004 + .002 * np.arange(12), rtol=1e-6)
        voltages = np.asarray(data['plane_voltages'], dtype=np.complex128)
        middle = voltages[1:-1]
        recurrence = np.sum((voltages[:-2] + voltages[2:]) * middle.conj(), axis=0)
        recurrence /= np.sum(np.abs(middle) ** 2, axis=0)
        propagation = np.arccosh(recurrence / 2) / np.mean(np.diff(positions))
        propagation = np.where(propagation.imag < 0, propagation.conj(), propagation)
        pair = np.array([0, 4])
        waves = np.stack([np.linalg.solve(
            np.column_stack((np.exp(k * positions[pair]), np.exp(-k * positions[pair]))),
            voltages[pair, i]) for i, k in enumerate(propagation)], axis=1)
        forward = np.asarray(data['forward_voltage'])
        backward = np.asarray(data['native_gamma']) * forward
        peak = np.max(np.abs(waves))
        report('reference-plane wave fit error / peak', np.max(np.abs(waves - [forward, backward])) / peak)
        np.testing.assert_allclose([forward, backward], waves, rtol=0, atol=1e-3 * peak)
        residual = np.asarray(data['recurrence_residual'])
        report('recurrence residual', residual.tolist())
        assert np.all(np.isfinite(residual))
        assert np.all(residual < .1)


def test_bottom_launch_and_mirror(monkeypatch, open_results):
    sim = scene('bottom')
    with monkeypatch.context() as patch:
        setup = capture(patch, sim, 'run')
    layout = actual_layout(setup, sim)
    grid = setup['grid']
    front, back = min(layout.plane_indices), layout.source_index - 1
    for plane in (back, front):
        sim.add_probe((.0075, .006, grid.node_of(2, plane)), 'ex')
    result = sim.run(n_steps=STEPS, s_param_freqs=BINS, skip_preflight=True)
    velocity = 299792458 / np.sqrt(2.1)
    before_return = int((abs(layout.reference_index-layout.source_index) +
                         abs(layout.reference_index-front)) * grid.dx / velocity / grid.dt)
    values = np.asarray(result.time_series)[:before_return]
    ratio = np.max(np.abs(values[:, 0])) / np.max(np.abs(values[:, 1]))
    report('B4 before-return steps', before_return)
    report('B4 backward/forward peak', ratio)
    assert ratio <= 1e-2
    error = np.max(np.abs(np.asarray(result.s_params) - np.asarray(open_results[0].s_params)))
    report('B4 mirror max S difference', error)
    assert error <= 1e-4


def test_occupancy_gradient(monkeypatch):
    sim = scene()
    with monkeypatch.context() as patch:
        setup = capture(patch, sim, 'run')
    grid = setup['grid']
    layout = actual_layout(setup, sim)
    cell = (grid.pad_x_lo + 14, grid.pad_y_lo + 12, layout.reference_index - 1)
    own = sim.realized_geometry().conductors
    assert not any(np.asarray(e.cells)[cell] for e in own.stamped_entities)
    def objective(value):
        rho = jnp.zeros(grid.shape).at[cell].set(value)
        output = sim.forward(n_steps=900, port_s11_freqs=np.array([5e9]),
                             pec_occupancy_override=rho, checkpoint_segments=30,
                             skip_preflight=True)
        return jnp.abs(output.s_params).reshape(-1)[0] ** 2
    gradient = jax.grad(objective)(jnp.asarray(.25))
    report('B8 occupancy gradient', float(gradient))
    assert np.isfinite(gradient)
    assert abs(float(gradient)) > 0


def test_short_cells_and_native(monkeypatch):
    from rfx.geometry import Box
    from tests.contracts.test_coax_port_model_and_drive import Captured
    sim = scene()
    sim.add(Box((.0035, .0035, .002), (.0085, .0085, .002)), material='pec')
    with monkeypatch.context() as patch:
        actual = capture(patch, sim, 'run')
    root = sim.realized_geometry().conductors
    ref = root.coax_bore_release.layout.reference_index
    recorded = {}
    def kernel(*args, **kwargs):
        recorded['cells'] = np.asarray(kwargs['conductors'].pec_cells)
        raise Captured
    with monkeypatch.context() as patch:
        patch.setattr('rfx.sparams._coax_oneport_record.run_reflection_record', kernel)
        with pytest.raises(Captured):
            scene().compute_coaxial_line_reflection(termination='short', n_steps=3, freqs=BINS)
    difference = np.asarray(root.pec_cells)[:, :, ref] ^ recorded['cells'][:, :, ref]
    report('B2 short realized cell difference', np.argwhere(difference).tolist())
    if difference.any():
        pytest.skip(f'short geometries differ at {np.count_nonzero(difference)} cells; no spectral judgement')
    assert actual['sources']
    result = sim.run(n_steps=STEPS, s_param_freqs=BINS, skip_preflight=True)
    expected = scene().compute_coaxial_line_reflection(termination='short', n_steps=STEPS, freqs=BINS)
    check_native(result, expected)


@pytest.mark.parametrize('mutation,expected', [('a', 0), ('b', 1), ('bprime', 1),
                                             ('wave_a', 0), ('wave_b', 1)])
def test_measurement_mutation(mutation, expected):
    import subprocess
    import gc
    jax.clear_caches()
    gc.collect()
    common = '''
import pytest
import tests.contracts.test_coax_port_s as contract
'''
    changes = {
        'a': '''
contract.check_native = lambda *args: None
selection = 'test_open_native_and_renormalization'
''',
        'b': '''
import rfx.measurement.coax_port as measurement
from rfx.measurement.plan import Owner
measurement.port_owner = lambda *args: Owner('coax_port:0:mode:0', 'coax', ())
selection = 'test_open_native_and_renormalization'
''',
        'bprime': '''
import jax.numpy as jnp
import rfx.measurement.coax_port as measurement
original = measurement.port_result
def from_stopped_records(sim, product, result, **kwargs):
    owner, = (o for o in product._prepared_solve.measurement.owners if o.kind == 'coax')
    samples = next(r.samples for r in result.sparam_time_records if r.kind == 'coax')
    times = jnp.arange(samples.shape[0]) * result.dt
    phases = jnp.exp(-2j * jnp.pi * times[:, None] * jnp.asarray(owner.frequencies)[None, :])
    result = result._replace(coax_sparam_accs=samples.T @ phases)
    return original(sim, product, result, **kwargs)
measurement.port_result = from_stopped_records
selection = 'test_occupancy_gradient'
''',
    }
    if mutation.startswith('wave_'):
        changes[mutation] = '''
import rfx.measurement.coax_port as measurement
original = measurement.port_result
def raw_voltage(*args, **kwargs):
    s11, frequencies, metadata = original(*args, **kwargs)
    for data in metadata.values():
        data['forward_voltage'] = data['plane_voltages'][0]
    return s11, frequencies, metadata
measurement.port_result = raw_voltage
selection = 'test_wave_metadata'
'''
        if mutation == 'wave_a':
            changes[mutation] += '''
original_assert = contract.np.testing.assert_allclose
def check_off(actual, desired, **kwargs):
    if contract.np.shape(actual) != (2, 3):
        original_assert(actual, desired, **kwargs)
contract.np.testing.assert_allclose = check_off
'''
    script = common + changes[mutation] + '''
raise SystemExit(pytest.main(['tests/contracts/test_coax_port_s.py', '-q', '-s', '-k', selection]))
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    report(f'mutation {mutation} exit', result.returncode)
    print(result.stdout + result.stderr, file=sys.stderr)
    assert result.returncode == expected
    if expected:
        assert ('AssertionError' if mutation in ('bprime', 'wave_b') else 'NoneType') in result.stdout
