"""#2449 A2/A5: reported-plane wiring and undefined own-port entries."""

import math
from types import SimpleNamespace
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.errors import ConfigurationWarning
from rfx.runners._admission import require_isolated_s_drive
from tests.oracle.test_waveguide_one_driven_column import (
    scene, grid_for, select_record_time, emit,
)


@pytest.fixture
def record_time(capfd):
    with capfd.disabled():
        return select_record_time()


def test_a2_uniform_raw_matrix_column(record_time, capfd):
    sim = scene('U1')
    steps = math.ceil(record_time / float(grid_for(sim, 'U1').dt))
    result = sim.forward(n_steps=steps, skip_preflight=True, checkpoint=False)
    actual = np.stack([result.waveguide_sparams[name].s_driven for name in ('p1', 'p2')])
    expected = np.asarray(scene('U1').compute_waveguide_s_matrix(
        n_steps=steps, normalize=False, skip_preflight=True).s_params)[:, 0]
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    bars = 1e-4 * np.max(np.abs(expected), axis=1)
    with capfd.disabled():
        emit('A2', steps=steps, max_difference=np.max(np.abs(actual - expected), axis=1), bar=bars)
    for i in range(2):
        assert (np.abs(actual[i] - expected[i]) <= bars[i]).all(), 'A2 per-entry peak bar'
        assert (actual[i][expected[i] == 0] == 0).all(), 'A2 exact zeros'


@pytest.mark.parametrize('mesh', ['U1', 'G1'])
@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('amplitudes', [(1., 0.), (0., 0.), (0., 1.)])
def test_a5_own_entries_and_warning(mesh, entry, amplitudes):
    sim = scene(mesh, amplitudes=amplitudes)
    options = {'checkpoint': False} if entry == 'forward' else {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = getattr(sim, entry)(n_steps=350, skip_preflight=True, **options)
    notices = [w for w in caught if 'no incident wave' in str(w.message)]
    assert len(notices) == 1, 'A5 exactly one no-incident-wave warning'
    assert notices[0].category is ConfigurationWarning
    driven = next((f'p{i + 1}' for i, amplitude in enumerate(amplitudes) if amplitude), None)
    for i, (name, record) in enumerate(result.waveguide_sparams.items()):
        assert record.driven_port == driven
        if not amplitudes[i]:
            assert name in str(notices[0].message)
            assert np.isnan(record.s11).all(), 'A5 undriven own s11 is NaN'
            assert np.isnan(record.s21).all(), 'A5 undriven own s21 is NaN'
        if driven is None:
            assert np.isnan(record.s_driven).all()
        else:
            assert np.isfinite(record.s_driven).all()
            if name == driven:
                assert (np.abs(record.s_driven - record.s11) <= 1e-6 * np.max(np.abs(record.s11))).all()


@pytest.mark.parametrize('amplitude', [0., 1.])
def test_single_port_fields(amplitude):
    sim = scene('U1', amplitudes=(amplitude, 0.))
    sim._waveguide_ports.pop()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = sim.forward(n_steps=350, skip_preflight=True, checkpoint=False)
    record = result.waveguide_sparams['p1']
    assert record.driven_port == ('p1' if amplitude else None)
    if amplitude:
        np.testing.assert_array_equal(record.s_driven, record.s11)
        assert not [w for w in caught if 'no incident wave' in str(w.message)]
    else:
        assert np.isnan(record.s_driven).all()
        assert len([w for w in caught if 'no incident wave' in str(w.message)]) == 1


def test_traced_zero_amplitude_counts_as_driven():
    def admitted(amplitude):
        sim = SimpleNamespace(_waveguide_ports=[SimpleNamespace(name='traced', amplitude=amplitude),
                                               SimpleNamespace(name='passive', amplitude=0.)], _ports=[])
        assert require_isolated_s_drive(sim)
        sim._waveguide_ports[1].amplitude = 1.
        with pytest.raises(ValueError, match='driven ports: traced, passive'):
            require_isolated_s_drive(sim)
        return amplitude
    assert float(jax.jit(admitted)(jnp.asarray(0.))) == 0.
