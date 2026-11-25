"""Requested coaxial frequency bins and the default frequency band."""
import numpy as np
import pytest

from rfx import Simulation
from rfx.errors import RfxRefusal


class Captured(Exception):
    pass


def scene(freq_max=10e9, freq_min=None):
    sim = Simulation(freq_max=freq_max, domain=(.012, .012, .052), dx=.0005,
                     boundary='cpml', cpml_layers=4, snap='declared')
    options = {} if freq_min is None else {'freq_min': freq_min}
    sim.add_coaxial_port((.006, .006, .002), face='top', pin_length=0, **options)
    return sim


@pytest.fixture
def captured(monkeypatch):
    owners = []

    def kernel(grid, materials, n_steps, **kwargs):
        owner, = (o for o in kwargs['measurement'].owners if o.kind == 'coax')
        owners.append(owner)
        raise Captured

    monkeypatch.setattr('rfx.simulation.run', kernel)
    return owners


def call(sim, entry, requested=None):
    options = ({'compute_s_params': True} if requested is None else
               {'s_param_freqs' if entry == 'run' else 'port_s11_freqs': requested})
    return getattr(sim, entry)(n_steps=3, skip_preflight=True, **options)


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_closed_band(captured, entry):
    requested = [1e9, 5e9, 10e9]
    with pytest.raises(Captured):
        call(scene(), entry, requested)
    np.testing.assert_array_equal(captured[0].frequencies, requested)


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_below_band(captured, entry):
    with pytest.raises(RfxRefusal) as caught:
        call(scene(), entry, [0.9e9, 5e9])
    message = str(caught.value)
    assert '1 below freq_min, the lowest at 0.9 GHz' in message
    assert 'freq_min=' in message
    assert 'above freq_max' not in message


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_above_band(captured, entry):
    with pytest.raises(RfxRefusal) as caught:
        call(scene(), entry, [5e9, 10.5e9])
    message = str(caught.value)
    assert '1 above freq_max, the highest at 10.5 GHz' in message
    assert 'freq_max' in message
    assert 'below freq_min' not in message


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_both_bounds(captured, entry):
    with pytest.raises(RfxRefusal) as caught:
        call(scene(), entry, [0.5e9, 5e9, 11e9, 12e9])
    message = str(caught.value)
    assert "the coaxial port's band 1 to 10 GHz" in message
    assert '1 below freq_min, the lowest at 0.5 GHz' in message
    assert '2 above freq_max, the highest at 12 GHz' in message


def test_default_band(captured):
    with pytest.raises(Captured):
        call(scene(freq_min=2e9), 'run')
    assert len(captured[0].frequencies) == 50
    assert captured[0].frequencies[0] == 2e9
    assert captured[0].frequencies[-1] == 10e9


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_float32_upper_bound(captured, entry):
    requested = None if entry == 'run' else np.asarray([10000000600.0], dtype=np.float32)
    with pytest.raises(Captured):
        call(scene(freq_max=10000000600.0), entry, requested)
    assert captured[0].frequencies[-1] == 10000001024.0
    if entry == 'run':
        assert len(captured[0].frequencies) == 50
        assert captured[0].frequencies[0] == 1000000064.0
    else:
        np.testing.assert_array_equal(captured[0].frequencies, [10000001024.0])


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_float32_lower_bound(captured, entry):
    requested = None if entry == 'run' else np.asarray([2000000030.0, 5e9], dtype=np.float32)
    with pytest.raises(Captured):
        call(scene(freq_min=2000000030.0), entry, requested)
    assert captured[0].frequencies[0] == 2000000000.0
    if entry == 'run':
        assert len(captured[0].frequencies) == 50
    else:
        np.testing.assert_array_equal(captured[0].frequencies, [2000000000.0, 5e9])
