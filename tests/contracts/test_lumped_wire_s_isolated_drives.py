"""S from one record needs one driven port; passive ports remain receivers."""
import warnings
import inspect

import jax
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.ringdown import RingdownSpec


FREQS = np.array([4e9, 5e9])
POSITIONS = ((.003, .004, .003), (.006, .004, .003))
REMEDY = ("set excite=False on every port but the driven one and run once per driven port, "
          "or use the uniform run(compute_s_params=True), which drives one port per run")


def _model(graded, second_excited, wire=True):
    profile = {"dz_profile": np.array([.001] * 4 + [.0005] * 8)} if graded else {}
    sim = Simulation(freq_max=10e9, domain=(.009, .008, .008), dx=.001,
                     boundary="pec", **profile)
    for index, position in enumerate(POSITIONS):
        sim.add_port(position, "ez", impedance=50,
                     waveform=GaussianPulse(f0=5e9, bandwidth=.8),
                     excite=index == 0 or second_excited,
                     **({"extent": .002} if wire else {}))
    sim.add_probe((.004, .004, .004), "ez")
    return sim


def _call(sim, path):
    if path == "graded_run":
        return sim.run(n_steps=12, compute_s_params=True,
                       s_param_freqs=FREQS, skip_preflight=True)
    return sim.forward(n_steps=12, port_s11_freqs=FREQS, skip_preflight=True)


def _assert_message(message):
    for index, position in enumerate(POSITIONS):
        assert f"port {index} at {position}" in message
    assert REMEDY in message


PATHS = [("uniform_forward", False, True), ("uniform_forward", False, False),
         ("graded_forward", True, True), ("graded_forward", True, False),
         ("graded_run", True, True)]


@pytest.mark.parametrize("path,graded,wire", PATHS)
def test_several_excited_ports_refused_before_step(monkeypatch, path, graded, wire):
    sim = _model(graded, True, wire)

    def stepping(*args, **kwargs):
        pytest.fail("stepping started before refusing mixed S")

    monkeypatch.setattr(jax.lax, "scan", stepping)
    monkeypatch.setattr(jax.lax, "while_loop", stepping)
    with pytest.raises(ValueError) as error:
        _call(sim, path)
    _assert_message(str(error.value))
    if path == "graded_run":
        assert "run(compute_s_params=False) for fields only" in str(error.value)


def test_graded_forward_ringdown_without_bins_refused_before_step(monkeypatch):
    sim = _model(True, True)

    def stepping(*args, **kwargs):
        pytest.fail("stepping started before refusing mixed ring-down S")

    monkeypatch.setattr(jax.lax, "scan", stepping)
    monkeypatch.setattr(jax.lax, "while_loop", stepping)
    with pytest.raises(ValueError) as error:
        sim.forward(n_steps=12, ringdown=RingdownSpec(), skip_preflight=True)
    _assert_message(str(error.value))


@pytest.mark.parametrize("path,graded,wire", PATHS)
def test_one_drive_and_passive_receiver_forms_s(path, graded, wire):
    result = _call(_model(graded, False, wire), path)
    assert result.s_params is not None
    expected_shape = (2, 2, len(FREQS)) if graded else (2, len(FREQS))
    assert np.asarray(result.s_params).shape == expected_shape
    assert np.all(np.isfinite(result.s_params))


def test_graded_forward_implicit_s_omitted_once():
    sim = _model(True, True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        call_line = inspect.currentframe().f_lineno + 1
        result = sim.forward(n_steps=12, skip_preflight=True)
    assert result.s_params is None
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    _assert_message(str(user_warnings[0].message))
    assert user_warnings[0].filename == __file__
    assert user_warnings[0].lineno == call_line
    reference = sim.run(n_steps=12, compute_s_params=False, skip_preflight=True)
    np.testing.assert_array_equal(result.time_series, reference.time_series)
    assert np.any(np.asarray(result.time_series))


def test_uniform_run_isolates_two_drives():
    result = _model(False, True).run(
        n_steps=12, compute_s_params=True, s_param_freqs=FREQS, skip_preflight=True)
    assert np.asarray(result.s_params).shape == (2, 2, len(FREQS))
    assert np.all(np.isfinite(result.s_params))
