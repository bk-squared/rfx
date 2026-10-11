"""Closed-form records must not acquire modes through window preprocessing."""

import warnings

import numpy as np
import pytest

import rfx


_DT = 1.9e-12
_BAND = (6e9, 14e9)
_ORDINARY = ((9e9, 40, 1, 0.3), (11e9, 200, 0.05, 1.1))
# Captured once on origin/main 2c32b7cf79b796fae825bfa08fc73d9914e61723,
# macOS, using find_resonances(source_decay_time=0) on case 3 below.
_FROZEN = (
    (8999999999.999994, 706858347.0577135, 39.9999999999994,
     0.5000000000000048, 0.30000000000002713, 0.01201452003571557),
    (10999999999.999994, 172787595.94741186, 200.0000000000309,
     0.02499999999999781, 1.1000000000000993, 0.002950307155427323),
)


def _record(n, dt, modes):
    t = np.arange(n, dtype=np.float64) * dt
    return sum(amplitude * np.exp(-np.pi * freq * t / quality)
               * np.cos(2 * np.pi * freq * t + phase)
               for freq, quality, amplitude, phase in modes)


@pytest.fixture(params=["find_resonances", "matrix_pencil_from_probe"])
def extract(request):
    def call(series, dt, **kwargs):
        if request.param == "matrix_pencil_from_probe":
            return rfx.matrix_pencil_from_probe(
                series, dt, _BAND, source_decay_time=0, **kwargs,
            )
        result = rfx.Result(state=None, s_params=None, freqs=None, dt=dt,
                            time_series=series[:, None], freq_range=_BAND)
        return result.find_resonances(source_decay_time=0, **kwargs)
    return request.param, call


def _assert_modes(modes, expected, freq_rtol, q_rtol):
    assert len(modes) == len(expected)
    for mode, (freq, quality) in zip(sorted(modes, key=lambda m: m.freq), expected):
        assert mode.freq == pytest.approx(freq, rel=freq_rtol)
        assert mode.Q == pytest.approx(quality, rel=q_rtol)


def test_case1_alias_is_not_a_record_mode(extract):
    _, call = extract
    series = _record(60_000, _DT, (
        (10e9, 30, 1, 0.3), (1 / (6 * _DT) - 10.5e9, 200, 0.1, 2.0),
    ))
    _assert_modes(call(series, _DT), [(10e9, 30)], 1e-6, 1e-4)


def test_case2_short_window_has_no_in_band_mode(extract):
    _, call = extract
    dt = 1.9066e-12
    series = _record(84, dt, ((30e9, 5, 1, 0.4),))
    assert call(series, dt) == []


def test_case3_ordinary_record_is_bitwise_unchanged(extract):
    name, call = extract
    modes = call(_record(8000, _DT, _ORDINARY), _DT)
    _assert_modes(modes, [(9e9, 40), (11e9, 200)], 1e-9, 1e-6)
    if name == "find_resonances":
        assert [tuple(mode) for mode in modes] == list(_FROZEN)
        assert np.asarray(modes, dtype=np.float64).tobytes() == np.asarray(
            _FROZEN, dtype=np.float64,
        ).tobytes()


def test_case4_long_record_keeps_both_record_modes(extract):
    _, call = extract
    modes = call(_record(200_000, _DT, _ORDINARY), _DT)
    _assert_modes(modes, [(9e9, 40), (11e9, 200)], 1e-6, 1e-3)


def test_case5_zero_window_is_empty_and_removed_options_warn(extract):
    name, call = extract
    series = np.zeros(84, dtype=np.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert call(series, _DT) == []
        if name == "find_resonances":
            assert call(series, _DT, bandpass=False) == []
    assert not caught
    options = {"bandpass": True} if name == "find_resonances" else {"max_samples": 10}
    with pytest.warns(DeprecationWarning) as caught:
        assert call(series, _DT, **options) == []
    assert len(caught) == 1
    assert caught[0].filename == __file__
