"""The ring-down window adds no resonance to a closed-form record.

find_resonances and matrix_pencil_from_probe used to put modes into the band that
the record does not contain: unfiltered subsampling of long windows folded
out-of-band content in (an alias reported as a mode), and the FFT band-pass
fallback made undamped modes at k/(N*dt) on short windows. This contract pins
that the window steps (start, mean removal, truncation in time) add none. The
estimator's own limits are outside it (rfx-archive ledger): its rank cut is
relative, so a noise-only record still yields poles; it reports modes up to
10 % beyond the band; its anti-aliasing filter is finite.
"""

import warnings

import numpy as np
import pytest

import importlib

import rfx

_harminv_module = importlib.import_module("rfx.harminv")


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


def test_case3_ordinary_record_is_unchanged(extract):
    name, call = extract
    series = _record(8000, _DT, _ORDINARY)
    modes = call(series, _DT)
    _assert_modes(modes, [(9e9, 40), (11e9, 200)], 1e-9, 1e-6)
    if name == "find_resonances":
        # Main's values, to the last bits a different BLAS can move (measured 4.5e-13).
        np.testing.assert_allclose(np.asarray(modes, dtype=np.float64),
                                   np.asarray(_FROZEN, dtype=np.float64), rtol=1e-10, atol=0)
        # Same process: bit for bit main's direct path, the window minus its mean.
        direct = rfx.matrix_pencil(series - np.mean(series), _DT, *_BAND)
        assert np.asarray(modes, dtype=np.float64).tobytes() == np.asarray(
            direct, dtype=np.float64).tobytes()


def test_case4_long_record_keeps_both_record_modes():
    series = _record(200_000, _DT, _ORDINARY)
    result = rfx.Result(state=None, s_params=None, freqs=None, dt=_DT,
                        time_series=series[:, None], freq_range=_BAND)
    _assert_modes(result.find_resonances(source_decay_time=0), [(9e9, 40), (11e9, 200)],
                  1e-6, 1e-3)


@pytest.mark.parametrize("entry, n, options", [
    ("find_resonances", 200_000, {}), ("find_resonances", 60_000, {}),
    ("matrix_pencil_from_probe", 200_000, {}), ("matrix_pencil_from_probe", 60_000, {}),
    ("matrix_pencil_from_probe", 200_000, {"decimate": False}),  # find_resonances takes no keywords
])
def test_truncation_keeps_the_longest_early_window(monkeypatch, entry, n, options):
    """The pencil gets the first n_keep samples, n_keep the largest kept-<=10,000 length."""
    series = _record(n, _DT, _ORDINARY)
    received = []
    monkeypatch.setattr(_harminv_module, "matrix_pencil",
                        lambda signal, *args, **kwargs: received.append(signal) or [])
    if entry == "find_resonances":
        rfx.Result(state=None, s_params=None, freqs=None, dt=_DT, time_series=series[:, None],
                   freq_range=_BAND).find_resonances(source_decay_time=0)
    else:
        rfx.matrix_pencil_from_probe(series, _DT, _BAND, source_decay_time=0, **options)
    plan = dict(decimate=options.get("decimate", "auto"), pencil_parameter=0.33, max_modes=50)
    n_keep = n
    while _harminv_module._decimation_plan(n_keep, _DT, _BAND[1], **plan)[1] > 10_000:
        n_keep -= 1
    (signal,) = received
    expected = (series - np.mean(series))[:n_keep]
    assert signal.tobytes() == expected.tobytes()


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
