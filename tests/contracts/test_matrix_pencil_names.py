"""Matrix pencil names and the deprecated 2.x aliases have identical results."""

import inspect
import warnings

import numpy as np
import pytest

import rfx
from rfx.harminv import harminv_record_duration, matrix_pencil_record_duration

_module = __import__("sys").modules["rfx.harminv"]  # ``rfx.harminv`` is the function


_DT = 1e-11  # 10 ps; 5.12 ns of float64 samples.
_BAND = (0.6e9, 1.4e9)


def _two_mode_signal():
    t = np.arange(512, dtype=np.float64) * _DT
    return (
        np.exp(-np.pi * 0.8e9 / 40 * t) * np.cos(2 * np.pi * 0.8e9 * t + 0.3)
        + 0.6 * np.exp(-np.pi * 1.2e9 / 25 * t)
        * np.cos(2 * np.pi * 1.2e9 * t + 0.8)
    )


def _legacy_call(old, new_name, *args, prefix="rfx.", **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = old(*args, **kwargs)
    assert len(caught) == 1
    assert caught[0].category is DeprecationWarning
    assert str(caught[0].message) == (
        f"{prefix}{old.__name__} is deprecated; use {prefix}{new_name} "
        "(same arguments, same result). It implements the matrix pencil method, not Harminv."
    )
    assert caught[0].filename == __file__  # stacklevel=2 identifies the caller.
    return result


def _assert_identical_modes(old, new):
    assert old == new
    assert new
    assert all(isinstance(mode, rfx.MatrixPencilMode) for mode in old + new)
    # Check every field's bytes too (tuple equality alone misses signed zero).
    assert np.asarray(old, dtype=np.float64).tobytes() == np.asarray(
        new, dtype=np.float64
    ).tobytes()


@pytest.mark.parametrize("settings", [{}, {"max_modes": 8, "decimate": False,
                                          "pencil_parameter": 0.4, "min_Q": 2.0,
                                          "sv_threshold": 1e-5}])
def test_matrix_pencil_legacy_modes_are_bitwise_identical(settings):
    signal = _two_mode_signal()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        new = rfx.matrix_pencil(signal, _DT, *_BAND, **settings)
    assert len(new) == 2
    old = _legacy_call(rfx.harminv, "matrix_pencil", signal, _DT, *_BAND, **settings)
    _assert_identical_modes(old, new)


@pytest.mark.parametrize("settings", [{}, {"max_modes": 8, "pencil_parameter": 0.4},
                                     {"max_modes": 8, "decimate": False}])
def test_matrix_pencil_record_duration_legacy_is_identical(settings):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        new = matrix_pencil_record_duration(512, _DT, _BAND[1], **settings)
    old = _legacy_call(harminv_record_duration, "matrix_pencil_record_duration",
                       512, _DT, _BAND[1], prefix="rfx.harminv.", **settings)
    assert old == new
    assert np.float64(old).tobytes() == np.float64(new).tobytes()


@pytest.mark.parametrize("settings", [{}, {"source_decay_time": 32 * _DT,
                                          "max_modes": 8, "max_samples": 400}])
def test_matrix_pencil_from_probe_legacy_modes_are_bitwise_identical(settings):
    # Result construction follows tests/contracts/test_settling_witness_per_bin_tail.py:360.
    result = rfx.Result(state=None, s_params=None, freqs=None, dt=_DT,
                        time_series=_two_mode_signal()[:, None], freq_range=_BAND)
    args = (result.time_series[:, 0], result.dt, result.freq_range)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        new = rfx.matrix_pencil_from_probe(*args, **settings)
        result.find_resonances(source_decay_time=0, bandpass=False)
    old = _legacy_call(rfx.harminv_from_probe, "matrix_pencil_from_probe", *args, **settings)
    _assert_identical_modes(old, new)


_PAIRS = (("harminv", "matrix_pencil"), ("harminv_from_probe", "matrix_pencil_from_probe"),
          ("harminv_record_duration", "matrix_pencil_record_duration"))


@pytest.mark.parametrize("old_name, new_name", _PAIRS)
def test_legacy_signature_equals_new_and_forwards_every_argument(monkeypatch, old_name, new_name):
    old, new = getattr(_module, old_name), getattr(_module, new_name)
    assert inspect.signature(old) == inspect.signature(new)
    # Call the old name with a distinct marker per parameter; the new function, replaced by a
    # recorder, must receive every one unchanged (a dropped keyword would arrive as its default).
    markers = {name: object() for name in inspect.signature(new).parameters if name != "kwargs"}
    received = {}

    def recorder(*args, **kwargs):
        received.update(inspect.signature(new).bind(*args, **kwargs).arguments)
        return []

    monkeypatch.setattr(_module, new_name, recorder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old(**markers)
    for name, marker in markers.items():
        assert received.get(name) is marker, name


def test_matrix_pencil_mode_alias_and_public_exports():
    assert rfx.HarminvMode is rfx.MatrixPencilMode
    for name in ("matrix_pencil", "matrix_pencil_from_probe", "MatrixPencilMode"):
        assert name in rfx.__all__
    for name in ("harminv", "harminv_from_probe", "HarminvMode"):
        assert name not in rfx.__all__  # deprecated: an attribute, not on the star surface
        assert hasattr(rfx, name)
