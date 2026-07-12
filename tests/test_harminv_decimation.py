"""Regression tests for band-limited Harminv auto-decimation."""

from __future__ import annotations

import time

import numpy as np

from rfx.harminv import HarminvMode, harminv

_DT = 1.2e-12
_F_MIN = 2.0e9
_F_MAX = 3.5e9
_TONES = (
    (2.2e9, 60.0, 1.0),
    (2.8e9, 40.0, 0.3),
)


def _ringdown(n: int, dt: float = _DT) -> np.ndarray:
    t = np.arange(n) * dt
    signal = np.zeros(n)
    for frequency, quality_factor, amplitude in _TONES:
        decay = np.pi * frequency / quality_factor
        signal += amplitude * np.exp(-decay * t) * np.cos(2 * np.pi * frequency * t)
    return signal


def _nearest(modes: list[HarminvMode], frequency: float) -> HarminvMode:
    return min(modes, key=lambda mode: abs(mode.freq - frequency))


def test_auto_decimation_preserves_modes_and_is_fast():
    undecimated = harminv(_ringdown(5000), _DT, _F_MIN, _F_MAX, decimate=False)

    start = time.perf_counter()
    automatic = harminv(_ringdown(20000), _DT, _F_MIN, _F_MAX, decimate="auto")
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0, f"auto-decimated Harminv took {elapsed:.2f} s"
    for frequency, quality_factor, _ in _TONES:
        short_mode = _nearest(undecimated, frequency)
        full_mode = _nearest(automatic, frequency)
        assert abs(full_mode.freq - short_mode.freq) / short_mode.freq < 2e-3
        assert abs(full_mode.Q - short_mode.Q) / short_mode.Q < 0.1
        assert (
            abs(full_mode.amplitude - short_mode.amplitude) / short_mode.amplitude
            < 0.05
        )
        assert abs(full_mode.freq - frequency) / frequency < 2e-3
        assert abs(full_mode.Q - quality_factor) / quality_factor < 0.1


def test_auto_decimation_is_bit_comparable_when_sampling_is_near_band():
    dt = 1.0 / (7.0 * _F_MAX)
    signal = _ringdown(1000, dt)

    automatic = harminv(signal, dt, _F_MIN, _F_MAX, decimate="auto")
    undecimated = harminv(signal, dt, _F_MIN, _F_MAX, decimate=False)

    assert automatic == undecimated
