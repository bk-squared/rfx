"""Regression tests for band-limited Harminv auto-decimation."""

from __future__ import annotations

import time

import numpy as np
import pytest

from rfx.harminv import HarminvMode, harminv, harminv_record_duration

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


@pytest.mark.parametrize("real_signal", [False, True])
@pytest.mark.parametrize("n,samples_per_period", [(512, 128), (2048, 128), (2048, 32)])
def test_fir_interior_preserves_a_known_damped_pole(n, samples_per_period, real_signal):
    """Zero padding is not part of an exponential's dynamics (#872).

    These well-conditioned clean signals have an exact answer. The 1e-7
    relative bar separates arithmetic error from filter transients; it is
    not an uncertainty bound on noisy experimental records.
    """
    f, q = 1e9, 100.0
    dt = 1 / (samples_per_period * f)
    t = np.arange(n) * dt
    signal = np.exp((-np.pi * f / q + 2j * np.pi * f) * t + 0.3j)
    if real_signal:
        signal = signal.real
    modes = harminv(signal, dt, 0.8e9, 1.2e9)
    assert len(modes) == 1
    assert modes[0].freq == pytest.approx(f, rel=1e-7)
    assert modes[0].Q == pytest.approx(q, rel=1e-7)


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_weak_pole_survives_the_same_relative_rank_threshold(scale):
    """A 0.003-amplitude mode is above the caller's 0.001 rank cut.

    Raising that cut by sqrt(22) loses it. Removing only the multiplier
    instead finds a contaminated pole (Q about 56.5 instead of 100): both
    changes are necessary. Joint fitting must also recover the known
    coefficients of this synthetic probe signal (not normalized energies).
    """
    dt = 1 / (128 * 1e9)
    t = np.arange(4096) * dt
    truth = [(0.8e9, 80.0, 1.0), (1.2e9, 100.0, 0.003)]
    signal = scale * sum(a * np.exp((-np.pi * f / q + 2j * np.pi * f) * t)
                         for f, q, a in truth)
    modes = sorted(harminv(signal, dt, 0.6e9, 1.4e9), key=lambda m: m.freq)
    assert len(modes) == len(truth)
    for mode, (f, q, amplitude) in zip(modes, truth):
        assert mode.freq == pytest.approx(f, rel=1e-7)
        assert mode.Q == pytest.approx(q, rel=1e-7)
        assert mode.amplitude == pytest.approx(scale * amplitude, rel=1e-7, abs=0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weak_pole_with_small_noise_does_not_add_fictitious_modes(seed):
    dt = 1 / (128 * 1e9)
    t = np.arange(4096) * dt
    truth = [(0.8e9, 80.0, 1.0), (1.2e9, 100.0, 0.003)]
    signal = sum(a * np.exp((-np.pi * f / q + 2j * np.pi * f) * t)
                 for f, q, a in truth)
    rng = np.random.default_rng(seed)
    signal += 3e-5 * (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t)))
    modes = sorted(harminv(signal, dt, 0.6e9, 1.4e9), key=lambda m: m.freq)
    assert len(modes) == len(truth)
    for mode, (f, q, _) in zip(modes, truth):
        assert mode.freq == pytest.approx(f, rel=2e-4)
        assert mode.Q == pytest.approx(q, rel=0.02)


def test_short_record_keeps_raw_samples_instead_of_disappearing():
    dt = 1 / (128 * 1e9)
    t = np.arange(80) * dt
    signal = np.exp((-np.pi * 1e9 / 100 + 2j * np.pi * 1e9) * t)
    assert harminv(signal, dt, 0.8e9, 1.2e9) == harminv(
        signal, dt, 0.8e9, 1.2e9, decimate=False)
    assert harminv_record_duration(len(signal), dt, 1.2e9) == 79 * dt


@pytest.mark.parametrize("n", [80, 512, 2048, 2053])
def test_reported_duration_matches_samples_actually_passed_to_svd(n, monkeypatch):
    import importlib
    module = importlib.import_module("rfx.harminv")
    decimate_original, svd_original = module.scipy_decimate, np.linalg.svd
    observed_factors, observed_shapes = [], []

    def decimate_spy(signal, factor, **kwargs):
        observed_factors.append(factor)
        return decimate_original(signal, factor, **kwargs)

    def svd_spy(matrix, **kwargs):
        observed_shapes.append(matrix.shape)
        return svd_original(matrix, **kwargs)

    monkeypatch.setattr(module, "scipy_decimate", decimate_spy)
    monkeypatch.setattr(np.linalg, "svd", svd_spy)
    dt = 1 / (128 * 1e9)
    t = np.arange(n) * dt
    harminv(np.exp((-np.pi * 1e9 / 100 + 2j * np.pi * 1e9) * t),
            dt, 0.8e9, 1.2e9)
    (shape,) = observed_shapes
    used_samples = sum(shape)  # Hankel M=N-L rows and L columns
    effective_dt = dt * np.prod(observed_factors, dtype=int)
    assert harminv_record_duration(n, dt, 1.2e9) == pytest.approx(
        (used_samples - 1) * effective_dt)
    if observed_factors:
        assert harminv_record_duration(n, dt, 1.2e9) < (n - 1) * dt
