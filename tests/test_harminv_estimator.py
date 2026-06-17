"""Estimator-level unit tests for ``rfx.harminv`` (Matrix Pencil).

Until now ``rfx.harminv`` had NO committed estimator-level test — its only
validation was indirect, through physics cross-vals (cv05) and the slow
issue-80 patch resonance gate (``tests/test_issue80_patch_resonance_harminv.py``).
The 2026-06-17 reference-verification confirmed (in throwaway scratch scripts)
that the issue-80 9.32 GHz witness is window-stable, estimator-parameter-stable,
and float32-insensitive — but those findings were not regression-locked.

This module locks the estimator's core contract against a SYNTHETIC
sum-of-decaying-exponentials with KNOWN frequencies, so a regression in the
Matrix-Pencil core (or a float32 precision surprise) is caught directly, not
only via an expensive FDTD run. CPU, fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx.harminv import HarminvMode, harminv

# A synthetic ring-down that mirrors the issue-80 spectrum shape: a dominant
# patch-mode tone with a couple of weaker neighbours, all decaying. The 9.30 GHz
# tone is given the largest amplitude so it must come out as the dominant mode.
_DT = 1.0e-12  # 1 ps -> fs = 1 THz, Nyquist 500 GHz >> 12 GHz
_N = 4000      # 4 ns record
_TONES = (
    # (freq_hz, amplitude, Q, phase)
    (8.80e9, 0.50, 35.0, 0.3),
    (9.30e9, 1.00, 44.0, 0.0),  # dominant
    (11.90e9, 0.60, 18.0, 1.1),
)


def _synthetic_ringdown(dtype=np.float64) -> np.ndarray:
    t = np.arange(_N) * _DT
    sig = np.zeros(_N, dtype=np.float64)
    for f, a, q, phi in _TONES:
        decay = np.pi * f / q  # 1/s, from Q = pi f / decay
        sig += a * np.exp(-decay * t) * np.cos(2 * np.pi * f * t + phi)
    return sig.astype(dtype)


def _freqs_ghz(modes: list[HarminvMode]) -> list[float]:
    return sorted(m.freq / 1e9 for m in modes)


def test_harminv_recovers_known_synthetic_frequencies():
    """Matrix Pencil recovers all three planted tones to <0.5% on a clean signal."""
    modes = harminv(_synthetic_ringdown(), _DT, 7.0e9, 14.0e9)
    got = _freqs_ghz(modes)
    for f_true_hz, _, _, _ in _TONES:
        f_true = f_true_hz / 1e9
        nearest = min(got, key=lambda g: abs(g - f_true))
        assert abs(nearest - f_true) < 0.05, (  # < 50 MHz (~0.5%)
            f"planted {f_true:.3f} GHz not recovered; got {got}"
        )


def test_harminv_dominant_is_largest_amplitude_tone():
    """The returned list is amplitude-sorted; the dominant must be the 9.30 GHz tone.

    This is exactly the selection the issue-80 gate relies on (max-by-amplitude,
    no analytic-nearest bias) — locks that 9.30 is dominant by ENERGY, not by
    proximity to an expected value.
    """
    modes = harminv(_synthetic_ringdown(), _DT, 7.0e9, 14.0e9)
    assert modes, "no modes recovered"
    # API contract: sorted by amplitude, strongest first.
    amps = [m.amplitude for m in modes]
    assert amps == sorted(amps, reverse=True), "modes not amplitude-sorted"
    assert abs(modes[0].freq / 1e9 - 9.30) < 0.05, (
        f"dominant mode {modes[0].freq/1e9:.3f} GHz is not the planted 9.30 GHz tone"
    )


def test_harminv_float32_roundtrip_is_frequency_stable():
    """Storing the signal as float32 (the FDTD field dtype) must not move the estimate.

    rfx FDTD fields are float32; ``harminv`` upcasts the signal to complex128
    internally. This locks that a float32 round-trip of the input leaves the
    dominant frequency essentially unchanged (the verification measured 9.3000
    under both float64 and float32-roundtrip).
    """
    dom_f64 = harminv(_synthetic_ringdown(np.float64), _DT, 7.0e9, 14.0e9)[0].freq
    dom_f32 = harminv(
        _synthetic_ringdown(np.float32).astype(np.float64), _DT, 7.0e9, 14.0e9
    )[0].freq
    assert abs(dom_f64 - dom_f32) / dom_f64 < 1e-3, (
        f"float32 roundtrip moved dominant {dom_f64/1e9:.4f} -> {dom_f32/1e9:.4f} GHz"
    )


def test_harminv_dominant_is_window_stable():
    """Dropping the first quarter of the record must not change the dominant tone.

    Window-stability is the property the issue-80 verification leaned on but had
    not regression-locked; a transient early-window artifact must not flip the
    dominant mode.
    """
    sig = _synthetic_ringdown()
    full = harminv(sig, _DT, 7.0e9, 14.0e9)[0].freq / 1e9
    tail = harminv(sig[_N // 4:], _DT, 7.0e9, 14.0e9)[0].freq / 1e9
    assert abs(full - 9.30) < 0.05 and abs(tail - 9.30) < 0.05, (
        f"dominant not window-stable: full={full:.3f}, tail={tail:.3f} GHz"
    )


def test_harminv_recovers_q_within_band():
    """Recovered Q of the dominant tone is in the right ballpark (planted Q=44)."""
    modes = harminv(_synthetic_ringdown(), _DT, 7.0e9, 14.0e9)
    dom = modes[0]
    # Matrix Pencil on a clean 3-tone signal recovers Q to a few %.
    assert 0.7 * 44.0 < dom.Q < 1.3 * 44.0, f"dominant Q={dom.Q:.1f} far from planted 44"


def test_harminv_modest_noise_does_not_move_dominant():
    """At a realistic SNR the dominant frequency is unchanged (seeded, deterministic)."""
    rng = np.random.default_rng(20260617)
    sig = _synthetic_ringdown()
    noise = rng.standard_normal(_N) * 0.01 * np.max(np.abs(sig))  # ~40 dB SNR
    dom = harminv(sig + noise, _DT, 7.0e9, 14.0e9)[0].freq / 1e9
    assert abs(dom - 9.30) < 0.05, f"noise moved dominant to {dom:.3f} GHz"
