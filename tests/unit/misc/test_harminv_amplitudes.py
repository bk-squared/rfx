"""Signal-coefficient checks for joint amplitude fitting, not RF power calibration."""
import numpy as np
import pytest

from rfx.harminv import _fit_mode_amplitudes, harminv


@pytest.mark.parametrize("real_signal", [False, True])
@pytest.mark.parametrize("start", [0, 137])
def test_amplitude_and_phase_use_the_input_origin_with_conjugates(real_signal, start):
    dt = 1 / 128e9
    times = (np.arange(4096) + start) * dt
    truth = [(0.8e9, 80.0, 1.0, 0.4), (1.2e9, 100.0, 0.003, -0.7)]
    signal = sum(a * np.exp((-np.pi * f / q + 2j * np.pi * f) * times + 1j * phase)
                 for f, q, a, phase in truth)
    if real_signal:
        signal = signal.real
    modes = sorted(harminv(signal, dt, 0.6e9, 1.4e9), key=lambda m: m.freq)
    assert len(modes) == len(truth)
    for mode, (f, q, a, phase) in zip(modes, truth):
        coefficient = a * np.exp((-np.pi * f / q + 2j * np.pi * f) * start * dt + 1j * phase)
        if real_signal:
            coefficient /= 2  # positive-frequency member of a real cosine
        assert mode.freq == pytest.approx(f, rel=1e-7)
        assert mode.amplitude == pytest.approx(abs(coefficient), rel=1e-7, abs=0)
        phase_error = np.angle(np.exp(1j * (mode.phase - np.angle(coefficient))))
        assert abs(phase_error) < 1e-7


def test_unreported_pole_still_participates_in_the_amplitude_fit():
    dt = 1 / 128e9
    times = np.arange(512) * dt
    signal = (0.01 * np.exp((-np.pi * 1e9 / 100 + 2j * np.pi * 1e9) * times)
              + np.exp((-np.pi * 1.3e9 / 10 + 2j * np.pi * 1.3e9) * times))
    modes = harminv(signal, dt, 0.9e9, 1.1e9)
    assert len(modes) == 1  # 1.3 GHz lies outside the expanded report band
    assert modes[0].freq == pytest.approx(1e9, rel=1e-7)
    assert modes[0].amplitude == pytest.approx(0.01, rel=1e-7)


def test_near_coincident_poles_expose_noise_sensitivity_despite_small_residual():
    # The public 1% reporting merge is unchanged. This isolates the linear
    # fit's conditioning on known poles, including an unresolvable noisy case.
    dt = 1e-11
    times = np.arange(256) * dt
    poles = np.array([-np.pi * 1e9 / 50 + 2j * np.pi * f for f in (1e9, 1e9 + 1e4)])
    truth = np.array([1 + 0.2j, -0.7 + 0.4j])
    basis = np.exp(times[:, None] * poles)
    signal = basis @ truth
    np.testing.assert_allclose(_fit_mode_amplitudes(signal, poles, dt), truth,
                               rtol=1e-7, atol=0)
    u, singular, _ = np.linalg.svd(basis, full_matrices=False)
    assert singular[0] / singular[-1] > 1e4
    noise = 1e-6 * np.linalg.norm(signal) * u[:, -1]
    fitted = _fit_mode_amplitudes(signal + noise, poles, dt)
    assert np.linalg.norm(fitted - truth) / np.linalg.norm(truth) > 1e-3
    assert np.linalg.norm(basis @ fitted - signal - noise) / np.linalg.norm(signal) < 1e-10


def test_growing_pole_does_not_overflow_and_keeps_the_original_origin():
    dt = 1e-9
    times = np.arange(1024) * dt
    growth, omega = 700 / times[-1], 2 * np.pi * 1e6
    signal = np.exp(growth * (times - times[-1]) + 1j * omega * times)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        fitted = _fit_mode_amplitudes(signal, np.array([growth + 1j * omega]), dt)
    # An absolute default tolerance would let a zero coefficient pass here.
    np.testing.assert_allclose(fitted, [np.exp(-700)], rtol=1e-10, atol=0)
