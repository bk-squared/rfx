"""Distinct synthetic poles survive even inside the former 1 % merge window."""

import numpy as np
import pytest

from rfx.harminv import matrix_pencil


@pytest.mark.parametrize("case", ["a_real", "b_complex", "c_zero_imag_complex_behaves_as_real"])
def test_matrix_pencil_keeps_close_modes(case):
    dt = 0.4e-9
    t = np.arange(1536) * dt
    # 1/T = 1.63 MHz < 0.2 * 9 MHz spacing; both tones have Q=1000.
    frequencies = [1e9] if case.startswith("c_") else [1e9, 1.009e9]
    if case == "b_complex":
        signal = sum(np.exp((-np.pi * f / 1000 + 2j * np.pi * f) * t)
                     for f in frequencies)
    else:
        signal = sum(np.exp(-np.pi * f * t / 1000) * np.cos(2 * np.pi * f * t)
                     for f in frequencies)
        if case.startswith("c_"):
            signal = signal.astype(np.complex128)

    modes = matrix_pencil(signal, dt, 0.95e9, 1.05e9, decimate=False)
    actual = sorted(mode.freq for mode in modes)
    assert len(actual) == len(frequencies), actual
    assert actual == pytest.approx(frequencies, rel=1e-4)


def test_complex_input_drops_only_the_conjugate_image():
    # A real damped cosine (a +f/-f conjugate pair) plus one complex tone at
    # +1.03 GHz: the -1 GHz image is dropped, both physical poles stay.
    dt = 0.4e-9
    t = np.arange(1536) * dt
    alpha = np.pi * 1e9 / 1000
    signal = (np.exp(-alpha * t) * np.cos(2 * np.pi * 1e9 * t)
              + 0.3 * np.exp((-alpha + 2j * np.pi * 1.03e9) * t))
    modes = matrix_pencil(signal, dt, 0.95e9, 1.05e9, decimate=False)
    actual = sorted(mode.freq for mode in modes)
    assert actual == pytest.approx([1e9, 1.03e9], rel=1e-4), actual
