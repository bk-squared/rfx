"""The report and completion entries resolve the same one-channel poles."""

import numpy as np
import pytest

from rfx.harminv import matrix_pencil
from rfx.ringdown import identify


@pytest.mark.parametrize("scale", [1.0, 1e-34])
def test_report_and_identification_share_poles(scale):
    dt = 1.9e-12
    t = np.arange(256) * dt
    expected = -np.pi * np.array([9e9, 11e9]) / [40, 200] + 2j * np.pi * np.array([9e9, 11e9])
    coefficients = scale * np.array([np.exp(0.3j), 0.05 * np.exp(1.1j)])
    signal = np.exp(t[:, None] * expected) @ coefficients
    # 1/dt < 8*f_max: both entries use the same original-rate record.
    report = matrix_pencil(signal, dt, 6e9, 70e9, sv_threshold=1e-3, max_modes=2)
    model = identify(signal, dt, 0, len(signal), freq_max=70e9, sv_rel=1e-3, unit_tol=0)
    assert len(report) == len(model.s) == 2, "both entries must retain the two closed-form poles"
    actual = np.array(sorted((-m.decay + 2j * np.pi * m.freq for m in report), key=lambda s: s.imag))
    identified = model.s[np.argsort(model.s.imag)]
    np.testing.assert_allclose(identified, expected, rtol=1e-12, atol=0,
                               err_msg="identified poles must match the closed form")
    # Check decay separately: the much larger frequency must not hide a drift.
    np.testing.assert_allclose(actual.real, identified.real, rtol=8*np.finfo(float).eps, atol=0,
                               err_msg="report and identify must share decay to rounding")
    np.testing.assert_allclose(actual.imag, identified.imag, rtol=8*np.finfo(float).eps, atol=0,
                               err_msg="report and identify must share frequency to rounding")
