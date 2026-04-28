"""Tests for continuous-coordinate waveguide box-probe post-processing."""

import numpy as np

from rfx.probes.waveguide_box import WaveguideBoxProbe


def test_waveguide_box_probe_integrates_smooth_te10_shape():
    a = 0.02286
    b = 0.01016
    y = np.linspace(-a / 2, a / 2, 24)
    z = np.linspace(-b / 2, b / 2, 11)
    u = y - y.min()
    ez_shape = np.sin(np.pi * u / a)[:, None] * np.ones((1, z.size))
    e = ez_shape[None, :, :].astype(complex)
    h = (0.5 * ez_shape)[None, :, :].astype(complex)

    probe = WaveguideBoxProbe(a=a, b=b, quad_per_interval=4)
    v, i = probe.modal_vi(e, h, y, z)

    # ∫_0^a sin²(pi u/a) du ∫_0^b dz = a*b/2 for the synthetic E field.
    expected_v = a * b / 2.0
    np.testing.assert_allclose(v[0].real, expected_v, rtol=3e-3, atol=0.0)
    np.testing.assert_allclose(i[0].real, 0.5 * expected_v, rtol=3e-3, atol=0.0)
    assert abs(v[0].imag) < 1e-14
    assert abs(i[0].imag) < 1e-14
