"""Fast analytic unit tests for the coaxial transmission-line reflection
extractor (broad-E5 redesign). No FDTD: feed the matrix-pencil estimator
synthetic two-wave modal voltages with a KNOWN propagation constant and
reflection, and assert it recovers them exactly. The end-to-end FDTD
calibration (short/open/matched on a real line) lives in the slow_physics
suite.
"""
import numpy as np
import pytest

from rfx.sources.coaxial_port import (
    coaxial_line_reflection_from_plane_voltages as extract,
)


def _synth(beta, gamma_load, *, ref_m, planes, A=1.0, alpha=0.0):
    """V(z)=A e^{+γz}+B e^{-γz} with γ=α+jβ, tuned so Γ(ref_m)=gamma_load
    when the load is BELOW the probe span."""
    g = alpha + 1j * beta
    # incident (toward -z) = A e^{+γz}; Γ = (B e^{-γ ref})/(A e^{+γ ref})
    B = gamma_load * A * np.exp(+2.0 * g * ref_m)
    z = np.asarray(planes, float)
    V = A * np.exp(+g * z) + B * np.exp(-g * z)
    return z, V


@pytest.mark.parametrize("gamma_load", [-1.0 + 0j, 1.0 + 0j, 0.3 + 0.4j, -0.2 - 0.5j, 0.0 + 0j])
def test_recovers_known_reflection_lossless(gamma_load):
    beta = 180.0  # rad/m
    ref = 0.000
    planes = ref + np.array([6, 10, 14, 18, 22, 26]) * 0.75e-3  # equally spaced, above load
    z, V = _synth(beta, gamma_load, ref_m=ref, planes=planes)
    out = extract(z, V, reference_plane_m=ref)
    assert out.reflection == pytest.approx(gamma_load, abs=1e-6)
    assert np.imag(out.gamma) == pytest.approx(beta, rel=1e-6)
    assert abs(np.real(out.gamma)) < 1e-6          # lossless => alpha ~ 0
    assert out.recurrence_residual < 1e-9
    assert out.fit_residual < 1e-9


def test_recovers_known_reflection_with_loss():
    beta, alpha, gamma_load = 240.0, 12.0, 0.5 - 0.3j
    ref = 0.0
    planes = ref + np.array([5, 9, 13, 17, 21]) * 0.6e-3
    z, V = _synth(beta, gamma_load, ref_m=ref, planes=planes, alpha=alpha)
    out = extract(z, V, reference_plane_m=ref)
    assert out.reflection == pytest.approx(gamma_load, abs=1e-6)
    assert np.imag(out.gamma) == pytest.approx(beta, rel=1e-6)
    assert np.real(out.gamma) == pytest.approx(alpha, rel=1e-6)
    assert out.recurrence_residual < 1e-9


def test_load_above_probes_branch():
    """Reference plane ABOVE the probe span: incident wave is +z (the B term)."""
    beta, gamma_load = 200.0, 0.6 + 0.0j
    g = 1j * beta
    ref = 0.030
    planes = np.array([6, 10, 14, 18, 22]) * 0.75e-3  # below ref
    # incident toward +z = B e^{-γz}; Γ = (A e^{+γ ref})/(B e^{-γ ref})
    B = 1.0
    A = gamma_load * B * np.exp(-2.0 * g * ref)
    z = planes
    V = A * np.exp(+g * z) + B * np.exp(-g * z)
    out = extract(z, V, reference_plane_m=ref)
    assert out.reflection == pytest.approx(gamma_load, abs=1e-6)
    assert abs(out.reflection) == pytest.approx(0.6, abs=1e-6)


def test_lossless_reflection_magnitude_is_unity_for_reactive_load():
    """A purely reactive (|Γ|=1) load is recovered with |Γ|=1 regardless of phase
    — the property that the single-plane V/I path violated (|S11|>1)."""
    beta = 150.0
    ref = 0.0
    planes = ref + np.arange(6) * 4 * 0.5e-3 + 0.004
    for phase_deg in (179, 120, 90, 30):
        gamma_load = np.exp(1j * np.radians(phase_deg))
        z, V = _synth(beta, gamma_load, ref_m=ref, planes=planes)
        out = extract(z, V, reference_plane_m=ref)
        assert abs(out.reflection) == pytest.approx(1.0, abs=1e-6)


def test_input_validation():
    z = np.array([1.0, 2.0, 3.0]) * 1e-3
    V = np.array([1 + 0j, 0.5 + 0j, 0.2 + 0j])
    # too few planes
    with pytest.raises(ValueError):
        extract(z[:2], V[:2], reference_plane_m=0.0)
    # unequal spacing
    with pytest.raises(ValueError):
        extract(np.array([1.0, 2.0, 4.0]) * 1e-3, V, reference_plane_m=0.0)
    # all-zero voltages
    with pytest.raises(ValueError):
        extract(z, np.zeros(3, complex), reference_plane_m=0.0)
