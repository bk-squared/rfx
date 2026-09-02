"""Fast analytic unit tests for the coaxial transmission-line reflection
extractor (broad-E5 redesign). No FDTD: feed the matrix-pencil estimator
synthetic two-wave modal voltages with a KNOWN propagation constant and
reflection, and assert it recovers them exactly. The end-to-end FDTD
calibration (short/open/matched on a real line) lives in the slow_physics
suite.
"""
import numpy as np
import jax
import jax.numpy as jnp
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
    # all-zero voltages (concrete path keeps the informative raise)
    with pytest.raises(ValueError):
        extract(z, np.zeros(3, complex), reference_plane_m=0.0)


# --- AD-traceability (coax AD-traceable extractor) --------------------------
# Traced voltages dispatch to a jax.numpy core; concrete voltages keep the
# byte-identical NumPy path. These gate the differentiable path.

def test_reflection_extractor_grad_matches_closed_form_and_fd():
    """On a planted two-wave profile V = θ·e^{+jβz} + 0.3·e^{-jβz} the load
    reflection is Γ = 0.3/θ, so d|Γ|/dθ = -0.3/θ². Gate the AD gradient against
    BOTH the closed form and a central finite difference."""
    z = np.linspace(0.002, 0.013, 12)

    def obj(theta):
        V = theta * jnp.exp(1j * 300.0 * jnp.asarray(z)) + 0.3 * jnp.exp(
            -1j * 300.0 * jnp.asarray(z)
        )
        return jnp.abs(extract(z, V, reference_plane_m=0.0).reflection)

    g = float(jax.grad(obj)(jnp.asarray(0.7)))
    assert np.isfinite(g), f"gradient not finite: {g}"
    closed_form = -0.3 / 0.7 ** 2
    assert abs(g - closed_form) < 1e-4, f"AD {g:.8f} vs closed form {closed_form:.8f}"

    h = 1e-4
    fd = (float(obj(0.7 + h)) - float(obj(0.7 - h))) / (2 * h)
    assert abs(g - fd) / max(abs(fd), 1e-12) < 1e-3, f"AD {g:.8f} vs FD {fd:.8f}"


def test_reflection_grad_finite_at_reactive_null():
    """Double-``where`` robustness: a purely reactive |Γ|=1 load — the match/null
    regime that would leak 0·inf=nan through a naive sqrt/divide — keeps a finite
    gradient."""
    z = np.linspace(0.002, 0.013, 12)

    def obj(theta):
        # B on the unit circle, A=θ  ->  |Γ| = 1/θ  (=1 at θ=1)
        V = theta * jnp.exp(1j * 150.0 * jnp.asarray(z)) + np.exp(
            1j * np.radians(120.0)
        ) * jnp.exp(-1j * 150.0 * jnp.asarray(z))
        return jnp.abs(extract(z, V, reference_plane_m=0.0).reflection)

    val = float(obj(jnp.asarray(1.0)))
    g = float(jax.grad(obj)(jnp.asarray(1.0)))
    assert abs(val - 1.0) < 1e-3, f"|Gamma| not unity: {val}"
    assert np.isfinite(g), f"gradient not finite at reactive null: {g}"
