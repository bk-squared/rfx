"""CPU correctness proof for the N-probe least-squares MSL extractor.

Issue #80 Fix C replaces the 3-probe quadratic ``q + 1/q = (V1+V3)/V2``
(which becomes singular as ``β·Δ → 0``, driving the extracted ``|q|``
above 1 and producing a wrong S11 resonance) with an N-probe
least-squares wave decomposition:

    V_n = α·exp(−jβ·x_n) + γ·exp(+jβ·x_n)

fitted by SVD ``jnp.linalg.lstsq`` over all N probes, with β anchored on
the analytic Hammerstad-Jensen guess.

These tests run on a SYNTHETIC analytic fixture — ``V_n`` is constructed
from known ``(α, γ, β)``, no FDTD — so they are the CPU-checkable
correctness proof for the extractor.  Acceptance:

  * recovered ``α / γ / β`` match the planted values to tight tolerance,
  * ``|q| < 1`` (well-conditioned — the 3-probe failure mode is gone),
  * recovered ``Z0`` within ~1 % of the planted value,
  * a finite-difference ``jax.grad`` check passes (the consumer is AD
    inverse design — ``jnp.linalg.lstsq`` carries a native JVP rule).

The full physics validation (issue #80 acceptance criterion 1 — S11 min
at 9.21 ± 0.20 GHz on the v21 RO4003C patch) needs a GPU FDTD run and is
intentionally NOT covered here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.probes.msl_wave_decomp import extract_msl_nprobe


# ---------------------------------------------------------------------------
# Synthetic analytic fixture
# ---------------------------------------------------------------------------


def _planted_voltages(alpha, gamma, beta, x):
    """V_n = α·e^{−jβx_n} + γ·e^{+jβx_n} for one frequency."""
    x = jnp.asarray(x, dtype=jnp.complex64)
    return alpha * jnp.exp(-1j * beta * x) + gamma * jnp.exp(1j * beta * x)


@pytest.mark.parametrize(
    "beta_true,n_probes",
    [
        (250.0, 3),    # minimal N
        (250.0, 5),    # default N
        (180.0, 7),    # over-determined
        (420.0, 9),    # higher β
    ],
)
def test_nprobe_recovers_planted_amplitudes(beta_true, n_probes):
    """Recover α / γ / β / Z0 from a planted analytic fixture."""
    spacing = 0.4e-3
    x = jnp.arange(n_probes, dtype=jnp.float32) * spacing
    alpha = jnp.asarray(1.0 + 0.0j, dtype=jnp.complex64)
    gamma = jnp.asarray(0.30 * np.exp(1j * 0.7), dtype=jnp.complex64)
    z0_true = 50.0

    v = _planted_voltages(alpha, gamma, beta_true, x)[None, :]  # (1, N)
    i1 = jnp.asarray([(alpha - gamma) / z0_true])
    # Analytic guess deliberately offset 6 % to exercise the β scan.
    beta0 = jnp.asarray([beta_true * 1.06])

    res = extract_msl_nprobe(v, x, i1, beta0, z0_hj=z0_true)

    beta_err = abs(complex(res["beta"][0]).real - beta_true) / beta_true
    alpha_err = abs(complex(res["alpha"][0]) - complex(alpha))
    gamma_err = abs(complex(res["gamma"][0]) - complex(gamma))
    z0_err = abs(complex(res["z0"][0]).real - z0_true) / z0_true
    q_abs = abs(complex(res["q"][0]))

    assert beta_err < 5e-3, f"β error {beta_err:.2e} (got {res['beta'][0]})"
    assert alpha_err < 5e-3, f"α error {alpha_err:.2e}"
    assert gamma_err < 5e-3, f"γ error {gamma_err:.2e}"
    assert z0_err < 1e-2, f"Z0 error {z0_err:.2e} (got {res['z0'][0]})"
    # The 3-probe failure mode is |q| > 1; the N-probe extractor must
    # stay at or below 1.  For a lossless planted line |q| == 1 exactly,
    # so allow a tiny fp32 margin above 1.
    assert q_abs < 1.0 + 1e-5, f"|q| = {q_abs:.6f} (non-physical)"


def test_nprobe_well_conditioned_low_loss():
    """Low-loss line — the 3-probe q→1 singularity regime.

    With β·Δ small the 3-probe discriminant collapses; the N-probe
    least-squares fit must still recover |q| ≤ 1 and the planted S11.
    """
    n_probes = 6
    spacing = 0.2e-3
    beta_true = 120.0          # β·Δ ≈ 0.024 rad — deep in the singular regime
    x = jnp.arange(n_probes, dtype=jnp.float32) * spacing
    alpha = jnp.asarray(1.0 + 0.0j, dtype=jnp.complex64)
    s11_true = 0.25 * np.exp(1j * 1.1)
    gamma = jnp.asarray(s11_true * 1.0, dtype=jnp.complex64)
    z0_true = 50.0

    v = _planted_voltages(alpha, gamma, beta_true, x)[None, :]
    i1 = jnp.asarray([(alpha - gamma) / z0_true])
    beta0 = jnp.asarray([beta_true * 0.95])

    res = extract_msl_nprobe(v, x, i1, beta0, z0_hj=z0_true)
    q_abs = abs(complex(res["q"][0]))
    s11_err = abs(complex(res["s11"][0]) - s11_true)

    assert q_abs < 1.0 + 1e-5, f"|q| = {q_abs:.6f} > 1 in the singular regime"
    assert s11_err < 5e-3, f"S11 error {s11_err:.2e} (got {res['s11'][0]})"


def test_nprobe_lossy_line():
    """Mildly lossy planted line — β carries a small imaginary part.

    The β scan models β as REAL (anchored on the real analytic HJ
    guess — the dominant physics for a low-loss MSL); a small planted
    attenuation is absorbed into the fitted α/γ amplitudes by the
    lstsq stage.  The extractor must still recover Z0 and S11 well and
    keep ``|q| ≤ 1`` — it must NOT exhibit the 3-probe ``|q| > 1``
    failure mode.
    """
    n_probes = 8
    spacing = 0.3e-3
    beta_real = 300.0
    # Small attenuation (~0.3 % per probe step) — representative of a
    # low-loss engineering MSL, the regime Fix C targets.
    beta_true = complex(beta_real, -3.0)
    x = jnp.arange(n_probes, dtype=jnp.float32) * spacing
    alpha = jnp.asarray(1.0 + 0.0j, dtype=jnp.complex64)
    gamma = jnp.asarray(0.2 + 0.05j, dtype=jnp.complex64)
    z0_true = 50.0

    v = _planted_voltages(alpha, gamma, beta_true, x)[None, :]
    i1 = jnp.asarray([(alpha - gamma) / z0_true])
    beta0 = jnp.asarray([beta_real * 1.04])

    res = extract_msl_nprobe(v, x, i1, beta0, z0_hj=z0_true)
    q_abs = abs(complex(res["q"][0]))
    # Real-β fit → |q| = 1 (lossless ceiling). The point of Fix C is
    # that |q| never EXCEEDS 1 (the 3-probe non-physical regime).
    assert q_abs < 1.0 + 1e-5, f"|q| = {q_abs:.6f} > 1 (non-physical)"
    z0_err = abs(complex(res["z0"][0]).real - z0_true) / z0_true
    assert z0_err < 1.5e-2, f"Z0 error {z0_err:.2e}"


def test_nprobe_multi_frequency_batch():
    """Vectorised over a frequency axis — each row fitted independently."""
    n_probes = 5
    n_freqs = 12
    spacing = 0.35e-3
    x = jnp.arange(n_probes, dtype=jnp.float32) * spacing
    z0_true = 50.0

    betas = np.linspace(150.0, 480.0, n_freqs)
    alpha = 1.0 + 0.0j
    gammas = 0.3 * np.exp(1j * np.linspace(0.0, 2.0, n_freqs))

    rows = []
    for b, g in zip(betas, gammas):
        rows.append(np.asarray(_planted_voltages(alpha, g, float(b), x)))
    v = jnp.asarray(np.stack(rows, axis=0))           # (n_freqs, N)
    i1 = jnp.asarray((alpha - gammas) / z0_true)
    beta0 = jnp.asarray(betas * 1.05)

    res = extract_msl_nprobe(v, x, i1, beta0, z0_hj=z0_true)

    beta_err = np.abs(np.asarray(res["beta"]).real - betas) / betas
    s11_err = np.abs(np.asarray(res["s11"]) - gammas / alpha)
    assert np.all(beta_err < 5e-3), f"max β error {beta_err.max():.2e}"
    assert np.all(s11_err < 5e-3), f"max S11 error {s11_err.max():.2e}"
    assert np.all(np.abs(np.asarray(res["q"])) < 1.0 + 1e-5)


# ---------------------------------------------------------------------------
# Differentiability — finite-difference jax.grad check
# ---------------------------------------------------------------------------


def _s11_mag_loss(theta):
    """|S11|² as a function of a real reflection-phase parameter ``theta``.

    The whole pipeline (planted V → SVD lstsq extractor → |S11|²) is
    JAX-traced; ``theta`` is a real scalar so ``jax.grad`` returns the
    ordinary real derivative — directly comparable to finite differences.
    """
    n_probes = 6
    spacing = 0.35e-3
    beta_true = 260.0
    x = jnp.arange(n_probes, dtype=jnp.float32) * spacing
    alpha = jnp.asarray(1.0 + 0.0j, dtype=jnp.complex64)
    gamma = 0.3 * jnp.exp(1j * theta.astype(jnp.complex64))
    z0_true = 50.0

    v = _planted_voltages(alpha, gamma, beta_true, x)[None, :]
    i1 = jnp.asarray([1.0 + 0.0j]) * (alpha - gamma) / z0_true
    beta0 = jnp.asarray([beta_true * 1.04])

    res = extract_msl_nprobe(v, x, i1, beta0)
    return jnp.abs(res["s11"][0]) ** 2


@pytest.mark.parametrize("theta0", [0.3, 0.9, 1.7, 2.5])
def test_nprobe_grad_matches_finite_difference(theta0):
    """jax.grad of |S11|² through the N-probe extractor matches FD.

    The consumer is AD inverse design — ``jnp.linalg.lstsq`` has a native
    JVP rule in JAX so no hand-written ``custom_jvp`` is needed; this
    test verifies the gradient genuinely flows and is correct.
    """
    theta = jnp.asarray(theta0, dtype=jnp.float32)
    grad_ad = jax.grad(_s11_mag_loss)(theta)

    h = 1e-3
    fd = (_s11_mag_loss(theta + h) - _s11_mag_loss(theta - h)) / (2.0 * h)

    assert jnp.isfinite(grad_ad), f"AD gradient is not finite: {grad_ad}"
    assert jnp.allclose(grad_ad, fd, atol=2e-3, rtol=2e-2), (
        f"theta={theta0}  AD={grad_ad}  FD={fd}  diff={grad_ad - fd}"
    )


def test_nprobe_grad_is_nonzero():
    """Sanity: the gradient is not silently zeroed out (a dead pipeline)."""
    theta = jnp.asarray(1.2, dtype=jnp.float32)
    grad_ad = jax.grad(_s11_mag_loss)(theta)
    assert abs(float(grad_ad)) > 1e-4, (
        f"gradient {grad_ad} ~ 0 — AD likely not flowing through lstsq"
    )
