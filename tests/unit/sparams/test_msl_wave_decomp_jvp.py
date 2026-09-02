"""Unit tests for the custom_jvp on `_solve_q` in msl_wave_decomp.

The 3-probe q-root solver previously used `jnp.where(use_plus, qp, qm)`
plus `jnp.sqrt(complex disc)`.  The √disc branch cut at |q|→1 caused
adjacent finite-difference samples to land on opposite sides of the
cut → reverse-mode AD propagated only one slope → wrong-sign gradient
in the near-resonance regime where MSL stub-notch optimization lives
(verified runs #605, #647 on 2026-05-08).

The custom_jvp bypasses √disc entirely via the implicit-function rule
``dq/dc = q² / (q² − 1)``.  These tests verify the JVP matches central
finite differences on synthetic complex inputs across both branches
and at the previously-broken near-resonance regime.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from rfx.probes.msl_wave_decomp import _solve_q


def _three_probes_from_amplitudes(alpha, gamma, q):
    """Build (v1, v2, v3) consistent with q² − c·q + 1 = 0 by construction.

    Wave decomposition:
        v_n = α · q^(n-1) + γ · q^(-(n-1))
    so v1 = α + γ, v2 = αq + γ/q, v3 = αq² + γ/q².
    """
    v1 = alpha + gamma
    v2 = alpha * q + gamma / q
    v3 = alpha * q * q + gamma / (q * q)
    return v1, v2, v3


def _fd_jvp(f, primal, tangent, *, h=1e-3):
    """Two-sided finite-difference JVP for complex inputs."""
    return (f(primal + h * tangent) - f(primal - h * tangent)) / (2.0 * h)


@pytest.mark.parametrize("theta_deg", [30.0, 60.0, 100.0, 150.0])
def test_solve_q_jvp_matches_fd_lossless(theta_deg):
    """Lossless (|q|=1, near-resonance regime where the old code FAILS).

    Old `_solve_3probe_jax` had wrong-sign AD here; the new custom_jvp
    must match FD to within fp32 noise."""
    theta = math.radians(theta_deg)
    q_true = jnp.exp(-1j * jnp.asarray(theta, dtype=jnp.complex64))
    alpha = jnp.asarray(1.0 + 0.0j, dtype=jnp.complex64)
    gamma = jnp.asarray(0.5 + 0.1j, dtype=jnp.complex64)
    v1, v2, v3 = _three_probes_from_amplitudes(alpha, gamma, q_true)

    # Tangent: perturb v1.real
    tangent = jnp.asarray(1.0 + 0.0j, dtype=jnp.complex64)
    fd = _fd_jvp(lambda x: _solve_q(x, v2, v3), v1, tangent, h=1e-3)
    _, ad = jax.jvp(lambda x: _solve_q(x, v2, v3), (v1,), (tangent,))

    assert jnp.allclose(fd, ad, atol=2e-3, rtol=1e-2), (
        f"theta={theta_deg}°  FD={fd}  AD={ad}  diff={fd - ad}"
    )


@pytest.mark.parametrize("loss", [0.05, 0.10, 0.20])
def test_solve_q_jvp_matches_fd_lossy(loss):
    """Lossy line — both old and new code work here, but verify the
    new JVP doesn't regress."""
    theta = math.radians(70.0)
    q_true = ((1.0 - loss) * jnp.exp(-1j * jnp.asarray(theta))).astype(
        jnp.complex64
    )
    alpha = jnp.asarray(1.0, dtype=jnp.complex64)
    gamma = jnp.asarray(0.3, dtype=jnp.complex64)
    v1, v2, v3 = _three_probes_from_amplitudes(alpha, gamma, q_true)

    tangent = jnp.asarray(1.0 + 0.5j, dtype=jnp.complex64)
    fd = _fd_jvp(lambda x: _solve_q(v1, x, v3), v2, tangent, h=1e-3)
    _, ad = jax.jvp(lambda x: _solve_q(v1, x, v3), (v2,), (tangent,))
    assert jnp.allclose(fd, ad, atol=2e-3, rtol=1e-2), (
        f"loss={loss}  FD={fd}  AD={ad}  diff={fd - ad}"
    )


def test_solve_q_returns_correct_root():
    """Primal forward returns the root whose phase matches v2/v1."""
    q_true = jnp.exp(-1j * jnp.asarray(0.4)).astype(jnp.complex64)
    alpha = jnp.asarray(1.0, dtype=jnp.complex64)
    gamma = jnp.asarray(0.0, dtype=jnp.complex64)  # pure forward wave
    v1, v2, v3 = _three_probes_from_amplitudes(alpha, gamma, q_true)
    q = _solve_q(v1, v2, v3)
    assert jnp.allclose(q, q_true, atol=1e-5), f"got q={q} expected {q_true}"


def test_solve_q_jvp_w_r_t_v3():
    """Tangent through v3 — exercises the third primal slot."""
    theta = math.radians(80.0)
    q_true = jnp.exp(-1j * jnp.asarray(theta)).astype(jnp.complex64)
    alpha = jnp.asarray(1.0, dtype=jnp.complex64)
    gamma = jnp.asarray(0.4, dtype=jnp.complex64)
    v1, v2, v3 = _three_probes_from_amplitudes(alpha, gamma, q_true)

    tangent = jnp.asarray(1.0, dtype=jnp.complex64)
    fd = _fd_jvp(lambda x: _solve_q(v1, v2, x), v3, tangent, h=1e-3)
    _, ad = jax.jvp(lambda x: _solve_q(v1, v2, x), (v3,), (tangent,))
    assert jnp.allclose(fd, ad, atol=2e-3, rtol=1e-2), (
        f"FD={fd}  AD={ad}  diff={fd - ad}"
    )


# ---------------------------------------------------------------- issue: G2 NaN
def _nprobe_cost(g, scale):
    """|S11| from extract_msl_nprobe for a synthetic 2-wave field whose
    backward amplitude is parameterized by the real latent ``g``."""
    from rfx.probes.msl_wave_decomp import extract_msl_nprobe
    delta = 381e-6
    x = jnp.array([0.0, delta, 2.0 * delta], dtype=jnp.float32)
    beta_true = jnp.asarray(213.0, dtype=jnp.float32)   # ~6 GHz MSL class
    alpha = scale * jnp.asarray(1.0 + 0.2j, dtype=jnp.complex64)
    gamma = scale * g * jnp.asarray(0.3 + 0.1j, dtype=jnp.complex64)
    ph = (beta_true * x).astype(jnp.complex64)
    v = (alpha * jnp.exp(-1j * ph) + gamma * jnp.exp(+1j * ph))[None, :]
    i1 = jnp.asarray([scale / 50.0 + 0j], dtype=jnp.complex64)
    res = extract_msl_nprobe(v, x, i1, beta_true[None])
    return jnp.abs(res["s11"][0])


def test_nprobe_grad_finite_and_scale_invariant_at_tiny_v():
    """2026-06-12 G2 regression lock (grad=nan on the Kottke path).

    With |V| ~ 1e-14 (measured on the density-PEC/Kottke path) the
    beta-scan residual curve is numerically flat in float32; the
    parabolic second-difference collapses below the 1e-20 guard and the
    OLD single-where division leaked nan through the backward pass
    (0 * nan = nan).  Lock three properties:

      1. grad is finite at scale 1e-14 (was nan),
      2. grad matches the scale=1.0 grad (s11 and d|s11|/dg are
         scale-invariant by construction),
      3. grad matches central finite differences.
    """
    g0 = 0.7

    grad_tiny = jax.grad(_nprobe_cost)(g0, 1e-14)
    assert jnp.isfinite(grad_tiny), f"grad at scale 1e-14 is {grad_tiny}"

    grad_unit = jax.grad(_nprobe_cost)(g0, 1.0)
    assert jnp.isfinite(grad_unit)
    assert jnp.allclose(grad_tiny, grad_unit, rtol=5e-2), (
        f"scale invariance broken: tiny={grad_tiny} unit={grad_unit}"
    )

    h = 1e-3
    fd = (_nprobe_cost(g0 + h, 1.0) - _nprobe_cost(g0 - h, 1.0)) / (2 * h)
    assert jnp.allclose(grad_unit, fd, rtol=2e-2, atol=1e-4), (
        f"AD={grad_unit} FD={fd}"
    )
