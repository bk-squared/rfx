"""Physical-correctness tests for minimize_s11_at_freq.

The existing unit test (tests/unit/autodiff/test_s11_at_freq.py) asserts that the
objective returns ~1.0 for an incident-only series. That value reveals
the current implementation computes |1 + S11|^2 (wave-sum ratio), not
|S11|^2. For a perfect match (S11 = 0) the objective should return 0.0,
not 1.0. For a perfect anti-phase reflection (S11 = -1) it should return
1.0, not 0.0.

These physical-correctness tests encode the true semantics. They fail
on the current implementation and will pass only once the objective
uses wave-decomposition S11 extracted from lumped-port V/I DFTs.

Fix branch: fix/lumped-port-s11-at-freq
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.optimize_objectives import minimize_s11_at_freq


class _FakeResult:
    def __init__(self, ts, dt):
        self.time_series = jnp.asarray(ts)
        self.dt = dt


def _gaussian_pulse(f0, dt, n, *, t0_frac=0.1, sigma_frac=0.04):
    t = np.arange(n) * dt
    T = t[-1]
    env = np.exp(-((t - t0_frac * T) / (sigma_frac * T)) ** 2)
    return env * np.cos(2 * np.pi * f0 * t)


def test_perfect_match_returns_zero():
    """S11 = 0 (no reflection) must give |S11|^2 ≈ 0, not ≈ 1."""
    dt = 1e-12
    n = 4000
    ts = _gaussian_pulse(10e9, dt, n)[:, None]
    r = _FakeResult(ts, dt)
    val = float(minimize_s11_at_freq(10e9)(r))
    assert val < 0.05, f"perfect match should give |S11|^2 ~ 0, got {val:.4f}"


def test_perfect_short_returns_one():
    """S11 = -1 (anti-phase reflection) must give |S11|^2 = 1, not 0."""
    dt = 1e-12
    n = 4000
    inc = _gaussian_pulse(10e9, dt, n)
    # Anti-phase reflection, arriving after incident pulse has passed.
    refl = np.zeros_like(inc)
    delay = n // 4
    refl[delay:] = -inc[:-delay]
    r = _FakeResult((inc + refl)[:, None], dt)
    val = float(minimize_s11_at_freq(10e9)(r))
    assert 0.8 <= val <= 1.2, (
        f"perfect short (S11 = -1) should give |S11|^2 ~ 1, got {val:.4f}"
    )


def test_half_amplitude_reflection_returns_quarter():
    """S11 = 0.5 should give |S11|^2 = 0.25. Current impl gives |1.5|^2 = 2.25."""
    dt = 1e-12
    n = 4000
    inc = _gaussian_pulse(10e9, dt, n)
    refl = np.zeros_like(inc)
    delay = n // 4
    refl[delay:] = 0.5 * inc[:-delay]
    r = _FakeResult((inc + refl)[:, None], dt)
    val = float(minimize_s11_at_freq(10e9)(r))
    assert 0.15 <= val <= 0.4, (
        f"S11 = 0.5 should give |S11|^2 ~ 0.25, got {val:.4f} "
        f"(buggy impl returns |1+S11|^2 ~ 2.25)"
    )


def test_gradient_finite():
    """Whatever the formula, jax.grad must produce a finite value."""
    dt = 1e-12
    n = 1000

    def loss(amp):
        t = jnp.arange(n) * dt
        ts = (amp * jnp.cos(2 * jnp.pi * 10e9 * t))[:, None]
        r = _FakeResult(ts, dt)
        return minimize_s11_at_freq(10e9)(r)

    g = float(jax.grad(loss)(jnp.float32(1.0)))
    assert np.isfinite(g), f"grad not finite: {g}"
