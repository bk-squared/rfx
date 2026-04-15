"""Issue #50: minimize_s11_at_freq — single-frequency S11 proxy for the
differentiable forward() path."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.optimize_objectives import minimize_s11_at_freq


class _FakeResult:
    """Stub with the minimal surface the objective reads."""

    def __init__(self, ts, dt):
        self.time_series = jnp.asarray(ts)
        self.dt = dt


def _cosine_series(freq, dt, n, reflection_gain=0.0, reflection_delay=0):
    """Build a test series where the incident pulse fits inside the
    first 25% window (so the objective's default incident_fraction is
    a reasonable proxy for it) and an optional reflection appears later.
    """
    t = np.arange(n) * dt
    T = t[-1]
    carrier = np.cos(2 * np.pi * freq * t)
    # Incident: Gaussian pulse centred at 10% of T, sigma 4%.
    win = np.exp(-((t - 0.10 * T) / (0.04 * T)) ** 2)
    incident = carrier * win
    if reflection_gain > 0 and reflection_delay > 0:
        refl = np.zeros_like(incident)
        refl[reflection_delay:] = reflection_gain * incident[:-reflection_delay]
        return (incident + refl)[:, None]
    return incident[:, None]


def test_no_reflection_gives_small_s11():
    """Incident-only series → |S11|² should be ~1 (total ≈ incident)."""
    dt = 1e-12
    n = 2000
    ts = _cosine_series(10e9, dt, n)
    r = _FakeResult(ts, dt)
    val = float(minimize_s11_at_freq(10e9, port_probe_idx=0)(r))
    # Without reflection, the windowed pulse peaks before the 25%
    # incident window ends, so total ≈ incident → ratio ≈ 1.
    assert 0.8 <= val <= 1.2, f"expected ~1.0, got {val}"


def test_large_reflection_gives_large_s11():
    dt = 1e-12
    n = 2000
    # Reflection with gain 1.0 and a 1ns delay
    ts = _cosine_series(10e9, dt, n, reflection_gain=1.0,
                        reflection_delay=1000)
    r = _FakeResult(ts, dt)
    val_refl = float(minimize_s11_at_freq(10e9, port_probe_idx=0)(r))
    # With a full-amplitude reflection the total power at 10 GHz should
    # exceed the incident-only case.
    val_norefl = float(minimize_s11_at_freq(10e9, port_probe_idx=0)(
        _FakeResult(_cosine_series(10e9, dt, n), dt)))
    assert val_refl > val_norefl + 0.1, (
        f"reflection should increase |S11|²; got refl={val_refl}, no={val_norefl}"
    )


def test_gradient_is_finite():
    """jax.grad through the full objective must return a finite value."""
    dt = 1e-12
    n = 1000

    def loss(amp):
        t = jnp.arange(n) * dt
        ts = amp * jnp.cos(2 * jnp.pi * 10e9 * t)[:, None]
        r = _FakeResult(ts, dt)
        return minimize_s11_at_freq(10e9, port_probe_idx=0)(r)

    g = float(jax.grad(loss)(jnp.float32(1.0)))
    assert np.isfinite(g), f"grad not finite: {g}"


def test_requires_time_series():
    """Empty time_series must produce a clear ValueError."""
    r = _FakeResult(jnp.zeros((10, 0)), 1e-12)
    with pytest.raises(ValueError, match="emit_time_series"):
        minimize_s11_at_freq(10e9)(r)
