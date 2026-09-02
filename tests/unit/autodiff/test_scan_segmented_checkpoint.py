"""Issue #73: segmented-checkpoint scan path on Simulation.forward().

The legacy ``checkpoint=True`` setting wraps the per-step body with
``jax.checkpoint`` but the outer ``jax.lax.scan`` still keeps every
step's full carry, so peak memory grows linearly with ``n_steps``.
This blocks #72's wave-decomposition |S11| objective on high-Q antennas
(e.g. ex1_dra DRA at 5 GHz) because the DFT integration window the
formula requires is ≥ a few thousand steps, which OOMs on a single
A6000.

The new ``checkpoint_segments=K`` API splits the n_steps scan into K
segments of size ``s = n_steps // K`` and rematerialises each segment
as a unit, dropping peak memory from ``O(n_steps · |carry|)`` to
``O((K + s) · |carry|)`` ≈ ``O(√n_steps · |carry|)`` for ``s ≈ √n_steps``.

These tests assert two properties that matter for downstream users:

1. **Forward equivalence** — ``forward(checkpoint_segments=K)`` returns
   the same time series, S-parameters, and (where applicable) NTFF data
   as the unsegmented forward, to numerical tolerance.
2. **Gradient equivalence** — ``jax.grad`` through the segmented path
   matches the unsegmented gradient.

A separate test verifies that a divisibility error is reported when
``checkpoint_segments`` does not divide ``n_steps`` evenly (we reject
padding rather than mutate the DFT integration window).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation, GaussianPulse
from rfx.optimize_objectives import minimize_s11_at_freq_wave_decomp
from rfx.simulation import _nearest_divisor, _suggest_checkpoint_segments


def _build_cavity():
    a, b, d = 0.05, 0.05, 0.025
    sim = Simulation(
        freq_max=5e9,
        domain=(a, b, d),
        dx=2.5e-3,
        boundary="pec",
    )
    sim.add_port(
        position=(a / 2, b / 2, d / 2),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3e9, bandwidth=0.8, amplitude=1.0),
    )
    return sim


def test_helpers():
    assert _nearest_divisor(800, 28) == 25  # divisors near 28: 25 (diff 3), 32 (diff 4) → 25 wins
    assert _nearest_divisor(3000, 55) == 50
    assert _nearest_divisor(100, 10) == 10
    # Pure prime n: only 1 and n divide it.
    assert _nearest_divisor(13, 3) == 1
    # Suggest is roughly √n.
    assert 20 <= _suggest_checkpoint_segments(800) <= 40
    assert 40 <= _suggest_checkpoint_segments(3000) <= 80


def test_segmented_forward_matches_unsegmented():
    """Time series + s_params with checkpoint_segments=K should match
    the unsegmented run on the same n_steps, to float32 tolerance."""
    sim = _build_cavity()
    n_steps = 200  # small enough for a CPU pytest run
    K = 10
    assert n_steps % K == 0
    freqs = jnp.linspace(2e9, 4e9, 5, dtype=jnp.float32)

    res_legacy = sim.forward(
        n_steps=n_steps, port_s11_freqs=freqs, skip_preflight=True,
    )
    res_segmented = sim.forward(
        n_steps=n_steps, port_s11_freqs=freqs, skip_preflight=True,
        checkpoint_segments=K,
    )

    # Time series shape should be (n_steps, n_probes).
    assert res_legacy.time_series.shape == res_segmented.time_series.shape
    # Float32 bit-exactness is too strict; use 1e-5 atol for the
    # potential reassociation by XLA when reshaping into segments.
    np.testing.assert_allclose(
        np.asarray(res_segmented.time_series),
        np.asarray(res_legacy.time_series),
        atol=1e-5, rtol=1e-4,
        err_msg="time_series differs between segmented and unsegmented",
    )
    np.testing.assert_allclose(
        np.asarray(res_segmented.s_params),
        np.asarray(res_legacy.s_params),
        atol=1e-5, rtol=1e-4,
        err_msg="s_params differs between segmented and unsegmented",
    )


def test_segmented_gradient_matches_unsegmented():
    """``jax.grad`` of an S11 objective wrt eps_override must match
    between segmented and unsegmented forward paths."""
    sim = _build_cavity()
    grid = sim._build_grid()
    n_steps = 200
    K = 10
    freqs = jnp.linspace(2e9, 4e9, 3, dtype=jnp.float32)
    obj = minimize_s11_at_freq_wave_decomp(target_freq=3.0e9, port_idx=0)

    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)

    def loss_legacy(eps):
        res = sim.forward(
            eps_override=eps, n_steps=n_steps, port_s11_freqs=freqs,
            skip_preflight=True,
        )
        return obj(res)

    def loss_segmented(eps):
        res = sim.forward(
            eps_override=eps, n_steps=n_steps, port_s11_freqs=freqs,
            skip_preflight=True, checkpoint_segments=K,
        )
        return obj(res)

    g_legacy = np.asarray(jax.grad(loss_legacy)(eps_base))
    g_segmented = np.asarray(jax.grad(loss_segmented)(eps_base))
    assert np.all(np.isfinite(g_segmented)), "segmented grad has NaN/Inf"
    # Allow a slightly looser tolerance on gradients than on outputs.
    np.testing.assert_allclose(
        g_segmented, g_legacy,
        atol=5e-5, rtol=1e-3,
        err_msg="gradient differs between segmented and unsegmented",
    )


def test_segmented_rejects_non_divisor():
    """When K does not divide n_steps, surface a clear ValueError
    pointing the user at a divisor.  Padding is intentionally rejected."""
    sim = _build_cavity()
    with pytest.raises(ValueError, match="does not divide n_steps"):
        sim.forward(n_steps=200, checkpoint_segments=7,
                    skip_preflight=True)


def test_segmented_rejects_non_positive():
    sim = _build_cavity()
    with pytest.raises(ValueError, match="must be ≥ 1"):
        sim.forward(n_steps=200, checkpoint_segments=0,
                    skip_preflight=True)


def test_segmented_rejects_nonuniform_path():
    """Issue #73: NU forward path is not yet wired for segmented remat.
    Reject loudly so the user does not get a silent fall-back to the
    linear-memory scan that this kwarg was meant to fix."""
    a, b, d = 0.05, 0.05, 0.025
    dx = 2.5e-3
    nx = int(round(a / dx))
    ny = int(round(b / dx))
    nz = int(round(d / dx))
    sim = Simulation(
        freq_max=5e9,
        domain=(0, 0, 0),
        dx=dx,
        dx_profile=np.full(nx, dx),
        dy_profile=np.full(ny, dx),
        dz_profile=np.full(nz, dx),
        boundary="pec",
    )
    sim.add_port(
        position=(a / 2, b / 2, d / 2),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3e9, bandwidth=0.8, amplitude=1.0),
    )
    with pytest.raises(NotImplementedError, match="uniform single-device"):
        sim.forward(n_steps=200, checkpoint_segments=10,
                    skip_preflight=True)
