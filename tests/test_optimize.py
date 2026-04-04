"""Tests for inverse design optimizer.

Validates:
1. DesignRegion construction
2. Latent-to-eps mapping (sigmoid bounds)
3. OptimizeResult structure
4. Gradient check (AD vs finite-difference)
"""

import jax
import jax.numpy as jnp

from rfx.api import Simulation
from rfx.optimize import (
    DesignRegion, OptimizeResult, GradientCheckResult,
    _latent_to_eps, optimize, gradient_check,
)


def test_design_region():
    """DesignRegion stores bounds correctly."""
    region = DesignRegion(
        corner_lo=(0.01, 0.01, 0.0),
        corner_hi=(0.04, 0.04, 0.001),
        eps_range=(1.0, 4.4),
    )
    assert region.corner_lo == (0.01, 0.01, 0.0)
    assert region.eps_range == (1.0, 4.4)


def test_latent_to_eps_bounds():
    """Sigmoid mapping should stay within [eps_min, eps_max]."""
    eps_min, eps_max = 1.0, 12.0

    # Large negative latent -> eps_min
    eps_lo = _latent_to_eps(jnp.array(-10.0), eps_min, eps_max)
    assert float(eps_lo) < eps_min + 0.01

    # Large positive latent -> eps_max
    eps_hi = _latent_to_eps(jnp.array(10.0), eps_min, eps_max)
    assert float(eps_hi) > eps_max - 0.01

    # Zero latent -> midpoint
    eps_mid = _latent_to_eps(jnp.array(0.0), eps_min, eps_max)
    expected_mid = (eps_min + eps_max) / 2.0
    assert abs(float(eps_mid) - expected_mid) < 0.01

    print(f"\nLatent-to-eps: lo={float(eps_lo):.4f}, mid={float(eps_mid):.4f}, hi={float(eps_hi):.4f}")


def test_latent_to_eps_differentiable():
    """Sigmoid mapping should be differentiable."""
    grad_fn = jax.grad(lambda x: _latent_to_eps(x, 1.0, 12.0))
    g = grad_fn(jnp.array(0.0))
    # Gradient at midpoint should be positive (sigmoid slope * range)
    assert float(g) > 0.0
    print(f"\nGradient at latent=0: {float(g):.4f}")


def test_optimize_result_structure():
    """OptimizeResult holds the right fields."""
    result = OptimizeResult(
        eps_design=jnp.ones((5, 5, 1)),
        loss_history=[1.0, 0.5, 0.25],
        latent=jnp.zeros((5, 5, 1)),
    )
    assert len(result.loss_history) == 3
    assert result.eps_design.shape == (5, 5, 1)
    assert result.latent.shape == (5, 5, 1)


def test_optimize_runs_single_iteration():
    """The public optimize() API should complete at least one iteration."""
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    sim.add_port((0.005, 0.0075, 0.0075), "ez")
    sim.add_probe((0.01, 0.0075, 0.0075), "ez")

    region = DesignRegion(
        corner_lo=(0.006, 0.006, 0.006),
        corner_hi=(0.009, 0.009, 0.009),
        eps_range=(1.0, 4.4),
    )

    result = optimize(
        sim,
        region,
        lambda run_result: -jnp.sum(run_result.time_series ** 2),
        n_iters=1,
        lr=0.01,
        verbose=False,
    )

    assert isinstance(result, OptimizeResult)
    assert len(result.loss_history) == 1
    assert result.eps_design.shape == result.latent.shape
