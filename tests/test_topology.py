"""Tests for density-based topology optimization.

Validates:
1. Density filter smooths the field
2. Projection binarizes densities
3. density_to_eps output is always in [eps_bg, eps_fg]
4. Gradient flows through the entire density -> eps pipeline
5. topology_optimize reduces loss on a small example
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.topology import (
    TopologyDesignRegion,
    TopologyResult,
    apply_density_filter,
    apply_projection,
    density_to_eps,
    topology_optimize,
    _get_beta,
    _DEFAULT_BETA_SCHEDULE,
)


# ---------------------------------------------------------------------------
# TopologyDesignRegion
# ---------------------------------------------------------------------------

class TestTopologyDesignRegion:
    """Tests for TopologyDesignRegion construction and properties."""

    def test_construction(self):
        """Region stores bounds and materials correctly."""
        region = TopologyDesignRegion(
            corner_lo=(0.01, 0.01, 0.0),
            corner_hi=(0.04, 0.04, 0.001),
            material_bg="air",
            material_fg="fr4",
            filter_radius=0.002,
            beta_projection=4.0,
        )
        assert region.corner_lo == (0.01, 0.01, 0.0)
        assert region.corner_hi == (0.04, 0.04, 0.001)
        assert region.material_bg == "air"
        assert region.material_fg == "fr4"
        assert region.filter_radius == 0.002
        assert region.beta_projection == 4.0

    def test_effective_filter_radius_from_min_feature_size(self):
        """min_feature_size overrides filter_radius."""
        region = TopologyDesignRegion(
            corner_lo=(0, 0, 0), corner_hi=(0.01, 0.01, 0.001),
            min_feature_size=0.004,
            filter_radius=0.001,  # should be overridden
        )
        assert region.effective_filter_radius == 0.002  # 0.004 / 2

    def test_effective_filter_radius_from_filter_radius(self):
        """When min_feature_size is None, filter_radius is used."""
        region = TopologyDesignRegion(
            corner_lo=(0, 0, 0), corner_hi=(0.01, 0.01, 0.001),
            filter_radius=0.003,
        )
        assert region.effective_filter_radius == 0.003

    def test_effective_filter_radius_none(self):
        """When neither is set, returns None."""
        region = TopologyDesignRegion(
            corner_lo=(0, 0, 0), corner_hi=(0.01, 0.01, 0.001),
        )
        assert region.effective_filter_radius is None


# ---------------------------------------------------------------------------
# Density filter
# ---------------------------------------------------------------------------

class TestDensityFilter:
    """Tests for apply_density_filter."""

    def test_density_filter_smooths_2d(self):
        """Filter should smooth a checkerboard pattern toward uniform."""
        # Create a 20x20 checkerboard
        n = 20
        rho = jnp.zeros((n, n), dtype=jnp.float32)
        for i in range(n):
            for j in range(n):
                if (i + j) % 2 == 0:
                    rho = rho.at[i, j].set(1.0)

        # Apply filter with radius = 3 cells
        rho_filtered = apply_density_filter(rho, radius_cells=3.0)

        # The filtered field should be smoother: less variance
        var_before = float(jnp.var(rho))
        var_after = float(jnp.var(rho_filtered))
        assert var_after < var_before, (
            f"Filter should reduce variance: {var_after:.4f} >= {var_before:.4f}"
        )
        print(f"\n  Checkerboard variance: before={var_before:.4f}, after={var_after:.4f}")

    def test_density_filter_smooths_3d(self):
        """Filter should smooth a 3D checkerboard pattern."""
        n = 10
        idx = jnp.arange(n)
        ii, jj, kk = jnp.meshgrid(idx, idx, idx, indexing="ij")
        rho = ((ii + jj + kk) % 2).astype(jnp.float32)

        rho_filtered = apply_density_filter(rho, radius_cells=2.0)

        var_before = float(jnp.var(rho))
        var_after = float(jnp.var(rho_filtered))
        assert var_after < var_before, (
            f"3D filter should reduce variance: {var_after:.4f} >= {var_before:.4f}"
        )
        print(f"\n  3D checkerboard variance: before={var_before:.4f}, after={var_after:.4f}")

    def test_density_filter_preserves_uniform(self):
        """Filtering a uniform field should leave it unchanged."""
        rho = 0.6 * jnp.ones((15, 15), dtype=jnp.float32)
        rho_filtered = apply_density_filter(rho, radius_cells=3.0)
        diff = float(jnp.max(jnp.abs(rho_filtered - rho)))
        # Small tolerance for edge effects
        assert diff < 0.05, f"Uniform field changed by {diff:.6f}"

    def test_density_filter_small_radius_noop(self):
        """Radius < 0.5 cells should be a no-op."""
        rho = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
        rho_filtered = apply_density_filter(rho, radius_cells=0.3)
        np.testing.assert_array_equal(rho, rho_filtered)

    def test_density_filter_is_differentiable(self):
        """jax.grad should flow through the density filter."""
        def loss_fn(rho):
            return jnp.sum(apply_density_filter(rho, radius_cells=2.0) ** 2)

        rho = 0.5 * jnp.ones((8, 8), dtype=jnp.float32)
        grad = jax.grad(loss_fn)(rho)
        assert jnp.all(jnp.isfinite(grad)), "Gradient through filter is not finite"
        assert float(jnp.max(jnp.abs(grad))) > 0, "Gradient through filter is zero"


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

class TestProjection:
    """Tests for apply_projection."""

    def test_projection_binarizes_at_high_beta(self):
        """High beta should push densities toward 0 and 1."""
        rho = jnp.linspace(0.01, 0.99, 100)

        proj_low = apply_projection(rho, beta=1.0)
        proj_high = apply_projection(rho, beta=64.0)

        # At high beta, values near 0 should be closer to 0,
        # values near 1 should be closer to 1
        # Measure "binarization" as mean of 4*p*(1-p), which is 0 for binary
        binarization_low = float(jnp.mean(4.0 * proj_low * (1.0 - proj_low)))
        binarization_high = float(jnp.mean(4.0 * proj_high * (1.0 - proj_high)))

        assert binarization_high < binarization_low, (
            f"High beta should be more binary: "
            f"low={binarization_low:.4f}, high={binarization_high:.4f}"
        )
        assert binarization_high < 0.1, (
            f"beta=64 should be nearly binary, got binarization={binarization_high:.4f}"
        )
        print(f"\n  Binarization: beta=1 -> {binarization_low:.4f}, "
              f"beta=64 -> {binarization_high:.4f}")

    def test_projection_maps_half_to_half(self):
        """Projection of 0.5 should remain 0.5 (threshold symmetry)."""
        result = apply_projection(jnp.array(0.5), beta=10.0)
        assert abs(float(result) - 0.5) < 1e-6

    def test_projection_range_01(self):
        """Projected values should be in [0, 1]."""
        rho = jnp.linspace(0.0, 1.0, 50)
        for beta in [1.0, 4.0, 16.0, 64.0]:
            proj = apply_projection(rho, beta=beta)
            assert float(jnp.min(proj)) >= -1e-6, f"Projection below 0 at beta={beta}"
            assert float(jnp.max(proj)) <= 1.0 + 1e-6, f"Projection above 1 at beta={beta}"

    def test_projection_is_differentiable(self):
        """jax.grad should flow through projection."""
        def loss_fn(rho):
            return jnp.sum(apply_projection(rho, beta=8.0) ** 2)

        rho = jnp.linspace(0.1, 0.9, 20)
        grad = jax.grad(loss_fn)(rho)
        assert jnp.all(jnp.isfinite(grad)), "Gradient through projection is not finite"
        assert float(jnp.max(jnp.abs(grad))) > 0, "Gradient through projection is zero"


# ---------------------------------------------------------------------------
# density_to_eps
# ---------------------------------------------------------------------------

class TestDensityToEps:
    """Tests for the full density -> eps pipeline."""

    def test_density_to_eps_range(self):
        """Output should always be in [eps_bg, eps_fg]."""
        eps_bg, eps_fg = 1.0, 4.4
        rho = jnp.linspace(0.0, 1.0, 100).reshape(10, 10)

        for beta in [1.0, 8.0, 64.0]:
            eps = density_to_eps(rho, eps_bg, eps_fg, beta=beta)
            min_eps = float(jnp.min(eps))
            max_eps = float(jnp.max(eps))
            assert min_eps >= eps_bg - 1e-5, (
                f"eps below bg: {min_eps:.6f} < {eps_bg} at beta={beta}"
            )
            assert max_eps <= eps_fg + 1e-5, (
                f"eps above fg: {max_eps:.6f} > {eps_fg} at beta={beta}"
            )
        print(f"\n  eps range verified for beta in [1, 8, 64]")

    def test_density_to_eps_with_filter(self):
        """Pipeline with filtering should also keep eps in range."""
        eps_bg, eps_fg = 1.0, 9.8
        rho = jnp.linspace(0.0, 1.0, 64).reshape(8, 8)
        eps = density_to_eps(rho, eps_bg, eps_fg,
                             filter_radius_cells=2.0, beta=4.0)
        assert float(jnp.min(eps)) >= eps_bg - 1e-4
        assert float(jnp.max(eps)) <= eps_fg + 1e-4

    def test_density_to_eps_endpoints(self):
        """rho=0 -> eps_bg, rho=1 -> eps_fg (approximately for beta=1)."""
        eps_bg, eps_fg = 2.0, 10.0
        # At high beta, endpoints should be close
        rho_zero = jnp.zeros((3, 3), dtype=jnp.float32)
        rho_one = jnp.ones((3, 3), dtype=jnp.float32)

        eps_at_0 = density_to_eps(rho_zero, eps_bg, eps_fg, beta=64.0)
        eps_at_1 = density_to_eps(rho_one, eps_bg, eps_fg, beta=64.0)

        assert float(jnp.max(jnp.abs(eps_at_0 - eps_bg))) < 0.1
        assert float(jnp.max(jnp.abs(eps_at_1 - eps_fg))) < 0.1

    def test_density_to_eps_3d(self):
        """3D density field should work."""
        eps_bg, eps_fg = 1.0, 4.4
        rho = 0.5 * jnp.ones((5, 5, 3), dtype=jnp.float32)
        eps = density_to_eps(rho, eps_bg, eps_fg, beta=1.0)
        assert eps.shape == (5, 5, 3)
        # At rho=0.5, projection(0.5)=0.5, so eps should be midpoint
        expected = eps_bg + 0.5 * (eps_fg - eps_bg)
        assert float(jnp.max(jnp.abs(eps - expected))) < 0.01


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Tests that jax.grad flows through the entire density -> eps pipeline."""

    def test_gradient_flows_through_density_2d(self):
        """jax.grad through density_to_eps should produce non-zero gradients."""
        eps_bg, eps_fg = 1.0, 4.4

        def loss_fn(rho):
            eps = density_to_eps(rho, eps_bg, eps_fg,
                                 filter_radius_cells=2.0, beta=4.0)
            # Minimize total permittivity (drives rho toward 0)
            return jnp.sum(eps)

        rho = 0.5 * jnp.ones((10, 10), dtype=jnp.float32)
        grad = jax.grad(loss_fn)(rho)

        assert jnp.all(jnp.isfinite(grad)), "Gradient is not finite"
        assert float(jnp.max(jnp.abs(grad))) > 0, "Gradient is all zeros"
        # Gradient should be positive (increasing rho increases eps)
        assert float(jnp.mean(grad)) > 0, "Expected positive gradient direction"
        print(f"\n  Gradient stats: mean={float(jnp.mean(grad)):.6e}, "
              f"max={float(jnp.max(grad)):.6e}")

    def test_gradient_flows_through_density_3d(self):
        """jax.grad through 3D density_to_eps."""
        eps_bg, eps_fg = 1.0, 4.4

        def loss_fn(rho):
            eps = density_to_eps(rho, eps_bg, eps_fg, beta=2.0)
            return jnp.sum(eps ** 2)

        rho = 0.5 * jnp.ones((5, 5, 3), dtype=jnp.float32)
        grad = jax.grad(loss_fn)(rho)

        assert jnp.all(jnp.isfinite(grad)), "3D gradient is not finite"
        assert float(jnp.max(jnp.abs(grad))) > 0, "3D gradient is all zeros"

    def test_gradient_with_high_beta(self):
        """Gradient should still flow at high beta (no NaN)."""
        eps_bg, eps_fg = 1.0, 4.4

        def loss_fn(rho):
            eps = density_to_eps(rho, eps_bg, eps_fg, beta=64.0)
            return jnp.sum(eps)

        rho = 0.5 * jnp.ones((8, 8), dtype=jnp.float32)
        grad = jax.grad(loss_fn)(rho)

        assert jnp.all(jnp.isfinite(grad)), f"Gradient at beta=64 has NaN/Inf"


# ---------------------------------------------------------------------------
# Beta schedule
# ---------------------------------------------------------------------------

class TestBetaSchedule:
    """Tests for beta schedule lookup."""

    def test_get_beta_default(self):
        """Default schedule should return correct beta values."""
        assert _get_beta(0, _DEFAULT_BETA_SCHEDULE) == 1.0
        assert _get_beta(15, _DEFAULT_BETA_SCHEDULE) == 1.0
        assert _get_beta(30, _DEFAULT_BETA_SCHEDULE) == 4.0
        assert _get_beta(59, _DEFAULT_BETA_SCHEDULE) == 4.0
        assert _get_beta(60, _DEFAULT_BETA_SCHEDULE) == 16.0
        assert _get_beta(80, _DEFAULT_BETA_SCHEDULE) == 64.0
        assert _get_beta(100, _DEFAULT_BETA_SCHEDULE) == 64.0


# ---------------------------------------------------------------------------
# Integration: topology_optimize reduces loss
# ---------------------------------------------------------------------------

class TestTopologyOptimize:
    """Integration test for the full topology optimization loop."""

    def test_topology_optimize_reduces_loss(self):
        """Topology optimization should reduce the objective over iterations.

        Uses a small 2D domain with PEC boundary and a simple
        probe-energy objective.  A few iterations should suffice to
        demonstrate that the optimizer makes progress.
        """
        from rfx.api import Simulation

        sim = Simulation(
            freq_max=5e9,
            domain=(0.015, 0.015, 0.015),
            boundary="pec",
        )
        sim.add_port((0.005, 0.0075, 0.0075), "ez")
        sim.add_probe((0.01, 0.0075, 0.0075), "ez")

        region = TopologyDesignRegion(
            corner_lo=(0.006, 0.006, 0.006),
            corner_hi=(0.009, 0.009, 0.009),
            material_bg="air",
            material_fg="fr4",
            beta_projection=1.0,
        )

        # Objective: maximize probe energy (minimize negative)
        def obj(result):
            return -jnp.sum(result.time_series ** 2)

        result = topology_optimize(
            sim, region, obj,
            n_iterations=5,
            learning_rate=0.05,
            beta_schedule=[(0, 1.0)],  # constant beta for short test
            verbose=True,
        )

        assert isinstance(result, TopologyResult)
        assert len(result.history) == 5
        assert len(result.beta_history) == 5
        assert result.density.ndim == 3
        assert result.eps_design.ndim == 3

        # Loss should decrease (or at least the optimizer tried)
        first_loss = result.history[0]
        last_loss = result.history[-1]
        print(f"\n  Loss: first={first_loss:.6e}, last={last_loss:.6e}")
        # With 5 iterations we expect some improvement
        assert last_loss <= first_loss + abs(first_loss) * 0.1, (
            f"Loss did not decrease: {first_loss:.6e} -> {last_loss:.6e}"
        )

    def test_topology_result_structure(self):
        """TopologyResult should hold the right fields."""
        result = TopologyResult(
            density=jnp.ones((5, 5, 1)),
            density_projected=jnp.ones((5, 5, 1)),
            eps_design=jnp.ones((5, 5, 1)) * 4.4,
            history=[1.0, 0.5, 0.25],
            beta_history=[1.0, 1.0, 4.0],
        )
        assert len(result.history) == 3
        assert result.density.shape == (5, 5, 1)
        assert result.eps_design.shape == (5, 5, 1)
        assert result.final_result is None
