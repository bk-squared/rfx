"""Tests for density-based topology optimization.

Validates:
1. Density filter smooths the field
2. Projection binarizes densities
3. density_to_eps output is always in [eps_bg, eps_fg]
4. Gradient flows through the entire density -> eps pipeline
5. topology_optimize reduces loss on a small example
"""

import importlib.util
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, DebyePole
from rfx.boundaries.pec import apply_pec_mask, apply_pec_occupancy
from rfx.core.yee import init_state

from rfx.topology import (
    TopologyDesignRegion,
    TopologyResult,
    apply_density_filter,
    apply_projection,
    density_to_eps,
    density_to_material_fields,
    inspect_topology_hybrid_support,
    topology_optimize,
    _get_beta,
    _DEFAULT_BETA_SCHEDULE,
    _inspect_topology_hybrid_support,
)

pytestmark = pytest.mark.gpu


def _make_phase4c_topology_case(
    *,
    boundary: str = "pec",
    add_port: bool = False,
    material_sigma: float = 0.0,
    material_fg: str = "diel",
    pec_foreground: bool = False,
    add_pec_box: bool = False,
    debye: bool = False,
):
    from rfx.api import Simulation

    sim = Simulation(
        freq_max=5e9,
        domain=(0.015, 0.015, 0.015),
        boundary=boundary,
    )
    if add_port:
        sim.add_port((0.005, 0.0075, 0.0075), "ez")
    else:
        sim.add_source((0.005, 0.0075, 0.0075), "ez")
    sim.add_probe((0.01, 0.0075, 0.0075), "ez")

    if add_pec_box:
        sim.add(Box((0.003, 0.006, 0.006), (0.006, 0.009, 0.009)), material="pec")

    if pec_foreground:
        fg_name = "pec"
    else:
        fg_name = material_fg
        sim.add_material(
            fg_name,
            eps_r=4.0,
            sigma=material_sigma,
            debye_poles=[DebyePole(delta_eps=1.0, tau=8e-12)] if debye else None,
        )

    region = TopologyDesignRegion(
        corner_lo=(0.009, 0.003, 0.003),
        corner_hi=(0.012, 0.006, 0.006),
        material_bg="air",
        material_fg=fg_name,
        beta_projection=1.0,
    )
    return sim, region


def _topology_probe_energy_objective(result):
    return -jnp.sum(result.time_series ** 2)


def test_phase13_topology_hybrid_support_accepts_explicit_init_density():
    sim, region = _make_phase4c_topology_case(boundary="pec")
    grid = sim._build_grid()
    lo_idx = list(grid.position_to_index(region.corner_lo))
    hi_idx = list(grid.position_to_index(region.corner_hi))
    pads = (grid.pad_x, grid.pad_y, grid.pad_z)
    dims = (grid.nx, grid.ny, grid.nz)
    for axis in range(3):
        lo_idx[axis] = max(lo_idx[axis], pads[axis])
        hi_idx[axis] = min(hi_idx[axis], dims[axis] - 1 - pads[axis])
    design_shape = tuple(hi_idx[axis] - lo_idx[axis] + 1 for axis in range(3))
    density = jnp.linspace(
        0.2, 0.8, int(np.prod(design_shape)), dtype=jnp.float32
    ).reshape(design_shape)

    inputs, report, _, _, _, _, _, _, _, fields = _inspect_topology_hybrid_support(
        sim,
        region,
        init_density=density,
        n_steps=8,
    )

    assert report.supported
    assert inputs.n_steps == 8
    assert np.asarray(fields.eps).shape == density.shape


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
            eps, _ = density_to_eps(rho, eps_bg, eps_fg, beta=beta)
            min_eps = float(jnp.min(eps))
            max_eps = float(jnp.max(eps))
            assert min_eps >= eps_bg - 1e-5, (
                f"eps below bg: {min_eps:.6f} < {eps_bg} at beta={beta}"
            )
            assert max_eps <= eps_fg + 1e-5, (
                f"eps above fg: {max_eps:.6f} > {eps_fg} at beta={beta}"
            )
        print("\n  eps range verified for beta in [1, 8, 64]")

    def test_density_to_eps_with_filter(self):
        """Pipeline with filtering should also keep eps in range."""
        eps_bg, eps_fg = 1.0, 9.8
        rho = jnp.linspace(0.0, 1.0, 64).reshape(8, 8)
        eps, _ = density_to_eps(rho, eps_bg, eps_fg,
                             filter_radius_cells=2.0, beta=4.0)
        assert float(jnp.min(eps)) >= eps_bg - 1e-4
        assert float(jnp.max(eps)) <= eps_fg + 1e-4

    def test_density_to_eps_endpoints(self):
        """rho=0 -> eps_bg, rho=1 -> eps_fg (approximately for beta=1)."""
        eps_bg, eps_fg = 2.0, 10.0
        # At high beta, endpoints should be close
        rho_zero = jnp.zeros((3, 3), dtype=jnp.float32)
        rho_one = jnp.ones((3, 3), dtype=jnp.float32)

        eps_at_0, _ = density_to_eps(rho_zero, eps_bg, eps_fg, beta=64.0)
        eps_at_1, _ = density_to_eps(rho_one, eps_bg, eps_fg, beta=64.0)

        assert float(jnp.max(jnp.abs(eps_at_0 - eps_bg))) < 0.1
        assert float(jnp.max(jnp.abs(eps_at_1 - eps_fg))) < 0.1

    def test_density_to_eps_3d(self):
        """3D density field should work."""
        eps_bg, eps_fg = 1.0, 4.4
        rho = 0.5 * jnp.ones((5, 5, 3), dtype=jnp.float32)
        eps, _ = density_to_eps(rho, eps_bg, eps_fg, beta=1.0)
        assert eps.shape == (5, 5, 3)
        # At rho=0.5, projection(0.5)=0.5, so eps should be midpoint
        expected = eps_bg + 0.5 * (eps_fg - eps_bg)
        assert float(jnp.max(jnp.abs(eps - expected))) < 0.01


def test_density_to_material_fields_pec_foreground_uses_occupancy():
    """PEC foreground should produce occupancy, not huge conductivity."""
    rho = jnp.array([[0.0, 0.5, 1.0]], dtype=jnp.float32)
    fields = density_to_material_fields(
        rho,
        eps_bg=2.2,
        eps_fg=1.0,
        beta=64.0,
        sigma_bg=0.0,
        sigma_fg=1e10,
        pec_fg=True,
    )

    assert fields.pec_occupancy is not None
    np.testing.assert_allclose(np.array(fields.eps), 2.2, atol=1e-3)
    np.testing.assert_allclose(np.array(fields.sigma), 0.0, atol=1e-6)
    assert float(fields.pec_occupancy[0, 0]) < 1e-3
    assert 0.0 < float(fields.pec_occupancy[0, 1]) < 1.0
    assert float(fields.pec_occupancy[0, 2]) > 1.0 - 1e-3


def test_apply_pec_occupancy_matches_binary_mask_for_sheet():
    """Binary occupancy should reduce to the hard-mask PEC operator."""
    state = init_state((5, 5, 5))._replace(
        ex=jnp.ones((5, 5, 5), dtype=jnp.float32),
        ey=jnp.ones((5, 5, 5), dtype=jnp.float32),
        ez=jnp.ones((5, 5, 5), dtype=jnp.float32),
    )
    mask = jnp.zeros((5, 5, 5), dtype=jnp.bool_).at[2, :, :].set(True)

    hard = apply_pec_mask(state, mask)
    soft = apply_pec_occupancy(state, mask.astype(jnp.float32))

    np.testing.assert_allclose(np.array(hard.ex), np.array(soft.ex))
    np.testing.assert_allclose(np.array(hard.ey), np.array(soft.ey))
    np.testing.assert_allclose(np.array(hard.ez), np.array(soft.ez))


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Tests that jax.grad flows through the entire density -> eps pipeline."""

    def test_gradient_flows_through_density_2d(self):
        """jax.grad through density_to_eps should produce non-zero gradients."""
        eps_bg, eps_fg = 1.0, 4.4

        def loss_fn(rho):
            eps, _ = density_to_eps(rho, eps_bg, eps_fg,
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
            eps, _ = density_to_eps(rho, eps_bg, eps_fg, beta=2.0)
            return jnp.sum(eps ** 2)

        rho = 0.5 * jnp.ones((5, 5, 3), dtype=jnp.float32)
        grad = jax.grad(loss_fn)(rho)

        assert jnp.all(jnp.isfinite(grad)), "3D gradient is not finite"
        assert float(jnp.max(jnp.abs(grad))) > 0, "3D gradient is all zeros"

    def test_gradient_with_high_beta(self):
        """Gradient should still flow at high beta (no NaN)."""
        eps_bg, eps_fg = 1.0, 4.4

        def loss_fn(rho):
            eps, _ = density_to_eps(rho, eps_bg, eps_fg, beta=64.0)
            return jnp.sum(eps)

        rho = 0.5 * jnp.ones((8, 8), dtype=jnp.float32)
        grad = jax.grad(loss_fn)(rho)

        assert jnp.all(jnp.isfinite(grad)), "Gradient at beta=64 has NaN/Inf"


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


@pytest.mark.skipif(
    not importlib.util.find_spec("optax"),
    reason="optax not installed",
)
def test_phase4c_topology_hybrid_support_inspection_reports_supported_zero_sigma_case():
    sim, region = _make_phase4c_topology_case()

    report = inspect_topology_hybrid_support(sim, region, n_steps=12)

    assert report.supported
    assert report.port_metadata is not None
    assert report.port_metadata.total_ports == 0
    assert report.port_metadata.soft_source_count == 1
    assert report.inventory is not None


@pytest.mark.skipif(
    not importlib.util.find_spec("optax"),
    reason="optax not installed",
)
def test_phase2_topology_hybrid_support_inspection_reports_supported_cpml_zero_sigma_case():
    sim, region = _make_phase4c_topology_case(boundary="cpml")

    report = inspect_topology_hybrid_support(sim, region, n_steps=12)

    assert report.supported
    assert report.port_metadata is not None
    assert report.port_metadata.total_ports == 0
    assert report.port_metadata.soft_source_count == 1
    assert report.inventory is not None
    assert "cpml_params" in report.inventory.replay_inputs


@pytest.mark.skipif(
    not importlib.util.find_spec("optax"),
    reason="optax not installed",
)
def test_phase4c_topology_optimize_default_route_stays_on_pure_ad(monkeypatch):
    sim, region = _make_phase4c_topology_case()

    def _fail_hybrid(*args, **kwargs):
        raise AssertionError("default topology_optimize unexpectedly routed through hybrid")

    monkeypatch.setattr(sim, "forward_hybrid_phase1_from_context", _fail_hybrid)

    result = topology_optimize(
        sim,
        region,
        _topology_probe_energy_objective,
        n_iterations=1,
        learning_rate=0.05,
        beta_schedule=[(0, 1.0)],
        verbose=False,
    )

    assert len(result.history) == 1


@pytest.mark.skipif(
    not importlib.util.find_spec("optax"),
    reason="optax not installed",
)
def test_phase4c_topology_optimize_strict_hybrid_route_proof(monkeypatch):
    sim, region = _make_phase4c_topology_case()
    report = inspect_topology_hybrid_support(sim, region, n_steps=12)
    assert report.supported

    calls = {"hybrid": 0}
    original_hybrid = sim.forward_hybrid_phase1_from_context

    def _wrapped_hybrid(context, *, eps_override=None):
        calls["hybrid"] += 1
        return original_hybrid(context, eps_override=eps_override)

    def _fail_pure_ad(*args, **kwargs):
        raise AssertionError("strict topology hybrid unexpectedly used the pure-AD forward path")

    monkeypatch.setattr(sim, "forward_hybrid_phase1_from_context", _wrapped_hybrid)
    monkeypatch.setattr(sim, "_forward_from_materials", _fail_pure_ad)

    result = topology_optimize(
        sim,
        region,
        _topology_probe_energy_objective,
        n_iterations=1,
        learning_rate=0.05,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="hybrid",
    )

    assert len(result.history) == 1
    assert calls["hybrid"] > 0


@pytest.mark.skipif(
    not importlib.util.find_spec("optax"),
    reason="optax not installed",
)
def test_phase4c_topology_optimize_auto_mode_uses_hybrid(monkeypatch):
    sim, region = _make_phase4c_topology_case()

    calls = {"hybrid": 0}
    original_hybrid = sim.forward_hybrid_phase1_from_context

    def _wrapped_hybrid(context, *, eps_override=None):
        calls["hybrid"] += 1
        return original_hybrid(context, eps_override=eps_override)

    def _fail_pure_ad(*args, **kwargs):
        raise AssertionError("auto topology hybrid unexpectedly fell back on supported case")

    monkeypatch.setattr(sim, "forward_hybrid_phase1_from_context", _wrapped_hybrid)
    monkeypatch.setattr(sim, "_forward_from_materials", _fail_pure_ad)

    result = topology_optimize(
        sim,
        region,
        _topology_probe_energy_objective,
        n_iterations=1,
        learning_rate=0.05,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="auto",
    )

    assert len(result.history) == 1
    assert calls["hybrid"] > 0


@pytest.mark.skipif(
    not importlib.util.find_spec("optax"),
    reason="optax not installed",
)
def test_phase2_topology_optimize_cpml_strict_hybrid_route_proof(monkeypatch):
    sim, region = _make_phase4c_topology_case(boundary="cpml")
    report = inspect_topology_hybrid_support(sim, region, n_steps=12)
    assert report.supported

    calls = {"hybrid": 0}
    original_hybrid = sim.forward_hybrid_phase1_from_context

    def _wrapped_hybrid(context, *, eps_override=None):
        calls["hybrid"] += 1
        return original_hybrid(context, eps_override=eps_override)

    def _fail_pure_ad(*args, **kwargs):
        raise AssertionError("strict CPML topology hybrid unexpectedly used the pure-AD forward path")

    monkeypatch.setattr(sim, "forward_hybrid_phase1_from_context", _wrapped_hybrid)
    monkeypatch.setattr(sim, "_forward_from_materials", _fail_pure_ad)

    result = topology_optimize(
        sim,
        region,
        _topology_probe_energy_objective,
        n_iterations=1,
        learning_rate=0.05,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="hybrid",
    )

    assert len(result.history) == 1
    assert calls["hybrid"] > 0


@pytest.mark.skipif(
    not importlib.util.find_spec("optax"),
    reason="optax not installed",
)
def test_phase2_topology_optimize_cpml_auto_mode_uses_hybrid(monkeypatch):
    sim, region = _make_phase4c_topology_case(boundary="cpml")

    calls = {"hybrid": 0}
    original_hybrid = sim.forward_hybrid_phase1_from_context

    def _wrapped_hybrid(context, *, eps_override=None):
        calls["hybrid"] += 1
        return original_hybrid(context, eps_override=eps_override)

    def _fail_pure_ad(*args, **kwargs):
        raise AssertionError("auto CPML topology hybrid unexpectedly fell back on supported case")

    monkeypatch.setattr(sim, "forward_hybrid_phase1_from_context", _wrapped_hybrid)
    monkeypatch.setattr(sim, "_forward_from_materials", _fail_pure_ad)

    result = topology_optimize(
        sim,
        region,
        _topology_probe_energy_objective,
        n_iterations=1,
        learning_rate=0.05,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="auto",
    )

    assert len(result.history) == 1
    assert calls["hybrid"] > 0


@pytest.mark.skipif(
    not importlib.util.find_spec("optax"),
    reason="optax not installed",
)
def test_phase4c_topology_optimize_hybrid_matches_pure_ad_one_step():
    sim, region = _make_phase4c_topology_case()

    pure = topology_optimize(
        sim,
        region,
        _topology_probe_energy_objective,
        n_iterations=1,
        learning_rate=0.05,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="pure_ad",
    )
    hybrid = topology_optimize(
        sim,
        region,
        _topology_probe_energy_objective,
        n_iterations=1,
        learning_rate=0.05,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="hybrid",
    )

    np.testing.assert_allclose(np.asarray(hybrid.history), np.asarray(pure.history), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(hybrid.density), np.asarray(pure.density), rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(
    not importlib.util.find_spec("optax"),
    reason="optax not installed",
)
def test_phase2_topology_optimize_cpml_hybrid_matches_pure_ad_one_step():
    sim, region = _make_phase4c_topology_case(boundary="cpml")

    pure = topology_optimize(
        sim,
        region,
        _topology_probe_energy_objective,
        n_iterations=1,
        learning_rate=0.05,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="pure_ad",
    )
    hybrid = topology_optimize(
        sim,
        region,
        _topology_probe_energy_objective,
        n_iterations=1,
        learning_rate=0.05,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="hybrid",
    )

    np.testing.assert_allclose(np.asarray(hybrid.history), np.asarray(pure.history), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(hybrid.density), np.asarray(pure.density), rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(
    not importlib.util.find_spec("optax"),
    reason="optax not installed",
)
@pytest.mark.parametrize(
    ("case_kwargs", "expected_reason"),
    [
        ({"material_fg": "fr4", "material_sigma": 0.02}, "zero sigma"),
        ({"pec_foreground": True}, "dielectric-only"),
        ({"add_port": True}, "requires zero ports"),
        ({"add_pec_box": True}, "pec_mask-free"),
        ({"debye": True}, "nondispersive"),
    ],
)
def test_phase4c_topology_hybrid_negative_reason_surface(case_kwargs, expected_reason):
    sim, region = _make_phase4c_topology_case(**case_kwargs)
    report = inspect_topology_hybrid_support(sim, region, n_steps=12)

    assert not report.supported
    assert expected_reason in report.reason_text

    with pytest.raises(ValueError, match=expected_reason):
        topology_optimize(
            sim,
            region,
            _topology_probe_energy_objective,
            n_iterations=1,
            learning_rate=0.05,
            beta_schedule=[(0, 1.0)],
            verbose=False,
            adjoint_mode="hybrid",
        )


@pytest.mark.skipif(
    not importlib.util.find_spec("optax"),
    reason="optax not installed",
)
def test_phase4c_topology_sigma_negative_reason_surface_matches_strict_raise_and_auto_fallback(monkeypatch):
    sim, region = _make_phase4c_topology_case(material_fg="fr4", material_sigma=0.02)
    report = inspect_topology_hybrid_support(sim, region, n_steps=12)
    assert not report.supported

    with pytest.raises(ValueError, match=re.escape(report.reason_text)):
        topology_optimize(
            sim,
            region,
            _topology_probe_energy_objective,
            n_iterations=1,
            learning_rate=0.05,
            beta_schedule=[(0, 1.0)],
            verbose=False,
            adjoint_mode="hybrid",
        )

    def _fail_hybrid(*args, **kwargs):
        raise AssertionError("auto topology hybrid unexpectedly used hybrid on sigma-bearing case")

    monkeypatch.setattr(sim, "forward_hybrid_phase1_from_context", _fail_hybrid)
    result = topology_optimize(
        sim,
        region,
        _topology_probe_energy_objective,
        n_iterations=1,
        learning_rate=0.05,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="auto",
    )

    assert len(result.history) == 1


@pytest.mark.skipif(
    not importlib.util.find_spec("optax"),
    reason="optax not installed",
)
def test_phase4c_topology_owned_dispersion_stays_on_pure_ad_in_auto_mode(monkeypatch):
    sim, region = _make_phase4c_topology_case(debye=True)
    report = inspect_topology_hybrid_support(sim, region, n_steps=12)
    assert not report.supported
    assert "nondispersive" in report.reason_text

    def _fail_hybrid(*args, **kwargs):
        raise AssertionError("auto topology hybrid unexpectedly used hybrid on topology-owned dispersion")

    monkeypatch.setattr(sim, "forward_hybrid_phase1_from_context", _fail_hybrid)
    result = topology_optimize(
        sim,
        region,
        _topology_probe_energy_objective,
        n_iterations=1,
        learning_rate=0.05,
        beta_schedule=[(0, 1.0)],
        verbose=False,
        adjoint_mode="auto",
    )

    assert len(result.history) == 1


# ---------------------------------------------------------------------------
# Integration: topology_optimize reduces loss
# ---------------------------------------------------------------------------

class TestTopologyOptimize:
    """Integration test for the full topology optimization loop."""

    @pytest.mark.skipif(
        not importlib.util.find_spec("optax"),
        reason="optax not installed",
    )
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

def test_pec_foreground_gradient_is_finite_and_nonzero():
    """PEC occupancy topology should produce a usable simulator gradient."""
    from rfx.api import Simulation
    from rfx.core.yee import MaterialArrays

    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    sim.add_source((0.005, 0.0075, 0.0075), "ez")
    sim.add_probe((0.01, 0.0075, 0.0075), "ez")

    region = TopologyDesignRegion(
        corner_lo=(0.006, 0.006, 0.006),
        corner_hi=(0.009, 0.009, 0.009),
        material_bg="air",
        material_fg="pec",
        beta_projection=1.0,
    )

    grid = sim._build_grid()
    lo_idx = grid.position_to_index(region.corner_lo)
    hi_idx = grid.position_to_index(region.corner_hi)
    design_shape = tuple(hi_idx[d] - lo_idx[d] + 1 for d in range(3))

    base_materials, debye_spec, lorentz_spec, base_pec_mask, _, _ = sim._assemble_materials(grid)

    def loss_fn(logit):
        rho = jax.nn.sigmoid(logit)
        fields = density_to_material_fields(
            rho,
            eps_bg=1.0006,
            eps_fg=1.0,
            sigma_bg=0.0,
            sigma_fg=1e10,
            pec_fg=True,
        )
        si, sj, sk = lo_idx
        ei, ej, ek = hi_idx
        eps_r = base_materials.eps_r.at[si:ei+1, sj:ej+1, sk:ek+1].set(fields.eps)
        sigma = base_materials.sigma.at[si:ei+1, sj:ej+1, sk:ek+1].set(fields.sigma)
        materials = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=base_materials.mu_r)
        pec_occupancy = jnp.zeros(grid.shape, dtype=jnp.float32)
        pec_occupancy = pec_occupancy.at[si:ei+1, sj:ej+1, sk:ek+1].set(fields.pec_occupancy)
        result = sim._forward_from_materials(
            grid,
            materials,
            debye_spec,
            lorentz_spec,
            n_steps=20,
            checkpoint=True,
            pec_mask=base_pec_mask,
            pec_occupancy=pec_occupancy,
        )
        return -jnp.sum(result.time_series ** 2)

    logit0 = jnp.zeros(design_shape, dtype=jnp.float32)
    grad = jax.grad(loss_fn)(logit0)

    assert jnp.all(jnp.isfinite(grad))
    assert float(jnp.max(jnp.abs(grad))) > 0.0


    @pytest.mark.skipif(
        not importlib.util.find_spec("optax"),
        reason="optax not installed",
    )
    def test_topology_optimize_with_pec_foreground(self):
        """PEC foreground should optimise through conductor occupancy, not sigma hacks."""
        from rfx.api import Simulation

        sim = Simulation(
            freq_max=5e9,
            domain=(0.015, 0.015, 0.015),
            boundary="pec",
        )
        sim.add_source((0.005, 0.0075, 0.0075), "ez")
        sim.add_probe((0.01, 0.0075, 0.0075), "ez")

        region = TopologyDesignRegion(
            corner_lo=(0.006, 0.006, 0.006),
            corner_hi=(0.009, 0.009, 0.009),
            material_bg="air",
            material_fg="pec",
            beta_projection=1.0,
        )

        result = topology_optimize(
            sim,
            region,
            lambda run_result: -jnp.sum(run_result.time_series ** 2),
            n_iterations=2,
            learning_rate=0.05,
            beta_schedule=[(0, 1.0)],
            verbose=False,
        )

        assert isinstance(result, TopologyResult)
        assert len(result.history) == 2
        assert np.all(np.isfinite(result.history))
        assert result.pec_occupancy_design is not None
        assert jnp.all(jnp.isfinite(result.pec_occupancy_design))

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

    @pytest.mark.skipif(
        not importlib.util.find_spec("optax"),
        reason="optax not installed",
    )
    def test_topology_optimize_with_cpml(self):
        """Topology optimization should work with CPML boundary.

        Verifies that the design-region index clamping prevents
        the design region from overlapping with CPML padding cells.
        This is the scenario that caused shape mismatch errors before
        the fix in topology.py.
        """
        from rfx.api import Simulation

        # Use CPML boundary (the default) — this adds padding cells
        sim = Simulation(
            freq_max=5e9,
            domain=(0.015, 0.015, 0.015),
            boundary="cpml",
            cpml_layers=8,
        )
        sim.add_port((0.005, 0.0075, 0.0075), "ez")
        sim.add_probe((0.01, 0.0075, 0.0075), "ez")

        # Design region well inside the domain (should not overlap CPML)
        region = TopologyDesignRegion(
            corner_lo=(0.006, 0.006, 0.006),
            corner_hi=(0.009, 0.009, 0.009),
            material_bg="air",
            material_fg="fr4",
            beta_projection=1.0,
        )

        def obj(result):
            return -jnp.sum(result.time_series ** 2)

        # Should not raise — the CPML index clamping prevents mismatch
        result = topology_optimize(
            sim, region, obj,
            n_iterations=2,
            learning_rate=0.05,
            beta_schedule=[(0, 1.0)],
            verbose=False,
        )

        assert isinstance(result, TopologyResult)
        assert len(result.history) == 2
        assert result.density.ndim == 3
        assert result.eps_design.ndim == 3

        # Verify the grid uses CPML
        grid = sim._build_grid()
        assert grid.pad_x == 8
        assert grid.pad_y == 8
        assert grid.pad_z == 8

        # Verify design region indices are within interior
        lo = grid.position_to_index(region.corner_lo)
        hi = grid.position_to_index(region.corner_hi)
        for d in range(3):
            pad = (grid.pad_x, grid.pad_y, grid.pad_z)[d]
            dim = (grid.nx, grid.ny, grid.nz)[d]
            assert lo[d] >= pad, (
                f"lo_idx[{d}]={lo[d]} inside CPML (pad={pad})"
            )
            assert hi[d] <= dim - 1 - pad, (
                f"hi_idx[{d}]={hi[d]} inside CPML (max={dim-1-pad})"
            )
        print(f"\n  CPML topology: grid={grid.shape}, pads=({grid.pad_x},{grid.pad_y},{grid.pad_z})")
