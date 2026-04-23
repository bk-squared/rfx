"""Tests for inverse design optimizer.

Validates:
1. DesignRegion construction
2. Latent-to-eps mapping (sigmoid bounds)
3. OptimizeResult structure
4. Gradient check (AD vs finite-difference)
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

from rfx import Box, GaussianPulse
from rfx.api import Simulation
from rfx.optimize import (
    DesignRegion, OptimizeResult, GradientCheckResult,
    _inspect_optimize_hybrid_support, _latent_to_eps, optimize, gradient_check,
)
from rfx.optimize_objectives import maximize_directivity

pytestmark = pytest.mark.gpu


_OPT_BOX_LO = (0.006, 0.006, 0.006)
_OPT_BOX_HI = (0.009, 0.009, 0.009)
_OPT_NTFF_BOX_LO = (0.004, 0.004, 0.004)
_OPT_NTFF_BOX_HI = (0.011, 0.011, 0.011)
_OPT_NTFF_FREQS = jnp.array([3e9], dtype=jnp.float32)


def _make_optimize_design_region() -> DesignRegion:
    return DesignRegion(
        corner_lo=_OPT_BOX_LO,
        corner_hi=_OPT_BOX_HI,
        eps_range=(1.0, 4.4),
    )


def _make_one_port_optimize_design_region() -> DesignRegion:
    return DesignRegion(
        corner_lo=(0.009, 0.003, 0.003),
        corner_hi=(0.012, 0.006, 0.006),
        eps_range=(1.0, 4.4),
    )


def _make_overlapping_one_port_optimize_design_region() -> DesignRegion:
    return DesignRegion(
        corner_lo=(0.004, 0.006, 0.006),
        corner_hi=(0.006, 0.009, 0.009),
        eps_range=(1.0, 4.4),
    )


def _make_overlapping_passive_port_optimize_design_region() -> DesignRegion:
    return DesignRegion(
        corner_lo=(0.009, 0.006, 0.006),
        corner_hi=(0.011, 0.009, 0.009),
        eps_range=(1.0, 4.4),
    )


def _make_source_optimize_sim(*, boundary: str = "pec", ntff: bool = False) -> Simulation:
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary=boundary)
    sim.add_source(
        (0.005, 0.0075, 0.0075),
        "ez",
        waveform=GaussianPulse(f0=3e9, bandwidth=0.5),
    )
    sim.add_probe((0.01, 0.0075, 0.0075), "ez")
    if ntff:
        sim.add_ntff_box(_OPT_NTFF_BOX_LO, _OPT_NTFF_BOX_HI, freqs=_OPT_NTFF_FREQS)
    return sim


def _make_port_optimize_sim() -> Simulation:
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    sim.add_port((0.005, 0.0075, 0.0075), "ez")
    sim.add_probe((0.01, 0.0075, 0.0075), "ez")
    return sim


def _make_passive_port_optimize_sim() -> Simulation:
    sim = _make_port_optimize_sim()
    sim.add_port((0.01, 0.0075, 0.0075), "ez", excite=False)
    return sim


def _make_multi_port_optimize_sim() -> Simulation:
    sim = _make_passive_port_optimize_sim()
    sim.add_port((0.012, 0.0075, 0.0075), "ez", excite=False)
    return sim


def _make_pec_mask_optimize_sim() -> Simulation:
    sim = _make_source_optimize_sim()
    sim.add(Box(_OPT_BOX_LO, _OPT_BOX_HI), material="pec")
    return sim


def _make_lossy_optimize_sim() -> Simulation:
    sim = _make_source_optimize_sim()
    sim.add_material("lossy", eps_r=2.0, sigma=5.0)
    sim.add(Box(_OPT_BOX_LO, _OPT_BOX_HI), material="lossy")
    return sim


def _probe_energy_objective(run_result) -> jnp.ndarray:
    return -jnp.sum(run_result.time_series ** 2)


def _port_cell(sim: Simulation) -> tuple[int, int, int]:
    grid = sim._build_grid()
    return tuple(int(v) for v in grid.position_to_index(sim._ports[0].position))


def _passive_port_cell(sim: Simulation) -> tuple[int, int, int]:
    grid = sim._build_grid()
    return tuple(int(v) for v in grid.position_to_index(sim._ports[1].position))


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
        _probe_energy_objective,
        n_iters=1,
        lr=0.01,
        verbose=False,
    )

    assert isinstance(result, OptimizeResult)
    assert len(result.loss_history) == 1
    assert result.eps_design.shape == result.latent.shape


def test_optimize_default_route_stays_on_pure_ad(monkeypatch):
    """Default optimize() mode should not silently switch to hybrid routing."""
    sim = _make_source_optimize_sim()
    region = _make_one_port_optimize_design_region()

    def _fail_hybrid(*args, **kwargs):
        raise AssertionError("default optimize() unexpectedly routed through hybrid")

    monkeypatch.setattr(sim, "forward_hybrid_phase1_from_context", _fail_hybrid)

    result = optimize(
        sim,
        region,
        _probe_energy_objective,
        n_iters=1,
        lr=0.01,
        verbose=False,
    )

    assert len(result.loss_history) == 1


def test_optimize_hybrid_supported_route_bypasses_pure_ad(monkeypatch):
    """Strict hybrid mode should use the hybrid seam on a supported source/probe case."""
    sim = _make_source_optimize_sim()
    report = sim.inspect_hybrid_phase1(n_steps=12)
    assert report.supported

    region = _make_optimize_design_region()
    calls = {"hybrid": 0}
    original_hybrid = sim.forward_hybrid_phase1_from_context

    def _wrapped_hybrid(context, *, eps_override=None):
        calls["hybrid"] += 1
        return original_hybrid(context, eps_override=eps_override)

    def _fail_pure_ad(*args, **kwargs):
        raise AssertionError("strict hybrid mode unexpectedly used the pure-AD forward path")

    monkeypatch.setattr(sim, "forward_hybrid_phase1_from_context", _wrapped_hybrid)
    monkeypatch.setattr(sim, "_forward_from_materials", _fail_pure_ad)

    result = optimize(
        sim,
        region,
        _probe_energy_objective,
        n_iters=1,
        lr=0.01,
        verbose=False,
        adjoint_mode="hybrid",
    )

    assert len(result.loss_history) == 1
    assert calls["hybrid"] > 0
    assert not np.allclose(np.asarray(result.latent), 0.0)


def test_optimize_hybrid_supported_one_excited_lumped_port_route_bypasses_pure_ad(monkeypatch):
    """Strict hybrid mode should support the approved one excited lumped-port subset."""
    sim = _make_port_optimize_sim()
    report = sim.inspect_hybrid_phase1(n_steps=12)
    assert report.supported
    assert report.port_metadata is not None
    assert report.port_metadata.total_ports == 1
    assert report.port_metadata.excited_ports == 1
    assert report.port_metadata.passive_ports == 0

    calls = {"hybrid": 0}
    original_hybrid = sim.forward_hybrid_phase1_from_context

    def _wrapped_hybrid(context, *, eps_override=None):
        calls["hybrid"] += 1
        return original_hybrid(context, eps_override=eps_override)

    def _fail_pure_ad(*args, **kwargs):
        raise AssertionError("strict hybrid one-port optimize unexpectedly used the pure-AD path")

    monkeypatch.setattr(sim, "forward_hybrid_phase1_from_context", _wrapped_hybrid)
    monkeypatch.setattr(sim, "_forward_from_materials", _fail_pure_ad)

    result = optimize(
        sim,
        _make_one_port_optimize_design_region(),
        _probe_energy_objective,
        n_iters=1,
        lr=0.01,
        verbose=False,
        adjoint_mode="hybrid",
    )

    assert len(result.loss_history) == 1
    assert calls["hybrid"] > 0


def test_optimize_hybrid_supported_one_excited_lumped_port_auto_mode_uses_hybrid(monkeypatch):
    """Auto mode should choose hybrid for the approved one-port subset."""
    sim = _make_port_optimize_sim()
    report = sim.inspect_hybrid_phase1(n_steps=12)
    assert report.supported

    calls = {"hybrid": 0}
    original_hybrid = sim.forward_hybrid_phase1_from_context

    def _wrapped_hybrid(context, *, eps_override=None):
        calls["hybrid"] += 1
        return original_hybrid(context, eps_override=eps_override)

    def _fail_pure_ad(*args, **kwargs):
        raise AssertionError("auto mode unexpectedly fell back on a supported one-port fixture")

    monkeypatch.setattr(sim, "forward_hybrid_phase1_from_context", _wrapped_hybrid)
    monkeypatch.setattr(sim, "_forward_from_materials", _fail_pure_ad)

    result = optimize(
        sim,
        _make_one_port_optimize_design_region(),
        _probe_energy_objective,
        n_iters=1,
        lr=0.01,
        verbose=False,
        adjoint_mode="auto",
    )

    assert len(result.loss_history) == 1
    assert calls["hybrid"] > 0


def test_optimize_hybrid_supported_one_excited_plus_one_passive_lumped_port_route_bypasses_pure_ad(monkeypatch):
    """Strict hybrid mode should support the approved one excited + one passive subset."""
    sim = _make_passive_port_optimize_sim()
    report = sim.inspect_hybrid_phase1(n_steps=12)
    assert report.supported
    assert report.port_metadata is not None
    assert report.port_metadata.total_ports == 2
    assert report.port_metadata.excited_ports == 1
    assert report.port_metadata.passive_ports == 1
    assert len(report.port_metadata.passive_lumped_port_cells) == 1

    calls = {"hybrid": 0}
    original_hybrid = sim.forward_hybrid_phase1_from_context

    def _wrapped_hybrid(context, *, eps_override=None):
        calls["hybrid"] += 1
        return original_hybrid(context, eps_override=eps_override)

    def _fail_pure_ad(*args, **kwargs):
        raise AssertionError("strict hybrid two-port optimize unexpectedly used the pure-AD path")

    monkeypatch.setattr(sim, "forward_hybrid_phase1_from_context", _wrapped_hybrid)
    monkeypatch.setattr(sim, "_forward_from_materials", _fail_pure_ad)

    result = optimize(
        sim,
        _make_one_port_optimize_design_region(),
        _probe_energy_objective,
        n_iters=1,
        lr=0.01,
        verbose=False,
        adjoint_mode="hybrid",
    )

    assert len(result.loss_history) == 1
    assert calls["hybrid"] > 0


def test_optimize_hybrid_supported_one_excited_plus_one_passive_lumped_port_auto_mode_uses_hybrid(monkeypatch):
    """Auto mode should choose hybrid for the approved one excited + one passive subset."""
    sim = _make_passive_port_optimize_sim()
    report = sim.inspect_hybrid_phase1(n_steps=12)
    assert report.supported

    calls = {"hybrid": 0}
    original_hybrid = sim.forward_hybrid_phase1_from_context

    def _wrapped_hybrid(context, *, eps_override=None):
        calls["hybrid"] += 1
        return original_hybrid(context, eps_override=eps_override)

    def _fail_pure_ad(*args, **kwargs):
        raise AssertionError("auto mode unexpectedly fell back on a supported two-port fixture")

    monkeypatch.setattr(sim, "forward_hybrid_phase1_from_context", _wrapped_hybrid)
    monkeypatch.setattr(sim, "_forward_from_materials", _fail_pure_ad)

    result = optimize(
        sim,
        _make_one_port_optimize_design_region(),
        _probe_energy_objective,
        n_iters=1,
        lr=0.01,
        verbose=False,
        adjoint_mode="auto",
    )

    assert len(result.loss_history) == 1
    assert calls["hybrid"] > 0


def test_optimize_supported_one_port_fixture_design_region_is_disjoint_from_port_cell():
    """The approved one-port optimize fixture must keep design-region cells away from the port cell."""
    sim = _make_port_optimize_sim()
    region = _make_one_port_optimize_design_region()
    grid = sim._build_grid()
    lo_idx = tuple(int(v) for v in grid.position_to_index(region.corner_lo))
    hi_idx = tuple(int(v) for v in grid.position_to_index(region.corner_hi))
    port_cell = _port_cell(sim)

    assert not (
        lo_idx[0] <= port_cell[0] <= hi_idx[0]
        and lo_idx[1] <= port_cell[1] <= hi_idx[1]
        and lo_idx[2] <= port_cell[2] <= hi_idx[2]
    )


def test_optimize_supported_two_port_fixture_design_region_is_disjoint_from_port_cells():
    """The approved two-port optimize fixture must keep design-region cells away from both port cells."""
    sim = _make_passive_port_optimize_sim()
    region = _make_one_port_optimize_design_region()
    grid = sim._build_grid()
    lo_idx = tuple(int(v) for v in grid.position_to_index(region.corner_lo))
    hi_idx = tuple(int(v) for v in grid.position_to_index(region.corner_hi))
    cells = (_port_cell(sim), _passive_port_cell(sim))

    for cell in cells:
        assert not (
            lo_idx[0] <= cell[0] <= hi_idx[0]
            and lo_idx[1] <= cell[1] <= hi_idx[1]
            and lo_idx[2] <= cell[2] <= hi_idx[2]
        )


def test_optimize_hybrid_inspection_reports_design_region_overlap_for_one_port_fixture():
    """Optimize-side hybrid inspection should surface design-region/port overlap in port metadata."""
    sim = _make_port_optimize_sim()
    overlap_region = _make_overlapping_one_port_optimize_design_region()
    grid = sim._build_grid()
    base_materials, _, _, _, _, _ = sim._assemble_materials(grid)
    lo_idx = tuple(int(v) for v in grid.position_to_index(overlap_region.corner_lo))
    hi_idx = tuple(int(v) for v in grid.position_to_index(overlap_region.corner_hi))
    _, report = _inspect_optimize_hybrid_support(
        sim,
        eps_r=base_materials.eps_r,
        n_steps=12,
        design_bounds=(lo_idx, hi_idx),
    )

    assert not report.supported
    assert report.port_metadata is not None
    assert report.port_metadata.design_region_overlaps_excited_port_cell
    assert "design region overlaps the excited lumped-port cell" in report.reason_text


def test_optimize_hybrid_inspection_reports_design_region_overlap_for_passive_port_fixture():
    """Optimize-side hybrid inspection should surface design-region/passive-port overlap."""
    sim = _make_passive_port_optimize_sim()
    overlap_region = _make_overlapping_passive_port_optimize_design_region()
    grid = sim._build_grid()
    base_materials, _, _, _, _, _ = sim._assemble_materials(grid)
    lo_idx = tuple(int(v) for v in grid.position_to_index(overlap_region.corner_lo))
    hi_idx = tuple(int(v) for v in grid.position_to_index(overlap_region.corner_hi))
    _, report = _inspect_optimize_hybrid_support(
        sim,
        eps_r=base_materials.eps_r,
        n_steps=12,
        design_bounds=(lo_idx, hi_idx),
    )

    assert not report.supported
    assert report.port_metadata is not None
    assert report.port_metadata.design_region_overlaps_passive_lumped_port_cell
    assert "design region overlaps a passive lumped-port cell" in report.reason_text


def test_optimize_hybrid_rejects_design_region_overlap_with_excited_port_cell():
    """Strict hybrid mode should fail closed when the design region touches the excited port cell."""
    sim = _make_port_optimize_sim()
    overlap_region = _make_overlapping_one_port_optimize_design_region()

    with pytest.raises(ValueError, match="design region overlaps the excited lumped-port cell"):
        optimize(
            sim,
            overlap_region,
            _probe_energy_objective,
            n_iters=1,
            lr=0.01,
            verbose=False,
            adjoint_mode="hybrid",
        )


def test_optimize_hybrid_rejects_design_region_overlap_with_passive_port_cell():
    """Strict hybrid mode should fail closed when the design region touches the passive port cell."""
    sim = _make_passive_port_optimize_sim()
    overlap_region = _make_overlapping_passive_port_optimize_design_region()

    with pytest.raises(ValueError, match="design region overlaps a passive lumped-port cell"):
        optimize(
            sim,
            overlap_region,
            _probe_energy_objective,
            n_iters=1,
            lr=0.01,
            verbose=False,
            adjoint_mode="hybrid",
        )


def test_optimize_auto_falls_back_for_design_region_overlap_with_excited_port_cell(monkeypatch):
    """Auto mode should fall back to pure AD when the design region overlaps the excited port cell."""
    sim = _make_port_optimize_sim()
    overlap_region = _make_overlapping_one_port_optimize_design_region()

    def _fail_hybrid(*args, **kwargs):
        raise AssertionError("auto mode unexpectedly used hybrid on a design-region/port overlap case")

    monkeypatch.setattr(sim, "forward_hybrid_phase1_from_context", _fail_hybrid)

    result = optimize(
        sim,
        overlap_region,
        _probe_energy_objective,
        n_iters=1,
        lr=0.01,
        verbose=False,
        adjoint_mode="auto",
    )

    assert len(result.loss_history) == 1


def test_optimize_auto_falls_back_for_design_region_overlap_with_passive_port_cell(monkeypatch):
    """Auto mode should fall back to pure AD when the design region overlaps the passive port cell."""
    sim = _make_passive_port_optimize_sim()
    overlap_region = _make_overlapping_passive_port_optimize_design_region()

    def _fail_hybrid(*args, **kwargs):
        raise AssertionError("auto mode unexpectedly used hybrid on a design-region/passive-port overlap case")

    monkeypatch.setattr(sim, "forward_hybrid_phase1_from_context", _fail_hybrid)

    result = optimize(
        sim,
        overlap_region,
        _probe_energy_objective,
        n_iters=1,
        lr=0.01,
        verbose=False,
        adjoint_mode="auto",
    )

    assert len(result.loss_history) == 1


def test_optimize_hybrid_single_step_matches_pure_ad():
    """Strict hybrid routing should match the pure-AD optimize step on a supported probe objective."""
    sim = _make_source_optimize_sim()
    region = _make_optimize_design_region()

    pure_result = optimize(
        sim,
        region,
        _probe_energy_objective,
        n_iters=1,
        lr=0.01,
        verbose=False,
        adjoint_mode="pure_ad",
    )
    hybrid_result = optimize(
        sim,
        region,
        _probe_energy_objective,
        n_iters=1,
        lr=0.01,
        verbose=False,
        adjoint_mode="hybrid",
    )

    np.testing.assert_allclose(
        np.asarray(hybrid_result.latent),
        np.asarray(pure_result.latent),
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(hybrid_result.eps_design),
        np.asarray(pure_result.eps_design),
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(hybrid_result.loss_history),
        np.asarray(pure_result.loss_history),
        rtol=1e-5,
        atol=1e-7,
    )
    assert not np.allclose(np.asarray(hybrid_result.latent), 0.0)


def test_optimize_hybrid_two_port_single_step_matches_pure_ad():
    """Strict hybrid routing should match pure AD on the supported two-lumped-port proxy subset."""
    sim = _make_passive_port_optimize_sim()
    region = _make_one_port_optimize_design_region()

    pure_result = optimize(
        sim,
        region,
        _probe_energy_objective,
        n_iters=1,
        lr=0.01,
        verbose=False,
        adjoint_mode="pure_ad",
    )
    hybrid_result = optimize(
        sim,
        region,
        _probe_energy_objective,
        n_iters=1,
        lr=0.01,
        verbose=False,
        adjoint_mode="hybrid",
    )

    np.testing.assert_allclose(
        np.asarray(hybrid_result.latent),
        np.asarray(pure_result.latent),
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(hybrid_result.loss_history),
        np.asarray(pure_result.loss_history),
        rtol=1e-5,
        atol=1e-7,
    )


def test_optimize_hybrid_ntff_directivity_single_step_matches_pure_ad():
    """Hybrid optimize should support NTFF/directivity objectives on supported source fixtures."""
    sim = _make_source_optimize_sim(ntff=True)
    report = sim.inspect_hybrid_phase1(n_steps=12)
    assert report.supported

    region = _make_optimize_design_region()
    objective = maximize_directivity(theta_target=jnp.pi / 2, phi_target=0.0)

    pure_result = optimize(
        sim,
        region,
        objective,
        n_iters=1,
        lr=0.01,
        verbose=False,
        adjoint_mode="pure_ad",
    )
    hybrid_result = optimize(
        sim,
        region,
        objective,
        n_iters=1,
        lr=0.01,
        verbose=False,
        adjoint_mode="hybrid",
    )

    np.testing.assert_allclose(
        np.asarray(hybrid_result.latent),
        np.asarray(pure_result.latent),
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(hybrid_result.loss_history),
        np.asarray(pure_result.loss_history),
        rtol=1e-5,
        atol=1e-7,
    )


def test_optimize_auto_falls_back_for_port_based_fixture(monkeypatch):
    """Auto mode should keep port-based optimize fixtures on the pure-AD lane."""
    sim = _make_multi_port_optimize_sim()
    region = _make_optimize_design_region()

    def _fail_hybrid(*args, **kwargs):
        raise AssertionError("auto mode unexpectedly routed an unsupported port fixture through hybrid")

    monkeypatch.setattr(sim, "forward_hybrid_phase1_from_context", _fail_hybrid)

    result = optimize(
        sim,
        region,
        _probe_energy_objective,
        n_iters=1,
        lr=0.01,
        verbose=False,
        adjoint_mode="auto",
    )

    assert len(result.loss_history) == 1


@pytest.mark.parametrize(
    ("sim_factory", "expected_error"),
    [
        (_make_multi_port_optimize_sim, "only one excited lumped port"),
        (_make_pec_mask_optimize_sim, "pec_mask"),
        (_make_lossy_optimize_sim, "lossy materials"),
    ],
)
def test_optimize_hybrid_raises_for_blocked_fixtures(sim_factory, expected_error):
    """Strict hybrid mode should raise on blocked source-family, pec_mask, and lossy cases."""
    sim = sim_factory()
    report = sim.inspect_hybrid_phase1(n_steps=12)
    assert not report.supported
    assert expected_error in report.reason_text

    with pytest.raises(ValueError, match=expected_error):
        optimize(
            sim,
            _make_optimize_design_region(),
            _probe_energy_objective,
            n_iters=1,
            lr=0.01,
            verbose=False,
            adjoint_mode="hybrid",
        )


def test_forward_returns_minimal_result_contract():
    """Simulation.forward should return only differentiable observables."""
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    sim.add_source((0.005, 0.0075, 0.0075), "ez")
    sim.add_probe((0.01, 0.0075, 0.0075), "ez")
    sim.add_ntff_box((0.003, 0.003, 0.003), (0.012, 0.012, 0.012), freqs=jnp.array([3e9]))

    result = sim.forward(n_steps=10, checkpoint=True)

    assert result.time_series.shape[0] == 10
    assert result.ntff_box is not None
    assert result.grid is not None
    assert not hasattr(result, "state")


# ---------------------------------------------------------------------------
# gradient_check: AD vs finite-difference
# ---------------------------------------------------------------------------

def test_gradient_check_matches():
    """AD gradient should agree with finite-difference gradient.

    Uses a tiny PEC simulation with a single scalar design parameter
    (uniform eps perturbation) so that the finite-difference loop is
    fast (only 2*N forward evaluations for N parameters).
    """
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    sim.add_port((0.005, 0.0075, 0.0075), "ez")
    sim.add_probe((0.01, 0.0075, 0.0075), "ez")

    grid = sim._build_grid()

    # Single scalar parameter: uniform eps perturbation
    design_params = jnp.zeros(grid.shape, dtype=jnp.float32)

    def obj(result):
        return jnp.sum(result.time_series ** 2)

    result = gradient_check(
        sim, design_params, obj,
        eps=1e-2,
        n_steps=20,
    )

    assert isinstance(result, GradientCheckResult)
    assert result.ad_grad.shape == design_params.shape
    assert result.fd_grad.shape == design_params.shape
    # Relative error should be reasonable (< 0.5 for such a coarse check)
    assert result.relative_error < 0.5, (
        f"AD/FD relative error too large: {result.relative_error:.4f}"
    )
    print(f"\n  gradient_check: rel_error = {result.relative_error:.6f}")
