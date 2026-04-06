"""Tests for auto_configure: source auto-selection, memory estimation."""

from rfx.auto_config import auto_configure, SimConfig


def test_auto_configure_source_recommendation_cpml():
    """CPML boundary should recommend J-source."""
    config = auto_configure([], freq_range=(1e9, 3e9))
    assert hasattr(config, "source_type")
    assert config.source_type == "j_source"


def test_auto_configure_source_recommendation_pec():
    """PEC boundary should recommend raw source."""
    config = auto_configure([], freq_range=(1e9, 3e9), boundary="pec")
    assert config.source_type == "raw"


def test_auto_configure_source_info_not_empty():
    """Source info should explain the recommendation."""
    config = auto_configure([], freq_range=(1e9, 3e9))
    assert config.source_info and len(config.source_info) > 0


def test_auto_configure_source_info_pec():
    """PEC source info should mention PEC."""
    config = auto_configure([], freq_range=(1e9, 3e9), boundary="pec")
    assert "PEC" in config.source_info


def test_auto_configure_source_info_cpml():
    """CPML source info should mention CPML or J-source."""
    config = auto_configure([], freq_range=(1e9, 3e9), boundary="cpml")
    assert "CPML" in config.source_info or "J-source" in config.source_info


def test_auto_configure_default_boundary_is_cpml():
    """Default boundary should be cpml, giving j_source."""
    config = auto_configure([], freq_range=(2e9, 5e9))
    assert config.source_type == "j_source"
    assert "CPML" in config.source_info


def test_auto_configure_summary_includes_source():
    """Summary string should include source type info."""
    config = auto_configure([], freq_range=(1e9, 3e9))
    summary = config.summary()
    assert "source" in summary.lower()
    assert "j_source" in summary


# ---------------------------------------------------------------------------
# Memory estimation and budget
# ---------------------------------------------------------------------------

def test_auto_configure_memory_estimate():
    """SimConfig.estimated_memory_mb should return a positive value and
    the grid_shape property should reflect CPML padding."""
    config = auto_configure([], freq_range=(1e9, 3e9))

    # grid_shape should be a 3-tuple of positive ints
    shape = config.grid_shape
    assert len(shape) == 3
    assert all(s > 0 for s in shape), f"Invalid grid_shape: {shape}"

    # estimated_memory_mb should be positive
    mem = config.estimated_memory_mb
    assert mem > 0, f"Expected positive memory estimate, got {mem}"
    print(f"\n  Grid shape: {shape}, estimated memory: {mem:.1f} MB")


def test_auto_configure_memory_budget_coarsens_dx():
    """When max_memory_mb is tight, dx should increase to fit budget."""
    from rfx.geometry.csg import Box
    # Use real geometry so the domain is large enough for coarsening to matter
    geometry = [(Box((0, 0, 0), (0.1, 0.1, 0.05)), "dielectric")]
    materials = {"dielectric": {"eps_r": 4.4, "sigma": 0.0}}
    config_unbounded = auto_configure(geometry, freq_range=(1e9, 3e9),
                                       materials=materials)
    # Use a very tight budget to force coarsening
    budget_mb = config_unbounded.estimated_memory_mb * 0.3
    config_bounded = auto_configure(
        geometry, freq_range=(1e9, 3e9), materials=materials,
        max_memory_mb=budget_mb,
    )

    # Bounded config should have coarser (larger) dx
    assert config_bounded.dx >= config_unbounded.dx, (
        f"Bounded dx ({config_bounded.dx}) should be >= unbounded ({config_unbounded.dx})"
    )

    # Bounded config should be close to budget (memory estimate is approximate,
    # allow 20% overshoot since the estimate includes ~10x AD overhead factor)
    assert config_bounded.estimated_memory_mb <= budget_mb * 1.2, (
        f"Bounded memory ({config_bounded.estimated_memory_mb:.1f} MB) exceeds "
        f"budget ({budget_mb:.1f} MB) by more than 20%"
    )
    print(
        f"\n  Unbounded: dx={config_unbounded.dx*1e3:.3f} mm, "
        f"mem={config_unbounded.estimated_memory_mb:.1f} MB\n"
        f"  Bounded({budget_mb:.0f}MB): dx={config_bounded.dx*1e3:.3f} mm, "
        f"mem={config_bounded.estimated_memory_mb:.1f} MB"
    )


def test_auto_configure_memory_budget_ignored_with_dx_override():
    """max_memory_mb should be ignored when dx_override is provided."""
    config = auto_configure(
        [], freq_range=(1e9, 3e9),
        dx_override=0.005,
        max_memory_mb=0.001,  # impossibly small
    )
    # dx should remain at the override value
    assert config.dx == 0.005


# ---------------------------------------------------------------------------
# P2: Smooth grading
# ---------------------------------------------------------------------------

def test_smooth_grading_noop_when_already_smooth():
    """Smooth grading should not modify cells that are already within ratio."""
    from rfx.auto_config import smooth_grading
    import numpy as np
    cells = [0.5e-3] * 4 + [1.0e-3] * 4  # ratio 2.0 at transition
    result = smooth_grading(cells, max_ratio=1.3)
    # Result should have more cells (transition inserted)
    assert len(result) >= len(cells)
    # But all-uniform should pass through unchanged
    uniform = [1.0e-3] * 10
    result_u = smooth_grading(uniform, max_ratio=1.3)
    assert len(result_u) == 10
    np.testing.assert_allclose(result_u, 1.0e-3, atol=1e-15)


def test_smooth_grading_enforces_max_ratio():
    """After smoothing, all adjacent ratios should be <= max_ratio."""
    from rfx.auto_config import smooth_grading
    import numpy as np
    # Abrupt 5x transition
    cells = [0.2e-3] * 3 + [1.0e-3] * 5
    result = smooth_grading(cells, max_ratio=1.3)
    ratios = result[1:] / result[:-1]
    max_r = float(np.max(np.maximum(ratios, 1.0 / ratios)))
    assert max_r <= 1.3 + 1e-6, f"Max ratio {max_r:.3f} exceeds 1.3"
    print(f"  Cells: {len(cells)} -> {len(result)}, max ratio: {max_r:.3f}")


def test_smooth_grading_single_cell():
    """Single cell should pass through unchanged."""
    from rfx.auto_config import smooth_grading
    import numpy as np
    result = smooth_grading([0.5e-3])
    assert len(result) == 1
    np.testing.assert_allclose(result[0], 0.5e-3)


def test_smooth_grading_preserves_boundary_values():
    """First and last cell sizes should be preserved."""
    from rfx.auto_config import smooth_grading
    import numpy as np
    cells = [0.1e-3, 0.1e-3, 2.0e-3, 2.0e-3]
    result = smooth_grading(cells, max_ratio=1.3)
    np.testing.assert_allclose(result[0], 0.1e-3, rtol=1e-10)
    np.testing.assert_allclose(result[-1], 2.0e-3, rtol=1e-10)


# ---------------------------------------------------------------------------
# P1: Auto mesh via Simulation
# ---------------------------------------------------------------------------

def test_simulation_auto_mesh_sets_dx():
    """When dx=None and geometry exists, run() should auto-set dx from features."""
    import warnings
    from rfx import Simulation, Box, GaussianPulse

    sim = Simulation(freq_max=5e9, domain=(0.05, 0.05, 0.02), boundary="pec")
    sim.add_material("fr4", eps_r=4.4)
    sim.add(Box((0, 0, 0), (0.05, 0.05, 0.002)), material="fr4")
    sim.add_source((0.025, 0.025, 0.01), "ez",
                    waveform=GaussianPulse(f0=3e9, bandwidth=0.5))
    sim.add_probe((0.025, 0.025, 0.01), "ez")

    # dx should be None before run
    assert sim._dx is None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.run(n_steps=10)

    # dx should be auto-set after run
    assert sim._dx is not None
    assert sim._dx > 0
    # Auto mesh accounts for material eps_r: finer than simple lambda/20
    # For FR4 (eps_r=4.4) at 5 GHz: lambda_min_medium = 60mm/2.1 ≈ 28.6mm
    # dx ≈ 28.6mm / 20 ≈ 1.4mm (rounded)
    assert 0.1e-3 < sim._dx < 5e-3, f"Auto dx={sim._dx*1e3:.3f}mm seems unreasonable"


def test_dz_profile_grading_warning():
    """User-supplied dz_profile with abrupt grading should warn."""
    import warnings
    from rfx import Simulation

    dz_profile = [0.1e-3] * 3 + [2.0e-3] * 3  # 20x ratio
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Simulation(
            freq_max=5e9,
            domain=(0.05, 0.05),
            boundary="pec",
            dx=2e-3,
            dz_profile=dz_profile,
        )
        grading_warnings = [x for x in w if "cell ratio" in str(x.message)]
        assert len(grading_warnings) >= 1, "Expected grading ratio warning"
