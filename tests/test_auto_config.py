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

    # Bounded config should fit within budget
    assert config_bounded.estimated_memory_mb <= budget_mb, (
        f"Bounded memory ({config_bounded.estimated_memory_mb:.1f} MB) exceeds "
        f"budget ({budget_mb:.1f} MB)"
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
