"""Tests for auto_configure source auto-selection."""

from rfx.auto_config import auto_configure


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
