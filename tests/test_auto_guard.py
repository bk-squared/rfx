"""Tests for empty-geometry guard in Simulation.auto() and auto_configure."""

import warnings


from rfx import Simulation
from rfx.auto_config import auto_configure


class TestAutoGuard:
    """Verify that empty geometry triggers warnings and keeps grid small."""

    def test_auto_empty_geometry_warns(self):
        """Simulation.auto() with no geometry must emit a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Simulation.auto(freq_range=(1e9, 4e9))
            geo_warnings = [
                x for x in w
                if "geometry" in str(x.message).lower()
                or "domain" in str(x.message).lower()
            ]
            assert len(geo_warnings) >= 1

    def test_auto_empty_geometry_reasonable_size(self):
        """Empty geometry grid must stay well under 10M cells."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = Simulation.auto(freq_range=(1e9, 4e9))
        grid = sim._build_grid()
        total_cells = grid.shape[0] * grid.shape[1] * grid.shape[2]
        assert total_cells < 10_000_000, f"Empty geometry: {total_cells:,} cells"

    def test_auto_configure_empty_geometry_warning(self):
        """auto_configure with [] geometry must emit a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            auto_configure([], freq_range=(1e9, 4e9))
            geo_warnings = [
                x for x in w
                if "empty geometry" in str(x.message).lower()
            ]
            assert len(geo_warnings) >= 1

    def test_auto_configure_empty_geometry_config_warnings(self):
        """SimConfig.warnings should mention empty geometry."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = auto_configure([], freq_range=(1e9, 4e9))
        assert any("empty geometry" in w.lower() for w in config.warnings)

    def test_auto_configure_empty_geometry_coarse_dx(self):
        """Empty geometry should use coarser dx than standard cpw=20 rule."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = auto_configure([], freq_range=(1e9, 4e9))
        # Standard accuracy would use lambda_min/20 = 3.75 mm.
        # Empty-geometry guard should pick lambda_min/10 = 7.5 mm (rounded
        # down by _round_dx to 5.0 mm). Either way, must be coarser than
        # the standard cpw=20 value.
        lambda_min = 299792458.0 / 4e9
        dx_standard = lambda_min / 20  # ~3.75 mm
        assert config.dx > dx_standard, (
            f"dx={config.dx*1e3:.3f} mm should be coarser than "
            f"standard {dx_standard*1e3:.3f} mm for empty geometry"
        )

    def test_auto_configure_draft_empty_still_reasonable(self):
        """Even draft accuracy with empty geometry should be small."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = auto_configure([], freq_range=(1e9, 10e9), accuracy="draft")
        nx = int(config.domain[0] / config.dx)
        ny = int(config.domain[1] / config.dx)
        nz = int(config.domain[2] / config.dx)
        total = nx * ny * nz
        assert total < 10_000_000, f"Draft empty geometry: {total:,} cells"

    def test_auto_configure_high_empty_still_reasonable(self):
        """Even high accuracy with empty geometry should stay bounded."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = auto_configure([], freq_range=(1e9, 10e9), accuracy="high")
        nx = int(config.domain[0] / config.dx)
        ny = int(config.domain[1] / config.dx)
        nz = int(config.domain[2] / config.dx)
        total = nx * ny * nz
        assert total < 10_000_000, f"High empty geometry: {total:,} cells"
