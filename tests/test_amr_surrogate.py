"""Tests for pre-AMR error indicator (rfx.amr) and neural surrogate export (rfx.surrogate)."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from rfx.api import Simulation, Result
from rfx.geometry.csg import Box
from rfx.amr import compute_error_indicator, suggest_refinement_regions, auto_refine
from rfx.surrogate import export_training_data, export_geometry_sdf
from rfx.sweep import parametric_sweep, SweepResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sim(eps_r=1.0):
    """Minimal PEC cavity with a source near one corner for asymmetric fields."""
    sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), boundary="pec")
    sim.add_material("fill", eps_r=eps_r)
    sim.add(Box((0.005, 0.005, 0.005), (0.025, 0.025, 0.025)), material="fill")
    # Source off-centre so the gradient is non-uniform
    sim.add_port((0.01, 0.01, 0.01), "ez")
    sim.add_probe((0.02, 0.02, 0.02), "ez")
    return sim


# ---------------------------------------------------------------------------
# Feature A: AMR error indicator
# ---------------------------------------------------------------------------

class TestErrorIndicator:
    """Tests for compute_error_indicator."""

    def test_error_indicator_nonzero(self):
        """Error indicator should be nonzero after a simulation with a source."""
        sim = _make_sim()
        result = sim.run(n_steps=50, compute_s_params=False)

        error = compute_error_indicator(result, component="ez")

        assert error.shape == result.state.ez.shape
        assert error.dtype == np.float64
        assert np.max(error) > 0, "Error indicator must be nonzero near the source"
        # Normalized to [0, 1]
        assert np.max(error) == pytest.approx(1.0)
        assert np.min(error) >= 0.0

    def test_error_indicator_different_components(self):
        """Should work for any field component."""
        sim = _make_sim()
        result = sim.run(n_steps=50, compute_s_params=False)

        for comp in ("ex", "ey", "ez", "hx", "hy", "hz"):
            error = compute_error_indicator(result, component=comp)
            assert error.shape == result.state.ez.shape


class TestSuggestRegions:
    """Tests for suggest_refinement_regions."""

    def test_suggest_regions_finds_source_area(self):
        """Should find at least one refinement region near the source."""
        sim = _make_sim()
        result = sim.run(n_steps=50, compute_s_params=False)

        error = compute_error_indicator(result, component="ez")
        boxes = suggest_refinement_regions(error, threshold=0.3, min_region_size=2)

        assert len(boxes) >= 1, "Should find at least one high-error region"
        for b in boxes:
            assert isinstance(b, Box)
            # Bounding box should have non-negative extents
            for i in range(3):
                assert b.corner_hi[i] >= b.corner_lo[i]

    def test_suggest_regions_with_grid(self):
        """When grid is provided, boxes should be in physical coordinates."""
        sim = _make_sim()
        grid = sim._build_grid()
        result = sim.run(n_steps=50, compute_s_params=False)

        error = compute_error_indicator(result, component="ez")
        boxes = suggest_refinement_regions(
            error, grid=grid, threshold=0.3, min_region_size=2,
        )

        assert len(boxes) >= 1
        # Physical coordinates should be in the domain range (allowing CPML offset)
        for b in boxes:
            for i in range(3):
                assert b.corner_lo[i] < b.corner_hi[i] or \
                    b.corner_lo[i] == b.corner_hi[i]

    def test_suggest_regions_empty_on_high_threshold(self):
        """A threshold of 1.0 should return no regions (max is exactly 1.0)."""
        sim = _make_sim()
        result = sim.run(n_steps=50, compute_s_params=False)

        error = compute_error_indicator(result, component="ez")
        boxes = suggest_refinement_regions(error, threshold=1.0)

        assert boxes == []

    def test_suggest_regions_empty_for_zero_field(self):
        """Zero error map should yield no regions."""
        error = np.zeros((20, 20, 20))
        boxes = suggest_refinement_regions(error, threshold=0.5)
        assert boxes == []


# ---------------------------------------------------------------------------
# Feature B: Neural surrogate data export
# ---------------------------------------------------------------------------

class TestExportTrainingData:
    """Tests for export_training_data."""

    def test_export_training_data_npz(self):
        """Should create a valid .npz file from a SweepResult."""
        values = [2.0, 4.0]
        sr = parametric_sweep(
            _make_sim,
            param_name="eps_r",
            param_values=values,
            n_steps=30,
            run_kwargs={"compute_s_params": False},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "train.npz"
            returned = export_training_data(sr, output_path=out_path)

            assert returned == out_path
            assert out_path.exists()

            data = np.load(str(out_path), allow_pickle=True)
            assert "inputs" in data
            assert "outputs" in data
            assert "param_name" in data

            inputs = data["inputs"]
            outputs = data["outputs"]

            assert inputs.shape == (2, 1)
            np.testing.assert_array_almost_equal(inputs[:, 0], [2.0, 4.0])

            # outputs: (n_samples, n_steps, n_probes)
            assert outputs.shape[0] == 2
            assert outputs.shape[1] == 30  # n_steps
            assert outputs.ndim == 3

    def test_export_unsupported_format(self):
        """Should raise ValueError for unsupported formats."""
        values = [2.0]
        sr = parametric_sweep(
            _make_sim,
            param_name="eps_r",
            param_values=values,
            n_steps=10,
            run_kwargs={"compute_s_params": False},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unsupported format"):
                export_training_data(sr, output_path=Path(tmpdir) / "x.h5", format="hdf5")


class TestExportGeometrySDF:
    """Tests for export_geometry_sdf."""

    def test_export_geometry_sdf(self):
        """SDF should be negative inside geometry, positive outside."""
        sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), boundary="pec")
        sim.add_material("block", eps_r=4.0)
        sim.add(
            Box((0.010, 0.010, 0.010), (0.020, 0.020, 0.020)),
            material="block",
        )

        sdf = export_geometry_sdf(sim, resolution=1e-3)

        assert sdf.ndim == 3
        assert sdf.dtype == np.float64

        # Check that some cells are negative (inside) and some positive (outside)
        assert np.any(sdf < 0), "SDF should have negative values inside geometry"
        assert np.any(sdf > 0), "SDF should have positive values outside geometry"

        # The centre of the box should be clearly inside (negative SDF)
        # resolution=1e-3, domain=0.03 → 30 cells; box from 10mm to 20mm
        cx, cy, cz = 15, 15, 15  # centre of box in grid indices
        assert sdf[cx, cy, cz] < 0, "SDF at box centre should be negative"

        # A corner far from the box should be positive
        assert sdf[0, 0, 0] > 0, "SDF at domain corner should be positive"

    def test_sdf_shape_matches_resolution(self):
        """SDF grid shape should match domain/resolution."""
        sim = Simulation(freq_max=5e9, domain=(0.02, 0.03, 0.01), boundary="pec")
        sdf = export_geometry_sdf(sim, resolution=1e-3)

        assert sdf.shape == (20, 30, 10)
