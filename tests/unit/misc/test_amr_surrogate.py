"""Tests for pre-AMR error indicator (rfx.amr) and neural surrogate export (rfx.surrogate)."""

import numpy as np
import pytest
import tempfile
import warnings
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

    def test_export_geometry_sdf_sees_a_sheet_declared_conductor(self):
        """A sheet must not vanish from the exported training data (#931).

        A sheet is a footprint on ONE node plane with zero thickness, so it
        has no interior and a naive "inside the shape" SDF is degenerate
        for it. A zero-thickness PEC Box IS the sheet declaration (design
        note §1.5), so the exporter realizes it the way the contract does:
        one sample layer, on the sample plane nearest its declared plane.
        This test pins that, because a silent drop would take every
        sheet-declared conductor out of the surrogate training set with no
        error anywhere.

        The ``add_thin_conductor`` twin of this gap is closed by
        ``test_export_geometry_sdf_sees_an_add_thin_conductor_sheet`` below.
        """
        sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03),
                         boundary="pec")
        # The exporter samples on ``linspace(0, L, ceil(L/resolution))``,
        # so pick a sheet plane that IS a sample point.
        z_sheet = float(np.linspace(0, 0.03, 30)[15])
        sim.add(Box((0.010, 0.010, z_sheet), (0.020, 0.020, z_sheet)),
                material="pec")

        sdf = export_geometry_sdf(sim, resolution=1e-3)

        assert np.any(sdf < 0), (
            "a sheet-declared conductor vanished from the exported SDF")
        assert sdf[15, 15, 15] < 0, "the sheet's own plane must read inside"

    def test_export_geometry_sdf_sees_an_add_thin_conductor_sheet(self):
        """An ``add_thin_conductor`` sheet must reach the exported SDF (#931).

        Sheets declared this way are not in ``sim._geometry``, and the
        exporter walked only that list — so a sheet ground plane, patch or
        trace was absent from the exported training data with no error
        anywhere. It is now realized the way the contract realizes it: one
        sample layer, on the sample plane nearest the declared mid-plane,
        with the drawn footprint sampled closed in-plane.

        The foil here is 35 um thick against a 1 mm sample pitch, i.e. it
        falls BETWEEN samples — the case a containment test cannot catch
        and the reason the plane is snapped rather than tested.
        """
        sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03),
                         boundary="pec")
        z_plane = float(np.linspace(0, 0.03, 30)[15])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.add_thin_conductor(
                Box((0.010, 0.010, z_plane), (0.020, 0.020, z_plane + 35e-6)),
                sigma_bulk=5.8e7, thickness=35e-6)

        sdf = export_geometry_sdf(sim, resolution=1e-3)

        assert np.any(sdf < 0), (
            "an add_thin_conductor sheet vanished from the exported SDF")
        # ONE sample layer, on the plane nearest the declared mid-plane
        # (mid = z_plane + 17.5 um, well inside the lower half-sample).
        inside = np.asarray(sdf < 0)
        planes = sorted(set(np.flatnonzero(inside.any(axis=(0, 1))).tolist()))
        assert planes == [15], planes
        # ... and the footprint is the drawn rectangle, sampled closed:
        # x, y run 0 .. 30 mm on 30 samples (pitch 30/29 mm), so the closed
        # [10, 20] mm window is exactly the samples inside it.
        xs = np.linspace(0, 0.03, 30)
        want = int(((xs >= 0.010 - 1e-12) & (xs <= 0.020 + 1e-12)).sum())
        assert int(inside[:, :, 15].sum()) == want * want, inside[:, :, 15].sum()

    def test_export_geometry_sdf_refuses_a_sheet_it_cannot_place(self):
        """A sheet too small for the sample pitch raises, never vanishes.

        The whole defect this closes is a silent drop, so the resolution
        failure mode must not be one either. The message names the knob.
        """
        sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03),
                         boundary="pec")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 100 um square patch between two 1 mm samples: no sample lands
            # in its footprint.
            sim.add_thin_conductor(
                Box((0.0154, 0.0154, 0.0155), (0.0155, 0.0155, 0.0155)),
                sigma_bulk=5.8e7, thickness=35e-6)
        with pytest.raises(ValueError, match=r"resolution="):
            export_geometry_sdf(sim, resolution=1e-3)

    def test_sdf_shape_matches_resolution(self):
        """SDF grid shape should match domain/resolution."""
        sim = Simulation(freq_max=5e9, domain=(0.02, 0.03, 0.01), boundary="pec")
        sdf = export_geometry_sdf(sim, resolution=1e-3)

        assert sdf.shape == (20, 30, 10)
