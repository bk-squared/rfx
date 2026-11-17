"""Tests for neural surrogate export (rfx.surrogate)."""

import numpy as np
import pytest
import tempfile
import warnings
from pathlib import Path

from rfx.api import Simulation, Result
from rfx.geometry.csg import Box, OrientedBox, PolylineWire
from rfx.geometry.curved import CurvedPatch
from rfx.surrogate import export_training_data, export_geometry_sdf
from rfx.sweep import parametric_sweep, SweepResult


def _make_sim(eps_r=1.0):
    """Minimal PEC cavity with a source near one corner for asymmetric fields."""
    sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), boundary="pec")
    sim.add_material("fill", eps_r=eps_r)
    sim.add(Box((0.005, 0.005, 0.005), (0.025, 0.025, 0.025)), material="fill")
    sim.add_port((0.01, 0.01, 0.01), "ez")
    sim.add_probe((0.02, 0.02, 0.02), "ez")
    return sim


class TestExportTrainingData:
    """Tests for export_training_data."""

    def test_export_training_data_npz(self):
        """Should create a valid .npz file from a SweepResult.
        Outputs have axes (n_samples, n_steps, n_probes).
        """
        EXPORT_RECORD_STEPS = 30
        values = [2.0, 4.0]
        sr = parametric_sweep(
            _make_sim,
            param_name="eps_r",
            param_values=values,
            n_steps=EXPORT_RECORD_STEPS,
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

            assert outputs.shape[0] == 2
            assert outputs.shape[1] == EXPORT_RECORD_STEPS
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

    @pytest.mark.parametrize("material", ["pec", "glass"])
    def test_sdf_snaps_nonzero_thin_box(self, material):
        """The 35 um slab's midpoint 15.0175 mm is closer to sample 15 (15*30/29 mm), so Box snaps."""
        sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), boundary="pec")
        sim.add_material("glass", eps_r=4.0)
        sim.add(Box((0.010, 0.010, 0.015), (0.020, 0.020, 0.015035)),
                material=material)
        sdf = export_geometry_sdf(sim)
        assert sdf[15, 15, 15] < 0
        np.testing.assert_array_equal(np.flatnonzero((sdf < 0).any(axis=(0, 1))), [15])

    @pytest.mark.parametrize("material", ["pec", "glass"])
    def test_sdf_refuses_unsampled_thin_oriented_box(self, material):
        """An already occupied region must not conceal the next entry's loss."""
        sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), boundary="pec")
        sim.add_material("glass", eps_r=4.0)
        sim.add(Box((0.002, 0.002, 0.002), (0.006, 0.006, 0.006)), material=material)
        sim.add(OrientedBox(center=(0.015, 0.015, 0.0150175),
                            size=(0.010, 0.010, 35e-6)), material=material)
        with pytest.raises(ValueError, match=(
            r"geometry\[1\] \(OrientedBox\) realizes no SDF sample at "
            "resolution=0.001 m; it would be missing — refine resolution"
        )):
            export_geometry_sdf(sim)

    def test_sdf_box_and_identity_oriented_box_occupancy_agree(self):
        """Binary-exact sample spacing 1/512 m places the faces at samples 4 and 12."""
        domain = (0.03125,) * 3
        shapes = [Box((0.0078125,) * 3, (0.0234375,) * 3),
                  OrientedBox(center=(0.015625,) * 3, size=(0.015625,) * 3)]
        masks = []
        for shape in shapes:
            sim = Simulation(freq_max=5e9, domain=domain, boundary="pec")
            sim.add(shape, material="pec")
            masks.append(export_geometry_sdf(sim, resolution=0.03125 / 17) < 0)
        np.testing.assert_array_equal(masks[0], masks[1])
        assert masks[0][4, 4, 4]
        assert not masks[0][12, 8, 8]

    def test_sdf_distance_uses_actual_sample_spacing(self):
        """The distance at (13,15,15) is three x samples from the only occupied x sample, 10."""
        sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), boundary="pec")
        x_plane = 10 * 0.03 / 29
        sim.add(Box((x_plane - 17.5e-6, 0.010, 0.010),
                    (x_plane + 17.5e-6, 0.020, 0.020)), material="pec")
        sdf = export_geometry_sdf(sim, resolution=1e-3)
        assert sdf[13, 15, 15] == pytest.approx(3 * 0.03 / 29, rel=0, abs=1e-12)

    def test_sdf_sees_curved_patch_occupancy(self):
        """At 1 mm sample spacing, (15,20,15) mm is inside the radius-10 mm sphere but outside the patch's y interval [13,17] mm given its 4 mm width."""
        sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), boundary="pec")
        sim.add(CurvedPatch(
            center=(0.015, 0.015, 0.015), length=0.008, width=0.004,
            radius=0.010, axis="x",
        ), material="pec")
        sdf = export_geometry_sdf(sim, resolution=0.03 / 31)
        assert sdf[15, 20, 15] > 0
        assert np.any(sdf < 0), "the curved patch must also appear in the SDF"

    def test_sdf_refuses_coaxial_port(self):
        sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), boundary="pec")
        sim.add_coaxial_port((0.015, 0.015, 0.03))
        with pytest.raises(ValueError, match="coaxial"):
            export_geometry_sdf(sim)

    @pytest.mark.parametrize("with_corners", [False, True])
    def test_sdf_sees_oriented_box(self, with_corners):
        """At 1 mm sample spacing, (19,11,15) has body coordinates (0,-4*sqrt(2),0) mm, outside the rotated box but inside its axis-aligned bounding box."""
        class OrientedBoxWithCorners(OrientedBox):
            @property
            def corner_lo(self):
                return self.bounding_box()[0]

            @property
            def corner_hi(self):
                return self.bounding_box()[1]

        sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), boundary="pec")
        c = np.sqrt(0.5)
        shape_type = OrientedBoxWithCorners if with_corners else OrientedBox
        sim.add(shape_type(
            center=(0.015, 0.015, 0.015), size=(0.016, 0.004, 0.008),
            rotation=((c, -c, 0), (c, c, 0), (0, 0, 1)),
        ), material="pec")

        sdf = export_geometry_sdf(sim, resolution=0.03 / 31)
        assert sdf[19, 19, 15] < 0
        assert sdf[19, 11, 15] > 0

    def test_sdf_sees_polyline_wire(self):
        """At 1 mm sample spacing, (15,19,15) mm is 4 mm from the segment, beyond its 2 mm radius."""
        sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), boundary="pec")
        sim.add(PolylineWire(
            points=((0.005, 0.015, 0.015), (0.025, 0.015, 0.015)),
            radius=0.002,
        ), material="pec")
        sdf = export_geometry_sdf(sim, resolution=0.03 / 31)
        assert sdf[15, 16, 15] < 0
        assert sdf[15, 19, 15] > 0

    @pytest.mark.parametrize("failure", ["missing", "raises", "wrong_shape"])
    def test_sdf_refuses_unevaluable_shape(self, failure):
        class UnevaluableShape:
            pass

        shape = UnevaluableShape()
        if failure == "raises":
            def mask_on_coords(x, y, z):
                raise RuntimeError("cannot sample this shape")
            shape.mask_on_coords = mask_on_coords
        elif failure == "wrong_shape":
            shape.mask_on_coords = lambda x, y, z: np.ones((1,), dtype=bool)
        sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), boundary="pec")
        sim.add(shape, material="pec")
        with pytest.raises(ValueError, match=(
            r"geometry\[0\] \(UnevaluableShape\) cannot be evaluated on the SDF "
            "samples; it would be missing from the exported geometry"
        )):
            export_geometry_sdf(sim)

    def test_sdf_refuses_pinned_sheet(self):
        sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), boundary="pec")
        sim.add_pinned_sheet(plane_index=5, i_range=(2, 8), j_range=(2, 8))
        with pytest.raises(ValueError, match="pinned"):
            export_geometry_sdf(sim)

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

        assert np.any(sdf < 0), "SDF should have negative values inside geometry"
        assert np.any(sdf > 0), "SDF should have positive values outside geometry"

        cx, cy, cz = 15, 15, 15
        assert sdf[cx, cy, cz] < 0, "SDF at box centre should be negative"

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
        The exporter samples on ``linspace(0, L, ceil(L/resolution))``.
        """
        sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03),
                         boundary="pec")
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
        inside = np.asarray(sdf < 0)
        planes = sorted(set(np.flatnonzero(inside.any(axis=(0, 1))).tolist()))
        assert planes == [15], planes
        xs = np.linspace(0, 0.03, 30)
        want = int(((xs >= 0.010 - 1e-12) & (xs <= 0.020 + 1e-12)).sum())
        assert int(inside[:, :, 15].sum()) == want * want, inside[:, :, 15].sum()

    def test_export_geometry_sdf_refuses_a_sheet_it_cannot_place(self):
        """A sheet too small for the sample pitch raises, never vanishes.

        The whole defect this closes is a silent drop, so the resolution
        failure mode must not be one either. The message names the knob.
        No sample lands in the footprint of the 100 um square patch between two 1 mm samples.
        """
        sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03),
                         boundary="pec")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
