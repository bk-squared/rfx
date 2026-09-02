"""Curved patch geometry tests."""

import numpy as np
import pytest

from rfx.geometry.curved import CurvedPatch
from rfx.geometry.csg import Box


class TestCurvedPatch:
    def test_creates_multiple_segments(self):
        """Curved patch generates multiple box slices."""
        patch = CurvedPatch(
            center=(0, 0, 0.02),
            length=38e-3,
            width=29e-3,
            radius=0.05,
            axis="x",
        )
        boxes = patch.to_staircase(dx=1e-3)
        assert len(boxes) >= 2
        assert all(isinstance(b, Box) for b in boxes)

    def test_segment_count_matches_resolution(self):
        """Number of segments ~ length / dx."""
        patch = CurvedPatch(
            center=(0, 0, 0),
            length=40e-3,
            width=20e-3,
            radius=0.1,
            axis="x",
        )
        boxes = patch.to_staircase(dx=1e-3)
        assert len(boxes) == 40  # 40mm / 1mm

    def test_segments_have_correct_width(self):
        """Each segment should span the full patch width in the non-curvature axis."""
        patch = CurvedPatch(
            center=(0, 0, 0),
            length=20e-3,
            width=10e-3,
            radius=0.05,
            axis="x",
        )
        boxes = patch.to_staircase(dx=1e-3)
        for box in boxes:
            # For x-axis curvature, width is along y
            y_extent = box.corner_hi[1] - box.corner_lo[1]
            assert abs(y_extent - 10e-3) < 1e-10, f"y-width should be 10mm, got {y_extent}"

    def test_curvature_creates_z_variation(self):
        """Segments at edges should be higher (z) than center."""
        patch = CurvedPatch(
            center=(0, 0, 0),
            length=20e-3,
            width=10e-3,
            radius=0.05,
            axis="x",
        )
        boxes = patch.to_staircase(dx=1e-3)
        z_values = [b.corner_lo[2] for b in boxes]
        # Center segments have lowest z, edge segments highest
        center_z = z_values[len(z_values) // 2]
        edge_z = z_values[0]
        assert edge_z > center_z, "Edge segments should be higher than center"

    def test_flat_patch_when_large_radius(self):
        """Very large radius ~ flat patch (all z nearly equal)."""
        patch = CurvedPatch(
            center=(0, 0, 0),
            length=20e-3,
            width=10e-3,
            radius=100.0,  # Very large
            axis="x",
        )
        boxes = patch.to_staircase(dx=1e-3)
        z_values = [b.corner_lo[2] for b in boxes]
        z_range = max(z_values) - min(z_values)
        assert z_range < 1e-6, f"Large radius should give flat patch, got range {z_range}"

    def test_y_axis_curvature(self):
        """Curvature along y-axis should vary segment y positions and z heights."""
        patch = CurvedPatch(
            center=(0, 0, 0),
            length=20e-3,
            width=10e-3,
            radius=0.05,
            axis="y",
        )
        boxes = patch.to_staircase(dx=1e-3)
        # For y-axis curvature, z still varies
        z_values = [b.corner_lo[2] for b in boxes]
        assert max(z_values) > min(z_values)

    def test_invalid_axis_raises(self):
        """Invalid axis should raise ValueError."""
        patch = CurvedPatch(
            center=(0, 0, 0),
            length=20e-3,
            width=10e-3,
            radius=0.05,
            axis="z",
        )
        with pytest.raises(ValueError):
            patch.to_staircase(dx=1e-3)

    def test_segments_tile_curvature_axis(self):
        """Segment boxes should tile the full length along the curvature axis."""
        patch = CurvedPatch(
            center=(0, 0, 0),
            length=20e-3,
            width=10e-3,
            radius=0.05,
            axis="x",
        )
        boxes = patch.to_staircase(dx=1e-3)
        # First segment should start at -length/2, last should end at +length/2
        x_lo = min(b.corner_lo[0] for b in boxes)
        x_hi = max(b.corner_hi[0] for b in boxes)
        assert abs(x_lo - (-10e-3)) < 1e-12
        assert abs(x_hi - 10e-3) < 1e-12

    def test_thin_z_extent(self):
        """Each segment is a thin sheet (corner_lo[2] == corner_hi[2])."""
        patch = CurvedPatch(
            center=(0, 0, 0),
            length=20e-3,
            width=10e-3,
            radius=0.05,
            axis="x",
        )
        boxes = patch.to_staircase(dx=1e-3)
        for box in boxes:
            assert box.corner_lo[2] == box.corner_hi[2], "Segment should be a thin sheet"

    def test_z_offset_formula(self):
        """Verify the arc height formula: z = R - sqrt(R^2 - s^2)."""
        radius = 0.05
        length = 20e-3
        patch = CurvedPatch(
            center=(0, 0, 0),
            length=length,
            width=10e-3,
            radius=radius,
            axis="x",
        )
        boxes = patch.to_staircase(dx=1e-3)
        n = len(boxes)
        seg_w = length / n
        half_len = length / 2
        for i, box in enumerate(boxes):
            s = -half_len + (i + 0.5) * seg_w
            expected_z = radius - np.sqrt(radius**2 - s**2)
            assert abs(box.corner_lo[2] - expected_z) < 1e-15
