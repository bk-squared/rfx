"""Via / through-hole geometry tests."""

import pytest
from rfx.geometry.via import Via
from rfx.geometry.csg import Box


class TestViaConstruction:
    """Basic construction and validation."""

    def test_basic_construction(self):
        via = Via(
            center=(0.01, 0.01),
            drill_radius=0.15e-3,
            pad_radius=0.3e-3,
            layers=[(0, 0.8e-3), (0.8e-3, 1.6e-3)],
        )
        assert via.center == (0.01, 0.01)
        assert via.drill_radius == 0.15e-3
        assert via.pad_radius == 0.3e-3
        assert via.material == "pec"

    def test_custom_material(self):
        via = Via(
            center=(0, 0),
            drill_radius=0.1e-3,
            pad_radius=0.2e-3,
            layers=[(0, 1e-3)],
            material="copper",
        )
        assert via.material == "copper"

    def test_invalid_drill_radius(self):
        with pytest.raises(ValueError, match="drill_radius must be positive"):
            Via(center=(0, 0), drill_radius=0, pad_radius=0.2e-3, layers=[(0, 1e-3)])

    def test_pad_smaller_than_drill_raises(self):
        with pytest.raises(ValueError, match="pad_radius.*must be >= drill_radius"):
            Via(
                center=(0, 0),
                drill_radius=0.3e-3,
                pad_radius=0.1e-3,
                layers=[(0, 1e-3)],
            )

    def test_empty_layers_raises(self):
        with pytest.raises(ValueError, match="layers must be non-empty"):
            Via(center=(0, 0), drill_radius=0.1e-3, pad_radius=0.2e-3, layers=[])


class TestViaShapes:
    """Shape generation from to_shapes()."""

    def test_two_layers_shape_count(self):
        """Two layers with shared boundary: 1 conductor + 3 unique z-boundaries."""
        via = Via(
            center=(0.01, 0.01),
            drill_radius=0.15e-3,
            pad_radius=0.3e-3,
            layers=[(0, 0.8e-3), (0.8e-3, 1.6e-3)],
        )
        shapes = via.to_shapes()
        # 1 conductor + 3 pads (z=0, z=0.8e-3, z=1.6e-3)
        assert len(shapes) == 4
        for shape, mat in shapes:
            assert isinstance(shape, Box)
            assert mat == "pec"

    def test_conductor_spans_all_layers(self):
        """Vertical conductor should span from z_min to z_max."""
        via = Via(
            center=(0.005, 0.005),
            drill_radius=0.15e-3,
            pad_radius=0.3e-3,
            layers=[(0, 0.8e-3), (0.8e-3, 1.6e-3)],
        )
        shapes = via.to_shapes()
        conductor = shapes[0][0]
        assert conductor.corner_lo[2] == pytest.approx(0.0)
        assert conductor.corner_hi[2] == pytest.approx(1.6e-3)

    def test_conductor_width_matches_drill(self):
        """Conductor cross-section should be 2 * drill_radius."""
        x, y = 0.005, 0.01
        r_drill = 0.15e-3
        via = Via(
            center=(x, y),
            drill_radius=r_drill,
            pad_radius=0.3e-3,
            layers=[(0, 1e-3)],
        )
        conductor = via.to_shapes()[0][0]
        assert conductor.corner_lo[0] == pytest.approx(x - r_drill)
        assert conductor.corner_hi[0] == pytest.approx(x + r_drill)
        assert conductor.corner_lo[1] == pytest.approx(y - r_drill)
        assert conductor.corner_hi[1] == pytest.approx(y + r_drill)

    def test_pad_width_matches_pad_radius(self):
        """Pads should be 2 * pad_radius wide."""
        x, y = 0.01, 0.01
        r_pad = 0.3e-3
        via = Via(
            center=(x, y),
            drill_radius=0.15e-3,
            pad_radius=r_pad,
            layers=[(0, 1e-3)],
        )
        # First pad is shapes[1]
        pad = via.to_shapes()[1][0]
        assert pad.corner_lo[0] == pytest.approx(x - r_pad)
        assert pad.corner_hi[0] == pytest.approx(x + r_pad)
        assert pad.corner_lo[1] == pytest.approx(y - r_pad)
        assert pad.corner_hi[1] == pytest.approx(y + r_pad)

    def test_pads_are_zero_thickness(self):
        """Pads should have corner_lo[z] == corner_hi[z]."""
        via = Via(
            center=(0, 0),
            drill_radius=0.1e-3,
            pad_radius=0.2e-3,
            layers=[(0, 1e-3)],
        )
        shapes = via.to_shapes()
        for shape, _ in shapes[1:]:  # skip conductor
            assert shape.corner_lo[2] == shape.corner_hi[2]

    def test_single_layer(self):
        """Via with single layer: 1 conductor + 2 pads (top and bottom)."""
        via = Via(
            center=(0, 0),
            drill_radius=0.1e-3,
            pad_radius=0.2e-3,
            layers=[(0, 1e-3)],
        )
        shapes = via.to_shapes()
        assert len(shapes) == 3  # 1 conductor + 2 pads

    def test_three_layers(self):
        """Via with three layers: 1 conductor + 4 unique boundary pads."""
        via = Via(
            center=(0, 0),
            drill_radius=0.1e-3,
            pad_radius=0.2e-3,
            layers=[(0, 0.5e-3), (0.5e-3, 1.0e-3), (1.0e-3, 1.5e-3)],
        )
        shapes = via.to_shapes()
        # z boundaries: 0, 0.5e-3, 1.0e-3, 1.5e-3 → 4 unique
        assert len(shapes) == 5  # 1 conductor + 4 pads

    def test_shared_boundary_deduplication(self):
        """Adjacent layers sharing a z-boundary produce only one pad there."""
        via_two = Via(
            center=(0, 0),
            drill_radius=0.1e-3,
            pad_radius=0.2e-3,
            layers=[(0, 1e-3), (1e-3, 2e-3)],
        )
        shapes = via_two.to_shapes()
        # Boundaries: 0, 1e-3, 2e-3 → 3 pads (not 4)
        pad_zs = [s.corner_lo[2] for s, _ in shapes[1:]]
        assert len(pad_zs) == 3
        assert len(set(pad_zs)) == 3  # all unique

    def test_custom_material_propagates(self):
        """All shapes should carry the specified material."""
        via = Via(
            center=(0, 0),
            drill_radius=0.1e-3,
            pad_radius=0.2e-3,
            layers=[(0, 1e-3)],
            material="copper",
        )
        for _, mat in via.to_shapes():
            assert mat == "copper"


class TestViaAlias:
    """to_simulation_items is an alias for to_shapes."""

    def test_alias_matches(self):
        via = Via(
            center=(0.01, 0.01),
            drill_radius=0.15e-3,
            pad_radius=0.3e-3,
            layers=[(0, 0.8e-3)],
        )
        assert via.to_simulation_items() == via.to_shapes()


class TestViaRepr:
    def test_repr(self):
        via = Via(
            center=(0, 0),
            drill_radius=0.1e-3,
            pad_radius=0.2e-3,
            layers=[(0, 1e-3)],
        )
        r = repr(via)
        assert "Via(" in r
        assert "drill_radius" in r
        assert "pad_radius" in r
