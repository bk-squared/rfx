"""Tests for the PCB stackup builder (rfx.pcb)."""

import math
import pytest

from rfx.pcb import PCBLayer, Stackup, resolve_pcb_material
from rfx.geometry.csg import Box


# ---------------------------------------------------------------------------
# PCBLayer basics
# ---------------------------------------------------------------------------

class TestPCBLayer:
    def test_defaults(self):
        layer = PCBLayer(thickness=0.035e-3)
        assert layer.material == "copper"
        assert layer.name is None

    def test_custom(self):
        layer = PCBLayer(thickness=1.0e-3, material="fr4", name="core")
        assert layer.thickness == 1.0e-3
        assert layer.material == "fr4"
        assert layer.name == "core"


# ---------------------------------------------------------------------------
# Material alias resolution
# ---------------------------------------------------------------------------

class TestMaterialAlias:
    def test_prepreg_alias(self):
        assert resolve_pcb_material("prepreg") == "fr4"

    def test_passthrough(self):
        assert resolve_pcb_material("copper") == "copper"
        assert resolve_pcb_material("rogers4003c") == "rogers4003c"


# ---------------------------------------------------------------------------
# Stackup
# ---------------------------------------------------------------------------

class TestStackup:
    """Core Stackup tests."""

    @pytest.fixture
    def simple_stackup(self):
        """A minimal 3-layer stackup for testing."""
        return Stackup([
            PCBLayer(thickness=0.035e-3, material="copper", name="bottom"),
            PCBLayer(thickness=1.6e-3, material="fr4", name="substrate"),
            PCBLayer(thickness=0.035e-3, material="copper", name="top"),
        ])

    @pytest.fixture
    def four_layer_stackup(self):
        return Stackup([
            PCBLayer(thickness=0.035e-3, material="copper", name="bottom"),
            PCBLayer(thickness=0.2e-3, material="prepreg"),
            PCBLayer(thickness=0.035e-3, material="copper", name="inner2"),
            PCBLayer(thickness=1.0e-3, material="fr4", name="core"),
            PCBLayer(thickness=0.035e-3, material="copper", name="inner1"),
            PCBLayer(thickness=0.2e-3, material="prepreg"),
            PCBLayer(thickness=0.035e-3, material="copper", name="top"),
        ])

    # -- total thickness ---------------------------------------------------

    def test_stackup_total_thickness(self, simple_stackup):
        expected = 0.035e-3 + 1.6e-3 + 0.035e-3
        assert math.isclose(simple_stackup.total_thickness, expected, rel_tol=1e-12)

    def test_stackup_total_thickness_4layer(self, four_layer_stackup):
        expected = 4 * 0.035e-3 + 2 * 0.2e-3 + 1.0e-3
        assert math.isclose(four_layer_stackup.total_thickness, expected, rel_tol=1e-12)

    # -- to_shapes count ---------------------------------------------------

    def test_stackup_to_shapes_count(self, simple_stackup):
        shapes = simple_stackup.to_shapes(center_xy=(0, 0), size_xy=(0.02, 0.02))
        assert len(shapes) == 3

    def test_stackup_to_shapes_count_4layer(self, four_layer_stackup):
        shapes = four_layer_stackup.to_shapes()
        assert len(shapes) == 7

    # -- z positions -------------------------------------------------------

    def _layer_z_bounds(self, stack):
        z = -stack.total_thickness / 2.0
        out = []
        for layer in stack.layers:
            out.append((z, z + layer.thickness))
            z = z + layer.thickness
        return out

    def test_stackup_z_positions(self, simple_stackup):
        """Dielectric layers tile z-space; conductor layers are SHEETS.

        #931 §1.5: a 35 um copper foil is thinner than any cell a board
        simulation uses, and a PEC Box with 0 < extent < one local cell is
        refused. ``to_shapes`` therefore emits each conductor layer as a
        ZERO-THICKNESS Box.  §4 rule 2 says WHERE: on the dielectric
        INTERFACE the foil bounds, not on its mid-plane.  On the mid-plane
        the sheet sits half a foil thickness off the laminate, and the mesh
        that puts a line at both the interface and the sheet opens a
        17.5 um vacuum cell in series with the board (the #702 slot, and a
        ~23x dt collapse on a standard 2-layer stack-up).  The stack's own
        z arithmetic (``total_thickness``, ``get_layer_z``) is unchanged, so
        the dielectric boxes still tile the interior continuously.
        """
        stack = simple_stackup
        shapes = stack.to_shapes()
        total = stack.total_thickness
        bounds = self._layer_z_bounds(stack)
        is_cond = [la.material == "copper" for la in stack.layers]

        z = -total / 2.0
        for idx, (layer, (box, _mat)) in enumerate(zip(stack.layers, shapes)):
            z_lo, z_hi = z, z + layer.thickness
            assert (z_lo, z_hi) == bounds[idx]
            if is_cond[idx]:
                if idx > 0 and not is_cond[idx - 1]:
                    z_sheet = z_lo
                elif idx + 1 < len(stack.layers) and not is_cond[idx + 1]:
                    z_sheet = z_hi
                else:
                    z_sheet = 0.5 * (z_lo + z_hi)
                assert math.isclose(box.corner_lo[2], z_sheet, rel_tol=1e-12)
                assert math.isclose(box.corner_hi[2], z_sheet, rel_tol=1e-12)
            else:
                assert math.isclose(box.corner_lo[2], z_lo, rel_tol=1e-12)
                assert math.isclose(box.corner_hi[2], z_hi, rel_tol=1e-12)
            z = z_hi
        assert math.isclose(z, total / 2.0, abs_tol=1e-15)

    def test_foil_sheets_land_on_the_dielectric_faces(self, simple_stackup):
        """No foil sits in vacuum beside its laminate (#702 slot class).

        Every conductor sheet plane must coincide with a face of an
        adjacent dielectric box, so a mesh line at the interface IS the
        sheet's line and no sliver cell is created between the two.
        """
        shapes = simple_stackup.to_shapes()
        faces = set()
        for layer, (box, _mat) in zip(simple_stackup.layers, shapes):
            if layer.material != "copper":
                faces.add(round(box.corner_lo[2], 15))
                faces.add(round(box.corner_hi[2], 15))
        for layer, (box, _mat) in zip(simple_stackup.layers, shapes):
            if layer.material == "copper":
                assert round(box.corner_lo[2], 15) in faces, (
                    f"foil {layer.name} at z={box.corner_lo[2]} is not on a "
                    f"dielectric face {sorted(faces)}")

    def test_four_layer_foils_land_on_the_dielectric_faces(
            self, four_layer_stackup):
        shapes = four_layer_stackup.to_shapes()
        faces = set()
        for layer, (box, _mat) in zip(four_layer_stackup.layers, shapes):
            if layer.material != "copper":
                faces.add(round(box.corner_lo[2], 15))
                faces.add(round(box.corner_hi[2], 15))
        for layer, (box, _mat) in zip(four_layer_stackup.layers, shapes):
            if layer.material == "copper":
                assert round(box.corner_lo[2], 15) in faces

    def test_conductor_layers_are_sheet_declarations(self, simple_stackup):
        """Every copper layer is a zero-thickness Box (a sheet, #931)."""
        for layer, (box, mat) in zip(simple_stackup.layers,
                                     simple_stackup.to_shapes()):
            is_zero = box.corner_hi[2] == box.corner_lo[2]
            assert is_zero == (mat == "copper"), (
                f"layer {layer.name!r} ({mat}): zero-thickness={is_zero}")

    def test_stackup_layers_realize_as_sheets_on_a_simulation(self):
        """A Stackup added to a Simulation collects one PEC sheet per foil.

        Before #931 a 35 um copper Box on any realistic dx rasterized to
        zero cells (silently) or, at HEAD, is refused as a sub-cell volume.
        The sheet declaration is what makes the public class usable.
        """
        import warnings

        import rfx

        stack = Stackup.standard_2layer()
        sim = rfx.Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.01),
                             dx=2e-4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for box, mat in stack.to_shapes(size_xy=(0.01, 0.01)):
                sim.add(box, material=mat)
            sheets: list = []
            _mats, _, _, pec_mask, *_ = sim._assemble_materials(
                sim._build_grid(), pec_sheets=sheets)
        assert len(sheets) == 2, "one sheet per copper layer"
        assert all(sp.normal_axis == 2 for sp in sheets)
        assert pec_mask is None, "a sheet owns no cell"

    # -- to_shapes geometry details ----------------------------------------

    def test_to_shapes_xy_extent(self, simple_stackup):
        cx, cy = 0.005, -0.003
        sx, sy = 0.04, 0.03
        shapes = simple_stackup.to_shapes(center_xy=(cx, cy), size_xy=(sx, sy))
        for box, _ in shapes:
            assert math.isclose(box.corner_lo[0], cx - sx / 2, rel_tol=1e-12)
            assert math.isclose(box.corner_hi[0], cx + sx / 2, rel_tol=1e-12)
            assert math.isclose(box.corner_lo[1], cy - sy / 2, rel_tol=1e-12)
            assert math.isclose(box.corner_hi[1], cy + sy / 2, rel_tol=1e-12)

    def test_to_shapes_material_names(self, simple_stackup):
        shapes = simple_stackup.to_shapes()
        materials = [mat for _, mat in shapes]
        assert materials == ["copper", "fr4", "copper"]

    def test_to_shapes_prepreg_resolved(self, four_layer_stackup):
        """Prepreg layers should resolve to 'fr4' in the output."""
        shapes = four_layer_stackup.to_shapes()
        materials = [mat for _, mat in shapes]
        # Indices 1 and 5 are prepreg layers
        assert materials[1] == "fr4"
        assert materials[5] == "fr4"

    def test_to_shapes_returns_box_instances(self, simple_stackup):
        shapes = simple_stackup.to_shapes()
        for box, _ in shapes:
            assert isinstance(box, Box)

    # -- get_layer_z -------------------------------------------------------

    def test_get_layer_z(self, simple_stackup):
        z_lo, z_hi = simple_stackup.get_layer_z("substrate")
        total = simple_stackup.total_thickness
        expected_lo = -total / 2 + 0.035e-3
        expected_hi = expected_lo + 1.6e-3
        assert math.isclose(z_lo, expected_lo, rel_tol=1e-12)
        assert math.isclose(z_hi, expected_hi, rel_tol=1e-12)

    def test_get_layer_z_first(self, simple_stackup):
        z_lo, z_hi = simple_stackup.get_layer_z("bottom")
        total = simple_stackup.total_thickness
        assert math.isclose(z_lo, -total / 2, rel_tol=1e-12)
        assert math.isclose(z_hi, -total / 2 + 0.035e-3, rel_tol=1e-12)

    def test_get_layer_z_last(self, simple_stackup):
        z_lo, z_hi = simple_stackup.get_layer_z("top")
        total = simple_stackup.total_thickness
        assert math.isclose(z_hi, total / 2, rel_tol=1e-12)

    def test_get_layer_z_missing(self, simple_stackup):
        with pytest.raises(KeyError, match="nonexistent"):
            simple_stackup.get_layer_z("nonexistent")


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

class TestStandard2Layer:
    def test_standard_2layer(self):
        stackup = Stackup.standard_2layer()
        assert stackup.num_layers == 3  # bottom Cu + substrate + top Cu
        expected = 2 * 0.035e-3 + 1.6e-3
        assert math.isclose(stackup.total_thickness, expected, rel_tol=1e-12)

    def test_standard_2layer_custom(self):
        stackup = Stackup.standard_2layer(
            substrate_thickness=0.8e-3,
            substrate_material="rogers4003c",
            copper_thickness=0.018e-3,
        )
        assert stackup.num_layers == 3
        # Verify material assignment
        shapes = stackup.to_shapes()
        materials = [mat for _, mat in shapes]
        assert materials == ["copper", "rogers4003c", "copper"]
        expected = 2 * 0.018e-3 + 0.8e-3
        assert math.isclose(stackup.total_thickness, expected, rel_tol=1e-12)

    def test_standard_2layer_layer_names(self):
        stackup = Stackup.standard_2layer()
        names = [l.name for l in stackup.layers]
        assert "top" in names
        assert "bottom" in names
        assert "substrate" in names


class TestStandard4Layer:
    def test_standard_4layer(self):
        stackup = Stackup.standard_4layer()
        assert stackup.num_layers == 7  # 4 Cu + 2 prepreg + 1 core
        expected = 4 * 0.035e-3 + 2 * 0.2e-3 + 1.0e-3
        assert math.isclose(stackup.total_thickness, expected, rel_tol=1e-12)

    def test_standard_4layer_custom(self):
        stackup = Stackup.standard_4layer(
            core_thickness=0.8e-3,
            prepreg_thickness=0.1e-3,
            copper_thickness=0.018e-3,
        )
        expected = 4 * 0.018e-3 + 2 * 0.1e-3 + 0.8e-3
        assert math.isclose(stackup.total_thickness, expected, rel_tol=1e-12)

    def test_standard_4layer_layer_names(self):
        stackup = Stackup.standard_4layer()
        names = [l.name for l in stackup.layers]
        assert "top" in names
        assert "bottom" in names
        assert "inner1" in names
        assert "inner2" in names
        assert "core" in names

    def test_standard_4layer_symmetry(self):
        """The stackup should be symmetric about the core centre."""
        stackup = Stackup.standard_4layer()
        thicknesses = [l.thickness for l in stackup.layers]
        n = len(thicknesses)
        for i in range(n // 2):
            assert math.isclose(thicknesses[i], thicknesses[n - 1 - i], rel_tol=1e-12)


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_info(self):
        stackup = Stackup.standard_2layer()
        r = repr(stackup)
        assert "3 layers" in r
        assert "mm" in r
        assert "copper" in r
        assert "fr4" in r
