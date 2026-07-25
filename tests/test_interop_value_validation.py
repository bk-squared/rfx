"""Regression tests for the design-interop value guards.

Every case here was a real hole found by an independent review of the codec
layer, most of them silent. Pure structural checks — no FDTD.
"""

from __future__ import annotations

import json

import pytest

from rfx.api._spec import MaterialSpec
from rfx.geometry.csg import Box, Cylinder, PolylineWire, Sphere
from rfx.geometry.via import Via
from rfx.interop import (
    UnsupportedDesignFeature,
    materials_to_dict,
    shape_from_dict,
    shape_to_dict,
)


def _points(n: int = 5) -> list[tuple[float, float, float]]:
    return [(i * 1e-4, 0.0, 1e-3) for i in range(n)]


# --------------------------------------------------------------------------
# The one that mattered: a one-shot iterator was consumed by exporting it, so
# the SECOND export emitted an empty sequence with no error, producing a
# schema-valid design document in which the conductor simply was not there.
# --------------------------------------------------------------------------

def test_generator_valued_sequence_field_is_refused_not_consumed():
    wire = PolylineWire(points=(p for p in _points()), radius=5e-5)
    with pytest.raises(UnsupportedDesignFeature, match="no length"):
        shape_to_dict(wire)


def test_export_is_idempotent_for_sequence_fields():
    """Export must not mutate what it exports."""
    wire = PolylineWire(points=tuple(_points()), radius=5e-5)
    first = shape_to_dict(wire)
    second = shape_to_dict(wire)
    assert first == second
    assert len(second["params"]["points"]) == 5


@pytest.mark.parametrize("empty", [[], ()])
def test_empty_point_sequence_is_refused_on_export(empty):
    with pytest.raises(UnsupportedDesignFeature, match="at least 1 element"):
        shape_to_dict(PolylineWire(points=empty, radius=5e-5))


def test_empty_point_sequence_is_refused_on_import():
    with pytest.raises(UnsupportedDesignFeature, match="at least 1 element"):
        shape_from_dict({"kind": "polyline_wire",
                         "params": {"points": [], "radius": 5e-5}})


def test_empty_via_layer_stack_is_refused_on_import():
    with pytest.raises(UnsupportedDesignFeature, match="at least 1 element"):
        shape_from_dict({"kind": "via", "params": {
            "center": [0.0, 0.0], "drill_radius": 1e-4, "pad_radius": 2e-4,
            "layers": [], "material": "pec"}})


def test_null_sequence_is_refused_before_iteration():
    with pytest.raises(UnsupportedDesignFeature, match="must be a sequence"):
        shape_from_dict({"kind": "polyline_wire",
                         "params": {"points": None, "radius": 5e-5}})


# --------------------------------------------------------------------------
# String fields: str(v) cannot fail, so an unguarded field records a repr.
# --------------------------------------------------------------------------

def test_none_axis_is_refused_rather_than_stringified():
    with pytest.raises(UnsupportedDesignFeature, match="must be a string"):
        shape_to_dict(Cylinder(center=(0.0, 0.0, 0.0), radius=1e-4,
                               height=1e-3, axis=None))


def test_material_spec_in_a_name_slot_is_refused():
    """Simulation.add(material=...) takes a NAME, so passing a spec to Via is a
    plausible confusion; it must not be recorded as a repr."""
    via = Via(center=(0.0, 0.0), drill_radius=1e-4, pad_radius=2e-4,
              layers=[(0.0, 1e-3)], material=MaterialSpec(eps_r=4.3))
    with pytest.raises(UnsupportedDesignFeature, match="must be a string"):
        shape_to_dict(via)


def test_traced_string_field_is_refused():
    jax = pytest.importorskip("jax")
    captured = {}

    def export_inside_trace(t):
        try:
            shape_to_dict(Cylinder(center=(0.0, 0.0, 0.0), radius=1e-4,
                                   height=1e-3, axis=t))
            captured["error"] = None
        except UnsupportedDesignFeature as exc:
            captured["error"] = str(exc)
        return t * 2.0

    jax.grad(export_inside_trace)(1.0)
    assert captured["error"] is not None, (
        "a traced value in a string slot must not be stringified into the JSON"
    )


# --------------------------------------------------------------------------
# JSON type discipline: a hand-edited or foreign-generated document.
# --------------------------------------------------------------------------

def test_digit_string_is_not_silently_read_as_a_vector():
    """'123' is iterable and each character floats, so an unguarded loader turns
    it into (1.0, 2.0, 3.0) — silent wrong geometry."""
    with pytest.raises(UnsupportedDesignFeature, match="got the string"):
        shape_from_dict({"kind": "box", "params": {
            "corner_lo": "123", "corner_hi": [1.0, 2.0, 3.0]}})


def test_quoted_numeric_is_refused():
    with pytest.raises(UnsupportedDesignFeature, match="not a number"):
        shape_from_dict({"kind": "sphere", "params": {
            "center": [0.0, 0.0, 0.0], "radius": "5e-4"}})


@pytest.mark.parametrize("value", [True, False])
def test_json_boolean_is_not_silently_read_as_a_number(value):
    """A bool is an int in Python, so float(True) == 1.0 would become a length."""
    with pytest.raises(UnsupportedDesignFeature, match="boolean"):
        shape_from_dict({"kind": "sphere", "params": {
            "center": [0.0, 0.0, 0.0], "radius": value}})


def test_primitive_invariant_violation_names_the_kind():
    """Via's own check would otherwise escape as a bare ValueError. Note
    UnsupportedDesignFeature IS a ValueError, so a caller catching the interop
    error specifically would have missed it."""
    with pytest.raises(UnsupportedDesignFeature,
                       match="via payload violates the primitive's own"):
        shape_from_dict({"kind": "via", "params": {
            "center": [0.0, 0.0], "drill_radius": 3e-4, "pad_radius": 1e-4,
            "layers": [[0.0, 1e-3]], "material": "pec"}})


def test_materials_to_dict_refuses_a_non_mapping():
    with pytest.raises(UnsupportedDesignFeature, match="must be a mapping"):
        materials_to_dict([MaterialSpec(eps_r=1.0)])


def test_stray_top_level_key_in_a_shape_payload_is_refused():
    """The unknown-key check covered `params` but not the payload itself, so a
    stray sibling key was dropped without a word."""
    payload = shape_to_dict(Box(corner_lo=(0.0, 0.0, 0.0),
                                corner_hi=(1e-3, 1e-3, 1e-3)))
    payload["note"] = "written by some other tool"
    with pytest.raises(UnsupportedDesignFeature,
                       match="unknown top-level keys"):
        shape_from_dict(payload)


def test_non_string_material_name_is_refused_on_import():
    """Symmetric with the export side, which uses check_text: a numeric JSON key
    must not quietly become a material named "5"."""
    from rfx.interop import materials_from_dict, materials_to_dict as _dump

    payload = _dump({"fr4": MaterialSpec(eps_r=4.3)})
    payload[5] = payload.pop("fr4")
    with pytest.raises(UnsupportedDesignFeature, match="must be a string"):
        materials_from_dict(payload)


# --------------------------------------------------------------------------
# The pin test's blind spots.
# --------------------------------------------------------------------------

def test_constructor_parameter_names_sees_init_var():
    """dataclasses.fields excludes InitVar, so a units InitVar consumed in
    __post_init__ would be invisible to the pin test and every export would
    silently carry values under an undeclared convention."""
    import dataclasses
    from rfx.interop._shapes import constructor_parameter_names

    @dataclasses.dataclass(frozen=True)
    class BoxV2:
        corner_lo: tuple[float, float, float]
        corner_hi: tuple[float, float, float]
        units: dataclasses.InitVar[str] = "m"

        def __post_init__(self, units):
            pass

    assert "units" in constructor_parameter_names(BoxV2)


def test_constructor_parameter_names_excludes_non_init_fields():
    """The inverse trap: a derived field(init=False) is not a constructor
    parameter and must not be handed back on import."""
    import dataclasses
    from rfx.interop._shapes import constructor_parameter_names

    @dataclasses.dataclass(frozen=True)
    class BoxV3:
        corner_lo: tuple[float, float, float]
        corner_hi: tuple[float, float, float]
        volume: float = dataclasses.field(init=False, default=0.0)

    assert "volume" not in constructor_parameter_names(BoxV3)


def test_private_cache_attributes_do_not_break_export():
    """MeshShape already stamps self._mask_cache; the extra-state guard must not
    make a shape unexportable the moment it grows a memo."""
    via = Via(center=(0.0, 0.0), drill_radius=1e-4, pad_radius=2e-4,
              layers=[(0.0, 1e-3)], material="pec")
    before = shape_to_dict(via)
    via._memo = {"cached": 1}
    assert shape_to_dict(via) == before


def test_public_extra_state_is_still_refused():
    """The guard must keep catching a genuinely new public field."""
    via = Via(center=(0.0, 0.0), drill_radius=1e-4, pad_radius=2e-4,
              layers=[(0.0, 1e-3)], material="pec")
    via.plating_thickness = 1e-5
    with pytest.raises(UnsupportedDesignFeature, match="plating_thickness"):
        shape_to_dict(via)


def test_named_tuple_shape_state_is_readable():
    """vars() raises on a NamedTuple; rfx/geometry/thin_wire.py already defines
    one, so this must not TypeError if such a class is ever registered."""
    import inspect

    from rfx.geometry.thin_wire import ThinWire
    from rfx.interop._shapes import _instance_state_names

    # Build from the class's own signature rather than guessing physics values.
    kwargs = {
        name: (0.0 if param.default is inspect.Parameter.empty else param.default)
        for name, param in inspect.signature(ThinWire).parameters.items()
    }
    assert _instance_state_names(ThinWire(**kwargs)) == tuple(ThinWire._fields)


# --------------------------------------------------------------------------
# Canonical input must keep working — the guards must not over-tighten.
# --------------------------------------------------------------------------

def test_canonical_shapes_still_round_trip():
    shapes = [
        Box(corner_lo=(0.0, 0.0, 0.0), corner_hi=(1e-3, 1e-3, 1e-3)),
        Sphere(center=(0.0, 0.0, 0.0), radius=1e-4),
        PolylineWire(points=tuple(_points()), radius=5e-5),
        Via(center=(0.0, 0.0), drill_radius=1e-4, pad_radius=2e-4,
            layers=[(0.0, 1e-3)], material="pec"),
    ]
    for shape in shapes:
        rebuilt = shape_from_dict(json.loads(json.dumps(shape_to_dict(shape))))
        assert type(rebuilt) is type(shape)


def test_numpy_scalars_and_arrays_still_accepted():
    np = pytest.importorskip("numpy")
    payload = shape_to_dict(Sphere(center=np.array([0.0, 0.0, 0.0]),
                                   radius=np.float64(5e-4)))
    assert payload["params"] == {"center": [0.0, 0.0, 0.0], "radius": 5e-4}
