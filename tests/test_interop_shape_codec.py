"""Contract tests for the design-interop shape codec.

These are pure structural checks — they do not run FDTD.  The point of the
codec is that a shape survives a JSON round trip *exactly*, because the
existing scene artifact records only a class name plus a bounding box and so
cannot tell a Cylinder from a Box with the same bounds.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from rfx.geometry.csg import Box, Cylinder, PolylineWire, Sphere
from rfx.geometry.curved import CurvedPatch
from rfx.geometry.via import Via
from rfx.interop import (
    SUPPORTED_SHAPE_TYPES,
    UnsupportedDesignFeature,
    shape_from_dict,
    shape_to_dict,
)
from rfx.interop._shapes import _CODECS, constructor_parameter_names


SHAPES = {
    "Box": Box(corner_lo=(0.0, 0.0, 0.0), corner_hi=(0.02, 0.012, 0.0015)),
    "Cylinder": Cylinder(
        center=(0.010, 0.006, 0.0008), radius=3e-4, height=1.5e-3, axis="z"),
    "Sphere": Sphere(center=(1e-3, 2e-3, 3e-3), radius=5e-4),
    "PolylineWire": PolylineWire(
        points=((0.0, 0.0, 1e-3), (5e-3, 0.0, 1e-3), (5e-3, 4e-3, 1e-3)),
        radius=1e-4),
    "Via": Via(
        center=(2e-3, 3e-3), drill_radius=1.5e-4, pad_radius=3e-4,
        layers=[(0.0, 8e-4), (8e-4, 1.6e-3)], material="pec"),
    "CurvedPatch": CurvedPatch(
        center=(0.0, 0.0, 1e-3), length=8e-3, width=4e-3, radius=0.05,
        axis="x"),
}


def test_every_supported_type_has_a_fixture():
    assert set(SHAPES) == set(SUPPORTED_SHAPE_TYPES)


@pytest.mark.parametrize("shape_type", sorted(SHAPES))
def test_registry_pins_live_constructor_parameters(shape_type):
    """Adding a parameter upstream must fail here, not vanish from exports."""
    codec = _CODECS[shape_type]
    assert set(codec.fields) == set(constructor_parameter_names(codec.cls)), (
        f"{shape_type} constructor and interop registry disagree; update "
        f"rfx/interop/_shapes.py"
    )


@pytest.mark.parametrize("shape_type", sorted(SHAPES))
def test_round_trip_through_json_text(shape_type):
    original = SHAPES[shape_type]
    payload = shape_to_dict(original)
    # Must survive real JSON, not just dict copying.
    rebuilt = shape_from_dict(json.loads(json.dumps(payload)))

    assert type(rebuilt) is type(original)
    if dataclasses.is_dataclass(original):
        assert rebuilt == original
    else:
        for name in constructor_parameter_names(type(original)):
            assert getattr(rebuilt, name) == getattr(original, name), name


@pytest.mark.parametrize("shape_type", sorted(SHAPES))
def test_round_trip_preserves_bounding_box(shape_type):
    """Numeric witness: the rebuilt shape occupies the same space."""
    original = SHAPES[shape_type]
    rebuilt = shape_from_dict(json.loads(json.dumps(shape_to_dict(original))))

    lo_a, hi_a = original.bounding_box()
    lo_b, hi_b = rebuilt.bounding_box()
    assert tuple(map(float, lo_a)) == pytest.approx(tuple(map(float, lo_b)))
    assert tuple(map(float, hi_a)) == pytest.approx(tuple(map(float, hi_b)))


def test_cylinder_records_what_the_scene_artifact_drops():
    """The gap this codec exists to close (see rfx/artifacts.py:274)."""
    payload = shape_to_dict(SHAPES["Cylinder"])
    assert payload["type"] == "Cylinder"
    assert payload["params"]["radius"] == pytest.approx(3e-4)
    assert payload["params"]["height"] == pytest.approx(1.5e-3)
    assert payload["params"]["axis"] == "z"


def test_via_material_is_recorded_alongside_geometry():
    """Via owns a material; the codec must not drop or resolve it silently."""
    payload = shape_to_dict(SHAPES["Via"])
    assert payload["params"]["material"] == "pec"
    assert payload["params"]["layers"] == [[0.0, 8e-4], [8e-4, 1.6e-3]]


def test_unknown_shape_class_is_refused_loudly():
    class Torus:
        def __init__(self):
            self.major_radius = 1.0

    with pytest.raises(UnsupportedDesignFeature, match="Torus"):
        shape_to_dict(Torus())


def test_subclass_of_supported_shape_is_refused():
    """A subclass may add state, so it is not silently treated as its base."""
    class TaggedSphere(Sphere):
        pass

    with pytest.raises(UnsupportedDesignFeature, match="TaggedSphere"):
        shape_to_dict(TaggedSphere(center=(0.0, 0.0, 0.0), radius=1e-3))


def test_same_named_impostor_class_is_refused():
    """Name lookup alone is not trusted: identity against the registry wins."""
    @dataclasses.dataclass(frozen=True)
    class Sphere:  # noqa: N801 - deliberately shadows the real primitive
        center: tuple[float, float, float]
        radius: float

    with pytest.raises(UnsupportedDesignFeature, match="registered"):
        shape_to_dict(Sphere(center=(0.0, 0.0, 0.0), radius=1e-3))


def test_unknown_type_in_payload_is_refused():
    with pytest.raises(UnsupportedDesignFeature, match="not supported"):
        shape_from_dict({"type": "Torus", "params": {}})


def test_missing_parameter_in_payload_is_refused():
    payload = shape_to_dict(SHAPES["Cylinder"])
    del payload["params"]["axis"]
    with pytest.raises(UnsupportedDesignFeature, match="missing parameters"):
        shape_from_dict(payload)


def test_extra_parameter_in_payload_is_refused():
    payload = shape_to_dict(SHAPES["Sphere"])
    payload["params"]["thickness"] = 1e-4
    with pytest.raises(UnsupportedDesignFeature, match="unknown parameters"):
        shape_from_dict(payload)


def test_wrong_vector_length_is_refused():
    payload = shape_to_dict(SHAPES["Box"])
    payload["params"]["corner_hi"] = [1.0, 2.0]
    with pytest.raises(UnsupportedDesignFeature, match="exactly 3 components"):
        shape_from_dict(payload)


def test_payload_shape_is_json_serialisable_scalars_only():
    for shape in SHAPES.values():
        payload = shape_to_dict(shape)
        # json.dumps would accept tuples too; assert the canonical list form so
        # emitted files stay byte-stable across writers.
        text = json.dumps(payload, sort_keys=True)
        assert "(" not in text
