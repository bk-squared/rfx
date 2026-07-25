"""Contract tests for the design-interop shape codec.

These are pure structural checks — they do not run FDTD.  The point of the
codec is that a shape survives a JSON round trip *exactly*, because every
existing setup-serialisation layer in rfx is bbox-or-box-level:

- ``rfx.artifacts.build_scene_artifact`` keeps ``shape_type`` + bounding box;
- ``rfx.io.export_geometry_json`` keeps the class name + bounding box;
- ``rfx.config._shapes`` supports ``("box",)`` only;
- ``rfx.experiments.canonical`` fails any geometry ``kind`` other than ``box``.

So none of them can tell a cylinder from a box with the same bounds.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from rfx.geometry.csg import Box, Cylinder, PolylineWire, Sphere
from rfx.geometry.curved import CurvedPatch
from rfx.geometry.via import Via
from rfx.interop import (
    SUPPORTED_SHAPE_KINDS,
    UnsupportedDesignFeature,
    shape_from_dict,
    shape_kind_of,
    shape_to_dict,
)
from rfx.interop._shapes import _CODECS, constructor_parameter_names


SHAPES = {
    "box": Box(corner_lo=(0.0, 0.0, 0.0), corner_hi=(0.02, 0.012, 0.0015)),
    "cylinder": Cylinder(
        center=(0.010, 0.006, 0.0008), radius=3e-4, height=1.5e-3, axis="z"),
    "sphere": Sphere(center=(1e-3, 2e-3, 3e-3), radius=5e-4),
    "polyline_wire": PolylineWire(
        points=((0.0, 0.0, 1e-3), (5e-3, 0.0, 1e-3), (5e-3, 4e-3, 1e-3)),
        radius=1e-4),
    "via": Via(
        center=(2e-3, 3e-3), drill_radius=1.5e-4, pad_radius=3e-4,
        layers=[(0.0, 8e-4), (8e-4, 1.6e-3)], material="pec"),
    "curved_patch": CurvedPatch(
        center=(0.0, 0.0, 1e-3), length=8e-3, width=4e-3, radius=0.05,
        axis="x"),
}


def test_every_supported_kind_has_a_fixture():
    assert set(SHAPES) == set(SUPPORTED_SHAPE_KINDS)


def test_kind_vocabulary_is_snake_case():
    for kind in SUPPORTED_SHAPE_KINDS:
        assert kind == kind.lower()
        assert " " not in kind and "-" not in kind


def test_kind_vocabulary_agrees_with_the_layers_it_claims_to_follow():
    """Bind the vocabulary claim to the other layers instead of asserting a
    literal written in this file.

    ``rfx/config/_shapes.py`` names shapes under the key ``shape`` and
    ``rfx/experiments/canonical.py`` under the key ``kind``; both spell the box
    ``"box"``. The codec follows those *values*, so its own name for a box must
    be importable-equal to theirs, not merely lowercase.
    """
    from rfx.config._shapes import _SUPPORTED_SHAPES

    assert set(_SUPPORTED_SHAPES).issubset(set(SUPPORTED_SHAPE_KINDS)), (
        "the config layer names a shape the codec cannot express"
    )
    assert "box" in _SUPPORTED_SHAPES and "box" in SUPPORTED_SHAPE_KINDS

    # The canonical experiment layer rejects any geometry kind other than box;
    # read its refusal rather than trusting a comment about it.
    repo_root = Path(__file__).resolve().parents[1]
    canonical = (repo_root / "rfx/experiments/canonical.py").read_text()
    assert 'P0 supports box geometry' in canonical, (
        "canonical.py's box-only fence moved; re-check the shared vocabulary"
    )


@pytest.mark.parametrize("kind", sorted(SHAPES))
def test_registry_pins_live_constructor_parameters(kind):
    """Adding a parameter upstream must fail here, not vanish from exports."""
    codec = _CODECS[kind]
    assert set(codec.fields) == set(constructor_parameter_names(codec.cls)), (
        f"{kind} constructor and interop registry disagree; update "
        f"rfx/interop/_shapes.py"
    )


@pytest.mark.parametrize("kind", sorted(SHAPES))
def test_shape_kind_of_matches_registry(kind):
    assert shape_kind_of(SHAPES[kind]) == kind
    assert shape_to_dict(SHAPES[kind])["kind"] == kind


@pytest.mark.parametrize("kind", sorted(SHAPES))
def test_round_trip_through_json_text(kind):
    original = SHAPES[kind]
    payload = shape_to_dict(original)
    # Must survive real JSON, not just dict copying.
    rebuilt = shape_from_dict(json.loads(json.dumps(payload)))

    assert type(rebuilt) is type(original)
    if dataclasses.is_dataclass(original):
        assert rebuilt == original
    else:
        for name in constructor_parameter_names(type(original)):
            assert getattr(rebuilt, name) == getattr(original, name), name


@pytest.mark.parametrize("kind", sorted(SHAPES))
def test_round_trip_preserves_bounding_box(kind):
    """Numeric witness: the rebuilt shape occupies the same space."""
    original = SHAPES[kind]
    rebuilt = shape_from_dict(json.loads(json.dumps(shape_to_dict(original))))

    lo_a, hi_a = original.bounding_box()
    lo_b, hi_b = rebuilt.bounding_box()
    assert tuple(map(float, lo_a)) == pytest.approx(tuple(map(float, lo_b)))
    assert tuple(map(float, hi_a)) == pytest.approx(tuple(map(float, hi_b)))


def test_cylinder_records_what_the_other_layers_drop():
    """The gap this codec exists to close (see rfx/artifacts.py:274)."""
    payload = shape_to_dict(SHAPES["cylinder"])
    assert payload["kind"] == "cylinder"
    assert payload["params"]["radius"] == pytest.approx(3e-4)
    assert payload["params"]["height"] == pytest.approx(1.5e-3)
    assert payload["params"]["axis"] == "z"


def test_cylinder_is_distinguishable_from_a_box_with_the_same_bounds():
    """The concrete failure the bbox-level layers cannot avoid."""
    cylinder = SHAPES["cylinder"]
    lo, hi = cylinder.bounding_box()
    look_alike = Box(corner_lo=tuple(map(float, lo)), corner_hi=tuple(map(float, hi)))

    assert look_alike.bounding_box() == cylinder.bounding_box()
    assert shape_to_dict(look_alike) != shape_to_dict(cylinder)
    assert shape_to_dict(look_alike)["kind"] == "box"


def test_via_material_is_recorded_alongside_geometry():
    """Via owns a material; the codec must not drop or resolve it silently."""
    payload = shape_to_dict(SHAPES["via"])
    assert payload["params"]["material"] == "pec"
    assert payload["params"]["layers"] == [[0.0, 8e-4], [8e-4, 1.6e-3]]


def test_unknown_shape_class_is_refused_loudly():
    class Torus:
        def __init__(self):
            self.major_radius = 1.0

    with pytest.raises(UnsupportedDesignFeature, match="Torus"):
        shape_to_dict(Torus())


def test_mesh_shape_is_refused_until_explicitly_supported():
    """CAD-sourced geometry (#358) is a real shape class with no IR mapping yet.

    ``MeshShape`` wraps a triangle mesh and rasterises host-side, so it is not
    parameter-describable the way the CSG primitives are.  It must refuse, not
    degrade to a bounding box.
    """
    trimesh = pytest.importorskip("trimesh")
    from rfx.geometry.mesh_import import MeshShape

    mesh = trimesh.creation.box(extents=(1e-3, 1e-3, 1e-3))
    with pytest.raises(UnsupportedDesignFeature, match="MeshShape"):
        shape_to_dict(MeshShape(mesh))


def test_subclass_of_supported_shape_is_refused():
    """A subclass may add state, so it is not silently treated as its base."""
    class TaggedSphere(Sphere):
        pass

    with pytest.raises(UnsupportedDesignFeature, match="TaggedSphere"):
        shape_to_dict(TaggedSphere(center=(0.0, 0.0, 0.0), radius=1e-3))


def test_same_named_impostor_class_is_refused():
    """Registry lookup is by class identity, not by name."""
    @dataclasses.dataclass(frozen=True)
    class Sphere:  # noqa: N801 - deliberately shadows the real primitive
        center: tuple[float, float, float]
        radius: float

    with pytest.raises(UnsupportedDesignFeature, match="Sphere"):
        shape_to_dict(Sphere(center=(0.0, 0.0, 0.0), radius=1e-3))


def test_unknown_kind_in_payload_is_refused():
    with pytest.raises(UnsupportedDesignFeature, match="not supported"):
        shape_from_dict({"kind": "torus", "params": {}})


def test_class_name_as_kind_is_refused():
    """Guards the vocabulary: 'Cylinder' is not a kind, 'cylinder' is."""
    with pytest.raises(UnsupportedDesignFeature, match="not supported"):
        shape_from_dict({"kind": "Cylinder", "params": {}})


def test_missing_kind_key_is_refused():
    with pytest.raises(UnsupportedDesignFeature, match="missing the 'kind' key"):
        shape_from_dict({"params": {}})


def test_missing_parameter_in_payload_is_refused():
    payload = shape_to_dict(SHAPES["cylinder"])
    del payload["params"]["axis"]
    with pytest.raises(UnsupportedDesignFeature, match="missing parameters"):
        shape_from_dict(payload)


def test_extra_parameter_in_payload_is_refused():
    payload = shape_to_dict(SHAPES["sphere"])
    payload["params"]["thickness"] = 1e-4
    with pytest.raises(UnsupportedDesignFeature, match="unknown parameters"):
        shape_from_dict(payload)


def test_wrong_vector_length_is_refused():
    payload = shape_to_dict(SHAPES["box"])
    payload["params"]["corner_hi"] = [1.0, 2.0]
    with pytest.raises(UnsupportedDesignFeature, match="exactly 3 components"):
        shape_from_dict(payload)


def test_long_point_sequences_are_never_truncated():
    """artifacts.py degrades any array over 16 elements to metadata
    (``_jsonable(max_array_values=16)``). The codec must emit every value.
    """
    points = tuple((i * 1e-4, (i % 3) * 1e-4, 1e-3) for i in range(64))
    payload = shape_to_dict(PolylineWire(points=points, radius=5e-5))
    assert len(payload["params"]["points"]) == 64

    rebuilt = shape_from_dict(json.loads(json.dumps(payload)))
    assert rebuilt.points == points


def test_long_via_layer_stacks_are_never_truncated():
    layers = [(i * 1e-4, (i + 1) * 1e-4) for i in range(40)]
    payload = shape_to_dict(Via(center=(1e-3, 1e-3), drill_radius=5e-5,
                                pad_radius=1e-4, layers=layers, material="pec"))
    assert len(payload["params"]["layers"]) == 40

    rebuilt = shape_from_dict(json.loads(json.dumps(payload)))
    assert [tuple(v) for v in rebuilt.layers] == [tuple(v) for v in layers]


def test_scene_artifact_really_does_lose_what_the_codec_keeps():
    """Pins the motivating contrast, so the claim cannot rot silently."""
    from rfx.artifacts import build_scene_artifact
    from rfx import Simulation

    points = tuple((i * 1e-4, 0.0, 1e-3) for i in range(64))
    wire = PolylineWire(points=points, radius=5e-5)

    sim = Simulation(freq_max=10e9, domain=(0.01, 0.01, 0.004), dx=1e-4,
                     boundary="cpml")
    sim.add(wire, material="pec")
    entry = build_scene_artifact(sim)["geometry"][0]

    assert "bounding_box" in entry
    assert "points" not in entry and "params" not in entry
    assert len(shape_to_dict(wire)["params"]["points"]) == 64


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_scalar_is_refused_at_the_codec(bad):
    """Not at json.dump time: allow_nan=False raises a bare ValueError with no
    field name, so the refusal must name the parameter instead."""
    with pytest.raises(UnsupportedDesignFeature, match="not a finite number"):
        shape_to_dict(Sphere(center=(0.0, 0.0, 0.0), radius=bad))


def test_non_finite_vector_component_is_refused_with_its_index():
    with pytest.raises(UnsupportedDesignFeature, match=r"corner_hi\[0\] is nan"):
        shape_to_dict(Box(corner_lo=(0.0, 0.0, 0.0),
                          corner_hi=(float("nan"), 1.0, 1.0)))


def test_non_finite_is_refused_on_import_too():
    payload = shape_to_dict(SHAPES["sphere"])
    payload["params"]["radius"] = float("inf")
    with pytest.raises(UnsupportedDesignFeature, match="not a finite number"):
        shape_from_dict(payload)


def test_traced_scalar_parameter_is_refused_naming_the_tracer():
    """A traced parameter means the geometry is a differentiable DoF, so there
    is no concrete value to record. The message must say so rather than blame
    the component count."""
    jax = pytest.importorskip("jax")
    captured = {}

    def export_inside_trace(t):
        shape = Sphere(center=(0.0, 0.0, 0.0), radius=t)
        try:
            shape_to_dict(shape)
            captured["error"] = None
        except UnsupportedDesignFeature as exc:
            captured["error"] = str(exc)
        return t * 2.0

    jax.grad(export_inside_trace)(1e-3)

    assert captured["error"] is not None, "a traced radius must be refused"
    assert "JAX tracer" in captured["error"]
    assert "components" not in captured["error"], (
        "the refusal must not misattribute a tracer to a bad component count"
    )


def test_traced_vector_parameter_is_refused():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    captured = {}

    def export_inside_trace(t):
        corner = jnp.stack([t, t, t])
        try:
            shape_to_dict(Box(corner_lo=(0.0, 0.0, 0.0), corner_hi=corner))
            captured["error"] = None
        except UnsupportedDesignFeature as exc:
            captured["error"] = str(exc)
        return t * 2.0

    jax.grad(export_inside_trace)(1e-3)
    assert captured["error"] is not None
    assert "JAX tracer" in captured["error"]


def test_numpy_array_parameters_are_still_accepted():
    """The tracer guard must not tighten into rejecting numpy input."""
    np = pytest.importorskip("numpy")
    payload = shape_to_dict(Box(corner_lo=np.array([0.0, 0.0, 0.0]),
                                corner_hi=np.array([0.02, 0.012, 0.0015])))
    assert payload["params"]["corner_hi"] == [0.02, 0.012, 0.0015]


def test_payload_shape_is_json_serialisable_scalars_only():
    for shape in SHAPES.values():
        payload = shape_to_dict(shape)
        # json.dumps would accept tuples too; assert the canonical list form so
        # emitted files stay byte-stable across writers.
        text = json.dumps(payload, sort_keys=True)
        assert "(" not in text
