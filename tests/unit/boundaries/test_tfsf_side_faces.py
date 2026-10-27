"""Declared TF/SF side faces; 24 x 16 x 16 mm vacuum probe scene."""
import jax
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


REFUSED = [
    pytest.param(BoundarySpec(y="cpml", z="cpml"), False,
                 {"y_lo": "cpml", "y_hi": "cpml", "z_lo": "cpml", "z_hi": "cpml"}, id="cpml"),
    pytest.param(BoundarySpec(y="pmc", z="pec"), False,
                 {"y_lo": "pmc", "y_hi": "pmc", "z_lo": "pec", "z_hi": "pec"}, id="walls"),
    pytest.param(BoundarySpec(y="periodic", z="cpml"), False,
                 {"z_lo": "cpml", "z_hi": "cpml"}, id="mixed-cpml"),
    pytest.param(BoundarySpec(y="periodic", z="pec"), False,
                 {"z_lo": "pec", "z_hi": "pec"}, id="mixed-pec"),
    pytest.param(BoundarySpec(y="cpml", z="cpml"), True,
                 {"z_lo": "cpml", "z_hi": "cpml"}, id="methodB"),
    pytest.param({"x": "cpml", "y": "pmc", "z": "pec"}, False,
                 {"y_lo": "pmc", "y_hi": "pmc", "z_lo": "pec", "z_hi": "pec"}, id="dict"),
    pytest.param(BoundarySpec(y="periodic", z=Boundary("pmc", "cpml")), False,
                 {"z_lo": "pmc", "z_hi": "cpml"}, id="asymmetric"),
]


def scene(boundary=None, *, dx=.001, graded=False, mode="3d"):
    kwargs = {} if boundary is None else {"boundary": boundary}
    if graded:
        kwargs["dz_profile"] = np.full(16, .001)
    return Simulation(freq_max=12e9, domain=(.024, .016, .016 if mode == "3d" else .001),
                      dx=dx, cpml_layers=6, mode=mode, **kwargs)


def source(sim, method_b=False, **kwargs):
    return sim.add_tfsf_source(f0=8e9, bandwidth=.5, margin=4,
                               angle_deg=20 if method_b else 0,
                               method="methodB" if method_b else "bloch", **kwargs)


def invoke(sim, entry):
    if entry == "preflight":
        return sim.preflight()
    if entry == "descriptor":
        return sim.boundary_model()
    if entry == "serialization":
        from rfx.boundaries.serialization import constructor_boundary
        return constructor_boundary(sim)
    return getattr(sim, entry.removesuffix("_skip"))(
        n_steps=20, skip_preflight=entry.endswith("_skip"))


def assert_conflicts(message, faces, method_b):
    for face in (f"{a}_{side}" for a in "xyz" for side in ("lo", "hi")):
        if face in faces:
            assert f"{face}={faces[face]!r}" in message
        else:
            assert face not in message
    assert f"axes {'z' if method_b else 'yz'!r} periodic" in message
    for remedy in ("Declare periodic", 'boundary="cpml"', "preflight warning",
                   "closed_box=True", "graded mesh"):
        assert remedy in message


@pytest.mark.parametrize("boundary,method_b,faces", REFUSED)
def test_registration_refuses(boundary, method_b, faces):
    with pytest.raises(ValueError) as error:
        source(scene(boundary), method_b)
    assert_conflicts(str(error.value), faces, method_b)


@pytest.mark.parametrize("boundary,method_b,faces", REFUSED[:-1])
@pytest.mark.parametrize("entry", ["run", "run_skip", "forward", "forward_skip",
                                   "preflight", "descriptor", "serialization"])
def test_deferred_mesh_refuses_before_stepping(boundary, method_b, faces, entry, monkeypatch):
    def stepped(*args, **kwargs):
        pytest.fail("stepping reached with an explicit nonperiodic TF/SF side face")

    monkeypatch.setattr("rfx.stepping.uniform.make_uniform_step", stepped)
    sim = source(scene(boundary, dx=None), method_b)
    assert sim._tfsf is not None  # Registration must not resolve an automatic mesh.
    with pytest.raises(ValueError) as error:
        invoke(sim, entry)
    assert_conflicts(str(error.value), faces, method_b)


@pytest.mark.parametrize("entry", ["run_skip", "forward_skip", "preflight", "descriptor", "serialization"])
def test_refresh_reads_current_declaration(entry):
    sim = source(scene())
    # No public boundary setter exists. Internal rebuilds replace the frozen
    # spec; refresh must check that current declaration rather than its cache.
    sim._boundary_spec = BoundarySpec(y="pmc", z="pec")
    with pytest.raises(ValueError, match="y_lo='pmc'.*z_hi='pec'"):
        invoke(sim, entry)


def trace(boundary, method_b=False, graded=False, entry="run"):
    sim = source(scene(boundary, graded=graded), method_b)
    sim.add_probe((.017, .005, .006), "ez")
    # Main's probe first becomes nonzero at index 20 (normal) / 27 (Method B).
    steps = 28 if method_b else 24
    result = getattr(sim, entry)(n_steps=steps, skip_preflight=True)
    values = np.asarray(result.time_series)
    assert values.shape == (steps, 1)
    assert np.isfinite(values).all()
    assert np.any(values != 0)
    return values


@pytest.mark.parametrize("boundary", [None, "cpml", BoundarySpec(y="periodic", z="periodic")],
                         ids=["default", "cpml", "periodic"])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_admitted_uniform_traces(boundary, entry):
    with jax.enable_x64(False):
        actual, expected = trace(boundary, entry=entry), trace(None, entry=entry)
    assert actual.dtype == expected.dtype == np.float32
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("entry", ["run", "forward"])
def test_method_b_keeps_y_absorber(entry):
    with jax.enable_x64(False):
        actual = trace(BoundarySpec(y="cpml", z="periodic"), method_b=True, entry=entry)
        expected = trace("cpml", method_b=True, entry=entry)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("boundary", [None, "cpml", BoundarySpec(y="cpml", z="cpml"),
                                      BoundarySpec(y="pmc", z="pec")],
                         ids=["default", "cpml", "spec-cpml", "walls"])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_admitted_graded_traces(boundary, entry):
    with jax.enable_x64(False):
        actual = trace(boundary, graded=True, entry=entry)
        expected = trace(None, graded=True, entry=entry)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode", ["2d_tmz", "2d_tez"])
def test_invariant_z_is_not_a_side_face(mode):
    sim = source(scene(BoundarySpec(y="periodic", z="cpml"), mode=mode),
                 polarization="ey" if mode == "2d_tez" else "ez")
    sim.boundary_model()
    with pytest.raises(ValueError) as error:
        source(scene(BoundarySpec(y="cpml", z="pec"), mode=mode),
               polarization="ey" if mode == "2d_tez" else "ez")
    assert "y_lo='cpml'" in str(error.value)
    assert "z_lo" not in str(error.value) and "z_hi" not in str(error.value)


def test_closed_box_keeps_explicit_absorbers():
    sim = source(scene(BoundarySpec.uniform("cpml")), closed_box=True)
    sim.boundary_model()


@pytest.mark.parametrize("boundary", [None, "cpml"])
def test_shorthand_keeps_preflight_warning(boundary):
    findings = source(scene(boundary)).preflight()
    assert any(item.code == "tfsf_transverse_periodic" for item in findings)


def test_declared_periodic_axes_are_in_faces():
    sim = source(scene({"x": "cpml", "y": "periodic", "z": "periodic"}))
    assert sim._periodic_axes == "yz"
    assert [(a, side, kind) for a, side, kind in sim._boundary_spec.faces() if a in "yz"] == [
        ("y", "lo", "periodic"), ("y", "hi", "periodic"),
        ("z", "lo", "periodic"), ("z", "hi", "periodic")]
    sim.boundary_model()
