"""Declared Floquet lateral faces; synthetic 15 x 15 x 60 mm vacuum cell."""
import jax
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

# CPU float32 probe bits measured before edits on main 038bcc8184e5ea91f59a5be4f4d2b0bde7e6fc5a.
# 64 steps, dx=1.5 mm, 8 CPML layers; f0=10 GHz, TE z port at 15 mm,
# Ex probe at (7.5, 7.5, 36) mm. Both run and forward gave these bits.
MAIN_CPML = np.array([
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 748890022,
    767298918, 795890815, 812488502, 832219240, 847393554, 861215280, 873473702, 883961344,
    893514388, 901722968, 908964492, 915482350, 920837588, 925480771, 930474390, 934235948,
    938999940, 942333664, 946033928, 949246599, 952084916, 955515278, 957914626, 960254466,
    962979656, 965411414, 967235604, 969339953, 971772038, 973823948, 975422001, 977246125,
    979320164, 981562262, 982859808, 984294521, 985878251, 987611905, 989501041, 990702022,
    991805530, 992987452, 994240894, 995561286, 996942274, 998309278, 999046146, 999796972,
], dtype=np.uint32)

MAIN_PEC = np.array([
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 748890022,
    767298918, 795890815, 812488502, 832219240, 847393554, 861215280, 873473702, 883961344,
    893514388, 901722968, 908964492, 915482350, 920837588, 925480771, 930474390, 934235948,
    938999940, 942333664, 946033928, 949246599, 952084916, 955515278, 957914626, 960254466,
    962979656, 965411414, 967235604, 969339953, 971772038, 973823948, 975422001, 977246125,
    979320163, 981562260, 982859806, 984294513, 985878230, 987611848, 989500904, 990701867,
    991805180, 992986721, 994239414, 995558414, 996936950, 998304536, 999038045, 999783704,
], dtype=np.uint32)


def scene(boundary=None):
    kwargs = {} if boundary is None else {"boundary": boundary}
    return Simulation(freq_max=10e9, domain=(.015, .015, .060), dx=.0015,
                      cpml_layers=8, **kwargs)


def port(sim, axis="z"):
    sim.add_floquet_port(.015, axis=axis, f0=10e9,
                         freqs=np.array([10e9]), name="lateral")


def declared(kind, placement):
    # Boundary requires symmetric periodic pairs. Use another nonperiodic
    # token on the companion face so the named low/high face is exercised.
    companion = "pmc" if kind == "cpml" else "cpml"
    if placement == "x_lo":
        return BoundarySpec(x=Boundary(kind, companion), y="periodic", z="cpml")
    if placement == "y_hi":
        return BoundarySpec(x="periodic", y=Boundary(companion, kind), z="cpml")
    return BoundarySpec(x=kind, y=kind, z="cpml")


def invoke(sim, entry):
    if entry == "preflight":
        return sim.preflight()
    if entry == "descriptor":
        return sim.boundary_model()
    method = "forward" if entry == "forward" else "run"
    return getattr(sim, method)(n_steps=2, **({"skip_preflight": True}
                                            if entry == "run_skip" else {}))


@pytest.mark.parametrize("kind", ["pec", "pmc", "cpml"])
@pytest.mark.parametrize("placement", ["x_lo", "y_hi", "all"])
@pytest.mark.parametrize("order", ["boundary_first", "port_first", "changed"])
@pytest.mark.parametrize("entry", ["run", "run_skip", "forward", "preflight", "descriptor"])
def test_refuses_before_stepping(kind, placement, order, entry, monkeypatch):
    def stepped(*args, **kwargs):
        pytest.fail("stepping reached with a nonperiodic Floquet lateral face")

    monkeypatch.setattr("rfx.stepping.uniform.make_uniform_step", stepped)
    spec = declared(kind, placement)
    sim = scene(spec if order == "boundary_first" else
                BoundarySpec(x="periodic", y="periodic", z="cpml")
                if order == "changed" else None)
    with pytest.raises(ValueError) as error:
        port(sim)
        if order != "boundary_first":
            # There is no public boundary setter. Replace the current frozen
            # spec, as internal rebuilds do; leave the descriptor cached.
            sim._boundary_spec = spec
        invoke(sim, entry)
    message = str(error.value)
    assert "axis='z'" in message
    faces = ("x_lo", "x_hi", "y_lo", "y_hi") if placement == "all" else (placement,)
    for face in faces:
        assert f"{face}={kind!r}" in message
    assert "Declare periodic on x and y, or omit those faces" in message
    assert "a Floquet port's modes need periodic lateral faces" in message


@pytest.mark.parametrize("row", ["periodic", "default", "z_pec"])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_main_probe_bits_unchanged(row, entry):
    with jax.enable_x64(False):
        sim = scene(None if row == "default" else
                    BoundarySpec(x="periodic", y="periodic",
                                 z="pec" if row == "z_pec" else "cpml"))
        port(sim)
        sim.add_probe((.0075, .0075, .036), component="ex")
        sim.preflight()
        result = getattr(sim, entry)(n_steps=64)
        trace = np.asarray(result.time_series)
        assert trace.shape == (64, 1)
        assert trace.dtype == np.float32
        np.testing.assert_array_equal(trace.view(np.uint32).ravel(),
                                      MAIN_PEC if row == "z_pec" else MAIN_CPML)


@pytest.mark.parametrize("axis,transverse", [("x", "yz"), ("y", "xz"), ("z", "xy")])
@pytest.mark.parametrize("kind", ["pec", "pmc", "cpml", "upml"])
def test_all_normals_and_nonperiodic_tokens(axis, transverse, kind):
    spec = BoundarySpec(**{a: kind if a in transverse else "pec" for a in "xyz"})
    with pytest.raises(ValueError) as error:
        port(scene(spec), axis)
    assert f"axis={axis!r}" in str(error.value)
    for a in transverse:
        for side in ("lo", "hi"):
            assert f"{a}_{side}={kind!r}" in str(error.value)


@pytest.mark.parametrize("kind", ["pec", "cpml", "upml"])
def test_explicit_scalar_is_not_omitted(kind):
    with pytest.raises(ValueError, match=f"x_lo='{kind}'"):
        port(scene(kind))


def test_equal_cpml_replacement_is_a_new_declaration():
    sim = scene()
    port(sim)
    sim._boundary_spec = BoundarySpec.uniform("cpml")
    with pytest.raises(ValueError, match="x_lo='cpml'"):
        sim.preflight()


def test_registration_refusal_does_not_autofill_or_append():
    sim = scene("pec")
    with pytest.raises(ValueError, match="x_lo='pec'"):
        port(sim)
    assert sim._periodic_axes == ""
    assert sim._floquet_ports == []


def test_periodic_axes_conflict_uses_same_message_family():
    sim = scene(BoundarySpec(x="periodic", y="cpml", z="cpml"))
    with pytest.raises(ValueError, match="axis='z'.*y_lo='cpml'.*periodic_axes='x'"):
        port(sim)


def test_graded_registration_refusal_precedes_lateral_rule():
    sim = Simulation(freq_max=10e9, domain=(.015, .015, .060), dx=.0015,
                     dz_profile=np.array([.001] * 20 + [.002] * 20), boundary="pec")
    with pytest.raises(ValueError, match="Floquet ports do not support non-uniform z mesh"):
        port(sim)


def test_rebuild_checks_before_omitted_boundary_is_recreated():
    from rfx.boundaries.serialization import constructor_boundary
    sim = scene()
    port(sim)
    sim._boundary_spec = BoundarySpec.uniform("cpml")
    with pytest.raises(ValueError, match="x_lo='cpml'"):
        constructor_boundary(sim)
