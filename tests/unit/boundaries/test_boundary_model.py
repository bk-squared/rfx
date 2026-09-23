"""Declared electric, magnetic, absorbing and paired faces (B1)."""

from itertools import product
from inspect import signature
from types import SimpleNamespace

import pytest

from rfx import Simulation
from rfx.boundaries import cpml, upml
from rfx.boundaries.model import (
    FACES, Features, Kind, electric_faces, magnetic_faces, realize, resolve_kinds,
)
from rfx.boundaries.pec import resolve_wall_faces
from rfx.boundaries.spec import Boundary, BoundarySpec


class BoundaryDeparture(AssertionError):
    pass


def compare_faces(expected, actual):
    if expected != actual:
        raise BoundaryDeparture(f"declared {expected}; realized {actual}")


@pytest.mark.parametrize("tokens", tuple(product(("pec", "pmc", "cpml", "periodic"), repeat=3)))
def test_per_face_wall_sets(tokens):
    spec = BoundarySpec(**dict(zip("xyz", tokens)))
    model = resolve_kinds(spec, mode="3d", features=Features())
    grid = SimpleNamespace(pec_faces=spec.pec_faces(), pmc_faces=spec.pmc_faces())
    expected_e = frozenset(f"{a}_{s}" for a, t in zip("xyz", tokens)
                           for s in ("lo", "hi") if t in ("pec", "cpml"))
    expected_h = frozenset(f"{a}_{s}" for a, t in zip("xyz", tokens)
                           for s in ("lo", "hi") if t == "pmc")
    assert electric_faces(model) == expected_e
    assert magnetic_faces(model) == expected_h
    compare_faces((electric_faces(model), magnetic_faces(model)),
                  resolve_wall_faces(grid, tuple(t == "periodic" for t in tokens), None))
    assert isinstance(hash(model), int)


@pytest.mark.xfail(strict=True, raises=BoundaryDeparture,
                   reason="d; graded periodic faces realized as electric; fixed in B1.5")
def test_graded_periodic_wall_argument_departure():
    model = resolve_kinds(BoundarySpec(x="periodic", y="pec", z="pec"),
                          mode="3d", features=Features())
    compare_faces((electric_faces(model), magnetic_faces(model)),
                  resolve_wall_faces(SimpleNamespace(pec_faces={"y_lo", "y_hi", "z_lo", "z_hi"},
                                                      pmc_faces=set()),
                                     (False, False, False), None))


def test_asymmetric_faces_and_absorber_terminal_metres():
    sim = Simulation(freq_max=20e9, domain=(.024, .020, .016), dx=.001,
                     cpml_layers=8, cpml_kappa_max=2,
                     boundary=BoundarySpec(x=Boundary("pmc", "cpml", hi_thickness=5),
                                           y=Boundary("pec", "pec", conformal=True), z="periodic"))
    model = sim.boundary_model()
    assert model is sim.boundary_model()
    assert tuple(f.kind for f in model.faces) == (Kind.PMC, Kind.ABSORBER, Kind.PEC,
                                                Kind.PEC, Kind.PERIODIC, Kind.PERIODIC)
    assert tuple(f.layers for f in model.faces) == (0, 5, 0, 0, 0, 0)
    assert tuple(f.conformal for f in model.faces) == (False, False, True, True, False, False)
    # The alpha profile stores float32; the other scalars come from the signature.
    assert dict(model.absorber_parameters) == {
        "kappa_max": 2, "order": 3, "R_asymptotic": 1e-15,
        "alpha_max": pytest.approx(.05, rel=1e-7, abs=0),
    }
    assert model.axes[2].pairing == ("z_lo", "z_hi")
    planes = realize(model, sim._build_grid())
    assert planes.periods == (("z", .016),)
    assert planes.faces[0].plane_m == 0
    assert planes.faces[1].plane_m == .024
    assert planes.faces[1].terminal_m == pytest.approx(.029, abs=1e-14)
    assert planes.faces[1].backing == Kind.PEC
    assert planes.faces[3].plane_m == .020


@pytest.mark.parametrize("absorber,profile,names", [
    ("cpml", cpml._cpml_profile, ("order", "R_asymptotic", "kappa_max")),
    ("upml", upml._sigma_profile_1d, ("order", "R_asymptotic")),
])
def test_absorber_parameters_match_profile_defaults(absorber, profile, names):
    expected = {name: signature(profile).parameters[name].default for name in names}
    if absorber == "cpml":
        expected["alpha_max"] = float(profile(8, 1e-12, .001).alpha.max())
    model = resolve_kinds(BoundarySpec.uniform(absorber), mode="3d", features=Features())
    compare_faces(expected, dict(model.absorber_parameters))
    for name, value in expected.items():
        changed = dict(model.absorber_parameters)
        changed[name] = value * 2
        with pytest.raises(BoundaryDeparture):
            compare_faces(expected, changed)


@pytest.mark.parametrize("mode,equivalent", [("2d_tmz", Kind.PEC), ("2d_tez", Kind.PMC)])
@pytest.mark.parametrize("token", ["pec", "pmc", "cpml", "upml", "periodic"])
def test_invariant_z_scalar_and_explicit(mode, equivalent, token):
    spec = BoundarySpec.uniform(token)
    scalar = resolve_kinds(spec, mode=mode, features=Features(explicit_faces=False))
    explicit = resolve_kinds(spec, mode=mode, features=Features(explicit_faces=True))
    assert scalar.axes[2].invariant
    assert scalar.axes[2].pairing is None
    assert scalar.faces[4].kind == scalar.faces[5].kind == equivalent
    assert scalar.faces[4].origin == "feature"
    assert scalar.departures == ()
    assert len(explicit.departures) == (0 if token.upper() == equivalent.value else 2)


def test_default_and_declared_origins():
    args = dict(freq_max=20e9, domain=(.024, .020, .016), dx=.001)
    assert {f.origin for f in Simulation(**args).boundary_model().faces} == {"default"}
    assert {f.origin for f in Simulation(**args, boundary="cpml").boundary_model().faces} == {"declared"}
    with pytest.warns(DeprecationWarning):
        legacy = Simulation(**args, pec_faces={"x_lo"}).boundary_model()
    assert legacy.face("x_lo").origin == "declared"
    assert legacy.face("x_hi").origin == "default"


def test_requirements_collect_without_rewriting():
    sim = Simulation(freq_max=20e9, domain=(.024, .020, .016), dx=.001, cpml_layers=8)
    before = sim.boundary_model()
    sim.add_tfsf_source(f0=10e9)
    model = sim.boundary_model(rcs=True)
    assert model.faces == before.faces
    assert model.axes == before.axes
    assert {r.feature for r in model.requirements} == {"TFSF", "RCS"}
    assert next(r for r in model.requirements if r.feature == "RCS").faces == FACES
    assert next(r for r in model.requirements if r.feature == "TFSF" and r.faces[0] == "y_lo").admissible == (Kind.PERIODIC, Kind.PMC)


def test_periodic_legacy_rebuild_and_bloch_descriptor():
    sim = Simulation(freq_max=20e9, domain=(.024, .020, .016), dx=.001)
    with pytest.warns(DeprecationWarning):
        sim.set_periodic_axes("xy")
    assert sim.boundary_model().axes[0].pairing == ("x_lo", "x_hi")
    model = resolve_kinds(BoundarySpec(x="periodic", y="pec", z="pec"),
                          mode="3d", features=Features(bloch_axes=("x",)))
    assert model.axes[0].bloch
    assert model.k_t is None
