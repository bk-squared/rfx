"""Every declaration is classified at both rebuild sites (rfx#2798).

Builders start from the declaration census. Public APIs are used except for
solver/nominal-profile states with no concrete public admission, and the
internal probe-index bookkeeping. Expected models never use rebuild helpers.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx import _declarations as declarations
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sparams.waveguide import _empty_waveguide_reference


DX = .001
DOMAIN = (.020, .016, .012)
DIP = [.9, .8, .8, .8, .8, .9]
PX = np.array([1.] * 7 + DIP + [1.] * 8) * DX
PY = np.array([1.] * 5 + DIP + [1.] * 6) * DX
PZ = np.array([1.] * 3 + DIP + [1.] * 4) * DX
DT = .5 * .99 * DX / (299792458. * np.sqrt(3.))
MSL_H, MSL_W, MSL_LX = 254e-6, 600e-6, .010
MSL_LY = MSL_W + 2 * (2 * MSL_H + 8 * MSL_H / 3)


def base(**kwargs):
    options = dict(freq_max=10e9, domain=DOMAIN, dx=DX, cpml_layers=8)
    options.update(kwargs)
    sim = Simulation(**options)
    sim.add_source((.007, .006, .005), "ez")
    sim.add_probe((.013, .009, .007), "ez")
    return sim


def guide():
    return Simulation(freq_max=10e9, domain=(.060, .024, .012), dx=DX,
                      cpml_layers=8, boundary={"x": "cpml", "y": "pec", "z": "pec"})


def msl_base(*, automatic=False):
    sim = Simulation(freq_max=5e9, domain=(MSL_LX, MSL_LY, MSL_H + .0015),
                     dx=None if automatic else MSL_H / 3, cpml_layers=8, snap="declared",
                     boundary=BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml")))
    sim.add_material("board", eps_r=3.66)
    sim.add(Box((0., 0., 0.), (MSL_LX, MSL_LY, MSL_H)), material="board")
    sim.add(Box((0., (MSL_LY - MSL_W) / 2, MSL_H),
                (MSL_LX, (MSL_LY + MSL_W) / 2, MSL_H)), material="pec")
    return sim


def add_msl(sim):
    sim.add_msl_port(position=(.002, MSL_LY / 2, 0.), width=MSL_W,
                     height=MSL_H, direction="+x", impedance=50.)


def wall_base():
    sim = base()
    sim.add(Box((.005, .004, .003), (.010, .008, .006)), material="pec")
    return sim


@dataclass(frozen=True)
class Case:
    build: object
    control: object = base
    classification: str = "CARRIED"
    public: str = ""


def kw(name, value, *, classification="CARRIED", public="", **extra):
    return Case(lambda: base(**{name: value}, **extra), lambda: base(**extra), classification, public)


def call(fn, make=base, *, classification="CARRIED", public=""):
    def build():
        sim = make()
        fn(sim)
        return sim
    return Case(build, make, classification, public)


def cases():
    # One explicit row per declaration; never derive this table from the registry.
    return {
        "_geometry": call(lambda s: s.add(Box((.005, .004, .003), (.010, .008, .006)), material="pec")),
        "_thin_conductors": call(lambda s: s.add_thin_conductor(Box((.005, .004, .006), (.012, .010, .006)))),
        "_pinned_sheets": call(lambda s: s.add_pinned_sheet(plane_index=14, i_range=(12, 18), j_range=(12, 16)),
                               classification="REFUSED", public="add_pinned_sheet"),
        "_wall_losses": call(lambda s: s.add_wall_loss(sigma_bulk=5.8e7, f0=5e9, shape=s._geometry[0].shape), wall_base),
        "_ports": call(lambda s: s.add_port((.010, .008, .006), "ez", impedance=50.)),
        "_waveguide_ports": call(lambda s: s.add_waveguide_port(.010, direction="+x", freqs=np.linspace(7e9, 10e9, 5)), guide),
        "_msl_ports": call(add_msl, msl_base),
        "_coaxial_ports": call(lambda s: s.add_coaxial_port((.004, .004, .015), face="top", pin_length=.005,
                                                          pin_radius=.000635, outer_radius=.002055),
                               lambda: Simulation(freq_max=10e9, domain=(.008, .008, .030), dx=.0004, boundary="cpml")),
        "_floquet_ports": call(lambda s: s.add_floquet_port(.008, axis="z", polarization="te", n_modes=1),
                               lambda: Simulation(freq_max=10e9, domain=(.015, .015, .030), boundary="cpml", cpml_layers=8)),
        "_lumped_rlc": call(lambda s: s.add_lumped_rlc((.010, .008, .006), "ez", R=50.)),
        "_probes": call(lambda s: s.add_probe((.012, .010, .008), "ex")),
        "_dft_planes": call(lambda s: s.add_dft_plane_probe(axis="z", coordinate=.006, n_freqs=5)),
        "_flux_monitors": call(lambda s: s.add_flux_monitor(axis="x", coordinate=.010, n_freqs=5)),
        "_materials": call(lambda s: s.add_material("contract_material", eps_r=3.)),
        "_dx_profile": kw("dx_profile", PX, classification="REFUSED", public="dx_profile="),
        "_dy_profile": kw("dy_profile", PY, classification="REFUSED", public="dy_profile="),
        "_dz_profile": kw("dz_profile", PZ, classification="REFUSED", public="dz_profile="),
        "_dx": kw("dx", .0008, classification="RESCALED"),
        "_domain": kw("domain", (.022, .016, .012)),
        "_freq_max": kw("freq_max", 12e9),
        "_boundary": kw("boundary", "pec"),
        "_boundary_spec": kw("boundary", BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="cpml", hi="cpml", lo_thickness=4))),
        "_boundary_explicit": kw("boundary", "cpml"),
        "_boundary_scalar_spec": kw("boundary", "upml"),
        "_pec_faces": kw("boundary", BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml"))),
        "_periodic_axes": kw("boundary", BoundarySpec(x="periodic", y="cpml", z="cpml")),
        "_cpml_layers": kw("cpml_layers", 12),
        "_cpml_kappa_max": kw("cpml_kappa_max", 3.),
        "_mode": kw("mode", "2d_tmz"),
        "_solver": call(lambda s: setattr(s, "_solver", "adi")),
        "_precision": kw("precision", "mixed"),
        "_stencil_order": kw("stencil_order", 4),
        "_interface_eps": kw("interface_eps", "dual_average"),
        "_pec_occupancy_model": kw("pec_occupancy_model", "kottke"),
        "_snap": kw("snap", "declared"),
        "_dt_pin": kw("dt", DT, classification="RESCALED"),
        "_dt_min_cell": kw("dt_min_cell", .0008, dt=DT * .8, dx_profile=PX,
                           classification="REFUSED", public="dt_min_cell="),
        "_nominal_profiles": call(lambda s: setattr(s, "_nominal_profiles", {"x": PX}),
                                  classification="REFUSED", public="nominal_profiles="),
        "_refinement": call(lambda s: s.add_refinement((.004, .008), ratio=2)),
        "_tfsf": call(lambda s: s.add_tfsf_source(f0=4e9, bandwidth=.5, amplitude=1., margin=3),
                      lambda: Simulation(freq_max=8e9, domain=(.080, .006, .006), boundary="cpml", cpml_layers=8, dx=DX)),
        "_ntff": call(lambda s: s.add_ntff_box((.003,) * 3, (.017, .013, .009), n_freqs=5)),
        "_current_moments": call(lambda s: s.add_current_moment_monitor((.004, .004, .005), (.016, .012, .007), .002, np.array([5e9]))),
        "_dft_plane_regions": call(lambda s: s.add_dft_plane_probe(axis="z", coordinate=.006, n_freqs=5, region=(2, 8, 2, 8)),
                                   classification="REFUSED", public="add_dft_plane_probe(region=)"),
        "_internal_probe_indices": call(lambda s: setattr(s, "_internal_probe_indices", {0})),
        "_time_recorded_dft_planes": call(lambda s: (s.add_dft_plane_probe(axis="z", coordinate=.006, n_freqs=5, name="timed"),
                                                     setattr(s, "_time_recorded_dft_planes", {"timed"}))),
        "_msl_auto_offset_min": call(add_msl, msl_base),
        "_msl_auto_probe_spacing": call(add_msl, msl_base),
        "_msl_auto_probe_lengths": call(add_msl, msl_base),
    }


def test_t1_registry_is_exhaustive():
    assert set(cases()) == {name for name, _ in declarations.DECLARATIONS}


@pytest.mark.parametrize("name", cases())
def test_t1_declaration_at_both_factors(name):
    case = cases()[name]
    sim = case.build()
    expected = declarations.declaration_key(sim)
    assert dict(expected)[name] != dict(declarations.declaration_key(case.control()))[name], name
    same = declarations.declaration_at_dx(sim, caller="contract")
    assert declarations.declaration_key(same) == expected, name
    assert same._declaration_root is same
    for attr, form in declarations.DECLARATIONS:
        value = sim.__dict__.get(attr)
        if attr not in {"_dx", "_domain", "_dx_profile", "_dy_profile", "_dz_profile"} and isinstance(value, (dict, set, list)):
            assert same.__dict__[attr] is not value, attr
    if case.classification == "REFUSED":
        with pytest.raises(ValueError) as error:
            declarations.declaration_at_dx(sim, .5, caller="contract")
        assert case.public in str(error.value)
        assert "contract" in str(error.value)
        assert "convergence_study" in str(error.value)
    else:
        realized_dx = float(sim._build_realized_grid().dx)
        scaled = declarations.declaration_at_dx(sim, .5, caller="contract")
        if case.classification == "CARRIED":
            assert dict(declarations.declaration_key(scaled))[name] == dict(expected)[name], name
        else:
            assert scaled._dx == .5 * realized_dx
            if name == "_dt_pin":
                assert scaled._dt_pin == .5 * sim._dt_pin
    assert declarations.declaration_key(sim) == expected


STATE_RULES = {
    "_realization_early_outcomes": "removed",
    "_pf_campaign_ctx": "released",
    "_realized_geometry_record": "released",
    "_coax_geometry": "released",
    "_mesh_resolution": "input-keyed",
    "_msl_default_memo": "released",
    "_boundary_model": "carried",
    "_frozen_mesh": "frozen",
    "_amplitude_kind_warned": "removed",
    "_boundary_default_warned": "removed",
    "_waveguide_call_overrides": "removed",
}


def test_t2_state_inventory_and_lifecycle():
    assert set(STATE_RULES) == {name for name, _ in declarations.NOT_DECLARATIONS}
    sim = base()
    tokens = {name: {} for name, rule in STATE_RULES.items() if rule not in {"frozen", "carried"}}
    sim.__dict__.update(tokens)
    same = declarations.declaration_at_dx(sim, caller="contract")
    for name, rule in STATE_RULES.items():
        if rule == "removed":
            assert name not in same.__dict__
        elif rule == "released":
            assert same.__dict__[name] is None
        elif rule in {"carried", "input-keyed"}:
            assert same.__dict__[name] is sim.__dict__[name]
    scaled = declarations.declaration_at_dx(sim, .5, caller="contract", _base_dx=DX)
    assert "_mesh_resolution" not in scaled.__dict__
    assert all(sim.__dict__[name] is token for name, token in tokens.items())


def test_t2_frozen_mesh():
    sim = base()
    sim.freeze_mesh()
    frozen = sim.__dict__["_frozen_mesh"]
    same = declarations.declaration_at_dx(sim, caller="contract")
    assert same.__dict__["_frozen_mesh"] is frozen
    assert declarations.declaration_key(same) == declarations.declaration_key(sim)
    with pytest.raises(ValueError, match=r"contract:.*freeze_mesh\(\).*Build a new Simulation to remesh"):
        declarations.declaration_at_dx(sim, .5, caller="contract")
    pinned = declarations.declaration_at_dx(sim, caller="contract", pin_resolved_mesh=True)
    assert "_frozen_mesh" not in pinned.__dict__
    assert sim.__dict__["_frozen_mesh"] is frozen
    assert pinned._build_realized_grid().shape == sim._build_realized_grid().shape


def test_t3_automatic_profile_is_inferred_and_refused():
    sim = msl_base(automatic=True)
    assert sim.__dict__["_dz_profile"] is None
    assert sim._resolve_mesh()["_dz_profile"] is not None
    same = declarations.declaration_at_dx(sim, caller="contract")
    assert declarations.declaration_key(same) == declarations.declaration_key(sim)
    with pytest.raises(ValueError, match=r"dz_profile=.*inferred"):
        declarations.declaration_at_dx(sim, .5, caller="contract")


def test_t4_quick_convergence_carries_msl_and_caches_factory(monkeypatch):
    import rfx.convergence as convergence
    sim = cases()["_msl_ports"].build()
    expected = declarations.declaration_key(sim)
    calls = []

    def study(*, sim_factory, dx_values, **kwargs):
        calls.append(dx_values)
        same = sim_factory(dx_values[0])
        assert declarations.declaration_key(same) == expected
        half = sim_factory(dx_values[1])
        assert half._build_realized_grid().dx == pytest.approx(MSL_H / 6)
        assert sim_factory(dx_values[1]) is half
        extra = sim_factory(MSL_H / 9)
        assert extra._dx == pytest.approx(MSL_H / 9)
        assert sim_factory(MSL_H / 9) is extra
        return "captured"

    monkeypatch.setattr(convergence, "convergence_study", study)
    assert convergence.quick_convergence(sim, dx_factors=[1., .5]) == "captured"
    assert len(calls) == 1
    assert declarations.declaration_key(sim) == expected


@pytest.mark.parametrize("name", [n for n, case in cases().items() if case.classification == "REFUSED"] + ["frozen", "inferred"])
def test_t4_refusal_precedes_study(monkeypatch, name):
    import rfx.convergence as convergence
    sim = base() if name == "frozen" else msl_base(automatic=True) if name == "inferred" else cases()[name].build()
    if name == "frozen":
        sim.freeze_mesh()
    calls = []
    monkeypatch.setattr(convergence, "convergence_study", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="quick_convergence:.*convergence_study"):
        convergence.quick_convergence(sim, dx_factors=[1., .5])
    assert not calls


# Independent expected reset inventory, including lazy attributes absent on bare.
RESET_NAMES = {
    "_geometry", "_thin_conductors", "_pinned_sheets", "_wall_losses", "_ports",
    "_waveguide_ports", "_msl_ports", "_coaxial_ports", "_floquet_ports", "_lumped_rlc",
    "_probes", "_dft_planes", "_flux_monitors", "_materials", "_refinement", "_tfsf",
    "_ntff", "_current_moments", "_dft_plane_regions", "_time_recorded_dft_planes",
    "_internal_probe_indices",
    "_msl_auto_offset_min", "_msl_auto_probe_spacing", "_msl_auto_probe_lengths",
}


@pytest.mark.parametrize("name", cases())
def test_t5_reference_carries_numerics_and_resets_device(name):
    sim = cases()[name].build()
    expected = dict(declarations.declaration_key(sim))
    built_dx = float(sim._build_realized_grid().dx)
    bare = type(sim)(freq_max=sim._freq_max, domain=tuple(sim._domain))
    defaults = dict(declarations.declaration_key(bare))
    reference = _empty_waveguide_reference(sim)
    actual = dict(declarations.declaration_key(reference))
    assert declarations.DEVICE_DECLARATIONS == RESET_NAMES
    carried = set(expected) - RESET_NAMES - {"_dx"}
    assert carried | RESET_NAMES | {"_dx"} == {n for n, _ in declarations.DECLARATIONS}
    for attr in carried:
        assert actual[attr] == expected[attr], attr
    for attr in RESET_NAMES:
        assert actual[attr] == defaults[attr], attr
    assert reference._dx == built_dx  # Pinned to built dx, including wall snaps.
    assert reference._declaration_root is reference
    assert reference._boundary_model is sim._boundary_model
    assert type(reference.__dict__["_domain"]) is tuple
    assert declarations.declaration_key(sim) == tuple(expected.items())


def test_t5_automatic_wr90_insert_preserves_built_grid():
    sim = Simulation(freq_max=12e9, domain=(.060, .02286, .01016),
                     boundary={"x": "cpml", "y": "pec", "z": "pec"})
    sim.add_material("insert", eps_r=2.2)
    sim.add(Box((.025, 0., .004), (.035, .02286, .0055)), material="insert")
    mesh = sim._resolve_mesh()
    assert mesh["_dz_profile"] is not None
    device_grid = sim._build_realized_grid()
    assert float(device_grid.dx) != mesh["_dx"]  # Exercise the pre-snap defect.
    reference = _empty_waveguide_reference(sim)
    ref_grid = reference._build_realized_grid()
    assert type(ref_grid) is type(device_grid)
    assert ref_grid.shape == device_grid.shape
    assert ref_grid.dx == device_grid.dx
    assert ref_grid.dt == device_grid.dt
    for attr in ("dx_arr", "dy_arr", "dz"):
        np.testing.assert_array_equal(getattr(ref_grid, attr, None), getattr(device_grid, attr, None))
    for attr in ("_dx_profile", "_dy_profile", "_dz_profile"):
        np.testing.assert_array_equal(getattr(reference, attr), mesh[attr])
    assert reference._geometry == []


@pytest.mark.parametrize("factor", [0, -1, np.inf, -np.inf, np.nan, None, "invalid"])
def test_invalid_factor(factor):
    with pytest.raises(ValueError, match="contract: factor must be finite and positive"):
        declarations.declaration_at_dx(base(), factor, caller="contract")


def test_pin_requires_factor_one():
    with pytest.raises(ValueError, match="pin_resolved_mesh requires factor=1.0"):
        declarations.declaration_at_dx(base(), .5, caller="contract", pin_resolved_mesh=True)


def test_mesh_inputs_are_not_redeclared_and_value_containers_are_private():
    sim = base(domain=list(DOMAIN))
    sim._refinement = {"z_range": (.004, .008), "ratio": 2}
    same = declarations.declaration_at_dx(sim, caller="contract")
    assert same.__dict__["_domain"] is sim.__dict__["_domain"]
    same._refinement["ratio"] = 3
    assert sim._refinement["ratio"] == 2


def test_refusals_are_collected_before_copy(monkeypatch):
    sim = base(dx_profile=PX, dy_profile=PY, dz_profile=PZ)
    sim.add_pinned_sheet(plane_index=14, i_range=(12, 18), j_range=(12, 16))
    sim.add_dft_plane_probe(axis="z", coordinate=.006, region=(2, 8, 2, 8))
    calls = []
    monkeypatch.setattr(declarations, "declaration_variant", lambda *a, **kw: calls.append(a))
    with pytest.raises(ValueError) as error:
        declarations.declaration_at_dx(sim, .5, caller="contract")
    for public in ("dx_profile=", "dy_profile=", "dz_profile=", "add_pinned_sheet", "add_dft_plane_probe(region=)"):
        assert public in str(error.value)
    assert not calls


def test_quick_convergence_builds_only_base_on_private_model(monkeypatch):
    import rfx.convergence as convergence
    sim = base(boundary="pec")
    built = []
    original = Simulation._build_realized_grid

    def build(current):
        built.append(current)
        current._realization_early_outcomes = ["grid finding"]
        current._boundary_default_warned = True
        return original(current)

    monkeypatch.setattr(Simulation, "_build_realized_grid", build)
    monkeypatch.setattr(convergence, "convergence_study", lambda **kwargs: kwargs)
    captured = convergence.quick_convergence(sim, dx_factors=[1., 1.5])
    assert len(built) == 1 and built[0] is not sim
    assert "_realization_early_outcomes" not in sim.__dict__
    assert "_boundary_default_warned" not in sim.__dict__
    # Strict wall mismatch is deferred until the rung actually builds a grid.
    with pytest.raises(ValueError, match="wall axis"):
        captured["sim_factory"](captured["dx_values"][1])._build_realized_grid()
