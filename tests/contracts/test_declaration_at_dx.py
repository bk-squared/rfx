"""Every declaration is classified at both rebuild sites (rfx#2798).

Builders start from the declaration census. Public APIs are used except for
solver/nominal-profile states with no concrete public admission, and the
internal probe-index bookkeeping. Expected models never use rebuild helpers.
"""
from dataclasses import dataclass
from math import floor

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
    return {
        "_geometry": call(lambda s: s.add(Box((.005, .004, .003), (.010, .008, .006)), material="pec")),
        "_thin_conductors": call(lambda s: s.add_thin_conductor(Box((.005, .004, .006), (.012, .010, .006)))),
        "_pinned_sheets": call(lambda s: s.add_pinned_sheet(plane_index=14, i_range=(12, 18), j_range=(12, 16)),
                               classification="REFUSED", public="add_pinned_sheet"),
        "_wall_losses": call(lambda s: s.add_wall_loss(sigma_bulk=5.8e7, f0=5e9, shape=s._geometry[0].shape), wall_base),
        "_ports": call(lambda s: s.add_port((.010, .008, .006), "ez", impedance=50.)),
        "_waveguide_ports": call(lambda s: s.add_waveguide_port(.010, direction="+x", freqs=np.linspace(7e9, 10e9, 5)), guide, classification="REWRITTEN"),
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
    entries = (tuple(sim._waveguide_ports), tuple(sim._ports))
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
        elif case.classification == "REWRITTEN":
            assert dict(declarations.declaration_key(scaled))[name] != dict(expected)[name], name
        else:
            assert scaled._dx == .5 * realized_dx
            if name == "_dt_pin":
                assert scaled._dt_pin == .5 * sim._dt_pin
    assert declarations.declaration_key(sim) == expected
    assert all(a is b for old, new in zip(entries, (sim._waveguide_ports, sim._ports)) for a, b in zip(old, new))


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
    base_grid = sim._build_realized_grid()
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
    scaled = declarations.declaration_at_dx(sim, .5, caller="contract", _base_grid=base_grid)
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
    assert reference._dx == built_dx
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
    assert float(device_grid.dx) != mesh["_dx"]
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
    with pytest.raises(ValueError, match="wall axis"):
        captured["sim_factory"](captured["dx_values"][1])._build_realized_grid()


def plane_guide(direction="+x", **kwargs):
    sim = Simulation(freq_max=10e9, domain=(.06096, .02286, .00762), dx=.00127,
                     boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=8)
    sim.add_waveguide_port(.0103 if direction == "+x" else .05066, direction=direction,
                           freqs=np.array([8e9, 10e9]), probe_offset=10, ref_offset=3, **kwargs)
    return sim


def built_reported(sim):
    from rfx.sparams._waveguide_planes import resolve_reported_planes
    entry = sim._waveguide_ports[0]
    grid = sim._build_realized_grid()
    cfg = sim._build_waveguide_port_config(entry, grid, entry.freqs, 1)
    return resolve_reported_planes(entry, cfg, grid=grid)


@pytest.mark.parametrize("factor", [2., 1.5, .75, .5])
@pytest.mark.parametrize("direction", ["+x", "-x"])
@pytest.mark.parametrize("preset", ["default", "measured", "reference", "both"])
def test_t6_reported_planes_keep_factor_one_metres(factor, direction, preset):
    options = {"calibration_preset": "measured"} if preset == "measured" else {}
    if preset in {"reference", "both"}:
        options["reference_plane"] = .030
    if preset == "both":
        options["probe_plane"] = .040
    sim = plane_guide(direction, **options)
    entry = sim._waveguide_ports[0]
    key = declarations.declaration_key(sim)
    original = built_reported(sim)
    literal = {("+x", "default"): (.01016, .02286), ("+x", "measured"): (.01397, .02286),
               ("-x", "default"): (.05080, .03810), ("-x", "measured"): (.04699, .03810)}
    expected = literal.get((direction, preset), (.030, .040 if preset == "both" else .02286 if direction == "+x" else .03810))
    assert original[:2] == pytest.approx(expected, abs=1e-12, rel=0)
    variant = declarations.declaration_at_dx(sim, factor, caller="contract")
    actual = built_reported(variant)
    assert actual[:2] == pytest.approx(original[:2], abs=1e-12, rel=0)
    assert actual[:2] == pytest.approx(expected, abs=1e-12, rel=0)
    assert actual.label == "explicit"
    rewritten = variant._waveguide_ports[0]
    assert rewritten.calibration_preset is None
    if preset in {"reference", "both"}:
        assert rewritten.reference_plane is entry.reference_plane
    if preset == "both":
        assert rewritten.probe_plane is entry.probe_plane
    for pin in (False, True):
        same = declarations.declaration_at_dx(sim, caller="contract", pin_resolved_mesh=pin)
        assert declarations.declaration_key(same) == key
        assert same._waveguide_ports[0] is entry
    assert declarations.declaration_key(sim) == key
    assert sim._waveguide_ports[0] is entry


@pytest.mark.parametrize("factor,probe,reference", [(f, 10, 3) for f in (2., 1.5, .75, .5)] + [(2., 5, 3), (2., 9, 3)])
def test_t7_sampling_counts_round_half_up(factor, probe, reference):
    from dataclasses import replace
    sim = plane_guide()
    sim._waveguide_ports[0] = replace(sim._waveguide_ports[0], probe_offset=probe, ref_offset=reference)
    entry = sim._waveguide_ports[0]
    key = declarations.declaration_key(sim)
    variant = declarations.declaration_at_dx(sim, factor, caller="contract")
    scaled = variant._waveguide_ports[0]
    assert scaled.probe_offset == max(1, floor(probe / factor + .5))
    assert scaled.ref_offset == max(1, floor(reference / factor + .5))
    if factor == 2. and probe in (5, 9):
        assert scaled.probe_offset == {5: 3, 9: 5}[probe]
    if factor == 1.5:
        assert scaled.probe_offset == 7
    assert declarations.declaration_key(sim) == key
    assert sim._waveguide_ports[0] is entry


def test_t7_collapsed_sampling_planes_refused_before_copy(monkeypatch):
    from dataclasses import replace
    sim = plane_guide()
    sim._waveguide_ports[0] = replace(sim._waveguide_ports[0], probe_offset=4)
    entry = sim._waveguide_ports[0]
    copies = []
    monkeypatch.setattr(declarations, "declaration_variant", lambda *a, **kw: copies.append(a))
    with pytest.raises(ValueError) as error:
        declarations.declaration_at_dx(sim, 2., caller="contract")
    for text in ("contract", entry.name, "ref_offset=3", "probe_offset=4", "convergence_study", "offsets further apart"):
        assert text in str(error.value)
    assert not copies
    assert sim._waveguide_ports[0] is entry


@pytest.mark.parametrize("factor,count", [(.75, 16), (1.5, 8), (2., 6)])
def test_t8_lumped_sampling_counts_scale(factor, count):
    sim = base(domain=(.080, .016, .012))
    for x in (.020, .060):
        sim.add_port((x, .008, .006), "ez", impedance=50., extent=.002,
                     direction="-x" if x == .020 else "+x", reference_plane_cells=12)
    entries = tuple(sim._ports)
    key = declarations.declaration_key(sim)
    variant = declarations.declaration_at_dx(sim, factor, caller="contract")
    opted_in = 0
    for original, scaled in zip(entries, variant._ports):
        if original.reference_plane_cells is None:
            assert scaled is original
        else:
            opted_in += 1
            assert scaled.position == original.position
            assert scaled.reference_plane_cells == count
    assert opted_in == 2
    assert declarations.declaration_key(sim) == key
    assert all(a is b for a, b in zip(entries, sim._ports))
