"""Public calculator entry checks, independent of the admission row lists."""

import jax
import numpy as np
import pytest

from rfx import Simulation
from rfx.runners import _admission as A
from tests.unit.runners.test_silent_routes import _sim, _mixed
from tests.unit.sparams.test_waveguide_solver_precision import _guide

PHYSICS_CASES = [
    (calculator, "kerr") for calculator in (
        "s_matrix_scan", "mixed_s_matrix", "topology_optimize",
        "waveguide_s_matrix", "coax_msl_transition", "vmap_sweep_batched")
] + [(calculator, "coax") for calculator in (
    "s_matrix_scan", "topology_optimize", "waveguide_s_matrix")
] + [("waveguide_s_matrix", feature) for feature in ("msl", "floquet")
] + [("coax_msl_transition", feature) for feature in ("debye", "lorentz", "drude")]
PHYSICS_CASES += [(calculator, "scan_angle") for calculator in
                  ("s_matrix_scan", "topology_optimize")]


def declare_physics(sim, feature):
    from dataclasses import replace
    from rfx import Box, DebyePole, drude_pole, lorentz_pole
    if feature in ("kerr", "debye", "lorentz", "drude"):
        names = [entry.material_name for entry in sim._geometry
                 if sim._resolve_material(entry.material_name).sigma < 1e6]
        if names:
            name = names[0]
        else:
            name = "admission_witness"
            sim.add_material(name, eps_r=2.)
            sim.add(Box(tuple(.3*x for x in sim._domain), tuple(.7*x for x in sim._domain)), material=name)
        change = ({"chi3": 1e-20} if feature == "kerr" else
                  {"debye_poles": [DebyePole(delta_eps=1., tau=1e-11)]} if feature == "debye" else
                  {"lorentz_poles": [lorentz_pole(1., 3e9, 1e8)]} if feature == "lorentz" else
                  {"lorentz_poles": [drude_pole(3e9, 1e8)]})
        sim._materials[name] = replace(sim._materials[name], **change)
        return ("_materials", feature)
    centre = tuple(.5*x for x in sim._domain)
    if feature == "coax":
        sim.add_coaxial_port(centre, pin_length=.002)
        return ("_coaxial_ports", "coax_port")
    if feature == "msl":
        sim.add_msl_port(centre, width=.002, height=.001, direction="+x", eps_r_sub=2.2)
        return ("_msl_ports", "msl_port")
    assert feature in ("floquet", "scan_angle")
    if feature == "scan_angle":
        # Keep the new witness's explicit cell size commensurate with
        # the transverse periods that add_floquet_port registers.
        sim._domain = tuple(round(length / sim._dx) * sim._dx for length in sim._domain)
    sim.add_floquet_port(centre[2], freqs=np.array([8e9, 9e9]),
                         scan_theta=30. if feature == "scan_angle" else 0.)
    return ("_floquet_ports", feature if feature == "scan_angle" else "floquet_port")


def calculator_case(calculator):
    """A registered model and one public call, with a four-step request."""
    if calculator == "waveguide_s_matrix":
        sim = _guide()
        call = lambda: sim.compute_waveguide_s_matrix(n_steps=4, normalize=True)
    elif calculator.startswith("coax"):
        if calculator == "coax_msl_transition":
            from dataclasses import replace
            from tests.unit.sparams.test_coax_msl_transition import _build_coax_msl_transition_sim
            sim = _build_coax_msl_transition_sim()
            # This witness uses the realized top of the volume ground
            # (2.5 mm) and the trace's lower surface (2.8 mm).
            port = sim._msl_ports[0]
            sim._msl_ports[0] = replace(
                port, position=(*port.position[:2], .0025), height=.0003)
        else:
            from tests.unit.sparams.test_coax_two_port_smatrix import _sim as coax_sim
            sim = coax_sim(dx=0.0005)
        method = getattr(sim, "compute_" + calculator)
        options = ({"junction_x": sim._coaxial_ports[0].position[0],
                    "probe_count": 6, "probe_start_cells": 4,
                    "probe_spacing_cells": 2, "skip_preflight": True}
                   if calculator == "coax_msl_transition" else {})
        call = lambda: method(n_steps=4, **options)
    elif calculator == "material_fit":
        from rfx.differentiable_material_fit import differentiable_material_fit
        from tests.unit.autodiff.test_differentiable_material_fit_inputs import _fixture
        sim = _fixture("supported", 2.0, [], [])
        call = lambda: differentiable_material_fit(
            lambda *args: sim, np.zeros((1, 1, 3), complex),
            np.array([2e9, 3e9, 4e9]), n_iterations=1, verbose=False)
    elif calculator == "mixed_s_matrix":
        sim = _mixed(False)
        call = lambda: sim.compute_mixed_s_matrix(
            n_steps=4, skip_preflight=True, magnitude_channel="wave")
    else:
        sim = _sim(ports="lumped" if calculator == "s_matrix_scan" else None)
        if calculator == "s_matrix_scan":
            from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
            call = lambda: compute_lumped_wire_s_matrix_via_scan(
                sim, np.array([4e9, 5e9, 6e9]), n_steps=4)
        elif calculator == "topology_optimize":
            from rfx.topology import TopologyDesignRegion, topology_optimize
            region = TopologyDesignRegion(
                corner_lo=(.002, .002, .002), corner_hi=(.003, .004, .004),
                material_bg="air", material_fg="design")
            call = lambda: topology_optimize(
                sim, region, lambda result: 0., n_iterations=1,
                verbose=False, skip_preflight=True)
        else:
            assert calculator == "vmap_sweep_batched"
            from rfx.vmap_sweep import vmap_material_sweep
            call = lambda: vmap_material_sweep(sim, "design.eps_r", [2., 3.], n_steps=4)
    return sim, call


@pytest.mark.parametrize("calculator", A.CALCULATORS)
@pytest.mark.parametrize("attr,value,words", [
    ("_interface_eps", "dual_average", "interface_eps"),
    ("_dt_pin", 1e-12, "pinned time step"),
    ("_dt_min_cell", 1e-4, "dt_min_cell"),
])
def test_calculator_refuses_unread_setting_before_scan(monkeypatch, calculator, attr, value, words):
    sim, call = calculator_case(calculator)
    # Uniform Yee fixtures: these settings are unread on the base routes
    # (except material fit's existing interface-rule refusal).
    setattr(sim, attr, value)

    def forbidden(*args, **kwargs):
        pytest.fail("a calculator started lax.scan before refusing its input")

    monkeypatch.setattr(jax.lax, "scan", forbidden)
    with pytest.raises(NotImplementedError) as caught:
        call()
    assert A.CALCULATOR_WORDS[calculator] in str(caught.value)
    assert words in str(caught.value)
    assert "does not carry" in str(caught.value)


@pytest.mark.parametrize("case", ["rlc", "wire", "msl", "source_field", "refinement"])
def test_fit_migrated_inputs_refused_before_assembly_or_scan(monkeypatch, case):
    from rfx.differentiable_material_fit import differentiable_material_fit
    from tests.unit.autodiff.test_differentiable_material_fit_inputs import _fixture

    def factory(eps, debye, lorentz):
        sim = _fixture(case if case != "refinement" else "supported", eps, debye, lorentz)
        if case == "refinement":
            sim.add_refinement(z_range=(0., .006), ratio=2)
        return sim

    def forbidden(*args, **kwargs):
        pytest.fail("material fit reached assembly or lax.scan before refusal")

    monkeypatch.setattr(Simulation, "_assemble_materials", forbidden)
    monkeypatch.setattr(jax.lax, "scan", forbidden)
    with pytest.raises(NotImplementedError, match="differentiable_material_fit.*does not carry"):
        differentiable_material_fit(factory, np.zeros((1, 1, 3), complex),
                                    np.array([2e9, 3e9, 4e9]), n_iterations=1, verbose=False)


@pytest.mark.parametrize("wall", ["pec", "pmc", "lid"])
def test_fit_hands_declared_faces_to_the_kernel(monkeypatch, wall):
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.differentiable_material_fit import differentiable_material_fit
    from tests.unit.autodiff.test_differentiable_material_fit_inputs import _acceptance_fixture

    class ReachedCore(Exception):
        pass

    seen = []

    def inspect(ctx, **kwargs):
        seen.append((ctx.pec_faces_frozen, ctx.pmc_faces_frozen))
        raise ReachedCore

    boundary = (BoundarySpec(x="pec", y="pec", z=Boundary(lo="pec", hi="cpml"))
                if wall == "lid" else BoundarySpec(x="cpml", y=wall, z="pec"))

    def factory(eps, debye, lorentz):
        return _acceptance_fixture(eps, debye, lorentz, boundary=boundary, cpml_layers=2)

    monkeypatch.setattr("rfx.simulation.make_core_step", inspect)
    with pytest.raises(ReachedCore):
        differentiable_material_fit(factory, np.zeros((1, 1, 3), complex),
                                    np.array([2e9, 3e9, 4e9]), n_iterations=1, verbose=False)
    assert len(seen) == 1
    pec, pmc = seen[0]
    if wall == "pmc":
        assert {"y_lo", "y_hi"} <= set(pmc)
        assert not {"y_lo", "y_hi"} & set(pec)
    else:
        assert {"y_lo", "y_hi", "z_lo"} <= set(pec)


@pytest.mark.parametrize("attr,value", [("_dt_pin", 1e-12), ("_dt_min_cell", 1e-4)])
@pytest.mark.parametrize("graded", [False, True])
def test_waveguide_timestep_gate(attr, value, graded, monkeypatch):
    sim = _guide(nonuniform=graded)
    setattr(sim, attr, value)
    assert ((attr, "") in A.refused(sim, "waveguide_s_matrix")) is not graded
    if not graded:
        def forbidden(*args, **kwargs):
            pytest.fail("uniform waveguide reached a scan with an unread dt control")
        monkeypatch.setattr(jax.lax, "scan", forbidden)
        with pytest.raises(NotImplementedError, match="compute_waveguide_s_matrix.*does not carry"):
            sim.compute_waveguide_s_matrix(n_steps=4)


@pytest.mark.parametrize("solver", ["yee", "adi"])
def test_topology_cfl_gate(solver):
    sim = _sim(solver=solver)
    sim._adi_cfl_factor = 2.718
    assert ("_adi_cfl_factor", "") not in A.refused(sim, "topology_optimize")


@pytest.mark.parametrize("calculator,feature", PHYSICS_CASES)
def test_calculator_refuses_dropped_physics_before_scan(monkeypatch, calculator, feature):
    sim, call = calculator_case(calculator)
    row = declare_physics(sim, feature)
    assert A.DETECTORS[row](sim)

    def forbidden(*args, **kwargs):
        pytest.fail("a calculator started lax.scan with a dropped physics input")

    monkeypatch.setattr(jax.lax, "scan", forbidden)
    if calculator == "waveguide_s_matrix" and feature == "floquet":
        with pytest.raises(ValueError, match="periodic-axis"):
            call()
        return
    with pytest.raises(NotImplementedError) as caught:
        call()
    assert "does not carry" in str(caught.value)
    assert A.ROW_WORDS[row] in str(caught.value)


EXTRA_SETTINGS = [
    ("waveguide_s_matrix", "_stencil_order", 4),
    ("vmap_sweep_batched", "_stencil_order", 4),
    ("vmap_sweep_batched", "_solver", "adi"),
    ("vmap_sweep_batched", "_precision", "mixed"),
]


@pytest.mark.parametrize("calculator,attr,value", EXTRA_SETTINGS)
def test_calculator_refuses_unread_setting_variant(monkeypatch, calculator, attr, value):
    sim, call = calculator_case(calculator)
    setattr(sim, attr, value)

    def forbidden(*args, **kwargs):
        pytest.fail("calculator started lax.scan before refusing its setting")

    monkeypatch.setattr(jax.lax, "scan", forbidden)
    with pytest.raises(NotImplementedError) as caught:
        call()
    assert A.ROW_WORDS[attr, ""] in str(caught.value)
    assert "does not carry" in str(caught.value)


def waveguide_upml_case():
    registered = _guide()
    sim = Simulation(freq_max=registered._freq_max, domain=registered._domain,
                     dx=registered._dx, boundary="upml", cpml_layers=4)
    sim._waveguide_ports = list(registered._waveguide_ports)
    return sim


def test_waveguide_refuses_upml_before_scan(monkeypatch):
    sim = waveguide_upml_case()

    def forbidden(*args, **kwargs):
        pytest.fail("waveguide started lax.scan with a substituted absorber")

    monkeypatch.setattr(jax.lax, "scan", forbidden)
    with pytest.raises(NotImplementedError, match="a UPML absorber"):
        sim.compute_waveguide_s_matrix(n_steps=4, normalize=False)


@pytest.mark.parametrize("calculator", A.CALCULATORS)
def test_yee_calculator_accepts_adi_cfl_setting(calculator):
    sim, _ = calculator_case(calculator)
    sim._adi_cfl_factor = 2.718
    assert ("_adi_cfl_factor", "") not in A.refused(sim, calculator)


@pytest.mark.parametrize("compute_s_params", [None, True])
def test_run_kerr_s_matrix_refuses_before_main_scan(monkeypatch, compute_s_params):
    sim, _ = calculator_case("s_matrix_scan")
    declare_physics(sim, "kerr")

    def forbidden(*args, **kwargs):
        pytest.fail("run started its main scan before S-matrix admission")

    monkeypatch.setattr(jax.lax, "scan", forbidden)
    with pytest.raises(NotImplementedError, match=r"run\(\).*Kerr.*compute_s_params=False"):
        sim.run(n_steps=4, compute_s_params=compute_s_params, skip_preflight=True)


def test_run_kerr_fields_only_reaches_main_scan(monkeypatch):
    sim, _ = calculator_case("s_matrix_scan")
    declare_physics(sim, "kerr")

    class Reached(Exception):
        pass

    def reached(*args, **kwargs):
        raise Reached

    monkeypatch.setattr(jax.lax, "scan", reached)
    with pytest.raises(Reached):
        sim.run(n_steps=4, compute_s_params=False, skip_preflight=True)


def test_devices_s_matrix_admission_precedes_distributed_runner(monkeypatch):
    from types import SimpleNamespace
    sim, _ = calculator_case("s_matrix_scan")
    sim._dt_pin = 1e-12
    monkeypatch.setattr(Simulation, "_dispatch_plan", lambda *args, **kwargs:
                        SimpleNamespace(lane="run_distributed", n_steps=4))

    def forbidden(*args, **kwargs):
        pytest.fail("distributed runner entered before S-matrix admission")

    monkeypatch.setattr("rfx.runners.distributed_v2.run_distributed", forbidden)
    with pytest.raises(NotImplementedError, match=r"run\(\).*pinned time step.*compute_s_params=False"):
        sim.run(n_steps=4, devices=[jax.devices()[0]], skip_preflight=True)


def test_material_fit_message_has_one_line_per_input():
    sim, _ = calculator_case("material_fit")
    sim._precision = "mixed"
    sim._stencil_order = 4
    rows = A.refused(sim, "material_fit")
    message = A.message("material_fit", rows, sim)
    assert len([line for line in message.splitlines() if line.startswith("  - ")]) == len(rows)
