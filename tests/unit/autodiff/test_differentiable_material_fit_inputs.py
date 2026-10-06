"""Input-validation regression tests for the material-fit entry point (#1290)."""

import re

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.core.jax_utils import is_tracer
from rfx.differentiable_material_fit import differentiable_material_fit


CASES = [
    ("wire", "extent=", "Remove"),
    ("source_field", "add_source()", "Remove"),
    ("source_current", "add_source()", "Remove"),
    ("source_legacy", "add_source()", "Remove"),
    ("passive", "excite=False", "Remove"),
    ("passive_waveform", "excite=False", "Remove"),
    ("msl", "add_msl_port()", "Remove"),
    ("rlc", "add_lumped_rlc()", "Remove"),
    ("tfsf", "add_tfsf_source()", "Remove"),
    ("waveguide", "add_waveguide_port()", "Remove"),
    ("coaxial", "add_coaxial_port()", "Remove"),
    ("floquet", "add_floquet_port()", "Remove"),
    ("periodic", "periodic axes", "Remove"),
    ("periodic_spec", "periodic axes", "Remove"),
    ("conformal", "conformal=True", "Remove"),
    ("kerr", "chi3=", "Remove"),
    ("adi", "solver='adi'", "Use solver='yee'"),
    ("mixed", "precision='mixed'", "Use precision='float32'"),
    ("float64", "precision='float64'", "Use precision='float32'"),
    ("stencil", "stencil_order=4", "Use stencil_order=2"),
    ("interface", "interface_eps='dual_average'", "Use interface_eps='sampled'"),
]


def _fixture(case, eps_inf, debye_poles, lorentz_poles):
    kwargs = {"boundary": "pec"}
    if case in ("tfsf", "waveguide"):
        kwargs.update(boundary="cpml", cpml_layers=2)
    elif case == "periodic_spec":
        kwargs["boundary"] = BoundarySpec(x="periodic", y="pec", z="pec")
    elif case == "conformal":
        kwargs["boundary"] = BoundarySpec(
            x="pec", y=Boundary(lo="pec", hi="pec", conformal=True), z="pec")
    elif case == "adi":
        kwargs["solver"] = "adi"
    elif case in ("mixed", "float64"):
        kwargs["precision"] = case
    elif case == "stencil":
        kwargs["stencil_order"] = 4
    elif case == "interface":
        kwargs["interface_eps"] = "dual_average"
    if case == "periodic":
        kwargs["boundary"] = BoundarySpec(x="periodic", y="cpml", z="cpml")
    sim = Simulation(freq_max=5e9, domain=(.012, .010, .008), dx=.002, **kwargs)
    sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles,
                     lorentz_poles=lorentz_poles, chi3=1e-20 if case == "kerr" else 0.)
    sim.add(Box((.004, .002, .002), (.008, .008, .006)), material="dut")
    pulse = GaussianPulse(f0=3e9, bandwidth=.5)
    if case == "tfsf":
        sim.add_tfsf_source(f0=3e9, margin=1)
    elif case == "waveguide":
        sim.add_waveguide_port(.002, freqs=np.array([2e9, 3e9, 4e9]),
                              ref_offset=1, probe_offset=2)
    elif case.startswith("source_"):
        kind = {"source_field": "field", "source_current": "current",
                "source_legacy": None}[case]
        sim.add_source((.002, .004, .002), waveform=pulse, amplitude_kind=kind)
    else:
        port_kwargs = {}
        if case == "wire":
            port_kwargs["extent"] = .004
        elif case.startswith("passive"):
            port_kwargs["excite"] = False
        if case != "passive":
            port_kwargs["waveform"] = pulse
        sim.add_port((.002, .004, .002), "ez", **port_kwargs)
    sim.add_probe((.008, .006, .004), "ez")
    if case == "msl":
        sim.add_msl_port((.002, .004, .002), width=.004, height=.002,
                         n_probe_offset=3, n_probe_spacing=2, n_probes=3,
                         eps_r_sub=2.)
    elif case == "rlc":
        sim.add_lumped_rlc((.006, .004, .002), R=75., L=1e-9, C=1e-12)
    elif case == "coaxial":
        sim.add_coaxial_port((.006, .004, .008), pin_length=.002)
    elif case == "floquet":
        sim.add_floquet_port(.002, freqs=np.array([2e9, 3e9, 4e9]))
    return sim


@pytest.mark.parametrize("case,input_name,remedy", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("declaration", ["static", "traced"])
def test_unsupported_fixture_is_refused_before_assembly(
        monkeypatch, case, input_name, remedy, declaration):
    """Exercise the public factory at static setup and at its traced rebuild."""
    calls = []

    def factory(eps_inf, debye_poles, lorentz_poles):
        traced = is_tracer(eps_inf)
        calls.append(traced)
        active = case if declaration == "static" or traced else "supported"
        sim = _fixture(active, eps_inf, debye_poles, lorentz_poles)
        if declaration == "static":
            monkeypatch.setattr(sim, "_build_grid", unexpected_grid)
        return sim

    def unexpected_assembly(*args, **kwargs):
        raise AssertionError("unsupported fixture reached material assembly before refusal")

    def unexpected_grid(*args, **kwargs):
        raise AssertionError("unsupported static fixture reached grid construction before refusal")

    # Assert at assembly entry as well as the scan: a missing refusal must
    # fail here, before an unrelated setup error (e.g. the zero-ohm source).
    monkeypatch.setattr(Simulation, "_assemble_materials", unexpected_assembly)
    monkeypatch.setattr("rfx.simulation.run", unexpected_assembly)
    match = re.escape(input_name) + r".*#1290.*" + re.escape(remedy)
    with pytest.raises(NotImplementedError, match=match):
        differentiable_material_fit(
            factory, np.zeros((1, 1, 3), complex), np.array([2e9, 3e9, 4e9]),
            n_iterations=1, verbose=False)
    assert calls == ([False] if declaration == "static" else [False, True])


@pytest.mark.parametrize("conductor", ["volume", "sheet", "pinned_sheet"])
def test_port_on_pec_edge_is_refused_before_fdtd(monkeypatch, conductor):
    def factory(eps_inf, debye_poles, lorentz_poles):
        sim = _fixture("supported", eps_inf, debye_poles, lorentz_poles)
        # The point port's Ez edge starts at (2, 4, 2) mm.
        if conductor == "volume":
            sim.add(Box((.0, .002, .0), (.004, .006, .006)), material="pec")
        elif conductor == "sheet":
            sim.add_thin_conductor(Box((.002, .002, .0), (.002, .008, .006)))
        else:
            sim.add_pinned_sheet(plane_index=1, i_range=(1, 4), j_range=(0, 3),
                                 normal_axis=0)
        return sim

    def unexpected_step(*args, **kwargs):
        raise AssertionError("point port on PEC reached FDTD before refusal")

    monkeypatch.setattr("rfx.simulation.run", unexpected_step)
    with pytest.raises(NotImplementedError, match=r"add_port\(\).*PEC edge.*#1290.*Move"):
        differentiable_material_fit(
            factory, np.zeros((1, 1, 3), complex), np.array([2e9, 3e9, 4e9]),
            n_iterations=1, verbose=False)


class _FDTDReached(RuntimeError):
    pass


def _fit_to_scan(monkeypatch, factory):
    def reached(grid, materials, *args, **kwargs):
        from rfx.model.materials import validate_components
        from rfx.simulation import resolve_periodic
        assert materials.components is not None
        validate_components(materials, grid=grid, periodic=resolve_periodic(grid, None))
        raise _FDTDReached("material fit reached FDTD")

    monkeypatch.setattr("rfx.simulation.run", reached)
    # NotImplementedError is also a RuntimeError: retain the exception so
    # the test's own assertion distinguishes refusal from reaching the scan.
    with pytest.raises(RuntimeError) as caught:
        differentiable_material_fit(
            factory, np.zeros((1, 1, 3), complex), np.array([2e9, 3e9, 4e9]),
            n_iterations=1, verbose=False)
    return caught.value


def _acceptance_fixture(eps_inf, debye_poles, lorentz_poles, *,
                        position=(.002, .004, .002), **kwargs):
    sim = Simulation(freq_max=5e9, domain=(.012, .010, .008), dx=.002, **kwargs)
    sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles,
                     lorentz_poles=lorentz_poles)
    sim.add(Box((.008, .006, .004), (.010, .008, .006)), material="dut")
    sim.add_port(position, "ez", waveform=GaussianPulse(f0=3e9, bandwidth=.5))
    sim.add_probe((.008, .006, .004), "ez")
    return sim


def test_2d_z_periodic_fixture_reaches_fdtd(monkeypatch):
    def factory(eps_inf, debye_poles, lorentz_poles):
        return _acceptance_fixture(
            eps_inf, debye_poles, lorentz_poles, mode="2d_tmz",
            boundary=BoundarySpec(x="pec", y="pec", z="periodic"))

    error = _fit_to_scan(monkeypatch, factory)
    assert isinstance(error, _FDTDReached), f"2D z-periodic fixture was refused: {error}"


def test_ez_port_on_z_normal_pec_sheet_reaches_fdtd(monkeypatch):
    """Ez starts at z = 2 mm on a PEC sheet spanning the x-y plane."""
    def factory(eps_inf, debye_poles, lorentz_poles):
        sim = _acceptance_fixture(eps_inf, debye_poles, lorentz_poles, boundary="pec")
        sim.add_thin_conductor(Box((.0, .002, .002), (.006, .008, .002)))
        return sim

    error = _fit_to_scan(monkeypatch, factory)
    assert isinstance(error, _FDTDReached), f"normal Ez port was refused: {error}"


@pytest.mark.parametrize("x,refused", [(.0076, False), (.0036, True)],
                         ids=["rounded_clear", "rounded_pec"])
def test_off_node_port_next_to_pec_block(monkeypatch, x, refused):
    """2 mm cells; the PEC block spans x = 4–6 mm, with two CPML pad cells."""
    def factory(eps_inf, debye_poles, lorentz_poles):
        sim = _acceptance_fixture(
            eps_inf, debye_poles, lorentz_poles, position=(x, .0044, .0024),
            boundary="cpml", cpml_layers=2)
        sim.add(Box((.004, .002, .002), (.006, .008, .006)), material="pec")
        return sim

    # 7.6 mm rounds to x = 8 mm; 3.6 mm rounds to x = 4 mm.
    # Flooring would instead select x = 6 mm and x = 2 mm, respectively.
    error = _fit_to_scan(monkeypatch, factory)
    if refused:
        assert isinstance(error, NotImplementedError), f"PEC port was accepted: {error}"
        assert re.search(r"add_port\(\).*PEC edge.*#1290.*Move", str(error))
    else:
        assert isinstance(error, _FDTDReached), f"clear off-node port was refused: {error}"
