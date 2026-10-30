"""B1b: reach every changed handler through its production caller.

Expected configuration errors skip unrecorded checks. Implementation failures
escape; recorded fallbacks retain their label but never absorb a timeout.
The two runner witnesses stop before FDTD, after the advisory under test.
"""
from types import SimpleNamespace as NS
import sys
import warnings

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.preflight import absorber, msl, ports, waveguide


class ReachedRunner(BaseException):
    """End a driver witness before field evolution."""


def _runner(*args, **kwargs):
    raise ReachedRunner


def _sim():
    return Simulation(freq_max=1e9, domain=(0.01, 0.01, 0.01),
                      dx=0.001, boundary="cpml", cpml_layers=2)


# id, expected configuration exception, recorded fallback
CASES = [
    ("ad_memory", ValueError, True),
    ("run_dc", ValueError, False),
    ("adi", ValueError, False),
    ("dielectric_grid", ValueError, False),
    ("dielectric_shapes", NotImplementedError, False),
    ("thin_depths", ValueError, False),
    ("msl_grid", ValueError, True),
    ("msl_geometry_grid", ValueError, True),
    ("half_node", ValueError, False),
    ("waveguide_setup", ValueError, True),
    ("profiling", ValueError, True),
    ("waveguide_sparams", ValueError, False),
    ("auto_probe_bounds", NotImplementedError, False),
    ("dispersive_material", KeyError, False),
    ("thin_bounds", NotImplementedError, False),
    ("msl_scan", ValueError, True),
    ("reflector_material", KeyError, True),
    ("reflector_bounds", NotImplementedError, True),
    ("wire_centers", IndexError, False),
    ("floating_material", KeyError, False),
    ("settling_cell", ValueError, False),
    ("visualize_bounds", NotImplementedError, False),
]


def _case(name, monkeypatch, fail):
    sim = _sim()
    shape = NS(bounding_box=lambda: ((0., 0., 0.), (.01, .01, .01)))
    sim._geometry = [NS(shape=shape, material_name="pec")]
    if name == "ad_memory":
        monkeypatch.setattr(sim, "_build_grid", fail)
        return sim._ad_memory_static_accounting, lambda r: r["grid_source"] == "estimated_from_domain"
    if name == "run_dc":
        import rfx.api._execute as execute
        monkeypatch.setattr(sim, "_auto_preflight", lambda **kw: NS(diagnostics=()))
        monkeypatch.setattr(sim, "_dispatch_plan", lambda **kw: NS(lane="run_nonuniform", n_steps=1))
        monkeypatch.setattr(sim, "_build_nonuniform_grid", fail)
        monkeypatch.setattr(sim, "_run_nonuniform", _runner)
        monkeypatch.setattr(execute, "bind_solve_call", lambda assembly, call: (call, None))
        return lambda: sim.run(n_steps=1, until_decay=1e-4, compute_s_params=False), lambda r: r == "runner"
    if name == "adi":
        import rfx.runners._admission as admission
        sim._solver = "adi"
        monkeypatch.setattr(admission, "_adi_homogeneous", fail)
        monkeypatch.setattr(sim, "_campaign_ctx", lambda: NS(error=None, realized=lambda: NS(empty=True)))
        return lambda: sim._validate_cfg_adi_interior_pec(warnings), lambda r: r is None
    if name.startswith("dielectric_"):
        if name == "dielectric_grid":
            monkeypatch.setattr(sim, "_build_grid", fail)
        else:
            import rfx.geometry.smoothing as smoothing
            monkeypatch.setattr(smoothing, "smoothed_shape_pairs", fail)
        return lambda: absorber._validate_cfg_dielectric_at_absorber_seam(sim, warnings), lambda r: r is None
    if name.startswith("thin_"):
        if name == "thin_depths":
            monkeypatch.setattr(sim, "_preflight_face_layers", fail)
        else:
            monkeypatch.setattr(shape, "bounding_box", fail)
        return lambda: absorber._conductor_in_thin_absorber_findings(sim, sim._dx), lambda r: r == []
    if name in ("msl_grid", "msl_geometry_grid", "msl_scan"):
        from tests.unit.ports.test_msl_clearance_diagnostic import _base
        sim, pe, grid = _base()
        if name == "msl_scan":
            monkeypatch.setattr(msl, "msl_probe_clearance_for_port", fail)
            label = "msl.probe_clearance_scan_failed"
        else:
            monkeypatch.setattr(sim, "_build_realized_grid", fail)
            label = ("msl.probe_clearance_unavailable" if name == "msl_grid"
                     else "msl.reflector_scan_unavailable")
        if name == "msl_geometry_grid":
            monkeypatch.setattr(sim, "_msl_assemble_once", lambda: None)
            call = lambda: msl._check_msl_port_geometry(sim, sim._dx, [0.002]*3, [0.002]*3)
        else:
            call = lambda: msl.preflight_msl_probe_clearance(sim, warnings)
        return call, lambda r: label  # warning's structured diagnostic is checked below
    if name == "half_node":
        sim._geometry = [NS(shape=Box((0, 0, .005), (.01, .01, .005)), material_name="pec")]
        sim.add_probe((.005, .005, .005), component="ez")
        monkeypatch.setattr(sim, "_build_realized_grid", fail)
        return lambda: ports.half_node_split_findings(sim), lambda r: r == []
    if name == "waveguide_setup":
        sim._waveguide_ports = [NS()]
        monkeypatch.setattr(sim, "_build_grid", fail)
        return lambda: waveguide._waveguide_setup_planes(sim, warnings, [1e9], 1, 1), lambda r: "waveguide_setup_audit_skipped"
    if name == "profiling":
        from rfx.profiling import profile_forward, format_report
        builds = []

        def build():
            if builds:
                return fail()
            builds.append(True)
            return NS(nx=2, ny=2, nz=2)

        sim = NS(_build_realized_grid=build, forward=lambda **kw: NS(time_series=np.zeros(1)))
        return lambda: profile_forward(sim, n_steps=1, warmup_trace=False), lambda r: r["cells"] is None and "cells=" not in format_report(r)
    if name == "waveguide_sparams":
        from tests.unit.nonuniform.test_dz_only_dispatch_contract import _dz_only_wg, _DZ_WR90_A
        sim = _dz_only_wg(_DZ_WR90_A)
        original_build = sim._build_nonuniform_grid

        def advisory_build():
            # Admission also builds a grid before the guarded advisory.
            if sys._getframe(1).f_code.co_name == "compute_waveguide_s_matrix":
                return fail()
            return original_build()

        monkeypatch.setattr(sim, "_build_nonuniform_grid", advisory_build)
        monkeypatch.setattr(sim, "_compute_waveguide_s_matrix_nu", _runner)
        return lambda: sim.compute_waveguide_s_matrix(n_steps=1, normalize="flux", skip_preflight=True), lambda r: r == "runner"
    if name == "auto_probe_bounds":
        from rfx.preflight._msl_probe_rules import msl_auto_probe_declaration
        monkeypatch.setattr(shape, "bounding_box", fail)
        return lambda: msl_auto_probe_declaration(sim, position=(.005, .005, 0), width=.002, height=.001, n_probes=3, eps_r_sub=None), lambda r: r[0] == 1.
    if name == "dispersive_material":
        monkeypatch.setattr(sim, "_resolve_material", fail)
        return lambda: absorber._validate_cfg_dispersive_pole_at_absorber_face(sim, warnings, sim._dx, [.002]*3, [.002]*3), lambda r: r is None
    if name.startswith("reflector_"):
        from rfx.preflight.msl_reflector import msl_nearest_downstream_reflector
        if name == "reflector_material":
            monkeypatch.setattr(sim, "_resolve_material", fail)
        else:
            monkeypatch.setattr(shape, "bounding_box", fail)
        return lambda: msl_nearest_downstream_reflector(sim._geometry, x_probe=.005, x_feed=.002, y_feed=.005, w_trace=.002, dx=.001, domain_y=.01, direction="+x", resolve_material=sim._resolve_material), lambda r: bool(r[2])
    if name == "wire_centers":
        monkeypatch.setattr(sim, "_build_realized_grid", fail)
        return lambda: ports._wire_port_cell_centers(sim, NS(component="ez")), lambda r: r is None
    if name == "floating_material":
        import rfx.model.materials as materials
        sim._ports = [NS(position=(.005, .005, .005), impedance=50., extent=None, excite=True, component="ez")]
        monkeypatch.setattr(materials, "assembled_as_pec", lambda *args: False)
        monkeypatch.setattr(sim, "_resolve_material", fail)
        return lambda: ports._validate_cfg_floating_single_cell_port(sim, warnings), lambda r: r is None
    if name == "settling_cell":
        from rfx.probes import settling
        monkeypatch.setattr(settling, "field_index", fail)
        return lambda: settling._cell_of(NS(), (.005, .005, .005), "ez"), lambda r: r is None
    if name == "visualize_bounds":
        import rfx.visualize as visualize
        # Exercise the geometry policy without the optional plotting package.
        fig = NS(data=[], update_layout=lambda **kw: None)
        monkeypatch.setattr(visualize, "_plotly", lambda: NS(Figure=lambda: fig))
        monkeypatch.setattr(shape, "bounding_box", fail)
        return lambda: visualize.visualize_structure(sim, include_sources=False, include_ntff=False, include_cpml=False), lambda r: len(r.data) == 0
    raise AssertionError(name)


@pytest.mark.parametrize("name,expected,recorded", CASES, ids=[row[0] for row in CASES])
@pytest.mark.parametrize("failure", ["expected", "unexpected", "signature", "timeout"])
def test_check_exception_policy(monkeypatch, name, expected, recorded, failure):
    # "signature" injects plain TypeError for a broken internal call, not
    # TracerArrayConversionError: supported traced inputs are checked below.
    error = {"expected": expected, "unexpected": RuntimeError, "signature": TypeError, "timeout": TimeoutError}[failure]
    reached = []

    def fail(*args, **kwargs):
        reached.append(True)
        raise error("B1b injected callee failure")

    call, verify = _case(name, monkeypatch, fail)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        if failure == "timeout" or (failure in ("unexpected", "signature") and not recorded):
            with pytest.raises(error, match="B1b injected callee failure"):
                call()
        else:
            try:
                result = call()
            except ReachedRunner:
                result = "runner"
            outcome = verify(result)
            if isinstance(outcome, str):
                assert any(outcome in str(w.message) or outcome == getattr(w.message, "code", None)
                           or outcome == getattr(getattr(w.message, "diagnostic", None), "code", None)
                           for w in seen), [str(w.message) for w in seen]
            else:
                assert outcome
            if name == "waveguide_sparams":
                assert any("UNCHECKED" in str(w.message) for w in seen)
            if name == "adi":
                assert not seen
    assert reached, f"{name}: injected callee was not reached"


@pytest.mark.parametrize("entry", ["preflight", "forward"])
def test_traced_wire_port_public_entries(entry):
    """A mesh design variable must survive the default preflight path."""
    import jax
    import jax.numpy as jnp
    from tests.unit.autodiff.test_wire_port_on_traced_mesh import _board, DX

    def traced(delta):
        sim = _board(delta)
        sim.add_probe((12 * DX, 10 * DX, 9 * DX), "ez")
        if entry == "preflight":
            sim.preflight()
            return delta
        result = sim.forward(n_steps=4)
        return jnp.sum(result.time_series ** 2)

    assert jax.make_jaxpr(traced)(jnp.float32(0.1 * DX)).jaxpr.eqns


class ProtocolOnlyShape:
    """The Shape protocol does not require a bounding box."""

    def mask_on_coords(self, x, y, z):
        return np.zeros((len(x), len(y), len(z)), dtype=bool)

    def mask(self, grid):
        return np.zeros(grid.shape, dtype=bool)


def _protocol_shape_sim(include_protocol_shape=True):
    sim = Simulation(freq_max=10e9, domain=(.03, .012, .006),
                     dx=.0005, boundary="cpml")
    sim.add_material("sub", eps_r=3.38)
    if include_protocol_shape:
        sim.add(ProtocolOnlyShape(), material="sub")
    sim.add(Box((0, 0, 0), (.03, .012, .001)), material="sub")
    return sim


def test_protocol_only_shape_add_msl_port():
    sim = _protocol_shape_sim()
    control = _protocol_shape_sim(include_protocol_shape=False)
    for board in (sim, control):
        board.add_msl_port((.005, .006, 0), width=.002, height=.001)
    assert len(sim._msl_ports) == 1
    assert sim._msl_auto_probe_lengths == control._msl_auto_probe_lengths
    assert sim._msl_auto_probe_spacing == control._msl_auto_probe_spacing
    assert sim._msl_auto_offset_min == control._msl_auto_offset_min


def test_protocol_only_shape_visualize_structure(monkeypatch):
    import rfx.visualize as visualize

    traces = []
    fig = NS(data=traces, add_trace=traces.append, update_layout=lambda **kw: None)
    monkeypatch.setattr(visualize, "_plotly", lambda: NS(Figure=lambda: fig, Mesh3d=NS))
    result = visualize.visualize_structure(
        _protocol_shape_sim(), include_sources=False, include_ntff=False,
        include_cpml=False,
    )
    assert result is fig
    assert len(result.data) == 1
    assert result.data[0].name == "sub"
    assert result.data[0].x == [0., 30., 30., 0., 0., 30., 30., 0.]
