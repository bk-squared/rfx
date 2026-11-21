"""MSL continuation, surviving refusals and mixed-port face precedence (#1512)."""
from dataclasses import replace
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from rfx import Box, Cylinder, Simulation
from rfx.errors import ConfigurationWarning
from rfx.geometry.smoothing import continued_conductor_shape
from rfx.preflight.line_stub import line_stub_findings, require_no_resonant_line_stub


DX = .0005


def line(overhang=.002, *, start=0., graded=False, thin=False, terminates=None,
         boundary="cpml"):
    """The issue's 1–3 mm tails at the unchanged default sixteen layers."""
    sim = Simulation(domain=(.012, .008, .004), dx=DX, freq_max=40e9,
                     boundary=boundary, snap="declared",
                     **({"dx_profile": np.r_[np.full(8, DX), np.full(16, DX/2),
                                            np.full(8, DX)]} if graded else {}))
    sim.add_material("board", eps_r=3.66)
    sim.add(Box((0, 0, 0), (.012, .008, .001)), material="board")
    sim.add(Box((0, 0, 0), (.012, .008, 0)), material="pec")
    strip = Box((start, .003, .001), (.012, .005, .001))
    if thin:
        sim.add_thin_conductor(strip)
    else:
        sim.add(strip, material="pec")
    sim.add_msl_port((start + overhang, .004, 0), width=.002, height=.001,
                     eps_r_sub=3.66, terminates=terminates, mode="uniform",
                     n_probe_offset=2, n_probe_spacing=1, n_probes=3)
    return sim


def strip_mask(sim):
    grid = sim._build_realized_grid()
    sheets = []
    assemble = sim._assemble_materials_nu if hasattr(grid, "dx_arr") else sim._assemble_materials
    assemble(grid, pec_sheets=sheets, pec_wires=[])
    return grid, np.asarray(sheets[-1].footprint)


def refusal(call):
    """Expose absence or wrong type as an assertion in the named witness."""
    try:
        call()
    except Exception as error:
        return type(error), str(error)
    return None, ""


@pytest.mark.parametrize("overhang", [.001, .002, .003])
@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("thin", [False, True])
@pytest.mark.parametrize("boundary", ["cpml", "upml"])
def test_overhang_continues(overhang, graded, thin, boundary):
    sim = line(overhang, graded=graded, thin=thin, boundary=boundary)
    grid, mask = strip_mask(sim)
    assert grid.pad_x_lo == grid.pad_x_hi == 16
    for pad in (mask[:grid.pad_x_lo], mask[-grid.pad_x_hi-1:-1]):
        assert pad.any()
        assert np.array_equal(pad, np.broadcast_to(mask[grid.pad_x_lo], pad.shape))
    assert line_stub_findings(sim, grid) == []
    assert refusal(lambda: require_no_resonant_line_stub(sim, (20e9, 40e9))) == (None, "")


def test_interior_gap_refused():
    sim = line(start=DX)
    findings = line_stub_findings(sim)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.overhang_m == pytest.approx(.002)
    kind, message = refusal(lambda: require_no_resonant_line_stub(sim, (20e9, 40e9)))
    assert kind is not None and issubclass(kind, ValueError)
    assert "is an open stub that shorts the port" in message
    assert "start the signal strip at that coordinate" in message


def test_realized_reach_boundary():
    reached = line()
    short = line(start=DX)
    short._msl_ports[0] = replace(short._msl_ports[0], position=reached._msl_ports[0].position)
    grid, mask = strip_mask(reached)
    _, gap = strip_mask(short)
    assert mask[grid.pad_x_lo].any() and mask[:grid.pad_x_lo].any()
    assert not gap[grid.pad_x_lo].any() and not gap[:grid.pad_x_lo].any()
    assert line_stub_findings(reached) == []
    kind, message = refusal(lambda: require_no_resonant_line_stub(short, (20e9, 40e9)))
    assert kind is not None and "open stub" in message


def test_unextendable_refused():
    sim = line()
    sim._geometry.pop()
    shape = Cylinder((.003, .004, .001), radius=.003, height=0., axis="z")
    sim.add_thin_conductor(shape)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solved = continued_conductor_shape(sim, sim._build_realized_grid(), shape)
    assert solved is shape
    assert any(isinstance(w.message, ConfigurationWarning)
               and "No continuation is applied" in str(w.message) for w in caught)
    kind, message = refusal(lambda: require_no_resonant_line_stub(sim, (20e9, 40e9)))
    assert kind is not None and "open stub" in message


@pytest.mark.parametrize("direction", ["+x", "-x", "+y", "-y"])
@pytest.mark.parametrize("thin", [False, True])
def test_mixed_hold_precedence(direction, thin):
    sim = Simulation(domain=(8., 8., 8.), dx=1., freq_max=1e6, boundary="cpml")
    strip = Box((0., 0., 3.), (8., 8., 3.))
    if thin:
        sim.add_thin_conductor(strip)
    else:
        sim.add(strip, material="pec")
    sim.add_port((4., 4., 0.), extent=3., terminates=strip)
    sim.add_msl_port((4., 4., 0.), width=2., height=3., direction=direction)
    grid = sim._build_realized_grid()
    solved = continued_conductor_shape(sim, grid, strip)
    axis = "xy".index(direction[-1])
    behind = 0 if direction.startswith("+") else 1
    bounds = (solved.corner_lo, solved.corner_hi)
    for a in (0, 1):
        for side in (0, 1):
            expected = (bounds[side][a] < 0 if side == 0 else bounds[side][a] > 8)
            assert expected == (a == axis and side == behind)
    assert solved.corner_lo[2] == solved.corner_hi[2] == 3.


@pytest.mark.parametrize("value", [[0], 0, [Box((0, 0, 0), (1, 1, 1))], False])
def test_registration_refusal(value, monkeypatch):
    import rfx.geometry.port_termination as pt
    sim = line()
    def forbidden(*args):
        raise AssertionError("conductor resolution must not precede MSL refusal")
    monkeypatch.setattr(pt, "conductor_entries", forbidden)
    kind, message = refusal(lambda: sim.add_msl_port(
        (.004, .004, 0), width=.002, height=.001, terminates=value))
    assert kind is ValueError
    assert message == (
        "add_msl_port at (0.004, 0.004, 0): terminates= is not accepted for MSL ports: "
        "a microstrip line continues through every absorbing face it reaches (#1512). "
        "To end the line at the port, start its strip at the port node.")
    assert len(sim._msl_ports) == 1


def test_empty_realizations_identical():
    masks = [strip_mask(line(terminates=value))[1] for value in (None, (), [])]
    assert all(np.array_equal(masks[0], mask) for mask in masks[1:])
    assert masks[0][:16].any() and masks[0][-17:-1].any()


def test_interop_refusal():
    from rfx.interop import design_to_dict, simulation_from_design
    document = design_to_dict(line())
    document["excitations"]["msl_ports"][0]["terminates"] = [["_geometry", 999]]
    kind, message = refusal(lambda: simulation_from_design(document))
    assert kind is ValueError
    assert "terminates= is not accepted for MSL ports" in message
    assert "start its strip at the port node" in message


def test_port_continues_warning():
    sim = line(terminates=())
    findings = []
    sink = SimpleNamespace(warn=lambda issue, **kwargs: findings.append(issue))
    sim._validate_cfg_port_conductor_continues(sink)
    assert findings == []
    sim._msl_ports.clear()
    sim.add_port((.002, .004, 0), extent=.001)
    sim._validate_cfg_port_conductor_continues(sink)
    assert len(findings) == 1
    assert findings[0].code == "port_conductor_continues"


def test_run_forward_fields_equal():
    sim = line(.001, thin=True)
    _, mask = strip_mask(sim)
    assert mask[:16].any() and mask[-17:-1].any()
    other = line(.001, thin=True)
    for board in (sim, other):
        board.add_vector_probe((.001, .004, .0005))
        board.add_vector_probe((0., .004, .0005))
    kwargs = dict(n_steps=8, skip_preflight=True)
    run = sim.run(compute_s_params=False, **kwargs)
    forward = other.forward(checkpoint=False, **kwargs)
    assert run.time_series.shape == forward.time_series.shape == (8, 12)
    assert np.array_equal(run.time_series, forward.time_series)
    assert np.asarray(run.time_series[:, 2]).any()
