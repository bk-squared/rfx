"""Pure-R shunt-network oracle on an internal TEM line through both absorbers.

D1: the cheap mesh checks the predicted one-cell residual; the slow ladder
also establishes first-order convergence to the unchanged pure-R answer.
"""
from functools import lru_cache

import json

import numpy as np
import pytest
import jax.numpy as jnp

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import GaussianPulse
from tests._interior_tem_line import (build, input_reflection, element_inductance,
                                      assert_predicted_residual, assert_first_order, residuals, assert_solved_ports)

ETA0 = 376.730313668
DX = 1e-3
N_NODES = 5
FREQS_HZ = np.array([1.0, 2.5, 5.0, 7.5, 10.0]) * 1e9
MESHES = (.25e-3, .125e-3, .0625e-3)


def _build(kind, r_over_zc):
    return _mesh(kind, r_over_zc, MESHES[0])[0]


def _magnetic_plane_advisory_fixture(kind, r_over_zc):
    sim = Simulation(
        freq_max=10e9,
        domain=((N_NODES - 1) * DX, DX, DX),
        dx=DX,
        boundary=BoundarySpec(
            x=Boundary(lo="pmc", hi="pmc"),
            y=Boundary(lo="pmc", hi="pmc"),
            z=Boundary(lo="pec", hi="pec"),
        ),
    )
    extra = {} if kind == "lumped" else {"extent": DX}
    sim.add_port(
        position=(1.0 * DX, 0.0, 0.0),
        component="ez",
        impedance=ETA0,
        waveform=GaussianPulse(f0=5e9, bandwidth=1.6),
        **extra,
    )
    sim.add_lumped_rlc(
        position=((N_NODES - 2) * DX, 0.0, 0.0),
        component="ez",
        R=r_over_zc * ETA0,
        topology="parallel",
    )
    return sim



def _mesh(kind, ratio, dx):
    return build(kind, ratio=ratio, dx=dx, cells=round(2.25e-3 / dx),
                 axial_positions=(.25e-3, 1.5e-3), declared_separation=1.25e-3)


@lru_cache(maxsize=None)
def _s11(kind, r_over_zc, dx=.25e-3):
    sim, line = _mesh(kind, r_over_zc, dx)
    res = sim.forward(
        port_s11_freqs=jnp.asarray(FREQS_HZ),
        num_periods=40.0,
        skip_preflight=True,
    )
    assert_solved_ports(res, line, kind)
    return np.asarray(res.s_params).reshape(-1)


@pytest.mark.parametrize("kind", ["lumped", "wire"])
def test_line_on_magnetic_plane_preflight_reports_in_plane_wave(kind):
    """The advisory must allow a TEM wave along the line on the y_lo plane."""
    report = _magnetic_plane_advisory_fixture(kind, 0.5).preflight()
    messages = [str(issue) for issue in report.issues
                if issue.code == "source_decoupled"]
    assert len(messages) == 1
    msg = messages[0]
    assert "sits on the magnetic-wall plane y_lo" in msg
    assert "magnetic image keeps tangential E" in msg
    assert "coupled to the interior" in msg
    assert "full symmetric object" in msg
    assert "Distributed kernels refuse magnetic faces" in msg



@pytest.mark.parametrize("kind", ["lumped", "wire"])
@pytest.mark.parametrize("r_over_zc", [0.5, 1.0, 2.0])
def test_lumped_port_s11_matches_the_closed_form_of_its_load(r_over_zc, kind, record_property):
    _, line = _mesh(kind, r_over_zc, MESHES[0])
    record_property("realized_d_m", line.length)
    pure = input_reflection(line, FREQS_HZ, r_over_zc * line.zc)
    predicted = input_reflection(line, FREQS_HZ, r_over_zc * line.zc,
                                 element_l=element_inductance(line))
    measured = _s11(kind, r_over_zc)
    record_property("s11", json.dumps(np.stack([measured.real, measured.imag], axis=-1).tolist()))
    record_property("residuals", json.dumps(residuals(measured, pure, predicted)))
    assert_predicted_residual(measured, pure, predicted)


@pytest.mark.parametrize("r_over_zc", [0.5, 1.0, 2.0])
def test_lumped_port_s11_is_passive_on_a_resistive_load(r_over_zc):
    """A resistively terminated passive line cannot reflect more than it gets.

    The pre-fix lane returned 1.248 on the matched load and 4.757 on
    R = 2 Zc, i.e. the reciprocal of the physical reflection.
    """
    s11 = np.abs(_s11("lumped", r_over_zc))
    assert s11.max() <= 1.0 + 1e-3, (
        f"R = {r_over_zc} Zc: |S11| = {np.round(s11, 5)} exceeds unity on a "
        "passive resistively terminated line"
    )


@pytest.mark.parametrize("r_over_zc", [0.5, 1.0, 2.0])
def test_lumped_and_one_cell_wire_port_agree_on_the_same_cell(r_over_zc):
    """The same cell declared two ways is one port, so it is one S11.

    ``extent=dx`` is the only difference between the two builds; the sigma
    folded into the cell and the voltage injected into it are identical
    (setup_wire_port / apply_wire_port both reduce to their lumped
    counterparts at n_live = 1).
    """
    lumped = _s11("lumped", r_over_zc)
    wire = _s11("wire", r_over_zc)
    assert np.allclose(lumped, wire, rtol=0.0, atol=1e-6), (
        f"R = {r_over_zc} Zc: lumped {np.round(np.abs(lumped), 6)} vs wire "
        f"{np.round(np.abs(wire), 6)} on the identical cell"
    )


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["lumped", "wire"])
@pytest.mark.parametrize("r_over_zc", [.5, 1., 2.])
def test_internal_coax_three_mesh_trend(kind, r_over_zc, record_property):
    errors, phases = [], []
    for dx in MESHES:
        _, line = _mesh(kind, r_over_zc, dx)
        record_property(f"realized_d_{dx}", line.length)
        measured = _s11(kind, r_over_zc, dx)
        pure = input_reflection(line, FREQS_HZ, r_over_zc * line.zc)
        predicted = input_reflection(line, FREQS_HZ, r_over_zc * line.zc,
                                     element_l=element_inductance(line))
        record_property(f"s11_{dx}", json.dumps(np.stack([measured.real, measured.imag], axis=-1).tolist()))
        record_property(f"residuals_{dx}", json.dumps(residuals(measured, pure, predicted)))
        assert_predicted_residual(measured, pure, predicted)
        errors.append(np.abs(measured - pure))
        phases.append(np.abs(np.angle(measured / pure)))
    record_property("complex_order", json.dumps(assert_first_order(MESHES, errors).tolist()))
    record_property("phase_order", json.dumps(assert_first_order(MESHES, phases).tolist()))


def test_residual_check_rejects_missing_inductance_and_shifted_load():
    """Arithmetic sensitivity: removing the check must make this test red."""
    from dataclasses import replace

    _, line = _mesh("lumped", 1., MESHES[0])
    pure = input_reflection(line, FREQS_HZ, line.zc)
    predicted = input_reflection(line, FREQS_HZ, line.zc,
                                 element_l=element_inductance(line))
    with pytest.raises(AssertionError, match="complex residual"):
        assert_predicted_residual(predicted, pure, pure)
    moved = input_reflection(replace(line, length=line.length + line.dx),
                             FREQS_HZ, line.zc, element_l=element_inductance(line))
    with pytest.raises(AssertionError, match="complex residual"):
        assert_predicted_residual(moved, pure, predicted)


def test_geometry_and_trend_checks_reject_restored_defects():
    from dataclasses import replace
    from tests._interior_tem_line import assert_realized_separation

    _, line = _mesh("lumped", 1., MESHES[0])
    with pytest.raises(AssertionError):
        assert_realized_separation(replace(line, length=line.length + line.dx), 1.25e-3)
    with pytest.raises(AssertionError):
        assert_first_order(MESHES, [1., 1., 1.])
    # A solved spec displaced from the declared snap must be observable even
    # before comparing the measured spectrum (no extra FDTD solve here).
    from types import SimpleNamespace
    spec = SimpleNamespace(i=line.port[0]+1, j=line.port[1], k=line.port[2])
    result = SimpleNamespace(lumped_port_sparams=((spec, None),))
    with pytest.raises(AssertionError, match="solved port cell"):
        assert_solved_ports(result, line)


@pytest.mark.parametrize("errors", [[.1, .5, .025], [.1, .2, .025], [.1, 1e-9, .025]],
                         ids=["middle-high", "nonmonotone", "middle-low"])
def test_order_rejects_reviewers_middle_mesh_mutations(errors):
    with pytest.raises(AssertionError, match="successive mesh orders"):
        assert_first_order([.1, .05, .025], errors)


@pytest.mark.parametrize("bad", [0., np.nan, np.inf, -1.])
def test_order_refuses_nonpositive_or_nonfinite_errors(bad):
    with pytest.raises(AssertionError, match="finite and strictly positive"):
        assert_first_order([.1, .05, .025], [.1, bad, .025])
