"""Known resistors on an internal PEC lattice coax after #1221 B3b.

The former one-cell PEC/PMC channel used magnetic side walls as its width.
The rebuilt solve uses tests._interior_tem_line: internal PEC outer sheets,
a filament inner conductor, asymmetric radial air edges, C'=3.75 epsilon0, Zc=eta0/3.75.
At dx=0.25 mm the grid is (10,9,9), port/load nodes (1,3,3)/(6,3,3),
with unequal 1/3-cell open end stubs. The declared separation is 5.08 cells;
the oracle uses its realized 5 cells. Each stub contributes j*tan(beta*l)/Zc.
The 0.05 magnitude bar, ordinary passivity and lane identity gates remain.
The magnetic-plane advisory below intentionally retains its separate probe.
"""
from functools import lru_cache

import json

import numpy as np
import pytest
import jax.numpy as jnp

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import GaussianPulse
from tests._interior_tem_line import build, input_reflection

ETA0 = 376.730313668
DX = 1e-3
N_NODES = 5
FREQS_HZ = np.array([1.0, 2.5, 5.0, 7.5, 10.0]) * 1e9
CLOSED_FORM_ATOL = 0.05


def _build(kind, r_over_zc):
    return build(kind, ratio=r_over_zc)[0]


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



@lru_cache(maxsize=None)
def _s11(kind, r_over_zc):
    res = _build(kind, r_over_zc).forward(
        port_s11_freqs=jnp.asarray(FREQS_HZ),
        num_periods=20.0,
        skip_preflight=True,
    )
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
    """The realized coax includes both open end stubs in its TEM answer."""
    _, line = build(kind, ratio=r_over_zc)
    gamma = np.abs(input_reflection(line, FREQS_HZ, r_over_zc * line.zc))
    s11 = np.abs(_s11(kind, r_over_zc))
    err = np.abs(s11 - gamma)
    record_property("measured_magnitude", json.dumps(s11.tolist()))
    record_property("closed_form_magnitude", json.dumps(gamma.tolist()))
    print(f"{kind} R/Zc={r_over_zc}: magnitude error {max(err):.9g}")
    assert err.max() <= CLOSED_FORM_ATOL, (
        f"R = {r_over_zc} Zc: closed form |S11| = {np.round(gamma, 5)}; "
        f"lumped read {np.round(s11, 5)} (per-bin error {np.round(err, 5)}) "
        f"at {FREQS_HZ / 1e9} GHz"
    )


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


def test_internal_coax_axial_mesh_trend(record_property):
    """Fixed transverse cells and declarations, three axial mesh resolutions.

    The transverse lattice TEM impedance stays eta0/3.75. The end cells stay
    0.25 mm as required by the NU grid; the seven interior cells subdivide
    by 1/2/4. Report the snapped lengths separately. This measures axial
    trend, not convergence of a continuum round coax or of the point feed.
    """
    errors = []
    for refinement in (1, 2, 4):
        profile = np.r_[.25e-3, np.full(7 * refinement, .25e-3 / refinement), .25e-3]
        sim, line = build(ratio=2., profile=profile)
        result = sim.forward(port_s11_freqs=FREQS_HZ, num_periods=40., skip_preflight=True)
        s11 = np.asarray(result.s_params).reshape(-1)
        expected = input_reflection(line, FREQS_HZ, 2 * line.zc)
        error = float(np.max(np.abs(np.abs(s11) - np.abs(expected))))
        errors.append(error)
        record_property(f"mesh_{refinement}", str((line.shape, line.left, line.length, line.right, error)))
        print(f"axial subdivision {refinement}: {line}, max magnitude error={error:.9g}")
    assert errors[2] < errors[1] < errors[0] <= CLOSED_FORM_ATOL
