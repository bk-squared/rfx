"""A one-cell lumped port on a line terminated in a known resistor.

Root cause this pins (scripts/diagnostics/lumped_port_known_load_line.py):
the lumped lane sampled its physical V/I BEFORE source injection, read its
diagonal with the PASSIVE port-branch algebra ``(V + Z0·I)/(V - Z0·I)`` on a
DRIVEN port, and withheld the Yee half-step current phase.  On the fixture
below it returned |S11| 0.714 / 1.248 / 4.757 where the closed form is
0.333 / 0 / 0.333 — two of the three grossly non-passive.  The wire family
had already been corrected on all three counts (#683, #764, half-step
phase); a one-cell WIRE port is the SAME cell (setup_wire_port with
n_live=1 folds the same sigma, apply_wire_port with n_live=1 adds the same
injection), so the two lanes must agree here, and both must land on the
closed form.

The fixture is a 4-cell air-filled parallel-plate channel: PEC plates on z,
magnetic walls on y and behind the port, one cell wide and one cell high so
the TEM line impedance is Zc = eta0.  The port sits at node 1 with its
reference impedance set to Zc; a resistor R sits at node 3 and the magnetic
wall beyond it carries no current, so the line is terminated in R alone.
Then |S11| = |(R - Zc)/(R + Zc)| at EVERY frequency, independent of line
length, of beta, and of the mesh's numerical dispersion.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import GaussianPulse

ETA0 = 376.730313668
DX = 1e-3
N_NODES = 5
FREQS_HZ = np.array([1.0, 2.5, 5.0, 7.5, 10.0]) * 1e9

# An EMPIRICAL envelope on a fixture whose convergence is not shown, not a
# discretization estimate.  Measured 2026-09-21: both lanes sit 0.00043 from
# the closed form at 1 GHz, rising to 0.04206 at 10 GHz.
#
# What is known about that residual: it is antisymmetric in R (+0.0339 at
# R = Zc/2, -0.0339 at R = 2 Zc, at 10 GHz) and fits ONE effective reference
# impedance — Zc_eff/eta0 = 1.0805 from one load and 1.0782 from the other.
# No cause is claimed.  The fixture cannot be mesh-refined: its cross-section
# is one cell in each transverse direction BY CONSTRUCTION, which is how
# Zc = eta0*h/w is arranged, so refining dx on the fixed structure leaves the
# one-cell port no longer bridging the gap and |S11| goes to about 1.0 at
# every load.  That the wire lane reads the same residual separates lane from
# lane on a shared fixture; it is not a second witness for the fixture.
#
# So this bound is a lock on measured behaviour, not a derived error bar.
# Widening it needs a written root cause, the same as any other gate; the
# evidence these tests carry is the low-frequency agreement and the passivity
# bound below, not the size of this number.
CLOSED_FORM_ATOL = 0.05


def _build(kind, r_over_zc):
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
    report = _build(kind, 0.5).preflight()
    messages = [str(issue) for issue in report.issues
                if issue.code == "source_decoupled"]
    assert len(messages) == 1
    msg = messages[0]
    assert "sits on the magnetic-wall plane y_lo" in msg
    assert "coupled only within the plane" in msg
    assert "a line lying in the plane carries its wave" in msg
    assert "nothing launched there reaches the volume off the plane" in msg
    assert "To radiate into the volume, place the source one cell (1mm) off the plane" in msg
    assert "no wave radiates" not in msg
    assert "silent zero field" not in msg


@pytest.mark.parametrize("r_over_zc", [0.5, 1.0, 2.0])
def test_lumped_port_s11_matches_the_closed_form_of_its_load(r_over_zc):
    """|S11| of a driven one-cell lumped port is |(R - Zc)/(R + Zc)|."""
    gamma = abs((r_over_zc - 1.0) / (r_over_zc + 1.0))
    s11 = np.abs(_s11("lumped", r_over_zc))
    err = np.abs(s11 - gamma)
    assert err.max() <= CLOSED_FORM_ATOL, (
        f"R = {r_over_zc} Zc: closed form |S11| = {gamma:.5f} at every bin; "
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
