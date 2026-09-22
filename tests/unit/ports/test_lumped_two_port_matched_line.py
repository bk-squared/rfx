"""Two one-cell ports on a line matched at both ends.

The single-port known-load line
(tests/unit/ports/test_lumped_port_known_load_line.py) gives the DIAGONAL an
exact answer. This fixture gives the OFF-DIAGONAL one, which the lumped family
did not have: two one-cell ports on the same parallel-plate channel, each
carrying ``impedance = Zc``, so each folds a ``Zc`` conductance into its own
cell and the line is matched at both ends. Then

    S11 = 0   and   S21 = exp(-j beta L),  |S21| = 1

at every frequency, independent of the line length.

What it pins
------------
Both entries, on both lanes, against the closed form — and lumped against wire
on the identical cells, where the two must agree because they ARE the same
port (same sigma, same injection, same V/I channels at ``n_live = 1``).

This fixture is what moved the lumped off-diagonal. Measured here across the
2026-09-21 work:

    lumped  |S11|  1.24831 -> 0.04206      (closed form 0)
    lumped  |S21|  2.24752 -> 1.00005      (closed form 1)
    wire    |S21|  1.00005 (unchanged)

The diagonal moved when a driven port started reading its terminal V/I pair.
The off-diagonal moved when the lumped N-port decomposition stopped being a
separately calibrated per-cell convention and became the wire family's
whole-port decomposition evaluated at one live cell — the per-cell frame built
its incident wave as ``(-V_ref + Z0 I)``, negated and on the pre-injection
sample, and read 2.25 where the closed form is 1 while the wire lane on the
same cells read 1.00005. One port cannot have two answers.
"""

import numpy as np
import pytest

C0 = 299792458.0

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import GaussianPulse

ETA0 = 376.730313668
DX = 1e-3
N_NODES = 5
FREQS_HZ = np.array([1.0, 2.5, 5.0, 7.5, 10.0]) * 1e9

# The same empirical envelope the single-port line uses, for the same reason:
# this cross-section is one cell by construction, so the fixture admits no mesh
# refinement and the bound is measured behaviour, not a derived error bar.
CLOSED_FORM_ATOL = 0.05


def _build(kind):
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
    for node in (1, N_NODES - 2):
        sim.add_port(
            position=(node * DX, 0.0, 0.0), component="ez", impedance=ETA0,
            waveform=GaussianPulse(f0=5e9, bandwidth=1.6), **extra,
        )
    return sim


def _s_matrix(kind):
    res = _build(kind).run(
        compute_s_params=True, s_param_freqs=FREQS_HZ, skip_preflight=True,
    )
    return np.asarray(res.s_params)


def _realized_port_separation(kind):
    """Port-to-port distance the GRID built, in metres, and its cell size.

    Read off the built grid rather than the declared geometry: the test is
    about where the ports ended up, and a rasterization change that moved
    them would otherwise be hidden by the declaration agreeing with itself.
    """
    sim = _build(kind)
    grid = sim._build_grid()
    idx = [grid.position_to_index(pe.position) for pe in sim._ports]
    n_cells = abs(int(idx[1][0]) - int(idx[0][0]))
    return n_cells * float(grid.dx), float(grid.dx)


@pytest.mark.parametrize("kind", ["lumped", "wire"])
def test_a_line_matched_at_both_ends_reflects_nothing(kind):
    """S11 = 0 in closed form, because each port terminates the line in Zc."""
    s11 = np.abs(_s_matrix(kind)[0, 0])
    assert s11.max() <= CLOSED_FORM_ATOL, (
        f"{kind}: closed form |S11| = 0 at every bin; read {np.round(s11, 5)} "
        f"at {FREQS_HZ / 1e9} GHz")


def test_the_two_lanes_agree_on_the_diagonal_of_the_same_cells():
    """Same cells, same sigma, same drive — so one diagonal, not two."""
    lumped = _s_matrix("lumped")[0, 0]
    wire = _s_matrix("wire")[0, 0]
    assert np.allclose(lumped, wire, rtol=0.0, atol=1e-6), (
        f"lumped {np.round(np.abs(lumped), 6)} vs wire "
        f"{np.round(np.abs(wire), 6)} on the identical cells")


def test_the_wire_off_diagonal_matches_the_closed_form():
    """|S21| = 1 on a matched lossless line. The wire lane reads it."""
    s21 = np.abs(_s_matrix("wire")[1, 0])
    assert np.abs(s21 - 1.0).max() <= 0.01, (
        f"closed form |S21| = 1 at every bin; wire read {np.round(s21, 5)}")


def test_the_lumped_off_diagonal_matches_the_closed_form():
    """|S21| = 1 on a matched lossless line. The lumped lane reads it too.

    The gate is the wire lane's own measured envelope on this fixture, 0.01
    against a worst bin of 0.00459. Before the lumped decomposition became
    the wire one at n_live = 1, this read 2.24752.
    """
    s21 = np.abs(_s_matrix("lumped")[1, 0])
    assert np.abs(s21 - 1.0).max() <= 0.01, (
        f"closed form |S21| = 1 at every bin; lumped read {np.round(s21, 5)}")


def test_the_two_lanes_agree_on_every_entry_of_the_same_cells():
    """One port, one S-matrix — diagonal and off-diagonal alike.

    This is the gate that makes the two lanes one implementation rather than
    two that happen to agree: the lumped decomposition IS the wire
    decomposition at one live cell, so every entry must match, not just the
    diagonal. Measured: exactly 0.0 on complex S, all four entries.
    """
    lumped = _s_matrix("lumped")
    wire = _s_matrix("wire")
    assert np.allclose(lumped, wire, rtol=0.0, atol=1e-6), (
        f"lumped {np.round(np.abs(lumped), 6)} vs wire "
        f"{np.round(np.abs(wire), 6)}")


# The port's reference plane is not the cell it occupies. MEASURED on this
# fixture: two one-cell ports two cells apart read as 1.5001 cells apart at
# 1 GHz, rising to 1.514 at 10 GHz, and the wire lane gives the same numbers
# to every digit. Magnitudes are unaffected (|S21| = 1.000055 ... 1.00459).
# The offset is half a cell, frequency-independent to four digits at the low
# end where the mesh's own dispersion is smallest; the climb above it is that
# dispersion. This is a statement of what was measured, on both lanes, before
# and after the 2026-09-21 work — NOT a claim about its cause.
#
# So the electrical length this line presents is the realized port separation
# less half a cell. Against that, the residual phase error is 0.0002 deg at
# 1 GHz and 0.1684 deg at 10 GHz. The gate below is 1.0 deg across the band,
# a measured envelope with 5.9x margin on the worst bin.
PHASE_GATE_DEG = 1.0
REFERENCE_PLANE_OFFSET_CELLS = 0.5


@pytest.mark.parametrize("kind", ["lumped", "wire"])
def test_s21_lags_by_the_electrical_length_of_the_line(kind):
    """S21 carries the phase of the line it crosses, not just its magnitude.

    |S21| = 1 alone cannot see a sign error in the incident wave: the whole
    point of the wave definition is which way the phase runs. This gates the
    COMPLEX S21 against exp(-j*beta*L_eff) on the air line, beta = omega/c.
    """
    l_realized, dx = _realized_port_separation(kind)
    l_eff = l_realized - REFERENCE_PLANE_OFFSET_CELLS * dx
    beta = 2.0 * np.pi * FREQS_HZ / C0
    s21 = _s_matrix(kind)[1, 0]

    measured_lag_deg = np.degrees(-np.angle(s21))
    expected_lag_deg = np.degrees(beta * l_eff)
    err_deg = np.degrees(np.angle(s21 * np.conj(np.exp(-1j * beta * l_eff))))

    worst = int(np.argmax(np.abs(err_deg)))
    assert np.abs(err_deg).max() <= PHASE_GATE_DEG, (
        f"{kind}: S21 lags beta*L_eff by "
        f"{measured_lag_deg[worst]:.4f} deg at "
        f"{FREQS_HZ[worst] / 1e9:.1f} GHz where the line's electrical length "
        f"is {expected_lag_deg[worst]:.4f} deg "
        f"(L_eff = {l_eff * 1e3:.4f} mm = realized {l_realized * 1e3:.4f} mm "
        f"less half a cell); error {err_deg[worst]:+.4f} deg against a "
        f"{PHASE_GATE_DEG} deg gate. Per bin: {np.round(err_deg, 4)} at "
        f"{FREQS_HZ / 1e9} GHz")
