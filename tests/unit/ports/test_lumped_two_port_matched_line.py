"""Two one-cell ports on a line matched at both ends.

The single-port known-load line
(tests/unit/ports/test_lumped_port_known_load_line.py) gives the DIAGONAL an
exact answer. This fixture gives the OFF-DIAGONAL one, which the lumped family
did not have: two one-cell ports on the same parallel-plate channel, each
carrying ``impedance = Zc``, so each folds a ``Zc`` conductance into its own
cell and the line is matched at both ends. Then

    S11 = 0   and   S21 = exp(-j beta L),  |S21| = 1

at every frequency, independent of the line length.

What it pins, and what it deliberately does not
-----------------------------------------------
The DIAGONAL is gated against the closed form, on both lanes, and lumped is
required to equal wire on the identical cells. That is a second, independent
fixture for the 2026-09-21 driven-diagonal change: measured here, the lumped
diagonal went from ``max |S11| = 1.24831`` before it to ``0.04206`` after,
which is the wire lane's number to every digit.

The OFF-DIAGONAL is NOT gated to a value, because the value is wrong and
pinning it would freeze the defect. Measured on this fixture:

    lumped  |S21|  before 2.24752 ... 2.08434   after 2.24739 ... 2.07659
    wire    |S21|  before 1.00005 ... 1.00459   after 1.00005 ... 1.00459

against a closed form of 1 at every bin. The lumped off-diagonal is a factor
2.25 out, before this change and after it — the change moves it by at most
7.8e-03 (0.37 %), which is the Yee half-step phase now carried on every port's
current. It neither caused that error nor fixed it. The wire lane, on the
identical cells, is within 0.5 %.

So the closed-form off-diagonal check below is ``xfail(strict=True)``: it
records the right answer, it does not lock the wrong one, and it turns red the
day someone fixes the lumped off-diagonal convention without updating this
file. No cause is claimed for the factor here.
"""

import numpy as np
import pytest

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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The lumped off-diagonal reads |S21| ~ 2.25 where the closed form is "
        "1, on this fixture, both before and after the 2026-09-21 driven-"
        "diagonal change (2.24752 -> 2.24739 at 1 GHz). The wire lane on the "
        "identical cells reads 1.00005. This records the right answer without "
        "locking the wrong one; when the lumped off-diagonal convention is "
        "fixed, this turns red and should become a plain assertion."
    ),
)
def test_the_lumped_off_diagonal_matches_the_closed_form():
    """|S21| = 1 on a matched lossless line. The lumped lane does not read it."""
    s21 = np.abs(_s_matrix("lumped")[1, 0])
    assert np.abs(s21 - 1.0).max() <= 0.01, (
        f"closed form |S21| = 1 at every bin; lumped read {np.round(s21, 5)}")
