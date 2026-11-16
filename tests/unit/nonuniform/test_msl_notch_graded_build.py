"""The graded-mesh notch filter's metal is solved at its drawn size.

Build both coarsest-rung arms (on-node and offset) and read their realized
metal widths from the lattice, without a time step.
"""

from __future__ import annotations

import pytest

from validation.research.multiband_nu import msl_notch_graded as ins
from tests.crossval.msl_notch_filter import test_msl_notch_filter as case


ARMS_BUILT = ("A_on", "A_off")


@pytest.fixture(scope="module")
def built():
    """Both coarsest-rung arms, built once: assembly dominates this file."""
    out = {}
    for key in ARMS_BUILT:
        arm = ins.ARMS[key]
        sim, prof, board = ins.build_graded(arm.rung, arm.placement,
                                            arm.arm_length_m)
        out[key] = (sim, prof, board)
    return out


@pytest.mark.parametrize("key", ARMS_BUILT)
def test_the_metal_is_solved_at_its_drawn_size(built, key):
    """The point of the two arms, read off the lattice.

    On the offset arm the metal's outermost node stands 0.35 of a fine cell
    inside the drawn edge, so its SOLVED edge lands on the drawn 600 um.  On
    the on-node arm the node is the drawn edge and the solved metal is
    0.7 cell wider.
    """
    sim, _prof, board = built[key]
    g = ins.realized_graded(sim)
    f = board["fine_cell_m"]
    solved_width = g["trace_width_node_span_m"] + 2.0 * ins.EDGE_OFFSET * f
    if board["edge_offset_m"] == 0.0:
        assert solved_width > case.TRACE_WIDTH_M * 1.05
    else:
        assert solved_width == pytest.approx(case.TRACE_WIDTH_M, abs=1e-12)
