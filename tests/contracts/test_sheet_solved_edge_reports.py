"""A PEC sheet's in-plane size is reported where the solve puts its edges (#1375).

A PEC sheet's in-plane edge is solved ``rfx.mesh_edges.EDGE_OFFSET`` (0.35)
of the OUTSIDE cell beyond its last covered node.  So a sheet whose last node
sits 0.35 cell inside the drawn edge (``edge_aware_profiles``) is solved at
its drawn size, and a sheet with a node ON each drawn edge is solved 0.35
cell too long at each end.  The reports must say the same: preflight's
``sheet_effective_size`` fires only for the second board, the off-lattice
census (``off_lattice_design_edges``) leaves a sheet's in-plane size to that
advisory in both, and ``fidelity_report`` gives the solved extent -- the drawn
size for the first board, drawn + 0.35 x (outside cell at each end) for the
second.  Before #1375 the census and the fidelity report measured each face
against its nearest node: they called the registered sheet 0.35 cell off and
the on-node sheet exact, the opposite of what is solved.

Asymmetric on purpose: the sheet is not centred, and on the on-node board the
outside cells differ from each other and from the inside cells, so an offset
taken from the wrong cell or mirrored between ends changes the numbers.
Build only, no time step.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.mesh_edges import EDGE_OFFSET, edge_aware_profiles

DX = 1e-3
DOMAIN = (20e-3, 16e-3, 6e-3)
X_LO, X_HI, Y_LO, Y_HI, Z = 5.7e-3, 13.7e-3, 4.1e-3, 9.6e-3, 3e-3

# Node ON each drawn edge.  Outside the sheet: 0.7 / 0.8 mm cells on x,
# 0.6 / 0.9 mm on y; inside: 0.5 mm on x, 0.55 mm on y.
ON_NODE_X = np.array([1, 1, 1, 1, 1, 0.7] + [0.5] * 16
                     + [0.8, 0.5, 1, 1, 1, 1, 1]) * 1e-3
ON_NODE_Y = np.array([1, 1, 1, 0.5, 0.6] + [0.55] * 10
                     + [0.9, 0.5, 1, 1, 1, 1, 1]) * 1e-3
OUTSIDE = {"x": (0.7e-3, 0.8e-3), "y": (0.6e-3, 0.9e-3)}
DRAWN = {"x": (X_LO, X_HI), "y": (Y_LO, Y_HI)}
TOL_UM = 1e-6 * DX * 1e6


def _sheet():
    return Box((X_LO, Y_LO, Z), (X_HI, Y_HI, Z))


def _reports(profiles):
    sim = Simulation(freq_max=10e9, domain=DOMAIN, dx=DX, boundary="pec",
                     **profiles)
    sim.add_thin_conductor(_sheet())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            codes = [issue.to_dict()["code"] for issue in sim.preflight()]
        report = sim.fidelity_report(print_report=False)
    (item,) = [it for it in report
               if str(it.get("entity", "")).startswith("thin_conductor")]
    return codes, item


def _axes(item):
    return {ax["axis"]: ax for ax in item["axes"]}


def test_registered_sheet_reports_its_drawn_size():
    profiles = edge_aware_profiles(DOMAIN, DX, sheets=[_sheet()], axes="xy")
    codes, item = _reports(profiles)
    assert "sheet_effective_size" not in codes, codes
    assert "off_lattice_design_edges" not in codes, codes
    axes = _axes(item)
    for a, (lo, hi) in DRAWN.items():
        r_lo, r_hi = axes[a]["realized_um"]
        assert abs(r_lo - lo * 1e6) <= TOL_UM and abs(r_hi - hi * 1e6) <= TOL_UM, (
            f"{a}: realized ({r_lo}, {r_hi}) um, drawn ({lo * 1e6}, {hi * 1e6}) um")
        assert abs(axes[a]["realized_extent_um"] - (hi - lo) * 1e6) <= TOL_UM
    kinds = [f["kind"] for f in item["findings"]]
    assert "off-lattice-face" not in kinds, item["findings"]


def test_on_node_sheet_reports_the_solved_overhang():
    codes, item = _reports(dict(dx_profile=ON_NODE_X, dy_profile=ON_NODE_Y))
    assert "sheet_effective_size" in codes, codes
    # A sheet's in-plane size is sheet_effective_size's report, not the census's.
    assert "off_lattice_design_edges" not in codes, codes
    axes = _axes(item)
    for a, (lo, hi) in DRAWN.items():
        c_lo, c_hi = OUTSIDE[a]
        want = ((lo - EDGE_OFFSET * c_lo) * 1e6, (hi + EDGE_OFFSET * c_hi) * 1e6)
        got = axes[a]["realized_um"]
        assert np.allclose(got, want, rtol=0.0, atol=TOL_UM), (
            f"{a}: realized {got} um, solved edges {want} um")
        assert np.allclose(axes[a]["face_residual_um"],
                           (EDGE_OFFSET * c_lo * 1e6, EDGE_OFFSET * c_hi * 1e6),
                           rtol=0.0, atol=TOL_UM), axes[a]["face_residual_um"]
    kinds = sorted((f["kind"], f["axis"]) for f in item["findings"])
    assert kinds == [("off-lattice-face", "x"), ("off-lattice-face", "y")], kinds


@pytest.mark.parametrize("axis", ["x", "y"])
def test_on_node_profile_puts_a_node_on_each_drawn_edge(axis):
    """The on-node board is what it claims: nodes at both drawn edges, and
    the outside cells named in OUTSIDE."""
    cells = {"x": ON_NODE_X, "y": ON_NODE_Y}[axis]
    nodes = np.concatenate([[0.0], np.cumsum(cells)])
    lo, hi = DRAWN[axis]
    i0 = int(np.argmin(np.abs(nodes - lo)))
    i1 = int(np.argmin(np.abs(nodes - hi)))
    assert abs(nodes[i0] - lo) < 1e-12 and abs(nodes[i1] - hi) < 1e-12
    assert np.allclose((cells[i0 - 1], cells[i1]), OUTSIDE[axis], atol=1e-15)
