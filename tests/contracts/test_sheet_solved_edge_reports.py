"""A PEC sheet's in-plane size is reported where the solve puts its edges (#1375).

A PEC sheet's in-plane edge is solved ``rfx.mesh_edges.EDGE_OFFSET`` (0.35)
of the OUTSIDE cell beyond its last covered node.  So a sheet whose last node
sits 0.35 cell inside the drawn edge (``edge_aware_profiles``) is solved at
its drawn size, and a sheet with a node ON each drawn edge is solved 0.35
cell too long at each end.  The reports must say the same, from one model
(``rfx.mesh_edges.solved_sheet_span``): preflight's ``sheet_effective_size``
fires only for the second board; the off-lattice census
(``off_lattice_design_edges``) and ``fidelity_report`` measure each in-plane
face as |solved edge - drawn edge|, so both are silent on the first board and
read 0.35 x the outside cell on the second, and for every sheet face the
census residual equals the fidelity report's.  Before #1375 both measured each
face against its nearest node: they called the registered sheet 0.35 cell off
and the on-node sheet exact, the opposite of what is solved.  A third board
has two strips of the right solved length, each shifted 0.35 cell away from
the gap between them (drawn gap 0.6 mm, solved gap 1.3 mm): sheet_effective_size cannot see it,
so the census must.

Asymmetric on purpose: the sheet is not centred, and on the on-node board the
outside cells differ from each other and from the inside cells, so an offset
taken from the wrong cell or mirrored between ends changes the numbers.
Build only, no time step.
"""

from __future__ import annotations

import contextlib
import io
import re
import warnings

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.mesh_edges import EDGE_OFFSET, edge_aware_profiles

MM = 1e-3
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


def _reports(sim):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            issues = [issue.to_dict() for issue in sim.preflight()]
        report = sim.fidelity_report(print_report=False)
    items = [it for it in report
             if str(it.get("entity", "")).startswith("thin_conductor")]
    return issues, items


def _board(profiles):
    sim = Simulation(freq_max=10e9, domain=DOMAIN, dx=DX, boundary="pec",
                     **profiles)
    sim.add_thin_conductor(_sheet())
    return sim


def _axes(item):
    return {ax["axis"]: ax for ax in item["axes"]}


_UNIT = {"m": 1.0, "mm": 1e-3, "µm": 1e-6, "nm": 1e-9}
_ROW = re.compile(r"(thin_conductor\[\d+\]) '[^']*' \(sheet\) ([xyz]): extent "
                  r"[^,]+, worst face residual ([\d.]+)(m|mm|µm|nm) ")


def _census_rows(issues):
    """{(entity label, axis): worst face residual in um} from the census."""
    rows = {}
    for d in issues:
        if d["code"] != "off_lattice_design_edges":
            continue
        head = d["message"].split(". OBSERVED")[0]
        for m in _ROW.finditer(head + " "):
            rows[(m.group(1), m.group(2))] = (
                float(m.group(3)) * _UNIT[m.group(4)] * 1e6)
    return rows


def _assert_census_is_the_fidelity_residual(issues, items):
    """For every sheet in-plane face: the census row (or its absence) is the
    fidelity report's face residual against the census's own 0.5 % bar."""
    rows = _census_rows(issues)
    for it in items:
        label = str(it["entity"]).split(" ")[0]
        for a, ax in _axes(it).items():
            ext = ax["declared_extent_um"]
            if ext <= 0.0:
                continue
            worst = max(ax["face_residual_um"])
            if worst / ext > 5e-3:
                assert (label, a) in rows, (label, a, worst, rows)
                # the census prints 4 significant digits
                assert rows[(label, a)] == pytest.approx(worst, rel=1e-3), (
                    label, a, rows[(label, a)], worst)
            else:
                assert (label, a) not in rows, (label, a, worst, rows)


def test_registered_sheet_reports_its_drawn_size():
    profiles = edge_aware_profiles(DOMAIN, DX, sheets=[_sheet()], axes="xy")
    issues, items = _reports(_board(profiles))
    codes = [d["code"] for d in issues]
    assert "sheet_effective_size" not in codes, codes
    assert "off_lattice_design_edges" not in codes, codes
    (item,) = items
    axes = _axes(item)
    for a, (lo, hi) in DRAWN.items():
        r_lo, r_hi = axes[a]["realized_um"]
        assert abs(r_lo - lo * 1e6) <= TOL_UM and abs(r_hi - hi * 1e6) <= TOL_UM, (
            f"{a}: realized ({r_lo}, {r_hi}) um, drawn ({lo * 1e6}, {hi * 1e6}) um")
        assert abs(axes[a]["realized_extent_um"] - (hi - lo) * 1e6) <= TOL_UM
    kinds = [f["kind"] for f in item["findings"]]
    assert "off-lattice-face" not in kinds, item["findings"]
    _assert_census_is_the_fidelity_residual(issues, items)


def test_sheet_continued_into_cpml_reports_only_the_free_end_residual():
    """The -1 mm end continues into CPML; compare it with the 0 mm domain face.

    The free end covers node 8 mm and is solved at 8.35 mm, 50 um beyond
    its drawn 8.3 mm end. Removing the census's domain clamp reads 1 mm.
    """
    sim = Simulation(freq_max=10e9, domain=DOMAIN, dx=DX, boundary="cpml",
                     cpml_layers=4)
    sim.add_thin_conductor(Box((-1 * MM, Y_LO, Z), (8.3 * MM, Y_HI, Z)))
    issues, items = _reports(sim)
    (item,) = items
    axis = _axes(item)["x"]
    assert np.allclose(axis["declared_um"], (0.0, 8300.0), rtol=0.0, atol=TOL_UM)
    assert np.allclose(axis["realized_um"], (0.0, 8350.0), rtol=0.0, atol=TOL_UM)
    assert np.allclose(axis["face_residual_um"], (0.0, 50.0), rtol=0.0, atol=TOL_UM)
    rows = _census_rows(issues)
    assert rows[("thin_conductor[0]", "x")] == pytest.approx(50.0, rel=1e-3), rows
    _assert_census_is_the_fidelity_residual(issues, items)


def test_on_node_sheet_reports_the_solved_overhang():
    issues, items = _reports(_board(dict(dx_profile=ON_NODE_X,
                                         dy_profile=ON_NODE_Y)))
    codes = [d["code"] for d in issues]
    assert "sheet_effective_size" in codes, codes
    (item,) = items
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
    rows = _census_rows(issues)
    assert rows == pytest.approx({
        ("thin_conductor[0]", "x"): EDGE_OFFSET * 0.8e-3 * 1e6,
        ("thin_conductor[0]", "y"): EDGE_OFFSET * 0.9e-3 * 1e6}, rel=1e-3), rows
    _assert_census_is_the_fidelity_residual(issues, items)


def test_shifted_strips_of_the_right_size_are_reported():
    """Two strips drawn 2.0-6.7 and 7.3-12.0 mm on 1 mm cells.  Each covers
    nodes 2-6 or 8-12 and is solved 4.7 mm long, its drawn length, so
    sheet_effective_size is silent on x; but each is shifted 0.35 mm
    away from the other, and the gap is solved 1.3 mm instead of 0.6."""
    sim = Simulation(freq_max=10e9, domain=(16 * MM, 8 * MM, 6 * MM), dx=MM,
                     boundary="pec")
    sim.add_thin_conductor(Box((2.0 * MM, 2 * MM, 3 * MM), (6.7 * MM, 5 * MM, 3 * MM)))
    sim.add_thin_conductor(Box((7.3 * MM, 2 * MM, 3 * MM), (12.0 * MM, 5 * MM, 3 * MM)))
    issues, items = _reports(sim)
    ses = [d["message"] for d in issues if d["code"] == "sheet_effective_size"]
    assert all(" x: drawn" not in m.split(". OBSERVED")[0] for m in ses), ses
    rows = _census_rows(issues)
    assert rows.get(("thin_conductor[0]", "x")) == pytest.approx(350.0, rel=1e-3), rows
    assert rows.get(("thin_conductor[1]", "x")) == pytest.approx(350.0, rel=1e-3), rows
    xs = sorted(_axes(it)["x"]["realized_um"] for it in items)
    assert np.allclose(xs, [(1650.0, 6350.0), (7650.0, 12350.0)],
                       rtol=0.0, atol=TOL_UM), xs
    _assert_census_is_the_fidelity_residual(issues, items)


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
