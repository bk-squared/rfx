"""Mesh lines that follow conductor edges (`rfx.mesh_edges`).

Two layers. The invariants of the line placement are pure arithmetic and run
in milliseconds. The physics test runs the 2-D fin the rule was measured on
(`scripts/diagnostics/pec_sheet_edge_offset.py`): a fin drawn at a length that
does not fit the grid, solved on the uniform grid and on the edge-aware
profile, each held against the fine-mesh reference.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from rfx.mesh_edges import (EDGE_OFFSET, SheetEdge, edge_aware_profile,
                            edge_aware_profiles)


def _straddling_cell(nodes, x):
    k = int(np.searchsorted(nodes, x, side="right")) - 1
    return nodes[k], nodes[k + 1]


@pytest.mark.parametrize("length_mm", [4.0, 4.125, 4.375, 4.75, 5.5, 5.9])
@pytest.mark.parametrize("dx_mm", [1.0, 0.5])
def test_last_metal_node_sits_edge_offset_of_its_own_cell_inside_the_edge(
        length_mm, dx_mm):
    """The invariant the rule exists for, read on the REALIZED cell.

    The ratio cap can split the cell that straddles the edge; the node must
    then sit 0.3 of the split cell inside, not 0.3 of the requested one.
    """
    edge, dx = length_mm * 1e-3, dx_mm * 1e-3
    prof = edge_aware_profile(0.0, 10e-3, dx, boundary_cell=dx,
                              sheet_edges=[SheetEdge(edge, -1)])
    lo, hi = _straddling_cell(prof.nodes, edge)
    assert (edge - lo) / (hi - lo) == pytest.approx(EDGE_OFFSET, abs=1e-6)
    assert prof.edge_cells[0] == pytest.approx(hi - lo, rel=1e-9)
    assert not np.any(np.isclose(prof.nodes, edge, atol=1e-9)), \
        "no mesh line may sit ON a sheet edge"


def test_metal_on_the_high_side_mirrors_the_rule():
    edge = 3.3e-3
    prof = edge_aware_profile(0.0, 10e-3, 1e-3, boundary_cell=1e-3,
                              sheet_edges=[SheetEdge(edge, +1)])
    lo, hi = _straddling_cell(prof.nodes, edge)
    assert (hi - edge) / (hi - lo) == pytest.approx(EDGE_OFFSET, abs=1e-6)


def test_faces_stay_on_nodes_and_the_profile_keeps_its_contract():
    faces = [9.0e-3]
    prof = edge_aware_profile(
        0.0, 12e-3, 1e-3, faces=faces, boundary_cell=1e-3, max_ratio=1.3,
        sheet_edges=[SheetEdge(4.4e-3, +1), SheetEdge(6.1e-3, -1)])
    for f in faces:
        assert np.min(np.abs(prof.nodes - f)) < 1e-12
    assert prof.cells.sum() == pytest.approx(12e-3, abs=1e-12)
    assert prof.cells[0] == prof.cells[-1] == 1e-3
    ratios = prof.cells[1:] / prof.cells[:-1]
    assert np.max(np.maximum(ratios, 1.0 / ratios)) <= 1.3 + 1e-9
    assert prof.cells.max() <= 1e-3 + 1e-15, "no cell above the target"


def test_a_strip_narrower_than_two_edge_cells_is_meshed_not_refused():
    """A 0.6 mm strip at dx = 1 mm: both edge cells land on one metal node,
    whose electrical width is 2 * EDGE_OFFSET cells = the drawn 0.6 mm."""
    prof = edge_aware_profile(
        0.0, 10e-3, 1e-3, boundary_cell=1e-3,
        sheet_edges=[SheetEdge(5.0e-3, +1), SheetEdge(5.6e-3, -1)])
    inside = prof.nodes[(prof.nodes > 5.0e-3) & (prof.nodes < 5.6e-3)]
    assert inside.size >= 1
    for edge, side in ((5.0e-3, +1), (5.6e-3, -1)):
        lo, hi = _straddling_cell(prof.nodes, edge)
        frac = (hi - edge) / (hi - lo) if side > 0 else (edge - lo) / (hi - lo)
        assert frac == pytest.approx(EDGE_OFFSET, abs=1e-6)


# --- chains: strips and gaps side by side, no face between them ------------

def _chain_drawing(strip, gap, n_strips, start=10e-3, margin=10e-3):
    """``n_strips`` coplanar strips of width ``strip`` separated by ``gap``,
    starting at ``start``; the domain ends ``margin`` beyond the last edge."""
    edges = []
    for j in range(n_strips):
        a = start + j * (strip + gap)
        edges += [SheetEdge(a, +1), SheetEdge(a + strip, -1)]
    return edges, edges[-1].position + margin


_CHAIN_CASES = [
    pytest.param(s, g, d, n, id=f"strip{s}-gap{g}-dx{d}-x{n}")
    for s in (0.5, 2.0, 16.35) for g in (0.3, 1.15, 5.0)
    for d in (1.0, 2.0, 3.0) for n in (2, 3)
] + [
    # The two-strip coupler that once meshed at 8.6 um and 3,985 cells.
    pytest.param(16.35, 1.15, d, 2, id=f"coupler-dx{d}") for d in (1.0, 2.0, 3.0)
]


@pytest.mark.parametrize("strip_mm, gap_mm, dx_mm, n_strips", _CHAIN_CASES)
def test_a_chain_of_strips_and_gaps_never_meshes_below_half_its_narrowest_feature(
        strip_mm, gap_mm, dx_mm, n_strips):
    """Strips and gaps side by side share every inner edge. One edge cell r
    for the whole run, an exact fit D = (n + k) r of one strip or gap, keeps
    every stretch a whole number of cells. Pairing each strip and gap on its
    own let a shared edge keep the last pair's r: two lines that should
    coincide ended 6.7 um apart and the ratio cap dragged the axis to 8.6 um.

    Bound, from the drawing alone: an exact fit with n = ceil(D/dx - k) has
    r > D dx / (D + dx) >= min(D, dx) / 2, and the cells next to it grade by at
    most max_ratio, so no cell is below min(dx, f_min) / (2 max_ratio), f_min
    the narrowest strip or gap."""
    max_ratio = 1.3
    strip, gap, dx = strip_mm * 1e-3, gap_mm * 1e-3, dx_mm * 1e-3
    edges, hi = _chain_drawing(strip, gap, n_strips)
    f_min = min(strip, gap)                       # drawn, not from the code
    r_floor = min(dx, f_min) / (2.0 * max_ratio)
    prof = edge_aware_profile(0.0, hi, dx, sheet_edges=edges,
                              max_ratio=max_ratio)
    cells, nodes = prof.cells, prof.nodes
    assert cells.min() >= r_floor * (1.0 - 1e-12), (
        f"smallest cell {cells.min() * 1e6:.2f} um below "
        f"{r_floor * 1e6:.2f} um ({cells.size} cells)")
    if strip_mm == 16.35 and gap_mm == 1.15 and n_strips == 2:
        assert cells.size < 200
    # The node placement still holds on every edge, read on the realized cell.
    for e, end in zip(edges, prof.metal_end_nodes):
        lo, hi_c = _straddling_cell(nodes, e.position)
        frac = ((e.position - lo) if e.metal_side < 0
                else (hi_c - e.position)) / (hi_c - lo)
        assert frac == pytest.approx(EDGE_OFFSET, abs=1e-6)
        assert np.min(np.abs(nodes - end)) < 1e-12
    assert not np.any(np.isclose(nodes[:, None],
                                 [e.position for e in edges], rtol=0,
                                 atol=1e-9)), "no mesh line ON a sheet edge"
    assert abs(nodes[-1] - hi) < 1e-12 and nodes[0] == 0.0
    ratios = cells[1:] / cells[:-1]
    assert np.max(np.maximum(ratios, 1.0 / ratios)) <= max_ratio + 1e-9
    assert cells.max() <= dx * (1.0 + 1e-12), "no cell above the target"


def test_a_face_inside_an_edge_cell_shrinks_the_cell_instead_of_moving_the_face():
    prof = edge_aware_profile(
        0.0, 10e-3, 1e-3, faces=[5.2e-3], boundary_cell=1e-3,
        sheet_edges=[SheetEdge(5.0e-3, -1)])
    assert np.min(np.abs(prof.nodes - 5.2e-3)) < 1e-12
    lo, hi = _straddling_cell(prof.nodes, 5.0e-3)
    assert hi <= 5.2e-3 + 1e-12
    assert (5.0e-3 - lo) / (hi - lo) == pytest.approx(EDGE_OFFSET, abs=1e-6)


def test_an_edge_outside_the_axis_is_refused():
    with pytest.raises(ValueError, match="outside"):
        edge_aware_profile(0.0, 10e-3, 1e-3,
                           sheet_edges=[SheetEdge(11e-3, -1)])


# --- from the drawing: seams, staggered ends, a face on an edge ------------

_DOM = (40e-3, 40e-3, 10e-3)


def _sheet(x0, x1, y0, y1):
    from rfx import Box
    return Box((x0, y0, 2e-3), (x1, y1, 2e-3))


def test_a_seam_between_two_abutting_sheets_is_not_an_edge():
    """One conductor drawn as two Boxes meshes like the same conductor drawn
    as one: the shared end is interior metal."""
    one = edge_aware_profiles(_DOM, 1e-3, sheets=[_sheet(10e-3, 30e-3, 10e-3, 20e-3)])
    two = edge_aware_profiles(_DOM, 1e-3, sheets=[
        _sheet(10e-3, 20e-3, 10e-3, 20e-3), _sheet(20e-3, 30e-3, 10e-3, 20e-3)])
    for key in one:
        np.testing.assert_allclose(two[key], one[key], rtol=0, atol=1e-15)


def test_a_feed_line_butting_a_wider_patch_keeps_the_patch_edge():
    patch = _sheet(10e-3, 20e-3, 10e-3, 20e-3)
    feed = _sheet(20e-3, 30e-3, 14e-3, 16e-3)
    prof = edge_aware_profiles(_DOM, 1e-3, sheets=[patch, feed], axes="x")
    nodes = np.concatenate([[0.0], np.cumsum(prof["dx_profile"])])
    lo, hi = _straddling_cell(nodes, 20e-3)      # the patch's edge survives
    assert (20e-3 - lo) / (hi - lo) == pytest.approx(EDGE_OFFSET, abs=1e-6)


def test_two_staggered_sheets_meeting_end_to_end_are_refused_in_words():
    with pytest.raises(ValueError, match="cannot sit inside both"):
        edge_aware_profiles(_DOM, 1e-3, sheets=[
            _sheet(10e-3, 20e-3, 10e-3, 20e-3), _sheet(20e-3, 30e-3, 15e-3, 25e-3)])


def test_sheets_on_different_planes_are_never_a_seam():
    """Same x, opposite ends, but 2 mm apart in z: both are free edges. They
    cannot share a line, so the layout is refused -- never merged as a seam
    (which would silently drop a real edge)."""
    from rfx import Box
    upper = Box((20e-3, 10e-3, 4e-3), (30e-3, 20e-3, 4e-3))
    with pytest.raises(ValueError, match="different planes"):
        edge_aware_profiles(_DOM, 1e-3, axes="x",
                            sheets=[_sheet(10e-3, 20e-3, 10e-3, 20e-3), upper])


def test_the_advisory_keeps_a_free_edge_under_a_sheet_on_another_plane():
    from rfx import Box
    lower = _sheet(10e-3, 20e-3, 10e-3, 20e-3)
    upper = Box((20e-3, 10e-3, 4e-3), (30e-3, 20e-3, 4e-3))
    rows = _sheet_size_rows([lower, upper])
    assert len(rows) == 1
    assert rows[0].count(f"(+{200 * EDGE_OFFSET * 1e-3 / 10e-3:.2f}%)") == 4


def test_coincident_opposite_edges_are_refused_not_divided_by():
    with pytest.raises(ValueError, match="same position"):
        edge_aware_profile(0.0, 10e-3, 1e-3, sheet_edges=[
            SheetEdge(5e-3, -1), SheetEdge(5e-3, +1)])


def test_a_declared_face_on_a_sheet_edge_warns():
    with pytest.warns(UserWarning, match="coincides with a sheet edge"):
        edge_aware_profiles(_DOM, 1e-3, axes="x", faces={"x": [20e-3]},
                            sheets=[_sheet(10e-3, 20e-3, 10e-3, 20e-3)])


def _sheet_size_rows(shapes):
    from rfx.api import Simulation
    sim = Simulation(freq_max=5e9, domain=_DOM, dx=1e-3, boundary="pec")
    for s in shapes:
        sim.add(s, material="pec")
    sim.add_source((5e-3, 5e-3, 5e-3), component="ez", amplitude_kind="field")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.preflight()
    return [str(i) for i in report.issues if "sheet dimension" in str(i)]


def test_the_advisory_does_not_count_a_seam_as_a_free_edge():
    """Two 10 mm halves of one 20 mm conductor have one free end each along
    x and two along y, so x reads half of y's relative error. Counting the
    seam would make them equal."""
    import re
    rows = _sheet_size_rows([_sheet(10e-3, 20e-3, 10e-3, 20e-3),
                             _sheet(20e-3, 30e-3, 10e-3, 20e-3)])
    assert len(rows) == 1
    rel = {ax: [float(v) for v in re.findall(
        rf"{ax}: drawn 10mm, nodes cover 10mm, solved as [0-9.]+mm "
        r"\(\+([0-9.]+)%\)", rows[0])] for ax in "xy"}
    assert rel["x"] == [pytest.approx(100 * EDGE_OFFSET * 1e-3 / 10e-3)] * 2
    assert rel["y"] == [pytest.approx(200 * EDGE_OFFSET * 1e-3 / 10e-3)] * 2



# --- physics: the fin the offset was measured on ---------------------------

#: Fine-mesh reference for a 4.75 mm fin in the 20 x 10 mm PEC box, 2-D TMz,
#: lowest resonance: dx = 0.25 mm 19.9241 GHz, dx = 0.125 mm 19.8883 GHz,
#: extrapolated at the tip's measured FIRST order 19.8525 GHz
#: (scripts/diagnostics/pec_sheet_edge_offset.py).
F_REF_HZ = 19.8525e9
FIN_M = 4.75e-3


def _fin_resonance(dy_profile, *, snap="strict"):
    from rfx import Box
    from rfx.api import Simulation
    from rfx.harminv import harminv
    a, b, dx = 0.020, 0.010, 1e-3
    kw = {} if dy_profile is None else {"dy_profile": dy_profile}
    # The graded lane refuses mode="2d_tmz" and names the one-cell 3-D box
    # between PEC walls, which is what it ran before, bit for bit (#1340). The
    # uniform arm keeps the uniform lane's own 2-D update: the 3-D box there
    # resonates 22 MHz lower (1.90 % against 1.79 % below the reference).
    mode = "2d_tmz" if dy_profile is None else "3d"
    # #1138: geometry[0] y solved -8.4211 % off; this test checks resonance after strict refusal.
    sim = Simulation(freq_max=30e9, domain=(a, b, dx), boundary="pec", dx=dx,
                     mode=mode, snap=snap, **kw)
    sim.add(Box((a / 2, 0.0, 0.0), (a / 2, FIN_M, dx)), material="pec")
    sim.add_source((0.0063, 0.0031, 0.0), component="ez")
    sim.add_probe((0.0137, 0.0069, 0.0), component="ez")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.run(num_periods=60, compute_s_params=False)
    ts = np.asarray(res.time_series)[:, 0]
    modes = [m for m in harminv(ts[int(0.15 * len(ts)):], float(res.dt),
                                12e9, 30e9) if m.amplitude > 0]
    top = max(m.amplitude for m in modes)
    return min(m.freq for m in modes if m.amplitude > 1e-3 * top)


def test_a_fin_that_does_not_fit_the_grid_is_solved_at_its_drawn_length():
    """Uniform grid: the 4.75 mm fin realizes 4 cells and resonates 1.8 %
    low against the reference. Edge-aware lines: 0.08 % -- what is left is
    the grid's own dispersion (0.10 % on the empty box at this cell size).
    Both bounds sit between the two measurements, so the test reds if the
    rule stops working AND if the uniform grid stops showing the defect
    (which would mean the rasterizer changed and the offset needs
    re-measuring)."""
    # #1138: first pin strict refusal, then retain the measured resonance control.
    with pytest.raises(ValueError, match="conductor sheet dimension"):
        _fin_resonance(None)
    f_uniform = _fin_resonance(None, snap="declared")
    prof = edge_aware_profile(0.0, 0.010, 1e-3, boundary_cell=1e-3,
                              sheet_edges=[SheetEdge(FIN_M, -1)])
    f_edge = _fin_resonance(prof.cells)
    err_uniform = abs(f_uniform / F_REF_HZ - 1.0)
    err_edge = abs(f_edge / F_REF_HZ - 1.0)
    assert err_uniform > 0.010, f"uniform grid error {100 * err_uniform:.3f} %"
    assert err_edge < 0.003, f"edge-aware error {100 * err_edge:.3f} %"
