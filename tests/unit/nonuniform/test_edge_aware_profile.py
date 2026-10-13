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

_CHAIN_CASES = [
    pytest.param((s,) + (g, s) * (n - 1), d, 10.0, 10.0, id=f"strip{s}-gap{g}-dx{d}-x{n}")
    for s in (0.5, 2.0, 16.35) for g in (0.3, 1.15, 5.0)
    for d in (1.0, 2.0, 3.0) for n in (2, 3)
] + [
    # The two-strip coupler that once meshed at 8.6 um and 3,985 cells.
    pytest.param((16.35, 1.15, 16.35), d, 10.0, 10.0, id=f"coupler-dx{d}")
    for d in (1.0, 2.0, 3.0)
] + [
    # Review rows (2026-10-13): the chain r goes below the first exact fit
    # when another pair's inner stretch cannot be split inside the window.
    pytest.param((0.3717, 0.3924, 0.3717), 3.0, 20.0, 20.0, id="equal-strips-second-fit"),
    pytest.param((0.441, 0.2745, 9.6755), 2.0, 20.0, 20.0, id="unequal-second-fit"),
    # Domain ends one to two cells from the outer edges.
    pytest.param((2.4806, 3.5003, 0.5475), 0.5, 2.3309, 0.8091, id="ends-near"),
]


@pytest.mark.parametrize("widths_mm, dx_mm, lo_mm, hi_mm", _CHAIN_CASES)
def test_a_chain_of_strips_and_gaps_never_meshes_below_a_fifth_of_its_narrowest_feature(
        widths_mm, dx_mm, lo_mm, hi_mm):
    """Strips and gaps side by side share every inner edge. One edge cell r
    for the whole run, an exact fit D = (n + k) r of one strip or gap, keeps
    every stretch a whole number of cells. Pairing each strip and gap on its
    own let a shared edge keep the last pair's r: two lines that should
    coincide ended 6.7 um apart and the ratio cap dragged the axis to 8.6 um.

    Bound, from the drawing alone, derived for outer edges at least 2 dx
    from the domain ends. A pair's inner stretch S = D - k r splits into
    equal cells in [r/1.3, min(cap, 1.3 r)] whenever S/r >= 2/1.3 and
    r <= cap/1.3, so every pair is closed once r <= min(cap/1.3,
    D/(k + 1.54)), k <= 1.3. Consecutive exact fits of the narrowest pair
    differ by at most (2.84 + 1)/2.84 and the start cap is at least 0.75 dx,
    so the chain r is at least min(0.43 dx, f_min/3.84), and its neighbours
    grade by at most 1.3: no cell below min(dx, f_min)/5. With an end closer
    than about 1.65 dx the end stretch is shorter than the boundary cell, the
    band profile splits the edge cells and the feedback halves the cap, so
    there the bound is measured, not derived: worst 0.191 min(dx, f_min) over
    4,500 random chains with ends 1-2.5 dx away (base 0.163 on that drawing),
    0.222-0.344 on 5,200 others (review, 2026-10-13). The "ends-near" row
    below is one such drawing."""
    max_ratio = 1.3
    widths = [w * 1e-3 for w in widths_mm]
    dx, lo_margin = dx_mm * 1e-3, lo_mm * 1e-3
    pos = lo_margin + np.concatenate([[0.0], np.cumsum(widths)])
    edges = [SheetEdge(p, +1 if i % 2 == 0 else -1) for i, p in enumerate(pos)]
    hi = pos[-1] + hi_mm * 1e-3
    f_min = min(widths)                           # drawn, not from the code
    r_floor = min(dx, f_min) / 5.0
    prof = edge_aware_profile(0.0, hi, dx, sheet_edges=edges,
                              max_ratio=max_ratio)
    cells, nodes = prof.cells, prof.nodes
    assert cells.min() >= r_floor * (1.0 - 1e-12), (
        f"smallest cell {cells.min() * 1e6:.2f} um below "
        f"{r_floor * 1e6:.2f} um ({cells.size} cells)")
    if tuple(widths_mm) == (16.35, 1.15, 16.35):
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


@pytest.mark.parametrize("dx_mm, edges_mm, faces_mm, hi_mm", [
    pytest.param(3.0, ((6.762, 1), (15.228, -1), (17.815, 1),
                       (18.412, -1)), (17.9977,), 26.74, id="review-worst"),
    pytest.param(2.0, ((4.22025048256538, 1), (5.265273516271296, -1), (12.934507880442927, 1), (17.255767034353962, -1), (18.752719482315815, 1), (25.281089049271475, -1), (31.538168602355498, 1), (32.48988483737529, -1)), (4.438897939065866, 17.441681738187118), 38.12990606308621, id="stop-measurement"),
] + [
    pytest.param(dx, ((5 * dx, 1),) if single else
                 ((5 * dx, 1), (10 * dx, -1)),
                 ((5 + side * distance) * dx,), 15 * dx,
                 id=f"{'edge' if single else 'strip'}-dx{dx}-side{side}-d{distance}")
    for dx in (1.0, 3.0) for single in (True, False)
    for side in (-1, 1) for distance in (0.05, 0.2, 0.4, 0.9)
])
def test_nearby_faces_respect_the_drawn_feature_cost_floor(
        dx_mm, edges_mm, faces_mm, hi_mm):
    edges = [SheetEdge(p * 1e-3, side) for p, side in edges_mm]
    faces = [f * 1e-3 for f in faces_mm]
    hi, dx = hi_mm * 1e-3, dx_mm * 1e-3
    # Strip/gap widths and edge-to-face/end distances from the drawing.
    features = [b.position - a.position for a, b in zip(edges, edges[1:])]
    features += [abs(e.position - f) for e in edges for f in (0.0, *faces, hi)]
    fixed = sorted({0.0, hi, *faces})
    features += [b - a for a, b in zip(fixed, fixed[1:])
                 if not any(a < e.position < b for e in edges)]
    floor = min(dx, min(features)) / 5.0
    prof = edge_aware_profile(0.0, hi, dx, faces=faces, sheet_edges=edges)
    in_scope = all(abs(e.position - f) >= 2 * dx
                   for e in edges for f in fixed)
    if in_scope:
        assert prof.cells.min() >= floor
    assert prof.cells.min() >= 0.5 * floor
    for face in faces:
        assert np.min(np.abs(prof.nodes - face)) < 1e-12


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
    from rfx.harminv import matrix_pencil
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
    modes = [m for m in matrix_pencil(ts[int(0.15 * len(ts)):], float(res.dt),
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


# Decision record 7: synthetic seed-1695 drawings and old minima transcribed
# from r5_failure_drawings.txt. These pins precede the partner-choice search.
@pytest.mark.parametrize("dx, hi, edges, faces, old_min", [
    pytest.param(0.003, 0.03532211431476114, [
        (0.008654214104124333, 1),
        (0.009050705016413873, -1),
        (0.009851073695366466, 1),
        (0.024008221913177273, -1),
        (0.024417597530647418, 1),
        (0.028228654201909988, -1),
    ], [0.01856126059377253, 0.024929284909628723], 0.0002267157379662616, id="r5-row-90"),
    pytest.param(0.003, 0.08906292420061195, [
        (0.009314433336613612, 1),
        (0.010532870393175797, -1),
        (0.013468374579264428, 1),
        (0.01554136599849749, -1),
        (0.018937348833576017, 1),
        (0.01963067107619123, -1),
        (0.04283647479374392, 1),
        (0.04807851593197778, -1),
        (0.048660157106444835, 1),
        (0.04981252262355404, -1),
        (0.052092143921565845, 1),
        (0.0524176378467934, -1),
        (0.06225336441615326, 1),
        (0.07092034375620213, -1),
        (0.07146605071706699, 1),
        (0.07707699790837289, -1),
    ], [0.010203600841302052, 0.041848729206730696, 0.049074688952570104, 0.07395837481942148], 0.0001302515074581683, id="r5-row-182"),
    pytest.param(0.001, 0.06642425028450952, [
        (0.0015129471729440653, 1),
        (0.0017238493630259107, -1),
        (0.009180540146513505, 1),
        (0.013604746039455311, -1),
        (0.02232535284637544, 1),
        (0.041742278001573555, -1),
        (0.04533397599502909, 1),
        (0.04933699773227546, -1),
        (0.056052462555443834, 1),
        (0.05633475351874272, -1),
        (0.06211965326319849, 1),
        (0.06247788579372889, -1),
        (0.06365595414970822, 1),
        (0.06423931466759547, -1),
    ], [0.008772684702136213, 0.043044909954510054, 0.04833379025383552], 9.894709486734528e-05, id="r5-row-271"),
    pytest.param(0.003, 0.025085304263882644, [
        (0.006358929587494869, 1),
        (0.007120022439285442, -1),
        (0.007820733872227214, 1),
        (0.012487578255044238, -1),
        (0.01526477749540771, 1),
        (0.01563265886982936, -1),
    ], [0.006714969400286977, 0.013363086025085732], 7.596218181818177e-05, id="r5-row-300"),
    pytest.param(0.003, 0.04397026129699939, [
        (0.0104153924822072, 1),
        (0.010773667447279103, -1),
        (0.021796502677138466, 1),
        (0.02857293113362723, -1),
        (0.028968361641595112, 1),
        (0.030763264558587675, -1),
        (0.032323023810560444, 1),
        (0.04089773196285489, -1),
    ], [0.020687993844888328, 0.02938866940838591, 0.038614096040335805, 0.04314067705169529], 0.00029834110621880674, id="r5-row-331"),
    pytest.param(0.003, 0.036546141654957384, [
        (0.009485872346158847, 1),
        (0.009953033606573816, -1),
        (0.011191857373718417, 1),
        (0.014560279194324068, -1),
        (0.02278719658188682, 1),
        (0.023141978818341992, -1),
        (0.023553196477984047, 1),
        (0.024820407594808126, -1),
    ], [0.02132748280576794], 0.0001653408757806632, id="r5-row-337"),
    pytest.param(0.002, 0.027437199011097965, [
        (0.0034634474296709513, 1),
        (0.0044991470549661785, -1),
        (0.005010113533622523, 1),
        (0.00624690164532531, -1),
        (0.006594411297344854, 1),
        (0.01961189147820107, -1),
    ], [0.005333285631814906], 0.00013448934189471892, id="r5-row-628"),
    pytest.param(0.002, 0.04317194250292493, [
        (0.006107385613726809, 1),
        (0.0064480171971208556, -1),
        (0.0076604712930671845, 1),
        (0.008089293824522989, -1),
        (0.009166572091907322, 1),
        (0.009815335263965835, -1),
        (0.012452110677065475, 1),
        (0.012684930742675226, -1),
        (0.012899504738551764, 1),
        (0.013210463387275917, -1),
        (0.013490310683103718, 1),
        (0.013763326924302752, -1),
        (0.01664469157834239, 1),
        (0.035095882884551524, -1),
        (0.0353058448630753, 1),
        (0.03554533192074853, -1),
        (0.037618601485402796, 1),
        (0.04085172392514105, -1),
    ], [0.008774103573822106, 0.016055002999534423], 7.698393719180689e-05, id="r5-row-808"),
    pytest.param(0.001, 0.030015252855695958, [
        (0.003270744414321898, 1),
        (0.0084512803645985, -1),
        (0.008651752925206611, 1),
        (0.027634965180380104, -1),
        (0.02870958176836188, 1),
        (0.029353292985627846, -1),
    ], [0.028281675388045022], 0.00015345744302251322, id="r5-row-939"),
])
def test_partner_choices_preserve_the_pinned_old_minimum(
        monkeypatch, dx, hi, edges, faces, old_min):
    kwargs = dict(sheet_edges=[SheetEdge(*e) for e in edges], faces=faces)
    result = edge_aware_profile(0.0, hi, dx, **kwargs)
    assert result.cells.min() >= old_min
    old = _without_face_partners(monkeypatch, 0.0, hi, dx, **kwargs)
    assert old.cells.min() == old_min


def _without_face_partners(monkeypatch, *args, **kwargs):
    import rfx.mesh_edges as mesh_edges
    build = mesh_edges._build

    def without(*args, **kwargs):
        kwargs["partner_edges"] = frozenset()
        return build(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(mesh_edges, "_build", without)
        return edge_aware_profile(*args, **kwargs)


def test_equal_partner_minimum_keeps_the_old_profile_bitwise(monkeypatch):
    # First r5_equal_min_nonbitwise.json drawing: row 72, old 108 cells,
    # partner 97 cells; both minima are 7.663882676742922e-05 m.
    edges = [
        SheetEdge(0.001988406310317549, 1),
        SheetEdge(0.004185881805127413, -1),
        SheetEdge(0.01015365127185391, 1),
        SheetEdge(0.010950558770448014, -1),
        SheetEdge(0.013317994423846069, 1),
        SheetEdge(0.013548811846919443, -1),
        SheetEdge(0.017771949625286842, 1),
        SheetEdge(0.03206307701105497, -1),
        SheetEdge(0.036537263260579156, 1),
        SheetEdge(0.03736431649437254, -1),
        SheetEdge(0.03756453798825852, 1),
        SheetEdge(0.03782917995853423, -1),
    ]
    kwargs = dict(sheet_edges=edges, faces=[0.017045347737791226, 0.029773683943158644, 0.03322500540389193])
    result = edge_aware_profile(0.0, 0.041366084591415164, 0.002, **kwargs)
    old = _without_face_partners(monkeypatch, 0.0, 0.041366084591415164, 0.002, **kwargs)
    assert old.cells.min() == 7.663882676742922e-05
    assert old.cells.size == 108
    for actual, expected in zip(result, old):
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


def test_rounding_level_partner_win_keeps_the_old_profile_bitwise(monkeypatch):
    # Seed 1696 row 767: a partner profile beat old's smallest cell by one
    # rounding step (both 222 cells); a tie within 1e-9*dx keeps old.
    edges = [SheetEdge(p, s) for p, s in [
        (0.006709112086251946, 1), (0.007629987692076592, -1),
        (0.007890505186735808, 1), (0.009539682778815351, -1),
        (0.010844874212193068, 1), (0.01121890889028278, -1),
        (0.016686096696858572, 1), (0.02764028622626831, -1),
        (0.0343569385751419, 1), (0.035129869154249274, -1),
        (0.038588667331245885, 1), (0.03974620115477226, -1),
        (0.0515626195357832, 1), (0.053631356409708786, -1),
        (0.06204123281909169, 1), (0.06226604252237799, -1)]]
    kwargs = dict(sheet_edges=edges, faces=[0.008538374838962243, 0.038989848678036385])
    result = edge_aware_profile(0.0, 0.07245005253591424, 0.003, **kwargs)
    old = _without_face_partners(monkeypatch, 0.0, 0.07245005253591424, 0.003, **kwargs)
    assert old.cells.min() == 0.00019257677558042033
    assert old.cells.size == 222
    for actual, expected in zip(result, old):
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


def test_a_partner_candidate_that_does_not_converge_keeps_the_old_profile(monkeypatch):
    # Faces 0.65 mm and 1.3 mm past an edge: the all-partners band fit does
    # not converge (RuntimeError); the parent mesh (183 cells) must come back.
    edges = [SheetEdge(p, s) for p, s in [
        (0.0036768344883638324, 1), (0.014861837802453046, -1),
        (0.02110442040643769, 1), (0.027286130767069382, -1),
        (0.03301433818587427, 1), (0.03732399437630228, -1)]]
    kwargs = dict(sheet_edges=edges, faces=[0.027936130767069384, 0.02858613076719938])
    result = edge_aware_profile(0.0, 0.049432886634875856, 0.002, **kwargs)
    old = _without_face_partners(monkeypatch, 0.0, 0.049432886634875856, 0.002, **kwargs)
    assert old.cells.min() == 0.0001587886065051022
    assert old.cells.size == 183
    for actual, expected in zip(result, old):
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


def test_review_worst_face_has_at_least_76_339_um_cells():
    result = edge_aware_profile(
        0.0, 0.02674, 0.003, faces=[0.0179977], sheet_edges=[
            SheetEdge(6.762e-3, 1), SheetEdge(15.228e-3, -1),
            SheetEdge(17.815e-3, 1), SheetEdge(18.412e-3, -1)])
    assert result.cells.min() >= 76.339e-6


def test_e6_empty_partners_reproduce_the_old_face_shrink(monkeypatch):
    # pinned_r5.json, test_a_face_inside_an_edge_cell...: old 30 cells.
    old = _without_face_partners(
        monkeypatch, 0.0, 10e-3, 1e-3, faces=[5.2e-3], boundary_cell=1e-3,
        sheet_edges=[SheetEdge(5.0e-3, -1)])
    assert old.cells.min() == 4.150613333333327e-05
    assert old.cells.size == 30
