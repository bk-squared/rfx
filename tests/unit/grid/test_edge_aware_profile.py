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

from rfx.mesh_edges import EDGE_OFFSET, SheetEdge, edge_aware_profile


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


# --- physics: the fin the offset was measured on ---------------------------

#: Fine-mesh reference for a 4.75 mm fin in the 20 x 10 mm PEC box, 2-D TMz,
#: lowest resonance: dx = 0.25 mm 19.9241 GHz, dx = 0.125 mm 19.8883 GHz,
#: Richardson 19.8764 GHz (scripts/diagnostics/pec_sheet_edge_offset.py).
F_REF_HZ = 19.8764e9
FIN_M = 4.75e-3


def _fin_resonance(dy_profile):
    from rfx import Box
    from rfx.api import Simulation
    from rfx.harminv import harminv
    a, b, dx = 0.020, 0.010, 1e-3
    kw = {} if dy_profile is None else {"dy_profile": dy_profile}
    sim = Simulation(freq_max=30e9, domain=(a, b, dx), boundary="pec", dx=dx,
                     mode="2d_tmz", **kw)
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
    """Uniform grid: the 4.75 mm fin realizes 4 cells and resonates 1.9 %
    low against the reference. Edge-aware lines: 0.09 % -- what is left is
    the grid's own dispersion (0.10 % on the empty box at this cell size).
    Both bounds sit between the two measurements, so the test reds if the
    rule stops working AND if the uniform grid stops showing the defect
    (which would mean the rasterizer changed and the offset needs
    re-measuring)."""
    f_uniform = _fin_resonance(None)
    prof = edge_aware_profile(0.0, 0.010, 1e-3, boundary_cell=1e-3,
                              sheet_edges=[SheetEdge(FIN_M, -1)])
    f_edge = _fin_resonance(prof.cells)
    err_uniform = abs(f_uniform / F_REF_HZ - 1.0)
    err_edge = abs(f_edge / F_REF_HZ - 1.0)
    assert err_uniform > 0.010, f"uniform grid error {100 * err_uniform:.3f} %"
    assert err_edge < 0.003, f"edge-aware error {100 * err_edge:.3f} %"
