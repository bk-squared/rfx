"""Preflight reads the SAME realized PEC edges the solver zeroes (#931 §1.7).

The lattice ownership contract has one function that turns conductor
geometry into PEC E edges (``rfx.boundaries.pec.realized_pec_edge_masks``),
and every consumer — the step functions and preflight alike — must read it
rather than re-derive metal from a cell mask or a bounding box (the #868
"40 mm guide reads 42 mm" and #767 "cavity check cannot see the far face"
classes were both a second derivation drifting from the first).

This file pins that identity from the OUTSIDE: for a one-cell PEC Box, a
declared sheet and a wire port starting on a volume ground, the planes and
liveness preflight REPORTS equal what one ``run()`` actually zeroes in the
final field state. No shared helper is used on the solver side — the field
is the witness.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import Box, Simulation

MM = 1e-3
DX = 1 * MM
N_STEPS = 60


def _sim():
    """10 mm PEC-walled cube, dx = 1 mm: node i at (i - pad)·dx with pad 0."""
    return Simulation(freq_max=10e9, domain=(10 * MM, 10 * MM, 10 * MM),
                     dx=DX, boundary="pec")


def _state(sim):
    return sim.run(n_steps=N_STEPS, skip_preflight=True).state


def _plane_is_dead(arr, k, axis, i_lo, i_hi, j_lo, j_hi):
    """Every entry of the in-plane window on plane ``k`` along ``axis`` is 0."""
    sub = np.take(np.asarray(arr), k, axis=axis)
    return float(np.abs(sub[i_lo:i_hi, j_lo:j_hi]).max()) == 0.0


def test_one_cell_volume_planes_reported_are_the_planes_zeroed():
    """A PEC Box one cell thick along z: preflight's ``pec_box_one_cell``
    names walls at z = 4 mm AND z = 5 mm, and the solved field has
    tangential E exactly zero on both planes over the footprint — and NOT
    zero one plane further out."""
    sim = _sim()
    sim.add(Box((2 * MM, 2 * MM, 4 * MM), (8 * MM, 8 * MM, 5 * MM)),
            material="pec")
    sim.add_source((5 * MM, 5 * MM, 7 * MM), "ex")
    rep = sim.preflight()
    hits = rep.by_code("pec_box_one_cell")
    assert len(hits) == 1
    assert "walls at 4mm and 5mm" in str(hits[0])
    # the context's own reading of the entry
    ctx = sim._campaign_ctx()
    (entry,) = [e for e in ctx.entry_realizations() if e.kind == "volume"]
    grid = ctx.grid
    planes = entry.wall_planes(2, ctx.periodic, tuple(grid.shape))
    k4 = grid.position_to_index((0.0, 0.0, 4 * MM))[2]
    assert planes == [k4, k4 + 1]

    st = _state(sim)
    i0, i1 = grid.position_to_index((2 * MM, 2 * MM, 0.0))[:2]
    j0 = grid.position_to_index((8 * MM, 8 * MM, 0.0))[0]
    # tangential E (ex over edges i0..j0-1, ey over i0..j0) on both planes
    for k in planes:
        assert _plane_is_dead(st.ex, k, 2, i0, j0, i1, j0 + 1), k
        assert _plane_is_dead(st.ey, k, 2, i0, j0 + 1, i1, j0), k
    # ... and the field is not trivially zero one plane above the slab
    assert not _plane_is_dead(st.ex, k4 + 2, 2, i0, j0, i1, j0 + 1)
    assert not _plane_is_dead(st.ex, k4 - 1, 2, i0, j0, i1, j0 + 1)


def test_sheet_plane_reported_is_the_plane_zeroed_and_its_normal_edge_lives():
    """A zero-thickness PEC Box at z = 4.3 mm is a sheet declaration:
    preflight's ``sheet_plane_realized`` reports node plane 4 mm (offset
    -0.3 cell), and the solved field has tangential E zero on that plane
    over the footprint, the normal Ez THROUGH the plane non-zero, and the
    planes on either side live."""
    sim = _sim()
    sim.add(Box((2 * MM, 2 * MM, 4.3 * MM), (8 * MM, 8 * MM, 4.3 * MM)),
            material="pec")
    sim.add_source((5 * MM, 5 * MM, 6 * MM), "ez")
    rep = sim.preflight()
    (hit,) = rep.by_code("sheet_plane_realized")
    assert "declared mid-plane 4.3mm, realized node plane" in str(hit)
    assert "(offset -0.300 cell = 300µm)" in str(hit)
    ctx = sim._campaign_ctx()
    (entry,) = [e for e in ctx.entry_realizations() if e.kind == "sheet"]
    grid = ctx.grid
    k = int(entry.sheet.plane)
    assert k == grid.position_to_index((0.0, 0.0, 4 * MM))[2]

    st = _state(sim)
    i0 = grid.position_to_index((2 * MM, 0.0, 0.0))[0]
    j0 = grid.position_to_index((8 * MM, 0.0, 0.0))[0]
    assert _plane_is_dead(st.ex, k, 2, i0, j0, i0, j0 + 1)
    assert _plane_is_dead(st.ey, k, 2, i0, j0 + 1, i0, j0)
    # normal E through the sheet (Ez[.., k] spans z_k .. z_{k+1}) is LIVE
    assert not _plane_is_dead(st.ez, k, 2, i0, j0 + 1, i0, j0 + 1)
    # the sheet is ONE plane: its neighbours carry tangential field
    assert not _plane_is_dead(st.ex, k + 1, 2, i0, j0, i0, j0 + 1)
    assert not _plane_is_dead(st.ex, k - 1, 2, i0, j0, i0, j0 + 1)


def test_wire_port_starting_on_a_volume_ground_face_is_galvanic():
    """A vertical Ez wire port whose start node lies ON the top face of a
    volume ground: every extent cell is live (the port's own Ez edges are
    above the face), no end-gap advisory (contact = the end node is on a
    realized wall plane), and the solved Ez is zero INSIDE the ground and
    non-zero on the port column above it. The remedy text of the end-gap
    advisory never tells a user to move such a feed off its ground."""
    sim = _sim()
    sim.add(Box((1 * MM, 1 * MM, 1 * MM), (9 * MM, 9 * MM, 3 * MM)),
            material="pec")                 # ground: top face at z = 3 mm
    sim.add_port(position=(5 * MM, 5 * MM, 3 * MM), component="ez",
                 impedance=50.0, extent=3 * MM)
    rep = sim.preflight()
    assert rep.by_code("wire_port_end_gap_to_conductor") == []
    assert rep.by_code("wire_port_dead_extent_cells") == []
    assert rep.by_code("wire_port_midpoint_in_pec") == []
    assert rep.by_code("port_in_pec") == []

    st = _state(sim)
    grid = sim._campaign_ctx().grid
    i, j, k3 = grid.position_to_index((5 * MM, 5 * MM, 3 * MM))
    ez = np.asarray(st.ez)
    assert float(abs(ez[i, j, k3 - 1])) == 0.0      # inside the ground
    assert float(abs(ez[i, j, k3 - 2])) == 0.0
    assert float(np.abs(ez[i, j, k3:k3 + 3]).max()) > 0.0   # the port's own edges


def test_wire_port_one_cell_short_of_a_volume_face_draws_the_gap_advisory():
    """The #556 signature under one definition of contact: the wire's end
    node carries no wall, the node one cell further IS the ground's face.
    The remedy names the wall plane, not a 'rasterized node'."""
    sim = _sim()
    sim.add(Box((1 * MM, 1 * MM, 1 * MM), (9 * MM, 9 * MM, 3 * MM)),
            material="pec")
    sim.add_port(position=(5 * MM, 5 * MM, 4 * MM), component="ez",
                 impedance=50.0, extent=3 * MM)
    rep = sim.preflight()
    (hit,) = rep.by_code("wire_port_end_gap_to_conductor")
    msg = str(hit)
    assert "-z-side end node" in msg
    assert "is a realized wall plane of ['pec']" in msg
    assert "its end node lands on that wall plane" in msg
    assert "rasterized node" not in msg


def test_wire_port_through_a_sheet_ground_keeps_its_normal_edges_live():
    """A sheet ground shorts only the edges IN its plane: a vertical Ez
    port passing through the sheet plane has every Ez edge live (no dead
    cell), and starting on the plane is galvanic (no gap advisory)."""
    sim = _sim()
    sim.add(Box((1 * MM, 1 * MM, 3 * MM), (9 * MM, 9 * MM, 3 * MM)),
            material="pec")                 # sheet at z = 3 mm
    sim.add_port(position=(5 * MM, 5 * MM, 2 * MM), component="ez",
                 impedance=50.0, extent=3 * MM)
    rep = sim.preflight()
    assert rep.by_code("wire_port_dead_extent_cells") == []
    assert rep.by_code("wire_port_end_gap_to_conductor") == []


def test_wire_port_inside_a_volume_ground_has_dead_cells_named_by_owner():
    """Cells whose Ez edge lies between a volume's faces are dead, and the
    advisory attributes them through the same realization (owner name)."""
    sim = _sim()
    sim.add(Box((1 * MM, 1 * MM, 1 * MM), (9 * MM, 9 * MM, 3 * MM)),
            material="pec")
    sim.add_port(position=(5 * MM, 5 * MM, 2 * MM), component="ez",
                 impedance=50.0, extent=3 * MM)
    rep = sim.preflight()
    (hit,) = rep.by_code("wire_port_dead_extent_cells")
    msg = str(hit)
    # extent 3 mm from z = 2 mm rasterizes to 3 Ez edges (production
    # _wire_port_cells, half-open in EDGES since #931 R8 — it gave 4 while
    # the extent was endpoint-inclusive, the fourth spanning one cell above
    # the declared end); only the edge lying between the ground's faces
    # (z = 2 -> 3 mm) is dead.
    assert "1 have their ez edge inside realized PEC ['pec']" in msg
    assert "n_live/n = 2/3" in msg
