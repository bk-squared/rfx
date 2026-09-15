"""A PEC body on a PERIODIC axis, end to end (#931 §1.2/§1.3, #689).

The wrap in ``rfx.boundaries.pec._shift`` exists for one geometry: a
conductor that straddles the seam of a periodic axis (the RIS/FSS unit
cell the ``pec.py`` docstring names as the reason). Until this file, the
wrap was exercised only through the raw mask API — no test put a conductor
body on a periodic lane and ran it, so removing the wrap would have left
the suite green and the unit cell wrong.

Two bodies, one drawn away from the seam and one straddling it, on an
x-periodic domain. The claims:

* a volume that straddles the seam realizes the SAME edge set as the same
  body translated into the interior, up to the roll — periodicity is a
  property of the lattice, not of where in the array the metal sits;
* a SHEET whose footprint straddles the seam realizes seamlessly: the two
  in-plane edges that cross the seam are PEC, so the footprint is one
  conductor and not two strips with a slit;
* the solver lane zeroes exactly those edges after a real run.

Fixture: 20 x 10 x 10 mm at dx = 1 mm, x periodic, y/z PEC, so node ``i``
reads as ``i`` mm on every axis and there is no CPML pad.
"""
from __future__ import annotations

import numpy as np

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.pec import (
    SheetSpec, realized_pec_edge_masks, realized_wall_planes,
)
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.rasterize_grid import (
    cell_centres_from_nodes, classify_pec_entry, coords_from_uniform_grid,
)

DX = 1e-3
DOM = (0.020, 0.010, 0.010)
PERIODIC = (True, False, False)


def _sim(shapes=()):
    sim = Simulation(
        freq_max=15e9, domain=DOM, dx=DX,
        boundary=BoundarySpec(x=Boundary(lo="periodic", hi="periodic"),
                              y="pec", z="pec"))
    for sh in shapes:
        sim.add(sh, material="pec")
    return sim


def _classify(sim, shape):
    grid = sim._build_grid()
    co = coords_from_uniform_grid(grid)
    return grid, classify_pec_entry(shape, co, cell_centres_from_nodes(co),
                                    name="pec")


def test_a_seam_straddling_volume_realizes_the_translated_body_rolled():
    """Two cells, one on each side of the seam, against the same two cells
    in the interior. Under the wrap the edge sets are one roll apart; under
    a zero pad the seam copy would lose the edges that join its halves."""
    grid = _sim()._build_grid()
    nx = grid.shape[0]
    seam = np.zeros(grid.shape, bool)
    seam[[nx - 1, 0], 4:7, 4:7] = True
    inner = np.roll(seam, 5, axis=0)

    e_seam = realized_pec_edge_masks(seam, periodic=PERIODIC)
    e_inner = realized_pec_edge_masks(inner, periodic=PERIODIC)
    for c in range(3):
        np.testing.assert_array_equal(
            np.roll(np.asarray(e_seam[c]), 5, axis=0),
            np.asarray(e_inner[c]), err_msg="xyz"[c])

    # A straddling body does NOT discriminate wrap from pad: each of its two
    # cells already puts a wall on its own near plane. Measured, so the
    # equality above is not credited to the wrap.
    assert all(bool(np.array_equal(np.asarray(a), np.asarray(b)))
               for a, b in zip(e_seam, realized_pec_edge_masks(seam)))
    # The volume case where the wrap IS load-bearing is a body on the LAST
    # cell alone: its far face is plane nx, which has no array entry under a
    # pad and is plane 0 under the wrap.
    last = np.zeros(grid.shape, bool)
    last[nx - 1, 4:7, 4:7] = True
    e_wrap = realized_pec_edge_masks(last, periodic=PERIODIC)
    e_pad = realized_pec_edge_masks(last)
    assert int(np.asarray(e_wrap[1]).sum()) > int(np.asarray(e_pad[1]).sum())
    assert realized_wall_planes(e_wrap, 0, periodic=PERIODIC) == [0, nx - 1]
    assert realized_wall_planes(e_pad, 0) == [nx - 1]


def test_a_seam_straddling_sheet_realizes_without_a_slit():
    """§1.3: the in-plane rule is ``F & shift(F)``, so the edge from the hi
    rim node to node 0 is PEC exactly when the footprint contains both —
    the unit cell's conductor is continuous across the seam."""
    grid = _sim()._build_grid()
    nx = grid.shape[0]
    fp = np.zeros(grid.shape, bool)
    fp[:, 3:8, 5] = True                  # full-width strip on node plane 5
    sheet = SheetSpec(normal_axis=2, plane=5, footprint=fp)

    per = realized_pec_edge_masks(None, sheets=[sheet], periodic=PERIODIC)
    pad = realized_pec_edge_masks(None, sheets=[sheet])
    ex_per = np.asarray(per[0], bool)
    ex_pad = np.asarray(pad[0], bool)
    # the seam edge Ex[nx-1, j, 5] joins node nx-1 to node 0
    assert ex_per[nx - 1, 3:8, 5].all(), "seam edge left live — a slit"
    assert not ex_pad[nx - 1, 3:8, 5].any(), (
        "fixture guard: the non-periodic reading must differ here, or the "
        "wrap is not what this test measures")
    assert int(ex_per.sum()) == int(ex_pad.sum()) + 5
    # the normal edge stays live either way (a sheet is not a slab)
    assert not np.asarray(per[2]).any()


def test_the_periodic_lane_run_zeroes_the_realized_edges():
    """End to end: a body drawn across the seam, one real run, and the
    field is zero on exactly the realized set."""
    body = Box((0.019, 0.004, 0.004), (0.021, 0.007, 0.007))   # crosses x = 20 mm
    sim = _sim()
    sim.add(body, material="pec")
    sim.add_source((0.010, 0.005, 0.005), "ez",
                   waveform=GaussianPulse(f0=8e9), amplitude_kind="field")
    sim.add_probe((0.012, 0.005, 0.005), "ez")
    res = sim.run(n_steps=40, skip_preflight=True)

    grid, (cells, sheet, wire) = _classify(sim, body)
    assert sheet is None and wire is None
    edges = realized_pec_edge_masks(cells, periodic=PERIODIC)
    for c, m in zip(("ex", "ey", "ez"), edges):
        f = np.asarray(getattr(res.state, c))
        m = np.asarray(m, bool)
        assert not np.any(f[m]), f"{c}: a realized PEC edge is not zero"
    assert float(np.abs(np.asarray(res.time_series)).max()) > 0.0
