"""Realized wall planes, in metres, for the fixtures in these four directories.

``tests/_realized_geometry`` is the single owner of the build-time
realization check under the lattice ownership contract (#931): it assembles
a ``Simulation`` without stepping it and reads the realized PEC edges from
``rfx.boundaries.pec.realized_pec_edge_masks``. Every fixture in
``tests/unit/{nonuniform,runners,grid,subgrid}`` reads the RULE from there.

It answers in NODE INDICES (``Realization.wall_planes``) and converts a
declared position into an index (``node_index``). Several fixtures here
want the other direction — "which physical z did the lattice put the wall
at" — because they compare against a drawn coordinate and print millimetres
when they fail. That conversion is all this module is. The node line comes
from the rasterizer's own coordinate builders, the same ones the shared
helper uses, so there is one node line and not two.

History, so the next reader does not re-derive it: this file started as a
lane shim. ``realized()`` used to decide uniform-vs-non-uniform from
``sim.dz_profile`` / ``sim._nu_axes``, attributes ``Simulation`` does not
have (the fields are ``_dx_profile`` / ``_dy_profile`` / ``_dz_profile``),
so a graded fixture was assembled on the UNIFORM lane and passed for the
wrong reason; and ``node_index`` fell back to ``(arange - pad) * grid.dx``,
where on a graded axis ``dx`` is the BOUNDARY cell size. Measured on
``test_inplane_grading_guards.py``'s fixture (a PEC Box drawn 3.0 -> 4.5 mm
on ``[12x250um, 8x500um, 8x125um, 12x250um]``): the wrong lane answered
``[18..24]`` where the real lane answers ``[18, 19, 20, 21]``. Both were
fixed in the shared helper on the branch (``115756db``), so the lane
override below is a convenience, not a workaround — pass nothing and the
helper picks the lane the run takes.
"""
from __future__ import annotations

import numpy as np

__all__ = ["wall_planes_m"]


def wall_planes_m(sim, axis, *, nonuniform=None, **kw):
    """``(Realization, [wall plane positions in metres])`` along ``axis``.

    ``nonuniform=None`` lets the shared helper pick the lane the run takes.
    """
    from rfx.geometry.rasterize_grid import (
        coords_from_nonuniform_grid, coords_from_uniform_grid)
    from rfx.nonuniform import NonUniformGrid
    from tests._realized_geometry import realized

    rz = realized(sim) if nonuniform is None else realized(
        sim, nonuniform=bool(nonuniform))
    coords = (coords_from_nonuniform_grid(rz.grid)
              if isinstance(rz.grid, NonUniformGrid)
              else coords_from_uniform_grid(rz.grid))
    nodes = np.asarray((coords.x, coords.y, coords.z)[axis], dtype=float)
    return rz, [float(nodes[k]) for k in rz.wall_planes(axis, **kw)]
