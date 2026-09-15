"""ONE realized-geometry reader for the two WR-90 inductive-iris cases.

Under the #931 lattice ownership contract a PEC volume is a set of primal
cells and its realized E edges come from exactly one function,
``rfx.boundaries.pec.realized_pec_edge_masks``.  Everything the two iris
cases need to know about what they built — how thick each iris came out,
where each aperture's bounding walls are — is a read of that edge set.

Before #931 both cases derived their geometry from the ``rasterize()`` sigma
MASK and reasoned about "masked node planes", each with its OWN convention:
case 18 fed its oracle ``t_c*dx`` while case 19 drew ``round(t/dx) + 1`` and
fed ``(t_c - 1)*dx``.  The suite stayed green with the two disagreeing, which
is the reason this module exists as ONE reader they both call: a future
divergence has to change a shared function, and
``tests/crossval/test_wr90_iris_realized_is_shared.py`` asserts the two cases
report the same realized thickness for the same drawn box.

Nothing here re-implements the rule.  It assembles the simulation's PEC
occupancy, hands it to ``realized_pec_edge_masks`` under the run's own
periodic flags, and reports wall planes through ``realized_wall_planes``.
"""
from __future__ import annotations

import numpy as np


def realized_edge_masks(sim):
    """The (Mx, My, Mz) this simulation will actually step (no solve)."""
    from rfx.boundaries.pec import realized_pec_edge_masks

    grid = sim._build_grid()
    sheets: list = []
    wires: list = []
    assembled = sim._assemble_materials(grid, pec_sheets=sheets,
                                        pec_wires=wires)
    pec_mask = assembled[3]
    if pec_mask is None and not sheets and not wires:
        raise AssertionError("no conductor in this build")
    edges = realized_pec_edge_masks(pec_mask, sheets=tuple(sheets),
                                    wires=tuple(wires),
                                    periodic=sim._periodic_flags())
    return grid, edges


def grid_plane(grid, axis, node_index):
    """Grid-array index of the node ``node_index`` cells from the domain origin.

    Wall planes come back from ``realized_wall_planes`` as indices into the
    padded field arrays, so a DECLARED plane (counted from the domain origin,
    which is how both cases draw) has to be shifted by the lo-side pad before
    the two can be compared.  Doing that arithmetic here, once, is the point:
    a declared-vs-realized assert that quietly compares two different index
    spaces passes for the wrong reason.
    """
    pad = (grid.pad_x_lo, grid.pad_y_lo, grid.pad_z_lo)[int(axis)]
    return int(node_index) + int(pad)


def wall_plane_runs(edges, axis, region=None):
    """Contiguous runs of realized wall planes along ``axis``.

    Returns a list of ``(first, last)`` node-plane index pairs.  A run's
    length in CELLS is ``last - first`` — the distance between the two
    bounding walls, which under the contract equals the drawn extent.
    """
    from rfx.boundaries.pec import realized_wall_planes

    planes = np.asarray(realized_wall_planes(edges, axis, region=region),
                        dtype=int)
    if planes.size == 0:
        return []
    cuts = np.where(np.diff(planes) != 1)[0] + 1
    return [(int(r[0]), int(r[-1])) for r in np.split(planes, cuts)]


def aperture_walls(edges, axis, region):
    """The two innermost wall planes bounding the gap along ``axis``.

    ``region`` selects one iris (a single-plane slice on the propagation
    axis).  The aperture in CELLS is the difference of the returned pair.
    Raises when the gap is not a single contiguous opening — the parasitic
    wall-slot class that case 18's setup defect (1) was.
    """
    runs = wall_plane_runs(edges, axis, region=region)
    if len(runs) != 2:
        raise AssertionError(
            f"expected exactly two fin runs along {'xyz'[axis]}, got "
            f"{len(runs)}: {runs} — an aperture that is not one contiguous "
            "opening is the parasitic wall-slot defect")
    return runs[0][1], runs[1][0]
