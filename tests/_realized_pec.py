"""Crossval-level assertions on realized PEC geometry — no solve (#931 §1.7).

``tests/_realized_geometry.py`` is the single shared spelling of "assemble a
``Simulation`` and realize its conductors": ``realized(sim)``, ``node_index``,
``assert_wall_planes``, ``assert_sheet_planes``. Everything here goes through
it and adds nothing to it — a second ``argwhere`` over some mask is exactly the
drift the single-owner rule exists to stop.

What this module adds is the four things a CROSSVAL case needs and a plane list
does not express:

``assert_walls_at``      a wall exists ACROSS THE CONDUCTOR'S FOOTPRINT, not
                         merely somewhere on the plane. cv15's #740 lesson is
                         that a wall check evaluated at the feed passes while
                         an in-plane rasterization edge case leaves a slit
                         somewhere else.
``assert_no_wall_at``    the falsifier arm. A check that only looks for the
                         walls it wants cannot see an extra one, and an extra
                         wall a cell away is exactly what a foil mis-declared
                         as a volume produces.
``assert_sheet_owns_no_cell`` / ``assert_normal_edge_live``
                         the (S) half of the contract, and the direct
                         replacement for the deleted #702 resample.
``wall_positions`` / ``realized_extent``
                         the answer in the case's own metres. Cases report
                         board widths and cavity thicknesses, not node indices.

Scope: PEC BODIES (volumes, sheets, wires). Domain-boundary PEC/PMC faces are a
separate convention and stay out (design note §1.8): cv09, cv10, cv14 and cv24
gate on ``BoundarySpec`` faces and nothing here reaches them.

Owner: the tests-crossval migration group (#931 phase 2, section T).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tests._realized_geometry import (assert_sheet_planes as _assert_sheet_planes,
                                      node_index, realized as _realized)

__all__ = [
    "RealizedPec",
    "realize",
    "nearest_plane",
    "wall_planes",
    "wall_positions",
    "assert_walls_at",
    "assert_no_wall_at",
    "assert_sheet_planes",
    "assert_sheet_owns_no_cell",
    "assert_normal_edge_live",
    "realized_extent",
]

_AXIS_NAME = ("x", "y", "z")


@dataclass(frozen=True)
class RealizedPec:  # noqa: D101 - documented below
    """A crossval-facing view of ``tests._realized_geometry.Realization``.

    ``edges`` is the ``(Mx, My, Mz)`` triple: ``True`` where that E component
    is zeroed every step. ``cells`` is the VOLUME occupancy (``pec_mask``);
    sheets and wires own no cell, so they are absent from it by construction
    and present only in ``edges`` / ``sheets`` / ``wires``.
    """

    inner: object
    sim: object

    @property
    def edges(self):
        return self.inner.edge_masks

    @property
    def grid(self):
        return self.inner.grid

    @property
    def cells(self):
        return self.inner.pec_mask

    @property
    def sheets(self):
        return tuple(self.inner.sheets)

    @property
    def wires(self):
        return tuple(self.inner.wires)

    @property
    def materials(self):
        """The assembled ``MaterialArrays`` on the SAME grid.

        The shared ``Realization`` does not keep them (it is about the edge
        set), but a board case has to be able to say "and the cells between
        those two walls are laminate, not vacuum" — the #693 defect was
        exactly a material reading, not an edge one. The lane is not
        re-decided here: it is read off the grid the shared reader already
        built.
        """
        import warnings as _warnings

        from rfx.nonuniform import NonUniformGrid
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            # Materials only — the realized conductors are already on
            # ``self.inner``. The #931 collectors are passed and dropped so
            # this read states "cells only" instead of tripping the
            # assembler's refusal on a sheet model.
            if isinstance(self.grid, NonUniformGrid):
                return self.sim._assemble_materials_nu(
                    self.grid, pec_sheets=[], pec_wires=[])[0]
            return self.sim._assemble_materials(
                self.grid, pec_sheets=[], pec_wires=[])[0]

    def nodes(self, axis: int) -> np.ndarray:
        """Node positions along ``axis`` in metres (host float64)."""
        from tests._realized_geometry import _node_line
        return np.asarray(_node_line(self.inner.grid, axis), dtype=np.float64)


def realize(sim, grid=None) -> RealizedPec:
    """Realize ``sim``'s conductors without stepping a single field.

    Thin wrapper on ``tests._realized_geometry.realized`` — it picks the lane
    the run itself would take and reads the run's own periodic flags, so the
    answer is the one the step function will apply.
    """
    if grid is not None:                       # pragma: no cover - callers pass none
        raise TypeError(
            "realize() takes the simulation only; the shared reader builds the "
            "grid the run would build (tests/_realized_geometry.realized)")
    rz = _realized(sim)
    if rz.pec_mask is None and not rz.sheets and not rz.wires:
        raise AssertionError(
            "realize(): this simulation declares no PEC body at all — no "
            "volume cells, no sheets, no wires. A case whose physics needs "
            "metal must not quote a result from a model that has none.")
    return RealizedPec(rz, sim)


def nearest_plane(realized: RealizedPec, axis: int, position: float,
                  *, tol_frac: float = 0.25) -> int:
    """Node index nearest ``position`` on ``axis``, refusing an off-lattice ask.

    A declared plane further than ``tol_frac`` of the local cell from any node
    is not a plane this lattice can carry: the case must be redrawn on-lattice
    (design note §1.3, "off-lattice interfaces"), not silently snapped. The
    index itself comes from the shared ``node_index``; the refusal is the part
    a crossval case needs and a generic reader should not impose.
    """
    nodes = realized.nodes(axis)
    k = node_index(realized.grid, axis, position)
    gaps = np.diff(nodes)
    d_local = float(gaps[min(k, gaps.size - 1)]) if gaps.size else 1.0
    offset = abs(float(nodes[k]) - float(position))
    if offset > tol_frac * abs(d_local):
        raise AssertionError(
            f"declared plane {_AXIS_NAME[axis]} = {position:.9g} m is "
            f"{offset / abs(d_local):.3f} of a local cell ({d_local:.6g} m) "
            f"from the nearest node ({nodes[k]:.9g} m, index {k}). Redraw the "
            "case on-lattice; do not let it snap.")
    return k


def wall_planes(realized: RealizedPec, axis: int, *, region=None, ij=None):
    """Sorted node-plane indices along ``axis`` carrying a tangential wall."""
    kw = {}
    if region is not None:
        kw["region"] = region
    if ij is not None:
        kw["ij"] = ij
    return realized.inner.wall_planes(axis, **kw)


def wall_positions(realized: RealizedPec, axis: int, *, region=None, ij=None):
    """:func:`wall_planes` in metres."""
    nodes = realized.nodes(axis)
    return [float(nodes[k]) for k in wall_planes(realized, axis, region=region,
                                                 ij=ij)]


def realized_extent(realized: RealizedPec, axis: int, *, region=None,
                    ij=None) -> float:
    """Distance in metres between the outermost realized wall planes.

    The contract's answer to "how wide is the realized guide / how wide is the
    realized trace" (§1.9): the span between the bounding walls, not a node
    count and not ``round(W / dx) * dx``. Under the pre-#931 rule a body's far
    face was never a wall, so every consumer that measured a conductor this way
    read one cell too many on one side and one too few on the other — the #868
    "40 mm guide reads 42 mm" case, and cv06b's 635.0 um for a 600 um trace.
    """
    pos = wall_positions(realized, axis, region=region, ij=ij)
    if len(pos) < 2:
        raise AssertionError(
            f"realized_extent: axis {_AXIS_NAME[axis]} carries "
            f"{len(pos)} wall plane(s) {pos} — an extent needs two.")
    return float(pos[-1] - pos[0])


def _footprint_mask(realized: RealizedPec, axis: int, footprint):
    """Normalize a footprint to a boolean mask over the two in-plane axes."""
    shape = tuple(realized.edges[0].shape)
    in_plane = tuple(a for a in range(3) if a != axis)
    want = (shape[in_plane[0]], shape[in_plane[1]])
    if footprint is None:
        return np.ones(want, dtype=bool)
    fp = np.asarray(footprint, dtype=bool)
    if fp.ndim == 3:
        fp = fp.any(axis=axis)
    if fp.shape != want:
        raise AssertionError(
            f"footprint shape {fp.shape} does not match the in-plane grid "
            f"{want} for a wall normal to {_AXIS_NAME[axis]}")
    return fp


def _full_wall(realized: RealizedPec, axis: int, k: int, footprint) -> bool:
    """True iff every tangential edge INSIDE the footprint is PEC on plane ``k``.

    "Inside" is the contract's own sentence (§1.3): the edge from node ``n`` to
    node ``n + 1`` belongs to the conductor iff BOTH of its end nodes are in
    the footprint. Asserting instead that every stored index under the
    footprint is PEC would demand an edge that leaves the footprint at its hi
    rim and would fail on a correctly realized sheet.
    """
    fp = _footprint_mask(realized, axis, footprint)
    in_plane = tuple(a for a in range(3) if a != axis)
    for t in in_plane:
        m = np.asarray(realized.edges[t], dtype=bool)
        idx = [slice(None)] * 3
        idx[axis] = k
        plane = m[tuple(idx)]                     # (n_p, n_q) in in-plane order
        along = 0 if t == in_plane[0] else 1
        if along == 0:
            need = fp[:-1, :] & fp[1:, :]
            have = plane[:-1, :]
        else:
            need = fp[:, :-1] & fp[:, 1:]
            have = plane[:, :-1]
        if need.any() and not bool(np.all(have[need])):
            return False
    return True


def assert_walls_at(realized: RealizedPec, axis: int, positions,
                    *, footprint=None, what: str = "conductor") -> list:
    """Assert a tangential wall is realized at every declared ``position``.

    The footprint form of ``tests._realized_geometry.assert_wall_planes``:
    that one compares the whole plane LIST, this one asks whether a named
    plane carries a wall across the conductor's own footprint. Both are
    useful; a case that has a footprint should use this, because the #740
    failure was a wall that existed at the feed and not elsewhere.

    Returns the realized plane indices, so a caller can record what it
    measured rather than what it asked for.
    """
    out = []
    for position in positions:
        k = nearest_plane(realized, axis, position)
        if not _full_wall(realized, axis, k, footprint):
            planes = wall_positions(realized, axis)
            raise AssertionError(
                f"{what}: declared wall at {_AXIS_NAME[axis]} = "
                f"{position:.9g} m (node {k}) is NOT realized across its "
                f"footprint. Realized wall planes on this axis: {planes}.")
        out.append(k)
    return out


def assert_no_wall_at(realized: RealizedPec, axis: int, positions,
                      *, footprint=None, what: str = "conductor") -> None:
    """Assert NO tangential wall exists at each of ``positions``.

    The falsifier half of :func:`assert_walls_at`.
    """
    for position in positions:
        k = nearest_plane(realized, axis, position)
        fp = _footprint_mask(realized, axis, footprint)
        for t in (a for a in range(3) if a != axis):
            m = np.asarray(realized.edges[t], dtype=bool)
            idx = [slice(None)] * 3
            idx[axis] = k
            if bool(np.any(m[tuple(idx)][fp])):
                raise AssertionError(
                    f"{what}: an UNDECLARED wall is realized at "
                    f"{_AXIS_NAME[axis]} = {position:.9g} m (node {k}), "
                    f"component E{_AXIS_NAME[t]}. Realized wall planes: "
                    f"{wall_positions(realized, axis)}.")


def assert_sheet_planes(realized: RealizedPec, axis: int, positions,
                        *, what: str = "sheet") -> list:
    """The declared sheets land on exactly the declared node planes.

    Delegates to the shared ``assert_sheet_planes`` so there is one
    implementation; this signature takes an already-realized object because a
    crossval case builds its simulation once and asserts several things.
    """
    return _assert_sheet_planes(realized.sim, axis, positions, what=what)


def assert_sheet_owns_no_cell(realized: RealizedPec, *,
                              what: str = "sheet") -> None:
    """The (S) half of the contract: a sheet adds no cell and no material.

    §1.3: "A sheet owns NO cell: it adds nothing to ``C``, writes no
    ``eps_r``/``sigma``, and is realized at exactly one plane." The direct
    replacement for the deleted #702 resample, which gave a one-node body its
    own cell's material at its live edge.
    """
    if not realized.sheets:
        raise AssertionError(f"{what}: this build declares no sheet at all.")
    cells = realized.cells
    if cells is None:
        return
    occ = np.asarray(cells, dtype=bool)
    for sp in realized.sheets:
        a = int(sp.normal_axis)
        k = int(sp.plane)
        fp = np.asarray(sp.footprint, dtype=bool)
        idx = [slice(None)] * 3
        idx[a] = k
        overlap = occ[tuple(idx)] & fp[tuple(idx)]
        if bool(overlap.any()):
            raise AssertionError(
                f"{what} {sp.name!r}: its footprint on plane "
                f"{_AXIS_NAME[a]}={k} overlaps {int(overlap.sum())} VOLUME "
                "cell(s). A sheet owns no cell; a body that owns cells is a "
                "volume and must be declared with add().")


def assert_normal_edge_live(realized: RealizedPec, *,
                            what: str = "sheet") -> None:
    """Normal E through a declared sheet stays live (§1.3, #690 semantics).

    The property that separates a sheet from a one-cell slab: a slab shorts
    the normal edge between its two faces, a sheet does not. Skipped for a
    sheet whose normal axis has length 1 (the 2-D lane), where the contract
    realizes the footprint as a 2-D volume and the "normal" component has no
    through-sheet edge.
    """
    if not realized.sheets:
        raise AssertionError(f"{what}: this build declares no sheet at all.")
    for sp in realized.sheets:
        a = int(sp.normal_axis)
        k = int(sp.plane)
        if realized.edges[a].shape[a] == 1:
            continue
        fp = np.asarray(sp.footprint, dtype=bool)
        m = np.asarray(realized.edges[a], dtype=bool)
        idx = [slice(None)] * 3
        idx[a] = k
        under = fp[tuple(idx)]
        if not under.any():                       # pragma: no cover
            continue
        if bool(np.all(m[tuple(idx)][under])):
            raise AssertionError(
                f"{what} {sp.name!r}: the normal component E{_AXIS_NAME[a]} is "
                f"zeroed everywhere under the footprint on plane "
                f"{_AXIS_NAME[a]}={k}. A sheet is zero-thickness — normal E "
                "through it stays live; a shorted normal edge means this was "
                "realized as a volume.")
