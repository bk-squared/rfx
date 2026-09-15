"""tests/_realized_geometry.py

Single shared spelling of the BUILD-TIME realization check every migrated
conductor fixture owes under the lattice ownership contract (#931).

The contract's whole point is that a declaration and its realization are
the same statement: a Box drawn ``z_a -> z_b`` on node planes realizes
tangential walls at BOTH ``z_a`` and ``z_b``; a sheet realizes exactly one
plane and leaves its normal edge live.  A fixture that says "the ground is
at z = 0.254 mm" and never checks where the lattice put it is how #702,
#706, #729 and #802 each got their own local repair — the compensations
this branch deletes.

So every fixture this branch migrates asserts, with NO solve, that the
planes the lattice realizes are the planes the fixture declared.  This
module is the one implementation of that check; a second hand-rolled
``argwhere`` over some mask is exactly the drift the single-owner rule
(design note §1.7) exists to stop.

Two entry points:

``realized(sim)``
    Assemble a ``Simulation`` (no solve) and hand back everything the
    realization is made of — the grid, the volume cell mask, the sheet and
    wire specs, and the ``(Mx, My, Mz)`` edge triple from
    ``rfx.boundaries.pec.realized_pec_edge_masks``.

``assert_wall_planes(sim, axis, expected, ...)``
    The check itself: ``realized_wall_planes`` along ``axis`` equals the
    node planes the fixture declares.  ``node_index`` converts a physical
    coordinate to the node index on the same grid, so a fixture states its
    expectation in metres and the arithmetic happens once, here.

Both are deliberately thin.  They read the shared owner and add nothing;
if a check here disagrees with the solver, the solver is right and this
module is the bug.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "realized",
    "assert_wall_planes",
    "assert_sheet_planes",
    "node_index",
    "domain_wall_positions",
    "Realization",
]


class Realization:
    """What one ``Simulation`` realizes as PEC, before any time step.

    Attributes
    ----------
    grid : Grid or NonUniformGrid
    pec_mask : array or None
        Primal-cell occupancy of the PEC VOLUMES (``None`` when the run
        declares no volume conductor).
    sheets, wires : list
        ``SheetSpec`` / ``WireSpec`` declarations; both own no cell.
    edge_masks : (Mx, My, Mz)
        The realized PEC E edges, from the single owner.
    """

    __slots__ = ("grid", "pec_mask", "sheets", "wires", "edge_masks")

    def __init__(self, grid, pec_mask, sheets, wires, edge_masks):
        self.grid = grid
        self.pec_mask = pec_mask
        self.sheets = sheets
        self.wires = wires
        self.edge_masks = edge_masks

    def wall_planes(self, axis, **kw):
        from rfx.boundaries.pec import realized_wall_planes
        return realized_wall_planes(self.edge_masks, axis, **kw)

    @property
    def sheet_planes(self):
        """``{normal_axis: sorted plane indices}`` of the declared sheets."""
        out: dict[int, list[int]] = {}
        for sp in self.sheets:
            out.setdefault(int(sp.normal_axis), []).append(int(sp.plane))
        return {a: sorted(v) for a, v in out.items()}


def _periodic_of(sim, grid):
    """The run's per-axis periodic flags (#689) — read from the one spelling.

    ``Simulation._periodic_flags`` is where the lanes get theirs; realizing
    edges under different flags than the step function is the #689 bug in a
    test's clothing.
    """
    flags = getattr(sim, "_periodic_flags", None)
    if callable(flags):
        return tuple(bool(v) for v in flags())
    mode = getattr(grid, "mode", "3d")
    if isinstance(mode, str) and mode.startswith("2d"):
        return (False, False, True)
    return (False, False, False)


def realized(sim, *, nonuniform: bool | None = None) -> Realization:
    """Assemble ``sim`` and realize its conductors — no time step.

    ``nonuniform=None`` picks the lane the simulation itself would take.
    """
    from rfx.boundaries.pec import realized_pec_edge_masks

    if nonuniform is None:
        # The lane the run takes, spelled the way rfx/api/_compile.py spells
        # it (the fields are _dx_profile / _dy_profile / _dz_profile; a
        # Simulation has no ``dz_profile`` attribute — found by group T3,
        # which measured the helper reading the UNIFORM grid for a graded
        # fixture and passing for the wrong reason).
        nonuniform = any(getattr(sim, f"_{a}_profile", None) is not None
                         for a in ("dx", "dy", "dz"))
    sheets: list = []
    wires: list = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if nonuniform:
            grid = sim._build_nonuniform_grid()
            _m, _d, _l, pec = sim._assemble_materials_nu(
                grid, pec_sheets=sheets, pec_wires=wires)
        else:
            grid = sim._build_grid()
            _m, _d, _l, pec, *_rest = sim._assemble_materials(
                grid, pec_sheets=sheets, pec_wires=wires)
        edges = realized_pec_edge_masks(
            pec, sheets=tuple(sheets), wires=tuple(wires),
            periodic=_periodic_of(sim, grid))
    return Realization(grid, pec, sheets, wires, edges)


def domain_wall_positions(grid, axis: int) -> tuple[float, float]:
    """The domain's REALIZED lo / hi face planes along ``axis`` (metres).

    ``Grid`` realizes a declared extent by ``ceil(extent / dx)`` cells, so a
    22.86 x 10.16 mm WR-90 declared on a 1 mm cell is a 23 x 11 mm guide and
    the domain-face PEC (design note §1.8, BC-owned) stands on the LAST
    INTERIOR node, not at the declared number. A conductor that is meant to
    reach a guide wall — a shorting plug, a full-width iris — must be drawn
    to THIS plane: a volume's face rounds to the nearest node (§1.1), so a
    plug drawn to the declared 10.16 mm realizes its top at 10.000 mm and
    leaves a one-cell vacuum slot under the wall at 11.000 mm. Measured
    2026-09-07 on cv11 (``scripts/diagnostics/pec_short_lane_ab.py``): that
    slot is the whole 0.0146 -> 0.0560 pec-short |S11| step.

    Read off the grid the run builds (``interior`` slices + the node line),
    never recomputed from ``dx`` by the caller.
    """
    line = _node_line(grid, axis)
    interior = getattr(grid, "interior", None)
    if interior is None:
        raise TypeError(
            "domain_wall_positions: this grid has no `interior` slices "
            "(non-uniform lane); read its node line directly")
    sl = interior[axis]
    return float(line[sl.start]), float(line[sl.stop - 1])


def node_index(grid, axis: int, position: float) -> int:
    """Node index nearest ``position`` (metres, user coordinates).

    Uniform grid: ``i = position / d + pad``.  Non-uniform: the nearest
    entry of that axis's node line.  Both are the same statement — the
    node line is where a wall can land.
    """
    line = _node_line(grid, axis)
    return int(np.argmin(np.abs(np.asarray(line, dtype=float) - float(position))))


def _node_line(grid, axis: int):
    """The axis's E-node line, from the library's OWN spelling.

    ``coords_from_nonuniform_grid`` for a NonUniformGrid (its ``dx`` is the
    boundary cell size, not a spacing — an ``(arange - pad) * dx`` fallback
    returns a node line that exists on no mesh; measured by group T3 on a
    250/500/125 um graded profile), ``coords_from_uniform_grid`` otherwise.
    """
    from rfx.geometry.rasterize_grid import (
        coords_from_nonuniform_grid, coords_from_uniform_grid)
    from rfx.nonuniform import NonUniformGrid

    coords = (coords_from_nonuniform_grid(grid) if isinstance(grid, NonUniformGrid)
              else coords_from_uniform_grid(grid))
    return np.asarray((coords.x, coords.y, coords.z)[axis], dtype=float)


def assert_wall_planes(sim, axis, expected_m=None, *, expected_planes=None,
                       ij=None, region=None, what="", tol_planes=0):
    """Realized tangential walls along ``axis`` == the declared planes.

    ``expected_m`` is a sequence of physical coordinates (metres) on the
    declared conductor's faces; ``expected_planes`` states node indices
    directly.  Exactly one of the two.  Returns the realized plane list so
    a caller can print or re-use it.
    """
    if (expected_m is None) == (expected_planes is None):
        raise TypeError("pass exactly one of expected_m= or expected_planes=")
    rz = realized(sim)
    got = rz.wall_planes(axis, **({"ij": ij} if ij is not None else
                                  {"region": region} if region is not None else {}))
    if expected_planes is None:
        expected_planes = [node_index(rz.grid, axis, p) for p in expected_m]
    want = sorted(int(p) for p in expected_planes)
    label = f"{what}: " if what else ""
    if tol_planes:
        assert len(got) == len(want), (
            f"{label}realized {len(got)} wall planes along {'xyz'[axis]} "
            f"({got}), declared {len(want)} ({want})")
        for g, w in zip(got, want):
            assert abs(g - w) <= tol_planes, (
                f"{label}realized wall plane {g} is more than {tol_planes} "
                f"node(s) from the declared {w} along {'xyz'[axis]}")
    else:
        assert got == want, (
            f"{label}realized wall planes along {'xyz'[axis]} are {got}, "
            f"the declaration says {want}. Under the lattice ownership "
            "contract (#931 §1.2/§1.3) drawn == realized; a mismatch is a "
            "geometry bug, not a tolerance.")
    return got


def assert_sheet_planes(sim, axis, expected_m=None, *, expected_planes=None,
                        what=""):
    """The DECLARED sheets on ``axis`` sit on the expected node planes.

    Complements :func:`assert_wall_planes`: that one reads the realized
    edge set (which a volume also feeds), this one reads the sheet
    declarations, so a fixture can pin "these planes are sheets, and no
    volume put them there".
    """
    if (expected_m is None) == (expected_planes is None):
        raise TypeError("pass exactly one of expected_m= or expected_planes=")
    rz = realized(sim)
    got = rz.sheet_planes.get(int(axis), [])
    if expected_planes is None:
        expected_planes = [node_index(rz.grid, axis, p) for p in expected_m]
    want = sorted(int(p) for p in expected_planes)
    label = f"{what}: " if what else ""
    assert got == want, (
        f"{label}declared sheet planes along {'xyz'[axis]} are {got}, "
        f"expected {want}")
    return got
