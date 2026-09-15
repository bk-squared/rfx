"""Crossval-side build-time realized-conductor gate (#931).

Thin ADAPTER over ``tests/_realized_geometry.py``, which is this branch's
one implementation of "did the conductor realize where the script says it
did".  Nothing here re-derives a realization: ``realize`` calls
``tests._realized_geometry.realized`` and the plane comparison calls its
``assert_wall_planes``.  A second hand-rolled ``argwhere`` over some mask
is exactly the drift the single-owner rule (design note §1.7) exists to
stop, and the pre-#931 tree had three of them (cv15, preflight, the MSL
trace detector) that disagreed.

What this module adds on top, and why it is not in the shared helper:

* a **physical column** (``at=(u, v)`` in metres on the two in-plane axes)
  instead of node indices — a crossval script states its geometry in
  metres, and converting per call site is where an off-by-one gets
  introduced;
* ``tol_m`` — the shared assertion matches a declared coordinate to its
  NEAREST node and then compares indices, so a declaration half a cell
  off-lattice still passes.  For a fixture that claims to be drawn on the
  node line, that is the thing worth catching, so the offset is checked
  explicitly;
* :func:`assert_wall_span` — a solid body is PEC on EVERY node plane it
  spans, not only its two faces, so a thick body's declaration is its two
  faces and the contiguous run between them is filled in here rather than
  spelled as an index list per mesh rung;
* :func:`assert_no_conductor` — the dielectric controls (cv22, cv23, cv24)
  claim bit-identity across the contract, and that claim is only checkable
  if "this case realizes no conductor at all" is asserted rather than
  assumed;
* :func:`footprint_nodes` — a realized trace/patch width, read off the
  edge set, for a script that wants to log it beside its own numbers.

Vocabulary (design note §1): **volume** = a set of primal cells, every
incident edge PEC, so a body drawn ``z_a -> z_b`` on node planes realizes
walls at BOTH; **sheet** = a footprint on ONE node plane, zero thickness,
normal edge live; **wire** = a 1-D lattice path of edges.

The falsifier discipline is cv24's (``nu_cavity_gates``
``extent_plus_one_fine_cell``): a gate that cannot fail by name on a
deliberately mis-realized geometry is not evidence.  Every assertion here
names the declared and realized plane lists, so the failure IS the
diagnosis.
"""

from __future__ import annotations

import numpy as np

from tests import _realized_geometry as _RG

__all__ = [
    "realize",
    "wall_planes",
    "sheet_planes",
    "footprint_nodes",
    "assert_wall_planes",
    "assert_wall_span",
    "assert_no_conductor",
]


def realize(sim, grid=None):
    """``Realization`` for ``sim`` — grid, cell mask, sheets, wires, edges.

    Build-time only (grid construction plus material assembly, no fields),
    and the LANE is chosen by the shared helper from the simulation's own
    ``_dx/_dy/_dz_profile``, not by a caller-supplied string: a gate that
    realizes on the uniform grid for a graded fixture passes for the wrong
    reason.  ``grid`` is accepted and ignored for call-site convenience.

    One case the shared helper does not cover, handled here rather than by
    editing it: a run that declares NO conductor at all gives
    ``realized_pec_edge_masks`` nothing to take a grid shape from, and it
    says so with a ``ValueError``. The controls (cv22/cv23/cv24) are
    exactly that case and want the empty answer, so it is spelled out --
    zero edges on the run's own grid -- instead of teaching the owner about
    emptiness. Reported upstream as a small gap in the helper.
    """
    try:
        return _RG.realized(sim)
    except ValueError as exc:
        if "no cell mask, sheets or wires" not in str(exc):
            raise
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nu = any(getattr(sim, f"_{a}_profile", None) is not None
                     for a in ("dx", "dy", "dz"))
            g = (sim._build_nonuniform_grid() if nu else sim._build_grid())
        zero = np.zeros(tuple(g.shape), dtype=bool)
        return _RG.Realization(
            g, None, [], [], (zero, zero.copy(), zero.copy()))


def _ij(rz, axis: int, at):
    """Physical ``(u, v)`` on the two in-plane axes -> node indices."""
    in_plane = [a for a in range(3) if a != axis]
    return tuple(_RG.node_index(rz.grid, a, float(u))
                 for a, u in zip(in_plane, at))


def wall_planes(sim, axis: int, *, at=None, region=None, grid=None):
    """``(planes, coords_m, grid)`` — realized tangential walls along ``axis``.

    ``at`` names ONE column by its physical ``(u, v)`` on the two in-plane
    axes (in axis order, skipping ``axis``); ``region`` is a tuple of three
    index slices; neither means the whole grid.
    """
    rz = realize(sim, grid)
    kw = {}
    if at is not None:
        kw["ij"] = _ij(rz, axis, at)
    elif region is not None:
        kw["region"] = tuple(region)
    planes = rz.wall_planes(axis, **kw)
    line = _RG._node_line(rz.grid, axis)
    return planes, [float(line[k]) for k in planes], rz.grid


def sheet_planes(sim, grid=None):
    """``[(name, normal_axis, plane, coordinate_m), ...]`` for every DECLARED
    sheet — what the SheetSpec says, before the edge rule is applied."""
    rz = realize(sim, grid)
    rows = []
    for sp in rz.sheets:
        line = _RG._node_line(rz.grid, int(sp.normal_axis))
        rows.append((sp.name, int(sp.normal_axis), int(sp.plane),
                     float(line[int(sp.plane)])))
    return rows


def footprint_nodes(sim, axis: int, plane: int, *, component: int = 0,
                    along: int | None = None, at: int | None = None,
                    grid=None):
    """Node indices of the realized ``component`` edges on ``(axis, plane)``.

    ``along`` is the axis to report node indices on and ``at`` the index on
    the remaining in-plane axis (default: that axis' mid-index). Used to
    check a realized trace/patch width against the drawn one.
    """
    rz = realize(sim, grid)
    m = np.asarray(rz.edge_masks[component], dtype=bool)
    in_plane = [a for a in range(3) if a != axis]
    if along is None:
        along = in_plane[0]
    other = [a for a in in_plane if a != along][0]
    if at is None:
        at = m.shape[other] // 2
    idx = [None, None, None]
    idx[axis] = int(plane)
    idx[other] = int(at)
    idx[along] = slice(None)
    return [int(v) for v in np.flatnonzero(m[tuple(idx)])]


def assert_wall_planes(sim, axis: int, declared, *, at=None, region=None,
                       grid=None, label: str = "conductor",
                       tol_m: float = 0.0):
    """Assert the realized wall planes along ``axis`` are exactly ``declared``.

    ``declared`` is a sequence of physical coordinates in metres when every
    entry is a float, otherwise node-plane INDICES.  The comparison itself
    is ``tests._realized_geometry.assert_wall_planes``; ``tol_m`` adds the
    off-lattice check that one cannot make (it matches to the nearest node
    first, so a declaration half a cell off the node line still passes).

    Returns the realized ``{"planes", "coords_m"}`` record so the caller can
    log it beside its own numbers.
    """
    rz = realize(sim, grid)
    declared = list(declared)
    as_metres = bool(declared) and all(isinstance(v, float) for v in declared)
    kw = {}
    if at is not None:
        kw["ij"] = _ij(rz, axis, at)
    elif region is not None:
        kw["region"] = tuple(region)
    if as_metres:
        planes = _RG.assert_wall_planes(
            sim, axis, list(declared), what=label, **kw)
    else:
        planes = _RG.assert_wall_planes(
            sim, axis, expected_planes=[int(v) for v in declared],
            what=label, **kw)
    line = _RG._node_line(rz.grid, axis)
    coords = [float(line[k]) for k in planes]
    if tol_m > 0.0 and as_metres:
        for k, target in zip(planes, sorted(float(v) for v in declared)):
            off = abs(float(line[k]) - target)
            if off > tol_m:
                raise AssertionError(
                    f"[{label}] realized wall plane {k} sits at "
                    f"{line[k] * 1e6:.4f} um, {off * 1e6:.4f} um from the "
                    f"declared {target * 1e6:.4f} um (tol {tol_m * 1e6:.4f} "
                    "um). The declared plane is off-lattice; redraw the "
                    "fixture on the node line (#931 §1.3 off-lattice "
                    "interfaces).")
    return {"planes": planes, "coords_m": coords}


def assert_wall_span(sim, axis: int, lo_m: float, hi_m: float, *, at=None,
                     region=None, grid=None, label: str = "conductor",
                     tol_m: float = 0.0):
    """Assert a SOLID body realizes walls on every node plane from ``lo_m``
    to ``hi_m``, and on no other.

    A volume more than one cell thick is PEC on every plane it spans: the
    tangential edges of every interior node plane are incident to an
    occupied cell.  The caller declares the two FACES and this fills in the
    contiguous run, so the declaration does not have to be re-derived as an
    index list at every mesh rung.  The falsifier property is unchanged: a
    body realized one plane short, one plane long, or with a hole loses
    exact equality and the failure names both lists.
    """
    rz = realize(sim, grid)
    k_lo = _RG.node_index(rz.grid, axis, float(lo_m))
    k_hi = _RG.node_index(rz.grid, axis, float(hi_m))
    want = list(range(min(k_lo, k_hi), max(k_lo, k_hi) + 1))
    out = assert_wall_planes(sim, axis, want, at=at, region=region,
                             grid=grid, label=label)
    if tol_m > 0.0:
        line = _RG._node_line(rz.grid, axis)
        for k, target in ((k_lo, float(lo_m)), (k_hi, float(hi_m))):
            off = abs(float(line[k]) - target)
            if off > tol_m:
                raise AssertionError(
                    f"[{label}] the declared face {target * 1e6:.4f} um is "
                    f"{off * 1e6:.4f} um from node {k} at "
                    f"{line[k] * 1e6:.4f} um (tol {tol_m * 1e6:.4f} um): "
                    "off-lattice, so drawn != realized whatever the plane "
                    "list says (#931 §1.3).")
    return out


def assert_no_conductor(sim, grid=None, label: str = "control"):
    """Assert the case realizes NO conductor at all — no PEC cell, no sheet,
    no wire, no PEC edge.

    The dielectric controls (cv22, cv23, cv24, and every slab-rig arm) are
    the regression that #931 did not leak outside the conductor path: the
    contract changes PEC sampling only, and dielectric sampling (node,
    half-open) is untouched.  Asserting "no conductor" is what makes their
    bit-identity claim checkable rather than assumed.
    """
    rz = realize(sim, grid)
    n_cells = 0 if rz.pec_mask is None else int(np.asarray(rz.pec_mask).sum())
    n_edges = int(sum(np.asarray(m, dtype=bool).sum() for m in rz.edge_masks))
    if n_cells or rz.sheets or rz.wires or n_edges:
        raise AssertionError(
            f"[{label}] expected a conductor-free case but realized "
            f"{n_cells} PEC cells, {len(rz.sheets)} sheets, "
            f"{len(rz.wires)} wires and {n_edges} PEC E edges. "
            "Domain-boundary PEC is applied by the boundary spec and is NOT "
            "a body (#931 §1.8), so it must not appear here.")
    return {"pec_cells": 0, "sheets": 0, "wires": 0, "pec_edges": 0}
