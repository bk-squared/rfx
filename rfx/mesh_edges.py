"""Mesh lines placed for conductor EDGES, not only for faces.

A flat PEC wall (a face) is exact on a node plane. The in-plane EDGE of a
PEC sheet is not: on the Yee grid the sheet behaves as if it ended about
0.3 of a cell BEYOND its last PEC node. Measured on a 2-D PEC box with a
thin fin, against a fine-mesh reference, for both polarizations and at two
cell sizes (``scripts/diagnostics/pec_sheet_edge_offset.py``): +0.29 ...
+0.32 cell. openEMS builds its meshes on the same fact -- its
``mesh_hint_from_box(metal_edge_res=r)`` puts lines at ``edge - r/3`` and
``edge + 2r/3`` and none on the edge (the "thirds rule").

So an edge drawn ON a node line is solved about 0.3 cell too long, an edge
drawn anywhere else is first moved to a node (up to a cell) and then solved
0.3 cell too long, and no choice of node removes the scatter: on the fin
the uniform grid is off by up to 2.4 % in frequency at a 1 mm cell, with a
sawtooth in the drawn length. Placing a node ``EDGE_OFFSET`` of a cell
INSIDE the metal removes it (0.13 % worst, which is the grid's own
dispersion).

:func:`edge_aware_profile` builds such an axis profile. It only computes
where the lines go; the cells between them come from
:func:`rfx.nonuniform.make_band_profile` (ratio cap, pinned boundary cell).
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np

from rfx.nonuniform import make_band_profile

#: Distance from the last PEC node to the electrical edge of a PEC sheet, in
#: cells. MEASURED on this solver (see the module docstring); openEMS uses 1/3.
EDGE_OFFSET = 0.30


class SheetEdge(NamedTuple):
    """One in-plane edge of a conductor sheet along the profiled axis.

    ``metal_side`` is ``-1`` when the metal lies on the LOWER-coordinate
    side of ``position`` (this is the sheet's upper end) and ``+1`` when it
    lies on the HIGHER side (its lower end).
    """

    position: float
    metal_side: int


class EdgeAwareProfile(NamedTuple):
    cells: np.ndarray
    #: node coordinates, ``lo`` first
    nodes: np.ndarray
    #: per SheetEdge: the node the sheet must END on (draw the sheet to it)
    metal_end_nodes: tuple
    #: per SheetEdge: the edge cell actually used (may be below ``dx``)
    edge_cells: tuple


def _edge_lines(edge: SheetEdge, res: float, offset: float):
    s = float(edge.metal_side)
    inside = edge.position + s * offset * res
    outside = edge.position - s * (1.0 - offset) * res
    return (inside, outside) if inside < outside else (outside, inside), inside


def _build(lo, hi, dx, fixed, edges, res, edge_offset, max_ratio,
           boundary_cell):
    """Lines for the given edge-cell sizes, then the cells between them."""
    res = list(res)
    # A strip (metal between two edges) or a gap (air between two metals) with
    # no face in between: pick ONE edge-cell size r for both edges so that the
    # stretch between their two nodes is a whole number of cells of that same
    # size -- D = (n + k) r, k = 2a inside a strip, 2(1 - a) across a gap.
    # Without this a stretch of 1.1 cells becomes two cells of 0.55 and the
    # ratio cap drags every neighbour down with it.
    paired = {}
    for i in range(len(edges) - 1):
        e1, e2 = edges[i], edges[i + 1]
        if any(e1.position < f < e2.position for f in fixed):
            continue
        if (e1.metal_side, e2.metal_side) == (1, -1):
            k = 2.0 * edge_offset
        elif (e1.metal_side, e2.metal_side) == (-1, 1):
            k = 2.0 * (1.0 - edge_offset)
        else:
            continue
        span = e2.position - e1.position
        n = max(0, int(np.ceil(span / min(res[i], res[i + 1]) - k - 1e-9)))
        r = span / (n + k) if (n + k) > 0 else None
        if r is not None and r <= min(res[i], res[i + 1]) * (1.0 + 1e-12):
            res[i] = res[i + 1] = r
            paired[i] = r
    # Shrink an edge cell until it clears its neighbours: other edge cells
    # and faces. Twelve steps of 0.8 take it to ~0.07 of its start; below
    # that the drawing, not the mesh, is the problem.
    for _ in range(12):
        spans = [_edge_lines(e, r, edge_offset)[0] for e, r in zip(edges, res)]
        clash = [False] * len(edges)
        for i, (a, b) in enumerate(spans):
            if a < lo or b > hi:
                clash[i] = True
            for f in fixed:
                if a + 1e-15 < f < b - 1e-15:
                    clash[i] = True
            if i + 1 < len(spans) and spans[i + 1][0] < b - 1e-15:
                clash[i] = clash[i + 1] = True
        if not any(clash):
            break
        res = [r * 0.8 if c else r for r, c in zip(res, clash)]
    else:
        bad = [edges[i].position for i, c in enumerate(clash) if c]
        raise ValueError(
            "edge cells still collide with a face or with each other at a "
            f"fraction of dx; conflicting sheet edges at {bad} m. Move the "
            "face off the edge, widen the strip or gap, or lower dx.")
    lines = set(fixed)
    protected_spans, ends = [], []
    for e, r in zip(edges, res):
        (a, b), inside = _edge_lines(e, r, edge_offset)
        lines.update((a, b))
        protected_spans.append((a, b))
        ends.append(inside)
    pts = sorted(lines)
    merged = [pts[0]]
    for p in pts[1:]:
        if p - merged[-1] > 1e-12:
            merged.append(p)
    sizes, prot = [], []
    for a, b in zip(merged[:-1], merged[1:]):
        is_edge_cell = any(abs(pa - a) < 1e-12 and abs(pb - b) < 1e-12
                           for pa, pb in protected_spans)
        inner = [r for i, r in paired.items()
                 if min(ends[i], ends[i + 1]) - 1e-12 <= a
                 and b <= max(ends[i], ends[i + 1]) + 1e-12]
        if is_edge_cell:
            sizes.append(b - a)
            prot.append(True)
        elif inner:
            sizes.append(inner[0])        # whole cells of the pair's size
            prot.append(True)
        else:
            sizes.append(dx)
            prot.append(False)
    cells = make_band_profile(merged, sizes, max_ratio=max_ratio,
                              protected=prot, boundary_cell=boundary_cell)
    return cells, merged, ends, res


def edge_aware_profile(
    lo: float,
    hi: float,
    dx: float,
    *,
    faces: Sequence[float] = (),
    sheet_edges: Sequence[SheetEdge] = (),
    max_ratio: float = 1.3,
    boundary_cell: float | None = None,
    edge_offset: float = EDGE_OFFSET,
) -> EdgeAwareProfile:
    """Cell sizes for one axis with a node ``edge_offset`` cell inside every
    sheet edge and a node ON every face.

    Parameters
    ----------
    lo, hi : float
        Axis extent (m). Both are faces.
    dx : float
        Target cell size (m).
    faces : sequence of float
        Coordinates that must lie ON a node: PEC walls, dielectric
        interfaces, port and probe planes.
    sheet_edges : sequence of SheetEdge
        In-plane edges of zero-thickness conductors along this axis.
    max_ratio, boundary_cell
        Passed to :func:`rfx.nonuniform.make_band_profile`.
    edge_offset : float
        Fraction of the edge cell between the last PEC node and the edge.

    Each edge gets ONE protected cell that straddles it, of size ``dx``
    unless a neighbouring constraint is closer than that, in which case the
    edge cell shrinks to fit (a strip narrower than ``(1 + 2*edge_offset)``
    cells, or a gap narrower than ``2*(1 - edge_offset)`` cells, is meshed
    with smaller edge cells rather than refused).

    Raises
    ------
    ValueError
        When two constraints cannot both be met -- a face inside an edge
        cell, or two edge cells that would overlap even after shrinking to
        a tenth of ``dx``.
    """
    lo, hi, dx = float(lo), float(hi), float(dx)
    if not (hi > lo and dx > 0.0):
        raise ValueError("need hi > lo and dx > 0")
    if not (0.0 <= edge_offset < 1.0):
        raise ValueError("edge_offset must be in [0, 1)")
    edges = sorted((SheetEdge(float(e.position), int(e.metal_side))
                    for e in sheet_edges), key=lambda e: e.position)
    for e in edges:
        if e.metal_side not in (-1, 1):
            raise ValueError("SheetEdge.metal_side must be -1 or +1")
        if not (lo < e.position < hi):
            raise ValueError(
                f"sheet edge at {e.position:g} m is outside ({lo:g}, {hi:g})")
    fixed = sorted({lo, hi, *(float(f) for f in faces)})

    # The ratio cap may make ``make_band_profile`` refine an edge cell (a
    # protected block next to smaller cells is split, e.g. a full-size edge
    # cell next to a stretch that only divides into 0.77-size cells). The
    # node then sits ``edge_offset`` of the WRONG cell from the edge, so the
    # realized edge-cell size is fed back until the cell that straddles each
    # edge is the one the lines were computed for. Starting from a slightly
    # smaller edge cell often avoids the split altogether, so a few starting
    # sizes are tried and the profile with the LARGEST smallest cell wins:
    # the time step follows the smallest cell.
    best = None
    last_error = None
    for scale in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75):
        res = [dx * scale] * len(edges)
        try:
            for _ in range(8):
                cells, merged, ends, res = _build(
                    lo, hi, dx, fixed, edges, res, edge_offset, max_ratio,
                    boundary_cell)
                nodes = lo + np.concatenate([[0.0], np.cumsum(cells)])
                realized = []
                for e in edges:
                    k = int(np.searchsorted(nodes, e.position, side="right")) - 1
                    realized.append(float(nodes[k + 1] - nodes[k]))
                if all(abs(a - b) <= 1e-9 * dx for a, b in zip(realized, res)):
                    break
                res = [min(a, b) for a, b in zip(realized, res)]
            else:
                raise ValueError(
                    "the edge cells did not settle under the ratio cap; "
                    "raise max_ratio or lower dx")
        except ValueError as exc:
            last_error = exc
            continue
        key = (float(cells.min()), -int(cells.size))
        if best is None or key > best[0]:
            best = (key, cells, nodes, ends, realized)
        if not edges:
            break
    if best is None:
        raise last_error
    _, cells, nodes, ends, realized = best
    return EdgeAwareProfile(cells=cells, nodes=nodes,
                            metal_end_nodes=tuple(ends),
                            edge_cells=tuple(realized))


def _box_corners(shape):
    lo = getattr(shape, "corner_lo", None)
    hi = getattr(shape, "corner_hi", None)
    if lo is None or hi is None:
        raise TypeError(
            "edge_aware_profiles reads axis-aligned Box shapes (corner_lo / "
            f"corner_hi); got {type(shape).__name__}. Pass its bounding faces "
            "through `faces=` instead.")
    return tuple(float(v) for v in lo), tuple(float(v) for v in hi)


def edge_aware_profiles(
    domain,
    dx: float,
    *,
    sheets: Sequence = (),
    solids: Sequence = (),
    faces=None,
    axes: str = "xy",
    max_ratio: float = 1.3,
    boundary_cell: float | None = None,
    edge_offset: float = EDGE_OFFSET,
) -> dict:
    """Profiles for ``Simulation(dx_profile=..., dy_profile=..., dz_profile=...)``
    straight from the drawing.

    Parameters
    ----------
    domain : (Lx, Ly, Lz)
        The same tuple the ``Simulation`` gets.
    dx : float
        Target cell size (m).
    sheets : sequence of Box
        Zero-thickness conductor Boxes (patches, strips, ground planes that
        end inside the domain). Along each in-plane axis both ends become
        sheet edges; a sheet edge that coincides with the domain boundary is
        a wall and is skipped. The sheet's own plane becomes a face on its
        normal axis.
    solids : sequence of Box
        Volumes -- dielectric slabs, PEC blocks. Every face goes ON a node.
    faces : dict, optional
        Extra node planes per axis, ``{"x": [...], "z": [...]}``: port
        planes, probe planes, the faces of a shape that is not a Box.
    axes : str
        Which profiles to build, any of ``"x"``, ``"y"``, ``"z"``.
    boundary_cell : float, optional
        Pinned first/last cell (default ``dx``; the x/y absorber contract
        needs the two end cells equal).

    Returns
    -------
    dict
        ``{"dx_profile": cells, "dy_profile": cells, ...}`` for the axes
        asked for -- pass it as ``Simulation(..., dx=dx, **profiles)`` and
        draw the geometry at its TRUE size.

    Example
    -------
    >>> patch = Box((xc - L/2, yc - W/2, h), (xc + L/2, yc + W/2, h))
    >>> slab = Box((0, 0, 0), (LX, LY, h))
    >>> prof = edge_aware_profiles((LX, LY, LZ), 1e-3, sheets=[patch],
    ...                            solids=[slab])
    >>> sim = Simulation(freq_max=5e9, domain=(LX, LY, LZ), dx=1e-3, **prof)
    >>> sim.add(slab, material="substrate"); sim.add(patch, material="pec")
    """
    names = {"x": 0, "y": 1, "z": 2}
    if not axes or any(a not in names for a in axes):
        raise ValueError("axes must be a non-empty subset of 'xyz'")
    extra = {k: [float(v) for v in vals] for k, vals in (faces or {}).items()}
    pin = float(dx) if boundary_cell is None else float(boundary_cell)
    out = {}
    for a in axes:
        k = names[a]
        hi_dom = float(domain[k])
        tol = 1e-9 * float(dx)
        axis_faces = list(extra.get(a, ()))
        axis_edges = {}
        for sh in solids:
            lo, hi = _box_corners(sh)
            axis_faces += [lo[k], hi[k]]
        for sh in sheets:
            lo, hi = _box_corners(sh)
            thin = [i for i in range(3) if abs(hi[i] - lo[i]) <= tol]
            if len(thin) != 1:
                raise ValueError(
                    "a sheet must be a Box with exactly one zero-thickness "
                    f"axis; got extents {tuple(h - l for l, h in zip(lo, hi))}")
            if k == thin[0]:
                axis_faces.append(lo[k])
                continue
            for pos, side in ((lo[k], +1), (hi[k], -1)):
                if tol < pos < hi_dom - tol:
                    axis_edges[(round(pos / tol), side)] = SheetEdge(pos, side)
        edge_pos = {round(e.position / tol) for e in axis_edges.values()}
        axis_faces = [f for f in axis_faces
                      if tol < f < hi_dom - tol and round(f / tol) not in edge_pos]
        prof = edge_aware_profile(
            0.0, hi_dom, dx, faces=axis_faces,
            sheet_edges=list(axis_edges.values()), max_ratio=max_ratio,
            boundary_cell=pin, edge_offset=edge_offset)
        out[f"d{a}_profile"] = prof.cells
    return out
