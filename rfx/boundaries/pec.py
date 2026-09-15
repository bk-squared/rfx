"""Perfect Electric Conductor (PEC): domain faces and conductor bodies.

Domain-face PEC (:func:`apply_pec`, :func:`apply_pec_faces`) keeps its own
convention (E_tan = 0 on the face plane at index 0 / N) — it is not a body.
Conductor bodies follow the lattice ownership contract (#931):
:func:`realized_pec_edge_masks` is the one source of PEC E edges for
volumes, sheets and wires.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from rfx.core.jax_utils import is_tracer as _is_tracer
from rfx.core.yee import _shift_bwd, _shift_fwd


def apply_pec(state, axes: str = "xyz") -> object:
    """Apply PEC (E_tan = 0) at domain boundaries.

    Parameters
    ----------
    state : FDTDState
    axes : str
        Which axes to apply PEC on. Default "xyz" = all 6 faces.
    """
    ex, ey, ez = state.ex, state.ey, state.ez

    if "x" in axes:
        # PEC at x=0 and x=end: Ey, Ez tangential → 0
        ey = ey.at[0, :, :].set(0.0)
        ey = ey.at[-1, :, :].set(0.0)
        ez = ez.at[0, :, :].set(0.0)
        ez = ez.at[-1, :, :].set(0.0)

    if "y" in axes:
        # PEC at y=0 and y=end: Ex, Ez tangential → 0
        ex = ex.at[:, 0, :].set(0.0)
        ex = ex.at[:, -1, :].set(0.0)
        ez = ez.at[:, 0, :].set(0.0)
        ez = ez.at[:, -1, :].set(0.0)

    if "z" in axes:
        # PEC at z=0 and z=end: Ex, Ey tangential → 0
        ex = ex.at[:, :, 0].set(0.0)
        ex = ex.at[:, :, -1].set(0.0)
        ey = ey.at[:, :, 0].set(0.0)
        ey = ey.at[:, :, -1].set(0.0)
        # Ez at k=nz-1 is a ghost cell outside the physical domain.
        ez = ez.at[:, :, -1].set(0.0)

    return state._replace(ex=ex, ey=ey, ez=ez)


def apply_pec_faces(state, faces: set[str]) -> object:
    """Apply PEC (E_tan = 0) on specific boundary faces.

    Parameters
    ----------
    state : FDTDState
    faces : set of str
        Which faces to enforce PEC on.  Valid names:
        ``"x_lo"``, ``"x_hi"``, ``"y_lo"``, ``"y_hi"``,
        ``"z_lo"``, ``"z_hi"``.
    """
    if not faces:
        return state
    ex, ey, ez = state.ex, state.ey, state.ez

    if "x_lo" in faces:
        ey = ey.at[0, :, :].set(0.0)
        ez = ez.at[0, :, :].set(0.0)
    if "x_hi" in faces:
        ey = ey.at[-1, :, :].set(0.0)
        ez = ez.at[-1, :, :].set(0.0)
    if "y_lo" in faces:
        ex = ex.at[:, 0, :].set(0.0)
        ez = ez.at[:, 0, :].set(0.0)
    if "y_hi" in faces:
        ex = ex.at[:, -1, :].set(0.0)
        ez = ez.at[:, -1, :].set(0.0)
    if "z_lo" in faces:
        ex = ex.at[:, :, 0].set(0.0)
        ey = ey.at[:, :, 0].set(0.0)
    if "z_hi" in faces:
        ex = ex.at[:, :, -1].set(0.0)
        ey = ey.at[:, :, -1].set(0.0)
        # Ez at k=nz-1 is a ghost cell at z=(nz-0.5)*dx, outside the
        # physical domain. Zero it to prevent ghost-layer accumulation.
        ez = ez.at[:, :, -1].set(0.0)

    return state._replace(ex=ex, ey=ey, ez=ez)


# ---------------------------------------------------------------------------
# Lattice ownership contract for conductors (#931)
#
# One sentence, three regions: an E component is PEC iff its own location is
# inside the closed conductor region — a VOLUME (a set of primal cells), a
# SHEET (a footprint on one node plane, zero thickness) or a WIRE (a 1-D path
# of edges).  :func:`realized_pec_edge_masks` is the ONLY function that turns
# geometry into PEC edges; every consumer reads its ``(Mx, My, Mz)`` or one
# of the two helpers built on it (:func:`realized_wall_planes`,
# :func:`edge_is_pec`).  Normative text:
# docs/design_notes/20260906_plan_realign_lattice_ownership.md (§1).
#
# Index conventions (§1.1): node ``i`` at ``x_i`` is the LOWER corner of
# primal cell ``i``; ``Ex[i,j,k]`` sits at ``(x_{i+1/2}, y_j, z_k)``,
# ``Ey[i,j,k]`` at ``(x_i, y_{j+1/2}, z_k)``, ``Ez[i,j,k]`` at
# ``(x_i, y_j, z_{k+1/2})``.  ``C[i,j,k]`` means primal cell ``(i,j,k)`` is
# conductor.
# ---------------------------------------------------------------------------


def _shift(arr, ax, periodic, direction):
    """THE single spelling of the #689 boundary convention.

    ``direction=+1`` returns ``arr[i-1]`` (backward neighbour), ``-1``
    returns ``arr[i+1]`` (forward neighbour).  Wrap (``jnp.roll``) on a
    periodic axis and on a length-1 axis (the 2-D lane's self-adjacency);
    explicit zero pad (``_shift_bwd`` / ``_shift_fwd``) otherwise — the
    same out-of-domain convention the solver's curl uses.  A hand-copied
    second neighbour rule is this repo's recurring defect (#689/#690
    class); volume, sheet and wire realization all come through here.
    """
    if arr.shape[ax] == 1 or periodic[ax]:
        return jnp.roll(arr, direction, axis=ax)
    return _shift_bwd(arr, ax) if direction > 0 else _shift_fwd(arr, ax)


@dataclass(frozen=True)
class SheetSpec:
    """One PEC sheet: a node footprint on ONE plane, zero thickness (§1.3).

    ``footprint`` is a boolean ``(nx, ny, nz)`` NODE mask that is True only
    on layer ``plane`` along ``normal_axis`` (the full-shape layout keeps it
    interchangeable with ``SheetImpedanceSpec.mask``, so a lossy f0 sheet
    and a PEC sheet are one footprint with a different operator — the #677
    G4 identity by construction).  ``plane`` is a static Python int: a
    sheet's plane is never a traced quantity (the ``argmin`` cliff).  A
    sheet owns NO cell — it adds nothing to the cell mask and writes no
    material.
    """
    normal_axis: int
    plane: int
    footprint: object
    name: str | None = None

    def __post_init__(self):
        a = int(self.normal_axis)
        if a not in (0, 1, 2):
            raise ValueError(f"SheetSpec.normal_axis must be 0/1/2, got {self.normal_axis!r}")
        object.__setattr__(self, "normal_axis", a)
        object.__setattr__(self, "plane", int(self.plane))
        fp = self.footprint
        if getattr(fp, "ndim", None) != 3:
            raise ValueError("SheetSpec.footprint must be a 3-D (nx, ny, nz) boolean array")
        if not (0 <= self.plane < fp.shape[a]):
            raise ValueError(
                f"SheetSpec.plane={self.plane} is outside the array along axis "
                f"{'xyz'[a]} (length {fp.shape[a]})")
        if not _is_tracer(fp):
            fp_np = np.asarray(fp, dtype=bool)
            other = tuple(b for b in range(3) if b != a)
            layers = np.flatnonzero(np.any(fp_np, axis=other))
            if layers.size and (layers.size != 1 or int(layers[0]) != self.plane):
                raise ValueError(
                    f"SheetSpec footprint must occupy exactly its own plane "
                    f"{self.plane} along {'xyz'[a]}; found layers {layers.tolist()}")
            object.__setattr__(self, "footprint", jnp.asarray(fp_np))


@dataclass(frozen=True)
class WireSpec:
    """One PEC filament: the E edges of an axis-aligned lattice path (§1.4).

    ``edges`` is the ``(Mx, My, Mz)`` boolean triple naming the edges on the
    path — built by :func:`wire_path_edge_masks` from the path's node
    indices.  A wire owns no cell.
    """
    edges: tuple
    name: str | None = None


def wire_path_edge_masks(nodes, shape):
    """E edges of the axis-aligned lattice path through ``nodes`` (§1.4).

    ``nodes`` is a sequence of integer ``(i, j, k)`` node indices; each
    consecutive pair must differ along exactly ONE axis (a diagonal
    segment raises — today such a wire silently rasterizes to nothing).
    The edge between node ``i`` and ``i+1`` along ``x`` is ``Ex[i, j, k]``,
    so a segment from ``a`` to ``b`` marks ``E_ax[min(a,b) .. max(a,b)-1]``
    — the interval BETWEEN the two indices, never the way round through
    the seam.  A wire that is meant to cross a periodic seam is drawn as
    two legs (… -> the hi rim node, then the lo rim node -> …); there is
    no periodic argument here, because a path is a list of edges the
    caller named, not a neighbour rule.
    """
    masks = [np.zeros(tuple(shape), dtype=bool) for _ in range(3)]
    pts = [tuple(int(v) for v in n) for n in nodes]
    if len(pts) < 2:
        raise ValueError("a PolylineWire filament needs at least two nodes")
    for a, b in zip(pts[:-1], pts[1:]):
        moving = [ax for ax in range(3) if a[ax] != b[ax]]
        if len(moving) != 1:
            raise ValueError(
                f"PolylineWire segment {a} -> {b} is not axis-aligned; only "
                "axis-aligned segments are supported (lattice ownership "
                "contract §1.4). Split the segment into axis-aligned legs.")
        ax = moving[0]
        lo, hi = sorted((a[ax], b[ax]))
        idx = [slice(v, v + 1) for v in a]
        idx[ax] = slice(lo, hi)
        masks[ax][tuple(idx)] = True
    return tuple(jnp.asarray(m) for m in masks)


def wire_node_footprint(wires, shape=None):
    """NODE mask covered by ``wires`` — both end nodes of every path edge.

    A filament owns no cell, so ``pec_mask`` cannot carry it and an
    occupancy / connectivity read (``Simulation.conductor_mask``, a
    footprint plot) that unions only cells and sheet footprints reports a
    wire-fed model as having no metal along the wire.  The node set is the
    honest cell-shaped answer for a 1-D region: edge ``c`` at index ``i``
    joins node ``i`` to node ``i+1`` along axis ``c``, so the nodes are
    ``M_c | shift_bwd(M_c)`` unioned over the three components.

    There is no ``periodic`` argument, for the same reason
    :func:`wire_path_edge_masks` has none: a path is the list of edges the
    caller named, and a wire meant to cross a seam is drawn as two legs.
    The zero-padded shift is therefore exact at both rims — node 0 is an
    endpoint of edge 0 only, node ``n-1`` of edge ``n-2`` only.

    ``shape`` seeds an all-False result when ``wires`` is empty; without
    it an empty list returns ``None`` rather than inventing a grid size.
    """
    out = None
    for w in wires or ():
        for c in range(3):
            m = jnp.asarray(w.edges[c], dtype=bool)
            nodes = m | _shift(m, c, (False, False, False), +1)
            out = nodes if out is None else (out | nodes)
    if out is None and shape is not None:
        return jnp.zeros(tuple(shape), dtype=bool)
    return out


def _volume_edge_masks(cell_mask, periodic):
    """§1.2: an edge is PEC iff it is incident to an occupied cell.

    ``Mx[i,j,k] = C[i,j,k] | C[i,j-1,k] | C[i,j,k-1] | C[i,j-1,k-1]`` and
    cyclically — the four cells sharing the edge, reached by the backward
    shifts along the two axes transverse to the component.
    """
    C = cell_mask
    out = []
    for c in range(3):
        m = C
        for t in range(3):
            if t == c:
                continue
            m = m | _shift(m, t, periodic, +1)
        out.append(m)
    return tuple(out)


def _volume_occupancy_masks(occ, periodic):
    """§1.6: noisy-OR of the four incident cells, ``M = 1 - Π(1 - o_c)``.

    Same shifts as :func:`_volume_edge_masks`, applied to the OCCUPANCY
    (so the zero pad of a non-periodic axis reads "no conductor outside",
    exactly as the hard rule's ``False`` pad does — shifting ``1 - o``
    would pad with "conductor").  The pairwise noisy-OR ``a ⊕ b =
    1 - (1-a)(1-b)`` chained over the two transverse axes is the
    four-cell product; at binary occupancy it is exactly 0 or 1, so the
    result is bit-identical to the hard rule.
    """
    out = []
    for c in range(3):
        m = occ
        for t in range(3):
            if t == c:
                continue
            m = 1.0 - (1.0 - m) * (1.0 - _shift(m, t, periodic, +1))
        out.append(m)
    return tuple(out)


def _sheet_edge_masks(sheets, shape, periodic):
    """§1.3: E_t (t != normal) on the sheet plane, both end nodes in F.

    Footprints are UNIONED per normal axis BEFORE the edge rule (two
    abutting sheets realize seamlessly; a per-sheet application would leave
    the shared edge live — a slit).  Because a footprint is a full-shape
    mask that is non-zero only on its own plane and the end-node shift runs
    along an IN-PLANE axis, the per-axis union is exactly the per-(axis,
    plane) union the design note asks for, and sheets on adjacent planes
    stay two films with the normal edge between them live (#690).

    On a length-1 normal axis (the 2-D lane) the region has no thickness
    direction, so the "normal" component is not a through-sheet edge: it
    is PEC exactly at the footprint nodes (the closed 2-D region evaluated
    for that component's own location), and the in-plane components take
    the usual both-end-nodes rule. That is the same set the volume rule
    gives the same drawn rectangle on the 2-D lane (design note §1.3:
    "realized as a 2-D volume of its footprint").
    """
    edge = [None, None, None]
    per_axis = {}
    for sp in sheets:
        fp = sp.footprint
        if tuple(fp.shape) != tuple(shape):
            raise ValueError(
                f"SheetSpec footprint shape {tuple(fp.shape)} does not match the "
                f"grid shape {tuple(shape)}")
        a = sp.normal_axis
        per_axis[a] = fp if a not in per_axis else (per_axis[a] | fp)
    for a, F in per_axis.items():
        for t in range(3):
            if t == a:
                if shape[a] == 1:
                    edge[t] = F if edge[t] is None else (edge[t] | F)
                continue
            m = F & _shift(F, t, periodic, -1)
            edge[t] = m if edge[t] is None else (edge[t] | m)
    return edge


def realized_pec_edge_masks(cell_mask, sheets=(), wires=(),
                            periodic=(False, False, False)):
    """The one function that turns conductor geometry into PEC E edges.

    Parameters
    ----------
    cell_mask : (nx, ny, nz) bool array or None
        Primal-cell occupancy of every PEC VOLUME (Box / Sphere / Cylinder
        via ``sim.add``, and thick ``PolylineWire``).  ``None`` when the run
        has no volume conductor.
    sheets : iterable of :class:`SheetSpec`
        PEC sheets (``add_thin_conductor`` with PEC conductivity).
    wires : iterable of :class:`WireSpec`
        PEC filaments.
    periodic : (bool, bool, bool)
        The run's per-axis periodic flags (#689).  Callers on a periodic
        lane MUST pass their own; the default is the non-periodic
        convention.

    Returns
    -------
    (Mx, My, Mz) boolean arrays, grid shape: True where that E component
    is zeroed.  Consequences the contract test pins: a Box drawn
    ``z_a -> z_b`` on node planes realizes walls at BOTH planes and shorts
    every normal edge between them; a 1-cell Box is a filled slab with two
    faces on every axis; a sheet is one plane with its normal edge live;
    the rule is invariant under mirror and axis permutation.
    """
    sheets = list(sheets or ())
    wires = list(wires or ())
    if cell_mask is None:
        src = next((sp.footprint for sp in sheets), None)
        if src is None:
            src = next((w.edges[0] for w in wires), None)
        if src is None:
            raise ValueError("realized_pec_edge_masks: no cell mask, sheets or wires given")
        shape = tuple(src.shape)
    else:
        shape = tuple(cell_mask.shape)
    sheet_edges = _sheet_edge_masks(sheets, shape, periodic)
    cells = cell_mask
    if cells is not None:
        out = list(_volume_edge_masks(cells.astype(bool), periodic))
    else:
        out = [jnp.zeros(shape, dtype=bool) for _ in range(3)]
    for c in range(3):
        if sheet_edges[c] is not None:
            out[c] = out[c] | sheet_edges[c]
        for w in wires:
            out[c] = out[c] | w.edges[c]
    return tuple(out)


def realized_wall_planes(edge_masks, axis, *, ij=None, region=None,
                         periodic=(False, False, False)):
    """Sorted node-plane indices along ``axis`` where a tangential wall exists.

    A plane ``k`` counts when some E component tangential to ``axis`` (the
    two components other than ``axis``) is PEC on that plane, inside the
    selection:

    * ``region`` — a tuple of three slices over the grid; ``None`` is the
      whole grid.  Any tangential edge whose index falls in the region.
    * ``ij`` — the two in-plane node indices of ONE column (in axis order,
      skipping ``axis``).  A wall counts on the column when any tangential
      edge INCIDENT to the node ``(ij, k)`` is PEC, i.e. ``E_t`` at the
      node's own index or at the backward index along ``t`` — so a node on
      the hi rim of a footprint, whose incident edges are stored at
      ``i-1``, is found.  The backward neighbour follows the run's #689
      convention: on a periodic (or length-1) in-plane axis the backward
      neighbour of index 0 is ``n-1``, so a wall on the seam node is
      found; on a non-periodic axis index 0 has no backward edge.

    ``periodic`` is the run's per-axis flags; callers on a periodic lane
    MUST pass their own (the default is the non-periodic convention).  It
    only affects the ``ij=`` column form — ``region=`` already scans every
    stored index.

    ``ij`` and ``region`` are exclusive.  Consumers: preflight cavity /
    guide-width checks, cv15 ``assert_realized_stack``, MSL trace
    detection, oracles.

    EVERY plane, not just the two faces.  A body three cells thick has a
    tangential wall on all four of its node planes, because the interior
    is shorted too, so this returns four indices and not two.  ``max -
    min`` is therefore the body's OUTER span and over-reads a guide or a
    cavity bounded by thick walls: a 40-cell guide inside 3-cell walls
    gives ``[0, 1, 2, 3]`` and ``[43, 44, 45]``, and ``45 - 0`` is 45, not
    40.  A consumer that wants the clear opening brackets the aperture
    between the hi plane of one wall group and the lo plane of the next
    (``_port_transverse_spans`` does, and reads 40).
    """
    if ij is not None and region is not None:
        raise ValueError("pass either ij= or region=, not both")
    tangential = [c for c in range(3) if c != axis]
    hit = None
    if ij is not None:
        i, j = (int(v) for v in ij)
        in_plane = tangential
        for t in tangential:
            m = np.asarray(edge_masks[t], dtype=bool)
            idx = [None, None, None]
            idx[in_plane[0]] = i
            idx[in_plane[1]] = j
            idx[axis] = slice(None)
            col = m[tuple(idx)]
            back = list(idx)
            if back[t] - 1 >= 0:
                back[t] = back[t] - 1
                col = col | m[tuple(back)]
            elif m.shape[t] == 1:
                pass  # length-1 axis: the node's own index IS the wrap neighbour
            elif periodic[t]:
                back[t] = m.shape[t] - 1   # #689 wrap: node 0's backward edge
                col = col | m[tuple(back)]
            hit = col if hit is None else (hit | col)
    else:
        sel = tuple(region) if region is not None else (slice(None),) * 3
        other = tuple(c for c in range(3) if c != axis)
        for t in tangential:
            m = np.asarray(edge_masks[t], dtype=bool)
            sub = np.zeros_like(m)
            sub[sel] = m[sel]
            col = np.any(sub, axis=other)
            hit = col if hit is None else (hit | col)
    return [int(k) for k in np.flatnonzero(hit)]


_COMPONENT_INDEX = {"ex": 0, "ey": 1, "ez": 2, "x": 0, "y": 1, "z": 2, 0: 0, 1: 1, 2: 2}


def edge_is_pec(edge_masks, component, i, j, k) -> bool:
    """True iff the E edge ``component`` at index ``(i, j, k)`` is PEC.

    ``component`` is ``"ex"``/``"ey"``/``"ez"``, ``"x"``/``"y"``/``"z"`` or
    0/1/2.  Wire-port live-cell logic (#556 end gap, #929 ``port_in_pec``),
    probes and sources read this instead of scanning a cell mask.
    """
    c = _COMPONENT_INDEX[component.lower() if isinstance(component, str) else int(component)]
    return bool(np.asarray(edge_masks[c])[int(i), int(j), int(k)])


def edges_are_pec(edge_masks, component, cells) -> list:
    """:func:`edge_is_pec` for a LIST of cells, with ONE host transfer.

    ``edge_is_pec`` pulls the whole component mask to the host per call,
    which the eager S-parameter loops used to pay once per wire cell per
    step.  Same rule, read once.
    """
    c = _COMPONENT_INDEX[
        component.lower() if isinstance(component, str) else int(component)]
    m = np.asarray(edge_masks[c], dtype=bool)
    return [bool(m[int(i), int(j), int(k)]) for (i, j, k) in cells]


def clear_edges(edge_masks, cells, component=None):
    """Un-zero E entries at the given cell indices (§1.9 port clearing).

    ``cells`` is either a boolean grid-shaped mask or an iterable of
    ``(i, j, k)`` index triples.  ``component`` names the ONE E component
    to release (``"ex"``/``"ey"``/``"ez"``, ``"x"``/``"y"``/``"z"`` or
    0/1/2); ``None`` releases all three.

    A port must pass its own component.  Under the ownership contract
    "live" is defined on the port's own component edge, so releasing the
    two TANGENTIAL edges at a port foot releases whatever conductor owns
    that node — a ground plane under an MSL feed, the top face of a body
    under a probe feed.  That is a hole the port never asked for, so the
    three-component form is only for callers that really mean "no
    conductor at this index".
    """
    if component is not None:
        c = _COMPONENT_INDEX[
            component.lower() if isinstance(component, str) else int(component)]
        comps = (c,)
    else:
        comps = (0, 1, 2)
    out = list(edge_masks)
    if (hasattr(cells, "shape")
            and tuple(getattr(cells, "shape", ())) == tuple(edge_masks[0].shape)):
        keep = ~jnp.asarray(cells, dtype=bool)
        for c in comps:
            out[c] = out[c] & keep
        return tuple(out)
    for (i, j, k) in cells:
        for c in comps:
            out[c] = out[c].at[int(i), int(j), int(k)].set(False)
    return tuple(out)


#: E is frozen where the Kottke Stage-2 inverse-permittivity tensor is this
#: small.  Same constant the step body's post-CPML re-enforcement and
#: ``apply_pec_h_mask`` selection use; named once so the fence below and the
#: step body cannot drift apart.
PEC_INV_THRESHOLD = 1e-9


def kottke_fenced_edge_masks(edge_masks, aniso_inv_eps, sheets=(), wires=(),
                             periodic=(False, False, False),
                             inv_threshold=PEC_INV_THRESHOLD):
    """§1.8 fence: under ``subpixel_smoothing="kottke_pec"`` the VOLUME's
    frozen set is the inverse-permittivity tensor's, not the ownership rule's.

    Stage 2 gives a partially filled edge a FRACTIONAL inverse permittivity —
    that fraction is the model.  Hard-zeroing such an edge because the §1.2
    ownership rule calls its cell occupied throws the subpixel result away and
    leaves a staircase.  §1.8 fences that path out of the contract, so on the
    Kottke lane an edge is applied only when

    * the tensor itself froze it (``inv < inv_threshold``) — the
      defense-in-depth re-zero that keeps float noise from accumulating in a
      fully-PEC cell — or
    * a SHEET or a WIRE owns it.  Those own no cell, so they contribute
      nothing to ``compute_inv_eps_tensor_diag`` and are invisible to the
      tensor; without this term a declared sheet would vanish on the Kottke
      lane.

    Volume edges the tensor left positive are released back to Kottke.  Port
    clearing survives: ``edge_masks`` arrives already cleared and the fence
    only removes entries.

    All-jnp and free of host-side decisions, so it also holds inside a trace.
    """
    sheets = tuple(sheets or ())
    wires = tuple(wires or ())
    shape = tuple(edge_masks[0].shape)
    if sheets or wires:
        owned = realized_pec_edge_masks(
            None, sheets=sheets, wires=wires, periodic=periodic)
    else:
        zero = jnp.zeros(shape, dtype=bool)
        owned = (zero, zero, zero)
    out = []
    for c in range(3):
        frozen = jnp.asarray(aniso_inv_eps[c]) < inv_threshold
        out.append(jnp.asarray(edge_masks[c], dtype=bool) & (frozen | owned[c]))
    return tuple(out)


def apply_pec_edges(state, edge_masks) -> object:
    """Zero E on the realized ``(Mx, My, Mz)`` edge masks."""
    mask_ex, mask_ey, mask_ez = edge_masks
    return state._replace(
        ex=state.ex * (1.0 - mask_ex.astype(state.ex.dtype)),
        ey=state.ey * (1.0 - mask_ey.astype(state.ey.dtype)),
        ez=state.ez * (1.0 - mask_ez.astype(state.ez.dtype)),
    )


def apply_pec_mask(state, pec_mask, periodic=(False, False, False),
                   sheets=()) -> object:
    """Zero E at the PEC edges realized from a cell mask (and sheets).

    Convenience wrapper: :func:`realized_pec_edge_masks` then
    :func:`apply_pec_edges`.  Step functions precompute the edge masks once
    at setup and call :func:`apply_pec_edges` directly; this wrapper is for
    tests and one-shot callers.  ``periodic`` is the run's flags (#689).
    """
    return apply_pec_edges(
        state, realized_pec_edge_masks(pec_mask, sheets=sheets, periodic=periodic))


def apply_pec_h_mask(state, pec_mask=None, *,
                     mask_hx=None, mask_hy=None, mask_hz=None) -> object:
    """Zero H-field components inside PEC cells.

    Stage 2 unified-path companion to ``apply_pec_mask``. The Stage 2
    inverse-permittivity tensor freezes E inside PEC (inv=0 → Ca=1,
    Cb=0) but does NOT damp H, which propagates freely via 1/μ. Over
    many periods (≥30·τ at typical RF parameters), this seeds late-
    time growth and float32 NaN. Stage 1's ``sigma=1e10`` fold
    provided implicit damping for both E and H via the
    sigma-coupled curl interaction; Stage 2 needs explicit H zeroing.

    Two modes:

    1. **Single cell-center mask** (``pec_mask``): zero all three H
       components at any cell where ``pec_mask`` is True. Use this
       when the "fully PEC" set has been pre-computed (all three
       Yee-staggered E components frozen at this cell index).

    2. **Per-component mask** (``mask_hx`` / ``mask_hy`` / ``mask_hz``,
       Stage 2 step B-v2): zero each H component independently
       according to its *driver* E components in the Yee curl. ``Hx``
       at (i, j+½, k+½) is updated by ``∂Ez/∂y - ∂Ey/∂z``; if both
       Ey (at index inv_yy) and Ez (at index inv_zz) are frozen at
       this cell, Hx has no driver → safe to zero. This catches
       boundary cells where one E component (perpendicular to the
       wall) has fractional inv but the two tangential components
       are zero — the mode that the all-zero ``pec_mask`` misses.

    Pass either ``pec_mask`` or all three of ``mask_h*``; a mix is
    accepted and the masks are OR'd.

    Parameters
    ----------
    state : FDTDState
    pec_mask : (nx, ny, nz) boolean array, optional
        True where the cell-center is inside a fully-PEC region.
    mask_hx, mask_hy, mask_hz : (nx, ny, nz) boolean arrays, optional
        Per-component zero masks (Yee-stagger aware). Stage 2 step
        B-v2 derives these from ``(inv_xx==0, inv_yy==0, inv_zz==0)``
        pairwise combinations.
    """
    dtype = state.hx.dtype
    # Build per-component boolean masks (default: nothing zeroed).
    zero_hx = mask_hx
    zero_hy = mask_hy
    zero_hz = mask_hz
    if pec_mask is not None:
        zero_hx = pec_mask if zero_hx is None else (zero_hx | pec_mask)
        zero_hy = pec_mask if zero_hy is None else (zero_hy | pec_mask)
        zero_hz = pec_mask if zero_hz is None else (zero_hz | pec_mask)
    keep_hx = (1.0 - zero_hx.astype(dtype)) if zero_hx is not None else 1.0
    keep_hy = (1.0 - zero_hy.astype(dtype)) if zero_hy is not None else 1.0
    keep_hz = (1.0 - zero_hz.astype(dtype)) if zero_hz is not None else 1.0
    return state._replace(
        hx=state.hx * keep_hx,
        hy=state.hy * keep_hy,
        hz=state.hz * keep_hz,
    )


def apply_pec_occupancy(state, pec_occupancy, periodic=(False, False, False),
                        sheet_edge_masks=None) -> object:
    """Differentiable PEC: relaxed occupancy on the §1.2 incident rule (§1.6).

    ``pec_occupancy`` is a float CELL field in ``[0, 1]``.  Each E component
    is scaled by ``1 - M`` with ``M`` the noisy-OR of its four incident
    cells, ``M = 1 - Π(1 - o_c)``, under the same #689 shifts as the hard
    rule — so at binary occupancy this is bit-identical to
    :func:`apply_pec_mask` (pinned on a battery that includes bodies on
    faces 0 and n-1 of a non-periodic axis and the periodic seam).  Sheets
    enter as STATIC edge masks (``sheet_edge_masks = (Sx, Sy, Sz)`` from
    :func:`realized_pec_edge_masks`) OR'd in; a sheet's plane is not a
    traced quantity.
    """
    occ = jnp.clip(pec_occupancy.astype(state.ex.dtype), 0.0, 1.0)
    m_ex, m_ey, m_ez = _volume_occupancy_masks(occ, periodic)
    if sheet_edge_masks is not None:
        sx, sy, sz = sheet_edge_masks
        m_ex = jnp.maximum(m_ex, sx.astype(m_ex.dtype))
        m_ey = jnp.maximum(m_ey, sy.astype(m_ey.dtype))
        m_ez = jnp.maximum(m_ez, sz.astype(m_ez.dtype))
    return state._replace(
        ex=state.ex * (1.0 - m_ex),
        ey=state.ey * (1.0 - m_ey),
        ez=state.ez * (1.0 - m_ez),
    )
