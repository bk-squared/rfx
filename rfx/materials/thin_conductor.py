"""Thin conductor subcell model.

Models conductors thinner than the grid cell size by modifying the
effective conductivity of the cells they occupy, without requiring
subcell grid refinement.

For a conductor of thickness t and bulk conductivity σ_bulk occupying
a cell of size Δx:
    σ_eff = σ_bulk · (t / Δx)

This preserves the correct sheet resistance R_s = 1/(σ_bulk · t)
while keeping the standard Yee cell size.

For PEC thin sheets (σ_bulk → ∞), the cells containing the sheet are
added to the PEC mask directly.  This implements the Thin Sheet Technique
(TST) from CST's Perfect Boundary Approximation, without requiring the
mesh to resolve the sheet volumetrically.

References:
    Taflove & Hagness, Ch. 10 — Subcell modeling techniques
    Weiland, AEÜ 31(3), 1977 — Thin sheet technique

MODEL CONTRACT (#711, measured 2026-08-26)
------------------------------------------
The lossy path is a PENETRABLE resistive film: one cell of folded volumetric
conductivity transmits, reflects and absorbs per the exact conductive-slab
solution (pinned by tests/oracle/test_sheet_film_rta_analytic.py: at Rs = eta0/2,
measured band-mean R/T land within 0.01 of the analytic 0.25/0.25). It is NOT
an opaque Leontovich boundary. The ``surface_impedance_f0`` path feeds this
same penetrable update a thick-conductor Leontovich Rs — a valid loss model
only while thickness >> skin depth, where the film's residual transmission
(~(2 Rs/eta0)^2, -70 dB class for copper foils) is negligible. The PEC branch
(sigma_bulk >= 1e6) is exact and opaque (pinned: T < 1e-3 in the same test).

CPML composition: the sheet update REPLACES the CPML-corrected tangential
update on edges it owns (runner order: bulk E -> CPML -> PEC -> sheet), so a
sheet inside the absorber would silently drop the stretching correction —
placement there is refused/flagged by preflight P1.9 (geometry-in-CPML).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import EPS_0, MU_0, MaterialArrays
from rfx.geometry.csg import Shape

# Threshold above which a thin conductor is treated as PEC sheet.
#
# This is a SPEC-LEVEL routing predicate on ThinConductor.sigma_bulk /
# MaterialSpec.sigma — NEVER compare assembled sigma ARRAYS against this
# threshold: a surface-impedance (Leontovich) sheet's folded sigma_eff can
# legally exceed it (sigma_eff = 1/(Rs0*d_norm) is unbounded as the mesh
# refines) while remaining a lossy sheet, not a PEC.
_PEC_SIGMA_THRESHOLD = 1e6


def leontovich_rs(f0, sigma_bulk):
    """Band-centre Leontovich surface resistance (ohms per square).

    Rs(f0) = sqrt(pi * f0 * mu0 / sigma_bulk) — the surface resistance of a
    good conductor much thicker than its skin depth, evaluated at the single
    frequency ``f0`` and held frequency-flat within a run (relative band
    error |sqrt(f/f0) - 1|). jnp-based so gradients flow through both
    ``f0`` and ``sigma_bulk``.
    """
    return jnp.sqrt(jnp.pi * jnp.asarray(f0) * MU_0 / sigma_bulk)


def sheet_bounds(shape):
    """Axis-aligned bounds of a thin-conductor shape, or ``(None, None)``.

    The ``Box`` fields are read FIRST so a Box takes bit-identically the same
    arithmetic it took before non-Box sheets were allowed (issue #674);
    everything else falls back to the ``Shape`` protocol's
    :meth:`~rfx.geometry.csg.Shape.bounding_box`. ``Box.bounding_box()``
    returns exactly ``(corner_lo, corner_hi)``, so the two routes agree there
    by construction — the ordering is about provable identity, not about
    disagreeing values.

    A shape that offers neither returns ``(None, None)``; the caller decides
    whether that is a skip (legacy DC fold) or a hard error (f0 mode).
    """
    lo = getattr(shape, "corner_lo", None)
    hi = getattr(shape, "corner_hi", None)
    if lo is not None and hi is not None:
        return lo, hi
    bbox = getattr(shape, "bounding_box", None)
    if bbox is None:
        return None, None
    try:
        lo, hi = bbox()
    except NotImplementedError:
        return None, None
    return lo, hi


def sheet_normal_axis(lo, hi) -> int:
    """Index (0/1/2) of the thinnest bounding-box axis — the sheet normal."""
    extents = [float(hi[i]) - float(lo[i]) for i in range(3)]
    return min(range(3), key=lambda i: extents[i])


def check_sheet_occupancy(mask, n_axis: int, *, lane: str) -> None:
    """Guard a surface-impedance sheet's RASTERIZED occupancy (issue #674).

    ``sigma_eff = 1/(Rs0*d_norm)`` is a per-occupied-cell fold and is
    shape-agnostic, which is what lets an arbitrary ``mask_on_coords`` shape
    replace the Box the #669 contract scoped it to. What is NOT shape-agnostic
    is the assumption behind the normalization: the sheet is realized on ONE E
    node along its normal. Two ways a non-Box shape breaks that, both silent
    without this check:

    - **zero cells** — a mesh slab thinner than a cell registers only where a
      grid node falls inside it, so a 35 um sheet that misses every node plane
      vaporizes (the #369 silently-vaporized-metal class, which is the failure
      the #669 Box guard was standing in for);
    - **more than one layer** — a 3-D body with height (an imported solid, an
      L-bracket, a slab thicker than a cell) would fold the sheet conductance
      once PER layer, multiplying the loss by the layer count while reporting
      the requested ``Rs0``.

    Concrete masks only: a traced mask (differentiable-mesh path) skips the
    check rather than raising a ``ConcretizationTypeError``. The reduction runs
    on-device first so a large grid pays one small host transfer, not a copy of
    the whole boolean field.
    """
    from rfx.core.jax_utils import is_tracer

    if is_tracer(mask) or getattr(mask, "ndim", 0) != 3:
        return
    other = tuple(a for a in range(3) if a != n_axis)
    occupied = np.asarray(jnp.any(mask, axis=other))
    layers = np.flatnonzero(occupied)
    axis_name = "xyz"[n_axis]
    if layers.size == 0:
        raise ValueError(
            f"add_thin_conductor(surface_impedance_f0=...): the sheet shape "
            f"rasterizes to ZERO cells on this grid ({lane} lane), so it would "
            f"silently vanish instead of loading the field (issue #369 class). "
            f"A slab thinner than one cell only registers where a grid node "
            f"falls inside it: centre it on a node plane, thicken it to at "
            f"least one cell, or draw it as a Box with equal lo/hi on its "
            f"normal axis (that snaps to the nearest node by construction).")
    if layers.size > 1:
        # If a DIFFERENT axis does resolve to one layer, the shape is probably
        # a flat sheet whose bounding box simply does not name its normal
        # (a footprint narrower than the thickness, or a cubic bounding box).
        # Saying which axis turns a confusing refusal into an actionable one.
        flat_axes = [
            "xyz"[a] for a in range(3) if a != n_axis
            and int(np.count_nonzero(np.asarray(jnp.any(
                mask, axis=tuple(b for b in range(3) if b != a))))) == 1]
        hint = (
            f" (The sheet normal is taken from the THINNEST bounding-box axis, "
            f"which is {axis_name}; the rasterized body does span exactly one "
            f"layer along {'/'.join(flat_axes)}, so if that is the intended "
            f"normal, redraw the sheet so its thickness is its smallest "
            f"bounding-box extent.)" if flat_axes else "")
        raise ValueError(
            f"add_thin_conductor(surface_impedance_f0=...): the sheet shape "
            f"rasterizes to {layers.size} cell layers along its normal "
            f"(axis {axis_name}, indices {int(layers[0])}..{int(layers[-1])}) "
            f"on the {lane} lane.{hint} A Leontovich surface-impedance sheet is "
            f"realized on ONE E node along its normal — sigma_eff = "
            f"1/(Rs0*d_norm) is the sheet conductance of a single node — so a "
            f"body with HEIGHT (an imported 3-D solid, a bent/L-shaped sheet, "
            f"a slab thicker than a cell) is not a sheet: folding it per cell "
            f"would multiply the sheet conductance by the layer count while "
            f"still reporting Rs0. Flat, sub-cell-thick sheets of any FOOTPRINT "
            f"(patterned planes, clearance holes, imported mesh outlines) are "
            f"supported; conformal and volumetric conductors are not.")


@dataclass(frozen=True)
class ThinConductor:
    """Thin conductor specification.

    Parameters
    ----------
    shape : Shape
        Geometric region of the thin conductor.
    sigma_bulk : float
        Bulk conductivity (S/m).  e.g. copper = 5.8e7.
        When >= 1e6, treated as PEC thin sheet (added to PEC mask).
    thickness : float
        Physical thickness in metres.
    eps_r : float
        Relative permittivity (default 1.0).
    surface_impedance_f0 : float | Array | None
        Band-centre frequency (Hz) for the opt-in Leontovich surface-
        impedance loss model (issue #669). ``None`` (default) keeps the
        exact legacy semantics: metal -> lossless PEC sheet, sub-threshold
        -> DC lossy fold ``sigma_bulk*t/d``. When set, the sheet is a
        resistive sheet of sheet resistance ``Rs0 = leontovich_rs(f0,
        sigma_bulk)`` for ANY ``sigma_bulk > 0``, realized as
        ``sigma_eff = 1/(Rs0*d_norm)``.
    """
    shape: Shape
    sigma_bulk: float
    thickness: float
    eps_r: float = 1.0
    surface_impedance_f0: float | jnp.ndarray | None = None

    @property
    def is_pec(self) -> bool:
        """Whether this thin conductor should be treated as PEC.

        ORDER IS LOAD-BEARING: the ``None``-check short-circuits BEFORE the
        float compare, so a traced (JAX tracer) ``sigma_bulk`` never reaches
        ``>=`` when ``surface_impedance_f0`` is set — that is what makes
        ``sigma_bulk`` a legal differentiable DoF in f0 mode. Pinned by
        tests/unit/materials/test_thin_conductor.py (is_pec-order pin).
        """
        return (self.surface_impedance_f0 is None) and (
            self.sigma_bulk >= _PEC_SIGMA_THRESHOLD)

    @property
    def r_s_leontovich(self):
        """Band-centre surface resistance Rs0 (ohms/sq); f0 mode only."""
        if self.surface_impedance_f0 is None:
            raise ValueError(
                "r_s_leontovich requires surface_impedance_f0 to be set")
        return leontovich_rs(self.surface_impedance_f0, self.sigma_bulk)

    @property
    def sheet_resistance(self) -> float:
        """Sheet resistance R_s = 1/(σ·t) in Ω/sq."""
        return 1.0 / (self.sigma_bulk * self.thickness)


def apply_thin_conductor(
    grid: Grid,
    conductor: ThinConductor,
    materials: MaterialArrays,
    pec_mask: jnp.ndarray | None = None,
    sheet_specs: list | None = None,
) -> tuple[MaterialArrays, jnp.ndarray | None]:
    """Apply thin conductor subcell correction to material arrays.

    For lossy conductors (σ < 1e6): modifies σ_eff in the material arrays.
    For PEC thin sheets (σ >= 1e6): adds cells to PEC mask instead.

    Parameters
    ----------
    grid : Grid
    conductor : ThinConductor
    materials : MaterialArrays
    pec_mask : bool array or None
        Existing PEC mask.  Updated in-place for PEC thin sheets.
    sheet_specs : list or None
        Collector for surface-impedance (``surface_impedance_f0``) sheets
        (#677). An f0-mode conductor no longer touches the material arrays;
        when a list is passed, a :class:`SheetImpedanceSpec` is appended to
        it for the caller to assemble into a runtime
        :class:`SheetImpedanceCtx`. When ``None`` (legacy callers), the f0
        branch still validates the sheet but emits nothing — the assembled
        arrays are sheet-free either way, and lanes that cannot apply the
        ctx must refuse f0 sheets at their entry point.

    Returns
    -------
    (materials, pec_mask) — updated material arrays and PEC mask.
    """
    mask = conductor.shape.mask(grid)

    if conductor.is_pec:
        # P4: Thin PEC sheet — add to PEC mask, no volumetric meshing needed
        if pec_mask is None:
            pec_mask = mask
        else:
            pec_mask = pec_mask | mask
        return materials, pec_mask

    if conductor.surface_impedance_f0 is not None:
        # Leontovich (band-centre) surface-impedance mode (issues #669/#677):
        # the sheet is a resistive sheet of sheet resistance Rs0 =
        # sqrt(pi*f0*mu0/sigma_bulk). Since #677 it is NOT folded into
        # ``materials.sigma`` (that realized the sheet as a full-cell slab —
        # the conductor surfaces moved half a cell each side and toggling f0
        # moved resonances by GEOMETRY — measured shifts of 1.01 and
        # 3.22 GHz on the #677 coupled-resonator A/B). Instead it is
        # emitted as a :class:`SheetImpedanceSpec` and realized NODE-THIN by
        # the per-step operator :func:`apply_sheet_impedance_e` on exactly
        # the tangential E edges ``apply_pec_mask`` would zero — for a
        # SINGLE sheet the PEC and f0 footprints are structurally identical,
        # so the f0 toggle changes loss, never geometry. That identity is
        # per sheet, not per stack: for two sheets on ADJACENT layers the f0
        # path leaves the vacuum gap edge lossless (#690) while the PEC path
        # still fuses the films, because ``pec_mask`` is a cell mask with no
        # normal attached. ``eps_r`` is deliberately left at the background
        # value (the sheet is a surface, not a dielectric fill).
        #
        # Thickness deliberately does NOT enter: Leontovich loss for a
        # conductor much thicker than its skin depth is thickness-
        # independent. ``sigma_sheet = G/d_dual`` with G = 1/Rs0 keeps
        # ``sigma_sheet * d_dual == 1/Rs0`` exactly per node (the O2 tooth;
        # grid.dx is both primal and dual on this uniform cubic lane).
        #
        # #674: any ``mask_on_coords`` shape may deliver ``mask``; it must
        # rasterize to ONE cell layer along its normal
        # (``check_sheet_occupancy``).
        lo, hi = sheet_bounds(conductor.shape)
        if lo is None or hi is None:
            raise ValueError(
                "surface-impedance (surface_impedance_f0) thin conductor "
                "requires a shape with an axis-aligned bounding box (Box "
                "corner_lo/corner_hi, or Shape.bounding_box()) — the sheet "
                "normal is read from it; refusing to fold it blind.")
        n_axis = sheet_normal_axis(lo, hi)
        check_sheet_occupancy(mask, n_axis, lane="uniform")
        g_sheet = 1.0 / leontovich_rs(conductor.surface_impedance_f0,
                                      conductor.sigma_bulk)
        if sheet_specs is not None:
            sigma_sheet = jnp.where(mask, g_sheet / grid.dx, 0.0)
            sheet_specs.append(SheetImpedanceSpec(
                mask=mask, normal_axis=n_axis, g_sheet=g_sheet,
                sigma_sheet=sigma_sheet))
        # Materials are returned UNCHANGED: assembled arrays are sheet-free
        # by design since #677. A caller that runs the fields without
        # applying the sheet ctx must refuse f0 sheets at its entry point
        # (see Simulation lanes) — the assembly itself is not the fence.
        return materials, pec_mask

    # Lossy thin conductor (DC fold): effective conductivity preserves
    # the DC sheet resistance R_s = 1/(sigma_bulk*t).
    sigma_eff = conductor.sigma_bulk * (conductor.thickness / grid.dx)

    eps_r = jnp.where(mask, conductor.eps_r, materials.eps_r)
    sigma = jnp.where(mask, sigma_eff, materials.sigma)

    return MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=materials.mu_r), pec_mask


# ---------------------------------------------------------------------------
# Surface-impedance sheet operator (issue #677, Design B)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SheetImpedanceSpec:
    """One rasterized surface-impedance (f0) sheet.

    Emitted by :func:`apply_thin_conductor` (uniform lane) and
    ``rfx.runners.nonuniform.assemble_materials_nu`` (NU lane) instead of a
    ``materials.sigma`` fold (#677). ``sigma_sheet`` is the per-node sheet
    conductivity ``G/d_dual`` (S/m) on the sheet cells and 0 elsewhere, with
    ``d_dual`` the E-node DUAL spacing along the sheet normal (== grid.dx on
    the uniform cubic lane; ``e_node_dual_spacings`` per node on NU — the
    #671 invariant, so ``sigma_sheet * Rs0 * d_dual == 1`` per node).
    ``g_sheet = 1/Rs0`` stays AD-live in ``f0`` and ``sigma_bulk``
    (d(1/Rs0)/df0 = -(1/Rs0)/(2 f0); d(1/Rs0)/dsigma_bulk =
    +(1/Rs0)/(2 sigma_bulk) — the #669 closed forms with flipped signs,
    since G = 1/Rs0).
    """
    mask: object          # (nx, ny, nz) bool — one-layer cell mask
    normal_axis: int      # 0/1/2
    g_sheet: object       # scalar sheet conductance per square, 1/Rs0 (S)
    sigma_sheet: object   # (nx, ny, nz) float — G/d_dual at sheet cells


@dataclass(frozen=True)
class SheetImpedanceCtx:
    """Assembled runtime context for all f0 sheets of one run.

    ``mask_ex/ey/ez`` are the TANGENTIAL edge masks from the shared
    ``rfx.boundaries.pec.tangential_edge_masks`` neighbor rule — the same
    edges ``apply_pec_mask`` would zero for the same cell mask (G4
    footprint identity), minus any edge the run's PEC mask already owns
    (PEC wins on overlap), and minus any component that is NORMAL to every
    sheet covering that cell.

    That last veto is what keeps the sheet-normal component all-False
    (#690). It used to be described as holding "by construction" for a
    one-layer sheet, and that was wrong: the neighbour rule sees only
    adjacency, so two one-layer sheets sharing a normal and sitting on
    ADJACENT cell layers looked like one two-cell body and the gap edge
    between them was loaded. Since #677 an f0 sheet is node-thin, so
    adjacent layers are two films one dual spacing apart with vacuum
    between — the gap edge must stay lossless. The veto uses each spec's
    declared ``normal_axis``, so it holds for a stack of any depth.

    Consequence worth knowing (and NOT re-litigating silently): for
    adjacent stacked sheets the f0 footprint now differs from the PEC
    thin-sheet footprint, which still fuses the two films because
    ``pec_mask`` is a cell mask with no normal attached. The #677
    PEC/f0 footprint identity holds per sheet, not for a stack.
    """
    mask_ex: object
    mask_ey: object
    mask_ez: object
    sigma_sheet: object   # accumulated over sheets; 0 off-sheet


def build_sheet_impedance_ctx(sheet_specs, pec_mask=None,
                             periodic=(False, False, False)):
    """Union a run's :class:`SheetImpedanceSpec` list into a runtime ctx.

    Refuses crossing/overlapping sheets with DIFFERENT normals (their
    edge-set union is not a sheet operator's contract — two planes crossing
    share edges whose loss would double-count with no defined normal).
    Same-normal OVERLAP (two sheets on the very same cells) ADDS
    conductance — parallel sheet admittances on one node, physical.
    Same-normal ADJACENCY (two one-layer sheets on neighbouring layers) is
    two independent films with a vacuum gap, so the sheet-normal edge
    between them is NOT loaded (#690); each component is kept only where
    some covering sheet has it tangential. PEC-owned edges are excluded so
    ``apply_pec_mask`` (which runs first) wins on overlap.

    That per-component veto is skipped on a LENGTH-1 axis, where "normal to
    the sheet" is not a direction the grid can represent — see the loop
    below. Note the exception is the degenerate axis ONLY, not
    ``periodic[c]``: on a periodic axis of length > 1 cells 0 and n-1 are
    genuinely different cells, so two one-layer films sitting either side of
    the seam are still two films and the edge between them is still their
    gap. Widening the exception to ``periodic[c]`` would re-open #690 across
    the seam; ``test_periodic_seam_gap_edge_stays_vetoed`` pins that.

    ``periodic`` is forwarded to ``tangential_edge_masks`` for BOTH the
    sheet union and the PEC subtraction (#689). Handing the two different
    flags would compute the #677 G4 footprint identity against two
    different neighbour rules; the caller must pass the same flags its
    ``apply_pec_mask`` gets.

    Returns ``None`` for an empty spec list.
    """
    from rfx.boundaries.pec import tangential_edge_masks
    from rfx.core.jax_utils import is_tracer

    specs = list(sheet_specs or ())
    if not specs:
        return None
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            a, b = specs[i], specs[j]
            if a.normal_axis == b.normal_axis:
                continue
            inter = a.mask & b.mask
            if not (is_tracer(inter) or getattr(inter, "ndim", 0) != 3):
                if bool(jnp.any(inter)):
                    raise ValueError(
                        "surface-impedance sheets with DIFFERENT normal axes "
                        f"('{'xyz'[a.normal_axis]}' and "
                        f"'{'xyz'[b.normal_axis]}') overlap on the grid. "
                        "Crossing f0 sheets share edges whose sheet normal "
                        "is undefined, so the operator refuses rather than "
                        "double-counting their loss. Split the geometry so "
                        "f0 sheets of different orientation do not share "
                        "cells (PEC sheets may still cross freely).")
    # #690: ``tangential_edge_masks`` classifies by ADJACENCY alone, so two
    # sheets that share a normal axis and land on ADJACENT cell layers look
    # like one two-cell body and the SHEET-NORMAL edge between them picks up
    # the resistive update.  That edge is the dielectric gap of a two-layer
    # board: since #677 an f0 sheet is node-thin, so adjacent layers are two
    # films one dual spacing apart, not one 2-cell slab.  Each spec carries
    # its own normal, so veto any component that is normal to EVERY sheet
    # covering that cell.  Grouping the masks by normal axis first keeps the
    # cost at n_specs cheap boolean ORs plus ONE classification.
    per_axis = [None, None, None]
    sigma_sheet = specs[0].sigma_sheet
    for k, sp in enumerate(specs):
        a = int(sp.normal_axis)
        per_axis[a] = sp.mask if per_axis[a] is None else (per_axis[a] | sp.mask)
        if k:
            # Coincident same-normal sheets are PARALLEL sheet admittances
            # (1/Rs_tot = 1/Rs_1 + 1/Rs_2); ``sigma_sheet = G/d_dual`` is
            # linear in G and both share the node's d_dual, so the per-node
            # sum IS that parallel combination.  Disjoint cells never
            # interact, so there the sum is only an assignment.  Summed in
            # the original spec order so the float result is unchanged.
            sigma_sheet = sigma_sheet + sp.sigma_sheet
    union = None
    for m in per_axis:
        if m is not None:
            union = m if union is None else (union | m)
    masks = list(tangential_edge_masks(union, periodic))
    for c in range(3):
        if union.shape[c] == 1:
            # DEGENERATE AXIS — leave the classification alone.  With one
            # cell along ``c`` there is no "normal to c" direction to be
            # normal to: every body spans the whole axis and is its own
            # neighbour through the wrap ``tangential_edge_masks``
            # deliberately keeps on a length-1 axis.  ``sheet_normal_axis``
            # still names ``c`` the thinnest bounding-box axis, so on the
            # 2-D lane (nz == 1) a flat patch reports ``normal_axis = 2``
            # and the veto below would zero ``mask_ez`` — the ONLY live E
            # component in ``mode="2d_tmz"``, making the sheet
            # bit-identically inert (measured on a 6x6-cell copper patch,
            # 20x20x1 grid: f0/none peak|Ez| = 1.000000 at the patch node
            # against pec/none = 0.000000).  Skipped whole, not narrowed to
            # ``masks[c] & allow``: with a second sheet of a different
            # normal present, that AND would drop this sheet's own edges.
            continue
        # ``allow``: cells covered by at least one sheet for which component
        # ``c`` is TANGENTIAL.  A component normal to every covering sheet is
        # a gap edge, not a sheet edge.
        allow = None
        for a in range(3):
            if a != c and per_axis[a] is not None:
                allow = per_axis[a] if allow is None else (allow | per_axis[a])
        masks[c] = (masks[c] & allow) if allow is not None \
            else jnp.zeros_like(masks[c])
    mask_ex, mask_ey, mask_ez = masks
    if pec_mask is not None:
        pex, pey, pez = tangential_edge_masks(pec_mask, periodic)
        mask_ex = mask_ex & ~pex
        mask_ey = mask_ey & ~pey
        mask_ez = mask_ez & ~pez
    return SheetImpedanceCtx(mask_ex=mask_ex, mask_ey=mask_ey,
                             mask_ez=mask_ez, sigma_sheet=sigma_sheet)


def sheet_update_coeffs(sigma_sheet, materials, dt):
    """Holland exponential-stepping E-update coefficients (A, B) (#677).

    Exact per-step integration of the Ampere ODE at a sheet edge,
    ``eps dE/dt + sigma_tot E = curlH`` with ``sigma_tot = sigma_bg +
    sigma_sheet`` held constant over the step::

        x2 = sigma_tot * dt / (eps0 * eps_r)
        A  = exp(-x2)
        B  = -expm1(-x2) / sigma_tot      (== (1 - exp(-x2)) / sigma_tot)

    Sign note (the #677 contract's documented flips): ``-expm1(-x2)`` is
    the POSITIVE quantity ``1 - exp(-x2)``; ``expm1`` is used so B keeps
    full precision at small x2 (B -> dt/eps as sigma_tot -> 0, matching
    the lossless update), while at large x2 the update reduces to
    ``E = curlH/sigma_tot`` — the resistive-sheet limit ``E_tan = Rs*Js``.
    Exact per-step decay at ANY sigma: no flip-mode (A > 0 always),
    unconditionally stable, unlike the semi-implicit (1-x)/(1+x) form
    whose ca goes negative past x = 1.

    Ambient precision (PROMOTE, never PIN): coefficients are computed in
    the dtype jnp promotes from the inputs; callers cast at apply time.
    ``eps_r`` is the BACKGROUND permittivity at the sheet cells (the f0
    branch no longer overwrites it, #677).
    """
    eps = materials.eps_r * EPS_0
    sigma_tot = materials.sigma + sigma_sheet
    x2 = sigma_tot * dt / eps
    a = jnp.exp(-x2)
    # Safe division: off-sheet vacuum cells have sigma_tot == 0; their
    # (unused there) B limit is dt/eps. The where-guard keeps forward and
    # AD free of 0/0.
    safe = jnp.where(sigma_tot > 0, sigma_tot, 1.0)
    b = jnp.where(sigma_tot > 0, -jnp.expm1(-x2) / safe, dt / eps)
    return a, b


def apply_sheet_impedance_e(state, e_prev, curls, ctx, coeffs):
    """Apply the node-thin resistive-sheet E update at tangential edges.

    Runs AFTER ``apply_pec_mask``/``apply_pec_occupancy`` and BEFORE
    S-parameter DFT sampling and source injection (same contract slot the
    PEC mask owns; see rfx/simulation.py make_core_step). At every masked
    tangential edge the standard E-update result is REPLACED by::

        E^{n+1} = A * E^n + B * curlH^{n+1/2}

    with ``curls`` computed by the SAME stencil helpers the kernels use
    (``rfx.core.yee.curl_h`` / ``curl_h_nu``) from the same H the E-update
    consumed. Everywhere off-mask the state passes through untouched; the
    sheet-normal edge is never touched (its mask is all-False for a
    one-layer sheet).

    Parameters
    ----------
    state : FDTDState after the standard E update + boundary operators.
    e_prev : (ex, ey, ez) tuple — the E fields BEFORE the E update (E^n).
    curls : (curl_x, curl_y, curl_z) from the shared curl helper.
    ctx : SheetImpedanceCtx
    coeffs : (A, B) from :func:`sheet_update_coeffs`.
    """
    a, b = coeffs
    _fdtype = state.ex.dtype
    _cdtype = jnp.promote_types(_fdtype, jnp.float32)
    ex = jnp.where(
        ctx.mask_ex,
        (a * e_prev[0].astype(_cdtype) + b * curls[0]).astype(_fdtype),
        state.ex)
    ey = jnp.where(
        ctx.mask_ey,
        (a * e_prev[1].astype(_cdtype) + b * curls[1]).astype(_fdtype),
        state.ey)
    ez = jnp.where(
        ctx.mask_ez,
        (a * e_prev[2].astype(_cdtype) + b * curls[2]).astype(_fdtype),
        state.ez)
    return state._replace(ex=ex, ey=ey, ez=ez)


def has_f0_sheets(thin_conductors) -> bool:
    """True when any registered thin conductor is in surface-impedance mode.

    The single predicate run lanes use to REFUSE f0 sheets they cannot
    apply (distributed, subgridded, ADI, ... — #677 G9). Since the sheet
    no longer rides ``materials.sigma``, a lane that ignores the ctx would
    silently simulate NO sheet at all (the #369 vaporized-metal class), so
    unsupported lanes must call this and raise loudly.
    """
    return any(
        getattr(tc, "surface_impedance_f0", None) is not None
        for tc in (thin_conductors or ()))


def refuse_f0_sheets(thin_conductors, lane: str) -> None:
    """Raise if ``thin_conductors`` contains an f0 sheet (#677 lane fence).

    ``lane`` must be UNIQUE per call site: it is what
    ``tests/unit/materials/test_sheet_lane_fences.py`` matches on to tell which of the
    fences fired, alongside a check on the raising frame. A new call site
    also needs a row in that file's ``FENCE_REGISTRY`` and a test entering
    its lane — the inventory guard there fails otherwise.
    """
    if has_f0_sheets(thin_conductors):
        raise ValueError(
            f"surface-impedance (surface_impedance_f0) thin conductors are "
            f"not supported on the {lane} lane: the sheet is realized by a "
            f"per-step node-thin operator (#677), which this lane does not "
            f"apply — running would silently simulate NO sheet at all. Use "
            f"the uniform or non-uniform run()/S-parameter lanes, or drop "
            f"surface_impedance_f0 (PEC or DC-fold sheets are fine here).")


# Default sigma floor for "this assembled cell is a conductor" (#695).
# Two decades BELOW ``_PEC_SIGMA_THRESHOLD`` on purpose: it is a test on
# assembled sigma ARRAYS (where a DC-fold sheet's sigma_eff or a bulk
# metal both land far above it) rather than the spec-level PEC routing
# predicate above, and ordinary lossy dielectrics (1e-3 - 1e0 S/m) must
# stay out.
CONDUCTOR_SIGMA_THRESHOLD = 1e3


def conductor_footprint(
    pec_mask=None,
    sigma=None,
    sheet_masks=(),
    *,
    sigma_threshold: float = CONDUCTOR_SIGMA_THRESHOLD,
    shape=None,
):
    """Union the three places a conductor can live into ONE cell mask (#695).

    Since #677 a surface-impedance (``surface_impedance_f0``) thin conductor
    is a node-thin per-step operator: it is absent from ``pec_mask`` AND
    contributes nothing to ``materials.sigma``.  The obvious hand-written
    conductor test ``pec_mask | (sigma > 1e3)`` therefore finds NOTHING on a
    board whose traces are all f0 sheets, and any check built on it reports
    a healthy model as having no metal.  This is the one spelling::

        pec_mask | (sigma > sigma_threshold) | union(sheet_masks)

    Every argument is optional so callers can pass only what they hold:
    ``pec_mask`` (bool array or None), ``sigma`` (float array or None), and
    ``sheet_masks`` (iterable of bool cell masks — ``SheetImpedanceSpec.mask``
    values, or ``ctx.sigma_sheet > 0`` when only the assembled ctx is in
    hand).  ``shape`` seeds an all-False result when every input is None.

    Raises ``ValueError`` when nothing at all was supplied and no ``shape``
    is given — returning a scalar False there would be the same silent
    "found no metal" the function exists to prevent.
    """
    out = None
    if sigma is not None:
        out = jnp.asarray(sigma) > float(sigma_threshold)
    if pec_mask is not None:
        pm = jnp.asarray(pec_mask).astype(bool)
        out = pm if out is None else (out | pm)
    for m in (sheet_masks or ()):
        if m is None:
            continue
        sm = jnp.asarray(m).astype(bool)
        out = sm if out is None else (out | sm)
    if out is None:
        if shape is None:
            raise ValueError(
                "conductor_footprint: nothing to union (pec_mask, sigma and "
                "sheet_masks are all None/empty) and no `shape` was given to "
                "seed an all-False mask. Refusing to invent a conductor-free "
                "answer the caller cannot distinguish from 'I was handed "
                "nothing' (#695).")
        return jnp.zeros(tuple(shape), dtype=bool)
    return out
