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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import MU_0, MaterialArrays
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
        tests/test_thin_conductor.py (is_pec-order pin).
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
        # Leontovich (band-centre) surface-impedance mode (issue #669): the
        # sheet is a resistive sheet of sheet resistance Rs0 = sqrt(pi*f0*
        # mu0/sigma_bulk), realized through the same lossy fold as
        # sigma_eff = 1/(Rs0 * d_norm) so that sigma_eff*d_norm = 1/Rs0
        # exactly. Thickness deliberately does NOT enter: Leontovich loss
        # for a conductor much thicker than its skin depth is thickness-
        # independent (that is the honest physics, not a gap). grid.dx is
        # the local normal spacing (Grid is uniform AND cubic on this lane,
        # so the normal axis does not enter the arithmetic here — it is read
        # only to guard the rasterized occupancy).
        #
        # #674: the fold itself is per-occupied-cell and shape-agnostic, so
        # any ``mask_on_coords`` shape may deliver ``mask``. What the shape
        # must still satisfy is that it rasterizes to ONE cell layer along its
        # normal; ``check_sheet_occupancy`` is the guard, and it is what the
        # #669 Box-only restriction was standing in for.
        lo, hi = sheet_bounds(conductor.shape)
        if lo is None or hi is None:
            raise ValueError(
                "surface-impedance (surface_impedance_f0) thin conductor "
                "requires a shape with an axis-aligned bounding box (Box "
                "corner_lo/corner_hi, or Shape.bounding_box()) — the sheet "
                "normal is read from it; refusing to fold it blind.")
        check_sheet_occupancy(mask, sheet_normal_axis(lo, hi), lane="uniform")
        sigma_eff = 1.0 / (
            leontovich_rs(conductor.surface_impedance_f0,
                          conductor.sigma_bulk) * grid.dx)
    else:
        # Lossy thin conductor (DC fold): effective conductivity preserves
        # the DC sheet resistance R_s = 1/(sigma_bulk*t).
        sigma_eff = conductor.sigma_bulk * (conductor.thickness / grid.dx)

    eps_r = jnp.where(mask, conductor.eps_r, materials.eps_r)
    sigma = jnp.where(mask, sigma_eff, materials.sigma)

    return MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=materials.mu_r), pec_mask
