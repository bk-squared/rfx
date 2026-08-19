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
        # the local normal spacing (Grid is uniform on this lane).
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
