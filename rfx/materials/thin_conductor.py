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
from rfx.core.yee import MaterialArrays
from rfx.geometry.csg import Shape

# Threshold above which a thin conductor is treated as PEC sheet
_PEC_SIGMA_THRESHOLD = 1e6


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
    """
    shape: Shape
    sigma_bulk: float
    thickness: float
    eps_r: float = 1.0

    @property
    def is_pec(self) -> bool:
        """Whether this thin conductor should be treated as PEC."""
        return self.sigma_bulk >= _PEC_SIGMA_THRESHOLD

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

    # Lossy thin conductor: effective conductivity preserves sheet resistance
    sigma_eff = conductor.sigma_bulk * (conductor.thickness / grid.dx)

    eps_r = jnp.where(mask, conductor.eps_r, materials.eps_r)
    sigma = jnp.where(mask, sigma_eff, materials.sigma)

    return MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=materials.mu_r), pec_mask
