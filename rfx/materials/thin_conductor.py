"""Thin conductor subcell model.

Models conductors thinner than the grid cell size by modifying the
effective conductivity of the cells they occupy, without requiring
subcell grid refinement.

For a conductor of thickness t and bulk conductivity σ_bulk occupying
a cell of size Δx:
    σ_eff = σ_bulk · (t / Δx)

This preserves the correct sheet resistance R_s = 1/(σ_bulk · t)
while keeping the standard Yee cell size.

References:
    Taflove & Hagness, Ch. 10 — Subcell modeling techniques
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import MaterialArrays
from rfx.geometry.csg import Shape


@dataclass(frozen=True)
class ThinConductor:
    """Thin conductor specification.

    Parameters
    ----------
    shape : Shape
        Geometric region of the thin conductor.
    sigma_bulk : float
        Bulk conductivity (S/m).  e.g. copper = 5.8e7
    thickness : float
        Physical thickness in metres.
    eps_r : float
        Relative permittivity (default 1.0).
    """
    shape: Shape
    sigma_bulk: float
    thickness: float
    eps_r: float = 1.0


def apply_thin_conductor(
    grid: Grid,
    conductor: ThinConductor,
    materials: MaterialArrays,
) -> MaterialArrays:
    """Apply thin conductor subcell correction to material arrays.

    Parameters
    ----------
    grid : Grid
    conductor : ThinConductor
    materials : MaterialArrays
        Existing material arrays to modify.

    Returns
    -------
    MaterialArrays with corrected conductivity in the conductor region.
    """
    mask = conductor.shape.mask(grid)
    sigma_eff = conductor.sigma_bulk * (conductor.thickness / grid.dx)

    eps_r = jnp.where(mask, conductor.eps_r, materials.eps_r)
    sigma = jnp.where(mask, sigma_eff, materials.sigma)

    return MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=materials.mu_r)
