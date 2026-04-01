"""Kerr nonlinear material for FDTD.

The Kerr effect: eps_r depends on E-field intensity:
  eps_r(E) = eps_r_linear + chi3 * |E|^2

Implemented as an auxiliary update applied after each E-field update.
All operations use JAX for differentiability.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class KerrMaterial(NamedTuple):
    """Kerr nonlinear material specification.

    Parameters
    ----------
    eps_r_linear : float
        Linear (low-field) relative permittivity.
    chi3 : float
        Third-order susceptibility in m^2/V^2.
    """
    eps_r_linear: float
    chi3: float


def apply_kerr_update(materials, state, kerr_regions):
    """Update eps_r based on instantaneous E-field intensity.

    Uses cell-center averaged |E|^2 to account for Yee grid staggering:
    each component is averaged to the cell center before squaring.

    Parameters
    ----------
    materials : MaterialArrays
    state : FDTDState
    kerr_regions : list of (mask_array, KerrMaterial)
        Each entry is a boolean mask and Kerr material spec.

    Returns
    -------
    Updated MaterialArrays with modified eps_r.

    Note
    ----
    This is an explicit (forward Euler) linearization: eps_r from the
    current E-field is used for the next E-update. Stable only when
    chi3 * |E_max|^2 << eps_r_linear. For strongly nonlinear regimes,
    a predictor-corrector scheme would be needed.
    """
    eps_r = materials.eps_r

    # Average E-components to cell centers for correct staggered-grid intensity
    # Ex at (i+1/2, j, k) → average in x
    ex_avg = 0.5 * (state.ex[:-1, :, :] + state.ex[1:, :, :])
    # Ey at (i, j+1/2, k) → average in y
    ey_avg = 0.5 * (state.ey[:, :-1, :] + state.ey[:, 1:, :])
    # Ez at (i, j, k+1/2) → average in z
    ez_avg = 0.5 * (state.ez[:, :, :-1] + state.ez[:, :, 1:])

    # Compute |E|^2 at cell centers (interior cells)
    nx, ny, nz = eps_r.shape
    sx = min(ex_avg.shape[0], nx)
    sy = min(ey_avg.shape[1], ny)
    sz = min(ez_avg.shape[2], nz)

    e_sq = jnp.zeros_like(eps_r)
    e_sq = e_sq.at[:sx, :sy, :sz].set(
        ex_avg[:sx, :sy, :sz] ** 2 +
        ey_avg[:sx, :sy, :sz] ** 2 +
        ez_avg[:sx, :sy, :sz] ** 2
    )

    for mask, kerr in kerr_regions:
        eps_nonlinear = kerr.eps_r_linear + kerr.chi3 * e_sq
        eps_r = jnp.where(mask, eps_nonlinear, eps_r)

    return materials._replace(eps_r=eps_r)
