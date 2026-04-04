"""Kerr nonlinear material for FDTD.

The Kerr effect: eps_r depends on E-field intensity:
  eps_r(E) = eps_r_linear + chi3 * |E|^2

Two implementations:

1. ``apply_kerr_update`` — modifies eps_r per timestep (legacy).
2. ``apply_kerr_ade`` — ADE polarisation-current correction applied
   directly to the E-field after the standard linear update.  This is
   the recommended approach for integration into the ``jax.lax.scan``
   time loop because it avoids mutating material arrays.

All operations use JAX for differentiability.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from rfx.core.yee import EPS_0


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


# ---------------------------------------------------------------------------
# ADE (Auxiliary Differential Equation) Kerr correction
# ---------------------------------------------------------------------------

def apply_kerr_ade(state, chi3_arr, dt):
    """Apply Kerr nonlinear correction to E-field (ADE formulation).

    After the standard linear E-update the nonlinear polarisation
    current introduces an additional term:

        P_NL = eps0 * chi3 * |E|^2 * E

    In discrete form the correction is:

        E^{n+1} -= (dt / eps0) * chi3 * |E^n|^2 * E^{n+1}

    where ``|E^n|^2 = Ex^2 + Ey^2 + Ez^2`` evaluated component-wise
    (co-located approximation suitable for weakly nonlinear media).

    Parameters
    ----------
    state : FDTDState
        State **after** the standard E-update (contains E^{n+1}).
        The previous-step |E^n|^2 is approximated by the updated
        field for an explicit scheme.
    chi3_arr : jnp.ndarray, shape (Nx, Ny, Nz)
        Third-order susceptibility at each cell (m^2/V^2).
        Zero where the medium is linear.
    dt : float
        Timestep in seconds.

    Returns
    -------
    FDTDState with corrected E-field components.
    """
    # |E|^2 from the just-updated field (explicit forward-Euler)
    e_sq = state.ex ** 2 + state.ey ** 2 + state.ez ** 2

    # Correction factor: dt * chi3 / eps0 * |E|^2
    # Negative sign because P_NL opposes field growth.
    factor = (dt / EPS_0) * chi3_arr * e_sq

    ex = state.ex - factor * state.ex
    ey = state.ey - factor * state.ey
    ez = state.ez - factor * state.ez

    return state._replace(ex=ex, ey=ey, ez=ez)


# ---------------------------------------------------------------------------
# Legacy eps_r-based Kerr update
# ---------------------------------------------------------------------------

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
