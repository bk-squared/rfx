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

def apply_kerr_ade(state, e_prev, chi3_arr, eps_r_arr):
    """Apply the instantaneous **reactive** Kerr correction to the E-field (#437).

    A Kerr χ³ nonlinearity is a REACTIVE, intensity-dependent permittivity
    (LOSSLESS — it changes the phase velocity, not the amplitude):

        ε_eff = ε_r + χ³·|E|²

    The linear E-update already produced ``E_lin = E^n + ΔE`` with
    ``ΔE = (dt/(ε0·ε_r))·(∇×H)``.  The ε_eff update is obtained algebraically by
    scaling the **increment** (not the whole field)::

        E^{n+1} = E^n + ΔE·ε_r/ε_eff = E^n + (E_lin − E^n) / (1 + χ³·|E^n|²/ε_r)

    Scaling the increment is what makes this reactive and lossless — the
    equilibrium field ``E^n`` is preserved and only the newly-integrated increment
    is slowed by the higher effective permittivity.  Scaling the whole field
    (``E → E/(1+factor)``, the pre-#437 behaviour) monotonically shrinks |E| and is
    a nonlinear **absorber**, not reactive χ³ (phase-neutral to first order — see
    docs/research_notes/2026-07-22_kerr_operator_defect.md).  The factor
    ``χ³·|E^n|²/ε_r`` is dimensionless and dt-independent, as a material index
    change must be.  ``|E^n|²`` is the co-located pre-update intensity (explicit
    ε_eff; suitable for weakly-to-moderately nonlinear media).

    Parameters
    ----------
    state : FDTDState
        State **after** the linear E-update (contains ``E_lin``).
    e_prev : tuple(jnp.ndarray, jnp.ndarray, jnp.ndarray)
        The E-field components (ex, ey, ez) **before** the linear update (``E^n``),
        used to form the increment and the pre-update intensity.
    chi3_arr : jnp.ndarray, shape (Nx, Ny, Nz)
        Third-order susceptibility at each cell (m^2/V^2).  Zero where linear;
        there the increment is scaled by 1 ⇒ the field is unchanged (byte-identical).
    eps_r_arr : jnp.ndarray, shape (Nx, Ny, Nz)
        Relative permittivity per cell (ε_r ≥ 1) — the ε_eff denominator.

    Returns
    -------
    FDTDState with the reactive-Kerr-corrected E-field components.
    """
    ex_p, ey_p, ez_p = e_prev
    # |E^n|^2 (co-located, pre-update) so ε_eff is explicit in the increment scale.
    e_sq = ex_p ** 2 + ey_p ** 2 + ez_p ** 2
    denom = 1.0 + chi3_arr * e_sq / eps_r_arr        # = ε_eff / ε_r  (>= 1)
    ex = ex_p + (state.ex - ex_p) / denom
    ey = ey_p + (state.ey - ey_p) / denom
    ez = ez_p + (state.ez - ez_p) / denom

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
