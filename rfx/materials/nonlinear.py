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

_KERR_FIXED_POINT_ITERS = 4   # fixed-point iterations for the reactive-Kerr constitutive solve


def apply_kerr_ade(state, e_prev, chi3_arr, eps_r_arr):
    """Apply the instantaneous **reactive** Kerr correction to the E-field (#437, #446).

    A Kerr χ³ nonlinearity is a REACTIVE, intensity-dependent permittivity
    (LOSSLESS — it changes the phase velocity, not the amplitude):

        D = ε0·(ε_r + χ³·|E|²)·E     ⇔     P_NL = ε0·χ³·|E|²·E

    This solves that constitutive relation SELF-CONSISTENTLY (the "D-based" update),
    which is the rigorous reactive Kerr.  The displacement is reconstructed from the
    pre-update field and advanced linearly, then E is recovered by solving the cubic::

        D_target/ε0 = ε_r·E_lin + χ³·|E^n|²·E^n           (advance D linearly)
        solve  ε_r·E^{n+1} + χ³·|E^{n+1}|²·E^{n+1} = D_target   (fixed-point)

    where ``E_lin`` is the linear (ε_r) E-update the scan already produced.  This is
    STATELESS — D^n is exactly reconstructable from E^n since E^n itself solved the
    constitutive relation — so no auxiliary D-field is carried.

    **Why the self-consistent solve, not the earlier increment scaling** (``E^n +
    ΔE/(1+χ³|E^n|²/ε_r)``): the increment form applies ε_eff to the E-update increment
    ``ΔE ∝ ∂ₓH``, which is 90° out of phase with E, so the χ³ correction hits the
    increment where it is smallest ⇒ it UNDERESTIMATES the self-phase-modulation index
    shift. The true-CW absolute oracle (tests/test_kerr_spm_absolute_oracle.py, #446, using
    ``waveform='continuous_wave'`` so ⟨E²⟩=A²/2 is unambiguous) pins the D-based operator at
    ratio ≈0.95 of the (3/8)χ³A² textbook (residual ~5% = Yee dispersion), while the increment
    form scores ~0.33 there — the gate cleanly discriminates them. (An earlier pulsed-source
    3D estimate read the pair as ~0.21× increment vs ~0.59× D-based — a comparator-independent
    2.8× relative gain, but its absolute scale was biased low by the pulsed-⟨E²⟩ ambiguity that
    #446 resolves.) See docs/research_notes/2026-07-23_kerr_quantitative_spm_finding.md.
    The pre-#437 form ``E → E/(1+factor)`` was worse still — a nonlinear
    ABSORBER (phase-neutral to first order). χ³=0 ⇒ D_target=ε_r·E_lin, denom=ε_r ⇒
    E=E_lin (byte-identical). Stable and lossless up to strong χ³ (verified χ³≤4).

    Parameters
    ----------
    state : FDTDState
        State **after** the linear ε_r E-update (contains ``E_lin``).
    e_prev : tuple(jnp.ndarray, jnp.ndarray, jnp.ndarray)
        The E-field components (ex, ey, ez) **before** the linear update (``E^n``),
        used to reconstruct D^n and form the co-located pre-update intensity.
    chi3_arr : jnp.ndarray, shape (Nx, Ny, Nz)
        Third-order susceptibility at each cell (m^2/V^2). Zero where linear ⇒ the
        cubic reduces to E=E_lin (byte-identical).
    eps_r_arr : jnp.ndarray, shape (Nx, Ny, Nz)
        Relative permittivity per cell (ε_r ≥ 1).

    Returns
    -------
    FDTDState with the reactive-Kerr-corrected E-field components.
    """
    ex_p, ey_p, ez_p = e_prev
    e_sq_p = ex_p ** 2 + ey_p ** 2 + ez_p ** 2       # |E^n|^2 (co-located, pre-update)
    # D_target/ε0 = ε_r·E_lin + χ³·|E^n|²·E^n  (advance the reconstructed D linearly)
    dtx = eps_r_arr * state.ex + chi3_arr * e_sq_p * ex_p
    dty = eps_r_arr * state.ey + chi3_arr * e_sq_p * ey_p
    dtz = eps_r_arr * state.ez + chi3_arr * e_sq_p * ez_p
    # solve  ε_r·E + χ³·|E|²·E = D_target  by fixed-point  E = D_target/(ε_r + χ³|E|²)
    ex, ey, ez = state.ex, state.ey, state.ez         # initial guess = E_lin
    for _ in range(_KERR_FIXED_POINT_ITERS):
        denom = eps_r_arr + chi3_arr * (ex ** 2 + ey ** 2 + ez ** 2)   # = ε_eff (>= ε_r >= 1)
        ex = dtx / denom
        ey = dty / denom
        ez = dtz / denom

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
