"""Uniaxial PML (UPML) helpers for uniform-grid Yee updates.

D/B-equivalent formulation: PML conductivity is applied as σ/ε₀ (material-
independent), matching Meep's approach where PML acts on D/B fields with
the constitutive relation D→E handling the material.

Mathematically equivalent to:
  D^{n+1} = Ca_D · D^n + Cb_D · curl(H)    (PML on D, material-independent)
  E^{n+1} = D^{n+1} / (ε_r · ε₀)           (constitutive relation)

Expressed as a direct E update:
  E^{n+1} = Ca_D · E^n + Cb_D/(ε_r·ε₀) · curl(H)

Ca_D = (1 − σ·dt/(2ε₀)) / (1 + σ·dt/(2ε₀))  — can be negative, which is
stable under Crank-Nicolson (|Ca_D| < 1 always).

This gives material-independent PML absorption: dielectric and vacuum cells
get the SAME PML damping, matching Meep's UPML behavior.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rfx.boundaries.cpml import _cpml_profile, _get_axis_cell_sizes
from rfx.core.yee import EPS_0, MU_0, FDTDState, MaterialArrays, _shift_bwd, _shift_fwd


class UPMLCoeffs(NamedTuple):
    """Precomputed component-aware UPML update coefficients."""
    ca_ex: jnp.ndarray
    ca_ey: jnp.ndarray
    ca_ez: jnp.ndarray
    cb_ex: jnp.ndarray
    cb_ey: jnp.ndarray
    cb_ez: jnp.ndarray
    da_hx: jnp.ndarray
    da_hy: jnp.ndarray
    da_hz: jnp.ndarray
    db_hx: jnp.ndarray
    db_hy: jnp.ndarray
    db_hz: jnp.ndarray


def _axis_sigma_full(grid, axis: str) -> jnp.ndarray:
    """Return graded PML σ for one axis over the full grid."""
    n = grid.cpml_layers
    if n <= 0:
        return jnp.zeros(grid.shape, dtype=jnp.float32)

    dx_x, dx_y, dz_lo, dz_hi = _get_axis_cell_sizes(grid)
    kappa_max = 1.0
    pec_faces = getattr(grid, "pec_faces", None) or set()

    if axis == "x":
        prof = _cpml_profile(n, grid.dt, dx_x, kappa_max=kappa_max)
        sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
        if grid.pad_x > 0:
            if "x_lo" not in pec_faces:
                sigma = sigma.at[:n, :, :].set(prof.sigma[:, None, None])
            if "x_hi" not in pec_faces:
                sigma = sigma.at[-n:, :, :].set(jnp.flip(prof.sigma)[:, None, None])
        return sigma
    if axis == "y":
        prof = _cpml_profile(n, grid.dt, dx_y, kappa_max=kappa_max)
        sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
        if grid.pad_y > 0:
            if "y_lo" not in pec_faces:
                sigma = sigma.at[:, :n, :].set(prof.sigma[None, :, None])
            if "y_hi" not in pec_faces:
                sigma = sigma.at[:, -n:, :].set(jnp.flip(prof.sigma)[None, :, None])
        return sigma
    if axis == "z":
        sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
        if grid.pad_z <= 0:
            return sigma
        prof_lo = _cpml_profile(n, grid.dt, dz_lo, kappa_max=kappa_max)
        prof_hi = _cpml_profile(n, grid.dt, dz_hi, kappa_max=kappa_max)
        if "z_lo" not in pec_faces:
            sigma = sigma.at[:, :, :n].set(prof_lo.sigma[None, None, :])
        if "z_hi" not in pec_faces:
            sigma = sigma.at[:, :, -n:].set(jnp.flip(prof_hi.sigma)[None, None, :])
        return sigma
    raise ValueError(f"Unsupported axis {axis!r}")


def init_upml(
    grid,
    materials: MaterialArrays,
    *,
    axes: str = "xyz",
    aniso_eps=None,
) -> UPMLCoeffs:
    """Build static UPML coefficients for uniform-grid Yee updates.

    D/B-equivalent formulation where PML loss is material-independent:
    - PML loss = σ_pml · dt / (2·ε₀)  — same for vacuum and dielectric
    - Material conductivity loss = σ_mat · dt / (2·ε_r·ε₀)
    - Ca = (1 − loss) / (1 + loss) — NO cap, can be negative (stable)

    Per-component anisotropic damping:
    - E_x / H_x use σ_pml from y + z PML
    - E_y / H_y use σ_pml from x + z PML
    - E_z / H_z use σ_pml from x + y PML

    Parameters
    ----------
    aniso_eps : tuple of 3 arrays, optional
        (eps_ex, eps_ey, eps_ez) per-component relative permittivity from
        subpixel smoothing.  When provided, E-field coefficients use
        per-component eps instead of the scalar materials.eps_r.
    """

    sigma_x = (
        _axis_sigma_full(grid, "x")
        if getattr(grid, "pad_x", 0) > 0 and "x" in axes
        else jnp.zeros(grid.shape, dtype=jnp.float32)
    )
    sigma_y = (
        _axis_sigma_full(grid, "y")
        if getattr(grid, "pad_y", 0) > 0 and "y" in axes
        else jnp.zeros(grid.shape, dtype=jnp.float32)
    )
    sigma_z = (
        _axis_sigma_full(grid, "z")
        if getattr(grid, "pad_z", 0) > 0 and "z" in axes
        else jnp.zeros(grid.shape, dtype=jnp.float32)
    )

    # Scale sigma to match Meep's convention.
    # Textbook: σ_max = -ln(R)·(m+1)/(2·η·d)     [continuous theory: R_cont = R_target]
    # Meep:     σ_max = -ln(R)·(m+1)/(4·η·dx)     [over-designed:    R_cont = R_target^(n/2)]
    # Ratio: n_layers/2.  This compensates for discretization error so that
    # the DISCRETE reflection ≈ R_target rather than R_target^(2/n).
    n_pml = grid.cpml_layers
    sigma_scale = jnp.float32(max(n_pml, 1) / 2.0)
    sigma_x = sigma_x * sigma_scale
    sigma_y = sigma_y * sigma_scale
    sigma_z = sigma_z * sigma_scale

    mu_abs = materials.mu_r * jnp.float32(MU_0)
    sigma_mat = materials.sigma.astype(jnp.float32)
    dt = jnp.float32(grid.dt)
    dx = jnp.float32(grid.dx)
    eps_0 = jnp.float32(EPS_0)

    # Per-component absolute permittivity (subpixel smoothing or scalar)
    if aniso_eps is not None:
        eps_ex, eps_ey, eps_ez = aniso_eps
        eps_abs_ex = eps_ex.astype(jnp.float32) * jnp.float32(EPS_0)
        eps_abs_ey = eps_ey.astype(jnp.float32) * jnp.float32(EPS_0)
        eps_abs_ez = eps_ez.astype(jnp.float32) * jnp.float32(EPS_0)
    else:
        eps_abs_scalar = materials.eps_r * jnp.float32(EPS_0)
        eps_abs_ex = eps_abs_scalar
        eps_abs_ey = eps_abs_scalar
        eps_abs_ez = eps_abs_scalar

    # PML σ per component (perpendicular directions only)
    sigma_pml_ex = sigma_y + sigma_z
    sigma_pml_ey = sigma_x + sigma_z
    sigma_pml_ez = sigma_x + sigma_y

    def _e_coeffs(sigma_pml_comp, eps_abs_comp):
        # D/B-equivalent: PML loss is material-independent (σ/ε₀)
        # Material conductivity loss uses local ε_r·ε₀
        loss_pml = sigma_pml_comp * dt / (jnp.float32(2.0) * eps_0)
        loss_mat = sigma_mat * dt / (jnp.float32(2.0) * eps_abs_comp)
        loss = loss_pml + loss_mat
        # NO cap — Ca < 0 is stable (|Ca| < 1 always, Crank-Nicolson)
        denom = jnp.float32(1.0) + loss
        ca = (jnp.float32(1.0) - loss) / denom
        cb = (dt / eps_abs_comp) / denom
        return ca.astype(jnp.float32), cb.astype(jnp.float32)

    ca_ex, cb_ex = _e_coeffs(sigma_pml_ex, eps_abs_ex)
    ca_ey, cb_ey = _e_coeffs(sigma_pml_ey, eps_abs_ey)
    ca_ez, cb_ez = _e_coeffs(sigma_pml_ez, eps_abs_ez)

    # H-field: impedance-matched PML (same loss as E: σ_pml·dt/(2ε₀))
    sigma_pml_hx = sigma_y + sigma_z
    sigma_pml_hy = sigma_x + sigma_z
    sigma_pml_hz = sigma_x + sigma_y

    def _h_coeffs(sigma_pml_comp):
        # Impedance matched: loss_B = σ_pml·dt/(2ε₀) = loss_D
        loss = sigma_pml_comp * dt / (jnp.float32(2.0) * eps_0)
        # NO cap
        denom = jnp.float32(1.0) + loss
        da = (jnp.float32(1.0) - loss) / denom
        db = (dt / mu_abs) / denom
        return da.astype(jnp.float32), db.astype(jnp.float32)

    da_hx, db_hx = _h_coeffs(sigma_pml_hx)
    da_hy, db_hy = _h_coeffs(sigma_pml_hy)
    da_hz, db_hz = _h_coeffs(sigma_pml_hz)

    return UPMLCoeffs(
        ca_ex=ca_ex, ca_ey=ca_ey, ca_ez=ca_ez,
        cb_ex=cb_ex / dx, cb_ey=cb_ey / dx, cb_ez=cb_ez / dx,
        da_hx=da_hx, da_hy=da_hy, da_hz=da_hz,
        db_hx=db_hx / dx, db_hy=db_hy / dx, db_hz=db_hz / dx,
    )


def apply_upml_h(
    state: FDTDState,
    coeffs: UPMLCoeffs,
    periodic: tuple = (False, False, False),
) -> FDTDState:
    """H-field update using precomputed UPML coefficients."""
    def fwd(arr, axis):
        if periodic[axis]:
            return jnp.roll(arr, -1, axis)
        return _shift_fwd(arr, axis)

    _fdtype = state.ex.dtype
    ex = state.ex.astype(jnp.float32)
    ey = state.ey.astype(jnp.float32)
    ez = state.ez.astype(jnp.float32)

    curl_x = (fwd(ez, 1) - ez) - (fwd(ey, 2) - ey)
    curl_y = (fwd(ex, 2) - ex) - (fwd(ez, 0) - ez)
    curl_z = (fwd(ey, 0) - ey) - (fwd(ex, 1) - ex)

    hx = (coeffs.da_hx * state.hx.astype(jnp.float32) - coeffs.db_hx * curl_x).astype(_fdtype)
    hy = (coeffs.da_hy * state.hy.astype(jnp.float32) - coeffs.db_hy * curl_y).astype(_fdtype)
    hz = (coeffs.da_hz * state.hz.astype(jnp.float32) - coeffs.db_hz * curl_z).astype(_fdtype)

    return state._replace(hx=hx, hy=hy, hz=hz)


def apply_upml_e(
    state: FDTDState,
    coeffs: UPMLCoeffs,
    periodic: tuple = (False, False, False),
) -> FDTDState:
    """E-field update using precomputed UPML coefficients."""
    def bwd(arr, axis):
        if periodic[axis]:
            return jnp.roll(arr, 1, axis)
        return _shift_bwd(arr, axis)

    _fdtype = state.ex.dtype
    hx = state.hx.astype(jnp.float32)
    hy = state.hy.astype(jnp.float32)
    hz = state.hz.astype(jnp.float32)

    curl_x = (hz - bwd(hz, 1)) - (hy - bwd(hy, 2))
    curl_y = (hx - bwd(hx, 2)) - (hz - bwd(hz, 0))
    curl_z = (hy - bwd(hy, 0)) - (hx - bwd(hx, 1))

    ex = (coeffs.ca_ex * state.ex.astype(jnp.float32) + coeffs.cb_ex * curl_x).astype(_fdtype)
    ey = (coeffs.ca_ey * state.ey.astype(jnp.float32) + coeffs.cb_ey * curl_y).astype(_fdtype)
    ez = (coeffs.ca_ez * state.ez.astype(jnp.float32) + coeffs.cb_ez * curl_z).astype(_fdtype)

    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)
