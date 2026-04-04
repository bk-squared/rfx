"""Convolutional Perfectly Matched Layer (CPML) absorbing boundary.

Implements the stretched-coordinate PML formulation with auxiliary
differential equation (ADE) update. Supports CFS-CPML with κ stretching
for improved evanescent wave absorption.

Standard CPML:  s(ω) = 1 + σ/(jω)
CFS-CPML:       s(ω) = κ + σ/(α + jω)

When kappa_max=1.0 (default), the CFS-CPML reduces exactly to standard CPML.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.grid import Grid
from rfx.core.yee import _shift_fwd, _shift_bwd


class CPMLParams(NamedTuple):
    """CPML profile parameters for one axis direction."""

    # Damping profile σ (n_layers,)
    sigma: jnp.ndarray
    # Frequency-shift parameter κ (n_layers,)
    kappa: jnp.ndarray
    # CFS parameter α (n_layers,)
    alpha: jnp.ndarray
    # Precomputed update coefficients
    b: jnp.ndarray  # exp(-(σ/κ + α) * dt/ε₀)
    c: jnp.ndarray  # σ * (b - 1) / (σ*κ + κ²*α)


class CPMLState(NamedTuple):
    """Auxiliary CPML fields (psi) for all 6 faces."""

    # Psi arrays for E-field update (from H-curl correction)
    psi_ex_ylo: jnp.ndarray
    psi_ex_yhi: jnp.ndarray
    psi_ex_zlo: jnp.ndarray
    psi_ex_zhi: jnp.ndarray
    psi_ey_xlo: jnp.ndarray
    psi_ey_xhi: jnp.ndarray
    psi_ey_zlo: jnp.ndarray
    psi_ey_zhi: jnp.ndarray
    psi_ez_xlo: jnp.ndarray
    psi_ez_xhi: jnp.ndarray
    psi_ez_ylo: jnp.ndarray
    psi_ez_yhi: jnp.ndarray
    # Psi arrays for H-field update (from E-curl correction)
    psi_hx_ylo: jnp.ndarray
    psi_hx_yhi: jnp.ndarray
    psi_hx_zlo: jnp.ndarray
    psi_hx_zhi: jnp.ndarray
    psi_hy_xlo: jnp.ndarray
    psi_hy_xhi: jnp.ndarray
    psi_hy_zlo: jnp.ndarray
    psi_hy_zhi: jnp.ndarray
    psi_hz_xlo: jnp.ndarray
    psi_hz_xhi: jnp.ndarray
    psi_hz_ylo: jnp.ndarray
    psi_hz_yhi: jnp.ndarray


def _cpml_profile(
    n_layers: int,
    dt: float,
    dx: float,
    order: int = 3,
    kappa_max: float = 1.0,
) -> CPMLParams:
    """Compute graded CPML profile using polynomial grading.

    Parameters
    ----------
    n_layers : int
        Number of CPML layers.
    dt : float
        Timestep (seconds).
    dx : float
        Cell size (meters).
    order : int
        Polynomial grading order.
    kappa_max : float
        Maximum κ stretching parameter. 1.0 = standard CPML (no stretching).
        Values > 1 improve evanescent wave absorption.
    """
    EPS_0 = 8.854187817e-12
    MU_0 = 1.2566370614e-6

    # Optimal σ_max (Taflove & Hagness, eq 7.67)
    # σ_opt = 0.8 * (m+1) / (η * Δx) where η = sqrt(μ/ε)
    eta = np.sqrt(MU_0 / EPS_0)  # ≈ 376.73 Ω
    # For CFS-CPML (κ>1), scale σ_max by κ_max (Gedney recommendation).
    # The κ stretching provides impedance matching that allows stronger σ
    # without increasing interface reflections, improving evanescent absorption.
    sigma_max = 0.8 * (order + 1) / (eta * dx) * kappa_max

    # Graded profiles: polynomial from max at outer boundary (index 0)
    # to 0 at interior edge (index n-1) for the lo face.
    # The hi face uses jnp.flip() to reverse this.
    rho = 1.0 - np.arange(n_layers, dtype=np.float64) / max(n_layers - 1, 1)
    sigma = sigma_max * rho**order
    # κ graded from kappa_max (outer) to 1.0 (inner): κ(ρ) = 1 + (κ_max - 1) * ρ^m
    kappa = 1.0 + (kappa_max - 1.0) * rho**order
    alpha = 0.05 * (1.0 - rho)  # α: small, decreasing toward outer boundary

    # Update coefficients
    denom = sigma * kappa + kappa**2 * alpha
    b = np.exp(-(sigma / kappa + alpha) * dt / EPS_0)
    c = np.where(denom > 1e-30, sigma * (b - 1.0) / denom, 0.0)

    return CPMLParams(
        sigma=jnp.array(sigma, dtype=jnp.float32),
        kappa=jnp.array(kappa, dtype=jnp.float32),
        alpha=jnp.array(alpha, dtype=jnp.float32),
        b=jnp.array(b, dtype=jnp.float32),
        c=jnp.array(c, dtype=jnp.float32),
    )


def init_cpml(grid: Grid, *, kappa_max: float | None = None) -> tuple[CPMLParams, CPMLState]:
    """Initialize CPML parameters and zero-valued auxiliary fields.

    Parameters
    ----------
    grid : Grid
        Simulation grid. If the grid carries a ``kappa_max`` attribute
        (set by the high-level Simulation API), it is used as the default
        when the caller does not pass an explicit *kappa_max*.
    kappa_max : float or None
        Maximum κ stretching parameter for CFS-CPML. When None, reads
        ``grid.kappa_max`` if available, otherwise defaults to 1.0
        (standard CPML, backward compatible). Values > 1 (e.g. 5)
        improve absorption of evanescent and low-frequency fields.
    """
    if kappa_max is None:
        kappa_max = getattr(grid, "kappa_max", None) or 1.0
    n = grid.cpml_layers
    params = _cpml_profile(n, grid.dt, grid.dx, kappa_max=kappa_max)

    nx, ny, nz = grid.shape

    def _zeros(dim_size: int, perp1: int, perp2: int) -> jnp.ndarray:
        return jnp.zeros((n, perp1, perp2), dtype=jnp.float32)

    state = CPMLState(
        # E-field psi (12 faces)
        psi_ex_ylo=_zeros(n, nx, nz), psi_ex_yhi=_zeros(n, nx, nz),
        psi_ex_zlo=_zeros(n, nx, ny), psi_ex_zhi=_zeros(n, nx, ny),
        psi_ey_xlo=_zeros(n, ny, nz), psi_ey_xhi=_zeros(n, ny, nz),
        psi_ey_zlo=_zeros(n, ny, nx), psi_ey_zhi=_zeros(n, ny, nx),
        psi_ez_xlo=_zeros(n, nz, ny), psi_ez_xhi=_zeros(n, nz, ny),
        psi_ez_ylo=_zeros(n, nz, nx), psi_ez_yhi=_zeros(n, nz, nx),
        # H-field psi (12 faces)
        psi_hx_ylo=_zeros(n, nx, nz), psi_hx_yhi=_zeros(n, nx, nz),
        psi_hx_zlo=_zeros(n, nx, ny), psi_hx_zhi=_zeros(n, nx, ny),
        psi_hy_xlo=_zeros(n, ny, nz), psi_hy_xhi=_zeros(n, ny, nz),
        psi_hy_zlo=_zeros(n, ny, nx), psi_hy_zhi=_zeros(n, ny, nx),
        psi_hz_xlo=_zeros(n, nz, ny), psi_hz_xhi=_zeros(n, nz, ny),
        psi_hz_ylo=_zeros(n, nz, nx), psi_hz_yhi=_zeros(n, nz, nx),
    )

    return params, state


def _kappa_correction(kappa, curl_slice, shape_broadcast):
    """Compute the κ correction term: (1/κ - 1) * curl.

    When κ=1, this returns 0 (no correction). When κ>1, this reduces
    the curl contribution inside the PML region by dividing by κ.

    Parameters
    ----------
    kappa : array (n_layers,)
        κ profile values.
    curl_slice : array
        The curl component slice in the PML region.
    shape_broadcast : str
        How to reshape kappa for broadcasting. One of 'x', 'y', 'z'.
    """
    # Reshape kappa for broadcasting to 3D slice
    k = kappa[:, None, None]  # (n, 1, 1)
    return (1.0 / k - 1.0) * curl_slice


def apply_cpml_e(
    state, cpml_params: CPMLParams, cpml_state: CPMLState, grid: Grid,
    axes: str = "xyz",
) -> tuple:
    """Apply CPML correction to E-field update on all 6 faces.

    E-field CPML uses backward-difference curl of H:
      curl_H_component = (H_slice - roll(H, +1, axis)) / dx
    Then: psi = b * psi_old + c * curl_H_component
          E += (dt / eps) * psi

    For CFS-CPML with κ>1, we also apply the coordinate stretching
    correction to the main curl contribution inside the PML:
          E += (dt / eps) * (1/κ - 1) * curl_H_component
    When κ=1 (default), this correction is exactly 0.
    """
    n = grid.cpml_layers
    dt = grid.dt
    dx = grid.dx
    EPS_0 = 8.854187817e-12
    coeff_e = dt / (EPS_0 * 1.0)

    b = cpml_params.b  # (n,)
    c = cpml_params.c  # (n,)
    kappa = cpml_params.kappa  # (n,)
    b_r = jnp.flip(b)  # reversed profile for hi faces
    c_r = jnp.flip(c)
    kappa_r = jnp.flip(kappa)

    ex = state.ex
    ey = state.ey
    ez = state.ez

    # =========================================================
    # X-axis CPML: Ey correction (dHz/dx) and Ez correction (dHy/dx)
    # =========================================================

    if "x" in axes:
        b_x = b[:, None, None]    # (n, 1, 1) → broadcasts to (n, ny, nz)
        c_x = c[:, None, None]
        b_xr = b_r[:, None, None]
        c_xr = c_r[:, None, None]
        k_x = kappa[:, None, None]
        k_xr = kappa_r[:, None, None]

        # --- X-lo: Ey correction from dHz/dx ---
        # The main FDTD update already computed the full curl. The CPML
        # correction replaces 1/dx with (1/κ)/dx + psi. The psi part is
        # the ADE update below. The κ part is the "kappa correction".
        hz_xlo = state.hz[:n, :, :]
        hz_shifted_xlo = _shift_bwd(state.hz, 0)[:n, :, :]
        curl_hz_dx_xlo = (hz_xlo - hz_shifted_xlo) / dx

        new_psi_ey_xlo = b_x * cpml_state.psi_ey_xlo + c_x * curl_hz_dx_xlo
        ey = ey.at[:n, :, :].add(-coeff_e * new_psi_ey_xlo)
        # κ correction: (1/κ - 1) applied to the curl already in the main update
        ey = ey.at[:n, :, :].add(-coeff_e * (1.0 / k_x - 1.0) * curl_hz_dx_xlo)

        # --- X-hi: Ey correction from dHz/dx ---
        hz_xhi = state.hz[-n:, :, :]
        hz_shifted_xhi = _shift_bwd(state.hz, 0)[-n:, :, :]
        curl_hz_dx_xhi = (hz_xhi - hz_shifted_xhi) / dx

        new_psi_ey_xhi = b_xr * cpml_state.psi_ey_xhi + c_xr * curl_hz_dx_xhi
        ey = ey.at[-n:, :, :].add(-coeff_e * new_psi_ey_xhi)
        ey = ey.at[-n:, :, :].add(-coeff_e * (1.0 / k_xr - 1.0) * curl_hz_dx_xhi)

        # --- X-lo: Ez correction from dHy/dx ---
        hy_xlo = state.hy[:n, :, :]
        hy_shifted_xlo = _shift_bwd(state.hy, 0)[:n, :, :]
        curl_hy_dx_xlo = (hy_xlo - hy_shifted_xlo) / dx          # (n, ny, nz)
        curl_hy_dx_xlo_t = jnp.transpose(curl_hy_dx_xlo, (0, 2, 1))  # (n, nz, ny)

        new_psi_ez_xlo = b_x * cpml_state.psi_ez_xlo + c_x * curl_hy_dx_xlo_t
        correction_ez_xlo = jnp.transpose(new_psi_ez_xlo, (0, 2, 1))  # (n, ny, nz)
        ez = ez.at[:n, :, :].add(coeff_e * correction_ez_xlo)
        ez = ez.at[:n, :, :].add(coeff_e * (1.0 / k_x - 1.0) * curl_hy_dx_xlo)

        # --- X-hi: Ez correction from dHy/dx ---
        hy_xhi = state.hy[-n:, :, :]
        hy_shifted_xhi = _shift_bwd(state.hy, 0)[-n:, :, :]
        curl_hy_dx_xhi = (hy_xhi - hy_shifted_xhi) / dx
        curl_hy_dx_xhi_t = jnp.transpose(curl_hy_dx_xhi, (0, 2, 1))

        new_psi_ez_xhi = b_xr * cpml_state.psi_ez_xhi + c_xr * curl_hy_dx_xhi_t
        correction_ez_xhi = jnp.transpose(new_psi_ez_xhi, (0, 2, 1))
        ez = ez.at[-n:, :, :].add(coeff_e * correction_ez_xhi)
        ez = ez.at[-n:, :, :].add(coeff_e * (1.0 / k_xr - 1.0) * curl_hy_dx_xhi)
    else:
        new_psi_ey_xlo = cpml_state.psi_ey_xlo
        new_psi_ey_xhi = cpml_state.psi_ey_xhi
        new_psi_ez_xlo = cpml_state.psi_ez_xlo
        new_psi_ez_xhi = cpml_state.psi_ez_xhi

    # =========================================================
    # Y-axis CPML: Ex correction (dHz/dy) and Ez correction (dHx/dy)
    # =========================================================

    b_yn = b[:, None, None]
    c_yn = c[:, None, None]
    b_yrn = b_r[:, None, None]
    c_yrn = c_r[:, None, None]
    k_yn = kappa[:, None, None]
    k_yrn = kappa_r[:, None, None]

    if "y" in axes:
        # --- Y-lo: Ex correction from dHz/dy ---
        hz_ylo = state.hz[:, :n, :]
        hz_shifted_ylo = _shift_bwd(state.hz, 1)[:, :n, :]
        curl_hz_dy_ylo = (hz_ylo - hz_shifted_ylo) / dx

        curl_hz_dy_ylo_t = jnp.transpose(curl_hz_dy_ylo, (1, 0, 2))

        new_psi_ex_ylo = b_yn * cpml_state.psi_ex_ylo + c_yn * curl_hz_dy_ylo_t
        correction_ex_ylo = jnp.transpose(new_psi_ex_ylo, (1, 0, 2))
        ex = ex.at[:, :n, :].add(coeff_e * correction_ex_ylo)
        # κ correction for Y-lo Ex
        kappa_corr_ylo = jnp.transpose((1.0 / k_yn - 1.0) * curl_hz_dy_ylo_t, (1, 0, 2))
        ex = ex.at[:, :n, :].add(coeff_e * kappa_corr_ylo)

        # --- Y-hi: Ex correction from dHz/dy ---
        hz_yhi = state.hz[:, -n:, :]
        hz_shifted_yhi = _shift_bwd(state.hz, 1)[:, -n:, :]
        curl_hz_dy_yhi = (hz_yhi - hz_shifted_yhi) / dx

        curl_hz_dy_yhi_t = jnp.transpose(curl_hz_dy_yhi, (1, 0, 2))

        new_psi_ex_yhi = b_yrn * cpml_state.psi_ex_yhi + c_yrn * curl_hz_dy_yhi_t
        correction_ex_yhi = jnp.transpose(new_psi_ex_yhi, (1, 0, 2))
        ex = ex.at[:, -n:, :].add(coeff_e * correction_ex_yhi)
        kappa_corr_yhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_hz_dy_yhi_t, (1, 0, 2))
        ex = ex.at[:, -n:, :].add(coeff_e * kappa_corr_yhi)

        # --- Y-lo: Ez correction from dHx/dy ---
        hx_ylo = state.hx[:, :n, :]
        hx_shifted_ylo = _shift_bwd(state.hx, 1)[:, :n, :]
        curl_hx_dy_ylo = (hx_ylo - hx_shifted_ylo) / dx

        curl_hx_dy_ylo_t = jnp.transpose(curl_hx_dy_ylo, (1, 2, 0))

        new_psi_ez_ylo = b_yn * cpml_state.psi_ez_ylo + c_yn * curl_hx_dy_ylo_t
        correction_ez_ylo = jnp.transpose(new_psi_ez_ylo, (2, 0, 1))
        ez = ez.at[:, :n, :].add(-coeff_e * correction_ez_ylo)
        # κ correction for Y-lo Ez (note the negative sign matches the curl term)
        kappa_corr_ez_ylo = jnp.transpose((1.0 / k_yn - 1.0) * curl_hx_dy_ylo_t, (2, 0, 1))
        ez = ez.at[:, :n, :].add(-coeff_e * kappa_corr_ez_ylo)

        # --- Y-hi: Ez correction from dHx/dy ---
        hx_yhi = state.hx[:, -n:, :]
        hx_shifted_yhi = _shift_bwd(state.hx, 1)[:, -n:, :]
        curl_hx_dy_yhi = (hx_yhi - hx_shifted_yhi) / dx

        curl_hx_dy_yhi_t = jnp.transpose(curl_hx_dy_yhi, (1, 2, 0))

        new_psi_ez_yhi = b_yrn * cpml_state.psi_ez_yhi + c_yrn * curl_hx_dy_yhi_t
        correction_ez_yhi = jnp.transpose(new_psi_ez_yhi, (2, 0, 1))
        ez = ez.at[:, -n:, :].add(-coeff_e * correction_ez_yhi)
        kappa_corr_ez_yhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_hx_dy_yhi_t, (2, 0, 1))
        ez = ez.at[:, -n:, :].add(-coeff_e * kappa_corr_ez_yhi)
    else:
        new_psi_ex_ylo = cpml_state.psi_ex_ylo
        new_psi_ex_yhi = cpml_state.psi_ex_yhi
        new_psi_ez_ylo = cpml_state.psi_ez_ylo
        new_psi_ez_yhi = cpml_state.psi_ez_yhi

    # =========================================================
    # Z-axis CPML: Ex correction (dHy/dz) and Ey correction (dHx/dz)
    # =========================================================

    if "z" in axes:
        # --- Z-lo: Ex correction from dHy/dz ---
        hy_zlo = state.hy[:, :, :n]
        hy_shifted_zlo = _shift_bwd(state.hy, 2)[:, :, :n]
        curl_hy_dz_zlo = (hy_zlo - hy_shifted_zlo) / dx

        curl_hy_dz_zlo_t = jnp.transpose(curl_hy_dz_zlo, (2, 0, 1))

        new_psi_ex_zlo = b_yn * cpml_state.psi_ex_zlo + c_yn * curl_hy_dz_zlo_t
        correction_ex_zlo = jnp.transpose(new_psi_ex_zlo, (1, 2, 0))
        ex = ex.at[:, :, :n].add(-coeff_e * correction_ex_zlo)
        # κ correction for Z-lo Ex
        kappa_corr_ex_zlo = jnp.transpose((1.0 / k_yn - 1.0) * curl_hy_dz_zlo_t, (1, 2, 0))
        ex = ex.at[:, :, :n].add(-coeff_e * kappa_corr_ex_zlo)

        # --- Z-hi: Ex correction from dHy/dz ---
        hy_zhi = state.hy[:, :, -n:]
        hy_shifted_zhi = _shift_bwd(state.hy, 2)[:, :, -n:]
        curl_hy_dz_zhi = (hy_zhi - hy_shifted_zhi) / dx

        curl_hy_dz_zhi_t = jnp.transpose(curl_hy_dz_zhi, (2, 0, 1))

        new_psi_ex_zhi = b_yrn * cpml_state.psi_ex_zhi + c_yrn * curl_hy_dz_zhi_t
        correction_ex_zhi = jnp.transpose(new_psi_ex_zhi, (1, 2, 0))
        ex = ex.at[:, :, -n:].add(-coeff_e * correction_ex_zhi)
        kappa_corr_ex_zhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_hy_dz_zhi_t, (1, 2, 0))
        ex = ex.at[:, :, -n:].add(-coeff_e * kappa_corr_ex_zhi)

        # --- Z-lo: Ey correction from dHx/dz ---
        hx_zlo = state.hx[:, :, :n]
        hx_shifted_zlo = _shift_bwd(state.hx, 2)[:, :, :n]
        curl_hx_dz_zlo = (hx_zlo - hx_shifted_zlo) / dx

        curl_hx_dz_zlo_t = jnp.transpose(curl_hx_dz_zlo, (2, 1, 0))

        new_psi_ey_zlo = b_yn * cpml_state.psi_ey_zlo + c_yn * curl_hx_dz_zlo_t
        correction_ey_zlo = jnp.transpose(new_psi_ey_zlo, (2, 1, 0))
        ey = ey.at[:, :, :n].add(coeff_e * correction_ey_zlo)
        kappa_corr_ey_zlo = jnp.transpose((1.0 / k_yn - 1.0) * curl_hx_dz_zlo_t, (2, 1, 0))
        ey = ey.at[:, :, :n].add(coeff_e * kappa_corr_ey_zlo)

        # --- Z-hi: Ey correction from dHx/dz ---
        hx_zhi = state.hx[:, :, -n:]
        hx_shifted_zhi = _shift_bwd(state.hx, 2)[:, :, -n:]
        curl_hx_dz_zhi = (hx_zhi - hx_shifted_zhi) / dx

        curl_hx_dz_zhi_t = jnp.transpose(curl_hx_dz_zhi, (2, 1, 0))

        new_psi_ey_zhi = b_yrn * cpml_state.psi_ey_zhi + c_yrn * curl_hx_dz_zhi_t
        correction_ey_zhi = jnp.transpose(new_psi_ey_zhi, (2, 1, 0))
        ey = ey.at[:, :, -n:].add(coeff_e * correction_ey_zhi)
        kappa_corr_ey_zhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_hx_dz_zhi_t, (2, 1, 0))
        ey = ey.at[:, :, -n:].add(coeff_e * kappa_corr_ey_zhi)
    else:
        new_psi_ex_zlo = cpml_state.psi_ex_zlo
        new_psi_ex_zhi = cpml_state.psi_ex_zhi
        new_psi_ey_zlo = cpml_state.psi_ey_zlo
        new_psi_ey_zhi = cpml_state.psi_ey_zhi

    state = state._replace(ex=ex, ey=ey, ez=ez)
    cpml_state = cpml_state._replace(
        # x-axis
        psi_ey_xlo=new_psi_ey_xlo,
        psi_ey_xhi=new_psi_ey_xhi,
        psi_ez_xlo=new_psi_ez_xlo,
        psi_ez_xhi=new_psi_ez_xhi,
        # y-axis
        psi_ex_ylo=new_psi_ex_ylo,
        psi_ex_yhi=new_psi_ex_yhi,
        psi_ez_ylo=new_psi_ez_ylo,
        psi_ez_yhi=new_psi_ez_yhi,
        # z-axis
        psi_ex_zlo=new_psi_ex_zlo,
        psi_ex_zhi=new_psi_ex_zhi,
        psi_ey_zlo=new_psi_ey_zlo,
        psi_ey_zhi=new_psi_ey_zhi,
    )

    return state, cpml_state


def apply_cpml_h(
    state, cpml_params: CPMLParams, cpml_state: CPMLState, grid: Grid,
    axes: str = "xyz",
) -> tuple:
    """Apply CPML correction to H-field update on all 6 faces.

    H-field CPML uses forward-difference curl of E:
      curl_E_component = (roll(E, -1, axis) - E_slice) / dx
    Then: psi = b * psi_old + c * curl_E_component
          H -= (dt / mu) * psi

    For CFS-CPML with κ>1, we also apply the coordinate stretching
    correction to the main curl contribution inside the PML:
          H -= (dt / mu) * (1/κ - 1) * curl_E_component
    When κ=1 (default), this correction is exactly 0.
    """
    n = grid.cpml_layers
    dt = grid.dt
    dx = grid.dx
    MU_0 = 1.2566370614e-6
    coeff_h = dt / MU_0

    b = cpml_params.b
    c = cpml_params.c
    kappa = cpml_params.kappa
    b_r = jnp.flip(b)
    c_r = jnp.flip(c)
    kappa_r = jnp.flip(kappa)

    hx = state.hx
    hy = state.hy
    hz = state.hz

    # =========================================================
    # X-axis CPML: Hy correction (dEz/dx) and Hz correction (dEy/dx)
    # =========================================================

    if "x" in axes:
        b_x = b[:, None, None]
        c_x = c[:, None, None]
        b_xr = b_r[:, None, None]
        c_xr = c_r[:, None, None]
        k_x = kappa[:, None, None]
        k_xr = kappa_r[:, None, None]

        # --- X-lo: Hy correction from dEz/dx ---
        ez_xlo = state.ez[:n, :, :]
        ez_shifted_xlo = _shift_fwd(state.ez, 0)[:n, :, :]
        curl_ez_dx_xlo = (ez_shifted_xlo - ez_xlo) / dx

        new_psi_hy_xlo = b_x * cpml_state.psi_hy_xlo + c_x * curl_ez_dx_xlo
        hy = hy.at[:n, :, :].add(coeff_h * new_psi_hy_xlo)
        hy = hy.at[:n, :, :].add(coeff_h * (1.0 / k_x - 1.0) * curl_ez_dx_xlo)

        # --- X-hi: Hy correction from dEz/dx ---
        ez_xhi = state.ez[-n:, :, :]
        ez_shifted_xhi = _shift_fwd(state.ez, 0)[-n:, :, :]
        curl_ez_dx_xhi = (ez_shifted_xhi - ez_xhi) / dx

        new_psi_hy_xhi = b_xr * cpml_state.psi_hy_xhi + c_xr * curl_ez_dx_xhi
        hy = hy.at[-n:, :, :].add(coeff_h * new_psi_hy_xhi)
        hy = hy.at[-n:, :, :].add(coeff_h * (1.0 / k_xr - 1.0) * curl_ez_dx_xhi)

        # --- X-lo: Hz correction from dEy/dx ---
        ey_xlo = state.ey[:n, :, :]
        ey_shifted_xlo = _shift_fwd(state.ey, 0)[:n, :, :]
        curl_ey_dx_xlo = (ey_shifted_xlo - ey_xlo) / dx          # (n, ny, nz)
        curl_ey_dx_xlo_t = jnp.transpose(curl_ey_dx_xlo, (0, 2, 1))  # (n, nz, ny)

        new_psi_hz_xlo = b_x * cpml_state.psi_hz_xlo + c_x * curl_ey_dx_xlo_t
        correction_hz_xlo = jnp.transpose(new_psi_hz_xlo, (0, 2, 1))  # (n, ny, nz)
        hz = hz.at[:n, :, :].add(-coeff_h * correction_hz_xlo)
        hz = hz.at[:n, :, :].add(-coeff_h * (1.0 / k_x - 1.0) * curl_ey_dx_xlo)

        # --- X-hi: Hz correction from dEy/dx ---
        ey_xhi = state.ey[-n:, :, :]
        ey_shifted_xhi = _shift_fwd(state.ey, 0)[-n:, :, :]
        curl_ey_dx_xhi = (ey_shifted_xhi - ey_xhi) / dx
        curl_ey_dx_xhi_t = jnp.transpose(curl_ey_dx_xhi, (0, 2, 1))

        new_psi_hz_xhi = b_xr * cpml_state.psi_hz_xhi + c_xr * curl_ey_dx_xhi_t
        correction_hz_xhi = jnp.transpose(new_psi_hz_xhi, (0, 2, 1))
        hz = hz.at[-n:, :, :].add(-coeff_h * correction_hz_xhi)
        hz = hz.at[-n:, :, :].add(-coeff_h * (1.0 / k_xr - 1.0) * curl_ey_dx_xhi)
    else:
        new_psi_hy_xlo = cpml_state.psi_hy_xlo
        new_psi_hy_xhi = cpml_state.psi_hy_xhi
        new_psi_hz_xlo = cpml_state.psi_hz_xlo
        new_psi_hz_xhi = cpml_state.psi_hz_xhi

    # =========================================================
    # Y-axis CPML: Hx correction (dEz/dy) and Hz correction (dEx/dy)
    # =========================================================

    b_yn = b[:, None, None]
    c_yn = c[:, None, None]
    b_yrn = b_r[:, None, None]
    c_yrn = c_r[:, None, None]
    k_yn = kappa[:, None, None]
    k_yrn = kappa_r[:, None, None]

    if "y" in axes:
        # --- Y-lo: Hx correction from dEz/dy ---
        ez_ylo = state.ez[:, :n, :]
        ez_shifted_ylo = _shift_fwd(state.ez, 1)[:, :n, :]
        curl_ez_dy_ylo = (ez_shifted_ylo - ez_ylo) / dx

        curl_ez_dy_ylo_t = jnp.transpose(curl_ez_dy_ylo, (1, 0, 2))

        new_psi_hx_ylo = b_yn * cpml_state.psi_hx_ylo + c_yn * curl_ez_dy_ylo_t
        correction_hx_ylo = jnp.transpose(new_psi_hx_ylo, (1, 0, 2))
        hx = hx.at[:, :n, :].add(-coeff_h * correction_hx_ylo)
        kappa_corr_hx_ylo = jnp.transpose((1.0 / k_yn - 1.0) * curl_ez_dy_ylo_t, (1, 0, 2))
        hx = hx.at[:, :n, :].add(-coeff_h * kappa_corr_hx_ylo)

        # --- Y-hi: Hx correction from dEz/dy ---
        ez_yhi = state.ez[:, -n:, :]
        ez_shifted_yhi = _shift_fwd(state.ez, 1)[:, -n:, :]
        curl_ez_dy_yhi = (ez_shifted_yhi - ez_yhi) / dx

        curl_ez_dy_yhi_t = jnp.transpose(curl_ez_dy_yhi, (1, 0, 2))

        new_psi_hx_yhi = b_yrn * cpml_state.psi_hx_yhi + c_yrn * curl_ez_dy_yhi_t
        correction_hx_yhi = jnp.transpose(new_psi_hx_yhi, (1, 0, 2))
        hx = hx.at[:, -n:, :].add(-coeff_h * correction_hx_yhi)
        kappa_corr_hx_yhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_ez_dy_yhi_t, (1, 0, 2))
        hx = hx.at[:, -n:, :].add(-coeff_h * kappa_corr_hx_yhi)

        # --- Y-lo: Hz correction from dEx/dy ---
        ex_ylo = state.ex[:, :n, :]
        ex_shifted_ylo = _shift_fwd(state.ex, 1)[:, :n, :]
        curl_ex_dy_ylo = (ex_shifted_ylo - ex_ylo) / dx

        curl_ex_dy_ylo_t = jnp.transpose(curl_ex_dy_ylo, (1, 2, 0))

        new_psi_hz_ylo = b_yn * cpml_state.psi_hz_ylo + c_yn * curl_ex_dy_ylo_t
        correction_hz_ylo = jnp.transpose(new_psi_hz_ylo, (2, 0, 1))
        hz = hz.at[:, :n, :].add(coeff_h * correction_hz_ylo)
        kappa_corr_hz_ylo = jnp.transpose((1.0 / k_yn - 1.0) * curl_ex_dy_ylo_t, (2, 0, 1))
        hz = hz.at[:, :n, :].add(coeff_h * kappa_corr_hz_ylo)

        # --- Y-hi: Hz correction from dEx/dy ---
        ex_yhi = state.ex[:, -n:, :]
        ex_shifted_yhi = _shift_fwd(state.ex, 1)[:, -n:, :]
        curl_ex_dy_yhi = (ex_shifted_yhi - ex_yhi) / dx

        curl_ex_dy_yhi_t = jnp.transpose(curl_ex_dy_yhi, (1, 2, 0))

        new_psi_hz_yhi = b_yrn * cpml_state.psi_hz_yhi + c_yrn * curl_ex_dy_yhi_t
        correction_hz_yhi = jnp.transpose(new_psi_hz_yhi, (2, 0, 1))
        hz = hz.at[:, -n:, :].add(coeff_h * correction_hz_yhi)
        kappa_corr_hz_yhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_ex_dy_yhi_t, (2, 0, 1))
        hz = hz.at[:, -n:, :].add(coeff_h * kappa_corr_hz_yhi)
    else:
        new_psi_hx_ylo = cpml_state.psi_hx_ylo
        new_psi_hx_yhi = cpml_state.psi_hx_yhi
        new_psi_hz_ylo = cpml_state.psi_hz_ylo
        new_psi_hz_yhi = cpml_state.psi_hz_yhi

    # =========================================================
    # Z-axis CPML: Hx correction (dEy/dz) and Hy correction (dEx/dz)
    # =========================================================

    if "z" in axes:
        # --- Z-lo: Hx correction from dEy/dz ---
        ey_zlo = state.ey[:, :, :n]
        ey_shifted_zlo = _shift_fwd(state.ey, 2)[:, :, :n]
        curl_ey_dz_zlo = (ey_shifted_zlo - ey_zlo) / dx

        curl_ey_dz_zlo_t = jnp.transpose(curl_ey_dz_zlo, (2, 0, 1))

        new_psi_hx_zlo = b_yn * cpml_state.psi_hx_zlo + c_yn * curl_ey_dz_zlo_t
        correction_hx_zlo = jnp.transpose(new_psi_hx_zlo, (1, 2, 0))
        hx = hx.at[:, :, :n].add(coeff_h * correction_hx_zlo)
        kappa_corr_hx_zlo = jnp.transpose((1.0 / k_yn - 1.0) * curl_ey_dz_zlo_t, (1, 2, 0))
        hx = hx.at[:, :, :n].add(coeff_h * kappa_corr_hx_zlo)

        # --- Z-hi: Hx correction from dEy/dz ---
        ey_zhi = state.ey[:, :, -n:]
        ey_shifted_zhi = _shift_fwd(state.ey, 2)[:, :, -n:]
        curl_ey_dz_zhi = (ey_shifted_zhi - ey_zhi) / dx

        curl_ey_dz_zhi_t = jnp.transpose(curl_ey_dz_zhi, (2, 0, 1))

        new_psi_hx_zhi = b_yrn * cpml_state.psi_hx_zhi + c_yrn * curl_ey_dz_zhi_t
        correction_hx_zhi = jnp.transpose(new_psi_hx_zhi, (1, 2, 0))
        hx = hx.at[:, :, -n:].add(coeff_h * correction_hx_zhi)
        kappa_corr_hx_zhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_ey_dz_zhi_t, (1, 2, 0))
        hx = hx.at[:, :, -n:].add(coeff_h * kappa_corr_hx_zhi)

        # --- Z-lo: Hy correction from dEx/dz ---
        ex_zlo = state.ex[:, :, :n]
        ex_shifted_zlo = _shift_fwd(state.ex, 2)[:, :, :n]
        curl_ex_dz_zlo = (ex_shifted_zlo - ex_zlo) / dx

        curl_ex_dz_zlo_t = jnp.transpose(curl_ex_dz_zlo, (2, 1, 0))

        new_psi_hy_zlo = b_yn * cpml_state.psi_hy_zlo + c_yn * curl_ex_dz_zlo_t
        correction_hy_zlo = jnp.transpose(new_psi_hy_zlo, (2, 1, 0))
        hy = hy.at[:, :, :n].add(-coeff_h * correction_hy_zlo)
        kappa_corr_hy_zlo = jnp.transpose((1.0 / k_yn - 1.0) * curl_ex_dz_zlo_t, (2, 1, 0))
        hy = hy.at[:, :, :n].add(-coeff_h * kappa_corr_hy_zlo)

        # --- Z-hi: Hy correction from dEx/dz ---
        ex_zhi = state.ex[:, :, -n:]
        ex_shifted_zhi = _shift_fwd(state.ex, 2)[:, :, -n:]
        curl_ex_dz_zhi = (ex_shifted_zhi - ex_zhi) / dx

        curl_ex_dz_zhi_t = jnp.transpose(curl_ex_dz_zhi, (2, 1, 0))

        new_psi_hy_zhi = b_yrn * cpml_state.psi_hy_zhi + c_yrn * curl_ex_dz_zhi_t
        correction_hy_zhi = jnp.transpose(new_psi_hy_zhi, (2, 1, 0))
        hy = hy.at[:, :, -n:].add(-coeff_h * correction_hy_zhi)
        kappa_corr_hy_zhi = jnp.transpose((1.0 / k_yrn - 1.0) * curl_ex_dz_zhi_t, (2, 1, 0))
        hy = hy.at[:, :, -n:].add(-coeff_h * kappa_corr_hy_zhi)
    else:
        new_psi_hx_zlo = cpml_state.psi_hx_zlo
        new_psi_hx_zhi = cpml_state.psi_hx_zhi
        new_psi_hy_zlo = cpml_state.psi_hy_zlo
        new_psi_hy_zhi = cpml_state.psi_hy_zhi

    state = state._replace(hx=hx, hy=hy, hz=hz)
    cpml_state = cpml_state._replace(
        # x-axis
        psi_hy_xlo=new_psi_hy_xlo,
        psi_hy_xhi=new_psi_hy_xhi,
        psi_hz_xlo=new_psi_hz_xlo,
        psi_hz_xhi=new_psi_hz_xhi,
        # y-axis
        psi_hx_ylo=new_psi_hx_ylo,
        psi_hx_yhi=new_psi_hx_yhi,
        psi_hz_ylo=new_psi_hz_ylo,
        psi_hz_yhi=new_psi_hz_yhi,
        # z-axis
        psi_hx_zlo=new_psi_hx_zlo,
        psi_hx_zhi=new_psi_hx_zhi,
        psi_hy_zlo=new_psi_hy_zlo,
        psi_hy_zhi=new_psi_hy_zhi,
    )

    return state, cpml_state
