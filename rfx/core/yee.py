"""Yee cell FDTD update equations — pure JAX functions.

All functions are jax.jit-compatible and operate on explicit state arrays.
No hidden mutable state.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


class FDTDState(NamedTuple):
    """Complete FDTD simulation state at one timestep."""

    # Electric field components (Nx, Ny, Nz)
    ex: jnp.ndarray
    ey: jnp.ndarray
    ez: jnp.ndarray
    # Magnetic field components (Nx, Ny, Nz)
    hx: jnp.ndarray
    hy: jnp.ndarray
    hz: jnp.ndarray
    # Timestep counter
    step: jnp.ndarray


class MaterialArrays(NamedTuple):
    """Material property arrays on the grid."""

    # Relative permittivity (Nx, Ny, Nz) — used in E update
    eps_r: jnp.ndarray
    # Conductivity S/m (Nx, Ny, Nz) — used in E update (lossy)
    sigma: jnp.ndarray
    # Relative permeability (Nx, Ny, Nz) — used in H update
    mu_r: jnp.ndarray


def init_state(shape: tuple[int, int, int], *, field_dtype=jnp.float32) -> FDTDState:
    """Initialize zero-valued FDTD state.

    Parameters
    ----------
    shape : (Nx, Ny, Nz)
    field_dtype : jnp dtype
        Data type for field arrays.  Use ``jnp.float16`` for mixed-
        precision mode (material coefficients and accumulators stay
        float32; only field storage is reduced).
    """
    zeros = jnp.zeros(shape, dtype=field_dtype)
    return FDTDState(
        ex=zeros, ey=zeros, ez=zeros,
        hx=zeros, hy=zeros, hz=zeros,
        step=jnp.array(0, dtype=jnp.int32),
    )


def init_materials(shape: tuple[int, int, int]) -> MaterialArrays:
    """Initialize free-space material arrays."""
    ones = jnp.ones(shape, dtype=jnp.float32)
    return MaterialArrays(
        eps_r=ones,
        sigma=jnp.zeros(shape, dtype=jnp.float32),
        mu_r=ones,
    )


# Physical constants
EPS_0 = 8.854187817e-12  # F/m
MU_0 = 1.2566370614e-6   # H/m


def _shift_fwd(arr, axis):
    """arr[i+1] with zero at the last position (replaces roll(arr, -1, axis))."""
    pad_widths = [(0, 0)] * arr.ndim
    pad_widths[axis] = (0, 1)
    padded = jnp.pad(arr, pad_widths)
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(1, None)
    return padded[tuple(slices)]


def _shift_bwd(arr, axis):
    """arr[i-1] with zero at the first position (replaces roll(arr, +1, axis))."""
    pad_widths = [(0, 0)] * arr.ndim
    pad_widths[axis] = (1, 0)
    padded = jnp.pad(arr, pad_widths)
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(None, -1)
    return padded[tuple(slices)]


@partial(jax.jit, static_argnums=(4,))
def update_h(state: FDTDState, materials: MaterialArrays, dt: float, dx: float,
             periodic: tuple = (False, False, False)) -> FDTDState:
    """Magnetic field half-step update (Faraday's law).

    H^{n+1/2} = H^{n-1/2} - (dt / μ) * curl(E^n)

    Note: uses a single ``dx`` for all three axes (cubic cells: dx=dy=dz).
    For non-uniform z grids, use ``update_h_nu`` / ``update_e_nu`` instead.

    periodic: tuple of 3 bools selecting periodic boundary per axis (x, y, z).
    """
    def fwd(arr, axis):
        if periodic[axis]:
            return jnp.roll(arr, -1, axis)
        return _shift_fwd(arr, axis)

    # Upcast to float32 for arithmetic when using reduced-precision fields
    _fdtype = state.ex.dtype
    ex = state.ex.astype(jnp.float32)
    ey = state.ey.astype(jnp.float32)
    ez = state.ez.astype(jnp.float32)
    mu = materials.mu_r * MU_0

    # curl E components via forward differences (zero-padded, no wraparound)
    # dEz/dy - dEy/dz
    curl_x = (
        (fwd(ez, 1) - ez) / dx
        - (fwd(ey, 2) - ey) / dx
    )
    # dEx/dz - dEz/dx
    curl_y = (
        (fwd(ex, 2) - ex) / dx
        - (fwd(ez, 0) - ez) / dx
    )
    # dEy/dx - dEx/dy
    curl_z = (
        (fwd(ey, 0) - ey) / dx
        - (fwd(ex, 1) - ex) / dx
    )

    hx = (state.hx.astype(jnp.float32) - (dt / mu) * curl_x).astype(_fdtype)
    hy = (state.hy.astype(jnp.float32) - (dt / mu) * curl_y).astype(_fdtype)
    hz = (state.hz.astype(jnp.float32) - (dt / mu) * curl_z).astype(_fdtype)

    return state._replace(hx=hx, hy=hy, hz=hz)


@partial(jax.jit, static_argnums=(4,))
def update_e(state: FDTDState, materials: MaterialArrays, dt: float, dx: float,
             periodic: tuple = (False, False, False)) -> FDTDState:
    """Electric field full-step update (Ampere's law).

    For lossy media with conductivity σ:
    E^{n+1} = Ca * E^n + Cb * curl(H^{n+1/2})

    Ca = (1 - σ*dt/(2ε)) / (1 + σ*dt/(2ε))
    Cb = (dt/ε) / (1 + σ*dt/(2ε))

    periodic: tuple of 3 bools selecting periodic boundary per axis (x, y, z).
    """
    def bwd(arr, axis):
        if periodic[axis]:
            return jnp.roll(arr, 1, axis)
        return _shift_bwd(arr, axis)

    # Upcast to float32 for arithmetic when using reduced-precision fields
    _fdtype = state.ex.dtype
    hx = state.hx.astype(jnp.float32)
    hy = state.hy.astype(jnp.float32)
    hz = state.hz.astype(jnp.float32)
    eps = materials.eps_r * EPS_0
    sigma = materials.sigma

    # Lossy update coefficients
    sigma_dt_2eps = sigma * dt / (2.0 * eps)
    ca = (1.0 - sigma_dt_2eps) / (1.0 + sigma_dt_2eps)
    cb = (dt / eps) / (1.0 + sigma_dt_2eps)

    # curl H components via backward differences (zero-padded, no wraparound)
    # dHz/dy - dHy/dz
    curl_x = (
        (hz - bwd(hz, 1)) / dx
        - (hy - bwd(hy, 2)) / dx
    )
    # dHx/dz - dHz/dx
    curl_y = (
        (hx - bwd(hx, 2)) / dx
        - (hz - bwd(hz, 0)) / dx
    )
    # dHy/dx - dHx/dy
    curl_z = (
        (hy - bwd(hy, 0)) / dx
        - (hx - bwd(hx, 1)) / dx
    )

    ex = (ca * state.ex.astype(jnp.float32) + cb * curl_x).astype(_fdtype)
    ey = (ca * state.ey.astype(jnp.float32) + cb * curl_y).astype(_fdtype)
    ez = (ca * state.ez.astype(jnp.float32) + cb * curl_z).astype(_fdtype)

    return state._replace(
        ex=ex, ey=ey, ez=ez,
        step=state.step + 1,
    )


# ---------------------------------------------------------------------------
# Non-uniform mesh updates
# ---------------------------------------------------------------------------

def update_h_nu(state: FDTDState, materials: MaterialArrays, dt: float,
                inv_dx_h: jnp.ndarray, inv_dy_h: jnp.ndarray, inv_dz_h: jnp.ndarray,
                ) -> FDTDState:
    """H update for non-uniform Yee grid.

    H uses forward differences with MEAN spacing between adjacent cells.

    Parameters
    ----------
    inv_dx_h, inv_dy_h, inv_dz_h : (N,) arrays
        Pre-padded: inv_d_h[j] = 2/(d[j]+d[j+1]) for j<N-1, inv_d_h[N-1]=0.
    """
    _fdtype = state.ex.dtype
    ex = state.ex.astype(jnp.float32)
    ey = state.ey.astype(jnp.float32)
    ez = state.ez.astype(jnp.float32)
    mu = materials.mu_r * MU_0

    # Forward differences with same shape (zero-pad via _shift_fwd)
    curl_x = (
        (_shift_fwd(ez, 1) - ez) * inv_dy_h[None, :, None]
        - (_shift_fwd(ey, 2) - ey) * inv_dz_h[None, None, :]
    )
    curl_y = (
        (_shift_fwd(ex, 2) - ex) * inv_dz_h[None, None, :]
        - (_shift_fwd(ez, 0) - ez) * inv_dx_h[:, None, None]
    )
    curl_z = (
        (_shift_fwd(ey, 0) - ey) * inv_dx_h[:, None, None]
        - (_shift_fwd(ex, 1) - ex) * inv_dy_h[None, :, None]
    )

    hx = (state.hx.astype(jnp.float32) - (dt / mu) * curl_x).astype(_fdtype)
    hy = (state.hy.astype(jnp.float32) - (dt / mu) * curl_y).astype(_fdtype)
    hz = (state.hz.astype(jnp.float32) - (dt / mu) * curl_z).astype(_fdtype)

    return state._replace(hx=hx, hy=hy, hz=hz)


def update_e_nu(state: FDTDState, materials: MaterialArrays, dt: float,
                inv_dx: jnp.ndarray, inv_dy: jnp.ndarray, inv_dz: jnp.ndarray,
                ) -> FDTDState:
    """E update for non-uniform Yee grid.

    E uses backward differences with LOCAL cell spacing.

    Parameters
    ----------
    inv_dx, inv_dy, inv_dz : (N,) arrays — 1/dx[i] per cell
    """
    _fdtype = state.ex.dtype
    hx = state.hx.astype(jnp.float32)
    hy = state.hy.astype(jnp.float32)
    hz = state.hz.astype(jnp.float32)
    eps = materials.eps_r * EPS_0
    sigma = materials.sigma

    sigma_dt_2eps = sigma * dt / (2.0 * eps)
    ca = (1.0 - sigma_dt_2eps) / (1.0 + sigma_dt_2eps)
    cb = (dt / eps) / (1.0 + sigma_dt_2eps)

    # Backward differences with same shape (zero-pad via _shift_bwd)
    curl_x = (
        (hz - _shift_bwd(hz, 1)) * inv_dy[None, :, None]
        - (hy - _shift_bwd(hy, 2)) * inv_dz[None, None, :]
    )
    curl_y = (
        (hx - _shift_bwd(hx, 2)) * inv_dz[None, None, :]
        - (hz - _shift_bwd(hz, 0)) * inv_dx[:, None, None]
    )
    curl_z = (
        (hy - _shift_bwd(hy, 0)) * inv_dx[:, None, None]
        - (hx - _shift_bwd(hx, 1)) * inv_dy[None, :, None]
    )

    ex = (ca * state.ex.astype(jnp.float32) + cb * curl_x).astype(_fdtype)
    ey = (ca * state.ey.astype(jnp.float32) + cb * curl_y).astype(_fdtype)
    ez = (ca * state.ez.astype(jnp.float32) + cb * curl_z).astype(_fdtype)

    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


# ---------------------------------------------------------------------------
# Pre-computed update coefficients for high-throughput scan loops
# ---------------------------------------------------------------------------

class UpdateCoeffs(NamedTuple):
    """Pre-computed FDTD update coefficients.

    Eliminates per-step coefficient recomputation and optionally bakes in
    PEC boundary enforcement (zero coefficients at boundary cells) to
    remove the need for separate ``apply_pec()`` calls.

    Use :func:`precompute_coeffs` to build.
    """
    # H-field coefficient: dt / (mu_r * MU_0 * dx)  — shape (Nx, Ny, Nz)
    ch: jnp.ndarray
    # E-field decay coefficient  — per-component (Nx, Ny, Nz)
    ca_ex: jnp.ndarray
    ca_ey: jnp.ndarray
    ca_ez: jnp.ndarray
    # E-field curl coefficient (includes 1/dx)  — per-component (Nx, Ny, Nz)
    cb_ex: jnp.ndarray
    cb_ey: jnp.ndarray
    cb_ez: jnp.ndarray


def precompute_coeffs(
    materials: MaterialArrays,
    dt: float,
    dx: float,
    *,
    pec_axes: str = "",
) -> UpdateCoeffs:
    """Pre-compute all FDTD update coefficients.

    Parameters
    ----------
    materials : MaterialArrays
    dt, dx : float
    pec_axes : str
        Axes on which to bake PEC (zero tangential E) into the
        coefficients.  For example ``"xyz"`` zeros Ca/Cb at all 6
        boundary faces so that ``apply_pec()`` is no longer needed.

    Returns
    -------
    UpdateCoeffs
    """
    ch = jnp.float32(dt / (MU_0 * dx)) / materials.mu_r

    eps = materials.eps_r * jnp.float32(EPS_0)
    sigma = materials.sigma
    loss = sigma * jnp.float32(dt) / (jnp.float32(2.0) * eps)
    denom = jnp.float32(1.0) + loss
    ca = (jnp.float32(1.0) - loss) / denom
    cb_over_dx = (jnp.float32(dt) / eps) / (denom * jnp.float32(dx))

    # Start with isotropic coefficients.
    ca_ex = ca
    ca_ey = ca
    ca_ez = ca
    cb_ex = cb_over_dx
    cb_ey = cb_over_dx
    cb_ez = cb_over_dx

    # Bake PEC boundary enforcement into the coefficients by zeroing
    # Ca and Cb at boundary faces for tangential E components.
    if pec_axes:
        nx, ny, nz = materials.eps_r.shape
        if "x" in pec_axes:
            # Ey, Ez tangential on x-faces
            ca_ey = ca_ey.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
            ca_ez = ca_ez.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
            cb_ey = cb_ey.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
            cb_ez = cb_ez.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
        if "y" in pec_axes:
            # Ex, Ez tangential on y-faces
            ca_ex = ca_ex.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
            ca_ez = ca_ez.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
            cb_ex = cb_ex.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
            cb_ez = cb_ez.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
        if "z" in pec_axes:
            # Ex, Ey tangential on z-faces
            ca_ex = ca_ex.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
            ca_ey = ca_ey.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
            cb_ex = cb_ex.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
            cb_ey = cb_ey.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)

    return UpdateCoeffs(
        ch=ch,
        ca_ex=ca_ex, ca_ey=ca_ey, ca_ez=ca_ez,
        cb_ex=cb_ex, cb_ey=cb_ey, cb_ez=cb_ez,
    )


def update_h_fast(state: FDTDState, ch: jnp.ndarray) -> FDTDState:
    """H update using pre-computed coefficient ``ch = dt/(mu*dx)``.

    Avoids recomputing material coefficients each timestep.
    Non-periodic boundaries only (uses ``_shift_fwd``).
    """
    _fdtype = state.ex.dtype
    ex = state.ex.astype(jnp.float32)
    ey = state.ey.astype(jnp.float32)
    ez = state.ez.astype(jnp.float32)
    hx = (state.hx.astype(jnp.float32) - ch * ((_shift_fwd(ez, 1) - ez) - (_shift_fwd(ey, 2) - ey))).astype(_fdtype)
    hy = (state.hy.astype(jnp.float32) - ch * ((_shift_fwd(ex, 2) - ex) - (_shift_fwd(ez, 0) - ez))).astype(_fdtype)
    hz = (state.hz.astype(jnp.float32) - ch * ((_shift_fwd(ey, 0) - ey) - (_shift_fwd(ex, 1) - ex))).astype(_fdtype)
    return state._replace(hx=hx, hy=hy, hz=hz)


def update_e_fast(
    state: FDTDState,
    ca_ex: jnp.ndarray, ca_ey: jnp.ndarray, ca_ez: jnp.ndarray,
    cb_ex: jnp.ndarray, cb_ey: jnp.ndarray, cb_ez: jnp.ndarray,
) -> FDTDState:
    """E update using pre-computed per-component coefficients.

    ``ca_*`` and ``cb_*`` already include PEC zeroing when built with
    :func:`precompute_coeffs` and ``pec_axes``, so no separate
    ``apply_pec()`` call is needed.

    Non-periodic boundaries only (uses ``_shift_bwd``).
    """
    _fdtype = state.ex.dtype
    hx = state.hx.astype(jnp.float32)
    hy = state.hy.astype(jnp.float32)
    hz = state.hz.astype(jnp.float32)
    ex = (ca_ex * state.ex.astype(jnp.float32) + cb_ex * ((hz - _shift_bwd(hz, 1)) - (hy - _shift_bwd(hy, 2)))).astype(_fdtype)
    ey = (ca_ey * state.ey.astype(jnp.float32) + cb_ey * ((hx - _shift_bwd(hx, 2)) - (hz - _shift_bwd(hz, 0)))).astype(_fdtype)
    ez = (ca_ez * state.ez.astype(jnp.float32) + cb_ez * ((hy - _shift_bwd(hy, 0)) - (hx - _shift_bwd(hx, 1)))).astype(_fdtype)
    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


def update_he_fast(state: FDTDState, coeffs: UpdateCoeffs) -> FDTDState:
    """Combined H + E update using pre-computed :class:`UpdateCoeffs`.

    Performs a full leapfrog step (H half-step then E full-step) with
    PEC baked into the coefficients.  This is the fastest path for the
    common case of non-periodic boundaries with uniform mesh.
    """
    _fdtype = state.ex.dtype
    # --- H update (upcast to float32 for arithmetic) ---
    ex = state.ex.astype(jnp.float32)
    ey = state.ey.astype(jnp.float32)
    ez = state.ez.astype(jnp.float32)
    ch = coeffs.ch
    hx = (state.hx.astype(jnp.float32) - ch * ((_shift_fwd(ez, 1) - ez) - (_shift_fwd(ey, 2) - ey))).astype(_fdtype)
    hy = (state.hy.astype(jnp.float32) - ch * ((_shift_fwd(ex, 2) - ex) - (_shift_fwd(ez, 0) - ez))).astype(_fdtype)
    hz = (state.hz.astype(jnp.float32) - ch * ((_shift_fwd(ey, 0) - ey) - (_shift_fwd(ex, 1) - ex))).astype(_fdtype)
    # --- E update (with PEC baked into coefficients) ---
    # Upcast newly computed H fields back to float32 for curl computation
    hx_f = hx.astype(jnp.float32)
    hy_f = hy.astype(jnp.float32)
    hz_f = hz.astype(jnp.float32)
    ex = (coeffs.ca_ex * ex + coeffs.cb_ex * ((hz_f - _shift_bwd(hz_f, 1)) - (hy_f - _shift_bwd(hy_f, 2)))).astype(_fdtype)
    ey = (coeffs.ca_ey * ey + coeffs.cb_ey * ((hx_f - _shift_bwd(hx_f, 2)) - (hz_f - _shift_bwd(hz_f, 0)))).astype(_fdtype)
    ez = (coeffs.ca_ez * ez + coeffs.cb_ez * ((hy_f - _shift_bwd(hy_f, 0)) - (hx_f - _shift_bwd(hx_f, 1)))).astype(_fdtype)
    return FDTDState(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz,
                     step=state.step + 1)


@jax.jit
def update_e_nu_aniso(state: FDTDState, materials: MaterialArrays,
                      eps_ex: jnp.ndarray, eps_ey: jnp.ndarray, eps_ez: jnp.ndarray,
                      dt: float,
                      inv_dx: jnp.ndarray, inv_dy: jnp.ndarray, inv_dz: jnp.ndarray,
                      ) -> FDTDState:
    """Non-uniform E update with per-component anisotropic permittivity.

    Same backward-difference structure as :func:`update_e_nu` (per-cell
    spacing via ``inv_dx``/``inv_dy``/``inv_dz``) but uses three separate
    permittivity arrays (Ex, Ey, Ez) so subpixel-smoothing can be applied
    on a non-uniform mesh. ``materials.sigma`` is still applied
    isotropically (matches the uniform-path :func:`update_e_aniso`).
    """
    _fdtype = state.ex.dtype
    hx = state.hx.astype(jnp.float32)
    hy = state.hy.astype(jnp.float32)
    hz = state.hz.astype(jnp.float32)
    sigma = materials.sigma

    abs_eps_ex = eps_ex * EPS_0
    abs_eps_ey = eps_ey * EPS_0
    abs_eps_ez = eps_ez * EPS_0

    loss_ex = sigma * dt / (2.0 * abs_eps_ex)
    ca_ex = (1.0 - loss_ex) / (1.0 + loss_ex)
    cb_ex = (dt / abs_eps_ex) / (1.0 + loss_ex)

    loss_ey = sigma * dt / (2.0 * abs_eps_ey)
    ca_ey = (1.0 - loss_ey) / (1.0 + loss_ey)
    cb_ey = (dt / abs_eps_ey) / (1.0 + loss_ey)

    loss_ez = sigma * dt / (2.0 * abs_eps_ez)
    ca_ez = (1.0 - loss_ez) / (1.0 + loss_ez)
    cb_ez = (dt / abs_eps_ez) / (1.0 + loss_ez)

    # Backward differences with per-cell inv-spacing (mirrors update_e_nu)
    curl_x = (
        (hz - _shift_bwd(hz, 1)) * inv_dy[None, :, None]
        - (hy - _shift_bwd(hy, 2)) * inv_dz[None, None, :]
    )
    curl_y = (
        (hx - _shift_bwd(hx, 2)) * inv_dz[None, None, :]
        - (hz - _shift_bwd(hz, 0)) * inv_dx[:, None, None]
    )
    curl_z = (
        (hy - _shift_bwd(hy, 0)) * inv_dx[:, None, None]
        - (hx - _shift_bwd(hx, 1)) * inv_dy[None, :, None]
    )

    ex = (ca_ex * state.ex.astype(jnp.float32) + cb_ex * curl_x).astype(_fdtype)
    ey = (ca_ey * state.ey.astype(jnp.float32) + cb_ey * curl_y).astype(_fdtype)
    ez = (ca_ez * state.ez.astype(jnp.float32) + cb_ez * curl_z).astype(_fdtype)

    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


@partial(jax.jit, static_argnums=(7,))
def update_e_aniso(state: FDTDState, materials: MaterialArrays,
                   eps_ex: jnp.ndarray, eps_ey: jnp.ndarray, eps_ez: jnp.ndarray,
                   dt: float, dx: float,
                   periodic: tuple = (False, False, False)) -> FDTDState:
    """Electric field update with per-component anisotropic permittivity.

    Same as :func:`update_e` but uses separate permittivity arrays for
    each E-field component (Ex, Ey, Ez) to support subpixel smoothing.

    The conductivity from ``materials.sigma`` is still applied isotropically.

    Parameters
    ----------
    state : FDTDState
    materials : MaterialArrays
        Used only for ``sigma`` (conductivity).
    eps_ex, eps_ey, eps_ez : jnp.ndarray
        Per-component relative permittivity arrays, each of shape (Nx, Ny, Nz).
    dt, dx : float
    periodic : tuple of 3 bools
    """
    def bwd(arr, axis):
        if periodic[axis]:
            return jnp.roll(arr, 1, axis)
        return _shift_bwd(arr, axis)

    _fdtype = state.ex.dtype
    hx = state.hx.astype(jnp.float32)
    hy = state.hy.astype(jnp.float32)
    hz = state.hz.astype(jnp.float32)
    sigma = materials.sigma

    # Per-component absolute permittivity
    abs_eps_ex = eps_ex * EPS_0
    abs_eps_ey = eps_ey * EPS_0
    abs_eps_ez = eps_ez * EPS_0

    # Per-component lossy update coefficients
    loss_ex = sigma * dt / (2.0 * abs_eps_ex)
    ca_ex = (1.0 - loss_ex) / (1.0 + loss_ex)
    cb_ex = (dt / abs_eps_ex) / (1.0 + loss_ex)

    loss_ey = sigma * dt / (2.0 * abs_eps_ey)
    ca_ey = (1.0 - loss_ey) / (1.0 + loss_ey)
    cb_ey = (dt / abs_eps_ey) / (1.0 + loss_ey)

    loss_ez = sigma * dt / (2.0 * abs_eps_ez)
    ca_ez = (1.0 - loss_ez) / (1.0 + loss_ez)
    cb_ez = (dt / abs_eps_ez) / (1.0 + loss_ez)

    # curl H (same as update_e)
    curl_x = (
        (hz - bwd(hz, 1)) / dx
        - (hy - bwd(hy, 2)) / dx
    )
    curl_y = (
        (hx - bwd(hx, 2)) / dx
        - (hz - bwd(hz, 0)) / dx
    )
    curl_z = (
        (hy - bwd(hy, 0)) / dx
        - (hx - bwd(hx, 1)) / dx
    )

    ex = (ca_ex * state.ex.astype(jnp.float32) + cb_ex * curl_x).astype(_fdtype)
    ey = (ca_ey * state.ey.astype(jnp.float32) + cb_ey * curl_y).astype(_fdtype)
    ez = (ca_ez * state.ez.astype(jnp.float32) + cb_ez * curl_z).astype(_fdtype)

    return state._replace(
        ex=ex, ey=ey, ez=ez,
        step=state.step + 1,
    )
