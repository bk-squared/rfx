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


def init_state(shape: tuple[int, int, int]) -> FDTDState:
    """Initialize zero-valued FDTD state."""
    zeros = jnp.zeros(shape, dtype=jnp.float32)
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

    periodic: tuple of 3 bools selecting periodic boundary per axis (x, y, z).
    """
    def fwd(arr, axis):
        if periodic[axis]:
            return jnp.roll(arr, -1, axis)
        return _shift_fwd(arr, axis)

    ex, ey, ez = state.ex, state.ey, state.ez
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

    hx = state.hx - (dt / mu) * curl_x
    hy = state.hy - (dt / mu) * curl_y
    hz = state.hz - (dt / mu) * curl_z

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

    hx, hy, hz = state.hx, state.hy, state.hz
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

    ex = ca * state.ex + cb * curl_x
    ey = ca * state.ey + cb * curl_y
    ez = ca * state.ez + cb * curl_z

    return state._replace(
        ex=ex, ey=ey, ez=ez,
        step=state.step + 1,
    )


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

    hx, hy, hz = state.hx, state.hy, state.hz
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

    ex = ca_ex * state.ex + cb_ex * curl_x
    ey = ca_ey * state.ey + cb_ey * curl_y
    ez = ca_ez * state.ez + cb_ez * curl_z

    return state._replace(
        ex=ex, ey=ey, ez=ez,
        step=state.step + 1,
    )
