"""Lorentz and Drude dispersive materials via ADE.

Lorentz model: ε(ω) = ε_∞ + Σ_p κ_p / (ω₀_p² - ω² + 2jδ_pω)

Drude model is Lorentz with ω₀ = 0:
    ε(ω) = ε_∞ - ω_p² / (ω² + jγω)

Each pole introduces a second-order auxiliary polarization P_p:
    d²P_p/dt² + 2δ_p dP_p/dt + ω₀_p² P_p = ε₀ κ_p E

Discretized (central difference + Crank-Nicolson damping):
    P^{n+1} = a_p P^n + b_p P^{n-1} + c_p E^n

    a_p = (2 - ω₀²Δt²) / (1 + δΔt)
    b_p = -(1 - δΔt) / (1 + δΔt)
    c_p = ε₀ κ_p Δt² / (1 + δΔt)

References:
    Taflove & Hagness, Ch. 9
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, FDTDState, _shift_bwd


class LorentzPole(NamedTuple):
    """Single Lorentz oscillator pole.

    omega_0 : float
        Resonant angular frequency (rad/s). Set to 0 for Drude.
    delta : float
        Damping coefficient (rad/s).
    kappa : float
        Coupling strength = Δε · ω₀² for Lorentz, ω_p² for Drude.
    """
    omega_0: float
    delta: float
    kappa: float


def drude_pole(omega_p: float, gamma: float) -> LorentzPole:
    """Create a Drude pole from plasma frequency and collision rate.

    Parameters
    ----------
    omega_p : float
        Plasma frequency (rad/s).
    gamma : float
        Collision rate (rad/s).
    """
    return LorentzPole(omega_0=0.0, delta=gamma / 2.0, kappa=omega_p ** 2)


def lorentz_pole(delta_eps: float, omega_0: float, delta: float) -> LorentzPole:
    """Create a Lorentz pole from physical parameters.

    Parameters
    ----------
    delta_eps : float
        Permittivity contribution Δε (dimensionless).
    omega_0 : float
        Resonant angular frequency (rad/s).
    delta : float
        Damping coefficient (rad/s).
    """
    return LorentzPole(omega_0=omega_0, delta=delta, kappa=delta_eps * omega_0 ** 2)


class LorentzCoeffs(NamedTuple):
    """Precomputed ADE coefficients for Lorentz/Drude poles.

    ca, cb : (nx, ny, nz) — E update coefficients
    a, b, c : (n_poles, nx, ny, nz) — P recurrence coefficients
    cc : (nx, ny, nz) — P→E coupling: 1/γ
    """
    ca: jnp.ndarray
    cb: jnp.ndarray
    a: jnp.ndarray
    b: jnp.ndarray
    c: jnp.ndarray
    cc: jnp.ndarray


class LorentzState(NamedTuple):
    """Auxiliary polarization state (current and previous step).

    p_{x,y,z} : (n_poles, nx, ny, nz) — P^n
    p_{x,y,z}_prev : (n_poles, nx, ny, nz) — P^{n-1}
    """
    px: jnp.ndarray
    py: jnp.ndarray
    pz: jnp.ndarray
    px_prev: jnp.ndarray
    py_prev: jnp.ndarray
    pz_prev: jnp.ndarray


def init_lorentz(
    poles: list[LorentzPole],
    materials,
    dt: float,
    mask: jnp.ndarray | list[jnp.ndarray] | tuple[jnp.ndarray, ...] | None = None,
) -> tuple[LorentzCoeffs, LorentzState]:
    """Initialize Lorentz/Drude ADE coefficients and state.

    Parameters
    ----------
    poles : list of LorentzPole
    materials : MaterialArrays (eps_r = ε_∞)
    dt : float
    mask : optional spatial mask

    Returns
    -------
    (LorentzCoeffs, LorentzState)
    """
    shape = materials.eps_r.shape
    n_poles = len(poles)

    if isinstance(mask, (list, tuple)):
        if len(mask) != n_poles:
            raise ValueError(
                f"Expected {n_poles} Lorentz masks, got {len(mask)}"
            )
        pole_masks = [jnp.asarray(mask_i, dtype=bool) for mask_i in mask]
    else:
        shared_mask = None if mask is None else jnp.asarray(mask, dtype=bool)
        pole_masks = [shared_mask] * n_poles

    eps_inf = materials.eps_r * EPS_0
    sigma = materials.sigma

    a_list, b_list, c_list = [], [], []

    for pole, pole_mask in zip(poles, pole_masks):
        w0, d, k = pole.omega_0, pole.delta, pole.kappa
        denom = 1.0 + d * dt

        a_val = (2.0 - w0 ** 2 * dt ** 2) / denom
        b_val = -(1.0 - d * dt) / denom
        c_val = EPS_0 * k * dt ** 2 / denom

        if pole_mask is not None:
            a_arr = jnp.where(pole_mask, a_val, 0.0)
            b_arr = jnp.where(pole_mask, b_val, 0.0)
            c_arr = jnp.where(pole_mask, c_val, 0.0)
        else:
            a_arr = jnp.full(shape, a_val, dtype=jnp.float32)
            b_arr = jnp.full(shape, b_val, dtype=jnp.float32)
            c_arr = jnp.full(shape, c_val, dtype=jnp.float32)

        a_list.append(a_arr)
        b_list.append(b_arr)
        c_list.append(c_arr)

    a = jnp.stack(a_list)
    b = jnp.stack(b_list)
    c = jnp.stack(c_list)

    gamma = eps_inf + sigma * dt / 2.0
    safe_gamma = jnp.maximum(gamma, EPS_0 * 1e-10)

    ca = (eps_inf - sigma * dt / 2.0) / safe_gamma
    cb = dt / safe_gamma
    cc = 1.0 / safe_gamma

    coeffs = LorentzCoeffs(ca=ca, cb=cb, a=a, b=b, c=c, cc=cc)

    zeros = jnp.zeros((n_poles,) + shape, dtype=jnp.float32)
    state = LorentzState(
        px=zeros, py=zeros.copy(), pz=zeros.copy(),
        px_prev=zeros.copy(), py_prev=zeros.copy(), pz_prev=zeros.copy(),
    )

    return coeffs, state


def update_e_lorentz(
    state: FDTDState,
    coeffs: LorentzCoeffs,
    lor_state: LorentzState,
    dt: float,
    dx: float,
    periodic: tuple = (False, False, False),
) -> tuple[FDTDState, LorentzState]:
    """E-field update with Lorentz/Drude ADE dispersion.

    Update order:
    1. P^{n+1} = a P^n + b P^{n-1} + c E^n  (explicit)
    2. E^{n+1} = Ca E^n + Cb curl(H) - Cc Σ(P^{n+1} - P^n)
    """
    def bwd(arr, axis):
        if periodic[axis]:
            return jnp.roll(arr, 1, axis)
        return _shift_bwd(arr, axis)

    hx, hy, hz = state.hx, state.hy, state.hz
    ca, cb, cc = coeffs.ca, coeffs.cb, coeffs.cc
    a, b, c = coeffs.a, coeffs.b, coeffs.c

    # curl(H)
    curl_x = ((hz - bwd(hz, 1)) - (hy - bwd(hy, 2))) / dx
    curl_y = ((hx - bwd(hx, 2)) - (hz - bwd(hz, 0))) / dx
    curl_z = ((hy - bwd(hy, 0)) - (hx - bwd(hx, 1))) / dx

    # P^{n+1} = a P^n + b P^{n-1} + c E^n (per pole)
    px_new = a * lor_state.px + b * lor_state.px_prev + c * state.ex[None]
    py_new = a * lor_state.py + b * lor_state.py_prev + c * state.ey[None]
    pz_new = a * lor_state.pz + b * lor_state.pz_prev + c * state.ez[None]

    # ΔP = P^{n+1} - P^n, summed over poles
    dpx = jnp.sum(px_new - lor_state.px, axis=0)
    dpy = jnp.sum(py_new - lor_state.py, axis=0)
    dpz = jnp.sum(pz_new - lor_state.pz, axis=0)

    # E^{n+1} = Ca E^n + Cb curl(H) - Cc ΔP
    ex_new = ca * state.ex + cb * curl_x - cc * dpx
    ey_new = ca * state.ey + cb * curl_y - cc * dpy
    ez_new = ca * state.ez + cb * curl_z - cc * dpz

    new_fdtd = state._replace(
        ex=ex_new, ey=ey_new, ez=ez_new,
        step=state.step + 1,
    )
    new_lor = LorentzState(
        px=px_new, py=py_new, pz=pz_new,
        px_prev=lor_state.px, py_prev=lor_state.py, pz_prev=lor_state.pz,
    )

    return new_fdtd, new_lor
