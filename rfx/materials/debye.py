"""Debye dispersive materials via Auxiliary Differential Equation (ADE).

Debye model: ε(ω) = ε_∞ + Σ_p Δε_p / (1 + jωτ_p)

Each Debye pole p introduces an auxiliary polarization field P_p that
satisfies τ_p · dP_p/dt + P_p = ε₀ · Δε_p · E.

Semi-implicit (Crank-Nicolson) discretization:
    P_p^{n+1} = α_p · P_p^n + β_p · (E^{n+1} + E^n)

    α_p = (2τ_p - dt) / (2τ_p + dt)
    β_p = ε₀ · Δε_p · dt / (2τ_p + dt)

The E update becomes (including conductivity σ):
    E^{n+1} = Ca · E^n + Cb · curl(H^{n+1/2}) + Σ_p Cc_p · P_p^n

    γ     = ε₀·ε_∞ + Σ_p β_p + σ·dt/2
    Ca    = (ε₀·ε_∞ - Σ_p β_p - σ·dt/2) / γ
    Cb    = dt / γ
    Cc_p  = (1 - α_p) / γ

References:
    Taflove & Hagness, "Computational Electrodynamics", 3rd ed., Ch. 9
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from rfx.core.yee import EPS_0, FDTDState, MaterialArrays, _shift_bwd


class DebyePole(NamedTuple):
    """Single Debye pole parameters.

    delta_eps : float
        Permittivity contribution Δε (dimensionless).
    tau : float
        Relaxation time in seconds.
    """
    delta_eps: float
    tau: float


class DebyeCoeffs(NamedTuple):
    """Precomputed ADE update coefficients for all Debye poles.

    ca : (nx, ny, nz) — E decay coefficient
    cb : (nx, ny, nz) — curl(H) coupling coefficient
    cc : (n_poles, nx, ny, nz) — P^n coupling into E update
    alpha : (n_poles, nx, ny, nz) — P decay coefficient
    beta : (n_poles, nx, ny, nz) — E coupling into P update
    """
    ca: jnp.ndarray
    cb: jnp.ndarray
    cc: jnp.ndarray      # (n_poles, nx, ny, nz)
    alpha: jnp.ndarray   # (n_poles, nx, ny, nz)
    beta: jnp.ndarray    # (n_poles, nx, ny, nz)


class DebyeState(NamedTuple):
    """Auxiliary polarization fields for Debye ADE.

    Each pole has 3 polarization components (px, py, pz).
    Stored as (n_poles, nx, ny, nz) for each component.
    """
    px: jnp.ndarray  # (n_poles, nx, ny, nz)
    py: jnp.ndarray
    pz: jnp.ndarray


def init_debye(
    poles: list[DebyePole],
    materials: MaterialArrays,
    dt: float,
    mask: jnp.ndarray | list[jnp.ndarray] | tuple[jnp.ndarray, ...] | None = None,
) -> tuple[DebyeCoeffs, DebyeState]:
    """Initialize Debye ADE coefficients and auxiliary state.

    Parameters
    ----------
    poles : list of DebyePole
        Debye relaxation poles.
    materials : MaterialArrays
        Base material arrays (eps_r = ε_∞, sigma, mu_r).
    dt : float
        Timestep in seconds.
    mask : (nx, ny, nz) bool array or per-pole mask list, optional
        Where to apply Debye dispersion. If a list/tuple is provided,
        it must align one-to-one with ``poles``.

    Returns
    -------
    coeffs : DebyeCoeffs
    state : DebyeState
    """
    shape = materials.eps_r.shape
    n_poles = len(poles)

    eps_inf = materials.eps_r * EPS_0  # (nx, ny, nz)
    sigma = materials.sigma

    if isinstance(mask, (list, tuple)):
        if len(mask) != n_poles:
            raise ValueError(
                f"Expected {n_poles} Debye masks, got {len(mask)}"
            )
        pole_masks = [jnp.asarray(mask_i, dtype=bool) for mask_i in mask]
    else:
        shared_mask = None if mask is None else jnp.asarray(mask, dtype=bool)
        pole_masks = [shared_mask] * n_poles

    # Per-pole coefficients
    alpha_list = []
    beta_list = []
    for pole, pole_mask in zip(poles, pole_masks):
        tau = pole.tau
        de = pole.delta_eps
        a = (2.0 * tau - dt) / (2.0 * tau + dt)
        b = EPS_0 * de * dt / (2.0 * tau + dt)

        if pole_mask is not None:
            a_arr = jnp.where(pole_mask, a, 0.0)
            b_arr = jnp.where(pole_mask, b, 0.0)
        else:
            a_arr = jnp.full(shape, a, dtype=jnp.float32)
            b_arr = jnp.full(shape, b, dtype=jnp.float32)

        alpha_list.append(a_arr)
        beta_list.append(b_arr)

    alpha = jnp.stack(alpha_list)  # (n_poles, nx, ny, nz)
    beta = jnp.stack(beta_list)

    # Sum of beta across poles
    beta_sum = jnp.sum(beta, axis=0)  # (nx, ny, nz)

    # Modified update coefficients
    gamma = eps_inf + beta_sum + sigma * dt / 2.0
    # Guard against zero (vacuum cells with no Debye)
    safe_gamma = jnp.maximum(gamma, EPS_0 * 1e-10)

    ca = (eps_inf - beta_sum - sigma * dt / 2.0) / safe_gamma
    cb = dt / safe_gamma

    # Cc for each pole: (1 - alpha_p) / gamma
    cc_list = []
    for p in range(n_poles):
        cc_p = (1.0 - alpha[p]) / safe_gamma
        cc_list.append(cc_p)
    cc = jnp.stack(cc_list)  # (n_poles, nx, ny, nz)

    coeffs = DebyeCoeffs(ca=ca, cb=cb, cc=cc, alpha=alpha, beta=beta)

    # Zero-initialized polarization state
    p_zeros = jnp.zeros((n_poles,) + shape, dtype=jnp.float32)
    state = DebyeState(px=p_zeros, py=p_zeros.copy(), pz=p_zeros.copy())

    return coeffs, state


def update_e_debye(
    state: FDTDState,
    coeffs: DebyeCoeffs,
    debye_state: DebyeState,
    dt: float,
    dx: float,
    periodic: tuple = (False, False, False),
) -> tuple[FDTDState, DebyeState]:
    """E-field update with Debye ADE dispersion.

    Replaces update_e() when Debye materials are present.
    Handles conductivity (via ca/cb) and Debye polarization simultaneously.

    Update order:
    1. Compute curl(H) (backward differences)
    2. E^{n+1} = Ca·E^n + Cb·curl(H) + Σ_p Cc_p·P_p^n
    3. P_p^{n+1} = α_p·P_p^n + β_p·(E^{n+1} + E^n)
    """
    def bwd(arr, axis):
        if periodic[axis]:
            return jnp.roll(arr, 1, axis)
        return _shift_bwd(arr, axis)

    hx, hy, hz = state.hx, state.hy, state.hz
    ca, cb, cc = coeffs.ca, coeffs.cb, coeffs.cc
    alpha, beta = coeffs.alpha, coeffs.beta

    # curl(H) via backward differences
    curl_x = ((hz - bwd(hz, 1)) - (hy - bwd(hy, 2))) / dx
    curl_y = ((hx - bwd(hx, 2)) - (hz - bwd(hz, 0))) / dx
    curl_z = ((hy - bwd(hy, 0)) - (hx - bwd(hx, 1))) / dx

    # Save old E for P update
    ex_old, ey_old, ez_old = state.ex, state.ey, state.ez

    # E^{n+1} = Ca·E^n + Cb·curl(H) + Σ_p Cc_p·P_p^n
    ex_new = ca * ex_old + cb * curl_x + jnp.sum(cc * debye_state.px, axis=0)
    ey_new = ca * ey_old + cb * curl_y + jnp.sum(cc * debye_state.py, axis=0)
    ez_new = ca * ez_old + cb * curl_z + jnp.sum(cc * debye_state.pz, axis=0)

    # P_p^{n+1} = α_p·P_p^n + β_p·(E^{n+1} + E^n)
    px_new = alpha * debye_state.px + beta * (ex_new[None] + ex_old[None])
    py_new = alpha * debye_state.py + beta * (ey_new[None] + ey_old[None])
    pz_new = alpha * debye_state.pz + beta * (ez_new[None] + ez_old[None])

    new_fdtd = state._replace(
        ex=ex_new, ey=ey_new, ez=ez_new,
        step=state.step + 1,
    )
    new_debye = DebyeState(px=px_new, py=py_new, pz=pz_new)

    return new_fdtd, new_debye
