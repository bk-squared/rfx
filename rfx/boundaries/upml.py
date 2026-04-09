"""Uniaxial PML (UPML) helpers for uniform-grid Yee updates.

Component-aware anisotropic PML using direct E/H conductivity with
material-scaled loss: σ_pml / (ε_r · ε₀).  This naturally gives stronger
damping in vacuum (correct) and gentler damping in dielectric (prevents
oscillation), matching the guided-mode impedance.

For vacuum cells where loss > 1 (Ca < 0), the loss is capped at 0.95
to keep Ca > 0.  This limits outermost-cell absorption but maintains
stability without D/B auxiliary fields.

Achieves SWR ~1.09 for eps=12 dielectric waveguide (vs CPML ~1.13,
Meep UPML ~1.01).  The remaining gap to Meep requires true D/B-field
UPML where Ca < 0 is stable.
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
) -> UPMLCoeffs:
    """Build static UPML coefficients for uniform-grid Yee updates.

    Per-component anisotropic damping:
    - E_x / H_x use σ from y + z PML
    - E_y / H_y use σ from x + z PML
    - E_z / H_z use σ from x + y PML

    Loss = σ_pml / (ε_r · ε₀) — material-scaled so dielectric cells get
    smooth decay (Ca > 0) while vacuum cells may hit the cap.
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

    eps_abs = materials.eps_r * jnp.float32(EPS_0)
    mu_abs = materials.mu_r * jnp.float32(MU_0)
    sigma_mat = materials.sigma.astype(jnp.float32)
    dt = jnp.float32(grid.dt)
    dx = jnp.float32(grid.dx)

    # Combined material + PML σ, divided by local ε_r·ε₀.
    # Capped at 0.95 to prevent Ca < 0 in vacuum outermost cells.
    sigma_ex = sigma_mat + sigma_y + sigma_z
    sigma_ey = sigma_mat + sigma_x + sigma_z
    sigma_ez = sigma_mat + sigma_x + sigma_y

    def _e_coeffs(sigma_comp):
        loss = sigma_comp * dt / (jnp.float32(2.0) * eps_abs)
        loss = jnp.minimum(loss, jnp.float32(0.95))
        denom = jnp.float32(1.0) + loss
        ca = (jnp.float32(1.0) - loss) / denom
        cb = (dt / eps_abs) / denom
        return ca.astype(jnp.float32), cb.astype(jnp.float32)

    ca_ex, cb_ex = _e_coeffs(sigma_ex)
    ca_ey, cb_ey = _e_coeffs(sigma_ey)
    ca_ez, cb_ez = _e_coeffs(sigma_ez)

    # Matched magnetic conductivity (impedance matched)
    sigma_hx = (sigma_y + sigma_z) * (mu_abs / eps_abs)
    sigma_hy = (sigma_x + sigma_z) * (mu_abs / eps_abs)
    sigma_hz = (sigma_x + sigma_y) * (mu_abs / eps_abs)

    def _h_coeffs(sigma_comp):
        loss = sigma_comp * dt / (jnp.float32(2.0) * mu_abs)
        loss = jnp.minimum(loss, jnp.float32(0.95))
        denom = jnp.float32(1.0) + loss
        da = (jnp.float32(1.0) - loss) / denom
        db = (dt / mu_abs) / denom
        return da.astype(jnp.float32), db.astype(jnp.float32)

    da_hx, db_hx = _h_coeffs(sigma_hx)
    da_hy, db_hy = _h_coeffs(sigma_hy)
    da_hz, db_hz = _h_coeffs(sigma_hz)

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
