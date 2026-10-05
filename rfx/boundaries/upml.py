"""Uniaxial PML (UPML) with D/B-equivalent formulation.

Two corrections over the original implementation:

1. **Half-cell offset**: σ_E evaluated at Yee E-positions (half-cell inside),
   σ_H evaluated at Yee H-positions (cell boundary). Prevents impedance
   mismatch inside PML that causes spurious reflections.

2. **No discrete scaling**: Uses textbook σ_max directly
   (``-ln(R)·(m+1)/(2·η·d)``). The previous n/2 factor caused 5x stronger
   damping than Meep, draining guided mode energy.

Design note — parallel component: E_x in x-PML has no inverse
stretching (which would require D-field storage).  The current approach
relies on indirect attenuation through curl coupling, the same strategy
used by CPML.  For uniform Cartesian grids at typical PML depths (8–12
layers), this is adequate: self-transmittance 0.995, integrated
absorption matches Meep to ratio 1.000000.  Full D/B split-field UPML
(6 extra 3-D arrays, ~50 % memory increase) would only be needed for
non-uniform meshes, very thick PML, or highly oblique corner incidence.

Per-component anisotropic damping:
  E_x / H_x: perpendicular σ from y + z axes
  E_y / H_y: perpendicular σ from x + z axes
  E_z / H_z: perpendicular σ from x + y axes
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.boundaries.cpml import _get_axis_cell_sizes
from rfx.core.yee import (EPS_0, MU_0, FDTDState, MaterialArrays, curl_h,
                          _shift_fwd, cell_owned_component_materials,
                          si_value_eps_r_grad)


class UPMLCoeffs(NamedTuple):
    """Precomputed component-aware UPML update coefficients.

    The ``cb_*`` / ``db_*`` coefficients intentionally exclude the stencil
    cell-size factor; per-axis inverse spacings below are applied to each
    curl component separately so the coefficients work on both uniform
    and non-uniform grids.  On a uniform grid the ``inv_*`` fields are
    scalars and the computation is bit-identical to the pre-split path.
    """
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
    inv_dx: jnp.ndarray    # E-position 1/dx (scalar or (nx,1,1))
    inv_dy: jnp.ndarray    # E-position 1/dy (scalar or (1,ny,1))
    inv_dz: jnp.ndarray    # E-position 1/dz (scalar or (1,1,nz))
    inv_dx_h: jnp.ndarray  # H-position 2/(dx[i]+dx[i+1])
    inv_dy_h: jnp.ndarray  # H-position
    inv_dz_h: jnp.ndarray  # H-position


def _sigma_profile_1d(
    n_layers: int,
    dt: float,
    dx: float,
    order: int = 2,
    R_asymptotic: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute graded σ at E and H positions (half-cell offset).

    Returns (sigma_E, sigma_H) arrays of shape (n_layers,).

    σ_E[i] at normalized position (n - 0.5 - i) / n  (half-cell inside).
    σ_H[i] at normalized position (n - i) / n         (cell boundary).
    """
    eta = np.sqrt(MU_0 / EPS_0)
    d = n_layers * dx
    sigma_max = -np.log(R_asymptotic) * (order + 1) / (2.0 * eta * d)
    # NO n/2 scaling — textbook formula directly

    u_E = np.clip((n_layers - 0.5 - np.arange(n_layers)) / n_layers, 0, 1)
    sigma_E = sigma_max * u_E ** order

    u_H = np.clip((n_layers - np.arange(n_layers)) / n_layers, 0, 1)
    sigma_H = sigma_max * u_H ** order

    return sigma_E.astype(np.float64), sigma_H.astype(np.float64)


def _axis_sigma_E_H(grid, axis: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (σ_E, σ_H) graded profiles for one axis over the full grid."""
    n = grid.cpml_layers
    if n <= 0:
        z = jnp.zeros(grid.shape, dtype=jnp.float32)
        return z, z

    # UPML runs on the uniform lane only (graded refuses it), where lo == hi.
    dx_x, _, dx_y, _, dz_lo, dz_hi = _get_axis_cell_sizes(grid)
    pec_faces = getattr(grid, "pec_faces", None) or set()
    pmc_faces = getattr(grid, "pmc_faces", None) or set()

    def _build(dx_cell, pad_lo, pad_hi, lo_face, hi_face, set_axis):
        sig_E_1d, sig_H_1d = _sigma_profile_1d(n, grid.dt, dx_cell)
        sE = jnp.zeros(grid.shape, dtype=jnp.float32)
        sH = jnp.zeros(grid.shape, dtype=jnp.float32)
        sE_lo = jnp.array(sig_E_1d, dtype=jnp.float32)
        sH_lo = jnp.array(sig_H_1d, dtype=jnp.float32)
        sE_hi = jnp.flip(sE_lo)
        sH_hi = jnp.flip(sH_lo)
        if (pad_lo > 0 and lo_face not in pec_faces
                and lo_face not in pmc_faces):
            sE = set_axis(sE, slice(None, pad_lo), sE_lo[-pad_lo:])
            sH = set_axis(sH, slice(None, pad_lo), sH_lo[-pad_lo:])
        if (pad_hi > 0 and hi_face not in pec_faces
                and hi_face not in pmc_faces):
            sE = set_axis(sE, slice(-pad_hi, None), sE_hi[:pad_hi])
            sH = set_axis(sH, slice(-pad_hi, None), sH_hi[:pad_hi])
        return sE, sH

    if axis == "x":
        def set_x(arr, sl, vals):
            return arr.at[sl, :, :].set(vals[:, None, None])
        return _build(dx_x, grid.pad_x_lo, grid.pad_x_hi,
                      "x_lo", "x_hi", set_x)
    if axis == "y":
        def set_y(arr, sl, vals):
            return arr.at[:, sl, :].set(vals[None, :, None])
        return _build(dx_y, grid.pad_y_lo, grid.pad_y_hi,
                      "y_lo", "y_hi", set_y)
    if axis == "z":
        if grid.pad_z_lo <= 0 and grid.pad_z_hi <= 0:
            z = jnp.zeros(grid.shape, dtype=jnp.float32)
            return z, z
        sig_E_lo, sig_H_lo = _sigma_profile_1d(n, grid.dt, dz_lo)
        sig_E_hi, sig_H_hi = _sigma_profile_1d(n, grid.dt, dz_hi)
        sE = jnp.zeros(grid.shape, dtype=jnp.float32)
        sH = jnp.zeros(grid.shape, dtype=jnp.float32)
        if (grid.pad_z_lo > 0 and "z_lo" not in pec_faces
                and "z_lo" not in pmc_faces):
            _nlo = grid.pad_z_lo
            sE = sE.at[:, :, :_nlo].set(
                jnp.array(sig_E_lo, dtype=jnp.float32)[-_nlo:][None, None, :])
            sH = sH.at[:, :, :_nlo].set(
                jnp.array(sig_H_lo, dtype=jnp.float32)[-_nlo:][None, None, :])
        if (grid.pad_z_hi > 0 and "z_hi" not in pec_faces
                and "z_hi" not in pmc_faces):
            _nhi = grid.pad_z_hi
            sE = sE.at[:, :, -_nhi:].set(
                jnp.flip(jnp.array(sig_E_hi, dtype=jnp.float32))[:_nhi][None, None, :])
            sH = sH.at[:, :, -_nhi:].set(
                jnp.flip(jnp.array(sig_H_hi, dtype=jnp.float32))[:_nhi][None, None, :])
        return sE, sH
    raise ValueError(f"Unsupported axis {axis!r}")


def _upml_e_coeffs_si(sigma_perp, eps_r, sigma_mat, dt):
    """``(Ca, Cb)`` of the UPML E update, SI spelling.

    ``sigma_perp`` is the PML's perpendicular conductivity (its loss is
    ``sigma_perp*dt/(2*EPS_0)``, material-independent), ``eps_r`` and
    ``sigma_mat`` the material's. ``dt`` is a float32 scalar.
    """
    eps_0 = jnp.float32(EPS_0)
    eps_abs = eps_r * eps_0
    loss_pml = sigma_perp * dt / (jnp.float32(2.0) * eps_0)
    loss_mat = sigma_mat * dt / (jnp.float32(2.0) * eps_abs)
    loss = loss_pml + loss_mat
    denom = jnp.float32(1.0) + loss
    ca = (jnp.float32(1.0) - loss) / denom
    cb = (dt / eps_abs) / denom
    return ca.astype(jnp.float32), cb.astype(jnp.float32)


def _upml_e_coeffs_eps_r(sigma_perp, eps_r, sigma_mat, dt):
    """:func:`_upml_e_coeffs_si` written in eps_r units (#1357).

    ``k = dt/(2*EPS_0)``: ``loss = sigma_perp*k + sigma_mat*k/eps_r`` and
    ``Cb = (dt/EPS_0) / (eps_r*(1 + sigma_perp*k) + sigma_mat*k)``, the SI
    ``(dt/eps_abs)/(1 + loss)`` with ``eps_abs = eps_r*EPS_0`` cancelled, so
    no SI-sized factor multiplies a cotangent.
    """
    k = dt / (2.0 * EPS_0)
    loss_pml = sigma_perp * k
    loss_mat = sigma_mat * k / eps_r
    loss = loss_pml + loss_mat
    ca = (1.0 - loss) / (1.0 + loss)
    cb = (dt / EPS_0) / (eps_r * (1.0 + loss_pml) + sigma_mat * k)
    return ca.astype(jnp.float32), cb.astype(jnp.float32)


def init_upml(
    grid,
    materials: MaterialArrays,
    *,
    axes: str = "xyz",
    aniso_eps=None,
) -> UPMLCoeffs:
    """Build static UPML coefficients for uniform-grid Yee updates.

    D/B-equivalent: PML loss is material-independent (σ/ε₀).
    Separate σ_E / σ_H with half-cell offset for impedance matching.
    No n/2 scaling — textbook σ_max.
    """
    if getattr(grid, "kappa_max", None) not in (None, 1):
        raise NotImplementedError(
            "cpml_kappa_max != 1 is not supported when building the UPML absorber: "
            "UPML does not read kappa. Use boundary='cpml', or set cpml_kappa_max=1."
        )
    z32 = jnp.zeros(grid.shape, dtype=jnp.float32)

    def _get_sigma(axis):
        if getattr(grid, f"pad_{axis}", 0) > 0 and axis in axes:
            return _axis_sigma_E_H(grid, axis)
        return z32, z32

    sEx, sHx = _get_sigma("x")
    sEy, sHy = _get_sigma("y")
    sEz, sHz = _get_sigma("z")

    from rfx.core.yee import component_h_materials
    from rfx.sources.wire_radius import require_radius_update
    require_radius_update(materials, lane="UPML", unsupported=True)
    mu_abs = tuple(m * jnp.float32(MU_0) for m in component_h_materials(
        materials, periodic=tuple(a in getattr(grid, "periodic_axes", "") for a in "xyz"),
        cell_sizes=(grid.dx_arr, grid.dy_arr, grid.dz)
        if hasattr(grid, "dx_arr") else None))
    # #1236: a lumped element (a port's load, an RLC R or C) loads its own E
    # edge only. This lane's E coefficients stay CELL-owned for the volume --
    # #1210's four-cell edge average was not carried into UPML, and doing it
    # here would move every inhomogeneous UPML result, a separate change --
    # so each component takes the cell total minus the stamps that belong to
    # the other two components, the rule the distributed slab update uses.
    # With no lumped record the three entries ARE materials.eps_r / .sigma.
    eps_r_c, sigma_c = cell_owned_component_materials(materials)
    sigma_mat_c = tuple(s_.astype(jnp.float32) for s_ in sigma_c)
    dt = jnp.float32(grid.dt)

    # Per-axis inverse cell-size broadcasts.  On NonUniformGrid these are
    # arrays we reshape into (nx,1,1) / (1,ny,1) / (1,1,nz); on the
    # uniform Grid they collapse to a single scalar 1/dx that broadcasts
    # identically — the uniform path stays bit-for-bit.
    _fallback_inv = jnp.float32(1.0) / jnp.float32(grid.dx)

    def _inv_broadcast(arr_attr: str, axis: int):
        arr = getattr(grid, arr_attr, None)
        if arr is None:
            return _fallback_inv
        shape = [1, 1, 1]
        shape[axis] = -1
        return jnp.asarray(arr, dtype=jnp.float32).reshape(tuple(shape))

    # E-position (integer-index) inverse spacings — consumed by E update.
    inv_dx = _inv_broadcast("inv_dx", 0)
    inv_dy = _inv_broadcast("inv_dy", 1)
    inv_dz = _inv_broadcast("inv_dz", 2)
    # H-position (half-integer) inverse spacings — consumed by H update.
    inv_dx_h = _inv_broadcast("inv_dx_h", 0)
    inv_dy_h = _inv_broadcast("inv_dy_h", 1)
    inv_dz_h = _inv_broadcast("inv_dz_h", 2)

    # Each component's relative permittivity; _upml_e_coeffs_si multiplies
    # it by eps_0 as this function used to here.
    if aniso_eps is not None:
        eps_ex, eps_ey, eps_ez = (e_.astype(jnp.float32) for e_ in aniso_eps)
    else:
        eps_ex, eps_ey, eps_ez = eps_r_c

    # Perpendicular σ: E_x gets damping from y,z PML (using E-position σ)
    sigma_perp_ex = sEy + sEz
    sigma_perp_ey = sEx + sEz
    sigma_perp_ez = sEx + sEy

    def _e_coeffs(sigma_perp, eps_r, sigma_mat):
        # #1357: the SI spelling's bits (:func:`_upml_e_coeffs_si`), the
        # eps_r-unit derivative. ``dt / eps_abs`` handed the cotangent a
        # factor (eps_r*EPS_0)**-2 ~ 1.3e22 and overflowed float32.
        return si_value_eps_r_grad(_upml_e_coeffs_si, _upml_e_coeffs_eps_r,
                                   sigma_perp, eps_r, sigma_mat, dt)

    ca_ex, cb_ex = _e_coeffs(sigma_perp_ex, eps_ex, sigma_mat_c[0])
    ca_ey, cb_ey = _e_coeffs(sigma_perp_ey, eps_ey, sigma_mat_c[1])
    ca_ez, cb_ez = _e_coeffs(sigma_perp_ez, eps_ez, sigma_mat_c[2])

    # H perpendicular: use H-position σ
    sigma_perp_hx = sHy + sHz
    sigma_perp_hy = sHx + sHz
    sigma_perp_hz = sHx + sHy

    def _h_coeffs(sigma_perp, mu):
        eps_0 = jnp.float32(EPS_0)
        loss = sigma_perp * dt / (jnp.float32(2.0) * eps_0)
        denom = jnp.float32(1.0) + loss
        da = (jnp.float32(1.0) - loss) / denom
        db = (dt / mu) / denom
        return da.astype(jnp.float32), db.astype(jnp.float32)

    da_hx, db_hx = _h_coeffs(sigma_perp_hx, mu_abs[0])
    da_hy, db_hy = _h_coeffs(sigma_perp_hy, mu_abs[1])
    da_hz, db_hz = _h_coeffs(sigma_perp_hz, mu_abs[2])

    return UPMLCoeffs(
        ca_ex=ca_ex, ca_ey=ca_ey, ca_ez=ca_ez,
        cb_ex=cb_ex, cb_ey=cb_ey, cb_ez=cb_ez,
        da_hx=da_hx, da_hy=da_hy, da_hz=da_hz,
        db_hx=db_hx, db_hy=db_hy, db_hz=db_hz,
        inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz,
        inv_dx_h=inv_dx_h, inv_dy_h=inv_dy_h, inv_dz_h=inv_dz_h,
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

    curl_x = ((fwd(ez, 1) - ez) * coeffs.inv_dy_h
              - (fwd(ey, 2) - ey) * coeffs.inv_dz_h)
    curl_y = ((fwd(ex, 2) - ex) * coeffs.inv_dz_h
              - (fwd(ez, 0) - ez) * coeffs.inv_dx_h)
    curl_z = ((fwd(ey, 0) - ey) * coeffs.inv_dx_h
              - (fwd(ex, 1) - ex) * coeffs.inv_dy_h)

    hx = (coeffs.da_hx * state.hx.astype(jnp.float32) - coeffs.db_hx * curl_x).astype(_fdtype)
    hy = (coeffs.da_hy * state.hy.astype(jnp.float32) - coeffs.db_hy * curl_y).astype(_fdtype)
    hz = (coeffs.da_hz * state.hz.astype(jnp.float32) - coeffs.db_hz * curl_z).astype(_fdtype)

    return state._replace(hx=hx, hy=hy, hz=hz)


def apply_upml_e(
    state: FDTDState,
    coeffs: UPMLCoeffs,
    periodic: tuple = (False, False, False),
    boundary=None,
) -> FDTDState:
    """E-field update using precomputed UPML coefficients."""
    _fdtype = state.ex.dtype
    hx = state.hx.astype(jnp.float32)
    hy = state.hy.astype(jnp.float32)
    hz = state.hz.astype(jnp.float32)

    curl_x, curl_y, curl_z = curl_h(
        hx, hy, hz, None, periodic, boundary=boundary,
        inv_spacing=(coeffs.inv_dx, coeffs.inv_dy, coeffs.inv_dz))

    ex = (coeffs.ca_ex * state.ex.astype(jnp.float32) + coeffs.cb_ex * curl_x).astype(_fdtype)
    ey = (coeffs.ca_ey * state.ey.astype(jnp.float32) + coeffs.cb_ey * curl_y).astype(_fdtype)
    ez = (coeffs.ca_ez * state.ez.astype(jnp.float32) + coeffs.cb_ez * curl_z).astype(_fdtype)

    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)
