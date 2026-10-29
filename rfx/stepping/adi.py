from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rfx import _realized
from rfx.boundaries.cpml import CPMLParams
from rfx.core.yee import (EPS_0, MU_0, MaterialArrays, component_h_materials,
                          si_value_eps_r_grad)
from rfx.model.adi_admission import _validate_interior_pec
from rfx.stepping import recorded_scan


# ---------------------------------------------------------------------------
# Tridiagonal Thomas solver — fully JAX-differentiable via lax.scan
# ---------------------------------------------------------------------------

def thomas_solve(a: jnp.ndarray, b: jnp.ndarray,
                 c: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """Solve tridiagonal system  A x = d  using the Thomas algorithm.

    A has sub-diagonal *a*, main diagonal *b*, and super-diagonal *c*.
    All inputs are 1-D arrays of length N.  a[0] and c[N-1] are ignored.

    The entire forward-elimination and back-substitution are expressed
    as ``jax.lax.scan`` so the function is JIT-compilable and
    differentiable (reverse-mode AD through the scan).

    Parameters
    ----------
    a, b, c, d : (N,) float arrays
        Tridiagonal system coefficients and right-hand side.

    Returns
    -------
    x : (N,) float array
        Solution vector.
    """
    N = b.shape[0]

    # --- Forward elimination ---
    c0 = c[0] / b[0]
    d0 = d[0] / b[0]

    def fwd_body(carry, idx):
        c_prev, d_prev = carry
        ai = a[idx]
        bi = b[idx]
        ci = c[idx]
        di = d[idx]
        denom = bi - ai * c_prev
        c_new = ci / denom
        d_new = (di - ai * d_prev) / denom
        return (c_new, d_new), (c_new, d_new)

    indices = jnp.arange(1, N)
    (_, _), (c_mod_rest, d_mod_rest) = jax.lax.scan(fwd_body, (c0, d0), indices)

    c_mod = jnp.concatenate([jnp.array([c0]), c_mod_rest])
    d_mod = jnp.concatenate([jnp.array([d0]), d_mod_rest])

    # --- Back substitution ---
    xN = d_mod[N - 1]

    def bwd_body(x_next, idx):
        x_cur = d_mod[idx] - c_mod[idx] * x_next
        return x_cur, x_cur

    bwd_indices = jnp.arange(N - 2, -1, -1)
    _, x_rest_rev = jax.lax.scan(bwd_body, xN, bwd_indices)

    x = jnp.concatenate([x_rest_rev[::-1], jnp.array([xN])])
    return x


# ---------------------------------------------------------------------------
# ADI core
# ---------------------------------------------------------------------------

def _apply_pec_2d(ez: jnp.ndarray, ez_pec_mask: jnp.ndarray | None = None) -> jnp.ndarray:
    """Apply PEC constraints for 2D TMz.

    TMz has only one electric component (Ez), so PEC means Ez=0 on
    domain boundaries and wherever the internal geometry realizes an Ez
    edge as PEC.  ``ez_pec_mask`` is that realized mask (#931 §1.7): the
    ``Mz`` plane from ``rfx.boundaries.pec.realized_pec_edge_masks``, NOT
    the primal-cell occupancy.  Zeroing Ez at the occupied CELL indices
    instead was this lane's own realization rule — a body one cell thick
    shorted one node plane and never its far one.
    """
    ez = ez.at[0, :].set(0.0)
    ez = ez.at[-1, :].set(0.0)
    ez = ez.at[:, 0].set(0.0)
    ez = ez.at[:, -1].set(0.0)
    if ez_pec_mask is not None:
        ez = jnp.where(ez_pec_mask, 0.0, ez)
    return ez


def _adi_coeffs_eps_r(eps_r, sigma, dt, *lengths):
    """The ADI E coefficients written in eps_r units (#1357).

    ``(damping, half_dt/eps_plus, *couplings)``, where ``couplings[i] =
    half_dt**2/(MU_0*eps_plus*d*d)`` for ``d = lengths[i]`` (``d = 1`` gives
    the 3-D lane's bare ``cc``). ``eps_plus`` is written as ``EPS_0*(eps_r +
    sigma*dt/(4*EPS_0))``, and the SI constants are grouped as the Python
    floats ``half_dt/EPS_0`` and ``half_dt**2/(MU_0*EPS_0*d*d)`` (about 1 at
    the ADI Courant number), so no SI-sized factor multiplies a cotangent.
    The SI spelling's ``(MU_0*eps_plus*dx*dx)**-2`` is about 1e46 and
    overflows float32 whatever the cotangent. It is the derivative half of
    :func:`rfx.core.yee.si_value_eps_r_grad` for :func:`adi_step_2d` and
    :func:`adi_step_3d`.
    """
    if not isinstance(dt, jax.Array):
        dt = float(dt)
    half_dt = dt / 2.0
    q = dt / (4.0 * EPS_0)
    eps_plus_r = eps_r + sigma * q
    out = [(eps_r - sigma * q) / eps_plus_r, (half_dt / EPS_0) / eps_plus_r]
    for d in lengths:
        out.append((half_dt * half_dt / (MU_0 * EPS_0) / (d * d)) / eps_plus_r)
    return tuple(out)


def _adi_2d_coeffs_si(eps_r, sigma, dt, dx, dy):
    """``(damping, half_dt/eps_plus, Cx, Cy)`` of :func:`adi_step_2d`, SI."""
    eps = eps_r * EPS_0
    half_dt = dt / 2.0

    # Lossy medium: implicit conductivity integration
    # (ε + σ*dt/4) * Ez^{n+1/2} = (ε - σ*dt/4) * Ez^n + ...
    sigma_term = sigma * dt / 4.0
    eps_plus = eps + sigma_term   # implicit damping factor
    eps_minus = eps - sigma_term  # explicit damping factor
    damping = eps_minus / eps_plus  # < 1 when sigma > 0

    # Courant-like coupling coefficient for the implicit direction
    Cx = half_dt * half_dt / (MU_0 * eps_plus * dx * dx)  # (Nx, Ny)
    Cy = half_dt * half_dt / (MU_0 * eps_plus * dy * dy)  # (Nx, Ny)
    return damping, half_dt / eps_plus, Cx, Cy


def _adi_3d_coeffs_si(eps_r, sigma, dt, dx, dy, dz):
    """``(damping, ce, cc, Cx, Cy, Cz)`` of :func:`adi_step_3d`, SI."""
    eps = eps_r * EPS_0
    half_dt = dt / 2.0

    # Semi-implicit conductivity integration per dt/2 sub-step
    # (ε + σ·dt/4) E' = (ε - σ·dt/4) E + ...   — mirrors adi_step_2d.
    sigma_term = sigma * dt / 4.0
    eps_plus = eps + sigma_term
    damping = (eps - sigma_term) / eps_plus

    ce = half_dt / eps_plus          # Ampère explicit-curl factor
    cc = half_dt * half_dt / (MU_0 * eps_plus)  # substituted mixed-term factor
    Cx = cc / (dx * dx)              # tridiagonal coupling per axis
    Cy = cc / (dy * dy)
    Cz = cc / (dz * dz)
    return damping, ce, cc, Cx, Cy, Cz


def _adi_3d_coeffs_eps_r(eps_r, sigma, dt, dx, dy, dz):
    return _adi_coeffs_eps_r(eps_r, sigma, dt, 1.0, dx, dy, dz)


def adi_step_2d(ez: jnp.ndarray, hx: jnp.ndarray, hy: jnp.ndarray,
                eps_r: jnp.ndarray, sigma: jnp.ndarray,
                dt: float, dx: float, dy: float,
                ez_pec_mask: jnp.ndarray | None = None):
    r"""Advance (Ez, Hx, Hy) by one full ADI timestep.

    Implements the Zheng et al. (2000) 2D TMz ADI-FDTD scheme.

    Yee grid staggering (2D TMz):
      Ez[i,j]   at integer points (i, j)
      Hx[i,j]   at (i, j+1/2)  — lives on y-edges
      Hy[i,j]   at (i+1/2, j)  — lives on x-edges

    Finite differences follow the standard Yee convention:
      dEz/dx|_{i+1/2,j} = (Ez[i+1,j] - Ez[i,j]) / dx   (forward)
      dHy/dx|_{i,j}     = (Hy[i,j] - Hy[i-1,j]) / dx   (backward)

    **Half-step 1** (n -> n+1/2, implicit in x):

    .. math::
        H_x^{n+1/2} = H_x^n - \frac{\Delta t}{2\mu_0\Delta y}
                       (E_z^n[i,j+1] - E_z^n[i,j])

        H_y^{n+1/2} = H_y^n + \frac{\Delta t}{2\mu_0\Delta x}
                       (E_z^{n+1/2}[i+1,j] - E_z^{n+1/2}[i,j])

        \varepsilon E_z^{n+1/2} = \varepsilon E_z^n
            + \frac{\Delta t}{2} \left(
                \frac{H_y^{n+1/2}[i,j] - H_y^{n+1/2}[i-1,j]}{\Delta x}
              - \frac{H_x^n[i,j] - H_x^n[i,j-1]}{\Delta y}
            \right)

    Substituting the Hy^{n+1/2} expression into the Ez equation and
    collecting terms in Ez^{n+1/2} gives a tridiagonal system along x.

    **Half-step 2** is symmetric with roles of x/y swapped.

    Parameters
    ----------
    ez, hx, hy : (Nx, Ny) field arrays
    eps_r, sigma : (Nx, Ny) material arrays
    dt, dx, dy : scalars

    Returns
    -------
    ez_new, hx_new, hy_new : updated fields
    """
    # This API accepts no magnetic materials; radius declarations are refused upstream.
    mu_abs = component_h_materials(MaterialArrays(None, None, 1.0), high_wall=(False, False, False))[0] * MU_0
    _validate_interior_pec(ez_pec_mask)
    Nx, Ny = ez.shape
    half_dt = dt / 2.0

    # Damping, the explicit-curl factor half_dt/(ε + σ*dt/4) and the
    # Courant-like couplings Cx/Cy (:func:`_adi_2d_coeffs_si`). #1357: those
    # bits, the eps_r-unit derivative -- ``(MU_0*eps_plus*dx*dx)`` is ~1e-23,
    # so its VJP's ``**-2`` overflowed float32 at ANY cotangent.
    damping, ce, Cx, Cy = si_value_eps_r_grad(
        _adi_2d_coeffs_si, _adi_coeffs_eps_r, eps_r, sigma, dt, dx, dy)

    # ===================================================================
    # Half-step 1: implicit in x, explicit in y
    # ===================================================================

    # Step 1a: Update Hx explicitly using Ez^n (explicit in y).
    # Hx[i,j] at (i, j+1/2): dEz/dy = (Ez[i,j+1] - Ez[i,j]) / dy
    dez_dy = jnp.zeros_like(ez)
    dez_dy = dez_dy.at[:, :-1].set((ez[:, 1:] - ez[:, :-1]) / dy)
    hx_half = hx - (half_dt / mu_abs) * dez_dy

    # Step 1b: Build tridiagonal for Ez^{n+1/2} along x.
    #
    # After substituting Hy^{n+1/2} = Hy^n + (dt/2)/(mu0*dx) * (Ez^{n+1/2}[i+1]-Ez^{n+1/2}[i])
    # into the Ampere update:
    #   eps*(Ez^{n+1/2} - Ez^n)/half_dt =
    #       (Hy^{n+1/2}[i] - Hy^{n+1/2}[i-1])/dx - (Hx^n[i,j] - Hx^n[i,j-1])/dy
    #
    # The implicit Hy contributes:
    #   (Hy^{n+1/2}[i] - Hy^{n+1/2}[i-1])/dx
    #     = (Hy^n[i]-Hy^n[i-1])/dx
    #       + (dt/2)/(mu0*dx^2) * (Ez^{n+1/2}[i+1] - 2*Ez^{n+1/2}[i] + Ez^{n+1/2}[i-1])
    #
    # So the full equation becomes:
    #   eps/half_dt * Ez^{n+1/2} - (dt/2)/(mu0*dx^2) * d²Ez^{n+1/2}/dx² =
    #       eps/half_dt * Ez^n + (Hy^n[i]-Hy^n[i-1])/dx - (Hx^n[i,j]-Hx^n[i,j-1])/dy
    #
    # Multiply both sides by half_dt/eps:
    #   Ez^{n+1/2} - Cx * (Ez^{n+1/2}[i+1] - 2*Ez^{n+1/2}[i] + Ez^{n+1/2}[i-1]) =
    #       Ez^n + (half_dt/eps) * curl_H^n_explicit
    #
    # In tridiagonal form:
    #   -Cx[i,j] * Ez^{n+1/2}[i-1,j]
    #     + (1 + 2*Cx[i,j]) * Ez^{n+1/2}[i,j]
    #     - Cx[i,j] * Ez^{n+1/2}[i+1,j]
    #   = Ez^n[i,j] + (half_dt/eps[i,j]) * (dHy_n/dx - dHx_n/dy)[i,j]

    # curl H^n (both components at time n, before any updates)
    # dHy^n/dx at (i,j): backward difference = (Hy^n[i,j] - Hy^n[i-1,j])/dx
    dhy_dx_n = jnp.zeros_like(hy)
    dhy_dx_n = dhy_dx_n.at[1:, :].set((hy[1:, :] - hy[:-1, :]) / dx)

    # dHx^n/dy at (i,j): backward difference = (Hx^n[i,j] - Hx^n[i,j-1])/dy
    dhx_dy_n = jnp.zeros_like(hx)
    dhx_dy_n = dhx_dy_n.at[:, 1:].set((hx[:, 1:] - hx[:, :-1]) / dy)

    curl_h_n = dhy_dx_n - dhx_dy_n  # (Nx, Ny)

    rhs1 = damping * ez + ce * curl_h_n  # (Nx, Ny)

    # Solve tridiagonal along x for each column j.
    # Interior points: i = 1 .. Nx-2. Boundary (i=0, i=Nx-1) are PEC: Ez=0.
    def _solve_x_column(j):
        cx = Cx[1:-1, j]  # (Nx-2,)
        rh = rhs1[1:-1, j]
        a_diag = -cx
        b_diag = 1.0 + 2.0 * cx
        c_diag = -cx
        return thomas_solve(a_diag, b_diag, c_diag, rh)

    interior_ez1 = jax.vmap(_solve_x_column)(jnp.arange(Ny))  # (Ny, Nx-2)
    ez_half = jnp.zeros_like(ez)
    ez_half = ez_half.at[1:-1, :].set(interior_ez1.T)
    ez_half = _apply_pec_2d(ez_half, ez_pec_mask)

    # Step 1c: Compute Hy^{n+1/2} using the implicit Ez^{n+1/2}
    dez_dx_half = jnp.zeros_like(ez_half)
    dez_dx_half = dez_dx_half.at[:-1, :].set(
        (ez_half[1:, :] - ez_half[:-1, :]) / dx
    )
    hy_half = hy + (half_dt / mu_abs) * dez_dx_half

    # ===================================================================
    # Half-step 2: implicit in y, explicit in x
    # ===================================================================

    # Step 2a: Update Hy explicitly using Ez^{n+1/2} (explicit in x).
    # Hy[i,j] at (i+1/2, j): dEz/dx = (Ez[i+1,j] - Ez[i,j]) / dx
    dez_dx_half2 = jnp.zeros_like(ez_half)
    dez_dx_half2 = dez_dx_half2.at[:-1, :].set(
        (ez_half[1:, :] - ez_half[:-1, :]) / dx
    )
    hy_new = hy_half + (half_dt / mu_abs) * dez_dx_half2

    # Step 2b: Build tridiagonal for Ez^{n+1} along y.
    # Same derivation as half-step 1, but with x<->y swapped.
    # Now implicit Hx update:
    #   Hx^{n+1}[i,j] = Hx^{n+1/2}[i,j] - (dt/2)/(mu0*dy) * (Ez^{n+1}[i,j+1] - Ez^{n+1}[i,j])
    # Substituting into Ampere:
    #   Ez^{n+1} - Cy * d²Ez^{n+1}/dy² =
    #       Ez^{n+1/2} + (half_dt/eps) * curl_H^{n+1/2}

    # curl H^{n+1/2}
    dhy_dx_half = jnp.zeros_like(hy_half)
    dhy_dx_half = dhy_dx_half.at[1:, :].set(
        (hy_half[1:, :] - hy_half[:-1, :]) / dx
    )
    dhx_dy_half = jnp.zeros_like(hx_half)
    dhx_dy_half = dhx_dy_half.at[:, 1:].set(
        (hx_half[:, 1:] - hx_half[:, :-1]) / dy
    )
    curl_h_half = dhy_dx_half - dhx_dy_half

    rhs2 = damping * ez_half + ce * curl_h_half  # (Nx, Ny)

    # Solve tridiagonal along y for each row i.
    # Interior: j = 1 .. Ny-2. Boundary j=0, j=Ny-1: PEC Ez=0.
    def _solve_y_row(i):
        cy = Cy[i, 1:-1]  # (Ny-2,)
        rh = rhs2[i, 1:-1]
        a_diag = -cy
        b_diag = 1.0 + 2.0 * cy
        c_diag = -cy
        return thomas_solve(a_diag, b_diag, c_diag, rh)

    interior_ez2 = jax.vmap(_solve_y_row)(jnp.arange(Nx))  # (Nx, Ny-2)
    ez_new = jnp.zeros_like(ez)
    ez_new = ez_new.at[:, 1:-1].set(interior_ez2)
    ez_new = _apply_pec_2d(ez_new, ez_pec_mask)

    # Step 2c: Compute Hx^{n+1} using the implicit Ez^{n+1}
    dez_dy_new = jnp.zeros_like(ez_new)
    dez_dy_new = dez_dy_new.at[:, :-1].set(
        (ez_new[:, 1:] - ez_new[:, :-1]) / dy
    )
    hx_new = hx_half - (half_dt / mu_abs) * dez_dy_new

    return ez_new, hx_new, hy_new


class ADICPMLState2D(NamedTuple):
    """Auxiliary psi arrays for 2D TMz CPML in ADI-FDTD."""
    # x-direction: in xlo/xhi CPML strips
    psi_ezy_xlo: jnp.ndarray  # (n_cpml, Ny) — Ez from dHy/dx
    psi_ezy_xhi: jnp.ndarray
    psi_hyx_xlo: jnp.ndarray  # (n_cpml, Ny) — Hy from dEz/dx
    psi_hyx_xhi: jnp.ndarray
    # y-direction: in ylo/yhi CPML strips
    psi_ezx_ylo: jnp.ndarray  # (Nx, n_cpml) — Ez from dHx/dy
    psi_ezx_yhi: jnp.ndarray
    psi_hxy_ylo: jnp.ndarray  # (Nx, n_cpml) — Hx from dEz/dy
    psi_hxy_yhi: jnp.ndarray


class ADICPMLParams2D(NamedTuple):
    """CPML profile coefficients for 2D ADI."""
    n_cpml: int
    # x-direction profiles (n_cpml,)
    bx: jnp.ndarray
    cx: jnp.ndarray
    # y-direction profiles (n_cpml,)
    by: jnp.ndarray
    cy: jnp.ndarray
    magnetic_xlo: CPMLParams
    magnetic_xhi: CPMLParams
    magnetic_ylo: CPMLParams
    magnetic_yhi: CPMLParams


def _inv_eps_si(eps_r):
    return 1.0 / (eps_r * EPS_0)


def _inv_eps_eps_r(eps_r):
    return (1.0 / EPS_0) / eps_r


def apply_adi_cpml_2d(ez, hx, hy, cpml_params, cpml_state, eps_r, dt, dx, dy):
    """Apply CPML ADE corrections after an ADI step.

    Updates psi auxiliary variables and adds corrections to fields
    in the CPML boundary strips.
    """
    # This API accepts no magnetic materials; radius declarations are refused upstream.
    mu_abs = component_h_materials(MaterialArrays(None, None, 1.0), high_wall=(False, False, False))[0] * MU_0
    n = cpml_params.n_cpml
    bx, cx = cpml_params.bx, cpml_params.cx
    by, cy = cpml_params.by, cpml_params.cy
    mxlo, mxhi = cpml_params.magnetic_xlo, cpml_params.magnetic_xhi
    mylo, myhi = cpml_params.magnetic_ylo, cpml_params.magnetic_yhi

    # #1357: 1/eps's bits, the eps_r-unit derivative (the VJP of 1/eps
    # multiplies the cotangent by eps**-2 ~ 1.3e22).
    inv_eps = si_value_eps_r_grad(_inv_eps_si, _inv_eps_eps_r, eps_r)

    # === Hy correction from dEz/dx (x-CPML) ===
    # xlo: forward diff dEz/dx at half-integer x in strip [0, n)
    dez_dx_xlo = (ez[1:n + 1, :] - ez[:n, :]) / dx
    psi_hyx_xlo = mxlo.b[:, None] * cpml_state.psi_hyx_xlo + mxlo.c[:, None] * dez_dx_xlo
    hy = hy.at[:n, :].add((dt / mu_abs) * psi_hyx_xlo)

    # Hy[i] lives at i+1/2, including in the high-face strip.
    dez_dx_xhi = (jnp.pad(ez, ((0, 1), (0, 0)))[-n:, :] - ez[-n:, :]) / dx
    bx_hi = jnp.flip(bx)
    cx_hi = jnp.flip(cx)
    psi_hyx_xhi = jnp.flip(mxhi.b)[:, None] * cpml_state.psi_hyx_xhi + jnp.flip(mxhi.c)[:, None] * dez_dx_xhi
    hy = hy.at[-n:, :].add((dt / mu_abs) * psi_hyx_xhi)

    # === Hx correction from dEz/dy (y-CPML) ===
    dez_dy_ylo = (ez[:, 1:n + 1] - ez[:, :n]) / dy
    psi_hxy_ylo = mylo.b[None, :] * cpml_state.psi_hxy_ylo + mylo.c[None, :] * dez_dy_ylo
    hx = hx.at[:, :n].add(-(dt / mu_abs) * psi_hxy_ylo)  # negative: Faraday sign

    dez_dy_yhi = (jnp.pad(ez, ((0, 0), (0, 1)))[:, -n:] - ez[:, -n:]) / dy
    by_hi = jnp.flip(by)
    cy_hi = jnp.flip(cy)
    psi_hxy_yhi = jnp.flip(myhi.b)[None, :] * cpml_state.psi_hxy_yhi + jnp.flip(myhi.c)[None, :] * dez_dy_yhi
    hx = hx.at[:, -n:].add(-(dt / mu_abs) * psi_hxy_yhi)

    # === Ez correction from dHy/dx (x-CPML) ===
    # Ez[i] and its integer-node profile share the strip [0, n).
    dhy_dx_xlo = (hy[:n, :] - jnp.concatenate([jnp.zeros_like(hy[:1, :]), hy[:n - 1, :]])) / dx
    psi_ezy_xlo = bx[:, None] * cpml_state.psi_ezy_xlo + cx[:, None] * dhy_dx_xlo
    inv_eps_xlo = inv_eps[:n, :]
    ez = ez.at[:n, :].add(dt * inv_eps_xlo * psi_ezy_xlo)

    dhy_dx_xhi = (hy[-n:, :] - hy[-n - 1:-1, :]) / dx
    psi_ezy_xhi = bx_hi[:, None] * cpml_state.psi_ezy_xhi + cx_hi[:, None] * dhy_dx_xhi
    inv_eps_xhi = inv_eps[-n:, :]
    ez = ez.at[-n:, :].add(dt * inv_eps_xhi * psi_ezy_xhi)

    # === Ez correction from dHx/dy (y-CPML) ===
    dhx_dy_ylo = (hx[:, :n] - jnp.concatenate([jnp.zeros_like(hx[:, :1]), hx[:, :n - 1]], axis=1)) / dy
    psi_ezx_ylo = by[None, :] * cpml_state.psi_ezx_ylo + cy[None, :] * dhx_dy_ylo
    inv_eps_ylo = inv_eps[:, :n]
    ez = ez.at[:, :n].add(-dt * inv_eps_ylo * psi_ezx_ylo)  # negative: Ampere sign

    dhx_dy_yhi = (hx[:, -n:] - hx[:, -n - 1:-1]) / dy
    psi_ezx_yhi = by_hi[None, :] * cpml_state.psi_ezx_yhi + cy_hi[None, :] * dhx_dy_yhi
    inv_eps_yhi = inv_eps[:, -n:]
    ez = ez.at[:, -n:].add(-dt * inv_eps_yhi * psi_ezx_yhi)

    # PEC at outer boundaries (outermost cells of CPML)
    ez = ez.at[0, :].set(0.0)
    ez = ez.at[-1, :].set(0.0)
    ez = ez.at[:, 0].set(0.0)
    ez = ez.at[:, -1].set(0.0)

    new_state = ADICPMLState2D(
        psi_ezy_xlo=psi_ezy_xlo, psi_ezy_xhi=psi_ezy_xhi,
        psi_hyx_xlo=psi_hyx_xlo, psi_hyx_xhi=psi_hyx_xhi,
        psi_ezx_ylo=psi_ezx_ylo, psi_ezx_yhi=psi_ezx_yhi,
        psi_hxy_ylo=psi_hxy_ylo, psi_hxy_yhi=psi_hxy_yhi,
    )
    return ez, hx, hy, new_state


def _apply_pec_3d(ex, ey, ez, pec_edge_masks=None):
    """Zero tangential E on all 6 domain faces + the realized PEC edges.

    ``pec_edge_masks`` is the ``(Mx, My, Mz)`` triple from
    ``rfx.boundaries.pec.realized_pec_edge_masks`` (#931 §1.7).  Zeroing
    all three components at the occupied CELL indices — this lane's own
    rule until #931 — is a fourth realization of the same geometry and is
    gone.
    """
    # x-faces: Ey, Ez = 0
    ey = ey.at[0, :, :].set(0.0)
    ey = ey.at[-1, :, :].set(0.0)
    ez = ez.at[0, :, :].set(0.0)
    ez = ez.at[-1, :, :].set(0.0)
    # y-faces: Ex, Ez = 0
    ex = ex.at[:, 0, :].set(0.0)
    ex = ex.at[:, -1, :].set(0.0)
    ez = ez.at[:, 0, :].set(0.0)
    ez = ez.at[:, -1, :].set(0.0)
    # z-faces: Ex, Ey = 0
    ex = ex.at[:, :, 0].set(0.0)
    ex = ex.at[:, :, -1].set(0.0)
    ey = ey.at[:, :, 0].set(0.0)
    ey = ey.at[:, :, -1].set(0.0)
    if pec_edge_masks is not None:
        mx, my, mz = pec_edge_masks
        ex = jnp.where(mx, 0.0, ex)
        ey = jnp.where(my, 0.0, ey)
        ez = jnp.where(mz, 0.0, ez)
    return ex, ey, ez


def _solve_tridiag_along(field_3d, C_3d, rhs_3d, axis):
    """Solve tridiagonal -C*f[k-1] + (1+2C)*f[k] - C*f[k+1] = rhs along *axis*.

    Interior points only (boundary = 0 from PEC).
    Uses thomas_solve vectorized via double vmap over the other two axes.
    """
    # Move solve axis to last position for uniform indexing
    # axis 0: already (Nx, Ny, Nz) → solve along first → transpose to (..., Nx)
    perm = list(range(3))
    perm.remove(axis)
    perm.append(axis)
    inv_perm = [0] * 3
    for i, p in enumerate(perm):
        inv_perm[p] = i

    C_t = jnp.transpose(C_3d, perm)       # (..., N_solve)
    rhs_t = jnp.transpose(rhs_3d, perm)   # (..., N_solve)

    n1, n2, ns = C_t.shape

    def _solve_line(c_line, rhs_line):
        c = c_line[1:-1]
        rh = rhs_line[1:-1]
        interior = thomas_solve(-c, 1.0 + 2.0 * c, -c, rh)
        return jnp.concatenate([jnp.zeros(1), interior, jnp.zeros(1)])

    # Double vmap over the two non-solve axes
    result_t = jax.vmap(jax.vmap(_solve_line))(C_t, rhs_t)  # (n1, n2, ns)
    return jnp.transpose(result_t, inv_perm)


def adi_step_3d(ex, ey, ez, hx, hy, hz,
                eps_r, sigma, dt, dx, dy, dz,
                pec_edge_masks=None):
    r"""Advance all 6 field components by one full 3D ADI timestep.

    Implements the Zheng–Chen–Zhang (ZCZ) 3D ADI-FDTD scheme (F. Zheng,
    Z. Chen, J. Zhang, IEEE Trans. MTT 48(9), 2000; identical scheme in
    Namiki's 3D extension): two sub-steps of ``dt/2``. In each sub-step
    every E component is implicit along exactly one axis (tridiagonal
    Thomas solve) and every H component is updated explicitly from the
    just-solved E; the two sub-steps swap which curl term is implicit.

    This is the Peaceman–Rachford alternation
    ``(I - dt/2·A) u' = (I + dt/2·B) u`` then
    ``(I - dt/2·B) u'' = (I + dt/2·A) u'`` with the curl split into the
    A/B term families. With the zero-padded difference operators used
    here, backward differences are exactly minus the transpose of forward
    differences (boundary rows included), so both split operators are
    skew-symmetric in the energy variables and the full-step map is
    similar to a product of two unitary Cayley transforms — the scheme is
    stable without the explicit CFL restriction for the homogeneous, lossless
    problem with compatible PEC domain boundaries. This argument does not
    establish stability for the internal PEC projection below.

    Accuracy envelope (12^3 PEC cavity, TE101 at ~15 cells per
    wavelength): eigenfrequency error is -1.4% at 2x the 3D Yee CFL
    limit (test-measured, ``tests/unit/misc/test_review_tier1_validation_battery.py``),
    growing ~dt^2 (Crank–Nicolson-like temporal lag) to ~ -6.7% at 5x CFL
    (von Neumann analysis — the 5x point is not test-measured). For that
    cavity without internal PEC, large factors are quantitative only for features
    much coarser than the timestep. Use large CFL factors for stiff
    meshes (thin substrates), not wavelength-scale resonances.

    Caveat: the *internal* ``pec_edge_masks`` are enforced by post-solve
    projection (as on the 2D path), not by exact Dirichlet rows. Both
    lanes exhibit growing solutions with interior PEC, even at unit CFL.
    A supplied internal mask is therefore refused at every timestep; the
    code remains here for research, not as a supported conductor solver.

    Parameters
    ----------
    ex, ey, ez, hx, hy, hz : (Nx, Ny, Nz) field arrays
    eps_r, sigma : (Nx, Ny, Nz) material arrays
    dt, dx, dy, dz : scalars

    Returns
    -------
    ex, ey, ez, hx, hy, hz : updated fields
    """
    # This API accepts no magnetic materials; radius declarations are refused upstream.
    mu_abs = component_h_materials(MaterialArrays(None, None, 1.0), high_wall=(False, False, False))[0] * MU_0
    _validate_interior_pec(pec_edge_masks)
    if _realized.ACTIVE is not None:
        observed = _realized.scalar_electric(
            MaterialArrays(eps_r, sigma, jnp.ones_like(eps_r)), "adi.E")
        eps_r, sigma = observed.eps_r, observed.sigma
    half_dt = dt / 2.0

    # Damping, the explicit-curl factor ce, the mixed-term factor cc and the
    # per-axis couplings (:func:`_adi_3d_coeffs_si`). #1357: those bits, the
    # eps_r-unit derivative -- the SI VJP of ``cc`` multiplies the cotangent
    # by (MU_0*eps_plus)**-2 ~ 8e33.
    damping, ce, cc, Cx, Cy, Cz = si_value_eps_r_grad(
        _adi_3d_coeffs_si, _adi_3d_coeffs_eps_r, eps_r, sigma, dt, dx, dy, dz)
    ch = half_dt / mu_abs              # Faraday factor (no magnetic loss)

    def _fwd(arr, ax):
        pw = [(0, 0)] * 3
        pw[ax] = (0, 1)
        s = [slice(None)] * 3
        s[ax] = slice(1, None)
        return jnp.pad(arr, pw)[tuple(s)]

    def _bwd(arr, ax):
        pw = [(0, 0)] * 3
        pw[ax] = (1, 0)
        s = [slice(None)] * 3
        s[ax] = slice(None, -1)
        return jnp.pad(arr, pw)[tuple(s)]

    def _dp(arr, ax, d):
        """Forward difference (E -> H staggering)."""
        return (_fwd(arr, ax) - arr) / d

    def _dm(arr, ax, d):
        """Backward difference (H -> E staggering)."""
        return (arr - _bwd(arr, ax)) / d

    # ===================================================================
    # Sub-step 1 (n -> n+1/2): implicit pairs (Ex,Hz|y), (Ey,Hx|z),
    # (Ez,Hy|x). Each implicit operator is obtained by substituting the
    # same-sub-step H update into the E equation; the -cc·Dm(Dp(E))
    # mixed terms are the explicit remainder of that substitution.
    # ===================================================================
    rhs_ex = damping * ex + ce * (_dm(hz, 1, dy) - _dm(hy, 2, dz)) \
        - cc * _dm(_dp(ey, 0, dx), 1, dy)
    rhs_ey = damping * ey + ce * (_dm(hx, 2, dz) - _dm(hz, 0, dx)) \
        - cc * _dm(_dp(ez, 1, dy), 2, dz)
    rhs_ez = damping * ez + ce * (_dm(hy, 0, dx) - _dm(hx, 1, dy)) \
        - cc * _dm(_dp(ex, 2, dz), 0, dx)
    # PEC on the RHS: a zero RHS line with zero Dirichlet ends solves to
    # exactly zero, so face-parallel lines inside PEC faces stay PEC.
    rhs_ex, rhs_ey, rhs_ez = _apply_pec_3d(rhs_ex, rhs_ey, rhs_ez, pec_edge_masks)

    ex1 = _solve_tridiag_along(ex, Cy, rhs_ex, axis=1)
    ey1 = _solve_tridiag_along(ey, Cz, rhs_ey, axis=2)
    ez1 = _solve_tridiag_along(ez, Cx, rhs_ez, axis=0)
    ex1, ey1, ez1 = _apply_pec_3d(ex1, ey1, ez1, pec_edge_masks)

    # Explicit H updates: implicit-partner E at n+1/2, other E at n.
    hx1 = hx + ch * (_dp(ey1, 2, dz) - _dp(ez, 1, dy))
    hy1 = hy + ch * (_dp(ez1, 0, dx) - _dp(ex, 2, dz))
    hz1 = hz + ch * (_dp(ex1, 1, dy) - _dp(ey, 0, dx))

    # ===================================================================
    # Sub-step 2 (n+1/2 -> n+1): implicit pairs (Ex,Hy|z), (Ey,Hz|x),
    # (Ez,Hx|y) — the curl-term roles swap.
    # ===================================================================
    rhs_ex = damping * ex1 + ce * (_dm(hz1, 1, dy) - _dm(hy1, 2, dz)) \
        - cc * _dm(_dp(ez1, 0, dx), 2, dz)
    rhs_ey = damping * ey1 + ce * (_dm(hx1, 2, dz) - _dm(hz1, 0, dx)) \
        - cc * _dm(_dp(ex1, 1, dy), 0, dx)
    rhs_ez = damping * ez1 + ce * (_dm(hy1, 0, dx) - _dm(hx1, 1, dy)) \
        - cc * _dm(_dp(ey1, 2, dz), 1, dy)
    rhs_ex, rhs_ey, rhs_ez = _apply_pec_3d(rhs_ex, rhs_ey, rhs_ez, pec_edge_masks)

    ex2 = _solve_tridiag_along(ex1, Cz, rhs_ex, axis=2)
    ey2 = _solve_tridiag_along(ey1, Cx, rhs_ey, axis=0)
    ez2 = _solve_tridiag_along(ez1, Cy, rhs_ez, axis=1)
    ex2, ey2, ez2 = _apply_pec_3d(ex2, ey2, ez2, pec_edge_masks)

    hx2 = hx1 + ch * (_dp(ey1, 2, dz) - _dp(ez2, 1, dy))
    hy2 = hy1 + ch * (_dp(ez1, 0, dx) - _dp(ex2, 2, dz))
    hz2 = hz1 + ch * (_dp(ex1, 1, dy) - _dp(ey2, 0, dx))

    return ex2, ey2, ez2, hx2, hy2, hz2


def run_adi_2d_loop(
    *,
    cpml_params,
    cpml_state,
    dt,
    dx,
    dy,
    eps_r,
    ez,
    ez_pec_mask,
    hx,
    hy,
    n_prb,
    n_src,
    n_steps,
    prb_meta,
    sigma,
    src_ij,
    src_waveforms,
    use_cpml,
):
    if use_cpml:
        def step_fn(state, step_idx):
            ez_s, hx_s, hy_s, cs = state

            # Inject sources
            def inject_one(carry, src_idx):
                ez_c = carry
                i = src_ij[src_idx, 0]
                j = src_ij[src_idx, 1]
                ez_c = ez_c.at[i, j].add(src_waveforms[src_idx, step_idx])
                return ez_c, None

            if n_src > 0:
                ez_s, _ = jax.lax.scan(inject_one, ez_s, jnp.arange(n_src))

            # ADI step (PEC at outer boundary handled by CPML)
            ez_s, hx_s, hy_s = adi_step_2d(
                ez_s, hx_s, hy_s, eps_r, sigma, dt, dx, dy, ez_pec_mask)

            # CPML correction (operator-splitting)
            ez_s, hx_s, hy_s, cs = apply_adi_cpml_2d(
                ez_s, hx_s, hy_s, cpml_params, cs, eps_r, dt, dx, dy)

            # Record probes
            if n_prb > 0:
                samples = []
                for pi, pj, component in prb_meta:
                    if component == "ez":
                        samples.append(ez_s[pi, pj])
                    elif component == "hx":
                        samples.append(hx_s[pi, pj])
                    else:
                        samples.append(hy_s[pi, pj])
                probe_vals = jnp.stack(samples)
            else:
                probe_vals = jnp.zeros(0)

            return (ez_s, hx_s, hy_s, cs), probe_vals

        init_state = (ez, hx, hy, cpml_state)
        (ez_f, hx_f, hy_f, _), probe_data = recorded_scan(
            step_fn, init_state, jnp.arange(n_steps))
    else:
        def step_fn(state, step_idx):
            ez_s, hx_s, hy_s = state

            def inject_one(carry, src_idx):
                ez_c = carry
                i = src_ij[src_idx, 0]
                j = src_ij[src_idx, 1]
                ez_c = ez_c.at[i, j].add(src_waveforms[src_idx, step_idx])
                return ez_c, None

            if n_src > 0:
                ez_s, _ = jax.lax.scan(inject_one, ez_s, jnp.arange(n_src))

            ez_s, hx_s, hy_s = adi_step_2d(
                ez_s, hx_s, hy_s, eps_r, sigma, dt, dx, dy, ez_pec_mask)

            if n_prb > 0:
                samples = []
                for pi, pj, component in prb_meta:
                    if component == "ez":
                        samples.append(ez_s[pi, pj])
                    elif component == "hx":
                        samples.append(hx_s[pi, pj])
                    else:
                        samples.append(hy_s[pi, pj])
                probe_vals = jnp.stack(samples)
            else:
                probe_vals = jnp.zeros(0)

            return (ez_s, hx_s, hy_s), probe_vals

        init_state = (ez, hx, hy)
        (ez_f, hx_f, hy_f), probe_data = recorded_scan(
            step_fn, init_state, jnp.arange(n_steps))
    return ez_f, hx_f, hy_f, probe_data


def run_adi_3d_loop(
    *,
    _prb_comp_indices,
    _src_by_comp,
    dt,
    dx,
    dy,
    dz,
    eps_r,
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    n_prb,
    n_steps,
    pec_edge_masks,
    prb_ijk,
    sigma,
    src_ijk,
    src_waveforms,
):
    def step_fn(carry, step_idx):
        fields = list(carry)

        # Source injection (soft source, before ADI step)
        for c_idx in range(6):
            for s_idx in _src_by_comp[c_idx]:
                si, sj, sk = src_ijk[s_idx]
                val = src_waveforms[s_idx, step_idx]
                fields[c_idx] = fields[c_idx].at[si, sj, sk].add(val)

        ex_, ey_, ez_, hx_, hy_, hz_ = fields

        # ADI step
        ex_, ey_, ez_, hx_, hy_, hz_ = adi_step_3d(
            ex_, ey_, ez_, hx_, hy_, hz_,
            eps_r, sigma, dt, dx, dy, dz,
            pec_edge_masks=pec_edge_masks,
        )

        # Probe sampling
        all_fields = (ex_, ey_, ez_, hx_, hy_, hz_)
        if n_prb > 0:
            samples = []
            for p_idx in range(n_prb):
                pi, pj, pk = prb_ijk[p_idx]
                samples.append(all_fields[_prb_comp_indices[p_idx]][pi, pj, pk])
            sample_vec = jnp.stack(samples)
        else:
            sample_vec = jnp.zeros(1)

        return (ex_, ey_, ez_, hx_, hy_, hz_), sample_vec

    carry_init = (ex, ey, ez, hx, hy, hz)
    final_carry, probe_data = recorded_scan(
        step_fn, carry_init, jnp.arange(n_steps))
    ex_f, ey_f, ez_f, hx_f, hy_f, hz_f = final_carry
    return ex_f, ey_f, ez_f, hx_f, hy_f, hz_f, probe_data
