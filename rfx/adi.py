"""ADI-FDTD: Alternating Direction Implicit FDTD solver.

Unconditionally stable — dt is not limited by CFL condition.
For thin substrates where standard Yee requires tiny dt, ADI allows
10-100x larger timesteps.

Supports 2D TMz (Ez, Hx, Hy) and 3D (all 6 components).

References:
    T. Namiki, IEEE MTT 1999
    F. Zheng et al., IEEE MGWL 2000
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rfx.core.yee import EPS_0, MU_0


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ADIState2D(NamedTuple):
    """2D TMz ADI-FDTD state: (Ez, Hx, Hy) on an (Nx, Ny) grid."""
    ez: jnp.ndarray   # (Nx, Ny)
    hx: jnp.ndarray   # (Nx, Ny)
    hy: jnp.ndarray   # (Nx, Ny)
    step: jnp.ndarray  # scalar int


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

def _apply_pec_2d(ez: jnp.ndarray, pec_mask: jnp.ndarray | None = None) -> jnp.ndarray:
    """Apply PEC constraints for 2D TMz.

    TMz has only one electric component (Ez), so PEC means Ez=0 on
    domain boundaries and on any internal PEC cells.
    """
    ez = ez.at[0, :].set(0.0)
    ez = ez.at[-1, :].set(0.0)
    ez = ez.at[:, 0].set(0.0)
    ez = ez.at[:, -1].set(0.0)
    if pec_mask is not None:
        ez = jnp.where(pec_mask, 0.0, ez)
    return ez


def adi_step_2d(ez: jnp.ndarray, hx: jnp.ndarray, hy: jnp.ndarray,
                eps_r: jnp.ndarray, sigma: jnp.ndarray,
                dt: float, dx: float, dy: float,
                pec_mask: jnp.ndarray | None = None):
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
    Nx, Ny = ez.shape
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

    # ===================================================================
    # Half-step 1: implicit in x, explicit in y
    # ===================================================================

    # Step 1a: Update Hx explicitly using Ez^n (explicit in y).
    # Hx[i,j] at (i, j+1/2): dEz/dy = (Ez[i,j+1] - Ez[i,j]) / dy
    dez_dy = jnp.zeros_like(ez)
    dez_dy = dez_dy.at[:, :-1].set((ez[:, 1:] - ez[:, :-1]) / dy)
    hx_half = hx - (half_dt / MU_0) * dez_dy

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

    rhs1 = damping * ez + (half_dt / eps_plus) * curl_h_n  # (Nx, Ny)

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
    ez_half = _apply_pec_2d(ez_half, pec_mask)

    # Step 1c: Compute Hy^{n+1/2} using the implicit Ez^{n+1/2}
    dez_dx_half = jnp.zeros_like(ez_half)
    dez_dx_half = dez_dx_half.at[:-1, :].set(
        (ez_half[1:, :] - ez_half[:-1, :]) / dx
    )
    hy_half = hy + (half_dt / MU_0) * dez_dx_half

    # ===================================================================
    # Half-step 2: implicit in y, explicit in x
    # ===================================================================

    # Step 2a: Update Hy explicitly using Ez^{n+1/2} (explicit in x).
    # Hy[i,j] at (i+1/2, j): dEz/dx = (Ez[i+1,j] - Ez[i,j]) / dx
    dez_dx_half2 = jnp.zeros_like(ez_half)
    dez_dx_half2 = dez_dx_half2.at[:-1, :].set(
        (ez_half[1:, :] - ez_half[:-1, :]) / dx
    )
    hy_new = hy_half + (half_dt / MU_0) * dez_dx_half2

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

    rhs2 = damping * ez_half + (half_dt / eps_plus) * curl_h_half  # (Nx, Ny)

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
    ez_new = _apply_pec_2d(ez_new, pec_mask)

    # Step 2c: Compute Hx^{n+1} using the implicit Ez^{n+1}
    dez_dy_new = jnp.zeros_like(ez_new)
    dez_dy_new = dez_dy_new.at[:, :-1].set(
        (ez_new[:, 1:] - ez_new[:, :-1]) / dy
    )
    hx_new = hx_half - (half_dt / MU_0) * dez_dy_new

    return ez_new, hx_new, hy_new


# ---------------------------------------------------------------------------
# CPML for ADI-FDTD (operator-splitting: ADI step + ADE correction)
# ---------------------------------------------------------------------------

def make_adi_absorbing_sigma(nx, ny, n_layers, dx, order=3):
    """Create a graded conductivity array for implicit ADI absorbing boundary.

    Unlike operator-splitting CPML, this uses conductivity folded into
    the ADI tridiagonal system — unconditionally stable at any dt.

    Parameters
    ----------
    nx, ny : grid dimensions
    n_layers : number of absorbing layers per face
    dx : cell size (used for optimal σ_max)
    order : polynomial grading order (default 3)

    Returns
    -------
    sigma : (nx, ny) conductivity array — zero in interior, graded in boundary
    """
    import numpy as np
    eta = float(np.sqrt(MU_0 / EPS_0))
    sigma_max = 0.8 * (order + 1) / (eta * dx)

    sigma = np.zeros((nx, ny), dtype=np.float64)

    for face_n in range(n_layers):
        # rho: 1.0 at outer boundary, 0.0 at inner edge
        rho = 1.0 - face_n / max(n_layers - 1, 1)
        s = sigma_max * rho**order

        # x-lo (column face_n)
        sigma[face_n, :] = np.maximum(sigma[face_n, :], s)
        # x-hi
        sigma[nx - 1 - face_n, :] = np.maximum(sigma[nx - 1 - face_n, :], s)
        # y-lo (row face_n)
        sigma[:, face_n] = np.maximum(sigma[:, face_n], s)
        # y-hi
        sigma[:, ny - 1 - face_n] = np.maximum(sigma[:, ny - 1 - face_n], s)

    return jnp.array(sigma, dtype=jnp.float32)


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


def init_adi_cpml_2d(n_cpml: int, dt: float, dx: float, dy: float,
                     nx: int, ny: int, kappa_max: float = 1.0):
    """Initialize CPML profiles and state for 2D TMz ADI-FDTD.

    Returns (params, state) where params has b/c coefficients and state
    has zero-initialized psi arrays.
    """
    import numpy as np

    eta = float(np.sqrt(MU_0 / EPS_0))
    order = 3

    def _make_bc(n_layers, ds):
        sigma_max = 0.8 * (order + 1) / (eta * ds) * kappa_max
        rho = 1.0 - np.arange(n_layers, dtype=np.float64) / max(n_layers - 1, 1)
        sigma = sigma_max * rho**order
        kappa = 1.0 + (kappa_max - 1.0) * rho**order
        alpha = 0.05 * (1.0 - rho)
        b = np.exp(-(sigma / kappa + alpha) * dt / EPS_0)
        denom = sigma * kappa + kappa**2 * alpha
        c = np.where(denom > 1e-30, sigma * (b - 1.0) / denom, 0.0)
        return jnp.array(b, dtype=jnp.float32), jnp.array(c, dtype=jnp.float32)

    bx, cx = _make_bc(n_cpml, dx)
    by, cy = _make_bc(n_cpml, dy)

    params = ADICPMLParams2D(n_cpml=n_cpml, bx=bx, cx=cx, by=by, cy=cy)

    state = ADICPMLState2D(
        psi_ezy_xlo=jnp.zeros((n_cpml, ny), dtype=jnp.float32),
        psi_ezy_xhi=jnp.zeros((n_cpml, ny), dtype=jnp.float32),
        psi_hyx_xlo=jnp.zeros((n_cpml, ny), dtype=jnp.float32),
        psi_hyx_xhi=jnp.zeros((n_cpml, ny), dtype=jnp.float32),
        psi_ezx_ylo=jnp.zeros((nx, n_cpml), dtype=jnp.float32),
        psi_ezx_yhi=jnp.zeros((nx, n_cpml), dtype=jnp.float32),
        psi_hxy_ylo=jnp.zeros((nx, n_cpml), dtype=jnp.float32),
        psi_hxy_yhi=jnp.zeros((nx, n_cpml), dtype=jnp.float32),
    )
    return params, state


def apply_adi_cpml_2d(ez, hx, hy, cpml_params, cpml_state, eps_r, dt, dx, dy):
    """Apply CPML ADE corrections after an ADI step.

    Updates psi auxiliary variables and adds corrections to fields
    in the CPML boundary strips.
    """
    n = cpml_params.n_cpml
    bx, cx = cpml_params.bx, cpml_params.cx
    by, cy = cpml_params.by, cpml_params.cy

    eps = eps_r * EPS_0

    # === Hy correction from dEz/dx (x-CPML) ===
    # xlo: forward diff dEz/dx at half-integer x in strip [0, n)
    dez_dx_xlo = (ez[1:n + 1, :] - ez[:n, :]) / dx
    psi_hyx_xlo = bx[:, None] * cpml_state.psi_hyx_xlo + cx[:, None] * dez_dx_xlo
    hy = hy.at[:n, :].add((dt / MU_0) * psi_hyx_xlo)

    # xhi: forward diff in strip [-n-1, -1)
    dez_dx_xhi = (ez[-n:, :] - ez[-n - 1:-1, :]) / dx
    bx_hi = jnp.flip(bx)
    cx_hi = jnp.flip(cx)
    psi_hyx_xhi = bx_hi[:, None] * cpml_state.psi_hyx_xhi + cx_hi[:, None] * dez_dx_xhi
    hy = hy.at[-n:, :].add((dt / MU_0) * psi_hyx_xhi)

    # === Hx correction from dEz/dy (y-CPML) ===
    dez_dy_ylo = (ez[:, 1:n + 1] - ez[:, :n]) / dy
    psi_hxy_ylo = by[None, :] * cpml_state.psi_hxy_ylo + cy[None, :] * dez_dy_ylo
    hx = hx.at[:, :n].add(-(dt / MU_0) * psi_hxy_ylo)  # negative: Faraday sign

    dez_dy_yhi = (ez[:, -n:] - ez[:, -n - 1:-1]) / dy
    by_hi = jnp.flip(by)
    cy_hi = jnp.flip(cy)
    psi_hxy_yhi = by_hi[None, :] * cpml_state.psi_hxy_yhi + cy_hi[None, :] * dez_dy_yhi
    hx = hx.at[:, -n:].add(-(dt / MU_0) * psi_hxy_yhi)

    # === Ez correction from dHy/dx (x-CPML) ===
    # xlo: backward diff dHy/dx at integer x in strip [1, n+1)
    dhy_dx_xlo = (hy[1:n + 1, :] - hy[:n, :]) / dx
    psi_ezy_xlo = bx[:, None] * cpml_state.psi_ezy_xlo + cx[:, None] * dhy_dx_xlo
    inv_eps_xlo = 1.0 / eps[1:n + 1, :]
    ez = ez.at[1:n + 1, :].add(dt * inv_eps_xlo * psi_ezy_xlo)

    dhy_dx_xhi = (hy[-n:, :] - hy[-n - 1:-1, :]) / dx
    psi_ezy_xhi = bx_hi[:, None] * cpml_state.psi_ezy_xhi + cx_hi[:, None] * dhy_dx_xhi
    inv_eps_xhi = 1.0 / eps[-n:, :]
    ez = ez.at[-n:, :].add(dt * inv_eps_xhi * psi_ezy_xhi)

    # === Ez correction from dHx/dy (y-CPML) ===
    dhx_dy_ylo = (hx[:, 1:n + 1] - hx[:, :n]) / dy
    psi_ezx_ylo = by[None, :] * cpml_state.psi_ezx_ylo + cy[None, :] * dhx_dy_ylo
    inv_eps_ylo = 1.0 / eps[:, 1:n + 1]
    ez = ez.at[:, 1:n + 1].add(-dt * inv_eps_ylo * psi_ezx_ylo)  # negative: Ampere sign

    dhx_dy_yhi = (hx[:, -n:] - hx[:, -n - 1:-1]) / dy
    psi_ezx_yhi = by_hi[None, :] * cpml_state.psi_ezx_yhi + cy_hi[None, :] * dhx_dy_yhi
    inv_eps_yhi = 1.0 / eps[:, -n:]
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


# ---------------------------------------------------------------------------
# Full simulation loop
# ---------------------------------------------------------------------------

def run_adi_2d(ez: jnp.ndarray, hx: jnp.ndarray, hy: jnp.ndarray,
               eps_r: jnp.ndarray, sigma: jnp.ndarray,
               dt: float, dx: float, dy: float,
               n_steps: int,
               sources: list | None = None,
               probes: list | None = None,
               pec_mask: jnp.ndarray | None = None,
               cpml_params: ADICPMLParams2D | None = None,
               cpml_state: ADICPMLState2D | None = None):
    """Run a 2D TMz ADI-FDTD simulation for *n_steps* timesteps.

    Parameters
    ----------
    ez, hx, hy : (Nx, Ny) initial field arrays
    eps_r : (Nx, Ny) relative permittivity
    sigma : (Nx, Ny) conductivity
    dt, dx, dy : timestep and cell sizes
    n_steps : number of full timesteps
    sources : list of (i, j, waveform_array) tuples.
    probes : list of probe tuples.
    pec_mask : (Nx, Ny) bool array or None
    cpml_params : ADICPMLParams2D or None
        CPML profile coefficients. When provided, CPML absorbing boundary
        is applied after each ADI step (operator-splitting).
    cpml_state : ADICPMLState2D or None
        Initial CPML auxiliary state. Required when cpml_params is set.

    Returns
    -------
    ez, hx, hy : final field arrays
    probe_data : (n_steps, n_probes) array or None
    """
    use_cpml = cpml_params is not None
    if use_cpml and cpml_state is None:
        raise ValueError("cpml_state is required when cpml_params is provided")

    if sources is None:
        sources = []
    if probes is None:
        probes = []

    n_src = len(sources)
    n_prb = len(probes)

    if n_src > 0:
        src_ij = jnp.array([[s[0], s[1]] for s in sources], dtype=jnp.int32)
        src_waveforms = jnp.stack([s[2] for s in sources], axis=0)
    else:
        src_ij = jnp.zeros((0, 2), dtype=jnp.int32)
        src_waveforms = jnp.zeros((0, n_steps))

    if n_prb > 0:
        prb_meta = []
        for probe in probes:
            if len(probe) == 2:
                prb_meta.append((int(probe[0]), int(probe[1]), "ez"))
            elif len(probe) == 3 and probe[2] in {"ez", "hx", "hy"}:
                prb_meta.append((int(probe[0]), int(probe[1]), probe[2]))
            else:
                raise ValueError(
                    "ADI probes must be (i, j) or (i, j, component) with "
                    "component in {'ez', 'hx', 'hy'}"
                )
    else:
        prb_meta = []

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
                ez_s, hx_s, hy_s, eps_r, sigma, dt, dx, dy, pec_mask)

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
        (ez_f, hx_f, hy_f, _), probe_data = jax.lax.scan(
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
                ez_s, hx_s, hy_s, eps_r, sigma, dt, dx, dy, pec_mask)

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
        (ez_f, hx_f, hy_f), probe_data = jax.lax.scan(
            step_fn, init_state, jnp.arange(n_steps))

    if n_prb == 0:
        probe_data = None

    return ez_f, hx_f, hy_f, probe_data


# ===========================================================================
# 3D ADI-FDTD (Namiki/Zheng scheme)
# ===========================================================================

class ADIState3D(NamedTuple):
    """3D ADI-FDTD state: all 6 field components on (Nx, Ny, Nz) grid."""
    ex: jnp.ndarray
    ey: jnp.ndarray
    ez: jnp.ndarray
    hx: jnp.ndarray
    hy: jnp.ndarray
    hz: jnp.ndarray
    step: jnp.ndarray


def _apply_pec_3d(ex, ey, ez, pec_mask=None):
    """Zero tangential E on all 6 domain faces + internal PEC cells."""
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
    if pec_mask is not None:
        ex = jnp.where(pec_mask, 0.0, ex)
        ey = jnp.where(pec_mask, 0.0, ey)
        ez = jnp.where(pec_mask, 0.0, ez)
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
                pec_mask=None):
    r"""Advance all 6 field components by one full 3D ADI timestep.

    **EXPERIMENTAL**: Implements the Zheng et al. (2000) / Namiki (1999)
    scheme with two sub-steps per timestep, each with 3 independent
    tridiagonal solves. Stable for dt < ~0.5× CFL; unconditional stability
    requires further validation of the H-update formulation.

    Sub-step 1: Ey implicit in z, Ez implicit in x, Ex implicit in y
    Sub-step 2: Ex implicit in z, Ey implicit in x, Ez implicit in y

    Parameters
    ----------
    ex, ey, ez, hx, hy, hz : (Nx, Ny, Nz) field arrays
    eps_r, sigma : (Nx, Ny, Nz) material arrays
    dt, dx, dy, dz : scalars

    Returns
    -------
    ex, ey, ez, hx, hy, hz : updated fields
    """
    eps = eps_r * EPS_0
    half_dt = dt / 2.0

    # Implicit sigma integration (unconditionally stable)
    sigma_term = sigma * dt / 4.0
    eps_plus = eps + sigma_term
    damping = (eps - sigma_term) / eps_plus

    Cx = half_dt ** 2 / (MU_0 * eps_plus * dx * dx)
    Cy = half_dt ** 2 / (MU_0 * eps_plus * dy * dy)
    Cz = half_dt ** 2 / (MU_0 * eps_plus * dz * dz)

    coeff_e = half_dt / eps_plus  # for curl_H → E update

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

    # ===================================================================
    # Sub-step 1: n → n+1/2
    # Implicit pairs: Hx↔Ey(z), Hy↔Ez(x), Hz↔Ex(y)
    # ===================================================================

    # RHS uses full curl(H^n)
    curl_hx = (hz - _bwd(hz, 1)) / dy - (hy - _bwd(hy, 2)) / dz
    curl_hy = (hx - _bwd(hx, 2)) / dz - (hz - _bwd(hz, 0)) / dx
    curl_hz = (hy - _bwd(hy, 0)) / dx - (hx - _bwd(hx, 1)) / dy

    rhs_ex = damping * ex + coeff_e * curl_hx
    rhs_ey = damping * ey + coeff_e * curl_hy
    rhs_ez = damping * ez + coeff_e * curl_hz

    ey_h = _solve_tridiag_along(ey, Cz, rhs_ey, axis=2)  # Ey implicit in z
    ez_h = _solve_tridiag_along(ez, Cx, rhs_ez, axis=0)   # Ez implicit in x
    ex_h = _solve_tridiag_along(ex, Cy, rhs_ex, axis=1)   # Ex implicit in y

    ex_h, ey_h, ez_h = _apply_pec_3d(ex_h, ey_h, ez_h, pec_mask)

    # H^{n+1/2}: each uses NEW E from implicit pair, OLD E otherwise
    hx_h = hx - (half_dt / MU_0) * (
        (_fwd(ez, 1) - ez) / dy - (_fwd(ey_h, 2) - ey_h) / dz)
    hy_h = hy - (half_dt / MU_0) * (
        (_fwd(ex, 2) - ex) / dz - (_fwd(ez_h, 0) - ez_h) / dx)
    hz_h = hz - (half_dt / MU_0) * (
        (_fwd(ey, 0) - ey) / dx - (_fwd(ex_h, 1) - ex_h) / dy)

    # ===================================================================
    # Sub-step 2: n+1/2 → n+1
    # Implicit pairs: Hx↔Ez(y), Hy↔Ex(z), Hz↔Ey(x)
    # ===================================================================

    curl_hx2 = (hz_h - _bwd(hz_h, 1)) / dy - (hy_h - _bwd(hy_h, 2)) / dz
    curl_hy2 = (hx_h - _bwd(hx_h, 2)) / dz - (hz_h - _bwd(hz_h, 0)) / dx
    curl_hz2 = (hy_h - _bwd(hy_h, 0)) / dx - (hx_h - _bwd(hx_h, 1)) / dy

    rhs_ex2 = damping * ex_h + coeff_e * curl_hx2
    rhs_ey2 = damping * ey_h + coeff_e * curl_hy2
    rhs_ez2 = damping * ez_h + coeff_e * curl_hz2

    ex_n = _solve_tridiag_along(ex_h, Cz, rhs_ex2, axis=2)
    ey_n = _solve_tridiag_along(ey_h, Cx, rhs_ey2, axis=0)
    ez_n = _solve_tridiag_along(ez_h, Cy, rhs_ez2, axis=1)

    ex_n, ey_n, ez_n = _apply_pec_3d(ex_n, ey_n, ez_n, pec_mask)

    # H^{n+1}: NEW E from implicit pair, OLD E^* otherwise
    hx_n = hx_h - (half_dt / MU_0) * (
        (_fwd(ez_n, 1) - ez_n) / dy - (_fwd(ey_h, 2) - ey_h) / dz)
    hy_n = hy_h - (half_dt / MU_0) * (
        (_fwd(ex_n, 2) - ex_n) / dz - (_fwd(ez_h, 0) - ez_h) / dx)
    hz_n = hz_h - (half_dt / MU_0) * (
        (_fwd(ey_n, 0) - ey_n) / dx - (_fwd(ex_h, 1) - ex_h) / dy)

    return ex_n, ey_n, ez_n, hx_n, hy_n, hz_n


def make_adi_absorbing_sigma_3d(nx, ny, nz, n_layers, dx, dy, dz, order=3):
    """Graded conductivity for implicit ADI absorbing boundary in 3D."""
    import numpy as np
    eta = float(np.sqrt(MU_0 / EPS_0))

    sigma = np.zeros((nx, ny, nz), dtype=np.float64)

    for face_n in range(n_layers):
        rho = 1.0 - face_n / max(n_layers - 1, 1)

        # x-faces
        sx = 0.8 * (order + 1) / (eta * dx) * rho ** order
        sigma[face_n, :, :] = np.maximum(sigma[face_n, :, :], sx)
        sigma[nx - 1 - face_n, :, :] = np.maximum(sigma[nx - 1 - face_n, :, :], sx)

        # y-faces
        sy = 0.8 * (order + 1) / (eta * dy) * rho ** order
        sigma[:, face_n, :] = np.maximum(sigma[:, face_n, :], sy)
        sigma[:, ny - 1 - face_n, :] = np.maximum(sigma[:, ny - 1 - face_n, :], sy)

        # z-faces
        sz = 0.8 * (order + 1) / (eta * dz) * rho ** order
        sigma[:, :, face_n] = np.maximum(sigma[:, :, face_n], sz)
        sigma[:, :, nz - 1 - face_n] = np.maximum(sigma[:, :, nz - 1 - face_n], sz)

    return jnp.array(sigma, dtype=jnp.float32)
