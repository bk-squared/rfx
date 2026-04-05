"""ADI-FDTD: Alternating Direction Implicit FDTD solver.

Unconditionally stable — dt is not limited by CFL condition.
For thin substrates where standard Yee requires tiny dt, ADI allows
10-100x larger timesteps.

Phase 1: 2D TMz (Ez, Hx, Hy) with PEC boundaries.

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

    # Courant-like coupling coefficient for the implicit direction
    Cx = half_dt * half_dt / (MU_0 * eps * dx * dx)  # (Nx, Ny)
    Cy = half_dt * half_dt / (MU_0 * eps * dy * dy)  # (Nx, Ny)

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

    rhs1 = ez + (half_dt / eps) * curl_h_n  # (Nx, Ny)

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

    rhs2 = ez_half + (half_dt / eps) * curl_h_half  # (Nx, Ny)

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
# Full simulation loop
# ---------------------------------------------------------------------------

def run_adi_2d(ez: jnp.ndarray, hx: jnp.ndarray, hy: jnp.ndarray,
               eps_r: jnp.ndarray, sigma: jnp.ndarray,
               dt: float, dx: float, dy: float,
               n_steps: int,
               sources: list | None = None,
               probes: list | None = None,
               pec_mask: jnp.ndarray | None = None):
    """Run a 2D TMz ADI-FDTD simulation for *n_steps* timesteps.

    Parameters
    ----------
    ez, hx, hy : (Nx, Ny) initial field arrays
    eps_r : (Nx, Ny) relative permittivity
    sigma : (Nx, Ny) conductivity
    dt, dx, dy : timestep and cell sizes
    n_steps : number of full timesteps
    sources : list of (i, j, waveform_array) tuples.
        ``waveform_array`` has length ``n_steps``.  At each step *n*,
        ``waveform_array[n]`` is added to ``Ez[i, j]``.
    probes : list of probe tuples.
        ``(i, j)`` records ``Ez`` for backward compatibility.
        ``(i, j, component)`` supports ``component`` in {``"ez"``, ``"hx"``, ``"hy"``}.
    pec_mask : (Nx, Ny) bool array or None
        Internal PEC occupancy for TMz. When provided, ``Ez`` is zeroed on
        those cells every half-step.

    Returns
    -------
    ez, hx, hy : final field arrays
    probe_data : (n_steps, n_probes) array of recorded Ez values,
        or None if no probes.
    """
    if float(jnp.max(jnp.abs(sigma))) > 0.0:
        raise NotImplementedError(
            "ADI-FDTD 2D integration currently supports lossless materials only; "
            "conductivity-aware ADI has not been implemented yet."
        )

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

    def step_fn(state, step_idx):
        ez_s, hx_s, hy_s = state

        # Inject sources (additive, soft source)
        def inject_one(carry, src_idx):
            ez_c = carry
            i = src_ij[src_idx, 0]
            j = src_ij[src_idx, 1]
            ez_c = ez_c.at[i, j].add(src_waveforms[src_idx, step_idx])
            return ez_c, None

        if n_src > 0:
            ez_s, _ = jax.lax.scan(inject_one, ez_s, jnp.arange(n_src))

        # ADI step
        ez_s, hx_s, hy_s = adi_step_2d(ez_s, hx_s, hy_s, eps_r, sigma, dt, dx, dy, pec_mask)

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

        return (ez_s, hx_s, hy_s), probe_vals

    init_state = (ez, hx, hy)
    (ez_f, hx_f, hy_f), probe_data = jax.lax.scan(
        step_fn, init_state, jnp.arange(n_steps)
    )

    if n_prb == 0:
        probe_data = None

    return ez_f, hx_f, hy_f, probe_data
