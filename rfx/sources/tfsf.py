"""Total-Field/Scattered-Field (TFSF) plane-wave source.

Normal incidence along +x with Ez polarization (Ez, Hy plane wave).
Uses a 1D auxiliary FDTD with CPML to generate the incident field.

TFSF boundary cleanly separates:
  - Total-field region (x_lo <= x <= x_hi): incident + scattered
  - Scattered-field region (outside): scattered only

Reference: Taflove & Hagness, Ch. 5.

Corrections derived from Yee update stencil mismatch at TFSF boundary:
  E correction (after update_e):
    Ez[x_lo]   -= (dt/(eps0*dx)) * Hy_inc[x_lo - 0.5]
    Ez[x_hi+1] += (dt/(eps0*dx)) * Hy_inc[x_hi + 0.5]
  H correction (after update_h):
    Hy[x_lo-1] -= (dt/(mu0*dx)) * Ez_inc[x_lo]
    Hy[x_hi]   += (dt/(mu0*dx)) * Ez_inc[x_hi + 1]
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0


class TFSFState(NamedTuple):
    """1D auxiliary FDTD state for TFSF plane-wave generation."""
    e1d: jnp.ndarray   # Ez incident (n_1d,)
    h1d: jnp.ndarray   # Hy incident (n_1d,)
    psi_e_lo: jnp.ndarray  # 1D CPML psi for E at lo end
    psi_e_hi: jnp.ndarray
    psi_h_lo: jnp.ndarray
    psi_h_hi: jnp.ndarray


class TFSFConfig(NamedTuple):
    """Static TFSF configuration (compile-time constants for JIT)."""
    x_lo: int       # TFSF box x-lo index (inclusive, total-field starts here)
    x_hi: int       # TFSF box x-hi index (inclusive, total-field ends here)
    i0: int         # 1D index that maps to 3D x_lo
    src_idx: int    # 1D source injection index
    n_cpml: int     # Number of 1D CPML layers
    # 1D CPML coefficients
    b_cpml: jnp.ndarray   # (n_cpml,)
    c_cpml: jnp.ndarray
    # Source waveform (inlined for JIT)
    src_amp: float
    src_t0: float
    src_tau: float


def init_tfsf(
    nx: int,
    dx: float,
    dt: float,
    cpml_layers: int = 0,
    tfsf_margin: int = 3,
    f0: float = 5e9,
    bandwidth: float = 0.5,
    amplitude: float = 1.0,
) -> tuple[TFSFConfig, TFSFState]:
    """Initialize TFSF source for normal +x incidence, Ez polarization.

    Parameters
    ----------
    nx : int
        Number of cells in x (3D grid).
    dx, dt : float
        Cell size and timestep.
    cpml_layers : int
        Number of 3D CPML layers (TFSF box is inset from these).
    tfsf_margin : int
        Extra cells between CPML edge and TFSF boundary.
    f0, bandwidth, amplitude : float
        Gaussian pulse parameters.

    Returns
    -------
    (TFSFConfig, TFSFState)
    """
    # TFSF box boundaries in x
    offset = cpml_layers + tfsf_margin
    x_lo = offset
    x_hi = nx - offset - 1

    # 1D auxiliary grid: spans x_lo..x_hi + margins for source + CPML
    n_cpml_1d = 20
    n_margin = 10
    n_tfsf = x_hi - x_lo + 2   # +2 for the correction at x_hi+1
    n_1d = n_cpml_1d + n_margin + n_tfsf + n_margin + n_cpml_1d

    # 1D source index: in the margin before the TFSF mapping region
    src_idx = n_cpml_1d + 3
    # i0: 1D index that maps to 3D x_lo
    i0 = n_cpml_1d + n_margin

    # 1D CPML profile (polynomial grading)
    eta = np.sqrt(MU_0 / EPS_0)
    sigma_max = 0.8 * 4.0 / (eta * dx)
    rho = 1.0 - np.arange(n_cpml_1d, dtype=np.float64) / max(n_cpml_1d - 1, 1)
    sigma_prof = sigma_max * rho**3
    alpha_prof = 0.05 * (1.0 - rho)
    denom = sigma_prof + alpha_prof
    b_prof = np.exp(-(sigma_prof + alpha_prof) * dt / EPS_0)
    c_prof = np.where(denom > 1e-30, sigma_prof * (b_prof - 1.0) / denom, 0.0)

    # Source waveform parameters
    tau = 1.0 / (f0 * bandwidth * np.pi)
    t0 = 3.0 * tau

    config = TFSFConfig(
        x_lo=x_lo,
        x_hi=x_hi,
        i0=i0,
        src_idx=src_idx,
        n_cpml=n_cpml_1d,
        b_cpml=jnp.array(b_prof, dtype=jnp.float32),
        c_cpml=jnp.array(c_prof, dtype=jnp.float32),
        src_amp=float(amplitude),
        src_t0=float(t0),
        src_tau=float(tau),
    )

    state = TFSFState(
        e1d=jnp.zeros(n_1d, dtype=jnp.float32),
        h1d=jnp.zeros(n_1d, dtype=jnp.float32),
        psi_e_lo=jnp.zeros(n_cpml_1d, dtype=jnp.float32),
        psi_e_hi=jnp.zeros(n_cpml_1d, dtype=jnp.float32),
        psi_h_lo=jnp.zeros(n_cpml_1d, dtype=jnp.float32),
        psi_h_hi=jnp.zeros(n_cpml_1d, dtype=jnp.float32),
    )

    return config, state


def update_tfsf_1d(cfg: TFSFConfig, st: TFSFState, dx: float, dt: float,
                   t: float) -> TFSFState:
    """Advance the 1D auxiliary FDTD by one full timestep (convenience wrapper).

    Calls update_tfsf_1d_h then update_tfsf_1d_e. For correct leapfrog
    interleaving with the 3D grid, use the split functions directly:

        loop:
          1. update_h(3D)
          2. apply_tfsf_h(3D)        ← uses e1d at time n
          3. update_tfsf_1d_h(1D)    ← h1d → time n+1/2
          4. update_e(3D)
          5. apply_tfsf_e(3D)        ← uses h1d at time n+1/2
          6. update_tfsf_1d_e(1D, t) ← e1d → time n+1, inject source
    """
    st = update_tfsf_1d_h(cfg, st, dx, dt)
    st = update_tfsf_1d_e(cfg, st, dx, dt, t)
    return st


def update_tfsf_1d_h(cfg: TFSFConfig, st: TFSFState, dx: float,
                      dt: float) -> TFSFState:
    """Advance 1D auxiliary H field: h1d^{n-1/2} → h1d^{n+1/2}.

    Uses e1d at the current time level (time n).
    Call AFTER apply_tfsf_h (which reads e1d at time n) and BEFORE
    update_e / apply_tfsf_e (which need h1d at time n+1/2).
    """
    n = cfg.n_cpml
    e1d, h1d = st.e1d, st.h1d

    # Faraday 1D: ∂Hy/∂t = +(1/μ₀) ∂Ez/∂x  →  h[i] += (dt/mu0) * (e[i+1] - e[i]) / dx
    de = (jnp.concatenate([e1d[1:], jnp.zeros(1)]) - e1d) / dx
    h1d = h1d + (dt / MU_0) * de

    # H CPML lo end
    psi_h_lo = cfg.b_cpml * st.psi_h_lo + cfg.c_cpml * de[:n]
    h1d = h1d.at[:n].add((dt / MU_0) * psi_h_lo)

    # H CPML hi end
    b_hi = jnp.flip(cfg.b_cpml)
    c_hi = jnp.flip(cfg.c_cpml)
    psi_h_hi = b_hi * st.psi_h_hi + c_hi * de[-n:]
    h1d = h1d.at[-n:].add((dt / MU_0) * psi_h_hi)

    return st._replace(h1d=h1d, psi_h_lo=psi_h_lo, psi_h_hi=psi_h_hi)


def update_tfsf_1d_e(cfg: TFSFConfig, st: TFSFState, dx: float,
                      dt: float, t: float) -> TFSFState:
    """Advance 1D auxiliary E field: e1d^{n} → e1d^{n+1} + source injection.

    Uses h1d at time n+1/2 (must call update_tfsf_1d_h first).
    Call AFTER apply_tfsf_e (which reads h1d at time n+1/2).
    """
    n = cfg.n_cpml
    e1d, h1d = st.e1d, st.h1d

    # e[i] += (dt/eps0) * (h[i] - h[i-1]) / dx
    dh = (h1d - jnp.concatenate([jnp.zeros(1), h1d[:-1]])) / dx
    e1d = e1d + (dt / EPS_0) * dh

    # E CPML lo end
    psi_e_lo = cfg.b_cpml * st.psi_e_lo + cfg.c_cpml * dh[:n]
    e1d = e1d.at[:n].add((dt / EPS_0) * psi_e_lo)

    # E CPML hi end
    b_hi = jnp.flip(cfg.b_cpml)
    c_hi = jnp.flip(cfg.c_cpml)
    psi_e_hi = b_hi * st.psi_e_hi + c_hi * dh[-n:]
    e1d = e1d.at[-n:].add((dt / EPS_0) * psi_e_hi)

    # Inject source (differentiated Gaussian)
    arg = (t - cfg.src_t0) / cfg.src_tau
    src_val = cfg.src_amp * (-2.0 * arg) * jnp.exp(-(arg ** 2))
    e1d = e1d.at[cfg.src_idx].add(src_val)

    return st._replace(e1d=e1d, psi_e_lo=psi_e_lo, psi_e_hi=psi_e_hi)


def apply_tfsf_e(state, cfg: TFSFConfig, tfsf_st: TFSFState,
                  dx: float, dt: float):
    """Apply TFSF E-field correction (call AFTER update_e).

    Corrects Ez at x_lo and x_hi+1 where the H-curl stencil
    crosses the TFSF boundary.
    """
    coeff = dt / (EPS_0 * dx)
    i0 = cfg.i0

    # Hy_inc at position x_lo - 0.5 → h1d[i0 - 1]
    # (h1d[i] represents Hy at (i + 0.5) in 1D grid coords;
    #  h1d[i0-1] is at position i0 - 0.5, which maps to x_lo - 0.5 in 3D)
    hy_inc_lo = tfsf_st.h1d[i0 - 1]

    # Hy_inc at position x_hi + 0.5 → h1d[i0 + (x_hi - x_lo)]
    hy_inc_hi = tfsf_st.h1d[i0 + (cfg.x_hi - cfg.x_lo)]

    ez = state.ez
    # At x_lo: curl used Hy_scat[x_lo-1] instead of Hy_total → over-estimated
    ez = ez.at[cfg.x_lo, :, :].add(-coeff * hy_inc_lo)
    # At x_hi+1: curl used Hy_total[x_hi] instead of Hy_scat → under-estimated
    ez = ez.at[cfg.x_hi + 1, :, :].add(coeff * hy_inc_hi)

    return state._replace(ez=ez)


def apply_tfsf_h(state, cfg: TFSFConfig, tfsf_st: TFSFState,
                  dx: float, dt: float):
    """Apply TFSF H-field correction (call AFTER update_h).

    Corrects Hy at x_lo-1 and x_hi where the E-curl stencil
    crosses the TFSF boundary.
    """
    coeff = dt / (MU_0 * dx)
    i0 = cfg.i0

    # Ez_inc at position x_lo → e1d[i0]
    ez_inc_lo = tfsf_st.e1d[i0]

    # Ez_inc at position x_hi+1 → e1d[i0 + (x_hi+1 - x_lo)]
    ez_inc_hi = tfsf_st.e1d[i0 + (cfg.x_hi + 1 - cfg.x_lo)]

    hy = state.hy
    # At x_lo-1 (scattered): curl used Ez_total[x_lo] → over-estimated dEz/dx
    hy = hy.at[cfg.x_lo - 1, :, :].add(-coeff * ez_inc_lo)
    # At x_hi (total): curl used Ez_scat[x_hi+1] → under-estimated dEz/dx
    hy = hy.at[cfg.x_hi, :, :].add(coeff * ez_inc_hi)

    return state._replace(hy=hy)
