"""2D auxiliary grid for oblique-incidence TFSF plane waves.

Supports two modes:

**TMz** (xy-plane): Ez out-of-plane, Hx/Hy in-plane.
  Used for ``ez`` polarization oblique incidence.
  Periodic in y, CFS-CPML on x boundaries.

**TEz** (xz-plane): Ey out-of-plane, Hx/Hz in-plane.
  Used for ``ey`` polarization oblique incidence.
  Periodic in z, CFS-CPML on x boundaries.

Both modes use the same cell size ``dx`` as the 3D simulation so
numerical dispersion matches exactly at any angle.

The oblique plane wave source is a soft line source at a fixed x
with per-cell transverse delay that encodes the oblique angle.

Reference: S. C. Tan and G. E. Potter, IEEE T-AP 58(9), 2010.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TFSF2DState(NamedTuple):
    """2D auxiliary grid state for oblique TFSF.

    Field semantics depend on mode:
      TMz: ez_2d=Ez, hx_2d=Hx, hy_2d=Hy  (xy-plane)
      TEz: ez_2d=Ey, hx_2d=Hx, hy_2d=Hz  (xz-plane)
    """
    ez_2d: jnp.ndarray   # (n2x, n2t) — E out-of-plane
    hx_2d: jnp.ndarray   # (n2x, n2t) — H transverse component 1
    hy_2d: jnp.ndarray   # (n2x, n2t) — H longitudinal / component 2
    # CFS-CPML psi arrays (x-direction only, 4 total)
    psi_ez_xlo: jnp.ndarray  # (n_cpml, n2t)
    psi_ez_xhi: jnp.ndarray  # (n_cpml, n2t)
    psi_hy_xlo: jnp.ndarray  # (n_cpml, n2t)
    psi_hy_xhi: jnp.ndarray  # (n_cpml, n2t)
    step: jnp.ndarray


class TFSF2DConfig(NamedTuple):
    """Configuration for 2D auxiliary grid TFSF."""
    x_lo: int
    x_hi: int
    n2x: int
    n2y: int          # periodic transverse dim (ny for TMz, nz for TEz)
    i0_x: int         # 2D x-index mapping to 3D x_lo
    i0_y: int         # always 0 (periodic transverse, no padding)
    src_x: int
    src_amp: float
    src_t0: float
    src_tau: float
    theta: float
    cos_theta: float
    sin_theta: float
    electric_component: str
    magnetic_component: str
    curl_sign: float
    direction: str
    direction_sign: float
    transverse_axis: str
    n_cpml: int
    b_cpml: jnp.ndarray
    c_cpml: jnp.ndarray
    kappa_cpml: jnp.ndarray
    grid_pad: int
    angle_deg: float
    dx_1d: float
    mode: str         # "TMz" or "TEz"


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_tfsf_2d(
    nx: int,
    ny: int,
    dx: float,
    dt: float,
    *,
    nz: int | None = None,
    cpml_layers: int = 0,
    tfsf_margin: int = 3,
    f0: float = 5e9,
    bandwidth: float = 0.5,
    amplitude: float = 1.0,
    polarization: str = "ez",
    direction: str = "+x",
    theta_deg: float = 0.0,
) -> tuple[TFSF2DConfig, TFSF2DState]:
    """Initialize 2D auxiliary grid for oblique-incidence TFSF.

    TMz mode (ez polarization): 2D grid in xy-plane, periodic in y.
    TEz mode (ey polarization): 2D grid in xz-plane, periodic in z.
    CFS-CPML on x boundaries only.

    Parameters
    ----------
    nz : int | None
        Number of cells in z (3D grid).  Required for ey oblique
        (TEz mode).  Ignored for ez polarization.
    """
    if polarization not in ("ez", "ey"):
        raise ValueError(f"polarization must be 'ez' or 'ey', got {polarization!r}")
    if direction not in ("+x", "-x"):
        raise ValueError(f"direction must be '+x' or '-x', got {direction!r}")
    if abs(theta_deg) >= 90.0:
        raise ValueError(f"|angle_deg| must be < 90, got {theta_deg}")

    # Determine mode
    if polarization == "ez":
        mode = "TMz"
    else:
        mode = "TEz"

    theta = np.radians(theta_deg)
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))

    # TFSF box boundaries in x
    offset = cpml_layers + tfsf_margin
    x_lo = offset
    x_hi = nx - offset - 1

    if x_lo <= 0 or x_hi >= nx - 1 or x_lo >= x_hi:
        raise ValueError(
            "TFSF margin/cpml_layers too large for the grid: "
            f"nx={nx}, cpml_layers={cpml_layers}, tfsf_margin={tfsf_margin}"
        )

    # ---- 2D grid sizing ----
    # Periodic transverse dimension: y for TMz, z for TEz
    if mode == "TMz":
        n2y = ny
    else:
        if nz is None:
            raise ValueError(
                "nz is required for ey oblique TFSF (TEz mode)"
            )
        n2y = nz  # periodic axis is z for TEz

    # x-direction: TFSF span + margins + CPML on both ends
    n_cpml_2d = 30
    n_tfsf_x = x_hi - x_lo + 2
    n_margin_x = 25

    n2x = n_cpml_2d + n_margin_x + n_tfsf_x + n_margin_x + n_cpml_2d

    # i0_x: 2D x-index that maps to 3D x_lo
    i0_x = n_cpml_2d + n_margin_x
    i0_y = 0  # periodic transverse, no padding

    # Source position
    if direction == "+x":
        src_x = n_cpml_2d + 3
    else:
        src_x = n2x - n_cpml_2d - 4

    # CFS-CPML profile (x-direction only), 4th order
    cpml_order = 4
    kappa_max = 7.0
    eta = np.sqrt(MU_0 / EPS_0)
    sigma_max = 0.8 * (cpml_order + 1) / (eta * dx) * kappa_max
    rho = 1.0 - np.arange(n_cpml_2d, dtype=np.float64) / max(n_cpml_2d - 1, 1)
    sigma_prof = sigma_max * rho ** cpml_order
    kappa_prof = 1.0 + (kappa_max - 1.0) * rho ** cpml_order
    alpha_prof = 0.05 * (1.0 - rho)
    denom = sigma_prof * kappa_prof + kappa_prof ** 2 * alpha_prof
    b_prof = np.exp(-(sigma_prof / kappa_prof + alpha_prof) * dt / EPS_0)
    c_prof = np.where(denom > 1e-30, sigma_prof * (b_prof - 1.0) / denom, 0.0)

    # Source waveform
    tau = 1.0 / (f0 * bandwidth * np.pi)
    t0 = 3.0 * tau

    if polarization == "ez":
        electric_component = "ez"
        magnetic_component = "hy"
        curl_sign = 1.0
        transverse_axis = "y"
    else:
        electric_component = "ey"
        magnetic_component = "hz"
        curl_sign = -1.0
        transverse_axis = "z"

    direction_sign = 1.0 if direction == "+x" else -1.0

    config = TFSF2DConfig(
        x_lo=x_lo, x_hi=x_hi,
        n2x=n2x, n2y=n2y,
        i0_x=i0_x, i0_y=i0_y,
        src_x=src_x,
        src_amp=float(amplitude),
        src_t0=float(t0),
        src_tau=float(tau),
        theta=float(theta),
        cos_theta=cos_theta,
        sin_theta=sin_theta,
        electric_component=electric_component,
        magnetic_component=magnetic_component,
        curl_sign=float(curl_sign),
        direction=direction,
        direction_sign=float(direction_sign),
        transverse_axis=transverse_axis,
        n_cpml=n_cpml_2d,
        b_cpml=jnp.array(b_prof, dtype=jnp.float32),
        c_cpml=jnp.array(c_prof, dtype=jnp.float32),
        kappa_cpml=jnp.array(kappa_prof, dtype=jnp.float32),
        grid_pad=cpml_layers,
        angle_deg=float(theta_deg),
        dx_1d=float(dx),
        mode=mode,
    )

    state = TFSF2DState(
        ez_2d=jnp.zeros((n2x, n2y), dtype=jnp.float32),
        hx_2d=jnp.zeros((n2x, n2y), dtype=jnp.float32),
        hy_2d=jnp.zeros((n2x, n2y), dtype=jnp.float32),
        psi_ez_xlo=jnp.zeros((n_cpml_2d, n2y), dtype=jnp.float32),
        psi_ez_xhi=jnp.zeros((n_cpml_2d, n2y), dtype=jnp.float32),
        psi_hy_xlo=jnp.zeros((n_cpml_2d, n2y), dtype=jnp.float32),
        psi_hy_xhi=jnp.zeros((n_cpml_2d, n2y), dtype=jnp.float32),
        step=jnp.array(0, dtype=jnp.int32),
    )

    return config, state


# ---------------------------------------------------------------------------
# CPML helper (shared by both modes)
# ---------------------------------------------------------------------------

def _apply_cpml_h(cfg, st, dez_dx, hy, coeff_h):
    """Apply CFS-CPML corrections to the x-derivative H component."""
    n = cfg.n_cpml

    b_lo = cfg.b_cpml[:, None]
    c_lo = cfg.c_cpml[:, None]
    k_lo = cfg.kappa_cpml[:, None]

    psi_hy_xlo = b_lo * st.psi_hy_xlo + c_lo * dez_dx[:n, :]
    hy = hy.at[:n, :].add(coeff_h * psi_hy_xlo)
    hy = hy.at[:n, :].add(coeff_h * (1.0 / k_lo - 1.0) * dez_dx[:n, :])

    b_hi = jnp.flip(cfg.b_cpml)[:, None]
    c_hi = jnp.flip(cfg.c_cpml)[:, None]
    k_hi = jnp.flip(cfg.kappa_cpml)[:, None]

    psi_hy_xhi = b_hi * st.psi_hy_xhi + c_hi * dez_dx[-n:, :]
    hy = hy.at[-n:, :].add(coeff_h * psi_hy_xhi)
    hy = hy.at[-n:, :].add(coeff_h * (1.0 / k_hi - 1.0) * dez_dx[-n:, :])

    return hy, psi_hy_xlo, psi_hy_xhi


def _apply_cpml_e(cfg, st, dhy_dx, ez, coeff_e):
    """Apply CFS-CPML corrections to the x-derivative E component."""
    n = cfg.n_cpml

    b_lo = cfg.b_cpml[:, None]
    c_lo = cfg.c_cpml[:, None]
    k_lo = cfg.kappa_cpml[:, None]

    psi_ez_xlo = b_lo * st.psi_ez_xlo + c_lo * dhy_dx[:n, :]
    ez = ez.at[:n, :].add(coeff_e * psi_ez_xlo)
    ez = ez.at[:n, :].add(coeff_e * (1.0 / k_lo - 1.0) * dhy_dx[:n, :])

    b_hi = jnp.flip(cfg.b_cpml)[:, None]
    c_hi = jnp.flip(cfg.c_cpml)[:, None]
    k_hi = jnp.flip(cfg.kappa_cpml)[:, None]

    psi_ez_xhi = b_hi * st.psi_ez_xhi + c_hi * dhy_dx[-n:, :]
    ez = ez.at[-n:, :].add(coeff_e * psi_ez_xhi)
    ez = ez.at[-n:, :].add(coeff_e * (1.0 / k_hi - 1.0) * dhy_dx[-n:, :])

    return ez, psi_ez_xlo, psi_ez_xhi


# ---------------------------------------------------------------------------
# 2D Yee update — TMz mode (Ez, Hx, Hy) — periodic y, CPML x
# ---------------------------------------------------------------------------

def _update_h_tmz(cfg, st, dx, dt):
    """TMz H update: Hx -= (dt/mu0)*dEz/dy,  Hy += (dt/mu0)*dEz/dx."""
    ez = st.ez_2d
    hx, hy = st.hx_2d, st.hy_2d
    coeff_h = dt / MU_0

    # dEz/dy -- periodic (roll -1 along y=axis1)
    dez_dy = (jnp.roll(ez, -1, axis=1) - ez) / dx

    # dEz/dx -- zero-pad forward difference along x=axis0
    dez_dx = (jnp.concatenate([ez[1:, :], jnp.zeros((1, ez.shape[1]))], axis=0) - ez) / dx

    hx = hx - coeff_h * dez_dy
    hy = hy + coeff_h * dez_dx

    # CFS-CPML for Hy (x-direction only)
    hy, psi_hy_xlo, psi_hy_xhi = _apply_cpml_h(cfg, st, dez_dx, hy, coeff_h)

    return st._replace(
        hx_2d=hx, hy_2d=hy,
        psi_hy_xlo=psi_hy_xlo, psi_hy_xhi=psi_hy_xhi,
    )


def _update_e_tmz(cfg, st, dx, dt, t):
    """TMz E update: Ez += (dt/eps0)*(dHy/dx - dHx/dy) + source."""
    ez = st.ez_2d
    hx, hy = st.hx_2d, st.hy_2d
    coeff_e = dt / EPS_0

    # dHy/dx -- zero-pad backward difference along x
    dhy_dx = (hy - jnp.concatenate([jnp.zeros((1, hy.shape[1])), hy[:-1, :]], axis=0)) / dx

    # dHx/dy -- periodic (roll +1 along y for backward difference)
    dhx_dy = (hx - jnp.roll(hx, 1, axis=1)) / dx

    ez = ez + coeff_e * (dhy_dx - dhx_dy)

    # CFS-CPML for Ez (x-direction, from dHy/dx)
    ez, psi_ez_xlo, psi_ez_xhi = _apply_cpml_e(cfg, st, dhy_dx, ez, coeff_e)

    # ---- Inject oblique plane-wave source along a line ----
    c0 = 1.0 / jnp.sqrt(jnp.float32(EPS_0 * MU_0))
    j_indices = jnp.arange(ez.shape[1], dtype=jnp.float32)
    y_offset = j_indices * dx
    t_delay = cfg.direction_sign * cfg.sin_theta * y_offset / c0
    arg = (t - t_delay - cfg.src_t0) / cfg.src_tau
    src_line = cfg.src_amp * (-2.0 * arg) * jnp.exp(-(arg ** 2))
    ez = ez.at[cfg.src_x, :].add(src_line)

    return st._replace(
        ez_2d=ez,
        psi_ez_xlo=psi_ez_xlo, psi_ez_xhi=psi_ez_xhi,
        step=st.step + 1,
    )


# ---------------------------------------------------------------------------
# 2D Yee update — TEz mode (Ey, Hx, Hz) — periodic z, CPML x
#
# State field mapping: ez_2d -> Ey, hx_2d -> Hx, hy_2d -> Hz
#
# Maxwell for TEz (xz-plane, Ey out-of-plane):
#   dHx/dt =  (1/mu0) * dEy/dz
#   dHz/dt = -(1/mu0) * dEy/dx
#   dEy/dt =  (1/eps0) * (dHx/dz - dHz/dx)
# ---------------------------------------------------------------------------

def _update_h_tez(cfg, st, dx, dt):
    """TEz H update: Hx += (dt/mu0)*dEy/dz,  Hz -= (dt/mu0)*dEy/dx."""
    ey = st.ez_2d   # Ey stored in ez_2d slot
    hx = st.hx_2d   # Hx
    hz = st.hy_2d   # Hz stored in hy_2d slot
    coeff_h = dt / MU_0

    # dEy/dz -- periodic (roll -1 along z=axis1)
    dey_dz = (jnp.roll(ey, -1, axis=1) - ey) / dx

    # dEy/dx -- zero-pad forward difference along x=axis0
    dey_dx = (jnp.concatenate([ey[1:, :], jnp.zeros((1, ey.shape[1]))], axis=0) - ey) / dx

    # TEz Faraday: Hx += coeff * dEy/dz,  Hz -= coeff * dEy/dx
    hx = hx + coeff_h * dey_dz
    hz = hz - coeff_h * dey_dx

    # CFS-CPML for Hz (x-direction only) — note the sign: Hz -= coeff*dEy/dx
    # so CPML correction sign is -coeff_h
    hz, psi_hy_xlo, psi_hy_xhi = _apply_cpml_h(cfg, st, dey_dx, hz, -coeff_h)

    return st._replace(
        hx_2d=hx, hy_2d=hz,
        psi_hy_xlo=psi_hy_xlo, psi_hy_xhi=psi_hy_xhi,
    )


def _update_e_tez(cfg, st, dx, dt, t):
    """TEz E update: Ey += (dt/eps0)*(dHx/dz - dHz/dx) + source."""
    ey = st.ez_2d   # Ey stored in ez_2d slot
    hx = st.hx_2d   # Hx
    hz = st.hy_2d   # Hz stored in hy_2d slot
    coeff_e = dt / EPS_0

    # dHx/dz -- periodic (roll +1 along z for backward difference)
    dhx_dz = (hx - jnp.roll(hx, 1, axis=1)) / dx

    # dHz/dx -- zero-pad backward difference along x
    dhz_dx = (hz - jnp.concatenate([jnp.zeros((1, hz.shape[1])), hz[:-1, :]], axis=0)) / dx

    # TEz Ampere: Ey += coeff * (dHx/dz - dHz/dx)
    ey = ey + coeff_e * (dhx_dz - dhz_dx)

    # CFS-CPML for Ey (x-direction, from -dHz/dx term)
    # The x-derivative contribution is -coeff_e * dHz/dx, so CPML sign is -coeff_e
    ey, psi_ez_xlo, psi_ez_xhi = _apply_cpml_e(cfg, st, dhz_dx, ey, -coeff_e)

    # ---- Inject oblique plane-wave source along a line ----
    c0 = 1.0 / jnp.sqrt(jnp.float32(EPS_0 * MU_0))
    j_indices = jnp.arange(ey.shape[1], dtype=jnp.float32)
    z_offset = j_indices * dx
    t_delay = cfg.direction_sign * cfg.sin_theta * z_offset / c0
    arg = (t - t_delay - cfg.src_t0) / cfg.src_tau
    src_line = cfg.src_amp * (-2.0 * arg) * jnp.exp(-(arg ** 2))
    ey = ey.at[cfg.src_x, :].add(src_line)

    return st._replace(
        ez_2d=ey,
        psi_ez_xlo=psi_ez_xlo, psi_ez_xhi=psi_ez_xhi,
        step=st.step + 1,
    )


# ---------------------------------------------------------------------------
# Mode-dispatching update entry points
# ---------------------------------------------------------------------------

def update_tfsf_2d_h(cfg: TFSF2DConfig, st: TFSF2DState,
                      dx: float, dt: float) -> TFSF2DState:
    """Advance 2D auxiliary H: H^{n-1/2} -> H^{n+1/2}.

    Dispatches to TMz or TEz based on ``cfg.mode``.
    """
    if cfg.mode == "TEz":
        return _update_h_tez(cfg, st, dx, dt)
    return _update_h_tmz(cfg, st, dx, dt)


def update_tfsf_2d_e(cfg: TFSF2DConfig, st: TFSF2DState,
                      dx: float, dt: float, t: float) -> TFSF2DState:
    """Advance 2D auxiliary E: E^n -> E^{n+1} + source injection.

    Dispatches to TMz or TEz based on ``cfg.mode``.
    """
    if cfg.mode == "TEz":
        return _update_e_tez(cfg, st, dx, dt, t)
    return _update_e_tmz(cfg, st, dx, dt, t)


def update_tfsf_2d(cfg: TFSF2DConfig, st: TFSF2DState,
                    dx: float, dt: float, t: float) -> TFSF2DState:
    """Advance 2D auxiliary grid by one full timestep (convenience)."""
    st = update_tfsf_2d_h(cfg, st, dx, dt)
    st = update_tfsf_2d_e(cfg, st, dx, dt, t)
    return st


# ---------------------------------------------------------------------------
# Sample incident fields from 2D grid
# ---------------------------------------------------------------------------

def _sample_e_at_x(cfg: TFSF2DConfig, st: TFSF2DState,
                    ix_2d: int, n_trans: int) -> jnp.ndarray:
    """Sample E (out-of-plane) from 2D grid at x-index. Returns (n_trans,)."""
    return st.ez_2d[ix_2d, :n_trans]


def _sample_h2_at_x(cfg: TFSF2DConfig, st: TFSF2DState,
                     ix_2d: int, n_trans: int) -> jnp.ndarray:
    """Sample H2 (x-derivative component: Hy for TMz, Hz for TEz).

    Returns (n_trans,).
    """
    return st.hy_2d[ix_2d, :n_trans]


# Keep old names as aliases for backward compatibility
_sample_ez_at_x = _sample_e_at_x
_sample_hy_at_x = _sample_h2_at_x


# ---------------------------------------------------------------------------
# 3D TFSF corrections using 2D auxiliary grid
# ---------------------------------------------------------------------------

def apply_tfsf_2d_e(state, cfg: TFSF2DConfig, tfsf_st: TFSF2DState,
                     dx: float, dt: float):
    """Apply E-field TFSF correction using 2D aux grid (call AFTER update_e).

    TMz mode: incident fields vary in y, broadcast along z.
      E_inc[x_lo, j, :] -= coeff * H_aux[x_lo-0.5, j]
      E_inc[x_hi+1, j, :] += coeff * H_aux[x_hi+0.5, j]

    TEz mode: incident fields vary in z, broadcast along y.
      E_inc[x_lo, :, k] -= coeff * H_aux[x_lo-0.5, k]
      E_inc[x_hi+1, :, k] += coeff * H_aux[x_hi+0.5, k]

    Signs are fixed (-coeff at x_lo, +coeff at x_hi+1) for both modes.
    """
    coeff = dt / (EPS_0 * dx)
    i0 = cfg.i0_x
    e_ref = getattr(state, cfg.electric_component)
    ny_3d = e_ref.shape[1]
    nz_3d = e_ref.shape[2]

    if cfg.mode == "TEz":
        # TEz: 2D grid transverse axis is z, broadcast along y
        h_inc_lo = _sample_h2_at_x(cfg, tfsf_st, i0 - 1, nz_3d)
        h_inc_hi = _sample_h2_at_x(cfg, tfsf_st, i0 + (cfg.x_hi - cfg.x_lo), nz_3d)

        h_lo_3d = jnp.broadcast_to(h_inc_lo[None, :], (ny_3d, nz_3d))
        h_hi_3d = jnp.broadcast_to(h_inc_hi[None, :], (ny_3d, nz_3d))
    else:
        # TMz: 2D grid transverse axis is y, broadcast along z
        h_inc_lo = _sample_h2_at_x(cfg, tfsf_st, i0 - 1, ny_3d)
        h_inc_hi = _sample_h2_at_x(cfg, tfsf_st, i0 + (cfg.x_hi - cfg.x_lo), ny_3d)

        h_lo_3d = jnp.broadcast_to(h_inc_lo[:, None], (ny_3d, nz_3d))
        h_hi_3d = jnp.broadcast_to(h_inc_hi[:, None], (ny_3d, nz_3d))

    # For TMz (Ez/Hy, curl_sign=+1): Ampere has +dHy/dx, so correction
    # sign is (-coeff, +coeff).  For TEz (Ey/Hz, curl_sign=-1): Ampere
    # has -dHz/dx, so correction sign is (+coeff, -coeff).  Using
    # curl_sign unifies both: (-curl_sign*coeff, +curl_sign*coeff).
    e_field = e_ref
    e_field = e_field.at[cfg.x_lo, :, :].add(-cfg.curl_sign * coeff * h_lo_3d)
    e_field = e_field.at[cfg.x_hi + 1, :, :].add(cfg.curl_sign * coeff * h_hi_3d)

    return state._replace(**{cfg.electric_component: e_field})


def apply_tfsf_2d_h(state, cfg: TFSF2DConfig, tfsf_st: TFSF2DState,
                     dx: float, dt: float):
    """Apply H-field TFSF correction using 2D aux grid (call AFTER update_h).

    TMz mode: incident fields vary in y, broadcast along z.
    TEz mode: incident fields vary in z, broadcast along y.

    H_mag[x_lo-1, ...] -= curl_sign * coeff * E_aux[x_lo, ...]
    H_mag[x_hi, ...]   += curl_sign * coeff * E_aux[x_hi+1, ...]
    """
    coeff = dt / (MU_0 * dx)
    i0 = cfg.i0_x
    h_ref = getattr(state, cfg.magnetic_component)
    ny_3d = h_ref.shape[1]
    nz_3d = h_ref.shape[2]

    if cfg.mode == "TEz":
        # TEz: 2D grid transverse axis is z, broadcast along y
        e_inc_lo = _sample_e_at_x(cfg, tfsf_st, i0, nz_3d)
        e_inc_hi = _sample_e_at_x(cfg, tfsf_st, i0 + (cfg.x_hi + 1 - cfg.x_lo), nz_3d)

        e_lo_3d = jnp.broadcast_to(e_inc_lo[None, :], (ny_3d, nz_3d))
        e_hi_3d = jnp.broadcast_to(e_inc_hi[None, :], (ny_3d, nz_3d))
    else:
        # TMz: 2D grid transverse axis is y, broadcast along z
        e_inc_lo = _sample_e_at_x(cfg, tfsf_st, i0, ny_3d)
        e_inc_hi = _sample_e_at_x(cfg, tfsf_st, i0 + (cfg.x_hi + 1 - cfg.x_lo), ny_3d)

        e_lo_3d = jnp.broadcast_to(e_inc_lo[:, None], (ny_3d, nz_3d))
        e_hi_3d = jnp.broadcast_to(e_inc_hi[:, None], (ny_3d, nz_3d))

    h_field = h_ref
    h_field = h_field.at[cfg.x_lo - 1, :, :].add(-cfg.curl_sign * coeff * e_lo_3d)
    h_field = h_field.at[cfg.x_hi, :, :].add(cfg.curl_sign * coeff * e_hi_3d)

    return state._replace(**{cfg.magnetic_component: h_field})
