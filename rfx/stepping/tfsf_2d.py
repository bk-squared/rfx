"""Two-dimensional auxiliary TF/SF field updates and Bloch phases."""

from __future__ import annotations

from rfx.core.yee import EPS_0
from rfx.core.yee import MU_0
from rfx.core.yee import MaterialArrays
from rfx.core.yee import component_h_materials
from rfx.sources.tfsf_2d import TFSF2DConfig
from rfx.sources.tfsf_2d import TFSF2DState
from rfx.sources.tfsf_2d import update_tfsf_2d_e
from rfx.sources.tfsf_2d import update_tfsf_2d_h
import jax.numpy as jnp


def update_tfsf_2d(cfg: TFSF2DConfig, st: TFSF2DState,
                    dx: float, dt: float, t: float) -> TFSF2DState:
    """Advance 2D auxiliary grid by one full timestep (convenience)."""
    st = update_tfsf_2d_h(cfg, st, dx, dt)
    st = update_tfsf_2d_e(cfg, st, dx, dt, t)
    return st


def _update_e_tez(cfg, st, dx, dt, t):
    """TEz E update: Ey += (dt/eps0)*(dHx/dz - dHz/dx) + source."""
    ey = st.ez_2d   # Ey stored in ez_2d slot
    hx = st.hx_2d   # Hx
    hz = st.hy_2d   # Hz stored in hy_2d slot
    coeff_e = dt / EPS_0

    # dHx/dz -- Bloch field-transform (#404): plain roll + exp(+j k_z dz) on the
    # backward neighbour (conjugate of the forward pshift; k_z=0 -> plain diff).
    pconj = jnp.exp(1j * cfg.k_transverse * dx)
    dhx_dz = (hx - jnp.roll(hx, 1, axis=1) * pconj) / dx

    # dHz/dx -- zero-pad backward difference along x
    dhz_dx = (hz - jnp.concatenate([jnp.zeros((1, hz.shape[1]), hz.dtype), hz[:-1, :]], axis=0)) / dx

    # TEz Ampere: Ey += coeff * (dHx/dz - dHz/dx)
    ey = ey + coeff_e * (dhx_dz - dhz_dx)

    # CFS-CPML for Ey (x-direction, from -dHz/dx term)
    # The x-derivative contribution is -coeff_e * dHz/dx, so CPML sign is -coeff_e
    ey, psi_ez_xlo, psi_ez_xhi = _apply_cpml_e(cfg, st, dhz_dx, ey, -coeff_e)

    # ---- Inject oblique plane wave: z-uniform complex analytic modulated
    # Gaussian (carrier at f0). The transverse k_z is carried by the transformed
    # derivatives above, not by a per-cell time delay. ----
    arg = (t - cfg.src_t0) / cfg.src_tau
    src_val = (cfg.src_amp
               * jnp.exp(-1j * 2.0 * jnp.pi * cfg.src_fcen * (t - cfg.src_t0))
               * jnp.exp(-(arg ** 2)))
    ey = ey.at[cfg.src_x, :].add(src_val)

    return st._replace(
        ez_2d=ey,
        psi_ez_xlo=psi_ez_xlo, psi_ez_xhi=psi_ez_xhi,
        step=st.step + 1,
    )


def _update_e_tmz(cfg, st, dx, dt, t):
    """TMz E update: Ez += (dt/eps0)*(dHy/dx - dHx/dy) + source."""
    ez = st.ez_2d
    hx, hy = st.hx_2d, st.hy_2d
    coeff_e = dt / EPS_0

    # dHy/dx -- zero-pad backward difference along x
    dhy_dx = (hy - jnp.concatenate([jnp.zeros((1, hy.shape[1]), hy.dtype), hy[:-1, :]], axis=0)) / dx

    # dHx/dy -- Bloch field-transform (#404): plain roll + exp(+j k_y dx) on the
    # backward neighbour (conjugate of the forward pshift; k_y=0 -> plain diff).
    pconj = jnp.exp(1j * cfg.k_transverse * dx)
    dhx_dy = (hx - jnp.roll(hx, 1, axis=1) * pconj) / dx

    ez = ez + coeff_e * (dhy_dx - dhx_dy)

    # CFS-CPML for Ez (x-direction, from dHy/dx)
    ez, psi_ez_xlo, psi_ez_xhi = _apply_cpml_e(cfg, st, dhy_dx, ez, coeff_e)

    # ---- Inject oblique plane wave: y-uniform complex analytic modulated
    # Gaussian (carrier at f0). The transverse k_y is carried by the transformed
    # derivatives above, not by a per-cell time delay. ----
    arg = (t - cfg.src_t0) / cfg.src_tau
    src_val = (cfg.src_amp
               * jnp.exp(-1j * 2.0 * jnp.pi * cfg.src_fcen * (t - cfg.src_t0))
               * jnp.exp(-(arg ** 2)))
    ez = ez.at[cfg.src_x, :].add(src_val)

    return st._replace(
        ez_2d=ez,
        psi_ez_xlo=psi_ez_xlo, psi_ez_xhi=psi_ez_xhi,
        step=st.step + 1,
    )


def _update_h_tez(cfg, st, dx, dt):
    """TEz H update: Hx += (dt/mu0)*dEy/dz,  Hz -= (dt/mu0)*dEy/dx."""
    mu_abs = component_h_materials(MaterialArrays(None, None, 1.0))[0] * MU_0
    ey = st.ez_2d   # Ey stored in ez_2d slot
    hx = st.hx_2d   # Hx
    hz = st.hy_2d   # Hz stored in hy_2d slot
    coeff_h = dt / mu_abs

    # dEy/dz -- Bloch field-transform (#404): plain roll + exp(-j k_z dz) on the
    # forward neighbour (transverse axis is z here; k_z=0 -> plain periodic diff).
    pshift = jnp.exp(-1j * cfg.k_transverse * dx)
    dey_dz = (jnp.roll(ey, -1, axis=1) * pshift - ey) / dx

    # dEy/dx -- zero-pad forward difference along x=axis0
    dey_dx = (jnp.concatenate([ey[1:, :], jnp.zeros((1, ey.shape[1]), ey.dtype)], axis=0) - ey) / dx

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


def _update_h_tmz(cfg, st, dx, dt):
    """TMz H update: Hx -= (dt/mu0)*dEz/dy,  Hy += (dt/mu0)*dEz/dx."""
    mu_abs = component_h_materials(MaterialArrays(None, None, 1.0))[0] * MU_0
    ez = st.ez_2d
    hx, hy = st.hx_2d, st.hy_2d
    coeff_h = dt / mu_abs

    # dEz/dy -- Bloch field-transform (#404): plain roll + exp(-j k_y dx) on the
    # forward neighbour. k_y=0 (normal) -> pshift=1 -> ordinary periodic diff.
    pshift = jnp.exp(-1j * cfg.k_transverse * dx)
    dez_dy = (jnp.roll(ez, -1, axis=1) * pshift - ez) / dx

    # dEz/dx -- zero-pad forward difference along x=axis0
    dez_dx = (jnp.concatenate([ez[1:, :], jnp.zeros((1, ez.shape[1]), ez.dtype)], axis=0) - ez) / dx

    hx = hx - coeff_h * dez_dy
    hy = hy + coeff_h * dez_dx

    # CFS-CPML for Hy (x-direction only)
    hy, psi_hy_xlo, psi_hy_xhi = _apply_cpml_h(cfg, st, dez_dx, hy, coeff_h)

    return st._replace(
        hx_2d=hx, hy_2d=hy,
        psi_hy_xlo=psi_hy_xlo, psi_hy_xhi=psi_hy_xhi,
    )


def _transverse_phase(cfg: TFSF2DConfig, n_trans: int, dx: float) -> jnp.ndarray:
    """Transverse Bloch phase exp(-j·k_y·y) to re-apply to the (complex,
    y-uniform) aux fields when injecting into the real 3D grid (#404).

    ``k_y = cfg.k_transverse`` is 0 for normal incidence (this path is oblique
    only), giving phase = 1 and recovering the plain broadcast.
    """
    y = jnp.arange(n_trans, dtype=jnp.float32) * dx
    return jnp.exp(-1j * cfg.k_transverse * y)


def bloch_phase_tuple(cfg: TFSF2DConfig, dx: float) -> tuple:
    """Per-axis Bloch phase ``exp(-j·k_transverse·dx)`` for the 3D
    ``update_e``/``update_h`` on the transverse periodic axis (#404 Phase-B).

    The transformed-frame 3D grid evolves the Bloch envelope ``P`` and must carry
    the SAME transverse phase on its periodic roll as the 2D-aux grid, on the axis
    matching ``cfg.transverse_axis`` (``y``=axis 1 for TMz, ``z``=axis 2 for TEz).
    All other axes are 1.0.  Pass the result as ``update_e/h(..., bloch=...)``.
    """
    ph = complex(jnp.exp(-1j * cfg.k_transverse * dx))
    if cfg.transverse_axis == "z":
        return (1.0 + 0j, 1.0 + 0j, ph)
    return (1.0 + 0j, ph, 1.0 + 0j)


def _apply_cpml_h(cfg, st, dez_dx, hy, coeff_h):
    """Apply CFS-CPML corrections to the x-derivative H component."""
    n = cfg.n_cpml

    b_lo = cfg.magnetic_lo.b[:, None]
    c_lo = cfg.magnetic_lo.c[:, None]
    k_lo = cfg.magnetic_lo.kappa[:, None]

    psi_hy_xlo = b_lo * st.psi_hy_xlo + c_lo * dez_dx[:n, :]
    hy = hy.at[:n, :].add(coeff_h * psi_hy_xlo)
    hy = hy.at[:n, :].add(coeff_h * (1.0 / k_lo - 1.0) * dez_dx[:n, :])

    b_hi = jnp.flip(cfg.magnetic_hi.b)[:, None]
    c_hi = jnp.flip(cfg.magnetic_hi.c)[:, None]
    k_hi = jnp.flip(cfg.magnetic_hi.kappa)[:, None]

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
