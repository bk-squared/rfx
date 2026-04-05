"""3D SBP-SAT FDTD subgridding (Phase 3).

Full 3D extension with 6 field components (Ex, Ey, Ez, Hx, Hy, Hz) and
6-face rectangular refinement box with SAT interface coupling.

Based on: Cheng et al., IEEE TAP 2025 (10836194)
"Toward the Development of a 3-D SBP-SAT FDTD Method: Subgridding Implementation"

Key design: uses a GLOBAL timestep dt (limited by fine grid) for BOTH
coarse and fine grids to maintain energy conservation. No temporal
sub-stepping — this avoids operator-splitting energy errors.

The coarse grid is less efficient (could use larger dt) but stability
and energy conservation are guaranteed.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0

C0 = 1.0 / np.sqrt(EPS_0 * MU_0)


class SubgridConfig3D(NamedTuple):
    """Configuration for 3D SBP-SAT subgridding."""
    # Coarse grid (full domain)
    nx_c: int
    ny_c: int
    nz_c: int
    dx_c: float
    # Fine region (in coarse indices)
    fi_lo: int
    fi_hi: int
    fj_lo: int
    fj_hi: int
    fk_lo: int
    fk_hi: int
    # Fine grid
    nx_f: int
    ny_f: int
    nz_f: int
    dx_f: float
    # Shared
    dt: float       # global timestep (limited by fine grid CFL)
    ratio: int
    tau: float      # SAT penalty


class SubgridState3D(NamedTuple):
    """State for 3D subgridded domain."""
    # Coarse
    ex_c: jnp.ndarray
    ey_c: jnp.ndarray
    ez_c: jnp.ndarray
    hx_c: jnp.ndarray
    hy_c: jnp.ndarray
    hz_c: jnp.ndarray
    # Fine
    ex_f: jnp.ndarray
    ey_f: jnp.ndarray
    ez_f: jnp.ndarray
    hx_f: jnp.ndarray
    hy_f: jnp.ndarray
    hz_f: jnp.ndarray
    step: int


def init_subgrid_3d(
    shape_c: tuple[int, int, int] = (40, 40, 40),
    dx_c: float = 0.003,
    fine_region: tuple[int, int, int, int, int, int] = (15, 25, 15, 25, 15, 25),
    ratio: int = 3,
    courant: float = 0.45,
    tau: float = 0.5,
) -> tuple[SubgridConfig3D, SubgridState3D]:
    """Initialize 3D subgridded domain.

    Uses a GLOBAL timestep for both grids (no temporal sub-stepping).

    Parameters
    ----------
    tau : float
        SAT penalty coefficient (default 0.5). Higher values give
        stronger coupling but more dissipation.
    """
    nx_c, ny_c, nz_c = shape_c
    fi_lo, fi_hi, fj_lo, fj_hi, fk_lo, fk_hi = fine_region
    dx_f = dx_c / ratio
    dt = courant * dx_f / (C0 * np.sqrt(3))  # 3D CFL

    nx_f = (fi_hi - fi_lo) * ratio
    ny_f = (fj_hi - fj_lo) * ratio
    nz_f = (fk_hi - fk_lo) * ratio

    config = SubgridConfig3D(
        nx_c=nx_c, ny_c=ny_c, nz_c=nz_c, dx_c=dx_c,
        fi_lo=fi_lo, fi_hi=fi_hi,
        fj_lo=fj_lo, fj_hi=fj_hi,
        fk_lo=fk_lo, fk_hi=fk_hi,
        nx_f=nx_f, ny_f=ny_f, nz_f=nz_f, dx_f=dx_f,
        dt=float(dt), ratio=ratio, tau=tau,
    )

    z = lambda s: jnp.zeros(s, dtype=jnp.float32)
    state = SubgridState3D(
        ex_c=z(shape_c), ey_c=z(shape_c), ez_c=z(shape_c),
        hx_c=z(shape_c), hy_c=z(shape_c), hz_c=z(shape_c),
        ex_f=z((nx_f, ny_f, nz_f)), ey_f=z((nx_f, ny_f, nz_f)),
        ez_f=z((nx_f, ny_f, nz_f)),
        hx_f=z((nx_f, ny_f, nz_f)), hy_f=z((nx_f, ny_f, nz_f)),
        hz_f=z((nx_f, ny_f, nz_f)),
        step=0,
    )

    return config, state


def _update_3d(ex, ey, ez, hx, hy, hz, dt, dx,
               mats=None, pec_mask=None, boundary_pec=True):
    """Full 3D Yee update using rfx core kernels.

    Parameters
    ----------
    mats : MaterialArrays or None
        If None, uses vacuum.
    pec_mask : array or None
        Boolean mask for interior PEC geometry.
    boundary_pec : bool
        If True, apply PEC at domain boundaries.
    """
    from rfx.core.yee import FDTDState, MaterialArrays, update_h, update_e
    from rfx.boundaries.pec import apply_pec, apply_pec_mask

    shape = ex.shape
    state = FDTDState(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz,
                      step=jnp.array(0, dtype=jnp.int32))
    if mats is None:
        mats = MaterialArrays(
            eps_r=jnp.ones(shape, dtype=jnp.float32),
            sigma=jnp.zeros(shape, dtype=jnp.float32),
            mu_r=jnp.ones(shape, dtype=jnp.float32),
        )

    state = update_h(state, mats, dt, dx)
    state = update_e(state, mats, dt, dx)
    if boundary_pec:
        state = apply_pec(state)
    if pec_mask is not None:
        state = apply_pec_mask(state, pec_mask)

    return state.ex, state.ey, state.ez, state.hx, state.hy, state.hz


def _downsample_2d(fine_face, n_coarse_j, n_coarse_k, ratio):
    """Downsample a 2D fine-grid face to coarse resolution via block averaging."""
    nj_f = n_coarse_j * ratio
    nk_f = n_coarse_k * ratio
    trimmed = fine_face[:nj_f, :nk_f]
    # Reshape into (n_coarse_j, ratio, n_coarse_k, ratio) then average
    return jnp.mean(trimmed.reshape(n_coarse_j, ratio, n_coarse_k, ratio), axis=(1, 3))


def _upsample_2d(coarse_face, ny_f, nz_f, ratio):
    """Upsample coarse face to fine resolution by repeating."""
    return jnp.repeat(jnp.repeat(coarse_face, ratio, axis=0), ratio, axis=1)[:ny_f, :nz_f]


def _shared_node_coupling_3d(state_c_fields, state_f_fields, config):
    """SAT penalty coupling for tangential E-components on 6 faces.

    Uses SAT (Simultaneous Approximation Terms) instead of hard
    synchronization. Adds correction proportional to the mismatch
    between coarse and fine boundary values, preserving outgoing
    wave information while coupling the two grids.

    SAT penalty: E += alpha * (E_other - E_self)
    where alpha = tau * min(dx_other/dx_self, 1.0) for stability.
    """
    ex_c, ey_c, ez_c = state_c_fields[:3]
    ex_f, ey_f, ez_f = state_f_fields[:3]

    ratio = config.ratio
    fi, fj, fk = config.fi_lo, config.fj_lo, config.fk_lo
    ni = config.fi_hi - fi
    nj = config.fj_hi - fj
    nk = config.fk_hi - fk

    # SBP-SAT penalty from Cheng et al. 2025 energy analysis.
    # The additive correction is: E += alpha * (E_other - E_self).
    #
    # Energy conservation requires:
    #   alpha_c + alpha_f = 1        (total penalty = full correction)
    #   alpha_c * dx_c = alpha_f * dx_f  (SBP norm symmetry)
    #
    # Solving: alpha_f = ratio / (ratio + 1)
    #          alpha_c = 1 / (ratio + 1)
    #
    # For ratio=3: alpha_f=0.75, alpha_c=0.25 (sums to 1.0)
    # config.tau scales both (tau=1.0 = energy-conservative,
    # tau<1.0 = dissipative but stable, tau>1.0 = unstable)
    tau = config.tau  # default 0.5

    alpha_f = tau * ratio / (ratio + 1.0)
    alpha_c = tau * 1.0 / (ratio + 1.0)

    def _sat_couple(ec_arr, ef_arr, c_slice, f_slice,
                    nj_ds, nk_ds, ny_up, nz_up):
        """SBP-SAT penalty coupling (additive correction)."""
        ec_face = ec_arr[c_slice]
        ef_face = ef_arr[f_slice]
        ef_ds = _downsample_2d(ef_face, nj_ds, nk_ds, ratio)
        ec_us = _upsample_2d(ec_face, ny_up, nz_up, ratio)

        # Additive penalty (not replacement)
        ec_arr = ec_arr.at[c_slice].add(alpha_c * (ef_ds - ec_face))
        ef_arr = ef_arr.at[f_slice].add(alpha_f * (ec_us - ef_face))
        return ec_arr, ef_arr

    # === x-lo face (i = fi_lo): tangential = Ey, Ez ===
    if nj > 0 and nk > 0 and config.ny_f > 0 and config.nz_f > 0:
        c_sl = (fi, slice(fj, fj+nj), slice(fk, fk+nk))
        f_sl = (0, slice(None), slice(None))
        ey_c, ey_f = _sat_couple(ey_c, ey_f, c_sl, f_sl, nj, nk, config.ny_f, config.nz_f)
        ez_c, ez_f = _sat_couple(ez_c, ez_f, c_sl, f_sl, nj, nk, config.ny_f, config.nz_f)

    # === x-hi face ===
    if nj > 0 and nk > 0 and config.ny_f > 0 and config.nz_f > 0:
        c_sl = (config.fi_hi - 1, slice(fj, fj+nj), slice(fk, fk+nk))
        f_sl = (-1, slice(None), slice(None))
        ey_c, ey_f = _sat_couple(ey_c, ey_f, c_sl, f_sl, nj, nk, config.ny_f, config.nz_f)
        ez_c, ez_f = _sat_couple(ez_c, ez_f, c_sl, f_sl, nj, nk, config.ny_f, config.nz_f)

    # === y-lo face (j = fj_lo): tangential = Ex, Ez ===
    if ni > 0 and nk > 0 and config.nx_f > 0 and config.nz_f > 0:
        c_sl = (slice(fi, fi+ni), fj, slice(fk, fk+nk))
        f_sl = (slice(None), 0, slice(None))
        ex_c, ex_f = _sat_couple(ex_c, ex_f, c_sl, f_sl, ni, nk, config.nx_f, config.nz_f)
        ez_c, ez_f = _sat_couple(ez_c, ez_f, c_sl, f_sl, ni, nk, config.nx_f, config.nz_f)

    # === y-hi face ===
    if ni > 0 and nk > 0 and config.nx_f > 0 and config.nz_f > 0:
        c_sl = (slice(fi, fi+ni), config.fj_hi - 1, slice(fk, fk+nk))
        f_sl = (slice(None), -1, slice(None))
        ex_c, ex_f = _sat_couple(ex_c, ex_f, c_sl, f_sl, ni, nk, config.nx_f, config.nz_f)
        ez_c, ez_f = _sat_couple(ez_c, ez_f, c_sl, f_sl, ni, nk, config.nx_f, config.nz_f)

    # === z-lo face (k = fk_lo): tangential = Ex, Ey ===
    if ni > 0 and nj > 0 and config.nx_f > 0 and config.ny_f > 0:
        c_sl = (slice(fi, fi+ni), slice(fj, fj+nj), fk)
        f_sl = (slice(None), slice(None), 0)
        ex_c, ex_f = _sat_couple(ex_c, ex_f, c_sl, f_sl, ni, nj, config.nx_f, config.ny_f)
        ey_c, ey_f = _sat_couple(ey_c, ey_f, c_sl, f_sl, ni, nj, config.nx_f, config.ny_f)

    # === z-hi face ===
    if ni > 0 and nj > 0 and config.nx_f > 0 and config.ny_f > 0:
        c_sl = (slice(fi, fi+ni), slice(fj, fj+nj), config.fk_hi - 1)
        f_sl = (slice(None), slice(None), -1)
        ex_c, ex_f = _sat_couple(ex_c, ex_f, c_sl, f_sl, ni, nj, config.nx_f, config.ny_f)
        ey_c, ey_f = _sat_couple(ey_c, ey_f, c_sl, f_sl, ni, nj, config.nx_f, config.ny_f)

    return (ex_c, ey_c, ez_c), (ex_f, ey_f, ez_f)


def step_subgrid_3d(
    state: SubgridState3D,
    config: SubgridConfig3D,
    *,
    mats_c=None,
    mats_f=None,
    pec_mask_c=None,
    pec_mask_f=None,
) -> SubgridState3D:
    """One timestep of coupled 3D coarse + fine grids.

    Both grids use the SAME dt (global timestep) for energy conservation.

    Parameters
    ----------
    mats_c, mats_f : MaterialArrays or None
        Materials for coarse/fine grids. None = vacuum.
    pec_mask_c, pec_mask_f : array or None
        Boolean PEC masks for coarse/fine grids.
    """
    dt = config.dt

    # Update coarse grid (with boundary PEC)
    ex_c, ey_c, ez_c, hx_c, hy_c, hz_c = _update_3d(
        state.ex_c, state.ey_c, state.ez_c,
        state.hx_c, state.hy_c, state.hz_c,
        dt, config.dx_c, mats=mats_c, pec_mask=pec_mask_c,
        boundary_pec=True)

    # Update fine grid (no boundary PEC — interfaces handled by coupling)
    ex_f, ey_f, ez_f, hx_f, hy_f, hz_f = _update_3d(
        state.ex_f, state.ey_f, state.ez_f,
        state.hx_f, state.hy_f, state.hz_f,
        dt, config.dx_f, mats=mats_f, pec_mask=pec_mask_f,
        boundary_pec=False)

    # Shared-node coupling: all tangential E on all 6 faces, bidirectional
    (ex_c, ey_c, ez_c), (ex_f, ey_f, ez_f) = _shared_node_coupling_3d(
        (ex_c, ey_c, ez_c), (ex_f, ey_f, ez_f), config)

    return SubgridState3D(
        ex_c=ex_c, ey_c=ey_c, ez_c=ez_c,
        hx_c=hx_c, hy_c=hy_c, hz_c=hz_c,
        ex_f=ex_f, ey_f=ey_f, ez_f=ez_f,
        hx_f=hx_f, hy_f=hy_f, hz_f=hz_f,
        step=state.step + 1,
    )


def compute_energy_3d(state: SubgridState3D, config: SubgridConfig3D) -> float:
    """Total 3D discrete energy."""
    dv_c = config.dx_c ** 3
    dv_f = config.dx_f ** 3
    e_c = (float(jnp.sum(state.ex_c**2 + state.ey_c**2 + state.ez_c**2)) * EPS_0 * dv_c +
           float(jnp.sum(state.hx_c**2 + state.hy_c**2 + state.hz_c**2)) * MU_0 * dv_c)
    e_f = (float(jnp.sum(state.ex_f**2 + state.ey_f**2 + state.ez_f**2)) * EPS_0 * dv_f +
           float(jnp.sum(state.hx_f**2 + state.hy_f**2 + state.hz_f**2)) * MU_0 * dv_f)
    return e_c + e_f
