"""3D SBP-SAT Overlapping FDTD Subgridding (Production Solver).

High-fidelity JAX-compatible overlapping FDTD subgridding solver incorporating
rigorous Yee-staggered cell-centered 2D projection operators and characteristic-scaled
SAT interface penalty coupling on all 6 faces.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0

C0 = 1.0 / np.sqrt(EPS_0 * MU_0)


class NonSplitSubgridConfig3D(NamedTuple):
    """Configuration for 3D SBP-SAT subgridding."""
    nx_c: int
    ny_c: int
    nz_c: int
    dx_c: float
    dt: float
    fi_lo: int
    fi_hi: int
    fj_lo: int
    fj_hi: int
    fk_lo: int
    fk_hi: int
    nx_f: int
    ny_f: int
    nz_f: int
    dx_f: float
    ratio: int
    tau: float      # SAT penalty coefficient

    # 1D norm-compatible projection operators along each axis
    T_W_x: jnp.ndarray
    T_W_hat_x: jnp.ndarray
    T_W_y: jnp.ndarray
    T_W_hat_y: jnp.ndarray
    T_W_z: jnp.ndarray
    T_W_hat_z: jnp.ndarray


class NonSplitSubgridState3D(NamedTuple):
    """State for 3D subgridded domain."""
    ex_c: jnp.ndarray
    ey_c: jnp.ndarray
    ez_c: jnp.ndarray
    hx_c: jnp.ndarray
    hy_c: jnp.ndarray
    hz_c: jnp.ndarray
    ex_f: jnp.ndarray
    ey_f: jnp.ndarray
    ez_f: jnp.ndarray
    hx_f: jnp.ndarray
    hy_f: jnp.ndarray
    hz_f: jnp.ndarray
    step: int


def build_interpolation_matrices(n_coarse: int, ratio: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the norm-compatible cell-centered 1D interpolation matrices T_c2f and T_f2c."""
    n_fine = n_coarse * ratio
    T_c2f = np.zeros((n_fine, n_coarse), dtype=np.float32)

    for k in range(n_fine):
        pos = (k + 0.5) / ratio
        c_left = int(np.floor(pos - 0.5))
        c_right = c_left + 1

        c_left_clamped = max(0, min(n_coarse - 1, c_left))
        c_right_clamped = max(0, min(n_coarse - 1, c_right))

        if c_left == c_right:
            T_c2f[k, c_left_clamped] = 1.0
        else:
            alpha = pos - (c_left + 0.5)
            T_c2f[k, c_left_clamped] += 1.0 - alpha
            T_c2f[k, c_right_clamped] += alpha

    # Norm compatibility for cell-centered SBP norms: T_f2c = (1 / ratio) * T_c2f^T
    T_f2c = (1.0 / ratio) * T_c2f.T
    return T_c2f, T_f2c


def init_nonsplit_subgrid_3d(
    shape_c: tuple[int, int, int] = (40, 40, 40),
    dx_c: float = 0.05,
    fine_region: tuple[int, int, int, int, int, int] = (15, 25, 15, 25, 15, 25),
    ratio: int = 3,
    courant: float = 0.45,
    tau: float = 0.5,
) -> tuple[NonSplitSubgridConfig3D, NonSplitSubgridState3D]:
    """Initialize 3D subgridding configuration and state."""
    nx_c, ny_c, nz_c = shape_c
    fi_lo, fi_hi = fine_region[0], fine_region[1]
    fj_lo, fj_hi = fine_region[2], fine_region[3]
    fk_lo, fk_hi = fine_region[4], fine_region[5]
    dx_f = dx_c / ratio
    dt = courant * dx_f / (C0 * np.sqrt(3.0))

    nx_f = (fi_hi - fi_lo) * ratio
    ny_f = (fj_hi - fj_lo) * ratio
    nz_f = (fk_hi - fk_lo) * ratio

    # Build 1D operators for each axis
    T_c2f_x, T_f2c_x = build_interpolation_matrices(fi_hi - fi_lo, ratio)
    T_c2f_y, T_f2c_y = build_interpolation_matrices(fj_hi - fj_lo, ratio)
    T_c2f_z, T_f2c_z = build_interpolation_matrices(fk_hi - fk_lo, ratio)

    config = NonSplitSubgridConfig3D(
        nx_c=nx_c, ny_c=ny_c, nz_c=nz_c, dx_c=dx_c, dt=dt,
        fi_lo=fi_lo, fi_hi=fi_hi,
        fj_lo=fj_lo, fj_hi=fj_hi,
        fk_lo=fk_lo, fk_hi=fk_hi,
        nx_f=nx_f, ny_f=ny_f, nz_f=nz_f, dx_f=dx_f,
        ratio=ratio, tau=tau,
        T_W_x=jnp.array(T_f2c_x), T_W_hat_x=jnp.array(T_c2f_x),
        T_W_y=jnp.array(T_f2c_y), T_W_hat_y=jnp.array(T_c2f_y),
        T_W_z=jnp.array(T_f2c_z), T_W_hat_z=jnp.array(T_c2f_z),
    )

    state = NonSplitSubgridState3D(
        ex_c=jnp.zeros(shape_c, dtype=jnp.float32),
        ey_c=jnp.zeros(shape_c, dtype=jnp.float32),
        ez_c=jnp.zeros(shape_c, dtype=jnp.float32),
        hx_c=jnp.zeros(shape_c, dtype=jnp.float32),
        hy_c=jnp.zeros(shape_c, dtype=jnp.float32),
        hz_c=jnp.zeros(shape_c, dtype=jnp.float32),
        ex_f=jnp.zeros((nx_f, ny_f, nz_f), dtype=jnp.float32),
        ey_f=jnp.zeros((nx_f, ny_f, nz_f), dtype=jnp.float32),
        ez_f=jnp.zeros((nx_f, ny_f, nz_f), dtype=jnp.float32),
        hx_f=jnp.zeros((nx_f, ny_f, nz_f), dtype=jnp.float32),
        hy_f=jnp.zeros((nx_f, ny_f, nz_f), dtype=jnp.float32),
        hz_f=jnp.zeros((nx_f, ny_f, nz_f), dtype=jnp.float32),
        step=0,
    )

    return config, state


def _project_c2f_2d(CoarseFace: jnp.ndarray, T_x: jnp.ndarray, T_y: jnp.ndarray) -> jnp.ndarray:
    """Project coarse 2D slice to fine grid boundary using 1D tensor products."""
    return jnp.matmul(T_x, jnp.matmul(CoarseFace, T_y.T))


def _project_f2c_2d(FineFace: jnp.ndarray, T_x: jnp.ndarray, T_y: jnp.ndarray) -> jnp.ndarray:
    """Restrict fine 2D slice to coarse grid boundary using 1D tensor products."""
    return jnp.matmul(T_x, jnp.matmul(FineFace, T_y.T))


def step_sbp_sat_nonsplit_3d(
    state: NonSplitSubgridState3D,
    config: NonSplitSubgridConfig3D,
    *,
    mats_c=None,
    mats_f=None,
    pec_mask_c=None,
    pec_mask_f=None,
) -> NonSplitSubgridState3D:
    """Take one time step of the production 3D overlapping SBP-SAT subgridding solver."""
    from rfx.subgridding.sbp_sat_3d import _update_h_only, _update_e_only

    dt = config.dt
    dx_c = config.dx_c
    dx_f = config.dx_f
    fi_lo, fi_hi = config.fi_lo, config.fi_hi
    fj_lo, fj_hi = config.fj_lo, config.fj_hi
    fk_lo, fk_hi = config.fk_lo, config.fk_hi

    # 1. Update H fields (standard Yee updates)
    hx_c, hy_c, hz_c = _update_h_only(
        state.ex_c, state.ey_c, state.ez_c,
        state.hx_c, state.hy_c, state.hz_c,
        dt, dx_c, mats=mats_c)

    hx_f, hy_f, hz_f = _update_h_only(
        state.ex_f, state.ey_f, state.ez_f,
        state.hx_f, state.hy_f, state.hz_f,
        dt, dx_f, mats=mats_f)

    # 2. Add H-field SAT boundary penalties (characteristic-scaled)
    alpha_f = config.tau * config.ratio / (config.ratio + 1.0)
    alpha_c = config.tau * 1.0 / (config.ratio + 1.0)

    coeff_h_c = jnp.float32(alpha_c * (dt / (MU_0 * dx_c)))
    coeff_h_f = jnp.float32(alpha_f * (dt / (MU_0 * dx_f)))

    # --- West Face (i = fi_lo) ---
    # Outward normal: +x
    # Coupled pairs: (Ey, Hz) and (Ez, Hy)
    ey_c_W = state.ey_c[fi_lo, fj_lo:fj_hi, fk_lo:fk_hi]
    ey_f_W = state.ey_f[0, :, :]
    J_ey_W_c = ey_c_W - _project_f2c_2d(ey_f_W, config.T_W_y, config.T_W_z)
    J_ey_W_f = ey_f_W - _project_c2f_2d(ey_c_W, config.T_W_hat_y, config.T_W_hat_z)

    ez_c_W = state.ez_c[fi_lo, fj_lo:fj_hi, fk_lo:fk_hi]
    ez_f_W = state.ez_f[0, :, :]
    J_ez_W_c = ez_c_W - _project_f2c_2d(ez_f_W, config.T_W_y, config.T_W_z)
    J_ez_W_f = ez_f_W - _project_c2f_2d(ez_c_W, config.T_W_hat_y, config.T_W_hat_z)

    hy_c = hy_c.at[fi_lo - 1, fj_lo:fj_hi, fk_lo:fk_hi].add(coeff_h_c * J_ez_W_c)
    hy_f = hy_f.at[0, :, :].add(-coeff_h_f * J_ez_W_f)
    hz_c = hz_c.at[fi_lo - 1, fj_lo:fj_hi, fk_lo:fk_hi].add(-coeff_h_c * J_ey_W_c)
    hz_f = hz_f.at[0, :, :].add(coeff_h_f * J_ey_W_f)

    # --- East Face (i = fi_hi) ---
    # Outward normal: -x
    # Coupled pairs: (Ey, Hz) and (Ez, Hy)
    ey_c_E = state.ey_c[fi_hi - 1, fj_lo:fj_hi, fk_lo:fk_hi]
    ey_f_E = state.ey_f[-1, :, :]
    J_ey_E_c = ey_c_E - _project_f2c_2d(ey_f_E, config.T_W_y, config.T_W_z)
    J_ey_E_f = ey_f_E - _project_c2f_2d(ey_c_E, config.T_W_hat_y, config.T_W_hat_z)

    ez_c_E = state.ez_c[fi_hi - 1, fj_lo:fj_hi, fk_lo:fk_hi]
    ez_f_E = state.ez_f[-1, :, :]
    J_ez_E_c = ez_c_E - _project_f2c_2d(ez_f_E, config.T_W_y, config.T_W_z)
    J_ez_E_f = ez_f_E - _project_c2f_2d(ez_c_E, config.T_W_hat_y, config.T_W_hat_z)

    hy_c = hy_c.at[fi_hi - 1, fj_lo:fj_hi, fk_lo:fk_hi].add(-coeff_h_c * J_ez_E_c)
    hy_f = hy_f.at[-1, :, :].add(coeff_h_f * J_ez_E_f)
    hz_c = hz_c.at[fi_hi - 1, fj_lo:fj_hi, fk_lo:fk_hi].add(coeff_h_c * J_ey_E_c)
    hz_f = hz_f.at[-1, :, :].add(-coeff_h_f * J_ey_E_f)

    # --- South Face (j = fj_lo) ---
    # Outward normal: +y
    # Coupled pairs: (Ez, Hx) and (Ex, Hz)
    ex_c_S = state.ex_c[fi_lo:fi_hi, fj_lo, fk_lo:fk_hi]
    ex_f_S = state.ex_f[:, 0, :]
    J_ex_S_c = ex_c_S - _project_f2c_2d(ex_f_S, config.T_W_x, config.T_W_z)
    J_ex_S_f = ex_f_S - _project_c2f_2d(ex_c_S, config.T_W_hat_x, config.T_W_hat_z)

    ez_c_S = state.ez_c[fi_lo:fi_hi, fj_lo, fk_lo:fk_hi]
    ez_f_S = state.ez_f[:, 0, :]
    J_ez_S_c = ez_c_S - _project_f2c_2d(ez_f_S, config.T_W_x, config.T_W_z)
    J_ez_S_f = ez_f_S - _project_c2f_2d(ez_c_S, config.T_W_hat_x, config.T_W_hat_z)

    hx_c = hx_c.at[fi_lo:fi_hi, fj_lo - 1, fk_lo:fk_hi].add(-coeff_h_c * J_ez_S_c)
    hx_f = hx_f.at[:, 0, :].add(coeff_h_f * J_ez_S_f)
    hz_c = hz_c.at[fi_lo:fi_hi, fj_lo - 1, fk_lo:fk_hi].add(coeff_h_c * J_ex_S_c)
    hz_f = hz_f.at[:, 0, :].add(-coeff_h_f * J_ex_S_f)

    # --- North Face (j = fj_hi) ---
    # Outward normal: -y
    # Coupled pairs: (Ez, Hx) and (Ex, Hz)
    ex_c_N = state.ex_c[fi_lo:fi_hi, fj_hi - 1, fk_lo:fk_hi]
    ex_f_N = state.ex_f[:, -1, :]
    J_ex_N_c = ex_c_N - _project_f2c_2d(ex_f_N, config.T_W_x, config.T_W_z)
    J_ex_N_f = ex_f_N - _project_c2f_2d(ex_c_N, config.T_W_hat_x, config.T_W_hat_z)

    ez_c_N = state.ez_c[fi_lo:fi_hi, fj_hi - 1, fk_lo:fk_hi]
    ez_f_N = state.ez_f[:, -1, :]
    J_ez_N_c = ez_c_N - _project_f2c_2d(ez_f_N, config.T_W_x, config.T_W_z)
    J_ez_N_f = ez_f_N - _project_c2f_2d(ez_c_N, config.T_W_hat_x, config.T_W_hat_z)

    hx_c = hx_c.at[fi_lo:fi_hi, fj_hi - 1, fk_lo:fk_hi].add(coeff_h_c * J_ez_N_c)
    hx_f = hx_f.at[:, -1, :].add(-coeff_h_f * J_ez_N_f)
    hz_c = hz_c.at[fi_lo:fi_hi, fj_hi - 1, fk_lo:fk_hi].add(-coeff_h_c * J_ex_N_c)
    hz_f = hz_f.at[:, -1, :].add(coeff_h_f * J_ex_N_f)

    # --- Bottom Face (k = fk_lo) ---
    # Outward normal: +z
    # Coupled pairs: (Ex, Hy) and (Ey, Hx)
    ex_c_B = state.ex_c[fi_lo:fi_hi, fj_lo:fj_hi, fk_lo]
    ex_f_B = state.ex_f[:, :, 0]
    J_ex_B_c = ex_c_B - _project_f2c_2d(ex_f_B, config.T_W_x, config.T_W_y)
    J_ex_B_f = ex_f_B - _project_c2f_2d(ex_c_B, config.T_W_hat_x, config.T_W_hat_y)

    ey_c_B = state.ey_c[fi_lo:fi_hi, fj_lo:fj_hi, fk_lo]
    ey_f_B = state.ey_f[:, :, 0]
    J_ey_B_c = ey_c_B - _project_f2c_2d(ey_f_B, config.T_W_x, config.T_W_y)
    J_ey_B_f = ey_f_B - _project_c2f_2d(ey_c_B, config.T_W_hat_x, config.T_W_hat_y)

    hx_c = hx_c.at[fi_lo:fi_hi, fj_lo:fj_hi, fk_lo - 1].add(coeff_h_c * J_ey_B_c)
    hx_f = hx_f.at[:, :, 0].add(-coeff_h_f * J_ey_B_f)
    hy_c = hy_c.at[fi_lo:fi_hi, fj_lo:fj_hi, fk_lo - 1].add(-coeff_h_c * J_ex_B_c)
    hy_f = hy_f.at[:, :, 0].add(coeff_h_f * J_ex_B_f)

    # --- Top Face (k = fk_hi) ---
    # Outward normal: -z
    # Coupled pairs: (Ex, Hy) and (Ey, Hx)
    ex_c_T = state.ex_c[fi_lo:fi_hi, fj_lo:fj_hi, fk_hi - 1]
    ex_f_T = state.ex_f[:, :, -1]
    J_ex_T_c = ex_c_T - _project_f2c_2d(ex_f_T, config.T_W_x, config.T_W_y)
    J_ex_T_f = ex_f_T - _project_c2f_2d(ex_c_T, config.T_W_hat_x, config.T_W_hat_y)

    ey_c_T = state.ey_c[fi_lo:fi_hi, fj_lo:fj_hi, fk_hi - 1]
    ey_f_T = state.ey_f[:, :, -1]
    J_ey_T_c = ey_c_T - _project_f2c_2d(ey_f_T, config.T_W_x, config.T_W_y)
    J_ey_T_f = ey_f_T - _project_c2f_2d(ey_c_T, config.T_W_hat_x, config.T_W_hat_y)

    hx_c = hx_c.at[fi_lo:fi_hi, fj_lo:fj_hi, fk_hi - 1].add(-coeff_h_c * J_ey_T_c)
    hx_f = hx_f.at[:, :, -1].add(coeff_h_f * J_ey_T_f)
    hy_c = hy_c.at[fi_lo:fi_hi, fj_lo:fj_hi, fk_hi - 1].add(coeff_h_c * J_ex_T_c)
    hy_f = hy_f.at[:, :, -1].add(-coeff_h_f * J_ex_T_f)

    # 3. Update E fields (Yee stencils)
    ex_c, ey_c, ez_c = _update_e_only(
        state.ex_c, state.ey_c, state.ez_c,
        hx_c, hy_c, hz_c,
        dt, dx_c, mats=mats_c, pec_mask=pec_mask_c,
        boundary_pec=True)

    ex_f, ey_f, ez_f = _update_e_only(
        state.ex_f, state.ey_f, state.ez_f,
        hx_f, hy_f, hz_f,
        dt, dx_f, mats=mats_f, pec_mask=pec_mask_f,
        boundary_pec=False)

    # 4. Add E-field SAT boundary penalties (Weak enforcement of tangential H mismatch)
    coeff_e_c = jnp.float32(alpha_c * (dt / (EPS_0 * dx_c)))
    coeff_e_f = jnp.float32(alpha_f * (dt / (EPS_0 * dx_f)))

    # --- West Face (i = fi_lo) ---
    # Outward normal: +x
    # Coupled pairs: (Ey, Hz) and (Ez, Hy)
    hy_c_W_e = hy_c[fi_lo - 1, fj_lo:fj_hi, fk_lo:fk_hi]
    hy_f_W_e = hy_f[0, :, :]
    J_hy_W_c = hy_c_W_e - _project_f2c_2d(hy_f_W_e, config.T_W_y, config.T_W_z)
    J_hy_W_f = hy_f_W_e - _project_c2f_2d(hy_c_W_e, config.T_W_hat_y, config.T_W_hat_z)

    hz_c_W_e = hz_c[fi_lo - 1, fj_lo:fj_hi, fk_lo:fk_hi]
    hz_f_W_e = hz_f[0, :, :]
    J_hz_W_c = hz_c_W_e - _project_f2c_2d(hz_f_W_e, config.T_W_y, config.T_W_z)
    J_hz_W_f = hz_f_W_e - _project_c2f_2d(hz_c_W_e, config.T_W_hat_y, config.T_W_hat_z)

    ey_c = ey_c.at[fi_lo, fj_lo:fj_hi, fk_lo:fk_hi].add(-coeff_e_c * J_hz_W_c)
    ey_f = ey_f.at[0, :, :].add(coeff_e_f * J_hz_W_f)
    ez_c = ez_c.at[fi_lo, fj_lo:fj_hi, fk_lo:fk_hi].add(coeff_e_c * J_hy_W_c)
    ez_f = ez_f.at[0, :, :].add(-coeff_e_f * J_hy_W_f)

    # --- East Face (i = fi_hi) ---
    # Outward normal: -x
    # Coupled pairs: (Ey, Hz) and (Ez, Hy)
    hy_c_E_e = hy_c[fi_hi - 1, fj_lo:fj_hi, fk_lo:fk_hi]
    hy_f_E_e = hy_f[-1, :, :]
    J_hy_E_c = hy_c_E_e - _project_f2c_2d(hy_f_E_e, config.T_W_y, config.T_W_z)
    J_hy_E_f = hy_f_E_e - _project_c2f_2d(hy_c_E_e, config.T_W_hat_y, config.T_W_hat_z)

    hz_c_E_e = hz_c[fi_hi - 1, fj_lo:fj_hi, fk_lo:fk_hi]
    hz_f_E_e = hz_f[-1, :, :]
    J_hz_E_c = hz_c_E_e - _project_f2c_2d(hz_f_E_e, config.T_W_y, config.T_W_z)
    J_hz_E_f = hz_f_E_e - _project_c2f_2d(hz_c_E_e, config.T_W_hat_y, config.T_W_hat_z)

    ey_c = ey_c.at[fi_hi - 1, fj_lo:fj_hi, fk_lo:fk_hi].add(coeff_e_c * J_hz_E_c)
    ey_f = ey_f.at[-1, :, :].add(-coeff_e_f * J_hz_E_f)
    ez_c = ez_c.at[fi_hi - 1, fj_lo:fj_hi, fk_lo:fk_hi].add(-coeff_e_c * J_hy_E_c)
    ez_f = ez_f.at[-1, :, :].add(coeff_e_f * J_hy_E_f)

    # --- South Face (j = fj_lo) ---
    # Outward normal: +y
    # Coupled pairs: (Ez, Hx) and (Ex, Hz)
    hx_c_S_e = hx_c[fi_lo:fi_hi, fj_lo - 1, fk_lo:fk_hi]
    hx_f_S_e = hx_f[:, 0, :]
    J_hx_S_c = hx_c_S_e - _project_f2c_2d(hx_f_S_e, config.T_W_x, config.T_W_z)
    J_hx_S_f = hx_f_S_e - _project_c2f_2d(hx_c_S_e, config.T_W_hat_x, config.T_W_hat_z)

    hz_c_S_e = hz_c[fi_lo:fi_hi, fj_lo - 1, fk_lo:fk_hi]
    hz_f_S_e = hz_f[:, 0, :]
    J_hz_S_c = hz_c_S_e - _project_f2c_2d(hz_f_S_e, config.T_W_x, config.T_W_z)
    J_hz_S_f = hz_f_S_e - _project_c2f_2d(hz_c_S_e, config.T_W_hat_x, config.T_W_hat_z)

    ex_c = ex_c.at[fi_lo:fi_hi, fj_lo, fk_lo:fk_hi].add(coeff_e_c * J_hz_S_c)
    ex_f = ex_f.at[:, 0, :].add(-coeff_e_f * J_hz_S_f)
    ez_c = ez_c.at[fi_lo:fi_hi, fj_lo, fk_lo:fk_hi].add(-coeff_e_c * J_hx_S_c)
    ez_f = ez_f.at[:, 0, :].add(coeff_e_f * J_hx_S_f)

    # --- North Face (j = fj_hi) ---
    # Outward normal: -y
    # Coupled pairs: (Ez, Hx) and (Ex, Hz)
    hx_c_N_e = hx_c[fi_lo:fi_hi, fj_hi - 1, fk_lo:fk_hi]
    hx_f_N_e = hx_f[:, -1, :]
    J_hx_N_c = hx_c_N_e - _project_f2c_2d(hx_f_N_e, config.T_W_x, config.T_W_z)
    J_hx_N_f = hx_f_N_e - _project_c2f_2d(hx_c_N_e, config.T_W_hat_x, config.T_W_hat_z)

    hz_c_N_e = hz_c[fi_lo:fi_hi, fj_hi - 1, fk_lo:fk_hi]
    hz_f_N_e = hz_f[:, -1, :]
    J_hz_N_c = hz_c_N_e - _project_f2c_2d(hz_f_N_e, config.T_W_x, config.T_W_z)
    J_hz_N_f = hz_f_N_e - _project_c2f_2d(hz_c_N_e, config.T_W_hat_x, config.T_W_hat_z)

    ex_c = ex_c.at[fi_lo:fi_hi, fj_hi - 1, fk_lo:fk_hi].add(-coeff_e_c * J_hz_N_c)
    ex_f = ex_f.at[:, -1, :].add(coeff_e_f * J_hz_N_f)
    ez_c = ez_c.at[fi_lo:fi_hi, fj_hi - 1, fk_lo:fk_hi].add(coeff_e_c * J_hx_N_c)
    ez_f = ez_f.at[:, -1, :].add(-coeff_e_f * J_hx_N_f)

    # --- Bottom Face (k = fk_lo) ---
    # Outward normal: +z
    # Coupled pairs: (Ex, Hy) and (Ey, Hx)
    hx_c_B_e = hx_c[fi_lo:fi_hi, fj_lo:fj_hi, fk_lo - 1]
    hx_f_B_e = hx_f[:, :, 0]
    J_hx_B_c = hx_c_B_e - _project_f2c_2d(hx_f_B_e, config.T_W_x, config.T_W_y)
    J_hx_B_f = hx_f_B_e - _project_c2f_2d(hx_c_B_e, config.T_W_hat_x, config.T_W_hat_y)

    hy_c_B_e = hy_c[fi_lo:fi_hi, fj_lo:fj_hi, fk_lo - 1]
    hy_f_B_e = hy_f[:, :, 0]
    J_hy_B_c = hy_c_B_e - _project_f2c_2d(hy_f_B_e, config.T_W_x, config.T_W_y)
    J_hy_B_f = hy_f_B_e - _project_c2f_2d(hy_c_B_e, config.T_W_hat_x, config.T_W_hat_y)

    ex_c = ex_c.at[fi_lo:fi_hi, fj_lo:fj_hi, fk_lo].add(-coeff_e_c * J_hy_B_c)
    ex_f = ex_f.at[:, :, 0].add(coeff_e_f * J_hy_B_f)
    ey_c = ey_c.at[fi_lo:fi_hi, fj_lo:fj_hi, fk_lo].add(coeff_e_c * J_hx_B_c)
    ey_f = ey_f.at[:, :, 0].add(-coeff_e_f * J_hx_B_f)

    # --- Top Face (k = fk_hi) ---
    # Outward normal: -z
    # Coupled pairs: (Ex, Hy) and (Ey, Hx)
    hx_c_T_e = hx_c[fi_lo:fi_hi, fj_lo:fj_hi, fk_hi - 1]
    hx_f_T_e = hx_f[:, :, -1]
    J_hx_T_c = hx_c_T_e - _project_f2c_2d(hx_f_T_e, config.T_W_x, config.T_W_y)
    J_hx_T_f = hx_f_T_e - _project_c2f_2d(hx_c_T_e, config.T_W_hat_x, config.T_W_hat_y)

    hy_c_T_e = hy_c[fi_lo:fi_hi, fj_lo:fj_hi, fk_hi - 1]
    hy_f_T_e = hy_f[:, :, -1]
    J_hy_T_c = hy_c_T_e - _project_f2c_2d(hy_f_T_e, config.T_W_x, config.T_W_y)
    J_hy_T_f = hy_f_T_e - _project_c2f_2d(hy_c_T_e, config.T_W_hat_x, config.T_W_hat_y)

    ex_c = ex_c.at[fi_lo:fi_hi, fj_lo:fj_hi, fk_hi - 1].add(coeff_e_c * J_hy_T_c)
    ex_f = ex_f.at[:, :, -1].add(-coeff_e_f * J_hy_T_f)
    ey_c = ey_c.at[fi_lo:fi_hi, fj_lo:fj_hi, fk_hi - 1].add(-coeff_e_c * J_hx_T_c)
    ey_f = ey_f.at[:, :, -1].add(coeff_e_f * J_hx_T_f)

    return NonSplitSubgridState3D(
        ex_c=ex_c, ey_c=ey_c, ez_c=ez_c,
        hx_c=hx_c, hy_c=hy_c, hz_c=hz_c,
        ex_f=ex_f, ey_f=ey_f, ez_f=ez_f,
        hx_f=hx_f, hy_f=hy_f, hz_f=hz_f,
        step=state.step + 1,
    )


def compute_nonsplit_energy_3d(state: NonSplitSubgridState3D, config: NonSplitSubgridConfig3D) -> float:
    """Compute the total physical discrete SBP energy of the 3D overlapping subgridding system."""
    fi_lo, fi_hi = config.fi_lo, config.fi_hi
    fj_lo, fj_hi = config.fj_lo, config.fj_hi
    fk_lo, fk_hi = config.fk_lo, config.fk_hi

    dv_c = config.dx_c ** 3
    dv_f = config.dx_f ** 3

    # Define masks for the coarse grid energy computation to exclude the fine region (overlapping energy accounting)
    mask = jnp.ones(state.ex_c.shape, dtype=jnp.bool_)
    mask = mask.at[fi_lo:fi_hi, fj_lo:fj_hi, fk_lo:fk_hi].set(False)

    # 1. Integrate Coarse Energy (excluding the fine region overlap)
    e_c = (
        0.5 * EPS_0 * jnp.sum(jnp.where(mask, state.ex_c ** 2 + state.ey_c ** 2 + state.ez_c ** 2, 0.0)) * dv_c +
        0.5 * MU_0 * jnp.sum(jnp.where(mask, state.hx_c ** 2 + state.hy_c ** 2 + state.hz_c ** 2, 0.0)) * dv_c
    )

    # 2. Integrate Fine Energy
    e_f = (
        0.5 * EPS_0 * jnp.sum(state.ex_f ** 2 + state.ey_f ** 2 + state.ez_f ** 2) * dv_f +
        0.5 * MU_0 * jnp.sum(state.hx_f ** 2 + state.hy_f ** 2 + state.hz_f ** 2) * dv_f
    )

    return float(e_c + e_f)
