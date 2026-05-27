"""2D TM SBP-SAT Non-Split FDTD Subgridding.

Strictly reproduces the mathematically energy-stable, non-split subgridding
formulation of Wang et al., "A Stable SBP-SAT FDTD Subgridding Method Without
Region Split", arXiv:2604.14618.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0

C0 = 1.0 / np.sqrt(EPS_0 * MU_0)


class NonSplitSubgridConfig(NamedTuple):
    """Configuration for 2D TM non-split SBP-SAT subgridding."""
    nx_c: int
    ny_c: int
    dx_c: float
    dt: float
    i1: int
    i2: int
    j1: int
    j2: int
    nx_f: int
    ny_f: int
    dx_f: float
    ratio: int
    T_W_x: jnp.ndarray
    T_W_hat_x: jnp.ndarray
    T_W_y: jnp.ndarray
    T_W_hat_y: jnp.ndarray


class NonSplitSubgridState(NamedTuple):
    """State for 2D TM non-split subgridded domain."""
    ez_c: jnp.ndarray   # (nx_c + 1, ny_c + 1)
    hx_c: jnp.ndarray   # (nx_c + 1, ny_c)
    hy_c: jnp.ndarray   # (nx_c, ny_c + 1)
    ez_f: jnp.ndarray   # (nx_f + 1, ny_f + 1)
    hx_f: jnp.ndarray   # (nx_f + 1, ny_f)
    hy_f: jnp.ndarray   # (nx_f, ny_f + 1)
    step: int


def build_interpolation_matrices(n_coarse: int, ratio: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the norm-compatible base interpolation matrices T_c2f and T_f2c."""
    n_fine = (n_coarse - 1) * ratio + 1
    dx_c = 1.0
    dx_f = dx_c / ratio

    T_c2f = np.zeros((n_fine, n_coarse), dtype=np.float32)
    for k in range(n_fine):
        c_cell = k // ratio
        rem = k % ratio
        alpha = rem / ratio
        if c_cell < n_coarse - 1:
            T_c2f[k, c_cell] = 1.0 - alpha
            T_c2f[k, c_cell + 1] = alpha
        else:
            T_c2f[k, c_cell] = 1.0

    P_coarse = np.full(n_coarse, dx_c, dtype=np.float32)
    P_coarse[0] = dx_c / 2.0
    P_coarse[-1] = dx_c / 2.0

    P_fine = np.full(n_fine, dx_f, dtype=np.float32)
    P_fine[0] = dx_f / 2.0
    P_fine[-1] = dx_f / 2.0

    T_f2c = np.zeros((n_coarse, n_fine), dtype=np.float32)
    for i in range(n_coarse):
        for k in range(n_fine):
            T_f2c[i, k] = (P_fine[k] / P_coarse[i]) * T_c2f[k, i]

    return T_c2f, T_f2c


def init_nonsplit_subgrid_2d(
    nx_c: int = 40,
    ny_c: int = 40,
    dx_c: float = 0.05,
    fine_region: tuple[int, int, int, int] = (15, 25, 15, 25),
    ratio: int = 3,
    courant: float = 0.05,
) -> tuple[NonSplitSubgridConfig, NonSplitSubgridState]:
    """Initialize non-split subgridding configuration and state."""
    i1, i2, j1, j2 = fine_region
    dx_f = dx_c / ratio
    dt = courant * dx_f / (C0 * np.sqrt(2.0))

    nx_f = (i2 - i1) * ratio
    ny_f = (j2 - j1) * ratio

    ny_c_sub = j2 - j1
    T_c2f_x, T_f2c_x = build_interpolation_matrices(ny_c_sub + 1, ratio)

    nx_c_sub = i2 - i1
    T_c2f_y, T_f2c_y = build_interpolation_matrices(nx_c_sub + 1, ratio)

    Bc_x = np.ones(ny_c_sub + 1, dtype=np.float32)
    Bc_x[0] = 0.5
    Bc_x[-1] = 0.5

    Bc_y = np.ones(nx_c_sub + 1, dtype=np.float32)
    Bc_y[0] = 0.5
    Bc_y[-1] = 0.5

    T_W_x = Bc_x[:, None] * T_f2c_x
    T_W_hat_x = T_c2f_x

    T_W_y = Bc_y[:, None] * T_f2c_y
    T_W_hat_y = T_c2f_y

    config = NonSplitSubgridConfig(
        nx_c=nx_c, ny_c=ny_c, dx_c=dx_c, dt=dt,
        i1=i1, i2=i2, j1=j1, j2=j2,
        nx_f=nx_f, ny_f=ny_f, dx_f=dx_f,
        ratio=ratio,
        T_W_x=jnp.array(T_W_x),
        T_W_hat_x=jnp.array(T_W_hat_x),
        T_W_y=jnp.array(T_W_y),
        T_W_hat_y=jnp.array(T_W_hat_y),
    )

    state = NonSplitSubgridState(
        ez_c=jnp.zeros((nx_c + 1, ny_c + 1), dtype=jnp.float32),
        hx_c=jnp.zeros((nx_c + 1, ny_c), dtype=jnp.float32),
        hy_c=jnp.zeros((nx_c, ny_c + 1), dtype=jnp.float32),
        ez_f=jnp.zeros((nx_f + 1, ny_f + 1), dtype=jnp.float32),
        hx_f=jnp.zeros((nx_f + 1, ny_f), dtype=jnp.float32),
        hy_f=jnp.zeros((nx_f, ny_f + 1), dtype=jnp.float32),
        step=0,
    )

    return config, state


def step_sbp_sat_nonsplit_2d(state: NonSplitSubgridState, config: NonSplitSubgridConfig) -> NonSplitSubgridState:
    """Take one time step of the stable SBP-SAT non-split 2D subgridding scheme."""
    dt = config.dt
    dx_c = config.dx_c
    dx_f = config.dx_f
    i1, i2 = config.i1, config.i2
    j1, j2 = config.j1, config.j2

    # 1. Update H fields (standard Yee updates)
    hx_c = state.hx_c - (dt / MU_0) * (state.ez_c[:, 1:] - state.ez_c[:, :-1]) / dx_c
    hy_c = state.hy_c + (dt / MU_0) * (state.ez_c[1:, :] - state.ez_c[:-1, :]) / dx_c

    hx_f = state.hx_f - (dt / MU_0) * (state.ez_f[:, 1:] - state.ez_f[:, :-1]) / dx_f
    hy_f = state.hy_f + (dt / MU_0) * (state.ez_f[1:, :] - state.ez_f[:-1, :]) / dx_f

    # 2. Add H-field SAT boundary penalties
    # West interface (i = i1, multiplier W: +1)
    ez_c_W = state.ez_c[i1, j1:j2+1]
    ez_f_W = state.ez_f[0, :]
    J_W_c = ez_c_W - config.T_W_x @ ez_f_W
    J_W_f = ez_f_W - config.T_W_hat_x @ ez_c_W

    hy_c = hy_c.at[i1 - 1, j1:j2+1].add(-0.25 * (dt / MU_0) * (1.0 / dx_c) * J_W_c)
    hy_f = hy_f.at[0, :].add(0.5 * (dt / MU_0) * (1.0 / dx_f) * J_W_f)

    # East interface (i = i2, multiplier E: -1 due to outward normal)
    ez_c_E = state.ez_c[i2, j1:j2+1]
    ez_f_E = state.ez_f[-1, :]
    J_E_c = ez_c_E - config.T_W_x @ ez_f_E
    J_E_f = ez_f_E - config.T_W_hat_x @ ez_c_E

    hy_c = hy_c.at[i2, j1:j2+1].add(0.25 * (dt / MU_0) * (1.0 / dx_c) * J_E_c)
    hy_f = hy_f.at[-1, :].add(-0.5 * (dt / MU_0) * (1.0 / dx_f) * J_E_f)

    # South interface (j = j1, multiplier S: +1)
    ez_c_S = state.ez_c[i1:i2+1, j1]
    ez_f_S = state.ez_f[:, 0]
    J_S_c = ez_c_S - config.T_W_y @ ez_f_S
    J_S_f = ez_f_S - config.T_W_hat_y @ ez_c_S

    hx_c = hx_c.at[i1:i2+1, j1 - 1].add(0.25 * (dt / MU_0) * (1.0 / dx_c) * J_S_c)
    hx_f = hx_f.at[:, 0].add(-0.5 * (dt / MU_0) * (1.0 / dx_f) * J_S_f)

    # North interface (j = j2, multiplier N: -1 due to outward normal)
    ez_c_N = state.ez_c[i1:i2+1, j2]
    ez_f_N = state.ez_f[:, -1]
    J_N_c = ez_c_N - config.T_W_y @ ez_f_N
    J_N_f = ez_f_N - config.T_W_hat_y @ ez_c_N

    hx_c = hx_c.at[i1:i2+1, j2].add(-0.25 * (dt / MU_0) * (1.0 / dx_c) * J_N_c)
    hx_f = hx_f.at[:, -1].add(0.5 * (dt / MU_0) * (1.0 / dx_f) * J_N_f)

    # 3. Update E fields (Yee stencils + SBP boundary derivatives)
    dhy_dx_c = (hy_c[1:, :] - hy_c[:-1, :]) / dx_c
    dhx_dy_c = (hx_c[:, 1:] - hx_c[:, :-1]) / dx_c

    ez_c_new = state.ez_c.at[1:-1, 1:-1].add((dt / EPS_0) * (dhy_dx_c[:, 1:-1] - dhx_dy_c[1:-1, :]))

    # West boundary (i = i1, j1 < j < j2)
    val_W = -2.0 / dx_c * hy_c[i1 - 1, j1+1:j2] - dhx_dy_c[i1, j1:j2-1]
    ez_c_new = ez_c_new.at[i1, j1+1:j2].set(state.ez_c[i1, j1+1:j2] + (dt / EPS_0) * val_W)

    # East boundary (i = i2, j1 < j < j2)
    val_E = 2.0 / dx_c * hy_c[i2, j1+1:j2] - dhx_dy_c[i2, j1:j2-1]
    ez_c_new = ez_c_new.at[i2, j1+1:j2].set(state.ez_c[i2, j1+1:j2] + (dt / EPS_0) * val_E)

    # South boundary (j = j1, i1 < i < i2)
    val_S = dhy_dx_c[i1:i2-1, j1] + 2.0 / dx_c * hx_c[i1+1:i2, j1 - 1]
    ez_c_new = ez_c_new.at[i1+1:i2, j1].set(state.ez_c[i1+1:i2, j1] + (dt / EPS_0) * val_S)

    # North boundary (j = j2, i1 < i < i2)
    val_N = dhy_dx_c[i1:i2-1, j2] - 2.0 / dx_c * hx_c[i1+1:i2, j2]
    ez_c_new = ez_c_new.at[i1+1:i2, j2].set(state.ez_c[i1+1:i2, j2] + (dt / EPS_0) * val_N)

    # Corners of the void:
    # SW: (i1, j1)
    val_SW = -2.0 / dx_c * hy_c[i1 - 1, j1] + 2.0 / dx_c * hx_c[i1, j1 - 1]
    ez_c_new = ez_c_new.at[i1, j1].set(state.ez_c[i1, j1] + (dt / EPS_0) * val_SW)
    # NW: (i1, j2)
    val_NW = -2.0 / dx_c * hy_c[i1 - 1, j2] - 2.0 / dx_c * hx_c[i1, j2]
    ez_c_new = ez_c_new.at[i1, j2].set(state.ez_c[i1, j2] + (dt / EPS_0) * val_NW)
    # SE: (i2, j1)
    val_SE = 2.0 / dx_c * hy_c[i2, j1] + 2.0 / dx_c * hx_c[i2, j1 - 1]
    ez_c_new = ez_c_new.at[i2, j1].set(state.ez_c[i2, j1] + (dt / EPS_0) * val_SE)
    # NE: (i2, j2)
    val_NE = 2.0 / dx_c * hy_c[i2, j2] - 2.0 / dx_c * hx_c[i2, j2]
    ez_c_new = ez_c_new.at[i2, j2].set(state.ez_c[i2, j2] + (dt / EPS_0) * val_NE)

    # -- Fine Grid updates --
    dhy_dx_f = (hy_f[1:, :] - hy_f[:-1, :]) / dx_f
    dhx_dy_f = (hx_f[:, 1:] - hx_f[:, :-1]) / dx_f

    ez_f_new = state.ez_f.at[1:-1, 1:-1].add((dt / EPS_0) * (dhy_dx_f[:, 1:-1] - dhx_dy_f[1:-1, :]))

    # West boundary (i = 0)
    val_W_f = 2.0 / dx_f * hy_f[0, 1:-1] - dhx_dy_f[0, :]
    ez_f_new = ez_f_new.at[0, 1:-1].set(state.ez_f[0, 1:-1] + (dt / EPS_0) * val_W_f)
    # East boundary (i = nx_f)
    val_E_f = -2.0 / dx_f * hy_f[-1, 1:-1] - dhx_dy_f[-1, :]
    ez_f_new = ez_f_new.at[-1, 1:-1].set(state.ez_f[-1, 1:-1] + (dt / EPS_0) * val_E_f)
    # South boundary (j = 0)
    val_S_f = dhy_dx_f[:, 0] - 2.0 / dx_f * hx_f[1:-1, 0]
    ez_f_new = ez_f_new.at[1:-1, 0].set(state.ez_f[1:-1, 0] + (dt / EPS_0) * val_S_f)
    # North boundary (j = ny_f)
    val_N_f = dhy_dx_f[:, -1] + 2.0 / dx_f * hx_f[1:-1, -1]
    ez_f_new = ez_f_new.at[1:-1, -1].set(state.ez_f[1:-1, -1] + (dt / EPS_0) * val_N_f)

    # Corners fine SW, NW, SE, NE:
    ez_f_new = ez_f_new.at[0, 0].set(state.ez_f[0, 0] + (dt / EPS_0) * (2.0 / dx_f * hy_f[0, 0] - 2.0 / dx_f * hx_f[0, 0]))
    ez_f_new = ez_f_new.at[0, -1].set(state.ez_f[0, -1] + (dt / EPS_0) * (2.0 / dx_f * hy_f[0, -1] + 2.0 / dx_f * hx_f[0, -1]))
    ez_f_new = ez_f_new.at[-1, 0].set(state.ez_f[-1, 0] + (dt / EPS_0) * (-2.0 / dx_f * hy_f[-1, 0] - 2.0 / dx_f * hx_f[-1, 0]))
    ez_f_new = ez_f_new.at[-1, -1].set(state.ez_f[-1, -1] + (dt / EPS_0) * (-2.0 / dx_f * hy_f[-1, -1] + 2.0 / dx_f * hx_f[-1, -1]))

    # PEC boundaries on coarse Ez outer bounding box
    ez_c_new = ez_c_new.at[0, :].set(0.0).at[-1, :].set(0.0)
    ez_c_new = ez_c_new.at[:, 0].set(0.0).at[:, -1].set(0.0)

    # 4. Add E-field SAT boundary penalties
    # West interface (i = i1, multiplier W: +1)
    hy_c_W = hy_c[i1 - 1, j1:j2+1]
    hy_f_W = hy_f[0, :]
    J_W_c_H = hy_c_W - config.T_W_x @ hy_f_W
    J_W_f_H = hy_f_W - config.T_W_hat_x @ hy_c_W

    ez_c_new = ez_c_new.at[i1, j1+1:j2].add(-0.5 * (dt / EPS_0) * (2.0 / dx_c) * J_W_c_H[1:-1])
    ez_c_new = ez_c_new.at[i1, j1].add(-0.5 * (dt / EPS_0) * (2.0 / dx_c) * J_W_c_H[0])
    ez_c_new = ez_c_new.at[i1, j2].add(-0.5 * (dt / EPS_0) * (2.0 / dx_c) * J_W_c_H[-1])
    ez_f_new = ez_f_new.at[0, :].add(0.5 * (dt / EPS_0) * (2.0 / dx_f) * J_W_f_H)

    # East interface (i = i2, multiplier E: -1 due to outward normal)
    hy_c_E = hy_c[i2, j1:j2+1]
    hy_f_E = hy_f[-1, :]
    J_E_c_H = hy_c_E - config.T_W_x @ hy_f_E
    J_E_f_H = hy_f_E - config.T_W_hat_x @ hy_c_E

    ez_c_new = ez_c_new.at[i2, j1+1:j2].add(0.5 * (dt / EPS_0) * (2.0 / dx_c) * J_E_c_H[1:-1])
    ez_c_new = ez_c_new.at[i2, j1].add(0.5 * (dt / EPS_0) * (2.0 / dx_c) * J_E_c_H[0])
    ez_c_new = ez_c_new.at[i2, j2].add(0.5 * (dt / EPS_0) * (2.0 / dx_c) * J_E_c_H[-1])
    ez_f_new = ez_f_new.at[-1, :].add(-0.5 * (dt / EPS_0) * (2.0 / dx_f) * J_E_f_H)

    # South interface (j = j1, multiplier S: +1)
    hx_c_S = hx_c[i1:i2+1, j1 - 1]
    hx_f_S = hx_f[:, 0]
    J_S_c_H = hx_c_S - config.T_W_y @ hx_f_S
    J_S_f_H = hx_f_S - config.T_W_hat_y @ hx_c_S

    ez_c_new = ez_c_new.at[i1+1:i2, j1].add(0.5 * (dt / EPS_0) * (2.0 / dx_c) * J_S_c_H[1:-1])
    ez_c_new = ez_c_new.at[i1, j1].add(0.5 * (dt / EPS_0) * (2.0 / dx_c) * J_S_c_H[0])
    ez_c_new = ez_c_new.at[i2, j1].add(0.5 * (dt / EPS_0) * (2.0 / dx_c) * J_S_c_H[-1])
    ez_f_new = ez_f_new.at[:, 0].add(-0.5 * (dt / EPS_0) * (2.0 / dx_f) * J_S_f_H)

    # North interface (j = j2, multiplier N: -1 due to outward normal)
    hx_c_N = hx_c[i1:i2+1, j2]
    hx_f_N = hx_f[:, -1]
    J_N_c_H = hx_c_N - config.T_W_y @ hx_f_N
    J_N_f_H = hx_f_N - config.T_W_hat_y @ hx_c_N

    ez_c_new = ez_c_new.at[i1+1:i2, j2].add(-0.5 * (dt / EPS_0) * (2.0 / dx_c) * J_N_c_H[1:-1])
    ez_c_new = ez_c_new.at[i1, j2].add(-0.5 * (dt / EPS_0) * (2.0 / dx_c) * J_N_c_H[0])
    ez_c_new = ez_c_new.at[i2, j2].add(-0.5 * (dt / EPS_0) * (2.0 / dx_c) * J_N_c_H[-1])
    ez_f_new = ez_f_new.at[:, -1].add(0.5 * (dt / EPS_0) * (2.0 / dx_f) * J_N_f_H)

    # Zero out all fields inside the topological void
    ez_c_new = ez_c_new.at[i1+1:i2, j1+1:j2].set(0.0)
    hx_c = hx_c.at[i1+1:i2, j1:j2].set(0.0)
    hy_c = hy_c.at[i1:i2, j1+1:j2].set(0.0)

    return NonSplitSubgridState(
        ez_c=ez_c_new,
        hx_c=hx_c,
        hy_c=hy_c,
        ez_f=ez_f_new,
        hx_f=hx_f,
        hy_f=hy_f,
        step=state.step + 1,
    )


def compute_nonsplit_energy_2d(state: NonSplitSubgridState, config: NonSplitSubgridConfig) -> float:
    """Compute the total physical discrete SBP energy of the non-split system."""
    i1, i2, j1, j2 = config.i1, config.i2, config.j1, config.j2
    dx_c = config.dx_c
    dx_f = config.dx_f

    # Create coarse indicator masks
    mask_e = np.ones_like(state.ez_c, dtype=bool)
    mask_e[i1+1:i2, j1+1:j2] = False

    mask_hx = np.ones_like(state.hx_c, dtype=bool)
    mask_hx[i1+1:i2, j1:j2] = False

    mask_hy = np.ones_like(state.hy_c, dtype=bool)
    mask_hy[i1:i2, j1+1:j2] = False

    da_c = dx_c ** 2
    P_ez_c = np.full_like(state.ez_c, da_c)
    P_ez_c[0, :] /= 2.0
    P_ez_c[-1, :] /= 2.0
    P_ez_c[:, 0] /= 2.0
    P_ez_c[:, -1] /= 2.0

    # Hole boundaries division
    P_ez_c[i1, j1:j2+1] /= 2.0
    P_ez_c[i2, j1:j2+1] /= 2.0
    P_ez_c[i1:i2+1, j1] /= 2.0
    P_ez_c[i1:i2+1, j2] /= 2.0

    P_hx_c = np.full_like(state.hx_c, da_c)
    P_hy_c = np.full_like(state.hy_c, da_c)

    # Amplify H-field SBP weights adjacent to void boundaries by 2.0
    P_hy_c[i1 - 1, j1:j2+1] *= 2.0
    P_hy_c[i2, j1:j2+1] *= 2.0
    P_hx_c[i1:i2+1, j1 - 1] *= 2.0
    P_hx_c[i1:i2+1, j2] *= 2.0

    e_c_ez = 0.5 * EPS_0 * np.sum(P_ez_c * (state.ez_c ** 2) * mask_e)
    e_c_hx = 0.5 * MU_0 * np.sum(P_hx_c * (state.hx_c ** 2) * mask_hx)
    e_c_hy = 0.5 * MU_0 * np.sum(P_hy_c * (state.hy_c ** 2) * mask_hy)

    da_f = dx_f ** 2
    P_ez_f = np.full_like(state.ez_f, da_f)
    P_ez_f[0, :] /= 2.0
    P_ez_f[-1, :] /= 2.0
    P_ez_f[:, 0] /= 2.0
    P_ez_f[:, -1] /= 2.0

    P_hx_f = np.full_like(state.hx_f, da_f)
    P_hy_f = np.full_like(state.hy_f, da_f)

    e_f_ez = 0.5 * EPS_0 * np.sum(P_ez_f * (state.ez_f ** 2))
    e_f_hx = 0.5 * MU_0 * np.sum(P_hx_f * (state.hx_f ** 2))
    e_f_hy = 0.5 * MU_0 * np.sum(P_hy_f * (state.hy_f ** 2))

    total_coarse = e_c_ez + e_c_hx + e_c_hy
    total_fine = e_f_ez + e_f_hx + e_f_hy
    return float(total_coarse + total_fine)
