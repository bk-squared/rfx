"""M0 dissipativity-certificate calculator for a 2-D TEz FDTD region.

Assembles the descriptor-system matrices R, F, B, L of arXiv:1606.08761
eqs. (17)-(20) for a rectangular Nx x Ny region with per-edge permittivity /
conductivity and per-cell permeability, and numerically checks the Theorem 1
dissipativity conditions:

  (29a) R = R^T > 0        (generalized CFL; dt_max = 2 / s_max(S), eq. (40)-(41))
  (29b) F + F^T >= 0       (per-edge sigma >= 0, eqs. (34)-(36))
  (29c) B = L (L^T B)      (input/output collocation, eq. (27))

Field ordering (row-major with i fastest):
  Ex[i, j], i in [0, Nx), j in [0, Ny]   -> index j*Nx + i
  Ey[i, j], i in [0, Nx], j in [0, Ny)   -> index j*(Nx+1) + i
  Hz[i, j], i in [0, Nx), j in [0, Ny)   -> index j*Nx + i

This is the M0 regression gate and the reviewer's re-computation target; it is
deliberately dense NumPy (small regions only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6


def _w(n: int) -> np.ndarray:
    """Paper eq. (7): n x (n+1) forward-difference stencil matrix."""
    w = np.zeros((n, n + 1), dtype=np.float64)
    for k in range(n):
        w[k, k] = -1.0
        w[k, k + 1] = 1.0
    return w


@dataclass
class RegionMatrices:
    nx: int
    ny: int
    dx: float
    dy: float
    dt: float
    R: np.ndarray
    F: np.ndarray
    B: np.ndarray
    L: np.ndarray
    S: np.ndarray  # eq. (40), for the generalized CFL


def build_region_matrices(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    dt: float,
    eps_x: np.ndarray,   # (nx, ny+1) per x-directed primary edge
    eps_y: np.ndarray,   # (nx+1, ny) per y-directed primary edge
    sigma_x: np.ndarray,  # (nx, ny+1)
    sigma_y: np.ndarray,  # (nx+1, ny)
    mu: np.ndarray,       # (nx, ny) per Hz sample (secondary edge)
) -> RegionMatrices:
    n_ex = nx * (ny + 1)
    n_ey = (nx + 1) * ny
    n_hz = nx * ny

    gx = np.kron(np.eye(ny), _w(nx))          # (n_hz, n_ey)
    gy = np.kron(_w(ny), np.eye(nx))          # (n_hz, n_ex)

    d_lx = dx * np.ones(n_ex)
    d_ly = dy * np.ones(n_ey)
    # Secondary-edge lengths: half at the region boundary (paper Sec. II-A).
    d_lpy = np.full((nx, ny + 1), dy)
    d_lpy[:, 0] = dy / 2.0
    d_lpy[:, -1] = dy / 2.0
    d_lpy = d_lpy.flatten(order="F")          # j-major with i fastest == column-major flatten
    d_lpx = np.full((nx + 1, ny), dx)
    d_lpx[0, :] = dx / 2.0
    d_lpx[-1, :] = dx / 2.0
    d_lpx = d_lpx.flatten(order="F")
    d_a = dx * dy * np.ones(n_hz)

    dex = np.asarray(eps_x, dtype=np.float64).flatten(order="F")
    dey = np.asarray(eps_y, dtype=np.float64).flatten(order="F")
    dsx = np.asarray(sigma_x, dtype=np.float64).flatten(order="F")
    dsy = np.asarray(sigma_y, dtype=np.float64).flatten(order="F")
    dmu = np.asarray(mu, dtype=np.float64).flatten(order="F")

    # --- R, eq. (17) ---
    r11 = np.diag(d_lx * d_lpy * dex / dt)
    r22 = np.diag(d_ly * d_lpx * dey / dt)
    r33 = np.diag(d_a * dmu / dt)
    r13 = 0.5 * (d_lx[:, None] * gy.T)
    r23 = -0.5 * (d_ly[:, None] * gx.T)
    R = np.block([
        [r11, np.zeros((n_ex, n_ey)), r13],
        [np.zeros((n_ey, n_ex)), r22, r23],
        [r13.T, r23.T, r33],
    ])

    # --- F, eq. (18) ---
    f11 = np.diag(d_lx * d_lpy * dsx / 2.0)
    f22 = np.diag(d_ly * d_lpx * dsy / 2.0)
    F = np.block([
        [f11, np.zeros((n_ex, n_ey)), r13],
        [np.zeros((n_ey, n_ex)), f22, r23],
        [-r13.T, -r23.T, np.zeros((n_hz, n_hz))],
    ])

    # --- B (19) and L (20) ---
    bs = np.zeros((n_ex, nx))
    bn = np.zeros((n_ex, nx))
    for i in range(nx):
        bs[0 * nx + i, i] = -1.0            # South rows: j = 0
        bn[ny * nx + i, i] = 1.0            # North rows: j = ny
    bw = np.zeros((n_ey, ny))
    be = np.zeros((n_ey, ny))
    for j in range(ny):
        bw[j * (nx + 1) + 0, j] = 1.0       # West rows: i = 0
        be[j * (nx + 1) + nx, j] = -1.0     # East rows: i = nx
    zx = np.zeros((n_ex, ny))
    zy = np.zeros((n_ey, nx))
    B = np.block([
        [dx * bs, dx * bn, zx, zx],
        [zy, zy, dy * bw, dy * be],
        [np.zeros((n_hz, 2 * nx + 2 * ny))],
    ])
    L = np.block([
        [-bs, bn, zx, zx],
        [zy, zy, bw, -be],
        [np.zeros((n_hz, 2 * nx + 2 * ny))],
    ])

    # --- S, eq. (40) ---
    pre = 1.0 / np.sqrt(d_a * dmu)
    s_left = pre[:, None] * (gy * (np.sqrt(d_lx / (d_lpy * dex)))[None, :])
    s_right = -pre[:, None] * (gx * (np.sqrt(d_ly / (d_lpx * dey)))[None, :])
    S = np.hstack([s_left, s_right])

    return RegionMatrices(nx, ny, dx, dy, dt, R, F, B, L, S)


def dt_max_certificate(m: RegionMatrices) -> float:
    """Generalized CFL bound: dt < 2 / s_max(S)  (eqs. (39)-(41))."""
    s_max = np.linalg.svd(m.S, compute_uv=False)[0]
    return 2.0 / s_max


def classical_cfl_dt(dx: float, dy: float, eps_min: float, mu_min: float) -> float:
    """Paper eq. (47) (sufficient condition; smallest material values, cf. 3-D (35))."""
    return np.sqrt(eps_min * mu_min) / np.sqrt(1.0 / dx**2 + 1.0 / dy**2)


def certify_region(m: RegionMatrices) -> dict:
    """Numerically evaluate Theorem 1 conditions (29a)-(29c) for the region."""
    r_sym = float(np.max(np.abs(m.R - m.R.T)) / np.max(np.abs(m.R)))
    eig_r_min = float(np.linalg.eigvalsh(0.5 * (m.R + m.R.T)).min())
    ff = m.F + m.F.T
    eig_ff_min = float(np.linalg.eigvalsh(0.5 * (ff + ff.T)).min())
    ltb = m.L.T @ m.B
    b_rec = m.L @ ltb
    b_res = float(np.max(np.abs(m.B - b_rec)) / max(np.max(np.abs(m.B)), 1e-300))
    # eq. (27): L^T B = diag(-dx I_nx, +dx I_nx, +dy I_ny, -dy I_ny)
    expected = np.diag(
        np.concatenate([
            -m.dx * np.ones(m.nx), m.dx * np.ones(m.nx),
            m.dy * np.ones(m.ny), -m.dy * np.ones(m.ny),
        ])
    )
    ltb_res = float(np.max(np.abs(ltb - expected)) / m.dx)
    dt_max = dt_max_certificate(m)
    scale_r = float(np.max(np.abs(m.R)))
    return {
        "R_symmetry_residual": r_sym,
        "R_eigmin": eig_r_min,
        "R_eigmin_rel": eig_r_min / scale_r,
        "R_positive_definite": eig_r_min > 0.0,
        "FFt_eigmin": eig_ff_min,
        "FFt_psd": eig_ff_min >= -1e-13 * max(np.max(np.abs(ff)), 1e-300),
        "B_LLTB_residual": b_res,
        "LTB_structure_residual": ltb_res,
        "dt_max_certificate": dt_max,
        "dissipative": (eig_r_min > 0.0)
        and (eig_ff_min >= -1e-13 * max(np.max(np.abs(ff)), 1e-300))
        and (b_res <= 1e-13),
    }
