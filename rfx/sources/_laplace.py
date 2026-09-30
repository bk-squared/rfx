"""Shared electrostatic cross-section solver for transmission-line sources."""

from __future__ import annotations

import numpy as np


def _solve_laplace_2d(
    eps_yz: np.ndarray,
    trace_mask: np.ndarray,
    ground_mask: np.ndarray,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Solve ∇·(ε ∇φ) = 0 with Dirichlet trace/ground via 5-point FV.

    Parameters
    ----------
    eps_yz : (n_y, n_z) array of cell-centred relative permittivity.
    trace_mask : (n_y, n_z) bool, True where φ = 1.
    ground_mask : (n_y, n_z) bool, True where φ = 0.
    dy, dz : float, cell sizes.

    Returns
    -------
    phi : (n_y, n_z) electrostatic potential.

    Notes
    -----
    Boundary at far y / top z is implicit Neumann (no flux) by truncating
    coefficients at the array edge. Caller must extend the box well past
    fringing fields (≥ 5W lateral, ≥ 4H above) for that to be accurate.
    """
    try:
        from scipy.sparse import lil_matrix, csr_matrix
        from scipy.sparse.linalg import spsolve
        _have_sparse = True
    except Exception:
        _have_sparse = False

    n_y, n_z = eps_yz.shape
    fixed_mask = trace_mask | ground_mask
    fixed_val = np.where(trace_mask, 1.0, 0.0)

    def _idx(j, k):
        return j * n_z + k

    n_unk = n_y * n_z

    if _have_sparse:
        A = lil_matrix((n_unk, n_unk), dtype=np.float64)
        b = np.zeros(n_unk, dtype=np.float64)
        for j in range(n_y):
            for k in range(n_z):
                p = _idx(j, k)
                if fixed_mask[j, k]:
                    A[p, p] = 1.0
                    b[p] = fixed_val[j, k]
                    continue
                # 5-point ε-weighted Laplacian. Off-diagonal coeffs use
                # harmonic-mean ε at the face (continuous flux).
                diag = 0.0
                # +y neighbour
                if j + 1 < n_y:
                    eps_face = 2.0 * eps_yz[j, k] * eps_yz[j + 1, k] / (
                        eps_yz[j, k] + eps_yz[j + 1, k] + 1e-30
                    )
                    coef = eps_face / (dy * dy)
                    A[p, _idx(j + 1, k)] = coef
                    diag -= coef
                # -y neighbour
                if j - 1 >= 0:
                    eps_face = 2.0 * eps_yz[j, k] * eps_yz[j - 1, k] / (
                        eps_yz[j, k] + eps_yz[j - 1, k] + 1e-30
                    )
                    coef = eps_face / (dy * dy)
                    A[p, _idx(j - 1, k)] = coef
                    diag -= coef
                # +z neighbour
                if k + 1 < n_z:
                    eps_face = 2.0 * eps_yz[j, k] * eps_yz[j, k + 1] / (
                        eps_yz[j, k] + eps_yz[j, k + 1] + 1e-30
                    )
                    coef = eps_face / (dz * dz)
                    A[p, _idx(j, k + 1)] = coef
                    diag -= coef
                # -z neighbour
                if k - 1 >= 0:
                    eps_face = 2.0 * eps_yz[j, k] * eps_yz[j, k - 1] / (
                        eps_yz[j, k] + eps_yz[j, k - 1] + 1e-30
                    )
                    coef = eps_face / (dz * dz)
                    A[p, _idx(j, k - 1)] = coef
                    diag -= coef
                A[p, p] = diag
                b[p] = 0.0
        phi = spsolve(csr_matrix(A), b)
        return phi.reshape(n_y, n_z)

    # Fallback: Jacobi/Gauss-Seidel iteration.
    phi = np.zeros((n_y, n_z), dtype=np.float64)
    phi[trace_mask] = 1.0
    for _ in range(20000):
        phi_new = phi.copy()
        max_d = 0.0
        for j in range(n_y):
            for k in range(n_z):
                if fixed_mask[j, k]:
                    continue
                num = 0.0
                den = 0.0
                if j + 1 < n_y:
                    eps_face = 2.0 * eps_yz[j, k] * eps_yz[j + 1, k] / (
                        eps_yz[j, k] + eps_yz[j + 1, k] + 1e-30
                    )
                    c = eps_face / (dy * dy)
                    num += c * phi_new[j + 1, k]
                    den += c
                if j - 1 >= 0:
                    eps_face = 2.0 * eps_yz[j, k] * eps_yz[j - 1, k] / (
                        eps_yz[j, k] + eps_yz[j - 1, k] + 1e-30
                    )
                    c = eps_face / (dy * dy)
                    num += c * phi_new[j - 1, k]
                    den += c
                if k + 1 < n_z:
                    eps_face = 2.0 * eps_yz[j, k] * eps_yz[j, k + 1] / (
                        eps_yz[j, k] + eps_yz[j, k + 1] + 1e-30
                    )
                    c = eps_face / (dz * dz)
                    num += c * phi_new[j, k + 1]
                    den += c
                if k - 1 >= 0:
                    eps_face = 2.0 * eps_yz[j, k] * eps_yz[j, k - 1] / (
                        eps_yz[j, k] + eps_yz[j, k - 1] + 1e-30
                    )
                    c = eps_face / (dz * dz)
                    num += c * phi_new[j, k - 1]
                    den += c
                new_val = num / (den + 1e-30)
                max_d = max(max_d, abs(new_val - phi_new[j, k]))
                phi_new[j, k] = new_val
        phi = phi_new
        if max_d < 1e-9:
            break
    return phi

