"""Numerical eigenmode solver for rectangular waveguide cross-sections.

Provides both analytical (uniform fill) and numerical (inhomogeneous fill)
eigenmode solvers for rectangular waveguides with PEC walls.

For uniform fills, delegates to the existing analytical formulas in
``rfx.sources.waveguide_port``.  For inhomogeneous cross-sections
(non-uniform eps_cross / mu_cross), solves the 2D scalar Helmholtz
eigenvalue problem on a Yee-like grid using ``scipy.sparse.linalg.eigsh``.

This is a one-time precomputation (SciPy/NumPy, not JAX).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from rfx.core.yee import EPS_0, MU_0
from rfx.sources.waveguide_port import (
    _te_mode_profiles,
    _tm_mode_profiles,
    cutoff_frequency,
    C0_LOCAL,
)


class WaveguideMode(NamedTuple):
    """A single waveguide eigenmode on the cross-section grid.

    Attributes
    ----------
    ey_profile : ndarray, shape (nu, nv)
        Transverse E-field, first local axis component.
    ez_profile : ndarray, shape (nu, nv)
        Transverse E-field, second local axis component.
    hy_profile : ndarray, shape (nu, nv)
        Transverse H-field, first local axis component.
    hz_profile : ndarray, shape (nu, nv)
        Transverse H-field, second local axis component.
    beta : ndarray, shape (n_freqs,)
        Propagation constant at each frequency. Real above cutoff,
        imaginary (stored as complex) below cutoff.
    f_cutoff : float
        Cutoff frequency in Hz.
    mode_type : str
        ``"TE"`` or ``"TM"``.
    mode_indices : tuple[int, int]
        ``(m, n)`` mode indices.
    """
    ey_profile: np.ndarray
    ez_profile: np.ndarray
    hy_profile: np.ndarray
    hz_profile: np.ndarray
    beta: np.ndarray
    f_cutoff: float
    mode_type: str
    mode_indices: tuple[int, int]


def _compute_beta_numpy(freqs: np.ndarray, f_cutoff: float,
                        eps_r_eff: float = 1.0, mu_r_eff: float = 1.0,
                        ) -> np.ndarray:
    """Propagation constant beta(f) for each frequency.

    beta = sqrt( (2*pi*f/c)^2 * eps_r * mu_r - kc^2 )
    where kc = 2*pi*f_cutoff / c.

    Returns complex array: real above cutoff, imaginary below.
    """
    omega = 2 * np.pi * freqs
    k = omega / C0_LOCAL
    kc = 2 * np.pi * f_cutoff / C0_LOCAL

    beta_sq = k**2 * eps_r_eff * mu_r_eff - kc**2
    result = np.zeros_like(beta_sq, dtype=complex)
    above = beta_sq >= 0
    result[above] = np.sqrt(beta_sq[above])
    result[~above] = 1j * np.sqrt(-beta_sq[~above])
    return result


def _analytical_modes(a: float, b: float, dx: float,
                      freqs: np.ndarray, n_modes: int,
                      ) -> list[WaveguideMode]:
    """Enumerate the lowest n_modes analytical modes for a uniform fill.

    Sorts all TE and TM modes by cutoff frequency and returns the first
    ``n_modes``.
    """
    ny = max(1, int(round(a / dx)))
    nz = max(1, int(round(b / dx)))
    y_coords = np.linspace(0.5 * dx, a - 0.5 * dx, ny)
    z_coords = np.linspace(0.5 * dx, b - 0.5 * dx, nz)

    # Collect candidate modes with their cutoff frequencies.
    candidates: list[tuple[float, str, int, int]] = []

    # TE modes: (m,n) with m>=0, n>=0, not both zero
    max_m = max(2 * n_modes, 5)
    max_n = max(2 * n_modes, 5)
    for m in range(0, max_m + 1):
        for n in range(0, max_n + 1):
            if m == 0 and n == 0:
                continue
            fc = cutoff_frequency(a, b, m, n)
            candidates.append((fc, "TE", m, n))
    # TM modes: m>=1, n>=1
    for m in range(1, max_m + 1):
        for n in range(1, max_n + 1):
            fc = cutoff_frequency(a, b, m, n)
            candidates.append((fc, "TM", m, n))

    # Sort by cutoff frequency
    candidates.sort(key=lambda x: x[0])

    modes: list[WaveguideMode] = []
    for fc, mode_type, m, n in candidates[:n_modes]:
        if mode_type == "TE":
            ey, ez, hy, hz = _te_mode_profiles(a, b, m, n, y_coords, z_coords)
        else:
            ey, ez, hy, hz = _tm_mode_profiles(a, b, m, n, y_coords, z_coords)

        beta = _compute_beta_numpy(freqs, fc)

        modes.append(WaveguideMode(
            ey_profile=ey,
            ez_profile=ez,
            hy_profile=hy,
            hz_profile=hz,
            beta=beta,
            f_cutoff=fc,
            mode_type=mode_type,
            mode_indices=(m, n),
        ))

    return modes


# ---------------------------------------------------------------------------
# Numerical eigenmode solver for inhomogeneous cross-sections
# ---------------------------------------------------------------------------

def _build_laplacian_2d_neumann(ny: int, nz: int, dy: float, dz: float,
                                 eps: np.ndarray | None = None,
                                 ) -> sparse.csr_matrix:
    """2D negative Laplacian with Neumann BC (for TE / Hz scalar problem).

    Solves: -div(1/eps * grad(Hz)) = kc^2 * Hz  (TE modes)

    When eps is uniform this reduces to the standard 5-point Laplacian.
    For inhomogeneous eps, the finite-difference stencil incorporates
    the spatially varying permittivity.

    Parameters
    ----------
    ny, nz : int
        Grid dimensions.
    dy, dz : float
        Cell sizes.
    eps : ndarray (ny, nz) or None
        Relative permittivity. None means uniform eps_r=1.

    Returns
    -------
    L : sparse matrix (ny*nz, ny*nz)
        The operator matrix.
    """
    N = ny * nz

    def idx(i, j):
        return i * nz + j

    rows, cols, vals = [], [], []

    for i in range(ny):
        for j in range(nz):
            k = idx(i, j)
            diag_val = 0.0

            # y-direction neighbors (Neumann: ghost = interior value)
            if i > 0:
                if eps is not None:
                    eps_face = 2.0 / (eps[i, j] + eps[i - 1, j])
                else:
                    eps_face = 1.0
                w = eps_face / (dy * dy)
                rows.append(k); cols.append(idx(i - 1, j)); vals.append(-w)
                diag_val += w
            # else: Neumann BC — no flux, no off-diag entry, no diag contribution

            if i < ny - 1:
                if eps is not None:
                    eps_face = 2.0 / (eps[i, j] + eps[i + 1, j])
                else:
                    eps_face = 1.0
                w = eps_face / (dy * dy)
                rows.append(k); cols.append(idx(i + 1, j)); vals.append(-w)
                diag_val += w

            # z-direction neighbors
            if j > 0:
                if eps is not None:
                    eps_face = 2.0 / (eps[i, j] + eps[i, j - 1])
                else:
                    eps_face = 1.0
                w = eps_face / (dz * dz)
                rows.append(k); cols.append(idx(i, j - 1)); vals.append(-w)
                diag_val += w

            if j < nz - 1:
                if eps is not None:
                    eps_face = 2.0 / (eps[i, j] + eps[i, j + 1])
                else:
                    eps_face = 1.0
                w = eps_face / (dz * dz)
                rows.append(k); cols.append(idx(i, j + 1)); vals.append(-w)
                diag_val += w

            rows.append(k); cols.append(k); vals.append(diag_val)

    return sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))


def _build_laplacian_2d_dirichlet(ny: int, nz: int, dy: float, dz: float,
                                   eps: np.ndarray | None = None,
                                   ) -> sparse.csr_matrix:
    """2D negative Laplacian with Dirichlet BC (for TM / Ez scalar problem).

    Solves: -div(1/eps * grad(Ez)) = kc^2 * Ez  with Ez=0 on boundary.

    The boundary nodes are interior grid points adjacent to the PEC walls.
    Since the waveguide cross-section grid represents the interior (PEC walls
    are at the edges), Dirichlet means the field is zero outside the grid.
    We implement this by keeping the standard 5-point stencil and treating
    out-of-bound neighbors as zero (which is the default — they simply
    don't appear in the stencil).
    """
    N = ny * nz

    def idx(i, j):
        return i * nz + j

    rows, cols, vals = [], [], []

    for i in range(ny):
        for j in range(nz):
            k = idx(i, j)
            diag_val = 0.0

            # y-direction: Dirichlet means ghost values = 0
            # Left neighbor
            if i > 0:
                if eps is not None:
                    eps_face = 2.0 / (eps[i, j] + eps[i - 1, j])
                else:
                    eps_face = 1.0
                w = eps_face / (dy * dy)
                rows.append(k); cols.append(idx(i - 1, j)); vals.append(-w)
                diag_val += w
            else:
                # Dirichlet: neighbor is zero, contributes to diagonal
                if eps is not None:
                    eps_face = 1.0 / eps[i, j]
                else:
                    eps_face = 1.0
                diag_val += eps_face / (dy * dy)

            # Right neighbor
            if i < ny - 1:
                if eps is not None:
                    eps_face = 2.0 / (eps[i, j] + eps[i + 1, j])
                else:
                    eps_face = 1.0
                w = eps_face / (dy * dy)
                rows.append(k); cols.append(idx(i + 1, j)); vals.append(-w)
                diag_val += w
            else:
                if eps is not None:
                    eps_face = 1.0 / eps[i, j]
                else:
                    eps_face = 1.0
                diag_val += eps_face / (dy * dy)

            # z-direction
            if j > 0:
                if eps is not None:
                    eps_face = 2.0 / (eps[i, j] + eps[i, j - 1])
                else:
                    eps_face = 1.0
                w = eps_face / (dz * dz)
                rows.append(k); cols.append(idx(i, j - 1)); vals.append(-w)
                diag_val += w
            else:
                if eps is not None:
                    eps_face = 1.0 / eps[i, j]
                else:
                    eps_face = 1.0
                diag_val += eps_face / (dz * dz)

            if j < nz - 1:
                if eps is not None:
                    eps_face = 2.0 / (eps[i, j] + eps[i, j + 1])
                else:
                    eps_face = 1.0
                w = eps_face / (dz * dz)
                rows.append(k); cols.append(idx(i, j + 1)); vals.append(-w)
                diag_val += w
            else:
                if eps is not None:
                    eps_face = 1.0 / eps[i, j]
                else:
                    eps_face = 1.0
                diag_val += eps_face / (dz * dz)

            rows.append(k); cols.append(k); vals.append(diag_val)

    return sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))


def _scalar_eigenmodes_to_vector(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    ny: int, nz: int,
    dy: float, dz: float,
    mode_type: str,
    a: float, b: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
    """Convert scalar eigenmodes to transverse (Ey, Ez, Hy, Hz) profiles.

    For TE modes, the scalar eigenfunction is Hz (longitudinal H).
    Transverse E is derived from: E_t = (-1/kc^2) * z_hat x grad(Hz)
        Ey = -(1/kc^2) * d(Hz)/dz   ... wait, careful with convention.

    Actually, following Pozar convention for propagation along x:
    TE modes: scalar eigenfunction psi = Hz.
        Ey = -(1/kc^2) * (d psi / dz) * (something)  — but for the
        eigenmode profile (not the full propagating field), the transverse
        E components from the scalar potential are:

    For TE (Neumann, psi = Hz):
        Ey_profile ~ -d(psi)/dz
        Ez_profile ~ +d(psi)/dy

    For TM (Dirichlet, psi = Ez):
        Ey_profile ~ +d(psi)/dy
        Ez_profile ~ +d(psi)/dz

    Then H_t for forward propagation: hy = -ez, hz = ey (same as analytical).

    Returns list of (ey, ez, hy, hz, kc_squared) tuples.
    """
    results = []
    n_modes = len(eigenvalues)

    for idx in range(n_modes):
        kc_sq = eigenvalues[idx]
        psi = eigenvectors[:, idx].reshape(ny, nz)

        if mode_type == "TE":
            # TE: psi = Hz pattern, Neumann BC
            # Ey ~ -dpsi/dz,  Ez ~ +dpsi/dy
            # Use central differences
            dpsi_dy = np.zeros_like(psi)
            dpsi_dz = np.zeros_like(psi)

            # Interior central differences
            dpsi_dy[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * dy)
            # Boundaries: one-sided (Neumann: dpsi/dn = 0 at walls,
            # but we compute gradient of the eigenfunction itself)
            dpsi_dy[0, :] = (psi[1, :] - psi[0, :]) / dy
            dpsi_dy[-1, :] = (psi[-1, :] - psi[-2, :]) / dy

            dpsi_dz[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * dz)
            dpsi_dz[:, 0] = (psi[:, 1] - psi[:, 0]) / dz
            dpsi_dz[:, -1] = (psi[:, -1] - psi[:, -2]) / dz

            ey = -dpsi_dz
            ez = dpsi_dy

        else:
            # TM: psi = Ez pattern, Dirichlet BC
            # Ey ~ +dpsi/dy,  Ez ~ +dpsi/dz
            dpsi_dy = np.zeros_like(psi)
            dpsi_dz = np.zeros_like(psi)

            dpsi_dy[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * dy)
            dpsi_dy[0, :] = (psi[1, :] - psi[0, :]) / dy
            dpsi_dy[-1, :] = (psi[-1, :] - psi[-2, :]) / dy

            dpsi_dz[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2 * dz)
            dpsi_dz[:, 0] = (psi[:, 1] - psi[:, 0]) / dz
            dpsi_dz[:, -1] = (psi[:, -1] - psi[:, -2]) / dz

            ey = dpsi_dy
            ez = dpsi_dz

        # H for forward +x propagation: hy = -ez, hz = ey
        hy = -ez.copy()
        hz = ey.copy()

        # Normalize: integral(Ey^2 + Ez^2) dA = 1
        power = np.sum(ey**2 + ez**2) * dy * dz
        if power > 1e-30:
            norm = np.sqrt(power)
            ey = ey / norm
            ez = ez / norm
            hy = hy / norm
            hz = hz / norm

        results.append((ey, ez, hy, hz, kc_sq))

    return results


def _numerical_modes(a: float, b: float, dx: float,
                     freqs: np.ndarray, n_modes: int,
                     eps_cross: np.ndarray | None,
                     mu_cross: np.ndarray | None,
                     ) -> list[WaveguideMode]:
    """Solve eigenmodes numerically for potentially inhomogeneous fill.

    Solves both the TE (Neumann) and TM (Dirichlet) scalar eigenvalue
    problems, collects all modes, sorts by cutoff frequency, and returns
    the lowest ``n_modes``.
    """
    ny = max(1, int(round(a / dx)))
    nz = max(1, int(round(b / dx)))
    dy = a / ny
    dz = b / nz

    # Default: uniform fill
    if eps_cross is None:
        eps = None
        eps_r_eff = 1.0
    else:
        eps = np.asarray(eps_cross, dtype=np.float64)
        if eps.shape != (ny, nz):
            raise ValueError(
                f"eps_cross shape {eps.shape} does not match grid ({ny}, {nz})"
            )
        eps_r_eff = float(np.mean(eps))

    if mu_cross is None:
        mu_r_eff = 1.0
    else:
        mu_arr = np.asarray(mu_cross, dtype=np.float64)
        if mu_arr.shape != (ny, nz):
            raise ValueError(
                f"mu_cross shape {mu_arr.shape} does not match grid ({ny}, {nz})"
            )
        mu_r_eff = float(np.mean(mu_arr))

    # Number of eigenmodes to request from each family.
    # Request more than needed since we merge TE + TM and sort.
    n_request = min(n_modes + 4, ny * nz - 1)
    n_request = max(n_request, 1)

    all_modes: list[WaveguideMode] = []

    # --- TE modes (Neumann) ---
    L_te = _build_laplacian_2d_neumann(ny, nz, dy, dz, eps)
    # eigsh finds smallest eigenvalues of symmetric positive semi-definite matrix.
    # The zero eigenvalue corresponds to the trivial constant mode (no cutoff).
    # We request n_request+1 to have room to skip the zero mode.
    n_te_req = min(n_request + 1, ny * nz - 1)
    if n_te_req >= 1:
        try:
            te_vals, te_vecs = eigsh(L_te, k=n_te_req, which='SM')
        except Exception:
            te_vals, te_vecs = np.array([]), np.zeros((ny * nz, 0))

        # Filter out near-zero eigenvalue (trivial constant mode)
        valid_mask = te_vals > 1e-6
        te_vals = te_vals[valid_mask]
        te_vecs = te_vecs[:, valid_mask]

        if len(te_vals) > 0:
            te_profiles = _scalar_eigenmodes_to_vector(
                te_vals, te_vecs, ny, nz, dy, dz, "TE", a, b
            )
            for ey, ez, hy, hz, kc_sq in te_profiles:
                kc = np.sqrt(max(kc_sq, 0.0))
                fc = kc * C0_LOCAL / (2 * np.pi)
                beta = _compute_beta_numpy(freqs, fc, eps_r_eff, mu_r_eff)

                # Try to identify (m, n) from the eigenvalue
                m_n = _identify_mode_indices(kc_sq, a, b)

                all_modes.append(WaveguideMode(
                    ey_profile=ey,
                    ez_profile=ez,
                    hy_profile=hy,
                    hz_profile=hz,
                    beta=beta,
                    f_cutoff=fc,
                    mode_type="TE",
                    mode_indices=m_n,
                ))

    # --- TM modes (Dirichlet) ---
    L_tm = _build_laplacian_2d_dirichlet(ny, nz, dy, dz, eps)
    n_tm_req = min(n_request, ny * nz - 1)
    if n_tm_req >= 1:
        try:
            tm_vals, tm_vecs = eigsh(L_tm, k=n_tm_req, which='SM')
        except Exception:
            tm_vals, tm_vecs = np.array([]), np.zeros((ny * nz, 0))

        valid_mask = tm_vals > 1e-6
        tm_vals = tm_vals[valid_mask]
        tm_vecs = tm_vecs[:, valid_mask]

        if len(tm_vals) > 0:
            tm_profiles = _scalar_eigenmodes_to_vector(
                tm_vals, tm_vecs, ny, nz, dy, dz, "TM", a, b
            )
            for ey, ez, hy, hz, kc_sq in tm_profiles:
                kc = np.sqrt(max(kc_sq, 0.0))
                fc = kc * C0_LOCAL / (2 * np.pi)
                beta = _compute_beta_numpy(freqs, fc, eps_r_eff, mu_r_eff)

                m_n = _identify_mode_indices(kc_sq, a, b)

                all_modes.append(WaveguideMode(
                    ey_profile=ey,
                    ez_profile=ez,
                    hy_profile=hy,
                    hz_profile=hz,
                    beta=beta,
                    f_cutoff=fc,
                    mode_type="TM",
                    mode_indices=m_n,
                ))

    # Sort by cutoff frequency and return lowest n_modes
    all_modes.sort(key=lambda m: m.f_cutoff)
    return all_modes[:n_modes]


def _identify_mode_indices(kc_sq: float, a: float, b: float,
                           ) -> tuple[int, int]:
    """Best-effort identification of (m, n) from numerical kc^2.

    Compares against analytical kc^2 = (m*pi/a)^2 + (n*pi/b)^2 for
    small mode indices. Returns (0, 0) if no good match is found.
    """
    best_err = float('inf')
    best_mn = (0, 0)
    for m in range(0, 8):
        for n in range(0, 8):
            if m == 0 and n == 0:
                continue
            kc_sq_analytical = (m * np.pi / a) ** 2 + (n * np.pi / b) ** 2
            err = abs(kc_sq - kc_sq_analytical) / max(kc_sq_analytical, 1e-30)
            if err < best_err:
                best_err = err
                best_mn = (m, n)
    # Accept match only if relative error < 10%
    if best_err < 0.1:
        return best_mn
    return (0, 0)


def solve_waveguide_modes(
    a: float,
    b: float,
    dx: float,
    freqs: np.ndarray,
    n_modes: int = 1,
    eps_cross: np.ndarray | None = None,
    mu_cross: np.ndarray | None = None,
) -> list[WaveguideMode]:
    """Solve for the lowest eigenmodes of a rectangular waveguide cross-section.

    Parameters
    ----------
    a : float
        Waveguide width (metres), along the first transverse axis.
    b : float
        Waveguide height (metres), along the second transverse axis.
    dx : float
        Cell size (metres).
    freqs : array_like, shape (n_freqs,)
        Frequency points at which to evaluate the propagation constant.
    n_modes : int
        Number of lowest-cutoff modes to return.
    eps_cross : ndarray, shape (ny, nz), optional
        Relative permittivity on the cross-section grid.  ``None`` means
        uniform vacuum (eps_r = 1).
    mu_cross : ndarray, shape (ny, nz), optional
        Relative permeability on the cross-section grid.  ``None`` means
        uniform (mu_r = 1).

    Returns
    -------
    modes : list[WaveguideMode]
        The ``n_modes`` lowest-cutoff eigenmodes, sorted by cutoff frequency.
    """
    freqs = np.asarray(freqs, dtype=np.float64)

    if eps_cross is None and mu_cross is None:
        # Uniform fill — use fast analytical path
        return _analytical_modes(a, b, dx, freqs, n_modes)
    else:
        # Inhomogeneous — numerical eigenvalue solve
        return _numerical_modes(a, b, dx, freqs, n_modes, eps_cross, mu_cross)
