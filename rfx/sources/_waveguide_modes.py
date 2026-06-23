"""Rectangular-waveguide transverse mode solver + mode-profile algebra.

Pure-NumPy helpers extracted verbatim from ``waveguide_port.py``. They sit
**off** the FDTD/JAX autodiff tape (all float64 NumPy, run at port setup), so
they live in their own module to keep ``waveguide_port.py`` focused on port
setup / injection / S-parameter extraction. ``waveguide_port`` re-imports every
name below, so existing import paths (e.g.
``rfx.sources.waveguide_port._second_diff_1d``) are unchanged.

Contents: the cell-centred finite-volume / Galerkin transverse-Laplacian
eigensolver (``_second_diff_1d``, ``_galerkin_*``, ``_cell_centred_gradient``),
eigenmode selection (``_pick_eigenmode_*``), and mode-profile algebra
(``_aperture_area``, ``_shift_profile_to_dual``,
``_orthonormalize_profile_arrays``, ``_scale_h_to_unit_cross``).
"""

from __future__ import annotations

import numpy as np


def _second_diff_1d(widths: np.ndarray, *, bc: str) -> np.ndarray:
    """Cell-centred finite-volume ∂²/∂u² operator with PEC-like BC.

    bc="neumann" uses ghost = same-as-boundary reflection (TE Hx, ∂Hx/∂n = 0).
    bc="dirichlet" uses ghost = negative-of-boundary (TM Ex / TE Ez tangential,
    field = 0 at wall, so for cell-centred values ghost[-1] = -cell[0] about
    a wall at the cell-face at u=0).

    The operator is symmetric-negative-semidefinite (Neumann) or symmetric-
    negative-definite (Dirichlet) so `eigh(-L)` yields non-negative kc².
    """
    n = len(widths)
    D = np.zeros((n, n), dtype=np.float64)
    w = np.asarray(widths, dtype=np.float64)
    for j in range(n):
        if j > 0:
            d_left = 0.5 * (w[j - 1] + w[j])
            coeff = 1.0 / (d_left * w[j])
            D[j, j - 1] += coeff
            D[j, j] -= coeff
        elif bc == "dirichlet":
            # Ghost cell beyond the wall reflects with opposite sign; the
            # wall lives at the outer face (u = 0). Face-spacing to the
            # mirrored ghost is w[0], and ghost value = -cell[0]:
            #   flux = (cell[0] - ghost) / w[0] = 2·cell[0] / w[0]
            coeff = 2.0 / (w[0] * w[0])
            D[j, j] -= coeff
        if j < n - 1:
            d_right = 0.5 * (w[j] + w[j + 1])
            coeff = 1.0 / (d_right * w[j])
            D[j, j + 1] += coeff
            D[j, j] -= coeff
        elif bc == "dirichlet":
            coeff = 2.0 / (w[-1] * w[-1])
            D[j, j] -= coeff
    return D


def _galerkin_stiffness_mass_1d(
    widths: np.ndarray, *, bc: str
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric finite-volume stiffness K and diagonal mass m for -d²/du².

    Galerkin / mixed-FV form of the 1-D operator on a cell-centred,
    possibly non-uniform mesh. Unlike ``_second_diff_1d`` (which builds the
    strong-form ``D = M⁻¹K`` and is ASYMMETRIC on a graded mesh, silently
    corrupting ``eigh``), this returns the SYMMETRIC stiffness ``K`` and the
    diagonal mass ``m = w`` separately so the generalized eigenproblem
    ``K φ = kc² M φ`` can be symmetrized exactly:

        K[j, j±1] = -1/d_face   (d_face = ½(w[j]+w[j±1]))     -- symmetric
        K[j, j]   = Σ 1/d_face  (+ Dirichlet wall terms)
        m[j]      = w[j]

    bc="neumann"   : ∂φ/∂n = 0 at the walls (no wall flux; TE Hx).
    bc="dirichlet" : φ = 0 at the wall faces (ghost = -cell; TM Ez / Ex).
    """
    n = len(widths)
    w = np.asarray(widths, dtype=np.float64)
    K = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        if j > 0:
            d_left = 0.5 * (w[j - 1] + w[j])
            c = 1.0 / d_left
            K[j, j] += c
            K[j, j - 1] -= c
        elif bc == "dirichlet":
            # Wall at the outer face; ghost = -cell[0], face spacing w[0].
            K[j, j] += 2.0 / w[0]
        if j < n - 1:
            d_right = 0.5 * (w[j] + w[j + 1])
            c = 1.0 / d_right
            K[j, j] += c
            K[j, j + 1] -= c
        elif bc == "dirichlet":
            K[j, j] += 2.0 / w[-1]
    return K, w


def _galerkin_eigh_separable_laplacian_2d(
    Ku: np.ndarray, mu: np.ndarray,
    Kv: np.ndarray, mv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric eigensolve of the separable 2-D transverse Laplacian.

    Solves ``(Ku⊗Mv + Mu⊗Kv) φ = kc² (Mu⊗Mv) φ`` (the Galerkin form of
    ``-∇_t² φ = kc² φ``) by symmetrizing with ``S = (Mu⊗Mv)^{-1/2}``:

        A = S (Ku⊗Mv + Mu⊗Kv) S ,   A ψ = kc² ψ ,   φ = S ψ

    ``A`` is symmetric by construction, so ``np.linalg.eigh`` is exact on a
    graded mesh (no silent lower-triangle symmetrization of an asymmetric
    strong-form operator). Returns ``(evals, evecs)`` with evecs as columns
    in the ORIGINAL φ basis (unit-mass-normalized), ascending eigenvalue.
    """
    Mv = np.diag(mv); Mu = np.diag(mu)
    big = np.kron(Ku, Mv) + np.kron(Mu, Kv)        # (nu*nv, nu*nv) symmetric
    m_diag = np.kron(mu, mv)                         # diagonal of (Mu⊗Mv)
    s = 1.0 / np.sqrt(m_diag)
    A = (s[:, None] * big) * s[None, :]             # S A S, symmetric
    A = 0.5 * (A + A.T)                              # clean tiny asymmetry
    evals, psi = np.linalg.eigh(A)
    evecs = psi * s[:, None]                         # φ = S ψ
    return evals, evecs


def _cell_centred_gradient(field: np.ndarray, widths: np.ndarray,
                            axis: int, *, bc: str) -> np.ndarray:
    """∂field/∂u at cell centres for cell-centred values with PEC-like BC.

    Uses face-derivatives averaged back to cell centres. Boundary handling:
      bc="neumann"   → face-derivative at the wall is 0 (mirrored-equal ghost)
      bc="dirichlet" → ghost value is -cell[0], face-derivative is
                       2·cell[0]/w[0] (signed)

    Returns an array the same shape as `field`.
    """
    n = field.shape[axis]
    w = np.asarray(widths, dtype=np.float64)
    field64 = np.asarray(field, dtype=np.float64)
    field_swap = np.moveaxis(field64, axis, 0)
    face = np.zeros_like(field_swap)
    face_spacing = 0.5 * (w[:-1] + w[1:])  # (n-1,)
    face[1:n, ...] = (field_swap[1:n, ...] - field_swap[:n - 1, ...]) \
        / face_spacing.reshape((-1,) + (1,) * (field_swap.ndim - 1))
    # Boundary faces (j = 0 and j = n) lie on the PEC walls.
    if bc == "dirichlet":
        # Ghost = -cell[0], face derivative = (cell[0] - (-cell[0])) / w[0]
        #                                    = 2·cell[0] / w[0]
        face0 = 2.0 * field_swap[0, ...] / w[0]
        face_n = -2.0 * field_swap[-1, ...] / w[-1]
    else:  # neumann
        face0 = np.zeros_like(field_swap[0, ...])
        face_n = np.zeros_like(field_swap[-1, ...])
    # Cell-centre gradient = half-width-weighted mean of two face values.
    grad_swap = np.zeros_like(field_swap)
    grad_swap[0, ...] = 0.5 * (face0 + face[1, ...])
    grad_swap[-1, ...] = 0.5 * (face[-1, ...] + face_n)
    if n > 2:
        grad_swap[1:n - 1, ...] = 0.5 * (face[1:n - 1, ...] + face[2:n, ...])
    return np.moveaxis(grad_swap, 0, axis)


def _pick_eigenmode_by_overlap(evecs: np.ndarray,
                                evals: np.ndarray,
                                analytic_flat: np.ndarray,
                                num_candidates: int = 40,
                                ) -> int:
    """Pick the eigenvector with the largest |overlap| with an analytic shape.

    Rank-based picking breaks when discrete kc² ordering swaps nearly-
    degenerate analytic modes (e.g. TE30 vs TE21 on a WR-90 grid). Overlap
    picks from the lowest `num_candidates` eigenvectors and returns the
    index with the largest absolute projection onto `analytic_flat`.
    """
    n = min(num_candidates, evecs.shape[1])
    analytic_unit = analytic_flat / max(float(np.linalg.norm(analytic_flat)),
                                         1e-30)
    overlaps = np.abs(evecs[:, :n].T @ analytic_unit)
    return int(np.argmax(overlaps))


def _pick_eigenmode_by_target_then_overlap(
    evecs: np.ndarray,
    evals: np.ndarray,
    analytic_flat: np.ndarray,
    target_kc2: float,
    num_candidates: int = 80,
) -> int:
    """Pick a mode by cutoff proximity, using overlap to break ties."""
    n = min(num_candidates, evecs.shape[1])
    target = max(float(target_kc2), 1e-30)
    rel = np.abs(evals[:n] - target) / target
    min_rel = float(np.min(rel))
    candidates = np.where(rel <= min_rel + 0.05)[0]
    if candidates.size == 1:
        return int(candidates[0])

    analytic_unit = analytic_flat / max(float(np.linalg.norm(analytic_flat)),
                                         1e-30)
    overlaps = np.abs(evecs[:, candidates].T @ analytic_unit)
    return int(candidates[int(np.argmax(overlaps))])


def _aperture_area(
    u_widths: np.ndarray,
    v_widths: np.ndarray,
    aperture_dA: np.ndarray | None,
) -> np.ndarray:
    """Return the physical aperture integration measure used by extraction."""
    if aperture_dA is None:
        dA = np.asarray(u_widths, dtype=np.float64)[:, None] * np.asarray(
            v_widths, dtype=np.float64
        )[None, :]
    else:
        dA = np.asarray(aperture_dA, dtype=np.float64)
    if dA.shape != (len(u_widths), len(v_widths)):
        raise ValueError(
            "aperture_dA must have shape "
            f"({len(u_widths)}, {len(v_widths)}), got {dA.shape}"
        )
    flat = dA.ravel()
    positive = flat[flat > 0.0]
    if positive.size == 0:
        raise ValueError("aperture_dA must contain at least one positive cell")
    return dA


def _shift_profile_to_dual(
    profile: np.ndarray,
    h_offset: tuple[float, float],
) -> np.ndarray:
    """Co-locate a transverse template to the E plane by a SYMMETRIC
    half-cell average.

    The Yee H sits a half cell from E. The legacy one-sided average
    ``0.5*(f + roll(f, -1))`` introduces a non-symmetric O(dx) bias that
    breaks the parity orthogonality between modes: the discrete modal
    cross-overlap ``∫(e_m × h_n)·n̂ dA`` becomes O(dx) and asymmetric
    (``O_12 != O_21``), leaking ~2 % power between TE10 and TE20 in an
    over-moded guide. The symmetric centered stencil
    ``0.25*(roll(+1) + 2f + roll(-1))`` keeps the field co-located,
    restores parity, and drives the cross-overlap to exactly zero for
    symmetric mode pairs (verified numerically: 0.029 -> 0 at dx=250 um).
    Edges are clamped (reflect), not wrapped, so PEC/Neumann walls do not
    leak.
    """
    out = np.asarray(profile, dtype=np.float64).copy()
    for axis, offset in enumerate(h_offset):
        if offset == 0.0:
            continue
        if offset != 0.5:
            raise ValueError(
                "h_offset currently supports only 0.0 or 0.5 per axis, "
                f"got {h_offset!r}"
            )
        # Edge-clamped neighbours (no wraparound at the aperture walls).
        plus = np.concatenate(
            [np.take(out, [0], axis=axis), np.take(out, range(out.shape[axis] - 1), axis=axis)],
            axis=axis,
        )
        minus = np.concatenate(
            [np.take(out, range(1, out.shape[axis]), axis=axis), np.take(out, [-1], axis=axis)],
            axis=axis,
        )
        out = 0.25 * (plus + 2.0 * out + minus)
    return out


def _orthonormalize_profile_arrays(
    ey: np.ndarray,
    ez: np.ndarray,
    hy: np.ndarray,
    hz: np.ndarray,
    dA: np.ndarray,
    previous: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormalize one stored profile against previous profiles."""
    ey = np.asarray(ey, dtype=np.float64).copy()
    ez = np.asarray(ez, dtype=np.float64).copy()
    hy = np.asarray(hy, dtype=np.float64).copy()
    hz = np.asarray(hz, dtype=np.float64).copy()
    for pey, pez, phy, phz in previous:
        c_prev = float(np.sum((pey * phz - pez * phy) * dA))
        if abs(c_prev) <= 1e-30:
            continue
        proj = float(np.sum((ey * phz - ez * phy) * dA)) / c_prev
        ey -= proj * pey
        ez -= proj * pez
        hy -= proj * phy
        hz -= proj * phz
    c_self = float(np.sum((ey * hz - ez * hy) * dA))
    if abs(c_self) <= 1e-30:
        raise ValueError("Cannot normalize waveguide mode with zero self-overlap")
    scale = np.sqrt(abs(c_self))
    return ey / scale, ez / scale, hy / scale, hz / scale


def _scale_h_to_unit_cross(
    ey: np.ndarray,
    ez: np.ndarray,
    hy: np.ndarray,
    hz: np.ndarray,
    dA: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale stored H templates so ∫(e×h)·n dA is unity."""
    c_self = float(np.sum((ey * hz - ez * hy) * dA))
    if abs(c_self) <= 1e-30:
        return hy, hz
    return hy / c_self, hz / c_self
