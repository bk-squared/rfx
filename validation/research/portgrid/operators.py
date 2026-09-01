"""M0 operator library: interface interpolation/restriction pair + P-norm adjoint.

Implements the 2-D interpolation rule of Bekmambetova/Zhang/Triverio,
arXiv:1606.08761 (IEEE TAP 65(2):751, 2017):

  (55)  E_hat_S = E_N * T          (fine tangential E replicated from coarse)
  (56)  H_N     = T^T H_hat_S / r  (coarse hanging H = average of fine hanging H)

for an interface line made of ``m`` coarse edges refined by an integer ratio
``r``.  With the P-norm weights

  P_c = ell * I_m         (coarse tangential edge length)
  P_f = (ell / r) * I_mr  (fine tangential edge length)

the pair satisfies the exact adjoint relation

  T_c2f = P_f^{-1} T_f2c^T P_c ,

which is the algebraic statement that the interpolation rule is lossless
(supply rate (62)-(63) cancels identically).

The 2-D paper supports arbitrary integer r (its own stability fixture uses
r = 4); the odd-only restriction belongs to the 3-D paper (arXiv:1705.02274,
Sec. V) and is enforced only when ``require_odd=True``.
"""

from __future__ import annotations

import numpy as np


def build_interface_operators(r: int, m: int, ell: float = 1.0, *, require_odd: bool = False):
    """Build the interface transfer pair and P-norm weights.

    Parameters
    ----------
    r : integer refinement ratio (>= 1).
    m : number of coarse edges along the interface line.
    ell : physical tangential length of one coarse edge (meters).
    require_odd : enforce the 3-D paper's odd-ratio restriction.

    Returns
    -------
    dict with dense float64 arrays:
      T_c2f : (m*r, m)  replication (paper eq. (55), per coarse edge)
      T_f2c : (m, m*r)  averaging   (paper eq. (56), per coarse edge)
      P_c   : (m,)      coarse boundary-length weights (ell)
      P_f   : (m*r,)    fine boundary-length weights (ell/r)
    """
    if r < 1 or int(r) != r:
        raise ValueError(f"refinement ratio must be a positive integer, got {r}")
    if require_odd and r % 2 == 0:
        raise ValueError(f"odd refinement ratio required (3-D paper rule), got {r}")
    if m < 1:
        raise ValueError(f"need at least one coarse edge, got {m}")
    if ell <= 0.0:
        raise ValueError(f"edge length must be positive, got {ell}")

    t_r = np.ones((r, 1), dtype=np.float64)  # T_r of the paper
    t_c2f = np.kron(np.eye(m, dtype=np.float64), t_r)          # (m*r, m)
    t_f2c = np.kron(np.eye(m, dtype=np.float64), t_r.T) / r    # (m, m*r)
    p_c = np.full(m, float(ell), dtype=np.float64)
    p_f = np.full(m * r, float(ell) / r, dtype=np.float64)
    return {"T_c2f": t_c2f, "T_f2c": t_f2c, "P_c": p_c, "P_f": p_f, "r": r, "m": m, "ell": float(ell)}


def adjoint_residual(ops) -> float:
    """Relative max residual of T_c2f - P_f^{-1} T_f2c^T P_c (exact identity)."""
    t_c2f, t_f2c = ops["T_c2f"], ops["T_f2c"]
    p_c, p_f = ops["P_c"], ops["P_f"]
    rhs = (t_f2c.T * p_c[np.newaxis, :]) / p_f[:, np.newaxis]
    return float(np.max(np.abs(t_c2f - rhs)) / np.max(np.abs(t_c2f)))


def pullback_residual(ops, rng: np.random.Generator) -> float:
    """Residual of the reverse-mode identity T_c2f^T w = P_c T_f2c P_f^{-1} w."""
    t_c2f, t_f2c = ops["T_c2f"], ops["T_f2c"]
    p_c, p_f = ops["P_c"], ops["P_f"]
    w = rng.standard_normal(t_c2f.shape[0])
    lhs = t_c2f.T @ w
    rhs = p_c * (t_f2c @ (w / p_f))
    scale = max(np.max(np.abs(lhs)), 1e-300)
    return float(np.max(np.abs(lhs - rhs)) / scale)


def supply_rate_residual(ops, rng: np.random.Generator) -> float:
    """Normalized supply rate of the interpolation rule on random fields.

    Paper eq. (62): with the rules (55)/(56) substituted, the energy flowing
    out of the fine side exactly equals the energy flowing into the coarse
    side, so s = 0 (eq. (63)).  dt factors cancel in the normalization and
    are omitted.
    """
    m, r, ell = ops["m"], ops["r"], ops["ell"]
    t_c2f, t_f2c = ops["T_c2f"], ops["T_f2c"]

    e_n = rng.standard_normal(m)        # coarse interface E at time n
    e_np1 = rng.standard_normal(m)      # ... at time n+1
    h_hat = rng.standard_normal(m * r)  # fine hanging variables (free)

    e_hat_n = t_c2f @ e_n               # rule (55) at both time levels
    e_hat_np1 = t_c2f @ e_np1
    h_coarse = t_f2c @ h_hat            # rule (56)

    # eq. (62): s = -dt*ell*<E_N>*H_N + dt*(ell/r)*<E_hat_S>^T H_hat_S
    term_coarse = -ell * (0.5 * (e_n + e_np1)) @ h_coarse
    term_fine = (ell / r) * (0.5 * (e_hat_n + e_hat_np1)) @ h_hat
    s = term_coarse + term_fine
    scale = abs(term_coarse) + abs(term_fine)
    return float(abs(s) / max(scale, 1e-300))
