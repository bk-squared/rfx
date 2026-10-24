"""Least-squares measurement of forward and backward MSL waves."""
from __future__ import annotations

import jax.numpy as jnp

from rfx._precision import HIGHEST


def _lstsq_alpha_gamma(
    v: jnp.ndarray, x: jnp.ndarray, beta: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve ``V_n = α e^{−jβx_n} + γ e^{+jβx_n}`` for one β by SVD lstsq.

    Parameters
    ----------
    v : (N,) complex
        Voltage phasors at the N probe positions.
    x : (N,) real
        Probe positions (metres), measured along the propagation axis.
    beta : scalar complex
        Trial propagation constant (rad/m).  Complex β allows a small
        imaginary (loss) part.

    Returns
    -------
    alpha, gamma : scalar complex
        Forward / backward wave amplitudes at ``x = 0``.
    residual : scalar real
        L2 norm of ``V − (α e^{−jβx} + γ e^{+jβx})`` — the fit quality.
    """
    x_c = x.astype(jnp.complex64)
    col_fwd = jnp.exp(-1j * beta * x_c)
    col_bwd = jnp.exp(+1j * beta * x_c)
    a_mat = jnp.stack([col_fwd, col_bwd], axis=-1)        # (N, 2)
    # SVD-based least squares; rcond=None uses the JAX default cutoff.
    sol, _, _, _ = jnp.linalg.lstsq(a_mat, v, rcond=None)
    alpha = sol[0]
    gamma = sol[1]
    pred = jnp.matmul(a_mat, sol, precision=HIGHEST)
    residual = jnp.sqrt(jnp.sum(jnp.abs(v - pred) ** 2))
    return alpha, gamma, residual
