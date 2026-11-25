"""Least-squares measurement of forward and backward MSL waves."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from rfx._internal._precision import HIGHEST


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
    a_mat = jnp.stack([col_fwd, col_bwd], axis=-1)
    sol, _, _, _ = jnp.linalg.lstsq(a_mat, v, rcond=None)
    alpha = sol[0]
    gamma = sol[1]
    pred = jnp.matmul(a_mat, sol, precision=HIGHEST)
    residual = jnp.sqrt(jnp.sum(jnp.abs(v - pred) ** 2))
    return alpha, gamma, residual


def _msl_fit_conditioning(x, beta, alpha, gamma):
    """Return phase span and per-probe fit conditioning, detached from AD.

    Centering changes only the wave amplitudes. Subtracting a multiple of
    sin(beta*y) from the beta column leaves its orthogonal residual unchanged:
    y*cos(beta*y) becomes y*(cos(beta*y) - sinc(beta*y/pi)). This exact
    identity removes the projection of a first-order column from the odd
    term y*cos(beta*y), but cos(beta*y) - sinc(beta*y/pi) ≈ -(beta*y)**2/3
    still subtracts two numbers near 1 in float32. The odd term dominates
    sigma_perp only near a voltage null at the centre (a + g ≈ 0); the even
    term -(a + g)*y*sin(beta*y) has no such cancellation. No small-phase
    series is used.
    QR uses the equivalent
    well-scaled wave basis {cos(beta*y), y*sinc(beta*y/pi)}.
    """
    x, beta, alpha, gamma = jax.tree.map(
        jax.lax.stop_gradient, (x, beta, alpha, gamma))
    centre = jnp.mean(x)
    y = x - centre

    def per_bin(b, a, g):
        a = a * jnp.exp(-1j * b * centre)
        g = g * jnp.exp(1j * b * centre)
        phase = b * y
        cosine = jnp.cos(phase)
        sinc = jnp.sinc(phase / jnp.pi)
        column = (-(a + g) * y * jnp.sin(phase)
                  + 1j * (g - a) * y * (cosine - sinc))
        basis = jnp.stack((cosine, y * sinc), axis=-1)
        q, _ = jnp.linalg.qr(basis, mode="reduced")
        residual = column - q @ (q.conj().T @ column)
        sigma = jnp.linalg.norm(residual)
        v_fit = a * jnp.exp(-1j * phase) + g * jnp.exp(1j * phase)
        return jnp.linalg.norm(v_fit) / (jnp.real(b) * sigma)

    return (jnp.real(beta) * jnp.abs(x[-1] - x[0]),
            jax.vmap(per_bin)(beta, alpha, gamma))
