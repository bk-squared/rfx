"""Compact wavelets for adjoint spectral targets."""

from __future__ import annotations

import jax.numpy as jnp
from rfx._precision import HIGHEST
from rfx.core.jax_utils import complex_solve


def _wavelet_basis(freqs, dt, length, n):
    """Compact discrete derivative: telescoping sum has no deposited DC."""
    def primitive(k):
        envelope = jnp.where((k >= 0) & (k < length - 1),
                            jnp.sin(jnp.pi * (k + 1) / length)**4, 0.)
        return envelope[..., None] * jnp.exp(
            2j * jnp.pi * ((k + 1) * dt)[..., None] * freqs)
    return primitive(n) - primitive(n - 1)


def _wavelet_coefficients(freqs, dt, length, target):
    """Real compact wavelet, exact complex bin targets via Nf-square solves."""
    t = (jnp.arange(length, dtype=freqs.dtype) + 1) * dt
    carrier = jnp.exp(2j * jnp.pi * t[:, None] * freqs)
    basis = _wavelet_basis(freqs, dt, length, jnp.arange(length, dtype=freqs.dtype))
    transform = jnp.conj(carrier).T * dt
    a = jnp.matmul(transform, basis, precision=HIGHEST)
    b = jnp.matmul(transform, jnp.conj(basis), precision=HIGHEST)
    flat = target.reshape((len(freqs), -1))
    cross = complex_solve(jnp.conj(a), jnp.conj(b))
    rhs = flat - jnp.matmul(b, complex_solve(jnp.conj(a), jnp.conj(flat)), precision=HIGHEST)
    return complex_solve(a - jnp.matmul(b, cross, precision=HIGHEST), rhs).reshape(target.shape)
