"""JAX record DFT and ringdown tail spectra."""

from __future__ import annotations

import numpy as np
from rfx._ringdown_jax import DFT_BLOCK, working_dtypes


def plain_dft(y, dt, freqs, block=DFT_BLOCK):
    """``dt * sum_n y_n exp(-j 2 pi f n dt)``, ``y (n, C)`` -> ``(nf, C)``.

    :func:`rfx.ringdown.plain_dft`'s convention. Blocked so no ``nf x n``
    kernel is formed: ``exp(-j w (b B + m)) = exp(-j w b B) exp(-j w m)``,
    both factors built on the host in float64 and cast to the working dtype.
    """
    import jax.numpy as jnp
    from rfx._precision import HIGHEST
    rdt, cdt = working_dtypes()
    n, C = int(y.shape[0]), int(y.shape[1])
    freqs = np.asarray(freqs, dtype=np.float64)
    B = int(min(block, n))
    nb = -(-n // B)
    w = -2.0 * np.pi * freqs * float(dt)
    k_blk = np.exp(1j * np.outer(w, np.arange(B, dtype=np.float64)))           # (nf, B)
    z_off = np.exp(1j * np.outer(np.arange(nb, dtype=np.float64) * B, w))      # (nb, nf)
    yp = jnp.zeros((nb * B, C), dtype=rdt).at[:n].set(y.astype(rdt))
    yb = yp.reshape(nb, B, C).astype(cdt)
    inner = jnp.einsum("fm,bmc->bfc", jnp.asarray(k_blk, dtype=cdt), yb, precision=HIGHEST)
    return float(dt) * jnp.einsum("bf,bfc->fc", jnp.asarray(z_off, dtype=cdt), inner,
                                      precision=HIGHEST)


def tail_dft(s, c, n_ref, n_last, dt, freqs, tail_arg=None):
    """:func:`rfx.ringdown.tail_dft` in JAX.

    ``dt sum_k c_k lambda_k**(N+1-n_ref) z**(N+1) / (1 - lambda_k z)`` with
    ``1 - lambda z = -expm1(s dt - j 2 pi f dt)``. ``c (K, C)``; returns ``(nf, C)``.
    ``tail_arg (nf, K)`` is ``s0 dt + log z`` from :func:`host_poles`, formed in
    float64; the device adds ``(s - stop_gradient(s)) dt`` to it, zero in value.
    ``None`` forms the sum on the device in the working precision.
    """
    import jax
    import jax.numpy as jnp
    from rfx._precision import HIGHEST
    _rdt, cdt = working_dtypes()
    freqs = np.asarray(freqs, dtype=np.float64)
    dt = float(dt)
    zN1 = jnp.asarray(np.exp(-2j * np.pi * freqs * dt * (int(n_last) + 1)), dtype=cdt)
    sdt = s * dt
    if tail_arg is None:
        logz = jnp.asarray(-2j * np.pi * freqs * dt, dtype=cdt)               # (nf,)
        one_minus_lz = -jnp.expm1(sdt[None, :] + logz[:, None])               # (nf, K)
    else:
        dsdt = (s - jax.lax.stop_gradient(s)) * dt
        one_minus_lz = -jnp.expm1(tail_arg + dsdt[None, :])
    amp = jnp.exp(sdt * float(int(n_last) + 1 - int(n_ref)))[:, None] * c     # (K, C)
    return dt * jnp.matmul(zN1[:, None] / one_minus_lz, amp, precision=HIGHEST)
