"""Differentiable sparse direct solve for frequency-domain problems.

``sparse_solve(data, rows, cols, b)`` returns ``x`` with ``A x = b`` where
``A`` is the sparse matrix with COO entries ``(rows[k], cols[k]) = data[k]``.
The sparsity pattern is static (NumPy); the entries and the right-hand side
are JAX arrays and BOTH are differentiable, in forward mode (``jax.jvp``,
``jax.jacfwd``) and reverse mode (``jax.grad``, ``jax.vjp``).

``b`` may be a single vector ``(n,)`` or a block of ``m`` right-hand sides
``(n, m)``; the result has the same shape. One LU factorisation serves all
``m`` columns, which is the N-port FDFD case: factor the Helmholtz operator
once, then solve for every port excitation.

Mechanism
---------
The factorisation itself runs on the host through ``scipy.sparse.linalg.splu``
inside ``jax.pure_callback``; JAX never sees its internals. Differentiation
comes from ``jax.lax.custom_linear_solve`` and the implicit-function rule:

    A x = b  =>  dx = A^-1 (db - dA x)                      (forward mode)
    L(x)     =>  lam = A^-T dL/dx,  dL/db = lam,
                 dL/dA_k = -lam[rows_k] * x[cols_k]          (reverse mode)

so a gradient with respect to every matrix entry and every rhs entry costs
ONE extra (transposed) solve, independent of the number of parameters and
with no stored time history. That is the property FDTD reverse-mode AD does
not have (checkpointing memory ~ sqrt(steps) * fields) and the reason this
module exists.

Precision
---------
The systems this is built for (Helmholtz with discrete DtN ports) have
condition numbers ~1e12; float32 is meaningless. Callers must enable x64
(``jax.config.update("jax_enable_x64", True)`` or the scoped
``enable_x64`` context) -- the solve raises otherwise rather than silently
downcasting.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

__all__ = ["sparse_solve", "sparse_matvec", "clear_factor_cache"]

# LU factors keyed by the matrix entries, so the forward solve and the
# transposed adjoint solve of one gradient evaluation share a factorisation.
_FACTOR_CACHE: OrderedDict[bytes, spl.SuperLU] = OrderedDict()
_FACTOR_CACHE_SIZE = 4


def clear_factor_cache() -> None:
    _FACTOR_CACHE.clear()


def _factor(data: np.ndarray, rows: np.ndarray, cols: np.ndarray, n: int) -> spl.SuperLU:
    key = hashlib.blake2b(data.tobytes(), digest_size=16).digest()
    lu = _FACTOR_CACHE.get(key)
    if lu is None:
        a = sp.csc_matrix((data, (rows, cols)), shape=(n, n))
        a.sum_duplicates()
        lu = spl.splu(a)
        _FACTOR_CACHE[key] = lu
        while len(_FACTOR_CACHE) > _FACTOR_CACHE_SIZE:
            _FACTOR_CACHE.popitem(last=False)
    else:
        _FACTOR_CACHE.move_to_end(key)
    return lu


def sparse_matvec(data: jax.Array, rows: np.ndarray, cols: np.ndarray, x: jax.Array) -> jax.Array:
    """``A @ x`` for the COO matrix ``(rows, cols, data)``; pure JAX, differentiable.

    ``x`` is ``(n,)`` or ``(n, m)``; in the block case every column is
    multiplied by the same matrix and the result is ``(n, m)``.
    """
    d = data if x.ndim == 1 else data[:, None]
    return jnp.zeros(x.shape, dtype=jnp.result_type(data, x)).at[rows].add(d * x[cols])


def sparse_solve(data: jax.Array, rows: np.ndarray, cols: np.ndarray, b: jax.Array) -> jax.Array:
    """Solve ``A x = b`` for the sparse COO matrix ``(rows, cols, data)``.

    ``rows``/``cols`` are static integer NumPy arrays (the pattern);
    ``data`` and ``b`` are JAX arrays and may be traced, jitted and
    differentiated in either mode. ``b`` is ``(n,)`` for one right-hand side
    or ``(n, m)`` for ``m`` of them; ``x`` has the same shape and the host
    factorises ``A`` once for all columns. Duplicate ``(row, col)`` pairs are
    summed, as in ``scipy.sparse``.
    """
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "rfx.fdfd.sparse_solve needs x64: the frequency-domain systems it "
            "targets have cond ~1e12. Enable jax_enable_x64 for this scope.")
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if b.ndim not in (1, 2):
        raise ValueError(f"b must have shape (n,) or (n, m), got {b.shape}")
    n = int(b.shape[0])
    if rows.shape != cols.shape:
        raise ValueError("rows and cols must have the same shape")
    if data.shape != rows.shape:
        raise ValueError(f"data has shape {data.shape}, pattern has {rows.shape}")
    data = jnp.asarray(data, dtype=jnp.complex128)
    b = jnp.asarray(b, dtype=jnp.complex128)
    out_spec = jax.ShapeDtypeStruct(tuple(b.shape), jnp.complex128)

    def _host(trans: bool):
        def f(d, rhs):
            d = np.asarray(d, dtype=np.complex128)
            rhs = np.asarray(rhs, dtype=np.complex128)
            lu = _factor(d, rows, cols, n)
            return lu.solve(rhs, trans="T" if trans else "N")
        return f

    host_solve = _host(False)
    host_solve_t = _host(True)

    def matvec(x):
        return sparse_matvec(data, rows, cols, x)

    def solve(_matvec, rhs):
        return jax.pure_callback(host_solve, out_spec, data, rhs)

    def transpose_solve(_vecmat, rhs):
        return jax.pure_callback(host_solve_t, out_spec, data, rhs)

    return jax.lax.custom_linear_solve(matvec, b, solve, transpose_solve)
