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

LU cache
--------
Factors are memoised in a small LRU keyed by the *whole* problem: the
sparsity pattern ``(n, rows, cols)``, the entry bytes, and the column
ordering. That is what lets the forward solve and the transposed adjoint
solve of one gradient evaluation share a single factorisation. The pattern
has to be in the key, because nothing else ties the entry values to the
matrix they belong to: byte-identical data on a different pattern hands the
second problem the first one's factors. The transpose is the sharp case
(same ``data``, ``rows`` and ``cols`` swapped, as any hand-written adjoint
system does) and so is one material array scattered onto two different
grids. Measured: under the old data-only key the transposed solve came back
bit-identical to the untransposed one and its relative residual was 0.41,
i.e. simply wrong. The pattern digest is computed once per
``sparse_solve`` call, not per host callback: one blake2b pass over the index
arrays, 0.18 ms at 2.2e4 nonzeros and 1.6 ms at 1.6e5 (~1.6 GB/s). End to
end that is +0.34 ms on a 5.5 ms cached hplane solve whose factorisation
costs 7.8 ms, and noise next to the 1.44 s spiral factorisation.

Ordering
--------
``permc_spec`` selects SuperLU's fill-reducing column ordering. The default
is SuperLU's own COLAMD, because it is the faster of the two candidates
where it matters. Measured in this environment (scipy splu on the assembled
matrices, median of 3 factorisations):

===========================  =======  =========================  =========================
matrix                             N  COLAMD                     MMD_AT_PLUS_A
===========================  =======  =========================  =========================
hplane two-iris, refine 2       3619  7.8 ms, 2.43e5 LU nnz      5.1 ms, 1.46e5 LU nnz
spiral nominal DUT fixture     12739  1.44 s,  1.17e7 LU nnz     13.81 s, 3.80e7 LU nnz
===========================  =======  =========================  =========================

MMD_AT_PLUS_A wins the 2-D 5-point system (1.5x faster, 1.7x less fill) and
loses the 3-D curl-curl one badly (9.6x slower, 3.3x the fill), so COLAMD
stays the default; the 2-D solves are milliseconds and the 3-D ones are the
budget. Pass ``permc_spec="MMD_AT_PLUS_A"`` explicitly for a 2-D sweep, or,
for a solver that does not thread the keyword through (``hplane.solve``,
``yee3d.solve``, ``ports3d.s_matrix``), wrap the call in the scoped default:

    with linear_solve.default_permc_spec("MMD_AT_PLUS_A"):
        s11, s21 = hplane.solve(model, freq)

That override is a ``contextvars.ContextVar``, so it is restored on exit
(including on an exception) and is private to the thread / asyncio task that
set it -- unlike a plain module global, which two concurrent sweeps would
race on. Neither the solution nor a gradient depends on the ordering beyond
roundoff; that is gated in ``tests/test_fdfd_linear_solve.py``.

Precision
---------
The systems this is built for (Helmholtz with discrete DtN ports) have
condition numbers ~1e12; float32 is meaningless. Callers must enable x64
(``jax.config.update("jax_enable_x64", True)`` or the scoped
``enable_x64`` context) -- the solve raises otherwise rather than silently
downcasting.
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
from collections import OrderedDict
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

__all__ = ["sparse_solve", "sparse_matvec", "clear_factor_cache",
           "default_permc_spec", "get_default_permc_spec", "PERMC_SPECS"]

# Column orderings SuperLU accepts. ``None`` means "scipy's own default",
# which is COLAMD -- verified identical here: ``splu(a, permc_spec=None)``
# and ``splu(a, permc_spec="COLAMD")`` return the same ``perm_c``, the same
# ``perm_r`` and the same LU nnz, so the two share one cache entry.
PERMC_SPECS = (None, "NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD")

# Ordering used when the caller passes ``permc_spec=None``; see "Ordering" in
# the module docstring for the factorisation times and fill that chose it,
# and ``default_permc_spec`` to override it for a scope.
_DEFAULT_PERMC_SPEC: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "rfx_fdfd_default_permc_spec", default=None)

# LU factors keyed by the pattern, the entries AND the ordering, so the
# forward solve and the transposed adjoint solve of one gradient evaluation
# share a factorisation while two different problems never collide.
_FACTOR_CACHE: OrderedDict[bytes, spl.SuperLU] = OrderedDict()
_FACTOR_CACHE_SIZE = 4


def get_default_permc_spec() -> str | None:
    """The ordering ``sparse_solve`` uses when the caller does not name one."""
    return _DEFAULT_PERMC_SPEC.get()


@contextlib.contextmanager
def default_permc_spec(spec: str | None) -> Iterator[None]:
    """Make ``spec`` the default SuperLU ordering inside this ``with`` block.

    For callers that reach ``sparse_solve`` through a solver that does not
    forward ``permc_spec`` (``hplane.solve``, ``yee3d.solve``,
    ``ports3d.s_matrix``). The previous default is restored on exit, also if
    the body raises, and the binding is per-thread / per-task
    (``contextvars``), so two concurrent sweeps do not race.
    """
    _check_permc_spec(spec)
    token = _DEFAULT_PERMC_SPEC.set(spec)
    try:
        yield
    finally:
        _DEFAULT_PERMC_SPEC.reset(token)


def clear_factor_cache() -> None:
    _FACTOR_CACHE.clear()


def _check_permc_spec(spec: str | None) -> None:
    if spec not in PERMC_SPECS:
        raise ValueError(f"permc_spec must be one of {PERMC_SPECS}, got {spec!r}")


def _pattern_digest(rows: np.ndarray, cols: np.ndarray, n: int,
                    permc_spec: str | None) -> bytes:
    """Digest of everything about the problem except the entry values.

    ``None`` and ``"COLAMD"`` are folded together because they are the same
    ordering (see :data:`PERMC_SPECS`), so naming it either way hits one
    cache entry instead of burning two LRU slots on one factorisation.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray([n, rows.size], dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(rows).tobytes())
    h.update(np.ascontiguousarray(cols).tobytes())
    h.update(("COLAMD" if permc_spec is None else permc_spec).encode())
    return h.digest()


def _factor(data: np.ndarray, rows: np.ndarray, cols: np.ndarray, n: int,
            permc_spec: str | None, pattern_key: bytes) -> spl.SuperLU:
    h = hashlib.blake2b(pattern_key, digest_size=16)
    h.update(data.tobytes())
    key = h.digest()
    lu = _FACTOR_CACHE.get(key)
    if lu is None:
        a = sp.csc_matrix((data, (rows, cols)), shape=(n, n))
        a.sum_duplicates()
        lu = spl.splu(a, permc_spec=permc_spec)
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


def sparse_solve(data: jax.Array, rows: np.ndarray, cols: np.ndarray, b: jax.Array, *,
                 permc_spec: str | None = None) -> jax.Array:
    """Solve ``A x = b`` for the sparse COO matrix ``(rows, cols, data)``.

    ``rows``/``cols`` are static integer NumPy arrays (the pattern);
    ``data`` and ``b`` are JAX arrays and may be traced, jitted and
    differentiated in either mode. ``b`` is ``(n,)`` for one right-hand side
    or ``(n, m)`` for ``m`` of them; ``x`` has the same shape and the host
    factorises ``A`` once for all columns. Duplicate ``(row, col)`` pairs are
    summed, as in ``scipy.sparse``.

    ``permc_spec`` is the SuperLU column ordering, one of
    :data:`PERMC_SPECS`. ``None`` (the default) means
    :func:`get_default_permc_spec`, itself ``None`` = SuperLU's own COLAMD,
    which the measurements in "Ordering" above say to keep;
    ``"MMD_AT_PLUS_A"``, ``"MMD_ATA"``, ``"NATURAL"`` and an explicit
    ``"COLAMD"`` are the alternatives, and :func:`default_permc_spec`
    overrides the default for a scope. The ordering is part of the LU cache
    key, so the forward and adjoint solve of one gradient share a
    factorisation while two orderings of the same matrix do not collide.
    Neither the solution nor a gradient depends on it, to roundoff. On a
    self-contained 2-D Helmholtz system (N = 3600, 1-norm condition estimate
    1.3e4) the four distinct orderings give fields agreeing to 3.6e-14
    relative, the objective to 5.1e-14 and its two-parameter gradient to
    3.9e-14; on the hplane two-iris system (N = 3619, cond ~1e12) the field
    agrees to 6.0e-13, |S11|^2 to 4.6e-14 and d|S11|^2/dwidth to 2.9e-12.
    Every one of those gradients matches a 4th-order central difference
    (2.9e-8 and 3.0e-7 relative, FD4 truncation dominating). See
    ``tests/test_fdfd_linear_solve.py``.
    """
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "rfx.fdfd.sparse_solve needs x64: the frequency-domain systems it "
            "targets have cond ~1e12. Enable jax_enable_x64 for this scope.")
    _check_permc_spec(permc_spec)
    if permc_spec is None:
        permc_spec = _DEFAULT_PERMC_SPEC.get()
        _check_permc_spec(permc_spec)
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
    # once per call, not once per (forward / adjoint / column-block) callback
    pattern_key = _pattern_digest(rows, cols, n, permc_spec)

    def _host(trans: bool):
        def f(d, rhs):
            d = np.asarray(d, dtype=np.complex128)
            rhs = np.asarray(rhs, dtype=np.complex128)
            lu = _factor(d, rows, cols, n, permc_spec, pattern_key)
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
