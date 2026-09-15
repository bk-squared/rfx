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
sparsity pattern ``(n, rows, cols)``, the entry bytes, the backend and
(for SuperLU) the column ordering. That is what lets the forward solve and
the transposed adjoint solve of one gradient evaluation share a single
factorisation. The pattern
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
roundoff; that is gated in ``tests/unit/fdfd/test_fdfd_linear_solve.py``.

Memory
------
Every factorisation, every triangular solve and every drop of a cached
factor runs on ONE dedicated thread (``_on_factor_thread``). That is not an
optimisation, it is a correctness requirement: ``scipy.sparse.linalg.SuperLU``
(1.18.1) does not return its factor to the allocator when the object is
deallocated on a thread other than the one that built it. Measured with no
JAX in the loop, on the N = 12739 spiral DUT operator, one 1.17e7-nonzero
factor, malloc's own live-byte counter after each of four build/destroy
rounds:

===============  ===============  ==============================
built on         destroyed on     malloc live bytes, per factor
===============  ===============  ==============================
worker thread    worker thread    0.0000 GB
main thread      main thread      0.0000 GB
worker thread    MAIN thread      0.2537 GB (the whole factor)
main thread      WORKER thread    0.2537 GB (the whole factor)
===============  ===============  ==============================

Under ``jit`` XLA runs ``pure_callback`` on its own worker pool, so the
factors were built there while ``clear_factor_cache()`` destroyed them on
the caller's thread and every one was lost. Whether that shows at all
depends on where XLA decides to put the callback: it keeps small programs
on the caller's thread and only moves to its pool above some size
(measured on a self-contained 3-D Laplacian: caller's thread at 13824
unknowns, pool thread at 17576), so a single jitted ``sparse_solve`` at
N = 12739 stayed on the caller's thread and never grew. The three-fixture
spiral solve is the case that did: it lost 2.54 of its three factors per
solve. At N = 12739, four jitted three-fixture solves at the SAME theta
with the cache emptied and ``gc.collect()`` forced between them, RSS after
each: 1.319 / 1.962 / 2.608 / 3.250 GB, +0.644 GB per solve, unbounded.
With the factor thread, six solves: 1.317 / 1.347 / 1.347 / 1.347 / 1.347 /
1.347 GB, +0.000042 GB per solve, i.e. 0.016 % of the first solve's
1.021 GB increment. ``jax.value_and_grad`` of the same behaves identically
(+0.63 GB per solve before, +0.000092 after).

At N = 33352 -- the design study's fixture, LU 5.54e7 nonzeros, 1.277 GB per
factor -- the old path lost 3.24 GB per solve; four solves now measure
5.22 / 5.33 / 5.08 / 5.02 GB, so a 16-solve design loop peaks at 5.33 GB
where the old one extrapolates to ~54 GB. That reconciles the two
contradictory claims in ``validation/fdfd/spiral_design.py``: its
"+3.6 GB per solve (5.6 / 9.2 / 12.8 / 16.4 GB after 1-4 solves)" was a
correct measurement of the JITTED path, and its "~0.5 GB per further solve"
was not the same experiment -- the eager path never leaked at all (a
six-solve eager loop is flat to 0.0002 GB per solve).

Nothing about the solution moves: the four N = 33352 solves above are
bit-identical to each other and to a one-solve-per-process control, and
match the design study's independently recorded nominal ``L_diff``
(5.090106167675779e-10 H, an eager solve written before this fix existed)
to 3.97e-12 relative.

The price is that the three fixtures' factorisations no longer overlap. A
jitted three-fixture ``value_and_grad`` at N = 12739 goes from 3.13 s to
4.42 s (x1.41); a forward solve is unchanged (4.61 -> 4.42 s, inside the
contention noise of this shared machine). Peak memory is now bounded by
``factor_cache_size()`` factors of 23.3 (N = 12739) to 24.7 (N = 33352)
bytes per LU nonzero, measured. The whole series is in
``validation/fdfd/memory_probe.json``; ``validation/fdfd/memory_probe.py``
re-measures it, block by block.

Backends
--------
``backend`` selects the sparse direct solver that actually factorises.
``"superlu"`` (the default, and the only one that needs nothing installed)
is scipy's SuperLU on the host, described above. ``"cudss"`` is NVIDIA
cuDSS through nvmath-python and ``"cusolver"`` is cuSOLVER's sparse QR
through cupy; both live in :mod:`rfx.fdfd._cudss`, which is imported only
when one of them is asked for, so a CPU-only install never touches them.
:func:`default_backend` is the scoped override (same mechanism as
:func:`default_permc_spec`) and the environment variable
``RFX_FDFD_BACKEND`` sets the process default at import, which is how a
batch job forces a whole pytest run or study onto the GPU without editing
anything.

The cache, the factor thread and the AD rule are the same for all three:
one factor object per ``(pattern, entries, backend)``, everything on the
one factor thread, ``custom_linear_solve`` on top. Two differences are
real and are the GPU backends' price:

* cuDSS has no transposed solve, so the adjoint system ``A^T lam = dL/dx``
  factorises ``A^T`` as a second problem (cached inside the same factor
  object, built on first use). A gradient therefore costs TWO cuDSS
  factorisations and holds two factors in device memory, against SuperLU's
  one of each. See "The transposed solve" in ``rfx/fdfd/_cudss.py``.
* ``"cusolver"`` is not a factor-once backend at all (cuSOLVER
  ``csrlsvqr`` is a one-shot sparse QR, and cupy's ``spsolve`` runs it per
  right-hand-side column), so it is the fallback for an image where
  nvmath-python cannot be installed, not a competitor.

What it buys, and what it does not. Measured on the spiral W/3 DUT
operator (N = 108898, 1.37e6 stored entries, the finest level of study
D2), the SAME ``(data, rows, cols, b)`` with one ``(n, 2)``
right-hand-side block, on an RTX 4090 against this repo's own SuperLU
path (validation/vessl/runs/fdfd-gpu-j0-*, gate G1):

==========  ==================  =============  ===============  ===========
backend     factor + solve      repeat solve   rel. residual    device
==========  ==================  =============  ===============  ===========
superlu       139.23 s            0.54 s       1.787e-08        (host)
cudss           0.95 s            0.08 s       3.841e-09        2.04 GB
==========  ==================  =============  ===============  ===========

147x on factor + solve (160x on the factorisation alone, 138.70 s against
0.867 s by difference), and the residuals say the rest: NEITHER is at
1e-16. This operator has cond ~1e12 and a row scaling to match, so a
backward-stable solve of it leaves ~1e-8, and the two solution VECTORS
differ by 2.395e-03 relative -- which is what two solves of such a system
at that residual are allowed to differ by, not a defect in either. That
is measured, not asserted: cuDSS's iterative refinement (``ir_num_steps``,
see :func:`rfx.fdfd._cudss.ir_steps`) takes its residual to 1.310e-09,
13x below SuperLU's, in 0.85 s -- and the difference from SuperLU's
solution stays at 2.392e-03. What DOES agree is the quantity the study
reports: ``L_dut`` to 9.2e-10 and its three-parameter gradient to 2.3e-08
relative between the two backends at this level. cuDSS's
own estimate of the factor is 1.38 GB at W/3, 2.43 GB at W/4 (N = 172747)
and 5.68 GB at W/6 (N = 352536), and its reordering (host, threaded
through the layer :func:`rfx.fdfd._cudss.threading_lib` finds) takes 0.3,
0.4 and 0.8 s -- and those two levels are what the backend is FOR: a
three-fixture ``value_and_grad`` of the spiral study's ``L_dut`` costs
432.9 s at W/3 on the lane's host SuperLU (304 s on the Mac that produced
the CPU study) against 6.9 s on cuDSS, and then 20.9 s at W/4
(N = 172747, peak 5.75 GB of device memory) and 38.1 s at W/6
(N = 352536, 12.72 GB) -- levels no host SuperLU in this project has
reached. What that difference does to the quantity a study reports
-- ``L_dut``, four port readings off that vector -- and what cuDSS's
iterative refinement changes, is in ``validation/fdfd/gpu_scaling.json``
(``gates.G1.conditioning``).

``permc_spec`` is SuperLU's column ordering and has no meaning for the GPU
backends: cuDSS does its own (nested-dissection) reordering. It is
therefore IGNORED there, and left out of their cache key so that a scoped
``default_permc_spec`` cannot silently burn two LRU slots on one cuDSS
factorisation.

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
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

__all__ = ["sparse_solve", "sparse_matvec", "clear_factor_cache",
           "factor_cache_size", "default_permc_spec", "get_default_permc_spec",
           "PERMC_SPECS", "default_backend", "get_default_backend", "BACKENDS",
           "backend_available"]

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

# Sparse direct solvers ``sparse_solve`` can dispatch to. ``"superlu"`` is
# scipy's, on the host, and needs nothing; ``"cudss"`` and ``"cusolver"`` are
# the GPU backends in :mod:`rfx.fdfd._cudss` and are imported only when named.
BACKENDS = ("superlu", "cudss", "cusolver")

# Process default backend, overridable for a scope by ``default_backend`` and
# for a process by the environment (``RFX_FDFD_BACKEND=cudss``), which is how
# a batch job forces a whole test run or study onto the GPU. An unknown value
# in the environment is a hard error here rather than a silent CPU run.
DEFAULT_BACKEND = os.environ.get("RFX_FDFD_BACKEND", "superlu").strip() or "superlu"
if DEFAULT_BACKEND not in BACKENDS:
    raise ValueError(
        f"RFX_FDFD_BACKEND must be one of {BACKENDS}, got {DEFAULT_BACKEND!r}")
_DEFAULT_BACKEND: contextvars.ContextVar[str] = contextvars.ContextVar(
    "rfx_fdfd_default_backend", default=DEFAULT_BACKEND)

# LU factors keyed by the pattern, the entries AND the ordering, so the
# forward solve and the transposed adjoint solve of one gradient evaluation
# share a factorisation while two different problems never collide.
_FACTOR_CACHE: OrderedDict[bytes, Any] = OrderedDict()
_FACTOR_CACHE_SIZE = 4

# ---------------------------------------------------------------------------
# The factor thread. See "Memory" in the module docstring: a
# ``scipy.sparse.linalg.SuperLU`` built on one thread and deallocated on
# another does not give its factor back to the allocator, and under ``jit``
# XLA runs the host callbacks on its own worker pool while
# ``clear_factor_cache()`` runs on the caller's thread. Every splu, every
# triangular solve and every drop of a cached factor therefore happens on
# ONE dedicated thread, so create and destroy always pair up.
_FACTOR_EXECUTOR: ThreadPoolExecutor | None = None
_FACTOR_TID: int | None = None
_FACTOR_EXECUTOR_LOCK = threading.Lock()
_T = TypeVar("_T")


def _on_factor_thread(fn: Callable[..., _T], *args: Any) -> _T:
    """Run ``fn(*args)`` on the one thread that owns every LU factor.

    Re-entrant (a call already on that thread runs inline, so a callback
    that reaches ``sparse_solve`` again cannot deadlock). The thread is
    created on first use and is a daemon-free pool worker, joined by
    ``concurrent.futures``' own atexit hook.
    """
    global _FACTOR_EXECUTOR, _FACTOR_TID
    if threading.get_ident() == _FACTOR_TID:
        return fn(*args)
    ex = _FACTOR_EXECUTOR
    if ex is None:
        with _FACTOR_EXECUTOR_LOCK:
            ex = _FACTOR_EXECUTOR
            if ex is None:
                ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rfx-fdfd-lu")
                _FACTOR_TID = ex.submit(threading.get_ident).result()
                _FACTOR_EXECUTOR = ex
    return ex.submit(fn, *args).result()


def _reset_factor_thread() -> None:
    """A forked child inherits the cache but not the thread that owns it."""
    global _FACTOR_EXECUTOR, _FACTOR_TID
    _FACTOR_EXECUTOR = None
    _FACTOR_TID = None
    _FACTOR_CACHE.clear()


if hasattr(os, "register_at_fork"):               # posix only
    os.register_at_fork(after_in_child=_reset_factor_thread)


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


def get_default_backend() -> str:
    """The backend ``sparse_solve`` uses when the caller does not name one."""
    return _DEFAULT_BACKEND.get()


@contextlib.contextmanager
def default_backend(backend: str) -> Iterator[None]:
    """Make ``backend`` the default sparse direct solver inside this block.

    The hook for the solvers that do not forward the keyword
    (``hplane.solve``, ``yee3d.solve``, ``ports3d.s_matrix``,
    ``spiral.solve_spiral``) and for a study that wants one line changed,
    not twenty. Same mechanism as :func:`default_permc_spec`: a
    ``ContextVar``, restored on exit including on an exception, private to
    the thread / task that set it. The name is validated here, so a typo
    raises at the ``with`` rather than at the first solve.
    """
    _check_backend(backend)
    token = _DEFAULT_BACKEND.set(backend)
    try:
        yield
    finally:
        _DEFAULT_BACKEND.reset(token)


def backend_available(backend: str) -> bool:
    """Can ``backend`` factorise in this process? ``"superlu"`` always can.

    For the GPU backends this imports :mod:`rfx.fdfd._cudss` and asks it
    (nvmath-python / cupy importable, a CUDA device visible);
    ``rfx.fdfd._cudss.unavailable_reason`` says why not. Tests use it to
    skip rather than fail on a machine without a GPU.
    """
    _check_backend(backend)
    if backend == "superlu":
        return True
    from rfx.fdfd import _cudss
    return _cudss.available(backend)


def _free_factor(lu: Any) -> None:
    """Give a dropped factor's memory back now, not at the next GC.

    SuperLU needs only to be dereferenced (on this thread -- see "Memory"),
    but a GPU factor holds device memory that ``__del__`` would return at
    an unpredictable time, and device memory is the scarce resource (24 GB
    on the run lane against 128 GB of host RAM).
    """
    free = getattr(lu, "free", None)
    if free is not None:
        try:
            free()
        except Exception:                       # pragma: no cover - defensive
            pass


def _trim_factor_cache() -> None:
    while len(_FACTOR_CACHE) > _FACTOR_CACHE_SIZE:
        _, lu = _FACTOR_CACHE.popitem(last=False)
        _free_factor(lu)


def _clear_factor_cache() -> None:
    for lu in list(_FACTOR_CACHE.values()):
        _free_factor(lu)
    _FACTOR_CACHE.clear()


def clear_factor_cache() -> None:
    """Drop every memoised LU factor.

    The drop is performed on the factor thread, which is the thread that
    built them: a ``SuperLU`` deallocated anywhere else keeps its whole
    factor (measured: 0.2537 GB for one 1.17e7-nonzero factor, i.e. all of
    it) for the life of the process. See "Memory" in the module docstring.

    It is not needed for correctness and not needed to bound the process:
    the cache is already capped at :func:`factor_cache_size` factors. Call
    it to give back the last solve's factors before doing something else
    large.
    """
    if _FACTOR_EXECUTOR is None:
        _clear_factor_cache()               # nothing was ever built off-thread
    else:
        _on_factor_thread(_clear_factor_cache)


def factor_cache_size(size: int | None = None) -> int:
    """Get, or (``size`` given) set, the number of LU factors kept.

    The cache is what lets the forward solve and the transposed adjoint
    solve of one gradient evaluation share a single factorisation, so
    ``size`` must be at least 1 for a gradient to cost one factorisation
    instead of two; ``0`` disables memoisation entirely. The default is 4.

    It is the knob for PEAK memory, not for the growth (there is none left;
    see "Memory" in the module docstring). A factor costs 23.3 bytes per LU
    nonzero measured -- 0.2538 GB for the 1.17e7-nonzero LU of the
    N = 12739 spiral fixture, 1.277 GB for the 5.54e7-nonzero LU of the
    N = 33352 one (24.7 bytes/nonzero) -- so the default bounds a process at
    4 x that: 1.02 GB and 5.11 GB of factors respectively. A three-fixture
    solve needs 3 live at once and a three-fixture *gradient* re-uses
    exactly those 3, so 3 is the smallest size that keeps the adjoint pass
    free of a second factorisation; below it every adjoint re-factorises.
    Measured at N = 12739, six jitted three-fixture solves at a MOVING theta
    and no ``clear_factor_cache()``: the plateau is 1.773 GB at size 4 and
    1.518 GB at size 3, a difference of 0.255 GB = one factor. Shrinking
    takes effect immediately (the excess is dropped on the factor thread).

    For the GPU backends the trade-off is the OPPOSITE way round and 1 is
    the right size for a multi-fixture gradient: cuDSS has no transposed
    solve, so the adjoint factorises ``A^T`` whether or not the forward
    factor is still cached, and a bigger cache buys no factorisation -- it
    only keeps more factors resident on a device with 24 GB. See "Device
    memory" in ``rfx/fdfd/_cudss.py``; measured at W/6 (N = 352536), one
    slot holds the peak at 12.72 GB.
    """
    global _FACTOR_CACHE_SIZE
    if size is None:
        return _FACTOR_CACHE_SIZE
    n = int(size)
    if n < 0:
        raise ValueError(f"factor_cache_size must be >= 0, got {size!r}")
    _FACTOR_CACHE_SIZE = n
    if _FACTOR_EXECUTOR is None:
        _trim_factor_cache()
    else:
        _on_factor_thread(_trim_factor_cache)
    return _FACTOR_CACHE_SIZE


def _check_permc_spec(spec: str | None) -> None:
    if spec not in PERMC_SPECS:
        raise ValueError(f"permc_spec must be one of {PERMC_SPECS}, got {spec!r}")


def _check_backend(backend: str | None) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")


def _pattern_digest(rows: np.ndarray, cols: np.ndarray, n: int,
                    permc_spec: str | None, backend: str = "superlu") -> bytes:
    """Digest of everything about the problem except the entry values.

    ``None`` and ``"COLAMD"`` are folded together because they are the same
    ordering (see :data:`PERMC_SPECS`), so naming it either way hits one
    cache entry instead of burning two LRU slots on one factorisation. The
    ordering is in the key only for ``"superlu"``: the GPU backends ignore
    it (cuDSS reorders itself), so keying on it there would split one
    factorisation across two slots for nothing. The backend itself IS in the
    key -- two backends' factors are not interchangeable objects, and a
    study that compares them holds both.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray([n, rows.size], dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(rows).tobytes())
    h.update(np.ascontiguousarray(cols).tobytes())
    h.update(backend.encode())
    if backend == "superlu":
        h.update(("COLAMD" if permc_spec is None else permc_spec).encode())
    return h.digest()


def _factor(data: np.ndarray, rows: np.ndarray, cols: np.ndarray, n: int,
            permc_spec: str | None, pattern_key: bytes,
            backend: str = "superlu") -> Any:
    """The memoised factorisation. Runs on the factor thread, always."""
    h = hashlib.blake2b(pattern_key, digest_size=16)
    h.update(data.tobytes())
    key = h.digest()
    lu = _FACTOR_CACHE.get(key)
    if lu is None:
        if backend == "superlu":
            a = sp.csc_matrix((data, (rows, cols)), shape=(n, n))
            a.sum_duplicates()
            lu = spl.splu(a, permc_spec=permc_spec)
        else:
            from rfx.fdfd import _cudss
            lu = _cudss.factor(data, rows, cols, n, backend)
        _FACTOR_CACHE[key] = lu
        _trim_factor_cache()
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
                 permc_spec: str | None = None,
                 backend: str | None = None) -> jax.Array:
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
    ``tests/unit/fdfd/test_fdfd_linear_solve.py``.

    ``backend`` names the sparse direct solver, one of :data:`BACKENDS`.
    ``None`` (the default) means :func:`get_default_backend`, itself
    ``"superlu"`` unless the environment (``RFX_FDFD_BACKEND``) or a
    :func:`default_backend` scope says otherwise -- so the resolved default
    is ``"superlu"`` and this argument is backward compatible. ``"cudss"``
    (NVIDIA cuDSS via nvmath-python) and ``"cusolver"`` (cuSOLVER sparse QR
    via cupy) factorise on the GPU; see "Backends" above for what they cost
    and what ``permc_spec`` means there (nothing). The backends are
    equal to their own roundoff, not to each other's bits: on a random
    complex 400 x 400 system with 3 right-hand sides (``_cudss.selftest``
    on the RTX 4090 lane) the relative residuals are 3.74e-16 (cuDSS),
    1.22e-15 (cuSOLVER) and 6.62e-16 (SuperLU) untransposed, 3.87e-16 /
    1.25e-15 / 6.94e-16 transposed, and the solutions differ from SuperLU's
    by 1.40e-15 and 2.75e-15. The same comparison on the real
    N = 108898 spiral W/3 fixture, and the ``value_and_grad`` equality, are
    in ``validation/fdfd/gpu_scaling.json`` under ``gates.G1``.
    """
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "rfx.fdfd.sparse_solve needs x64: the frequency-domain systems it "
            "targets have cond ~1e12. Enable jax_enable_x64 for this scope.")
    _check_permc_spec(permc_spec)
    if permc_spec is None:
        permc_spec = _DEFAULT_PERMC_SPEC.get()
        _check_permc_spec(permc_spec)
    if backend is None:
        backend = _DEFAULT_BACKEND.get()
    _check_backend(backend)
    if backend != "superlu" and not backend_available(backend):
        # fail here, at trace time, with the reason -- not inside a
        # ``pure_callback`` on a worker thread with an ImportError traceback
        from rfx.fdfd import _cudss
        raise RuntimeError(
            f"sparse_solve(backend={backend!r}) is not available in this "
            f"process: {_cudss.unavailable_reason(backend)}")
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
    pattern_key = _pattern_digest(rows, cols, n, permc_spec, backend)

    def _host(trans: bool):
        tr = "T" if trans else "N"

        def on_thread(d, rhs):
            # factorise, solve and drop the local reference all on the ONE
            # thread that owns the factors (see ``_on_factor_thread``). The
            # GPU backends are on that thread too: one CUDA context, one
            # stream, one allocation order, wherever XLA runs the callback.
            lu = _factor(d, rows, cols, n, permc_spec, pattern_key, backend)
            return lu.solve(rhs, trans=tr)

        def f(d, rhs):
            d = np.asarray(d, dtype=np.complex128)
            rhs = np.asarray(rhs, dtype=np.complex128)
            return _on_factor_thread(on_thread, d, rhs)
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
