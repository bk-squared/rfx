"""GPU sparse-direct factorisations for :mod:`rfx.fdfd.linear_solve`.

Optional module: nothing here is imported unless a caller asks for a GPU
backend (``sparse_solve(..., backend="cudss")`` or
``linear_solve.default_backend("cudss")``), so a CPU-only install never
pays for it and never sees an ImportError. :func:`availability` reports,
per backend, whether it can run here and *why not* when it cannot.

Two backends, both factor-once / solve-many, both exposing the same
``factor(...).solve(rhs, trans=...)`` interface as ``scipy``'s SuperLU so
``linear_solve`` dispatches on a string and nothing else changes:

``"cudss"``
    NVIDIA cuDSS through nvmath-python
    (``nvmath.sparse.advanced.DirectSolver``): reorder + symbolic
    factorisation (``plan``), numeric factorisation (``factorize``), then
    any number of triangular solves (``solve``) against a right-hand-side
    block that is reset in place. CSR, complex128, int32 indices (cuDSS
    requires int32; that caps this path at 2^31 nonzeros, ~4e5 unknowns of
    the 3-D curl-curl operator, far above anything here).

``"cusolver"``
    The documented fallback for images where nvmath-python cannot be
    installed: ``cupyx.scipy.sparse.linalg.spsolve``, i.e. cuSOLVER's
    ``csrlsvqr`` sparse QR. It is NOT a factor-once solver -- cupy's
    wrapper calls the one-shot QR per right-hand-side column and keeps
    nothing -- so a block of ``m`` right-hand sides costs ``m``
    factorisations and an adjoint ``m`` more. It exists so a run is
    possible at all, not because it is competitive. Measured, on the real
    paper-geometry DUT operator (N = 85785, 1.08e6 stored entries, ONE
    right-hand-side column): 20.69 s and a relative residual of 1.505e-08
    against cuDSS's 0.71 s and 8.061e-09 on the same system -- so it DOES
    work at this size, at 29x the time for one column, and a block of
    ``m`` columns costs ``m`` of those with an adjoint costing ``m`` more
    (``validation/fdfd/gpu_scaling.json``, ``paper_geometry.cusolver_probe``).

The transposed solve
--------------------
``linear_solve``'s reverse mode needs ``A^T x = b`` (the adjoint system)
with the SAME factorisation the forward solve used. cuDSS has no
transposed solve: neither ``cudssConfigParam_t`` nor ``cudssPhase_t``
carries a transpose/conjugate flag (checked against the enums shipped in
nvmath-python 1.0.0's ``nvmath.bindings.cudss``: phases are
REORDERING / SYMBOLIC_FACTORIZATION / ANALYSIS / FACTORIZATION /
REFACTORIZATION / SOLVE{,_FWD,_DIAG,_BWD,_FWD_PERM,_BWD_PERM,_REFINEMENT},
and ``DirectSolverOptions`` exposes only ``sparse_system_type`` and
``sparse_system_view``), and ``DirectSolver`` exposes no such keyword. The
conjugate-transpose identity does not help either: ``A^-T b =
conj(A^-H conj(b))`` needs an ``A^H`` solve, which cuDSS does not have for
a GENERAL matrix any more than it has ``A^T``.

So a transposed solve factorises ``A^T`` -- the same values on the
transposed pattern -- as a SECOND cuDSS problem inside the same factor
object, built lazily on first use and cached there. The cost is stated
where it is paid: a *gradient* (one forward + one adjoint solve) costs two
cuDSS factorisations against SuperLU's one, and holds two factors on the
device at once. It is not 2x the gradient's wall time, because the
transposed pattern of the curl-curl operator is the pattern of ``A``
itself (the FDFD operator is structurally symmetric, its values are not),
so the reordering and the symbolic factorisation cost the same as the
forward one; and it is bounded, unlike an iterative adjoint. The numbers
this actually costs at N = 108898 are in ``validation/fdfd/gpu_scaling.json``
under ``gates.G3``.

Device memory
-------------
A 3-D curl-curl LU is large: cuDSS's own estimate on the spiral W/3 DUT
operator (N = 108898, 1.37e6 stored entries) is in
``validation/fdfd/gpu_scaling.json`` (``gates.G3``), and the reverse pass
holds a second one for ``A^T``. Three knobs, all off by default, all read
from the environment so a batch lane can set them without touching code:

``RFX_FDFD_CUDSS_HYBRID=1``
    cuDSS hybrid memory mode: the factor lives in host memory and is
    streamed to the device. The way to run a level whose factor does not
    fit on the device at all; slower, and the amount it slows is measured
    (below) rather than guessed.

``RFX_FDFD_CUDSS_HYBRID_LIMIT=<limit>``
    The device-memory limit for hybrid mode, passed to nvmath-python as
    ``HybridMemoryModeOptions(hybrid_device_memory_limit=...)`` (cuDSS's
    ``CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT``). Accepted forms are
    nvmath's own (:func:`parse_memory_limit` checks them on read, so a typo
    fails before a factorisation, not inside one): ``"40GiB"`` / ``"40 GB"``
    / ``"512MiB"`` (binary or decimal units, ``B`` required), ``"80%"`` of
    the device, a plain byte count, or a fraction in ``[0, 1]`` of the
    device. It bounds ALL of cuDSS's device memory (the factor portion,
    the workspace), not just the factor; it must be at least cuDSS's
    ``hybrid_min_device_memory`` (:func:`plan_estimate` reports it).
    nvmath-python passes it at ``DirectSolver`` construction, earlier than
    the window the cuDSS documentation names, and it binds there --
    :func:`hybrid_record` has the measurement, the provenance key that says
    the constructor is where it came from, and what happens if you try to set
    it again later. It is read only when ``RFX_FDFD_CUDSS_HYBRID=1``; unset, hybrid mode keeps
    cuDSS's internal heuristic, which per the cuDSS documentation "will
    assume that it can use the entire GPU memory and set the device memory
    limit based on the device properties" -- the whole card, including what
    the right-hand side, the CSR copies and cupy's pool already hold. That
    is what the study-P m = 5 attempt ran with (no limit) and it died with
    ``ALLOC_FAILED`` at 47.52 of 47.54 GB in use
    (``validation/vessl/runs/fdfd-gpu-p1c-20260915T161317Z``). The defaults
    are unchanged: hybrid mode stays off, and nothing here is read unless
    it is switched on.

    HOST MEMORY is the hard limit of hybrid mode: cuDSS keeps "the entire
    factors only in the host memory" and "Factors L and U (together with
    all necessary internal arrays) must fit into the host memory" (cuDSS
    documentation, "Hybrid memory mode"). So a factor of ``F`` GB needs
    ``F`` GB of host memory for the PROCESS -- in a container that is the
    cgroup memory limit, not ``/proc/meminfo`` or ``free -g``, which report
    the whole node. nvmath also pins it (``register_cuda_memory=True``,
    ``cudaHostRegister``) by default.

    MEASURED (gate H1 of ``validation/fdfd/invariant_ladder.py``, the
    ``hybrid`` block of ``invariant_ladder.json``, run 369367261289; the
    level-3 spiral DUT operator, N = 466833, an 11.37 GB factor, with a
    6 GiB limit, against the in-memory run of the same level in the same job
    on one RTX A6000): cuDSS's plan drops from 11.37 GB of device memory to
    6.00 GB -- the limit, to 1.9e-09 relative -- and asks for 18.79 GB of
    host memory instead of 0.04 GB; the process's peak resident set grows by
    7.90 GB (13.55 -> 21.45 GB); one factorisation costs 29.5 s against
    23.1 s (1.27x), a triangular solve 1.30 s against 0.58 s (2.26x) and the
    whole three-fixture ``value_and_grad`` 200.7 s against 161.1 s (1.25x);
    L_dut agrees to 9.0e-10 relative and the gradient to 1.9e-09 (as a
    vector) -- against 1.7e-09 between two IN-CORE runs of the same thing in
    the same job, and 5.4e-09 between two in-core runs on different cards,
    so that difference is the solver's own reproducibility rather than the
    memory mode. Gate H1 asks for 1e-9 on both and is recorded FAILING on
    the gradient; the numbers, the controls and the job-to-job spread of the
    comparison itself are in the study.

``RFX_FDFD_CUDSS_FREE_FORWARD=1``
    Free the forward solver when the transposed one is built. In a
    reverse-mode pass every forward solve happens before every adjoint
    solve, so ``A``'s factor is dead once ``A^T``'s exists; freeing it
    halves the peak for a multi-fixture gradient. It costs a
    re-factorisation if the SAME matrix is then solved forward again
    (counted in ``factorizations``), which a single value-and-gradient
    never does.

And one consequence for ``linear_solve.factor_cache_size`` that is the
opposite of SuperLU's: with cuDSS, size 1 is the right setting for a
multi-fixture gradient. The cache exists so the adjoint can re-use the
forward factorisation, and for SuperLU it does -- one ``splu`` serves both
directions, so a three-fixture gradient needs 3 slots or every adjoint
re-factorises. cuDSS has no transposed solve, so the adjoint pays for its
own ``A^T`` factorisation whether or not the forward factor is still
cached: the factorisation COUNT is 6 either way (3 forward + 3 adjoint),
and the only thing a bigger cache changes is how many of those are
resident at the peak. Size 1 keeps one, size 3 keeps up to four. A factor
object whose ``A`` has been evicted and is then asked for a transposed
solve builds ``A^T`` alone and never ``A``, which is why nothing is
wasted.

Threads
-------
``linear_solve`` runs every factor operation on one dedicated thread
because scipy's SuperLU leaks a factor freed off its creating thread. That
rule is kept for the GPU backends and costs nothing: CUDA's primary
context is per process and shared by all threads, cupy's memory pool is
thread-safe, and doing all of plan / factorize / solve / free on ONE
thread means one stream and one allocation order no matter where XLA
decides to run the host callback. ``selftest`` checks cross-thread use
explicitly (same factor object driven from two threads) so that claim is
measured rather than assumed: on the RTX 4090 lane the second thread's
solution agrees to 4.2e-16 relative (1.665e-16 max-abs) -- correct, but
NOT bit-identical, because cuDSS's triangular solve is not
bit-reproducible across calls. cuSOLVER's QR path returns exactly the same
bits. Nothing here depends on bit-reproducibility; the LU cache is keyed on
the INPUT bytes, not on the output.

Host threads
------------
cuDSS reorders and symbolically factorises on the HOST, and does it on one
core unless it is given a threading layer -- nvmath prints "No
multithreading interface library was specified ... The performance of CPU
operations like planning will be significantly lower". :func:`threading_lib`
finds ``libcudss_mtlayer_gomp.so`` inside the installed
``nvidia-cudss-cu12`` wheel and passes it as
``DirectSolverOptions.multithreading_lib``, and :func:`host_nthreads`
passes ``OMP_NUM_THREADS`` (or ``RFX_FDFD_CUDSS_NTHREADS``) as the plan's
``host_nthreads``. ``RFX_FDFD_CUDSS_MTLIB=off`` turns it off,
``RFX_FDFD_CUDSS_MTLIB=<path>`` names another library.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any

import numpy as np

__all__ = ["BACKENDS", "availability", "available", "factor", "selftest",
           "device_memory", "unavailable_reason", "plan_estimate",
           "hybrid_memory", "hybrid_device_memory_limit", "parse_memory_limit",
           "memory_limit_bytes", "hybrid_record", "free_forward",
           "threading_lib", "host_nthreads", "ir_steps"]

# The GPU backends this module implements. ``linear_solve.BACKENDS`` is this
# tuple plus ``"superlu"``.
BACKENDS = ("cudss", "cusolver")

# nvmath-python's DirectSolver needs cuDSS >= this; it raises on its own if
# the installed cuDSS is older. Recorded here only so the probe JSON can say
# what it found.
_MIN_CUDSS = "0.4.0"

_AVAIL: dict[str, dict[str, Any]] = {}
_AVAIL_LOCK = threading.Lock()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def hybrid_memory() -> bool:
    """Is cuDSS hybrid (host-backed) memory mode on? See "Device memory"."""
    return _env_flag("RFX_FDFD_CUDSS_HYBRID")


# nvmath-python's own memory-limit grammar (nvmath/internal/mem_limit.py,
# 1.0.0), mirrored so a bad RFX_FDFD_CUDSS_HYBRID_LIMIT is rejected on read,
# on a machine with no nvmath at all, instead of inside the first factorisation
_LIMIT_PCT = re.compile(r"(?P<value>[+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*%\s*$")
_LIMIT_VAL = re.compile(
    r"(?P<value>[+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(?P<units>[kmg])?"
    r"(?P<binary>(?<=[kmg])i)?b\s*$", re.IGNORECASE)
_LIMIT_NUM = re.compile(r"[+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?$")


def parse_memory_limit(value: str | int | float) -> str | int | float:
    """A device-memory limit in one of nvmath-python's forms, validated.

    Returns what ``HybridMemoryModeOptions(hybrid_device_memory_limit=)``
    accepts: an ``int`` byte count, a ``float`` (a fraction of the device if
    in ``[0, 1]``, else bytes), or the string itself for ``"40GiB"`` /
    ``"40 GB"`` / ``"80%"``. A string holding a plain number becomes that
    number. Raises ``ValueError`` on anything else, and on a negative value
    or a percentage outside ``(0, 100]``.
    """
    if isinstance(value, bool):
        raise ValueError(f"memory limit must be a number or a string, got {value!r}")
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"memory limit must be >= 0, got {value!r}")
        return value
    text = str(value).strip()
    if _LIMIT_NUM.match(text):
        num = float(text)
        if "." in text or "e" in text.lower():
            return num
        return int(num)
    m = _LIMIT_PCT.match(text)
    if m:
        pct = float(m.group("value"))
        if not 0.0 < pct <= 100.0:
            raise ValueError(f"memory limit percentage must be in (0, 100], got {text!r}")
        return text
    if _LIMIT_VAL.match(text):
        return text
    raise ValueError(
        f"memory limit {text!r} is not one of nvmath's forms: a byte count, a fraction "
        "in [0, 1], a value with a unit ('40GiB', '40 GB', '512MiB', '1e9 B') or a "
        "percentage ('80%')")


def memory_limit_bytes(value: str | int | float, total_bytes: int) -> int:
    """``value`` (any :func:`parse_memory_limit` form) in bytes, for a device
    of ``total_bytes`` -- the same arithmetic as nvmath-python's
    ``_get_memory_limit`` (decimal units are powers of 1000, ``iB`` units
    powers of 1024, a float in ``[0, 1]`` and a percentage are fractions of
    the device). Used to record the limit and to check it against a factor
    size before a run; nvmath does its own conversion for the solve."""
    v = parse_memory_limit(value)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v * total_bytes) if v <= 1.0 else int(v)
    m = _LIMIT_PCT.match(v)
    if m:
        return int(float(m.group("value")) / 100.0 * total_bytes)
    m = _LIMIT_VAL.match(v)
    assert m is not None
    num = float(m.group("value"))
    units = (m.group("units") or "").lower()
    base = 1024 if m.group("binary") else 1000
    power = {"": 0, "k": 1, "m": 2, "g": 3}[units]
    return int(num * base ** power)


def hybrid_device_memory_limit() -> str | int | float | None:
    """The hybrid-mode device-memory limit (``RFX_FDFD_CUDSS_HYBRID_LIMIT``),
    validated, or ``None`` (cuDSS's heuristic: the whole device). See
    "Device memory"; only consulted when :func:`hybrid_memory` is on."""
    raw = os.environ.get("RFX_FDFD_CUDSS_HYBRID_LIMIT", "").strip()
    if not raw:
        return None
    return parse_memory_limit(raw)


def free_forward() -> bool:
    """Free ``A``'s factor when ``A^T``'s is built? See "Device memory"."""
    return _env_flag("RFX_FDFD_CUDSS_FREE_FORWARD")


def threading_lib() -> str | None:
    """Path to cuDSS's host threading layer, or ``None`` if it is not there.

    cuDSS does its reordering and symbolic factorisation on the HOST, and
    without this library it does them on one core: nvmath-python prints
    "No multithreading interface library was specified ... The performance
    of CPU operations like planning will be significantly lower". The
    library ships inside the ``nvidia-cudss-cu12`` wheel that
    ``nvmath-python[cu12]`` pulls in, so it is found rather than
    configured; ``CUDSS_THREADING_LIB`` (nvmath's own variable) overrides
    the search and ``RFX_FDFD_CUDSS_MTLIB=off`` disables it.
    """
    override = os.environ.get("RFX_FDFD_CUDSS_MTLIB", "").strip()
    if override.lower() in ("off", "none", "0"):
        return None
    if override and os.path.isfile(override):
        return override
    env = os.environ.get("CUDSS_THREADING_LIB", "")
    if env and os.path.isfile(env):
        return env
    import glob
    import site
    roots: list[str] = list(site.getsitepackages()) if hasattr(site, "getsitepackages") else []
    try:
        import nvidia                                   # type: ignore[import-not-found]
        roots += [os.path.dirname(os.path.dirname(p)) for p in (nvidia.__path__ or [])]
    except Exception:
        pass
    for root in roots:
        hits = sorted(glob.glob(os.path.join(
            root, "nvidia", "**", "libcudss_mtlayer_gomp.so*"), recursive=True))
        if hits:
            os.environ.setdefault("CUDSS_THREADING_LIB", hits[0])
            return hits[0]
    return None


def host_nthreads() -> int | None:
    """Host threads for cuDSS's planning (``RFX_FDFD_CUDSS_NTHREADS``, else
    ``OMP_NUM_THREADS``, else nothing -- cuDSS's own default)."""
    for name in ("RFX_FDFD_CUDSS_NTHREADS", "OMP_NUM_THREADS"):
        v = os.environ.get(name, "").strip()
        if v.isdigit() and int(v) > 0:
            return int(v)
    return None


def ir_steps() -> int | None:
    """cuDSS iterative-refinement steps (``RFX_FDFD_CUDSS_IR``), or ``None``.

    cuDSS pivots STATICALLY (matching + a pivot epsilon), where SuperLU
    pivots partially, so on a badly scaled operator its solution can carry
    a larger backward error than the factorisation's own roundoff.
    Iterative refinement is cuDSS's answer to that and it is almost free
    here (a solve at N = 108898 is 0.08 s against a 0.95 s factorisation).
    ``None`` leaves cuDSS's own default.

    MEASURED on the spiral W/3 DUT operator (N = 108898), sweeping this
    knob against SuperLU's solution of the same system
    (``validation/fdfd/gpu_scaling.json``, ``gates.G1.conditioning``):

    ====  ================  ==========================  =======
    ir    rel. residual     rel. diff from SuperLU      solve
    ====  ================  ==========================  =======
    0     3.858e-09         2.396e-03                   3.78 s
    1     1.310e-09         2.392e-03                   0.85 s
    2     1.315e-09         2.395e-03                   0.85 s
    4     1.317e-09         2.399e-03                   0.86 s
    ====  ================  ==========================  =======

    One step buys 3x on the residual and then it saturates at 1.3e-09 --
    13x below SuperLU's 1.787e-08 on the same system -- and the difference
    between the two solutions does not move at all. So that 2.4e-03 is not
    cuDSS's static pivoting: it is the operator, whose conditioning leaves
    the solution VECTOR undetermined at that level once the residual is at
    its floor. ``L_dut``, the functional the study reports, agrees to
    9.2e-10 between the backends regardless.
    """
    v = os.environ.get("RFX_FDFD_CUDSS_IR", "").strip()
    return int(v) if v.isdigit() else None


def _solver_options() -> Any:
    """``options=`` for ``DirectSolver``: the threading layer, if found."""
    mt = threading_lib()
    if mt is None:
        return None
    from nvmath.sparse.advanced import DirectSolverOptions
    return DirectSolverOptions(multithreading_lib=mt)


def _execution_options() -> Any:
    """``execution=`` for ``DirectSolver``: default (``None``), or hybrid
    memory mode with the device limit of :func:`hybrid_device_memory_limit`
    when one is set (read -- and validated -- before nvmath is imported)."""
    if not hybrid_memory():
        return None
    limit = hybrid_device_memory_limit()
    from nvmath.sparse.advanced import ExecutionCUDA, HybridMemoryModeOptions
    opts: dict[str, Any] = {"hybrid_memory_mode": True}
    if limit is not None:
        opts["hybrid_device_memory_limit"] = limit
    return ExecutionCUDA(hybrid_memory_mode_options=HybridMemoryModeOptions(**opts))


def hybrid_record() -> dict[str, Any]:
    """The hybrid-memory settings in force, for a measurement record.

    Not a setter, and the ONLY place the limit is set is
    :func:`_execution_options`, at ``DirectSolver`` construction:
    ``hybrid_limit_applied`` in the record says exactly that
    (``"constructor"``), so a measurement taken with the limit applied any
    other way cannot be mistaken for this path's.

    Construction is BEFORE the analysis, while the cuDSS documentation says
    the limit "can be set after the analysis phase but must be set before
    the factorization phase". Measured on the constructor path ALONE, it
    binds anyway (gate H1, ``validation/fdfd/invariant_ladder.json``,
    ``hybrid.h1``): with ``RFX_FDFD_CUDSS_HYBRID_LIMIT=6GiB`` at study P's
    level 3, cuDSS's own plan reports 5.99999999 GB of permanent DEVICE
    memory -- the limit, 1.9e-09 relative under it -- against 11.37 GB for
    the same matrix in-core, and 18.79 GB of host memory against 0.04 GB.

    Re-setting it in cuDSS's own window is NOT done here. The only route is
    nvmath's private config interface (``solver._internal_config`` /
    ``cudssConfigSet``), and it behaves differently on the two threads this
    module runs on: on the CALLING thread (``plan_estimate``) it succeeds and
    reads back, but on ``linear_solve``'s dedicated factor thread -- where
    every real solve happens -- it raises ``AttributeError: '_thread._local'
    object has no attribute 'size_written'`` from inside nvmath 1.0.0. An
    earlier revision of this module did both (set at construction AND again
    after ``plan()``), which is why the plan records of run
    ``fdfd-gpu-p3-20260915T233751Z`` carry ``hybrid_limit_set_after_plan``
    and cannot say which of the two setters the plan's numbers came from.
    They are not the numbers quoted anywhere here: those come from run
    369367261289 (``fdfd-gpu-p3-20260916T004810Z``), whose plan records carry
    ``hybrid_limit_applied = "constructor"`` and no second setter, and which
    reproduced the same 5.999999988824129 GB / 18.78843168914318 GB
    byte for byte with the constructor as the only setter. The same
    nvmath thread-local is why the LU stats below carry no
    ``memory_estimates``: those come from :func:`plan_estimate`, which runs
    on the calling thread.
    """
    out: dict[str, Any] = {"hybrid_memory": hybrid_memory()}
    if not out["hybrid_memory"]:
        return out
    limit = hybrid_device_memory_limit()
    out["hybrid_device_memory_limit"] = limit
    if limit is None:
        # cuDSS's own heuristic: "the entire GPU memory" (see "Device memory")
        out["hybrid_limit_applied"] = "none"
        return out
    # PROVENANCE, written into every artifact: the limit reached cuDSS at
    # DirectSolver construction and nowhere else
    out["hybrid_limit_applied"] = "constructor"
    total = int(device_memory()["total"] * 2.0 ** 30)
    if total > 0:                                       # pragma: no cover - no GPU here
        out["hybrid_device_memory_limit_gb"] = memory_limit_bytes(limit, total) / 2.0 ** 30
    return out


def _make_solver(nvs: Any, a: Any, b: Any) -> Any:
    """``DirectSolver`` with this process's options, planning threads set.

    Every keyword is optional and every one of them is behind its own
    try/except: an nvmath that does not accept one must not take the solve
    with it.
    """
    kwargs: dict[str, Any] = {}
    opts = _solver_options()
    if opts is not None:
        kwargs["options"] = opts
    execution = _execution_options()
    if execution is not None:
        kwargs["execution"] = execution
    try:
        solver = nvs.DirectSolver(a, b, **kwargs)
    except TypeError:                                   # pragma: no cover - no GPU here
        if execution is not None:
            # hybrid mode was ASKED for: dropping it silently would run the
            # factor in device memory and fail (or succeed) for the wrong reason
            raise
        solver = nvs.DirectSolver(a, b)
    n = host_nthreads()
    if n is not None:
        try:
            solver.plan_config.host_nthreads = n
        except Exception:                               # pragma: no cover - no GPU here
            pass
    ir = ir_steps()
    if ir is not None:
        try:
            solver.solution_config.ir_num_steps = ir
        except Exception:                               # pragma: no cover - no GPU here
            pass
    return solver


def _probe_cudss() -> dict[str, Any]:
    out: dict[str, Any] = {"backend": "cudss", "available": False, "reason": None}
    try:
        import nvmath                                  # noqa: F401
        from nvmath.sparse.advanced import DirectSolver  # noqa: F401
    except Exception as exc:                            # pragma: no cover - no GPU here
        out["reason"] = f"nvmath-python unavailable: {type(exc).__name__}: {exc}"
        return out
    out["nvmath_version"] = getattr(nvmath, "__version__", "?")
    try:
        import cupy
        out["cupy_version"] = cupy.__version__
        out["device_count"] = int(cupy.cuda.runtime.getDeviceCount())
        if out["device_count"] < 1:
            out["reason"] = "no CUDA device"
            return out
        out["device_name"] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
        out["device_arrays"] = True
    except Exception as exc:                            # pragma: no cover - no GPU here
        # nvmath can take scipy/numpy operands and do the transfers itself;
        # that path is slower per solve but correct, so cupy is not required.
        out["cupy_version"] = None
        out["device_arrays"] = False
        out["cupy_reason"] = f"{type(exc).__name__}: {exc}"
        try:
            import ctypes
            ctypes.CDLL("libcuda.so.1")
        except Exception as exc2:                       # pragma: no cover - no GPU here
            out["reason"] = f"no CUDA driver: {type(exc2).__name__}: {exc2}"
            return out
    out["available"] = True
    out["min_cudss"] = _MIN_CUDSS
    out["threading_lib"] = threading_lib()
    out["host_nthreads"] = host_nthreads()
    out["ir_steps"] = ir_steps()
    return out


def _probe_cusolver() -> dict[str, Any]:
    out: dict[str, Any] = {"backend": "cusolver", "available": False, "reason": None}
    try:
        import cupy
        import cupyx.scipy.sparse as csp                # noqa: F401
        from cupyx.scipy.sparse.linalg import spsolve   # noqa: F401
    except Exception as exc:                            # pragma: no cover - no GPU here
        out["reason"] = f"cupy unavailable: {type(exc).__name__}: {exc}"
        return out
    out["cupy_version"] = cupy.__version__
    try:
        out["device_count"] = int(cupy.cuda.runtime.getDeviceCount())
        out["device_name"] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception as exc:                            # pragma: no cover - no GPU here
        out["reason"] = f"no CUDA device: {type(exc).__name__}: {exc}"
        return out
    if out["device_count"] < 1:
        out["reason"] = "no CUDA device"
        return out
    out["available"] = True
    return out


def availability(backend: str | None = None, refresh: bool = False) -> dict[str, Any]:
    """What this machine can run, and the reason it cannot when it cannot.

    Cached (the import and the device query are not free); ``refresh=True``
    re-probes. With ``backend`` given, the record for that one backend;
    without, a dict of every GPU backend's record. Every record has
    ``available`` (bool) and, when false, ``reason`` (str) -- the string the
    probe JSON and the test skip messages print, so an unavailable backend
    says *why* instead of vanishing.
    """
    with _AVAIL_LOCK:
        if refresh:
            _AVAIL.clear()
        if not _AVAIL:
            _AVAIL["cudss"] = _probe_cudss()
            _AVAIL["cusolver"] = _probe_cusolver()
        if backend is None:
            return dict(_AVAIL)
        if backend not in _AVAIL:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        return _AVAIL[backend]


def available(backend: str) -> bool:
    """``True`` if ``backend`` can factorise on this machine right now."""
    return bool(availability(backend)["available"])


def unavailable_reason(backend: str) -> str:
    """Why ``backend`` cannot run here (empty string if it can)."""
    rec = availability(backend)
    return "" if rec["available"] else str(rec.get("reason") or "unknown")


def device_memory() -> dict[str, float]:
    """Device memory in GB: what the driver reports and what cupy's pool holds.

    ``used`` is ``total - free`` from ``cudaMemGetInfo``, which counts cuDSS's
    own allocations and the context, not just cupy's pool; ``pool_used`` and
    ``pool_total`` are cupy's mempool. Zeros if no GPU is visible, so callers
    can record it unconditionally.
    """
    out = {"used": 0.0, "free": 0.0, "total": 0.0, "pool_used": 0.0, "pool_total": 0.0}
    try:                                                # pragma: no cover - no GPU here
        import cupy
        free, total = cupy.cuda.runtime.memGetInfo()
        gb = 1.0 / 2.0 ** 30
        out["free"] = free * gb
        out["total"] = total * gb
        out["used"] = (total - free) * gb
        pool = cupy.get_default_memory_pool()
        out["pool_used"] = pool.used_bytes() * gb
        out["pool_total"] = pool.total_bytes() * gb
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# cuDSS
# ---------------------------------------------------------------------------

class CudssFactor:
    """One cuDSS problem: ``A`` on the device, factorised, solved many times.

    Mirrors the slice of ``scipy.sparse.linalg.SuperLU`` that
    ``linear_solve`` uses -- ``solve(rhs, trans="N" | "T")`` taking and
    returning host arrays of shape ``(n,)`` or ``(n, m)`` -- so the caller
    does not know which backend it holds. ``trans="T"`` factorises ``A^T``
    separately and caches it here; see "The transposed solve" in the module
    docstring for why and what it costs.

    One solver object is kept per ``trans``; if the width ``m`` of the
    right-hand-side block changes, that solver is freed and re-planned
    (``replans`` counts it). Within one gradient evaluation the forward and
    the adjoint solve have the same ``m`` by construction, so that does not
    happen on the FDFD path.
    """

    def __init__(self, data: np.ndarray, rows: np.ndarray, cols: np.ndarray, n: int):
        import nvmath.sparse.advanced as nvs
        self._nvs = nvs
        self._n = int(n)
        self._data = np.ascontiguousarray(data, dtype=np.complex128)
        nnz = self._data.size
        if nnz > 2 ** 31 - 1:
            raise ValueError(
                f"cuDSS uses int32 indices; this matrix has {nnz} nonzeros")
        self._rows = np.ascontiguousarray(rows, dtype=np.int32)
        self._cols = np.ascontiguousarray(cols, dtype=np.int32)
        try:
            import cupy
            self._cp: Any = cupy
        except Exception:                               # pragma: no cover - no GPU here
            self._cp = None
        self._mat: dict[bool, Any] = {}
        self._solvers: dict[bool, tuple[int, Any, Any]] = {}
        self.factorizations = 0
        self.replans = 0
        self.solves = 0
        self.freed_forward = 0
        self.stats: dict[str, Any] = {}

    # -- matrix -------------------------------------------------------------
    def _matrix(self, trans: bool) -> Any:
        """CSR of ``A`` (or ``A^T``) with duplicate entries summed."""
        mat = self._mat.get(trans)
        if mat is None:
            r, c = (self._cols, self._rows) if trans else (self._rows, self._cols)
            n = self._n
            if self._cp is not None:
                import cupyx.scipy.sparse as csp
                coo = csp.coo_matrix(
                    (self._cp.asarray(self._data),
                     (self._cp.asarray(r), self._cp.asarray(c))), shape=(n, n))
                mat = coo.tocsr()                       # tocsr() sums duplicates
                mat.sum_duplicates()
                # cuDSS wants int32 indices; cupy already uses int32 here, but
                # do not rely on it.
                mat.indices = mat.indices.astype(self._cp.int32, copy=False)
                mat.indptr = mat.indptr.astype(self._cp.int32, copy=False)
            else:                                       # pragma: no cover - no GPU here
                import scipy.sparse as sp
                coo = sp.coo_matrix((self._data, (r, c)), shape=(n, n))
                mat = coo.tocsr()
                mat.sum_duplicates()
                mat.indices = mat.indices.astype(np.int32, copy=False)
                mat.indptr = mat.indptr.astype(np.int32, copy=False)
            self._mat[trans] = mat
        return mat

    # -- solver -------------------------------------------------------------
    def _rhs_buffer(self, shape: tuple[int, ...]) -> Any:
        """A zeroed right-hand-side block in the layout cuDSS requires
        (column-major for a matrix; a vector is both)."""
        if self._cp is not None:
            return self._cp.zeros(shape, dtype=self._cp.complex128, order="F")
        return np.zeros(shape, dtype=np.complex128, order="F")   # pragma: no cover

    def _solver(self, trans: bool, shape: tuple[int, ...]) -> tuple[int, Any, Any]:
        m = 1 if len(shape) == 1 else int(shape[1])
        cur = self._solvers.get(trans)
        if cur is not None and cur[0] == m and cur[2].shape == shape:
            return cur
        if cur is not None:                             # width changed: re-plan
            self.replans += 1
            try:
                cur[1].free()
            except Exception:
                pass
            self._solvers.pop(trans, None)
        b = self._rhs_buffer(shape)
        b[...] = 1.0                                    # never factorise against 0
        solver = _make_solver(self._nvs, self._matrix(trans), b)
        plan_info = solver.plan()
        fac_info = solver.factorize()
        self.factorizations += 1
        self._record_stats(trans, plan_info, fac_info)
        self._solvers[trans] = (m, solver, b)
        if trans and free_forward():
            # a reverse pass is forward-then-adjoint: A's factor is dead now
            old_ = self._solvers.pop(False, None)
            if old_ is not None:
                try:
                    old_[1].free()
                except Exception:                       # pragma: no cover
                    pass
                self._mat.pop(False, None)
                self.freed_forward += 1
        return self._solvers[trans]

    def _record_stats(self, trans: bool, plan_info: Any, fac_info: Any) -> None:
        """cuDSS's own account of the factor: LU nonzeros and device bytes.

        Every field is optional -- the attributes come straight from
        ``cudssDataParam_t`` and a future version may drop or rename one --
        so a failure here must not fail a solve.
        """
        tag = "AT" if trans else "A"
        rec: dict[str, Any] = {}
        for name, obj in (("plan", plan_info), ("factorization", fac_info)):
            try:
                est = obj.memory_estimates
                rec[f"{name}_permanent_device_gb"] = \
                    float(est["permanent_device_memory"]) / 2.0 ** 30
                rec[f"{name}_peak_device_gb"] = \
                    float(est["peak_device_memory"]) / 2.0 ** 30
            except Exception:
                pass
        try:
            est = plan_info.memory_estimates
            for k in ("permanent_host_memory", "peak_host_memory",
                      "hybrid_min_device_memory", "hybrid_max_device_memory"):
                rec[f"plan_{k}_gb"] = float(est[k]) / 2.0 ** 30
        except Exception:
            pass
        for attr in ("lu_nnz", "npivots"):
            try:
                rec[attr] = int(getattr(fac_info, attr))
            except Exception:
                pass
        rec.update(hybrid_record())
        rec["device_memory_gb"] = device_memory()["used"]
        self.stats[tag] = rec

    # -- the SuperLU-shaped entry point -------------------------------------
    def solve(self, rhs: np.ndarray, trans: str = "N") -> np.ndarray:
        if trans not in ("N", "T"):
            raise ValueError(f"trans must be 'N' or 'T', got {trans!r}")
        rhs = np.asarray(rhs, dtype=np.complex128)
        shape = tuple(int(s) for s in rhs.shape)
        if shape[0] != self._n:
            raise ValueError(f"rhs has {shape[0]} rows, matrix has {self._n}")
        _, solver, b = self._solver(trans == "T", shape)
        if self._cp is not None:
            b[...] = self._cp.asarray(np.asfortranarray(rhs))
        else:                                           # pragma: no cover - no GPU here
            solver.reset_operands(b=np.asfortranarray(rhs))
        x = solver.solve()
        self.solves += 1
        if self._cp is not None:
            x = self._cp.asnumpy(x)
        return np.ascontiguousarray(np.asarray(x, dtype=np.complex128))

    def free(self) -> None:
        for _, solver, _ in self._solvers.values():
            try:
                solver.free()
            except Exception:
                pass
        self._solvers.clear()
        self._mat.clear()

    def __del__(self) -> None:                          # pragma: no cover
        try:
            self.free()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# cuSOLVER QR (the fallback)
# ---------------------------------------------------------------------------

class CusolverFactor:
    """``cupyx.scipy.sparse.linalg.spsolve`` behind the same interface.

    NOT a factorisation: cuSOLVER's ``csrlsvqr`` is a one-shot sparse QR
    solve and cupy's wrapper runs it once per right-hand-side column, so
    this object caches the device matrix (and its transpose) and nothing
    else. ``factorizations`` counts the QR solves it has paid for, which is
    the honest count for a cost table: ``m`` per ``(n, m)`` block.
    """

    def __init__(self, data: np.ndarray, rows: np.ndarray, cols: np.ndarray, n: int):
        import cupy
        import cupyx.scipy.sparse as csp
        from cupyx.scipy.sparse.linalg import spsolve
        self._cp = cupy
        self._csp = csp
        self._spsolve = spsolve
        self._n = int(n)
        self._data = np.ascontiguousarray(data, dtype=np.complex128)
        self._rows = np.ascontiguousarray(rows, dtype=np.int32)
        self._cols = np.ascontiguousarray(cols, dtype=np.int32)
        self._mat: dict[bool, Any] = {}
        self.factorizations = 0
        self.replans = 0
        self.solves = 0
        self.stats: dict[str, Any] = {}

    def _matrix(self, trans: bool) -> Any:
        mat = self._mat.get(trans)
        if mat is None:
            r, c = (self._cols, self._rows) if trans else (self._rows, self._cols)
            coo = self._csp.coo_matrix(
                (self._cp.asarray(self._data),
                 (self._cp.asarray(r), self._cp.asarray(c))),
                shape=(self._n, self._n))
            mat = coo.tocsr()
            mat.sum_duplicates()
            self._mat[trans] = mat
            self.stats["AT" if trans else "A"] = {
                "nnz": int(mat.nnz), "device_memory_gb": device_memory()["used"]}
        return mat

    def solve(self, rhs: np.ndarray, trans: str = "N") -> np.ndarray:
        if trans not in ("N", "T"):
            raise ValueError(f"trans must be 'N' or 'T', got {trans!r}")
        rhs = np.asarray(rhs, dtype=np.complex128)
        a = self._matrix(trans == "T")
        b = self._cp.asarray(rhs)
        if b.ndim == 1:
            x = self._spsolve(a, b)
            self.factorizations += 1
        else:
            cols = []
            for j in range(b.shape[1]):
                cols.append(self._spsolve(a, self._cp.ascontiguousarray(b[:, j])))
                self.factorizations += 1
            x = self._cp.stack(cols, axis=1)
        self.solves += 1
        return np.ascontiguousarray(self._cp.asnumpy(x).astype(np.complex128, copy=False))

    def free(self) -> None:
        self._mat.clear()


def factor(data: np.ndarray, rows: np.ndarray, cols: np.ndarray, n: int,
           backend: str) -> Any:
    """Factorise ``A = (rows, cols, data)`` on the GPU with ``backend``.

    Returns an object with SuperLU's ``solve(rhs, trans=...)``. Called only
    from ``linear_solve._factor``, i.e. on the one factor thread, and the
    result is memoised in that module's LRU exactly like a SuperLU factor.
    """
    if backend == "cudss":
        return CudssFactor(data, rows, cols, n)
    if backend == "cusolver":
        return CusolverFactor(data, rows, cols, n)
    raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")


def plan_estimate(data: np.ndarray, rows: np.ndarray, cols: np.ndarray, n: int,
                  m: int = 1) -> dict[str, Any]:
    """cuDSS's symbolic pass only: what factorising this matrix would cost.

    Reorders and symbolically factorises (``plan()``) WITHOUT the numeric
    factorisation, then reports ``memory_estimates`` -- permanent and peak
    device bytes -- and frees everything. That is how a lane decides,
    cheaply and before committing two hours, whether a level fits in 24 GB
    or needs ``RFX_FDFD_CUDSS_HYBRID=1`` (and, then, how much host memory:
    in hybrid mode the ``*_host_memory_gb`` fields are cuDSS's estimate of
    what the process must be allowed to hold).
    """
    import time
    f = CudssFactor(data, rows, cols, n)
    # ``hybrid_record()`` carries the PROVENANCE of the limit
    # (``hybrid_limit_applied``: "constructor", i.e. :func:`_execution_options`
    # and nothing else), so a plan record can be attributed to a code path
    out: dict[str, Any] = {"n": int(n), "nnz": int(np.asarray(data).size),
                           "hybrid_device_memory_limit": None, **hybrid_record()}
    try:
        b = f._rhs_buffer((n, m) if m > 1 else (n,))
        solver = _make_solver(f._nvs, f._matrix(False), b)
        t0 = time.time()
        info = solver.plan()
        out["plan_seconds"] = time.time() - t0
        try:
            est = info.memory_estimates
            for k in ("permanent_device_memory", "peak_device_memory",
                      "permanent_host_memory", "peak_host_memory",
                      "hybrid_min_device_memory", "hybrid_max_device_memory"):
                out[k + "_gb"] = float(est[k]) / 2.0 ** 30
        except Exception as exc:
            out["memory_estimates_error"] = f"{type(exc).__name__}: {exc}"
        out["device_memory_gb"] = device_memory()
        solver.free()
    except Exception as exc:                            # pragma: no cover - no GPU here
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        f.free()
    return out


def selftest(backend: str = "cudss", n: int = 400, m: int = 3,
             seed: int = 0) -> dict[str, Any]:
    """Solve a random complex system with ``backend`` and with SuperLU.

    Self-contained (no rfx model, no JAX): a diagonally dominant complex
    ``n x n`` sparse matrix and an ``(n, m)`` right-hand-side block, solved
    forward and transposed by both. Returns the residuals and the two
    solutions' difference, plus the factor object's own counters -- the
    numbers the J0 probe prints before anything expensive runs, and the
    cross-thread check (the same factor object driven from a second thread)
    that backs the "Threads" claim in the module docstring.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    rng = np.random.default_rng(seed)
    a = sp.random(n, n, density=min(1.0, 8.0 / n), format="coo", random_state=seed)
    a = (a + sp.eye(n, format="coo") * 4.0).tocoo()
    data = a.data + 0.3j * rng.standard_normal(a.data.size)
    rhs = (rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m)))
    acsc = sp.csc_matrix((data, (a.row, a.col)), shape=(n, n))
    acsc.sum_duplicates()
    lu = spl.splu(acsc)
    out: dict[str, Any] = {"backend": backend, "n": n, "m": m}
    f = factor(data, a.row, a.col, n, backend)
    for tag, tr in (("N", "N"), ("T", "T")):
        x_gpu = f.solve(rhs, trans=tr)
        x_cpu = lu.solve(rhs, trans=tr)
        mat = acsc.T if tr == "T" else acsc
        nb = float(np.linalg.norm(rhs))
        out[f"residual_{tag}_gpu"] = float(np.linalg.norm(mat @ x_gpu - rhs)) / nb
        out[f"residual_{tag}_superlu"] = float(np.linalg.norm(mat @ x_cpu - rhs)) / nb
        out[f"diff_{tag}"] = float(np.max(np.abs(x_gpu - x_cpu))
                                   / max(np.max(np.abs(x_cpu)), 1e-300))
    # cross-thread: the same factor object used from another thread must give
    # the same answer (CUDA's primary context is per process, not per thread)
    box: dict[str, Any] = {}

    def other() -> None:
        try:
            box["x"] = f.solve(rhs, trans="N")
        except Exception as exc:                        # pragma: no cover - no GPU here
            box["error"] = f"{type(exc).__name__}: {exc}"

    t = threading.Thread(target=other)
    t.start()
    t.join()
    if "x" in box:
        x0 = f.solve(rhs, trans="N")
        scale = max(float(np.max(np.abs(x0))), 1e-300)
        out["cross_thread_diff"] = float(np.max(np.abs(box["x"] - x0)))
        out["cross_thread_rel"] = out["cross_thread_diff"] / scale
        # MEASURED (RTX 4090, nvmath 1.0.0, cuDSS): the same factor object
        # solved from a second thread agrees to 1.665e-16 max-abs, i.e.
        # 4.2e-16 relative -- NOT bit-identical, which cuSOLVER's QR path is
        # (exactly 0.0). cuDSS's triangular solve is not bit-reproducible
        # across calls, so the gate is roundoff, not equality; 1e-12 is four
        # orders above what was measured and still catches a wrong answer
        # (a cross-context failure would be O(1), not O(1e-16)).
        out["cross_thread_ok"] = out["cross_thread_rel"] <= 1e-12
    else:                                               # pragma: no cover - no GPU here
        out["cross_thread_ok"] = False
        out["cross_thread_error"] = box.get("error")
    out["factorizations"] = int(f.factorizations)
    out["replans"] = int(f.replans)
    out["freed_forward"] = int(getattr(f, "freed_forward", 0))
    out["hybrid_memory"] = hybrid_memory()
    out["hybrid_device_memory_limit"] = hybrid_device_memory_limit() if hybrid_memory() else None
    out["free_forward"] = free_forward()
    out["threading_lib"] = threading_lib()
    out["host_nthreads"] = host_nthreads()
    out["solves"] = int(f.solves)
    out["stats"] = f.stats
    out["device_memory_gb"] = device_memory()
    f.free()
    return out
