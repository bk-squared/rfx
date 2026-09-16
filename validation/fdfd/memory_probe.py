"""Per-solve memory growth of ``rfx.fdfd.linear_solve.sparse_solve``: what
grows, why, and what it costs after the fix.

The design-loop study (``validation/fdfd/spiral_design.py``) reported that
repeated three-fixture spiral solves at N = 33352 grow the process without
bound -- 5.6 / 9.2 / 12.8 / 16.4 GB of resident set after 1-4 solves at the
SAME theta, with ``clear_factor_cache()`` and ``gc.collect()`` between them
-- while a comment in the same file put the growth at ~0.5 GB per further
solve. This script settles that, finds the mechanism and measures the fix.

WHAT IT IS (block ``cross_thread``, no JAX involved)
----------------------------------------------------
``scipy.sparse.linalg.SuperLU`` does not return its factor to the allocator
when the object is deallocated on a thread other than the one that built it.
Four pure-scipy loops over the same N = 12739 matrix, ``malloc_zone_statistics``
live bytes after each, one factor of 1.17e7 LU nonzeros:

=============================  ==========================================
build / destroy                 malloc live bytes per iteration
=============================  ==========================================
worker thread / worker thread   flat
main thread / main thread       flat
worker thread / MAIN thread     +0.2537 GB  (the whole factor)
main thread / worker thread     +0.2537 GB  (the whole factor)
=============================  ==========================================

Under ``jax.jit`` XLA runs ``pure_callback`` on its own worker pool, so the
factors were built there and ``clear_factor_cache()`` -- called by the design
loop from the main thread -- destroyed them on the wrong thread. Every one
was lost. That is why the growth was invisible to every Python-level check:
``gc`` sees no live ``SuperLU``, ``jax.live_arrays()`` is empty, and the
bytes are gone from malloc's free lists as well as from Python.

Whether it shows at all depends on where XLA puts the callback: it keeps
small programs on the caller's thread and only moves to its pool above some
size (measured on a self-contained 3-D Laplacian: caller's thread at 13824
unknowns, pool thread at 17576), so a single jitted ``sparse_solve`` at
N = 12739 stayed on the caller's thread and never grew, and so did every
eager call. The three-fixture spiral solve is the case that did grow: XLA
spreads its three callbacks over its worker threads, and 2.54 of the three
factors are lost per solve.

THE FIX (``rfx/fdfd/linear_solve.py``)
--------------------------------------
Every factorisation, every triangular solve and every drop of a cached
factor now runs on ONE dedicated thread (``_on_factor_thread``), so create
and destroy always pair up. Measured here at N = 12739, jitted three-fixture
solves at fixed theta with the cache emptied and ``gc.collect()`` forced
between them, RSS after each solve:

* before (block ``pre_fix``, 4 solves): 1.319 / 1.962 / 2.608 / 3.250 GB,
  +0.644 GB per solve, unbounded -- 2.54 of the three factors lost per solve
* after (block ``c_jit``, 6 solves): 1.317 / 1.347 / 1.347 / 1.347 / 1.347 /
  1.347 GB, +0.000042 GB per solve, 0.016 % of the first solve's 1.021 GB

The cost is that the three fixtures' factorisations no longer overlap: a
jitted three-fixture ``value_and_grad`` goes from 3.13 s to 4.42 s (x1.41)
at N = 12739, while a forward solve is unchanged (4.61 -> 4.42 s, inside
this shared machine's contention noise). At N = 33352 the factor is 1.277 GB
and the old path lost 3.24 GB per solve; four solves now measure 5.22 /
5.33 / 5.08 / 5.02 GB, so a 16-solve design loop peaks at 5.33 GB against
the ~54 GB the old path extrapolates to. Every number above is in
``memory_probe.json`` (``gates`` and ``conclusion``).

Nothing about the solution moves: the four N = 33352 solves are bit-identical
to each other and to the one-solve-per-process control (block ``subprocess``),
and match the design study's own nominal ``L_diff`` -- an eager solve recorded
in ``spiral_design.json`` before this fix existed -- to 3.97e-12 relative
(gates ``n33k_value_bit_identical``, ``isolated_process_same_value``,
``n33k_matches_study_nominal``).

It also settles the study's two contradictory claims: "+3.6 GB per solve
(5.6 / 9.2 / 12.8 / 16.4 GB after 1-4 solves)" was a correct measurement of
the JITTED path -- this run gets +3.24 GB per solve for it -- and
"~0.5 GB per further solve" was not the same experiment: the study's own
``memory_probe`` block calls ``sm.solve_spiral`` EAGERLY, and the eager path
never leaked (block ``b_eager``: flat to 0.0002 GB per solve).

Blocks (each runs in its own subprocess, so the RSS series are clean and the
run is resumable under this machine's ~530 s watchdog on background jobs):

``factors``       LU nonzeros and measured bytes per nonzero, N = 12739 and 33352
``a_scipy``       (a) pure scipy control: splu + solve + del, 6 times
``b_eager``       (b) eager ``sparse_solve``, 6 identical and 6 perturbed
``c_jit``         (c) jitted three-fixture ``solve_spiral``, 6 fixed and 6 perturbed
``d_grad``        (d) jitted ``value_and_grad`` of the same, 6 times
``cross_thread``  the diagnosis above, pure scipy
``pre_fix``       (c) and (d) with the pre-fix release path, for the delta
``subprocess``    one solve per process, three processes (allocator control)
``no_clear``      what bounds the peak with no ``clear_factor_cache()``: the LRU
``n33k``          four jitted three-fixture solves of the W/1 design fixture

Run::

    .venv/bin/python validation/fdfd/memory_probe.py --from-json
    .venv/bin/python validation/fdfd/memory_probe.py --block c_jit   # one block
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
JSON_PATH = HERE / "memory_probe.json"

# The small fixture everything but ``n33k`` uses: ``spiral.SpiralSpec()``
# defaults, base_dx = W, 16 x 18 x 13 cells, N = 12739 unknowns.
FREQ = 2.4e9
SIGMA_CU = 3e7
SIGMA_SI = 10.0
N_SOLVES = 6
N_SOLVES_PRE_FIX = 4          # the pre-fix path leaks 0.64 GB/solve; 4 keeps it under 4 GB


# ---------------------------------------------------------------------------
# measurement primitives

def rss_gb() -> float:
    """Resident set size in GiB (psutil, the metric the report asked for)."""
    import psutil
    return psutil.Process().memory_info().rss / 2.0 ** 30


class _MallocStats(ctypes.Structure):
    _fields_ = [("blocks_in_use", ctypes.c_uint), ("size_in_use", ctypes.c_size_t),
                ("max_size_in_use", ctypes.c_size_t), ("size_allocated", ctypes.c_size_t)]


def _libc():
    try:
        return ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    except OSError:                              # pragma: no cover - this machine is darwin
        return None


def malloc_live_gb() -> float:
    """malloc's own LIVE bytes (``malloc_zone_statistics``, all zones), GiB.

    This is the number that separates a leak from allocator retention: RSS
    counts pages the allocator holds after ``free``, this counts only blocks
    that have not been freed. NaN off darwin.
    """
    lib = _libc()
    if lib is None or sys.platform != "darwin":  # pragma: no cover
        return float("nan")
    s = _MallocStats()
    lib.malloc_zone_statistics.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MallocStats)]
    lib.malloc_zone_statistics(None, ctypes.byref(s))
    return s.size_in_use / 2.0 ** 30


def series_stats(rss: list[float], base: float) -> dict[str, Any]:
    """First-solve increment, later growth, and the ratio the gate uses.

    ``first_increment`` is solve 1 (it also pays for the jit compilation and
    the first factorisation); ``growth_per_solve`` is the mean increment over
    solves 3..n, i.e. after the second, which is what "does this process grow
    without bound" means.
    """
    first = rss[0] - base
    later = (rss[-1] - rss[1]) if len(rss) > 2 else float("nan")
    per = later / (len(rss) - 2) if len(rss) > 2 else float("nan")
    return {"rss_gb": rss, "rss_gb_base": base,
            "first_increment_gb": first, "growth_after_second_gb": later,
            "growth_gb_per_solve": per,
            "growth_fraction_of_first": (later / first) if first > 0 else float("nan")}


# ---------------------------------------------------------------------------
# fixtures

def _x64() -> None:
    import jax
    jax.config.update("jax_enable_x64", True)


def small_model():
    from rfx.fdfd import spiral as sm
    return sm.build_spiral(sm.SpiralSpec())


def w1_model():
    """The design study's W/1 fixture, N = 33352 (``spiral_design.build_level``)."""
    import importlib.util
    path = HERE / "spiral_design.py"
    spec = importlib.util.spec_from_file_location("_spiral_design_for_probe", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_level(1.0)


def capture_matrix(model) -> dict[str, Any]:
    """One three-fixture solve, keeping the COO entries of each fixture.

    The blocks that need a matrix (the scipy control and the eager solve)
    must use the SAME operator the jitted path factorises, not a lookalike.
    """
    import jax
    import numpy as np

    from rfx.fdfd import linear_solve as ls
    from rfx.fdfd import spiral as sm
    # keyed by the ENTRY bytes, not by ``key``: ``key`` here is the pattern
    # digest and the three fixtures share one pattern
    got: dict[bytes, tuple] = {}
    orig = ls._factor

    def spy(data, rows, cols, n, permc, key):
        lu = orig(data, rows, cols, n, permc, key)
        tag = data.tobytes()
        if tag not in got:
            got[tag] = (np.array(data), rows, cols, n, int(lu.nnz))
        return lu

    ls._factor = spy
    try:
        t0 = time.time()
        res = sm.solve_spiral(model, FREQ, sigma_metal=SIGMA_CU, sigma_si=SIGMA_SI)
        jax.block_until_ready(res.L_diff)
        sec = time.time() - t0
    finally:
        ls._factor = orig
    ls.clear_factor_cache()
    mats = list(got.values())
    return {"mats": mats, "seconds": sec, "n": int(mats[0][3]),
            "nnz": int(mats[0][0].size), "lu_nnz": [int(m[4]) for m in mats]}


# ---------------------------------------------------------------------------
# blocks

def block_factors() -> dict[str, Any]:
    """Factor size, measured: malloc live bytes across one ``splu``.

    The bytes-per-LU-nonzero this returns is what ``factor_cache_size``'s
    docstring quotes and what the N = 33352 peak is computed from.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    _x64()
    out: dict[str, Any] = {}
    for tag, build in (("N12739", small_model), ("N33352", w1_model)):
        model = build()
        cap = capture_matrix(model)
        d, r, c, n, _ = cap["mats"][0]           # the DUT fixture
        a = sp.csc_matrix((d, (r, c)), shape=(n, n))
        a.sum_duplicates()
        gc.collect()
        m0, s0 = malloc_live_gb(), rss_gb()
        t0 = time.time()
        lu = spl.splu(a)
        sec = time.time() - t0
        m1, s1 = malloc_live_gb(), rss_gb()
        nnz = int(lu.nnz)
        del lu, a
        gc.collect()
        out[tag] = {"n": n, "coo_nnz": int(d.size), "lu_nnz": nnz,
                    "lu_nnz_all_fixtures": cap["lu_nnz"],
                    "splu_seconds": sec,
                    "factor_gb_malloc": m1 - m0, "factor_gb_rss": s1 - s0,
                    "bytes_per_lu_nnz": (m1 - m0) * 2.0 ** 30 / nnz,
                    "three_fixture_solve_seconds": cap["seconds"],
                    "shape": list(model.shape)}
        del model, cap
        gc.collect()
    return out


def block_a_scipy() -> dict[str, Any]:
    """(a) the control: splu + solve + del, same thread, no JAX."""
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    _x64()
    cap = capture_matrix(small_model())
    d, r, c, n, _ = cap["mats"][0]
    a = sp.csc_matrix((d, (r, c)), shape=(n, n))
    a.sum_duplicates()
    b = np.ones((n, 2), dtype=np.complex128)
    gc.collect()
    base, mbase = rss_gb(), malloc_live_gb()
    rss, mal, secs = [], [], []
    for _ in range(N_SOLVES):
        t0 = time.time()
        lu = spl.splu(a)
        x = lu.solve(b)
        del lu, x
        gc.collect()
        secs.append(time.time() - t0)
        rss.append(rss_gb())
        mal.append(malloc_live_gb())
    out = series_stats(rss, base)
    out.update({"malloc_live_gb": mal, "malloc_live_gb_base": mbase,
                "seconds": secs, "n": n})
    return out


def block_b_eager() -> dict[str, Any]:
    """(b) eager ``sparse_solve``, identical data (same key) and perturbed."""
    import jax
    import jax.numpy as jnp

    from rfx.fdfd import linear_solve as ls
    _x64()
    cap = capture_matrix(small_model())
    d, r, c, n, _ = cap["mats"][0]
    dj = jnp.asarray(d)
    b = jnp.ones((n, 2), dtype=jnp.complex128)
    out: dict[str, Any] = {"n": n}
    for tag, perturb in (("identical", False), ("perturbed", True)):
        gc.collect()
        base, mbase = rss_gb(), malloc_live_gb()
        rss, mal, secs = [], [], []
        for i in range(N_SOLVES):
            t0 = time.time()
            dd = dj * (1.0 + 1e-3 * i) if perturb else dj
            x = ls.sparse_solve(dd, r, c, b)
            jax.block_until_ready(x)
            del x, dd
            ls.clear_factor_cache()
            gc.collect()
            secs.append(time.time() - t0)
            rss.append(rss_gb())
            mal.append(malloc_live_gb())
        s = series_stats(rss, base)
        s.update({"malloc_live_gb": mal, "malloc_live_gb_base": mbase, "seconds": secs})
        out[tag] = s
    return out


def _jit_loop(model, mode: str, perturb: bool, n_solves: int, inline: bool,
              clear: bool = True, cache_size: int | None = None) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from rfx.fdfd import linear_solve as ls
    from rfx.fdfd import spiral as sm
    if inline:
        # the pre-fix path exactly: factor, solve and release all happen on
        # whatever thread XLA hands the callback to
        ls._on_factor_thread = lambda fn, *a: fn(*a)
    if cache_size is not None:
        ls.factor_cache_size(cache_size)
    th0 = jnp.asarray(model.theta_nominal)

    def metric(th):
        return jnp.real(sm.solve_spiral(model, FREQ, theta=th,
                                        sigma_metal=SIGMA_CU, sigma_si=SIGMA_SI).L_diff)

    fn = jax.jit(jax.value_and_grad(metric)) if mode == "grad" else jax.jit(metric)
    gc.collect()
    base, mbase = rss_gb(), malloc_live_gb()
    rss, mal, secs, vals = [], [], [], []
    for i in range(n_solves):
        th = th0 * (1.0 + 1e-3 * i) if perturb else th0
        t0 = time.time()
        v = fn(th)
        jax.block_until_ready(v)
        secs.append(time.time() - t0)
        vals.append(float(v[0]) if mode == "grad" else float(v))
        del v, th
        if clear:
            ls.clear_factor_cache()
        gc.collect()
        rss.append(rss_gb())
        mal.append(malloc_live_gb())
    out = series_stats(rss, base)
    out.update({"malloc_live_gb": mal, "malloc_live_gb_base": mbase,
                "seconds": secs, "value": vals, "mode": mode,
                "perturbed": bool(perturb), "inline_release": bool(inline),
                "clear_between": bool(clear),
                "factor_cache_size": ls.factor_cache_size(),
                "n": int(model.n_unknowns)})
    return out


def block_c_jit() -> dict[str, Any]:
    """(c) the jitted three-fixture solve, fixed theta and perturbed theta."""
    _x64()
    model = small_model()
    return {"fixed_theta": _jit_loop(model, "forward", False, N_SOLVES, False),
            "perturbed_theta": _jit_loop(model, "forward", True, N_SOLVES, False)}


def block_d_grad() -> dict[str, Any]:
    """(d) ``jax.value_and_grad`` of the same: one forward pass plus the
    adjoint, which re-uses the forward pass's three factorisations."""
    _x64()
    model = small_model()
    return {"fixed_theta": _jit_loop(model, "grad", False, N_SOLVES, False)}


def block_pre_fix() -> dict[str, Any]:
    """(c) and (d) with the pre-fix release path, for the delta.

    ``_on_factor_thread`` is replaced by a direct call, which is line for
    line what the module did before: the host callback factorises and solves
    on whichever XLA worker thread it lands on, and ``clear_factor_cache()``
    destroys the factors on the caller's thread.
    """
    _x64()
    model = small_model()
    return {"forward_fixed_theta": _jit_loop(model, "forward", False, N_SOLVES_PRE_FIX, True),
            "grad_fixed_theta": _jit_loop(model, "grad", False, N_SOLVES_PRE_FIX, True)}


def block_n33k() -> dict[str, Any]:
    """The design fixture itself: four jitted three-fixture solves at
    N = 33352, fixed theta, after the fix."""
    _x64()
    model = w1_model()
    return {"fixed_theta": _jit_loop(model, "forward", False, 4, False)}


def block_no_clear() -> dict[str, Any]:
    """What bounds the peak when the caller does NOT empty the cache.

    Six jitted three-fixture solves at a MOVING theta (three new factors
    each) and no ``clear_factor_cache()``: the LRU is the only bound, so the
    plateau is ``factor_cache_size()`` factors. Repeated with the knob set
    to 3 -- the smallest size that still lets a three-fixture adjoint re-use
    the forward pass's factorisations -- to show the peak move by one
    factor.
    """
    _x64()
    model = small_model()
    return {"cache4": _jit_loop(model, "forward", True, N_SOLVES, False,
                                clear=False, cache_size=4),
            "cache3": _jit_loop(model, "forward", True, N_SOLVES, False,
                                clear=False, cache_size=3)}


def block_cross_thread() -> dict[str, Any]:
    """THE diagnosis, with no JAX in it: a ``SuperLU`` built on one thread and
    destroyed on another keeps its whole factor."""
    import scipy
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    _x64()
    cap = capture_matrix(small_model())
    d, r, c, n, _ = cap["mats"][0]

    def build(eps: float):
        a = sp.csc_matrix((d * (1.0 + eps), (r, c)), shape=(n, n))
        a.sum_duplicates()
        return spl.splu(a)

    held: dict[str, Any] = {}
    out: dict[str, Any] = {"n": n, "scipy_version": scipy.__version__,
                           "lu_nnz": int(cap["lu_nnz"][0])}
    reps = 4

    def record(tag: str, step: Callable[[int], None]) -> None:
        gc.collect()
        mal = [malloc_live_gb()]
        rss = [rss_gb()]
        for i in range(reps):
            step(i)
            gc.collect()
            mal.append(malloc_live_gb())
            rss.append(rss_gb())
        out[tag] = {"malloc_live_gb": mal, "rss_gb": rss,
                    "malloc_growth_gb_per_factor": (mal[-1] - mal[1]) / (reps - 1)}

    with ThreadPoolExecutor(max_workers=1) as ex:
        def worker_worker(i: int) -> None:
            ex.submit(lambda: (build(1e-3 * i), None)[1]).result()

        def worker_main(i: int) -> None:
            held["lu"] = ex.submit(build, 1e-3 * i).result()
            held.clear()

        def main_main(i: int) -> None:
            held["lu"] = build(1e-3 * i)
            held.clear()

        def main_worker(i: int) -> None:
            held["lu"] = build(1e-3 * i)
            ex.submit(held.clear).result()

        record("build_worker_destroy_worker", worker_worker)
        record("build_worker_destroy_main", worker_main)
        record("build_main_destroy_main", main_main)
        record("build_main_destroy_worker", main_worker)
    return out


_ONE_SOLVE = """
import gc, json, sys, time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import psutil
from rfx.fdfd import spiral as sm
p = psutil.Process()
t0 = time.time()
m = sm.build_spiral(sm.SpiralSpec())
f = jax.jit(lambda th: jnp.real(sm.solve_spiral(m, 2.4e9, theta=th,
            sigma_metal=3e7, sigma_si=10.0).L_diff))
v = f(jnp.asarray(m.theta_nominal)); jax.block_until_ready(v)
print("RESULT " + json.dumps({"rss_gb": p.memory_info().rss / 2.0 ** 30,
      "value": float(v), "seconds": time.time() - t0}))
"""


def block_subprocess() -> dict[str, Any]:
    """The allocator control: one solve per process, three processes.

    If the growth were macOS holding freed blocks, an isolated solve would
    cost the same as the first solve of a repeated run; it does, and the
    repeated run (after the fix) now costs the same too.
    """
    runs = []
    for _ in range(3):
        t0 = time.time()
        p = subprocess.run([sys.executable, "-c", _ONE_SOLVE], cwd=str(REPO),
                           capture_output=True, text=True, timeout=600)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
        if not line:                             # pragma: no cover - diagnostics
            raise RuntimeError(f"subprocess solve failed: {p.stdout[-2000:]}\n{p.stderr[-2000:]}")
        rec = json.loads(line[0][len("RESULT "):])
        rec["wall_seconds"] = time.time() - t0
        runs.append(rec)
    return {"runs": runs, "rss_gb": [r["rss_gb"] for r in runs],
            "rss_gb_mean": sum(r["rss_gb"] for r in runs) / len(runs)}


BLOCKS: dict[str, Callable[[], dict[str, Any]]] = {
    "factors": block_factors,
    "a_scipy": block_a_scipy,
    "b_eager": block_b_eager,
    "c_jit": block_c_jit,
    "d_grad": block_d_grad,
    "pre_fix": block_pre_fix,
    "cross_thread": block_cross_thread,
    "subprocess": block_subprocess,
    "no_clear": block_no_clear,
    "n33k": block_n33k,
}


# ---------------------------------------------------------------------------
# gates and assembly

def gates(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """The numeric claims this study makes, each with the number it is made
    from. Thresholds are stated with the measurement that set them."""
    g: list[dict[str, Any]] = []

    def add(name: str, measured: float, threshold: float, passed: bool,
            units: str, what: str) -> None:
        g.append({"name": name, "measured": measured, "threshold": threshold,
                  "passed": bool(passed), "units": units, "what": what})

    c = doc.get("c_jit", {}).get("fixed_theta")
    if c:
        frac = c["growth_fraction_of_first"]
        add("jit_solve_growth_bounded", frac, 0.15, frac < 0.15, "fraction",
            "RSS growth over solves 2..6 of the jitted three-fixture solve, "
            "as a fraction of solve 1's increment (the task's gate; measured "
            f"{c['growth_after_second_gb']:.4f} GB on {c['first_increment_gb']:.3f} GB)")
    d = doc.get("d_grad", {}).get("fixed_theta")
    if d:
        frac = d["growth_fraction_of_first"]
        add("grad_growth_bounded", frac, 0.15, frac < 0.15, "fraction",
            "the same for value_and_grad")
    p = doc.get("pre_fix", {}).get("forward_fixed_theta")
    if p and c:
        ratio = p["growth_gb_per_solve"] / max(c["growth_gb_per_solve"], 1e-9)
        add("fix_beats_pre_fix", ratio, 10.0, ratio > 10.0, "x",
            "pre-fix growth per solve divided by post-fix growth per solve "
            f"({p['growth_gb_per_solve']:.4f} vs {c['growth_gb_per_solve']:.4f} GB)")
    ct = doc.get("cross_thread")
    if ct:
        same = max(ct["build_worker_destroy_worker"]["malloc_growth_gb_per_factor"],
                   ct["build_main_destroy_main"]["malloc_growth_gb_per_factor"])
        cross = min(ct["build_worker_destroy_main"]["malloc_growth_gb_per_factor"],
                    ct["build_main_destroy_worker"]["malloc_growth_gb_per_factor"])
        add("cross_thread_dealloc_leaks", cross / max(same, 1e-6), 50.0,
            cross / max(same, 1e-6) > 50.0, "x",
            "malloc live bytes per factor, cross-thread destroy over same-thread "
            f"destroy ({cross:.4f} vs {same:.4f} GB) -- the diagnosis")
    a = doc.get("a_scipy")
    if a:
        add("scipy_control_flat", a["growth_gb_per_solve"], 0.01,
            abs(a["growth_gb_per_solve"]) < 0.01, "GB/solve",
            "the pure-scipy control never grew, so the operator and the "
            "factorisation are not the problem")
    b = doc.get("b_eager", {}).get("perturbed")
    if b:
        add("eager_solve_flat", b["growth_gb_per_solve"], 0.01,
            abs(b["growth_gb_per_solve"]) < 0.01, "GB/solve",
            "eager sparse_solve never grew either -- it takes jit, and the "
            "XLA callback pool it brings, to show the bug")
    nc = doc.get("no_clear", {})
    f12 = doc.get("factors", {}).get("N12739")
    if nc and f12:
        drop = (max(nc["cache4"]["rss_gb"]) - max(nc["cache3"]["rss_gb"])) / f12["factor_gb_malloc"]
        add("factor_cache_size_knob", drop, 0.5, drop > 0.5, "factors",
            "dropping factor_cache_size from 4 to 3 lowers the un-cleared plateau "
            f"by {max(nc['cache4']['rss_gb']) - max(nc['cache3']['rss_gb']):.3f} GB, "
            f"i.e. this many factors of {f12['factor_gb_malloc']:.3f} GB")
    n33 = doc.get("n33k", {}).get("fixed_theta")
    if n33:
        # the fix must not move a single digit of the physics: every repeat
        # bit-identical, and the value equal to the design study's own
        # independently recorded nominal (an EAGER solve of the same model,
        # written by validation/fdfd/spiral_design.py before this fix existed)
        spread = max(n33["value"]) - min(n33["value"])
        add("n33k_value_bit_identical", spread, 0.0, spread == 0.0, "H",
            "spread of L_diff over the four repeated solves")
        ref_path = HERE / "spiral_design.json"
        if ref_path.exists():
            ref = json.loads(ref_path.read_text()).get("nominal", {}).get("L_diff")
            if ref:
                rel = abs(n33["value"][0] - ref) / abs(ref)
                add("n33k_matches_study_nominal", rel, 1e-9, rel < 1e-9, "relative",
                    "L_diff of the jitted three-fixture solve against the design "
                    f"study's recorded nominal ({n33['value'][0]:.15e} vs {ref:.15e})")
    sub = doc.get("subprocess", {}).get("runs")
    c_val = doc.get("c_jit", {}).get("fixed_theta", {}).get("value")
    if sub and c_val:
        worst = max(abs(r["value"] - c_val[0]) / abs(c_val[0]) for r in sub)
        add("isolated_process_same_value", worst, 0.0, worst == 0.0, "relative",
            "one-solve-per-process control returns the same L_diff bit for bit "
            "as the repeated in-process loop")
    if n33:
        add("n33k_growth_bounded", n33["growth_fraction_of_first"], 0.15,
            n33["growth_fraction_of_first"] < 0.15, "fraction",
            "the design fixture itself, N = 33352, four jitted solves")
    return g


def conclusion(doc: dict[str, Any]) -> dict[str, Any]:
    """The peak a 16-solve design loop at N = 33352 reaches, after and before.

    "16 solves" is the design study's gradient loop: the first solve plus 15
    more in the same process. After the fix the series is flat, so the peak
    is the measured plateau; before it, each solve lost ``leaked_factors_per_solve``
    factors, which at N = 33352 is ``factor_gb_N33352`` each.
    """
    out: dict[str, Any] = {}
    fa = doc.get("factors", {})
    f33, f12 = fa.get("N33352"), fa.get("N12739")
    n33 = doc.get("n33k", {}).get("fixed_theta")
    c = doc.get("c_jit", {}).get("fixed_theta")
    p = doc.get("pre_fix", {}).get("forward_fixed_theta")
    d = doc.get("d_grad", {}).get("fixed_theta")
    pg = doc.get("pre_fix", {}).get("grad_fixed_theta")
    if f12:
        out["lu_nnz_N12739"] = f12["lu_nnz"]
        out["factor_gb_N12739"] = f12["factor_gb_malloc"]
        out["bytes_per_lu_nnz_N12739"] = f12["bytes_per_lu_nnz"]
    if f33:
        out["lu_nnz_N33352"] = f33["lu_nnz"]
        out["factor_gb_N33352"] = f33["factor_gb_malloc"]
        out["bytes_per_lu_nnz_N33352"] = f33["bytes_per_lu_nnz"]
        out["factor_cache_gb_N33352_size4"] = 4 * f33["factor_gb_malloc"]
        out["factor_cache_gb_N33352_size3"] = 3 * f33["factor_gb_malloc"]
    if p and f12:
        leaked = p["growth_gb_per_solve"] / f12["factor_gb_malloc"]
        out["growth_gb_per_solve_N12739_before"] = p["growth_gb_per_solve"]
        out["leaked_factors_per_solve"] = leaked
        if f33:
            out["growth_gb_per_solve_N33352_before_scaled"] = leaked * f33["factor_gb_malloc"]
    if c:
        out["growth_gb_per_solve_N12739_after"] = c["growth_gb_per_solve"]
    if n33:
        out["n33k_rss_gb"] = n33["rss_gb"]
        out["n33k_growth_gb_per_solve"] = n33["growth_gb_per_solve"]
        peak = max(n33["rss_gb"]) + 15.0 * max(0.0, n33["growth_gb_per_solve"])
        out["peak_gb_16_solve_loop_after_fix"] = peak
        if "growth_gb_per_solve_N33352_before_scaled" in out:
            out["peak_gb_16_solve_loop_before_fix"] = (
                n33["rss_gb"][0] + 15.0 * out["growth_gb_per_solve_N33352_before_scaled"])
    nc = doc.get("no_clear", {})
    if nc and f33:
        out["no_clear_plateau_gb_N12739_cache4"] = max(nc["cache4"]["rss_gb"])
        out["no_clear_plateau_gb_N12739_cache3"] = max(nc["cache3"]["rss_gb"])
        out["no_clear_saving_gb_N12739"] = (max(nc["cache4"]["rss_gb"])
                                            - max(nc["cache3"]["rss_gb"]))
    if d and pg:
        after = sorted(d["seconds"])[len(d["seconds"]) // 2]
        before = sorted(pg["seconds"])[len(pg["seconds"]) // 2]
        out["grad_seconds_N12739_before"] = before
        out["grad_seconds_N12739_after"] = after
        out["grad_slowdown"] = after / before
    if c and p:
        after = sorted(c["seconds"])[len(c["seconds"]) // 2]
        before = sorted(p["seconds"])[len(p["seconds"]) // 2]
        out["forward_seconds_N12739_before"] = before
        out["forward_seconds_N12739_after"] = after
        out["forward_slowdown"] = after / before
    return out


def load() -> dict[str, Any]:
    if JSON_PATH.exists():
        return json.loads(JSON_PATH.read_text())
    return {}


def save(doc: dict[str, Any]) -> None:
    doc["gates"] = gates(doc)
    doc["conclusion"] = conclusion(doc)
    doc["written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    JSON_PATH.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")


def run_block_subprocess(name: str) -> dict[str, Any]:
    """Each block in its own process: clean RSS series, and a block killed by
    the machine's watchdog costs only itself."""
    t0 = time.time()
    p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()),
                        "--block", name, "--stdout"],
                       cwd=str(REPO), capture_output=True, text=True)
    marks = [ln for ln in p.stdout.splitlines() if ln.startswith("BLOCK_JSON ")]
    if p.returncode != 0 or not marks:
        raise RuntimeError(f"block {name} failed (rc {p.returncode}):\n"
                           f"{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    out = json.loads(marks[-1][len("BLOCK_JSON "):])
    out["_block_seconds"] = time.time() - t0
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block", help="run one block in this process")
    ap.add_argument("--stdout", action="store_true", help="print the block as JSON")
    ap.add_argument("--from-json", action="store_true",
                    help="reuse blocks already in memory_probe.json")
    ap.add_argument("--only", nargs="*", help="run only these blocks")
    args = ap.parse_args(argv)

    if args.block:
        sys.path.insert(0, str(REPO))
        res = BLOCKS[args.block]()
        if args.stdout:
            print("BLOCK_JSON " + json.dumps(res))
        return 0

    doc = load() if args.from_json else {}
    names = args.only if args.only else list(BLOCKS)
    for name in names:
        if args.from_json and name in doc:
            print(f"[{name}] reused", flush=True)
            continue
        print(f"[{name}] running", flush=True)
        t0 = time.time()
        doc[name] = run_block_subprocess(name)
        save(doc)                                # incremental: a kill keeps the rest
        print(f"[{name}] {time.time() - t0:.1f} s", flush=True)
    save(doc)
    for g in doc["gates"]:
        print(f"  {'PASS' if g['passed'] else 'FAIL'}  {g['name']}: "
              f"{g['measured']:.4g} vs {g['threshold']:.4g} {g['units']}", flush=True)
    print(json.dumps(doc["conclusion"], indent=1))
    return 0 if all(g["passed"] for g in doc["gates"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
