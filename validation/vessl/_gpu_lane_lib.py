"""Shared measurement helpers for the VESSL GPU lanes (J0 / J1 / J2).

Imported by the lane scripts in this directory, which run inside the
container against the repo tarball the build script uploads. Nothing here
touches the cluster: it is the measurement code, kept in one place so the
three lanes report the same fields and ``validation/fdfd/gpu_scaling.py``
can merge them without special cases.

The lane scripts are embedded in the generated YAML as base64 (a quoted
heredoc inside a VESSL ``run:`` block kills the job at parse time) and
this module is imported from the untarred repo, so both are lintable
Python in the repository rather than strings in a YAML.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SC_PATH = REPO / "validation" / "fdfd" / "spiral_convergence.py"
SC_JSON = REPO / "validation" / "fdfd" / "spiral_convergence.json"


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def out_dir() -> pathlib.Path:
    """Where the job writes its artifacts (``$RFX_OUT``, or the cwd)."""
    d = pathlib.Path(os.environ.get("RFX_OUT", "."))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.void,)):                      # a numpy record (cuDSS)
        return {k: _default(o[k]) for k in (o.dtype.names or ())}
    return str(o)


class Recorder:
    """A JSON file rewritten after every block.

    The lanes are wall-clock bounded and a level can be killed by the job
    timeout or by the device running out of memory; rewriting after each
    block means a killed run still leaves every measurement it finished.
    """

    def __init__(self, name: str, lane: str):
        self.path = out_dir() / name
        self.t0 = time.time()
        self.rec: dict[str, Any] = {"lane": lane, "started_utc": _utc(),
                                    "artifact_dir": str(out_dir())}

    def __setitem__(self, k: str, v: Any) -> None:
        self.rec[k] = v
        self.dump()

    def __getitem__(self, k: str) -> Any:
        return self.rec[k]

    def get(self, k: str, default: Any = None) -> Any:
        return self.rec.get(k, default)

    def dump(self) -> None:
        self.rec["seconds"] = time.time() - self.t0
        self.rec["finished_utc"] = _utc()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.rec, indent=1, default=_default))
        tmp.replace(self.path)

    def note(self, *args: Any) -> None:
        print(*args, flush=True)


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

def env_record() -> dict[str, Any]:
    """Versions, device, and the backend availability probe with reasons."""
    import jax
    import scipy

    from rfx.fdfd import _cudss
    import rfx.fdfd.linear_solve as ls
    rec: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "jax": jax.__version__,
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS"),
        "jax_devices": [str(d) for d in jax.devices()],
        "jax_default_backend": jax.default_backend(),
        "default_backend": ls.get_default_backend(),
        "env_backend": os.environ.get("RFX_FDFD_BACKEND"),
        "cudss_hybrid_memory": _cudss.hybrid_memory(),
        "cudss_free_forward": _cudss.free_forward(),
        "backends": _cudss.availability(refresh=True),
        "device_memory_gb": _cudss.device_memory(),
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    for mod in ("nvmath", "cupy"):
        try:
            m = __import__(mod)
            rec[f"{mod}_version"] = getattr(m, "__version__", "?")
        except Exception as exc:
            rec[f"{mod}_version"] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        rec["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:
        rec["nvidia_smi"] = f"unavailable: {type(exc).__name__}: {exc}"
    return rec


# ---------------------------------------------------------------------------
# the study module (imported by path: validation/ is not a package)
# ---------------------------------------------------------------------------

def load_study() -> Any:
    """``validation/fdfd/spiral_convergence.py`` as a module, unmodified.

    The GPU lanes reuse its fixture constants, its ``build_level`` and its
    ``richardson`` so a GPU level is the SAME measurement as a CPU level;
    nothing in that file is edited (it owns the CPU study).
    """
    spec = importlib.util.spec_from_file_location("spiral_convergence_lane", SC_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["spiral_convergence_lane"] = mod
    spec.loader.exec_module(mod)
    return mod


def cpu_study() -> dict[str, Any]:
    """The CPU study's JSON (referee, levels W/1..W/3) as the reference."""
    return json.loads(SC_JSON.read_text())


# ---------------------------------------------------------------------------
# the linear system of one fixture, without solving it
# ---------------------------------------------------------------------------

class _Captured(Exception):
    pass


def capture_system(solve_call: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(data, rows, cols, rhs)`` of the FIRST system ``solve_call()`` builds.

    Spies on ``rfx.fdfd.yee3d.sparse_solve`` and aborts the call as soon as
    the assembled system reaches it, so this costs the assembly and not the
    factorisation. That is how gate G1 gets the EXACT ``(data, rows, cols,
    b)`` of the real W/3 DUT fixture to hand to two backends, instead of a
    reimplementation of the assembly that could differ from the study's.
    """
    import rfx.fdfd.yee3d as y3
    box: dict[str, Any] = {}
    orig = y3.sparse_solve

    def spy(data: Any, rows: Any, cols: Any, b: Any, **kw: Any) -> Any:
        box["sys"] = (np.asarray(data, dtype=np.complex128),
                      np.asarray(rows), np.asarray(cols),
                      np.asarray(b, dtype=np.complex128))
        raise _Captured

    y3.sparse_solve = spy
    try:
        solve_call()
    except _Captured:
        pass
    finally:
        y3.sparse_solve = orig
    if "sys" not in box:
        raise RuntimeError("no sparse system reached yee3d.sparse_solve")
    return box["sys"]


def dut_system_at(_study: Any, model: Any, freq: float, sigma: float
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The volumetric-metal DUT system at an explicit frequency and sigma."""
    from rfx.fdfd import spiral as sm
    return capture_system(lambda: sm.solve_spiral(
        model, freq, sigma_volumetric=sigma, fixtures=("dut",)))


def dut_system(study: Any, model: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The volumetric-metal DUT system of one level (the study's own fixture)."""
    return dut_system_at(study, model, study.FREQ, study.SIGMA_VOL)


# ---------------------------------------------------------------------------
# gate G1: two backends, one system
# ---------------------------------------------------------------------------

def _csr(data: np.ndarray, rows: np.ndarray, cols: np.ndarray, n: int) -> Any:
    import scipy.sparse as sp
    a = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    a.sum_duplicates()
    return a


def last_factor_stats() -> dict[str, Any]:
    """cuDSS's own account of the factor most recently built (LU nnz, GB)."""
    import rfx.fdfd.linear_solve as ls
    for lu in reversed(list(ls._FACTOR_CACHE.values())):
        st = getattr(lu, "stats", None)
        if st:
            return {"stats": st,
                    "factorizations": int(getattr(lu, "factorizations", 0)),
                    "solves": int(getattr(lu, "solves", 0)),
                    "freed_forward": int(getattr(lu, "freed_forward", 0))}
    return {}


def solve_both(data: np.ndarray, rows: np.ndarray, cols: np.ndarray, rhs: np.ndarray,
               backends: tuple[str, ...] = ("superlu", "cudss"),
               reference: str = "superlu") -> dict[str, Any]:
    """Gate G1: solve ONE system with several backends and compare.

    Per backend: the cold wall time (factor + solve), the warm wall time of
    a repeat (a factor-cache hit, so triangular solves only), hence the
    factorisation time by difference, the relative residual
    ``||A x - b|| / ||b||`` of its own solution, the device memory in use
    after it, and cuDSS's LU nonzero count. Then every backend's solution
    against ``reference``'s, as a relative max-norm difference.

    Neither backend is "the answer": both are direct solves with their own
    pivoting, so what is gated is that each residual is at the roundoff
    level and that the difference between the solutions is of that order
    too (rule 2 of the study).
    """
    import jax.numpy as jnp

    from rfx.fdfd import _cudss
    import rfx.fdfd.linear_solve as ls
    n = int(rhs.shape[0])
    a = _csr(data, rows, cols, n)
    nb = float(np.linalg.norm(rhs))
    out: dict[str, Any] = {"n": n, "nnz": int(data.size),
                           "rhs_shape": list(rhs.shape), "reference": reference,
                           "per_backend": {}}
    xs: dict[str, np.ndarray] = {}
    jd, jb = jnp.asarray(data), jnp.asarray(rhs)
    for be in backends:
        ls.clear_factor_cache()
        mem0 = _cudss.device_memory()
        t0 = time.time()
        x = np.asarray(ls.sparse_solve(jd, rows, cols, jb, backend=be))
        t1 = time.time()
        x_again = np.asarray(ls.sparse_solve(jd, rows, cols, jb, backend=be))
        t2 = time.time()
        mem1 = _cudss.device_memory()
        rec: dict[str, Any] = {
            "cold_seconds": t1 - t0,
            "warm_seconds": t2 - t1,
            "factor_seconds": (t1 - t0) - (t2 - t1),
            "solve_seconds": t2 - t1,
            "residual": float(np.linalg.norm(a @ x - rhs)) / nb,
            "residual_warm": float(np.linalg.norm(a @ x_again - rhs)) / nb,
            "warm_is_bit_identical": bool(np.array_equal(x, x_again)),
            "device_memory_gb_before": mem0,
            "device_memory_gb_after": mem1,
            "device_memory_gb_delta": mem1["used"] - mem0["used"],
        }
        rec.update(last_factor_stats())
        out["per_backend"][be] = rec
        xs[be] = x
        print(f"    {be:9s} cold {rec['cold_seconds']:8.2f} s  warm "
              f"{rec['warm_seconds']:7.2f} s  residual {rec['residual']:.3e}  "
              f"dev {rec['device_memory_gb_after']['used']:.2f} GB", flush=True)
        ls.clear_factor_cache()
    ref = xs[reference]
    scale = float(np.max(np.abs(ref)))
    for be, x in xs.items():
        if be == reference:
            continue
        d = out["per_backend"][be]
        d["max_abs_diff_vs_reference"] = float(np.max(np.abs(x - ref)))
        d["rel_diff_vs_reference"] = float(np.max(np.abs(x - ref))) / scale
        d["rel_l2_diff_vs_reference"] = float(np.linalg.norm(x - ref)
                                              / np.linalg.norm(ref))
        print(f"    {be} vs {reference}: rel diff "
              f"{d['rel_diff_vs_reference']:.3e}", flush=True)
    out["reference_scale"] = scale
    return out


# ---------------------------------------------------------------------------
# a level: value_and_grad of L_dut, per backend
# ---------------------------------------------------------------------------

def value_and_grad_level(study: Any, model: Any, backend: str,
                         theta: Any = None) -> dict[str, Any]:
    """``L_dut`` and ``dL_dut/dtheta`` of one level through one backend.

    The same quantity ``spiral_convergence.run_level`` records, computed
    the same way (eager ``jax.value_and_grad`` of the real three-fixture
    de-embedded ``L_diff``), with the backend selected by the scoped
    default because ``solve_spiral`` does not forward the keyword.
    """
    import jax
    import jax.numpy as jnp

    from rfx.fdfd import _cudss
    import rfx.fdfd.linear_solve as ls
    theta0 = study.THETA0 if theta is None else theta

    def scalar(t: Any) -> Any:
        return jnp.real(study.l_dut(model, t))

    ls.clear_factor_cache()
    mem0 = _cudss.device_memory()
    t0 = time.time()
    with ls.default_backend(backend):
        val, grad = jax.value_and_grad(scalar)(jnp.asarray(theta0, dtype=jnp.float64))
        jax.block_until_ready(grad)
    dt = time.time() - t0
    rec = {"backend": backend, "L_dut": float(val), "grad": [float(v) for v in grad],
           "value_and_grad_seconds": dt,
           "device_memory_gb_before": mem0,
           "device_memory_gb_peak_after": _cudss.device_memory(),
           "factor_cache_size": ls.factor_cache_size(),
           "factor_cache_entries": len(ls._FACTOR_CACHE)}
    rec.update(last_factor_stats())
    print(f"    {backend:9s} L_dut={rec['L_dut'] * 1e12:.4f} pH  "
          f"grad={['%.6e' % g for g in rec['grad']]}  {dt:.1f} s", flush=True)
    ls.clear_factor_cache()
    return rec


def forward_level(study: Any, model: Any, backend: str,
                  theta: Any = None) -> dict[str, Any]:
    """``L_dut`` only (no gradient): the half-memory, half-time fallback."""
    import jax.numpy as jnp

    from rfx.fdfd import _cudss
    import rfx.fdfd.linear_solve as ls
    theta0 = study.THETA0 if theta is None else theta
    ls.clear_factor_cache()
    t0 = time.time()
    with ls.default_backend(backend):
        val = float(jnp.real(study.l_dut(model, jnp.asarray(theta0, dtype=jnp.float64))))
    dt = time.time() - t0
    rec = {"backend": backend, "L_dut": val, "solve_seconds": dt,
           "device_memory_gb_peak_after": _cudss.device_memory()}
    rec.update(last_factor_stats())
    print(f"    {backend:9s} L_dut={val * 1e12:.4f} pH  {dt:.1f} s (forward only)",
          flush=True)
    ls.clear_factor_cache()
    return rec


def model_record(study: Any, model: Any) -> dict[str, Any]:
    """The same grid record the CPU study writes (so the blocks merge)."""
    return study.model_record(model)


def fd4_one_parameter(study: Any, model: Any, backend: str, k: int,
                      theta: Any = None) -> dict[str, Any]:
    """FD4 of ``L_dut`` in parameter ``k`` through ``backend`` -- gate G2.

    Four extra forward solves (``+2h, +h, -h, -2h``) with the study's own
    1 % step, i.e. the same stencil and the same step the CPU study used,
    so the number is comparable with ``levels[...]["fd"]`` in
    ``spiral_convergence.json``.
    """
    import jax.numpy as jnp

    import rfx.fdfd.linear_solve as ls
    theta0 = list(study.THETA0 if theta is None else theta)
    h = study.FD_STEP_REL * theta0[k]
    t0 = time.time()
    with ls.default_backend(backend):
        def f(v: float) -> float:
            t = list(theta0)
            t[k] = v
            ls.clear_factor_cache()
            return float(jnp.real(study.l_dut(model, jnp.asarray(t, dtype=jnp.float64))))
        fd = float(study.fd4(f, theta0[k], h))
    ls.clear_factor_cache()
    return {"backend": backend, "parameter": k, "step": h, "fd_order": 4,
            "fd": fd, "fd_seconds": time.time() - t0}


def ir_sweep(data: np.ndarray, rows: np.ndarray, cols: np.ndarray, rhs: np.ndarray,
             steps: tuple[int, ...] = (0, 1, 2, 4),
             reference: str = "superlu") -> dict[str, Any]:
    """Does cuDSS iterative refinement close the gap to SuperLU's solution?

    The J0 probe measured, on the real W/3 DUT system (N = 108898): SuperLU
    relative residual 1.787e-08, cuDSS 3.841e-09, and the two SOLUTIONS
    differing by 2.395e-03 relative. Both residuals are eight orders above
    the 1e-16 of a well-scaled system, which is the operator talking (the
    curl-curl matrix with discrete DtN ports has cond ~1e12 and a huge row
    scaling), and a solution difference of 1e-3 is what two 1e-8-residual
    solves of it are allowed to have. cuDSS pivots statically where SuperLU
    pivots partially, and its answer to that is iterative refinement.

    This sweeps ``ir_num_steps`` and reports what each step buys: the
    residual, the difference from the reference solution, and the wall time
    -- a solve is 0.08 s against a 0.95 s factorisation, so refinement is
    nearly free if it works. ``RFX_FDFD_CUDSS_IR`` is set around each solve
    and restored afterwards; the factor cache is emptied between them so
    each one re-plans with its own configuration.
    """
    import jax.numpy as jnp

    from rfx.fdfd import _cudss
    import rfx.fdfd.linear_solve as ls
    n = int(rhs.shape[0])
    a = _csr(data, rows, cols, n)
    nb = float(np.linalg.norm(rhs))
    jd, jb = jnp.asarray(data), jnp.asarray(rhs)
    ls.clear_factor_cache()
    t0 = time.time()
    x_ref = np.asarray(ls.sparse_solve(jd, rows, cols, jb, backend=reference))
    out: dict[str, Any] = {
        "what": ("cuDSS ir_num_steps against the reference backend's solution on the "
                 "same system: residual, solution difference, wall time"),
        "reference": reference, "n": n, "nnz": int(data.size),
        "reference_seconds": time.time() - t0,
        "reference_residual": float(np.linalg.norm(a @ x_ref - rhs)) / nb,
        "steps": {},
    }
    scale = float(np.max(np.abs(x_ref)))
    prev = os.environ.get("RFX_FDFD_CUDSS_IR")
    try:
        for s in steps:
            os.environ["RFX_FDFD_CUDSS_IR"] = str(int(s))
            ls.clear_factor_cache()
            t0 = time.time()
            x = np.asarray(ls.sparse_solve(jd, rows, cols, jb, backend="cudss"))
            dt = time.time() - t0
            rec = {"ir_num_steps": int(s), "seconds": dt,
                   "residual": float(np.linalg.norm(a @ x - rhs)) / nb,
                   "rel_diff_vs_reference": float(np.max(np.abs(x - x_ref))) / scale,
                   "rel_l2_diff_vs_reference": float(np.linalg.norm(x - x_ref)
                                                     / np.linalg.norm(x_ref)),
                   "ir_seen_by_backend": _cudss.ir_steps()}
            rec.update(last_factor_stats())
            out["steps"][str(int(s))] = rec
            print(f"    ir={s}: residual {rec['residual']:.3e}  diff "
                  f"{rec['rel_diff_vs_reference']:.3e}  {dt:.2f} s", flush=True)
            ls.clear_factor_cache()
    finally:
        if prev is None:
            os.environ.pop("RFX_FDFD_CUDSS_IR", None)
        else:
            os.environ["RFX_FDFD_CUDSS_IR"] = prev
    return out


# ---------------------------------------------------------------------------
# gate G1's own control: ONE backend, several elimination paths
# ---------------------------------------------------------------------------

def _row_permuted(data: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                  rhs: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, Any]:
    """``(P A) x = P b`` -- the same system, its equations in another order.

    A permutation of the equations changes SuperLU's partial-pivot choices
    and its COLAMD input, hence the elimination path and its rounding,
    while leaving the exact solution ``x`` untouched. ``(P A)[i, :] =
    A[p[i], :]``, so an entry in row ``r`` moves to row ``pinv[r]``.
    """
    p = np.random.default_rng(seed).permutation(int(rhs.shape[0]))
    pinv = np.empty_like(p)
    pinv[p] = np.arange(p.size)
    return data, pinv[rows], (cols, rhs[p], None)


def _col_scaled(data: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                rhs: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, Any]:
    """``(A D) y = b``, ``x = D y`` -- the same system in scaled unknowns.

    ``D`` is diagonal and strictly positive, so the exact solution is
    recovered exactly; the scale factors are deliberately NOT powers of two
    (which would be exact in binary floating point and change nothing) so
    every multiply rounds and the elimination takes a different path.
    """
    d = np.random.default_rng(seed).uniform(0.6, 1.7, size=int(rhs.shape[0]))
    return data * d[cols], rows, (cols, rhs, d)


_VARIANTS = {"row_permuted": _row_permuted, "col_scaled": _col_scaled}


def superlu_ordering_control(data: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                             rhs: np.ndarray, cases: tuple[tuple[str, str], ...],
                             reference: str = "COLAMD", seed: int = 20260915,
                             cond_estimate: bool = True) -> dict[str, Any]:
    """Gate G1's same-backend control: how far apart are two HOST LU solves?

    G1 asks the two backends' solution VECTORS to agree to 1e-8 relative.
    The control for that claim is not another backend -- it is the SAME
    backend told to eliminate in a different order. Each case here is
    scipy's SuperLU through ``rfx.fdfd.linear_solve.sparse_solve`` (the
    shipped path, factor thread and cache included) on a system whose exact
    solution is identical to the reference's:

    * a different fill-reducing column ordering (``permc_spec``);
    * ``row_permuted``: the same equations in a random order, which changes
      the partial-pivot sequence;
    * ``col_scaled``: the same unknowns rescaled by a positive diagonal.

    Reported per case: the relative residual ``||A x - b|| / ||b||`` of its
    solution in the ORIGINAL system, and its distance from the reference
    case's solution in exactly the norm G1 gates on
    (``max|x - x_ref| / max|x_ref|``, plus the L2 form). ``cond_estimate``
    adds a 1-norm condition estimate of the operator via the reference
    factorisation, which is what sets the floor on all of it.

    ``cases`` is a tuple of ``(name, spec)``: ``spec`` is a ``permc_spec``
    (``"COLAMD"``, ``"MMD_ATA"``, ``"MMD_AT_PLUS_A"``, ``"NATURAL"``) or
    ``"<variant>:<permc_spec>"`` for one of the variants above. They run in
    the order given, sequentially, with the factor cache emptied between
    them so that only one host factorisation is ever resident.
    """
    import jax.numpy as jnp

    import rfx.fdfd.linear_solve as ls
    n = int(rhs.shape[0])
    a = _csr(data, rows, cols, n)
    nb = float(np.linalg.norm(rhs))
    out: dict[str, Any] = {
        "what": ("one backend (host SuperLU), several elimination paths on the same "
                 "system: the control for G1's solution-difference clause"),
        "backend": "superlu", "n": n, "nnz": int(data.size),
        "rhs_shape": list(rhs.shape), "reference": reference, "seed": seed,
        "diff_norm": "max|x - x_ref| / max|x_ref|, the same norm G1 gates on",
        "cases": {},
    }
    xs: dict[str, np.ndarray] = {}
    for name, spec in cases:
        variant, _, permc = spec.partition(":")
        if variant in _VARIANTS:
            d, r, (c, b, scale) = _VARIANTS[variant](data, rows, cols, rhs, seed)
        else:
            variant, permc, d, r, c, b, scale = "", spec, data, rows, cols, rhs, None
        permc_spec = None if permc in ("", "COLAMD") else permc
        rec: dict[str, Any] = {"permc_spec": permc or "COLAMD", "variant": variant or None}
        ls.clear_factor_cache()
        t0 = time.time()
        try:
            x = np.asarray(ls.sparse_solve(jnp.asarray(d), r, c, jnp.asarray(b),
                                           permc_spec=permc_spec, backend="superlu"))
        except Exception as exc:                       # out of memory, mostly
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["seconds"] = time.time() - t0
            out["cases"][name] = rec
            print(f"    {name:22s} FAILED {rec['error']}", flush=True)
            ls.clear_factor_cache()
            continue
        rec["seconds"] = time.time() - t0
        if scale is not None:
            x = x * scale[:, None] if x.ndim == 2 else x * scale
        rec["residual"] = float(np.linalg.norm(a @ x - rhs)) / nb
        xs[name] = x
        print(f"    {name:22s} {rec['seconds']:8.1f} s  residual {rec['residual']:.3e}",
              flush=True)
        if cond_estimate and name == reference:
            out["cond_1_estimate"] = _cond_1_estimate(a, permc_spec)
            print(f"    cond_1 estimate {out['cond_1_estimate'].get('cond_1'):.4e}",
                  flush=True)
        ls.clear_factor_cache()
        out["cases"][name] = rec
    ref = xs.get(reference)
    if ref is not None:
        scale = float(np.max(np.abs(ref)))
        for name, x in xs.items():
            rec = out["cases"][name]
            rec["rel_diff_vs_reference"] = float(np.max(np.abs(x - ref))) / scale
            rec["rel_l2_diff_vs_reference"] = float(np.linalg.norm(x - ref)
                                                    / np.linalg.norm(ref))
            if name != reference:
                print(f"    {name:22s} vs {reference}: rel diff "
                      f"{rec['rel_diff_vs_reference']:.3e}", flush=True)
        diffs = [v["rel_diff_vs_reference"] for k, v in out["cases"].items()
                 if k != reference and "rel_diff_vs_reference" in v]
        res = [v["residual"] for v in out["cases"].values() if "residual" in v]
        if diffs:
            out["rel_diff_span"] = [min(diffs), max(diffs)]
            out["worst_rel_diff"] = max(diffs)
        if res:
            out["residual_span"] = [min(res), max(res)]
    return out


def _cond_1_estimate(a: Any, permc_spec: str | None) -> dict[str, Any]:
    """``||A||_1 * ||A^-1||_1``, both estimated (Higham-Tisseur, scipy).

    ``A^-1`` is a ``LinearOperator`` backed by one SuperLU factorisation --
    a handful of triangular solves, next to nothing against the
    factorisation itself -- so the estimate is essentially free wherever a
    factorisation is already affordable. The whole thing runs on the one
    dedicated factor thread, like every other factor operation in this
    repository (scipy's SuperLU leaks a factor freed off the thread that
    created it; see ``rfx.fdfd.linear_solve``).
    """
    import scipy.sparse.linalg as spl

    import rfx.fdfd.linear_solve as ls

    def work() -> dict[str, Any]:
        t0 = time.time()
        try:
            lu = spl.splu(a.tocsc(), permc_spec=permc_spec)
            n = int(a.shape[0])
            inv = spl.LinearOperator(
                (n, n), dtype=a.dtype,
                matvec=lambda v: lu.solve(v),
                rmatvec=lambda v: lu.solve(v, trans="H"))
            norm_a = float(spl.onenormest(a))
            norm_inv = float(spl.onenormest(inv))
            return {"norm_1_A": norm_a, "norm_1_A_inverse": norm_inv,
                    "cond_1": norm_a * norm_inv, "seconds": time.time() - t0,
                    "method": ("scipy.sparse.linalg.onenormest of A and of a "
                               "SuperLU-backed inverse operator")}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}",
                    "seconds": time.time() - t0}

    return ls._on_factor_thread(work)
