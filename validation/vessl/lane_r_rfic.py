"""The GPU lane of study R (``validation/fdfd/rfic_spiral.py`` parts A, B and
``validation/fdfd/rfic_design.py`` part C): the paper-scale square spiral on
the level-invariant fixture with a graded silicon, a Leontovich metal and
lossy silicon at 2.45 GHz, through cuDSS.

The blocks run in the order listed in ``RFX_R_BLOCKS`` and every result is
written to ``$RFX_OUT/$RFX_R_JSON`` after every solve:

* ``plans``: cuDSS plan-only memory estimates of the captured DUT operator
  for ``RFX_R_PLANS`` ("t_si_um:grade:m", grade 0 = layer-uniform silicon);
* ``tsi1``: level-1 forward solves at every thickness of ``T_SI_SENS``
  (graded) plus the chosen thickness with the layer-uniform silicon;
* ``levels``: forward solves at ``RFX_R_LEVELS`` at 2.45 GHz;
* ``tsi2``: the resolved level at ``T_SI_SENS_L2``;
* ``sweep``: the resolved level at ``SWEEP_GHZ``;
* ``grad``: L_diff, Q_diff and both gradients at the resolved level (one
  forward, two reverse passes);
* ``fd``: FD4 of L_diff and Q_diff in ``FD_PARAMS`` at the resolved level;
* ``csweep``: part C's 3 x 3 x 3 baseline sweep (27 forward solves);
* ``design``: part C's L-BFGS-B loop (``rfic_design.run_design``);
* ``fdopt``: FD4 at the optimum ``RFX_R_THETA_OPT`` (m or um) for ``RFX_R_FDOPT``
  parameters, plus the AD gradients there;
* ``fine``: forward solves of the cases in ``RFX_R_FINE_CASES`` (finer
  in-plane / finer vertical grids and their references, at the optimum or
  the paper values; see ``b_fine``). ``RFX_R_HYBRID_LIMIT`` (with
  ``RFX_FDFD_CUDSS_HYBRID=1``) would run cuDSS in hybrid memory mode with a
  device limit, set here by wrapping ``rfx.fdfd._cudss._execution_options``
  (the backend is not edited); no case of this study uses it.
"""
from __future__ import annotations

import functools
import os
import pathlib
import sys
import time
import traceback
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _gpu_lane_lib as lib                                        # noqa: E402

FDFD = lib.REPO / "validation" / "fdfd"


def _load(name: str) -> Any:
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, FDFD / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def host_memory() -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                out[k] = float(v.split()[0]) / 2 ** 20
    except Exception:
        pass
    try:
        import resource
        out["peak_rss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    except Exception:
        pass
    return out


def set_hybrid_limit(limit: str) -> None:
    """cuDSS hybrid memory mode WITH a device-memory limit (the study P
    attempt at N = 2.1e6 ran it without one and died with ALLOC_FAILED at
    47.52 GB on the 48 GB card; the likely cause was never verified)."""
    from rfx.fdfd import _cudss

    def execution() -> Any:
        if not _cudss.hybrid_memory():
            return None
        from nvmath.sparse.advanced import ExecutionCUDA, HybridMemoryModeOptions
        return ExecutionCUDA(hybrid_memory_mode_options=HybridMemoryModeOptions(
            hybrid_memory_mode=True, hybrid_device_memory_limit=limit))

    _cudss._execution_options = execution


def main() -> int:
    import jax
    jax.config.update("jax_enable_x64", True)
    import numpy as np

    from rfx.fdfd import _cudss
    import rfx.fdfd.linear_solve as ls
    rs = _load("rfic_spiral")
    sys.modules["rfic_spiral_for_design"] = rs      # rfic_design.load_rs() -> the same module
    rd = _load("rfic_design")
    rec = lib.Recorder(os.environ.get("RFX_R_JSON", "r_lane.json"),
                       os.environ.get("RFX_R_LANE", "R"))
    rec["env"] = lib.env_record()
    rec["host_memory_gb"] = host_memory()
    backend = os.environ.get("RFX_FDFD_BACKEND", "cudss")
    blocks = os.environ.get("RFX_R_BLOCKS", "plans tsi1").split()
    deadline = rec.t0 + float(os.environ.get("RFX_R_DEADLINE_S", "4800"))
    ls.factor_cache_size(int(os.environ.get("RFX_R_FACTOR_CACHE", "0")))
    hyb_limit = os.environ.get("RFX_R_HYBRID_LIMIT", "").strip()
    if hyb_limit and _cudss.hybrid_memory():
        set_hybrid_limit(hyb_limit)
    rec["config"] = {"backend": backend, "blocks": blocks, "deadline_s": deadline - rec.t0,
                     "factor_cache_size": ls.factor_cache_size(),
                     "free_forward": _cudss.free_forward(),
                     "hybrid_memory": _cudss.hybrid_memory(), "hybrid_limit": hyb_limit or None,
                     "t_si": rs.T_SI_R, "si_grade": rs.SI_GRADE, "freq": rs.FREQ,
                     "sigma_metal": rs.SIGMA_METAL, "sigma_si": rs.SIGMA_SI,
                     "resolved_level": rs.RESOLVED, "box": [list(b) for b in rd.BOX],
                     "lam": rd.LAM}
    if backend != "superlu" and not ls.backend_available(backend):
        rec["blocked"] = {"backend": backend, "reason": _cudss.unavailable_reason(backend)}
        return 3

    def log(msg: str) -> None:
        print(msg, flush=True)

    def left() -> bool:
        return time.time() < deadline

    def dev() -> float:
        return float(_cudss.device_memory().get("used", float("nan")))

    def solve(model: Any, **kw: Any) -> dict[str, Any]:
        ls.clear_factor_cache()
        with ls.default_backend(backend):
            r = rs.forward(model, **kw)
        r["device_memory_gb_after"] = dev()
        r.update(lib.last_factor_stats())
        ls.clear_factor_cache()
        return r

    cache: dict[Any, Any] = {}
    # RFX_R_SMOKE=1: every model is rfic_spiral.build_cheap() and the design box
    # is shrunk around its theta -- a local CPU check of this lane's plumbing
    # (superlu backend), never a result
    smoke = os.environ.get("RFX_R_SMOKE", "0") == "1"
    if smoke:
        th_c = rs.build_cheap().theta_nominal
        rs.THETA0 = th_c
        rd.BOX = tuple((0.95 * v, 1.05 * v) for v in th_c)
        rd.SWEEP_AXES = tuple((lo, 0.5 * (lo + hi), hi) for lo, hi in rd.BOX)
        rs.SWEEP_GHZ = (2.0, 2.45)
        rec["smoke"] = True

    def model_at(m: int, t_si: float = rs.T_SI_R, grade: float | None = rs.SI_GRADE) -> Any:
        key = (m, t_si, grade)
        if key not in cache:
            cache.clear()
            cache[key] = rs.build_cheap() if smoke else rs.build_r(m, t_si=t_si, grade=grade)
        return cache[key]

    def guarded(name: str, fn: Any) -> None:
        if not left():
            rec[name] = {"skipped": "deadline"}
            return
        t0 = time.time()
        try:
            fn()
        except Exception as exc:
            err = {"error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-3000:],
                   "device_memory_gb_at_error": _cudss.device_memory()}
            cur = rec.get(name)
            if isinstance(cur, dict):
                cur.update(err)
                rec[name] = cur
            else:
                rec[name] = err
            log(f"  block {name} FAILED: {err['error']}")
        ls.clear_factor_cache()
        log(f"  block {name}: {time.time() - t0:.0f} s")
        rec.dump()

    res = rs.RESOLVED

    # ---- plans
    def b_plans() -> None:
        from rfx.fdfd import spiral as sm
        out: dict[str, Any] = {}
        rec["plans"] = out
        for item in os.environ.get("RFX_R_PLANS", "190:1.5:2").split():
            if not left():
                break
            t_um, g, m = item.split(":")
            grade = float(g) or None
            t0 = time.time()
            model = (rs.build_cheap() if smoke else
                     rs.build_r(int(m), t_si=float(t_um) * 1e-6, grade=grade))
            d, r, c, b = lib.capture_system(functools.partial(
                sm.solve_spiral, model, rs.FREQ, sigma_metal=rs.SIGMA_METAL,
                sigma_si=rs.SIGMA_SI, fixtures=("dut",)))
            p = _cudss.plan_estimate(d, r, c, int(b.shape[0]),
                                     m=int(b.shape[1]) if b.ndim > 1 else 1)
            p["grid"] = rs.grid_record(model)
            p["assembly_seconds"] = time.time() - t0
            out[item] = p
            log(f"  plan {item}: N={p['grid']['n_unknowns']} perm "
                f"{p.get('permanent_device_memory_gb', float('nan')):.2f} GB peak "
                f"{p.get('peak_device_memory_gb', float('nan')):.2f} GB hybrid_min "
                f"{p.get('hybrid_min_device_memory_gb', float('nan')):.2f} GB")
            del model, d, r, c, b
            rec["plans"] = out

    # ---- level-1 thickness sensitivity (+ graded vs uniform)
    def b_tsi1() -> None:
        out: dict[str, Any] = {}
        rec["tsi_level1"] = out
        for t in rs.T_SI_SENS:
            if not left():
                break
            model = model_at(1, t)
            r = solve(model)
            r["grid"] = rs.grid_record(model)
            out[f"{t * 1e6:g}"] = r
            log(f"  tsi {t * 1e6:g} um level 1: N={r['grid']['n_unknowns']} "
                f"L={r['L_diff'] * 1e12:.2f} pH Q={r['Q_diff']:.4f} ({r['seconds']:.0f} s)")
            rec["tsi_level1"] = out
        model = model_at(1, rs.T_SI_R, None)
        r = solve(model)
        r["grid"] = rs.grid_record(model)
        rec["uniform_vs_graded"] = {"level": 1, "uniform": r}
        log(f"  uniform silicon level 1: N={r['grid']['n_unknowns']} "
            f"L={r['L_diff'] * 1e12:.2f} pH Q={r['Q_diff']:.4f}")

    # ---- levels at 2.45 GHz
    def b_levels() -> None:
        out: dict[str, Any] = rec.get("levels") or {}
        rec["levels"] = out
        for m in [int(v) for v in os.environ.get("RFX_R_LEVELS", "1 2").split()]:
            if not left():
                break
            model = model_at(m)
            r = solve(model)
            r["grid"] = rs.grid_record(model)
            r["host_memory_gb"] = host_memory()
            out[str(m)] = r
            log(f"  level {m}: N={r['grid']['n_unknowns']} W cells "
                f"{r['grid']['cells_across_width']} L={r['L_diff'] * 1e12:.3f} pH "
                f"Q={r['Q_diff']:.5f} L_se={r['L_se'] * 1e12:.3f} pH Q_se={r['Q_se']:.5f} "
                f"({r['seconds']:.0f} s, dev {r['device_memory_gb_after']:.2f} GB)")
            rec["levels"] = out

    def b_tsi2() -> None:
        model = model_at(res, rs.T_SI_SENS_L2)
        r = solve(model)
        r["grid"] = rs.grid_record(model)
        rec["tsi_resolved"] = r
        log(f"  tsi {rs.T_SI_SENS_L2 * 1e6:g} um level {res}: L={r['L_diff'] * 1e12:.3f} pH "
            f"Q={r['Q_diff']:.5f}")

    def b_sweep() -> None:
        out: dict[str, Any] = rec.get("sweep") or {}
        rec["sweep"] = out
        model = model_at(res)
        for f in rs.SWEEP_GHZ:
            if not left():
                break
            k = f"{f:g}"
            lv = (rec.get("levels") or {}).get(str(res))
            if abs(f * 1e9 - rs.FREQ) < 1.0 and lv:
                out[k] = dict(lv, reused="levels.%d (the same solve)" % res)
            else:
                out[k] = solve(model, freq=f * 1e9)
            log(f"  sweep {k} GHz: L={out[k]['L_diff'] * 1e12:.3f} pH Q={out[k]['Q_diff']:.4f}")
            rec["sweep"] = out

    def b_grad() -> None:
        model = model_at(res)
        ls.clear_factor_cache()
        with ls.default_backend(backend):
            g = rs.metrics_with_grads(model)
        g["device_memory_gb_after"] = dev()
        rec["grad"] = g
        log(f"  grad level {res}: L={g['L_diff'] * 1e12:.3f} pH Q={g['Q_diff']:.5f} dL={g['dL']} "
            f"dQ={g['dQ']} ({g['seconds']['total']:.0f} s)")

    def b_fd() -> None:
        out: dict[str, Any] = rec.get("fd") or {}
        rec["fd"] = out
        model = model_at(res)
        for name in os.environ.get("RFX_R_FD", " ".join(rs.FD_PARAMS)).split():
            if not left():
                break
            k = rs.PARAMS.index(name)
            with ls.default_backend(backend):
                out[name] = rs.fd4_param(model, k, log=log)
            ls.clear_factor_cache()
            log(f"  FD4 {name}: dL={out[name]['fd_L']:.8e} dQ={out[name]['fd_Q']:.8e}")
            rec["fd"] = out

    def b_csweep() -> None:
        out: dict[str, Any] = rec.get("csweep") or {"points": []}
        rec["csweep"] = out
        model = model_at(res)
        from rfx.fdfd import spiral as sm
        t0 = time.time()
        done = {tuple(p["theta"]) for p in out["points"]}
        for th in rd.sweep_points():
            if not left():
                break
            if th in done:
                continue
            sm.check_feasible(model, np.asarray(th))
            r = solve(model, theta=th)
            out["points"].append(r)
            log(f"  csweep {len(out['points']):2d}/27 ({th[0] * 1e6:.0f}, {th[1] * 1e6:.0f}, "
                f"{th[2] * 1e6:.0f}) um: L={r['L_diff'] * 1e12:.2f} pH Q={r['Q_diff']:.4f} "
                f"({r['seconds']:.0f} s)")
            out["seconds"] = out.get("seconds_prior", 0.0) + time.time() - t0
            rec["csweep"] = out

    def b_design() -> None:
        model = model_at(res)
        q_ref = float(os.environ.get("RFX_R_QREF", "0") or 0) or float(rd.Q_REF)
        ls.clear_factor_cache()

        def record(d: dict[str, Any]) -> None:
            rec["design"] = d

        with ls.default_backend(backend):
            rd.run_design(model, q_ref, record, deadline=deadline, log=log)

    def theta_opt() -> list[float]:
        """``RFX_R_THETA_OPT`` (um), else this job's own design result."""
        env = [float(v) for v in os.environ.get("RFX_R_THETA_OPT", "").split()]
        if env:
            # metres (the design result's own repr, bit-exact) or micrometres
            return env if max(env) < 1e-2 else [v * 1e-6 for v in env]
        return [float(v) for v in rec["design"]["result"]["theta"]]

    def b_fdopt() -> None:
        model = model_at(res)
        th = theta_opt()
        out: dict[str, Any] = {"theta": th, "fd": {}}
        rec["fdopt"] = out
        ls.clear_factor_cache()
        with ls.default_backend(backend):
            out["grad"] = rs.metrics_with_grads(model, theta=th)
        rec["fdopt"] = out
        log(f"  grad at optimum: dL={out['grad']['dL']} dQ={out['grad']['dQ']}")
        for name in os.environ.get("RFX_R_FDOPT", "r_out width").split():
            if not left():
                break
            k = rs.PARAMS.index(name)
            with ls.default_backend(backend):
                out["fd"][name] = rs.fd4_param(model, k, theta0=th, log=log)
            ls.clear_factor_cache()
            rec["fdopt"] = out

    def b_fine() -> None:
        """The optimum one level finer. The full level 3 is a 106.71 GB cuDSS
        factor (R1 plan) and gpu-a6000-1 has 48 GB of device memory and a
        32 GiB host-memory limit, so it can run neither in-core nor in hybrid
        mode there. ``RFX_R_FINE_CASES`` lists what can: items
        ``name:refine:z_refine:wall_w:opt|nom`` (z_refine 0 = follow refine),
        each built, planned (skipped above ``RFX_R_INCORE_MAX_GB``) and solved
        forward at the design optimum or the paper values."""
        out: dict[str, Any] = rec.get("fine") or {"cases": {}}
        rec["fine"] = out
        from rfx.fdfd import spiral as sm
        cap = float(os.environ.get("RFX_R_INCORE_MAX_GB", "44"))
        for item in os.environ.get("RFX_R_FINE_CASES", "").split():
            if not left():
                break
            name, m_, zr_, w_, which = item.split(":")
            m, zr, w = int(m_), int(zr_), float(w_)
            th = theta_opt() if which == "opt" else list(rs.THETA0)
            case: dict[str, Any] = {"refine": m, "z_refine": zr or m, "wall_w": w, "at": which}
            out["cases"][name] = case
            try:
                model = rs.build_cheap() if smoke else rs.build_r(m, wall_w=w, z_refine=zr)
                case["grid"] = rs.grid_record(model)
                if not smoke:
                    d, r, c, b = lib.capture_system(functools.partial(
                        sm.solve_spiral, model, rs.FREQ, sigma_metal=rs.SIGMA_METAL,
                        sigma_si=rs.SIGMA_SI, fixtures=("dut",)))
                    p = _cudss.plan_estimate(d, r, c, int(b.shape[0]),
                                             m=int(b.shape[1]) if b.ndim > 1 else 1)
                    del d, r, c, b
                    case["plan_gb"] = float(p.get("permanent_device_memory_gb", float("nan")))
                    if case["plan_gb"] > cap:
                        case["skipped"] = f"plan {case['plan_gb']:.2f} GB > {cap:g} GB in-core cap"
                        log(f"  fine {name}: SKIPPED ({case['skipped']})")
                        rec["fine"] = out
                        continue
                r = solve(model, theta=th)
                r["host_memory_gb"] = host_memory()
                case.update(r)
                log(f"  fine {name}: N={case['grid']['n_unknowns']} plan "
                    f"{case.get('plan_gb', float('nan')):.2f} GB L={r['L_diff'] * 1e12:.3f} pH "
                    f"Q={r['Q_diff']:.5f} ({r['seconds']:.0f} s)")
            except Exception as exc:
                case["error"] = f"{type(exc).__name__}: {exc}"
                log(f"  fine {name} FAILED: {case['error']}")
            ls.clear_factor_cache()
            rec["fine"] = out

    table = {"plans": b_plans, "tsi1": b_tsi1, "levels": b_levels, "tsi2": b_tsi2,
             "sweep": b_sweep, "grad": b_grad, "fd": b_fd, "csweep": b_csweep,
             "design": b_design, "fdopt": b_fdopt, "fine": b_fine}
    for name in blocks:
        log(f"=== block {name} ===")
        guarded(name, table[name])
    rec["host_memory_gb_end"] = host_memory()
    rec.dump()
    log(f"lane done in {rec['seconds']:.0f} s -> {rec.path}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.exit(main())
