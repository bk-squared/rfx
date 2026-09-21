"""P3, the hybrid-memory lane of study P (``validation/fdfd/invariant_ladder.py``).

cuDSS hybrid (host + device) memory mode with an EXPLICIT device-memory
limit (``RFX_FDFD_CUDSS_HYBRID_LIMIT``, ``rfx/fdfd/_cudss.py`` "Device
memory"), on the level-invariant spiral fixture (10 W walls, metal_cells 1).
Blocks, in this order, each written to the JSON as soon as it finishes:

``host``  what this container may actually hold: ``free -g`` and
          ``/proc/meminfo`` (the NODE), the cgroup memory limit and usage
          (v2 ``memory.max`` / v1 ``memory.limit_in_bytes``: the PROCESS),
          ``nvidia-smi``. Hybrid mode keeps the whole factor in host memory,
          so the cgroup limit, not the node total, decides what can run.
``h1``    gate H1 at level ``RFX_P3_H1_LEVEL`` (3; an 11.37 GB cuDSS
          factor by the in-core plan): cuDSS plan estimates in-core and
          hybrid (limit ``RFX_P3_H1_LIMIT``, below the factor size), then a
          one-fixture factor probe (factor and solve wall times, residual,
          the solution difference) and ``value_and_grad`` of the
          three-fixture de-embedded L_dut, first in-core and then hybrid,
          with the device memory sampled every 0.25 s and cupy's pool emptied
          before every measurement. Both comparisons get their CONTROL, a
          THIRD run of the same thing in-core, because cuDSS is not
          bit-reproducible across calls and in-core-vs-in-core is the floor
          any hybrid-vs-in-core number is read against:
          ``RFX_P3_PROBE_CONTROL=1`` (default) for the solution VECTOR of the
          probe, ``RFX_P3_CONTROL=1`` (default) for the value_and_grad. Gate:
          L and the gradient vector equal to 1e-9 relative; the slowdown is
          the wall-time ratio.
``m5``    level ``RFX_P3_M5_LEVEL`` (5; N = 2128175, 91.87 GB in-core factor
          by P0's plan) in hybrid mode with the limit ``RFX_P3_M5_LIMIT``: the
          DUT operator is assembled and PLANNED (in-core and hybrid) -- cheap,
          and it states cuDSS's own host-memory estimate -- and the solve is
          attempted ONLY if the process's host limit holds the factor plus
          what it already uses plus ``RFX_P3_HOST_MARGIN_GB``. Otherwise the
          numbers are written and the block stops, rather than letting the
          kernel OOM-kill the job. If it runs: the three-fixture forward
          solve first (written), then ``value_and_grad``; stored under
          ``levels`` in the P1 lanes' format so ``invariant_ladder.py``
          merges it as a level.

``RFX_P3_SMOKE=1`` runs every code path on the Mac before a GPU job is
spent: SuperLU instead of cuDSS, levels 1, cuDSS plans replaced by stubs
(``RFX_P3_SMOKE_ALLOWED_GB`` fakes the host limit to take the "not run"
branch).
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _gpu_lane_lib as lib                                        # noqa: E402
from lane_p0_probe import forward, host_memory, load_il             # noqa: E402

GB = 2.0 ** 30
SMOKE = os.environ.get("RFX_P3_SMOKE", "") == "1"
BE = "superlu" if SMOKE else "cudss"


def _read(path: str) -> str | None:
    try:
        return pathlib.Path(path).read_text().strip()
    except Exception:
        return None


def cgroup_memory() -> dict[str, Any]:
    """The container's memory limit and usage in GB (cgroup v2, else v1).

    ``limit_gb`` is ``None`` when the cgroup says "max" (no limit) or no
    cgroup file is readable; then ``/proc/meminfo`` is all there is.
    """
    out: dict[str, Any] = {"version": None, "limit_gb": None}
    v2 = _read("/sys/fs/cgroup/memory.max")
    if v2 is not None:
        out["version"] = 2
        out["memory.max"] = v2
        out["limit_gb"] = None if v2 == "max" else int(v2) / GB
        for f in ("memory.current", "memory.peak"):
            v = _read(f"/sys/fs/cgroup/{f}")
            if v is not None and v.isdigit():
                out[f + "_gb"] = int(v) / GB
        return out
    v1 = _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1 is not None and v1.isdigit():
        out["version"] = 1
        out["memory.limit_in_bytes"] = v1
        lim = int(v1)
        out["limit_gb"] = None if lim >= 2 ** 60 else lim / GB
        for f in ("memory.usage_in_bytes", "memory.max_usage_in_bytes"):
            v = _read(f"/sys/fs/cgroup/memory/{f}")
            if v is not None and v.isdigit():
                out[f.replace("_in_bytes", "") + "_gb"] = int(v) / GB
    return out


def host_record() -> dict[str, Any]:
    rec: dict[str, Any] = {"meminfo": host_memory(), "cgroup": cgroup_memory()}
    for name, cmd in (("free_g", ["free", "-g"]),
                      ("nvidia_smi", ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                                      "--format=csv,noheader"])):
        try:
            rec[name] = subprocess.run(cmd, capture_output=True, text=True,
                                       timeout=30).stdout.strip()
        except Exception as exc:
            rec[name] = f"unavailable: {type(exc).__name__}: {exc}"
    lim = rec["cgroup"].get("limit_gb")
    total = rec["meminfo"].get("MemTotal")
    rec["process_limit_gb"] = min(v for v in (lim, total) if v is not None) \
        if (lim is not None or total is not None) else None
    return rec


def free_pool() -> dict[str, float]:
    """Return cupy's cached blocks to the driver and report the device state.

    ``cudaMemGetInfo`` cannot see the difference between memory a library is
    USING and memory cupy's pool is holding for reuse -- and nvmath hands
    cuDSS the operand package's allocator, so a freed factor goes back to
    that pool, not to the driver. Without this call the device "in use"
    reading after one factorisation is the high-water mark of everything
    before it (measured: the first P3 run reported 12.46 GB in-core and
    12.49 GB hybrid at level 3, i.e. the pool, not the hybrid factor)."""
    from rfx.fdfd import _cudss
    try:
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    return _cudss.device_memory()


@contextmanager
def device_sampler(period: float = 0.25) -> Iterator[dict[str, float]]:
    """Peak device memory in use (``cudaMemGetInfo``: every allocation on the
    card, cuDSS's included) and peak cupy-pool bytes, sampled on a side thread
    while the body runs. Call :func:`free_pool` first: the reading is the
    whole process's, so anything the pool still holds counts."""
    from rfx.fdfd import _cudss
    box = {"peak_used_gb": 0.0, "peak_pool_used_gb": 0.0, "peak_pool_total_gb": 0.0,
           "samples": 0.0}
    stop = threading.Event()

    def run() -> None:
        while not stop.is_set():
            d = _cudss.device_memory()
            box["peak_used_gb"] = max(box["peak_used_gb"], d["used"])
            box["peak_pool_used_gb"] = max(box["peak_pool_used_gb"], d["pool_used"])
            box["peak_pool_total_gb"] = max(box["peak_pool_total_gb"], d["pool_total"])
            box["samples"] += 1
            stop.wait(period)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    try:
        yield box
    finally:
        stop.set()
        t.join(timeout=5)


@contextmanager
def hybrid(limit: str | None) -> Iterator[None]:
    """``RFX_FDFD_CUDSS_HYBRID`` / ``_LIMIT`` for the body only (``None``:
    in-core). The factor cache is cleared on both sides: it is keyed on the
    matrix and the backend, not on the memory mode."""
    import rfx.fdfd.linear_solve as ls
    keep = {k: os.environ.get(k) for k in ("RFX_FDFD_CUDSS_HYBRID", "RFX_FDFD_CUDSS_HYBRID_LIMIT")}
    os.environ["RFX_FDFD_CUDSS_HYBRID"] = "0" if limit is None else "1"
    if limit is None:
        os.environ.pop("RFX_FDFD_CUDSS_HYBRID_LIMIT", None)
    else:
        os.environ["RFX_FDFD_CUDSS_HYBRID_LIMIT"] = limit
    ls.clear_factor_cache()
    free_pool()
    try:
        yield
    finally:
        ls.clear_factor_cache()
        free_pool()
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def factor_probe(d: Any, r: Any, c: Any, b: Any) -> tuple[dict[str, Any], Any]:
    """One cuDSS factorisation + solve of the DUT system, then the same solve
    again (a cache hit: triangular solves only), under the current mode."""
    import jax.numpy as jnp
    import numpy as np

    from rfx.fdfd import _cudss
    import rfx.fdfd.linear_solve as ls
    ls.factor_cache_size(1)
    ls.clear_factor_cache()
    jd, jb = jnp.asarray(d), jnp.asarray(b)
    base = free_pool()
    with device_sampler() as dev:
        t0 = time.time()
        x = np.asarray(ls.sparse_solve(jd, r, c, jb, backend=BE))
        t1 = time.time()
        np.asarray(ls.sparse_solve(jd, r, c, jb, backend=BE))
        t2 = time.time()
    stats = lib.last_factor_stats()
    a = lib._csr(d, r, c, int(b.shape[0]))
    rec = {"cold_seconds": t1 - t0, "solve_seconds": t2 - t1,
           "factor_seconds": (t1 - t0) - (t2 - t1),
           "residual": float(np.linalg.norm(a @ x - b) / np.linalg.norm(b)),
           "device_peak_used_gb": dev["peak_used_gb"],
           "device_peak_pool_used_gb": dev["peak_pool_used_gb"],
           "device_peak_pool_total_gb": dev["peak_pool_total_gb"],
           "device_baseline_used_gb": base["used"], "device_samples": int(dev["samples"]),
           "hybrid_memory": _cudss.hybrid_memory(),
           "hybrid_device_memory_limit": _cudss.hybrid_device_memory_limit()}
    rec.update(stats)
    ls.clear_factor_cache()
    ls.factor_cache_size(0)
    return rec, x


def value_and_grad(il: Any, model: Any) -> dict[str, Any]:
    from rfx.fdfd import _cudss
    import rfx.fdfd.linear_solve as ls
    ls.factor_cache_size(0)
    base = free_pool()
    host0 = cgroup_memory()
    with device_sampler() as dev:
        v = lib.value_and_grad_level(il, model, BE)
    v["device_peak_used_gb"] = dev["peak_used_gb"]
    v["device_peak_pool_used_gb"] = dev["peak_pool_used_gb"]
    v["device_baseline_used_gb"] = base["used"]
    v["cgroup_before"] = host0
    v["hybrid_memory"] = _cudss.hybrid_memory()
    v["hybrid_device_memory_limit"] = _cudss.hybrid_device_memory_limit()
    v["host_memory_gb"] = host_memory()
    v["cgroup"] = cgroup_memory()
    return v


def plans(d: Any, r: Any, c: Any, b: Any, limit: str) -> dict[str, Any]:
    from rfx.fdfd import _cudss
    m = int(b.shape[1]) if b.ndim > 1 else 1
    out: dict[str, Any] = {}
    for name, lim in (("in_core", None), ("hybrid", limit)):
        with hybrid(lim):
            if SMOKE:       # no cuDSS on the Mac: the fields, with stand-in numbers
                out[name] = {"n": int(b.shape[0]), "permanent_device_memory_gb": 0.5,
                             "permanent_host_memory_gb": 0.01, "peak_host_memory_gb": 0.02,
                             "hybrid_min_device_memory_gb": 0.1, "smoke": True}
            else:
                out[name] = _cudss.plan_estimate(d, r, c, int(b.shape[0]), m=m)
    return out


def compare(ref: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """``other`` against ``ref``: the relative difference of L_dut, of each
    gradient component, and of the gradient VECTOR (the metric the linear-solve
    tests use for "the same gradient": component ratios blow up where a
    component is small)."""
    import numpy as np
    lc, lo = ref["L_dut"], other["L_dut"]
    gc, go = np.asarray(ref["grad"]), np.asarray(other["grad"])
    return {"rel_L": abs(lo - lc) / abs(lc),
            "rel_grad": [abs(a - b) / abs(b) for a, b in zip(go, gc)],
            "rel_grad_worst": float(np.max(np.abs(go - gc) / np.abs(gc))),
            "rel_grad_l2": float(np.linalg.norm(go - gc) / np.linalg.norm(gc))}


def h1_gate(incore: dict[str, Any], hyb: dict[str, Any], tol: float,
            control: dict[str, Any] | None = None) -> dict[str, Any]:
    """Gate H1, plus its CONTROL: the same in-memory computation run twice.

    cuDSS's triangular solve is not bit-reproducible across calls, and a
    value_and_grad here is six factorisations and twelve solves, so the
    in-core-vs-in-core difference is the floor any hybrid-vs-in-core number
    has to be read against."""
    out = {"L_in_core": incore["L_dut"], "L_hybrid": hyb["L_dut"],
           "grad_in_core": incore["grad"], "grad_hybrid": hyb["grad"],
           **compare(incore, hyb), "tolerance": tol,
           "slowdown_value_and_grad": hyb["value_and_grad_seconds"]
           / incore["value_and_grad_seconds"]}
    if control is not None:
        out["control_in_core_repeat"] = {
            "L": control["L_dut"], "grad": control["grad"],
            "seconds": control["value_and_grad_seconds"], **compare(incore, control)}
    out["passed"] = bool(out["rel_L"] <= tol and out["rel_grad_l2"] <= tol)
    return out


def main() -> int:
    import jax
    jax.config.update("jax_enable_x64", True)
    import numpy as np

    from rfx.fdfd import _cudss
    import rfx.fdfd.linear_solve as ls
    lane = os.environ.get("RFX_P_LANE_NAME", "P3")
    rec = lib.Recorder(os.environ.get("RFX_P_JSON", "p3_ladder.json"), lane)
    rec["env"] = lib.env_record()
    blocks = os.environ.get("RFX_P3_BLOCKS", "host h1 m5").split()
    wall = float(os.environ.get("RFX_P_WALL", "10"))
    h1_level = int(os.environ.get("RFX_P3_H1_LEVEL", "1" if SMOKE else "3"))
    h1_limit = os.environ.get("RFX_P3_H1_LIMIT", "6GiB")
    m5_level = int(os.environ.get("RFX_P3_M5_LEVEL", "1" if SMOKE else "5"))
    m5_limit = os.environ.get("RFX_P3_M5_LIMIT", "40GiB")
    margin = float(os.environ.get("RFX_P3_HOST_MARGIN_GB", "4"))
    deadline = rec.t0 + float(os.environ.get("RFX_P_DEADLINE_S", "6000"))
    # the P1 lanes' setting: one factor resident (cuDSS refactors A^T anyway)
    ls.factor_cache_size(0)
    rec["config"] = {"backend": BE, "smoke": SMOKE, "blocks": blocks, "wall_w": wall,
                     "h1_level": h1_level, "h1_limit": h1_limit,
                     "m5_level": m5_level, "m5_limit": m5_limit,
                     "host_margin_gb": margin, "deadline_s": deadline - rec.t0,
                     "factor_cache_size": 0, "free_forward": _cudss.free_forward(),
                     "h1_tolerance": 1e-9}
    if not ls.backend_available(BE):
        rec["blocked"] = {"backend": "cudss", "reason": _cudss.unavailable_reason("cudss")}
        return 3
    # validate both limits before anything expensive (a typo fails here)
    _cudss.parse_memory_limit(h1_limit)
    _cudss.parse_memory_limit(m5_limit)
    il = load_il()
    mc = il.METAL_CELLS
    rec["fixture"] = {"freq": il.FREQ, "sigma": il.SIGMA, "metal_cells": mc, "wall_w": wall}
    total_dev = _cudss.device_memory()["total"] * GB or 48.0 * GB     # 48: smoke only

    def log(msg: str) -> None:
        print(msg, flush=True)

    if "host" in blocks:
        rec["host"] = host_record()
        log(f"host: {rec['host']['free_g']}\n  cgroup {rec['host']['cgroup']}")

    if "h1" in blocks:
        out: dict[str, Any] = {"level": h1_level, "limit": h1_limit,
                               "limit_gb": _cudss.memory_limit_bytes(h1_limit, int(total_dev)) / GB}
        rec["h1"] = out
        try:
            model = il.build(il.spec_new(h1_level, wall, metal_cells=mc))
            out["grid"] = il.model_record(model)
            d, r, c, b = lib.dut_system_at(None, model, il.FREQ, il.SIGMA)
            out["plans"] = plans(d, r, c, b, h1_limit)
            fac = out["plans"]["in_core"].get("permanent_device_memory_gb")
            hmin = out["plans"]["hybrid"].get("hybrid_min_device_memory_gb")
            out["limit_below_in_core_factor"] = bool(fac is not None and out["limit_gb"] < fac)
            out["limit_at_least_hybrid_min"] = bool(hmin is not None and out["limit_gb"] >= hmin)
            log(f"  h1 plans: in-core {fac} GB, hybrid min {hmin} GB, limit {out['limit_gb']:.2f} GB")
            rec["h1"] = out
            probe: dict[str, Any] = {}
            xs = {}
            probe_runs = [("in_core", None), ("hybrid", h1_limit)]
            if os.environ.get("RFX_P3_PROBE_CONTROL", "1") == "1":
                # the control for the SOLUTION VECTOR: the same in-core solve
                # again. cuDSS's reordering and its triangular solve are not
                # bit-reproducible across calls, so hybrid-vs-in-core on x has
                # to be read against in-core-vs-in-core on x, exactly as the
                # value_and_grad comparison is
                probe_runs.append(("in_core_repeat", None))
            for name, lim in probe_runs:
                with hybrid(lim):
                    probe[name], xs[name] = factor_probe(d, r, c, b)
                log(f"  h1 probe {name}: factor {probe[name]['factor_seconds']:.1f} s solve "
                    f"{probe[name]['solve_seconds']:.2f} s residual {probe[name]['residual']:.3e} "
                    f"device peak {probe[name]['device_peak_used_gb']:.2f} GB")
                out["probe"] = probe
                rec["h1"] = out
            scale = float(np.max(np.abs(xs["in_core"])))
            probe["solution_rel_diff"] = float(np.max(np.abs(xs["hybrid"] - xs["in_core"]))) / scale
            if "in_core_repeat" in xs:
                probe["solution_rel_diff_control"] = float(
                    np.max(np.abs(xs["in_core_repeat"] - xs["in_core"]))) / scale
                log(f"  h1 probe solution: hybrid {probe['solution_rel_diff']:.3e} vs control "
                    f"(in-core twice) {probe['solution_rel_diff_control']:.3e}")
            probe["factor_slowdown"] = probe["hybrid"]["factor_seconds"] / probe["in_core"]["factor_seconds"]
            probe["solve_slowdown"] = probe["hybrid"]["solve_seconds"] / probe["in_core"]["solve_seconds"]
            del d, r, c, b, xs
            out["probe"] = probe
            rec["h1"] = out
            vg: dict[str, Any] = {}
            runs = [("in_core", None), ("hybrid", h1_limit)]
            if os.environ.get("RFX_P3_CONTROL", "1") == "1":
                runs.append(("in_core_repeat", None))
            for name, lim in runs:
                with hybrid(lim):
                    vg[name] = value_and_grad(il, model)
                out["value_and_grad"] = vg
                rec["h1"] = out
            out["gate"] = h1_gate(vg["in_core"], vg["hybrid"], 1e-9,
                                  vg.get("in_core_repeat"))
            ctl = (out["gate"].get("control_in_core_repeat") or {}).get("rel_grad_l2")
            log(f"  H1: rel L {out['gate']['rel_L']:.3e} rel grad (l2) "
                f"{out['gate']['rel_grad_l2']:.3e} worst component "
                f"{out['gate']['rel_grad_worst']:.3e} control (in-core twice) "
                f"{ctl if ctl is None else format(ctl, '.3e')} slowdown "
                f"{out['gate']['slowdown_value_and_grad']:.2f}x passed {out['gate']['passed']}")
            del model
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
            out["device_memory_gb_at_error"] = _cudss.device_memory()
            out["cgroup_at_error"] = cgroup_memory()
            log(f"  h1 FAILED: {out['error']}")
        ls.clear_factor_cache()
        rec["h1"] = out

    if "m5" in blocks and time.time() < deadline:
        m = m5_level
        lim_gb = _cudss.memory_limit_bytes(m5_limit, int(total_dev)) / GB
        feas: dict[str, Any] = {"level": m, "limit": m5_limit, "limit_gb": lim_gb}
        rec["m5_feasibility"] = feas
        levels: dict[str, Any] = {}
        try:
            model = il.build(il.spec_new(m, wall, metal_cells=mc))
            grid = il.model_record(model)
            feas["grid"] = grid
            d, r, c, b = lib.dut_system_at(None, model, il.FREQ, il.SIGMA)
            feas["plans"] = plans(d, r, c, b, m5_limit)
            del d, r, c, b
            fac = feas["plans"]["in_core"].get("permanent_device_memory_gb") or float("nan")
            hp = feas["plans"]["hybrid"]
            host_est = max(hp.get("permanent_host_memory_gb") or 0.0,
                           hp.get("peak_host_memory_gb") or 0.0)
            # cuDSS: "Factors L and U (together with all necessary internal
            # arrays) must fit into the host memory" -- the factor is at least
            # its in-core size, whatever the hybrid estimate says
            need = max(fac, host_est)
            hr = host_record()
            rss = float(hr["cgroup"].get("memory.current_gb")
                        or hr["cgroup"].get("memory.usage_gb")
                        or hr["meminfo"].get("peak_rss") or 0.0)
            allowed = hr["process_limit_gb"]
            if SMOKE:     # (macOS ru_maxrss is bytes, not kB: the RSS is meaningless here)
                allowed = float(os.environ.get("RFX_P3_SMOKE_ALLOWED_GB", "1000"))
                rss = 0.0
            feas.update({"host": hr, "in_core_factor_gb": fac,
                         "hybrid_plan_host_estimate_gb": host_est,
                         "host_needed_gb": need, "host_in_use_gb": rss,
                         "host_allowed_gb": allowed, "margin_gb": margin,
                         "hybrid_min_device_memory_gb": hp.get("hybrid_min_device_memory_gb"),
                         # cuDSS gives a DIFFERENT hybrid minimum depending on
                         # whether hybrid mode was on when it planned; both are
                         # recorded, and the feasibility test uses neither (the
                         # blocker is host memory)
                         "hybrid_min_device_memory_gb_in_core_plan": (
                             (feas["plans"].get("in_core") or {}).get("hybrid_min_device_memory_gb")),
                         "limit_at_least_hybrid_min": bool(
                             (hp.get("hybrid_min_device_memory_gb") or 0.0) <= lim_gb)})
            fits = bool(allowed is not None and need + rss + margin <= allowed)
            feas["fits"] = fits
            rec["m5_feasibility"] = feas
            log(f"  m5: factor {fac:.2f} GB, hybrid host estimate {host_est:.2f} GB, "
                f"in use {rss:.2f} GB, allowed {allowed} GB -> fits {fits}")
            if not fits:
                feas["not_run"] = (
                    f"host memory: the level-{m} factor needs >= {need:.2f} GB of host memory in "
                    f"hybrid mode (in-core factor {fac:.2f} GB, cuDSS hybrid host estimate "
                    f"{host_est:.2f} GB) + {rss:.2f} GB in use + {margin:g} GB margin, and this "
                    f"container may hold {allowed} GB (cgroup limit "
                    f"{hr['cgroup'].get('limit_gb')} GB; node MemTotal "
                    f"{hr['meminfo'].get('MemTotal')} GB)")
                rec["m5_feasibility"] = feas
                log(f"  m5 NOT RUN: {feas['not_run']}")
            else:
                lv: dict[str, Any] = {"m": m, "metal_cells": mc, "hybrid_memory": True,
                                      "hybrid_device_memory_limit": m5_limit, "grid": grid}
                levels[str(m)] = lv
                rec["levels"] = levels
                with hybrid(m5_limit):
                    with device_sampler() as dev:
                        lv["forward"] = forward(il, model, BE)
                    lv["forward"]["device_peak_used_gb"] = dev["peak_used_gb"]
                    lv["L_dut"] = lv["forward"]["L_dut"]
                    lv["host_memory_gb"] = host_memory()
                    rec["levels"] = levels
                    log(f"  m5 forward: L_dut={lv['L_dut'] * 1e12:.4f} pH "
                        f"({lv['forward']['seconds']:.0f} s)")
                    if time.time() < deadline:
                        v = value_and_grad(il, model)
                        lv.update({k: val for k, val in v.items() if k != "backend"})
                        lv["rel_L_forward_vs_value_and_grad"] = abs(
                            v["L_dut"] - lv["forward"]["L_dut"]) / abs(lv["forward"]["L_dut"])
                        rec["levels"] = levels
                    else:
                        lv["grad_skipped"] = "deadline"
        except Exception as exc:
            feas["error"] = f"{type(exc).__name__}: {exc}"
            feas["device_memory_gb_at_error"] = _cudss.device_memory()
            feas["cgroup_at_error"] = cgroup_memory()
            if levels.get(str(m)) is not None:
                levels[str(m)]["error"] = feas["error"]
                rec["levels"] = levels
            log(f"  m5 FAILED: {feas['error']}")
        ls.clear_factor_cache()
        rec["m5_feasibility"] = feas
    rec["host_end"] = host_record()
    rec.dump()
    log(f"{lane} done in {rec['seconds']:.0f} s -> {rec.path}")
    h1 = (rec.get("h1") or {}).get("gate") or {}
    return 0 if h1.get("passed") is not None else 1


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.exit(main())
