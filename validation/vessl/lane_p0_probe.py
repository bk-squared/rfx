"""P0, the probe lane of study P (``validation/fdfd/invariant_ladder.py``).

What one submission measures, in this order (each block is written to the
JSON as soon as it finishes, so a killed job keeps what it did):

``plans``     cuDSS plan-only memory estimates of the DUT operator for the
              level-invariant fixture at the configurations in
              ``RFX_P_PLANS`` (``wall_w:metal_cells:m`` triples) -- which
              levels fit a 24 GB card, which need the 48 GB one, which do
              not fit at all.
``walls``     the wall ladder at level 2 (margins ``RFX_P_WALLS`` in widths
              W, the lid the same distance above the stack): forward L_dut,
              and the chosen margin = the smallest whose change to the next
              is below 0.2 %.
``p6``        gate P6: the OLD cell-defined fixture at W/1 and W/2 with each
              level-dependence (a walls, b short standard, c vertical grid)
              switched to its physical definition one at a time, all three,
              and the new fixture.
``plateau``   the uniform-current plateau at level ``RFX_P_PLATEAU_LEVEL``
              (3 cells across W): L_dut over ``PLATEAU_SIGMAS`` at 100 MHz,
              plus the RC twin (10 MHz, 2e7 S/m: the SAME skin depth as
              100 MHz, 2e6 S/m, a ten times smaller resistance).
``ladder``    value_and_grad of L_dut at the levels ``RFX_P_LEVELS``.
``bigplans``  more plan estimates (``RFX_P_BIGPLANS``), LAST because the
              host assembly of the largest operators is the part most likely
              to exhaust the container's memory.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _gpu_lane_lib as lib                                        # noqa: E402

IL_PATH = lib.REPO / "validation" / "fdfd" / "invariant_ladder.py"


def load_il() -> Any:
    import importlib.util
    spec = importlib.util.spec_from_file_location("invariant_ladder_lane", IL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["invariant_ladder_lane"] = mod
    spec.loader.exec_module(mod)
    return mod


def host_memory() -> dict[str, float]:
    """``/proc/meminfo`` totals and this process's peak RSS, GB."""
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


def forward(il: Any, model: Any, backend: str, sigma: float | None = None,
            freq: float | None = None, theta: Any = None) -> dict[str, Any]:
    """One three-fixture forward solve: L_dut, L_raw, Re Z, time, memory."""
    import numpy as np

    from rfx.fdfd import _cudss
    from rfx.fdfd import spiral as sm
    import rfx.fdfd.linear_solve as ls
    sigma = il.SIGMA if sigma is None else sigma
    freq = il.FREQ if freq is None else freq
    ls.clear_factor_cache()
    t0 = time.time()
    with ls.default_backend(backend):
        r = sm.solve_spiral(model, freq, theta=theta, sigma_volumetric=sigma)
        rec = {"L_dut": float(r.L_diff), "L_raw": float(r.L_raw), "Q_diff": float(r.Q_diff),
               "re_z_diff": float(np.real(r.z_dut[0, 0] - r.z_dut[0, 1] - r.z_dut[1, 0]
                                          + r.z_dut[1, 1])),
               "sigma": sigma, "freq": freq}
    rec["seconds"] = time.time() - t0
    rec["device_memory_gb_after"] = _cudss.device_memory()["used"]
    rec.update(lib.last_factor_stats())
    ls.clear_factor_cache()
    return rec


def plan_one(il: Any, wall_w: float, mc: int, m: int) -> dict[str, Any]:
    from rfx.fdfd import _cudss
    t0 = time.time()
    model = il.build(il.spec_new(m, wall_w, metal_cells=mc))
    rec: dict[str, Any] = {"wall_w": wall_w, "metal_cells": mc, "m": m,
                           "grid": il.model_record(model)}
    d, r, c, b = lib.dut_system_at(None, model, il.FREQ, il.SIGMA)
    rec["assembly_seconds"] = time.time() - t0
    p = _cudss.plan_estimate(d, r, c, int(b.shape[0]), m=int(b.shape[1]) if b.ndim > 1 else 1)
    rec.update(p)
    rec["host_memory_gb"] = host_memory()
    print(f"  plan W{wall_w:g} mc{mc} m{m}: N={rec['grid']['n_unknowns']} "
          f"perm {p.get('permanent_device_memory_gb', float('nan')):.2f} GB "
          f"peak {p.get('peak_device_memory_gb', float('nan')):.2f} GB "
          f"hybrid_min {p.get('hybrid_min_device_memory_gb', float('nan')):.2f} GB", flush=True)
    del model, d, r, c, b
    return rec


def main() -> int:
    import jax
    jax.config.update("jax_enable_x64", True)

    from rfx.fdfd import _cudss
    import rfx.fdfd.linear_solve as ls
    rec = lib.Recorder("p0_probe.json", "P0")
    rec["env"] = lib.env_record()
    rec["host_memory_gb"] = host_memory()
    backend = os.environ.get("RFX_FDFD_BACKEND", "cudss")
    blocks = os.environ.get("RFX_P_BLOCKS", "plans walls p6 plateau ladder bigplans").split()
    deadline = rec.t0 + float(os.environ.get("RFX_P_DEADLINE_S", "4500"))
    ls.factor_cache_size(int(os.environ.get("RFX_P_FACTOR_CACHE", "1")))
    rec["config"] = {"backend": backend, "blocks": blocks, "deadline_s": deadline - rec.t0,
                     "factor_cache_size": ls.factor_cache_size(),
                     "free_forward": _cudss.free_forward(),
                     "hybrid_memory": _cudss.hybrid_memory()}
    if backend != "superlu" and not ls.backend_available(backend):
        rec["blocked"] = {"backend": backend, "reason": _cudss.unavailable_reason(backend)}
        return 3
    il = load_il()
    rec["fixture"] = {"freq": il.FREQ, "sigma": il.SIGMA, "short_gap": il.SHORT_GAP,
                      "port_gap": il.PORT_GAP, "base_dz": il.BASE_DZ,
                      "metal_cells": il.METAL_CELLS, "pad_ratio": il.PAD_RATIO,
                      "skin_depth": il.skin_depth(il.FREQ, il.SIGMA)}

    def triples(name: str, default: str) -> list[tuple[float, int, int]]:
        out = []
        for tok in os.environ.get(name, default).split():
            w, mc, m = tok.split(":")
            out.append((float(w), int(mc), int(m)))
        return out

    def plans_block(key: str, env: str, default: str) -> None:
        plans: dict[str, Any] = rec.get(key) or {}
        for w, mc, m in triples(env, default):
            if time.time() > deadline:
                break
            tag = f"W{w:g}_mc{mc}_m{m}"
            try:
                plans[tag] = plan_one(il, w, mc, m)
            except Exception as exc:
                plans[tag] = {"error": f"{type(exc).__name__}: {exc}",
                              "host_memory_gb": host_memory()}
                print(f"  plan {tag} FAILED: {plans[tag]['error']}", flush=True)
            ls.clear_factor_cache()
            rec[key] = plans

    chosen = float(os.environ.get("RFX_P_WALL_DEFAULT", "20"))
    if "plans" in blocks:
        plans_block("plans", "RFX_P_PLANS", "10:2:3 20:2:3 10:2:4 20:2:4 10:1:4")

    if "walls" in blocks and time.time() < deadline:
        walls: dict[str, Any] = {"level": 2, "margins_w": [], "L_dut": [], "grid": [],
                                 "runs": {}}
        rec["walls"] = walls
        for w in [float(v) for v in os.environ.get("RFX_P_WALLS", "10 20 40 80").split()]:
            if time.time() > deadline:
                break
            try:
                model = il.build(il.spec_new(2, w))
                r = forward(il, model, backend)
                r["grid"] = il.model_record(model)
                walls["runs"][f"{w:g}"] = r
                walls["margins_w"].append(w)
                walls["L_dut"].append(r["L_dut"])
                print(f"  walls {w:g} W: N={r['grid']['n_unknowns']} L={r['L_dut'] * 1e12:.4f} pH "
                      f"({r['seconds']:.0f} s)", flush=True)
                del model
            except Exception as exc:
                walls["runs"][f"{w:g}"] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  walls {w:g} FAILED: {exc}", flush=True)
            rec["walls"] = walls
        ws, ls_ = walls["margins_w"], walls["L_dut"]
        walls["rel_change_to_next"] = [ls_[i + 1] / ls_[i] - 1.0 for i in range(len(ls_) - 1)]
        ok = [ws[i] for i, v in enumerate(walls["rel_change_to_next"]) if abs(v) < il.WALL_TOL]
        if ok:
            chosen = ok[0]
        walls["chosen_margin_w"] = chosen
        walls["rule"] = ("the smallest margin whose change to the next (doubled) margin is "
                         f"below {il.WALL_TOL:g}")
        rec["walls"] = walls
    rec["chosen_margin_w"] = chosen

    if "p6" in blocks and time.time() < deadline:
        p6: dict[str, Any] = {"wall_w": chosen, "levels": {}}
        rec["p6"] = p6
        for lvl in [int(v) for v in os.environ.get("RFX_P_P6_LEVELS", "1 2").split()]:
            cases = {"old": il.spec_old(lvl), "a_walls": il.spec_old(lvl, walls=True, wall_w=chosen),
                     "b_short": il.spec_old(lvl, short=True),
                     "c_vertical": il.spec_old(lvl, vertical=True),
                     "abc": il.spec_old(lvl, walls=True, short=True, vertical=True, wall_w=chosen),
                     "new": il.spec_new(lvl, chosen)}
            out: dict[str, Any] = {}
            p6["levels"][str(lvl)] = out
            for name, spec in cases.items():
                if time.time() > deadline:
                    break
                try:
                    model = il.build(spec)
                    r = forward(il, model, backend)
                    r["grid"] = il.model_record(model)
                    out[name] = r
                    print(f"  P6 level {lvl} {name:10s}: N={r['grid']['n_unknowns']} "
                          f"L={r['L_dut'] * 1e12:.4f} pH", flush=True)
                    del model
                except Exception as exc:
                    out[name] = {"error": f"{type(exc).__name__}: {exc}"}
                    print(f"  P6 {lvl} {name} FAILED: {exc}", flush=True)
                rec["p6"] = p6

    if "plateau" in blocks and time.time() < deadline:
        lvl = int(os.environ.get("RFX_P_PLATEAU_LEVEL", "3"))
        pw = float(os.environ.get("RFX_P_PLATEAU_WALL", str(chosen)))
        pl: dict[str, Any] = {"level": lvl, "wall_w": pw, "runs": []}
        rec["plateau"] = pl
        try:
            model = il.build(il.spec_new(lvl, pw))
            pl["grid"] = il.model_record(model)
            cases = [(il.FREQ, s) for s in il.PLATEAU_SIGMAS] + [(il.FREQ_RC, il.SIGMA_RC)]
            for f, s in cases:
                if time.time() > deadline:
                    break
                r = forward(il, model, backend, sigma=s, freq=f)
                r["skin_depth"] = il.skin_depth(f, s)
                r["skin_over_width"] = r["skin_depth"] / il.WIDTH
                pl["runs"].append(r)
                print(f"  plateau f={f:.0e} sigma={s:.0e} delta={r['skin_depth'] * 1e6:.1f} um "
                      f"L={r['L_dut'] * 1e12:.4f} pH Re Z={r['re_z_diff']:.4g} ({r['seconds']:.0f} s)",
                      flush=True)
                rec["plateau"] = pl
            del model
        except Exception as exc:
            pl["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  plateau FAILED: {exc}", flush=True)
        rec["plateau"] = pl

    if "ladder" in blocks and time.time() < deadline:
        lad: dict[str, Any] = {"wall_w": chosen, "levels": {}}
        rec["ladder"] = lad
        for m in [int(v) for v in os.environ.get("RFX_P_LEVELS", "1 2").split()]:
            if time.time() > deadline:
                break
            try:
                model = il.build(il.spec_new(m, chosen))
                r = lib.value_and_grad_level(il, model, backend)
                r["grid"] = il.model_record(model)
                lad["levels"][str(m)] = r
                del model
            except Exception as exc:
                lad["levels"][str(m)] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  ladder m={m} FAILED: {exc}", flush=True)
            ls.clear_factor_cache()
            rec["ladder"] = lad

    if "bigplans" in blocks:
        plans_block("bigplans", "RFX_P_BIGPLANS", "10:2:5 10:1:5 10:1:6")
    rec["host_memory_gb_end"] = host_memory()
    rec.dump()
    print(f"P0 done in {rec['seconds']:.0f} s -> {rec.path}", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.exit(main())
