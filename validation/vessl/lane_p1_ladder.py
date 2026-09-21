"""P1, the ladder lane of study P (``validation/fdfd/invariant_ladder.py``).

The level-invariant fixture (physical walls at ``RFX_P_WALL`` widths, the
physical short standard and port gap) jointly refined in x, y AND z, at the
levels ``RFX_P_LEVELS``. Per level, in this order:

* ``factor``: the DUT operator captured and factorised once through cuDSS
  (cold / warm wall time, hence the factorisation time; cuDSS's own LU
  nonzeros and device bytes; the relative residual) -- the "N, factor time,
  device GB" record the study asks for;
* ``value_and_grad`` of the de-embedded L_dut in ``(r_out, spacing,
  width)`` (six cuDSS factorisations: three fixtures, forward + adjoint --
  cuDSS has no transposed solve, so the adjoint refactors A^T);
* the RC twin (``RFX_P_TWIN``, default on): a forward solve at 10 MHz,
  2e7 S/m -- the same skin depth as 100 MHz / 2e6, a tenth of the resistance;
* then, after every level, FD4 of ``dL/dwidth`` at ``RFX_P_FD_LEVEL``
  through cuDSS (four more three-fixture solves, the study's 1 % step);
* and the thru probe at ``RFX_P_THRU_LEVELS``: the SAME invariant fixture
  with a straight M2 bar identical to the short standard's bridge as the
  DUT, whose de-embedded L is the post-short residual.

``RFX_P_HYBRID_LEVELS`` switches cuDSS's host-backed memory mode on for the
listed levels only. Everything is written after every block.
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
from lane_p0_probe import forward, host_memory, load_il             # noqa: E402


def _ints(name: str, default: str) -> list[int]:
    return [int(v) for v in os.environ.get(name, default).split()]


def main() -> int:
    import jax
    jax.config.update("jax_enable_x64", True)

    from rfx.fdfd import _cudss
    import rfx.fdfd.linear_solve as ls
    lane = os.environ.get("RFX_P_LANE_NAME", "P1")
    rec = lib.Recorder(os.environ.get("RFX_P_JSON", "p1_ladder.json"), lane)
    rec["env"] = lib.env_record()
    rec["host_memory_gb"] = host_memory()
    backend = os.environ.get("RFX_FDFD_BACKEND", "cudss")
    levels = _ints("RFX_P_LEVELS", "1 2 3")
    # "mc:m" pairs: extra levels of the OTHER level-1 family (metal_cells mc),
    # run after the primary ones and stored under "xlevels"
    xlevels = [tuple(int(t) for t in v.split(":"))
               for v in os.environ.get("RFX_P_XLEVELS", "").split()]
    wall = float(os.environ.get("RFX_P_WALL", "20"))
    fd_level = os.environ.get("RFX_P_FD_LEVEL", "").strip()
    thru_levels = _ints("RFX_P_THRU_LEVELS", "")
    hybrid_levels = set(_ints("RFX_P_HYBRID_LEVELS", ""))
    deadline = rec.t0 + float(os.environ.get("RFX_P_DEADLINE_S", "4500"))
    ls.factor_cache_size(int(os.environ.get("RFX_P_FACTOR_CACHE", "1")))
    mc = int(os.environ.get("RFX_P_METAL_CELLS", "0")) or None
    rec["config"] = {"backend": backend, "levels": levels, "xlevels": xlevels, "wall_w": wall,
                     "fd_level": fd_level, "thru_levels": thru_levels,
                     "hybrid_levels": sorted(hybrid_levels),
                     "deadline_s": deadline - rec.t0,
                     "factor_cache_size": ls.factor_cache_size(),
                     "free_forward": _cudss.free_forward()}
    if backend != "superlu" and not ls.backend_available(backend):
        rec["blocked"] = {"backend": backend, "reason": _cudss.unavailable_reason(backend)}
        return 3
    il = load_il()
    rec["fixture"] = {"freq": il.FREQ, "sigma": il.SIGMA, "short_gap": il.SHORT_GAP,
                      "port_gap": il.PORT_GAP, "base_dz": il.BASE_DZ,
                      "metal_cells": mc or il.METAL_CELLS, "pad_ratio": il.PAD_RATIO,
                      "wall_w": wall, "skin_depth": il.skin_depth(il.FREQ, il.SIGMA)}
    hyb0 = os.environ.get("RFX_FDFD_CUDSS_HYBRID", "0")
    out: dict[str, Any] = {}
    xout: dict[str, Any] = {}
    rec["levels"] = out
    todo = [(m, mc or il.METAL_CELLS, out, str(m)) for m in levels]
    todo += [(m, c, xout, f"mc{c}_m{m}") for c, m in xlevels]
    for m, mcl, dest, key in todo:
        if dest is xout:
            rec["xlevels"] = xout
        if time.time() > deadline:
            dest[key] = {"skipped": "deadline"}
            rec.dump()
            continue
        os.environ["RFX_FDFD_CUDSS_HYBRID"] = "1" if m in hybrid_levels else hyb0
        lv: dict[str, Any] = {"m": m, "metal_cells": mcl, "hybrid_memory": _cudss.hybrid_memory()}
        dest[key] = lv
        try:
            t0 = time.time()
            model = il.build(il.spec_new(m, wall, metal_cells=mcl))
            lv["grid"] = il.model_record(model)
            lv["build_seconds_here"] = time.time() - t0
            rec.dump()
            print(f"  level {m}: {lv['grid']['shape']} N={lv['grid']['n_unknowns']}", flush=True)
            if os.environ.get("RFX_P_FACTOR_PROBE", "1") == "1":
                d, r, c, b = lib.dut_system_at(None, model, il.FREQ, il.SIGMA)
                lv["factor"] = lib.solve_both(d, r, c, b, backends=(backend,),
                                              reference=backend)
                del d, r, c, b
                rec.dump()
            if os.environ.get("RFX_P_GRAD", "1") == "1":
                v = lib.value_and_grad_level(il, model, backend)
            else:
                v = forward(il, model, backend)
                v["value_only"] = "RFX_P_GRAD=0: a three-fixture forward solve, no gradient"
            lv.update({k: val for k, val in v.items() if k != "backend"})
            lv["host_memory_gb"] = host_memory()
            rec.dump()
            if os.environ.get("RFX_P_TWIN", "1") == "1" and time.time() < deadline:
                # the RC twin: 10 MHz, 2e7 S/m -- the same skin depth, a ten
                # times smaller metal resistance, so the shunt-C contamination
                # of Im Z / omega (~ C R^2) drops a hundredfold
                lv["rc_twin"] = forward(il, model, backend, sigma=il.SIGMA_RC, freq=il.FREQ_RC)
                print(f"    RC twin: L={lv['rc_twin']['L_dut'] * 1e12:.4f} pH", flush=True)
                rec.dump()
            del model
        except Exception as exc:
            lv["error"] = f"{type(exc).__name__}: {exc}"
            lv["device_memory_gb_at_error"] = _cudss.device_memory()
            print(f"  level {m} FAILED: {lv['error']}", flush=True)
        ls.clear_factor_cache()
        rec.dump()
    os.environ["RFX_FDFD_CUDSS_HYBRID"] = hyb0

    if fd_level and time.time() < deadline:
        m = int(fd_level)
        lv = out.get(str(m)) or {}
        if lv.get("grad"):
            os.environ["RFX_FDFD_CUDSS_HYBRID"] = "1" if m in hybrid_levels else hyb0
            try:
                model = il.build(il.spec_new(m, wall, metal_cells=mc or il.METAL_CELLS))
                fd = lib.fd4_one_parameter(il, model, backend, 2)
                fd["ad"] = lv["grad"][2]
                fd["rel"] = abs(fd["ad"] - fd["fd"]) / abs(fd["fd"])
                lv["fd_width"] = fd
                print(f"  FD4 dL/dwidth at level {m}: AD {fd['ad']:.8e} FD {fd['fd']:.8e} "
                      f"rel {fd['rel']:.2e} ({fd['fd_seconds']:.0f} s)", flush=True)
                del model
            except Exception as exc:
                lv["fd_error"] = f"{type(exc).__name__}: {exc}"
                print(f"  FD4 FAILED: {exc}", flush=True)
            os.environ["RFX_FDFD_CUDSS_HYBRID"] = hyb0
            ls.clear_factor_cache()
            rec.dump()

    if thru_levels:
        thru: dict[str, Any] = {"bar_length": None, "levels": {}}
        rec["thru"] = thru
        # the bar spans the two lead-column footprint CENTRES of the spiral
        # (40 um); its own post then sits at x in [-5, 5] um, the same 10 um
        # wide, 10 um from each column as the spiral's
        r_out, spacing, width = il.THETA0
        a0 = r_out - 0.5 * width
        l_bar = (width + spacing) * il.N_TURNS
        thru["bar_length"] = l_bar
        thru["spiral_columns_x"] = [a0 - l_bar, a0]
        for m in thru_levels:
            if time.time() > deadline:
                break
            try:
                model = il.build(il.spec_new(m, wall, metal_cells=mc or il.METAL_CELLS,
                                             dut_kind="bar", bar_length=l_bar))
                t = forward(il, model, backend)
                t["grid"] = il.model_record(model)
                thru["levels"][str(m)] = t
                print(f"  thru level {m}: N={t['grid']['n_unknowns']} L_deembedded="
                      f"{t['L_dut'] * 1e12:+.4f} pH L_raw={t['L_raw'] * 1e12:.4f} pH", flush=True)
                del model
            except Exception as exc:
                thru["levels"][str(m)] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  thru level {m} FAILED: {exc}", flush=True)
            ls.clear_factor_cache()
            rec["thru"] = thru
    rec["host_memory_gb_end"] = host_memory()
    rec.dump()
    print(f"{lane} done in {rec['seconds']:.0f} s -> {rec.path}", flush=True)
    ok = [k for k, v in out.items() if v.get("L_dut") is not None]
    return 0 if ok else 1


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.exit(main())
