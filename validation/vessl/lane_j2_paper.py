"""J2, the paper-geometry lane: the level the CPU study built but could not solve.

``validation/fdfd/rfic_spiral.py`` (study H) takes the 2.4 GHz LC-VCO
paper's geometry -- 3 turns, ``r_out`` 218 um, ``W`` 30 um, ``S`` 14 um,
TopMetal2 3 um -- as a SQUARE spiral on an SG13G2-like ``SmallStack``, and
runs part A (100 MHz, uniform-current volumetric metal ``sigma`` = 3e6,
vacuum dielectrics, ``pad_cells`` 4) at ``base_dx`` = W and W/2. Its own
module doc records why that is the end of the CPU road and corrects the
"3 cells across W ~ 190k unknowns" estimate this lane was commissioned
with: at a 14 um turn-to-turn gap next to a 30 um strip the mesher's
mandatory edge lines make ``base_dx`` = W and W/2 the SAME grid (2 cells
across the strip, N = 85785 / 83820), and the next grid it will actually
give is ``base_dx`` = W/3 -> 5 um cells, SIX across the strip, and
N = 509580. So this lane runs

* a CONTROL at the study's own level (``RFX_LANE_DIVS`` default "1"): the
  same L at 100 MHz through cuDSS, against the bridge-corrected referee
  and against study H's own recorded value if its JSON is in the tarball;
* ``plan_estimate`` at W/3 (N = 509580): cuDSS's reordering and symbolic
  factorisation only, which answers "does this fit in 24 GB" with a number
  instead of an extrapolation;
* and, if that estimate fits the configuration, the forward solve there.

The geometry, the stack and the referee come from ``rfic_spiral.py``
itself (imported by path, never edited) so that the two studies cannot
drift; if that file is not in the tarball the lane reports that and stops
rather than inventing a second set of conventions.
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

RFIC = lib.REPO / "validation" / "fdfd" / "rfic_spiral.py"
RFIC_JSON = lib.REPO / "validation" / "fdfd" / "rfic_spiral.json"


def _load_rfic() -> Any:
    import importlib.util
    spec = importlib.util.spec_from_file_location("rfic_spiral_lane", RFIC)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rfic_spiral_lane"] = mod
    spec.loader.exec_module(mod)
    return mod


def _floats(name: str, default: str) -> list[float]:
    return [float(v) for v in os.environ.get(name, default).split()]


def main() -> int:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from rfx.fdfd import _cudss, spiral as sm
    import rfx.fdfd.linear_solve as ls
    rec = lib.Recorder("j2_paper.json", "J2")
    rec["env"] = lib.env_record()
    backend = os.environ.get("RFX_FDFD_BACKEND", "cudss")
    divs = _floats("RFX_LANE_DIVS", "1")
    plan_divs = _floats("RFX_LANE_PLAN_DIVS", "3")
    deadline = rec.t0 + float(os.environ.get("RFX_LANE_DEADLINE_S", "6300"))
    cache = int(os.environ.get("RFX_LANE_FACTOR_CACHE", "0"))
    if cache:
        ls.factor_cache_size(cache)
    rec["config"] = {"backend": backend, "divs": divs, "plan_divs": plan_divs,
                     "factor_cache_size": ls.factor_cache_size(),
                     "hybrid_memory": _cudss.hybrid_memory(),
                     "free_forward": _cudss.free_forward(),
                     "deadline_seconds": deadline - rec.t0}
    if not RFIC.exists():
        rec["blocked"] = {"what": "validation/fdfd/rfic_spiral.py not in the tarball",
                          "why": "the paper geometry, stack and referee are defined there"}
        rec.dump()
        return 3
    if backend != "superlu" and not ls.backend_available(backend):
        rec["blocked"] = {"backend": backend,
                          "reason": _cudss.unavailable_reason(backend)}
        rec.dump()
        return 3

    h = _load_rfic()
    rec["fixture"] = {
        "n_turns": h.N_TURNS, "r_out": h.R_OUT, "spacing": h.SPACING, "width": h.WIDTH,
        "lead": h.LEAD, "margin": h.MARGIN, "t_m2": h.T_M2, "t_m1": h.T_M1,
        "t_via": h.T_VIA, "t_si": h.T_SI, "t_ox_low": h.T_OX_LOW,
        "t_ox_high": h.T_OX_HIGH, "t_air": h.T_AIR, "base_dz": h.BASE_DZ,
        "metal_cells": h.METAL_CELLS, "pad_cells": h.PAD_CELLS,
        "pad_cells_z": h.PAD_CELLS_Z, "pad_ratio": h.PAD_RATIO,
        "freq": h.FREQ_A, "sigma_volumetric": h.SIGMA_VOL,
        "eps_dielectrics": "vacuum (eps=False), part A's protocol",
        "source": "validation/fdfd/rfic_spiral.py (study H), imported unmodified",
    }
    try:
        rec["referee"] = h.referee_block()
        ref = rec["referee"]["deembedded"] if "deembedded" in rec["referee"] \
            else rec["referee"].get("total")
    except Exception as exc:
        rec["referee"] = {"error": f"{type(exc).__name__}: {exc}"}
        ref = None
    if RFIC_JSON.exists():
        import json
        try:
            hj = json.loads(RFIC_JSON.read_text())
            rec["cpu_study_H"] = {"A": hj.get("A", {}).get("levels", {}),
                                  "referee": hj.get("referee", {})}
            if ref is None:
                ref = hj.get("referee", {}).get("deembedded")
        except Exception as exc:
            rec["cpu_study_H"] = {"error": f"{type(exc).__name__}: {exc}"}
    rec["referee_deembedded"] = ref

    plans: dict[str, Any] = {}
    rec["plan_estimates"] = plans
    for div in plan_divs:
        key = f"{div:g}"
        if time.time() > deadline:
            plans[key] = {"skipped": "deadline"}
            rec.dump()
            continue
        try:
            model = h.build_level(div, eps=False)
            grid = h.model_record(model)
            data, rows, cols, rhs = lib.dut_system_at(h, model, h.FREQ_A, h.SIGMA_VOL)
            p = _cudss.plan_estimate(data, rows, cols, int(rhs.shape[0]),
                                     m=int(rhs.shape[1]) if rhs.ndim > 1 else 1)
            p["grid"] = grid
            plans[key] = p
            print(f"  plan W/{key}: N={grid['n_unknowns']} "
                  f"permanent {p.get('permanent_device_memory_gb', float('nan')):.2f} GB "
                  f"peak {p.get('peak_device_memory_gb', float('nan')):.2f} GB "
                  f"({p.get('plan_seconds', float('nan')):.1f} s)", flush=True)
            del model, data, rows, cols, rhs
        except Exception as exc:
            plans[key] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  plan W/{key} FAILED: {plans[key]['error']}", flush=True)
        rec.dump()

    levels: dict[str, Any] = {}
    rec["levels"] = levels
    for div in divs:
        key = f"{div:g}"
        if time.time() > deadline:
            levels[key] = {"skipped": "deadline"}
            rec.dump()
            continue
        t0 = time.time()
        model = h.build_level(div, eps=False)
        lv: dict[str, Any] = {"div": div, "base_dx": h.WIDTH / div,
                              "grid": h.model_record(model),
                              "build_seconds": time.time() - t0, "backend": backend}
        levels[key] = lv
        rec.dump()
        print(f"  W/{key}: N={lv['grid']['n_unknowns']} "
              f"{lv['grid'].get('cells_across_width')} cells across W", flush=True)
        try:
            ls.clear_factor_cache()
            t0 = time.time()
            with ls.default_backend(backend):
                res = sm.solve_spiral(model, h.FREQ_A, sigma_volumetric=h.SIGMA_VOL)
                lv["L_diff"] = float(jnp.real(res.L_diff))
                lv["L_raw"] = float(jnp.real(res.L_raw))
                lv["Q_diff"] = float(jnp.real(res.Q_diff))
            lv["solve_seconds"] = time.time() - t0
            lv["device_memory_gb"] = _cudss.device_memory()
            lv.update(lib.last_factor_stats())
            if ref:
                lv["referee_deembedded"] = ref
                lv["gap"] = lv["L_diff"] / ref - 1.0
            print(f"    L_diff={lv['L_diff'] * 1e12:.2f} pH"
                  + (f"  gap {100 * lv['gap']:+.2f} %" if ref else "")
                  + f"  ({lv['solve_seconds']:.0f} s)", flush=True)
        except Exception as exc:
            lv["error"] = f"{type(exc).__name__}: {exc}"
            lv["device_memory_gb_at_error"] = _cudss.device_memory()
            print(f"    FAILED: {lv['error']}", flush=True)
        ls.clear_factor_cache()
        rec.dump()
        del model

    # --- the cuSOLVER fallback, at scale, once, last ---------------------
    # The task asks which backend actually works at N ~ 100-200k. cuDSS is
    # answered everywhere above; this is the one bounded measurement of the
    # fallback on a real system of that size: ONE right-hand-side column
    # through cupy's spsolve, i.e. one cuSOLVER csrlsvqr sparse QR. It is
    # last because a sparse QR of a 3-D curl-curl operator can be much more
    # expensive than its LU (QR fill >= LU fill) and may simply not fit;
    # whatever happens is recorded, including the exception.
    cus_div = os.environ.get("RFX_LANE_CUSOLVER_DIV", "").strip()
    if cus_div and time.time() < deadline and ls.backend_available("cusolver"):
        block: dict[str, Any] = {"div": float(cus_div), "backend": "cusolver",
                                 "what": ("one (n, 1) solve of the paper geometry's DUT "
                                          "system through cuSOLVER csrlsvqr, against "
                                          "cuDSS on the same system")}
        rec["cusolver_probe"] = block
        try:
            import numpy as np

            import jax.numpy as jnp2
            model = h.build_level(float(cus_div), eps=False)
            data, rows, cols, rhs = lib.dut_system_at(h, model, h.FREQ_A, h.SIGMA_VOL)
            b1 = np.ascontiguousarray(rhs[:, 0] if rhs.ndim > 1 else rhs)
            block["n"] = int(b1.shape[0])
            block["nnz"] = int(data.size)
            for be in ("cudss", "cusolver"):
                ls.clear_factor_cache()
                t0 = time.time()
                x = np.asarray(ls.sparse_solve(jnp2.asarray(data), rows, cols,
                                               jnp2.asarray(b1), backend=be))
                dt = time.time() - t0
                import scipy.sparse as sp
                a = sp.coo_matrix((data, (rows, cols)),
                                  shape=(block["n"], block["n"])).tocsr()
                a.sum_duplicates()
                block[be] = {"seconds": dt,
                             "residual": float(np.linalg.norm(a @ x - b1))
                             / float(np.linalg.norm(b1)),
                             "device_memory_gb": _cudss.device_memory()}
                print(f"  cusolver probe {be}: {dt:.1f} s residual "
                      f"{block[be]['residual']:.3e}", flush=True)
                ls.clear_factor_cache()
                rec.dump()
            del model, data, rows, cols, rhs
        except Exception as exc:
            block["error"] = f"{type(exc).__name__}: {exc}"
            block["device_memory_gb_at_error"] = _cudss.device_memory()
            print(f"  cusolver probe FAILED: {block['error']}", flush=True)
        ls.clear_factor_cache()
        rec.dump()

    rec.dump()
    print(f"J2 done in {rec['seconds']:.0f} s -> {rec.path}", flush=True)
    return 0 if any(v.get("L_diff") is not None for v in levels.values()) else 1


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.exit(main())
