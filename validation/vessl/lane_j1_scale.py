"""J1, the scale lane: the spiral levels the Mac cannot reach.

W/4 (N = 172747) and W/6 (N = 352536) of the SAME fixture the CPU study
measured at W/1, W/2 and W/3 -- same uniform-current volumetric metal,
same pad_cells 4, same graded wall padding, same 1 % FD steps -- so the
five levels form one Richardson series. Each level is written as a block
with the CPU study's own keys (``base_dx``, ``div``, ``grid``, ``L_dut``,
``grad``, ``value_and_grad_seconds``, ...) under ``levels``, plus the
referee block copied from the CPU JSON, so
``validation/fdfd/gpu_scaling.py`` can merge them into the extended study
without editing ``spiral_convergence.py``.

Everything is env-driven so the generated YAML, not this file, decides
what one submission does (the J0 probe's measured memory estimates choose
the configuration):

``RFX_LANE_DIVS``            levels, space separated (default "4 6")
``RFX_LANE_GRAD``            "1" for value_and_grad, "0" for value only
``RFX_LANE_GRAD_MAX_DIV``    value-only above this level (default: all)
``RFX_LANE_PEC``             "1" to add the PEC fixture (gate V1b)
``RFX_LANE_FD_DIV``          level for the one-parameter FD4 (gate G2)
``RFX_LANE_FD_PARAM``        which parameter (default 2 = width)
``RFX_LANE_DEADLINE_S``      stop starting new blocks after this (default 6300)
``RFX_LANE_IR_DIV``          level for the cuDSS iterative-refinement sweep
``RFX_LANE_IR_SWEEP``        ir_num_steps values to sweep ("" = skip)
``RFX_LANE_IR_PHYSICS``      ir_num_steps values for the L_dut comparison
``RFX_FDFD_BACKEND``         the backend under test (cudss)
``RFX_FDFD_CUDSS_HYBRID``    cuDSS host-backed memory mode
``RFX_FDFD_CUDSS_FREE_FORWARD``  free A's factor when A^T's is built
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


def _floats(name: str, default: str) -> list[float]:
    return [float(v) for v in os.environ.get(name, default).split()]


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def main() -> int:
    import jax
    jax.config.update("jax_enable_x64", True)

    from rfx.fdfd import _cudss
    import rfx.fdfd.linear_solve as ls
    rec = lib.Recorder("j1_scale.json", "J1")
    rec["env"] = lib.env_record()
    backend = os.environ.get("RFX_FDFD_BACKEND", "cudss")
    divs = _floats("RFX_LANE_DIVS", "4 6")
    want_grad = _flag("RFX_LANE_GRAD", "1")
    grad_max_div = float(os.environ.get("RFX_LANE_GRAD_MAX_DIV", "1e9"))
    want_pec = _flag("RFX_LANE_PEC", "0")
    fd_div = os.environ.get("RFX_LANE_FD_DIV", "").strip()
    fd_param = int(os.environ.get("RFX_LANE_FD_PARAM", "2"))
    deadline = rec.t0 + float(os.environ.get("RFX_LANE_DEADLINE_S", "6300"))
    cache = int(os.environ.get("RFX_LANE_FACTOR_CACHE", "0"))
    if cache:
        ls.factor_cache_size(cache)
    rec["config"] = {"backend": backend, "divs": divs, "grad": want_grad,
                     "grad_max_div": grad_max_div, "pec": want_pec,
                     "fd_div": fd_div, "fd_param": fd_param,
                     "deadline_seconds": deadline - rec.t0,
                     "factor_cache_size": ls.factor_cache_size(),
                     "hybrid_memory": _cudss.hybrid_memory(),
                     "free_forward": _cudss.free_forward()}
    if backend != "superlu" and not ls.backend_available(backend):
        rec["blocked"] = {"backend": backend,
                          "reason": _cudss.unavailable_reason(backend)}
        rec.dump()
        return 3

    study = lib.load_study()
    cpu = lib.cpu_study()
    # the referee and the fixture are the CPU study's, copied verbatim: the
    # GPU levels are compared against the SAME numbers, not a re-derivation
    rec["fixture"] = cpu["fixture"]
    rec["referee"] = cpu["referee"]
    rec["cpu_levels"] = {k: {kk: v[kk] for kk in
                             ("base_dx", "div", "L_dut", "L_pec", "grad", "fd",
                              "value_and_grad_seconds")
                             if kk in v} | {"n_unknowns": v["grid"]["n_unknowns"]}
                         for k, v in cpu["levels"].items()}
    # --- the iterative-refinement sweep (gate G1's open question) --------
    # J0 measured SuperLU and cuDSS agreeing to 2.395e-03 on the W/3
    # solution VECTOR with residuals 1.787e-08 and 3.841e-09. This asks
    # whether cuDSS's iterative refinement closes that, and what it costs.
    # Cheap: one host factorisation (139 s) plus a second of GPU time per
    # step. RFX_LANE_IR_SWEEP="" turns it off.
    ir_div = os.environ.get("RFX_LANE_IR_DIV", "3").strip()
    ir_steps = [int(v) for v in os.environ.get("RFX_LANE_IR_SWEEP", "0 1 2 4").split()]
    if ir_div and ir_steps:
        try:
            m = study.build_level(float(ir_div))
            d, r, c, b = lib.dut_system(study, m)
            print(f"  IR sweep at W/{ir_div}: N={b.shape[0]}", flush=True)
            rec["ir_sweep"] = lib.ir_sweep(d, r, c, b, tuple(ir_steps))
            rec["ir_sweep"]["div"] = float(ir_div)
            del m, d, r, c, b
        except Exception as exc:
            rec["ir_sweep"] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  IR sweep FAILED: {rec['ir_sweep']['error']}", flush=True)
        rec.dump()

    # --- what the refinement does to the PHYSICS, not just to x ----------
    # The solution vector of this operator is only determined to ~1e-3 by a
    # 1e-8-residual solve (see ir_sweep). L_dut is not the solution vector:
    # it is two port voltages and two port currents read off it, and the
    # question the study actually needs answered is whether THAT moves. One
    # cuDSS value_and_grad at W/3 costs six factorisations at ~1 s each, so
    # this asks it at several refinement settings and compares each against
    # the CPU study's recorded W/3 level.
    ir_phys = [int(v) for v in os.environ.get("RFX_LANE_IR_PHYSICS", "0 2 4").split()]
    if ir_div and ir_phys:
        cpu3 = cpu["levels"].get(f"{float(ir_div):g}") or {}
        block: dict[str, Any] = {
            "what": ("value_and_grad of L_dut at this level through cuDSS at several "
                     "ir_num_steps, against the CPU study's recorded level"),
            "div": float(ir_div), "cpu": {k: cpu3.get(k) for k in ("L_dut", "grad", "fd")},
            "steps": {}}
        rec["l_dut_vs_ir"] = block
        prev_ir = os.environ.get("RFX_FDFD_CUDSS_IR")
        try:
            m = study.build_level(float(ir_div))
            for step in ir_phys:
                if time.time() > deadline:
                    break
                os.environ["RFX_FDFD_CUDSS_IR"] = str(step)
                try:
                    r = lib.value_and_grad_level(study, m, backend)
                    if cpu3.get("grad"):
                        r["L_dut_rel_vs_cpu"] = abs(
                            r["L_dut"] - cpu3["L_dut"]) / abs(cpu3["L_dut"])
                        r["grad_rel_vs_cpu"] = [
                            abs(a - b) / max(abs(b), 1e-300)
                            for a, b in zip(r["grad"], cpu3["grad"])]
                        r["grad_rel_vs_cpu_fd4"] = [
                            abs(a - b) / max(abs(b), 1e-300)
                            for a, b in zip(r["grad"], cpu3.get("fd") or cpu3["grad"])]
                        r["worst_vs_cpu"] = max([r["L_dut_rel_vs_cpu"]]
                                                + r["grad_rel_vs_cpu"])
                        print(f"      ir={step}: L_dut rel vs CPU "
                              f"{r['L_dut_rel_vs_cpu']:.3e}  grad worst "
                              f"{max(r['grad_rel_vs_cpu']):.3e}", flush=True)
                    block["steps"][str(step)] = r
                except Exception as exc:
                    block["steps"][str(step)] = {"error": f"{type(exc).__name__}: {exc}"}
                    print(f"      ir={step} FAILED: "
                          f"{block['steps'][str(step)]['error']}", flush=True)
                rec.dump()
            del m
        except Exception as exc:
            block["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if prev_ir is None:
                os.environ.pop("RFX_FDFD_CUDSS_IR", None)
            else:
                os.environ["RFX_FDFD_CUDSS_IR"] = prev_ir
            ls.clear_factor_cache()
        rec.dump()

    levels: dict[str, Any] = {}
    rec["levels"] = levels

    for div in divs:
        key = f"{div:g}"
        if time.time() > deadline:
            levels[key] = {"skipped": "deadline"}
            rec.dump()
            print(f"  W/{key}: skipped (deadline)", flush=True)
            continue
        t0 = time.time()
        model = study.build_level(div)
        grid = lib.model_record(study, model)
        lv: dict[str, Any] = {"base_dx": study.WIDTH / div, "div": div,
                              "grid": grid, "backend": backend,
                              "build_seconds": time.time() - t0}
        levels[key] = lv
        rec.dump()
        print(f"  W/{key}: {grid['shape']} N={grid['n_unknowns']} "
              f"nnz={len(model.yee.rows)}", flush=True)
        try:
            if want_grad and div <= grad_max_div:
                r = lib.value_and_grad_level(study, model, backend)
                lv.update({k: v for k, v in r.items() if k != "backend"})
            else:
                r = lib.forward_level(study, model, backend)
                lv.update({k: v for k, v in r.items() if k != "backend"})
                lv["grad"] = None
                lv["no_gradient_because"] = (
                    "value only at this level (RFX_LANE_GRAD / GRAD_MAX_DIV): the "
                    "adjoint holds a second cuDSS factorisation per fixture")
        except Exception as exc:
            lv["error"] = f"{type(exc).__name__}: {exc}"
            lv["device_memory_gb_at_error"] = _cudss.device_memory()
            print(f"  W/{key} FAILED: {lv['error']}", flush=True)
            ls.clear_factor_cache()
            rec.dump()
            continue
        rec.dump()
        if want_pec and time.time() < deadline:
            try:
                from rfx.fdfd import spiral as sm
                t0 = time.time()
                with ls.default_backend(backend):
                    lv["L_pec"] = float(sm.solve_spiral(
                        model, study.FREQ, sigma_volumetric=None).L_diff)
                lv["pec_seconds"] = time.time() - t0
                lv["L_pec_over_L_uniform_minus_1"] = lv["L_pec"] / lv["L_dut"] - 1.0
                ls.clear_factor_cache()
                print(f"    L_pec={lv['L_pec'] * 1e12:.4f} pH "
                      f"({lv['pec_seconds']:.0f} s)", flush=True)
            except Exception as exc:
                lv["pec_error"] = f"{type(exc).__name__}: {exc}"
                ls.clear_factor_cache()
            rec.dump()
        del model

    # --- gate G2: FD4 in ONE parameter, AFTER every level ----------------
    # Last on purpose: it is four more forward solves (twelve cuDSS
    # factorisations at W/4) and it is worth nothing if the levels
    # themselves did not land. The levels are the study; this is the check
    # on the gradient the study reports.
    if fd_div and time.time() < deadline:
        key = f"{float(fd_div):g}"
        lv = levels.get(key)
        if lv is None or lv.get("L_dut") is None:
            rec["fd_skipped"] = f"level W/{key} has no value to check"
        else:
            try:
                model = study.build_level(float(fd_div))
                fd = lib.fd4_one_parameter(study, model, backend, fd_param)
                lv["fd_one_parameter"] = fd
                if lv.get("grad"):
                    g = lv["grad"][fd_param]
                    fd["ad"] = g
                    fd["rel"] = abs(g - fd["fd"]) / max(abs(fd["fd"]), 1e-300)
                    print(f"    FD4 param {fd_param} at W/{key}: AD {g:.8e} vs FD "
                          f"{fd['fd']:.8e} rel {fd['rel']:.2e} "
                          f"({fd['fd_seconds']:.0f} s)", flush=True)
                del model
            except Exception as exc:
                lv["fd_error"] = f"{type(exc).__name__}: {exc}"
                print(f"    FD4 FAILED: {lv['fd_error']}", flush=True)
            ls.clear_factor_cache()
        rec.dump()

    # --- the extended Richardson, on whatever levels exist ---------------
    all_levels: dict[str, Any] = {}
    for k, v in cpu["levels"].items():
        all_levels[k] = {"div": v["div"], "base_dx": v["base_dx"],
                         "L_dut": v["L_dut"], "grad": v["grad"], "source": "cpu"}
    for k, v in levels.items():
        if v.get("L_dut") is not None:
            all_levels[k] = {"div": v["div"], "base_dx": v["base_dx"],
                             "L_dut": v["L_dut"], "grad": v.get("grad"),
                             "source": f"gpu:{backend}"}
    keys = sorted(all_levels, key=float)
    h = [all_levels[k]["base_dx"] for k in keys]
    vals = [all_levels[k]["L_dut"] for k in keys]
    dee = cpu["referee"]["deembedded"]["total"]
    rich = study.richardson(h, vals)
    rec["extended"] = {
        "levels": keys, "h": h, "L_dut": vals,
        "referee_deembedded": dee,
        "per_level_gap": {k: all_levels[k]["L_dut"] / dee - 1.0 for k in keys},
        "richardson": rich,
        "rel_range": [v / dee - 1.0 for v in rich["range"]],
        "source": {k: all_levels[k]["source"] for k in keys},
        "dL_dwidth": {k: (all_levels[k]["grad"] or [None, None, None])[2] for k in keys},
    }
    print("extended Richardson:", rec["extended"]["richardson"]["estimates"],
          "rel", rec["extended"]["rel_range"], flush=True)
    rec.dump()
    print(f"J1 done in {rec['seconds']:.0f} s -> {rec.path}", flush=True)
    ok = [k for k, v in levels.items() if v.get("L_dut") is not None]
    return 0 if ok else 1


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.exit(main())
