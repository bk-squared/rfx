"""J0, the GPU backend probe: does cuDSS work, and does it agree with SuperLU?

Runs inside the VESSL container (``validation/vessl/build_gpu_lanes.py``
generates the YAML that embeds this file). Blocks, in order, each written
to the lane JSON as soon as it finishes so a timeout still harvests what
ran:

1. environment: versions, device, backend availability with reasons
2. ``_cudss.selftest`` for every available GPU backend: a random complex
   system solved forward and transposed against SuperLU, plus the
   cross-thread check
3. gate G1 on the REAL W/3 spiral DUT fixture (N = 108898): the exact
   ``(data, rows, cols, b)`` the study assembles, solved by SuperLU and by
   cuDSS -- residuals, solution difference, factor and solve wall times,
   device memory, cuDSS's LU nonzero count
4. cuDSS plan-only memory estimates at W/3, W/4 and W/6, which is what
   decides whether J1's levels fit in 24 GB or need hybrid memory mode
5. ``value_and_grad`` of ``L_dut`` at W/3 with both backends: equality to
   1e-8 relative (rule 2), against each other AND against the CPU study's
   recorded value; the cuDSS gradient is also compared with the CPU
   study's recorded FD4 gradient, which is gate G2 at W/3 for free (that
   FD4 cost 5033 s on the Mac and is already in the JSON).

The pytest runs (this file's suite and the hplane suite, with the backend
forced both ways) happen in the shell before this script, so their exit
codes are the job's, not this script's.
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

# W/3 is the finest CPU level: N = 108898, L_dut 300.5577 pH, a
# value_and_grad of 304 s (347 s as recorded) on the Mac. It is the level
# where the two backends can be compared against a number that already
# exists, which is why the probe uses it and not a toy.
DIV_G1 = 3.0
PLAN_DIVS = (3.0, 4.0, 6.0)
# rule 2: the two backends' value and gradient must agree to 1e-8 relative
EQUALITY_TOL = 1e-8
# rule 1 / gate G2: AD vs FD4, 1e-4 relative (FD4 truncation dominates)
FD_TOL = 1e-4


def main() -> int:
    import jax
    jax.config.update("jax_enable_x64", True)
    rec = lib.Recorder("j0_probe.json", "J0")
    rec["env"] = lib.env_record()
    print("env:", {k: rec["env"][k] for k in ("python", "jax", "scipy", "numpy",
                                              "jax_devices", "nvidia_smi")}, flush=True)
    print("backends:", rec["env"]["backends"], flush=True)

    from rfx.fdfd import _cudss
    import rfx.fdfd.linear_solve as ls
    have = [b for b in ("cudss", "cusolver") if ls.backend_available(b)]
    rec["gpu_backends_available"] = have
    if not have:
        rec["blocked"] = {"what": "no GPU backend available",
                          "reasons": {b: _cudss.unavailable_reason(b)
                                      for b in ("cudss", "cusolver")}}
        rec.dump()
        print("BLOCKED: no GPU backend", flush=True)
        return 3

    # -- 2. selftest ------------------------------------------------------
    st: dict[str, Any] = {}
    for be in have:
        t0 = time.time()
        try:
            st[be] = _cudss.selftest(be, n=400, m=3)
            st[be]["seconds"] = time.time() - t0
            print(f"  selftest {be}: residual_N {st[be]['residual_N_gpu']:.2e} "
                  f"residual_T {st[be]['residual_T_gpu']:.2e} "
                  f"diff_N {st[be]['diff_N']:.2e} diff_T {st[be]['diff_T']:.2e} "
                  f"cross_thread_ok {st[be]['cross_thread_ok']} "
                  f"({st[be]['seconds']:.1f} s)", flush=True)
        except Exception as exc:
            st[be] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  selftest {be} FAILED: {st[be]['error']}", flush=True)
    rec["selftest"] = st
    # The heavy blocks (a real N = 108898 system, and value_and_grad of the
    # three-fixture study quantity) run on SuperLU and cuDSS only.
    # ``"cusolver"`` is in the selftest, where it is honest and cheap, and
    # NOT here: cupy's ``spsolve`` is cuSOLVER's one-shot sparse QR per
    # right-hand-side column, so one (108898, 2) solve is two full sparse QR
    # factorisations with QR's fill, a repeat is two more, and a
    # three-fixture value_and_grad would be twelve. It is the fallback for an
    # image without nvmath-python, not a candidate at this size.
    # ``RFX_LANE_HEAVY_BACKENDS`` overrides.
    heavy = os.environ.get("RFX_LANE_HEAVY_BACKENDS", "superlu cudss").split()
    backends = tuple(b for b in heavy
                     if b == "superlu" or (b in have and not st.get(b, {}).get("error")))
    rec["backends_compared"] = list(backends)
    rec["backends_selftested"] = list(have)

    # -- 3. gate G1 on the real W/3 DUT system ----------------------------
    study = lib.load_study()
    cpu = lib.cpu_study()
    rec["cpu_reference"] = {
        "levels": {k: {kk: cpu["levels"][k][kk] for kk in
                       ("base_dx", "div", "L_dut", "grad", "fd", "fd_order",
                        "value_and_grad_seconds", "fd_seconds", "grad_vs_fd_rel")
                       if kk in cpu["levels"][k]}
                   for k in sorted(cpu["levels"], key=float)},
        "referee_deembedded": cpu["referee"]["deembedded"]["total"],
        "n_unknowns": {k: cpu["levels"][k]["grid"]["n_unknowns"]
                       for k in sorted(cpu["levels"], key=float)},
    }
    t0 = time.time()
    model = study.build_level(DIV_G1)
    grid = lib.model_record(study, model)
    grid["build_seconds_here"] = time.time() - t0
    rec["g1_grid"] = grid
    print(f"  W/{DIV_G1:g}: {grid['shape']} N={grid['n_unknowns']}", flush=True)
    data, rows, cols, rhs = lib.dut_system(study, model)
    print(f"  DUT system captured: N={rhs.shape[0]} nnz={data.size} "
          f"rhs={rhs.shape}", flush=True)
    rec["g1"] = lib.solve_both(data, rows, cols, rhs, backends=backends)

    # -- 4. plan-only memory estimates ------------------------------------
    if "cudss" in backends:
        plans: dict[str, Any] = {}
        for div in PLAN_DIVS:
            try:
                m = model if div == DIV_G1 else study.build_level(div)
                d, r, c, b = (data, rows, cols, rhs) if div == DIV_G1 \
                    else lib.dut_system(study, m)
                p = _cudss.plan_estimate(d, r, c, int(b.shape[0]), m=int(b.shape[1])
                                         if b.ndim > 1 else 1)
                p["n_unknowns"] = int(b.shape[0])
                p["shape"] = list(m.shape)
                plans[f"{div:g}"] = p
                print(f"  plan W/{div:g}: N={p['n_unknowns']} "
                      f"permanent {p.get('permanent_device_memory_gb', float('nan')):.2f} GB "
                      f"peak {p.get('peak_device_memory_gb', float('nan')):.2f} GB "
                      f"({p.get('plan_seconds', float('nan')):.1f} s)", flush=True)
                del m
            except Exception as exc:
                plans[f"{div:g}"] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  plan W/{div:g} FAILED: {plans[f'{div:g}']['error']}", flush=True)
            rec["plan_estimates"] = plans

    # -- 5. value_and_grad with both backends -----------------------------
    del data, rows, cols, rhs
    vg: dict[str, Any] = {}
    for be in backends:
        try:
            vg[be] = lib.value_and_grad_level(study, model, be)
        except Exception as exc:
            vg[be] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  value_and_grad {be} FAILED: {vg[be]['error']}", flush=True)
        rec["value_and_grad"] = vg

    # -- the gate arithmetic ----------------------------------------------
    gates: dict[str, Any] = {}
    g1 = rec["g1"]
    res = {b: g1["per_backend"][b]["residual"] for b in g1["per_backend"]}
    diffs = {b: g1["per_backend"][b].get("rel_diff_vs_reference")
             for b in g1["per_backend"] if b != "superlu"}
    ok_eq = True
    vg_pairs: dict[str, Any] = {}
    ref = vg.get("superlu", {})
    for be, r in vg.items():
        if be == "superlu" or "error" in r or "error" in ref:
            continue
        dl = abs(r["L_dut"] - ref["L_dut"]) / abs(ref["L_dut"])
        dg = [abs(a - b) / max(abs(b), 1e-300) for a, b in zip(r["grad"], ref["grad"])]
        vg_pairs[be] = {"L_dut_rel": dl, "grad_rel": dg,
                        "worst": max([dl] + dg)}
        ok_eq = ok_eq and max([dl] + dg) <= EQUALITY_TOL
    gates["G1"] = {
        "what": ("same (data, rows, cols, b) of the W/3 DUT fixture solved by "
                 "every backend: each residual at roundoff, the solutions equal "
                 "there too, and value_and_grad of L_dut equal to 1e-8 relative"),
        "n": g1["n"], "nnz": g1["nnz"],
        "residuals": res, "rel_diff_vs_superlu": diffs,
        "residual_tolerance": 1e-9,
        "solution_diff_tolerance": 1e-8,
        "value_and_grad_equality": vg_pairs,
        "value_and_grad_tolerance": EQUALITY_TOL,
        "passed": bool(all(v is not None and v <= 1e-9 for v in res.values())
                       and all(v is not None and v <= 1e-8 for v in diffs.values())
                       and bool(vg_pairs) and ok_eq),
    }
    # G2 at W/3 for free: the CPU study's FD4 gradient at this level was
    # measured over 5033 s and is in the JSON; the cuDSS AD gradient is
    # compared with it directly (rule 1 on the GPU path).
    cpu3 = cpu["levels"].get(f"{DIV_G1:g}", {})
    if "cudss" in vg and "error" not in vg["cudss"] and cpu3.get("fd"):
        rel = [abs(a - b) / max(abs(b), 1e-300)
               for a, b in zip(vg["cudss"]["grad"], cpu3["fd"])]
        gates["G2_at_W3_vs_recorded_cpu_fd4"] = {
            "what": ("cuDSS AD gradient vs the CPU study's recorded FD4 gradient "
                     "at the same level, same 1 % steps"),
            "ad": vg["cudss"]["grad"], "fd4": cpu3["fd"], "rel": rel,
            "worst": max(rel), "tolerance": FD_TOL,
            "passed": bool(max(rel) <= FD_TOL)}
    if "cudss" in vg and "error" not in vg["cudss"] and cpu3.get("L_dut"):
        gates["L_dut_vs_cpu_study"] = {
            "cpu": cpu3["L_dut"], "gpu": vg["cudss"]["L_dut"],
            "rel": abs(vg["cudss"]["L_dut"] - cpu3["L_dut"]) / abs(cpu3["L_dut"]),
            "what": ("the same level's L_dut as recorded by the CPU study on a "
                     "different machine, python and scipy: a version-drift check"),
        }
    rec["gates"] = gates
    print("gates:", {k: v.get("passed") for k, v in gates.items()}, flush=True)
    rec.dump()
    print(f"J0 done in {rec['seconds']:.0f} s -> {rec.path}", flush=True)
    return 0 if gates["G1"]["passed"] else 1


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.exit(main())
