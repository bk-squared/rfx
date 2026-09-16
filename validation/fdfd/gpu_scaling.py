"""Study V: the GPU sparse-direct backend, and the spiral levels it buys.

    .venv/bin/python validation/fdfd/gpu_scaling.py            # JSON + PNG
    .venv/bin/python validation/fdfd/gpu_scaling.py --no-figure
    .venv/bin/python validation/fdfd/gpu_scaling.py --ordering-control 1,3
                              # G1's same-backend control, host only, ~20 min

WHAT THIS IS
------------
``rfx.fdfd.linear_solve`` grew a ``backend`` argument:
``"superlu"`` (scipy, host, the default and unchanged) and ``"cudss"``
(NVIDIA cuDSS through nvmath-python, GPU) behind one interface, with the
same LU cache, the same one factor thread and the same
``custom_linear_solve`` AD rule. This script does NOT run any of that: the
solves happen on the VESSL RTX 4090 lane (``validation/vessl/``, lanes
J0/J1/J2) and land as one JSON per job under ``validation/vessl/runs/``.
Here they are merged, the extended convergence study is recomputed, and
the gates are decided.

WHY A GPU BACKEND AT ALL
------------------------
The spiral study (``spiral_convergence.py``, study D2) is bounded by one
number: a ``value_and_grad`` of ``L_dut`` at W/3 (N = 108898) costs 304 s
and ~5 GB on this Mac, and W/4 (N = 172747) and W/6 (N = 352536) -- the
levels that would settle the Richardson extrapolation and the shape
derivative's convergence -- are out of reach of a host SuperLU. Its
verdict V1 rests on an extrapolation from three levels whose finest is
-9.8 % from the referee, with a Richardson range of -7.9..-5.2 %. Two more
levels either close that or do not.

WHAT IS EXTENDED, AND WHAT IS NOT
---------------------------------
``spiral_convergence.py`` is NOT edited. Its ``build_level``, ``l_dut``,
``richardson`` and its recorded referee are imported and reused, so a GPU
level is the same measurement as a CPU level -- same fixture, same
uniform-current volumetric metal, same ``pad_cells`` 4, same 1 % FD steps
-- and the five levels W/1, W/2, W/3 (CPU, from its JSON) plus W/4, W/6
(GPU, from the lane JSONs) form ONE series. Only the two verdicts that
depend on the level ladder are recomputed here: V1 (the absolute gap and
its extrapolation) and V4 (the width derivative's convergence). The
per-level gradient check V2, the PEC deficit V1b, the referee gradient V3,
the wall gate V5 and the thru probe V6 stay where they are.

THE GATES
---------
G1  backend equality on the W/3 spiral DUT fixture: the same
    ``(data, rows, cols, b)`` through both backends -- each residual at its
    own roundoff, the two solutions equal there, and ``value_and_grad`` of
    ``L_dut`` equal to 1e-8 relative. It FAILS, and the numbers say exactly
    where: the residuals are 1.787e-08 (SuperLU) and 3.841e-09 (cuDSS) --
    not 1e-16, because this operator's row scaling does not allow it -- the
    solution VECTORS differ by 2.395e-03, and ``value_and_grad`` agrees to
    9.2e-10 on ``L_dut`` but only to 2.29e-08 on the worst gradient
    component. The conditioning block under it isolates the cause: cuDSS's
    iterative refinement takes its residual to 1.310e-09, 13x below
    SuperLU's, and the difference between the solutions does not move
    (2.392e-03), so the solution vector is simply not determined to better
    than that by a solve of this system -- while ``L_dut`` moves by 3e-10
    and its gradient by 4e-09 over the same sweep. The functional the study
    reports is insensitive to the difference; the vector is not; and the
    gradient's cross-backend agreement (1.5e-08) is an order BELOW the
    study's own AD-versus-FD4 floor at this level (9.3e-08), which is why
    none of it moves a physics conclusion. Its ``same_backend_control``
    block carries the control for that clause, measured on this Mac by
    ``--ordering-control``: ONE backend (host SuperLU) solving those same
    systems along different elimination paths -- other fill-reducing
    orderings, the equations permuted, the unknowns rescaled -- all with the
    same exact solution. At the gate's own level (W/3, N = 108898) those
    paths differ from the COLAMD reference by 2.873e-03 (MMD_ATA),
    3.262e-03 (a random row permutation) and 3.682e-03 (a diagonal
    rescaling) -- residuals 1.15e-08..1.94e-08 -- i.e. host SuperLU
    disagrees with itself MORE than cuDSS disagrees with it (2.395e-03), at a
    1-norm condition estimate of 3.05e+13. The 1e-8 clause is therefore
    unreachable for any pair of direct solves of this operator; it is left at
    1e-8 and G1 is still reported failing.
G2  AD through cuDSS against a 4th-order finite difference, one parameter,
    1e-4 relative.
G3  the cost table: N, factorisation seconds, solve seconds and device GB
    per level, against the CPU W/3 reference (304 s ``value_and_grad``).
G4  the extended Richardson: per-level gap, the observed order over the
    finest three levels, and the extrapolated range against the
    bridge-corrected referee 333.055 pH, at 5 %.
G5  V4 on the extended levels: is the level-to-level change of
    ``dL/dwidth`` still decreasing when the ladder is twice as long?
G6  every run accounted for -- completed or terminated, none left running.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS_DIR = REPO / "validation" / "vessl" / "runs"
LEDGER = RUNS_DIR / "run_ledger.json"
JSON_PATH = HERE / "gpu_scaling.json"
PNG_PATH = HERE / "gpu_scaling.png"
CPU_JSON = HERE / "spiral_convergence.json"

# 5 %: study D2's acceptance criterion for V1, unchanged and not widened.
TOL_V1 = 0.05
# 1e-8 relative: two direct solves of the same system at their own roundoff
# (rule 2 of the task). 1e-4: AD vs FD4, where the FD truncation dominates.
TOL_EQUALITY = 1e-8
TOL_FD = 1e-4
# The CPU reference this study is measured against: study D2's W/3 level on
# this Mac (jax 0.11.1, scipy 1.18.1, SuperLU/COLAMD).
CPU_W3_VALUE_AND_GRAD_SECONDS = 304.0
CPU_W3_RECORDED_SECONDS = 347.0937809944153

# G1's same-backend control (--ordering-control, this Mac, host SuperLU only):
# the cached measurement and the case list per level. W/1 can afford every
# ordering SuperLU has; at W/3 MMD_AT_PLUS_A and NATURAL cost 16x and 34x
# COLAMD's factorisation (measured at W/1: 5.4 s / 87.7 s / 183.4 s), which is
# hours and tens of GB there, so the gate's own level is controlled with the
# two orderings that are affordable plus the two path variants -- a row
# permutation and a diagonal rescaling, both of which leave the exact solution
# untouched and change only the elimination path.
CONTROL_JSON = REPO / "validation" / "vessl" / "controls" / "superlu_ordering_control.json"
CONTROL_CASES: dict[str, tuple[tuple[str, str], ...]] = {
    "1": (("COLAMD", "COLAMD"), ("MMD_ATA", "MMD_ATA"),
          ("MMD_AT_PLUS_A", "MMD_AT_PLUS_A"), ("NATURAL", "NATURAL"),
          ("row_permuted", "row_permuted:COLAMD"),
          ("col_scaled", "col_scaled:COLAMD")),
    "3": (("COLAMD", "COLAMD"), ("MMD_ATA", "MMD_ATA"),
          ("row_permuted", "row_permuted:COLAMD"),
          ("col_scaled", "col_scaled:COLAMD")),
}


def load_study() -> Any:
    """``spiral_convergence.py`` as a module -- imported, never edited."""
    path = HERE / "spiral_convergence.py"
    spec = importlib.util.spec_from_file_location("spiral_convergence_v", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["spiral_convergence_v"] = mod
    spec.loader.exec_module(mod)
    return mod


def lane_records(runs_dir: pathlib.Path) -> dict[str, Any]:
    """Every harvested lane JSON, newest run per lane winning.

    ``validation/vessl/runs/fdfd-gpu-<lane>-<utc>/j<lane>_*.json``; a lane
    that never ran is simply absent and every downstream gate says so.
    """
    out: dict[str, Any] = {}
    if not runs_dir.exists():
        return out
    for d in sorted(runs_dir.glob("fdfd-gpu-*")):
        for f in sorted(d.glob("j*.json")):
            try:
                rec = json.loads(f.read_text())
            except Exception as exc:
                print(f"  skip {f.name}: {type(exc).__name__}: {exc}")
                continue
            lane = str(rec.get("lane", f.stem)).lower()
            rec["_dir"] = d.name
            rec["_file"] = str(f.relative_to(REPO))
            out[lane] = rec
    return out


def merged_levels(cpu: dict[str, Any], j1: dict[str, Any] | None) -> dict[str, Any]:
    """The five-level ladder: CPU W/1..W/3 plus whatever the GPU lane got.

    ``div`` is the number of cells across the strip width (``base_dx =
    W / div``), which is the x axis the bar study's 2 % point is quoted on.
    """
    lv: dict[str, Any] = {}
    for k, v in cpu["levels"].items():
        lv[k] = {"div": v["div"], "base_dx": v["base_dx"],
                 "n_unknowns": v["grid"]["n_unknowns"],
                 "L_dut": v["L_dut"], "grad": v["grad"],
                 "L_pec": v.get("L_pec"),
                 "value_and_grad_seconds": v.get("value_and_grad_seconds"),
                 "backend": "superlu", "source": "cpu:spiral_convergence.json"}
    for k, v in ((j1 or {}).get("levels") or {}).items():
        if v.get("L_dut") is None:
            continue
        lv[k] = {"div": v["div"], "base_dx": v["base_dx"],
                 "n_unknowns": v["grid"]["n_unknowns"],
                 "L_dut": v["L_dut"], "grad": v.get("grad"),
                 "L_pec": v.get("L_pec"),
                 "value_and_grad_seconds": v.get("value_and_grad_seconds"),
                 "solve_seconds": v.get("solve_seconds"),
                 "device_memory_gb_peak_after": v.get("device_memory_gb_peak_after"),
                 "stats": v.get("stats"),
                 "backend": v.get("backend", "cudss"),
                 "source": f"gpu:{(j1 or {}).get('_file', 'j1')}"}
    return lv


def _g1_from_raw(j0: dict[str, Any]) -> dict[str, Any]:
    """G1 from the raw blocks, for a lane that was cut off before its own
    gate arithmetic (the lane writes its JSON after every block, and the
    gates are the last one)."""
    g1 = j0.get("g1") or {}
    per = g1.get("per_backend") or {}
    if not per:
        return {}
    res = {b: r.get("residual") for b, r in per.items()}
    diffs = {b: r.get("rel_diff_vs_reference") for b, r in per.items() if b != "superlu"}
    vg = j0.get("value_and_grad") or {}
    ref = vg.get("superlu", {})
    pairs: dict[str, Any] = {}
    for be, r in vg.items():
        if be == "superlu" or "error" in r or "error" in ref or not ref:
            continue
        dl = abs(r["L_dut"] - ref["L_dut"]) / abs(ref["L_dut"])
        dg = [abs(a - b) / max(abs(b), 1e-300) for a, b in zip(r["grad"], ref["grad"])]
        pairs[be] = {"L_dut_rel": dl, "grad_rel": dg, "worst": max([dl] + dg)}
    ok_res = all(v is not None and v <= 1e-9 for v in res.values())
    ok_diff = all(v is not None and v <= 1e-8 for v in diffs.values())
    ok_vg = bool(pairs) and all(v["worst"] <= TOL_EQUALITY for v in pairs.values())
    return {"n": g1.get("n"), "nnz": g1.get("nnz"), "residuals": res,
            "rel_diff_vs_superlu": diffs, "residual_tolerance": 1e-9,
            "solution_diff_tolerance": 1e-8,
            "value_and_grad_equality": pairs,
            "value_and_grad_tolerance": TOL_EQUALITY,
            "recomputed_here": True,
            "passed": bool(ok_res and ok_diff and ok_vg)}


def _vg_equality_vs_cpu(j0: dict[str, Any], cpu: dict[str, Any],
                        div: str = "3") -> dict[str, Any]:
    """The value_and_grad equality against the CPU study's RECORDED level.

    The in-job pair (SuperLU and cuDSS in the same process) is the primary
    form of the equality, but the SuperLU half costs three host
    factorisations at N = 108898 and is the first thing a wall-clock cap
    cuts. The recorded W/3 level is the same computation on this Mac
    (jax 0.11.1, scipy 1.18.1, SuperLU/COLAMD), so comparing against it is
    a WEAKER claim in one way -- a different machine, python, numpy, scipy
    and jax -- and a stronger one in another, and it is stated as what it
    is rather than substituted silently.
    """
    lv = (cpu.get("levels") or {}).get(div) or {}
    out: dict[str, Any] = {}
    for be, r in (j0.get("value_and_grad") or {}).items():
        if be == "superlu" or "error" in r or not lv.get("grad"):
            continue
        dl = abs(r["L_dut"] - lv["L_dut"]) / abs(lv["L_dut"])
        dg = [abs(a - b) / max(abs(b), 1e-300) for a, b in zip(r["grad"], lv["grad"])]
        out[be] = {"L_dut_rel": dl, "grad_rel": dg, "worst": max([dl] + dg),
                   "cpu_L_dut": lv["L_dut"], "cpu_grad": lv["grad"],
                   "gpu_L_dut": r["L_dut"], "gpu_grad": r["grad"],
                   "what": ("cuDSS on the RTX 4090 lane (jax 0.6.2, scipy 1.15.3, "
                            "python 3.10) against the recorded CPU level (jax 0.11.1, "
                            "scipy 1.18.1, python 3.12, SuperLU)")}
    return out


def g1_gate(j0: dict[str, Any] | None) -> dict[str, Any]:
    if not j0:
        return {"what": "backend equality on the W/3 spiral DUT fixture",
                "passed": False, "reason": "no J0 lane JSON harvested"}
    g = dict((j0.get("gates") or {}).get("G1") or {})
    if not g:
        g = _g1_from_raw(j0)
    if not g:
        g = {"passed": False, "reason": "J0 ran but got no further than the selftest"}
    g.setdefault("what", "backend equality on the W/3 spiral DUT fixture")
    per = ((j0.get("g1") or {}).get("per_backend") or {})
    g["per_backend_timing"] = {
        b: {k: r.get(k) for k in ("cold_seconds", "warm_seconds", "factor_seconds",
                                  "solve_seconds", "residual", "stats")}
        for b, r in per.items()}
    g["selftest"] = j0.get("selftest")
    return g


def _g1_conditioning(j1: dict[str, Any] | None) -> dict[str, Any]:
    """Is the 2.4e-03 solution difference the backend, or the operator?

    The J1 lane sweeps cuDSS's ``ir_num_steps`` on the same W/3 system and
    re-reads ``L_dut`` at several settings. Two readings come out of it:
    whether refinement moves cuDSS's OWN solution by the same ~1e-3 (then
    the solution vector is simply not determined to better than that by a
    solve of this operator, and the two backends' difference is the
    operator talking), and whether ``L_dut`` -- the four port readings the
    study actually reports -- moves at all.
    """
    out: dict[str, Any] = {}
    sweep = (j1 or {}).get("ir_sweep") or {}
    if sweep.get("steps"):
        steps = sweep["steps"]
        base = steps.get("0") or next(iter(steps.values()))
        out["ir_sweep"] = {
            "reference": sweep.get("reference"),
            "reference_residual": sweep.get("reference_residual"),
            "reference_seconds": sweep.get("reference_seconds"),
            "per_step": {k: {kk: v.get(kk) for kk in
                             ("ir_num_steps", "residual", "rel_diff_vs_reference",
                              "rel_l2_diff_vs_reference", "seconds")}
                         for k, v in steps.items()},
            "residual_span": [min(v["residual"] for v in steps.values()),
                              max(v["residual"] for v in steps.values())],
            "diff_span": [min(v["rel_diff_vs_reference"] for v in steps.values()),
                          max(v["rel_diff_vs_reference"] for v in steps.values())],
            "refinement_moves_the_solution_like_the_backend_gap": bool(
                base and max(abs(v["rel_diff_vs_reference"]
                                 - base["rel_diff_vs_reference"])
                             for v in steps.values()) > 0.1 * base["rel_diff_vs_reference"]),
        }
    phys = (j1 or {}).get("l_dut_vs_ir") or {}
    if phys.get("steps"):
        ok = {k: v for k, v in phys["steps"].items() if "error" not in v}
        if ok:
            out["l_dut_vs_ir"] = {
                "div": phys.get("div"), "cpu": phys.get("cpu"),
                "per_step": {k: {"L_dut": v.get("L_dut"),
                                 "L_dut_rel_vs_cpu": v.get("L_dut_rel_vs_cpu"),
                                 "grad_rel_vs_cpu": v.get("grad_rel_vs_cpu"),
                                 "worst_vs_cpu": v.get("worst_vs_cpu"),
                                 "seconds": v.get("value_and_grad_seconds")}
                             for k, v in ok.items()},
                "L_dut_span_over_ir": [min(v["L_dut"] for v in ok.values()),
                                       max(v["L_dut"] for v in ok.values())],
                "worst_vs_cpu": max((v.get("worst_vs_cpu") or 0.0) for v in ok.values()),
                "best_vs_cpu": min((v.get("worst_vs_cpu") or 0.0) for v in ok.values()),
                "L_dut_rel_span": [min(v["L_dut_rel_vs_cpu"] for v in ok.values()),
                                   max(v["L_dut_rel_vs_cpu"] for v in ok.values())],
            }
    return out


def _g1_with_cpu_fallback(j0: dict[str, Any] | None, cpu: dict[str, Any],
                          j1: dict[str, Any] | None = None) -> dict[str, Any]:
    """G1, and -- when the in-job SuperLU value_and_grad did not run -- the
    same equality against the recorded CPU level, which then decides it."""
    g = g1_gate(j0)
    if not j0:
        return g
    vs_cpu = _vg_equality_vs_cpu(j0, cpu)
    if vs_cpu:
        g["value_and_grad_equality_vs_recorded_cpu"] = vs_cpu
    cond = _g1_conditioning(j1)
    if cond:
        g["conditioning"] = cond
        ldut = cond.get("l_dut_vs_ir") or {}
        if ldut.get("worst_vs_cpu") is not None:
            g["functional_worst_vs_cpu"] = ldut["worst_vs_cpu"]
            g["functional_best_vs_cpu"] = ldut.get("best_vs_cpu")
            g["functional_L_dut_rel_span"] = ldut.get("L_dut_rel_span")
            g["functional_note"] = (
                "worst and best over the ir_num_steps sweep; L_dut itself agrees to "
                "1.5e-10 with refinement on, and the number that misses 1e-8 is the "
                "worst GRADIENT component -- which sits an order below the study's own "
                "AD-vs-FD4 floor at this level (9.28e-08, spiral_convergence.json "
                "levels.3.grad_vs_fd_rel)")
            g["passed_functional"] = bool(ldut["worst_vs_cpu"] <= TOL_EQUALITY)
    if not g.get("value_and_grad_equality") and vs_cpu:
        worst = max(v["worst"] for v in vs_cpu.values())
        g["value_and_grad_fallback_used"] = (
            "the in-job SuperLU value_and_grad did not run (wall-clock cap); the "
            "equality is decided against the recorded CPU W/3 level instead")
        res_ok = all(v is not None and v <= 1e-9 for v in (g.get("residuals") or {}).values())
        diff_ok = all(v is not None and v <= 1e-8
                      for v in (g.get("rel_diff_vs_superlu") or {}).values())
        g["value_and_grad_worst"] = worst
        g["passed"] = bool(res_ok and diff_ok and worst <= TOL_EQUALITY)
    return g


def load_lane_lib() -> Any:
    """``validation/vessl/_gpu_lane_lib.py`` -- the lanes' measurement code.

    The control below is a HOST measurement (no GPU, no cluster), but it has
    to build the system exactly as gate G1 did, so it uses the same capture
    helper the lanes used rather than a second copy of it.
    """
    path = REPO / "validation" / "vessl" / "_gpu_lane_lib.py"
    spec = importlib.util.spec_from_file_location("gpu_lane_lib_v", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gpu_lane_lib_v"] = mod
    spec.loader.exec_module(mod)
    return mod


def run_ordering_control(study: Any, divs: tuple[float, ...],
                         path: pathlib.Path = CONTROL_JSON) -> dict[str, Any]:
    """Measure G1's same-backend control on this Mac and cache it.

    G1's solution-difference clause asks two SOLVES of the W/3 DUT system to
    agree to 1e-8 in ``max|x - x_ref| / max|x_ref|``. Whether that is a
    statement about the two backends or about the operator is decided by the
    same measurement with the backend held fixed: host SuperLU, several
    elimination paths, one system whose exact solution is common to all of
    them. Runs one level at a time and merges into the cached JSON, because
    the W/3 cases are minutes each.
    """
    import jax
    jax.config.update("jax_enable_x64", True)   # as every lane and the CPU study do
    lib = load_lane_lib()
    path.parent.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {}
    if path.exists():
        rec = json.loads(path.read_text())
    rec.setdefault("what", ("gate G1's same-backend control: host SuperLU, several "
                            "elimination paths, one system -- measured on this Mac"))
    rec.setdefault("levels", {})
    rec["host"] = {"python": sys.version.split()[0], "machine": _host_tag()}
    for div in divs:
        key = f"{div:g}"
        cases = CONTROL_CASES.get(key)
        if cases is None:
            raise SystemExit(f"no control case list for W/{key}; add one to CONTROL_CASES")
        print(f"W/{key}: building the DUT system ...", flush=True)
        t0 = time.time()
        model = study.build_level(float(div))
        data, rows, cols, rhs = lib.dut_system(study, model)
        print(f"  N={int(rhs.shape[0])} nnz={int(data.size)} "
              f"assembled in {time.time() - t0:.1f} s", flush=True)
        out = lib.superlu_ordering_control(data, rows, cols, rhs, cases)
        out["div"] = float(div)
        out["base_dx"] = study.WIDTH / float(div)
        rec["levels"][key] = out
        rec["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        path.write_text(json.dumps(rec, indent=1, default=str))
        print(f"  wrote {path}", flush=True)
    return rec


def _host_tag() -> str:
    import platform
    return f"{platform.system()} {platform.machine()}"


def g1_control(control: dict[str, Any] | None,
               g1: dict[str, Any] | None = None) -> dict[str, Any]:
    """G1's control block: what one backend does to itself on this operator.

    Reported per level: every case's residual and its distance from the
    reference case's solution in G1's own norm, the spans over the cases, and
    a 1-norm condition estimate. Two readings come out of it.
    ``bounds_the_gate``: one backend's own elimination paths already differ by
    more than G1's 1e-8 solution tolerance, so no PAIR of direct solvers can
    meet that clause on this system. And ``cudss_gap_vs_same_backend_spread``:
    where the cuDSS-versus-SuperLU number of G1 sits relative to the spread
    SuperLU shows against itself at the SAME level -- a ratio at or below 1
    says the cross-backend gap is not distinguishable from the ordinary
    disagreement of two direct solves of this operator.
    """
    if not control or not control.get("levels"):
        return {"passed": None, "reason": "no ordering control measured "
                                          "(--ordering-control)"}
    per: dict[str, Any] = {}
    for key, lv in sorted(control["levels"].items(), key=lambda kv: float(kv[0])):
        cases = lv.get("cases") or {}
        per[f"W/{key}"] = {
            "n": lv.get("n"), "nnz": lv.get("nnz"),
            "reference": lv.get("reference"),
            "cases": {k: {kk: v.get(kk) for kk in
                          ("permc_spec", "variant", "seconds", "residual",
                           "rel_diff_vs_reference", "rel_l2_diff_vs_reference",
                           "error")}
                      for k, v in cases.items()},
            "residual_span": lv.get("residual_span"),
            "rel_diff_span": lv.get("rel_diff_span"),
            "worst_rel_diff": lv.get("worst_rel_diff"),
            "cond_1_estimate": (lv.get("cond_1_estimate") or {}).get("cond_1"),
        }
    worst = max((v["worst_rel_diff"] for v in per.values()
                 if v.get("worst_rel_diff") is not None), default=None)
    gate_level = per.get("W/3") or {}
    # where G1's cross-backend number sits inside the same-backend spread at
    # the same level, which is the whole point of measuring the control there
    cross: dict[str, Any] = {}
    span = gate_level.get("rel_diff_span")
    for be, v in ((g1 or {}).get("rel_diff_vs_superlu") or {}).items():
        if v is None or not span:
            continue
        cross[be] = {
            "cross_backend_rel_diff": v,
            "same_backend_rel_diff_span": span,
            "ratio_to_worst_same_backend": v / span[1],
            "ratio_to_closest_same_backend": v / span[0],
            "where": ("below every same-backend pair" if v <= span[0] else
                      "inside the same-backend spread" if v <= span[1] else
                      "above the same-backend spread"),
            "not_worse_than_same_backend": bool(v <= span[1]),
        }
    # the control runs on THIS Mac (scipy 1.18.1) and G1 ran on the lane
    # (scipy 1.15.3): the two SuperLU/COLAMD residuals on the same W/3 system
    # say whether that is a comparison of like with like
    lane_res = ((g1 or {}).get("residuals") or {}).get("superlu")
    here_res = ((gate_level.get("cases") or {}).get("COLAMD") or {}).get("residual")
    same_solver = {"superlu_colamd_residual_here": here_res,
                   "superlu_colamd_residual_on_the_lane": lane_res,
                   "ratio": (here_res / lane_res) if (here_res and lane_res) else None,
                   "what": ("the same system and the same solver on the two machines: "
                            "the control's reference solve against G1's SuperLU half")}
    return {
        "what": ("host SuperLU against itself on the same systems -- different "
                 "fill-reducing orderings, a random permutation of the equations, and "
                 "a diagonal rescaling of the unknowns, all with the same exact "
                 "solution"),
        "source": str(CONTROL_JSON.relative_to(REPO)),
        "host": control.get("host"),
        "per_level": per,
        "worst_same_backend_rel_diff": worst,
        "gate_tolerance": 1e-8,
        "bounds_the_gate": bool(worst is not None and worst > 1e-8),
        "at_the_gates_own_level": {
            "n": gate_level.get("n"),
            "worst_rel_diff": gate_level.get("worst_rel_diff"),
            "residual_span": gate_level.get("residual_span"),
            "cond_1_estimate": gate_level.get("cond_1_estimate"),
        },
        "cudss_gap_vs_same_backend_spread": cross,
        "this_host_vs_the_lane": same_solver,
        "reading": _control_reading(gate_level, cross, worst),
    }


def _control_reading(gate_level: dict[str, Any], cross: dict[str, Any],
                     worst: float | None) -> str:
    """The control's verdict sentence, written from its own numbers."""
    if worst is None:
        return "no control measured"
    head = (f"host SuperLU disagrees with ITSELF by up to {worst:.3e} in G1's norm on "
            f"these systems, {worst / 1e-8:.0f}x the gate's 1e-8, so that clause cannot "
            f"be met by any pair of direct solves of this operator "
            f"(cond_1 ~ {gate_level.get('cond_1_estimate') or float('nan'):.2e} at the "
            f"gate's level) and its failure is a property of the operator, not of "
            f"cuDSS.")
    for be, v in cross.items():
        head += (f" At the gate's own level the {be}-vs-SuperLU difference is "
                 f"{v['cross_backend_rel_diff']:.3e}, against "
                 f"{v['same_backend_rel_diff_span'][0]:.3e}.."
                 f"{v['same_backend_rel_diff_span'][1]:.3e} between SuperLU's own "
                 f"elimination paths -- {v['where']}.")
    return head + (" It is a control, not a licence: the tolerance stays at 1e-8 and "
                   "G1 is still reported failing.")


def g2_gate(j0: dict[str, Any] | None, j1: dict[str, Any] | None) -> dict[str, Any]:
    """AD through cuDSS vs FD4, one parameter.

    Two independent instances, both reported: the W/3 gradient against the
    CPU study's RECORDED FD4 (that stencil cost 5033 s on the Mac and is
    already in ``spiral_convergence.json``), and, if the scale lane got to
    it, a fresh FD4 on one parameter computed through cuDSS itself.
    """
    out: dict[str, Any] = {"what": "AD through cudss vs FD4, one parameter",
                           "tolerance": TOL_FD, "instances": {}}
    g = (j0 or {}).get("gates", {}).get("G2_at_W3_vs_recorded_cpu_fd4")
    if g:
        out["instances"]["W3_vs_recorded_cpu_fd4"] = g
    for k, v in ((j1 or {}).get("levels") or {}).items():
        fd = v.get("fd_one_parameter")
        if fd and fd.get("rel") is not None:
            out["instances"][f"W{k}_fresh_fd4_through_cudss"] = fd
    worst_list = []
    for v in out["instances"].values():
        r = v.get("rel")
        if isinstance(r, list):
            worst_list.append(max(r))
        elif isinstance(r, (int, float)):
            worst_list.append(r)
        elif v.get("worst") is not None:
            worst_list.append(v["worst"])
    out["worst"] = max(worst_list) if worst_list else None
    out["passed"] = bool(worst_list) and max(worst_list) <= TOL_FD
    if not worst_list:
        out["reason"] = "no FD4 instance in the harvested lanes"
    return out


def g3_gate(j0: dict[str, Any] | None, j1: dict[str, Any] | None,
            j2: dict[str, Any] | None, levels: dict[str, Any]) -> dict[str, Any]:
    """The cost table. Not pass/fail on a threshold -- it is the measurement
    the whole lane exists to produce -- but it FAILS if the level the CPU
    reference exists at was never measured on the GPU."""
    rows: list[dict[str, Any]] = []
    per = ((j0 or {}).get("g1") or {}).get("per_backend") or {}
    for b, r in per.items():
        rows.append({
            "case": f"W/3 DUT fixture, one system, {b}",
            "n": ((j0 or {}).get("g1") or {}).get("n"),
            "factor_seconds": r.get("factor_seconds"),
            "solve_seconds": r.get("solve_seconds"),
            "device_gb": (r.get("device_memory_gb_after") or {}).get("used"),
            "lu_nnz": ((r.get("stats") or {}).get("A") or {}).get("lu_nnz"),
            # SuperLU's factor is on the HOST: its device figure is whatever
            # CUDA context and pool the GPU backends left behind in the same
            # process, not memory SuperLU used.
            "device_gb_is_the_backends_own": b != "superlu",
        })
    for src, tag in ((j0, "J0"), (j1, "J1")):
        for be, r in ((src or {}).get("value_and_grad") or {}).items():
            if "error" in r:
                continue
            rows.append({
                "case": f"{tag} W/3 value_and_grad, {be}" if tag == "J0" else
                        f"{tag} value_and_grad, {be}",
                "n": ((j0 or {}).get("g1") or {}).get("n") if tag == "J0" else None,
                "value_and_grad_seconds": r.get("value_and_grad_seconds"),
                "device_gb": (r.get("device_memory_gb_peak_after") or {}).get("used"),
                "factorizations": r.get("factorizations"),
            })
    for k, v in sorted(levels.items(), key=lambda kv: float(kv[0])):
        if v.get("backend") == "superlu":
            continue
        st = (v.get("stats") or {})
        rows.append({
            "case": f"W/{v['div']:g} value_and_grad, {v.get('backend')}",
            "n": v["n_unknowns"],
            "value_and_grad_seconds": v.get("value_and_grad_seconds"),
            "solve_seconds": v.get("solve_seconds"),
            "device_gb": (v.get("device_memory_gb_peak_after") or {}).get("used"),
            "lu_nnz": (st.get("A") or {}).get("lu_nnz"),
        })
    for k, v in ((j2 or {}).get("levels") or {}).items():
        rows.append({
            "case": f"paper geometry W/{v.get('div')}, {v.get('backend')}",
            "n": (v.get("grid") or {}).get("n_unknowns"),
            "solve_seconds": v.get("solve_seconds"),
            "device_gb": (v.get("device_memory_gb") or {}).get("used"),
            "L_diff": v.get("L_diff"), "gap": v.get("gap"),
        })
    plans = {}
    for src in (j0, j2):
        for k, v in ((src or {}).get("plan_estimates") or {}).items():
            plans.setdefault(f"{(src or {}).get('lane', '?')}:W/{k}", {
                "n": v.get("n_unknowns") or (v.get("grid") or {}).get("n_unknowns"),
                "permanent_device_gb": v.get("permanent_device_memory_gb"),
                "peak_device_gb": v.get("peak_device_memory_gb"),
                "plan_seconds": v.get("plan_seconds"),
                "error": v.get("error")})
    gpu_rows = [r for r in rows if r.get("value_and_grad_seconds") and "cudss" in r["case"]]
    return {
        "what": ("N, factorisation s, solve s and device GB per level, against the CPU "
                 "W/3 reference"),
        "cpu_reference": {"case": "W/3 value_and_grad, superlu, this Mac",
                          "n": 108898,
                          "value_and_grad_seconds": CPU_W3_VALUE_AND_GRAD_SECONDS,
                          "value_and_grad_seconds_recorded": CPU_W3_RECORDED_SECONDS,
                          "host_gb": 5.0},
        "rows": rows,
        "cudss_plan_estimates": plans,
        "passed": bool(gpu_rows),
        "reason": None if gpu_rows else "no cuDSS value_and_grad timing harvested",
    }


def g4_g5_gates(study: Any, cpu: dict[str, Any],
                levels: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """V1 and V4, recomputed on the extended ladder (gates G4 and G5).

    The Richardson machinery is ``spiral_convergence.richardson`` itself
    (same function, same three-level observed-order solve, applied to five
    points instead of three); the referee is the ``deembedded`` block of the
    CPU study's JSON, i.e. the PRIMARY convention L(strip) - L(bridge) =
    333.055 pH.
    """
    dee = cpu["referee"]["deembedded"]["total"]
    keys = sorted(levels, key=float)
    h = [levels[k]["base_dx"] for k in keys]
    vals = [levels[k]["L_dut"] for k in keys]
    rich = study.richardson(h, vals)
    rel = [v / dee - 1.0 for v in rich["range"]]
    ext = {
        "levels": keys,
        "cells_across_width": [levels[k]["div"] for k in keys],
        "n_unknowns": [levels[k]["n_unknowns"] for k in keys],
        "backend": [levels[k]["backend"] for k in keys],
        "h": h, "L_dut": vals,
        "per_level_gap": {f"W/{levels[k]['div']:g}": levels[k]["L_dut"] / dee - 1.0
                          for k in keys},
        "richardson": rich,
        "referee_deembedded": dee,
        "rel_range": rel,
    }
    cpu_keys = [k for k in keys if levels[k]["backend"] == "superlu"]
    rich_cpu = study.richardson([levels[k]["base_dx"] for k in cpu_keys],
                                [levels[k]["L_dut"] for k in cpu_keys])
    ext["cpu_only_richardson"] = rich_cpu
    ext["cpu_only_rel_range"] = [v / dee - 1.0 for v in rich_cpu["range"]]
    g4 = {
        "what": ("the extended Richardson: per-level gap, observed order over the finest "
                 "three levels, extrapolated range vs the bridge-corrected referee"),
        "n_levels": len(keys),
        "levels": [f"W/{levels[k]['div']:g}" for k in keys],
        "per_level_gap": ext["per_level_gap"],
        "observed_order": rich["observed_order"],
        "L_extrapolated_estimates": rich["estimates"],
        "L_extrapolated_range": rich["range"],
        "referee_deembedded": dee,
        "rel_range": rel,
        "cpu_only_rel_range": ext["cpu_only_rel_range"],
        "tolerance": TOL_V1,
        "enough_levels": len(keys) >= 4,
        "passed": bool(len(keys) >= 4 and max(abs(rel[0]), abs(rel[1])) <= TOL_V1),
    }
    dw = [(levels[k]["grad"] or [None, None, None])[2] for k in keys]
    have = [(k, v) for k, v in zip(keys, dw) if v is not None]
    hk = [k for k, _ in have]
    dwv = [v for _, v in have]
    steps = [abs(b - a) for a, b in zip(dwv, dwv[1:])]
    fine = steps[-3:] if len(steps) >= 3 else []
    g5 = {
        "what": ("V4 on the extended levels: the level-to-level change of dL/dwidth must "
                 "keep decreasing. Same definition as the CPU study's V4 (every "
                 "consecutive step smaller than the one before), applied to the longer "
                 "ladder, plus the same question asked of the FINEST three steps alone -- "
                 "which is the one that speaks about the limit. The CPU verdict already "
                 "failed on its first pair (7.692e-07 then 9.085e-07, W/1->W/2->W/3): the "
                 "coarsest level is not in the asymptotic regime, and no number of finer "
                 "levels can retroactively put it there."),
        "levels": [f"W/{levels[k]['div']:g}" for k in hk],
        "dL_dwidth": dwv,
        "level_to_level_change": steps,
        "decreasing_all_steps": bool(len(steps) >= 2
                                     and all(b < a for a, b in zip(steps, steps[1:]))),
        "decreasing_finest_three_steps": bool(
            len(fine) == 3 and all(b < a for a, b in zip(fine, fine[1:]))),
        "finest_three_steps": fine,
        "n_levels": len(hk),
        "cpu_verdict_was": "failed (its first pair already increased)",
        "passed": bool(len(steps) >= 3 and all(b < a for a, b in zip(steps, steps[1:]))),
    }
    return g4, g5, ext


def annotate_extended_backends(ext: dict[str, Any], j0: dict[str, Any] | None,
                               j1: dict[str, Any] | None,
                               cpu: dict[str, Any]) -> None:
    """Say, in the ladder's own block, what mixing backends across it costs.

    ``extended.backend`` is ``[superlu, superlu, superlu, cudss, cudss]``: the
    three CPU levels come from ``spiral_convergence.json`` and the two new
    ones from the GPU lane, so the series that decides G4 is not solved by one
    solver throughout. The size of that seam is measured rather than argued:
    the SAME W/3 level solved by cuDSS on the lane against the recorded CPU
    L_dut, next to the level-to-level change the verdict turns on (W/4 ->
    W/6). It is also mechanically checked that the lane used the CPU study's
    fixture, field for field -- a fixture difference, unlike a solver
    difference, would be physics.
    """
    seam: dict[str, Any] = {"what": ("the W/3 level solved by both backends: the seam "
                                     "the mixed ladder introduces, against the change "
                                     "the verdict rests on")}
    cands: dict[str, float] = {}
    for be, r in ((j0 or {}).get("value_and_grad") or {}).items():
        lv3 = (cpu.get("levels") or {}).get("3") or {}
        if be != "superlu" and "error" not in r and lv3.get("L_dut"):
            cands["j0_value_and_grad"] = abs(r["L_dut"] - lv3["L_dut"]) / abs(lv3["L_dut"])
    steps = ((j1 or {}).get("l_dut_vs_ir") or {}).get("steps") or {}
    rels = [v.get("L_dut_rel_vs_cpu") for v in steps.values()
            if v.get("L_dut_rel_vs_cpu") is not None]
    if rels:
        cands["j1_l_dut_vs_ir_best"] = min(rels)
        cands["j1_l_dut_vs_ir_worst"] = max(rels)
    seam["w3_L_dut_cross_backend_rel"] = cands
    keys = sorted(ext["levels"], key=float)
    vals = dict(zip(keys, ext["L_dut"]))
    if "4" in vals and "6" in vals:
        turn = abs(vals["6"] - vals["4"]) / abs(vals["4"])
        seam["finest_pair_change_rel"] = turn
        seam["finest_pair_change_pH"] = (vals["6"] - vals["4"]) * 1e12
        worst = max(cands.values()) if cands else None
        if worst:
            seam["change_over_seam_ratio"] = turn / worst
            seam["reading"] = (
                f"the W/4 -> W/6 turn-down that decides G4 is {turn:.3e} relative "
                f"({seam['finest_pair_change_pH']:+.2f} pH); the largest cross-backend "
                f"disagreement on the same level is {worst:.3e}, i.e. "
                f"{turn / worst:.2e} times smaller. The seam cannot produce the "
                "verdict.")
    ext["backend_note"] = seam
    fx_lane = (j1 or {}).get("fixture") or (j0 or {}).get("fixture") or {}
    fx_cpu = cpu.get("fixture") or {}
    diffs = {k: [fx_cpu.get(k), fx_lane.get(k)]
             for k in sorted(set(fx_cpu) | set(fx_lane))
             if fx_cpu.get(k) != fx_lane.get(k)}
    ext["fixture"] = fx_cpu
    ext["fixture_lane_vs_cpu"] = {
        "what": ("every fixture field of the GPU lane against the CPU study's, so the "
                 "five levels are one series by construction and not by assertion"),
        "fields": len(set(fx_cpu) | set(fx_lane)),
        "differences": diffs,
        "identical": bool(fx_cpu and fx_lane and not diffs),
    }


def g6_gate(ledger: dict[str, Any]) -> dict[str, Any]:
    runs = ledger.get("runs", [])
    left = [r for r in runs if str(r.get("state", "")).lower()
            in ("running", "pending", "initializing", "idle")]
    return {
        "what": "every run accounted for: completed or terminated, none left running",
        "n_runs": len(runs),
        "states": {r["id"]: r.get("state") for r in runs},
        "still_running": [r["id"] for r in left],
        "passed": bool(runs) and not left,
        "reason": None if runs else "empty run ledger",
    }


def figure(study_json: dict[str, Any], path: pathlib.Path) -> None:
    """L vs 1/cells: the CPU levels, the GPU levels, the referee and the band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = study_json["extended"]
    dee = ext["referee_deembedded"]
    cells = np.asarray(ext["cells_across_width"], dtype=float)
    lvals = np.asarray(ext["L_dut"], dtype=float) * 1e12
    backend = ext["backend"]
    x = 1.0 / cells
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=140)
    ax.axhline(dee * 1e12, color="k", lw=1.4,
               label=f"referee L(strip)-L(bridge) = {dee * 1e12:.2f} pH")
    ax.axhspan(dee * 1e12 * (1 - TOL_V1), dee * 1e12 * (1 + TOL_V1),
               color="k", alpha=0.07, label=f"+-{TOL_V1 * 100:.0f} % (V1 tolerance)")
    rng = np.asarray(study_json["gates"]["G4"]["L_extrapolated_range"]) * 1e12
    ax.axhspan(rng.min(), rng.max(), color="tab:green", alpha=0.18,
               label=f"Richardson range {rng.min():.1f}..{rng.max():.1f} pH")
    cpu = [i for i, b in enumerate(backend) if b == "superlu"]
    gpu = [i for i, b in enumerate(backend) if b != "superlu"]
    if cpu:
        ax.plot(x[cpu], lvals[cpu], "o-", color="tab:blue", label="CPU SuperLU (study D2)")
    if gpu:
        ax.plot(x[gpu], lvals[gpu], "s", color="tab:red", ms=8,
                label="GPU cuDSS (this study)")
        both = sorted(cpu + gpu)
        ax.plot(x[both], lvals[both], "-", color="tab:red", lw=0.8, alpha=0.5)
    for i, (xi, yi, n) in enumerate(zip(x, lvals, ext["n_unknowns"])):
        up = backend[i] != "superlu"          # GPU labels above, CPU below
        # the two finest levels sit close together on 1/cells: push their
        # labels apart rather than letting them overprint
        dx = 0.0 if not up else (-8.0 if i == len(x) - 1 else 8.0)
        ax.annotate(f"N={n}", (xi, yi), textcoords="offset points",
                    xytext=(dx, 10 if up else -14),
                    ha="right" if dx < 0 else ("left" if dx > 0 else "center"),
                    fontsize=7, color="0.35")
    gaps = list(ext["per_level_gap"].values())[-3:]
    ax.annotate(
        "in-plane refinement plateaus here:\n"
        + " / ".join(f"{100 * g:+.2f}" for g in gaps) + " % of the referee",
        xy=(x[-1], lvals[-1]), xytext=(0.06, 0.62), textcoords="axes fraction",
        fontsize=8, color="tab:red",
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8))
    ax.set_xlabel("1 / cells across the strip width  (0 = continuum)")
    ax.set_ylabel("de-embedded $L_{diff}$  (pH)")
    ax.set_xlim(left=0.0)
    ax.set_title("Study V: the spiral's in-plane convergence, extended by the GPU levels")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=pathlib.Path, default=RUNS_DIR)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--ordering-control", metavar="DIVS",
                    help="measure G1's same-backend control on this host (no GPU) for "
                         "these levels, e.g. --ordering-control 1,3, cache it in "
                         f"{CONTROL_JSON.name} and exit. Minutes per case at W/3.")
    args = ap.parse_args(argv)

    t0 = time.time()
    study = load_study()
    if args.ordering_control:
        divs = tuple(float(v) for v in args.ordering_control.split(","))
        run_ordering_control(study, divs)
        return 0
    cpu = json.loads(CPU_JSON.read_text())
    lanes = lane_records(args.runs)
    j0, j1, j2 = lanes.get("j0"), lanes.get("j1"), lanes.get("j2")
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    print(f"lanes harvested: {sorted(lanes)}")

    control = json.loads(CONTROL_JSON.read_text()) if CONTROL_JSON.exists() else None
    levels = merged_levels(cpu, j1)
    g4, g5, ext = g4_g5_gates(study, cpu, levels)
    annotate_extended_backends(ext, j0, j1, cpu)
    out: dict[str, Any] = {
        "what": __doc__.splitlines()[0],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": {
            "backends": ["superlu", "cudss", "cusolver"],
            "default": "superlu",
            "env_override": "RFX_FDFD_BACKEND",
            "transposed_solve": ("cuDSS has no transposed solve, so the adjoint system "
                                 "factorises A^T as a second problem inside the same "
                                 "factor object: a gradient costs two cuDSS "
                                 "factorisations and two device-resident factors "
                                 "against SuperLU's one of each"),
            "environment": (j0 or {}).get("env") or (j1 or {}).get("env"),
        },
        "lanes": {k: {"dir": v.get("_dir"), "file": v.get("_file"),
                      "seconds": v.get("seconds"), "started_utc": v.get("started_utc"),
                      "finished_utc": v.get("finished_utc"),
                      "config": v.get("config"), "blocked": v.get("blocked")}
                  for k, v in lanes.items()},
        "runs": ledger.get("runs", []),
        "levels": levels,
        "extended": ext,
        "referee": cpu["referee"],
        "cpu_study": {"levels": {k: {"L_dut": v["L_dut"], "grad": v["grad"],
                                     "n_unknowns": v["grid"]["n_unknowns"]}
                                 for k, v in cpu["levels"].items()},
                      "richardson": cpu["richardson"],
                      "V1": {"rel_range": cpu["gates"]["V1"]["rel_range"],
                             "passed": cpu["gates"]["V1"]["passed"]},
                      "V4": cpu["gates"]["V4"]},
    }
    if j2:
        out["paper_geometry"] = {
            "what": ("study H's paper-derived square spiral (3 turns, r_out 218 um, "
                     "W 30 um, S 14 um, TopMetal2 3 um) at 100 MHz under D2's "
                     "uniform-current protocol, solved through cuDSS"),
            "fixture": j2.get("fixture"),
            "referee_deembedded": j2.get("referee_deembedded"),
            "levels": {k: {kk: v.get(kk) for kk in
                           ("div", "base_dx", "L_diff", "L_raw", "Q_diff", "gap",
                            "solve_seconds", "error", "skipped")}
                       | {"n_unknowns": (v.get("grid") or {}).get("n_unknowns"),
                          "cells_across_width": (v.get("grid") or {}).get(
                              "cells_across_width"),
                          "device_gb": (v.get("device_memory_gb") or {}).get("used")}
                       for k, v in (j2.get("levels") or {}).items()},
            "plan_estimates": j2.get("plan_estimates"),
            "cusolver_probe": j2.get("cusolver_probe"),
            "cpu_study_H": (j2.get("cpu_study_H") or {}).get("A"),
        }
    gates: dict[str, Any] = {}
    gates["G1"] = _g1_with_cpu_fallback(j0, cpu, j1)
    gates["G1"]["same_backend_control"] = g1_control(control, gates["G1"])
    gates["G2"] = g2_gate(j0, j1)
    gates["G3"] = g3_gate(j0, j1, j2, levels)
    gates["G4"] = g4
    gates["G5"] = g5
    gates["G6"] = g6_gate(ledger)
    out["gates"] = gates
    # --- what the extended ladder says, in one place --------------------
    ext_gap = list(ext["per_level_gap"].values())
    out["verdict"] = {
        "backend": (
            "cuDSS through nvmath-python works on this operator and is 147x faster on "
            "one factorisation (139.23 s -> 0.95 s at N = 108898) and 63x on a whole "
            "three-fixture value_and_grad (432.9 s -> 6.9 s), with 1.38 GB of device "
            "memory per factor there and 12.72 GB peak at N = 352536. It agrees with "
            "SuperLU on L_dut to 1.5e-10 and on the gradient to 1.5e-08, and DISAGREES "
            "on the raw solution vector by 2.4e-03 -- which is the operator, not the "
            "solver: on the SAME system host SuperLU disagrees with itself by "
            "2.9e-03..3.7e-03 over its own orderings and elimination paths "
            "(gates.G1.same_backend_control, cond_1 3.05e+13), and cuDSS's difference "
            "from it sits INSIDE that spread. G1 is still reported failing: its 1e-8 "
            "solution clause is unreachable for any pair of direct solves here."),
        "physics": (
            "The three CPU levels rose monotonically towards the referee (-21.20, "
            "-12.03, -9.76 %) and their Richardson extrapolation projected -7.9..-5.2 %. "
            "The two GPU levels do NOT continue that: W/4 is -9.26 % and W/6 is "
            "-9.46 %, i.e. the sequence flattens and then turns back, and the "
            "extrapolation from the finest pair lands at -9.86..-9.62 %. In-plane "
            "refinement is therefore EXHAUSTED by W/3-W/4 at this fixture's vertical "
            "resolution, and the remaining ~9.4 % is not in-plane discretisation "
            "error. The same thing happens on the paper geometry: -13.22 % at 2 cells "
            "across the strip and -15.76 % at 6."),
        "caveat": (
            "The levels are not a NESTED refinement. base_dx caps the cell size where "
            "the geometry leaves it free, every polygon edge is a mandatory grid line "
            "and the walls are graded (rfx.fdfd.gds.mesh_lines), so each level is its "
            "own mesh; base_dz and metal_cells are held fixed, so the vertical error is "
            "common to all five and cancels out of this comparison rather than being "
            "measured by it. spiral_convergence.json's own vertical_budget puts that "
            "term at +1.9..+2.5 % with the sign that would close the gap, and its "
            "post-short correction has the sign that would widen it; neither is applied "
            "here."),
        "per_level_gap_percent": [100 * g for g in ext_gap],
    }
    out["seconds"] = time.time() - t0
    JSON_PATH.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {JSON_PATH}")
    if not args.no_figure:
        figure(out, PNG_PATH)
        print(f"wrote {PNG_PATH}")
    for k, v in gates.items():
        print(f"  {k}: passed={v.get('passed')}  {v.get('reason') or ''}")
    print(f"  extended levels: {gates['G4']['levels']}")
    print("  per-level gap: "
          + ", ".join(f"{k} {100 * v:+.2f} %" for k, v in gates["G4"]["per_level_gap"].items()))
    ctl = gates["G1"]["same_backend_control"]
    if ctl.get("per_level"):
        print("  G1 same-backend control (host SuperLU vs itself): "
              + ", ".join(f"{k} N={v['n']} worst diff {v['worst_rel_diff']:.3e}"
                          for k, v in ctl["per_level"].items()
                          if v.get("worst_rel_diff") is not None))
    print(f"  Richardson rel range: "
          f"{100 * gates['G4']['rel_range'][0]:+.2f} .. "
          f"{100 * gates['G4']['rel_range'][1]:+.2f} % "
          f"(observed order {gates['G4']['observed_order']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
