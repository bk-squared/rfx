"""Gradient descent through the FDTD solve for a three-layer anti-reflection coating on eps_r = 12.

A normal-incidence plane wave in a 1-D-equivalent FDTD model (periodic y and z,
CPML along x) meets three quarter-wave layers in front of an eps_r = 12
substrate; the three layer permittivities are the design variables and the cost
is the mean reflected power |R(f)|^2 over 8-12 GHz.  The model, the cost and
the Adam settings are those of ``examples/inverse_design/multilayer_ar_coating.py``,
imported by path (its builders and constants, never copied, never edited).  The
transfer-matrix method (TMM) gives the same cost in closed form.

What is recorded, per the pre-declaration
(``docs/design_notes/20260927_showcase_predeclaration.md``, section "AR
coating"): every iterate's three eps_r, FDTD cost, FDTD R(f) on the FFT band
bins, gradient, and TMM R(f) at the same eps_r; the TMM optimum; the geometric
ladder; wall time per iteration; autodiff against central differences at the
start point; FDTD against TMM R(f) at the final iterate.  The example's own
``main()`` is also run once, and its printed trace is compared with this
script's iterates.

    python scripts/showcase/ar_coating.py --out DIR
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))       # this checkout's rfx, not an installed one
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _record  # noqa: E402

CASE_ID = "ar-coating"
QUESTION = ("Does Adam through the differentiable FDTD solve reach the transfer-matrix "
            "optimum's X-band mean reflection for a three-layer coating on eps_r = 12, "
            "with the gradient checked against central differences?")
EXAMPLE = REPO / "examples" / "inverse_design" / "multilayer_ar_coating.py"

# ------------------------------------------------------------ pre-declared
FD_STEPS = (0.02, 0.01, 0.005)   # in eps_r, per component
FD_JUDGED_STEP = 0.01
FD_REL_BAR = 0.05
WITNESS_FACTOR = 1.5             # record-length witness: 1.5 x the example's N_STEPS
WITNESS_TOL = 0.05               # Amendment 1 (lane leader): judged per component
# The example's gates (examples/inverse_design/multilayer_ar_coating.py, main()):
GATE_TMM_RATIO = 1.2
GATE_GEO_RATIO = 0.7
GATE_MIN_INTERMEDIATE = 2
# Values that live inside the example's main() rather than at module level; they
# are restated here and the example's own run is compared with this one.
TMM_X0 = (2.5, 4.0, 7.0)         # its L-BFGS-B start
N_ITERS = 60
LR = 0.15
BETA1, BETA2, ADAM_EPS = 0.9, 0.999, 1e-8


def load_example(name: str = "_showcase_ar_example"):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_record._jsonable) + "\n")
    tmp.replace(path)


class Pipeline:
    """The example's reflection pipeline, parameterised by record length."""

    def __init__(self, ex, n_steps: int, nfft: int | None = None):
        import jax.numpy as jnp
        self.ex = ex
        self.n_steps = n_steps
        self.nfft = nfft or int(2 ** np.ceil(np.log2(n_steps)))
        ref = ex._build_simulation().forward(eps_override=ex._vacuum_eps_array(),
                                             n_steps=n_steps, skip_preflight=True)
        self.ts_inc = jnp.asarray(ref.time_series[:, 0])
        self.freqs = np.fft.rfftfreq(self.nfft, d=ex.DT)
        self.band = (self.freqs >= ex.F_LO) & (self.freqs <= ex.F_HI)
        self.band_j = jnp.asarray(self.band)
        self.band_norm = float(np.sum(self.band))
        self.sim = ex._build_simulation()

    def cost_and_R(self, layer_eps):
        import jax.numpy as jnp
        eps = self.ex._render_design_eps(layer_eps)
        ts = self.sim.forward(eps_override=eps, n_steps=self.n_steps,
                              skip_preflight=True).time_series
        s_inc = jnp.fft.rfft(self.ts_inc, n=self.nfft)
        s_scat = jnp.fft.rfft(ts[:, 0] - self.ts_inc, n=self.nfft)
        R = (jnp.abs(s_scat) / (jnp.abs(s_inc) + 1e-30)) ** 2
        return jnp.sum(R * self.band_j) / self.band_norm, R


def run_example_main(ex) -> dict:
    """The example's own ``main()``, its stdout parsed into a trace."""
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        rc = ex.main()
    wall = time.perf_counter() - t0
    text = buf.getvalue()
    trace = {}
    for m in re.finditer(r"iter\s+(\d+)\s+cost = ([0-9.eE+-]+)\s+εr = \[([^\]]*)\]", text):
        trace[int(m.group(1))] = {"cost": float(m.group(2)),
                                  "eps": [float(v) for v in m.group(3).split(",")]}
    fin = re.search(r"εr \(rfx final\): \[([^\]]*)\]", text)
    return {"rc": int(rc), "wall_s": wall, "trace": trace,
            "final_eps": [float(v) for v in fin.group(1).split(",")] if fin else None,
            "stdout": text}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repo-dir", type=Path, default=REPO)
    ap.add_argument("--iters", type=int, default=N_ITERS, help="smoke runs only")
    ap.add_argument("--skip-example-main", action="store_true", help="smoke runs only")
    a = ap.parse_args(argv)
    out = a.out
    out.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize as scimin

    # the tree as the job received it, before anything below writes a file
    source = _record.source_block(a.repo_dir, precision="float32")
    ex = load_example()
    wall = {}

    # ---- closed form: TMM optimum and the geometric ladder -------------------
    res_tmm = scimin(ex.tmm_band_cost, x0=np.array(TMM_X0),
                     bounds=[(1.0, ex.EPS_SUB)] * ex.N_LAYERS, method="L-BFGS-B")
    tmm_eps = np.asarray(res_tmm.x)
    tmm_cost = float(res_tmm.fun)
    geo = np.array([ex.EPS_SUB ** ((i + 1) / (ex.N_LAYERS + 1)) for i in range(ex.N_LAYERS)])
    geo_cost = ex.tmm_band_cost(geo)
    fs_tmm51 = np.linspace(ex.F_LO, ex.F_HI, 51)
    realized_thk = ex.LAYER_CELLS[0] * ex.DX
    tmm = {"optimum_eps_r": tmm_eps, "optimum_cost": tmm_cost, "lbfgsb_success": bool(res_tmm.success),
           "lbfgsb_x0": TMM_X0, "geometric_ladder_eps_r": geo, "geometric_ladder_cost": geo_cost,
           "cost_grid_hz": fs_tmm51, "layer_thickness_m": ex.LAYER_THK,
           "layer_cells": ex.LAYER_CELLS, "realized_layer_thickness_m": [c * ex.DX for c in ex.LAYER_CELLS],
           "optimum_cost_at_realized_thickness": float(np.mean(ex.tmm_R(tmm_eps, fs_tmm51,
                                                                        layer_thk=realized_thk)))}
    _save_json(out / "tmm.json", tmm)
    _log(f"TMM optimum eps_r {np.round(tmm_eps, 4)} cost {tmm_cost:.4e}; geometric ladder "
         f"{np.round(geo, 4)} cost {geo_cost:.4e}")

    model = {"structure": "1-D-equivalent FDTD: vacuum | 3 layers | eps_r 12 substrate "
                          "(examples/inverse_design/multilayer_ar_coating.py, imported by path)",
             "example": str(EXAMPLE.relative_to(REPO)),
             "eps_sub": ex.EPS_SUB, "n_layers": ex.N_LAYERS, "band_hz": [ex.F_LO, ex.F_HI],
             "dx_m": ex.DX, "nx": ex.NX, "dt_s": ex.DT, "n_steps": ex.N_STEPS,
             "n_periods_at_f0": ex.N_PERIODS, "layer_edge_cells": ex.LAYER_EDGES_C,
             "layer_cells": ex.LAYER_CELLS, "tmm_layer_thickness_m": ex.LAYER_THK,
             "realized_layer_thickness_m": [c * ex.DX for c in ex.LAYER_CELLS],
             "substrate_cells": [ex.i_sub_lo, ex.i_sub_hi],
             "cost": "mean over the FFT band bins of |FFT(ts_total - ts_inc)|^2 / |FFT(ts_inc)|^2 "
                     "at the reflection probe",
             "adam": {"iters": a.iters, "lr": LR, "beta1": BETA1, "beta2": BETA2, "eps": ADAM_EPS,
                      "start": "geometric ladder, latent = logit((eps - 1)/(eps_sub - 1))"},
             "jit": "jax.jit(jax.value_and_grad(cost, has_aux=True)); the example calls "
                    "value_and_grad without jit"}
    _save_json(out / "model.json", model)

    # ---- the pipeline at the example's record length ---------------------------
    t0 = time.perf_counter()
    pipe = Pipeline(ex, ex.N_STEPS)
    wall["vacuum_reference_s"] = time.perf_counter() - t0
    freqs_band = pipe.freqs[pipe.band]
    # the closed-form curves: the TMM optimum and the geometric ladder, on the
    # FDTD's band bins and on the example's 51-point cost grid
    np.savez(out / "tmm_curves.npz", freqs_band_hz=freqs_band, freqs_51_hz=fs_tmm51,
             R_opt_band=ex.tmm_R(tmm_eps, freqs_band), R_geo_band=ex.tmm_R(geo, freqs_band),
             R_opt_51=ex.tmm_R(tmm_eps, fs_tmm51), R_geo_51=ex.tmm_R(geo, fs_tmm51),
             eps_opt=tmm_eps, eps_geo=geo)
    model["nfft"] = pipe.nfft
    model["n_band_bins"] = int(pipe.band.sum())
    _save_json(out / "model.json", model)

    def cost_latent(latent):
        return pipe.cost_and_R(ex._latent_to_eps(latent))

    def cost_eps(layer_eps):
        return pipe.cost_and_R(layer_eps)[0]

    vg = jax.jit(jax.value_and_grad(cost_latent, has_aux=True))
    cost_eps_j = jax.jit(cost_eps)
    grad_eps_j = jax.jit(jax.grad(cost_eps))

    # ---- AD against central differences at the start point (eps_r space) --------
    eps0 = jnp.asarray(geo, dtype=jnp.float32)
    t0 = time.perf_counter()
    ad = np.asarray(grad_eps_j(eps0), dtype=float)
    wall["grad_eps_first_call_s"] = time.perf_counter() - t0
    c0 = float(cost_eps_j(eps0))
    fd = {"precision": "float32", "eps0": geo, "cost0": c0, "ad": ad, "fd": {}, "rows": []}
    for h in FD_STEPS:
        comp = []
        for i in range(ex.N_LAYERS):
            e = np.zeros(ex.N_LAYERS, dtype=np.float32)
            e[i] = h
            cp, cm = float(cost_eps_j(eps0 + e)), float(cost_eps_j(eps0 - e))
            comp.append({"c_plus": cp, "c_minus": cm, "fd": (cp - cm) / (2 * h)})
        fd["fd"][str(h)] = comp
    for i in range(ex.N_LAYERS):
        f_j = fd["fd"][str(FD_JUDGED_STEP)][i]["fd"]
        rel = abs(ad[i] - f_j) / abs(f_j) if f_j else float("inf")
        fd["rows"].append({"component": i, "ad": float(ad[i]), "fd_judged_step": f_j,
                           "rel": rel, "passed": rel <= FD_REL_BAR,
                           "rel_by_step": {str(h): abs(ad[i] - fd["fd"][str(h)][i]["fd"])
                                           / abs(fd["fd"][str(h)][i]["fd"]) for h in FD_STEPS}})
    _save_json(out / "fd_start.json", fd)
    _log("AD vs FD at the start point: " + "; ".join(
        f"eps{r['component']+1}: AD {r['ad']:+.5e} FD {r['fd_judged_step']:+.5e} rel {r['rel']:.3e}"
        for r in fd["rows"]))

    # ---- record-length witness (Amendment 1): rfx's helper, both arms on one FFT ------
    from rfx import gradient_record_length_witness
    n_long = int(np.ceil(WITNESS_FACTOR * ex.N_STEPS))   # the helper's own ceil(factor * n)
    nfft_w = int(2 ** np.ceil(np.log2(n_long)))
    pipes = {n: Pipeline(ex, n, nfft=nfft_w) for n in (ex.N_STEPS, n_long)}

    def witness_objective(e, n):
        return pipes[n].cost_and_R(e)[0]

    w = gradient_record_length_witness(witness_objective, eps0, ex.N_STEPS, tol=WITNESS_TOL,
                                       factor=WITNESS_FACTOR)
    g1 = np.asarray(next(iter(w.grad.values())))[0].astype(float)
    g15 = np.asarray(next(iter(w.grad_long.values())))[0].astype(float)
    rel = np.abs(g15 - g1) / np.abs(g15)
    wit = {"helper": "rfx.gradient_record_length_witness", "nfft": nfft_w,
           "n_steps": {"1.0x": w.n_steps, "1.5x": w.n_steps_long}, "tol": WITNESS_TOL,
           "ad": {"1.0x": g1, "1.5x": g15}, "rel_change": rel.tolist(),
           "passed": [bool(r <= WITNESS_TOL) for r in rel],
           "helper_norm_rel_change": float(w.worst), "helper_cosine": float(w.cosine_by_bin[0]),
           "helper_passed": bool(w.passed),
           "cost": {"1.0x": float(np.real(w.value[0])), "1.5x": float(np.real(w.value_long[0]))},
           "how": "objective(eps, n) = the example's cost from an n-step record, both arms "
                  f"zero-padded to one {nfft_w}-point FFT so only the record length differs"}
    _save_json(out / "record_length_witness.json", wit)
    _log(f"record-length witness (nfft {nfft_w}, {w.n_steps} vs {w.n_steps_long} steps): "
         f"|g_1.5 - g_1.0| / |g_1.5| = {np.round(rel, 6)}, helper norm {w.worst:.3e}")
    witness_claims = [
        _record.claim(f"record-length witness: |g_1.5x - g_1.0x| / |g_1.5x|, d cost / d eps_r{i + 1}, "
                      "start point", float(rel[i]), "1", "record_length_witness.json",
                      threshold=WITNESS_TOL, rule="Amendment 1 (lane leader)")
        for i in range(ex.N_LAYERS)]
    witness_claims.append(_record.claim(
        "record-length witness, helper norm ||g_1.5x - g_1.0x|| / ||g_1.5x||", float(w.worst), "1",
        "record_length_witness.json", note=f"helper passed at tol 0.05: {bool(w.passed)}"))
    if not all(wit["passed"]):
        # Amendment 1: a failed witness stops the case; the record is not lengthened
        fd_claims = [_record.claim(
            f"|AD - FD| / |FD| at h = {FD_JUDGED_STEP}, d cost / d eps_r{r['component'] + 1}, start point",
            r["rel"], "1", "fd_start.json", threshold=FD_REL_BAR,
            rule="pre-declared: central differences at h = 0.01 in eps_r") for r in fd["rows"]]
        _record.write_result(out, {
            "schema": _record.SCHEMA, "id": CASE_ID, "question": QUESTION, "source": source,
            "run": {"platform": "VESSL", "preset": os.environ.get("RFX_SHOWCASE_PRESET"),
                    "run_id": None, "wall_s": wall},
            "model": model, "claims": witness_claims + fd_claims, "derived": [],
            "out_of_scope": ["the Adam descent: not run, the record-length witness failed"]},
            _record.data_files(out))
        _log("record-length witness FAILED: the case stops here (Amendment 1)")
        return 3

    # ---- Adam, as the example runs it, every iterate recorded ----------------------
    p_geo = np.clip((geo - 1.0) / (ex.EPS_SUB - 1.0), 1e-3, 1.0 - 1e-3)
    latent = jnp.asarray(np.log(p_geo / (1.0 - p_geo)), dtype=jnp.float32)
    m = jnp.zeros_like(latent)
    v = jnp.zeros_like(latent)
    rec = {k: [] for k in ("eps_r", "latent", "cost", "R_band", "grad_latent", "grad_eps",
                           "R_tmm_band", "tmm_cost51", "wall_s")}

    def dEps_dLatent(lat):
        s = 1.0 / (1.0 + np.exp(-np.asarray(lat, dtype=float)))
        return (ex.EPS_SUB - 1.0) * s * (1.0 - s)

    def record(latent, cost, R, grad, dt):
        eps_now = np.asarray(ex._latent_to_eps(latent), dtype=float)
        rec["eps_r"].append(eps_now)
        rec["latent"].append(np.asarray(latent, dtype=float))
        rec["cost"].append(float(cost))
        rec["R_band"].append(np.asarray(R)[pipe.band].astype(float))
        rec["grad_latent"].append(np.full(ex.N_LAYERS, np.nan) if grad is None
                                  else np.asarray(grad, dtype=float))
        rec["grad_eps"].append(np.full(ex.N_LAYERS, np.nan) if grad is None
                               else np.asarray(grad, dtype=float) / dEps_dLatent(latent))
        rec["R_tmm_band"].append(ex.tmm_R(eps_now, freqs_band))
        rec["tmm_cost51"].append(ex.tmm_band_cost(eps_now))
        rec["wall_s"].append(dt)

    def persist():
        np.savez(out / "iterations.npz", freqs_band_hz=freqs_band,
                 **{k: np.asarray(v) for k, v in rec.items()})

    for it in range(a.iters):
        t0 = time.perf_counter()
        (cost, R), grad = vg(latent)
        jax.block_until_ready(grad)
        dt = time.perf_counter() - t0
        record(latent, cost, R, grad, dt)
        m = BETA1 * m + (1 - BETA1) * grad
        v = BETA2 * v + (1 - BETA2) * grad ** 2
        m_hat = m / (1 - BETA1 ** (it + 1))
        v_hat = v / (1 - BETA2 ** (it + 1))
        latent = latent - LR * m_hat / (jnp.sqrt(v_hat) + ADAM_EPS)
        persist()
        if it % 5 == 0 or it == a.iters - 1:
            _log(f"iter {it:3d} cost {float(cost):.4e} eps_r {np.round(rec['eps_r'][-1], 3)} "
                 f"({dt:.2f} s)")
    # the final iterate: its cost and R, no step taken
    t0 = time.perf_counter()
    cost_f, R_f = jax.jit(cost_latent)(latent)
    record(latent, cost_f, R_f, None, time.perf_counter() - t0)
    persist()
    wall["adam_total_s"] = float(np.sum(rec["wall_s"]))

    # ---- the example's gates, computed the way the example computes them -------------
    losses = rec["cost"][:a.iters]
    final_cost = losses[-1]              # the example: losses[-1], the cost before the last step
    final_eps = rec["eps_r"][-1]         # the example: eps after the last step
    n_inter = int(np.sum((final_eps > 1.5) & (final_eps < ex.EPS_SUB - 0.5)))
    dR = np.abs(rec["R_band"][-1] - rec["R_tmm_band"][-1])
    gates = {"final_cost": final_cost, "final_eps_r": final_eps, "tmm_cost": tmm_cost,
             "geo_cost": geo_cost, "ratio_to_tmm": final_cost / tmm_cost,
             "ratio_to_geo": final_cost / geo_cost, "n_intermediate": n_inter,
             "cost_at_final_eps": rec["cost"][-1],
             "final_iterate_R_fdtd_vs_tmm": {"mean_abs_dR": float(dR.mean()),
                                             "max_abs_dR": float(dR.max()),
                                             "f_at_max_hz": float(freqs_band[int(np.argmax(dR))])}}
    _save_json(out / "gates.json", gates)
    _log(f"gates: ratio to TMM {gates['ratio_to_tmm']:.3f}, to geometric {gates['ratio_to_geo']:.3f}, "
         f"intermediate layers {n_inter}; final R: mean |dR| {dR.mean():.3e}, max {dR.max():.3e}")

    # ---- the example's own main(), for comparison with this script's iterates ----------
    if not a.skip_example_main:
        _log("running the example's main() (its own loop, unjitted)")
        exm = run_example_main(ex)
        (out / "example_main_stdout.txt").write_text(exm.pop("stdout"))
        cmp_rows = []
        for it, t in sorted(exm["trace"].items()):
            if it < len(losses):
                cmp_rows.append({"iter": it, "example_cost": t["cost"], "this_cost": losses[it],
                                 "rel": abs(t["cost"] - losses[it]) / abs(losses[it]),
                                 "max_abs_eps_diff": float(np.max(np.abs(
                                     np.asarray(t["eps"]) - np.round(rec["eps_r"][it], 3))))})
        exm["compare"] = cmp_rows
        exm["max_rel_cost_diff"] = max((r["rel"] for r in cmp_rows), default=None)
        _save_json(out / "example_main.json", exm)
        wall["example_main_s"] = exm["wall_s"]
        _log(f"example main(): rc {exm['rc']}, max relative cost difference at its printed "
             f"iterations {exm['max_rel_cost_diff']}")

    # ---- result.json -----------------------------------------------------------------
    claims = [
        _record.claim("TMM optimum X-band mean |R|^2", tmm_cost, "1", "tmm.json"),
        _record.claim("geometric-ladder X-band mean |R|^2 (TMM)", geo_cost, "1", "tmm.json"),
        _record.claim("FDTD X-band mean |R|^2 at iterate 0", losses[0], "1", "iterations.npz"),
        _record.claim("final cost / TMM optimum cost (example gate)", gates["ratio_to_tmm"], "1",
                      "gates.json", threshold=GATE_TMM_RATIO,
                      rule="the example's gate: final cost <= 1.2 x TMM optimum"),
        _record.claim("final cost / geometric-ladder cost (example gate)", gates["ratio_to_geo"], "1",
                      "gates.json", threshold=GATE_GEO_RATIO,
                      rule="the example's gate: final cost <= 0.7 x geometric ladder"),
        _record.claim("layers with 1.5 < eps_r < eps_sub - 0.5 (example gate)", n_inter, "layers",
                      "gates.json", threshold=GATE_MIN_INTERMEDIATE, op=">=",
                      rule="the example's gate: >= 2 intermediate layers"),
        _record.claim("FDTD X-band mean |R|^2 at the final eps_r", rec["cost"][-1], "1",
                      "iterations.npz"),
        _record.claim("final iterate: mean |R_FDTD - R_TMM| over the band bins",
                      gates["final_iterate_R_fdtd_vs_tmm"]["mean_abs_dR"], "1", "gates.json"),
        _record.claim("final iterate: max |R_FDTD - R_TMM| over the band bins",
                      gates["final_iterate_R_fdtd_vs_tmm"]["max_abs_dR"], "1", "gates.json"),
        _record.claim("median wall time per Adam iteration (jitted value_and_grad)",
                      float(np.median(rec["wall_s"][1:a.iters])) if a.iters > 1 else rec["wall_s"][0],
                      "s", "iterations.npz"),
    ]
    for r in fd["rows"]:
        claims.append(_record.claim(
            f"|AD - FD| / |FD| at h = {FD_JUDGED_STEP}, d cost / d eps_r{r['component'] + 1}, start point",
            r["rel"], "1", "fd_start.json", threshold=FD_REL_BAR,
            rule="pre-declared: central differences at h = 0.01 in eps_r"))
    claims += witness_claims
    if (out / "example_main.json").is_file():
        claims.append(_record.claim(
            "max relative cost difference, example main() vs this script, at its printed iterations",
            json.loads((out / "example_main.json").read_text())["max_rel_cost_diff"], "1",
            "example_main.json"))
    derived = [
        _record.derived("final cost / TMM optimum cost", gates["ratio_to_tmm"], "1",
                        "losses[59] / tmm_cost", {"losses[59]": final_cost, "tmm_cost": tmm_cost}),
        _record.derived("final cost / geometric-ladder cost", gates["ratio_to_geo"], "1",
                        "losses[59] / geo_cost", {"losses[59]": final_cost, "geo_cost": geo_cost}),
        _record.derived("reflection drop, iterate 0 to 59", 10 * np.log10(losses[0] / final_cost), "dB",
                        "10 log10(losses[0] / losses[59])", {"losses[0]": losses[0],
                                                             "losses[59]": final_cost}),
    ]
    files = _record.data_files(out)
    result = {
        "schema": _record.SCHEMA, "id": CASE_ID, "question": QUESTION,
        "source": source,
        "run": {"platform": "VESSL", "preset": os.environ.get("RFX_SHOWCASE_PRESET"),
                "run_id": None, "wall_s": wall},
        "model": model, "claims": claims, "derived": derived,
        "out_of_scope": ["Meep or any other solver (the example's optional Meep overlay is not run)",
                         "oblique incidence, loss, dispersion",
                         "mesh convergence: one dx, the example's"],
    }
    path = _record.write_result(out, result, files)
    _log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
