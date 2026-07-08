"""Optimizer bake-off at a FIXED forward-solve budget (external-reviewer WP 4-D).

DESIGN-LOOP CAPABILITY EVIDENCE ONLY — never physics validation.  Nothing here
claims a physical result or touches a committed validation gate.  A NEGATIVE
result (adopt no new optimizer) is a first-class outcome.

Question
--------
Should rfx adopt a second optimizer beyond its shipped default
(hand-rolled Adam + multi-start + best-iterate, promoted in PR #286 as
``rfx.optimize(..., n_starts>1, best_iterate=True)``)?

The four contenders each consume the SAME scalar objective ``cost_fn(latent) ->
loss`` per benchmark (reusing the example sim builders + objectives) at an EQUAL
total forward-solve budget:

1. ``adam_rfx``            — rfx's shipped hand-rolled Adam, single start
                            (``rfx.optimize._adam_multistart`` with one init,
                            ``best_iterate=False``; the ``optimize()`` default).
2. ``optax_adam``         — ``optax.adam`` at the same lr, manual loop.
3. ``optax_lbfgs``        — ``optax.lbfgs`` (zoom line search), manual loop.
4. ``adam_multistart_bi`` — rfx's shipped multi-start best-iterate Adam
                            (``_adam_multistart`` with N inits,
                            ``best_iterate=True``; the PR #286 / 4-C default).

Benchmarks (reuse the committed example sim builders + objectives)
------------------------------------------------------------------
* ``ar_coating``      SMOOTH, 3 DoF — multilayer AR coating, X-band mean |R|^2.
* ``msl_stub_notch``  SHARP RESONANT NULL, 1 DoF — MSL open-stub |S21(6 GHz)|^2.
  Coarsened to dx = h_sub (1 substrate cell) so a CPU bake-off is affordable;
  the full-res multimodality collapses to a single sharp -30 dB well here, so
  this case exercises BEST-ITERATE overshoot-guarding (design-loop capability,
  not MSL physics — see ``build_msl_stub_notch``).
* ``waveguide_taper`` BAND-AVERAGED, 12 DoF — WR-90 dielectric taper (SMOKE),
  <|S11|^2> over X-band via the differentiable modal S-matrix.

Budgeting
---------
Budget ``B`` = number of value-and-grad-equivalent objective evaluations (each is
one primal FDTD forward solve; the reverse pass is the adjoint, not a forward
solve, but every contender pays it identically per eval so counting evals is
fair).  Multi-start's N restarts COUNT against ``B`` (``B // N`` iters per start,
so it gets fewer iters per start — equal TOTAL forward solves).  ``optax.lbfgs``
line-search evaluations are counted EXACTLY from ``info.num_linesearch_steps`` in
the optax state (1 initial fresh eval + sum of per-iteration line-search steps),
and the loop stops at the first iterate whose cumulative eval count reaches ``B``.

Adoption gate
-------------
Adopt a second optimizer ONLY if a CANDIDATE (``optax_adam`` or ``optax_lbfgs``)
beats the incumbent (``adam_multistart_bi``, the 4-C default) by >= 3 dB on the
benchmark's natural metric (all three are "minimize a mean power", so dB =
10*log10(loss)) at EQUAL budget on >= 2 of 3 benchmarks, with clean descent
curves.  Otherwise the verdict is "no-adopt".  The gate test
``tests/test_optimizer_bakeoff_gates.py`` re-derives this from the committed raw
curves.

Run
---
  JAX_PLATFORMS=cpu python scripts/diagnostics/optimizer_bakeoff/run_bakeoff.py
  # subset while iterating:
  JAX_PLATFORMS=cpu python .../run_bakeoff.py --only ar_coating

Writes ``tests/fixtures/optimizer_bakeoff/bakeoff_results.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

# float32 is the regime of study; no module-level x64 flip (scoped-x64 only).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")
os.environ.setdefault("SMOKE", "1")  # taper example reads this at import

import numpy as np
import jax
import jax.numpy as jnp
import optax

REPO = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO / "tests" / "fixtures" / "optimizer_bakeoff"
C0 = 2.998e8

# ---- fixed bake-off configuration (deterministic) -------------------------
ADOPT_GATE_DB = 3.0          # candidate must beat incumbent by >= this many dB
ADOPT_MIN_BENCHMARKS = 2     # ... on >= this many of the 3 benchmarks
N_STARTS = 3                 # multi-start restarts (matches the 4-C default)
SEED = 0                     # PRNG seed for the extra multi-start inits
INCUMBENT = "adam_multistart_bi"
CANDIDATES = ("optax_adam", "optax_lbfgs")
OPT_ORDER = ("adam_rfx", "optax_adam", "optax_lbfgs", "adam_multistart_bi")

# Per-benchmark forward-solve budget (divisible by N_STARTS) + Adam lr (the
# example's own lr).  Budgets sized so total CPU stays < ~2.5-3 h; the light AR
# case gets the deepest budget, the heavy waveguide case the shallowest.
BENCH_CFG = {
    "ar_coating":     dict(budget=60, lr=0.15, metric="mean_R_band"),
    "msl_stub_notch": dict(budget=9, lr=0.15, metric="abs_S21_sq"),
    "waveguide_taper": dict(budget=12, lr=0.20, metric="mean_S11_sq"),
}
METRIC_DB_FACTOR = 10.0      # 10*log10(power) for all three "mean power" metrics


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# Benchmark builders — reuse the committed example sim builders + objectives.
# Each returns (cost_fn, init_latent, n_dof, source_note).  ``cost_fn`` is a
# pure scalar-latent -> scalar-loss callable (jittable, differentiable).
# ===========================================================================
def build_ar_coating():
    """SMOOTH: multilayer AR coating, 3 layer permittivities, band-mean |R|^2.

    Reuses ``multilayer_ar_coating.py`` builders (``_build_simulation``,
    ``_render_design_eps``, ``_latent_to_eps``).  The cost closure lives inside
    that file's ``main()`` (not importable); it is reconstructed here 1:1 from
    lines 268-311 (vacuum reference forward, scattered-field FFT, X-band mean
    |R(f)|^2).  Init = zeros latent (flat mid-eps block) so every optimizer has
    real smooth descent work (the example warm-starts from the geometric ladder;
    we do not, to exercise the optimizers).
    """
    ar = _load(REPO / "examples/inverse_design/multilayer_ar_coating.py", "bo_ar")
    sim_ref = ar._build_simulation()
    res_ref = sim_ref.forward(eps_override=ar._vacuum_eps_array(),
                              n_steps=ar.N_STEPS, skip_preflight=True)
    ts_inc_refl = jnp.asarray(res_ref.time_series[:, 0])
    nfft = int(2 ** np.ceil(np.log2(ar.N_STEPS)))
    freqs_fft = jnp.fft.rfftfreq(nfft, d=ar.DT)
    band_mask = (freqs_fft >= ar.F_LO) & (freqs_fft <= ar.F_HI)
    band_norm = float(jnp.sum(band_mask))
    S_inc = jnp.fft.rfft(ts_inc_refl, n=nfft)
    sim_design = ar._build_simulation()

    def cost_fn(latent):
        layer_eps = ar._latent_to_eps(latent)
        eps = ar._render_design_eps(layer_eps)
        res = sim_design.forward(eps_override=eps, n_steps=ar.N_STEPS,
                                 skip_preflight=True)
        ts_scat = res.time_series[:, 0] - ts_inc_refl
        S_scat = jnp.fft.rfft(ts_scat, n=nfft)
        R = (jnp.abs(S_scat) / (jnp.abs(S_inc) + 1e-30)) ** 2
        return jnp.sum(R * band_mask) / band_norm

    init = jnp.zeros(3, dtype=jnp.float32)
    return cost_fn, init, 3, "multilayer_ar_coating.py:_build_simulation/_render_design_eps"


def _load_msl_coarse(dx_new: str):
    """Re-exec ``msl_stub_notch_tuning.py`` with a COARSER ``dx`` patched in.

    The full-res demo (dx = 127 um = h_sub/2, 2 substrate cells, 954 k cells)
    costs ~150 s PER forward on CPU — a 4-optimizer bake-off would blow the CPU
    budget.  This bake-off is DESIGN-LOOP CAPABILITY evidence, not MSL physics
    (framing fence), so the mesh is coarsened to dx = 254 um (exactly 1 substrate
    cell, ~238 k cells, ~14 s/forward).  Every builder (``build_sim``,
    ``build_stub_occ``, the derived geometry) is REUSED verbatim at the coarser
    resolution — only the ``dx`` literal is patched.  Intermediate meshes
    (dx = 190 um) were rejected: the MSL plane-extractor's known dx-fragility
    made |S21| flat/garbage there; dx = h_sub and dx = h_sub/2 are the clean
    points, and h_sub is the affordable one.
    """
    src = (REPO / "examples/inverse_design/msl_stub_notch_tuning.py").read_text()
    patched = src.replace("DX = 127e-6", f"DX = {dx_new}", 1)
    if patched == src:
        raise RuntimeError("MSL dx patch did not apply")
    import types
    mod = types.ModuleType("bo_msl_coarse")
    mod.__file__ = str(REPO / "examples/inverse_design/msl_stub_notch_tuning.py")
    sys.modules["bo_msl_coarse"] = mod
    exec(compile(patched, mod.__file__, "exec"), mod.__dict__)
    return mod


def build_msl_stub_notch():
    """SHARP RESONANT NULL: MSL open-stub notch, scalar stub length, |S21(6 GHz)|^2.

    Reuses ``msl_stub_notch_tuning.py`` builders (``build_sim``,
    ``build_stub_occ``) and the JAX N-probe extractor at a COARSENED mesh
    (dx = 254 um; see ``_load_msl_coarse``).  The cost closure lives inside that
    file's ``main()`` (not importable); reconstructed here 1:1 from lines
    537-580.

    Character note (R5, honest): the full-res (dx = h_sub/2) |S21|^2(L) surface
    is physically MULTIMODAL (in-band lambda/4 notch ~7 mm vs a below-band
    longer-stub valley ~9.5 mm, the #171 trap).  At the bake-off's coarse mesh
    that multimodality COLLAPSES to a single sharp well at L ~ 7 mm
    (scan: -16 dB @4.5mm -> -30 dB @7mm -> -14 dB @11.5mm, monotone flanks), so
    this benchmark exercises BEST-ITERATE overshoot-guarding on a sharp -30 dB
    resonant null (the other half of the 4-C default), not multi-start basin-
    hopping.  Init = L = 9.5 mm (the example's legacy seed), here on the well's
    descending flank -- Adam must descend into the sharp null and best-iterate
    must catch the overshoot.
    """
    msl = _load_msl_coarse("254e-6")
    from rfx.probes.msl_wave_decomp import (_v_from_plane, _i_from_plane,
                                            extract_msl_nprobe)
    f_target_arr = jnp.asarray([msl.F_TARGET], dtype=jnp.float32)
    sim, y_trace, trace_y_hi, d_set, p_set = msl.build_sim(f_target_arr)
    grid = sim._build_grid()
    num_periods = 6.0
    period = 1.0 / float(sim._freq_max)
    n_steps_raw = int(math.ceil(num_periods * period / float(grid.dt)))
    K = max(8, int(math.isqrt(n_steps_raw)))
    n_steps_use = ((n_steps_raw + K - 1) // K) * K
    beta0 = (2.0 * jnp.pi * f_target_arr
             * jnp.sqrt(jnp.asarray(msl.EPS_EFF, jnp.float32))
             / jnp.asarray(C0, jnp.float32))
    x_probes = jnp.array([0.0, d_set.delta, 2.0 * d_set.delta], jnp.float32)

    def cost_fn(latent):
        L_stub = msl.L_MIN + (msl.L_MAX - msl.L_MIN) * jax.nn.sigmoid(latent)
        occ = msl.build_stub_occ(grid, trace_y_hi, L_stub)
        fr = sim.forward(pec_occupancy_override=occ, n_steps=n_steps_use,
                         checkpoint_segments=K, skip_preflight=True)
        v_d = jnp.stack([_v_from_plane(fr, d_set.ez1_name, d_set),
                         _v_from_plane(fr, d_set.ez2_name, d_set),
                         _v_from_plane(fr, d_set.ez3_name, d_set)], axis=-1)
        i1_d = _i_from_plane(fr, d_set.hy_name, d_set)
        v_p = jnp.stack([_v_from_plane(fr, p_set.ez1_name, p_set),
                         _v_from_plane(fr, p_set.ez2_name, p_set),
                         _v_from_plane(fr, p_set.ez3_name, p_set)], axis=-1)
        i1_p = _i_from_plane(fr, p_set.hy_name, p_set)
        res_d = extract_msl_nprobe(v_d, x_probes, i1_d, beta0)
        res_p = extract_msl_nprobe(v_p, x_probes, i1_p, beta0)
        s21 = res_p["alpha"] / (res_d["alpha"] + 1e-30)
        return jnp.abs(s21[0]) ** 2

    # latent for L = 9.5 mm (the example's legacy single-start trap seed).
    u = (9.5e-3 - msl.L_MIN) / (msl.L_MAX - msl.L_MIN)
    init = jnp.asarray(math.log(u / (1.0 - u)), dtype=jnp.float32)
    return cost_fn, init, 1, "msl_stub_notch_tuning.py:build_sim/build_stub_occ"


def build_waveguide_taper():
    """BAND-AVERAGED: WR-90 dielectric taper (SMOKE), per-section eps, <|S11|^2>.

    Reuses ``waveguide_dielectric_taper.py`` directly: ``build_sim`` +
    ``design_layout`` + ``make_eps_builder`` + ``make_objective`` -> ``loss_fn``.
    Init = zeros theta (the example's flat mid-eps block).
    """
    tp = _load(REPO / "examples/tap_paper/waveguide_dielectric_taper.py", "bo_tp")
    sim = tp.build_sim()
    grid = sim._build_grid()
    layout = tp.design_layout(grid)
    _s11_fn, loss_fn = tp.make_objective(sim, tp.make_eps_builder(grid, layout), grid)
    n_sec = layout["n_sec"]
    init = jnp.zeros(n_sec, dtype=jnp.float32)
    return loss_fn, init, n_sec, "waveguide_dielectric_taper.py:make_objective"


BUILDERS = {
    "ar_coating": build_ar_coating,
    "msl_stub_notch": build_msl_stub_notch,
    "waveguide_taper": build_waveguide_taper,
}


# ===========================================================================
# The four optimizers.  Each returns a dict with:
#   curve: [[cumulative_forward_solves, loss], ...]  (raw; R5 inspectable)
#   n_solves, wall_s, starts, failures:[str], (lbfgs) linesearch_steps:[int]
# ===========================================================================
def _finite(x) -> bool:
    return bool(np.isfinite(np.asarray(x)).all())


def run_adam_rfx(cost_fn, init, budget, lr, best_iterate, n_starts, seed):
    """rfx's shipped hand-rolled Adam via ``_adam_multistart`` (single or multi).

    Faithful to ``rfx.optimize``: single start + best_iterate=False is the
    ``optimize()`` default; N starts + best_iterate=True is the PR #286 4-C
    default.  Extra starts are i.i.d. standard-normal latents from ``seed`` —
    exactly ``optimize()``'s scheme.  Per-start iters = budget // n_starts so the
    TOTAL forward solves equal ``budget`` (multi-start pays for its restarts).
    """
    from rfx.optimize import _adam_multistart
    # Run the objective EAGER (no jit): rfx's forward has a host-side
    # ``bool(jnp.any(pec_mask))`` branch in ``_assemble_materials`` that only
    # resolves under eager execution (grid-derived arrays stay concrete);
    # ``jax.jit(cost_fn)`` tracerises it and raises. The shipped ``optimize()``
    # runs eager too, so this is faithful.
    n_iters = budget // n_starts
    latent_inits = [init]
    if n_starts > 1:
        keys = jax.random.split(jax.random.PRNGKey(seed), n_starts - 1)
        latent_inits += [jax.random.normal(k, init.shape, dtype=jnp.float32)
                         for k in keys]
    t0 = time.time()
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always")
        best_lat, best_lh, all_histories, best_start = _adam_multistart(
            cost_fn, latent_inits, n_iters=n_iters, lr=lr,
            best_iterate=best_iterate, verbose=False)
    wall = time.time() - t0
    # Build a monotone budget curve: consume starts sequentially, plot the
    # running-best loss across all evals so far (best-iterate selection).
    curve = []
    running_best = math.inf
    n = 0
    per_start = []
    for lh in all_histories:
        per_start.append([float(x) for x in lh])
        for loss in lh:
            n += 1
            running_best = min(running_best, float(loss))
            curve.append([n, float(loss)])
    failures = [str(w.message) for w in wlog
                if "non-finite" in str(w.message).lower()]
    # early stop (a start produced < n_iters finite iterates) is a NaN failure
    if any(len(lh) < n_iters for lh in all_histories) and not failures:
        failures.append("a start stopped early (non-finite loss/gradient)")
    return dict(curve=curve, per_start=per_start, n_solves=n, wall_s=wall,
                starts=n_starts, best_start=int(best_start), failures=failures)


def run_optax_adam(cost_fn, init, budget, lr):
    """optax.adam, manual loop, 1 forward solve per iteration (eager)."""
    jvg = jax.value_and_grad(cost_fn)  # eager (forward is not jit-traceable)
    opt = optax.adam(lr)
    x = init
    state = opt.init(x)
    curve = []
    failures = []
    t0 = time.time()
    for it in range(budget):
        v, g = jvg(x)
        vf = float(v)
        curve.append([it + 1, vf])
        if not (_finite(v) and _finite(g)):
            failures.append(f"non-finite loss/gradient at iter {it}")
            break
        updates, state = opt.update(g, state)
        x = optax.apply_updates(x, updates)
    wall = time.time() - t0
    return dict(curve=curve, n_solves=len(curve), wall_s=wall, starts=1,
                failures=failures)


def run_optax_lbfgs(cost_fn, init, budget):
    """optax.lbfgs (zoom line search), manual loop; exact forward-solve count.

    INTEGRATION FINDING: optax's zoom line search LINEARISES (traces) the
    objective inside a ``lax.while_loop`` (optax/_src/linesearch.py:369,
    ``jax.linearize(value_fn, ...)``).  rfx's forward is NOT jit-traceable —
    ``_assemble_materials`` runs a host-side ``bool(jnp.any(pec_mask))`` — so
    optax.lbfgs cannot be driven through the rfx eager forward directly (the
    Adam family sidesteps this by only ever running the forward eagerly).  To
    give L-BFGS a fair NUMERICAL run anyway (and NOT change any rfx source), the
    objective is routed through a host callback: ``jax.pure_callback`` +
    ``custom_jvp`` evaluate the eager FDTD on-host and hand optax a
    traceable/differentiable scalar.  This host-shim requirement is itself an
    adoption cost, recorded in the results.

    Forward-solve count is the EXACT number of on-host FDTD evaluations (the
    callback increments a counter on every real execution, including the ones
    inside the line-search while_loop).  Stops at the first outer iterate whose
    cumulative solve count reaches ``budget``.
    """
    from optax import tree_utils as otu
    eager_vg = jax.value_and_grad(cost_fn)
    solves = [0]

    def host_vg(x_np):
        v, g = eager_vg(jnp.asarray(np.asarray(x_np, dtype=np.float32)))
        solves[0] += 1
        return np.asarray(v, np.float32), np.asarray(g, np.float32).reshape(init.shape)

    sds_v = jax.ShapeDtypeStruct((), jnp.float32)
    sds_g = jax.ShapeDtypeStruct(init.shape, jnp.float32)

    @jax.custom_jvp
    def obj(x):
        v, _g = jax.pure_callback(host_vg, (sds_v, sds_g), x)
        return v

    @obj.defjvp
    def obj_jvp(primals, tangents):
        (x,), (t,) = primals, tangents
        v, g = jax.pure_callback(host_vg, (sds_v, sds_g), x)
        return v, jnp.vdot(g, t)

    opt = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(obj)

    # JIT the update so the zoom-linesearch ``lax.while_loop`` (with the opaque
    # host callback inside) compiles ONCE and is reused every outer iteration.
    # The eager alternative recompiled the while_loop each ``opt.update`` and
    # exhausted the 64 GB cgroup / 65 k mmap limit (LLVM "cannot allocate
    # memory").  The between-step ``value_and_grad`` stays eager (a plain
    # callback, no while_loop) so it is cheap.
    @jax.jit
    def do_update(g, state, x, v):
        return opt.update(g, state, x, value=v, grad=g, value_fn=obj)

    x = init
    state = opt.init(x)
    curve = []
    ls_steps = []
    failures = []
    t0 = time.time()
    # iter 0: fresh value-and-grad (state has no cached value) -> 1 host solve.
    v0, g0 = value_and_grad(x, state=state)
    curve.append([solves[0], float(v0)])
    it = 0
    while solves[0] < budget:
        prev = solves[0]
        updates, state = do_update(g0, state, x, v0)
        x = optax.apply_updates(x, updates)
        # value/grad at the new point are cached by the line search (0 solves).
        v0, g0 = value_and_grad(x, state=state)
        n_ls = solves[0] - prev
        ls_steps.append(int(n_ls))
        v_here = float(v0)
        curve.append([solves[0], v_here])
        dec = float(np.asarray(otu.tree_get(state, "decrease_error")))
        cur = float(np.asarray(otu.tree_get(state, "curvature_error")))
        if not math.isfinite(v_here):
            failures.append(f"non-finite value at outer iter {it}")
            break
        if n_ls == 0 and it > 0:
            failures.append(f"line search made no progress at outer iter {it} "
                            f"(decrease_err={dec:.2e}, curvature_err={cur:.2e})")
            break
        it += 1
        if it > 4 * budget:  # safety valve
            break
    wall = time.time() - t0
    return dict(curve=curve, n_solves=int(solves[0]), wall_s=wall, starts=1,
                linesearch_steps=ls_steps, host_shim=True, failures=failures)


# ===========================================================================
# Curve reductions + gate arithmetic (also re-derived independently by the
# committed gate test — keep these in lock-step).
# ===========================================================================
def within_budget(curve, budget):
    return [(int(n), float(v)) for n, v in curve if int(n) <= budget]


def final_loss(curve, budget):
    pts = within_budget(curve, budget)
    return pts[-1][1] if pts else math.inf


def best_loss(curve, budget):
    pts = within_budget(curve, budget)
    return min((v for _, v in pts), default=math.inf)


def to_db(loss):
    return METRIC_DB_FACTOR * math.log10(max(float(loss), 1e-30))


def run_benchmark(name):
    cfg = BENCH_CFG[name]
    budget, lr = cfg["budget"], cfg["lr"]
    print(f"\n{'='*72}\n[{name}] budget={budget} lr={lr} metric={cfg['metric']}\n{'='*72}")
    cost_fn, init, n_dof, src = BUILDERS[name]()

    runs = {}
    runs["adam_rfx"] = run_adam_rfx(cost_fn, init, budget, lr,
                                    best_iterate=False, n_starts=1, seed=SEED)
    runs["optax_adam"] = run_optax_adam(cost_fn, init, budget, lr)
    runs["optax_lbfgs"] = run_optax_lbfgs(cost_fn, init, budget)
    runs["adam_multistart_bi"] = run_adam_rfx(cost_fn, init, budget, lr,
                                              best_iterate=True,
                                              n_starts=N_STARTS, seed=SEED)

    for opt_name in OPT_ORDER:
        r = runs[opt_name]
        r["final_loss"] = final_loss(r["curve"], budget)
        r["best_loss"] = best_loss(r["curve"], budget)
        r["final_db"] = to_db(r["final_loss"])
        r["best_db"] = to_db(r["best_loss"])
        fail = "; ".join(r["failures"]) if r["failures"] else "none"
        print(f"  {opt_name:20s} solves={r['n_solves']:3d} "
              f"best={r['best_loss']:.4e} ({r['best_db']:+.2f} dB) "
              f"final={r['final_loss']:.4e} ({r['final_db']:+.2f} dB) "
              f"wall={r['wall_s']:.1f}s  fail={fail}")

    incumbent_best_db = runs[INCUMBENT]["best_db"]
    winner = min(OPT_ORDER, key=lambda o: runs[o]["best_loss"])
    margins = {c: incumbent_best_db - runs[c]["best_db"] for c in CANDIDATES}
    beats = {c: bool(margins[c] >= ADOPT_GATE_DB) for c in CANDIDATES}
    print(f"  -> winner(min best-loss)={winner}  "
          f"incumbent({INCUMBENT}) best={incumbent_best_db:+.2f} dB")
    for c in CANDIDATES:
        print(f"     candidate {c}: margin_vs_incumbent={margins[c]:+.2f} dB "
              f"(>= {ADOPT_GATE_DB} → beats={beats[c]})")

    return dict(metric=cfg["metric"], metric_db_factor=METRIC_DB_FACTOR,
                budget=budget, lr=lr, n_dof=n_dof, n_starts=N_STARTS,
                source=src, init=[float(x) for x in np.atleast_1d(np.asarray(init))],
                runs=runs, winner=winner,
                incumbent_best_db=incumbent_best_db,
                candidate_margins_db=margins, candidate_beats=beats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of benchmark names to run")
    args = ap.parse_args()

    names = args.only or list(BUILDERS.keys())
    t0 = time.time()
    benchmarks = {}
    for n in names:
        benchmarks[n] = run_benchmark(n)
        jax.clear_caches()  # release compiled executables between benchmarks (64 GB cgroup)

    # ---- decision arithmetic (re-derived in the gate test) ----------------
    full = len(names) == len(BUILDERS)
    per_bench_beats = {c: [benchmarks[n]["candidate_beats"][c] for n in names]
                       for c in CANDIDATES}
    n_beats = {c: int(sum(per_bench_beats[c])) for c in CANDIDATES}
    adopts = {c: bool(n_beats[c] >= ADOPT_MIN_BENCHMARKS) for c in CANDIDATES}
    any_adopt = any(adopts.values())
    verdict = "adopt" if (full and any_adopt) else "no-adopt"
    if full:
        adopters = [c for c in CANDIDATES if adopts[c]]
        if verdict == "no-adopt":
            verdict_text = (
                "no-adopt: Adam + multi-start + best-iterate (the PR #286 / 4-C "
                "default) is the right shipped optimizer. Neither optax.adam nor "
                "optax.lbfgs beats best-iterate multi-start Adam by >= 3 dB on >= 2 "
                "of 3 benchmarks at equal forward-solve budget. This is design-loop "
                "capability evidence, not physics.")
        else:
            verdict_text = (
                f"adopt (record-only follow-up, do NOT wire in this PR): {adopters} "
                f"cleared the >= 3 dB on >= 2/3 bar vs the incumbent.")
    else:
        verdict_text = f"partial run ({names}) — verdict not final."

    decision = dict(
        adopt_gate_db=ADOPT_GATE_DB, adopt_min_benchmarks=ADOPT_MIN_BENCHMARKS,
        incumbent=INCUMBENT, candidates=list(CANDIDATES),
        per_benchmark_candidate_beats=per_bench_beats,
        candidate_n_beats=n_beats, candidate_adopts=adopts,
        any_candidate_adopts=any_adopt, verdict=verdict,
        verdict_text=verdict_text)

    meta = dict(
        work_package="4-D", date="2026-07-08", platform="cpu",
        precision="float32", jax_version=jax.__version__,
        optax_version=optax.__version__, numpy_version=np.__version__,
        rfx_version=__import__("rfx").__version__,
        n_starts=N_STARTS, seed=SEED,
        budgets={n: BENCH_CFG[n]["budget"] for n in BENCH_CFG},
        optimizers=list(OPT_ORDER), benchmark_names=list(BUILDERS.keys()),
        metric_db_factor=METRIC_DB_FACTOR, total_wall_s=time.time() - t0)

    out = dict(meta=meta, benchmarks=benchmarks, decision=decision)
    if full:
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        path = FIXTURE_DIR / "bakeoff_results.json"
        path.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"\nWrote {path}")
    else:
        print(f"\n[partial: {names}] not written (run full for the fixture)")
    print(f"\nVERDICT: {verdict}\n{decision['verdict_text']}")
    print(f"total wall {meta['total_wall_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
