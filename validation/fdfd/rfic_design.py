"""Study R, part C: gradient design of the paper-scale square spiral on the
GPU at the resolved level -- L_diff = 4.000 nH within 1 % with Q_diff
maximised at 2.45 GHz over (r_out, spacing, width) -- against a 3 x 3 x 3
sweep of the same box.

The model is ``rfic_spiral.py``'s (parts A and B): the level-invariant
fixture, graded 190 um silicon (eps 11.9, 2 S/m), Leontovich 3.05e7 S/m metal,
level 2 = 4 cells across W, N = 772892, cuDSS on an RTX 4090. The paper's
4.000 nH / 16.80 belong to a symmetric OCTAGON; this is a square single-ended
spiral (its L is higher at equal size), so 4.000 nH is a design TARGET here
and the paper's Q is context, never a gate.

THE PROBLEM. Box (um): r_out [200, 260], spacing [10, 18], width [22, 32]
around the paper values (218, 14, 30); every corner and every sweep point is
accepted by ``rfx.fdfd.spiral.check_feasible`` (worst margin r_out - 4 W - 3 S
= 18 um). Loss on box-normalised u in [0, 1]^3: -Q_diff / Q_ref + 100
(L_diff / 4 nH - 1)^2, Q_ref = 23.6308 (the paper values on this level).
L-BFGS-B (scipy, jac=True) on ``jax.value_and_grad`` through cuDSS, from the
paper values, inside ONE GPU job (lane r2); every evaluation and every
accepted iterate is in ``design``.

RESULTS (``rfic_design.json``; GPU runs in ``runs``)
----------------------------------------------------
The loop: 22 objective evaluations (147 s median, each one forward and one
reverse pass, i.e. 6 cuDSS factorisations: 132 cuDSS factorisations), 14
accepted iterates, 54.2 min of wall time; it stopped on scipy's relative
loss-reduction test. From (218, 14, 30) um, L = 3506.98 pH (-12.3 %), Q =
23.631, its first step went to the far corner (260, 10, 22) um (L = 6867 pH),
the line search came back to the target in two evaluations, and the rest of
the loop walked along the L = 4 nH valley toward small r_out and narrow W.
Returned theta* = (200.177, 11.832, 22.000) um: width on its lower bound,
r_out 0.18 um above its own.
R4 L_diff(theta*) = 4000.04 pH, +0.0011 % from 4.000 nH, on the model it was
   designed on. Q_diff = 23.914. PASSES (1 %).
R3 AD vs FD4 at theta* (1 % steps, through cuDSS): dL/dr_out 4.30e-09,
   dQ/dr_out 3.88e-07, dL/dwidth 4.19e-09, dQ/dwidth 2.59e-07; worst
   3.88e-07. PASSES (<= 1e-4).
R6 the sweep (27 forward solves, 72 s each, 81 cuDSS factorisations, 32.5 min)
   has two points inside the 1 % band: (230, 10, 32) um, L = 3999.30 pH,
   Q = 22.861, and (230, 18, 27) um, L = 4020.00 pH, Q = 23.240. The design
   beats the better one by 0.674 (2.9 %). The best-Q sweep point overall,
   (200, 10, 22) um, is at L = 4152.24 pH (+3.8 %): the sweep cannot hit the
   target and its best in-band point is 2.9 % down on Q. PASSES.
R2 every S-matrix of part C (68 solves: 22 design evaluations, 27 sweep
   points, the 9 solves of the FD4 check, the 10 finer-grid solves; 272
   S-matrices): |S12 - S21| <= 2.69e-12, largest singular value <= 0.99782.
   PASSES (<= 1e-8, <= 1 + 1e-8).
Cost against the sweep: the design used 132 cuDSS factorisations and
   54.2 min; the sweep 81 and 32.5 min, and the sweep's best in-band point
   is 2.9 % below the design on Q.

HOW OPTIMAL. The stop was scipy's relative-reduction test (ftol), not the
projected gradient (gtol 1e-5), and theta* is NOT a KKT point (``kkt``): in
box units the loss gradient there is (+0.048, -0.0006, -0.018) and the
projected gradient P(u - g) - u has an infinity norm of 0.0182 -- the width
sits on its lower bound with a NEGATIVE gradient (the loss would still fall
if the width grew), r_out 0.003 box units above its bound with a positive
one. A first-order move from the AD gradient at theta* (checked by R3) at
fixed L -- spacing to its 10 um bound, r_out to its own, the width up to
keep L -- lands at (200, 10, 23.07) um and predicts +0.19 % in Q. That is an
extrapolation from one gradient, not a measured bound: no solve was made
there, so how far theta* is from the box's constrained optimum on Q is
estimated (about 0.2 %, first order), not established.

R7 ONE LEVEL FINER (``gates.R7``; a reported shift). The full level 3 (6
   cells across W, z refined 3x; N = 2577198) is a 106.71 GB cuDSS factor
   (plan): it fits neither the 48 GB gpu-a6000-1 in-core nor in hybrid mode
   (that preset's host-memory limit is 32 GiB); only the cluster's H200
   could hold it, and this study may not use it. So the finer re-solve of
   theta* is done in its two directions, each on the A6000 in-core:
   IN-PLANE (x, y at level 3 = 6 cells across W, z at level 2; N = 1253063,
   34.71 GB), against level 2 on the same box -- walls at 3 W to keep the
   factor on the card (at 10 W this grid is N = 1730643 and was not run):
   L_diff +0.663 %, Q_diff +2.810 %;
   VERTICAL (z at level 3, x, y at level 2, the study's 10 W walls;
   N = 1150967, 38.77 GB), against the design's own evaluation: L_diff
   +1.061 %, Q_diff +2.935 %.
   Summed, the one-level-finer shift at theta* is L_diff +1.723 %, Q_diff
   +5.745 %: one level finer the design would sit at ~4068.98 pH, 1.72 %
   ABOVE the target -- outside the 1 % band it was designed into. Two checks
   bound the sum, both at the paper values and the 1 -> 2 step: the level
   shift depends on the box by 0.86 points on L (6.46 % at 10 W, 7.32 % at
   3 W) and -0.98 on Q, and the two directions ADD on L to 0.018 points
   (x, y only +2.41 %, z only +4.04 %, together 6.46 %) but not on Q
   (-1.84 points: the sum over-states Q's step). The level 1 -> 2 step at
   theta* itself was L_diff +6.21 %, Q_diff +12.43 %, so the shifts are
   shrinking (6.2 % -> ~1.7 % on L) but the grid is not converged, as the
   ladder track found for this fixture. The design's L is a statement about
   the level-2 model; a design meant to hold 4.000 nH one level finer would
   aim ~1.7 % low.
   WHAT LIMITS THE ESTIMATE (``gates.R7.caveats``). (1) The in-plane half
   was measured on 3 W walls, where the level-2 L at theta* is 8.28 % below
   the design's 10 W value (3668.69 against 4000.04 pH); the box check above
   is the only handle on what that does to a shift. (2) Level 3 is not a
   refinement of level 2: level m cuts every level-1 interval into m, so
   levels 2 and 3 both refine level 1 but only the level-1 lines are shared
   -- "one level finer" is h/3 against h/2, not a nested halving. (3) Taking
   the level 1 -> 2 step and the estimated 2 -> 3 step at theta* as a
   sequence L* - C h^p on h = 1, 1/2, 1/3 gives p = 1.21 and a limit of
   4177.66 pH: the level-2 L would be 4.25 % below its grid limit and the
   design +4.44 % above the target there -- an estimate on an estimate (the
   level-3 point is the R7 sum). On Q the same construction gives p = 0.21,
   not an asymptotic sequence (the sum over-states Q's step, see the
   additivity check), and no Q limit is extrapolated.

WALLS AND RE-MESHING (``walls``, ``remesh``; level-1 CPU checks recorded
in ``rfic_spiral.json``). Every design evaluation, sweep point and finer
grid is the NOMINAL mesh deformed to theta by the traced metric, which pins
the 10 W walls in space: the wall-to-strip distance runs from 317.8 um at
theta* to 258.0 um at r_out = 260 um, and the walls' low bias moves with it
-- moving them to 20 W raises L_diff at level 1 by +0.66 % at theta*,
+0.82 % at the paper values and +1.72 % at (260, 14, 27) um. Carried to the
design as an estimate, theta* on a 20 W box would sit at 4026.33 pH,
+0.66 % from the target: still inside the 1 % band. The bias's two-point
slope in r_out (0.0178 % per um) is 1.8 % of L's own relative slope at
theta* (1.01 % per um), so it barely tilts the trade the optimiser made;
Q_diff moves by at most 0.38 % across the three points, far less than the
design's 2.9 % lead over the in-band sweep points (R6). Which sweep points
are in band is less robust: at r_out = 230 um their L would rise by
somewhere between the measured +0.82 % and +1.72 % (not measured there).
Re-meshing: a fresh level-1 mesh built at theta* gives L_diff +0.52 % and
Q_diff +2.18 % against the deformed one (``remesh``: the fresh mesh has 3
cells across the 22 um strip, the deformed 2), inside the level 1 -> 2
step. Neither is applied to R4.

ACCURACY. The absolute L and Q carry the fixture's bias measured by the
ladder track (``accuracy``): L on this fixture family is low (the paper
geometry 6.27 % below its Greenhouse referee at this resolution, in a
quasi-static protocol -- vacuum, 10 MHz, 120 um silicon -- so the transfer
to this model is approximate; against this model's own 190 um referee the
paper-value level-2 L is 7.27 % low, dielectrics and the sheet's internal
inductance included), the 10 W walls add the low bias above, and the
Leontovich sheet under-states the strip's resistance by ~1/5 at delta / t =
0.614 (``rfic_spiral.json`` leontovich). 4.000 nH is met on the model it was
designed on, which is what R4 gates.

RUNNING IT. ``validation/vessl/lane_r_rfic.py`` does the GPU work (lanes
r2 = the loop, r3 = FD4 at theta* and the sweep, r4 = the finer grids;
ledger ``validation/vessl/runs/r_run_ledger.json``); this file holds the
problem (box, loss, loop) and assembles the JSON and figure::

    .venv/bin/python validation/fdfd/rfic_design.py [--no-figure]

Figure: (a) Q against L for the sweep, every evaluation and the design,
(b) L error and Q per evaluation, (c) the path in (r_out, width) over the
sweep, (d) the one-level-finer shifts.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import pathlib
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
JSON_PATH = HERE / "rfic_design.json"
PNG_PATH = HERE / "rfic_design.png"
RS_PATH = HERE / "rfic_spiral.py"


def load_rs() -> Any:
    """``rfic_spiral.py`` (parts A and B): the fixture, the solves, the ledger."""
    name = "rfic_spiral_for_design"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RS_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# the design problem

L_TARGET = 4.000e-9          # the paper's L_diff, as a TARGET (never a physics gate)
L_BAND = 0.01                # gate R4: |L - 4.000 nH| <= 1 %
# The design box around the paper values (218, 14, 30) um: every corner is
# accepted by rfx.fdfd.spiral.check_feasible on the resolved model (asserted by
# the test on the level-1 model, whose breakpoints are the same); the binding
# margin a_in - width / 2 = r_out - 4 width - 3 spacing is 18 um at the worst
# corner (200, 18, 32) um.
BOX = ((200e-6, 260e-6), (10e-6, 18e-6), (22e-6, 32e-6))
SWEEP_AXES = tuple((lo, 0.5 * (lo + hi), hi) for lo, hi in BOX)   # 3 x 3 x 3 = 27
# Penalty weight of the soft L constraint. At a penalty optimum the relative L
# error is e* = (grad Qhat . grad e) / (2 LAM |grad e|^2) over the FREE
# parameters (Qhat = Q / Q_ref, e = L / L_target - 1). Measured at the paper
# values on the resolved level (R1 lane, rfic_spiral.json "gradients"), in
# box units: dQhat/du = (-0.0450, +0.0135, -0.0132), de/du = (+0.540, -0.136,
# -0.288). Width raises BOTH L and Q as it narrows, so it is expected on its
# lower bound, leaving r_out (and possibly spacing) free: e* = -0.0450 /
# (2 LAM 0.540) = -0.042 % at LAM = 100 (r_out alone), against the 1 % band.
LAM = 100.0
# Q_ref normalises the Q term of the loss (a constant, so the loss is one fixed
# function for the whole run): Q_diff at the paper values on the resolved level
# (R1 lane run 369367261263, r1_study.json levels["2"]["Q_diff"])
Q_REF = 23.630784354814097
MAXFUN = 28                  # objective evaluations (each = 3 forward + 3 adjoint factorisations);
                             # scipy checks it per iteration, the job deadline is the hard stop
MAXITER = 25
FTOL = 1e-7                  # relative loss decrease (Q moves < 1e-7 Q_ref)
GTOL = 1e-5                  # projected gradient in box-normalised units


def to_u(theta: Sequence[float]) -> np.ndarray:
    return np.array([(t - lo) / (hi - lo) for t, (lo, hi) in zip(theta, BOX)])


def to_theta(u: Sequence[float]) -> np.ndarray:
    return np.array([lo + float(v) * (hi - lo) for v, (lo, hi) in zip(u, BOX)])


def sweep_points() -> list[tuple[float, float, float]]:
    return [tuple(float(v) for v in p) for p in itertools.product(*SWEEP_AXES)]


def loss_of(l_value, q_value, q_ref: float, lam: float = LAM, l_target: float | None = None):
    """``-Q / Q_ref + lam (L / L_target - 1)^2`` (works on jnp and floats;
    ``l_target`` defaults to the module's :data:`L_TARGET` at call time)."""
    lt = L_TARGET if l_target is None else l_target
    return -q_value / q_ref + lam * (l_value / lt - 1.0) ** 2


def objective(model, q_ref: float, lam: float = LAM, freq: float | None = None):
    """``u -> (loss, (L, Q, S-matrices))`` on the box-normalised parameters,
    for ``jax.value_and_grad(..., has_aux=True)``."""
    import jax.numpy as jnp

    from rfx.fdfd import spiral as sm
    rs = load_rs()
    f = rs.FREQ if freq is None else freq
    lo = jnp.asarray([b[0] for b in BOX])
    span = jnp.asarray([b[1] - b[0] for b in BOX])

    def fn(u):
        theta = lo + u * span
        r = sm.solve_spiral(model, f, theta=theta, sigma_metal=rs.SIGMA_METAL,
                            sigma_si=rs.SIGMA_SI)
        lv, qv = jnp.real(r.L_diff), jnp.real(r.Q_diff)
        return loss_of(lv, qv, q_ref, lam), (lv, qv, r.s_raw, r.s_open, r.s_short, r.s_dut)

    return fn


class DeadlineStop(Exception):
    """The job's wall-clock budget ran out inside the loop."""


def run_design(model, q_ref: float, record: Callable[[dict[str, Any]], None],
               lam: float = LAM, deadline: float | None = None,
               maxfun: int = MAXFUN, maxiter: int = MAXITER,
               log: Callable[[str], None] = print) -> dict[str, Any]:
    """L-BFGS-B (scipy) on ``jax.value_and_grad`` of :func:`objective` from the
    paper values. Every evaluation and every accepted iterate is logged into
    the returned dict and ``record(rec)`` is called after each one (the lane
    rewrites its JSON there)."""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    from rfx.fdfd import spiral as sm
    rs = load_rs()
    vg = jax.value_and_grad(objective(model, q_ref, lam), has_aux=True)
    rec: dict[str, Any] = {"method": "L-BFGS-B (scipy.optimize.minimize, jac=True)",
                           "box": [list(b) for b in BOX], "u0": to_u(rs.THETA0).tolist(),
                           "theta0": list(rs.THETA0), "L_target": L_TARGET, "q_ref": q_ref,
                           "lam": lam, "options": {"maxfun": maxfun, "maxiter": maxiter,
                                                   "ftol": FTOL, "gtol": GTOL},
                           "evals": [], "iterates": []}
    t_start = time.time()

    def fun(u: np.ndarray) -> tuple[float, np.ndarray]:
        if deadline is not None and time.time() > deadline:
            raise DeadlineStop()
        theta = to_theta(u)
        sm.check_feasible(model, theta)
        t0 = time.time()
        (val, (lv, qv, *mats)), g = vg(jnp.asarray(u, dtype=jnp.float64))
        jax.block_until_ready(g)
        ev = {"n": len(rec["evals"]), "u": [float(v) for v in u],
              "theta": [float(v) for v in theta], "loss": float(val),
              "grad_u": [float(v) for v in g], "L_diff": float(lv), "Q_diff": float(qv),
              "L_err": float(lv) / L_TARGET - 1.0, "seconds": time.time() - t0,
              "t_since_start": time.time() - t_start}
        ev.update(rs.s_block(*mats))
        rec["evals"].append(ev)
        log(f"  eval {ev['n']:2d}: theta = ({theta[0] * 1e6:.3f}, {theta[1] * 1e6:.3f}, "
            f"{theta[2] * 1e6:.3f}) um  L = {ev['L_diff'] * 1e12:.2f} pH ({100 * ev['L_err']:+.3f} %)"
            f"  Q = {ev['Q_diff']:.4f}  loss = {ev['loss']:.6f}  ({ev['seconds']:.0f} s)")
        record(rec)
        return float(val), np.asarray(g, dtype=np.float64)

    def callback(xk: np.ndarray) -> None:
        # the iterate L-BFGS-B accepted: the evaluation at exactly xk
        hit = [e["n"] for e in rec["evals"] if np.array_equal(np.asarray(e["u"]), xk)]
        rec["iterates"].append({"k": len(rec["iterates"]) + 1, "u": xk.tolist(),
                                "theta": to_theta(xk).tolist(),
                                "eval": hit[-1] if hit else None})
        record(rec)

    u0 = to_u(rs.THETA0)
    try:
        res = minimize(fun, u0, jac=True, method="L-BFGS-B", bounds=[(0.0, 1.0)] * 3,
                       callback=callback,
                       options={"maxfun": maxfun, "maxiter": maxiter, "ftol": FTOL,
                                "gtol": GTOL})
        rec["result"] = {"success": bool(res.success), "status": int(res.status),
                         "message": str(res.message), "nit": int(res.nit),
                         "nfev": int(res.nfev), "u": np.asarray(res.x).tolist(),
                         "theta": to_theta(res.x).tolist(), "loss": float(res.fun)}
    except DeadlineStop:
        rec["result"] = {"success": False, "status": -1, "message": "stopped: job deadline"}
    rec["seconds"] = time.time() - t_start
    best = min(rec["evals"], key=lambda e: e["loss"]) if rec["evals"] else None
    rec["best_eval"] = best["n"] if best else None
    record(rec)
    return rec


# ---------------------------------------------------------------------------
# assembly (from the harvested GPU lane JSONs) and gates

R_TOL = 1e-4                 # AD vs FD4 (rule 1)


def final_eval(design: dict[str, Any]) -> dict[str, Any] | None:
    """The evaluation at L-BFGS-B's returned point (``result.u``)."""
    res = design.get("result") or {}
    u = res.get("u")
    if u is None:
        return None
    hits = [e for e in design.get("evals", []) if np.array_equal(np.asarray(e["u"]), np.asarray(u))]
    return hits[-1] if hits else None


def kkt_block(study: dict[str, Any], fe: dict[str, Any] | None) -> dict[str, Any]:
    """How far the returned point is from a constrained optimum, to first
    order, from the AD gradients AT it (``fdopt.grad``, checked against FD4 by
    R3). The move considered keeps L_diff fixed (linearised): r_out and
    spacing to their lower bounds and the width re-solved for the same L. A
    positive predicted dQ means Q was left on the table when L-BFGS-B stopped."""
    gr = (study.get("fdopt") or {}).get("grad") or {}
    if not fe or not gr:
        return {}
    th = fe["theta"]
    dl, dq = gr["dL"], gr["dQ"]
    dr, ds = BOX[0][0] - th[0], BOX[1][0] - th[1]
    dw = -(dl[0] * dr + dl[1] * ds) / dl[2]
    dqv = dq[0] * dr + dq[1] * ds + dq[2] * dw
    gu = np.asarray(study["design"]["evals"][fe["n"]]["grad_u"], dtype=float)
    u = np.asarray(fe["u"], dtype=float)
    # L-BFGS-B's projected gradient, P(u - g) - u on [0, 1]^3: zero at a KKT point
    pg = np.clip(u - gu, 0.0, 1.0) - u
    return {"theta": th, "u_at_return": u.tolist(), "grad_u_at_return": gu.tolist(),
            "projected_grad_u": pg.tolist(),
            "projected_grad_inf_norm": float(np.max(np.abs(pg))), "gtol": GTOL,
            "kkt_point": bool(np.max(np.abs(pg)) <= GTOL),
            "kkt_reading": "NOT a KKT point: at a lower bound the loss gradient must be >= 0 "
                           "and an interior component must vanish; here the width sits on "
                           "its lower bound with a NEGATIVE gradient (the loss still falls if "
                           "the width grows) and r_out sits 0.003 box units above its bound "
                           "with a positive one. The stop was scipy's relative-reduction "
                           "test (ftol), not the projected gradient (gtol)",
            "dL_dtheta": dl, "dQ_dtheta": dq,
            "move": [dr, ds, dw],
            "predicted_theta": [th[0] + dr, th[1] + ds, th[2] + dw],
            "predicted_dQ": dqv, "predicted_dQ_rel": dqv / fe["Q_diff"],
            "what": "first-order (AD gradient at the returned point) move at fixed L_diff: "
                    "r_out and spacing to their lower bounds, width re-solved for the same "
                    "L; predicted_dQ > 0 is the Q the stopping rule left. An EXTRAPOLATION "
                    "from one gradient, not a measured bound on the distance to the "
                    "box's optimum (no solve was made at predicted_theta)"}


def cost_block(design: dict[str, Any], csweep: dict[str, Any]) -> dict[str, Any]:
    ev = design.get("evals", [])
    pts = csweep.get("points", [])
    n_ev = len(ev)
    return {
        "design": {"objective_evaluations": n_ev, "accepted_iterates": len(design.get("iterates", [])),
                   "forward_three_fixture_solves": n_ev, "reverse_passes": n_ev,
                   "cudss_factorisations": 6 * n_ev,
                   "wall_seconds_loop": design.get("seconds"),
                   "seconds_per_evaluation_median": float(np.median([e["seconds"] for e in ev]))
                   if ev else None},
        "sweep": {"forward_three_fixture_solves": len(pts), "cudss_factorisations": 3 * len(pts),
                  "wall_seconds": float(sum(p["seconds"] for p in pts)) if pts else None,
                  "seconds_per_solve_median": float(np.median([p["seconds"] for p in pts]))
                  if pts else None},
        "note": "a design evaluation is one jax.value_and_grad: the three fixtures forward and "
                "their three adjoints; cuDSS has no transposed solve, so each adjoint is its "
                "own factorisation (6 per evaluation); a sweep point is one forward (3)"}


def evaluate_gates(study: dict[str, Any]) -> dict[str, Any]:
    rs = load_rs()
    g: dict[str, Any] = {}
    solves: dict[str, dict[str, float]] = {}
    for part in ("design", "csweep", "fdopt", "fine"):
        rs.collect_s(part, study.get(part, {}), solves)
    g["R2"] = rs.r2_gate(solves)

    fo = study.get("fdopt", {})
    gr = fo.get("grad", {})
    checked: dict[str, Any] = {}
    for name, rec in (fo.get("fd") or {}).items():
        if "fd_L" not in rec or not gr:
            continue
        k = rec["k"]
        for q, key in (("L", "dL"), ("Q", "dQ")):
            ad, fdv = gr[key][k], rec[f"fd_{q}"]
            checked[f"d{q}_d{name}"] = {"ad": ad, "fd4": fdv, "rel": abs(ad - fdv) / abs(fdv)}
    worst = max((v["rel"] for v in checked.values()), default=None)
    g["R3"] = {"at": "the design optimum (result theta), resolved level", "tolerance": R_TOL,
               "fd_order": 4, "fd_step_rel": rs.FD_STEP_REL, "checked": checked,
               "n_checked": len(checked), "worst_rel": worst,
               "passed": worst is not None and len(checked) >= 2 and worst <= R_TOL}

    d = study.get("design", {})
    fe = final_eval(d)
    r4: dict[str, Any] = {"L_target": L_TARGET, "band": L_BAND,
                          "model": f"the resolved level ({rs.RESOLVED}) it was designed on"}
    if fe:
        r4.update(theta=fe["theta"], L_diff=fe["L_diff"], Q_diff=fe["Q_diff"],
                  L_err=fe["L_err"], eval_index=fe["n"],
                  passed=abs(fe["L_err"]) <= L_BAND)
    else:
        r4["passed"] = None
    g["R4"] = r4

    pts = study.get("csweep", {}).get("points", [])
    band = [p for p in pts if abs(p["L_diff"] / L_TARGET - 1.0) <= L_BAND]
    r6: dict[str, Any] = {"n_sweep_points": len(pts), "band": L_BAND,
                          "sweep_points_in_band": [{"theta": p["theta"], "L_diff": p["L_diff"],
                                                    "Q_diff": p["Q_diff"]} for p in band],
                          "sweep_L_range": [min(p["L_diff"] for p in pts),
                                            max(p["L_diff"] for p in pts)] if pts else None}
    if fe and pts:
        best_any = max(pts, key=lambda p: p["Q_diff"])
        r6["design_Q"] = fe["Q_diff"]
        r6["sweep_best_Q_any_L"] = {"theta": best_any["theta"], "L_diff": best_any["L_diff"],
                                    "Q_diff": best_any["Q_diff"]}
        # the nearest sweep points to the target, whatever their L
        near = sorted(pts, key=lambda p: abs(p["L_diff"] / L_TARGET - 1.0))[:3]
        r6["sweep_nearest_to_target"] = [{"theta": p["theta"], "L_err": p["L_diff"] / L_TARGET - 1.0,
                                          "Q_diff": p["Q_diff"]} for p in near]
        if band:
            r6["best_in_band_Q"] = max(p["Q_diff"] for p in band)
            r6["design_minus_best_in_band"] = fe["Q_diff"] - r6["best_in_band_Q"]
            r6["passed"] = fe["Q_diff"] > r6["best_in_band_Q"] and abs(fe["L_err"]) <= L_BAND
            r6["reading"] = "the design beats every sweep point inside the 1 % L band on Q"
        else:
            r6["passed"] = abs(fe["L_err"]) <= L_BAND
            r6["reading"] = ("NO sweep point qualifies: none of the 27 lies within 1 % of "
                             "4.000 nH, so the sweep cannot deliver the target at all; the "
                             "design does (R4)")
    else:
        r6["passed"] = None
    g["R6"] = r6

    g["R7"] = r7_block(study, fe)
    return g


def _shift(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, float] | None:
    """Relative change of L_diff and Q_diff from record ``b`` to record ``a``."""
    if not a or not b or "L_diff" not in a or "L_diff" not in b:
        return None
    return {"dL_rel": a["L_diff"] / b["L_diff"] - 1.0, "dQ_rel": a["Q_diff"] / b["Q_diff"] - 1.0}


def r7_block(study: dict[str, Any], fe: dict[str, Any] | None) -> dict[str, Any]:
    """The one-level-finer shift at the optimum, from what fits the allowed cards.

    The full level 3 (6 cells across W, z refined 3x) is a 106.71 GB cuDSS
    factor (R1 plan): it fits neither gpu-a6000-1 in-core (48 GB) nor in
    hybrid mode (its 32 GiB host-memory limit). So the shift is measured in
    its two directions separately: IN-PLANE (x, y at level 3 = 6 cells across
    W, z at level 2) on walls at 3 W, against level 2 on the same 3 W box, and
    VERTICAL (z at level 3, x, y at level 2) on the study's 10 W box, against
    the design's own level-2 evaluation. Two checks say how far the pieces
    can be trusted: whether a level shift depends on the box (level 1 -> 2 at
    the paper values, 3 W vs 10 W) and whether the two directions add (level
    1 -> 2 at the paper values: the full step against x,y-only plus z-only)."""
    rs = load_rs()
    cs = (study.get("fine") or {}).get("cases", {})
    nom = study.get("nominal_levels", {})
    out: dict[str, Any] = {
        "full_level3": {"n_unknowns": 2577198, "plan_gb": study.get("plan_level3_gb"),
                        "status": "not run: a 106.71 GB factor exceeds gpu-a6000-1 (48 GB device, "
                                  "32 GiB host limit); of this cluster's presets only the "
                                  "H200 (141 GB) could hold it, and this study may not use it"},
        "cases": {k: {kk: v.get(kk) for kk in ("refine", "z_refine", "wall_w", "at", "L_diff",
                                               "Q_diff", "plan_gb", "seconds", "skipped",
                                               "error")}
                  | {"n_unknowns": (v.get("grid") or {}).get("n_unknowns"),
                     "cells_across_width": (v.get("grid") or {}).get("cells_across_width")}
                  for k, v in cs.items()}}
    lvl2_opt = ({"L_diff": fe["L_diff"], "Q_diff": fe["Q_diff"]} if fe else None)
    out["inplane_at_optimum"] = _shift(cs.get("opt_m3z2_w3"), cs.get("opt_m2_w3"))
    out["vertical_at_optimum"] = _shift(cs.get("opt_m2z3_w10"), lvl2_opt)
    out["level1_to_2_at_optimum"] = _shift(lvl2_opt, cs.get("opt_m1_w10"))
    out["inplane_at_nominal"] = _shift(cs.get("nom_m3z2_w3"), cs.get("nom_m2_w3"))
    out["vertical_at_nominal"] = _shift(cs.get("nom_m2z3_w10"), nom.get("2"))
    box10 = _shift(nom.get("2"), nom.get("1"))
    box3 = _shift(cs.get("nom_m2_w3"), cs.get("nom_m1_w3"))
    if box10 and box3:
        out["box_check_level1_to_2_nominal"] = {
            "walls_10W": box10, "walls_3W": box3,
            "dL_points": box3["dL_rel"] - box10["dL_rel"],
            "dQ_points": box3["dQ_rel"] - box10["dQ_rel"]}
    l11, l22 = nom.get("1"), nom.get("2")
    l21, l12 = cs.get("nom_m2z1_w10"), cs.get("nom_m1z2_w10")
    if l11 and l22 and l21 and l12 and all("L_diff" in r for r in (l21, l12)):
        full = _shift(l22, l11)
        xy, zz = _shift(l21, l11), _shift(l12, l11)
        assert full and xy and zz
        out["additivity_check_level1_to_2_nominal"] = {
            "full": full, "xy_only": xy, "z_only": zz,
            "dL_sum": xy["dL_rel"] + zz["dL_rel"], "dQ_sum": xy["dQ_rel"] + zz["dQ_rel"],
            "dL_points_full_minus_sum": full["dL_rel"] - xy["dL_rel"] - zz["dL_rel"],
            "dQ_points_full_minus_sum": full["dQ_rel"] - xy["dQ_rel"] - zz["dQ_rel"]}
    ip, vt = out["inplane_at_optimum"], out["vertical_at_optimum"]
    if ip and vt:
        out["dL_rel"] = ip["dL_rel"] + vt["dL_rel"]
        out["dQ_rel"] = ip["dQ_rel"] + vt["dQ_rel"]
        out["combined_what"] = ("the one-level-finer shift at the optimum as the SUM of the "
                                "measured in-plane and vertical shifts (their additivity and "
                                "box independence are the two checks above)")
        if fe:
            out["L_finer_estimate"] = fe["L_diff"] * (1.0 + out["dL_rel"])
            out["Q_finer_estimate"] = fe["Q_diff"] * (1.0 + out["dQ_rel"])
            out["L_err_finer_estimate"] = out["L_finer_estimate"] / L_TARGET - 1.0
        out["passed"] = True
        out["gate"] = "report (R7 is a reported shift, not a tolerance)"
    else:
        out["passed"] = None
    out["resolved_level"] = rs.RESOLVED
    out["inplane_grid_at_10W_n_unknowns"] = study.get("inplane_10w_n")
    out["caveats"] = r7_caveats(out, cs, fe)
    return out


# the smallest implied order from which a three-point limit is reported; below
# it the sequence is not in an asymptotic regime (L gives 1.21, Q 0.21 here)
ORDER_MIN = 0.5


def implied_order(ratio: float) -> float:
    """``p`` with ``(2^-p - 3^-p) / (1 - 2^-p) = ratio``: the order a
    sequence L(h) = L* - C h^p on h = 1, 1/2, 1/3 (levels 1, 2, 3; each a
    cut of the level-1 intervals) must have for its step 2 -> 3 to be
    ``ratio`` times its step 1 -> 2. The left side falls monotonically from
    1 (p -> 0) to 0 (p -> inf); bisection."""
    def f(p: float) -> float:
        return (2.0 ** -p - 3.0 ** -p) / (1.0 - 2.0 ** -p) - ratio
    lo, hi = 1e-6, 20.0
    if not (f(lo) > 0 > f(hi)):
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


def r7_caveats(r7: dict[str, Any], cs: dict[str, Any],
               fe: dict[str, Any] | None) -> dict[str, Any]:
    """What limits the R7 estimate, as numbers: (1) the in-plane half's box
    (3 W) against the design's (10 W), in absolute L at the optimum; (2)
    level 3 is not a refinement of level 2 (h = 1/3 against 1/2 of the
    level-1 cell, not nested); (3) the order the level 1 -> 2 step and the
    estimated 2 -> 3 step imply, and where L would then converge -- a
    three-point extrapolation whose third point is itself an estimate."""
    out: dict[str, Any] = {}
    w3 = cs.get("opt_m2_w3")
    if fe and w3 and "L_diff" in w3:
        out["inplane_box_3W_vs_10W_at_optimum"] = {
            "L_diff_3W": w3["L_diff"], "L_diff_10W": fe["L_diff"],
            "rel": w3["L_diff"] / fe["L_diff"] - 1.0,
            "what": "level 2 at the optimum on the in-plane half's 3 W box against the design's "
                    "10 W evaluation: the in-plane shift was measured on a model whose absolute "
                    "L is this much lower"}
    out["nesting"] = ("level m cuts every level-1 interval into m: level 3 (h = 1/3) and "
                      "level 2 (h = 1/2) both refine level 1, but no level-2 line other than "
                      "the level-1 ones is a level-3 line; 'one level finer' is h/3 against "
                      "h/2, a 2/3 step, not a nested halving")
    l1 = cs.get("opt_m1_w10")
    if fe and l1 and "L_diff" in l1 and r7.get("dL_rel") is not None:
        ext: dict[str, Any] = {"what": "L(h) = L* - C h^p through level 1 (h = 1), level 2 "
                                       "(h = 1/2) and the R7 estimate of level 3 (h = 1/3), all "
                                       "at the optimum; the level-3 value is the SUM of two "
                                       "halves (R7), so this is an estimate on an estimate"}
        for q, key in (("L", "dL_rel"), ("Q", "dQ_rel")):
            v1, v2 = l1[f"{q}_diff"], fe[f"{q}_diff"]
            v3 = v2 * (1.0 + r7[key])
            ratio = (v3 - v2) / (v2 - v1)
            p = implied_order(ratio)
            vinf = v2 + (v2 - v1) * 2.0 ** -p / (1.0 - 2.0 ** -p) if p == p else float("nan")
            ext[q] = {"level1": v1, "level2": v2, "level3_estimate": v3,
                      "step_ratio_23_over_12": ratio, "implied_order": p}
            if p == p and p >= ORDER_MIN:
                ext[q].update(limit=vinf, level2_rel_to_limit=v2 / vinf - 1.0)
            else:
                ext[q].update(limit=None, level2_rel_to_limit=None,
                              reading=f"implied order {p:.2f} < {ORDER_MIN:g}: not an "
                                      "asymptotic sequence, no limit is extrapolated (on Q the "
                                      "R7 sum over-states the step, see the additivity check)")
        if "L" in ext and ext["L"]["limit"] == ext["L"]["limit"]:
            ext["L_err_at_limit"] = ext["L"]["limit"] / L_TARGET - 1.0
        out["extrapolation"] = ext
    return out


def walls_design_block(walls: dict[str, Any], fe: dict[str, Any] | None,
                       kkt: dict[str, Any] | None = None) -> dict[str, Any]:
    """The 10 W wall bias across the design box, from ``rfic_spiral.json``
    ``walls`` (level 1, CPU): the walls are pinned in space while theta
    moves, so the wall-to-strip distance and the bias change with r_out.
    Applied to the design's level-2 L_diff as an ESTIMATE (the level-1
    relative change carried to level 2)."""
    pp = walls.get("per_point", {})
    if not pp:
        return {}
    out: dict[str, Any] = {
        "source": "rfic_spiral.json walls (level 1, SuperLU on the CPU, walls and lid at "
                  "5 / 10 / 20 W)",
        "per_point": {k: {kk: v.get(kk) for kk in ("dL_rel_10_to_20W", "dQ_rel_10_to_20W",
                                                   "dL_rel_5_to_10W", "wall_x_minus_r_out")}
                      for k, v in pp.items()},
        "points": walls.get("points"),
        "dL_rel_10_to_20W_range_over_box": walls.get("dL_rel_10_to_20W_range_over_box"),
        "design_minus_nominal_points": walls.get("design_minus_nominal_points")}
    dq = [abs(v["dQ_rel_10_to_20W"]) for v in pp.values() if "dQ_rel_10_to_20W" in v]
    if dq:
        out["max_abs_dQ_rel_10_to_20W"] = max(dq)
    d = pp.get("design", {})
    hi = pp.get("r_out_hi", {})
    pts = walls.get("points") or {}
    if "dL_rel_10_to_20W" in d and "dL_rel_10_to_20W" in hi and "design" in pts:
        dr = pts["r_out_hi"][0] - pts["design"][0]
        slope = (hi["dL_rel_10_to_20W"] - d["dL_rel_10_to_20W"]) / dr
        out["bias_slope_per_m_r_out"] = slope
        out["bias_slope_what"] = ("two-point slope of the 10 -> 20 W change in r_out, theta* "
                                  "to the r_out = 260 um sweep point (the two also differ in "
                                  "spacing and width)")
        if fe and kkt and kkt.get("dL_dtheta"):
            rel = kkt["dL_dtheta"][0] / fe["L_diff"]
            out["dL_rel_dr_out_at_theta_star"] = rel
            out["bias_slope_over_dL_rel_dr_out"] = slope / rel
    if fe and "dL_rel_10_to_20W" in d:
        est = fe["L_diff"] * (1.0 + d["dL_rel_10_to_20W"])
        out["design_L_at_20W_estimate"] = est
        out["design_L_err_at_20W_estimate"] = est / L_TARGET - 1.0
        out["design_Q_at_20W_estimate"] = fe["Q_diff"] * (1.0 + d["dQ_rel_10_to_20W"])
        out["what"] = ("the design's level-2 L_diff times (1 + its level-1 10 -> 20 W change): "
                       "an estimate of the design on a box with walls twice as far; stated, "
                       "not applied (R4 gates the 10 W model it was designed on)")
    return out


def assemble() -> dict[str, Any]:
    rs = load_rs()
    t0 = time.time()
    p1, r1 = rs.lane_json("r1", "r1_study.json")
    p2, r2 = rs.lane_json("r2", "r2_design.json")
    p3, r3 = rs.lane_json("r3", "r3_fdopt.json")
    p4, r4 = rs.lane_json("r4", "r4_fine.json")
    study: dict[str, Any] = {
        "what": "study R part C: L-BFGS-B design of the paper-scale square spiral to "
                "L_diff = 4.000 nH (1 %) maximising Q_diff at 2.45 GHz on the resolved level, "
                "against a 3x3x3 sweep of the same box; cuDSS on the GPU",
        "paper_context": {"L_diff": 4.000e-9, "Q_diff": rs.Q_PAPER,
                          "what": "arXiv 2607.08852: a symmetric OCTAGONAL spiral with the same "
                                  "(r_out, W, S); this is a square single-ended spiral, so its L "
                                  "is higher at equal size and the numbers are NOT comparable one "
                                  "to one -- context only, never a gate"},
        "problem": {"L_target": L_TARGET, "band": L_BAND, "box": [list(b) for b in BOX],
                    "sweep_axes": [list(a) for a in SWEEP_AXES], "lam": LAM, "q_ref": Q_REF,
                    "loss": "-Q_diff / q_ref + lam (L_diff / L_target - 1)^2 on the "
                            "box-normalised u in [0, 1]^3",
                    "start": list(rs.THETA0), "level": rs.RESOLVED},
        "sources": {k: str(p.relative_to(REPO)) if p else None
                    for k, p in (("r1", p1), ("r2", p2), ("r3", p3), ("r4", p4))},
        "runs": [row for row in rs.load_ledger() if row.get("lane", "").startswith("r")],
    }
    rsj = json.loads(rs.JSON_PATH.read_text()) if rs.JSON_PATH.exists() else {}
    study["accuracy"] = rsj.get("accuracy") or rs.accuracy_statement()
    lv1 = r1.get("levels", {})
    study["nominal_levels"] = {k: {kk: lv1[k].get(kk) for kk in ("L_diff", "Q_diff", "theta")}
                               for k in ("1", "2") if isinstance(lv1.get(k), dict)}
    study["plan_level3_gb"] = (r1.get("plans", {}).get("190:2:3") or {}).get(
        "permanent_device_memory_gb")
    # the in-plane-finer grid on the study's own 10 W box (built here, not solved)
    study["inplane_10w_n"] = int(rs.build_r(3, z_refine=2).n_unknowns)
    study["csweep"] = r3.get("csweep") or r1.get("csweep", {})
    study["design"] = r2.get("design", {})
    fo = r2.get("fdopt") if (r2.get("fdopt") or {}).get("fd") else r3.get("fdopt", {})
    study["fdopt"] = fo or {}
    study["fine"] = r4.get("fine", {})
    study["cost"] = cost_block(study["design"], study["csweep"])
    study["gates"] = evaluate_gates(study)
    study["kkt"] = kkt_block(study, final_eval(study["design"]))
    study["walls"] = walls_design_block(rsj.get("walls") or {}, final_eval(study["design"]),
                                        study["kkt"])
    study["remesh"] = rsj.get("remesh") or {}
    study["seconds_assembly"] = time.time() - t0
    return study


def figure(study: dict[str, Any], path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    pts = study.get("csweep", {}).get("points", [])
    ev = study.get("design", {}).get("evals", [])
    fe = final_eval(study.get("design", {}))
    ax = axs[0, 0]
    ax.axvspan(L_TARGET * 1e9 * (1 - L_BAND), L_TARGET * 1e9 * (1 + L_BAND), color="#3b7d3b",
               alpha=0.18, label="4.000 nH +- 1 %")
    if pts:
        ax.scatter([p["L_diff"] * 1e9 for p in pts], [p["Q_diff"] for p in pts], c="0.45", s=24,
                   label="3x3x3 sweep (27 forward solves)")
    if ev:
        ax.plot([e["L_diff"] * 1e9 for e in ev], [e["Q_diff"] for e in ev], ".-", color="#1f5fa8",
                lw=0.8, label=f"L-BFGS-B evaluations ({len(ev)})")
    if fe:
        ax.plot([fe["L_diff"] * 1e9], [fe["Q_diff"]], "*", color="#c2571a", ms=15, label="design")
    ax.set_xlabel("L_diff (nH)")
    ax.set_ylabel("Q_diff")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("(a) sweep vs design, 2.45 GHz, resolved level")
    ax = axs[0, 1]
    if ev:
        n = [e["n"] for e in ev]
        ax.plot(n, [100 * e["L_err"] for e in ev], "o-", color="#1f5fa8", ms=4)
        ax.axhspan(-100 * L_BAND, 100 * L_BAND, color="#3b7d3b", alpha=0.18)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("objective evaluation")
        ax.set_ylabel("L_diff / 4 nH - 1 (%)", color="#1f5fa8")
        ax2 = ax.twinx()
        ax2.plot(n, [e["Q_diff"] for e in ev], "s--", color="#c2571a", ms=4)
        ax2.set_ylim(23.0, 24.1)
        ax2.set_ylabel("Q_diff (evaluation 1 is off scale)", color="#c2571a")
        ax.set_title("(b) every evaluation")
    ax = axs[1, 0]
    if pts:
        sc = ax.scatter([p["theta"][0] * 1e6 for p in pts], [p["theta"][2] * 1e6 for p in pts],
                        c=[p["L_diff"] * 1e9 for p in pts], cmap="viridis", s=36)
        fig.colorbar(sc, ax=ax, label="sweep L_diff (nH), all spacings")
    if ev:
        ax.plot([e["theta"][0] * 1e6 for e in ev], [e["theta"][2] * 1e6 for e in ev], ".-",
                color="#c2571a", lw=0.8)
    if fe:
        ax.plot([fe["theta"][0] * 1e6], [fe["theta"][2] * 1e6], "*", color="#c2571a", ms=15)
    ax.set_xlabel("r_out (um)")
    ax.set_ylabel("width (um)")
    ax.set_title("(c) the path in (r_out, width)")
    ax = axs[1, 1]
    r7 = study.get("gates", {}).get("R7", {})
    rows = [("in-plane\n(optimum)", r7.get("inplane_at_optimum")),
            ("vertical\n(optimum)", r7.get("vertical_at_optimum")),
            ("sum\n(optimum)", {"dL_rel": r7.get("dL_rel"), "dQ_rel": r7.get("dQ_rel")}
             if r7.get("dL_rel") is not None else None),
            ("level 1->2\n(optimum)", r7.get("level1_to_2_at_optimum"))]
    rows = [(k, v) for k, v in rows if v]
    if rows:
        x = np.arange(len(rows))
        ax.bar(x - 0.2, [100 * v["dL_rel"] for _, v in rows], 0.4, color="#1f5fa8", label="L_diff")
        ax.bar(x + 0.2, [100 * v["dQ_rel"] for _, v in rows], 0.4, color="#c2571a", label="Q_diff")
        ax.set_xticks(x)
        ax.set_xticklabels([k for k, _ in rows], fontsize=8)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("relative shift (%)")
        ax.legend(fontsize=8)
        ax.set_title("(d) R7: one level finer at the optimum")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="study R part C (assembly)")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    study = assemble()
    JSON_PATH.write_text(json.dumps(study, indent=1, default=float) + "\n")
    print(f"wrote {JSON_PATH}")
    for k, v in study["gates"].items():
        print(f"  {k}: passed={v.get('passed')}")
    if not args.no_figure:
        figure(study, PNG_PATH)
        print(f"wrote {PNG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
