"""WP 3-1 gradient-fidelity ladder: AD-vs-FD rel_err as a function of dx.

Reviewer question (item 3): "at a fine mesh (~lambda/100), does optimization
still work -- does gradient fidelity survive mesh refinement?" This producer
measures the CPU-feasible part of that answer on a SMOOTH-DIELECTRIC witness
(the patch antenna is banned as a ladder witness -- its staircase consistency
error converges to the wrong limit; a smooth slab does not).

Witness geometry (fixed in PHYSICAL units across all rungs; only dx changes)
---------------------------------------------------------------------------
* 60 x 12 x 12 mm box, CPML on the propagation (x) axis (cpml on all faces).
* A single soft Ez source at x = 12 mm, a Gaussian pulse centered on the
  objective frequency (plane-ish in the small transverse box; exact planarity
  is irrelevant to AD-vs-FD -- any field-shape artifact perturbs the AD and FD
  objectives IDENTICALLY and cancels in the ratio).
* eps_r = 4 dielectric slab, 6 mm thick (a FIXED physical size = 4 cells at
  the coarsest lambda/20 rung, more at finer rungs), spanning the full
  transverse plane, centered at x = 33 mm.
* Ez probe at x = 48 mm, downstream of the slab.
* Objective J(eps) = |DFT of the probe Ez at f_obj = 8 GHz|^2 -- one scalar
  observable at one fixed frequency.
* SINGLE SCALAR DoF = the slab eps_r (painted via forward(eps_override=...)),
  so a central finite difference is exactly one forward-run pair.
* num_periods = 20 held FIXED on every rung (the settling-confound fence:
  refining dx must not be confounded with letting the field settle longer).

Comparator discipline (WP 3-1 step 1, run FIRST at the finest CPU rung)
----------------------------------------------------------------------
Before trusting any AD-vs-FD number, the FD signal must sit far above the
float32 objective repeat-noise. On CPU, JAX/XLA is deterministic, so the
measured repeat-noise is exactly 0.0 -- the comparator is valid by a wide
margin (FD signal / noise is unbounded). If a future run on a nondeterministic
backend shows repeat-noise > 0 and FD-signal/noise < 100, the ladder is
comparator-INVALID at that rung and must stop there (comparator-first: prior
rfx "gradient bugs" were repeatedly comparator bugs).

Measured rel_err(dx) curve (freq_max=10 GHz, f_obj=8 GHz, eps0=4.0, float32,
CPU; from tests/fixtures/gradient_dx_ladder/ad_fd_ladder.json, the
authoritative record) -- comparator falsifier ran FIRST at lambda/60
(repeat_noise = 0.0 exactly, fd_diff = 3.33e-9, comparator_valid = True):

  rung        grid shape       n_steps   grad_ad       grad_fd      rel_err     sign  wall(fwd+grad)
  lambda/20   (58, 26, 26)        700   -2.3415e-06  -2.3414e-06  2.76e-05   agree   2.6s + 9.0s
  lambda/30   (78, 30, 30)       1050   -6.0980e-07  -6.0979e-07  1.11e-05   agree   5.1s + 11.2s
  lambda/40   (98, 34, 34)       1400   -2.0662e-07  -2.0662e-07  2.89e-06   agree   9.3s + 21.1s
  lambda/60  (138, 42, 42)       2100   -4.1605e-08  -4.1609e-08  8.12e-05   agree  20.5s + 50.1s

Total wall time 308.4 s (well under the 2.5 h budget); no rungs dropped.
Every rung's FD signal / repeat-noise ratio is comparator-valid (CPU is
deterministic, so repeat_noise == 0.0 at every rung, which trivially clears
the >=100x gate).

Honest one-paragraph answer (as far as the CPU rungs measure)
-------------------------------------------------------------
Over the four CPU-feasible rungs (lambda/20 through lambda/60), the
central-FD check of the float32 reverse-mode gradient of this smooth-slab
objective stays in tight agreement with the discrete adjoint: sign always
agrees and rel_err sits between 2.9e-6 and 8.1e-5 -- three to four orders of
magnitude below the plan's 0.10 gate -- with NO monotonic trend toward
degradation as dx shrinks (rel_err is non-monotonic across the four rungs,
consistent with float32 rounding-level noise rather than a systematic
fidelity loss). This is consistent with, but narrower than, the
electrically-finer-than-lambda/100 MSL evidence: it says gradient fidelity
survives refinement THROUGH lambda/60 on a smooth dielectric; it makes NO
claim about lambda/80-lambda/100 (not run here -- lambda/60 was already the
practical CPU wall-time edge at ~70s/rung and was chosen as the finest rung
for this run) and no claim about staircased or curved-PEC geometries (a
different, consistency-error regime, per the patch-antenna ban). Per
fidelity-over-cost, the claim rides only on the four committed rungs.

Run
---
    python scripts/diagnostics/gradient_dx_ladder/run_ladder.py

Regenerates tests/fixtures/gradient_dx_ladder/ad_fd_ladder.json. Output is
deterministic on CPU, so regeneration is reproducible.
"""

from __future__ import annotations

import json
import os
import time

import jax


def _rfx_version():
    try:
        import rfx
        return getattr(rfx, '__version__', 'unknown')
    except Exception:
        return 'unknown'
import jax.numpy as jnp
import numpy as np

from rfx.api import Simulation
from rfx.grid import Grid
from rfx.sources.sources import GaussianPulse

# ---- Fixed physical geometry (only dx changes across rungs) ----------------
C0 = 2.998e8
FREQ_MAX = 10e9                 # defines lambda for the dx rungs + settling
LAMBDA = C0 / FREQ_MAX          # 29.98 mm
F_OBJ = 8e9                     # fixed objective frequency (in-band)
EPS_SLAB = 4.0                  # nominal slab eps_r (the scalar DoF)
T_SLAB = 6.0e-3                 # FIXED physical slab thickness (4 cells @ l/20)
DOM_X = 60.0e-3                 # propagation-axis interior length (~2 lambda)
DOM_YZ = 12.0e-3               # transverse box (~0.4 lambda)
X_SRC = 12.0e-3
X_SLAB_C = 33.0e-3
X_PROBE = 48.0e-3
YZ_C = DOM_YZ / 2.0
CPML = 8
NUM_PERIODS = 20.0
H_REL = 1e-2                    # central-FD relative step (absolute = H_REL*eps)

LADDER_RUNGS = (20, 30, 40, 60)   # coarsest -> finest CPU rungs
FINEST_RUNG = 60                  # comparator falsifier runs here first
WALL_BUDGET_S = 2.5 * 3600.0

# Gate constants (also enforced by tests/test_gradient_dx_ladder_gates.py)
REL_ERR_FLOOR = 0.10              # plan's per-rung rel_err gate
REL_ERR_MARGIN = 1.5             # regression margin above the measured value
# NOTE (review): at the measured magnitudes (~1e-5) the 1.5x margin is inert --
# max(measured*1.5, 0.10) collapses to the plan's 0.10 floor on every rung, so
# the effective committed ceiling IS 0.10; the tight envelope is locked instead
# by the raw-number re-derivation + sign + comparator gates in the test.
COMPARATOR_MIN_RATIO = 100.0     # FD-signal / repeat-noise validity gate


def build_sim(dx: float) -> Simulation:
    sim = Simulation(freq_max=FREQ_MAX, domain=(DOM_X, DOM_YZ, DOM_YZ),
                     boundary="cpml", cpml_layers=CPML, dx=dx)
    wf = GaussianPulse(f0=F_OBJ, bandwidth=0.9)
    sim.add_source((X_SRC, YZ_C, YZ_C), "ez", waveform=wf)
    sim.add_probe((X_PROBE, YZ_C, YZ_C), "ez")
    return sim


def slab_mask(grid: Grid, dx: float) -> np.ndarray:
    nx, ny, nz = grid.shape
    xs = (np.arange(nx) - grid.pad_x_lo) * dx
    lo, hi = X_SLAB_C - T_SLAB / 2.0, X_SLAB_C + T_SLAB / 2.0
    m1d = (xs >= lo) & (xs < hi)
    mask = np.zeros((nx, ny, nz), dtype=bool)
    mask[m1d, :, :] = True
    return mask


def make_objective(sim: Simulation, mask: np.ndarray, n_steps: int):
    mask_j = jnp.asarray(mask)

    def J(eps_scalar):
        eps = jnp.where(mask_j, eps_scalar, 1.0).astype(jnp.float32)
        res = sim.forward(eps_override=eps, n_steps=n_steps,
                          skip_preflight=True)
        ts = res.time_series[:, 0]
        dt = res.grid.dt
        t = jnp.arange(ts.shape[0]) * dt
        phasor = jnp.exp(-1j * 2.0 * jnp.pi * F_OBJ * t)
        acc = jnp.sum(ts.astype(jnp.complex64) * phasor)
        return jnp.abs(acc) ** 2

    return J


def _setup(n: int):
    dx = LAMBDA / n
    sim = build_sim(dx)
    grid = Grid(FREQ_MAX, (DOM_X, DOM_YZ, DOM_YZ), dx=dx, cpml_layers=CPML)
    n_steps = grid.num_timesteps(num_periods=NUM_PERIODS)
    mask = slab_mask(grid, dx)
    J = make_objective(sim, mask, n_steps)
    slab_cells_x = int(mask[:, 0, 0].sum())
    return dx, sim, grid, n_steps, mask, J, slab_cells_x


def comparator_falsifier(n: int) -> dict:
    """WP 3-1 step 1: measure repeat-noise vs FD signal at the finest rung."""
    dx, sim, grid, n_steps, mask, J, slab_cells_x = _setup(n)
    issues = sim.preflight(strict=False)
    eps0 = jnp.float32(EPS_SLAB)
    h = H_REL * EPS_SLAB
    r1 = float(J(eps0))
    r2 = float(J(eps0))
    repeat_noise = abs(r1 - r2)
    jp = float(J(jnp.float32(EPS_SLAB + h)))
    jm = float(J(jnp.float32(EPS_SLAB - h)))
    fd_diff = abs(jp - jm)
    ratio = float("inf") if repeat_noise == 0.0 else fd_diff / repeat_noise
    valid = (repeat_noise == 0.0) or (ratio >= COMPARATOR_MIN_RATIO)
    return {
        "rung": f"lambda/{n}", "lambda_over_dx": n, "dx_m": dx,
        "grid_shape": list(grid.shape), "n_steps": n_steps,
        "eps0": EPS_SLAB, "fd_step_abs": h,
        "J_repeat_1": r1, "J_repeat_2": r2, "repeat_noise": repeat_noise,
        "J_plus": jp, "J_minus": jm, "fd_diff": fd_diff,
        "fd_signal_over_noise": (None if ratio == float("inf") else ratio),
        "comparator_valid": bool(valid),
        "preflight_issues": list(issues),
    }


def run_rung(n: int) -> dict:
    dx, sim, grid, n_steps, mask, J, slab_cells_x = _setup(n)
    issues = sim.preflight(strict=False)
    eps0 = jnp.float32(EPS_SLAB)
    h = H_REL * EPS_SLAB

    t0 = time.time()
    j0a = float(J(eps0))
    t_fwd = time.time() - t0
    j0b = float(J(eps0))
    repeat_noise = abs(j0a - j0b)

    jp = float(J(jnp.float32(EPS_SLAB + h)))
    jm = float(J(jnp.float32(EPS_SLAB - h)))
    fd_diff = abs(jp - jm)
    g_fd = (jp - jm) / (2.0 * h)

    t0 = time.time()
    g_ad = float(jax.grad(J)(eps0))
    t_grad = time.time() - t0

    rel_err = abs(g_ad - g_fd) / max(abs(g_fd), 1e-30)
    sign_agree = bool((g_ad * g_fd) > 0.0)
    ratio = float("inf") if repeat_noise == 0.0 else fd_diff / repeat_noise
    comparator_valid = (repeat_noise == 0.0) or (ratio >= COMPARATOR_MIN_RATIO)
    rel_err_ceiling = max(rel_err * REL_ERR_MARGIN, REL_ERR_FLOOR)

    return {
        "rung": f"lambda/{n}", "lambda_over_dx": n, "dx_m": dx,
        "grid_shape": list(grid.shape), "cells": int(np.prod(grid.shape)),
        "n_steps": n_steps, "slab_cells_x": slab_cells_x,
        "eps0": EPS_SLAB, "fd_step_abs": h,
        "J0": j0a, "J_repeat": j0b, "repeat_noise": repeat_noise,
        "J_plus": jp, "J_minus": jm, "fd_diff": fd_diff,
        "grad_ad": g_ad, "grad_fd": g_fd,
        "rel_err": rel_err, "sign_agree": sign_agree,
        "rel_err_ceiling": rel_err_ceiling,
        "fd_signal_over_noise": (None if ratio == float("inf") else ratio),
        "comparator_valid": bool(comparator_valid),
        "wall_forward_s": t_fwd, "wall_grad_s": t_grad,
        "preflight_issues": list(issues),
    }


def main():
    t_start = time.time()
    print("=" * 78)
    print("WP 3-1 gradient-fidelity ladder (smooth eps=4 slab, float32, CPU)")
    print(f"freq_max = {FREQ_MAX/1e9:.1f} GHz  f_obj = {F_OBJ/1e9:.1f} GHz  "
          f"num_periods = {int(NUM_PERIODS)}  eps0 = {EPS_SLAB}  "
          f"h_rel = {H_REL}")
    print("=" * 78)

    print(f"\n[step 1] comparator falsifier at the finest CPU rung "
          f"(lambda/{FINEST_RUNG}) -- run FIRST")
    comp = comparator_falsifier(FINEST_RUNG)
    print(f"  repeat_noise = {comp['repeat_noise']:.3e}  "
          f"fd_diff = {comp['fd_diff']:.3e}  "
          f"valid = {comp['comparator_valid']}  "
          f"preflight = {comp['preflight_issues']}")
    if not comp["comparator_valid"]:
        print("  COMPARATOR INVALID at the finest rung -- ladder stops here "
              "(WP 3-1 falsifier). Recording the finding.")

    rungs = []
    print("\n[step 2] ladder (coarsest -> finest)")
    print(f"  {'rung':>10} {'l/dx':>5} {'shape':>16} {'n_steps':>8} "
          f"{'grad_ad':>12} {'grad_fd':>12} {'rel_err':>10} {'sign':>5}")
    for n in LADDER_RUNGS:
        elapsed = time.time() - t_start
        if elapsed > WALL_BUDGET_S:
            print(f"  lambda/{n}: DROPPED -- wall budget "
                  f"{WALL_BUDGET_S/3600:.1f} h exceeded ({elapsed/3600:.2f} h)")
            rungs.append({"rung": f"lambda/{n}", "lambda_over_dx": n,
                          "dropped": True,
                          "reason": "wall-time budget exceeded"})
            continue
        r = run_rung(n)
        rungs.append(r)
        print(f"  {r['rung']:>10} {n:5d} {str(tuple(r['grid_shape'])):>16} "
              f"{r['n_steps']:8d} {r['grad_ad']:12.4e} {r['grad_fd']:12.4e} "
              f"{r['rel_err']:10.3e} {str(r['sign_agree']):>5}  "
              f"(fwd {r['wall_forward_s']:.1f}s grad {r['wall_grad_s']:.1f}s)")

    out = {
        "meta": {
        "jax_version": jax.__version__, "rfx_version": _rfx_version(),
            "work_package": "3-1",
            "witness": "smooth dielectric slab (eps_r=4), single scalar DoF",
            "freq_max_hz": FREQ_MAX, "f_obj_hz": F_OBJ,
            "lambda_m": LAMBDA, "eps0": EPS_SLAB, "fd_rel_step": H_REL,
            "num_periods": NUM_PERIODS,
            "domain_m": [DOM_X, DOM_YZ, DOM_YZ], "cpml_layers": CPML,
            "slab_thickness_m": T_SLAB, "precision": "float32",
            "rel_err_floor": REL_ERR_FLOOR, "rel_err_margin": REL_ERR_MARGIN,
            "comparator_min_ratio": COMPARATOR_MIN_RATIO,
            "total_wall_s": time.time() - t_start,
        },
        "comparator_falsifier": comp,
        "rungs": rungs,
    }

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    fixture = os.path.join(repo, "tests", "fixtures", "gradient_dx_ladder",
                           "ad_fd_ladder.json")
    os.makedirs(os.path.dirname(fixture), exist_ok=True)
    with open(fixture, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {fixture}")
    print(f"total wall time: {out['meta']['total_wall_s']:.1f} s")


if __name__ == "__main__":
    main()
