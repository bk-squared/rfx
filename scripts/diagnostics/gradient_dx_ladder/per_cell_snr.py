"""WP 3-2 per-cell gradient-SNR ladder: does topology-scale sensitivity sink
below the float32 noise floor as the mesh refines?

Reviewer question (item 3-2, the topology-optimization variant of the dx
ladder). For differentiable-FDTD TOPOLOGY optimization the design variable is
PER-CELL permittivity, not a single scalar. Hypothesis: a single cell's
sensitivity ``|dJ/d(eps_cell)|`` is a volume integral of a smooth continuous
density over that one cell, so it should scale ~ cell volume (dx^3), while any
FIXED (dx-independent) float32 noise floor stays put. If so, at fine dx the
per-cell gradients would sink below the noise floor and topology-scale
optimization would lose per-cell SNR. This had NEVER been measured in rfx.
This producer measures the CPU-feasible part of that answer on a SMOOTH
DIELECTRIC block (the patch antenna is banned as a ladder witness -- its
staircase consistency error converges to the wrong limit; a smooth block does
not) and extends the WP 3-1 SINGLE-SCALAR ladder to a PER-CELL design region.

Witness geometry (FIXED in PHYSICAL units across all rungs; only dx changes)
---------------------------------------------------------------------------
* 30 x 12 x 12 mm interior box, CPML (8 cells) on every face (absorbing on the
  propagation x-axis; the transverse walls are far enough from the block to be
  irrelevant to the gradient-scaling measurement).
* One soft Ez source at x = 6 mm, a Gaussian pulse centered on the objective
  frequency.
* A smooth eps_r = 4 dielectric BLOCK filling the design region
  x in [13, 19] mm, y,z in [3, 9] mm -- a FIXED physical 6 x 6 x 6 mm cube
  (4^3 = 64 cells at lambda/20, 8^3 = 512 cells at lambda/40, i.e. the SAME
  physical region carries ~8x more cells per halving of dx).
* Ez probe at x = 24 mm, downstream of the block.
* Objective J(eps) = |DFT of the probe Ez at f_obj = 8 GHz|^2 -- one scalar
  observable at one fixed frequency.
* DoF = PER-CELL eps_r over the design box (a 3D jnp array painted via
  ``base_eps_r.at[si:ei+1, sj:ej+1, sk:ek+1].set(eps_design)`` -- the exact
  contiguous-box painting the production topology path uses in
  ``rfx/optimize.py``). ``g = jax.grad(J)(eps_design)`` is the full per-cell
  gradient FIELD (one reverse pass yields every cell's sensitivity).
* num_periods = 20 held FIXED on every rung (the settling-confound fence:
  refining dx must not be confounded with letting the field settle longer).

R2/R5 redesign note -- why the metric is the NORMALIZED sensitivity
-------------------------------------------------------------------
The first 2-rung pilot measured the RAW per-cell |g| and found it decays as
dx^6.9 (median|g| falls ~120x per halving of dx), NOT dx^3. Per R5 (no
surface-metric verdict) that surprising exponent was decomposed before any
verdict: the DFT-accumulator objective J is itself NOT mesh-invariant (its own
magnitude scales ~dx^3.9 because dt, the step count at fixed num_periods, and
the discrete field all drift with dx). The raw per-cell decay therefore
FACTORS as ``p_raw = p_J + 3`` -- the objective's own mesh-scaling TIMES the
per-cell volume dx^3. Dividing the objective scaling back out
(``g_rel = |dJ/deps_cell| / J``, the per-cell derivative of ``ln J``, a
mesh-invariant relative sensitivity) isolates the hypothesis's actual
mechanism: median ``g_rel`` decays as dx^3, cleanly. The RAW |g| metric is a
confounded comparator, so the falsifier + verdict ride on the NORMALIZED
metric; the raw numbers and the decomposition are recorded for transparency.

Noise-floor discipline (WP 3-2 comparator)
------------------------------------------
Two distinct floors matter, and they answer the reviewer's question
differently for reverse-mode AD vs finite differences:

* AD repeat-noise floor (the floor the rfx differentiable path actually sees).
  Measured as ``max|g_run1 - g_run2|`` over two bit-identical AD runs. On CPU,
  JAX/XLA is deterministic, so this is exactly 0.0 -- there is NO fixed absolute
  floor for the analytic gradient to cross on this backend. float32's remaining
  imprecision is RELATIVE (~2^-23 of each value), i.e. scale-invariant, so the
  per-cell AD SNR does not degrade as |g| shrinks. A finite crossing for AD
  would require a genuinely nondeterministic reduction (GPU atomic scatters, a
  mixed-precision tape), which is NOT measured here.
* FD-detectability floor (the floor a finite-difference topology method would
  hit). A central FD can only resolve a cell whose full-range perturbation
  moves J by more than float32 can represent, i.e. when the mesh-invariant
  relative sensitivity ``g_rel`` drops below ~``FLOAT32_ULP = 2^-23`` ~ 1.2e-7.
  Because ``g_rel`` decays as dx^3, this floor gives a genuine, FIXED,
  dx-independent crossing -- reported as an extrapolated lambda/dx.

A directional finite-difference cross-check (all cells +/- h) confirms the
per-cell AD field is a real gradient, not noise: ``sum(g)`` must match
``(J(+h) - J(-h)) / 2h``.

Falsifier FIRST (2-rung pilot: lambda/20 and lambda/40)
-------------------------------------------------------
Compare median ``g_rel`` at lambda/20 vs lambda/40. The dx^3 prediction is
``median g_rel(l/40) / median g_rel(l/20) ~ (dx_fine/dx_coarse)^3 = 0.125``.
The measured decay EXPONENT is ``p = log(ratio) / log(dx_fine/dx_coarse)``
(p = 3 is exact dx^3). Verdict:
* "decay-seen"    -- p in [2, 4] (roughly cubic per-cell decay is present);
* "null-no-decay" -- p outside [2, 4] (no dx^3-like per-cell decay).
A clear NULL is a DELIVERABLE finding, not a failure. If (and only if) decay is
seen, one intermediate rung (lambda/30) is added and the FD-detectability
crossing is characterized.

Measured per-rung table (freq_max=10 GHz, f_obj=8 GHz, eps_block=4.0, float32,
CPU; from tests/fixtures/gradient_dx_ladder/per_cell_snr.json, the authoritative
record). ``g_rel = |g| / J0`` is the mesh-invariant relative sensitivity:

  rung        grid shape     design   median|g|    J0          median g_rel  AD repeat-noise  dir_rel_err
  lambda/20   (38, 26, 26)   4^3= 64  8.8567e-08  2.7222e-04  3.2536e-04    0.0             2.81e-04
  lambda/30   (48, 30, 30)   6^3=216  6.1801e-09  5.6762e-05  1.0888e-04    0.0             1.53e-03
  lambda/40   (58, 34, 34)   8^3=512  7.3637e-10  1.8729e-05  3.9318e-05    0.0             1.41e-03

Falsifier (NORMALIZED, median g_rel): ratio median g_rel(l/40)/median g_rel(l/20)
= 0.1208 (predicted dx^3 = 0.125), measured exponent p = 3.05 -> VERDICT =
decay-seen. R5 decomposition: raw median|g| decays p_raw = 6.91 = objective
p_J = 3.86 + per-cell volume (residual 3.05, expected 3.0). AD repeat-noise
floor is 0.0 on all three rungs (deterministic CPU); the FD-detectability
crossing (median g_rel < float32 ULP 1.19e-7) is extrapolated to ~lambda/268
(NOT measured beyond lambda/40 -- and the DISTRIBUTION TAIL already shows it:
at lambda/40 the minimum per-cell g_rel = 9.3e-8 has just dipped below the FD
floor, so frac_rel_above_fd_floor = 0.998 there, vs 1.000 at lambda/20-30).

Honest one-paragraph answer -- "does per-cell gradient SNR degrade at fine dx
for topology optimization?" (measured rungs only)
-----------------------------------------------------------------------------
The per-cell RELATIVE sensitivity (mesh-invariant ``g_rel = |dJ/deps_cell| / J``)
DOES decay ~dx^3 across lambda/20 -> lambda/30 -> lambda/40 on a smooth eps=4
block -- the reviewer's "each cell is 1/8 the volume so carries 1/8 the
sensitivity" mechanism is confirmed once the objective's own mesh-scaling is
divided out (the RAW |g| decays even faster, ~dx^7, because the unnormalized
DFT objective is itself not mesh-invariant). Whether that shrinking signal
COLLAPSES the SNR depends entirely on the noise model, and the two paths differ:
for a finite-difference topology method there is a FIXED float32
detectability floor (~2^-23 relative), so the dx^3 decay predicts a real
crossing at an extrapolated fine mesh (labeled below, not measured beyond
lambda/40). For rfx's reverse-mode AD -- the actual differentiable-FDTD path --
the gradient repeat-noise floor is exactly 0.0 on deterministic CPU and
float32's residual imprecision is scale-invariant, so per-cell AD SNR does NOT
degrade as the mesh refines here: there is no measured SNR collapse for AD.
Per fidelity-over-cost this rides only on the three committed rungs
(lambda/20..40); it makes no claim about lambda/60-lambda/100, about staircased /
curved-PEC geometries, or about nondeterministic (e.g. GPU) reductions.

Run
---
    python scripts/diagnostics/gradient_dx_ladder/per_cell_snr.py

Regenerates tests/fixtures/gradient_dx_ladder/per_cell_snr.json. Output is
deterministic on CPU, so regeneration is reproducible.
"""

from __future__ import annotations

import json
import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from rfx.api import Simulation
from rfx.grid import Grid
from rfx.sources.sources import GaussianPulse


def _rfx_version():
    try:
        import rfx
        return getattr(rfx, "__version__", "unknown")
    except Exception:
        return "unknown"


# ---- Fixed physical geometry (only dx changes across rungs) ----------------
C0 = 2.998e8
FREQ_MAX = 10e9                 # defines lambda for the dx rungs + settling
LAMBDA = C0 / FREQ_MAX          # 29.98 mm
F_OBJ = 8e9                     # fixed objective frequency (in-band)
EPS_BLOCK = 4.0                 # nominal block eps_r (the per-cell DoF value)
DOM_X = 30.0e-3                 # propagation-axis interior length (~1 lambda)
DOM_YZ = 12.0e-3                # transverse box (~0.4 lambda)
X_SRC = 6.0e-3
X_PROBE = 24.0e-3
# Design region: a FIXED physical 6 x 6 x 6 mm block (topology DoF = per cell).
DES_X = (13.0e-3, 19.0e-3)
DES_Y = (3.0e-3, 9.0e-3)
DES_Z = (3.0e-3, 9.0e-3)
YZ_C = DOM_YZ / 2.0
CPML = 8
NUM_PERIODS = 20.0
H_REL = 1e-2                    # directional-FD relative step (abs = H_REL*eps)

PILOT_RUNGS = (20, 40)          # 2-rung falsifier pilot (coarse, fine)
INTERMEDIATE_RUNG = 30          # added ONLY if decay is seen
WALL_BUDGET_S = 2.5 * 3600.0

# Verdict band on the measured decay exponent p = log(ratio)/log(dx_f/dx_c) of
# the NORMALIZED per-cell sensitivity g_rel = |g|/J0.
DECAY_EXP_LO = 2.0
DECAY_EXP_HI = 4.0
DX3_EXPONENT = 3.0
# The raw metric decomposes as p_raw = p_J + 3 (objective scaling + volume);
# this is the tolerance on the residual (p_raw - p_J) matching the volume 3.
DECOMP_RESIDUAL_TOL = 0.6
COMPARATOR_MIN_RATIO = 100.0    # directional-FD-signal / J-repeat-noise gate
FLOAT32_ULP = 2.0 ** -23        # ~1.19e-7 relative granularity / FD floor


def build_sim(dx: float) -> Simulation:
    sim = Simulation(freq_max=FREQ_MAX, domain=(DOM_X, DOM_YZ, DOM_YZ),
                     boundary="cpml", cpml_layers=CPML, dx=dx)
    wf = GaussianPulse(f0=F_OBJ, bandwidth=0.9)
    sim.add_source((X_SRC, YZ_C, YZ_C), "ez", waveform=wf)
    sim.add_probe((X_PROBE, YZ_C, YZ_C), "ez")
    return sim


def design_box(grid: Grid, dx: float):
    """Contiguous cell-index box for the FIXED physical design region.

    Uses the same node-coordinate convention as the WP 3-1 slab mask
    (coord = (idx - pad_lo) * dx), then takes the inclusive [lo, hi] cell
    range on each axis -- the exact contiguous box the topology path paints.
    """
    nx, ny, nz = grid.shape
    pads = (grid.pad_x_lo, grid.pad_y_lo, grid.pad_z_lo)
    bounds = (DES_X, DES_Y, DES_Z)
    ns = (nx, ny, nz)
    box = []
    for axis in range(3):
        coord = (np.arange(ns[axis]) - pads[axis]) * dx
        lo, hi = bounds[axis]
        sel = np.flatnonzero((coord >= lo) & (coord < hi))
        if sel.size == 0:
            raise ValueError(f"design region empty on axis {axis} at dx={dx}")
        box.append((int(sel[0]), int(sel[-1])))
    return box  # [(si, ei), (sj, ej), (sk, ek)]


def make_objective(sim: Simulation, grid: Grid, box, n_steps: int):
    (si, ei), (sj, ej), (sk, ek) = box
    base = jnp.ones(grid.shape, dtype=jnp.float32)

    def J(eps_design):
        eps = base.at[si:ei + 1, sj:ej + 1, sk:ek + 1].set(
            eps_design.astype(jnp.float32))
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
    box = design_box(grid, dx)
    (si, ei), (sj, ej), (sk, ek) = box
    design_shape = (ei - si + 1, ej - sj + 1, ek - sk + 1)
    J = make_objective(sim, grid, box, n_steps)
    return dx, sim, grid, n_steps, box, design_shape, J


def _percentiles(gabs: np.ndarray) -> dict:
    # Aggregate in float64 over the (exactly float32->float64 widened) values so
    # the committed stats are bit-reproducible from the committed JSON list; the
    # gradient VALUES stay float32 (the object of study), only the aggregation
    # is float64 (an even-length median averages two middle cells, which would
    # otherwise differ at the float32 rounding scale between producer and gate).
    gabs = np.asarray(gabs, dtype=np.float64)
    return {
        "min": float(np.min(gabs)),
        "p10": float(np.percentile(gabs, 10)),
        "p25": float(np.percentile(gabs, 25)),
        "median": float(np.percentile(gabs, 50)),
        "p75": float(np.percentile(gabs, 75)),
        "p90": float(np.percentile(gabs, 90)),
        "max": float(np.max(gabs)),
    }


def run_rung(n: int) -> dict:
    dx, sim, grid, n_steps, box, design_shape, J = _setup(n)
    issues = sim.preflight(strict=False)
    eps0 = jnp.full(design_shape, EPS_BLOCK, dtype=jnp.float32)
    h = H_REL * EPS_BLOCK

    # Objective repeat-noise (float32 determinism witness).
    t0 = time.time()
    j0a = float(J(eps0))
    t_fwd = time.time() - t0
    j0b = float(J(eps0))
    j_repeat_noise = abs(j0a - j0b)

    # Full per-cell gradient FIELD, measured TWICE for the repeat-noise floor.
    t0 = time.time()
    g1 = np.asarray(jax.grad(J)(eps0))
    t_grad = time.time() - t0
    g2 = np.asarray(jax.grad(J)(eps0))
    grad_repeat_noise = float(np.max(np.abs(g1 - g2)))

    g1abs = np.abs(g1).astype(np.float64).ravel()
    pct = _percentiles(g1abs)
    median_grad = pct["median"]
    max_grad = pct["max"]

    # Mesh-invariant relative sensitivity g_rel = |g| / J0 (the PRIMARY metric;
    # divides out the non-mesh-invariant objective magnitude, see the R2/R5
    # redesign note). d ln J / d eps_cell.
    grel = g1abs / abs(j0a)
    pct_rel = _percentiles(grel)
    median_grad_rel = pct_rel["median"]
    max_grad_rel = pct_rel["max"]

    # Directional FD cross-check: sum(g) must equal (J(+h) - J(-h)) / 2h.
    jp = float(J(eps0 + h))
    jm = float(J(eps0 - h))
    fd_dir = (jp - jm) / (2.0 * h)
    ad_dir = float(np.sum(g1))
    dir_rel_err = abs(ad_dir - fd_dir) / max(abs(fd_dir), 1e-30)
    fd_diff = abs(jp - jm)
    comp_ratio = (float("inf") if j_repeat_noise == 0.0
                  else fd_diff / j_repeat_noise)
    comparator_valid = (j_repeat_noise == 0.0) or (
        comp_ratio >= COMPARATOR_MIN_RATIO)

    # Fraction of design cells whose RAW |g| clears the MEASURED AD repeat-noise
    # floor. On deterministic CPU grad_repeat_noise == 0.0, so this is the
    # fraction with |g| > 0 (AD path -- no absolute floor to sink below).
    frac_above_noise = float(np.mean(g1abs > grad_repeat_noise))
    # Fraction whose mesh-invariant g_rel clears the float32 FD-detectability
    # floor (the floor a finite-difference topology method would hit).
    frac_rel_above_fd_floor = float(np.mean(grel > FLOAT32_ULP))

    return {
        "rung": f"lambda/{n}", "lambda_over_dx": n, "dx_m": dx,
        "grid_shape": list(grid.shape), "cells": int(np.prod(grid.shape)),
        "n_steps": n_steps, "num_periods": NUM_PERIODS,
        "design_shape": list(design_shape),
        "design_cells": int(np.prod(design_shape)),
        "dx3_m3": dx ** 3,
        "eps_block": EPS_BLOCK, "fd_step_abs": h,
        "J0": j0a, "J_repeat": j0b, "j_repeat_noise": j_repeat_noise,
        "per_cell_grad_abs": [float(x) for x in g1abs],
        "grad_abs_stats": pct,
        "grad_rel_stats": pct_rel,
        "median_grad": median_grad, "max_grad": max_grad,
        "median_grad_rel": median_grad_rel, "max_grad_rel": max_grad_rel,
        "grad_repeat_noise": grad_repeat_noise,
        "noise_floor": grad_repeat_noise,
        "fd_detect_floor": FLOAT32_ULP,
        "frac_above_noise": frac_above_noise,
        "frac_rel_above_fd_floor": frac_rel_above_fd_floor,
        "directional_ad": ad_dir, "directional_fd": fd_dir,
        "directional_rel_err": dir_rel_err,
        "fd_diff": fd_diff,
        "fd_signal_over_noise": (None if comp_ratio == float("inf")
                                 else comp_ratio),
        "comparator_valid": bool(comparator_valid),
        "wall_forward_s": t_fwd, "wall_grad_s": t_grad,
        "preflight_issues": list(issues),
    }


def _exponent(coarse_val: float, fine_val: float, dx_ratio: float) -> float:
    return math.log(fine_val / coarse_val) / math.log(dx_ratio)


def _falsifier(coarse: dict, fine: dict) -> dict:
    """Primary falsifier: NORMALIZED median g_rel ratio vs (dx_f/dx_c)^3."""
    dx_ratio = fine["dx_m"] / coarse["dx_m"]
    r_med = fine["median_grad_rel"] / coarse["median_grad_rel"]
    predicted = dx_ratio ** DX3_EXPONENT
    exponent = _exponent(coarse["median_grad_rel"], fine["median_grad_rel"],
                         dx_ratio)
    seen = DECAY_EXP_LO <= exponent <= DECAY_EXP_HI
    return {
        "metric": "median_grad_rel (|g|/J0, mesh-invariant)",
        "coarse_rung": coarse["lambda_over_dx"],
        "fine_rung": fine["lambda_over_dx"],
        "median_rel_coarse": coarse["median_grad_rel"],
        "median_rel_fine": fine["median_grad_rel"],
        "median_ratio": r_med,
        "dx_ratio": dx_ratio,
        "predicted_dx3_ratio": predicted,
        "measured_exponent": exponent,
        "decay_seen": bool(seen),
        "verdict": "decay-seen" if seen else "null-no-decay",
    }


def _decomposition(coarse: dict, fine: dict) -> dict:
    """R5 record: p_raw(|g|) = p_J(objective) + 3(volume). The residual
    p_raw - p_J must match the per-cell volume exponent 3."""
    dx_ratio = fine["dx_m"] / coarse["dx_m"]
    p_J = _exponent(coarse["J0"], fine["J0"], dx_ratio)
    p_raw = _exponent(coarse["median_grad"], fine["median_grad"], dx_ratio)
    p_rel = _exponent(coarse["median_grad_rel"], fine["median_grad_rel"],
                      dx_ratio)
    residual = p_raw - p_J
    return {
        "p_objective_J": p_J,
        "p_raw_median_grad": p_raw,
        "p_rel_median_grad": p_rel,
        "residual_raw_minus_J": residual,
        "volume_exponent_expected": DX3_EXPONENT,
        "residual_matches_volume": bool(
            abs(residual - DX3_EXPONENT) <= DECOMP_RESIDUAL_TOL),
    }


def _crossing(rungs: list, falsifier: dict) -> dict:
    """FD-detectability crossing on the NORMALIZED metric, plus the AD
    no-collapse statement.

    For AD: measured repeat-noise floor is 0.0 on deterministic CPU -> no
    crossing (recorded honestly). For a finite-difference method: median g_rel
    decays as dx^p toward the fixed FLOAT32_ULP floor; extrapolate the
    lambda/dx where it crosses. Clearly labeled extrapolated-only.
    """
    ad_floor_zero = all(r["grad_repeat_noise"] == 0.0 for r in rungs)
    out = {
        "ad_repeat_noise_floor": (0.0 if ad_floor_zero
                                  else max(r["grad_repeat_noise"]
                                           for r in rungs)),
        "ad_crossing_lambda_over_dx": None,
        "ad_note": (
            "AD gradient repeat-noise floor is 0.0 on deterministic CPU -> the "
            "analytic per-cell gradient never sinks below a fixed absolute "
            "floor here; float32 residual imprecision is RELATIVE (scale-"
            "invariant), so per-cell AD SNR does not degrade with dx. A finite "
            "AD crossing would require a nondeterministic reduction (GPU atomic "
            "scatter / mixed precision), NOT measured here."),
    }
    if not falsifier["decay_seen"]:
        out["fd_applicable"] = False
        out["fd_note"] = "no dx^3 decay -> no FD crossing to characterize"
        return out
    # Extrapolate the FD-detectability crossing from the finest rung's g_rel.
    finest = max(rungs, key=lambda r: r["lambda_over_dx"])
    p = falsifier["measured_exponent"]
    n_ref = finest["lambda_over_dx"]
    g_ref = finest["median_grad_rel"]
    # median g_rel(n) = g_ref * (n_ref / n)^p ; solve = FLOAT32_ULP.
    n_cross = n_ref * (g_ref / FLOAT32_ULP) ** (1.0 / p)
    out["fd_applicable"] = True
    out["fd_detect_floor"] = FLOAT32_ULP
    out["fd_ref_rung"] = n_ref
    out["fd_ref_median_grad_rel"] = g_ref
    out["fd_crossing_lambda_over_dx_extrapolated"] = n_cross
    out["fd_note"] = (
        f"EXTRAPOLATED (not measured beyond lambda/{n_ref}): a central-FD "
        f"topology method would lose per-cell detectability (median g_rel < "
        f"float32 ULP {FLOAT32_ULP:.2e}) near ~lambda/{n_cross:.0f}, assuming "
        f"the measured dx^{p:.2f} decay continues. AD is unaffected (see "
        f"ad_note).")
    return out


def main():
    t_start = time.time()
    print("=" * 78)
    print("WP 3-2 per-cell gradient-SNR ladder (smooth eps=4 block, float32, "
          "CPU)")
    print(f"freq_max = {FREQ_MAX/1e9:.1f} GHz  f_obj = {F_OBJ/1e9:.1f} GHz  "
          f"num_periods = {int(NUM_PERIODS)}  eps_block = {EPS_BLOCK}")
    print("=" * 78)

    print("\n[step 1] falsifier pilot (2 rungs: coarse -> fine)")
    print(f"  {'rung':>10} {'l/dx':>5} {'shape':>16} {'design':>7} "
          f"{'median|g|':>12} {'J0':>12} {'med g_rel':>12} {'dir_relerr':>10}")
    rungs = []
    for n in PILOT_RUNGS:
        r = run_rung(n)
        rungs.append(r)
        print(f"  {r['rung']:>10} {n:5d} {str(tuple(r['grid_shape'])):>16} "
              f"{r['design_cells']:7d} {r['median_grad']:12.4e} "
              f"{r['J0']:12.4e} {r['median_grad_rel']:12.4e} "
              f"{r['directional_rel_err']:10.2e}  "
              f"(fwd {r['wall_forward_s']:.1f}s grad {r['wall_grad_s']:.1f}s)")

    fals = _falsifier(rungs[0], rungs[-1])
    decomp = _decomposition(rungs[0], rungs[-1])
    print(f"\n[step 2] NORMALIZED falsifier: median g_rel ratio = "
          f"{fals['median_ratio']:.4e} (predicted dx^3 = "
          f"{fals['predicted_dx3_ratio']:.4e}), exponent p = "
          f"{fals['measured_exponent']:.2f} -> VERDICT = {fals['verdict']}")
    print(f"          decomposition: raw|g| p={decomp['p_raw_median_grad']:.2f} "
          f"= objective p_J={decomp['p_objective_J']:.2f} + volume "
          f"(residual {decomp['residual_raw_minus_J']:.2f}, expect 3.0; "
          f"matches={decomp['residual_matches_volume']})")

    if fals["decay_seen"]:
        elapsed = time.time() - t_start
        if elapsed < WALL_BUDGET_S:
            print(f"\n[step 3] decay seen -> adding lambda/{INTERMEDIATE_RUNG} "
                  f"+ characterizing the FD-detectability crossing")
            r_mid = run_rung(INTERMEDIATE_RUNG)
            rungs.append(r_mid)
            print(f"  {r_mid['rung']:>10} {INTERMEDIATE_RUNG:5d} "
                  f"{str(tuple(r_mid['grid_shape'])):>16} "
                  f"{r_mid['design_cells']:7d} {r_mid['median_grad']:12.4e} "
                  f"{r_mid['J0']:12.4e} {r_mid['median_grad_rel']:12.4e} "
                  f"{r_mid['directional_rel_err']:10.2e}")
    else:
        print("\n[step 3] NULL result (no dx^3 decay) -> stopping per the "
              "falsifier protocol; not adding rungs to force a trend")

    rungs.sort(key=lambda r: r["lambda_over_dx"])
    crossing = _crossing(rungs, fals)
    if crossing.get("fd_applicable"):
        print(f"          {crossing['fd_note']}")
    print(f"          {crossing['ad_note']}")

    out = {
        "meta": {
            "jax_version": jax.__version__, "rfx_version": _rfx_version(),
            "work_package": "3-2",
            "witness": ("smooth dielectric block (eps_r=4), per-cell "
                        "permittivity DoF over a fixed physical design region"),
            "freq_max_hz": FREQ_MAX, "f_obj_hz": F_OBJ,
            "lambda_m": LAMBDA, "eps_block": EPS_BLOCK, "fd_rel_step": H_REL,
            "num_periods": NUM_PERIODS,
            "domain_m": [DOM_X, DOM_YZ, DOM_YZ], "cpml_layers": CPML,
            "design_region_m": {"x": list(DES_X), "y": list(DES_Y),
                                "z": list(DES_Z)},
            "precision": "float32",
            "primary_metric": "median_grad_rel (|g|/J0, mesh-invariant)",
            "decay_exp_lo": DECAY_EXP_LO, "decay_exp_hi": DECAY_EXP_HI,
            "dx3_exponent": DX3_EXPONENT,
            "decomp_residual_tol": DECOMP_RESIDUAL_TOL,
            "comparator_min_ratio": COMPARATOR_MIN_RATIO,
            "float32_ulp": FLOAT32_ULP,
            "verdict": fals["verdict"],
            "total_wall_s": time.time() - t_start,
        },
        "falsifier_normalized": fals,
        "decomposition": decomp,
        "crossing": crossing,
        "rungs": rungs,
    }

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    fixture = os.path.join(repo, "tests", "fixtures", "gradient_dx_ladder",
                           "per_cell_snr.json")
    os.makedirs(os.path.dirname(fixture), exist_ok=True)
    with open(fixture, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {fixture}")
    print(f"VERDICT = {fals['verdict']}  "
          f"total wall time: {out['meta']['total_wall_s']:.1f} s")


if __name__ == "__main__":
    main()
