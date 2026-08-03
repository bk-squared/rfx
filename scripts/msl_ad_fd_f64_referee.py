#!/usr/bin/env python3
"""Float64 finite-difference referee for the MSL AD gate (issue #527).

WHY THIS EXISTS
---------------
``test_msl_ad_fd_converged_tight`` compares reverse-mode AD against a central
finite difference whose two loss evaluations are computed in **float32**::

    alpha0  = jnp.float32(1.0)
    f_plus  = float(objective(jnp.float32(float(alpha0) + _FD_H)))
    f_minus = float(objective(jnp.float32(float(alpha0) - _FD_H)))
    g_fd    = (f_plus - f_minus) / (2.0 * _FD_H)

A float32 value near ``loss = 16.005951`` can only move in steps of one ULP,
``1.907349e-06``. At the gate's ``h = 1e-3`` the true difference is
``2h|g| = 8.4723e-06`` — **4.4 ULP wide**. The subtraction therefore returns an
integer number of quantisation steps, not a derivative.

Measured on the gate's own fixture (VESSL 369367250742, gpu-rtx4090), the span
stayed between 1 and 76 ULP across the whole sweep, while the FD value scattered
2017% and changed sign:

    h        g_fd            span (ULP)
    3.0e-04  +4.450480e-02       14
    5.0e-04  +1.907349e-03        1     <- the entire measurement is ONE ULP
    1.0e-03  -2.861023e-02       30     <- the gate's h
    2.0e-03  -3.337860e-03        7
    5.0e-03  -5.912781e-03       31
    1.0e-02  -4.959106e-03       52
    2.0e-02  -3.623962e-03       76

The sign flip that looked like a physics puzzle is just which side of a
representable value each evaluation landed on.

CAUTION for anyone reusing this reasoning: those spans are exact integers, and
an earlier draft of this docstring offered that integrality as the proof. It is
not evidence of anything — ANY two float32 values in the same binade differ by
an integer number of ULPs (200k random pairs, including deliberately
well-resolved ones 47186 ULP apart, all reproduce it). Integrality shows only
that the values were float32. The load-bearing fact is the MAGNITUDE of the
span. Do not carry the integer-multiple test forward as a diagnostic; it fires
on healthy comparators too.

No h repairs this: 1% quantisation resolution needs ``2h|g| > 100 ULP``, i.e.
``h > 0.0225`` — a 2.3% perturbation of ``alpha``, where ``h^2`` truncation
dominates instead. The usable window has closed, so Richardson extrapolation
over h would extrapolate over quantisation noise.

This did not used to bite. Before PR #516 the gradient was ``2.1143e-01``; the
extractor fix moved ``|S11|`` from 0.2233 to ~0.05, shrinking it 50x to
``4.2362e-03`` while the loss magnitude — and therefore the ULP — stayed put.
The signal fell off the resolvable end of float32:

    2h|g| at the gate's h:   pre-#516 ~222 ULP    post-#516 4.4 ULP

Independently corroborated by a number recorded before that change: the #477
work logged a "27% NON-monotone FD scatter" on this fixture; the same sweep now
reads 2017%, and 27% x 50 ~ 1350% is the same order.

WHAT THIS SCRIPT DOES
---------------------
Evaluates the loss under ``jax.experimental.enable_x64()`` (a scoped context —
module-level ``jax.config.update('jax_enable_x64', True)`` is forbidden here:
it is process-global, flips at pytest collection, and reds every same-process
pytest-split shard). ``rfx/probes/probes.py`` already keys its DFT accumulator
dtype off ``jax.config.x64_enabled``, so x64 reaches the S-matrix: a plumbing
smoke at num_periods=3 returned ``S.dtype=complex128``, ``loss.dtype=float64``,
ULP ``8.88e-16`` — and an f32/f64 loss pair agreeing to 7 digits (7.94825983 vs
7.94825835), confirming the f32 *forward* is sound and only the *subtraction*
was unresolvable.

Reports, per h, the ULP headroom alongside every number, so the comparator's own
validity is visible in the same table as the result it is used to judge.

PRE-DECLARED READS
------------------
  R1 f64 FD is h-stable (spread <= a few %) AND rel_err vs AD <= 0.10
     -> the AD is correct; #527 is purely a comparator defect. Switch the gate
        to this referee, leave the 0.10 bound untouched.
  R2 f64 FD is h-stable but rel_err stays well above 0.10
     -> a genuine AD systematic, now measurable for the first time since #516.
        #527 reopens on its original framing with a trustworthy instrument.
  R3 f64 FD is still h-unstable
     -> the residual scatter is not float32 quantisation; suspect the objective
        itself (a non-smooth dependence of S on alpha, e.g. a mesh/rasterisation
        threshold crossing as eps changes) and investigate that before the AD.

Report-only. Modifies no gate.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np  # noqa: E402

DEFAULT_H = (3.0e-4, 1.0e-3, 2.0e-3, 5.0e-3, 1.0e-2)


def _closest_divisor(n: int, target: int) -> int:
    divs = [d for d in range(1, n + 1) if n % d == 0]
    return min(divs, key=lambda d: (abs(d - target), d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-periods", type=int, default=None,
                    help="default: the gate's own _NUM_PERIODS")
    ap.add_argument("--n-freqs", type=int, default=None)
    ap.add_argument("--h", type=float, nargs="*", default=list(DEFAULT_H))
    ap.add_argument("--f64-ad", action="store_true",
                    help="also compute the AD gradient in f64 (slower; the "
                         "shipped gate's AD is f32, so f32 is the default)")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from jax.experimental import enable_x64

    import rfx
    print(f"rfx     : {rfx.__file__}")
    print(f"jax     : {jax.__version__}   devices: {jax.devices()}")
    if not str(Path(rfx.__file__).resolve()).startswith(str(REPO)):
        print(f"FATAL: imported rfx is not this checkout ({REPO})")
        return 2

    from test_msl_ad_fd_converged import (  # noqa: E402
        _FD_H,
        _N_FREQS,
        _NUM_PERIODS,
        _REL_ERR_THRESHOLD,
        _build_msl_sim,
    )

    num_periods = args.num_periods if args.num_periods is not None else _NUM_PERIODS
    n_freqs = args.n_freqs if args.n_freqs is not None else _N_FREQS
    print(f"fixture : num_periods={num_periods} n_freqs={n_freqs} "
          f"(gate ships {_NUM_PERIODS}/{_N_FREQS}, h={_FD_H}, "
          f"threshold={_REL_ERR_THRESHOLD})")

    # ---- AD gradient: f32 by default, matching what the gate ships ----------
    sim32 = _build_msl_sim()
    grid = sim32._build_grid()
    n_steps = int(grid.num_timesteps(num_periods=num_periods))
    cseg = _closest_divisor(n_steps, int(np.sqrt(n_steps)))
    assert n_steps % cseg == 0
    print(f"grid    : {grid.shape}  n_steps={n_steps}  checkpoint_segments={cseg}")

    def _objective(sim, dtype):
        eps_base = jnp.ones(grid.shape, dtype=dtype)

        def obj(alpha):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = sim.compute_msl_s_matrix(
                    n_freqs=n_freqs, num_periods=num_periods,
                    eps_override=eps_base * alpha, checkpoint_segments=cseg,
                )
            return jnp.real(jnp.sum(jnp.abs(r.S) ** 2))
        return obj

    print("\n--- AD (float32, as shipped) ---", flush=True)
    t0 = time.perf_counter()
    loss32, g32 = jax.value_and_grad(_objective(sim32, jnp.float32))(jnp.float32(1.0))
    g_ad32, loss32 = float(g32), float(loss32)
    ulp32 = float(np.spacing(np.float32(loss32)))
    print(f"  loss = {loss32:.8f}   g_ad = {g_ad32:.6e}   ({time.perf_counter()-t0:.1f}s)")
    print(f"  float32 ULP of this loss = {ulp32:.4e}")
    print(f"  the SHIPPED f32 FD at h={_FD_H} would span "
          f"{2*_FD_H*abs(g_ad32)/ulp32:.1f} ULP  <- why it cannot resolve 10%")

    g_ad = g_ad32
    with enable_x64():
        if args.f64_ad:
            print("\n--- AD (float64) ---", flush=True)
            t0 = time.perf_counter()
            sim64 = _build_msl_sim()
            l64, g64 = jax.value_and_grad(
                _objective(sim64, jnp.float64))(jnp.float64(1.0))
            g_ad = float(g64)
            print(f"  loss = {float(l64):.14f}   g_ad = {g_ad:.10e}   "
                  f"({time.perf_counter()-t0:.1f}s)")
            print(f"  f32-vs-f64 AD relative difference: "
                  f"{abs(g_ad32-g_ad)/max(abs(g_ad),1e-300):.3e}")

        # ---- f64 central differences ---------------------------------------
        print("\n--- central FD, float64 loss ---", flush=True)
        sim_fd = _build_msl_sim()
        obj64 = _objective(sim_fd, jnp.float64)
        print(f"  {'h':>9} {'g_fd':>18} {'rel_err':>10} {'signal (ULP)':>14} {'s':>6}")
        rows = []
        for h in sorted(args.h):
            t0 = time.perf_counter()
            fp = float(obj64(jnp.float64(1.0 + h)))
            fm = float(obj64(jnp.float64(1.0 - h)))
            g_fd = (fp - fm) / (2.0 * h)
            ulp64 = float(np.spacing(np.float64(0.5 * (fp + fm))))
            rel = abs(g_ad - g_fd) / (abs(g_fd) + 1e-300)
            rows.append((h, g_fd, rel, abs(fp - fm) / ulp64))
            print(f"  {h:9.1e} {g_fd:18.10e} {rel:10.4f} "
                  f"{abs(fp-fm)/ulp64:14.3g} {time.perf_counter()-t0:6.1f}",
                  flush=True)

    # ---- verdict ------------------------------------------------------------
    gs = np.array([r[1] for r in rows])
    rels = np.array([r[2] for r in rows])
    spread = float(np.ptp(gs) / max(abs(np.mean(gs)), 1e-300))
    best_i = int(np.argmin(rels))
    print(f"\n  g_ad                 : {g_ad:.10e}")
    print(f"  f64 FD spread over h : {spread*100:.3f}%")
    print(f"  best rel_err         : {rels[best_i]:.5f} at h = {rows[best_i][0]:.1e}")
    print(f"  rel_err at gate's h  : "
          f"{dict((r[0], r[2]) for r in rows).get(_FD_H, float('nan')):.5f}")

    stable = spread <= 0.05
    if stable and rels.min() <= _REL_ERR_THRESHOLD:
        print("\nVERDICT R1: f64 FD is h-stable and agrees with AD inside the gate's "
              "bound -> the AD is correct; #527 is a comparator defect. Switch the "
              "gate to this referee and leave the 0.10 threshold alone.")
    elif stable:
        print("\nVERDICT R2: f64 FD is h-stable but disagrees with AD beyond the "
              "gate's bound -> a genuine AD systematic, measurable for the first "
              "time since #516. #527 reopens on its original framing.")
    else:
        print("\nVERDICT R3: f64 FD is STILL h-unstable -> the scatter is not float32 "
              "quantisation. Suspect a non-smooth dependence of S on alpha (a "
              "mesh/rasterisation threshold crossing) before suspecting the AD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
