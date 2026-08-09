#!/usr/bin/env python3
"""Owner-platform measurement for the #530 objective replacement.

WHY THIS SCRIPT EXISTS
-----------------------
``test_msl_ad_fd_converged_tight`` (issue #530) now differentiates band-mean
``|S21|**2`` (``tests._msl_ad_objective.msl_band_mean_s21_sq``) instead of
``sum_ij|S_ij|**2``. The gate's own rule (repo CLAUDE.md, "no silent gate
change") requires the new threshold to come from a MEASURED envelope on the
gate's OWNER platform (gpu-rtx4090 via VESSL — issue #477 lesson: a CPU
execution of this test carries an f32/f64 evaluation-noise envelope that does
not represent the platform the gate actually runs on).

This script is a STANDALONE measurement harness, deliberately NOT run via
``scripts/vessl_msl_ad_gate.yaml`` (which executes against the MOUNTED
primary checkout's ``main`` — a personal-workspace NFS volume mount, not a
git checkout the job controls). The objective-replacement work lives on a
branch that is not ``main`` yet, so this script is meant to be invoked from a
FRESH ``git clone`` of that branch inside the VESSL job (see
``scripts/vessl_msl_ad_band_mean_owner_measurement.yaml``), independent of
whatever the mounted checkout happens to have.

WHAT IT MEASURES
-----------------
1. **Baseline envelope**: AD (float32, as the gate ships) vs a central FD
   reference (float64, scoped ``enable_x64``) across an h-sweep, on the
   gate's own fixture (``tests.test_msl_ad_fd_converged._build_msl_sim``,
   ``_NUM_PERIODS=20``, ``_N_FREQS=8``) — reusing the SAME
   ``msl_band_mean_s21_sq`` reduction the gate itself calls, imported from
   ``tests._msl_ad_objective`` rather than re-derived here (the repeated
   drift defect this repo's ``tests/_gate_policy.py`` docstring documents
   for the envelope->gate multiplier; the same discipline applies to the
   objective itself). Also reports the FD reference's resolving power in
   ULPs of the loss (issue #527's comparator-validity check), reusing
   ``_fd_ulp_span`` from the gate's own test module.

2. **Resolving-power falsifier** (``--plant-defect``, on by default): plants
   an issue-#483-CLASS defect — "the fixture sampled the override under FD
   while the tape saw it frozen" — by building the AD-side objective with
   ``eps_override`` FROZEN at a concrete ``alpha0`` computed before tracing,
   so the traced ``alpha`` argument never enters the AD computation at all.
   The FD reference is untouched (always the correct, alpha-dependent
   function). ``jax.grad`` of a function that does not use its argument
   returns exactly 0.0, so this must read ``g_ad = 0.0`` and
   ``rel_err = 1.0`` — the gate must red on this with enormous margin
   (#529 discipline: prove the gate actually has resolving power, not only
   that it is well-calibrated on the happy path).

Report-only. Modifies no gate; the PR that adds this script hand-computes
the new threshold via ``tests._gate_policy.gate_from_envelope`` from this
script's printed baseline numbers.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np  # noqa: E402

DEFAULT_H = (3.0e-4, 1.0e-3, 2.0e-3, 5.0e-3, 1.0e-2)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
        ).strip()
    except Exception as exc:  # pragma: no cover - provenance-only
        return f"<unavailable: {exc}>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-periods", type=int, default=None,
                     help="default: the gate's own _NUM_PERIODS (20)")
    ap.add_argument("--n-freqs", type=int, default=None,
                     help="default: the gate's own _N_FREQS (8)")
    ap.add_argument("--h", type=float, nargs="*", default=list(DEFAULT_H))
    ap.add_argument("--no-plant-defect", action="store_true",
                     help="skip the #483-class falsifier (baseline only)")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from tests._x64_compat import enable_x64

    import rfx
    print(f"rfx        : {rfx.__file__}")
    print(f"git SHA    : {_git_sha()}")
    print(f"jax        : {jax.__version__}   devices: {jax.devices()}")
    if not str(Path(rfx.__file__).resolve()).startswith(str(REPO)):
        print(f"FATAL: imported rfx is not this checkout ({REPO})")
        return 2

    from test_msl_ad_fd_converged import (  # noqa: E402
        _FD_H,
        _N_FREQS,
        _NUM_PERIODS,
        _build_msl_sim,
        _closest_divisor,
        _fd_ulp_span,
    )
    from tests._msl_ad_objective import msl_band_mean_s21_sq  # noqa: E402

    num_periods = args.num_periods if args.num_periods is not None else _NUM_PERIODS
    n_freqs = args.n_freqs if args.n_freqs is not None else _N_FREQS
    plant_defect = not args.no_plant_defect
    print(f"fixture    : num_periods={num_periods} n_freqs={n_freqs} "
          f"(gate's h={_FD_H}); plant_defect={plant_defect}")

    sim32 = _build_msl_sim()
    grid = sim32._build_grid()
    n_steps = int(grid.num_timesteps(num_periods=num_periods))
    cseg = _closest_divisor(n_steps, int(np.sqrt(n_steps)))
    assert n_steps % cseg == 0
    print(f"grid       : {grid.shape}  n_steps={n_steps}  checkpoint_segments={cseg}")

    def _make_objective(sim, dtype, *, frozen_alpha=None):
        """Band-mean |S21|^2 objective (issue #530). If ``frozen_alpha`` is
        given, ``eps_override`` is built from that CONCRETE value instead of
        the traced ``alpha`` argument -- the #483-class defect: the argument
        never enters the trace, so jax.grad of this returns exactly 0.0.
        """
        eps_base = jnp.ones(grid.shape, dtype=dtype)

        def obj(alpha):
            eps_ov = (
                eps_base * jnp.asarray(frozen_alpha, dtype=dtype)
                if frozen_alpha is not None
                else eps_base * alpha
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = sim.compute_msl_s_matrix(
                    n_freqs=n_freqs, num_periods=num_periods,
                    eps_override=eps_ov, checkpoint_segments=cseg,
                )
            return msl_band_mean_s21_sq(r.S)
        return obj

    # ---- AD gradient(s): float32, as the gate ships -------------------------
    print("\n--- AD (float32, as shipped) ---", flush=True)
    t0 = time.perf_counter()
    loss32, g32 = jax.value_and_grad(_make_objective(sim32, jnp.float32))(jnp.float32(1.0))
    g_ad, loss32 = float(g32), float(loss32)
    ulp32 = float(np.spacing(np.float32(loss32)))
    print(f"  loss = {loss32:.8f}   g_ad = {g_ad:.6e}   ({time.perf_counter()-t0:.1f}s)")
    print(f"  float32 ULP of this loss = {ulp32:.4e}")

    g_ad_defect = None
    if plant_defect:
        print("\n--- AD (float32, PLANTED #483-CLASS DEFECT: eps_override frozen) ---",
              flush=True)
        sim32_defect = _build_msl_sim()
        t0 = time.perf_counter()
        loss32_d, g32_d = jax.value_and_grad(
            _make_objective(sim32_defect, jnp.float32, frozen_alpha=1.0)
        )(jnp.float32(1.0))
        g_ad_defect, loss32_d = float(g32_d), float(loss32_d)
        print(f"  loss = {loss32_d:.8f}   g_ad(defect) = {g_ad_defect:.6e}   "
              f"({time.perf_counter()-t0:.1f}s)")
        print("  (expect exactly 0.0 -- alpha never enters the traced computation)")

    # ---- central FD, float64 loss (always the CORRECT function) -------------
    print("\n--- central FD, float64 loss (correct function; unaffected by the "
          "planted defect above) ---", flush=True)
    with enable_x64():
        sim_fd = _build_msl_sim()
        obj64 = _make_objective(sim_fd, jnp.float64)
        print(f"  {'h':>9} {'g_fd':>18} {'rel_err (correct AD)':>22} "
              f"{'rel_err (defect AD)':>22} {'signal (ULP)':>14} {'s':>6}")
        rows = []
        for h in sorted(args.h):
            t0 = time.perf_counter()
            fp = float(obj64(jnp.float64(1.0 + h)))
            fm = float(obj64(jnp.float64(1.0 - h)))
            g_fd = (fp - fm) / (2.0 * h)
            rel = abs(g_ad - g_fd) / (abs(g_fd) + 1e-300)
            rel_d = (abs(g_ad_defect - g_fd) / (abs(g_fd) + 1e-300)
                     if g_ad_defect is not None else float("nan"))
            span = _fd_ulp_span(fp, fm, np.float64)
            rows.append((h, g_fd, rel, rel_d, span))
            print(f"  {h:9.1e} {g_fd:18.10e} {rel:22.4f} {rel_d:22.4f} "
                  f"{span:14.3g} {time.perf_counter()-t0:6.1f}", flush=True)

    gs = np.array([r[1] for r in rows])
    spread = float(np.ptp(gs) / max(abs(np.mean(gs)), 1e-300))
    gate_h_row = dict((r[0], r) for r in rows).get(_FD_H)

    print(f"\n  g_ad (correct)       : {g_ad:.10e}")
    if g_ad_defect is not None:
        print(f"  g_ad (planted defect): {g_ad_defect:.10e}")
    print(f"  f64 FD spread over h : {spread*100:.3f}%")
    if gate_h_row is not None:
        print(f"  rel_err at gate's h={_FD_H:g} (correct AD) : {gate_h_row[2]:.5f}")
        if g_ad_defect is not None:
            print(f"  rel_err at gate's h={_FD_H:g} (defect AD)  : {gate_h_row[3]:.5f}")
        print(f"  FD resolving power at gate's h              : {gate_h_row[4]:.3g} ULP")

    print("\n=== SUMMARY (for tests._gate_policy.gate_from_envelope) ===")
    print(f"  measured envelope (rel_err at gate's h, correct AD) = "
          f"{gate_h_row[2] if gate_h_row is not None else float('nan'):.4f}")
    if g_ad_defect is not None:
        defect_rel = gate_h_row[3] if gate_h_row is not None else float("nan")
        print(f"  falsifier rel_err (planted #483-class defect)       = {defect_rel:.4f}")
        print("  -> the gate must RED on this with margin over whatever "
              "threshold gate_from_envelope derives above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
