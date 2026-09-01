#!/usr/bin/env python3
"""F-M1a measurement: >=1e6-step lossless energy audit (paper Sec. V-A fixture).

Fixture (pre-declared): 60 x 40 mm PEC cavity, coarse dx=1mm dy=2mm, centered
40 x 20 mm fine island; arms r=4 (paper-exact) and r=5 (odd lane); vacuum,
f64, dt = 0.99 x fine CFL; modulated-Gaussian magnetic-current source
f0=3.75 GHz, HWHM 0.74 GHz, compact support.

Window: for all n > n_off: (E_n - E_ref)/E_ref <= +1e-8, E_ref = E_{n_off+1}.

Run:
  PYTHONPATH=<worktree> .venv/bin/python \
      validation/research/portgrid/m1_energy_audit.py [--steps N] [--ratios 4,5]
"""

from __future__ import annotations

import argparse
import json
import time

import jax

jax.config.update("jax_enable_x64", True)  # script entrypoint: process-global is intended

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--ratios", type=str, default="4,5")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    from validation.research.portgrid import sim2d

    results = {}
    fired = False
    for r in [int(t) for t in args.ratios.split(",")]:
        nx, ny = 60, 20
        spec = sim2d.TwoRegionSpec(
            nx=nx, ny=ny, dx=1e-3, dy=2e-3, i0=10, i1=50, j0=5, j1=15, r=r,
            dt=np.nan, src_ij=(3, 2), probe_ij=(55, 17),
        )
        spec.dt = 0.99 * sim2d.fine_cfl_dt(spec)
        wf = sim2d.gaussian_modulated(args.steps, spec.dt, 3.75e9, 0.74e9)
        n_off = int(np.max(np.nonzero(wf)[0])) + 1
        step, init, _ = sim2d.make_stepper(spec)
        run = jax.jit(lambda s, w, _step=step: sim2d.run_scan(_step, s, w))

        t0 = time.perf_counter()
        _, energies, probe = run(init(), wf)
        energies = np.asarray(energies)
        probe = np.asarray(probe)
        wall = time.perf_counter() - t0

        finite = bool(np.all(np.isfinite(energies)) and np.all(np.isfinite(probe)))
        e_ref = float(energies[n_off + 1])
        drift = (energies[n_off + 1:] - e_ref) / e_ref
        max_growth = float(np.max(drift))
        max_abs_drift = float(np.max(np.abs(drift)))
        window = 1e-8
        passed = finite and (max_growth <= window)
        fired |= not passed
        results[f"r={r}"] = {
            "steps": args.steps,
            "dt_s": spec.dt,
            "n_off": n_off,
            "E_ref_J_per_m": e_ref,
            "max_rel_growth_after_off": max_growth,
            "max_abs_rel_drift_after_off": max_abs_drift,
            "probe_abs_max_last_10pct": float(np.max(np.abs(probe[-args.steps // 10:]))),
            "probe_abs_max_first_10pct_after_off": float(
                np.max(np.abs(probe[n_off:n_off + args.steps // 10]))),
            "all_finite": finite,
            "window": window,
            "F_M1a_fired": not passed,
            "wall_s": wall,
        }
        print(f"[r={r}] steps={args.steps} dt={spec.dt:.4e}s n_off={n_off} "
              f"wall={wall:.1f}s finite={finite}")
        print(f"        E_ref={e_ref:.6e} max_rel_growth={max_growth:+.3e} "
              f"max_abs_drift={max_abs_drift:.3e} window=+{window:.0e} "
              f"-> {'FIRE' if not passed else 'PASS'}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
    return 1 if fired else 0


if __name__ == "__main__":
    raise SystemExit(main())
