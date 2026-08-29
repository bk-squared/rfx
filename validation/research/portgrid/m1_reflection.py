#!/usr/bin/env python3
"""F-M1b measurement: interface-only reflection spectrum vs the paper's class.

Fixture (pre-declared, geometry re-derived for time gating instead of PML):
parallel-plate waveguide, PEC at y = 0 and y = 40 mm, vacuum, coarse
dx = dy = 1 mm, domain 700 mm x 40 mm, PEC x-ends.  Magnetic-current line
source spanning all y at x = 300 mm; probe at (360 mm, 20 mm); vacuum fine
island of 20 mm x 20 mm (coarse cells [440,460) x [10,30)).  Reference run:
identical, uniform coarse grid, SAME dt.  Reflected = probe_island -
probe_ref; gates chosen from ray arrival times (derivation in the
pre-declaration + below) so that only first-incidence island scattering is
in the reflection gate.

Gate derivation (c = 299792458 m/s, distances from probe at x=360mm):
  direct incident:      60 mm  -> 0.200 ns, pulse support +-0.113 ns
  island front return: 140+80 = 220 mm -> 0.734 ns
  island back-face:     260 mm -> 0.867 ns (+ tail < 1.10 ns)
  second incidence (backward wave via left wall, reflected at island):
                        820 mm -> 2.74 ns   (far outside gate)
  right-end transit difference: 740 mm -> 2.47 ns (outside gate)
  => incident gate [0, 0.45] ns; reflection gate [0.50, 1.35] ns.

|S11|(f) = |FFT(gated refl)| / |FFT(gated incident)|.

Windows (frozen): for every r in {2,3,4,5,6}:
  max over [2, 20] GHz <= -45 dB  AND  max over [2, 30] GHz <= -35 dB.

Run:
  PYTHONPATH=<worktree> .venv/bin/python \
      validation/research/portgrid/m1_reflection.py
"""

from __future__ import annotations

import argparse
import json

import jax

jax.config.update("jax_enable_x64", True)  # script entrypoint

import numpy as np

C0 = 299792458.0


def s11_for_ratio(sim2d, r: int, island=(440, 460, 10, 30), scale: int = 1):
    """scale = integer refinement of the WHOLE coarse mesh at fixed physical
    geometry (dx-scaling diagnostic; scale=2 is the Delta=0.5mm arm recorded
    in the phase-1 results note).  All indices scale with it; the physical
    fixture, gates, and windows are unchanged."""
    nx, ny = 700 * scale, 40 * scale
    dx = dy = 1e-3 / scale
    i0, i1, j0, j1 = (v * scale for v in island)
    spec = sim2d.TwoRegionSpec(
        nx=nx, ny=ny, dx=dx, dy=dy, i0=i0, i1=i1, j0=j0, j1=j1, r=r,
        dt=np.nan, probe_ij=(360 * scale, 20 * scale),
    )
    spec.dt = 0.99 * sim2d.fine_cfl_dt(spec)
    src = np.zeros((nx, ny))
    src[300 * scale, :] = 1.0
    spec.src_mask = src

    t_total = 1.40e-9
    n_steps = int(np.ceil(t_total / spec.dt))
    wf = sim2d.gaussian_modulated(n_steps, spec.dt, 16e9, 10e9)

    step, init, _ = sim2d.make_stepper(spec)
    _, _, p_island = jax.jit(lambda s, w: sim2d.run_scan(step, s, w))(init(), wf)

    ustep, uinit = sim2d.make_uniform_stepper(nx, ny, dx, dy, spec.dt, src, spec.probe_ij)
    _, _, p_ref = jax.jit(lambda s, w: sim2d.run_scan(ustep, s, w))(uinit(), wf)

    p_island = np.asarray(p_island)
    p_ref = np.asarray(p_ref)
    t = np.arange(n_steps) * spec.dt

    inc = np.where(t <= 0.45e-9, p_ref, 0.0)
    refl = np.where((t >= 0.50e-9) & (t <= 1.35e-9), p_island - p_ref, 0.0)

    nfft = 1 << int(np.ceil(np.log2(4 * n_steps)))
    f = np.fft.rfftfreq(nfft, spec.dt)
    s11 = np.abs(np.fft.rfft(refl, nfft)) / np.maximum(
        np.abs(np.fft.rfft(inc, nfft)), 1e-300)
    return f, s11, spec.dt, n_steps, float(np.max(np.abs(p_island - p_ref)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", type=str, default="2,3,4,5,6")
    ap.add_argument("--island", type=str, default="440,460,10,30",
                    help="i0,i1,j0,j1 in coarse cells (diagnostic arms only; "
                         "the pre-declared F-M1b fixture is the default)")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--scale", type=int, default=1,
                    help="integer refinement of the whole coarse mesh at fixed "
                         "physical geometry (dx-scaling diagnostic; reviewer nb)")
    args = ap.parse_args()

    from validation.research.portgrid import sim2d

    island = tuple(int(t) for t in args.island.split(","))
    results = {"island_coarse_cells": list(island), "mesh_scale": args.scale}
    fired = False
    for r in [int(t) for t in args.ratios.split(",")]:
        f, s11, dt, n_steps, dmax = s11_for_ratio(sim2d, r, island, args.scale)
        band20 = (f >= 2e9) & (f <= 20e9)
        band30 = (f >= 2e9) & (f <= 30e9)
        db = 20.0 * np.log10(np.maximum(s11, 1e-300))
        max20 = float(np.max(db[band20]))
        max30 = float(np.max(db[band30]))
        passed = (max20 <= -45.0) and (max30 <= -35.0)
        fired |= not passed
        results[f"r={r}"] = {
            "dt_s": dt, "n_steps": n_steps,
            "max_s11_db_2_20GHz": max20,
            "max_s11_db_2_30GHz": max30,
            "windows_db": {"2-20GHz": -45.0, "2-30GHz": -35.0},
            "F_M1b_fired": not passed,
            "max_abs_diff_timeseries": dmax,
        }
        print(f"[r={r}] max|S11| 2-20GHz = {max20:7.2f} dB (win -45) | "
              f"2-30GHz = {max30:7.2f} dB (win -35) -> "
              f"{'FIRE' if not passed else 'PASS'}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
    return 1 if fired else 0


if __name__ == "__main__":
    raise SystemExit(main())
