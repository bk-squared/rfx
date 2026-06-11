#!/usr/bin/env python3
"""Experiment 10: does doubling rfx resolution (dx 1 mm → 0.5 mm) close
the Fabry-Perot peak |S21| deficit?

After β_d bug in analytic was fixed, rfx shows ~4-5 % |S21| deficit at
the Fabry-Perot transmission peak (~11.5 GHz) vs analytic. CPML layer
doubling (20→40) moved it only 0.8 %; subpixel smoothing moved it
0.5 %. The remaining leverage is resolution.

Meep's reference at r3 uses dx=333 μm — three times finer than rfx's
1 mm. Yee numerical dispersion scales O((β·dx)²), which at 10 GHz
β≈200 is ~4 % at dx=1 mm. If the |S21| deficit is dispersion at the
slab discontinuity (wave scattering into higher modes absorbed by
CPML), halving dx should quarter the deficit.

Run:
    python scripts/slab_resolution_sweep_magnitude.py
Compute time: ~3 min (2× run, doubled-resolution 2nd run ~8×).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "0")

import numpy as np


def _load_cv11():
    cv11_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "examples", "crossval", "11_waveguide_port_wr90.py",
    )
    spec = importlib.util.spec_from_file_location("cv11", cv11_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    cv11 = _load_cv11()
    cv11.CPML_LAYERS = 20

    eps_r, slab_L = 2.0, 0.010
    results = {}
    for dx in (0.001, 0.0005):
        # Scale CPML layers so physical CPML thickness stays ~20 mm.
        cv11.DX_M = dx
        cv11.CPML_LAYERS = int(round(0.020 / dx))
        print(
            f"\n=== rfx slab run with DX = {dx*1e3:.2f} mm, "
            f"CPML_LAYERS = {cv11.CPML_LAYERS} ===", flush=True)
        t0 = time.time()
        f_hz, _s11, s21 = cv11.run_rfx_slab(eps_r, slab_L)
        elapsed = time.time() - t0
        print(f"   run time: {elapsed:.1f}s", flush=True)
        results[dx] = (f_hz, s21, elapsed)

    _, s21_airy = cv11.analytic_slab_s(results[0.001][0], eps_r, slab_L)

    print()
    print(" f/GHz  |S21|_dx1mm  |S21|_dx0.5mm  |S21|_analytic   gap_1mm    gap_0.5mm")
    print("-" * 90)
    for i in range(len(results[0.001][0])):
        f = results[0.001][0][i] / 1e9
        a1 = abs(results[0.001][1][i])
        a05 = abs(results[0.0005][1][i])
        ana = abs(s21_airy[i])
        g1 = (a1 - ana) * 100.0
        g05 = (a05 - ana) * 100.0
        print(
            f" {f:5.2f}    {a1:.4f}       {a05:.4f}         {ana:.4f}      "
            f"{g1:+6.2f} %   {g05:+6.2f} %"
        )

    rms_1 = float(np.sqrt(np.mean(
        (np.abs(results[0.001][1]) - np.abs(s21_airy)) ** 2))) * 100.0
    rms_05 = float(np.sqrt(np.mean(
        (np.abs(results[0.0005][1]) - np.abs(s21_airy)) ** 2))) * 100.0
    print()
    print(f"RMS |S21| error vs analytic:")
    print(f"  dx = 1.0 mm:  {rms_1:.2f} %")
    print(f"  dx = 0.5 mm:  {rms_05:.2f} %")
    print(f"  wall clock: {results[0.001][2]:.1f}s → {results[0.0005][2]:.1f}s")
    print()
    if rms_05 < rms_1 * 0.5:
        print("Verdict: Yee numerical dispersion (resolution) IS the dominant")
        print("         cause. Halving dx more than halves the |S21| error —")
        print("         consistent with O((β·dx)²) dispersion scaling.")
    elif rms_05 < rms_1 * 0.85:
        print("Verdict: Resolution contributes, but not the whole story.")
    else:
        print("Verdict: Resolution is NOT the cause — gap persists at finer dx.")
        print("         Candidate: mode projection at slab interface,")
        print("         or higher-mode coupling not captured by V/I extraction.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
