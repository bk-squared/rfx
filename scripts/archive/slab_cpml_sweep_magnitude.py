#!/usr/bin/env python3
"""Experiment 9: is the rfx slab |S21| Fabry-Perot peak deficit CPML absorption?

With crossval/11's analytic_slab_s fixed (β_d now uses geometric kc,
not kc/sqrt(εr)), the phase residual between rfx and analytic is
RMS 0.27° — physics-exact. But |S21| shows a frequency-growing
deficit:

    8.2 GHz:   |S21|_rfx 0.8431  vs analytic 0.8393   (+0.4 %)
    11.6 GHz:  |S21|_rfx 0.9509  vs analytic 1.0000   (−4.9 %, peak)

The known rfx CPML guided-mode reflection profile is:
    10 layers → 11.7 %,  20 → 4.2 %,  40 → 1.8 %.
Two-run normalisation partially cancels the ref-run CPML leak, but
any residual asymmetry shows up as a |S21| deficit.

Test: rerun the slab case with CPML_LAYERS bumped 20 → 40 and see if
the Fabry-Perot peak approaches unity. If yes, the deficit is CPML
absorption. If no, it's something intrinsic (Yee dispersion at the
slab discontinuity, mode projection at the interface, etc.).

Run:
    python scripts/slab_cpml_sweep_magnitude.py
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

    eps_r, slab_L = 2.0, 0.010
    results = {}
    for n_cpml in (20, 40):
        cv11.CPML_LAYERS = n_cpml
        print(f"\n=== rfx slab run with CPML_LAYERS = {n_cpml} ===", flush=True)
        t0 = time.time()
        f_hz, _s11, s21 = cv11.run_rfx_slab(eps_r, slab_L)
        print(f"   run time: {time.time()-t0:.1f}s", flush=True)
        results[n_cpml] = (f_hz, s21)

    _, s21_airy = cv11.analytic_slab_s(results[20][0], eps_r, slab_L)

    print()
    print(" f/GHz   |S21|_cpml20  |S21|_cpml40   |S21|_analytic   gap@20      gap@40")
    print("-" * 92)
    for i in range(len(results[20][0])):
        f = results[20][0][i] / 1e9
        a20 = abs(results[20][1][i])
        a40 = abs(results[40][1][i])
        ana = abs(s21_airy[i])
        g20 = (a20 - ana) * 100.0
        g40 = (a40 - ana) * 100.0
        print(
            f" {f:5.2f}     {a20:.4f}        {a40:.4f}         {ana:.4f}       "
            f"{g20:+6.2f} %   {g40:+6.2f} %"
        )

    max_gap_20 = float(np.max(np.abs(
        np.abs(results[20][1]) - np.abs(s21_airy))))
    max_gap_40 = float(np.max(np.abs(
        np.abs(results[40][1]) - np.abs(s21_airy))))
    print()
    print(f"Max |S21| gap vs analytic:")
    print(f"  CPML 20 layers:  {max_gap_20 * 100:.2f} %")
    print(f"  CPML 40 layers:  {max_gap_40 * 100:.2f} %")
    print()
    if max_gap_40 < max_gap_20 * 0.5:
        print("Verdict: CPML guided-mode absorption IS the dominant cause.")
        print("         Gap halved or more with 2× CPML layers.")
    elif max_gap_40 < max_gap_20 * 0.9:
        print("Verdict: CPML contributes some, but not the only source.")
    else:
        print("Verdict: CPML is NOT the dominant cause — gap unchanged.")
        print("         Candidate: Yee dispersion at slab discontinuity")
        print("         or mode projection at material interface.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
