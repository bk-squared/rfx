"""CPML reflectivity sweep — find minimum layers for -40 dB.

Uses the proven reference-comparison method from test_pml_reflectivity.py:
large PEC domain (reflections don't reach probe) vs smaller CPML domain.

Sweep: 7 layer counts x 4 frequencies x 2 kappa_max = 56 runs.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rfx.grid import Grid
from rfx.core.yee import init_state, init_materials, update_e, update_h
from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
from rfx.boundaries.pec import apply_pec
from rfx.sources.sources import GaussianPulse

LAYERS = [4, 6, 8, 10, 12, 16, 20]
FREQS = [1e9, 2.4e9, 5e9, 10e9]
KAPPAS = [1.0, 5.0]
TARGET_DB = -40.0
N_STEPS = 400

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def measure_reflectivity(f0, n_cpml_layers, kappa_max):
    """Measure CPML reflectivity in dB."""
    freq_max = f0 * 2.5
    pulse = GaussianPulse(f0=f0, bandwidth=0.5)

    # Reference: large PEC domain — reflections don't reach probe in N_STEPS
    grid_ref = Grid(freq_max=freq_max, domain=(0.20, 0.20, 0.20), cpml_layers=0)
    state_ref = init_state(grid_ref.shape)
    mats_ref = init_materials(grid_ref.shape)
    cx, cy, cz = grid_ref.nx // 2, grid_ref.ny // 2, grid_ref.nz // 2
    probe_ref = (cx + 3, cy, cz)

    ts_ref = np.zeros(N_STEPS)
    for n in range(N_STEPS):
        state_ref = update_h(state_ref, mats_ref, grid_ref.dt, grid_ref.dx)
        state_ref = update_e(state_ref, mats_ref, grid_ref.dt, grid_ref.dx)
        state_ref = apply_pec(state_ref)
        ez = state_ref.ez.at[cx, cy, cz].add(pulse(n * grid_ref.dt))
        state_ref = state_ref._replace(ez=ez)
        ts_ref[n] = float(state_ref.ez[probe_ref])

    # CPML domain: smaller, same dx
    grid_cpml = Grid(freq_max=freq_max, domain=(0.06, 0.06, 0.06),
                     cpml_layers=n_cpml_layers, kappa_max=kappa_max)
    state_cpml = init_state(grid_cpml.shape)
    mats_cpml = init_materials(grid_cpml.shape)
    cpml_params, cpml_state = init_cpml(grid_cpml)
    cx2, cy2, cz2 = grid_cpml.nx // 2, grid_cpml.ny // 2, grid_cpml.nz // 2
    probe_cpml = (cx2 + 3, cy2, cz2)

    ts_cpml = np.zeros(N_STEPS)
    for n in range(N_STEPS):
        state_cpml = update_h(state_cpml, mats_cpml, grid_cpml.dt, grid_cpml.dx)
        state_cpml, cpml_state = apply_cpml_h(
            state_cpml, cpml_params, cpml_state, grid_cpml)
        state_cpml = update_e(state_cpml, mats_cpml, grid_cpml.dt, grid_cpml.dx)
        state_cpml, cpml_state = apply_cpml_e(
            state_cpml, cpml_params, cpml_state, grid_cpml)
        ez = state_cpml.ez.at[cx2, cy2, cz2].add(pulse(n * grid_cpml.dt))
        state_cpml = state_cpml._replace(ez=ez)
        ts_cpml[n] = float(state_cpml.ez[probe_cpml])

    peak_ref = np.max(np.abs(ts_ref))
    diff = ts_cpml - ts_ref
    peak_diff = np.max(np.abs(diff))
    return 20 * np.log10(peak_diff / max(peak_ref, 1e-30))


def main():
    t_start = time.time()
    print("=" * 70)
    print("CPML Reflectivity Sweep")
    print("=" * 70)
    print(f"Layers: {LAYERS}")
    print(f"Frequencies: {[f'{f/1e9:.1f}' for f in FREQS]} GHz")
    print(f"Kappa: {KAPPAS}")
    print(f"Target: {TARGET_DB} dB")
    print(f"Runs: {len(LAYERS) * len(FREQS) * len(KAPPAS)}")
    print()

    results = {}
    for kappa in KAPPAS:
        label = "CFS-CPML" if kappa > 1 else "Standard"
        print(f"\n--- {label} (kappa_max={kappa}) ---")
        for f0 in FREQS:
            for n_layers in LAYERS:
                t0 = time.time()
                db = measure_reflectivity(f0, n_layers, kappa)
                dt = time.time() - t0
                key = f"k{kappa}_f{f0/1e9:.1f}_n{n_layers}"
                results[key] = {"kappa": kappa, "f_ghz": f0/1e9,
                                "layers": n_layers, "db": round(db, 1)}
                ok = "PASS" if db < TARGET_DB else "FAIL"
                print(f"  f={f0/1e9:5.1f}GHz  n={n_layers:2d}  "
                      f"R={db:7.1f}dB  [{ok}]  {dt:.1f}s")

    # Summary table
    print(f"\n{'='*70}")
    print("MINIMUM LAYERS FOR -40 dB")
    print(f"{'='*70}")
    header = f"{'freq':>8}"
    for k in KAPPAS:
        header += f"  {'std' if k==1 else 'CFS':>6}"
    print(header)

    for f0 in FREQS:
        row = f"  {f0/1e9:4.1f}GHz"
        for kappa in KAPPAS:
            min_n = ">20"
            for n in LAYERS:
                key = f"k{kappa}_f{f0/1e9:.1f}_n{n}"
                if results[key]["db"] < TARGET_DB:
                    min_n = str(n)
                    break
            row += f"  {min_n:>6}"
        print(row)

    print(f"\nTotal: {time.time()-t_start:.0f}s")

    # Save
    out = os.path.join(SCRIPT_DIR, "..", "docs", "research_notes",
                       "20260405_cpml_reflectivity_sweep", "results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
