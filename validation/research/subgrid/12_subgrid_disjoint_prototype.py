#!/usr/bin/env python3
"""Research example: disjoint 3-D subgrid prototype smoke.

This example exercises the Stage-2 disjoint subgrid runner
(``rfx.runners.disjoint.run_disjoint_stage2_path``), which is **research-only**
and **not long-time energy-stable**. It is not a public
``Simulation.add_refinement`` feature claim and it is not a Meep/openEMS
cross-solver pass. The short-window run below is a research smoke, *not* a
"COMPLETE" or production-ready status: the disjoint topology still grows energy
over long integration windows and remains outside the validated production
envelope. Guarded one-sided production subgrid crossval evidence lives in
``scripts/subgrid_external_crossval_audit.py`` and related guarded-envelope
artifacts; this disjoint prototype remains research-only.

Run:
    python validation/research/subgrid/12_subgrid_disjoint_prototype.py
"""

from __future__ import annotations

import numpy as np

from rfx import GaussianPulse, Simulation
from rfx.runners.disjoint import run_disjoint_stage2_path


def _build_disjoint_simulation() -> Simulation:
    """Return a small research-only Stage-2 disjoint subgrid simulation."""
    sim = Simulation(
        freq_max=8e9,
        domain=(0.04, 0.04, 0.024),
        boundary="pec",
        dx=0.002,
    )
    sim.add_refinement(
        z_range=(0.006, 0.018),
        ratio=2,
        validation="research",
        topology="stage2_disjoint_3d",
    )
    sim.add_source(
        (0.04 / 3.0, 0.04 / 3.0, 0.0114),
        "ez",
        waveform=GaussianPulse(f0=4e9, bandwidth=0.6),
    )
    sim.add_probe((2.0 * 0.04 / 3.0, 2.0 * 0.04 / 3.0, 0.0126), "ez")
    return sim


def main() -> int:
    sim = _build_disjoint_simulation()
    grid = sim._build_grid()
    n_steps = 100
    result = run_disjoint_stage2_path(sim, grid, n_steps=n_steps)

    series = np.asarray(result.time_series)
    finite = bool(np.all(np.isfinite(series)))
    abs_max = float(np.max(np.abs(series))) if series.size else 0.0
    print(
        "Disjoint subgrid research prototype: short-window smoke "
        "(research-only; NOT a COMPLETE/production status -- the disjoint "
        "topology is not long-time energy-stable)"
    )
    print(f"  steps run:          {n_steps}")
    print(f"  probe series shape: {series.shape}")
    print(f"  probe abs max:      {abs_max:.6e}")
    print(f"  all finite:         {finite}")
    print(f"  timestep dt:        {result.dt:.6e}")
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
