"""Issue #31 — AD memory estimate sweep across inverse-design problem sizes.

Purpose
-------
Answer the question: does Phase A (+ C) alone fit realistic 3D inverse
design jobs into a 24 GB consumer GPU (RTX 4090), or is Phase B (bf16
field precision) actually needed?

The sim.estimate_ad_memory() formula hardcodes bytes_per_cell=4 (fp32)
and approximates checkpointed-AD as 4x the forward working set. We
apply the same formula to three representative geometries × three
n_steps values and compare against a 24 GB budget.

Run: python scripts/issue31_ad_memory_sweep.py
"""

from __future__ import annotations

import numpy as np

from rfx import Simulation


BUDGET_GB = 24.0  # RTX 4090 / single-GPU consumer ceiling


def build_sim(*, freq_max, extent_mm, dx_mm, cpml_layers=8):
    """Uniform-NU-equivalent sim: cell count driven by dx and extent."""
    dx = dx_mm * 1e-3
    ext = extent_mm * 1e-3
    sim = Simulation(
        freq_max=freq_max,
        domain=(ext, ext, ext),
        dx=dx,
        cpml_layers=cpml_layers,
    )
    sim.add_source((ext / 2, ext / 2, dx * 2), "ez")
    sim.add_probe((ext / 2, ext / 2, ext / 2), "ez")
    return sim


def row(label, sim, n_steps):
    est = sim.estimate_ad_memory(n_steps=n_steps, available_memory_gb=BUDGET_GB)
    # Phase A+C delta: time-series tape removed. Estimator doesn't track
    # it, so the A+C number equals ad_checkpointed_gb on this formula.
    ac_gb = est.ad_checkpointed_gb
    # Phase B projection: fields 4 → 2 bytes/cell. Forward working set
    # drops by (6 field *2B)/(6 field *4B + 6 mat *4B + 15% CPML *4B),
    # roughly 30%. Apply as a simple multiplier on the field component.
    # Estimator mixes field+mat+CPML in forward_bytes, so approximate:
    forward_gb = est.forward_gb
    # bf16 halves field bytes (40% of forward), so forward_b drops ~20%.
    forward_b_bf16 = forward_gb * 0.80
    ad_b_bf16 = 4 * forward_b_bf16 + est.ntff_dft_gb
    fits = lambda gb: "OK " if gb <= BUDGET_GB else "OOM"
    print(
        f"  {label:<40} fwd={forward_gb:>7.3f}GB  "
        f"A+C={ac_gb:>7.3f}GB [{fits(ac_gb)}]  "
        f"A+B+C={ad_b_bf16:>7.3f}GB [{fits(ad_b_bf16)}]  "
        f"fullAD={est.ad_full_gb:>9.3f}GB [{fits(est.ad_full_gb)}]"
    )


def main():
    print("=" * 110)
    print(f"Issue #31 memory sweep — budget = {BUDGET_GB} GB (RTX 4090)")
    print("=" * 110)

    cases = [
        # label,           freq_max, ext_mm, dx_mm
        ("Patch (baseline)", 10e9, 30.0, 0.5),         # ~0.2 M cells
        ("Horn / WR-90 mid", 10e9, 60.0, 0.5),         # ~1.8 M cells
        ("Metasurface",      10e9, 100.0, 0.6),        # ~5 M cells
    ]
    for label, fmax, ext, dx in cases:
        sim = build_sim(freq_max=fmax, extent_mm=ext, dx_mm=dx)
        # Grid shape for header
        est = sim.estimate_ad_memory(n_steps=1)
        nx = int(round(ext * 1e-3 / (dx * 1e-3))) + 1 + 2 * 8
        cells_m = nx ** 3 / 1e6
        print(f"\n[{label}] nx≈{nx}, cells≈{cells_m:.2f}M, fwd={est.forward_gb:.3f}GB")
        for n_steps in (2000, 5000, 10000, 30000):
            row(f"n_steps={n_steps}", sim, n_steps)

    print("\nLegend: fwd = forward working set, A+C = checkpointed AD, "
          "A+B+C = bf16-field projection, fullAD = no-checkpoint baseline.")


if __name__ == "__main__":
    main()
