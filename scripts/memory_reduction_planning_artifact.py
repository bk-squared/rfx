#!/usr/bin/env python3
"""Generate a memory-reduction planning artifact for the non-uniform AD lane.

This script is intentionally lightweight: it does not run FDTD.  It builds the
same kind of smooth non-uniform z-profile used by the Stage-1 physical gate,
then records the cell-savings comparator and the recommended segmented-AD
``checkpoint_every`` for a requested memory budget.

Run:
    python scripts/memory_reduction_planning_artifact.py --available-memory-gb 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from rfx import GaussianPulse, Simulation
from rfx.auto_config import smooth_grading
from rfx.grid import C0


def build_memory_reduction_case() -> Simulation:
    """Return a smooth non-uniform PEC cavity planning case.

    The geometry mirrors the Stage-1 NU cavity gate: coarse x/y cells with a
    locally fine z band.  A uniform-fine grid at the smallest z spacing would
    refine all axes, so this gives a large cell-count saving while staying in a
    clean vacuum/PEC support envelope.
    """
    a = b = 40e-3
    dz_raw = np.array([1.0e-3] * 14 + [0.25e-3] * 8 + [1.0e-3] * 14)
    dz_profile = smooth_grading(dz_raw, max_ratio=1.3)
    d = float(np.sum(dz_profile))
    f_tm110 = float((C0 / 2.0) * np.sqrt((1.0 / a) ** 2 + (1.0 / b) ** 2))

    sim = Simulation(
        freq_max=2.0 * f_tm110,
        domain=(a, b, d),
        dx=1.0e-3,
        dz_profile=dz_profile,
        boundary="pec",
        cpml_layers=0,
    )
    sim.add_source(
        (a / 3.0, b / 3.0, d / 2.0),
        "ez",
        waveform=GaussianPulse(f0=f_tm110, bandwidth=0.8),
    )
    sim.add_probe((2.0 * a / 3.0, 2.0 * b / 3.0, d / 2.0), "ez")
    return sim


def build_artifact(
    *,
    n_steps: int = 5_000,
    available_memory_gb: float = 1.0,
    target_fraction: float = 0.85,
) -> dict[str, Any]:
    """Build a JSON-serializable memory-reduction planning artifact."""
    sim = build_memory_reduction_case()
    plan = sim.plan_ad_memory(
        n_steps=n_steps,
        available_memory_gb=available_memory_gb,
        target_fraction=target_fraction,
    )
    report = sim.mesh_intelligence_report(
        n_steps=n_steps,
        checkpoint_every=plan.checkpoint_every,
        available_memory_gb=available_memory_gb,
    )
    segmented_gb = (
        None if report.ad_memory is None else report.ad_memory.ad_segmented_gb
    )
    return {
        "status": "memory_reduction_planning_ready",
        "description": (
            "Non-uniform z mesh plus segmented AD planning artifact for a "
            "smooth PEC cavity case; physics validation is provided separately "
            "by scripts/stage1_nu_cavity_physics_gate.py."
        ),
        "inputs": {
            "n_steps": int(n_steps),
            "available_memory_gb": float(available_memory_gb),
            "target_fraction": float(target_fraction),
        },
        "plan": plan.to_dict(),
        "mesh_report": report.to_dict(),
        "gates": {
            "no_preflight_issues": len(report.preflight_issues) == 0,
            "cell_savings_at_least_40x": report.cell_savings_factor >= 40.0,
            "segmented_ad_fits_budget": bool(plan.segmented_fits),
            "segmented_ad_below_full_ad": (
                segmented_gb is not None
                and report.ad_memory is not None
                and segmented_gb < report.ad_memory.ad_full_gb
            ),
        },
        "next_validation": "Run scripts/stage1_nu_cavity_physics_gate.py for RF accuracy evidence.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=5_000)
    parser.add_argument("--available-memory-gb", type=float, default=1.0)
    parser.add_argument("--target-fraction", type=float, default=0.85)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/research_notes/memory_reduction_planning_artifact.json"),
    )
    args = parser.parse_args(argv)

    artifact = build_artifact(
        n_steps=args.n_steps,
        available_memory_gb=args.available_memory_gb,
        target_fraction=args.target_fraction,
    )
    if not all(artifact["gates"].values()):
        failed = [k for k, v in artifact["gates"].items() if not v]
        raise SystemExit(f"memory-reduction planning gates failed: {failed}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    plan = artifact["plan"]
    report = artifact["mesh_report"]
    print("Memory-reduction planning artifact: PASS")
    print(f"  output:              {args.output}")
    print(f"  checkpoint_every:    {plan['checkpoint_every']}")
    print(f"  segmented AD memory: {plan['selected_estimate']['ad_segmented_gb']:.4f} GB")
    print(f"  target memory:       {plan['target_memory_gb']:.4f} GB")
    print(f"  cell savings:        {report['cell_savings_factor']:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
