#!/usr/bin/env python3
"""Empirical compile-memory sweep for supported-safe optimizer lanes.

This script measures compiled gradient memory through the public preflight API
for the supported-safe built-in proxy-objective lanes in `optimize()` and
`topology_optimize()`.
"""

from __future__ import annotations

import argparse
import json

from rfx import (
    DesignRegion,
    GaussianPulse,
    Simulation,
    TopologyDesignRegion,
    maximize_transmitted_energy,
)


def build_optimize_case(domain_x: float = 0.02, *, boundary: str = "pec", cpml_layers: int = 8, dx: float | None = None):
    sim_kwargs = {
        "freq_max": 5e9,
        "domain": (domain_x, 0.01, 0.01),
        "boundary": boundary,
    }
    if boundary == "cpml":
        sim_kwargs["cpml_layers"] = cpml_layers
        sim_kwargs["dx"] = 0.002 if dx is None else dx
        edge_clearance = max(5 * sim_kwargs["dx"], 0.010)
        port_x, probe_x = edge_clearance, domain_x - edge_clearance
    else:
        if dx is not None:
            sim_kwargs["dx"] = dx
        port_x, probe_x = 0.005, domain_x - 0.005

    sim = Simulation(**sim_kwargs)
    sim.add_port((port_x, 0.005, 0.005), "ez", impedance=50.0, waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5))
    sim.add_probe((probe_x, 0.005, 0.005), "ez")
    region = DesignRegion(
        corner_lo=(0.008 if boundary == "cpml" else 0.007, 0.0, 0.0),
        corner_hi=(min(domain_x - (0.008 if boundary == "cpml" else 0.004), 0.013), 0.01, 0.01),
        eps_range=(1.0, 4.4),
    )
    objective = maximize_transmitted_energy(output_probe_idx=0)
    return sim, region, objective


def build_topology_case(domain_x: float = 0.02, *, boundary: str = "pec", cpml_layers: int = 8, dx: float | None = None):
    sim_kwargs = {
        "freq_max": 5e9,
        "domain": (domain_x, 0.01, 0.01),
        "boundary": boundary,
    }
    if boundary == "cpml":
        sim_kwargs["cpml_layers"] = cpml_layers
        sim_kwargs["dx"] = 0.002 if dx is None else dx
        edge_clearance = max(5 * sim_kwargs["dx"], 0.010)
        port_x, probe_x = edge_clearance, domain_x - edge_clearance
    else:
        if dx is not None:
            sim_kwargs["dx"] = dx
        port_x, probe_x = 0.005, domain_x - 0.005

    sim = Simulation(**sim_kwargs)
    sim.add_port((port_x, 0.005, 0.005), "ez", impedance=50.0, waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5))
    sim.add_probe((probe_x, 0.005, 0.005), "ez")
    region = TopologyDesignRegion(
        corner_lo=(0.008 if boundary == "cpml" else 0.007, 0.0, 0.0),
        corner_hi=(min(domain_x - (0.008 if boundary == "cpml" else 0.004), 0.013), 0.01, 0.01),
        material_bg="air",
        material_fg="fr4",
    )
    objective = maximize_transmitted_energy(output_probe_idx=0)
    return sim, region, objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", nargs="+", type=int, default=[20, 50, 100, 200])
    parser.add_argument("--domain-x", nargs="+", type=float, default=[0.03, 0.04])
    parser.add_argument("--boundaries", nargs="+", default=["pec", "cpml"])
    args = parser.parse_args()

    records = []
    for boundary in args.boundaries:
        for domain_x in args.domain_x:
            sim, region, objective = build_optimize_case(domain_x, boundary=boundary)
            for n_steps in args.n_steps:
                report = sim.preflight_optimize(
                    region,
                    objective,
                    n_steps=n_steps,
                    memory_budget_mb=1e9,
                )
                records.append(
                    {
                        "lane": "optimize",
                        "boundary": boundary,
                        "domain_x": domain_x,
                        "n_steps": n_steps,
                        "compiled_memory_mb": report.compiled_memory_mb,
                        "trace_cost": report.trace_cost,
                        "issues": [issue.code for issue in report.issues],
                    }
                )

    for boundary in args.boundaries:
        for domain_x in args.domain_x:
            sim, region, objective = build_topology_case(domain_x, boundary=boundary)
            for n_steps in args.n_steps:
                report = sim.preflight_topology_optimize(
                    region,
                    objective,
                    n_steps=n_steps,
                    memory_budget_mb=1e9,
                )
                records.append(
                    {
                        "lane": "topology",
                        "boundary": boundary,
                        "domain_x": domain_x,
                        "n_steps": n_steps,
                        "compiled_memory_mb": report.compiled_memory_mb,
                        "trace_cost": report.trace_cost,
                        "issues": [issue.code for issue in report.issues],
                    }
                )

    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
