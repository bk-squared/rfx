#!/usr/bin/env python3
"""Runtime validation harness for issue #13 supported-safe lanes.

This script complements the compile-time preflight gate by executing a small set
of representative optimizer/topology runs and reporting:
- whether preflight passed
- the compiled-memory estimate from preflight
- whether a strict one-iteration run succeeded

It is designed to run locally or on GPU/VESSL without code changes.
"""

from __future__ import annotations

import json
import time

import jax

from rfx import (
    DesignRegion,
    GaussianPulse,
    Simulation,
    TopologyDesignRegion,
    maximize_transmitted_energy,
    optimize,
    topology_optimize,
)


def build_sim(*, boundary: str, domain_x: float, dx: float | None = None) -> Simulation:
    kwargs = {
        "freq_max": 5e9,
        "domain": (domain_x, 0.01, 0.01),
        "boundary": boundary,
    }
    if boundary == "cpml":
        kwargs["cpml_layers"] = 8
        kwargs["dx"] = 0.002 if dx is None else dx
        edge_clearance = max(5 * kwargs["dx"], 0.010)
        port_x, probe_x = edge_clearance, domain_x - edge_clearance
    else:
        if dx is not None:
            kwargs["dx"] = dx
        port_x, probe_x = 0.005, domain_x - 0.005

    sim = Simulation(**kwargs)
    sim.add_port(
        (port_x, 0.005, 0.005),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5),
    )
    sim.add_probe((probe_x, 0.005, 0.005), "ez")
    return sim


def run_case(case: dict) -> dict:
    sim = build_sim(boundary=case["boundary"], domain_x=case["domain_x"])
    objective = maximize_transmitted_energy(output_probe_idx=0)

    if case["lane"] == "optimize":
        region = DesignRegion(
            corner_lo=(0.008 if case["boundary"] == "cpml" else 0.007, 0.0, 0.0),
            corner_hi=(min(case["domain_x"] - (0.008 if case["boundary"] == "cpml" else 0.004), 0.013), 0.01, 0.01),
            eps_range=(1.0, 4.4),
        )
        report = sim.preflight_optimize(
            region,
            objective,
            n_steps=case["n_steps"],
            memory_budget_mb=case["memory_budget_mb"],
        )
        output = {
            "case": case,
            "backend": jax.default_backend(),
            "preflight_ok": report.ok,
            "preflight_strict_ok": report.strict_ok,
            "issues": [issue.code for issue in report.issues],
            "issue_messages": [issue.message for issue in report.issues],
            "compiled_memory_mb": report.compiled_memory_mb,
        }
        if report.strict_ok:
            t0 = time.time()
            result = optimize(
                sim,
                region,
                objective,
                n_iters=1,
                lr=0.05,
                n_steps=case["n_steps"],
                memory_budget_mb=case["memory_budget_mb"],
                preflight_mode="strict",
                verbose=False,
            )
            output["runtime_seconds"] = time.time() - t0
            output["loss_history"] = result.loss_history
        return output

    region = TopologyDesignRegion(
        corner_lo=(0.008 if case["boundary"] == "cpml" else 0.007, 0.0, 0.0),
        corner_hi=(min(case["domain_x"] - (0.008 if case["boundary"] == "cpml" else 0.004), 0.013), 0.01, 0.01),
        material_bg="air",
        material_fg=case["material_fg"],
    )
    report = sim.preflight_topology_optimize(
        region,
        objective,
        n_steps=case["n_steps"],
        memory_budget_mb=case["memory_budget_mb"],
    )
    output = {
        "case": case,
        "backend": jax.default_backend(),
        "preflight_ok": report.ok,
        "preflight_strict_ok": report.strict_ok,
        "issues": [issue.code for issue in report.issues],
        "issue_messages": [issue.message for issue in report.issues],
        "compiled_memory_mb": report.compiled_memory_mb,
    }
    if report.strict_ok:
        t0 = time.time()
        result = topology_optimize(
            sim,
            region,
            objective,
            n_iterations=1,
            learning_rate=0.05,
            n_steps=case["n_steps"],
            memory_budget_mb=case["memory_budget_mb"],
            preflight_mode="strict",
            verbose=False,
        )
        output["runtime_seconds"] = time.time() - t0
        output["loss_history"] = result.history
    return output


def main() -> None:
    cases = [
        {
            "lane": "optimize",
            "boundary": "pec",
            "domain_x": 0.02,
            "n_steps": 100,
            "memory_budget_mb": 10.0,
        },
        {
            "lane": "optimize",
            "boundary": "cpml",
            "domain_x": 0.03,
            "n_steps": 20,
            "memory_budget_mb": 32.0,
        },
        {
            "lane": "topology",
            "boundary": "cpml",
            "domain_x": 0.03,
            "n_steps": 20,
            "material_fg": "fr4",
            "memory_budget_mb": 32.0,
        },
        {
            "lane": "topology",
            "boundary": "cpml",
            "domain_x": 0.03,
            "n_steps": 20,
            "material_fg": "pec",
            "memory_budget_mb": 32.0,
        },
    ]
    results = [run_case(case) for case in cases]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
