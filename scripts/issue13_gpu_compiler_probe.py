#!/usr/bin/env python3
"""Probe JAX compile diagnostics for a supported-safe issue #13 lane."""

from __future__ import annotations

import json

import jax

from rfx import DesignRegion, GaussianPulse, Simulation, maximize_transmitted_energy
from rfx.preflight import _compile_memory_gate_optimize


def main() -> None:
    sim = Simulation(freq_max=5e9, domain=(0.03, 0.01, 0.01), boundary="cpml", cpml_layers=8, dx=0.002)
    sim.add_port(
        (0.010, 0.005, 0.005),
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5),
    )
    sim.add_probe((0.020, 0.005, 0.005), "ez")
    region = DesignRegion(
        corner_lo=(0.012, 0.0, 0.0),
        corner_hi=(0.018, 0.01, 0.01),
        eps_range=(1.0, 4.4),
    )
    objective = maximize_transmitted_energy(output_probe_idx=0)

    compiled_stats, reason = _compile_memory_gate_optimize(
        sim,
        region,
        objective,
        resolved_n_steps=20,
    )

    payload = {
        "backend": jax.default_backend(),
        "compiled_stats": compiled_stats,
        "reason": reason,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
