"""Targeted regression for the agent-friendly workflow example.

This test keeps the problem small but verifies the intended physical/story-level
behavior of example 9:
- a uniform air slab should outperform a higher-eps slab in the chosen PEC guide
- strict-mode optimization should improve over the midpoint uniform slab baseline
- the optimized design should move its mean permittivity downward from the
  midpoint starting value
"""

import numpy as np

from rfx import Box, DesignRegion, GaussianPulse, Simulation, maximize_transmitted_energy, optimize

F0 = 5e9
DOMAIN = (0.04, 0.01, 0.01)
PORT = (0.008, 0.005, 0.005)
OUT = (0.030, 0.005, 0.005)
REGION_LO = (0.015, 0.0, 0.0)
REGION_HI = (0.025, 0.01, 0.01)
N_STEPS = 200


def _uniform_slab_output_energy(eps_r: float) -> float:
    sim = Simulation(freq_max=10e9, domain=DOMAIN, boundary="pec")
    sim.add_material("slab", eps_r=eps_r, sigma=0.0)
    sim.add(Box(REGION_LO, REGION_HI), material="slab")
    sim.add_port(
        PORT,
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=F0, bandwidth=0.5),
    )
    sim.add_probe(PORT, "ez")
    sim.add_probe(OUT, "ez")
    result = sim.run(n_steps=N_STEPS, compute_s_params=False)
    ts = np.asarray(result.time_series)
    return float(np.sum(ts[:, 1] ** 2))


def test_agent_friendly_workflow_example_physics_anchor_and_optimization():
    air_energy = _uniform_slab_output_energy(1.0)
    high_eps_energy = _uniform_slab_output_energy(4.0)
    midpoint_energy = _uniform_slab_output_energy(2.5)

    assert air_energy > high_eps_energy, (
        "In the example-9 PEC guide setup, the matched air slab should transmit "
        "more output energy than a strongly mismatched high-eps slab."
    )
    assert air_energy > midpoint_energy

    sim = Simulation(freq_max=10e9, domain=DOMAIN, boundary="pec")
    sim.add_port(
        PORT,
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=F0, bandwidth=0.5),
    )
    sim.add_probe(PORT, "ez")
    sim.add_probe(OUT, "ez")

    region = DesignRegion(corner_lo=REGION_LO, corner_hi=REGION_HI, eps_range=(1.0, 4.0))
    obj = maximize_transmitted_energy(output_probe_idx=1)

    report = sim.preflight_optimize(region, obj, n_steps=N_STEPS)
    assert report.ok
    assert report.strict_ok

    result = optimize(
        sim,
        region,
        obj,
        n_iters=10,
        lr=0.1,
        n_steps=N_STEPS,
        preflight_mode="strict",
        verbose=False,
    )

    optimized_energy = -float(result.loss_history[-1])
    optimized_mean_eps = float(np.mean(np.asarray(result.eps_design)))

    assert optimized_energy > midpoint_energy
    assert optimized_mean_eps < 2.5
