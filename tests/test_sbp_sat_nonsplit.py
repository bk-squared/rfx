"""Unit tests for the non-split SBP-SAT FDTD subgridding solver."""

from __future__ import annotations

import jax
import numpy as np

from rfx.subgridding.sbp_sat_nonsplit import (
    compute_nonsplit_energy_2d,
    init_nonsplit_subgrid_2d,
    step_sbp_sat_nonsplit_2d,
)

# Enable 64-bit precision for high-fidelity energy tracking
jax.config.update("jax_enable_x64", True)


def test_nonsplit_subgrid_2d_stability():
    """Verify that the 2D non-split SBP-SAT subgridding is stable over 200 steps."""
    config, state = init_nonsplit_subgrid_2d(
        nx_c=40,
        ny_c=40,
        dx_c=0.05,
        fine_region=(15, 25, 15, 25),
        ratio=3,
        courant=0.05,
    )

    t0 = 40.0 * config.dt
    pulse_width = 10.0 * config.dt

    energies = []
    steps = 200

    # JIT compile the update step as a closure capturing static config constants
    @jax.jit
    def step_jit(s):
        return step_sbp_sat_nonsplit_2d(s, config)

    for step in range(steps):
        t = step * config.dt
        pulse = np.exp(-((t - t0) / pulse_width) ** 2)
        state = state._replace(
            ez_c=state.ez_c.at[5, 5].add(pulse)
        )

        state = step_jit(state)

        energy = compute_nonsplit_energy_2d(state, config)
        energies.append(energy)

    energies = np.array(energies)
    post_source_idx = 100
    max_energy_post_source = np.max(energies[post_source_idx:])

    growth_ratio = max_energy_post_source / energies[post_source_idx]
    print(f"\nJAX non-split subgrid max energy growth ratio: {growth_ratio:.6f}")
    assert growth_ratio <= 25.0
