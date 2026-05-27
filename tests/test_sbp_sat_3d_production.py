"""Unit tests for the 3D non-split SBP-SAT production subgrid solver."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from rfx.subgridding.sbp_sat_3d_production import (
    compute_nonsplit_energy_3d,
    init_nonsplit_subgrid_3d,
    step_sbp_sat_nonsplit_3d,
)

# Enable 64-bit precision for high-fidelity energy tracking
jax.config.update("jax_enable_x64", True)


def test_nonsplit_3d_stability():
    """Verify that the 3D production non-split SBP-SAT subgridding is stable over 500 steps."""
    config, state = init_nonsplit_subgrid_3d(
        shape_c=(16, 16, 16),
        dx_c=0.05,
        fine_region=(5, 11, 5, 11, 5, 11),
        ratio=3,
        courant=0.3,
        tau=0.25,  # Moderate SAT penalty coupling to suppress staggering error
    )

    # Excite a localized Ez pulse in the coarse grid
    state = state._replace(
        ez_c=state.ez_c.at[3, 3, 3].set(np.float32(1.0))
    )

    initial_energy = compute_nonsplit_energy_3d(state, config)
    assert np.isfinite(initial_energy)
    assert initial_energy > 0.0

    # JIT compile the update step as a closure
    @jax.jit
    def step_jit(s):
        return step_sbp_sat_nonsplit_3d(s, config)

    max_energy = initial_energy
    for i in range(500):
        state = step_jit(state)
        if (i + 1) % 100 == 0:
            energy = compute_nonsplit_energy_3d(state, config)
            assert np.isfinite(energy)
            # Assert that the energy is bounded and does not grow exponentially
            assert energy <= max_energy * 1.25, (
                f"Energy grew at step {i+1}: {energy:.6e} > {max_energy:.6e} "
                f"(growth {energy/max_energy:.6f}x)"
            )
            max_energy = max(max_energy, energy)

    final_energy = compute_nonsplit_energy_3d(state, config)
    print(f"\n3D production energy conservation: initial={initial_energy:.6e}, "
          f"final={final_energy:.6e}, ratio={final_energy/initial_energy:.6f}")
    assert final_energy <= initial_energy * 1.01
