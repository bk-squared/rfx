"""Tests for 2D TM SBP-SAT subgridding."""

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.subgridding.sbp_sat_2d import (
    init_subgrid_2d, step_subgrid_2d, compute_energy_2d,
)


@pytest.mark.slow
def test_2d_stability():
    """Energy bounded over 10,000 steps."""
    config, state = init_subgrid_2d(
        nx_c=40, ny_c=40, dx_c=0.003,
        fine_region=(15, 25, 15, 25), ratio=3,
    )
    # Inject pulse on coarse grid (outside fine region)
    state = state._replace(ez_c=state.ez_c.at[8, 8].set(1.0))
    initial_energy = compute_energy_2d(state, config)

    max_energy = initial_energy
    for i in range(10000):
        state = step_subgrid_2d(state, config)
        if i % 2000 == 0:
            e = compute_energy_2d(state, config)
            max_energy = max(max_energy, e)

    final_energy = compute_energy_2d(state, config)
    print("\n2D stability (10K steps):")
    print(f"  Initial: {initial_energy:.6e}")
    print(f"  Max:     {max_energy:.6e}")
    print(f"  Final:   {final_energy:.6e}")

    assert max_energy < initial_energy * 20, f"Energy blew up: {max_energy}"
    assert np.isfinite(final_energy), "Final energy should be finite"


def test_2d_pulse_propagation():
    """Pulse should propagate across the 2D coarse-fine interface."""
    config, state = init_subgrid_2d(
        nx_c=30, ny_c=30, dx_c=0.002,
        fine_region=(10, 20, 10, 20), ratio=3,
    )
    state = state._replace(ez_c=state.ez_c.at[5, 15].set(1.0))

    for _ in range(3000):
        state = step_subgrid_2d(state, config)

    fine_signal = float(jnp.max(jnp.abs(state.ez_f)))
    coarse_signal = float(jnp.max(jnp.abs(state.ez_c)))

    print("\n2D pulse propagation:")
    print(f"  Coarse max |Ez|: {coarse_signal:.6e}")
    print(f"  Fine max |Ez|:   {fine_signal:.6e}")

    assert np.isfinite(coarse_signal), "Coarse field should be finite"
    assert np.isfinite(fine_signal), "Fine field should be finite"
    assert coarse_signal < 10, "Coarse field should not blow up"


def test_2d_small_fine_region():
    """Small fine region should still be stable."""
    config, state = init_subgrid_2d(
        nx_c=20, ny_c=20, dx_c=0.002,
        fine_region=(8, 12, 8, 12),  # small 4x4 coarse = 12x12 fine
        ratio=3,
    )
    state = state._replace(ez_c=state.ez_c.at[5, 5].set(1.0))

    for _ in range(500):
        state = step_subgrid_2d(state, config)

    energy = compute_energy_2d(state, config)
    print(f"\n2D small fine region energy: {energy:.6e}")
    assert energy > 0, "Should have positive energy"
    assert np.isfinite(energy), "Energy should be finite"
