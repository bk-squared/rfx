"""Tests for Dey-Mittra conformal PEC boundaries."""

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import init_state, init_materials, update_e, update_h
from rfx.geometry.csg import Box, Sphere
from rfx.geometry.conformal import compute_conformal_weights, apply_conformal_pec
from rfx.boundaries.pec import apply_pec
from rfx.sources.sources import GaussianPulse


def test_conformal_weights_pec_box():
    """Weights should be exactly 0 inside box, 1 well outside."""
    grid = Grid(freq_max=5e9, domain=(0.05, 0.05, 0.05), dx=0.005, cpml_layers=0)
    box = Box((0.015, 0.015, 0.015), (0.035, 0.035, 0.035))

    w_ex, w_ey, w_ez = compute_conformal_weights(grid, [box])

    mask = np.array(box.mask(grid))

    # Inside PEC: weights = 0
    assert float(jnp.max(w_ex[mask])) == 0.0, "Ex weight inside PEC should be 0"
    assert float(jnp.max(w_ey[mask])) == 0.0, "Ey weight inside PEC should be 0"
    assert float(jnp.max(w_ez[mask])) == 0.0, "Ez weight inside PEC should be 0"

    # Well outside PEC (corner cells far from boundary): weights = 1
    assert float(w_ex[0, 0, 0]) == 1.0, "Far exterior Ex weight should be 1"
    assert float(w_ey[0, 0, 0]) == 1.0, "Far exterior Ey weight should be 1"

    print("\nConformal weights for PEC box:")
    print(f"  Interior zeros: {int(np.sum(np.array(w_ex) == 0))}")
    print(f"  Exterior ones:  {int(np.sum(np.array(w_ex) == 1))}")
    print(f"  Fractional:     {int(np.sum((np.array(w_ex) > 0) & (np.array(w_ex) < 1)))}")


def test_conformal_sphere_has_fractional_weights():
    """A PEC sphere should produce fractional conformal weights at its surface."""
    grid = Grid(freq_max=5e9, domain=(0.05, 0.05, 0.05), dx=0.005, cpml_layers=0)
    sphere = Sphere((0.025, 0.025, 0.025), 0.015)

    w_ex, w_ey, w_ez = compute_conformal_weights(grid, [sphere], n_sub=4)

    w_ex_np = np.array(w_ex)
    n_zero = int(np.sum(w_ex_np == 0))
    n_one = int(np.sum(w_ex_np == 1.0))
    n_frac = int(np.sum((w_ex_np > 0) & (w_ex_np < 1.0)))

    print("\nConformal weights for PEC sphere:")
    print(f"  Interior (w=0): {n_zero}")
    print(f"  Exterior (w=1): {n_one}")
    print(f"  Fractional:     {n_frac}")

    # Sphere should produce fractional weights at surface cells
    assert n_zero > 0, "Should have interior PEC cells"
    assert n_one > 0, "Should have exterior cells"
    assert n_frac > 0, "Sphere should produce fractional boundary weights"

    # Fractional weights should be between 0 and 1
    frac_vals = w_ex_np[(w_ex_np > 0) & (w_ex_np < 1.0)]
    assert np.all(frac_vals > 0) and np.all(frac_vals < 1.0)


def test_conformal_disabled_matches_original():
    """With all weights=1, conformal should give identical results to no conformal."""
    grid = Grid(freq_max=5e9, domain=(0.03, 0.03, 0.03), dx=0.003, cpml_layers=0)
    w_ones = (jnp.ones(grid.shape), jnp.ones(grid.shape), jnp.ones(grid.shape))

    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    pulse = GaussianPulse(f0=3e9)
    src = (grid.nx // 2, grid.ny // 2, grid.nz // 2)

    # Run with conformal weights = 1 (should be identity)
    for n in range(20):
        t = n * grid.dt
        state = update_h(state, materials, grid.dt, grid.dx)
        state = update_e(state, materials, grid.dt, grid.dx)
        state = apply_pec(state)
        state = apply_conformal_pec(state, *w_ones)
        state = state._replace(ez=state.ez.at[src].add(pulse(t)))

    ez_conformal = float(jnp.sum(state.ez ** 2))

    # Run without conformal
    state2 = init_state(grid.shape)
    for n in range(20):
        t = n * grid.dt
        state2 = update_h(state2, materials, grid.dt, grid.dx)
        state2 = update_e(state2, materials, grid.dt, grid.dx)
        state2 = apply_pec(state2)
        state2 = state2._replace(ez=state2.ez.at[src].add(pulse(t)))

    ez_original = float(jnp.sum(state2.ez ** 2))

    diff = abs(ez_conformal - ez_original)
    print(f"\nConformal disabled vs original: diff={diff:.2e}")
    assert diff < 1e-10, f"Conformal with w=1 should match original: diff={diff}"
