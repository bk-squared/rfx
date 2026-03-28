"""Tests for thin conductor subcell model.

Validates:
1. Effective conductivity computation
2. Selective application via shape mask
3. Integration with Simulation API
"""

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import init_materials
from rfx.geometry.csg import Box
from rfx.materials.thin_conductor import ThinConductor, apply_thin_conductor


def test_thin_conductor_sigma_eff():
    """σ_eff = σ_bulk · (t / Δx) should be correct."""
    grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.002))
    dx = grid.dx

    sigma_bulk = 5.8e7  # copper
    thickness = 35e-6  # 35 µm = 1 oz copper
    expected_sigma_eff = sigma_bulk * (thickness / dx)

    shape_box = Box((0.005, 0.005, 0.0), (0.015, 0.015, 0.001))
    tc = ThinConductor(shape=shape_box, sigma_bulk=sigma_bulk, thickness=thickness)

    materials = init_materials(grid.shape)
    materials = apply_thin_conductor(grid, tc, materials)

    # Check a cell inside the conductor region
    mask = shape_box.mask(grid)
    inside_idx = np.argwhere(np.array(mask))
    if len(inside_idx) > 0:
        i, j, k = inside_idx[len(inside_idx) // 2]
        sigma_val = float(materials.sigma[i, j, k])
        assert abs(sigma_val - expected_sigma_eff) / expected_sigma_eff < 1e-4, \
            f"σ_eff={sigma_val:.4e}, expected={expected_sigma_eff:.4e}"
        print(f"\nThin conductor σ_eff = {sigma_val:.4e} S/m "
              f"(expected {expected_sigma_eff:.4e})")

    # Check a cell outside — should be zero
    outside_idx = np.argwhere(~np.array(mask))
    if len(outside_idx) > 0:
        i, j, k = outside_idx[0]
        assert float(materials.sigma[i, j, k]) == 0.0


def test_thin_conductor_preserves_outside():
    """Thin conductor should not modify material outside its shape."""
    grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.002))

    # Pre-fill with some background
    materials = init_materials(grid.shape)
    materials = materials._replace(
        eps_r=jnp.full(grid.shape, 4.4, dtype=jnp.float32),
        sigma=jnp.full(grid.shape, 0.025, dtype=jnp.float32),
    )

    shape_box = Box((0.005, 0.005, 0.0), (0.01, 0.01, 0.001))
    tc = ThinConductor(shape=shape_box, sigma_bulk=5.8e7, thickness=35e-6, eps_r=1.0)
    materials = apply_thin_conductor(grid, tc, materials)

    mask = shape_box.mask(grid)

    # Inside: eps_r should be 1.0 (conductor), sigma should be thin-conductor value
    inside_idx = np.argwhere(np.array(mask))
    if len(inside_idx) > 0:
        i, j, k = inside_idx[len(inside_idx) // 2]
        assert float(materials.eps_r[i, j, k]) == 1.0

    # Outside: eps_r should still be 4.4, sigma should still be 0.025
    outside_idx = np.argwhere(~np.array(mask))
    if len(outside_idx) > 0:
        i, j, k = outside_idx[len(outside_idx) // 2]
        assert abs(float(materials.eps_r[i, j, k]) - 4.4) < 1e-4
        assert abs(float(materials.sigma[i, j, k]) - 0.025) < 1e-4


def test_thin_conductor_api_integration():
    """ThinConductor works through the Simulation API."""
    from rfx.api import Simulation

    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.002), boundary="pec")
    sim.add_material("substrate", eps_r=4.4, sigma=0.025)
    sim.add(Box((0, 0, 0), (0.02, 0.02, 0.001)), material="substrate")
    sim.add_thin_conductor(
        Box((0.005, 0.005, 0.001), (0.015, 0.015, 0.001)),
        sigma_bulk=5.8e7,
        thickness=35e-6,
    )

    # Should build without error
    grid = sim._build_grid()
    materials, debye, lorentz = sim._build_materials(grid)

    # Verify thin conductor region has high conductivity
    tc_box = Box((0.005, 0.005, 0.001), (0.015, 0.015, 0.001))
    mask = tc_box.mask(grid)
    inside_idx = np.argwhere(np.array(mask))
    if len(inside_idx) > 0:
        i, j, k = inside_idx[len(inside_idx) // 2]
        assert float(materials.sigma[i, j, k]) > 1e3, \
            "Thin conductor should have high effective sigma"

    print(f"\nThin conductor API integration: OK, grid={grid.shape}")
