"""The coax termination adds radial conductance and preserves its host fill.

Root cause: writing vacuum at the sheet removed a shunt capacitance and
artificially lowered the matched reflection floor. See rfx-archive record
``rfx/records/20261107-coax-oneport-floor/PREDECLARE-load-fill.md`` and its
2026-11-10 14:25 KST addendum. The edge ownership remains that of #1373.
"""
from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import (
    cell_owned_component_materials,
    component_e_materials,
    init_materials,
)
from rfx.geometry.rasterize_grid import unpadded_high_walls
from rfx.grid import Grid
from rfx.sources.coaxial_port import (
    SMA_OUTER_RADIUS,
    SMA_PIN_RADIUS,
    stamp_coaxial_annular_resistor,
    stamp_coaxial_line,
)


@pytest.mark.parametrize("reader", [component_e_materials, cell_owned_component_materials])
@pytest.mark.parametrize("host_eps", [2.1, 3.0])
def test_annular_resistor_loads_only_ex_and_ey_on_its_node_plane(reader, host_eps):
    grid = Grid(freq_max=40e9, domain=(8e-3, 8e-3, 8e-3),
                dx=0.4e-3, cpml_layers=4)
    center = (4e-3, 4e-3)
    z = grid.pad_z_lo + 8
    base, shell_inner, pec = stamp_coaxial_line(
        grid, init_materials(grid.shape), center_xy=center,
        z_lo_index=grid.pad_z_lo + 2, z_hi_index=z + 3)
    if host_eps == 3.0:
        base = base._replace(eps_r=jnp.full_like(base.eps_r, host_eps))
    resistance = 25.0
    loaded = stamp_coaxial_annular_resistor(
        grid, base, center_xy=center, z_index=z,
        target_impedance=resistance, shell_inner_radius=shell_inner,
        pec_cell_mask=pec)

    x = (np.arange(grid.nx) - grid.pad_x_lo) * grid.dx - center[0]
    y = (np.arange(grid.ny) - grid.pad_y_lo) * grid.dx - center[1]
    radius = np.hypot(x[:, None], y[None, :])
    annulus = ((radius >= SMA_PIN_RADIUS) & (radius <= shell_inner)
               & ~np.asarray(pec)[:, :, z])
    assert annulus.any()
    assert shell_inner == SMA_OUTER_RADIUS
    # Closed-form sheet conductivity, independent of the stamp's helpers.
    sigma_load = np.log(2.055e-3 / 0.635e-3) / (
        2 * np.pi * 0.4e-3 * resistance)
    if reader is component_e_materials:
        reader = functools.partial(reader, high_wall=unpadded_high_walls(grid))
    _, before = reader(base)
    _, after = reader(loaded)
    for axis in range(3):
        original = np.asarray(before[axis])
        actual = np.asarray(after[axis])
        own = np.zeros(grid.shape, dtype=bool)
        if axis < 2:
            own[:, :, z] = annulus
            np.testing.assert_allclose(actual[own], sigma_load, rtol=1e-6)
        # Every other conductivity, including both adjacent node planes and
        # the entire Ez component, is bit-identical to the unstamped line.
        assert actual[~own].tobytes() == original[~own].tobytes()
        for plane in (z - 1, z + 1):
            assert actual[:, :, plane].tobytes() == original[:, :, plane].tobytes()
    own = np.zeros(grid.shape, dtype=bool)
    own[:, :, z] = annulus
    expected_eps = np.asarray(base.eps_r)
    actual_eps = np.asarray(loaded.eps_r)
    print(f"host_eps={host_eps:g}; loaded_eps={np.unique(actual_eps[own])}; "
          f"expected_edge_sigma={sigma_load:.9f}")
    np.testing.assert_array_equal(expected_eps[own], np.float32(host_eps))
    np.testing.assert_array_equal(actual_eps[own], expected_eps[own])
    np.testing.assert_array_equal(actual_eps[~own], expected_eps[~own])
    np.testing.assert_array_equal(np.asarray(loaded.sigma)[~own],
                                  np.asarray(base.sigma)[~own])
