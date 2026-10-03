"""The coax termination loads radial edges at one node plane (#1373)."""
from __future__ import annotations

import numpy as np
import pytest

from rfx.core.yee import (
    cell_owned_component_materials,
    component_e_materials,
    init_materials,
)
from rfx.grid import Grid
from rfx.sources.coaxial_port import (
    SMA_OUTER_RADIUS,
    SMA_PIN_RADIUS,
    stamp_coaxial_annular_resistor,
    stamp_coaxial_line,
)


@pytest.mark.parametrize("reader", [component_e_materials, cell_owned_component_materials])
def test_annular_resistor_loads_only_ex_and_ey_on_its_node_plane(reader):
    grid = Grid(freq_max=40e9, domain=(8e-3, 8e-3, 8e-3),
                dx=0.4e-3, cpml_layers=4)
    center = (4e-3, 4e-3)
    z = grid.pad_z_lo + 8
    base, shell_inner, pec = stamp_coaxial_line(
        grid, init_materials(grid.shape), center_xy=center,
        z_lo_index=grid.pad_z_lo + 2, z_hi_index=z + 3)
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
    sigma_load = np.log(shell_inner / SMA_PIN_RADIUS) / (
        2 * np.pi * grid.dx * resistance)
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
    # Permittivity retains the existing cell-volume assignment independently
    # of the per-edge conductivity contract above.
    expected_eps = np.array(base.eps_r)
    expected_eps[:, :, z][annulus] = 1.0
    np.testing.assert_array_equal(loaded.eps_r, expected_eps)
