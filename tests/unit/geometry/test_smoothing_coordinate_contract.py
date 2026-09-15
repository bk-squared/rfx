"""Cheap coordinate ownership checks for the opt-in Kottke path (#833)."""

import numpy as np

from rfx.geometry.rasterize_grid import coords_from_uniform_grid
from rfx.geometry.smoothing import _yee_coords
from rfx.grid import Grid


def test_smoothing_uses_shared_uniform_node_spine():
    grid = Grid(
        freq_max=1.0e9,
        domain=(7.0e-3, 5.0e-3, 3.0e-3),
        dx=2.54e-4,
        cpml_layers=2,
    )
    expected = coords_from_uniform_grid(grid)
    actual = _yee_coords(grid)

    # The smoothing path may use the active JAX storage precision, but it
    # must consume the same host-built node values as the binary rasterizer.
    for got, want in zip(actual, (expected.x, expected.y, expected.z)):
        got_array = np.asarray(got)
        np.testing.assert_array_equal(got_array, np.asarray(want, dtype=got_array.dtype))
