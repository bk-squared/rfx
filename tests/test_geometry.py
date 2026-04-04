"""Tests for CSG geometry primitives."""

import jax.numpy as jnp
import pytest

from rfx.grid import Grid
from rfx.geometry.csg import Box, Sphere, union, difference, rasterize


@pytest.fixture
def grid():
    return Grid(freq_max=3e9, domain=(0.1, 0.1, 0.1), cpml_layers=0)


def test_box_mask(grid):
    """Box should create a rectangular region."""
    box = Box(corner_lo=(0.02, 0.02, 0.02), corner_hi=(0.08, 0.08, 0.08))
    mask = box.mask(grid)
    assert mask.shape == grid.shape
    assert mask.dtype == jnp.bool_
    # Should have non-zero interior
    assert float(mask.sum()) > 0
    # Corners should be outside
    assert not mask[0, 0, 0]


def test_sphere_mask(grid):
    """Sphere centered in domain."""
    sphere = Sphere(center=(0.05, 0.05, 0.05), radius=0.02)
    mask = sphere.mask(grid)
    # Center should be inside
    ci = grid.nx // 2
    cj = grid.ny // 2
    ck = grid.nz // 2
    assert mask[ci, cj, ck]


def test_union(grid):
    """Union should combine two shapes."""
    a = Box(corner_lo=(0.0, 0.0, 0.0), corner_hi=(0.04, 0.04, 0.04))
    b = Box(corner_lo=(0.06, 0.06, 0.06), corner_hi=(0.1, 0.1, 0.1))
    result = union(a, b, grid)
    assert float(result.sum()) >= float(a.mask(grid).sum())


def test_difference(grid):
    """Difference should subtract shape b from a."""
    a = Box(corner_lo=(0.0, 0.0, 0.0), corner_hi=(0.1, 0.1, 0.1))
    b = Box(corner_lo=(0.03, 0.03, 0.03), corner_hi=(0.07, 0.07, 0.07))
    result = difference(a, b, grid)
    assert float(result.sum()) < float(a.mask(grid).sum())


def test_rasterize(grid):
    """Rasterize should produce eps_r and sigma arrays."""
    box = Box(corner_lo=(0.02, 0.02, 0.02), corner_hi=(0.08, 0.08, 0.08))
    eps_r, sigma = rasterize(grid, [(box, 4.0, 0.0)])
    assert eps_r.shape == grid.shape
    # Interior of box should have eps_r = 4.0
    ci = grid.nx // 2
    assert float(eps_r[ci, ci, ci]) == 4.0
    # Outside should be background (1.0)
    assert float(eps_r[0, 0, 0]) == 1.0
