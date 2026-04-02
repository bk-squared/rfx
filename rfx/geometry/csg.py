"""Constructive Solid Geometry (CSG) primitives.

Defines geometric shapes and boolean operations that produce material arrays
from geometric descriptions. All shapes operate on grid coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax.numpy as jnp
import numpy as np

from rfx.grid import Grid


class Shape(Protocol):
    """Protocol for CSG shapes — must implement a signed distance or mask."""

    def mask(self, grid: Grid) -> jnp.ndarray:
        """Return boolean mask (True inside shape) on the given grid."""
        ...


@dataclass(frozen=True)
class Box:
    """Axis-aligned box defined by two corners (meters)."""

    corner_lo: tuple[float, float, float]
    corner_hi: tuple[float, float, float]

    def mask(self, grid: Grid) -> jnp.ndarray:
        nx, ny, nz = grid.shape
        dx = grid.dx
        pad_x, pad_y, pad_z = grid.axis_pads

        ix = jnp.arange(nx) - pad_x
        iy = jnp.arange(ny) - pad_y
        iz = jnp.arange(nz) - pad_z

        x = ix * dx
        y = iy * dx
        z = iz * dx

        # For each axis: if extent <= dx (thin feature), snap to single
        # cell nearest to corner_lo. Otherwise, standard containment.
        def _axis_mask(coords, lo, hi):
            extent = hi - lo
            if extent <= dx * 1.01:
                # Thin feature (≤ 1 cell): snap to cell nearest lo
                return jnp.abs(coords - lo) < dx * 0.51
            else:
                # Thick feature: standard [lo, hi) containment
                return (coords >= lo) & (coords < hi)

        mx = _axis_mask(x, self.corner_lo[0], self.corner_hi[0])
        my = _axis_mask(y, self.corner_lo[1], self.corner_hi[1])
        mz = _axis_mask(z, self.corner_lo[2], self.corner_hi[2])

        return mx[:, None, None] & my[None, :, None] & mz[None, None, :]


@dataclass(frozen=True)
class Cylinder:
    """Cylinder along a given axis."""

    center: tuple[float, float, float]
    radius: float
    height: float
    axis: str = "z"  # "x", "y", or "z"

    def mask(self, grid: Grid) -> jnp.ndarray:
        nx, ny, nz = grid.shape
        dx = grid.dx
        pad_x, pad_y, pad_z = grid.axis_pads

        x = (jnp.arange(nx) - pad_x) * dx - self.center[0]
        y = (jnp.arange(ny) - pad_y) * dx - self.center[1]
        z = (jnp.arange(nz) - pad_z) * dx - self.center[2]

        x3 = x[:, None, None]
        y3 = y[None, :, None]
        z3 = z[None, None, :]

        if self.axis == "z":
            r2 = x3**2 + y3**2
            h = z3
        elif self.axis == "y":
            r2 = x3**2 + z3**2
            h = y3
        else:
            r2 = y3**2 + z3**2
            h = x3

        return (r2 <= self.radius**2) & (jnp.abs(h) <= self.height / 2)


@dataclass(frozen=True)
class Sphere:
    """Sphere defined by center and radius."""

    center: tuple[float, float, float]
    radius: float

    def mask(self, grid: Grid) -> jnp.ndarray:
        nx, ny, nz = grid.shape
        dx = grid.dx
        pad_x, pad_y, pad_z = grid.axis_pads

        x = (jnp.arange(nx) - pad_x) * dx - self.center[0]
        y = (jnp.arange(ny) - pad_y) * dx - self.center[1]
        z = (jnp.arange(nz) - pad_z) * dx - self.center[2]

        r2 = x[:, None, None]**2 + y[None, :, None]**2 + z[None, None, :]**2
        return r2 <= self.radius**2


def union(a: Shape, b: Shape, grid: Grid) -> jnp.ndarray:
    return a.mask(grid) | b.mask(grid)


def difference(a: Shape, b: Shape, grid: Grid) -> jnp.ndarray:
    return a.mask(grid) & ~b.mask(grid)


def intersection(a: Shape, b: Shape, grid: Grid) -> jnp.ndarray:
    return a.mask(grid) & b.mask(grid)


def rasterize(
    grid: Grid,
    shapes: list[tuple[Shape, float, float]],
    background_eps: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Rasterize shapes onto grid, producing (eps_r, sigma) arrays.

    Parameters
    ----------
    grid : Grid
    shapes : list of (Shape, eps_r, sigma) tuples
        Applied in order; later shapes overwrite earlier ones.
    background_eps : float
        Background relative permittivity.

    Returns
    -------
    eps_r, sigma : jnp.ndarray
    """
    eps_r = jnp.full(grid.shape, background_eps, dtype=jnp.float32)
    sigma = jnp.zeros(grid.shape, dtype=jnp.float32)

    for shape, er, sig in shapes:
        m = shape.mask(grid)
        eps_r = jnp.where(m, er, eps_r)
        sigma = jnp.where(m, sig, sigma)

    return eps_r, sigma
