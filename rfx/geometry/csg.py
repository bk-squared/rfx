"""Constructive Solid Geometry (CSG) primitives.

Defines geometric shapes and boolean operations that produce material arrays
from geometric descriptions. All shapes operate on grid coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid


class Shape(Protocol):
    """Protocol for CSG shapes — must implement mask and mask_on_coords."""

    def mask(self, grid: Grid) -> jnp.ndarray:
        """Return boolean mask (True inside shape) on the given grid."""
        ...

    def mask_on_coords(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        z: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate shape occupancy on explicit 1D coordinate arrays.

        Parameters
        ----------
        x, y, z : 1D arrays of physical coordinates (metres)

        Returns
        -------
        (Nx, Ny, Nz) boolean array — True inside the shape.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement mask_on_coords(). "
            f"This shape cannot be used on nonuniform or subgridded meshes."
        )

    def bounding_box(self) -> tuple[tuple[float, float, float],
                                     tuple[float, float, float]]:
        """Return (corner_lo, corner_hi) axis-aligned bounding box."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement bounding_box()."
        )


def _grid_coords(grid: Grid):
    """Extract 1D physical coordinate arrays from a uniform Grid."""
    nx, ny, nz = grid.shape
    dx = grid.dx
    pad_x, pad_y, pad_z = grid.axis_pads
    x = (jnp.arange(nx) - pad_x) * dx
    y = (jnp.arange(ny) - pad_y) * dx
    z = (jnp.arange(nz) - pad_z) * dx
    return x, y, z


@dataclass(frozen=True)
class Box:
    """Axis-aligned box defined by two corners (meters)."""

    corner_lo: tuple[float, float, float]
    corner_hi: tuple[float, float, float]

    def bounding_box(self):
        return (self.corner_lo, self.corner_hi)

    def mask_on_coords(self, x, y, z):
        def _axis_mask(coords, lo, hi):
            # Use per-axis cell size (critical for non-uniform z mesh)
            dc = float(coords[1] - coords[0]) if len(coords) > 1 else 1e-3
            extent = hi - lo
            if extent <= dc * 1.01:
                # Thin geometry: snap to nearest cell center
                mid = (lo + hi) * 0.5
                return jnp.abs(coords - mid) < dc * 0.51
            else:
                return (coords >= lo) & (coords <= hi)

        mx = _axis_mask(x, self.corner_lo[0], self.corner_hi[0])
        my = _axis_mask(y, self.corner_lo[1], self.corner_hi[1])
        mz = _axis_mask(z, self.corner_lo[2], self.corner_hi[2])
        return mx[:, None, None] & my[None, :, None] & mz[None, None, :]

    def mask(self, grid: Grid) -> jnp.ndarray:
        x, y, z = _grid_coords(grid)
        return self.mask_on_coords(x, y, z)


@dataclass(frozen=True)
class Cylinder:
    """Cylinder along a given axis."""

    center: tuple[float, float, float]
    radius: float
    height: float
    axis: str = "z"  # "x", "y", or "z"

    def bounding_box(self):
        r = self.radius
        h = self.height / 2
        cx, cy, cz = self.center
        if self.axis == "z":
            return ((cx - r, cy - r, cz - h), (cx + r, cy + r, cz + h))
        elif self.axis == "y":
            return ((cx - r, cy - h, cz - r), (cx + r, cy + h, cz + r))
        else:
            return ((cx - h, cy - r, cz - r), (cx + h, cy + r, cz + r))

    def mask_on_coords(self, x, y, z):
        xc = x - self.center[0]
        yc = y - self.center[1]
        zc = z - self.center[2]

        x3 = xc[:, None, None]
        y3 = yc[None, :, None]
        z3 = zc[None, None, :]

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

    def mask(self, grid: Grid) -> jnp.ndarray:
        x, y, z = _grid_coords(grid)
        return self.mask_on_coords(x, y, z)


@dataclass(frozen=True)
class Sphere:
    """Sphere defined by center and radius."""

    center: tuple[float, float, float]
    radius: float

    def bounding_box(self):
        r = self.radius
        cx, cy, cz = self.center
        return ((cx - r, cy - r, cz - r), (cx + r, cy + r, cz + r))

    def mask_on_coords(self, x, y, z):
        xc = x - self.center[0]
        yc = y - self.center[1]
        zc = z - self.center[2]
        r2 = xc[:, None, None]**2 + yc[None, :, None]**2 + zc[None, None, :]**2
        return r2 <= self.radius**2

    def mask(self, grid: Grid) -> jnp.ndarray:
        x, y, z = _grid_coords(grid)
        return self.mask_on_coords(x, y, z)


@dataclass(frozen=True)
class PolylineWire:
    """Wire defined by a polyline path with constant circular cross-section.

    Voxelizes by computing the distance from each grid point to the nearest
    line segment of the polyline.  Grid points within ``radius`` of any
    segment are marked as inside.

    Parameters
    ----------
    points : tuple of tuple[float, float, float]
        Ordered vertices in metres, e.g. ((x0,y0,z0), (x1,y1,z1), ...).
    radius : float
        Wire radius in metres.
    """

    points: tuple[tuple[float, float, float], ...]
    radius: float

    def bounding_box(self):
        pts = np.array(self.points)
        lo = tuple(float(v) for v in pts.min(axis=0) - self.radius)
        hi = tuple(float(v) for v in pts.max(axis=0) + self.radius)
        return (lo, hi)

    def mask_on_coords(self, x, y, z):
        r2_thresh = self.radius ** 2
        pts = np.array(self.points, dtype=np.float64)
        n_seg = len(pts) - 1

        # 3D coordinate grids (Nx, Ny, Nz)
        X = x[:, None, None]
        Y = y[None, :, None]
        Z = z[None, None, :]

        mask = jnp.zeros((len(x), len(y), len(z)), dtype=jnp.bool_)

        for i in range(n_seg):
            # Segment from A to B
            ax, ay, az = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
            bx, by, bz = float(pts[i + 1, 0]), float(pts[i + 1, 1]), float(pts[i + 1, 2])

            dx_s = bx - ax
            dy_s = by - ay
            dz_s = bz - az
            seg_len2 = dx_s ** 2 + dy_s ** 2 + dz_s ** 2

            if seg_len2 < 1e-30:
                continue

            # Parameter t: projection of (P-A) onto (B-A), clamped to [0,1]
            t = ((X - ax) * dx_s + (Y - ay) * dy_s + (Z - az) * dz_s) / seg_len2
            t = jnp.clip(t, 0.0, 1.0)

            # Closest point on segment
            cx = ax + t * dx_s
            cy = ay + t * dy_s
            cz = az + t * dz_s

            dist2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
            mask = mask | (dist2 <= r2_thresh)

        return mask

    def mask(self, grid: Grid) -> jnp.ndarray:
        x, y, z = _grid_coords(grid)
        return self.mask_on_coords(x, y, z)


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
