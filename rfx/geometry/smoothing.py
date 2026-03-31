"""Anisotropic subpixel smoothing at dielectric interfaces.

Implements the Meep-style averaging scheme: at Yee voxels that straddle a
dielectric boundary, the effective permittivity depends on the interface
orientation relative to each E-field component.

- **Parallel** to the interface: arithmetic mean  eps_par = f*eps1 + (1-f)*eps2
- **Perpendicular** to the interface: harmonic mean  eps_perp = [f/eps1 + (1-f)/eps2]^{-1}

This gives second-order convergence instead of the first-order stairstepping.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rfx.grid import Grid
from rfx.geometry.csg import Shape


# ---------------------------------------------------------------------------
# Signed distance functions for supported shapes
# ---------------------------------------------------------------------------

def _sdf_sphere(x, y, z, shape) -> jnp.ndarray:
    """Signed distance: negative inside, positive outside."""
    cx, cy, cz = shape.center
    r = jnp.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
    return r - shape.radius


def _sdf_box(x, y, z, shape) -> jnp.ndarray:
    """Signed distance for axis-aligned box."""
    lo = jnp.array(shape.corner_lo)
    hi = jnp.array(shape.corner_hi)
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0

    dx = jnp.abs(x - center[0]) - half[0]
    dy = jnp.abs(y - center[1]) - half[1]
    dz = jnp.abs(z - center[2]) - half[2]

    # Outside distance
    ox = jnp.maximum(dx, 0.0)
    oy = jnp.maximum(dy, 0.0)
    oz = jnp.maximum(dz, 0.0)
    outside = jnp.sqrt(ox**2 + oy**2 + oz**2)

    # Inside distance (negative)
    inside = jnp.minimum(jnp.maximum(jnp.maximum(dx, dy), dz), 0.0)

    return outside + inside


def _sdf_cylinder(x, y, z, shape) -> jnp.ndarray:
    """Signed distance for a cylinder along a given axis."""
    cx, cy, cz = shape.center

    if shape.axis == "z":
        r = jnp.sqrt((x - cx)**2 + (y - cy)**2) - shape.radius
        h = jnp.abs(z - cz) - shape.height / 2.0
    elif shape.axis == "y":
        r = jnp.sqrt((x - cx)**2 + (z - cz)**2) - shape.radius
        h = jnp.abs(y - cy) - shape.height / 2.0
    else:
        r = jnp.sqrt((y - cy)**2 + (z - cz)**2) - shape.radius
        h = jnp.abs(x - cx) - shape.height / 2.0

    outside = jnp.sqrt(jnp.maximum(r, 0.0)**2 + jnp.maximum(h, 0.0)**2)
    inside = jnp.minimum(jnp.maximum(r, h), 0.0)
    return outside + inside


def _get_sdf_fn(shape: Shape):
    """Return the SDF function for a known shape type, or None."""
    from rfx.geometry.csg import Box, Sphere, Cylinder
    if isinstance(shape, Sphere):
        return _sdf_sphere
    elif isinstance(shape, Box):
        return _sdf_box
    elif isinstance(shape, Cylinder):
        return _sdf_cylinder
    return None


# ---------------------------------------------------------------------------
# Coordinate helpers for Yee-offset positions
# ---------------------------------------------------------------------------

def _yee_coords(grid: Grid):
    """Return cell-center coordinates (x, y, z) as 3D arrays.

    On the Yee grid:
    - Ex lives at (i+0.5, j, k)
    - Ey lives at (i, j+0.5, k)
    - Ez lives at (i, j, k+0.5)

    We return the integer-grid positions; the caller applies +0.5*dx offsets.
    """
    nx, ny, nz = grid.shape
    dx = grid.dx
    pad_x, pad_y, pad_z = grid.axis_pads

    x = (jnp.arange(nx) - pad_x) * dx
    y = (jnp.arange(ny) - pad_y) * dx
    z = (jnp.arange(nz) - pad_z) * dx

    return x, y, z


# ---------------------------------------------------------------------------
# Core smoothing
# ---------------------------------------------------------------------------

def _compute_fill_fraction_and_normal(
    sdf_vals: jnp.ndarray,
    dx: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """From SDF values, compute fill fraction and interface normal.

    The fill fraction is estimated from the signed distance: for a voxel
    of size dx, if the SDF value at the center is d, the fraction inside
    is approximately clamp(0.5 - d/dx, 0, 1).

    Returns
    -------
    f : filling fraction (fraction inside the shape)
    is_boundary : boolean mask of boundary voxels
    nx, ny, nz : interface normal components (pointing outward)
    """
    # Fill fraction: linear approximation from SDF
    f = jnp.clip(0.5 - sdf_vals / dx, 0.0, 1.0)

    # Gradient of SDF gives the outward normal
    # Use central differences
    grad_x = jnp.gradient(sdf_vals, dx, axis=0)
    grad_y = jnp.gradient(sdf_vals, dx, axis=1)
    grad_z = jnp.gradient(sdf_vals, dx, axis=2)

    grad_mag = jnp.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-30)
    nx = grad_x / grad_mag
    ny = grad_y / grad_mag
    nz = grad_z / grad_mag

    # Boundary voxels: where 0 < f < 1
    is_boundary = (f > 0.0) & (f < 1.0)

    return f, is_boundary, nx, ny, nz


def _anisotropic_eps(
    f: jnp.ndarray,
    eps_inside: float,
    eps_outside: float,
    n_component: jnp.ndarray,
) -> jnp.ndarray:
    """Compute effective eps for one E-field component at boundary voxels.

    For an E-field component with unit vector e_hat, and interface normal n_hat:
    - The component of e_hat perpendicular to the interface: |e_hat . n_hat|
    - The component parallel to the interface: sqrt(1 - (e_hat . n_hat)^2)

    For e_hat = x_hat, |e_hat . n_hat| = |nx|, so:
    eps_eff = cos^2(theta) * eps_parallel + sin^2(theta) * eps_perp

    where theta is the angle between e_hat and the interface plane,
    eps_parallel = f*eps1 + (1-f)*eps2  (arithmetic)
    eps_perp = [f/eps1 + (1-f)/eps2]^{-1}  (harmonic)
    and sin^2(theta) = n_component^2 (perpendicular fraction).

    Parameters
    ----------
    f : filling fraction (fraction inside)
    eps_inside, eps_outside : permittivity values
    n_component : |n_hat . e_hat| for this E-field component
    """
    # Arithmetic (parallel) mean
    eps_par = f * eps_inside + (1.0 - f) * eps_outside

    # Harmonic (perpendicular) mean
    eps_perp = 1.0 / (f / eps_inside + (1.0 - f) / eps_outside + 1e-30)

    # Weight by how much the E-field aligns with the normal
    sin2 = n_component**2  # perpendicular fraction
    cos2 = 1.0 - sin2      # parallel fraction

    return cos2 * eps_par + sin2 * eps_perp


def compute_smoothed_eps(
    grid: Grid,
    shapes: list[tuple[Shape, float]],
    background_eps: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute smoothed permittivity arrays for each E-field component.

    At interface voxels, uses anisotropic arithmetic/harmonic averaging
    based on the local interface normal orientation. Interior voxels get
    the bulk permittivity value.

    Parameters
    ----------
    grid : Grid
    shapes : list of (shape, eps_r) pairs
        Applied in order; later shapes overwrite earlier ones.
    background_eps : float
        Background relative permittivity.

    Returns
    -------
    eps_ex, eps_ey, eps_ez : jnp.ndarray
        Per-component permittivity arrays, each of shape grid.shape.
    """
    x, y, z = _yee_coords(grid)
    dx = grid.dx
    half = dx * 0.5

    # Integer-grid 3D coordinate arrays
    X = x[:, None, None] * jnp.ones((1, len(y), 1))
    Y = jnp.ones((len(x), 1, 1)) * y[None, :, None] * jnp.ones((1, 1, len(z)))
    Z = jnp.ones((len(x), len(y), 1)) * z[None, None, :]

    # Start with background eps for all three components
    eps_ex = jnp.full(grid.shape, background_eps, dtype=jnp.float32)
    eps_ey = jnp.full(grid.shape, background_eps, dtype=jnp.float32)
    eps_ez = jnp.full(grid.shape, background_eps, dtype=jnp.float32)

    # Also track the scalar (staircased) eps for interior assignment
    eps_scalar = jnp.full(grid.shape, background_eps, dtype=jnp.float32)

    for shape, eps_r in shapes:
        sdf_fn = _get_sdf_fn(shape)

        if sdf_fn is None:
            # Fallback: no SDF available, use staircased mask
            mask = shape.mask(grid)
            eps_ex = jnp.where(mask, eps_r, eps_ex)
            eps_ey = jnp.where(mask, eps_r, eps_ey)
            eps_ez = jnp.where(mask, eps_r, eps_ez)
            eps_scalar = jnp.where(mask, eps_r, eps_scalar)
            continue

        # For each E-component, evaluate SDF at the Yee-offset position
        # Ex at (i+0.5, j, k)
        sdf_ex = sdf_fn(X + half, Y, Z, shape)
        f_ex, bnd_ex, nx_ex, ny_ex, nz_ex = _compute_fill_fraction_and_normal(sdf_ex, dx)

        # Ey at (i, j+0.5, k)
        sdf_ey = sdf_fn(X, Y + half, Z, shape)
        f_ey, bnd_ey, nx_ey, ny_ey, nz_ey = _compute_fill_fraction_and_normal(sdf_ey, dx)

        # Ez at (i, j, k+0.5)
        sdf_ez = sdf_fn(X, Y, Z + half, shape)
        f_ez, bnd_ez, nx_ez, ny_ez, nz_ez = _compute_fill_fraction_and_normal(sdf_ez, dx)

        # The "outside" eps is whatever was there before (could be another shape)
        eps_outside_ex = eps_ex
        eps_outside_ey = eps_ey
        eps_outside_ez = eps_ez

        # For Ex: e_hat = x_hat, so n_component = |nx|
        smooth_ex = _anisotropic_eps(f_ex, eps_r, eps_outside_ex, jnp.abs(nx_ex))
        # For Ey: e_hat = y_hat, so n_component = |ny|
        smooth_ey = _anisotropic_eps(f_ey, eps_r, eps_outside_ey, jnp.abs(ny_ey))
        # For Ez: e_hat = z_hat, so n_component = |nz|
        smooth_ez = _anisotropic_eps(f_ez, eps_r, eps_outside_ez, jnp.abs(nz_ez))

        # Interior voxels (fully inside): just use eps_r
        inside_ex = f_ex >= 1.0
        inside_ey = f_ey >= 1.0
        inside_ez = f_ez >= 1.0

        # Apply: boundary voxels get smoothed, interior get bulk, exterior unchanged
        eps_ex = jnp.where(inside_ex, eps_r, jnp.where(bnd_ex, smooth_ex, eps_ex))
        eps_ey = jnp.where(inside_ey, eps_r, jnp.where(bnd_ey, smooth_ey, eps_ey))
        eps_ez = jnp.where(inside_ez, eps_r, jnp.where(bnd_ez, smooth_ez, eps_ez))

    return eps_ex, eps_ey, eps_ez
