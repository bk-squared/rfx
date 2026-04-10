"""Kottke tensor-averaged subpixel smoothing at dielectric interfaces.

Implements the Kottke–Farjadpour–Johnson scheme (PRE 77, 036612, 2008):
at every Yee voxel that straddles a dielectric boundary the effective
inverse-permittivity *tensor* is

    (ε_eff)⁻¹ = Pₙ [f/ε₁ + (1−f)/ε₂]  +  Pₜ / [f·ε₁ + (1−f)·ε₂]

where  Pₙ = n̂·n̂ᵀ  is the projection onto the interface normal and
Pₜ = I − Pₙ  is the tangential projection.  The diagonal elements of
ε_eff are then extracted for each E-field component on the Yee grid.

Three design improvements over the original rfx linear-SDF scheme:

1. **Same-material union corner fix** — voxels fully inside the union
   of same-material shapes get bulk ε directly; no gradient needed.

2. **Analytic normals per shape** — Box (nearest face), Sphere
   (radial), Cylinder (radial + axial); eliminates the central-
   difference artefact at SDF union seams.

3. **Full Kottke tensor** — replaces the scalar cos²/sin² mixing
   formula with the proper 3×3 projection, giving second-order
   convergence for arbitrarily oriented interfaces.
"""

from __future__ import annotations

import jax.numpy as jnp

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

    ox = jnp.maximum(dx, 0.0)
    oy = jnp.maximum(dy, 0.0)
    oz = jnp.maximum(dz, 0.0)
    outside = jnp.sqrt(ox**2 + oy**2 + oz**2)
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
# Analytic normals per shape (outward-pointing)
# ---------------------------------------------------------------------------

def _normal_sphere(x, y, z, shape):
    """Analytic outward normal for a sphere: radial direction."""
    cx, cy, cz = shape.center
    dx = x - cx
    dy = y - cy
    dz = z - cz
    r = jnp.sqrt(dx**2 + dy**2 + dz**2 + 1e-30)
    return dx / r, dy / r, dz / r


def _normal_box(x, y, z, shape):
    """Analytic outward normal for an axis-aligned box.

    At each point the normal is determined by which face is nearest
    (the axis with the smallest penetration distance).
    """
    lo = jnp.array(shape.corner_lo)
    hi = jnp.array(shape.corner_hi)
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0

    # Signed distance to each face pair (positive = outside that pair)
    rx = jnp.abs(x - center[0]) - half[0]
    ry = jnp.abs(y - center[1]) - half[1]
    rz = jnp.abs(z - center[2]) - half[2]

    # The nearest face corresponds to the axis with the largest
    # (least-negative inside, or least-positive outside) component.
    ax = jnp.abs(rx)
    ay = jnp.abs(ry)
    az = jnp.abs(rz)

    # For interior points: nearest face = axis with *max* signed dist
    # For exterior points: same logic (largest component dominates SDF)
    max_r = jnp.maximum(jnp.maximum(rx, ry), rz)

    # Use soft selection: pick the axis closest to the surface.
    # For box interior, max_r < 0 and we want the axis where r is
    # closest to zero (largest / least negative).
    is_x = (rx >= ry) & (rx >= rz)
    is_y = (ry > rx) & (ry >= rz)
    # is_z = everything else

    sign_x = jnp.sign(x - center[0])
    sign_y = jnp.sign(y - center[1])
    sign_z = jnp.sign(z - center[2])

    # Default to z-face normal
    nx = jnp.where(is_x, sign_x, 0.0)
    ny = jnp.where(is_y, sign_y, jnp.where(is_x, 0.0, 0.0))
    nz = jnp.where(is_x | is_y, 0.0, sign_z)

    # Ensure unit length (should already be 1 for pure axis normals,
    # but guard against degenerate zero-sign cases)
    mag = jnp.sqrt(nx**2 + ny**2 + nz**2 + 1e-30)
    return nx / mag, ny / mag, nz / mag


def _normal_cylinder(x, y, z, shape):
    """Analytic outward normal for a cylinder.

    The radial component dominates on the curved surface; the axial
    component dominates on the caps.
    """
    cx, cy, cz = shape.center
    R = shape.radius
    H2 = shape.height / 2.0

    if shape.axis == "z":
        dx, dy = x - cx, y - cy
        r = jnp.sqrt(dx**2 + dy**2 + 1e-30)
        dh = jnp.abs(z - cz) - H2
        dr = r - R
        on_cap = dh > dr
        nx = jnp.where(on_cap, 0.0, dx / r)
        ny = jnp.where(on_cap, 0.0, dy / r)
        nz = jnp.where(on_cap, jnp.sign(z - cz), 0.0)
    elif shape.axis == "y":
        dx, dz = x - cx, z - cz
        r = jnp.sqrt(dx**2 + dz**2 + 1e-30)
        dh = jnp.abs(y - cy) - H2
        dr = r - R
        on_cap = dh > dr
        nx = jnp.where(on_cap, 0.0, dx / r)
        ny = jnp.where(on_cap, jnp.sign(y - cy), 0.0)
        nz = jnp.where(on_cap, 0.0, dz / r)
    else:  # axis == "x"
        dy, dz = y - cy, z - cz
        r = jnp.sqrt(dy**2 + dz**2 + 1e-30)
        dh = jnp.abs(x - cx) - H2
        dr = r - R
        on_cap = dh > dr
        nx = jnp.where(on_cap, jnp.sign(x - cx), 0.0)
        ny = jnp.where(on_cap, 0.0, dy / r)
        nz = jnp.where(on_cap, 0.0, dz / r)

    mag = jnp.sqrt(nx**2 + ny**2 + nz**2 + 1e-30)
    return nx / mag, ny / mag, nz / mag


def _get_normal_fn(shape: Shape):
    """Return the analytic normal function for a known shape type."""
    from rfx.geometry.csg import Box, Sphere, Cylinder
    if isinstance(shape, Sphere):
        return _normal_sphere
    elif isinstance(shape, Box):
        return _normal_box
    elif isinstance(shape, Cylinder):
        return _normal_cylinder
    return None


# ---------------------------------------------------------------------------
# Coordinate helpers for Yee-offset positions
# ---------------------------------------------------------------------------

def _yee_coords(grid: Grid):
    """Return cell-center coordinates (x, y, z) as 1-D arrays.

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
# Kottke tensor averaging
# ---------------------------------------------------------------------------

def _kottke_tensor_eps(
    f: jnp.ndarray,
    eps_inside: float,
    eps_outside: jnp.ndarray,
    nx: jnp.ndarray,
    ny: jnp.ndarray,
    nz: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the Kottke tensor-averaged ε and return its diagonals.

    The effective inverse-permittivity tensor is:

        (ε_eff)⁻¹ = Pₙ · [f/ε₁ + (1−f)/ε₂]  +  Pₜ · 1/[f·ε₁ + (1−f)·ε₂]

    where  Pₙ = n̂ n̂ᵀ,  Pₜ = I − Pₙ.

    We invert element-wise to get ε_eff and extract the diagonal for
    each E-field component.  Off-diagonal terms are discarded (standard
    practice for Yee FDTD; Meep does the same).

    Returns (eps_xx, eps_yy, eps_zz).
    """
    # Harmonic (perpendicular) inverse-eps
    inv_perp = f / eps_inside + (1.0 - f) / (eps_outside + 1e-30)

    # Arithmetic (parallel) eps, then invert
    eps_par = f * eps_inside + (1.0 - f) * eps_outside
    inv_par = 1.0 / (eps_par + 1e-30)

    # Projection tensor Pₙ = n̂ n̂ᵀ   (only need diagonal + cross terms)
    nxnx = nx * nx
    nyny = ny * ny
    nznz = nz * nz
    nxny = nx * ny
    nxnz = nx * nz
    nynz = ny * nz

    # (ε_eff)⁻¹ diagonal elements:
    #   inv_xx = nxnx * inv_perp + (1 - nxnx) * inv_par
    inv_xx = nxnx * inv_perp + (1.0 - nxnx) * inv_par
    inv_yy = nyny * inv_perp + (1.0 - nyny) * inv_par
    inv_zz = nznz * inv_perp + (1.0 - nznz) * inv_par

    # Invert to get ε_eff diagonals
    eps_xx = 1.0 / (inv_xx + 1e-30)
    eps_yy = 1.0 / (inv_yy + 1e-30)
    eps_zz = 1.0 / (inv_zz + 1e-30)

    return eps_xx, eps_yy, eps_zz


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_smoothed_eps(
    grid: Grid,
    shapes: list[tuple[Shape, float]],
    background_eps: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute Kottke tensor-averaged permittivity for each E-field component.

    At interface voxels, uses the full Kottke inverse-permittivity
    tensor with analytic interface normals.  Interior voxels get the
    bulk permittivity.

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

    # Group shapes by eps_r so overlapping same-material shapes use
    # union SDF (min of individual SDFs) instead of double-smoothing.
    from collections import OrderedDict
    groups: OrderedDict[float, list] = OrderedDict()
    for shape, eps_r in shapes:
        groups.setdefault(eps_r, []).append(shape)

    for eps_r, group_shapes in groups.items():
        sdf_shapes = []
        fallback_shapes = []
        for shape in group_shapes:
            sdf_fn = _get_sdf_fn(shape)
            normal_fn = _get_normal_fn(shape)
            if sdf_fn is not None and normal_fn is not None:
                sdf_shapes.append((shape, sdf_fn, normal_fn))
            else:
                fallback_shapes.append(shape)

        # Fallback shapes: staircased mask
        for shape in fallback_shapes:
            mask = shape.mask(grid)
            eps_ex = jnp.where(mask, eps_r, eps_ex)
            eps_ey = jnp.where(mask, eps_r, eps_ey)
            eps_ez = jnp.where(mask, eps_r, eps_ez)

        if not sdf_shapes:
            continue

        # --- For each E-component position, compute union SDF and
        #     select the best analytic normal from the nearest shape ---
        for comp, offset in [("ex", (half, 0., 0.)),
                             ("ey", (0., half, 0.)),
                             ("ez", (0., 0., half))]:
            Xc = X + offset[0]
            Yc = Y + offset[1]
            Zc = Z + offset[2]

            # Union SDF + per-voxel nearest-shape tracking
            sdf_union = None
            # For analytic normals we pick the shape whose SDF is
            # closest to zero (the one whose boundary is nearest).
            best_abs_sdf = None
            best_nx = None
            best_ny = None
            best_nz = None

            for shape, sdf_fn, normal_fn in sdf_shapes:
                s = sdf_fn(Xc, Yc, Zc, shape)
                n_x, n_y, n_z = normal_fn(Xc, Yc, Zc, shape)

                if sdf_union is None:
                    sdf_union = s
                    best_abs_sdf = jnp.abs(s)
                    best_nx = n_x
                    best_ny = n_y
                    best_nz = n_z
                else:
                    sdf_union = jnp.minimum(sdf_union, s)
                    # Update normal where this shape's surface is closer
                    closer = jnp.abs(s) < best_abs_sdf
                    best_abs_sdf = jnp.where(closer, jnp.abs(s), best_abs_sdf)
                    best_nx = jnp.where(closer, n_x, best_nx)
                    best_ny = jnp.where(closer, n_y, best_ny)
                    best_nz = jnp.where(closer, n_z, best_nz)

            # Fill fraction from union SDF
            f = jnp.clip(0.5 - sdf_union / dx, 0.0, 1.0)

            # Fix #1: fully-inside voxels get bulk eps, no smoothing
            inside = f >= 1.0
            # Boundary voxels
            bnd = (f > 0.0) & (f < 1.0)

            # The "outside" eps is whatever was there before
            if comp == "ex":
                eps_outside = eps_ex
            elif comp == "ey":
                eps_outside = eps_ey
            else:
                eps_outside = eps_ez

            # Fix #3: full Kottke tensor averaging
            kt_xx, kt_yy, kt_zz = _kottke_tensor_eps(
                f, eps_r, eps_outside,
                best_nx, best_ny, best_nz,
            )

            if comp == "ex":
                smooth = kt_xx
            elif comp == "ey":
                smooth = kt_yy
            else:
                smooth = kt_zz

            # Apply: interior → bulk, boundary → Kottke, exterior → unchanged
            if comp == "ex":
                eps_ex = jnp.where(inside, eps_r,
                         jnp.where(bnd, smooth, eps_ex))
            elif comp == "ey":
                eps_ey = jnp.where(inside, eps_r,
                         jnp.where(bnd, smooth, eps_ey))
            else:
                eps_ez = jnp.where(inside, eps_r,
                         jnp.where(bnd, smooth, eps_ez))

    return eps_ex, eps_ey, eps_ez
