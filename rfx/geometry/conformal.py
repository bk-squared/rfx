"""Dey-Mittra conformal FDTD for PEC boundaries.

When a PEC surface cuts through a Yee cell at an angle, standard stairstepping
gives first-order convergence. The Dey-Mittra method computes fractional cell
areas and scales the E-field update accordingly, restoring second-order accuracy.

Reference: Dey & Mittra, IEEE MGWL 7(9), 273-275, 1997.

For each E-field component, we compute a weight in [0, 1]:
  - 1.0: cell face fully outside PEC (normal update)
  - 0.0: cell face fully inside PEC (E forced to zero)
  - 0 < w < 1: cell face partially cut by PEC surface

The weight is applied as: E_new = w * (Ca * E + Cb * curl_H)
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid


def _signed_distance_sphere(x, y, z, center, radius):
    """Signed distance from point to sphere surface. Negative inside."""
    dx = x - center[0]
    dy = y - center[1]
    dz = z - center[2]
    return np.sqrt(dx**2 + dy**2 + dz**2) - radius


def _signed_distance_cylinder(x, y, z, center, radius, height, axis="z"):
    """Signed distance from point to cylinder. Negative inside."""
    if axis == "z":
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        d_radial = r - radius
        d_axial = np.maximum(abs(z - center[2]) - height / 2, 0)
    elif axis == "y":
        r = np.sqrt((x - center[0])**2 + (z - center[2])**2)
        d_radial = r - radius
        d_axial = np.maximum(abs(y - center[1]) - height / 2, 0)
    else:
        r = np.sqrt((y - center[1])**2 + (z - center[2])**2)
        d_radial = r - radius
        d_axial = np.maximum(abs(x - center[0]) - height / 2, 0)
    return np.where(d_radial > 0, np.sqrt(d_radial**2 + d_axial**2),
                    np.maximum(d_radial, d_axial))


def compute_conformal_weights(
    grid: Grid,
    pec_shapes: list,
    n_sub: int = 4,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute Dey-Mittra conformal weights for E-field components.

    Parameters
    ----------
    grid : Grid
    pec_shapes : list of shape objects (must have .mask(grid) method)
    n_sub : int
        Sub-sampling per cell face for fractional area computation.

    Returns
    -------
    (w_ex, w_ey, w_ez) : each shape grid.shape, values in [0, 1]
        1.0 = fully outside PEC, 0.0 = fully inside PEC.
    """
    dx = grid.dx
    nx, ny, nz = grid.shape

    # Compute PEC mask at sub-cell resolution for each E-component face
    # Ex at (i+0.5, j, k): faces in y-z plane
    # Ey at (i, j+0.5, k): faces in x-z plane
    # Ez at (i, j, k+0.5): faces in x-y plane

    w_ex = np.ones((nx, ny, nz), dtype=np.float32)
    w_ey = np.ones((nx, ny, nz), dtype=np.float32)
    w_ez = np.ones((nx, ny, nz), dtype=np.float32)

    if not pec_shapes:
        return jnp.array(w_ex), jnp.array(w_ey), jnp.array(w_ez)

    # Get combined PEC mask
    combined_mask = np.zeros((nx, ny, nz), dtype=bool)
    for shape in pec_shapes:
        combined_mask |= np.array(shape.mask(grid))

    # Build a point-test function from all PEC shapes
    # For sub-cell sampling we need continuous geometry, not voxelized mask
    def _point_inside_pec(px, py, pz):
        """Check if physical point (px, py, pz) is inside any PEC shape."""
        # Convert to grid index (fractional) and test each shape
        px / dx
        py / dx
        pz / dx
        for shape in pec_shapes:
            # Use shape's mask at fractional position via SDF or bounds check
            if hasattr(shape, 'center') and hasattr(shape, 'radius'):
                # Sphere
                c = shape.center
                r = shape.radius
                if (px - c[0])**2 + (py - c[1])**2 + (pz - c[2])**2 <= r**2:
                    return True
            elif hasattr(shape, 'corner_lo') and hasattr(shape, 'corner_hi'):
                # Box
                lo, hi = shape.corner_lo, shape.corner_hi
                if lo[0] <= px <= hi[0] and lo[1] <= py <= hi[1] and lo[2] <= pz <= hi[2]:
                    return True
            elif hasattr(shape, 'axis') and hasattr(shape, 'radius'):
                # Cylinder — simplified check
                c = shape.center
                r = shape.radius
                h = shape.height
                ax = getattr(shape, 'axis', 'z')
                if ax == 'z':
                    if (px-c[0])**2 + (py-c[1])**2 <= r**2 and abs(pz-c[2]) <= h/2:
                        return True
        return False

    # For cells near PEC boundary, compute fractional area via sub-sampling
    from scipy.ndimage import binary_dilation
    boundary = binary_dilation(combined_mask, iterations=1) & ~combined_mask
    boundary |= combined_mask & binary_dilation(~combined_mask, iterations=1)

    # Sub-sample offsets within a cell [0, 1)
    sub = np.linspace(0.5 / n_sub, 1.0 - 0.5 / n_sub, n_sub)

    # Pad offsets for each axis
    pad = np.array(grid.axis_pads) if hasattr(grid, 'axis_pads') else np.zeros(3)

    boundary_indices = np.argwhere(boundary)

    for idx in boundary_indices:
        i, j, k = idx

        # Ex face at (i+0.5, j, k): sub-sample in y-z plane
        n_inside = 0
        px = (i + 0.5 - pad[0]) * dx
        for sy in sub:
            for sz in sub:
                py = (j + sy - pad[1]) * dx
                pz = (k + sz - pad[2]) * dx
                if _point_inside_pec(px, py, pz):
                    n_inside += 1
        w_ex[i, j, k] = 1.0 - n_inside / (n_sub * n_sub)

        # Ey face at (i, j+0.5, k): sub-sample in x-z plane
        n_inside = 0
        py = (j + 0.5 - pad[1]) * dx
        for sx in sub:
            for sz in sub:
                px = (i + sx - pad[0]) * dx
                pz = (k + sz - pad[2]) * dx
                if _point_inside_pec(px, py, pz):
                    n_inside += 1
        w_ey[i, j, k] = 1.0 - n_inside / (n_sub * n_sub)

        # Ez face at (i, j, k+0.5): sub-sample in x-y plane
        n_inside = 0
        pz = (k + 0.5 - pad[2]) * dx
        for sx in sub:
            for sy in sub:
                px = (i + sx - pad[0]) * dx
                py = (j + sy - pad[1]) * dx
                if _point_inside_pec(px, py, pz):
                    n_inside += 1
        w_ez[i, j, k] = 1.0 - n_inside / (n_sub * n_sub)

    # Interior PEC cells: weight = 0
    w_ex[combined_mask] = 0.0
    w_ey[combined_mask] = 0.0
    w_ez[combined_mask] = 0.0

    return jnp.array(w_ex), jnp.array(w_ey), jnp.array(w_ez)


def apply_conformal_pec(state, w_ex, w_ey, w_ez):
    """Apply conformal PEC: zero E inside PEC, leave partial cells to update coefficients.

    For cells with w=0 (fully inside PEC): force E=0.
    For cells with 0<w<1 (partial PEC): do NOT multiply E by w every step.
    The Dey-Mittra correction for partial cells should be applied by modifying
    the material eps_r via `conformal_eps_correction()` once at setup time.

    Call AFTER update_e each timestep.
    """
    # Only zero out fully-PEC cells (w == 0), not partial cells
    pec_mask_ex = (w_ex == 0.0)
    pec_mask_ey = (w_ey == 0.0)
    pec_mask_ez = (w_ez == 0.0)
    return state._replace(
        ex=jnp.where(pec_mask_ex, 0.0, state.ex),
        ey=jnp.where(pec_mask_ey, 0.0, state.ey),
        ez=jnp.where(pec_mask_ez, 0.0, state.ez),
    )


def conformal_eps_correction(eps_r, w_ex, w_ey, w_ez):
    """Modify eps_r to account for Dey-Mittra conformal PEC.

    For partial PEC cells (0 < w < 1), the effective permittivity is
    scaled by 1/w to account for the reduced cell area. This is applied
    ONCE at setup, not every timestep.

    Returns (eps_ex, eps_ey, eps_ez) per-component arrays.
    """
    safe_wx = jnp.where(w_ex > 0, w_ex, 1.0)
    safe_wy = jnp.where(w_ey > 0, w_ey, 1.0)
    safe_wz = jnp.where(w_ez > 0, w_ez, 1.0)
    return eps_r / safe_wx, eps_r / safe_wy, eps_r / safe_wz
