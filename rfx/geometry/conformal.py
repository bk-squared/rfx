"""Dey-Mittra conformal FDTD for PEC boundaries.

When a PEC surface cuts through a Yee cell at an angle, standard stairstepping
gives first-order convergence. The Dey-Mittra method computes fractional cell
areas and scales the E-field update accordingly, restoring second-order accuracy.

Reference: Dey & Mittra, IEEE MGWL 7(9), 273-275, 1997.

For each E-field component, we compute a weight in [0, 1]:
  - 1.0: cell face fully outside PEC (normal update)
  - 0.0: cell face fully inside PEC (E forced to zero)
  - 0 < w < 1: cell face partially cut by PEC surface

The conformal correction is applied via the effective permittivity:
  eps_eff = eps_r / w_clamped

This is equivalent to scaling Cb by 1/w, as described by Dey-Mittra.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.geometry.smoothing import _sdf_sphere, _sdf_box, _sdf_cylinder, _get_sdf_fn, _yee_coords


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


# ---------------------------------------------------------------------------
# Vectorized SDF-based conformal weight computation (Phase 1)
# ---------------------------------------------------------------------------

def compute_conformal_weights_sdf(
    grid: Grid,
    pec_shapes: list,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Vectorized conformal weight computation using SDF.

    For each E-field component (Ex, Ey, Ez), compute the area fraction
    of the corresponding cell face that is outside PEC.

    w = 1.0: fully free (normal update)
    w = 0.0: fully PEC (E forced to zero)
    0 < w < 1: partial PEC (conformal Cb = Cb_standard / w)

    Uses SDF: w = clip(0.5 + sdf/dx, 0, 1) evaluated at Yee-offset positions.
    For multiple PEC shapes, weights are combined via element-wise minimum
    (union of PEC regions).

    Parameters
    ----------
    grid : Grid
    pec_shapes : list of Shape objects with known SDF (Box, Sphere, Cylinder).
        Shapes without SDF fall back to mask-based binary weights.

    Returns
    -------
    (w_ex, w_ey, w_ez) : each shape grid.shape, values in [0, 1]
        1.0 = fully outside PEC, 0.0 = fully inside PEC.
    """
    nx, ny, nz = grid.shape
    dx = grid.dx
    half = dx * 0.5

    if not pec_shapes:
        ones = jnp.ones((nx, ny, nz), dtype=jnp.float32)
        return ones, ones, ones

    # Build Yee-offset 3D coordinate arrays
    x, y, z = _yee_coords(grid)
    X = x[:, None, None] * jnp.ones((1, len(y), 1))
    Y = jnp.ones((len(x), 1, 1)) * y[None, :, None] * jnp.ones((1, 1, len(z)))
    Z = jnp.ones((len(x), len(y), 1)) * z[None, None, :]

    # Start with all free space
    w_ex = jnp.ones((nx, ny, nz), dtype=jnp.float32)
    w_ey = jnp.ones((nx, ny, nz), dtype=jnp.float32)
    w_ez = jnp.ones((nx, ny, nz), dtype=jnp.float32)

    for shape in pec_shapes:
        sdf_fn = _get_sdf_fn(shape)

        if sdf_fn is not None:
            # Vectorized SDF evaluation at Yee-offset positions
            # SDF convention: negative inside shape, positive outside
            # Weight = fraction outside PEC = clip(0.5 + sdf/dx, 0, 1)
            sdf_ex = sdf_fn(X + half, Y, Z, shape)  # Ex at (i+0.5, j, k)
            sdf_ey = sdf_fn(X, Y + half, Z, shape)  # Ey at (i, j+0.5, k)
            sdf_ez = sdf_fn(X, Y, Z + half, shape)  # Ez at (i, j, k+0.5)

            w_ex_shape = jnp.clip(0.5 + sdf_ex / dx, 0.0, 1.0)
            w_ey_shape = jnp.clip(0.5 + sdf_ey / dx, 0.0, 1.0)
            w_ez_shape = jnp.clip(0.5 + sdf_ez / dx, 0.0, 1.0)
        else:
            # Fallback: binary weights from mask (staircase for unknown shapes)
            mask = shape.mask(grid).astype(jnp.float32)
            w_ex_shape = 1.0 - mask
            w_ey_shape = 1.0 - mask
            w_ez_shape = 1.0 - mask

        # Union of PEC: take minimum weight (most PEC)
        w_ex = jnp.minimum(w_ex, w_ex_shape)
        w_ey = jnp.minimum(w_ey, w_ey_shape)
        w_ez = jnp.minimum(w_ez, w_ez_shape)

    return w_ex, w_ey, w_ez


def clamp_conformal_weights(
    w_ex: jnp.ndarray,
    w_ey: jnp.ndarray,
    w_ez: jnp.ndarray,
    w_min: float = 0.1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Clamp small conformal weights for CFL stability.

    Very small weights (w -> 0+) amplify the effective Cb by 1/w,
    which can violate the CFL stability condition. Following Dey-Mittra
    (1997), weights are clamped to a minimum threshold.

    w_clamped = where(w > 0, max(w, w_min), 0)

    Cells with w=0 (fully inside PEC) remain at zero.
    Cells with w=1 (fully outside PEC) remain at one.

    Parameters
    ----------
    w_ex, w_ey, w_ez : jnp.ndarray
        Conformal weights in [0, 1].
    w_min : float
        Minimum weight threshold. Default 0.1 (standard practice in
        production FDTD codes). Recommended range: 0.05-0.3.

    Returns
    -------
    (w_ex_clamped, w_ey_clamped, w_ez_clamped) : each same shape as input.
    """
    def _clamp(w):
        return jnp.where(w > 0.0, jnp.maximum(w, w_min), 0.0)

    return _clamp(w_ex), _clamp(w_ey), _clamp(w_ez)


# ---------------------------------------------------------------------------
# Legacy sub-sampling weight computation (fallback)
# ---------------------------------------------------------------------------

def compute_conformal_weights(
    grid: Grid,
    pec_shapes: list,
    n_sub: int = 4,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute Dey-Mittra conformal weights for E-field components.

    Legacy sub-sampling implementation. Prefer ``compute_conformal_weights_sdf``
    for production use (100x faster, JAX-differentiable).

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
    def _point_inside_pec(px, py, pz):
        """Check if physical point (px, py, pz) is inside any PEC shape."""
        for shape in pec_shapes:
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
                # Cylinder
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

    sub = np.linspace(0.5 / n_sub, 1.0 - 0.5 / n_sub, n_sub)
    pad = np.array(grid.axis_pads) if hasattr(grid, 'axis_pads') else np.zeros(3)

    boundary_indices = np.argwhere(boundary)

    for idx in boundary_indices:
        i, j, k = idx

        # Ex face at (i+0.5, j, k)
        n_inside = 0
        px = (i + 0.5 - pad[0]) * dx
        for sy in sub:
            for sz in sub:
                py = (j + sy - pad[1]) * dx
                pz = (k + sz - pad[2]) * dx
                if _point_inside_pec(px, py, pz):
                    n_inside += 1
        w_ex[i, j, k] = 1.0 - n_inside / (n_sub * n_sub)

        # Ey face at (i, j+0.5, k)
        n_inside = 0
        py = (j + 0.5 - pad[1]) * dx
        for sx in sub:
            for sz in sub:
                px = (i + sx - pad[0]) * dx
                pz = (k + sz - pad[2]) * dx
                if _point_inside_pec(px, py, pz):
                    n_inside += 1
        w_ey[i, j, k] = 1.0 - n_inside / (n_sub * n_sub)

        # Ez face at (i, j, k+0.5)
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
    scaled by 1/w to account for the reduced cell area:

        eps_eff = eps_r / w_clamped

    This is equivalent to scaling Cb by 1/w (Dey-Mittra method).
    Applied ONCE at setup, not every timestep.

    Weights should be clamped via ``clamp_conformal_weights`` before
    calling this function to ensure CFL stability.

    Parameters
    ----------
    eps_r : jnp.ndarray or float
        Background relative permittivity (scalar or array).
    w_ex, w_ey, w_ez : jnp.ndarray
        Clamped conformal weights.

    Returns
    -------
    (eps_ex, eps_ey, eps_ez) : per-component permittivity arrays.
    """
    safe_wx = jnp.where(w_ex > 0, w_ex, 1.0)
    safe_wy = jnp.where(w_ey > 0, w_ey, 1.0)
    safe_wz = jnp.where(w_ez > 0, w_ez, 1.0)
    return eps_r / safe_wx, eps_r / safe_wy, eps_r / safe_wz
