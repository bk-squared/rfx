"""Conformal PEC as partially filled cells: E free lengths, H free areas, and the SC clamp (#1651).

Each E edge cut by a PEC surface gets permittivity eps/lbar, lbar its PEC-free length (from the signed distances at its
two end nodes); each H face gets permeability mu*sbar, sbar its PEC-free area (marching squares on its corner nodes);
edges with lbar = 0 are metal. This is the partially-filled-cell form of Dey and Mittra (Zagorodnov, Schuhmann and
Weiland, JCP 225 (2007) eq. 12). The SC clamp lbar <= 2 min(sbar of the four faces at the edge) (ibid. eqs. 25-26)
keeps the ordinary time step. Measured: a flat wall sits at its drawn plane within 1e-2 cell; on a PEC sphere the
error is first order, a radius about 0.2 cell large against the staircase's 0.7.
"""

from __future__ import annotations

from rfx.errors import AccuracyWarning
import warnings

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.geometry.conformal_cells import compute_conformal_fractions
from rfx.stepping.conductors import apply_conformal_pec as _apply_conformal_pec


def apply_conformal_pec(state, w_ex, w_ey, w_ez):
    """Compatibility entry for weight callers; production forms masks at setup."""
    return _apply_conformal_pec(state, w_ex == 0.0, w_ey == 0.0, w_ez == 0.0)


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


CONFORMAL_MODEL_ADVISORY = (
    "conformal PEC uses partially filled cells (eps/lbar on cut E edges, mu*sbar on cut H "
    "faces, the SC clamp): a flat wall sits at its drawn plane, but a curved body converges "
    "only first order, solved about 0.2 cell larger in radius on a PEC sphere (staircase "
    "0.7); judge a curved body by a mesh series (docs/guides/known_limitations.md).")


def warn_conformal_model():
    """Warn on a conformal run that a curved body converges only first order (#1651).

    Under Python's default warning filter it is shown once per calling
    location, not on every run.
    """
    warnings.warn(CONFORMAL_MODEL_ADVISORY, AccuracyWarning, stacklevel=3)


# ---------------------------------------------------------------------------
# Vectorized SDF-based conformal weight computation (Phase 1)
# ---------------------------------------------------------------------------

def compute_conformal_weights_sdf(grid: Grid, pec_shapes: list):
    """Compatibility E operands: node free lengths after the SC clamp.

    Call ``compute_conformal_fractions`` when forming H operands too.
    """
    return compute_conformal_fractions(grid, pec_shapes)[0]


def validate_conformal_min_weight(value):
    """The old floor is no longer a physics input; refuse nondefault requests."""
    if value != 0.1:
        raise NotImplementedError(
            'conformal_min_weight is replaced by the SC clamp; only the ignored default 0.1 is accepted')


def clamp_conformal_weights(w_ex, w_ey, w_ez, w_min=0.1):
    """Accept the former default; node geometry already applies the SC clamp."""
    validate_conformal_min_weight(w_min)
    return w_ex, w_ey, w_ez


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

    from rfx.geometry.csg import OrientedBox
    if any(isinstance(shape, OrientedBox) for shape in pec_shapes):
        raise NotImplementedError("OrientedBox conformal weights are not implemented")
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


def conformal_eps_correction(eps_r, w_ex, w_ey, w_ez):
    """Form eps/lbar once before stepping; zero-length edges keep eps.

    The zero masks enforce metal edges. The SC clamp is applied to the
    lengths during geometry formation, together with the H face areas.

    Parameters
    ----------
    eps_r : tuple of three jnp.ndarray, jnp.ndarray or float
        Volume relative permittivity per E component, already averaged
        onto the Yee edges (``rfx.core.yee.edge_mean_components``), or one
        scalar / array shared by all three components. A heterogeneous
        cell-centred array must be edge-averaged first (#1373), and a
        lumped capacitor is added afterwards, never divided by ``w``.
    w_ex, w_ey, w_ez : jnp.ndarray
        Clamped conformal weights.

    Returns
    -------
    (eps_ex, eps_ey, eps_ez) : per-component permittivity arrays.
    """
    safe_wx = jnp.where(w_ex > 0, w_ex, 1.0)
    safe_wy = jnp.where(w_ey > 0, w_ey, 1.0)
    safe_wz = jnp.where(w_ez > 0, w_ez, 1.0)
    if isinstance(eps_r, tuple):
        eps_x, eps_y, eps_z = eps_r
    else:
        eps_x = eps_y = eps_z = eps_r
    return eps_x / safe_wx, eps_y / safe_wy, eps_z / safe_wz
