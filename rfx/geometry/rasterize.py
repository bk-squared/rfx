"""Unified geometry rasterization for all grid types.

Extracts the material-assembly loop from api.py / nonuniform.py /
subgridded.py into a single function that accepts any grid type
via a coordinate-provider abstraction.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from rfx.core.yee import MaterialArrays


class GridCoords(NamedTuple):
    """Physical cell-center coordinates for rasterization."""
    x: jnp.ndarray  # (nx,)
    y: jnp.ndarray  # (ny,)
    z: jnp.ndarray  # (nz,)
    shape: tuple[int, int, int]


def coords_from_uniform_grid(grid) -> GridCoords:
    """Extract cell-center coordinates from a uniform Grid."""
    nx, ny, nz = grid.shape
    dx = grid.dx
    pad_x, pad_y, pad_z = grid.axis_pads
    x = jnp.asarray((np.arange(nx) - pad_x) * dx, dtype=jnp.float32)
    y = jnp.asarray((np.arange(ny) - pad_y) * dx, dtype=jnp.float32)
    z = jnp.asarray((np.arange(nz) - pad_z) * dx, dtype=jnp.float32)
    return GridCoords(x=x, y=y, z=z, shape=(nx, ny, nz))


def coords_from_nonuniform_grid(grid) -> GridCoords:
    """Extract cell-center coordinates from a NonUniformGrid.

    x, y are uniform; z comes from cumulative dz.
    """
    cpml = grid.cpml_layers
    dx = grid.dx
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    x = jnp.asarray((np.arange(nx) - cpml) * dx, dtype=jnp.float32)
    y = jnp.asarray((np.arange(ny) - cpml) * dx, dtype=jnp.float32)

    dz_np = np.array(grid.dz)
    z_cumsum = np.cumsum(dz_np)
    z_cumsum = np.insert(z_cumsum, 0, 0.0)
    z_offset = z_cumsum[cpml]
    z_centers = (z_cumsum[:-1] + z_cumsum[1:]) / 2.0 - z_offset
    z = jnp.asarray(z_centers, dtype=jnp.float32)

    return GridCoords(x=x, y=y, z=z, shape=(nx, ny, nz))


def coords_from_fine_grid(nx_f, ny_f, nz_f, dx_f, x_off, y_off, z_off) -> GridCoords:
    """Extract cell-center coordinates for a subgridded fine region.

    Uses cell centers (offset by dx_f/2), not cell edges.
    """
    x = jnp.asarray(x_off + (np.arange(nx_f) + 0.5) * dx_f, dtype=jnp.float32)
    y = jnp.asarray(y_off + (np.arange(ny_f) + 0.5) * dx_f, dtype=jnp.float32)
    z = jnp.asarray(z_off + (np.arange(nz_f) + 0.5) * dx_f, dtype=jnp.float32)
    return GridCoords(x=x, y=y, z=z, shape=(nx_f, ny_f, nz_f))


def rasterize_geometry(
    geometry_entries,
    material_resolver,
    coords: GridCoords,
    *,
    pec_sigma_threshold: float = 1e6,
    thin_conductors=None,
    thin_conductor_applier=None,
    grid=None,
):
    """Rasterize geometry entries onto material arrays.

    This is the single shared implementation used by all runner paths
    (uniform, non-uniform, subgridded).

    Parameters
    ----------
    geometry_entries : list of _GeometryEntry
        Each has .shape (Shape) and .material_name (str).
    material_resolver : callable(name) -> MaterialSpec
        Resolves material name to MaterialSpec.
    coords : GridCoords
        Cell-center coordinates from any grid type.
    pec_sigma_threshold : float
        Conductivity above which a material is treated as PEC.
    thin_conductors : list or None
        ThinConductor entries to apply after geometry.
    thin_conductor_applier : callable or None
        Function(grid, tc, materials, pec_mask) -> (materials, pec_mask).
    grid : Grid or NonUniformGrid or None
        Original grid object, needed by thin_conductor_applier.

    Returns
    -------
    materials : MaterialArrays
    debye_spec : (poles, masks) or None
    lorentz_spec : (poles, masks) or None
    pec_mask : bool array or None
    pec_shapes : list of Shape
    kerr_chi3 : float array or None
    """
    from rfx.materials.debye import DebyePole
    from rfx.materials.lorentz import LorentzPole

    shape = coords.shape
    eps_r = jnp.ones(shape, dtype=jnp.float32)
    sigma = jnp.zeros(shape, dtype=jnp.float32)
    mu_r = jnp.ones(shape, dtype=jnp.float32)
    chi3_arr = jnp.zeros(shape, dtype=jnp.float32)
    pec_mask = jnp.zeros(shape, dtype=jnp.bool_)
    pec_shapes = []
    has_kerr = False

    debye_masks_by_pole: dict[DebyePole, jnp.ndarray] = {}
    lorentz_masks_by_pole: dict[LorentzPole, jnp.ndarray] = {}

    for entry in geometry_entries:
        mat = material_resolver(entry.material_name)
        mask = entry.shape.mask_on_coords(coords.x, coords.y, coords.z)

        if mat.sigma >= pec_sigma_threshold:
            pec_mask = pec_mask | mask
            pec_shapes.append(entry.shape)
        else:
            eps_r = jnp.where(mask, mat.eps_r, eps_r)
            sigma = jnp.where(mask, mat.sigma, sigma)
            mu_r = jnp.where(mask, mat.mu_r, mu_r)

        if mat.chi3 != 0.0:
            chi3_arr = jnp.where(mask, mat.chi3, chi3_arr)
            has_kerr = True

        if mat.debye_poles:
            for pole in mat.debye_poles:
                if pole in debye_masks_by_pole:
                    debye_masks_by_pole[pole] = debye_masks_by_pole[pole] | mask
                else:
                    debye_masks_by_pole[pole] = mask

        if mat.lorentz_poles:
            for pole in mat.lorentz_poles:
                if pole in lorentz_masks_by_pole:
                    lorentz_masks_by_pole[pole] = lorentz_masks_by_pole[pole] | mask
                else:
                    lorentz_masks_by_pole[pole] = mask

    materials = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)

    # Thin conductors (P4)
    if thin_conductors and thin_conductor_applier and grid is not None:
        for tc in thin_conductors:
            materials, pec_mask = thin_conductor_applier(
                grid, tc, materials, pec_mask=pec_mask)
            if tc.is_pec:
                pec_shapes.append(tc.shape)

    debye_spec = None
    if debye_masks_by_pole:
        debye_poles = list(debye_masks_by_pole)
        debye_masks = [debye_masks_by_pole[pole] for pole in debye_poles]
        debye_spec = (debye_poles, debye_masks)

    lorentz_spec = None
    if lorentz_masks_by_pole:
        lorentz_poles = list(lorentz_masks_by_pole)
        lorentz_masks = [lorentz_masks_by_pole[pole] for pole in lorentz_poles]
        lorentz_spec = (lorentz_poles, lorentz_masks)

    has_pec = bool(jnp.any(pec_mask))
    kerr_chi3 = chi3_arr if has_kerr else None
    return materials, debye_spec, lorentz_spec, pec_mask if has_pec else None, pec_shapes, kerr_chi3
