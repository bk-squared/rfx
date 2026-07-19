"""Unified geometry rasterization for all grid types.

Extracts the material-assembly loop from api.py / nonuniform.py /
subgridded.py into a single function that accepts any grid type
via a coordinate-provider abstraction.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from rfx.core.jax_utils import is_tracer
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


def _axis_cell_centers(d_arr: np.ndarray, cpml: int) -> np.ndarray:
    """Cell-center positions for a padded cell-size array.

    Matches the existing ``coords_from_nonuniform_grid`` z convention:
    the first interior cell's LEFT edge is at physical position 0, so
    its CENTER is at ``d[cpml]/2``. This is off by half a cell from
    the legacy uniform-Grid convention (cell[cpml] center at 0) — the
    two conventions are not unified anywhere in rfx today.
    """
    d = np.asarray(d_arr, dtype=np.float64)
    edges = np.insert(np.cumsum(d), 0, 0.0)           # len = n+1
    offset = edges[cpml]                              # first-interior left edge
    centers = (edges[:-1] + edges[1:]) / 2.0 - offset
    return centers


def coords_from_nonuniform_grid(grid) -> GridCoords:
    """Extract cell-center coordinates from a NonUniformGrid.

    All three axes use the per-cell spacing arrays (``dx_arr``,
    ``dy_arr``, ``dz``). The first interior cell on each axis is
    placed at physical position 0, matching the convention that a
    ``Box((0,0,0), (Lx,Ly,Lz))`` should tile the interior domain
    exactly.
    """
    # Per-axis pad — respects PEC/PMC faces which have pad=0 even when
    # ``grid.cpml_layers`` is nonzero. Using the scalar ``cpml_layers``
    # here hit IndexError on axes that are PEC on both sides and shorter
    # than ``cpml_layers + 1`` cells (e.g. WR-90's narrow b-axis at
    # dx=1 mm: 11 cells, cpml_layers=20 → edges[20] out of bounds).
    pad_x_lo = int(getattr(grid, "pad_x_lo", grid.cpml_layers))
    pad_y_lo = int(getattr(grid, "pad_y_lo", grid.cpml_layers))
    pad_z_lo = int(getattr(grid, "pad_z_lo", grid.cpml_layers))
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    def _axis_centers(d_arr, pad_lo):
        # Mesh-as-design-variable path: any axis cell-size profile may be
        # a JAX tracer. Route the cumsum / offset arithmetic through jnp
        # in-trace; fall back to the numpy path on concrete inputs to keep
        # the host-float behaviour the rest of the codebase depends on.
        if is_tracer(d_arr):
            d_j = jnp.asarray(d_arr)
            cum = jnp.concatenate([jnp.zeros((1,), dtype=d_j.dtype),
                                   jnp.cumsum(d_j)])
            offset = cum[pad_lo]
            centers = (cum[:-1] + cum[1:]) / 2.0 - offset
            return centers.astype(jnp.float32)
        d_np = np.asarray(d_arr)
        return jnp.asarray(_axis_cell_centers(d_np, pad_lo), dtype=jnp.float32)

    x = _axis_centers(grid.dx_arr, pad_x_lo)
    y = _axis_centers(grid.dy_arr, pad_y_lo)
    z = _axis_centers(grid.dz, pad_z_lo)

    return GridCoords(x=x, y=y, z=z, shape=(nx, ny, nz))


def _pole_key(pole):
    """Mask-dict key for a dispersion pole (issue #274).

    Key by VALUE when the pole is hashable: equal-valued poles from
    different ``add_material`` calls merge into one ``(pole, mask)``
    entry, so overlapping geometry cannot apply the same pole twice
    (``init_debye`` / ``init_lorentz`` sum contributions over entries —
    a duplicate entry would silently double delta_eps on overlap
    cells). Fall back to object IDENTITY only for unhashable poles
    (JAX-traced fields), where value equality is undecidable at trace
    time and identity is the only stable key. Plain id-keying for all
    poles is the recorded do-not-repeat (PR #272 branch: overlap-cell
    beta ratio 2.000 vs 1.000).
    """
    try:
        hash(pole)
        return pole
    except TypeError:
        return id(pole)


def _accumulate_pole_mask(masks_by_pole: dict, pole, mask) -> None:
    """Merge ``mask`` into the entry for ``pole`` (see ``_pole_key``).

    Values are ``(pole, mask)`` pairs so iteration yields the pole
    object regardless of whether the key is the pole itself (concrete)
    or ``id(pole)`` (traced). The first-seen pole object is kept on
    merge, matching the historical dict-key behaviour byte-for-byte.
    """
    key = _pole_key(pole)
    prev = masks_by_pole.get(key)
    if prev is not None:
        masks_by_pole[key] = (prev[0], prev[1] | mask)
    else:
        masks_by_pole[key] = (pole, mask)


def _spec_from_pole_masks(masks_by_pole: dict):
    """Build a ``(poles, masks)`` spec tuple, or None when empty."""
    if not masks_by_pole:
        return None
    entries = list(masks_by_pole.values())
    return ([p for p, _ in entries], [m for _, m in entries])


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

    # Keyed per _pole_key (#274): pole value when hashable, id(pole) for
    # traced poles. Values are (pole, mask) pairs.
    debye_masks_by_pole: dict[DebyePole | int, tuple[DebyePole, jnp.ndarray]] = {}
    lorentz_masks_by_pole: dict[LorentzPole | int, tuple[LorentzPole, jnp.ndarray]] = {}

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
                _accumulate_pole_mask(debye_masks_by_pole, pole, mask)

        if mat.lorentz_poles:
            for pole in mat.lorentz_poles:
                _accumulate_pole_mask(lorentz_masks_by_pole, pole, mask)

    materials = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)

    # Thin conductors (P4)
    if thin_conductors and thin_conductor_applier and grid is not None:
        for tc in thin_conductors:
            materials, pec_mask = thin_conductor_applier(
                grid, tc, materials, pec_mask=pec_mask)
            if tc.is_pec:
                pec_shapes.append(tc.shape)

    debye_spec = _spec_from_pole_masks(debye_masks_by_pole)
    lorentz_spec = _spec_from_pole_masks(lorentz_masks_by_pole)

    has_pec = bool(jnp.any(pec_mask))
    kerr_chi3 = chi3_arr if has_kerr else None
    return materials, debye_spec, lorentz_spec, pec_mask if has_pec else None, pec_shapes, kerr_chi3
