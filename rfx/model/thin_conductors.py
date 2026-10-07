"""One PEC/DC/f0 fold on the realized grid, in declaration order."""
import jax.numpy as jnp

from rfx.core.jax_utils import is_tracer
from rfx.geometry.csg import _grid_coords
from rfx.geometry.rasterize_grid import (
    GridCoords, cell_sizes_from_nonuniform_grid, cell_sizes_from_uniform_grid,
    sheet_end_rows, sheet_footprint_traced, sheet_spec_from_shape,
)
from rfx.materials.thin_conductor import (
    SheetImpedanceSpec, check_sheet_occupancy, leontovich_rs, sheet_bounds,
    sheet_normal_axis,
)
from rfx.nonuniform import e_node_dual_spacings


def dc_cell_mask(shape, grid):
    """Sample DC occupancy, retaining periodic plane identification."""
    if getattr(grid, 'periodic_axes', ''):
        from rfx._periodic import periodic_mask, plane_coordinate
        lo, hi = sheet_bounds(shape)
        normal = None if lo is None or hi is None else sheet_normal_axis(lo, hi)
        if normal is not None and hi[normal] - lo[normal] <= float(grid.cells(normal)[0]):
            mid = .5 * (lo[normal] + hi[normal])
            sample = list(_grid_coords(grid))
            shift = mid - plane_coordinate(grid, normal, mid)
            if shift:
                sample[normal] = sample[normal] + shift
            return periodic_mask(grid, shape, sample,
                                 axes=tuple(a for a in range(3) if a != normal))
    if hasattr(grid, 'dx_arr'):
        return shape.mask_on_coords(*_grid_coords(grid))
    return shape.mask(grid)


def fold_thin_conductor(grid, conductor, materials, pec_mask=None, *,
                        sheet_specs=None, sheets=None, geometry_masks=None,
                        geometry_key=None):
    """Emit PEC/f0 products or fold DC sigma*t/dual; never overwrite PEC cells."""
    nonuniform = hasattr(grid, 'dx_arr')
    lane = 'non-uniform' if nonuniform else 'uniform'
    coords = GridCoords(*_grid_coords(grid), shape=tuple(grid.shape))
    sizes = (cell_sizes_from_nonuniform_grid(grid) if nonuniform
             else cell_sizes_from_uniform_grid(grid))
    f0 = conductor.surface_impedance_f0
    if conductor.is_pec or f0 is not None:
        if f0 is not None:
            lo, hi = sheet_bounds(conductor.shape)
            if lo is None or hi is None:
                detail = ("to locate its normal; refusing to skip it on the non-uniform path."
                          if nonuniform else
                          "— the sheet normal is read from it; refusing to fold it blind.")
                raise ValueError(
                    "surface-impedance (surface_impedance_f0) thin conductor requires "
                    "a shape with an axis-aligned bounding box (Box corner_lo/corner_hi, "
                    "or Shape.bounding_box()) " + detail)
            normal = sheet_normal_axis(lo, hi)
        traced = any(is_tracer(c) for c in coords[:3])
        if f0 is not None and traced:
            mask = sheet_footprint_traced(conductor.shape, coords, normal)
            plane, unwrapped = None, None
            end_rows = sheet_end_rows(conductor.shape, coords, normal)
        else:
            spec = sheet_spec_from_shape(
                conductor.shape, coords, sizes, name='thin_conductor',
                lane=lane, refuse_thick=True, grid=None if nonuniform else grid)
            if conductor.is_pec:
                if sheets is None:
                    raise ValueError(
                        "PEC thin conductor (add_thin_conductor with sigma_bulk >= "
                        "1e6 and no surface_impedance_f0) is a SHEET (#931) and this "
                        "lane collects no sheets (apply_thin_conductor(sheets=None)); "
                        "refusing to drop it silently.")
                sheets.append(spec)
                return materials, pec_mask
            mask, plane = spec.footprint, spec.plane
            unwrapped, end_rows = spec.unwrapped_footprint, spec.end_rows
        check_sheet_occupancy(mask, normal, lane=lane)
    else:
        # G5 pre-declaration, decision record 4: keep each path's DC admission.
        if nonuniform:
            from rfx.runners.nonuniform import nu_thin_conductor_refusal
            refusal = nu_thin_conductor_refusal(conductor)
            if refusal is not None:
                raise NotImplementedError(refusal)
            normal = sheet_normal_axis(conductor.shape.corner_lo, conductor.shape.corner_hi)
            mask = conductor.shape.mask_on_coords(*coords[:3])
        else:
            mask = dc_cell_mask(conductor.shape, grid)
    if geometry_masks is not None:
        geometry_masks.append((geometry_key, mask))
    # Keep the path's existing metric and precision: a Python scalar on
    # uniform grids, the solver's dual array on profiled grids.
    # Uniform grid: one cell size on every axis, so any axis's dual spacing is the divisor (a mask-shape DC
    # conductor has no declared normal here). Asked of the grid, not read from the scalar attribute, and kept a
    # Python float so the arithmetic is main's exactly (decision record 3).
    dual = None if nonuniform else float(grid.duals(0)[0])
    if nonuniform:
        dual = e_node_dual_spacings((grid.dx_arr, grid.dy_arr, grid.dz)[normal])
        view = [1, 1, 1]
        view[normal] = len(dual)
        dual = dual.reshape(view)
    if f0 is not None:
        conductance = 1.0 / leontovich_rs(f0, conductor.sigma_bulk)
        if sheet_specs is not None:
            density = conductance / dual
            if nonuniform:
                # Preserve main's promotion when material sigma is float64.
                density = density * jnp.ones_like(materials.sigma)
            sigma_sheet = jnp.where(mask, density, 0.0)
            sheet_specs.append(SheetImpedanceSpec(
                mask=mask, normal_axis=normal, g_sheet=conductance,
                sigma_sheet=sigma_sheet, plane=plane,
                unwrapped_footprint=unwrapped, end_rows=end_rows))
        return materials, pec_mask
    sigma_eff = conductor.sigma_bulk * (conductor.thickness / dual)
    return materials._replace(eps_r=jnp.where(mask, conductor.eps_r, materials.eps_r),
                              sigma=jnp.where(mask, sigma_eff, materials.sigma)), pec_mask
