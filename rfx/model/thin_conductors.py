"""One PEC/DC/f0 fold on the realized grid, in declaration order."""
from dataclasses import dataclass
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.geometry.csg import Box, _grid_coords
from rfx.geometry.rasterize_grid import (
    GridCoords, cell_sizes_from_nonuniform_grid, cell_sizes_from_uniform_grid,
    sheet_end_rows, sheet_footprint_traced, sheet_spec_from_shape,
)
from rfx.materials.thin_conductor import (
    SheetImpedanceSpec, check_sheet_occupancy, leontovich_rs, sheet_bounds,
    sheet_normal_axis,
)
from rfx.nonuniform import e_node_dual_spacings


class DCFilmAdmissionError(ValueError):
    """A geometry-only DC film refusal, also carried by preflight."""
    code = 'dc_film_area'
    source = 'admit_dc_film'

    def __init__(self, message, *, code='dc_film_area'):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class DCFilmGeometry:
    mask: object
    normal: int
    realized_area: float | None
    declared_area: float | None


def dc_footprint_areas(shape, grid, mask, normal):
    """Area of the actual cell footprint, independent of folded conductivity."""
    tangents = [a for a in range(3) if a != normal]
    widths = [np.asarray(grid.cells(a), dtype=float) for a in tangents]
    footprint = np.asarray(mask, dtype=bool).any(axis=normal)
    realized = float(np.sum(footprint * widths[0][:, None] * widths[1][None, :]))
    area = getattr(shape, 'footprint_area', None)
    try:
        declared = None if area is None else area(normal)
    except NotImplementedError:
        declared = None
    if declared is not None:
        declared = float(declared)
        if not np.isfinite(declared) or declared <= 0:
            declared = None
    return realized, declared


def admit_dc_film(shape, grid, *, snap='strict', emit=True):
    """One non-Box DC admission and plane realization on every grid.

    The Box DC sampler owns layer selection. Only its normal-axis occupancy
    is borrowed; the shape still owns its in-plane cross-section and area.
    Static geometry remains host-evaluable under parameter differentiation.
    """
    from rfx._periodic import periodic_mask, plane_coordinate
    from rfx.geometry.rasterize_grid import _local_cell
    from rfx.preflight._common import PreflightWarning
    from rfx.preflight.pec_geometry import _SHEET_EFFECTIVE_SIZE_TOL

    name = type(shape).__name__
    lo, hi = sheet_bounds(shape)
    if lo is None or hi is None:
        raise DCFilmAdmissionError(
            f'{name} DC film: cannot locate layers without a bounding_box(); '
            'provide bounds and footprint_area(normal), or draw a Box.', code='dc_film_layers')
    if any(is_tracer(v) for v in jax.tree_util.tree_leaves((lo, hi))):
        raise DCFilmAdmissionError(
            f'{name} DC film: cannot judge traced shape bounds (including radius); '
            'keep shape parameters concrete for layer/area admission.', code='dc_film_unchecked')
    extents = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    normal = int(np.argmin(extents))
    if (np.any(~np.isfinite(extents)) or np.any(extents < 0)
            or np.count_nonzero(np.isclose(extents, extents[normal], rtol=1e-12, atol=0)) != 1):
        raise DCFilmAdmissionError(
            f'{name} DC film: ambiguous normal/layers for bounding extents '
            f'{extents.tolist()}; draw a single planar film or use add() for a volume.',
            code='dc_film_layers')
    coords = list(_grid_coords(grid))
    mid = .5 * (lo[normal] + hi[normal])
    traced = any(is_tracer(c) for c in coords)
    if traced:
        if snap != 'declared':
            raise DCFilmAdmissionError(
                f'{name} DC film: cannot judge layer/area on a traced mesh; '
                'use a Box, or snap="declared" to run without these checks.',
                code='dc_film_unchecked')
        if emit:
            warnings.warn(PreflightWarning(
                f'{name} DC film: area/layer check not evaluated on traced coordinates; '
                'snap="declared" runs without checking area or the one-layer rule; '
                'check the same geometry on a concrete mesh.', code='dc_film_unchecked',
                source='admit_dc_film'), stacklevel=2)
        # Keep the coordinate-dependent Box selection and cross-section traced.
        plane_lo, plane_hi = list(lo), list(hi)
        plane_lo[normal] = plane_hi[normal] = mid
        layer = Box(tuple(plane_lo), tuple(plane_hi)).mask_on_coords(*coords).any(
            axis=tuple(a for a in range(3) if a != normal))
        sample = list(coords)
        sample[normal] = jnp.asarray([mid])
        footprint = shape.mask_on_coords(*sample)
        view = [1, 1, 1]
        view[normal] = len(layer)
        return DCFilmGeometry(footprint & layer.reshape(view), normal, None, None)
    with jax.ensure_compile_time_eval():
        local = _local_cell(coords[normal], np.asarray(grid.cells(normal)),
                            plane_coordinate(grid, normal, mid))
        if extents[normal] > local * (1 + 1e-9):
            raise DCFilmAdmissionError(
                f'{name} DC film: multiple layers, normal extent {extents[normal]:.12g} m '
                f'exceeds local cell {local:.12g} m; use add() for a volume.', code='dc_film_layers')
        plane_lo, plane_hi = list(lo), list(hi)
        plane_lo[normal] = plane_hi[normal] = mid
        control = dc_cell_mask(Box(tuple(plane_lo), tuple(plane_hi)), grid)
        layer = np.asarray(control).any(axis=tuple(a for a in range(3) if a != normal))
        sample = list(coords)
        sample[normal] = np.asarray([mid])
        footprint = np.asarray(periodic_mask(
            grid, shape, sample, axes=tuple(a for a in range(3) if a != normal)))
        view = [1, 1, 1]
        view[normal] = len(layer)
        mask = footprint & layer.reshape(view)
        layers = int(mask.any(axis=tuple(a for a in range(3) if a != normal)).sum())
        if layers != 1:
            raise DCFilmAdmissionError(
                f'{name} DC film: {layers} occupied layers (empty or nonplanar); '
                'a film requires exactly one nonempty layer.', code='dc_film_layers')
        realized, declared = dc_footprint_areas(shape, grid, mask, normal)
    error = None if declared is None else np.sqrt(realized / declared) - 1
    if error is None or abs(error) > _SHEET_EFFECTIVE_SIZE_TOL:
        clipped = any(
            'xyz'[a] not in getattr(grid, 'periodic_axes', '')
            and (lo[a] < float(coords[a][0]) or hi[a] > float(coords[a][-1]))
            for a in range(3) if a != normal)
        reason = ('cannot judge: no positive footprint_area(normal)' if error is None
                  else f'e=sqrt(A_r/A_d)-1={error:+.12g}, bar={_SHEET_EFFECTIVE_SIZE_TOL:.12g}')
        remedy = ('Implement footprint_area(normal) to make this custom shape judgeable'
                  if error is None else
                  'The in-plane bounding box is clipped by the grid extent; enlarge the '
                  'domain/grid or move the film inside it' if clipped else
                  'Refine the in-plane mesh or draw a Box')
        message = (f'{name} DC film: A_d={declared!r} m^2, A_r={realized:.12g} m^2; '
                   f'{reason}. {remedy}, or use '
                   'snap="declared" to accept the declared geometry.')
        if snap != 'declared':
            raise DCFilmAdmissionError(message)
        if emit:
            warnings.warn(PreflightWarning(message, code='dc_film_area', severity='warning',
                                          source='admit_dc_film'), stacklevel=2)
    return DCFilmGeometry(jnp.asarray(mask), normal, realized, declared)


def dc_film_refusals(sim, grid):
    """Audit the same admission without dropping a refused declaration's identity."""
    refused = {}
    for i, tc in enumerate(sim._thin_conductors):
        if tc.is_pec or tc.surface_impedance_f0 is not None or isinstance(tc.shape, Box):
            continue
        try:
            admit_dc_film(tc.shape, grid, snap=sim._snap, emit=False)
        except DCFilmAdmissionError as exc:
            refused[i] = str(exc)
    return refused


def warn_dc_films(sim, warn):
    """Expose assembly refusals through the existing structured finding carrier."""
    from rfx.preflight._common import PreflightWarning
    if not any(not tc.is_pec and tc.surface_impedance_f0 is None
               and not isinstance(tc.shape, Box) for tc in sim._thin_conductors):
        return
    grid = sim._build_realized_grid()
    for i, tc in enumerate(sim._thin_conductors):
        if tc.is_pec or tc.surface_impedance_f0 is not None or isinstance(tc.shape, Box):
            continue
        try:
            # Emit declared findings directly, including when the diagnostic
            # assembly cache contains a previous strict refusal.
            with warnings.catch_warnings(record=True) as findings:
                warnings.simplefilter('always')
                admit_dc_film(tc.shape, grid, snap=sim._snap)
        except DCFilmAdmissionError as exc:
            warn.warn(PreflightWarning(str(exc), code=exc.code, severity='error',
                                       source=exc.source, loc=f'thin_conductors[{i}]'))
        else:
            for finding in findings:
                message = finding.message
                warn.warn(PreflightWarning(str(message), code=message.code,
                    severity=message.severity, source=message.source,
                    loc=f'thin_conductors[{i}]'))


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
                        geometry_key=None, snap='strict'):
    """Emit PEC/f0 products or fold DC sigma*t/cell width; never overwrite PEC cells."""
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
        if not isinstance(conductor.shape, Box):
            film = admit_dc_film(conductor.shape, grid, snap=snap)
            mask, normal = film.mask, film.normal
        elif nonuniform:
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
    # The weighted E-edge mean integrates cell-owned DC sigma over its primal width.
    divisor = ((grid.dx_arr, grid.dy_arr, grid.dz)[normal].reshape(view)
               if nonuniform else dual)
    sigma_eff = conductor.sigma_bulk * (conductor.thickness / divisor)
    return materials._replace(eps_r=jnp.where(mask, conductor.eps_r, materials.eps_r),
                              sigma=jnp.where(mask, sigma_eff, materials.sigma)), pec_mask
