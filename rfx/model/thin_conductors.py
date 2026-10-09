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


def dc_film_plane(shape, grid):
    """The declared midpoint's nearest E node, with the shared lower tie rule."""
    from rfx._grid_metric import nearest_node_index
    from rfx._periodic import plane_coordinate
    from rfx.geometry.rasterize_grid import _local_cell
    lo, hi = sheet_bounds(shape)
    normal = sheet_normal_axis(lo, hi)
    nodes = _grid_coords(grid)[normal]
    mid = .5 * (lo[normal] + hi[normal])
    periodic = 'xyz'[normal] in getattr(grid, 'periodic_axes', '')
    if periodic:
        mid = plane_coordinate(grid, normal, mid)
        # Periodic stores omit the duplicate high node. Include that candidate
        # for nearest-node selection, then identify it with node zero.
        xp = jnp if is_tracer(nodes) else np
        nodes = xp.concatenate((nodes, nodes[-1:] + grid.cells(normal)[-1]))
    if is_tracer(nodes):
        widths = grid.cells(normal)
        local = widths[jnp.clip(jnp.searchsorted(nodes, mid) - 1, 0, len(widths) - 1)]
        plane = nearest_node_index(nodes, mid, local, xp=jnp)
    else:
        local = _local_cell(nodes, np.asarray(grid.cells(normal)), mid)
        plane = nearest_node_index(nodes, mid, local)
    if periodic:
        plane = jnp.where(plane == len(nodes) - 1, 0, plane) if is_tracer(plane) else (
            0 if plane == len(nodes) - 1 else plane)
    return normal, plane


def admit_dc_plane(conductor, grid, *, pmc_faces=None, mode=None):
    """Admission shared by Box/non-Box folds and the diagnostic audit."""
    if conductor.is_pec or conductor.surface_impedance_f0 is not None:
        return
    if not is_tracer(conductor.eps_r) and conductor.eps_r != 1:
        raise DCFilmAdmissionError(
            "a film on one node plane has no volume; its eps_r is not modelled",
            code='dc_film_permittivity')
    normal, plane = dc_film_plane(conductor.shape, grid)
    if mode is None:
        mode = getattr(grid, 'mode', '3d')
    if mode != '3d' and normal == 2:
        raise DCFilmAdmissionError(
            "a film normal to a 2-D model's invariant axis is not supported",
            code='dc_film_invariant')
    faces = getattr(grid, 'pmc_faces', ()) if pmc_faces is None else pmc_faces
    if not is_tracer(plane):
        face = ('xyz'[normal] + '_lo' if plane == 0 else
                'xyz'[normal] + '_hi' if plane == grid.shape[normal] - 1 else None)
        if face in faces:
            raise DCFilmAdmissionError("a film on a PMC domain face is not supported",
                                       code='dc_film_pmc')


def _fold_dc_plane(grid, conductor, materials, mask, normal):
    """In-plane cell coverage, carried only by tangential edges on one node."""
    from rfx.core.yee import edge_averaged_materials, lumped_components
    from rfx.model.materials import electric_cell_sizes
    normal, plane = dc_film_plane(conductor.shape, grid)
    footprint = mask.any(axis=normal)
    field = jnp.broadcast_to(jnp.expand_dims(footprint, normal), materials.sigma.shape)
    periodic = tuple(a in getattr(grid, 'periodic_axes', '') for a in 'xyz')
    _, weights = edge_averaged_materials(jnp.ones_like(materials.eps_r),
        field.astype(materials.sigma.dtype), periodic, cell_sizes=electric_cell_sizes(grid))
    dual = (e_node_dual_spacings((grid.dx_arr, grid.dy_arr, grid.dz)[normal])[plane]
            if hasattr(grid, 'dx_arr') else float(grid.duals(normal)[0]))
    density = conductor.sigma_bulk * conductor.thickness / dual
    view = [1, 1, 1]
    view[normal] = grid.shape[normal]
    on_plane = (jnp.arange(grid.shape[normal]) == plane).reshape(view)
    merged = []
    for component, previous in enumerate(lumped_components(materials.sigma_film)):
        if component == normal:
            merged.append(previous)
            continue
        weight = jnp.where(on_plane, weights[component], 0)
        merged.append(jnp.where(weight > 0, density * weight,
                                0 if previous is None else previous))
    n_cells = grid.shape[normal] - (not periodic[normal])
    layer = jnp.minimum(plane, n_cells - 1)
    cell_mask = jnp.broadcast_to(jnp.expand_dims(footprint, normal), mask.shape)
    cell_mask = cell_mask & (jnp.arange(grid.shape[normal]) == layer).reshape(view)
    return materials._replace(sigma_film=tuple(merged)), cell_mask


def dc_film_refusals(sim, grid):
    """Audit the same admission without dropping a refused declaration's identity."""
    refused = {}
    for i, tc in enumerate(sim._thin_conductors):
        if tc.is_pec or tc.surface_impedance_f0 is not None:
            continue
        try:
            admit_dc_plane(tc, grid, pmc_faces=sim._boundary_spec.pmc_faces(), mode=sim._mode)
            if not isinstance(tc.shape, Box):
                admit_dc_film(tc.shape, grid, snap=sim._snap, emit=False)
        except DCFilmAdmissionError as exc:
            refused[i] = str(exc)
    return refused


def warn_dc_films(sim, warn):
    """Expose assembly refusals through the existing structured finding carrier."""
    from rfx.preflight._common import PreflightWarning
    if not any(not tc.is_pec and tc.surface_impedance_f0 is None
               for tc in sim._thin_conductors):
        return
    grid = sim._build_realized_grid()
    for i, tc in enumerate(sim._thin_conductors):
        if tc.is_pec or tc.surface_impedance_f0 is not None:
            continue
        try:
            # Emit declared findings directly, including when the diagnostic
            # assembly cache contains a previous strict refusal.
            with warnings.catch_warnings(record=True) as findings:
                warnings.simplefilter('always')
                admit_dc_plane(tc, grid, pmc_faces=sim._boundary_spec.pmc_faces(), mode=sim._mode)
                if not isinstance(tc.shape, Box):
                    admit_dc_film(tc.shape, grid, snap=sim._snap)
        except DCFilmAdmissionError as exc:
            warn.warn(PreflightWarning(str(exc), code=exc.code, severity='error',
                                       source=exc.source, loc=f'thin_conductors[{i}]'))
        else:
            normal, plane = dc_film_plane(tc.shape, grid)
            face = ('xyz'[normal] + '_lo' if plane == 0 else
                    'xyz'[normal] + '_hi' if plane == grid.shape[normal] - 1 else None)
            if face in sim._boundary_spec.pec_faces():
                warn.warn(PreflightWarning(
                    f"lossy thin conductor {i} on PEC domain face {face} has no effect",
                    code='dc_film_pec_face', severity='info', source='admit_dc_film',
                    loc=f'thin_conductors[{i}]'))
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
    """Emit PEC/f0 products or place a DC film on tangential edges of one node."""
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
        admit_dc_plane(conductor, grid)
        if not isinstance(conductor.shape, Box):
            film = admit_dc_film(conductor.shape, grid, snap=snap)
            mask, normal = film.mask, film.normal
        elif nonuniform:
            normal = sheet_normal_axis(conductor.shape.corner_lo, conductor.shape.corner_hi)
            mask = conductor.shape.mask_on_coords(*coords[:3])
        else:
            normal = sheet_normal_axis(conductor.shape.corner_lo, conductor.shape.corner_hi)
            mask = dc_cell_mask(conductor.shape, grid)
        materials, mask = _fold_dc_plane(grid, conductor, materials, mask, normal)
        if geometry_masks is not None:
            geometry_masks.append((geometry_key, mask))
        return materials, pec_mask
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


def conductivity_envelope(materials):
    """Report/refusal view: a cell or any film component marks its index."""
    from rfx.core.yee import lumped_components
    out = materials.sigma
    for part in lumped_components(materials.sigma_film):
        if part is not None:
            out = jnp.maximum(out, part)
    return out


def splice_film(materials, junction, window):
    """Preserve each junction film component through the coax row splice."""
    from rfx.core.yee import lumped_components
    result = []
    for old, new in zip(lumped_components(materials.sigma_film),
                        lumped_components(junction.sigma_film)):
        if old is None and new is None:
            result.append(None)
            continue
        old = jnp.zeros_like(materials.sigma) if old is None else old
        result.append(old.at[window].set(0 if new is None else new[window]))
    return None if all(p is None for p in result) else tuple(result)


def refuse_design_films(sim, grid, bounds):
    """Name the declaration whose film edges intersect the design write window."""
    from dataclasses import replace
    from rfx.core.yee import init_materials, lumped_components
    from rfx.geometry.smoothing import continued_conductor_shape
    from rfx.simulation import _design_box_window
    write, _, _, _ = _design_box_window(bounds, grid.shape)
    sl = tuple(slice(write[a], write[a + 1]) for a in (0, 2, 4))
    for index, tc in enumerate(sim._thin_conductors):
        if tc.is_pec or tc.surface_impedance_f0 is not None:
            continue
        # Geometry decides admission even when conductance is differentiated.
        geometric = replace(tc, sigma_bulk=1.0, thickness=1.0,
            shape=continued_conductor_shape(sim, grid, tc.shape, entry=tc))
        mats, _ = fold_thin_conductor(grid, geometric, init_materials(grid.shape), snap=sim._snap)
        if any(part is not None and bool(jnp.any(part[sl] != 0))
               for part in lumped_components(mats.sigma_film)):
            raise ValueError(
                f"the design box (cells {bounds}) contains edges of thin conductor {index}; "
                "a design region over a declared film is not supported. Move the box or the film.")


def splice_junction_materials(materials, junction, z_index):
    """Retain the declared coax junction above its final stub row."""
    window = (slice(None), slice(None), slice(z_index, None))
    return materials._replace(
        eps_r=materials.eps_r.at[window].set(junction.eps_r[window]),
        sigma=materials.sigma.at[window].set(junction.sigma[window]),
        sigma_film=splice_film(materials, junction, window))
