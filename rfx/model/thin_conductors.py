"""One PEC/DC/f0 fold on the realized grid, in declaration order."""

from rfx.errors import AccuracyWarning, ConfigurationWarning, RfxRefusal
from dataclasses import dataclass, replace
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.geometry.csg import Box, _grid_coords
from rfx.geometry.rasterize_grid import (
    GridCoords, cell_sizes_from_nonuniform_grid, cell_sizes_from_uniform_grid,
    footprint_sample_coords, sheet_end_rows, sheet_footprint_traced, sheet_spec_from_shape,
)
from rfx.materials.thin_conductor import (
    _PEC_SIGMA_THRESHOLD, ThinConductor,
    SheetImpedanceSpec, check_sheet_occupancy, leontovich_rs, sheet_bounds,
    sheet_normal_axis,
)
from rfx.model.graded_grid import e_node_dual_spacings


def admit_thin_conductor(
    shape, *, sigma_bulk, thickness, eps_r, surface_impedance_f0,
) -> ThinConductor:
    """Admit one ``Simulation.add_thin_conductor`` declaration and build its record.

    Raises on the concrete-value checks, then warns which model the record
    selects (PEC sheet or Leontovich sheet); the warnings name the user's call.
    """
    if surface_impedance_f0 is not None:
        # Concrete-value checks only: traced f0 / sigma_bulk are legal
        # differentiable DoFs and skip these (documented on Simulation.add_thin_conductor).
        if (not is_tracer(surface_impedance_f0)
                and float(surface_impedance_f0) <= 0.0):
            raise ValueError(
                f"add_thin_conductor: surface_impedance_f0="
                f"{surface_impedance_f0!r} must be a positive frequency "
                f"in Hz (band centre for the Leontovich surface "
                f"resistance).")
        if not is_tracer(sigma_bulk) and float(sigma_bulk) <= 0.0:
            raise ValueError(
                f"add_thin_conductor: sigma_bulk={sigma_bulk!r} must be "
                f"> 0 when surface_impedance_f0 is set — Rs0 = "
                f"sqrt(pi*f0*mu0/sigma_bulk) is undefined otherwise.")
        # #674: any shape that can rasterize itself is allowed — the
        # fold is per occupied cell and shape-agnostic. Two structural
        # requirements survive, and both fail loud on BOTH lanes rather
        # than repeat the former NU warn-and-skip (the #369
        # silently-vaporized-metal class: a sheet that vanishes on one
        # lane is a wrong answer, not a degraded one).
        if not callable(getattr(shape, "mask_on_coords", None)):
            raise ValueError(
                f"add_thin_conductor: surface_impedance_f0 requires a "
                f"shape that implements mask_on_coords(x, y, z) — the "
                f"sheet is rasterized from that mask on both lanes, and "
                f"{type(shape).__name__} does not provide it.")
        if sheet_bounds(shape) == (None, None):
            raise ValueError(
                f"add_thin_conductor: surface_impedance_f0 requires a "
                f"shape with an axis-aligned bounding box (Box "
                f"corner_lo/corner_hi, or Shape.bounding_box()) — the "
                f"sheet NORMAL is read from it, and on a graded mesh that "
                f"normal decides which dual spacing normalizes the fold. "
                f"{type(shape).__name__} provides neither.")
    tc = ThinConductor(
        shape=shape, sigma_bulk=sigma_bulk,
        thickness=thickness, eps_r=eps_r,
        surface_impedance_f0=surface_impedance_f0,
    )
    # Tell the caller which model they actually got. The predicate is
    # sigma_bulk ALONE (materials/thin_conductor.py:60-62) — an earlier
    # version of this warning printed sigma_bulk*thickness/dx and claimed
    # it "exceeds 1e6", which was a false statement for ordinary inputs
    # (e.g. aluminium at 17 um and dx = 1 mm gives 5.95e5, announced as
    # exceeding 1e6 while the conductor was routed to PEC anyway).
    if tc.is_pec:
        import warnings
        _thin_dflt = 35e-6
        _asked = (
            f" You passed thickness={thickness:.3g} m, which is not used"
            if thickness != _thin_dflt else
            " thickness is not used"
        )
        warnings.warn(
            f"add_thin_conductor: sigma_bulk={sigma_bulk:.2e} S/m is at or "
            f"above the {_PEC_SIGMA_THRESHOLD:.0e} S/m PEC threshold, so "
            f"this is a LOSSLESS PEC sheet.{_asked} here, and neither is "
            f"eps_r. Every metal is above this threshold — a lower "
            f"sigma_bulk would be a different material, not thinner "
            f"copper. For conductor loss pass "
            f"surface_impedance_f0=<band centre Hz> (Leontovich sheet"
            f").", ConfigurationWarning,
            stacklevel=3,
        )
    if surface_impedance_f0 is not None:
        import warnings
        if not (is_tracer(surface_impedance_f0) or is_tracer(sigma_bulk)):
            _rs0 = float(tc.r_s_leontovich)
            _rs_txt = f"Rs0 = {_rs0:.4g} ohm/sq"
        else:
            _rs_txt = ("Rs0 = sqrt(pi*f0*mu0/sigma_bulk) (traced value "
                       "not evaluated here)")
        warnings.warn(
            f"add_thin_conductor: surface_impedance_f0="
            f"{surface_impedance_f0!r} Hz — Leontovich surface-"
            f"resistance sheet with {_rs_txt}. Rs is FREQUENCY-FLAT "
            f"within the run, frozen at f0, with relative band error "
            f"|sqrt(f/f0)-1|; the reactive part Xs = Rs (internal "
            f"inductance) is NOT modeled, so resonances read slightly "
            f"high.", AccuracyWarning,
            stacklevel=3,
        )
    return tc


class DCFilmAdmissionError(RfxRefusal):
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


@dataclass(frozen=True)
class FilmFootprint:
    """One declaration's cell coverage and drawn join authority."""
    mask: object
    normal: int
    plane: object
    drawn: object
    order: int

    @property
    def footprint(self):
        return self.mask


def separate_film_footprints(films, grid):
    """Leave an empty cell at each unauthorized face between DC films.

    Each transverse column is independent. Compare the two incident rows'
    drawn coverage fractions; a floating-point tie belongs to the earlier
    declaration. The shared sheet predicate owns tolerance and image joins.
    """
    from rfx.boundaries.pec import _sheet_pair_touches
    from rfx.model.film_geometry import refuse_traced_film_pairs
    if len(films) < 2:
        return films
    if any(is_tracer(v) for v in jax.tree_util.tree_leaves(
            (_grid_coords(grid), [(f.mask, f.plane, f.drawn) for f in films]))):
        refuse_traced_film_pairs(films, grid)
        return films
    masks = [np.array(f.mask, dtype=bool) for f in films]
    periodic = tuple(a in getattr(grid, 'periodic_axes', '') for a in 'xyz')
    nodes = _grid_coords(grid)

    def coverage(film, axis, row):
        lo, hi, _, length = film.drawn[axis]
        x = float(nodes[axis][row])
        width = float(grid.cells(axis)[row])
        return max(max(0., min(x + width, hi + shift) - max(x, lo + shift)) / width
                   for shift in ((0., length, -length) if periodic[axis] else (0.,)))

    for i, left in enumerate(films):
        for j in range(i + 1, len(films)):
            right = films[j]
            if left.normal != right.normal or left.plane != right.plane:
                continue
            # Compare footprints on their solved node plane even when their
            # original normal cell layers differ.
            a, b = masks[i].any(axis=left.normal), masks[j].any(axis=left.normal)
            for axis in range(3):
                if axis == left.normal or _sheet_pair_touches(left, right, axis, periodic):
                    continue
                t = axis - (axis > left.normal)
                aa, bb = np.moveaxis(a, t, 0), np.moveaxis(b, t, 0)
                n = len(aa)
                # Include common cells as well as both face orientations.
                # Removing cells cannot create another conflicting face.
                for row in range(n):
                    for other in (row - 1, row, row + 1):
                        if not periodic[axis] and not 0 <= other < n:
                            continue
                        other %= n
                        conflict = aa[row] & bb[other]
                        if not conflict.any():
                            continue
                        ca, cb = coverage(left, axis, row), coverage(right, axis, other)
                        earlier_loses = (ca < cb and not np.isclose(ca, cb, rtol=0, atol=1e-12))
                        if np.isclose(ca, cb, rtol=0, atol=1e-12):
                            earlier_loses = left.order > right.order
                        if earlier_loses:
                            aa[row, conflict] = False
                        else:
                            bb[other, conflict] = False
            masks[i] &= np.expand_dims(a, left.normal)
            masks[j] &= np.expand_dims(b, left.normal)
    return [replace(f, mask=jnp.asarray(m)) for f, m in zip(films, masks)]


def dc_footprint_areas(shape, grid, mask, normal):
    """Area of the actual cell footprint, independent of folded conductivity."""
    tangents = [a for a in range(3) if a != normal]
    widths = [np.asarray(grid.cells(a), dtype=float) for a in tangents]
    footprint = np.asarray(mask, dtype=bool).any(axis=normal)
    if isinstance(shape, Box):
        coords = _grid_coords(grid)
        spans = []
        for i, a in enumerate(tangents):
            if len(coords[a]) == 1:
                widths[i] = np.ones_like(widths[i])
                spans.append(1.0)
                continue
            lo, hi = shape.corner_lo[a], shape.corner_hi[a]
            if 'xyz'[a] in getattr(grid, 'periodic_axes', ''):
                spans.append(min(max(hi - lo, 0.0), float(np.sum(grid.cells(a)))))
            else:
                # The final node's metric is a fence-post provider, not
                # another cell beyond the sampled coordinate extent.
                widths[i] = np.minimum(widths[i], np.maximum(
                    float(coords[a][-1]) - np.asarray(coords[a]), 0.0))
                spans.append(max(0.0, min(hi, float(coords[a][-1]))
                                 - max(lo, float(coords[a][0]))))
        realized = float(np.sum(footprint * widths[0][:, None] * widths[1][None, :]))
        return realized, float(np.prod(spans))
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
    """DC area admission using cell-centred in-plane occupancy samples.

    The Box DC sampler owns layer selection. Only its normal-axis occupancy
    is borrowed; the shape still owns its in-plane cross-section and area.
    Static geometry remains host-evaluable under parameter differentiation.
    Box drawings are clipped to the sampled tangential extent (one period
    on periodic axes). A Box on traced coordinates keeps its existing fold
    without a new refusal; preflight judges the concrete realized grid.
    """
    from rfx._periodic import periodic_mask, plane_coordinate
    from rfx.geometry.rasterize_grid import _local_cell
    from rfx.preflight._common import PreflightWarning
    name = type(shape).__name__
    lo, hi = sheet_bounds(shape)
    if isinstance(shape, Box):
        normal = sheet_normal_axis(lo, hi)
        with jax.ensure_compile_time_eval():
            mask = dc_cell_mask(shape, grid)
            if any(is_tracer(v) for v in jax.tree_util.tree_leaves(
                    (lo, hi, _grid_coords(grid), mask))):
                return DCFilmGeometry(mask, normal, None, None)
            realized, declared = dc_footprint_areas(shape, grid, mask, normal)
        _check_dc_area(shape, grid, normal, realized, declared, snap=snap, emit=emit)
        return DCFilmGeometry(mask, normal, realized, declared)
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
    tangents = tuple(a for a in range(3) if a != normal)
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
        sample = list(footprint_sample_coords(grid, tangents, carrier="cell"))
        layer = Box(tuple(plane_lo), tuple(plane_hi)).mask_on_coords(*sample).any(
            axis=tuple(a for a in range(3) if a != normal))
        sample[normal] = jnp.asarray([mid])
        footprint = shape.mask_on_coords(*sample)
        view = [1, 1, 1]
        view[normal] = len(layer)
        return DCFilmGeometry(footprint & layer.reshape(view), normal, None, None)
    with jax.ensure_compile_time_eval():
        local = _local_cell(coords[normal], np.asarray(grid.cells(normal)),
                            plane_coordinate(grid, normal, mid))
        from rfx._grid_metric import node_tie_tol
        if extents[normal] > (local + node_tie_tol(lo[normal], local)
                              + node_tie_tol(hi[normal], local)):
            raise DCFilmAdmissionError(
                f'{name} DC film: multiple layers, normal extent {extents[normal]:.12g} m '
                f'exceeds local cell {local:.12g} m; use add() for a volume.', code='dc_film_layers')
        plane_lo, plane_hi = list(lo), list(hi)
        plane_lo[normal] = plane_hi[normal] = mid
        control = dc_cell_mask(Box(tuple(plane_lo), tuple(plane_hi)), grid)
        layer = np.asarray(control).any(axis=tuple(a for a in range(3) if a != normal))
        sample = list(footprint_sample_coords(grid, tangents, carrier="cell"))
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
    _check_dc_area(shape, grid, normal, realized, declared, snap=snap, emit=emit)
    return DCFilmGeometry(jnp.asarray(mask), normal, realized, declared)


def _check_dc_area(shape, grid, normal, realized, declared, *, snap, emit):
    from rfx.preflight._common import PreflightWarning
    from rfx.preflight.pec_geometry import _SHEET_EFFECTIVE_SIZE_TOL
    name = type(shape).__name__
    lo, hi = sheet_bounds(shape)
    coords = _grid_coords(grid)
    k = sum(len(coords[a]) > 1 for a in range(3) if a != normal) if isinstance(shape, Box) else 2
    exponent = 'sqrt(A_r/A_d)' if k == 2 else f'(A_r/A_d)^(1/{k})'
    error = (None if declared is None or declared <= 0 or k == 0 else
             np.sqrt(realized / declared) - 1 if k == 2 else (realized / declared)**(1 / k) - 1)
    if error is None or abs(error) > _SHEET_EFFECTIVE_SIZE_TOL:
        clipped = any(
            'xyz'[a] not in getattr(grid, 'periodic_axes', '')
            and (lo[a] < float(coords[a][0]) or hi[a] > float(coords[a][-1]))
            for a in range(3) if a != normal)
        reason = ('cannot judge: no positive footprint_area(normal)' if error is None
                  else f'e={exponent}-1={error:+.12g}, bar={_SHEET_EFFECTIVE_SIZE_TOL:.12g}')
        box = isinstance(shape, Box)
        remedy = ('The Box has no in-plane extent inside the grid; draw it inside the domain'
                  if error is None and box else
                  'Implement footprint_area(normal) to make this custom shape judgeable'
                  if error is None else
                  'The in-plane bounding box is clipped by the grid extent; enlarge the '
                  'domain/grid or move the film inside it' if clipped else
                  'Put the in-plane edges on mesh nodes or refine the in-plane mesh' if box else
                  'Refine the in-plane mesh or draw a Box')
        message = (f'{name} DC film: A_d={declared!r} m^2, A_r={realized:.12g} m^2; '
                   f'{reason}. {remedy}, or use '
                   'snap="declared" to accept the declared geometry.')
        if snap != 'declared':
            raise DCFilmAdmissionError(message)
        if emit:
            warnings.warn(PreflightWarning(message, code='dc_film_area', severity='warning',
                                          source='admit_dc_film'), stacklevel=2)


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
    if is_tracer(conductor.eps_r) or conductor.eps_r != 1:
        raise DCFilmAdmissionError(
            "a film on one node plane has no volume; its eps_r is not modelled",
            code='dc_film_permittivity')
    lo, hi = sheet_bounds(conductor.shape)
    if not isinstance(conductor.shape, Box) and (
            lo is None or hi is None or
            any(is_tracer(v) for v in jax.tree_util.tree_leaves((lo, hi)))):
        # Preserve the non-Box admission's descriptive bounds refusal before
        # the node selector attempts any concrete geometry arithmetic.
        admit_dc_film(conductor.shape, grid, emit=False)
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
    from rfx.core.yee import lumped_components
    from rfx.model.materials import edge_coverage_weights
    normal, plane = dc_film_plane(conductor.shape, grid)
    footprint = mask.any(axis=normal)
    field = jnp.broadcast_to(jnp.expand_dims(footprint, normal), materials.sigma.shape)
    periodic = tuple(a in getattr(grid, 'periodic_axes', '') for a in 'xyz')
    weights = edge_coverage_weights(grid, field.astype(materials.sigma.dtype), periodic)
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
        merged.append(density * weight + (0 if previous is None else previous))
    n_cells = grid.shape[normal] - (not periodic[normal])
    layer = jnp.minimum(plane, n_cells - 1)
    cell_mask = jnp.broadcast_to(jnp.expand_dims(footprint, normal), mask.shape)
    cell_mask = cell_mask & (jnp.arange(grid.shape[normal]) == layer).reshape(view)
    return materials._replace(sigma_film=tuple(merged)), cell_mask


def dc_film_refusals(sim, grid):
    """Audit the same admission without dropping a refused declaration's identity."""
    from rfx.model.film_geometry import diagnostic_film_footprints, check_film_area
    conductors, films = diagnostic_film_footprints(sim, grid)
    refused = {}
    for i, tc in enumerate(conductors):
        if tc.is_pec or tc.surface_impedance_f0 is not None:
            continue
        try:
            admit_dc_plane(tc, grid, pmc_faces=sim._boundary_spec.pmc_faces(), mode=sim._mode)
            if i in films:
                check_film_area(tc.shape, grid, films[i], snap=sim._snap, emit=False)
            else:
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
    from rfx.model.film_geometry import diagnostic_film_footprints, check_film_area
    conductors, films = diagnostic_film_footprints(sim, grid)
    for i, tc in enumerate(conductors):
        if tc.is_pec or tc.surface_impedance_f0 is not None:
            continue
        try:
            # Emit declared findings directly, including when the diagnostic
            # assembly cache contains a previous strict refusal.
            with warnings.catch_warnings(record=True) as findings:
                warnings.simplefilter('always')
                admit_dc_plane(tc, grid, pmc_faces=sim._boundary_spec.pmc_faces(), mode=sim._mode)
                if i in films:
                    check_film_area(tc.shape, grid, films[i], snap=sim._snap)
                else:
                    admit_dc_film(tc.shape, grid, snap=sim._snap)
        except DCFilmAdmissionError as exc:
            warn.warn(PreflightWarning(str(exc), code=exc.code, severity='error',
                                       source=exc.source, loc=f'thin_conductors[{i}]'))
        else:
            normal, plane = dc_film_plane(tc.shape, grid)
            face = None
            if not is_tracer(plane):
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
    """Sample in-plane cell centres, retaining normal plane identification."""
    lo, hi = sheet_bounds(shape)
    normal = None if lo is None or hi is None else sheet_normal_axis(lo, hi)
    tangents = tuple(a for a in range(3) if a != normal)
    sample = list(footprint_sample_coords(grid, tangents, carrier="cell"))
    if getattr(grid, 'periodic_axes', ''):
        from rfx._periodic import periodic_mask, plane_coordinate
        if normal is not None and hi[normal] - lo[normal] <= float(grid.cells(normal)[0]):
            mid = .5 * (lo[normal] + hi[normal])
            shift = mid - plane_coordinate(grid, normal, mid)
            if shift:
                sample[normal] = sample[normal] + shift
            return periodic_mask(grid, shape, sample,
                                 axes=tuple(a for a in range(3) if a != normal))
        return periodic_mask(grid, shape, sample)
    return shape.mask_on_coords(*sample)


def fold_thin_conductor(grid, conductor, materials, pec_mask=None, *,
                        sheet_specs=None, sheets=None, geometry_masks=None,
                        geometry_key=None, snap='strict', film_geometry=None):
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
            mask, drawn = sheet_footprint_traced(
                conductor.shape, coords, normal, sizes, with_drawn=True)
            plane, unwrapped = None, None
            end_rows = sheet_end_rows(conductor.shape, coords, normal)
        else:
            spec = sheet_spec_from_shape(
                conductor.shape, coords, sizes, name='thin_conductor',
                lane=lane, refuse_thick=True, grid=grid)
            if conductor.is_pec:
                if sheets is None:
                    raise ValueError(
                        "PEC thin conductor (add_thin_conductor with sigma_bulk >= "
                        "1e6 and no surface_impedance_f0) is a SHEET and this "
                        "lane collects no sheets (apply_thin_conductor(sheets=None)); "
                        "refusing to drop it silently.")
                sheets.append(spec)
                return materials, pec_mask
            mask, plane = np.asarray(spec.footprint), spec.plane
            unwrapped = (None if spec.unwrapped_footprint is None
                         else np.asarray(spec.unwrapped_footprint))
            end_rows = spec.end_rows
            drawn = spec.drawn
        check_sheet_occupancy(mask, normal, lane=lane)
    else:
        if getattr(grid, 'periodic_axes', '') and any(is_tracer(c) for c in coords[:3]):
            raise NotImplementedError(
                f"a lossy thin-conductor film on a traced graded mesh with periodic axes {grid.periodic_axes!r} "
                "is not implemented (its footprint would get no seam image); use a concrete mesh profile")
        admit_dc_plane(conductor, grid)
        film = film_geometry
        if film is None:
            film = admit_dc_film(conductor.shape, grid, snap=snap)
        else:
            from rfx.model.film_geometry import check_film_area
            check_film_area(conductor.shape, grid, film, snap=snap)
        mask, normal = film.mask, film.normal
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
                unwrapped_footprint=unwrapped, end_rows=end_rows, drawn=drawn))
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
    from rfx.model.design_box import design_box_window
    from rfx.model.film_geometry import prepare_film_footprints
    conductors = [replace(tc, shape=continued_conductor_shape(
        sim, grid, tc.shape, entry=tc)) for tc in sim._thin_conductors]
    films = prepare_film_footprints(conductors, grid)
    write, _, _, _ = design_box_window(bounds, grid.shape)
    sl = tuple(slice(write[a], write[a + 1]) for a in (0, 2, 4))
    f0_hits = set()
    if any(tc.surface_impedance_f0 is not None and not tc.is_pec for tc in sim._thin_conductors):
        from rfx.model.conductors import realized_conductors, publish_port_release
        from rfx.model.design_admission import cells_in_box, sheet_edges
        root = realized_conductors(sim, grid, nonuniform=hasattr(grid, 'dx_arr'))
        for index, component, edges in sheet_edges(root, sim):
            released = publish_port_release(root, edges, component=component,
                entity_id=f'thin_conductor[{index}]', kind='f0_sheet').released_cells
            if cells_in_box(released, bounds, grid.shape):
                f0_hits.add(index)
    for index, tc in enumerate(conductors):
        if index in f0_hits:
            raise RfxRefusal(
                f'thin_conductor[{index}] (surface_impedance_f0 sheet): the design box (cells {bounds}) '
                'contains cells incident to its tangential edges. Move the box or the sheet.')
        if tc.is_pec or tc.surface_impedance_f0 is not None:
            continue
        # Geometry decides admission even when conductance is differentiated.
        geometric = replace(tc, sigma_bulk=1.0, thickness=1.0)
        mats, _ = fold_thin_conductor(grid, geometric, init_materials(grid.shape),
                                    snap=sim._snap, film_geometry=films.get(index))
        if any(part is not None and bool(jnp.any(part[sl] != 0))
               for part in lumped_components(mats.sigma_film)):
            raise RfxRefusal(
                f"the design box (cells {bounds}) contains edges of thin conductor {index}; "
                "a design region over a declared film is not supported. Move the box or the film.")


def splice_junction_materials(materials, junction, z_index):
    """Retain the declared coax junction above its final stub row."""
    window = (slice(None), slice(None), slice(z_index, None))
    return materials._replace(
        eps_r=materials.eps_r.at[window].set(junction.eps_r[window]),
        sigma=materials.sigma.at[window].set(junction.sigma[window]),
        sigma_film=splice_film(materials, junction, window))
