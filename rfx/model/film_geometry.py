"""Assembly-wide DC film geometry, shared by solve and diagnostic readers."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.geometry.csg import _grid_coords
from rfx.geometry.rasterize_grid import _sheet_drawn_bounds
from rfx.materials.thin_conductor import sheet_bounds


def prepare_film_footprints(conductors, grid):
    """Return solved masks before any material fold or area verdict.

    Structural admission remains per declaration. Area admission is deferred
    until the fold so that it judges the separated, solved footprint.
    """
    from rfx.model.thin_conductors import (
        FilmFootprint, admit_dc_film, dc_film_plane, dc_footprint_areas,
        separate_film_footprints,
    )
    if sum(not tc.is_pec and tc.surface_impedance_f0 is None for tc in conductors) < 2:
        return {}
    films, geometry = [], {}
    with jax.ensure_compile_time_eval():
        for i, tc in enumerate(conductors):
            if tc.is_pec or tc.surface_impedance_f0 is not None:
                continue
            film = admit_dc_film(tc.shape, grid, snap='declared', emit=False)
            geometry[i] = film
            normal, plane = dc_film_plane(tc.shape, grid)
            lo, hi = sheet_bounds(tc.shape)
            sizes = tuple(grid.cells(a) for a in range(3))
            traced = any(is_tracer(v) for v in jax.tree_util.tree_leaves((lo, hi, sizes)))
            drawn = _sheet_drawn_bounds(lo, hi, normal, sizes, grid=grid,
                                       xp=jnp if traced else np)
            films.append(FilmFootprint(film.mask, normal, plane, drawn, i))
        separated = separate_film_footprints(films, grid)
        for film in separated:
            original = geometry[film.order]
            if original.realized_area is None:
                continue
            realized, declared = dc_footprint_areas(
                conductors[film.order].shape, grid, film.mask, film.normal)
            geometry[film.order] = replace(original, mask=film.mask,
                                           realized_area=realized, declared_area=declared)
    return geometry


def check_film_area(shape, grid, film, *, snap, emit=True):
    from rfx.model.thin_conductors import _check_dc_area, admit_dc_film
    if film.realized_area is not None:
        _check_dc_area(shape, grid, film.normal, film.realized_area,
                       film.declared_area, snap=snap, emit=emit)
    else:
        admit_dc_film(shape, grid, snap=snap, emit=emit)


def diagnostic_film_footprints(sim, grid):
    """Continue exactly as assembly does; retain per-declaration refusals."""
    from rfx.geometry.smoothing import continued_conductor_shape
    from rfx.model.thin_conductors import DCFilmAdmissionError
    conductors = [replace(tc, shape=continued_conductor_shape(
        sim, grid, tc.shape, entry=tc)) for tc in sim._thin_conductors]
    try:
        return conductors, prepare_film_footprints(conductors, grid)
    except DCFilmAdmissionError:
        # A structural refusal in one declaration must not hide the others.
        return conductors, {}


def refuse_traced_film_pairs(films, grid):
    """Refuse nearby pairs by name, retaining the check under mesh tracing."""
    from rfx._traced_check import traced_check
    from rfx.model.thin_conductors import DCFilmAdmissionError
    from rfx.geometry.rasterize_grid import _local_cell
    coords = _grid_coords(grid)
    for i, a in enumerate(films):
        for b in films[i + 1:]:
            if a.normal != b.normal:
                continue
            if not is_tracer(a.plane) and not is_tracer(b.plane) and a.plane != b.plane:
                continue
            tangents = tuple(t for t in range(3) if t != a.normal)
            bounds = tuple((*a.drawn[t][:2], *b.drawn[t][:2]) for t in tangents)
            periods = tuple(float(grid.domain[t]) if 'xyz'[t] in getattr(grid, 'periodic_axes', '')
                            else None for t in tangents)

            def check(planes, intervals, nodes, widths, *, orders=(a.order, b.order), periods=periods):
                if int(planes[0]) != int(planes[1]):
                    return
                for (alo, ahi, blo, bhi), x, d, length in zip(intervals, nodes, widths, periods):
                    near = False
                    for shift in ((0., length, -length) if length is not None else (0.,)):
                        gap = max(blo + shift - ahi, alo - bhi - shift)
                        local = max(_local_cell(np.asarray(x), np.asarray(d), edge)
                                    for edge in (alo, ahi, blo + shift, bhi + shift))
                        near |= gap <= local
                    if not near:
                        return
                raise DCFilmAdmissionError(
                    f'thin_conductor[{orders[0]}] and thin_conductor[{orders[1]}]: '
                    'DC films on one plane within one local cell require concrete '
                    'coordinates for drawn-gap separation. Use a concrete mesh and bounds.',
                    code='dc_film_traced_pair')

            traced_check('DC film drawn gap', 'nearby films', check,
                         (a.plane, b.plane), bounds, tuple(coords[t] for t in tangents),
                         tuple(grid.cells(t) for t in tangents), ordered=True)
