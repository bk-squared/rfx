"""Shared sheet-plane setup and allocation-bounded plane lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import numpy as np

if TYPE_CHECKING:
    from rfx.geometry.rasterize_grid import GridCoords


class _SheetPlaneSetup(NamedTuple):
    """Resolved coordinates, bounds and plane shared by sheet consumers."""

    coords: GridCoords
    cell_sizes: tuple
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    normal_axis: int
    node_axes: list
    requested_mid: float
    plane: int


def _sheet_plane_setup(shape, coords: GridCoords, cell_sizes=None, *,
                       normal_axis=None, name=None, lane: str = "",
                       refuse_thick: bool = False, grid=None, nominal_grid=None):
    """Resolve the shared static sheet geometry before footprint sampling.

    On a periodic in-plane axis the node line keeps its endpoint at the
    period: endpoints stay distinct until edge enumeration, because folding
    L into 0 here would either lose the final edge or add an edge near 0.
    """
    from rfx._internal._grid_metric import node_tie_tol
    from rfx._internal._periodic import plane_coordinate
    from rfx.core.jax_utils import is_tracer
    from rfx.errors import RfxRefusal
    from rfx.geometry.csg import OrientedBox
    from rfx.geometry.rasterize_grid import (
        _is_traced_coords, _local_cell, _nearest_plane, axis_cell_sizes,
        cell_sizes_from_nonuniform_grid, coords_from_nonuniform_grid,
        footprint_sample_coords,
    )
    from rfx.materials.thin_conductor import sheet_bounds

    if isinstance(shape, OrientedBox):
        raise NotImplementedError(
            "OrientedBox is a volume; axis-aligned sheet conversion is not implemented")
    traced = _is_traced_coords(coords)
    reference_grid = nominal_grid if nominal_grid is not None else grid
    if traced and reference_grid is None:
        raise RfxRefusal(
            f"sheet {name!r}: a sheet's plane is a static integer, and this "
            "mesh is a JAX tracer (mesh-as-design-variable), so the nearest "
            "node plane cannot be resolved. Pin the mesh, or declare the "
            "conductor as a volume.")
    lo, hi = sheet_bounds(shape)
    if lo is None or hi is None:
        raise ValueError(
            f"sheet {name!r}: shape {type(shape).__name__} has no axis-aligned "
            "bounding box (Box corner_lo/corner_hi or Shape.bounding_box()), "
            "so its normal and plane cannot be located.")
    lo = tuple(float(v) for v in lo)
    hi = tuple(float(v) for v in hi)
    extents = [hi[i] - lo[i] for i in range(3)]
    if normal_axis is None:
        normal_axis = min(range(3), key=lambda i: extents[i])
    a = int(normal_axis)
    if traced:
        if is_tracer(coords[a]):
            raise RfxRefusal(
                f"sheet {name!r}: a sheet's plane is a static integer, and "
                f"normal axis {'xyz'[a]} is a JAX tracer; pin that axis or "
                "declare the conductor as a volume.")
        for axis in range(3):
            if axis != a and is_tracer(coords[axis]) and 'xyz'[axis] not in (
                    getattr(reference_grid, "nominal_cells", None) or {}):
                raise RfxRefusal(
                    f"sheet {name!r}: traced in-plane axis {'xyz'[axis]} needs "
                    f"nominal_profiles; declare nominal_profiles for axis {'xyz'[axis]} "
                    "or use add_pinned_sheet.")
        from rfx.model.graded_grid import host_nominal_grid
        with jax.ensure_compile_time_eval():
            nominal = host_nominal_grid(reference_grid)
            coords = coords_from_nonuniform_grid(nominal)
            cell_sizes = cell_sizes_from_nonuniform_grid(nominal)
    node_axes = footprint_sample_coords(
        coords, tuple(t for t in range(3) if t != a), carrier="node")
    if cell_sizes is None:
        cell_sizes = tuple(axis_cell_sizes(n) for n in node_axes)
    requested_mid = 0.5 * (lo[a] + hi[a])
    mid = plane_coordinate(grid, a, requested_mid)
    d_local = _local_cell(node_axes[a], cell_sizes[a], mid)
    if refuse_thick and extents[a] > (d_local + node_tie_tol(lo[a], d_local)
                                    + node_tie_tol(hi[a], d_local)):
        raise ValueError(
            f"add_thin_conductor: shape {name!r} is {extents[a]:.6g} m thick "
            f"along {'xyz'[a]} against a local cell of {d_local:.6g} m — not a "
            "sheet; use add() for a volume (the lattice ownership contract "
            "realizes a volume with both faces and a shorted interior).")
    plane = _nearest_plane(node_axes[a], mid, d_local,
                           what="sheet", name=name, axis=a)
    node_axes = list(node_axes)
    periodic = getattr(grid, 'periodic_axes', '')
    for t in range(3):
        if t != a and 'xyz'[t] in periodic:
            node_axes[t] = np.append(node_axes[t], float(grid.domain[t]))
    return _SheetPlaneSetup(
        coords, cell_sizes, lo, hi, a, node_axes, requested_mid, plane)


def sheet_plane_from_shape(shape, coords: GridCoords, cell_sizes=None, *,
                           normal_axis=None, name=None, lane: str = "",
                           refuse_thick: bool = False, grid=None, nominal_grid=None):
    """Resolve a nonempty sheet's node plane without a volume footprint."""
    from rfx._internal._periodic import _fill_image_range, periodic_mask
    from rfx.geometry.rasterize_grid import _box_axis_closed, _local_cell

    setup = _sheet_plane_setup(
        shape, coords, cell_sizes, normal_axis=normal_axis, name=name, lane=lane,
        refuse_thick=refuse_thick, grid=grid, nominal_grid=nominal_grid)
    _, cell_sizes, lo, hi, a, node_axes, requested_mid, plane = setup
    periodic = getattr(grid, 'periodic_axes', '')
    if getattr(shape, "corner_lo", None) is not None:
        occupied = True
        for t in range(3):
            if t == a:
                continue
            line = node_axes[t]
            d_t = _local_cell(line, cell_sizes[t], 0.5 * (lo[t] + hi[t]))
            images = (_fill_image_range(grid, shape, t, closed_footprint=True)
                      if 'xyz'[t] in periodic else (0,))
            folded = np.zeros(len(line), dtype=bool)
            for k in images:
                sample = line + k * float(grid.domain[t]) if k else line
                folded |= np.asarray(_box_axis_closed(sample, lo[t], hi[t], d_t))
            occupied = occupied and bool(folded.any())
    else:
        sample_coords = [np.asarray(n, dtype=np.float64) for n in node_axes]
        sample_coords[a] = np.asarray([requested_mid], dtype=np.float64)
        cross = np.asarray(periodic_mask(
            grid, shape, sample_coords, axes=tuple(t for t in range(3) if t != a),
            closed_footprint=True), dtype=bool)
        occupied = bool(cross.any())
    if not occupied:
        raise ValueError(
            f"sheet {name!r}: the footprint rasterizes to ZERO nodes on this "
            f"grid{(' (' + lane + ' lane)') if lane else ''} at plane "
            f"{'xyz'[a]}={plane}; it would silently vanish. "
            "Widen the footprint to reach a node line or refine the mesh.")
    return plane

