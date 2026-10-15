"""Physical lengths for S-parameter geometry advisories."""
import numpy as np


def node_distance(grid, axis, first, second):
    """Node separation; retain uniform multiplication's exact rounding."""
    if grid.is_constant(axis):
        return abs(second - first) * float(grid.boundary_cell(axis, "lo"))
    return abs(float(grid.node_of(axis, second)) - float(grid.node_of(axis, first)))


def absorber_depth(grid, axis, side, n_cells):
    """Sum physical pad cells, excluding the final bounding-node provider.

    Uniform multiplication preserves the old operation order. Graded pads
    use their own axis and face, including unequal low/high boundary cells.
    """
    if grid.is_constant(axis):
        return n_cells * float(grid.boundary_cell(axis, side))
    cells = np.asarray(grid.cells(axis), dtype=float)
    pad = cells[:n_cells] if side == "lo" else cells[len(cells) - 1 - n_cells:-1]
    return float(pad.sum())


def junction_absorber_depth(grid, axis, default_cells):
    """Thinnest active propagation pad, including per-face layer overrides."""
    depths = []
    for side in ("lo", "hi"):
        n = int(getattr(grid, f"pad_{axis}_{side}", default_cells))
        if n > 0:
            depths.append(absorber_depth(grid, axis, side, n))
    return min(depths, default=float("inf"))


def _calculator_source_plane(grid, port):
    """Keep the calculator's derived half-cell source on its original node.

    This is an internal one-cell descriptor, not a declared coordinate.
    Preserve its float64 half-to-even arithmetic independently of the
    declared-coordinate tie rule.
    """
    direction = 1 if port.face == "bottom" else -1
    centre = port.position[2] + direction * port.pin_length / 2.0
    dz = float(grid.cells(2)[0])   # uniform z only (coax calculators refuse a graded mesh)
    return int(round(centre / dz)) + grid.pad_z_lo


def _msl_axis_spacing(grid, axis: int):
    """Cell spacing along one grid axis, and whether that axis is GRADED.

    Returns ``(spacing_m, graded, evaluable)``:

    * uniform :class:`~rfx.grid.Grid` — ``(grid.dx, False, True)``: every
      axis carries the one scalar spacing.
    * :class:`~rfx.nonuniform.NonUniformGrid` — the axis's own interior
      cell-size array decides. ``graded`` is True when max/min differ by
      more than 1e-6 relative.  ``spacing_m`` is the (single) interior
      cell size when the axis is ungraded, ``None`` when it is graded.
    * traced (mesh-as-design-variable) profiles — ``(None, None, False)``:
      the answer is not available host-side.

    Issue #686: the #469 probe-offset interval solve used to bail on ANY
    non-uniform grid (``getattr(grid, "dz", None) is not None``). Its
    stated reason — "cell-counted intervals are ill-defined under graded
    dx" — is a statement about the PROPAGATION axis, and a ``dz_profile``
    does not grade dx. For a microstrip the propagation axis is x or y,
    so on a z-graded mesh (the boundary-fitted stackup case) the interval
    is perfectly well defined and the solve simply never ran.
    """
    from rfx.core.jax_utils import is_tracer
    from rfx.nonuniform import NonUniformGrid, interior_cells

    if not isinstance(grid, NonUniformGrid):
        return float(grid.boundary_cell(axis, "lo")), False, True
    arr = (grid.dx_arr, grid.dy_arr, grid.dz)[axis]
    pad_lo = (grid.pad_x_lo, grid.pad_y_lo, grid.pad_z_lo)[axis]
    pad_hi = (grid.pad_x_hi, grid.pad_y_hi, grid.pad_z_hi)[axis]
    if is_tracer(arr):
        return None, None, False
    cells = np.asarray(interior_cells(np.asarray(arr), pad_lo, pad_hi),
                       dtype=np.float64)
    if cells.size == 0:
        return None, None, False
    lo, hi = float(cells.min()), float(cells.max())
    if lo <= 0.0:
        return None, None, False
    graded = (hi - lo) / lo > 1e-6
    return (None if graded else lo), graded, True


def _msl_axis_cell_f64(grid, axis: int) -> float:
    """The cell of an ungraded axis, from the float64 cell spine.

    ``_msl_axis_spacing`` reads the float32 solver store, which is what the
    interval solve has always counted in. A count compared with the one
    ``add_msl_port`` made in its float64 scalar is read here instead, so an
    axis whose cell IS that scalar reproduces the stored count exactly.
    """
    pad_lo = (grid.pad_x_lo, grid.pad_y_lo, grid.pad_z_lo)[axis]
    return float(np.asarray(grid.cells(axis), dtype=np.float64)[pad_lo])
