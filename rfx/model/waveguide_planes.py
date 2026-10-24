"""Where a waveguide port measures, computed from the declaration for admission (#1913)."""
from __future__ import annotations

import numpy as np


def with_segment_declaration(config, entry):
    """Carry the declared reference to setup, including every mode of a port."""
    if isinstance(config, list):
        return [with_segment_declaration(cfg, entry) for cfg in config]
    return config._replace(
        segment_planes=() if entry.reference_plane is None else (entry.reference_plane,),
        port_name=entry.name or "")


def offset_plane(source_m, source_index, offset, dx, *, grid=None, axis="x"):
    """Signed cell offset, integrating primal spacing on a graded axis."""
    if grid is None:
        return source_m + offset * dx
    cells = np.asarray(grid.cells(axis))
    target = source_index + offset
    lo, hi = sorted((source_index, target))
    return source_m + float(np.sum(cells[lo:hi])) * (1.0 if offset >= 0 else -1.0)


def entry_planes(entry, grid):
    """Source and measured reflection plane, without solving a mode."""
    axis = entry.direction[1]
    index = grid.index_of(axis, entry.x_position)
    if hasattr(grid, "dx_arr"):
        # Match the NU builder's interior-edge cumulative sum exactly.
        pad = getattr(grid, f"pad_{axis}_lo")
        cells = np.asarray(grid.cells(axis))
        source = float(np.insert(np.cumsum(cells[pad:]), 0, 0.0)[index - pad])
        metric_grid = grid
    else:
        source = grid.node_of(axis, index)
        metric_grid = None
    sign = 1 if entry.direction.startswith("+") else -1
    reference = offset_plane(source, index, sign * entry.ref_offset,
                             float(grid.cells(axis)[index]),
                             grid=metric_grid, axis=axis)
    return source, reference
