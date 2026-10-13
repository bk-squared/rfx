"""Admission of a waveguide port's measurement planes on the realized grid."""
from __future__ import annotations

import numpy as np


def admit_waveguide_planes(sim, *, axis_idx, x_position, direction,
                           ref_offset, probe_offset):
    """Refuse a waveguide port whose snapped planes leave the domain.

    The source plane snaps to the nearest grid node on the port axis; the
    reference and probe planes sit ``ref_offset`` and ``probe_offset`` cells
    from it in the port's direction. All three must stay inside the physical
    domain and inside the grid's index range.
    """
    axis_name = "xyz"[axis_idx]
    pos_vec = [0.0, 0.0, 0.0]
    pos_vec[axis_idx] = x_position
    step_sign = 1 if direction.startswith("+") else -1
    if sim._uses_nonuniform_mesh:
        from rfx.nonuniform import position_to_index
        from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid
        grid = sim._build_nonuniform_grid()
        x_index = position_to_index(grid, tuple(pos_vec))[axis_idx]
        coords = coords_from_nonuniform_grid(grid)
        nodes = np.asarray((coords.x, coords.y, coords.z)[axis_idx])
        measured_reference_plane = float(nodes[np.clip(
            x_index + step_sign * ref_offset, 0, len(nodes) - 1)])
        measured_probe_plane = float(nodes[np.clip(
            x_index + step_sign * probe_offset, 0, len(nodes) - 1)])
    else:
        grid = sim._build_grid(extra_waveguide_axes=axis_name)
        x_index = grid.position_to_index(tuple(pos_vec))[axis_idx]
        axis_pad = grid.axis_pads[axis_idx]
        # A uniform grid's cells are np.full(n, dx): the same float as grid.dx.
        d = float(grid.cells(axis_idx)[x_index])
        snapped_source_plane = (x_index - axis_pad) * d
        measured_reference_plane = snapped_source_plane + step_sign * ref_offset * d
        measured_probe_plane = snapped_source_plane + step_sign * probe_offset * d
    axis_domain = sim._domain[axis_idx]
    if (
        measured_reference_plane < 0.0
        or measured_reference_plane > axis_domain
        or measured_probe_plane < 0.0
        or measured_probe_plane > axis_domain
        or x_index + step_sign * ref_offset < 0
        or x_index + step_sign * ref_offset >= grid.shape[axis_idx]
        or x_index + step_sign * probe_offset < 0
        or x_index + step_sign * probe_offset >= grid.shape[axis_idx]
    ):
        raise ValueError(
            "Waveguide port measurement planes exceed the physical "
            f"{axis_name}-domain after grid snapping; reduce ref_offset/probe_offset, "
            "flip direction, or move x_position inward"
        )
