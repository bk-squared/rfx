"""Whole Yee edges between the realized coax conductor faces."""
import jax
import numpy as np


def coax_voltage_weights(grid, *, center_xy, pec_cell_mask, plane_index):
    """Return (x index, y index, length) on the centre's positive x ray.

    Ex[i,j,k] occupies the complete edge from node i to node i+1.
    The first dielectric edge after the pin and the last before the shell
    both have full length. PEC ownership includes the preceding z cell.
    """
    from rfx.boundaries.pec import realized_pec_edge_masks
    if pec_cell_mask is None or plane_index is None:
        raise ValueError('coax voltage requires realized PEC cells and a plane index')
    with jax.ensure_compile_time_eval():
        cells = np.asarray(pec_cell_mask, dtype=bool)
        if cells.shape != grid.shape:
            raise ValueError('coax voltage conductor mask must match the grid shape')
        k = int(plane_index)
        if not 0 < k < grid.shape[2]:
            raise ValueError('coax voltage plane must be inside the grid')
        i0, j, _ = grid.position_to_index((*center_xy, grid.node_of(2, k)))
        edges = realized_pec_edge_masks(cells[:, :, k-1:k+1], periodic=tuple(
            a in getattr(grid, 'periodic_axes', '') for a in 'xy') + (False,))
        metal = np.asarray(edges[0])[:, int(j), 1]
        if not metal[int(i0)]:
            raise ValueError('coax voltage ray has no realized centre pin')
        free = np.flatnonzero(~metal[int(i0):])
        if not free.size:
            raise ValueError('coax voltage ray has no dielectric gap')
        first = int(i0) + int(free[0])
        shell = np.flatnonzero(metal[first:])
        if not shell.size:
            raise ValueError('coax voltage ray has no realized outer conductor')
        stop = first + int(shell[0])
        return tuple((i, int(j), float(grid.dx)) for i in range(first, stop))
