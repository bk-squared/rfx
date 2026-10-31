"""Turn body geometry into conformal update operands on the realized grid."""

import numpy as np
import jax.numpy as jnp


def sc_clamp(lengths, areas):
    """Bound each edge by twice the least of its four incident face areas.

    Arrays start at the same node. Missing low-side faces have no constraint;
    geometry supplies a halo so every returned physical edge has four faces.
    """
    result = []
    for axis, length in enumerate(lengths):
        bound = np.ones_like(length)
        for normal in range(3):
            if normal == axis:
                continue
            transverse = 3 - axis - normal
            previous = np.roll(areas[normal], 1, axis=transverse).copy()
            low = [slice(None)] * 3
            low[transverse] = 0
            previous[tuple(low)] = 1.
            bound = np.minimum(bound, np.minimum(areas[normal], previous))
        result.append(np.minimum(length, 2 * bound))
    return tuple(result)


def conformal_b_floor(lengths, areas, q):
    """Raise cut H faces to q**2 times their largest bounding E edge.

    A face normal to ``normal`` at a node has two edges along each
    in-plane axis: at that node and one node higher along the other
    in-plane axis. Missing high-side edges supply zero; geometry supplies
    a high halo so every returned physical face has all four edges.
    Neither lengths nor uncut (zero or one) areas are changed.
    """
    result = []
    for normal, area in enumerate(areas):
        largest = np.zeros_like(area)
        for axis in range(3):
            if axis == normal:
                continue
            transverse = 3 - axis - normal
            high = np.roll(lengths[axis], -1, axis=transverse).copy()
            end = [slice(None)] * 3
            end[transverse] = -1
            high[tuple(end)] = 0.
            largest = np.maximum(largest, np.maximum(lengths[axis], high))
        result.append(np.where((area > 0) & (area < 1),
                               np.maximum(area, q**2 * largest), area))
    return tuple(result)


def conformal_fractions_from_table(table, grid, conformal_pec, *, clamp=True):
    """Apply the operand rule, physical slice, then the float32 device cast."""
    lengths, areas = table.lengths, table.areas
    if conformal_pec == "cut_cell_floor":
        from rfx.grid import Grid
        q = float(grid.dt) / Grid.courant_dt(grid.dx, 2 if grid.is_2d else 3)
        areas = conformal_b_floor(lengths, areas, q)
        physical = table.floor_physical_slice
    else:
        lengths, areas = (tuple(a[table.sc_lattice_slice] for a in arrays)
                          for arrays in (lengths, areas))
        if clamp:
            lengths = sc_clamp(lengths, areas)
        physical = table.sc_physical_slice
    return tuple(tuple(jnp.asarray(a[physical], dtype=jnp.float32) for a in arrays)
                 for arrays in (lengths, areas))
