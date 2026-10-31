"""Host node positions shared by rasterization and cell geometry."""

import numpy as np


def _uniform_axis_nodes(n: int, pad: int, dx: float) -> np.ndarray:
    """Exact E-node positions for a uniform axis: host float64, flag-independent.

    ``(i - pad) * dx`` per node in float64 — bit-identical to the x64=1
    realization verified against the Box half-open convention. This is
    THE uniform node formula: ``csg._grid_coords``, ``coords_from_uniform_grid``
    and the uniform-valued branch of ``_axis_node_positions`` all call it, so
    the uniform and non-uniform lanes cannot rasterize a uniform-valued axis
    differently. Before this builder existed the repo carried
    three hand-copies that each rounded differently in float32.
    """
    return (np.arange(n, dtype=np.float64) - pad) * float(dx)


def _axis_node_positions(d_arr: np.ndarray, cpml: int) -> np.ndarray:
    """E-node positions for a padded cell-size array.

    The nodes this grid steps sit on cell EDGES, not centres: the
    non-uniform E update divides by ``2/(d[i-1]+d[i])``, the dual spacing
    of a node straddling cells ``i-1`` and ``i``. Node ``cpml`` (the first
    interior one) is the origin, so the interior spans
    ``[0, sum(interior d)]`` and the last interior node lands exactly on
    the requested domain face — the same convention
    ``coords_from_uniform_grid`` uses (``(arange - pad) * dx``).

    Previously this returned cell CENTRES, half a cell off the nodes the
    stencil differences and half a cell off the uniform builder, which put
    every rasterized material half a cell away from the fields acting on
    it and (with the missing bounding node) made a PEC-bounded guide one
    cell narrower than requested.
    """
    d = np.asarray(d_arr, dtype=np.float64)
    if d.size and bool(np.all(d == d[0])):
        # Uniform-valued axis: closed form, bitwise-equal to the uniform
        # builder by construction — the lane-equality guarantee.
        return _uniform_axis_nodes(d.size, cpml, float(d[0]))
    edges = np.insert(np.cumsum(d), 0, 0.0)           # len = n+1
    return edges[:-1] - edges[cpml]                   # n nodes, origin at cpml
