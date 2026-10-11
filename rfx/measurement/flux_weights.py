"""Flux quadrature on realized nodes, including full-plane storage conventions."""
import numpy as np


def axis_weights(nodes, lo, hi, *, nodal, periodic=False, extrusion=None):
    """Integrate a cell window; full planes retain a zero-weight high ghost.

    Periodic coordinates include the closing node, supplied by the grid adapter.
    A collapsed axis carries only its extruded width and is not a quadrature axis.
    """
    if extrusion is not None:
        return np.asarray([extrusion], dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    size = len(nodes) - int(periodic)
    widths = np.diff(nodes)
    if periodic and lo == 0 and hi == size:
        return .5 * (np.roll(widths, 1) + widths) if nodal else widths
    full = hi == size and not periodic
    end = hi - 1 if full else hi
    cells = widths[lo:end]
    if nodal:
        return np.r_[.5*cells[0], .5*(cells[:-1]+cells[1:]), .5*cells[-1]]
    return np.r_[cells, 0.] if full else cells


def flux_weights(nodes, bounds, tangential, components, magnetic_faces=(),
                 periodic_axes='', extrusions=(None, None)):
    """Return one area array for each E/H product.

    Magnetic-face window endpoints use the same clipped dual widths as every
    other endpoint; they require no second halving of the trapezoid.
    """
    areas = []
    for component in components:
        weights = [axis_weights(x, lo, hi, nodal=component != axis,
                                periodic='xyz'[axis] in periodic_axes, extrusion=extrusion)
                   for x, (lo, hi), axis, extrusion in
                   zip(nodes, bounds, tangential, extrusions)]
        areas.append(weights[0][:, None] * weights[1][None, :])
    return tuple(areas)


def grid_flux_monitor(grid, *, axis, index, freqs, pmc_faces=(), periodic=None, **kwargs):
    """Adapt either grid's realized coordinates to the sole monitor constructor."""
    from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid, coords_from_uniform_grid
    from rfx.nonuniform import NonUniformGrid
    from rfx.probes.probes import init_flux_monitor

    coords = (coords_from_nonuniform_grid(grid) if isinstance(grid, NonUniformGrid)
              else coords_from_uniform_grid(grid))
    tangential = [a for a in range(3) if a != axis]
    periodic_axes = (getattr(grid, 'periodic_axes', '') if periodic is None else
                     ''.join(a for a, enabled in zip('xyz', periodic) if enabled))
    nodes = []
    for a in tangential:
        x = np.asarray(getattr(coords, 'xyz'[a]), dtype=float)
        if 'xyz'[a] in periodic_axes and len(x) > 1:
            x = np.r_[x, x[-1] + np.asarray(grid.cells(a))[-1]]
        nodes.append(x)
    return init_flux_monitor(axis, index, freqs, grid.shape,
                             np.asarray(grid.cells(tangential[0])),
                             np.asarray(grid.cells(tangential[1])),
                             node_coords=tuple(nodes), periodic_axes=periodic_axes,
                             pmc_faces=pmc_faces, **kwargs)


def flux_regions(mon):
    """Each stored product has its own tangential sampling extent."""
    return tuple((mon.lo1, mon.lo1 + acc.shape[-2], mon.lo2, mon.lo2 + acc.shape[-1])
                 for acc in (mon.e1_dft, mon.e2_dft, mon.h1_dft, mon.h2_dft))
