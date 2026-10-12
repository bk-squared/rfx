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


def flux_weights(nodes, bounds, tangential, components,
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


def normal_weights(nodes, index, *, uniform=False):
    """Distance weights of the H half-cell planes bracketing an E node.

    The uniform arithmetic is retained exactly. At the low outer node the
    weights stay equal; the sampler selects the clamped or wrapped neighbour.
    """
    if uniform or index == 0 or len(nodes) <= 2:
        return (.5, .5)
    left, right = np.diff(np.asarray(nodes, dtype=float)[index-1:index+2])
    return (float(right/(left+right)), float(left/(left+right)))


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
    normal = np.asarray(getattr(coords, 'xyz'[axis]), dtype=float)
    normal_cells = np.asarray(grid.cells(axis))
    # Include the grid's existing high fence-post/ghost provider, as for the
    # tangential periodic closing coordinate. It has no physical box extent.
    normal = np.r_[normal, normal[-1] + normal_cells[-1]]
    h_weights = normal_weights(normal, index, uniform=np.all(normal_cells == normal_cells[0]))
    return init_flux_monitor(axis, index, freqs, grid.shape,
                             np.asarray(grid.cells(tangential[0])),
                             np.asarray(grid.cells(tangential[1])),
                             node_coords=tuple(nodes), periodic_axes=periodic_axes,
                             pmc_faces=pmc_faces, normal_h_weights=h_weights, **kwargs)


def flux_regions(mon):
    """Each stored product has its own tangential sampling extent."""
    return tuple((mon.lo1, mon.lo1 + acc.shape[-2], mon.lo2, mon.lo2 + acc.shape[-1])
                 for acc in (mon.e1_dft, mon.e2_dft, mon.h1_dft, mon.h2_dft))


def monitor_periodic_flags(sim, grid):
    """Use the uniform stepper's TF/SF override; graded stepping does not wrap."""
    from rfx.nonuniform import NonUniformGrid
    if isinstance(grid, NonUniformGrid):
        return (False, False, False)
    if sim._tfsf is None:
        return sim._periodic_flags()
    from rfx.sources.tfsf import init_tfsf
    from rfx.boundaries.tfsf import boundary_flags
    entry = sim._tfsf
    cfg, _ = init_tfsf(
        grid.nx, float(grid.cells('x')[0]), grid.dt,
        ny=grid.ny, nz=grid.nz, cpml_layers=grid.cpml_layers,
        tfsf_margin=entry.margin,
        f0=entry.f0 if entry.f0 is not None else sim._freq_max / 2,
        bandwidth=entry.bandwidth, amplitude=entry.amplitude,
        polarization=entry.polarization, direction=entry.direction,
        angle_deg=entry.angle_deg, waveform=entry.waveform,
        method=entry.method, closed_box=entry.closed_box,
    )
    return boundary_flags(cfg, grid)[0]


def normal_h_neighbor(index, size, periodic):
    """H half-cell immediately before the node, including a periodic seam."""
    return (index-1) % size if periodic else max(index-1, 0)


def build_flux_monitor_cfgs(sim, grid, n_steps, entries=None, *, periodic=None):
    """Materialize ``sim._flux_monitors`` entries into scan monitor configs.

    Shared extraction of the ``run_uniform`` inline block so
    the low-level forward lane (``_forward_from_materials``, issue-#488
    mixed-family flux magnitude channel) can register the SAME monitors
    the run() lane does. Pure function of (sim registrations, grid,
    n_steps). Finite regions resolve to interior cells via the same geometry
    resolver used by preflight; ``size=None`` keeps the legacy full plane.

    ``entries`` overrides the entry list (same ``_FluxMonitorEntry``
    objects ``add_flux_monitor`` registers) without touching the sim's
    own registrations — the ``extra_flux_monitors=`` opt-in on the coax
    extractors (#589 flux-adjudication instrument) passes caller-built
    entries here while the registered-monitor guard stays intact. ``sim``
    is still consulted for ``_freq_max``/``_domain``.
    """
    import jax.numpy as jnp
    periodic = monitor_periodic_flags(sim, grid) if periodic is None else periodic
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    from rfx.probes.flux_region import resolve_flux_region
    from rfx.boundaries.pec import resolve_wall_faces
    flux_monitors = []
    if entries is None:
        entries = getattr(sim, '_flux_monitors', [])
    for pe in entries:
        axis_idx = axis_to_index[pe.axis]
        plane_pos = [0.0, 0.0, 0.0]
        plane_pos[axis_idx] = pe.coordinate
        grid_index = grid.position_to_index(tuple(plane_pos))[axis_idx]
        freqs_arr = (
            pe.freqs
            if pe.freqs is not None
            else jnp.linspace(sim._freq_max / 10, sim._freq_max, pe.n_freqs)
        )
        # Compute tangential index bounds from size (finite flux region)
        tangential_axes = [a for a in range(3) if a != axis_idx]
        grid_ns = [grid.shape[a] for a in tangential_axes]
        region = resolve_flux_region(grid, pe, sim._domain)
        if region is not None:
            (lo1, hi1), (lo2, hi2) = region["cell_slices"]
            grid_index = region["normal_index"]
        else:
            lo1, hi1 = 0, grid_ns[0]
            lo2, hi2 = 0, grid_ns[1]
        flux_monitors.append(
            grid_flux_monitor(
                grid, axis=axis_idx, periodic=periodic,
                index=grid_index,
                freqs=freqs_arr,
                dft_total_steps=n_steps,
                dft_window=getattr(pe, 'dft_window', 'rect'),
                dft_window_alpha=getattr(pe, 'dft_window_alpha', 0.25),
                lo1=lo1, hi1=hi1, lo2=lo2, hi2=hi2,
                pmc_faces=resolve_wall_faces(grid, periodic)[1],
            )
        )
    return flux_monitors
