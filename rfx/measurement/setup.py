"""Registration and construction of measurement plane channels."""
import jax
import jax.numpy as jnp


def register_plane(sim, axis, coordinate, component, freqs, n_freqs, name, region):
    from rfx.api._spec import _DFTPlaneEntry
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
    if component not in ("ex", "ey", "ez", "hx", "hy", "hz"):
        raise ValueError(f"component must be a field name, got {component!r}")

    sim._validate_declared_plane_coordinate(axis, coordinate)
    if freqs is None:
        if n_freqs <= 0:
            raise ValueError(f"n_freqs must be positive, got {n_freqs}")
        freqs_arr = None
    else:
        freqs_arr = jnp.asarray(freqs)
        if freqs_arr.ndim != 1 or freqs_arr.size == 0:
            raise ValueError("freqs must be a non-empty 1-D array")

    if name is None:
        name = f"{component}_{axis}_{len(sim._dft_planes)}"

    sim._dft_planes.append(_DFTPlaneEntry(
        name=name,
        axis=axis,
        coordinate=coordinate,
        component=component,
        freqs=freqs_arr,
        n_freqs=n_freqs,
    ))
    if region is not None:
        if not hasattr(sim, "_dft_plane_regions"):
            sim._dft_plane_regions = {}
        sim._dft_plane_regions[name] = tuple(region)
    return sim


def build_planes(sim, grid, n_steps):
    from rfx.probes.probes import init_dft_plane_probe
    from rfx.measurement.plan import _index
    probes = []
    for entry in sim._dft_planes:
        axis = 'xyz'.index(entry.axis)
        position = [0., 0., 0.]
        position[axis] = entry.coordinate
        with jax.ensure_compile_time_eval():
            bins = (entry.freqs if entry.freqs is not None else
                    jnp.linspace(sim._freq_max/10, sim._freq_max, entry.n_freqs))
        probes.append(init_dft_plane_probe(
            axis=axis, index=_index(grid, tuple(position))[axis], component=entry.component,
            freqs=bins, grid_shape=grid.shape, dft_total_steps=n_steps,
            region=getattr(sim, '_dft_plane_regions', {}).get(entry.name)))
    return probes
