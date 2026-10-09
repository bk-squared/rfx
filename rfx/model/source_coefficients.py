"""Run-specific source coefficient selection and absorber admission."""
from rfx._grid_metric import field_index
from dataclasses import replace


def drive_table(series, meta, n_steps, *, occupancy=None, periodic=(False,) * 3,
                sheet_edge_masks=None, design=None, shape=None, pad=False):
    """Stack resolved drives, storing each electric edge's occupancy weight.

    Occupancy is the released input used by this lane's step. The optional
    design window replaces its background weight, including static masks.
    Traced occupancy stays traced through the gather and table product.
    With neither occupancy nor a design window, return the original stack
    without any multiplication. Magnetic columns are never multiplied.
    ``pad`` retains the decay runner's truncation/zero-padding to n_steps.
    """
    import jax.numpy as jnp
    from rfx.model.occupancy import build_edge_keep

    if pad:
        series = [s[:n_steps] if s.shape[0] >= n_steps else
                  jnp.pad(s, (0, n_steps - s.shape[0])) for s in series]
    table = (jnp.stack(series, axis=-1) if series else
             jnp.zeros((n_steps, 0), dtype=jnp.float32))
    if not series or (occupancy is None and design is None):
        return table
    keep = build_edge_keep(occupancy, dtype=table.dtype, periodic=periodic,
                           sheet_edge_masks=sheet_edge_masks,
                           design=design, shape=shape)
    for column, (i, j, k, component) in enumerate(meta):
        if component in ('ex', 'ey', 'ez'):
            weight = keep[('ex', 'ey', 'ez').index(component)][i, j, k]
            table = table.at[:, column].set(table[:, column] * weight)
    return table


def source_table(samples, kind=None, *, native="raw", coefficient=None, volume=1.0):
    """Build per-step E increments from samples and a final edge coefficient.

    A field increment has no material operand, including under AD. A callable
    coefficient is read only for a material-dependent drive, after all stamps.
    ``native`` preserves low-level helpers' historical None contracts and the
    current-drive product order (open uniform: Cb*w then 1/dV). Distributed
    slabs may supply their already resolved Cb/dV as a native ``cb`` factor;
    the same multiplication then builds their table inside the program.
    """
    from rfx.api._source_semantics import needs_scale
    needs_scale(kind, native)  # validate Python-level dispatch, never a tracer
    if kind == "field" or (kind is None and native == "raw"):
        return samples
    cb = coefficient() if callable(coefficient) else coefficient
    if kind == "current" and native == "cb":
        return (1.0 / volume) * (cb * samples)
    if kind == "current" or native == "cb_over_dv":
        return (cb / volume) * samples
    return cb * samples


def uniform_source_table(grid, cell, component, waveform, n_steps, materials,
                         kind=None, *, native="raw", periodic=(False,) * 3):
    """Uniform and sweep adapters share samples, edge reads and product order."""
    import jax
    import jax.numpy as jnp
    from rfx.model.materials import e_update_coefficient_at, electric_grid_kwargs
    from rfx.simulation import _uniform_cell_volume
    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    return source_table(
        jax.vmap(waveform)(times), kind, native=native,
        volume=_uniform_cell_volume(grid),
        coefficient=lambda: e_update_coefficient_at(
            materials, cell, component, grid.dt, periodic,
            **electric_grid_kwargs(grid)))


def vacuum_source_table(samples, kind, dt, volume):
    """The disjoint research stepper updates homogeneous vacuum only."""
    from rfx.core.yee import e_update_coeffs
    return source_table(samples, kind, volume=volume,
                        coefficient=lambda: e_update_coeffs(1.0, 0.0, dt)[1])


def field_source_samples(waveform, n_steps, dt):
    """ADI's prescribed field table, using its solver-specific timestep."""
    import jax
    import jax.numpy as jnp
    return source_table(jax.vmap(waveform)(jnp.arange(n_steps, dtype=jnp.float32) * dt), "field")


def scale_source_columns(xs, scales, columns):
    """Finish distributed current tables from slab-local Cb/dV reads."""
    steps, table = xs
    for n, col in enumerate(columns):
        table = table.at[:, col].set(source_table(
            table[:, col], native="cb", coefficient=scales[n]))
    return steps, table


def subgrid_source_tables(requests, grid, materials, n_steps, *, native,
                          fine_scale, volume, shadow_entries=None):
    """Resolve fine-grid drives after every load; project completed soft tables.

    Port descriptors retain their own Norton load and unit-voltage shape;
    their coefficient, like a soft current's, reads the final fine materials.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from rfx.model.materials import e_update_coefficient_at
    from rfx.sources.port_drive import port_drive_waveform

    fine, coarse = [], []
    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    for pe, cell, drive in requests:
        if drive is None:
            waveform = source_table(
                jax.vmap(pe.waveform)(times), pe.amplitude_kind, native=native,
                volume=volume,
                coefficient=lambda: float(e_update_coefficient_at(
                    materials, cell, pe.component, grid.dt)))
            fine.append((*cell, pe.component, np.array(waveform) * fine_scale))
            if shadow_entries is not None:
                coarse.extend(shadow_entries(pe.position, cell, pe.component, waveform))
        else:
            sigma, unit_field = drive
            waveform = port_drive_waveform(
                grid, cell, pe.component, pe.waveform, n_steps, materials,
                sigma_port=sigma, unit_field=unit_field)
            fine.append((*cell, pe.component, np.array(waveform)))
    return fine, coarse


def edge_in_absorber_pad(grid, cell, component=None):
    """Whether the UPML update on this edge carries an absorber conductivity.

    The UPML E update of a component uses only the conductivities of the
    axes perpendicular to it, so an edge inside the pad of its own axis
    alone keeps the interior coefficient. ``component=None`` asks about any
    pad (a launch that drives more than one component).
    """
    axes = getattr(grid, "cpml_axes", "xyz")
    own = component[-1] if component else None
    return any(axis in axes and axis != own and (
        cell[a] < getattr(grid, f"pad_{axis}_lo", 0) or
        cell[a] >= grid.shape[a] - getattr(grid, f"pad_{axis}_hi", 0))
        for a, axis in enumerate("xyz"))


def _refuse_pad_edge(grid, cell, label, component=None):
    if edge_in_absorber_pad(grid, cell, component):
        raise ValueError(
            f"{label} edge {tuple(cell)} lies inside a UPML absorber pad; "
            "its source coefficient does not include the absorber operator. "
            "Move the source/port edge into the interior.")


def resolve_run_sources(queue, materials, sim, grid, *, tensor=False):
    """Resolve after all stamps; preserve tensor branches pending their audit."""
    upml = sim._boundary == "upml" and grid.cpml_layers > 0
    undriven = set()
    if upml:
        from rfx.sources.sources import _wire_port_cells, wire_port_from_entry
        from rfx.sources.msl_port import _msl_yz_cells, msl_port_from_entry
        # Only a current drive multiplies the update coefficient. A source
        # that prescribes the field increment and a port that is a load only
        # inject nothing through it and are admitted anywhere.
        for n, port in enumerate(sim._ports):
            cells = (_wire_port_cells(grid, wire_port_from_entry(port))
                     if port.extent is not None else
                     [field_index(grid, port.position, port.component)])
            prescribed = (port.impedance == 0.0
                          and port.amplitude_kind == "field")
            if prescribed or not getattr(port, "excite", True):
                undriven.update((tuple(int(c) for c in cell), port.component)
                                for cell in cells)
                continue
            for cell in cells:
                _refuse_pad_edge(grid, cell, f"source/port[{n}]", port.component)
        for n, port in enumerate(sim._msl_ports):
            for cell in _msl_yz_cells(grid, msl_port_from_entry(port)):
                _refuse_pad_edge(grid, cell, f"msl_port[{n}]")
        for n, port in enumerate(sim._floquet_ports):
            position = [length / 2 for length in sim._domain]
            position["xyz".index(port.axis)] = port.position
            _refuse_pad_edge(grid, grid.position_to_index(tuple(position)),
                             f"floquet_port[{n}]")
        materials = materials._replace(components=replace(
            materials.components, source_upml=not tensor))
    sources = queue.resolve(materials)
    if upml:
        for source in sources:
            cell = (int(source.i), int(source.j), int(source.k))
            if (cell, source.component) not in undriven:
                _refuse_pad_edge(grid, cell, f"{source.component} source",
                                 source.component)
    return sources


def dispersive_drive_model(materials, debye_spec, lorentz_spec, grid=None):
    """Materials a drive reads its coefficient from on a lane that keeps no
    realization of its own (the distributed lanes). Nothing whole-domain is
    built: a dispersive model carries its pole specs, read edge by edge."""
    if debye_spec is None and lorentz_spec is None and grid is None:
        return materials
    from rfx.model.materials import EdgePoles
    return materials._replace(components=EdgePoles(debye_spec, lorentz_spec, grid))


def debye_pole_term(drive_model, cell, component, dt):
    """The Debye poles' part of one edge's ADE coefficient, as the
    conductivity that adds the same term: ``Cb = dt/(eps + sum(beta) +
    sigma*dt/2)``, so ``sum(beta)`` equals a conductivity ``2*sum(beta)/dt``.
    It does not depend on eps_inf or sigma, so it stays a host constant
    under a traced override."""
    from rfx.model.materials import EdgePoles
    c = getattr(drive_model, "components", None)
    debye = c.at(cell, dt)[0] if isinstance(c, EdgePoles) else None
    if debye is None:
        return 0.0
    axis = {"ex": 0, "ey": 1, "ez": 2}[str(component).lower()]
    return 2.0 * float(debye[1][axis].sum()) / float(dt)


def tensor_replaced_edges(materials, *, aniso_eps=None, aniso_inv_eps=None, upml=False):
    """Audit predicate: edges whose tensor epsilon differs from the plain edge.

    This observes finished kernel operands, including the same final stamps.
    It does not refuse sources or change smoothing. Compare inverse values
    for Stage 2 so a frozen PEC edge (inverse=0) remains representable.
    """
    baseline = materials.components.upml_eps if upml else materials.components.eps_update
    if aniso_inv_eps is not None:
        return tuple(tensor != 1.0 / plain for tensor, plain in
                     zip(aniso_inv_eps, baseline))
    if aniso_eps is not None:
        return tuple(tensor != plain for tensor, plain in
                     zip(aniso_eps, baseline))
    return None
