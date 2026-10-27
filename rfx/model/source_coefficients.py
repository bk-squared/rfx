"""Run-specific source coefficient selection and absorber admission."""
from __future__ import annotations

from rfx._grid_metric import field_index
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.core.yee import EPS_0
from rfx.grid import Grid
from rfx.model.graded_grid import NonUniformGrid, e_node_dual_spacing_at


def drive_columns(eps, sigma, keep, cells, owned, samples, volumes, dt, *, cb=None):
    """Build graded increments from local realized edges in source order.

    Run eagerly (including eager vmap over ranks): compiling these products
    may replace division by a constant with reciprocal multiplication.
    ``cb`` supplies the unchanged ADE coefficients on dispersive edges.
    A None volume denotes an already resolved field increment.
    """
    import jax.numpy as jnp

    columns = []
    for n, ((axis, cell), volume) in enumerate(zip(cells, volumes)):
        if volume is None:
            column = jnp.where(owned[n], samples[:, n], 0.)
        else:
            if cb is None:
                e, s = eps[axis][cell], sigma[axis][cell]
                pole = volume[1] if isinstance(volume, tuple) else 0.
                coefficient = current_source_cb(jnp.where(owned[n], e, 1.),
                    jnp.where(owned[n], s + pole, 0.), dt)
            else:
                coefficient = cb[n]
            if isinstance(volume, tuple) and len(volume) == 4:
                scale = jnp.where(owned[n], coefficient * volume[2] * volume[3], 0.)
                column = scale * samples[:, n]
            else:
                volume = volume[0] if isinstance(volume, tuple) else volume
                column = jnp.where(owned[n], (1.0 / volume) * (coefficient * samples[:, n]), 0.)
        if keep is not None:
            column = column * jnp.where(owned[n], keep[axis][cell], 0.)
        columns.append(column)
    return (jnp.stack(columns, axis=-1) if columns else
            jnp.zeros((samples.shape[0], 0), dtype=jnp.float32))


def drive_table(series, meta, n_steps, *, occupancy=None, periodic=(False,) * 3,
                sheet_edge_masks=None, design=None, shape=None, pad=False, keep=None):
    """Stack resolved drives, storing each electric edge's occupancy weight.

    Occupancy is the released input used by this lane's step. The optional
    design window replaces its background weight, including static masks.
    Traced occupancy stays traced through the gather and table product.
    A supplied ``keep`` reuses the published factor only without a design window.
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
    if not series or (keep is None and occupancy is None and design is None):
        return table
    if keep is None or design is not None:
        keep = build_edge_keep(occupancy, dtype=table.dtype, periodic=periodic,
                               sheet_edge_masks=sheet_edge_masks,
                               design=design, shape=shape)
    weights = [keep[('ex', 'ey', 'ez').index(component)][i, j, k]
               if component in ('ex', 'ey', 'ez') else None
               for i, j, k, component in meta]
    return weight_drive_columns(table, weights, meta)


def weight_drive_columns(table, weights, meta):
    """Store per-column electric products; magnetic columns stay untouched.

    Weights are already resolved by the caller, in the table's dtype.
    This helper has no ownership, occupancy or material reads.
    """
    for column, (i, j, k, component) in enumerate(meta):
        if component in ('ex', 'ey', 'ez'):
            weight = weights[column]
            table = table.at[:, column].set(table[:, column] * weight)
    return table


def source_table(samples, kind=None, *, native="raw", coefficient=None, volume=1.0):
    """Build per-step E increments from samples and a final edge coefficient.

    A field increment has no material operand, including under AD. A callable
    coefficient is read only for a material-dependent drive, after all stamps.
    Every current path forms ``(1/dV)*(Cb*w)``, including native
    ``cb_over_dv`` and multi-device slabs in :func:`drive_columns`.
    On concrete profiles dV comes from declared float64 widths, multiplied
    in x*y*z order; traced profiles retain their float32 volume. Keeping Cb
    in the waveform product also makes eager and jit agree with traced Cb.
    """
    from rfx.api._source_semantics import needs_scale
    needs_scale(kind, native)  # validate Python-level dispatch, never a tracer
    if kind == "field" or (kind is None and native == "raw"):
        return samples
    cb = coefficient() if callable(coefficient) else coefficient
    if native == "cb_over_dv":
        return (1.0 / volume) * (cb * samples)
    if kind == "current":
        return (1.0 / volume) * (cb * samples)
    return cb * samples


def uniform_source_table(grid, cell, component, waveform, n_steps, materials,
                         kind=None, *, native="raw", periodic=(False,) * 3):
    """Uniform and sweep adapters share samples, edge reads and product order."""
    import jax
    import jax.numpy as jnp
    from rfx.model.materials import e_update_coefficient_at, electric_grid_kwargs
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


def current_source_samples(grid, waveform_fn, n_steps):
    """The current ``I(t)`` in amperes at the ``n_steps`` step times
    ``t = n*dt`` (float32), before any ``Cb/dV`` normalisation."""
    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    return jax.vmap(waveform_fn)(times)


def current_source_cb(eps_r_c, sigma_c, dt):
    """``Cb = (dt/eps) / (1 + sigma*dt/(2*eps))`` of the edge a current
    source feeds, from that edge's ``(eps_r, sigma)``.

    ``eps_r_c`` / ``sigma_c`` are the edge's own values -- the mean over the
    four cells incident to it, :func:`rfx.core.yee.cell_component_e_materials`
    (#1210). Concrete and traced inputs use the same jnp arithmetic,
    keeping a traced permittivity on the tape (GEO-C3).
    The one spelling of the drive coefficient: :func:`make_current_source`
    and the distributed graded forward, which builds it inside its jitted
    program from the slabs its E update receives (#1279), both call it.
    """
    # In eps_r units the reverse pass avoids squaring eps ~ 1e-11,
    # which overflowed float32 cotangents into NaN gradients (#1279).
    eps_r = jnp.asarray(eps_r_c)
    sigma = jnp.asarray(sigma_c)
    return (dt / EPS_0) / (eps_r + sigma * (dt / (2.0 * EPS_0)))


def current_source_volume(grid, position_ijk, component):
    """``(dV, grid_traced)``: the control volume of the E node a current
    source at ``position_ijk`` injects into, and whether the profile is
    traced (then ``dV`` is a jnp scalar, else a Python float)."""
    i, j, k = position_ijk

    # Control volume of the E node this source injects into (issue #672).
    # An E_a component is an EDGE along its own axis a and sits ON a node on
    # the two transverse axes, so its control volume is MIXED:
    #   axis == a  -> primal per-cell width d[idx]
    #   axis != a  -> dual spacing (d[idx-1]+d[idx])/2, which is exactly the
    #                 metric the NU E update divides curl H by (CORE-C2,
    #                 ``_profile_to_inv_arrays``); the Ex update, for
    #                 instance, uses inv_dy and inv_dz and never inv_dx.
    # Using the primal width on all three axes (pre-#672) is right only where
    # both neighbour pairs are equal. On a 2:1 grading step it mis-normalizes
    # the injected current moment by the local cell ratio on each graded
    # TRANSVERSE axis (measured: dV too large by 1.333 with one graded
    # transverse axis, 1.778 with two).
    # Stay in jnp when the profile is traced so a mesh-as-design variable
    # propagates the gradient into the waveform normalisation.
    _axis_of = {"ex": 0, "ey": 1, "ez": 2}
    if component not in _axis_of:
        raise ValueError(
            f"make_current_source: unknown component {component!r} "
            f"(expected one of {sorted(_axis_of)}). The control volume is "
            f"component-dependent on a non-uniform mesh, so there is no safe "
            f"fallback (issue #672)."
        )
    p_axis = _axis_of[component]
    grid_traced = (
        is_tracer(grid.dx_arr) or is_tracer(grid.dy_arr) or is_tracer(grid.dz)
    )
    _profiles = tuple(store if grid_traced or exact is None else exact for store, exact in zip((grid.dx_arr, grid.dy_arr, grid.dz), (grid.dx_arr_f64, grid.dy_arr_f64, grid.dz_f64)))
    _idx = (i, j, k)
    _widths = []
    for _a in range(3):
        if _a == p_axis:
            _w = (jnp.asarray(_profiles[_a])[_idx[_a]] if grid_traced
                  else float(np.asarray(_profiles[_a])[_idx[_a]]))
        elif grid_traced:
            _w = e_node_dual_spacing_at(jnp.asarray(_profiles[_a]), _idx[_a])
        else:
            _w = float(e_node_dual_spacing_at(
                np.asarray(_profiles[_a], dtype=np.float64), _idx[_a]))
        _widths.append(_w)
    dx_local, dy_local, dz_local = _widths
    return dx_local * dy_local * dz_local, grid_traced


def make_current_source(grid: NonUniformGrid, position_ijk, component,
                        waveform_fn, n_steps, materials, amplitude_kind=None):
    """Create a properly normalized current source for non-uniform grid.

    The waveform specifies CURRENT (Amperes). The E-field addition is:
    E += (dt/ε) × I_source / dV
    where dV is the E node's control volume: the PRIMAL per-cell width on
    the component's own axis and the DUAL spacing ``(d[k-1]+d[k])/2`` on the
    two transverse axes. On a uniform mesh the two coincide and
    this is the familiar ``dx × dy × dz``.

    This gives resolution-independent injected POWER regardless of cell size.
    Same approach as Meep's internal source normalization.

    Native amplitude convention: ``'current'`` —
    ``amplitude_kind`` of None or ``'current'`` is bit-identical to the
    historical output (no extra multiply). ``amplitude_kind='field'``
    returns the waveform directly, without reading a material coefficient.
    The shared builder preserves the current drive's differentiable Cb/dV.
    """
    i, j, k = position_ijk

    # Read the edge's own eps and sigma, averaged over its four cells
    # (#1210), with the same differentiable drive formula on every trace.
    from rfx.model import e_update_coefficient_at
    if str(component).lower() not in ("ex", "ey", "ez"):
        raise ValueError(
            f"unknown component {component!r}: a current source injects an "
            f"electric field component, one of 'ex', 'ey', 'ez'.")
    from rfx.model import electric_materials_traced
    materials_traced = electric_materials_traced(materials)
    from rfx.model import source_table
    dV, grid_traced = current_source_volume(grid, (i, j, k), component)
    any_traced = materials_traced or grid_traced
    waveform = source_table(
        current_source_samples(grid, waveform_fn, n_steps), amplitude_kind,
        native="cb_over_dv", volume=dV,
        coefficient=lambda: e_update_coefficient_at(
            materials, (i, j, k), component, grid.dt, periodic=tuple(a in grid.periodic_axes for a in "xyz"), host=True, grid=grid))

    waveform_out = waveform if any_traced or is_tracer(waveform) else np.array(waveform)
    return (i, j, k, component, waveform_out)


def _uniform_cell_volume(grid: Grid) -> float:
    """dV = dx*dy*dz with missing axes duck-typed to dx (engineering
    principle 3). A 2D grid is treated as ONE CELL DEEP (dz -> dx), so
    dV = dx**3 there too — the documented #571 convention
    (``rfx/api/_source_semantics.py``), not per-unit-length dx**2."""
    return grid.dx * getattr(grid, "dy", grid.dx) * getattr(grid, "dz", grid.dx)

def scale_source_columns(xs, scales, columns, volumes=()):
    """Finish distributed current tables from slab-local Cb and each dV."""
    steps, table = xs
    for n, col in enumerate(columns):
        table = table.at[:, col].set(source_table(
            table[:, col], 'current', coefficient=scales[n], volume=volumes[n]))
    return steps, table


def slab_drive_table(xs, mesh, device_ids, meta, *, ranks, scales=None,
                     columns=(), volumes=(), weights=None):
    """Store (step, slab, source) increments before any time scan.

    Material products precede per-column weights. The local owner mask
    supplies unit weights until a caller supplies published edge weights.
    Each product is stored in the series, never formed next to an E add.
    """
    from functools import partial
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from rfx.stepping.rank import rank_shard_map
    from rfx.stepping.slab import source_owner_mask

    steps, table = xs
    if scales is None:
        scales = jnp.zeros((mesh.size, 0), dtype=table.dtype)
    if weights is None:
        weights = jnp.ones((mesh.size, table.shape[1]), dtype=table.dtype)

    @partial(rank_shard_map, mesh=mesh, in_specs=(P(), P('x'), P('x')),
             out_specs=P(None, 'x'), check_rep=False)
    def build(samples, material, weight, *, rank):
        _, samples = scale_source_columns((steps, samples), material[0], columns, volumes)
        weight = jnp.where(source_owner_mask(rank, device_ids), weight[0], 0)
        samples = weight_drive_columns(samples, weight, meta)
        return samples[:, None, :]

    return steps, build(ranks, table, scales, weights)


def subgrid_source_tables(requests, grid, materials, n_steps, *, high_wall, native,
                          fine_scale, volume, shadow_entries=None):
    """Resolve fine-grid drives after every load; project completed soft tables.

    Port descriptors retain their own Norton load and unit-voltage shape;
    their coefficient, like a soft current's, reads the final fine materials.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from rfx.model.materials import e_update_coefficient_at, with_components
    from rfx.sources.port_drive import port_drive_waveform

    materials = with_components(materials, grid, periodic=(False,) * 3, high_wall=high_wall)
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


def dispersive_drive_model(materials, debye_spec, lorentz_spec, grid=None, *,
                           stability_grid=None, in_program=True):
    """Materials a drive reads its coefficient from on a lane that keeps no
    realization of its own (the distributed lanes). Nothing whole-domain is
    built: a dispersive model carries its pole specs, read edge by edge."""
    judged = grid if grid is not None else stability_grid
    if judged is not None and lorentz_spec is not None:
        from rfx.model.pole_stability import courant_budget, refuse_unstable_cells
        refuse_unstable_cells(materials, lorentz_spec, judged.dt, courant_budget(judged),
                              grid_kind=type(judged).__name__, debye_spec=debye_spec,
                              in_program=in_program)
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
    import jax
    from rfx.model.materials import EdgePoles
    c = getattr(drive_model, "components", None)
    with jax.ensure_compile_time_eval():
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
