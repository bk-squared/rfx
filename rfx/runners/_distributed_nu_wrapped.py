"""Adapter from the established graded distributed entry to the slab wrapper."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from rfx.nonuniform import NonUniformGrid, _build_nu_scan
from rfx.stepping.rank import mesh_ranks
from rfx.stepping.slab import Slab
from rfx.stepping.slab_wrap import split, wrap, reduce
from rfx.runners._distributed_common import unstack_and_gather


def run_wrapped(sg, materials, pec_mask, n_steps, sources, probes, *,
                devices, occupancy, checkpoint_every, n_warmup,
                emit_time_series, gather_final_state, material_drive,
                debye=None, lorentz=None):
    """Reuse canonical setup, then scan the same step on each rank."""
    devices = jax.devices()[:sg.n_devices] if devices is None else devices
    if len(devices) != sg.n_devices:
        raise ValueError(f'need {sg.n_devices} devices, got {len(devices)}')
    mesh = Mesh(np.asarray(devices), ('x',))
    incoming = Slab.from_grid(sg)

    def gather_input(a):
        if np.ndim(a) == 0:
            return a
        return unstack_and_gather(a, incoming.n_devices, incoming.nx_local,
                                 incoming.ghost_width, incoming.pad_x, incoming.nx)

    materials = jax.tree.map(gather_input, materials)
    pec_mask = jax.tree.map(gather_input, pec_mask)
    occupancy = jax.tree.map(gather_input, occupancy)
    grid = sg.original_grid or NonUniformGrid(
        sg.nx, sg.ny, sg.nz, float(sg.dx_padded[0]), float(sg.dy_cells[0]),
        np.asarray(sg.dx_padded[:sg.nx]), np.asarray(sg.dy_cells), np.asarray(sg.dz_cells),
        sg.dt, 0, np.asarray(sg.inv_dx_global[:sg.nx]), sg.inv_dy, sg.inv_dz,
        np.asarray(sg.inv_dx_h_global[:sg.nx]), sg.inv_dy_h, sg.inv_dz_h)
    def gather_poles(a):
        poles = a.shape[0] // incoming.n_devices
        rows = a.reshape((incoming.n_devices, poles, incoming.nx_local) + a.shape[2:])
        rows = rows[:, :, incoming.ghost_width:incoming.ghost_width + incoming.nx_per_rank]
        return rows.transpose(1, 0, 2, 3, 4).reshape((poles, incoming.nx_padded) + a.shape[2:])[:, :incoming.nx]

    def gather_dispersion(bundle, family):
        if bundle is None:
            return None
        coeffs, state = bundle
        spatial = ('ca', 'cb') if family == 'debye' else ('ca', 'cb', 'cc')
        coeffs = type(coeffs)(*(jax.tree.map(gather_input if name in spatial else gather_poles, value)
                               for name, value in zip(coeffs._fields, coeffs)))
        if coeffs.ca is None:
            # Compatibility with callers supplying fixed, staged pole terms.
            from rfx.model.materials import with_components
            from rfx.materials.debye import debye_e_coeffs
            from rfx.materials.lorentz import lorentz_e_coeffs
            components = with_components(materials, grid, periodic=(False,) * 3).components
            means = components.eps_update, components.sigma_update
            coeffs = (debye_e_coeffs(means, grid.dt, coeffs.alpha, coeffs.beta)
                      if family == 'debye' else
                      lorentz_e_coeffs(means, grid.dt, coeffs.a, coeffs.b, coeffs.c))
        return coeffs, jax.tree.map(gather_poles, state)

    debye = gather_dispersion(debye, 'debye')
    lorentz = gather_dispersion(lorentz, 'lorentz')
    material_drive = (None,) * len(sources) if material_drive is None else material_drive
    if len(material_drive) != len(sources):
        raise ValueError('material_drive must contain one entry per source')
    source_rows = []
    for source, volume in zip(sources, material_drive):
        samples = source.waveform
        if volume is not None:
            # The public distributed entry supplies raw current samples for
            # overrides. Resolve them with the single-device graded producer
            # before _build_nu_scan publishes its occupancy-weighted table.
            from rfx.model.materials import e_update_coefficient_at
            from rfx.model.source_coefficients import source_table
            volume = volume[0] if isinstance(volume, tuple) else volume
            def coefficient():
                cell = (source.i, source.j, source.k)
                axis = ('ex', 'ey', 'ez').index(source.component)
                if debye is not None and lorentz is not None:
                    from rfx.materials.lorentz import mixed_e_component_coeffs
                    return mixed_e_component_coeffs(debye[0], lorentz[0], axis, grid.dt)[1][cell]
                if debye is not None or lorentz is not None:
                    return (debye or lorentz)[0].cb[axis][cell]
                return e_update_coefficient_at(materials, cell, source.component,
                                               grid.dt, host=True, grid=grid)
            samples = source_table(samples, native='cb_over_dv', volume=volume,
                coefficient=coefficient)
        source_rows.append((source.i, source.j, source.k, source.component, samples))
    edge_masks = pec_mask if isinstance(pec_mask, tuple) else None
    setup = _build_nu_scan(grid, materials, n_steps, cpml_axes='',
        debye=debye, lorentz=lorentz,
        pec_mask=None if edge_masks is not None else pec_mask, pec_edge_masks=edge_masks,
        pec_occupancy=occupancy, sources=source_rows,
        probes=[(p.i, p.j, p.k, p.component) for p in probes],
        pec_faces={f'{a}_{s}' for a in 'xyz' for s in ('lo', 'hi')},
        emit_time_series=emit_time_series)
    layout = incoming
    invariants, carry = split(setup.invariants, setup.carry_init, layout, mesh,
                               waveforms=setup.src_waveforms, dt=grid.dt)
    waves = invariants['drives'].waves
    invariants['drives'] = invariants['drives']._replace(waves=None)
    advance = wrap(setup.step_fn_inv, layout, mesh, table=setup.invariants['probe_table'])

    from rfx.runners import distributed_nu as nu

    @jax.jit
    def run_fn(carry, invariants, waves, ranks):
        step = partial(advance, invariants=invariants, ranks=ranks)
        xs = (jnp.arange(n_steps, dtype=jnp.int32), waves)
        warm = None
        if n_warmup:
            carry, warm = nu.lax.scan(step, carry, jax.tree.map(lambda a: a[:n_warmup], xs))
            carry, warm = jax.tree.map(jax.lax.stop_gradient, (carry, warm))
        xs = jax.tree.map(lambda a: a[n_warmup:], xs)
        count = n_steps - n_warmup
        if checkpoint_every is not None and checkpoint_every < count:
            chunk = int(checkpoint_every)
            segments = (count + chunk - 1) // chunk
            padding = segments * chunk - count
            padded = (jnp.arange(n_warmup, n_warmup + segments * chunk, dtype=jnp.int32),
                      jnp.pad(xs[1], ((0, padding), (0, 0), (0, 0))))
            segmented = jax.tree.map(lambda a: a.reshape((segments, chunk) + a.shape[1:]), padded)
            def segment(c, x):
                return nu.lax.scan(step, c, x)
            carry, records = nu.lax.scan(jax.checkpoint(segment), carry, segmented)
            records = records.reshape((-1,) + records.shape[2:])[:count]
        else:
            carry, records = nu.lax.scan(step, carry, xs)
        if warm is not None:
            records = jnp.concatenate((warm, records))
        return carry, records

    final, records = run_fn(carry, invariants, waves, mesh_ranks(mesh))
    ts, state = reduce(records, final['fdtd'], layout, gather=gather_final_state)
    return dict(time_series=ts if emit_time_series else None, final_state=state,
                final_state_sharded=final['fdtd'], cpml_state_sharded=None,
                debye_state_sharded=final.get('debye'), lorentz_state_sharded=final.get('lorentz'))
