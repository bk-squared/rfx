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
                emit_time_series, gather_final_state, material_drive):
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

    raw_materials = materials
    materials = jax.tree.map(gather_input, materials)
    pec_mask = jax.tree.map(gather_input, pec_mask)
    occupancy = jax.tree.map(gather_input, occupancy)
    grid = sg.original_grid or NonUniformGrid(
        sg.nx, sg.ny, sg.nz, float(sg.dx_padded[0]), float(sg.dy_cells[0]),
        np.asarray(sg.dx_padded[:sg.nx]), np.asarray(sg.dy_cells), np.asarray(sg.dz_cells),
        sg.dt, 0, np.asarray(sg.inv_dx_global[:sg.nx]), sg.inv_dy, sg.inv_dz,
        np.asarray(sg.inv_dx_h_global[:sg.nx]), sg.inv_dy_h, sg.inv_dz_h)
    source_rows = [(s.i, s.j, s.k, s.component, s.waveform) for s in sources]
    material_drive = (None,) * len(sources) if material_drive is None else material_drive
    if len(material_drive) != len(sources):
        raise ValueError('material_drive must contain one entry per source')
    owners = [incoming.owner(s.i) for s in sources]
    local_specs = [(incoming.local_index(s.i), s.j, s.k, s.component) for s in sources]
    nodes = tuple((owner, *spec) for owner, spec in zip(owners, local_specs))
    drive_columns, drives = [], []
    for col, (s, volume) in enumerate(zip(sources, material_drive)):
        if volume is not None:
            row0 = incoming.ghost_width if s.i == 0 else 0
            drives.append((owners[col], row0,
                (local_specs[col][0] - row0, s.j, s.k), s.component,
                *(volume if isinstance(volume, tuple) else (float(volume), 0.))))
            drive_columns.append(col)
    edge_masks = pec_mask if isinstance(pec_mask, tuple) else None
    setup = _build_nu_scan(grid, materials, n_steps, cpml_axes='',
        pec_mask=None if edge_masks is not None else pec_mask, pec_edge_masks=edge_masks,
        pec_occupancy=occupancy, sources=source_rows,
        probes=[(p.i, p.j, p.k, p.component) for p in probes],
        pec_faces={f'{a}_{s}' for a in 'xyz' for s in ('lo', 'hi')},
        emit_time_series=emit_time_series)
    layout = incoming
    invariants, carry = split(setup.invariants, setup.carry_init, layout, mesh)
    advance = wrap(setup.step_fn_inv, layout, mesh, table=setup.invariants['probe_table'])

    from rfx.runners import distributed_nu as nu
    from rfx.model.source_coefficients import slab_drive_table
    from rfx.model.electric_metrics import material_drive_scales, SlabElectricMetrics

    @jax.jit
    def run_fn(carry, invariants, waves, ranks, raw_materials):
        step = partial(advance, invariants=invariants, ranks=ranks)
        scales = None
        if drives:
            scales = material_drive_scales(raw_materials.eps_r, raw_materials.sigma,
                mesh, tuple(drives), sg.dt, ranks=ranks,
                grid=SlabElectricMetrics(sg.e_cell_sizes, sg.nx, sg.nx_per_rank, sg.nx_local),
                sigma_film=raw_materials.sigma_film)
        keep = invariants['materials'].edge_keep
        weights = nu.owned_values(keep, mesh, nodes, ranks=ranks) if keep is not None and sources else None
        xs = slab_drive_table((jnp.arange(n_steps, dtype=jnp.int32), waves), mesh,
            owners, local_specs, ranks=ranks, scales=scales, columns=drive_columns, weights=weights)
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

    waves = jnp.stack([s.waveform for s in sources], axis=-1) if sources else jnp.zeros((n_steps, 0), jnp.float32)
    final, records = run_fn(carry, invariants, waves, mesh_ranks(mesh), raw_materials)
    ts, state = reduce(records, final['fdtd'], layout, gather=gather_final_state)
    return dict(time_series=ts if emit_time_series else None, final_state=state,
                final_state_sharded=final['fdtd'], cpml_state_sharded=None,
                debye_state_sharded=None, lorentz_state_sharded=None)
