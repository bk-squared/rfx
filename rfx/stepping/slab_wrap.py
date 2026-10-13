"""Placement and exchanges around the canonical graded step; no field arithmetic."""
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from rfx.core.yee import FDTDState
from rfx.measurement.projector import ComponentTerms, SampleTable
from rfx.stepping.graded import make_graded_step
from rfx.stepping.rank import rank_shard_map
from rfx.stepping.sequence import HookPoint
from rfx.stepping.slab import cut


def select_step(*, cpml=False, debye=False, lorentz=False, ports=False,
                dft=False, flux=False, ntff=False, pmc=False, periodic_x=False,
                point_sources=True, point_probes=True, pec=True):
    """The W1 first-slice admission decision (upstream refusals still apply)."""
    return "wrapped" if (pec and point_sources and point_probes and not any((
        cpml, debye, lorentz, ports, dft, flux, ntff, pmc, periodic_x))) else "legacy"


def _place(value, mesh):
    return jax.device_put(value, NamedSharding(mesh, P('x')))


def _nodes(indices, layout):
    owners = np.asarray(indices[0]) // layout.nx_per_rank
    rows = []
    for rank in range(layout.n_devices):
        keep = np.flatnonzero(owners == rank)
        rows.append(keep)
    width = max(map(len, rows), default=0)
    nodes, selections, valid = [], [], []
    for rank, keep in enumerate(rows):
        n = len(keep)
        local = np.zeros((3, width), np.int32)
        local[0] = layout.ghost_width
        for axis in range(3):
            local[axis, :n] = np.asarray(indices[axis])[keep]
        local[0, :n] += layout.ghost_width - rank * layout.nx_per_rank
        nodes.append(local)
        selections.append(np.pad(keep, (0, width-n)))
        valid.append(np.arange(width) < n)
    return np.asarray(nodes), np.asarray(selections), np.asarray(valid)


def cut_drives(drives, layout, mesh, *, waveforms):
    """Select canonical drive rows and their stored samples; never rescale them."""
    routed = {component: _nodes(indices, layout)
              for component, indices in drives.nodes.items()}
    # Preserve the canonical waveform order on each owner. Several rows may
    # reference the same waveform; the local table retains one such column.
    wave_rows = []
    for rank in range(layout.n_devices):
        ids = [np.asarray(drives.wave_id[c])[selected[rank, valid[rank]]]
               for c, (_, selected, valid) in routed.items()]
        wave_rows.append(np.unique(np.concatenate(ids)).astype(np.int32)
                         if ids else np.empty(0, np.int32))
    width = max(map(len, wave_rows), default=0)
    selected_waves = np.zeros((layout.n_devices, width), np.int32)
    wave_valid = np.zeros_like(selected_waves, dtype=bool)
    nodes, coef, wave_id, owner = {}, {}, {}, {}
    for rank, rows in enumerate(wave_rows):
        selected_waves[rank, :len(rows)] = rows
        wave_valid[rank, :len(rows)] = True
    for component, (local, ids, valid) in routed.items():
        nodes[component] = tuple(_place(local[:, axis], mesh) for axis in range(3))
        coef[component] = _place(jnp.where(valid, drives.coef[component][ids], 0), mesh)
        local_waves = np.zeros_like(ids)
        for rank, rows in enumerate(wave_rows):
            local_waves[rank, valid[rank]] = np.searchsorted(
                rows, np.asarray(drives.wave_id[component])[ids[rank, valid[rank]]])
        wave_id[component] = _place(local_waves, mesh)
        owner[component] = _place(drives.owner[component][ids], mesh)
    samples = jnp.where(wave_valid[None, :, :], waveforms[:, selected_waves],
                        jnp.zeros((), waveforms.dtype))
    samples = jax.device_put(samples, NamedSharding(mesh, P(None, 'x')))
    return drives._replace(nodes=nodes, coef=coef, wave_id=wave_id, owner=owner, waves=samples)


def cut_probe_table(table, layout, mesh):
    """Array payload; component names and channel count stay static in wrap."""
    terms, all_ids = [], []
    for term in table.components:
        local, ids, valid = _nodes(term.indices, layout)
        channels = np.where(valid, term.channel_ids[ids], table.n_channels)
        weights = jnp.where(valid, term.weights[ids], 0)
        terms.append((tuple(local[:, axis] for axis in range(3)), channels, weights))
        all_ids.append(channels)
    ids = np.concatenate(all_ids, axis=1) if all_ids else np.empty((layout.n_devices, 0), np.int32)
    order = np.argsort(ids, axis=1, kind='stable').astype(np.int32)
    sorted_ids = np.take_along_axis(ids, order, axis=1)
    assert np.all(np.diff(sorted_ids, axis=1) >= 0), 'slab segment ids must be sorted'
    return jax.tree.map(lambda a: _place(a, mesh), (tuple(terms), order, sorted_ids))


def split(invariants, carry, layout, mesh, *, waveforms):
    """Cut every spatial operand with the shared FILL policy, and route nodes."""
    tags = dict(inv_dx='inv_dx', inv_dx_h='inv_dx_h', dx_arr='width',
                pec_edge_masks='pec_edge', pec_static_edge_masks='pec_edge',
                pec_occupancy='pec_occupancy')
    result = {}
    for key, value in invariants.items():
        if key in tags:
            result[key] = cut(value, layout, tags[key], mesh=mesh)
        elif key == 'materials':
            fields = {}
            for name, field in zip(value._fields, value):
                if name == 'components':
                    fields[name] = field._replace(**{
                        part: cut(getattr(field, part), layout, kind, mesh=mesh)
                        for part, kind in (('eps_update', 'eps_r'), ('sigma_update', 'sigma'),
                                           ('mu_update', 'mu_r'))})
                elif name == 'edge_keep':
                    fields[name] = cut(field, layout, 'edge_keep', mesh=mesh)
                elif field is None or np.ndim(field) == 0:
                    fields[name] = field
                else:
                    fields[name] = cut(field, layout, name if name in ('eps_r', 'sigma', 'mu_r') else 'lumped', mesh=mesh)
            result[key] = type(value)(**fields)
        elif key == 'drives':
            result[key] = cut_drives(value, layout, mesh, waveforms=waveforms)
        elif key == 'probe_table':
            result[key] = cut_probe_table(value, layout, mesh)
        elif value is None:
            result[key] = None
        else:
            raise ValueError(f'unsupported wrapped invariant: {key}')
    result['owns_x'] = {face: _place(np.asarray([pair[side] for pair in layout.faces]), mesh)
                        for side, face in enumerate(('x_lo', 'x_hi'))}
    state = carry['fdtd']
    local = {name: cut(getattr(state, name), layout, 'state', mesh=mesh)
             for name in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')}
    return result, {'fdtd': FDTDState(**local, step=state.step)}


def wrap(step, layout, mesh, *, table):
    """Execute the supplied graded step on each slab, attaching two copies."""
    from rfx.runners.distributed_nu import _exchange_h_ghosts_nu, _exchange_e_ghosts_nu
    context = replace(step.context, x_face_frame={'x': (
        layout.ghost_width, layout.ghost_width + layout.nx_per_rank - layout.pad_x)})

    def local(carry, xs, invariants, *, rank):
        def h(frame):
            frame.st = _exchange_h_ghosts_nu(frame.st, mesh, layout.n_devices, ranks=rank, local=True)

        def e(frame):
            frame.st = _exchange_e_ghosts_nu(frame.st, mesh, layout.n_devices, ranks=rank, local=True)

        xs = (xs[0], xs[1][0])
        inputs = dict(invariants)
        inputs['drives'] = jax.tree.map(lambda a: a[0], inputs['drives'])
        terms, order, ids = jax.tree.map(lambda a: a[0], inputs['probe_table'])
        inputs['probe_table'] = SampleTable(tuple(
            ComponentTerms(host.component, *part) for host, part in zip(table.components, terms)),
            table.n_channels, table.channel_names, order, ids)
        inputs['owns_x'] = jax.tree.map(lambda a: a[0], inputs['owns_x'])
        advance = make_graded_step(context, hooks={HookPoint.AFTER_H: (h,), HookPoint.STEP_END: (e,)})
        carry, record = advance(carry, xs, inputs)
        return carry, record[None, :]

    def mapped(carry, xs, invariants, *, ranks):
        specs = jax.tree.map(lambda a: P() if np.ndim(a) == 0 else P('x'), invariants)
        state_spec = FDTDState(*(P('x'),)*6, step=P())
        return rank_shard_map(local, mesh=mesh,
            in_specs=({'fdtd': state_spec}, (P(), P('x')), specs),
            out_specs=({'fdtd': state_spec}, P('x')), check_rep=False)(ranks, carry, xs, invariants)
    return mapped


def reduce(records, state, layout, *, gather=True):
    """Reduce owner samples once and retain only owned field rows."""
    from rfx.runners._distributed_common import unstack_and_gather
    fields = None
    if gather:
        fields = state._replace(**{name: unstack_and_gather(getattr(state, name),
            layout.n_devices, layout.nx_local, layout.ghost_width, layout.pad_x, layout.nx)
            for name in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')})
    return jnp.sum(records, axis=1), fields
