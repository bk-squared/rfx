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
from rfx.stepping.remat import segmented_scan
from rfx.stepping.sequence import HookPoint
from rfx.stepping.slab import cut, cut_pole_coeffs, cut_poles, place


def select_step(*, cpml=False, debye=False, lorentz=False, ports=False,
                dft=False, flux=False, ntff=False, pmc=False, periodic_x=False,
                point_sources=True, point_probes=True, pec=True):
    """Wrap PEC and plain CPML; keep dispersive CPML on the legacy step.

    The legacy-form CPML call is bit-equal to one device on four graded CPML
    boxes up to 1600 steps on two and three devices (rfx-archive
    records/20261017-item5-first-divergence). Upstream refusals still apply.
    """
    kind = "wrapped" if (pec and point_sources and point_probes and not any((
        dft, flux, ntff, pmc, periodic_x, cpml and (debye or lorentz)))) else "legacy"
    if ports and kind != 'wrapped':
        raise NotImplementedError('Graded multi-device ports require the wrapped step; CPML with poles requires rfx #1402 item 5 dispersive; PMC and periodic x are not supported.')
    return kind


def record_step_kind(grid, debye_spec, lorentz_spec, boundary_spec, *, ports=False):
    """Record the runner's choice before any slab operands are staged."""
    if grid.step_kind is not None:
        return grid
    return grid._replace(step_kind=select_step(
        cpml=bool(grid.cpml_layers), debye=debye_spec is not None,
        lorentz=lorentz_spec is not None, pmc=bool(boundary_spec.pmc_faces()), ports=ports))


def runner_step_kind(grid, **flags):
    """The step a direct runner call takes: the one the grid recorded, else the decision from flags."""
    step_kind = grid.step_kind
    if step_kind is None:
        step_kind = select_step(**flags)
    if flags.get('ports') and step_kind != 'wrapped':
        raise NotImplementedError('Graded multi-device ports require the wrapped step (rfx #1402 item 5 dispersive for CPML with poles).')
    return step_kind


def _place(value, mesh):
    return place(value, NamedSharding(mesh, P('x')))


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


def cut_drives(drives, layout, mesh, *, waveforms, per_rank=False):
    """Select canonical drive rows and their stored samples; never rescale them.

    ``waveforms`` is the replicated ``(n_steps, n_src)`` table, or with
    ``per_rank=True`` the slab-built ``(n_steps, n_devices * n_src)`` table on
    ``P(None, 'x')`` whose rank block is zero for sources that rank does not own.
    """
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
    if per_rank:
        # A rank block holds only the sources that rank owns, so every wave
        # must be read by rows on one rank only; and one row per source.
        n_src = sum(len(a) for a in drives.wave_id.values())
        if waveforms.shape[1] != layout.n_devices * n_src:
            raise ValueError(f'per-rank drive table has {waveforms.shape[1]} columns; '
                             f'expected n_devices * n_src = {layout.n_devices} * {n_src}')
        readers = {}
        for rank, rows in enumerate(wave_rows):
            for wave in rows.tolist():
                readers.setdefault(wave, []).append(rank)
        shared = {wave: ranks for wave, ranks in readers.items() if len(ranks) > 1}
        if shared:
            raise ValueError(f'per-rank drive table: waves read on several ranks {shared}; '
                             'each rank block holds only the sources it owns')
    if per_rank and layout.n_devices > 1 and n_src > 0:
        @partial(jax.shard_map, mesh=mesh,
                 in_specs=(P(None, 'x'), P('x'), P('x')),
                 out_specs=P(None, 'x'), check_vma=False)
        def select(local, ids, valid):
            return jnp.where(valid[None], local[:, ids], jnp.zeros((), local.dtype))
        samples = select(waveforms, _place(selected_waves, mesh), _place(wave_valid, mesh))
    else:
        samples = jnp.where(wave_valid[None, :, :], waveforms[:, selected_waves],
                            jnp.zeros((), waveforms.dtype))
    samples = place(samples, NamedSharding(mesh, P(None, 'x')))
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


def split(invariants, carry, layout, mesh, *, waveforms, dt=None):
    """Cut every spatial operand with the shared FILL policy, and route nodes."""
    tags = dict(inv_dx='inv_dx', inv_dx_h='inv_dx_h', dx_arr='width',
                pec_edge_masks='pec_edge', pec_static_edge_masks='pec_edge',
                pec_occupancy='pec_occupancy')
    result = {}
    for key, value in invariants.items():
        if key in tags:
            result[key] = cut(value, layout, tags[key], mesh=mesh)
        elif key == 'e_loss' and value is not None:
            eps, sigma, inverse = value
            result[key] = (cut(eps, layout, 'eps_r', mesh=mesh),
                           cut(sigma, layout, 'sigma', mesh=mesh))
            assert inverse is False, 'graded CPML loss uses relative epsilon'
        elif key == 'cpml_inv_eps_r':
            result[key] = cut(value, layout, 'eps_r', mesh=mesh)
        elif key == 'materials':
            fields = {}
            for name, field in zip(value._fields, value):
                if name == 'components':
                    # Kernel operands (KernelComponents) or, under a realized-model
                    # capture, the whole realization: cut every grid-shaped leaf,
                    # padded by the field family its top-level name belongs to.
                    def cut_component(path, a):
                        top = str(getattr(path[0], 'name', getattr(path[0], 'key', path[0])))
                        if np.ndim(a) != 3 or np.shape(a)[0] != layout.nx:
                            return a
                        kind = ('eps_r' if 'eps' in top else 'sigma' if 'sigma' in top
                                else 'mu_r' if top.startswith('mu') else 'lumped')
                        return cut(a, layout, kind, mesh=mesh)
                    fields[name] = None if field is None else jax.tree_util.tree_map_with_path(
                        cut_component, field)
                elif name == 'edge_keep':
                    fields[name] = cut(field, layout, 'edge_keep', mesh=mesh)
                else:
                    # Lumped records (sigma_film, eps_r_lumped, ...) are per-component
                    # tuples; a realized-model capture keeps them in the materials.
                    kind = name if name in ('eps_r', 'sigma', 'mu_r') else 'lumped'
                    fields[name] = jax.tree.map(
                        lambda a, kind=kind: a if np.ndim(a) == 0 else cut(a, layout, kind, mesh=mesh),
                        field)
            result[key] = type(value)(**fields)
        elif key in ('debye_coeffs', 'lorentz_coeffs'):
            result[key] = cut_pole_coeffs(value, layout, key.removesuffix('_coeffs'),
                                          dt=dt, mesh=mesh)
        elif key == 'drives':
            result[key] = cut_drives(value, layout, mesh, waveforms=waveforms)
        elif key in ('probe_table', 'port_table'):
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
    local_carry = {'fdtd': FDTDState(**local, step=state.step)}
    if 'cpml' in carry:
        local_carry['cpml'] = cut(carry['cpml'], layout, 'cpml_psi', mesh=mesh)
    for family in ('debye', 'lorentz'):
        if family in carry:
            local_carry[family] = cut_poles(carry[family], layout, family + '_state', mesh=mesh)
    for key in ('wire_sparams', 'sparam_time_records'):
        if key in carry:
            local_carry[key] = jax.tree.map(lambda a: _place(
                jnp.broadcast_to(a, (layout.n_devices,) + a.shape), mesh), carry[key])
    return result, local_carry


def wrap(step, layout, mesh, *, table, port_table=None):
    """Execute the supplied graded step on each slab, attaching two copies."""
    from rfx.runners.distributed_nu import _exchange_h_ghosts_nu, _exchange_e_ghosts_nu
    components = tuple(term.component for term in table.components)
    n_channels, channel_names = table.n_channels, table.channel_names
    context = replace(step.context, x_face_frame={'x': (
        layout.ghost_width, layout.ghost_width + layout.nx_per_rank - layout.pad_x)})

    def local(carry, xs, invariants, *, rank):
        def h(frame):
            frame.st = _exchange_h_ghosts_nu(frame.st, mesh, layout.n_devices, ranks=rank, local=True)

        def e(frame):
            frame.st = _exchange_e_ghosts_nu(frame.st, mesh, layout.n_devices, ranks=rank, local=True)

        xs = (xs[0], xs[1][0])
        inputs = dict(invariants)
        if inputs.get('e_loss') is not None:
            inputs['e_loss'] = (*inputs['e_loss'], False)
        inputs['drives'] = jax.tree.map(lambda a: a[0], inputs['drives'])
        terms, order, ids = jax.tree.map(lambda a: a[0], inputs['probe_table'])
        inputs['probe_table'] = SampleTable(tuple(
            ComponentTerms(component, *part) for component, part in zip(components, terms)),
            n_channels, channel_names, order, ids)
        if port_table is not None:
            parts, port_order, port_ids = jax.tree.map(lambda a: a[0], inputs['port_table'])
            inputs['port_table'] = SampleTable(tuple(
                ComponentTerms(term.component, *part) for term, part in zip(port_table.components, parts)),
                port_table.n_channels, port_table.channel_names, port_order, port_ids)
            carry = dict(carry)
            for key in ('wire_sparams', 'sparam_time_records'):
                if key in carry:
                    carry[key] = jax.tree.map(lambda a: a[0], carry[key])
        inputs['owns_x'] = jax.tree.map(lambda a: a[0], inputs['owns_x'])
        advance = make_graded_step(context, hooks={HookPoint.AFTER_H: (h,), HookPoint.STEP_END: (e,)})
        from rfx import _realized
        if _realized.ACTIVE is None:
            carry, record = advance(carry, xs, inputs)
        else:
            start = rank * layout.nx_per_rank
            with _realized.owned(start, jnp.minimum(layout.nx_per_rank, layout.nx - start)):
                carry, record = advance(carry, xs, inputs)
        # Owner-masked samples summed here, as the legacy step does: a record
        # leaving per device would make its replicated cotangent be sliced by
        # device identity in the backward program (#1441).
        if port_table is not None:
            for key in ('wire_sparams', 'sparam_time_records'):
                if key in carry:
                    carry[key] = jax.tree.map(lambda a: a[None], carry[key])
        return carry, jax.lax.psum(record, 'x')

    def mapped(carry, xs, invariants, *, ranks):
        specs = jax.tree.map(lambda a: P() if np.ndim(a) == 0 else P('x'), invariants)
        carry_spec = jax.tree.map(lambda a: P() if np.ndim(a) == 0 else P('x'), carry)
        return rank_shard_map(local, mesh=mesh,
            in_specs=(carry_spec, (P(), P('x')), specs),
            out_specs=(carry_spec, P()), check_rep=False)(ranks, carry, xs, invariants)
    return mapped


def reduce(records, state, layout, *, gather=True):
    """Records arrive summed over devices in wrap; retain only owned field rows."""
    from rfx.runners._distributed_common import unstack_and_gather
    fields = None
    if gather:
        fields = state._replace(**{name: unstack_and_gather(getattr(state, name),
            layout.n_devices, layout.nx_local, layout.ghost_width, layout.pad_x, layout.nx)
            for name in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')})
    return records, fields


def run_steps(carry, invariants, waves, ranks, advance, *, n_steps, n_warmup, checkpoint_every,
              compile_loop=jax.jit, scan_loop=jax.lax.scan):
    """Drive canonical slab steps, including warmup and checkpoint segments."""
    @compile_loop
    def run_fn(carry, invariants, waves, ranks):
        step = partial(advance, invariants=invariants, ranks=ranks)
        xs = (jnp.arange(n_steps, dtype=jnp.int32), waves)
        warm = None
        if n_warmup:
            carry, warm = scan_loop(step, carry, jax.tree.map(lambda a: a[:n_warmup], xs))
            carry, warm = jax.tree.map(jax.lax.stop_gradient, (carry, warm))
        xs = jax.tree.map(lambda a: a[n_warmup:], xs)
        # One remat implementation (rfx.stepping.remat): full checkpointed
        # segments and a real tail, never padded steps.
        carry, records = segmented_scan(step, carry, xs, checkpoint_every, scan=scan_loop)
        if warm is not None:
            records = jnp.concatenate((warm, records))
        return carry, records

    return run_fn(carry, invariants, waves, ranks)


def validate_scan_controls(n_steps, n_warmup, checkpoint_every):
    """Validate warmup and checkpoint controls before selecting a runner."""
    if n_warmup < 0:
        raise ValueError(f"n_warmup must be >= 0; got {n_warmup}")
    if n_warmup >= n_steps:
        raise ValueError(
            f"n_warmup ({n_warmup}) must be < n_steps ({n_steps})"
        )
    if checkpoint_every is not None and int(checkpoint_every) <= 0:
        raise ValueError(
            f"checkpoint_every must be a positive int or None; "
            f"got {checkpoint_every}"
        )
