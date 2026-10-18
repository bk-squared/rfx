"""Global port metadata and small per-slab partial measurements."""
from dataclasses import dataclass, replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from rfx.stepping.slab import place


@dataclass(frozen=True)
class PortPlan:
    specs: tuple
    acc_indices: tuple
    wp_meta: list
    sp_freqs: object
    use_wire_ports: bool
    initial: dict


def global_plan(sim, grid, specs, lane, freqs, compute_s_params):
    """Build all metric-dependent metadata once, on the global host grid."""
    from rfx.measurement.port_registration import graded_specs, graded_acc_indices, graded_accumulators
    if not specs:
        return None
    if lane == 'run_distributed_nu':
        if compute_s_params is False:
            freqs = None
        elif freqs is None:
            from rfx.measurement.monitors import default_frequency_grid
            freqs = default_frequency_grid(sim._freq_max, 50, host=True)
    initial = {}
    with jax.ensure_compile_time_eval():
        bins, meta, use = graded_accumulators(specs, grid, freqs, initial)
    return PortPlan(graded_specs(specs), graded_acc_indices(specs), meta, bins, use, initial)


def setup_ports(context, carry, invariants, plan, layout, mesh, n_steps):
    """Broadcast only zero accumulators/records; never use slab indices as global."""
    from rfx.measurement.projector import port_table
    from rfx.measurement.time_records import init_graded_records
    from rfx.stepping.graded import make_graded_step
    if plan is None:
        return context
    context = replace(context, port_record_specs=plan.specs,
        port_acc_indices=plan.acc_indices, wp_meta=plan.wp_meta,
        sp_freqs=plan.sp_freqs, use_wire_ports=plan.use_wire_ports)
    with jax.ensure_compile_time_eval():
        invariants['port_table'] = port_table(context.grid, plan.specs, nu=True,
            boundary=context.curl_boundary, dtype=carry['fdtd'].ex.dtype)
    initial = dict(plan.initial)
    initial['fdtd'] = carry['fdtd']
    init_graded_records(initial, n_steps, SimpleNamespace(step_fn_inv=make_graded_step(context)))
    for key in ('wire_sparams', 'sparam_time_records'):
        if key in initial:
            carry[key] = jax.tree.map(lambda a: place(
                jnp.broadcast_to(a, (layout.n_devices,) + a.shape), NamedSharding(mesh, P('x'))), initial[key])
    return context


def sum_partials(final, mesh, ranks):
    """One post-scan psum inside shard_map; its transpose broadcasts cotangents."""
    from rfx.stepping.rank import rank_shard_map
    partials = {key: final[key] for key in ('wire_sparams', 'sparam_time_records') if key in final}
    if not partials:
        return final
    def local(values, *, rank):
        totals = jax.tree.map(lambda a: jax.lax.psum(a[0], 'x'), values)
        if 'sparam_time_records' in values:
            records = []
            for partial, total in zip(values['sparam_time_records'], totals['sparam_time_records']):
                ref = partial[0, :, -1]
                # Only the midpoint owner contributes a reference; other
                # ranks contribute +0. Preserve its signed zero in the
                # stop-gradient time record, which ordinary psum erases.
                negative_zero = jax.lax.psum(((ref == 0) & jnp.signbit(ref)).astype(jnp.int32), 'x') > 0
                ref_total = total[:, -1]
                ref_total = jnp.where((ref_total == 0) & negative_zero,
                                      jnp.asarray(-0., dtype=ref.dtype), ref_total)
                records.append(jnp.concatenate((total[:, :-1], ref_total[:, None]), axis=-1))
            totals['sparam_time_records'] = tuple(records)
        return totals
    totals = rank_shard_map(local, mesh=mesh, in_specs=P('x'), out_specs=P(), check_rep=False)(ranks, partials)
    return dict(final, **totals)
