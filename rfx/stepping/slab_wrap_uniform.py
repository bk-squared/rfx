"""Placement and exchanges around the canonical uniform step."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from rfx.core.drives import StepDrives
from rfx.stepping.rank import rank_shard_map
from rfx.stepping.sequence import HookPoint
from rfx.stepping.slab_exchange import exchange_e_yee_shmap, exchange_h_yee_shmap
from rfx.stepping.uniform import make_uniform_step


_PORT_CARRY = ('wire_sparam_accs', 'lumped_sparam_accs', 'sparam_time_records')


def wrap_uniform(context, layout, mesh, *, table, port_table=None):
    """Attach exchanges to each local uniform step, retaining owner samples.

    E exchange follows the PEC mask and sources: a neighbour's next H update
    reads the completed seam plane. H exchange precedes the local E update.
    """
    from rfx.stepping.slab_measurements import local_table
    context = replace(context, x_face_frame={'x': (
        layout.ghost_width, layout.ghost_width + layout.nx_per_rank - layout.pad_x)})

    def local(carry, xs, invariants, *, rank):
        def h(frame):
            frame.st = exchange_h_yee_shmap(
                frame.st, mesh, layout.n_devices, ranks=rank, local=True)

        def e(frame):
            frame.st = exchange_e_yee_shmap(
                frame.st, mesh, layout.n_devices, ranks=rank, local=True)

        inputs = dict(invariants)
        inputs['owns_x'] = jax.tree.map(lambda a: a[0], inputs['owns_x'])
        inputs['probe_table'] = local_table(table, inputs['probe_table'])
        if port_table is not None:
            inputs['port_table'] = local_table(port_table, inputs['port_table'])
        if inputs.get('e_loss') is not None:
            inputs['e_loss'] = (*inputs['e_loss'], False)
        drives = jax.tree.map(lambda a: a[0], inputs['drives'])
        ctx = replace(context, materials=inputs['materials'],
            pec_edge_masks=inputs['pec_edge_masks'],
            debye_coeffs=inputs.get('debye_coeffs'),
            lorentz_coeffs=inputs.get('lorentz_coeffs'),
            drives=StepDrives(drives, None))
        carry = dict(carry)
        for key in _PORT_CARRY:
            if key in carry:
                carry[key] = jax.tree.map(lambda a: a[0], carry[key])
        advance = make_uniform_step(ctx, inputs,
            hooks={HookPoint.AFTER_H: (h,), HookPoint.STEP_END: (e,)})
        from rfx import _realized
        if _realized.ACTIVE is None:
            carry, samples, _ = advance(carry, xs[0], xs[1][0], None)
        else:
            start = rank * layout.nx_per_rank
            with _realized.owned(start, jnp.minimum(layout.nx_per_rank, layout.nx - start)):
                carry, samples, _ = advance(carry, xs[0], xs[1][0], None)
        for key in _PORT_CARRY:
            if key in carry:
                carry[key] = jax.tree.map(lambda a: a[None], carry[key])
        return carry, samples[None]

    def mapped(carry, xs, invariants, *, ranks):
        specs = jax.tree.map(lambda a: P() if np.ndim(a) == 0 else P('x'), invariants)
        specs['magnetic_factors'] = tuple(
            None if a is None else P('x') if axis == 0 else P()
            for axis, a in enumerate(invariants['magnetic_factors']))
        carry_spec = jax.tree.map(lambda a: P() if np.ndim(a) == 0 else P('x'), carry)
        return rank_shard_map(local, mesh=mesh,
            in_specs=(carry_spec, (P(), P('x')), specs),
            out_specs=(carry_spec, P('x')), check_rep=False)(ranks, carry, xs, invariants)

    return mapped


def prepare_uniform(grid, materials, pec_edges, sources, probes, layout, mesh,
                    *, n_steps, boundary, debye_spec, lorentz_spec, port_plan,
                    record_vref=True):
    """Build native operands once and publish their per-slab cuts."""
    from types import SimpleNamespace
    from rfx._uniform_specs import _StepContext
    from rfx.core.drives import drive_layout
    from rfx.measurement import init_uniform_records
    from rfx.stepping.graded import graded_probe_table
    from rfx.stepping.slab import cut, place
    from rfx.stepping.slab_setup import stage_dispersion
    from rfx.stepping.slab_wrap import split
    from rfx.stepping.uniform_setup import _build_step_setup, core_step_invariants
    from jax.sharding import NamedSharding

    debye = stage_dispersion(materials, grid.dt, debye_spec, grid, layout, mesh, 'debye')
    lorentz = stage_dispersion(materials, grid.dt, lorentz_spec, grid, layout, mesh,
                              'lorentz', debye_spec=debye_spec)
    if lorentz is not None:
        from rfx.materials.lorentz import mark_judged
        lorentz = (mark_judged(lorentz[0]), lorentz[1])
    wires = [m for m, s in zip(port_plan.metadata, port_plan.specs) if s[2] == 'wire']
    lumped = [m for m, s in zip(port_plan.metadata, port_plan.specs) if s[2] == 'lumped']
    setup = _build_step_setup(grid, materials, boundary=boundary,
        cpml_axes='xyz', pec_axes=None, periodic=(False,) * 3,
        debye=debye, lorentz=lorentz, tfsf=None, sources=sources, probes=probes,
        dft_planes=[], flux_monitors=[], waveguide_ports=[], ntff=None,
        aniso_eps=None, aniso_inv_eps=None, aniso_inv_eps_smooth=False,
        pec_mask=None, pec_sheets=None, pec_wires=None, pec_edge_masks=pec_edges,
        pec_occupancy=None, conformal_weights=None, wire_port_sparams=wires,
        lumped_port_sparams=lumped, lumped_rlc=[], kerr_chi3=None,
        field_dtype=jnp.float32, mag_sources=[])
    ctx = _StepContext(**setup.ctx_kwargs, use_fast_he=False, use_snapshot=False,
        use_monitor=False, use_flux_window=False, cpml_grid=grid,
        port_record_specs=port_plan.specs, port_record_vref=record_vref)
    inputs = core_step_invariants(ctx)
    ctx = inputs.pop('ctx')
    factors = inputs.pop('magnetic_factors')
    inputs.update(materials=ctx.materials, pec_edge_masks=ctx.pec_edge_masks,
        drives=drive_layout(setup.src_meta, jnp.float32),
        probe_table=graded_probe_table(setup.prb_meta, grid.shape, dtype=jnp.float32))
    if port_plan.specs:
        inputs['port_table'] = port_plan.table
    table, port_table = inputs['probe_table'], inputs.get('port_table')
    carry = dict(setup.carry_init)
    init_uniform_records(carry, n_steps, wires, lumped, (), jnp.float32, specs=port_plan.specs)
    staged = {key: carry.pop(key) for key in ('debye', 'lorentz') if key in carry}
    profiles = {}
    if ctx.use_cpml:
        from rfx.stepping.slab_cpml import publish
        shell = SimpleNamespace(step_fn_inv=SimpleNamespace(context=ctx),
                                invariants={}, carry_init=carry)
        ctx, profiles, published = publish(shell, layout, mesh)
        staged['cpml'] = published['cpml']
        carry.pop('cpml')
    waves = (jnp.stack([s.waveform for s in sources], axis=-1) if sources
             else jnp.zeros((n_steps, 0), jnp.float32))
    port_carry = {key: carry.pop(key) for key in _PORT_CARRY if key in carry}
    inputs, carry = split(inputs, carry, layout, mesh, waveforms=waves, dt=grid.dt)
    carry.update(staged)
    carry.update(jax.tree.map(lambda a: place(jnp.broadcast_to(a,
        (layout.n_devices,) + a.shape), NamedSharding(mesh, P('x'))), port_carry))
    inputs.update(profiles)
    inputs['magnetic_factors'] = tuple(
        cut(a, layout, 'pmc_h_factor', mesh=mesh) if axis == 0 and a is not None else a
        for axis, a in enumerate(factors))
    for family, bundle in (('debye', debye), ('lorentz', lorentz)):
        if bundle is not None:
            inputs[family + '_coeffs'] = bundle[0]
    waves = inputs['drives'].waves
    inputs['drives'] = inputs['drives']._replace(waves=None)
    ctx = replace(ctx, materials=None, pec_edge_masks=None, pec_static_edge_masks=None,
                  debye_coeffs=None, lorentz_coeffs=None, drives=None)
    return carry, inputs, waves, wrap_uniform(ctx, layout, mesh,
                                             table=table, port_table=port_table)


def run_prepared(prepared, records, mesh, *, n_steps):
    """Keep monitor records outside the single-map canonical field step."""
    from rfx.stepping.rank import mesh_ranks
    from rfx.stepping.slab_measurements import sum_partials
    from rfx.stepping.slab_wrap import run_steps
    carry, invariants, waves, field_advance = prepared
    carry = dict(carry, records=records.initial())

    def advance(carry, xs, invariants, *, ranks):
        fields = {key: value for key, value in carry.items() if key != 'records'}
        previous = records.prepare(fields['fdtd'], ranks=ranks)
        final, samples = field_advance(fields, xs, invariants, ranks=ranks)
        final['records'] = records.update(carry['records'], final['fdtd'],
            records.grid.dt, xs[0], ranks=ranks, previous=previous)
        return final, samples

    ranks = mesh_ranks(mesh)
    final, samples = run_steps(carry, invariants, waves, ranks, advance,
        n_steps=n_steps, n_warmup=0, checkpoint_every=None)
    final = sum_partials(final, mesh, ranks)
    ports = {key: final[key] for key in _PORT_CARRY if key in final}
    return dict(final, ports=ports), jnp.sum(samples, axis=1), final['fdtd'], records
