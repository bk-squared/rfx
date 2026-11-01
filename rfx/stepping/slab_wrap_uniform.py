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
    """Form every spatial step operand from its owning slab's material rows."""
    from types import SimpleNamespace
    from jax.sharding import NamedSharding
    from rfx._uniform_specs import _StepContext
    from rfx.boundaries.pec import resolve_wall_faces
    from rfx.core.drives import drive_layout
    from rfx.core.yee import FDTDState, CurlBoundary
    from rfx.model.materials import kernel_materials
    from rfx.stepping.graded import graded_probe_table
    from rfx.stepping.slab_setup import realize_materials, stage_dispersion, input_materials
    from rfx.stepping.slab_wrap import cut, split

    sharding = NamedSharding(mesh, P('x'))
    replicated = NamedSharding(mesh, P())
    cells = materials._replace(components=None, edge_keep=None)
    cells = type(cells)(**{name: jax.tree.map(
        lambda a: a if np.ndim(a) == 0 else cut(a, layout,
            name if name in ('eps_r', 'sigma', 'mu_r') else 'lumped', mesh=mesh),
        value) for name, value in zip(cells._fields, cells)})
    pole_materials = input_materials(cells, layout, mesh)
    debye = stage_dispersion(pole_materials, grid.dt, debye_spec, grid, layout, mesh,
                            'debye', uniform=True)
    lorentz = stage_dispersion(pole_materials, grid.dt, lorentz_spec, grid, layout, mesh,
                              'lorentz', debye_spec=debye_spec, uniform=True)
    use_cpml = boundary == 'cpml' and grid.cpml_layers > 0
    local_materials, inverse, loss = realize_materials(
        pole_materials, grid, layout, mesh, cpml=use_cpml,
        debye=debye is not None, lorentz=lorentz is not None, uniform=True)
    local_materials = kernel_materials(local_materials,
        electric=debye is None and lorentz is None)
    pec_faces, pmc_faces = resolve_wall_faces(grid, (False,) * 3, None)
    wires = tuple(m for m, s in zip(port_plan.metadata, port_plan.specs) if s[2] == 'wire')
    lumped = tuple(m for m, s in zip(port_plan.metadata, port_plan.specs) if s[2] == 'lumped')
    ctx = _StepContext(grid=grid, materials=None, dt=grid.dt, dx=grid.dx,
        periodic=(False,) * 3, pec_axes=''.join(a for a in 'xyz'
            if {a + '_lo', a + '_hi'} <= pec_faces), stencil_order=2,
        use_fast_he=False, use_upml=False, use_cpml=use_cpml, use_tfsf=False,
        use_debye=debye is not None, use_lorentz=lorentz is not None,
        use_ntff=False, use_dft_planes=False, use_flux_monitors=False,
        use_waveguide_ports=False, use_pec_faces=bool(pec_faces),
        use_pmc_faces=bool(pmc_faces), use_aniso_inv=False, aniso_inv_eps_smooth=False,
        use_pec_edges=pec_edges is not None, use_pec_occupancy=False,
        use_conformal=False, use_wire_sparams=bool(wires), use_lumped_sparams=bool(lumped),
        use_lumped_rlc=False, use_kerr=False, use_snapshot=False, use_monitor=False,
        use_flux_window=False, cpml_grid=grid, cpml_axes='xyz',
        pec_faces_frozen=pec_faces, pmc_faces_frozen=pmc_faces,
        curl_boundary=CurlBoundary(pec_faces, pmc_faces, (False,) * 3),
        port_record_specs=port_plan.specs, port_record_vref=record_vref,
        wire_sparam_meta=wires, lumped_sparam_meta=lumped)
    field_shape = (layout.n_devices * layout.nx_local, *grid.shape[1:])
    carry = {'fdtd': FDTDState(**{name: jnp.zeros(field_shape, jnp.float32, device=sharding)
        for name in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')},
        step=jax.device_put(jnp.int32(0), replicated))}
    profiles = {}
    if use_cpml:
        from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
        from rfx.stepping.slab_cpml import publish

        def allocate_psi(name, shape, dtype):
            local_x = shape[0] if name.split('_')[-1][0] == 'x' else layout.nx_local
            return jnp.zeros((layout.n_devices * local_x, *shape[1:]), dtype, device=sharding)

        params, carry['cpml'] = init_cpml(grid, state_allocator=allocate_psi)
        ctx = replace(ctx, cpml_params=params, apply_cpml_e=apply_cpml_e, apply_cpml_h=apply_cpml_h)
        shell = SimpleNamespace(step_fn_inv=SimpleNamespace(context=ctx),
                                invariants={}, carry_init=carry)
        ctx, profiles, carry = publish(shell, layout, mesh, staged_state=True)
    for family, bundle in (('debye', debye), ('lorentz', lorentz)):
        if bundle is not None:
            carry[family] = bundle[1]
    if port_plan.specs:
        carry['sparam_time_records'] = tuple(jnp.zeros(
            (layout.n_devices, n_steps, 4 if s[2] == 'wire' else 3),
            jnp.float32, device=sharding) for s in port_plan.specs)
    for kind, metas, count in (('wire', wires, 5), ('lumped', lumped, 3)):
        if metas:
            carry[kind + '_sparam_accs'] = tuple(tuple(jnp.zeros(
                (layout.n_devices, len(m.freqs)),
                jnp.result_type(jnp.complex64, jnp.float64), device=sharding)
                for _ in range(count)) for m in metas)
    table = graded_probe_table([(p.i, p.j, p.k, p.component) for p in probes],
                               grid.shape, dtype=jnp.float32)
    inputs = dict(materials=local_materials,
        pec_edge_masks=cut(pec_edges, layout, 'pec_edge', mesh=mesh),
        cpml_inv_eps_r=inverse, e_loss=loss, sheet_coeffs=None, conformal_zero_masks=None,
        drives=drive_layout([(s.i, s.j, s.k, s.component) for s in sources], jnp.float32),
        probe_table=table)
    port_table = port_plan.table if port_plan.specs else None
    if port_table is not None:
        inputs['port_table'] = port_table
    waves = (jnp.stack([s.waveform for s in sources], axis=-1) if sources
             else jnp.zeros((n_steps, 0), jnp.float32))
    inputs, carry = split(inputs, carry, layout, mesh, waveforms=waves, dt=grid.dt, staged=True)
    if ctx.use_pec_edges:
        row = jnp.arange(layout.n_devices * layout.nx_local) % layout.nx_local
        real = (row >= layout.ghost_width) & (row < layout.nx_local - layout.ghost_width)
        inputs['pec_edge_masks'] = jax.tree.map(
            lambda mask: mask & real[:, None, None], inputs['pec_edge_masks'])
    inputs.update(profiles)
    from rfx.boundaries.pmc import magnetic_update_factors
    factors = magnetic_update_factors(pmc_faces, grid.shape, ctx.periodic)
    inputs['magnetic_factors'] = tuple(
        cut(a, layout, 'pmc_h_factor', mesh=mesh) if axis == 0 and a is not None else a
        for axis, a in enumerate(factors))
    for family, bundle in (('debye', debye), ('lorentz', lorentz)):
        if bundle is not None:
            inputs[family + '_coeffs'] = bundle[0]
    waves = inputs['drives'].waves
    inputs['drives'] = inputs['drives']._replace(waves=None)
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
