"""Frames equal STEP_END states observed in the same-length one-device run."""
import jax
import numpy as np
import pytest
from rfx import SnapshotSpec
from tests._multi_device_scene import host_devices
from tests.unit.runners._graded_records_scene import model, measure, record, mutation_result, with_bars


@pytest.mark.parametrize('n_devices,boundary', [(2, 'pec'), (3, 'pec'), (2, 'cpml'), (3, 'cpml')])
@pytest.mark.parametrize('interval,axis', [(1, None), (3, 0), (3, 1)])
def test_snapshots(n_devices, boundary, interval, axis):
    sim, grid, seam = model(boundary, n_devices)
    index = None if axis is None else seam if axis == 0 else grid.shape[1] // 2
    spec = SnapshotSpec(interval=interval, components=('ez', 'hy'), slice_axis=axis, slice_index=index)
    steps = 7
    expected = {c: [] for c in spec.components}
    from rfx.stepping import graded
    from rfx.stepping.sequence import HookPoint
    original = graded.make_graded_step
    def observe(context, **kwargs):
        def capture(frame):
            def save(n, *fields):
                if (int(n) + 1) % interval == 0:
                    for c, value in zip(spec.components, fields):
                        value = np.asarray(value)
                        expected[c].append(value.copy() if axis is None else np.take(value, index, axis=axis))
            jax.debug.callback(save, frame.step_idx, *(getattr(frame.st, c) for c in spec.components), ordered=True)
        return original(context, hooks={HookPoint.STEP_END: (capture,)}, **kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(graded, 'make_graded_step', observe)
        one = sim.run(n_steps=steps, skip_preflight=True)
        jax.block_until_ready(one.state)
        jax.effects_barrier()
    many = sim.run(n_steps=steps, devices=host_devices(n_devices), snapshot=spec, skip_preflight=True)
    jax.block_until_ready(many.snapshots)
    values = {f'{c}/step{(i + 1) * interval}': measure(actual, reference)
              for c in spec.components
              for i, (actual, reference) in enumerate(zip(many.snapshots[c], expected[c]))}
    assert all(len(many.snapshots[c]) == len(expected[c]) for c in spec.components)
    # CPML wrapped step: compile-form rounding, operands byte-equal (rfx-archive DESIGN.md 4g amendment 1).
    with_bars(values, 'snapshot_ulp' if boundary == 'cpml' else 'array_equal')
    record('snapshots', dict(boundary=boundary, devices=n_devices, interval=interval, axis=axis), values)
    assert all(v['passed'] for v in values.values()), ('snapshot bar', values)
    for c, axes in many.snapshot_axes.items():
        for a, name in enumerate('xyz'):
            if name not in axes.coords:
                continue
            widths = np.asarray(one.grid.cells(a), np.float64)
            nodes = np.asarray([one.grid.node_of(a, i) for i in range(one.grid.shape[a])])
            staggered = (c == 'ez' and a == 2) or (c == 'hy' and a in (0, 2))
            expected_coords = nodes + (widths * .5 if staggered else 0.)
            np.testing.assert_array_equal(axes.coords[name], expected_coords)
        np.testing.assert_array_equal(axes.steps, np.arange(interval, steps + 1, interval))
        np.testing.assert_array_equal(axes.times_s, (axes.steps - (0.5 if c.startswith('h') else 0)) * grid.dt)


@pytest.mark.parametrize('mutation', ['one_step_late', 'skip'])
@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
def test_snapshot_mutations(mutation, boundary, monkeypatch):
    from rfx.runners._distributed_records import SnapshotRecords
    original = SnapshotRecords.update
    monkeypatch.setattr(SnapshotRecords, 'update',
        (lambda self, buffer, state, dt, step, **kw: original(self, buffer, state, dt, step - 1, **kw))
        if mutation == 'one_step_late' else (lambda self, buffer, *a, **kw: buffer))
    with pytest.raises(AssertionError, match='snapshot bar') as failure:
        test_snapshots(2, boundary, 3, 0)
    mutation_result('snapshots', f'{mutation}/{boundary}', failure.value)


@pytest.mark.parametrize('mutation', ['skipped_cpml_psi_exchange', 'x_cpml_profile_cut_offset'])
def test_cpml_seam_snapshot_mutations(mutation, monkeypatch):
    """Measure Hy halo and x-profile cut defects on the asymmetric CPML scene.

    skipped_cpml_psi_exchange uses the skipped-Hy exchange fallback: the
    wrapped step has no psi exchange to skip.
    """
    import jax.numpy as jnp
    from rfx.runners import distributed_nu
    from rfx.stepping import graded, slab_cpml
    from rfx.stepping.sequence import HookPoint
    from tests.unit.runners.test_distributed_graded_records_ntff import ntff_model

    devices = host_devices(2)
    sim, grid = ntff_model('cpml', 2)
    sim._probes.clear()
    spec = SnapshotSpec(interval=1, components=('ez', 'hy'))
    steps = 24
    expected = {c: [] for c in spec.components}
    original_step = graded.make_graded_step

    def observe(context, **kwargs):
        def capture(frame):
            def save(*fields):
                for c, value in zip(spec.components, fields):
                    expected[c].append(np.asarray(value).copy())
            jax.debug.callback(save, *(getattr(frame.st, c) for c in spec.components), ordered=True)
        return original_step(context, hooks={HookPoint.STEP_END: (capture,)}, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(graded, 'make_graded_step', observe)
        one = sim.run(n_steps=steps, skip_preflight=True)
        jax.block_until_ready(one.state)
        jax.effects_barrier()

    def run():
        result = sim.run(n_steps=steps, devices=devices, snapshot=spec, skip_preflight=True)
        jax.block_until_ready(result.snapshots)
        jax.effects_barrier()
        return result.snapshots

    def metrics(snapshots):
        assert all(len(snapshots[c]) == len(expected[c]) == steps for c in spec.components)
        values = {}
        for c in spec.components:
            for i, (actual, reference) in enumerate(zip(snapshots[c], expected[c])):
                value = measure(actual, reference)
                value['ulp'] = value['max_abs'] / float(np.spacing(np.float32(value['peak'])))
                delta = np.abs(np.asarray(actual, np.float64) - reference.astype(np.float64))
                value['cell'] = [int(v) for v in np.unravel_index(np.argmax(delta), delta.shape)]
                value.update(step=i + 1, component=c)
                values[f'{c}/step{i + 1}'] = value
        return with_bars(values, 'snapshot_ulp')

    # CPML wrapped step: compile-form rounding, operands byte-equal (rfx-archive DESIGN.md 4g amendment 1).
    control = run()
    control_values = metrics(control)
    record('snapshot_seam_control', dict(mutation=mutation, devices=2, boundary='cpml'), control_values)
    assert all(v['passed'] for v in control_values.values()), ('snapshot control bar', control_values)
    reached = []
    profile_operands = {}
    if mutation == 'skipped_cpml_psi_exchange':
        # slab_wrap.wrap exchanges only frame.st: psi has no halo exchange.
        # Authorized fallback: retain rank 1's pre-exchange Hy low ghost only.
        original_exchange = distributed_nu._exchange_h_ghosts_nu

        def skip_hy_exchange_fallback(state, mesh, n_devices, *, ranks, local=False):
            """No psi exchange exists; skip only rank 1's Hy low ghost exchange."""
            assert local and n_devices == 2
            reached.append('hy')
            exchanged = original_exchange(state, mesh, n_devices, ranks=ranks, local=local)
            row = jnp.where(ranks == 1, state.hy[0], exchanged.hy[0])
            return exchanged._replace(hy=exchanged.hy.at[0].set(row))

        monkeypatch.setattr(distributed_nu, '_exchange_h_ghosts_nu', skip_hy_exchange_fallback)
    else:
        # slab_cpml.publish broadcasts whole profiles. Mutate after apply has
        # formed the rank-local profile, immediately before the boundary call.
        original_apply = slab_cpml.apply

        def shift_x_hi(kernel, ctx, invariants, state, psi, **kwargs):
            reached.append(kernel.__name__)
            rank = jax.lax.axis_index('x')

            def shifted_kernel(local_state, params, local_psi, local_grid, **kernel_kwargs):
                owner = kernel_kwargs['owned']['x_hi']

                def shift(a):
                    assert a.ndim == 1 and a.shape[0] >= grid.pad_x_hi
                    # Hi profiles are stored in increasing x, outermost last;
                    # the kernel clips any leading allocation padding.
                    return jnp.where(owner, jnp.concatenate((a[1:], a[-1:])), a)

                shifted = params._replace(x_hi=jax.tree.map(shift, params.x_hi),
                    magnetic=params.magnetic._replace(x_hi=jax.tree.map(shift, params.magnetic.x_hi)))

                def save(r, electric, shifted_electric, magnetic, shifted_magnetic):
                    for prefix, before, after in (('electric', electric, shifted_electric),
                                                  ('magnetic', magnetic, shifted_magnetic)):
                        for name, a, b in zip(before._fields, before, after):
                            a, b = np.asarray(a), np.asarray(b)
                            profile_operands[f'rank{int(r)}/{prefix}/x_hi/{name}'] = dict(
                                before=a.tolist(), after=b.tolist(),
                                max_abs=float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))

                jax.debug.callback(save, rank, params.x_hi, shifted.x_hi,
                                   params.magnetic.x_hi, shifted.magnetic.x_hi)
                return kernel(local_state, shifted, local_psi, local_grid, **kernel_kwargs)

            return original_apply(shifted_kernel, ctx, invariants, state, psi, **kwargs)

        monkeypatch.setattr(slab_cpml, 'apply', shift_x_hi)

    mutated = run()
    assert reached, 'wrapped mutation was not reached'
    values = metrics(mutated)
    failures = sorted((v for v in values.values() if not v['passed']), key=lambda v: (v['step'], v['component']))
    first = failures[0] if failures else None
    effect = {c: measure(mutated[c], control[c]) for c in spec.components}
    metadata = dict(mutation=mutation, devices=2, boundary='cpml', interval=1, steps=steps,
        shape=grid.shape, fallback='skipped_hy_low_ghost' if mutation == 'skipped_cpml_psi_exchange' else None,
        first_exceedance=first, max_ulp=max(v['ulp'] for v in values.values()),
        x_hi_profile_operands=profile_operands, mutation_vs_control=effect)
    record('snapshot_seam_mutation', metadata, values)
    if failures:
        mutation_result('snapshots', mutation, metadata)
    else:
        record('mutation', dict(record='snapshots', mutation=mutation, status='GREEN'), metadata)
    assert failures, ('seam mutation did not exceed 9 ULP', metadata)
