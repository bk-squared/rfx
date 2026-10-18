"""D4 graded port parity against the same public one-device declaration."""
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.sources.sources import GaussianPulse, _axis_node_position
from rfx._grid_metric import field_index

pytestmark = pytest.mark.distributed
BINS = np.array([3e9, 7e9, 13e9], np.float32)


def scene(case, devices, boundary='pec'):
    if case.startswith('C4'):
        boundary = 'cpml'
        case = 'C2' if case == 'C4-wire' else 'C1-mid'
    dx, dy, dz = (np.linspace(.8e-3, 1.2e-3, 24),
                  np.linspace(.9e-3, 1.3e-3, 12),
                  np.linspace(.7e-3, .95e-3, 10))
    dx[[0, -1]] = dy[[0, -1]] = dz[[0, -1]] = 1e-3
    sim = Simulation(freq_max=25e9, domain=tuple(float(a.sum()) for a in (dx, dy, dz)),
        dx=1e-3, dx_profile=dx, dy_profile=dy, dz_profile=dz,
        boundary=boundary, cpml_layers=2)
    grid = sim._build_nonuniform_grid()
    size = (grid.nx + devices - 1) // devices
    seam = size
    x = seam if case == 'C1-seam' else seam + 2
    component = 'ex' if case in ('C1b', 'C2', 'C5') else 'ez'
    if component == 'ex':
        x = seam - 2
    cell = (x, grid.ny // 2 - 1, grid.nz // 2 - 1)
    pos = tuple(_axis_node_position(grid, axis, idx) for axis, idx in enumerate(cell))
    extent = None
    if component == 'ex':
        extent = _axis_node_position(grid, 0, seam + 3) - pos[0]
    sim.add_port(pos, component, impedance=50., extent=extent,
        waveform=GaussianPulse(f0=20e9, bandwidth=.9))
    if case == 'C3':
        receiver = (x + 1, cell[1] + 1, cell[2])
        location = tuple(_axis_node_position(grid, a, i) for a, i in enumerate(receiver))
        sim.add_port(location, 'ez', impedance=65., excite=False)
    sim.add_probe(pos, component)
    assert grid.nx % devices != 0
    if component == 'ez':
        assert field_index(grid, pos, component)[0] == x
    return sim


def measure(a, b):
    a, b = np.asarray(a), np.asarray(b)
    finite = np.isfinite(a)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.array_equal(np.isinf(a), np.isinf(b))
    peak = float(np.max(np.abs(a[finite]), initial=0))
    delta = float(np.max(np.abs(a.astype(np.complex128)[finite] - b.astype(np.complex128)[finite]), initial=0))
    unequal = (a != b) & ~(np.isnan(a) & np.isnan(b))
    first = np.argwhere(unequal)
    return dict(bit_equal=a.tobytes() == b.tobytes(), peak=peak, max_abs_difference=delta,
        ulp_peak=delta / float(np.spacing(np.float32(peak))),
        relative=delta / peak if peak else (0. if not delta else float('inf')),
        first_difference=first[0].tolist() if len(first) else None)


def record(row):
    path = os.environ.get('RFX_STEP2_REPORT_DIR')
    if path:
        with (Path(path) / 'port-measurements.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')


def evaluate(sim, entry, n, devices=None, eps=None):
    if entry == 'run':
        wire = any(p.extent is not None for p in sim._ports)
        return sim.run(n_steps=n, devices=devices, skip_preflight=True, compute_s_params=wire,
                       s_param_freqs=BINS if wire else None)
    return sim.forward(n_steps=n, distributed=devices is not None, devices=devices,
        checkpoint=False, skip_preflight=True, port_s11_freqs=BINS, eps_override=eps)


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('n', [128, 192, 288])
@pytest.mark.parametrize('case', ['C1-mid', 'C1-seam', 'C1b', 'C2', 'C3', 'C4-mid', 'C4-wire'])
def test_port_records_and_s(case, n, entry):
    devices = jax.devices('cpu')
    if len(devices) < 2:
        pytest.skip('requires 2 or 3 host devices')
    sim = scene(case, len(devices))
    one = evaluate(sim, entry, n)
    multi = evaluate(sim, entry, n, devices)
    row = dict(case=case, devices=len(devices), entry=entry, n=n, records={}, accumulators={}, S=None)
    if entry == 'run':
        row['final_fields'] = {field: measure(getattr(one.state, field), getattr(multi.state, field))
            for field in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')}
        for metric in row['final_fields'].values():
            assert metric['ulp_peak'] <= 9, metric
    for port, (a, b) in enumerate(zip(one.sparam_time_records, multi.sparam_time_records, strict=True)):
        assert a.channels == b.channels
        for channel, name in enumerate(a.channels):
            row['records'][f'{port}/{name}'] = measure(a.samples[:, channel], b.samples[:, channel])
    if one.s_params is not None:
        row['S'] = measure(one.s_params, multi.s_params)
        row['S_entries'] = {str(i): measure(one.s_params[i], multi.s_params[i]) for i in np.ndindex(one.s_params.shape[:-1])}
        for port, ((_, a), (_, b)) in enumerate(zip(one.wire_port_sparams, multi.wire_port_sparams, strict=True)):
            for name, aa, bb in zip(('v_dft', 'i_dft', 'v_inc_dft', 'v_port_dft'), a, b):
                row['accumulators'][f'{port}/{name}'] = measure(aa, bb)
    record(row)
    p1 = case in ('C1-mid', 'C3', 'C4-mid')
    for name, measured in row['records'].items():
        assert measured['ulp_peak'] <= 9, (name, measured)
        if p1:
            assert measured['bit_equal'], (name, measured)
    for name, measured in row['accumulators'].items():
        assert measured['relative'] <= 1e-4, (name, measured)
        if p1:
            assert measured['bit_equal'], (name, measured)
    for metric in row.get('S_entries', {}).values():
        assert metric['relative'] <= 1e-4, metric
    if row['S'] is not None:
        assert row['S']['relative'] <= 1e-4, row['S']
        if p1:
            assert row['S']['bit_equal'], row['S']


@pytest.mark.parametrize('n', [128, 192, 288])
def test_c5_gradient(n):
    devices = jax.devices('cpu')
    if len(devices) < 2:
        pytest.skip('requires multiple devices')
    sim = scene('C5', len(devices))
    eps = jnp.full(sim._build_nonuniform_grid().shape, 1.7, jnp.float32)
    def objective(e, selected):
        result = evaluate(sim, 'forward', n, selected, e)
        return jnp.sum(jnp.abs(result.s_params[:, 0]) ** 2), result.s_params
    (_, one), g_one = jax.value_and_grad(lambda e: objective(e, None), has_aux=True)(eps)
    (_, multi), g_multi = jax.value_and_grad(lambda e: objective(e, devices), has_aux=True)(eps)
    row = dict(case='C5', devices=len(devices), entry='forward', n=n,
               S=measure(one, multi), gradient=measure(g_one, g_multi))
    record(row)
    assert row['S']['relative'] <= 1e-4, row
    assert row['gradient']['relative'] <= 1e-4, row


def test_run_default_s_grid_equals_one_device():
    """run() with no s_param_freqs: the same 50 default bins and S as one device."""
    devices = jax.devices('cpu')
    if len(devices) < 2:
        pytest.skip('requires multiple devices')
    sim = scene('C2', len(devices))
    one = sim.run(n_steps=128, skip_preflight=True)
    multi = sim.run(n_steps=128, devices=devices, skip_preflight=True)
    assert np.asarray(one.freqs).shape == (50,)
    np.testing.assert_array_equal(np.asarray(multi.freqs), np.asarray(one.freqs))
    row = dict(case='C2-default-bins', devices=len(devices), entry='run', n=128,
               S=measure(one.s_params, multi.s_params))
    record(row)
    assert row['S']['relative'] <= 1e-4, row


def test_c5_gradient_under_user_jit():
    """Port assembly reads host cell sizes; an outer jit must not trace them."""
    devices = jax.devices('cpu')
    if len(devices) < 2:
        pytest.skip('requires multiple devices')
    sim = scene('C5', len(devices))
    eps = jnp.full(sim._build_nonuniform_grid().shape, 1.7, jnp.float32)
    def objective(e, selected):
        result = evaluate(sim, 'forward', 128, selected, e)
        return jnp.sum(jnp.abs(result.s_params[:, 0]) ** 2), result.s_params
    (_, one), g_one = jax.value_and_grad(lambda e: objective(e, None), has_aux=True)(eps)
    (_, multi), g_multi = jax.jit(jax.value_and_grad(lambda e: objective(e, devices), has_aux=True))(eps)
    row = dict(case='C5-jit', devices=len(devices), entry='forward', n=128,
               S=measure(one, multi), gradient=measure(g_one, g_multi))
    record(row)
    assert row['S']['relative'] <= 1e-4, row
    assert row['gradient']['relative'] <= 1e-4, row


def test_c2_vmap(monkeypatch):
    devices = jax.devices('cpu')[:2]
    if len(devices) < 2:
        pytest.skip('requires two devices')
    sim = scene('C2', 2)
    eps = jnp.stack([jnp.full(sim._build_nonuniform_grid().shape, e, jnp.float32) for e in (1.4, 1.8)])
    from rfx.runners import _distributed_nu_wrapped as wrapped
    build, captured = wrapped.build_setup, {}
    def capture(*args, **kwargs):
        setup = build(*args, **kwargs)
        captured['waves'] = setup.src_waveforms
        return setup
    monkeypatch.setattr(wrapped, 'build_setup', capture)
    def f(e):
        result = evaluate(sim, 'forward', 128, devices, e)
        return (result.s_params, tuple(r.samples for r in result.sparam_time_records),
                tuple(a for _, a in result.wire_port_sparams), captured['waves'])
    one = jax.tree.map(lambda *a: jnp.stack(a), *[f(e) for e in eps])
    batch = jax.vmap(f)(eps)
    row = dict(case='C2-vmap', devices=2, S=measure(one[0], batch[0]), records={}, accumulators={},
        drives={str(c): measure(one[3][..., c], batch[3][..., c]) for c in range(one[3].shape[-1])})
    for port, (a, b) in enumerate(zip(one[1], batch[1])):
        for channel in range(a.shape[-1]):
            row['records'][f'{port}/{channel}'] = measure(a[..., channel], b[..., channel])
    for port, (a, b) in enumerate(zip(one[2], batch[2])):
        for channel, (aa, bb) in enumerate(zip(a, b)):
            row['accumulators'][f'{port}/{channel}'] = measure(aa, bb)
    record(row)
    assert row['S']['relative'] <= 1e-4, row
    for metric in (*row['records'].values(), *row['drives'].values()):
        assert metric['ulp_peak'] <= 9, metric
    for metric in row['accumulators'].values():
        assert metric['relative'] <= 1e-4, metric


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('kind', ['cpml_poles', 'pmc', 'reference_plane'])
def test_refuses_before_staging(entry, kind, monkeypatch):
    from rfx import Box, LorentzPole
    from rfx.boundaries.spec import BoundarySpec
    from rfx.model import electric_metrics
    sim = scene('C2', len(jax.devices()), boundary='cpml' if kind == 'cpml_poles' else 'pec')
    if kind == 'cpml_poles':
        sim.add_material('pole', eps_r=3., lorentz_poles=[LorentzPole(1., 2*np.pi*3e9, 1e9)])
        sim.add(Box((.002, .002, .002), (.004, .004, .004)), material='pole')
        message = 'item 5 dispersive'
    elif kind == 'pmc':
        sim._boundary_spec = BoundarySpec(x='pec', y='pmc', z='pec')
        message = 'PMC'
    else:
        from dataclasses import replace
        sim._ports[0] = replace(sim._ports[0], reference_plane_cells=2)
        message = 'reference_plane_cells'
    def staging(*args, **kwargs):
        pytest.fail('refusal reached material staging')
    monkeypatch.setattr(electric_metrics, 'stage_forward_materials', staging)
    with pytest.raises(NotImplementedError, match=message):
        evaluate(sim, entry, 8, jax.devices())


@pytest.mark.parametrize('kind', ['debye', 'lorentz'])
def test_override_pole_edge_refuses_before_staging(kind, monkeypatch):
    from rfx import Box, DebyePole, LorentzPole
    from rfx.model import electric_metrics
    sim = scene('C2', len(jax.devices()))
    poles = {'debye_poles': [DebyePole(1., 1e-11)]} if kind == 'debye' else {
        'lorentz_poles': [LorentzPole(1., 2*np.pi*3e9, 1e9)]}
    sim.add_material('pole', eps_r=3., **poles)
    sim.add(Box((0., 0., 0.), sim._domain), material='pole')
    def staging(*args, **kwargs):
        pytest.fail('refusal reached material staging')
    monkeypatch.setattr(electric_metrics, 'stage_forward_materials', staging)
    with pytest.raises(NotImplementedError, match='step2-graded-port-s design'):
        evaluate(sim, 'forward', 8, jax.devices(), jnp.ones(sim._build_nonuniform_grid().shape))


def test_two_excited_wire_ports_refuse_run_s_before_staging(monkeypatch):
    from rfx.model import electric_metrics
    def two_excited():
        sim = scene('C2', len(jax.devices()))
        grid = sim._build_nonuniform_grid()
        cell = (4, grid.ny // 2 - 1, grid.nz // 2 - 1)
        pos = tuple(_axis_node_position(grid, a, i) for a, i in enumerate(cell))
        sim.add_port(pos, 'ex', impedance=50., extent=_axis_node_position(grid, 0, 7) - pos[0],
            waveform=GaussianPulse(f0=20e9, bandwidth=.9))
        return sim
    with pytest.raises(ValueError, match='isolated drive') as one:
        two_excited().run(n_steps=8, compute_s_params=True, s_param_freqs=BINS, skip_preflight=True)
    def staging(*args, **kwargs):
        pytest.fail('refusal reached material staging')
    monkeypatch.setattr(electric_metrics, 'stage_forward_materials', staging)
    with pytest.raises(ValueError, match='isolated drive') as multi:
        two_excited().run(n_steps=8, devices=jax.devices(), compute_s_params=True,
            s_param_freqs=BINS, skip_preflight=True)
    assert str(multi.value) == str(one.value)


@pytest.mark.parametrize('entry', ['run', 'forward'])
@pytest.mark.parametrize('n', [128, 192, 288])
@pytest.mark.parametrize('case', ['C1-mid', 'C1-seam', 'C1b', 'C2', 'C3', 'C4-mid', 'C4-wire', 'C5'])
def test_model_drives_and_cut(case, n, entry, monkeypatch):
    if case == 'C5' and entry == 'run':
        pytest.skip('C5 specifies traced forward overrides')
    import rfx.stepping.graded_loop as nu
    from rfx.runners import _distributed_nu_wrapped as wrapped
    from rfx.stepping import slab_setup
    from rfx.stepping.slab import cut
    from rfx.model.materials import e_update_coefficient_at
    from rfx.nonuniform import current_source_cb
    from jax.sharding import Mesh
    selected = jax.devices('cpu')
    sim = scene(case, len(selected))
    seen = {}
    canonical, build, finish = nu._build_nu_scan, wrapped.build_setup, wrapped.split
    def one_setup(*args, **kwargs):
        setup = canonical(*args, **kwargs)
        seen['one'] = setup
        seen['one_material'] = args[1]
        return setup
    def slab_setup_spy(*args, **kwargs):
        setup = build(*args, **kwargs)
        seen['multi'] = setup
        seen['layout'] = args[2]
        return setup
    def cut_spy(*args, **kwargs):
        output = finish(*args, **kwargs)
        seen['cut'] = output[0]
        return output
    monkeypatch.setattr(nu, '_build_nu_scan', one_setup)
    monkeypatch.setattr(wrapped, 'build_setup', slab_setup_spy)
    monkeypatch.setattr(wrapped, 'split', cut_spy)
    eps = jnp.full(sim._build_nonuniform_grid().shape, 1.7, jnp.float32) if case == 'C5' else None
    one = evaluate(sim, entry, n, eps=eps)
    multi = evaluate(sim, entry, n, selected, eps)
    expected, actual, layout = seen['one'], seen['multi'], seen['layout']
    waves_one = np.asarray(expected.src_waveforms)
    waves_multi = np.asarray(actual.src_waveforms).reshape(n, len(selected), -1).sum(axis=1)
    row = dict(case=case, devices=len(selected), entry=entry, n=n,
               drives={str(c): measure(waves_one[:, c], waves_multi[:, c]) for c in range(waves_one.shape[1])},
               A1=[], A3=[], padding=[])
    global_mat = seen['one_material']
    slab_mat = actual.invariants['materials']
    for spec in expected.step_fn_inv.context.port_record_specs:
        component, live, _, _, _ = spec
        axis = ('ex', 'ey', 'ez').index(component)
        for cell in live:
            owner = layout.owner(cell[0])
            local = (owner * layout.nx_local + layout.local_index(cell[0]), *cell[1:])
            a = e_update_coefficient_at(global_mat, cell, component, expected.step_fn_inv.context.dt)
            b = current_source_cb(slab_mat.components.eps_update[axis][local],
                                 slab_mat.components.sigma_update[axis][local], expected.step_fn_inv.context.dt)
            row['A1'].append(dict(cell=list(cell), component=component, owner=owner,
                one_bytes=np.asarray(a).tobytes().hex(), slab_bytes=np.asarray(b).tobytes().hex(), **measure(a, b)))
    a_edges = cut(expected.invariants['pec_edge_masks'], layout, 'pec_edge', mesh=Mesh(np.asarray(selected), ('x',)))
    b_edges = actual.invariants['pec_edge_masks']
    if a_edges is None:
        assert b_edges is None
        row['A3'] = dict(differing_entries=0, entries=0)
    else:
        row['A3'] = dict(differing_entries=sum(int(np.count_nonzero(np.asarray(a) != np.asarray(b))) for a,b in zip(a_edges, b_edges)),
                        entries=sum(int(a.size) for a in a_edges))
    parts = seen['cut']['port_table'][0]
    for term, (indices, channels, weights) in zip(actual.invariants['port_table'].components, parts):
        indices = [np.asarray(i) for i in indices]
        weights = np.asarray(weights)
        pad = weights == 0
        ranges = []
        for axis, index in enumerate(indices):
            size = layout.nx_local if axis == 0 else sim._build_nonuniform_grid().shape[axis]
            assert np.all((index >= 0) & (index < size))
            ranges.append([int(index[pad].min()), int(index[pad].max())] if np.any(pad) else None)
        row['padding'].append(dict(component=term.component, count=int(pad.sum()), index_ranges=ranges))
    if case == 'C5':
        row['records'] = {f'{p}/{c}': measure(a.samples[:, c], b.samples[:, c])
            for p, (a, b) in enumerate(zip(one.sparam_time_records, multi.sparam_time_records))
            for c in range(a.samples.shape[-1])}
        row['accumulators'] = {f'{p}/{c}': measure(aa, bb)
            for p, ((_, a), (_, b)) in enumerate(zip(one.wire_port_sparams, multi.wire_port_sparams))
            for c, (aa, bb) in enumerate(zip(a, b))}
        for metric in row['records'].values():
            assert metric['ulp_peak'] <= 9, metric
        for metric in row['accumulators'].values():
            assert metric['relative'] <= 1e-4, metric
    record(row)
    for column, metric in row['drives'].items():
        assert metric['ulp_peak'] <= 9, (column, metric)
        if case != 'C5':
            assert metric['bit_equal'], (column, metric)
    assert row['A3']['differing_entries'] == 0, row['A3']
    for cell in row['A1']:
        assert cell['ulp_peak'] <= 9, cell
    assert np.any(np.asarray(one.sparam_time_records[0].samples) != 0)
    assert np.all(np.isfinite(np.asarray(multi.sparam_time_records[0].samples)))


@pytest.fixture(autouse=True)
def restored_defect(monkeypatch):
    """Opt-in restored defects; each is run in isolation, then without the flag."""
    defect = os.environ.get('RFX_STEP2_MUTATION')
    if not defect:
        return
    from rfx.stepping import slab_ports, slab_setup, slab_wrap
    from rfx.model import distributed_ports
    if defect in ('M1', 'M1b', 'M1c'):
        original = slab_ports.sum_partials
        def broken(final, mesh, ranks):
            totals = original(final, mesh, ranks)
            for key in ('wire_sparams', 'sparam_time_records'):
                if key in final:
                    if defect == 'M1':
                        totals[key] = jax.tree.map(lambda a: a[0], final[key])
                    elif defect == 'M1b':
                        totals[key] = jax.tree.map(lambda a: jnp.sum(a, axis=0), final[key])
                    else:
                        from rfx.stepping.rank import rank_shard_map
                        from jax.sharding import PartitionSpec as P
                        def again(values, *, rank):
                            return jax.tree.map(lambda a: jax.lax.psum(a, 'x'), values)
                        totals[key] = rank_shard_map(again, mesh=mesh, in_specs=P(),
                            out_specs=P(), check_rep=False)(ranks, totals[key])
            return totals
        monkeypatch.setattr(slab_ports, 'sum_partials', broken)
    elif defect in ('M2', 'M7'):
        from rfx.measurement import projector
        from rfx.stepping.sequence import HookPoint
        original, project = slab_wrap.make_graded_step, projector.project
        def broken(context, *, hooks):
            captured = {}
            def before(frame):
                captured['state'] = frame.st
                if defect == 'M7':
                    frame.step_idx = frame.step_idx + (jax.lax.axis_index('x') == 1)
            def stale(state, table):
                return project(captured['state'] if defect == 'M2' and 'V' in table.channel_names else state, table)
            with monkeypatch.context() as patch:
                patch.setattr(projector, 'project', stale)
                return original(context, hooks={**hooks, HookPoint.BEFORE_SOURCES: (before,)})
        monkeypatch.setattr(slab_wrap, 'make_graded_step', broken)
    elif defect == 'M3':
        original = slab_setup.realize_materials
        def broken(materials, grid, layout, mesh, *args, **kwargs):
            staged, inverse, loss = original(materials, grid, layout, mesh, *args, **kwargs)
            # These vacuum fixtures have no other sigma: remove the port load
            # on rank 1, retaining all calls and every other rank's operands.
            sigma = tuple(a.at[layout.nx_local:2*layout.nx_local].set(0.)
                          for a in staged.components.sigma_update)
            from dataclasses import replace
            staged = staged._replace(components=replace(staged.components, sigma_update=sigma))
            return staged, inverse, loss
        monkeypatch.setattr(slab_setup, 'realize_materials', broken)
    elif defect == 'M4':
        from rfx.runners import _admission
        for lane in ('run_distributed_nu', 'fwd_distributed_nu'):
            monkeypatch.setitem(_admission.ADMITS, lane,
                _admission.ADMITS[lane] - {('_ports', kind) for kind in ('lumped_port', 'wire_port', 'passive_port')})
    elif defect == 'M5':
        original = slab_wrap.cut_probe_table
        def broken(table, layout, mesh):
            result = original(table, layout, mesh)
            if 'V' not in table.channel_names:
                return result
            from jax.sharding import NamedSharding, PartitionSpec as P
            terms = tuple((term.indices, term.channel_ids, term.weights) for term in table.components)
            ids = np.concatenate([t.channel_ids for t in table.components])
            order = np.argsort(ids, kind='stable')
            payload = (terms, order, ids[order])
            return jax.tree.map(lambda a: jax.device_put(jnp.broadcast_to(a,
                (layout.n_devices,) + np.shape(a)), NamedSharding(mesh, P('x'))), payload)
        monkeypatch.setattr(slab_wrap, 'cut_probe_table', broken)
    elif defect in ('M6', 'M8'):
        original = distributed_ports.assemble_graded_ports
        def broken(sim, grid, materials, materials_drive, conductors, **kwargs):
            result = original(sim, grid, materials, materials_drive, conductors, **kwargs)
            if defect == 'M6':
                from rfx.model.materials import with_components
                before = with_components(materials, grid, periodic=(False,) * 3)
                from rfx.sources.port_drive import PortSourceQueue
                resolve = PortSourceQueue.resolve
                monkeypatch.setattr(PortSourceQueue, 'resolve', lambda self, mats: resolve(self, before))
            else:
                from copy import copy
                active = copy(sim)
                active._ports = [p for p in sim._ports if p.excite]
                loads = original(active, grid, materials, materials_drive, conductors, **kwargs)
                result = result._replace(materials=loads.materials, materials_drive=loads.materials_drive)
            return result
        monkeypatch.setattr(distributed_ports, 'assemble_graded_ports', broken)
    else:
        raise ValueError(defect)


def _port_memory_worker(devices, ports, mode):
    """Fresh-process item-6 accounting at the first scan, same scene and drive count."""
    from dataclasses import replace
    sim = scene('C1-mid', devices)
    if not ports:
        sim._ports = [replace(p, impedance=0.) for p in sim._ports]
    grid = sim._build_nonuniform_grid()
    size = int(np.prod(grid.shape))
    eps = jnp.full(grid.shape, 1.7, jnp.float32) if mode == 'grad' else None
    rows, scan = [], jax.lax.scan
    def observe(*args, **kwargs):
        if not rows:
            per, whole, leaves = {}, [], []
            for a in jax.live_arrays():
                for shard in a.addressable_shards:
                    name = str(shard.device)
                    per[name] = per.get(name, 0) + shard.data.nbytes
                    leaves.append(dict(device=name, shape=list(shard.data.shape), bytes=shard.data.nbytes))
                    if shard.data.size >= size and a is not eps:
                        whole.append(dict(device=name, shape=list(a.shape), shard_shape=list(shard.data.shape)))
            rows.append(dict(bytes=per, whole=whole, leaves=leaves))
        return scan(*args, **kwargs)
    jax.lax.scan = observe
    def forward(e):
        result = sim.forward(n_steps=128, distributed=True, devices=jax.devices()[:devices],
            checkpoint=False, skip_preflight=True, port_s11_freqs=BINS if ports else None, eps_override=e)
        return jnp.sum(jnp.abs(result.s_params if ports else result.time_series) ** 2)
    result = jax.grad(forward)(eps) if mode == 'grad' else forward(None)
    jax.block_until_ready(result)
    assert len(rows) == 1
    return dict(devices=devices, ports=ports, mode=mode, grid=list(grid.shape), **rows[0])


@pytest.mark.parametrize('mode', ['plain', 'grad'])
def test_port_first_scan_memory(mode):
    import subprocess
    import sys
    devices = len(jax.devices())
    rows = []
    for ports in (False, True):
        code = ('import json; from tests.unit.runners.test_distributed_nu_ports import _port_memory_worker; '
                f'print("PORT_MEMORY " + json.dumps(_port_memory_worker({devices}, {ports}, {mode!r})))')
        p = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=os.environ)
        assert p.returncode == 0, p.stdout + p.stderr
        rows.append(json.loads(next(s[12:] for s in p.stdout.splitlines() if s.startswith('PORT_MEMORY '))))
    from collections import Counter
    counts = [Counter((v['device'], tuple(v['shape']), v['bytes']) for v in row['leaves']) for row in rows]
    added = [dict(device=d, shape=list(shape), bytes=size, count=counts[1][d,shape,size]-counts[0][d,shape,size])
             for d,shape,size in counts[1]
             if len(shape) >= 3 and shape[-2:] == tuple(rows[1]['grid'][1:])
             and counts[1][d,shape,size] > counts[0][d,shape,size]]
    record(dict(case='memory', devices=devices, entry=mode, rows=rows, added_spatial_arrays=added,
        delta={d: rows[1]['bytes'].get(d, 0) - rows[0]['bytes'].get(d, 0) for d in rows[1]['bytes']}))
    assert not rows[1]['whole'], rows[1]['whole']
    assert not added, added



@pytest.mark.parametrize('case', ['C1-seam', 'C2', 'C3'])
def test_summed_spectrum(case):
    sim = scene(case, len(jax.devices()))
    one = evaluate(sim, 'forward', 128)
    multi = evaluate(sim, 'forward', 128, jax.devices())
    metric = measure(one.s_params, multi.s_params)
    assert metric['relative'] <= 1e-4, ('S', metric)
    for port, ((_, a), (_, b)) in enumerate(zip(one.wire_port_sparams, multi.wire_port_sparams)):
        for name, aa, bb in zip(('v_dft', 'i_dft', 'v_inc_dft', 'v_port_dft'), a, b):
            metric = measure(aa, bb)
            assert metric['relative'] <= 1e-4, (port, name, metric)


def test_checkpoint_uses_global_port_phase():
    sim = scene('C2', len(jax.devices()))
    kwargs = dict(n_steps=128, checkpoint=True, checkpoint_every=37, n_warmup=7,
                  skip_preflight=True, port_s11_freqs=BINS)
    one = sim.forward(**kwargs)
    multi = sim.forward(distributed=True, devices=jax.devices(), **kwargs)
    row = dict(case='C2-remat', devices=len(jax.devices()), S=measure(one.s_params, multi.s_params), accumulators={})
    for port, ((_, a), (_, b)) in enumerate(zip(one.wire_port_sparams, multi.wire_port_sparams)):
        for channel, (aa, bb) in enumerate(zip(a, b)):
            row['accumulators'][f'{port}/{channel}'] = measure(aa, bb)
    record(row)
    for metric in (row['S'], *row['accumulators'].values()):
        assert metric['relative'] <= 1e-4, metric


def test_removed_admission_rows_name_the_port(monkeypatch):
    from rfx.runners import _admission
    sim = scene('C1-mid', len(jax.devices()))
    for lane in ('run_distributed_nu', 'fwd_distributed_nu'):
        monkeypatch.setitem(_admission.ADMITS, lane,
            _admission.ADMITS[lane] - {('_ports', kind) for kind in ('lumped_port', 'wire_port', 'passive_port')})
    with pytest.raises(NotImplementedError, match='a driven lumped port .* is not carried'):
        evaluate(sim, 'forward', 8, jax.devices())


def test_ringdown_request_uses_shared_graded_result():
    from types import SimpleNamespace
    from rfx.measurement.ringdown_port_s import _assemble_core
    result = evaluate(scene('C2', 2), 'forward', 128)
    pairs = result.wire_port_sparams
    pms = [SimpleNamespace(meta=meta, owner=SimpleNamespace(kind=record.kind))
           for (meta, _), record in zip(pairs, result.sparam_time_records)]
    actual = _assemble_core('graded', pms, [a[3] for _, a in pairs],
        [a[1] for _, a in pairs], BINS, result.grid.dt, v_mid=[a[0] for _, a in pairs])
    assert measure(result.s_params, actual)['bit_equal']
