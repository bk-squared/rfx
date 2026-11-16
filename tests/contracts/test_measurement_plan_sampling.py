"""One-device steps sample probes and lumped/wire ports from the attached plan.

The expected side is always the runner's own formation, called with the
arguments the runner passed to its stepper; the plan never supplies it.
"""
import sys
from dataclasses import replace
from functools import lru_cache
from unittest.mock import patch

import jax
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.measurement.projector import plan_port_owners, port_owners, port_table
from rfx.measurement.time_records import uniform_specs
from rfx.stepping.graded import graded_probe_table
from tests.contracts.path_equivalence.builders import build, DX, FREQS, point, waveform

SCENES = ('periodic', '2d', 'wire_pec', 'wire_lumped', 'no_probe')
# A graded periodic axis needs a constant profile (refused), and 2-D runs uniform.
CASES = [(scene, mesh, entry) for scene in SCENES for mesh in ('uniform', 'graded')
         for entry in ('run', 'forward') if not (scene in ('2d', 'periodic') and mesh == 'graded')]


def scene(name, graded):
    lane = 'run_nonuniform' if graded else 'run_uniform'
    if name == 'periodic':
        return build(('_periodic_axes', 'periodic'), lane, graded=graded)
    if name == '2d':
        sim = Simulation(freq_max=15e9, domain=point(10, 8, 1), dx=DX,
                         mode='2d_tmz', boundary='cpml', cpml_layers=2)
        sim.add_source(point(3.1, 3.2, 0), 'ez', waveform=waveform)
        for p in ((4.1, 2.3, 0), (6.1, 5.2, 0)):
            sim.add_probe(point(*p), 'ez')
        return sim
    if name == 'wire_pec':
        sim = build(('_ports', 'wire_port'), lane, graded=graded)
        edge = 3.9 if graded else 4
        sim.add(Box(point(edge, 2, 2), point(6, edge, 3)), material='pec')
        return sim
    if name == 'wire_lumped':
        # Plan order (as declared: lumped, wire) differs from spec order (wires first).
        # One driven port: S from one record needs an isolated drive.
        sim = build(('_ports', 'passive_port'), lane, graded=graded)
        sim.add_port(point(7, 4, 2), 'ez', impedance=50., extent=2*DX, waveform=waveform)
        return sim
    sim = build(('_ports', 'lumped_port'), lane, graded=graded)
    sim._probes = []
    return sim


def call(sim, entry, n_steps=12):
    ports = [p for p in sim._ports if p.impedance]
    if entry == 'run':
        # The S scan refuses a mixed lumped + wire set; the main scan still records both.
        mixed = len({p.extent is None for p in ports}) > 1
        return sim.run(n_steps=n_steps, skip_preflight=True, compute_s_params=bool(ports) and not mixed,
                       **({'s_param_freqs': FREQS} if ports else {}))
    return sim.forward(n_steps=n_steps, checkpoint=False, skip_preflight=True,
                       # A plain source fires in every port drive: no S request then.
                       **({'port_s11_freqs': FREQS} if any(p.excite for p in ports)
                          and len(ports) == len(sim._ports) else {}))


def capture(sim, entry, graded, edit=None, n_steps=12):
    """Run one entry; keep the plan, the stepper call and what the step read."""
    from rfx.stepping import prepared, uniform, graded_loop
    seen = dict(calls=[], invariants=[], setups=[])
    advance, make_step, build_scan = prepared.advance, uniform.make_uniform_step, graded_loop._build_nu_scan

    def spy(product, stepper, *args, **kwargs):
        solve = product._prepared_solve
        if edit is not None:
            solve.measurement = edit(solve.measurement)
        seen['plan'] = solve.measurement

        def stepped(*a, **k):
            seen['calls'].append((stepper, a, k))
            return stepper(*a, **k)
        return advance(product, stepped, *args, **kwargs)

    def step(ctx, invariants, **kwargs):
        seen['invariants'].append(invariants)
        return make_step(ctx, invariants, **kwargs)

    def scan(*args, **kwargs):
        result = build_scan(*args, **kwargs)
        seen['setups'].append((result, kwargs))
        return result

    with patch.object(prepared, 'advance', spy), patch.object(uniform, 'make_uniform_step', step), \
            patch.object(graded_loop, '_build_nu_scan', scan):
        call(sim, entry, n_steps)
    assert seen['calls'] and all(k['measurement'] is seen['plan'] for _, _, k in seen['calls'])
    return seen


@lru_cache(maxsize=None)
def case(name, mesh, entry):
    graded = mesh == 'graded'
    seen = capture(scene(name, graded), entry, graded)
    # Today's formation: the same stepper and arguments, with no plan.
    pairs = [(stepper(*args, **kwargs),
              stepper(*args, **{k: v for k, v in kwargs.items() if k != 'measurement'}))
             for stepper, args, kwargs in seen['calls']]
    return seen, pairs


def same_table(actual, expected):
    """Integer indices equal, weights bitwise, channel names, order and segment ids."""
    assert [t.component for t in actual.components] == [t.component for t in expected.components]
    for a, b in zip(actual.components, expected.components, strict=True):
        for x, y in zip((*a.indices, a.channel_ids), (*b.indices, b.channel_ids), strict=True):
            assert x.dtype == y.dtype and np.array_equal(x, y)
        wa, wb = np.asarray(a.weights), np.asarray(b.weights)
        assert wa.dtype == wb.dtype and wa.shape == wb.shape and wa.tobytes() == wb.tobytes()
    assert actual.n_channels == expected.n_channels
    assert actual.channel_names == expected.channel_names
    for x, y in ((actual.order, expected.order), (actual.segment_ids, expected.segment_ids)):
        assert x.dtype == y.dtype and np.array_equal(x, y)


def same_owners(actual, expected):
    assert [(o.id, o.kind, tuple(c.name for c in o.channels)) for o in actual] == \
        [(o.id, o.kind, tuple(c.name for c in o.channels)) for o in expected]
    for a, b in zip(actual, expected, strict=True):
        for ca, cb in zip(a.channels, b.channels, strict=True):
            assert [(n.component, n.i, n.j, n.k) for n in ca.nodes] == \
                [(n.component, n.i, n.j, n.k) for n in cb.nodes]
            assert np.asarray([n.weight for n in ca.nodes]).tobytes() == \
                np.asarray([n.weight for n in cb.nodes]).tobytes()


def direct_nodes(table):
    """The direct-index list a one-node, weight-one table stands for, in record order."""
    nodes = [None] * table.n_channels
    for t in table.components:
        for channel, i, j, k, w in zip(t.channel_ids, *t.indices, np.asarray(t.weights)):
            assert w == 1.0 and nodes[channel] is None
            nodes[channel] = (int(i), int(j), int(k), t.component)
    return nodes


def formations(seen, mesh):
    """Each step the entry built: (invariants, ctx, dtype, specs, probes, selected owners)."""
    plan = seen['plan']
    if mesh == 'uniform':
        dtype = np.dtype(seen['calls'][0][2].get('field_dtype') or np.float32)
        for invariants in seen['invariants']:
            ctx = invariants['ctx']
            specs = uniform_specs(ctx)
            yield invariants, ctx, dtype, specs, list(ctx.prb_meta), plan_port_owners(plan, specs)
    else:
        for setup, kwargs in seen['setups']:
            ctx = setup.step_fn_inv.context
            yield (setup.invariants, ctx, setup.carry_init['fdtd'].ex.dtype, ctx.port_record_specs,
                   list(kwargs['probes']), setup.step_fn_inv.port_owners)


@pytest.mark.parametrize('name,mesh,entry', CASES)
def test_b2_tables_from_plan(name, mesh, entry):
    seen, _ = case(name, mesh, entry)
    plan = seen['plan']
    steps = list(formations(seen, mesh))
    assert len(steps) == len(seen['calls'])
    nu = mesh == 'graded'
    for invariants, ctx, dtype, specs, probes, selected in steps:
        grid = ctx.grid
        if probes:
            same_table(invariants['probe_table'], graded_probe_table(probes, grid.shape, dtype=dtype))
            if not nu:
                assert direct_nodes(invariants['probe_table']) == [tuple(p) for p in probes]
        else:
            assert nu or 'probe_table' not in invariants
        if specs:
            with jax.ensure_compile_time_eval():
                expected = port_table(grid, specs, nu=nu, boundary=ctx.curl_boundary, dtype=dtype)
                owners = port_owners(grid, specs, nu=nu, boundary=ctx.curl_boundary)
            same_table(invariants['port_table'], expected)
            same_owners(selected, owners)
        else:
            assert 'port_table' not in invariants
        if name in ('wire_lumped', 'no_probe', 'wire_pec'):
            assert specs
        if name == 'wire_lumped':
            assert [s[2] for s in specs] == ['wire', 'lumped']
            assert [o.kind for o in plan.owners if o.kind in ('wire', 'lumped')] == ['lumped', 'wire']
        if name == 'no_probe':
            ids = [o.id for o in plan.owners if o.kind == 'probe']
            # Only the uniform forward() adds probes at port positions.
            assert ids == (['probe:port0'] if (entry, mesh) == ('forward', 'uniform') else [])
            assert len(probes) == len(ids)


def signed_zero_count(actual, expected):
    """Bits equal, or the reference holds -0.0 where S1 holds +0.0; return that count."""
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.dtype == b.dtype == np.float32 and a.shape == b.shape
    ua, ub = a.view(np.uint32), b.view(np.uint32)
    differ = ua != ub
    zero = differ & (ub == np.uint32(0x80000000)) & (ua == np.uint32(0))
    assert not np.any(differ & ~zero), f'{int(np.count_nonzero(differ & ~zero))} samples differ in bits'
    return int(np.count_nonzero(zero))


def bits(tree):
    return [np.asarray(x) for x in jax.tree.leaves(tree) if isinstance(x, (jax.Array, np.ndarray))]


@pytest.mark.parametrize('name,mesh,entry', CASES)
def test_b1_records_bitwise(name, mesh, entry):
    _, pairs = case(name, mesh, entry)
    for s1, reference in pairs:
        compare_records(name, mesh, entry, s1, reference)


def compare_records(name, mesh, entry, s1, reference):
    if mesh == 'uniform':
        series, ref_series = s1.time_series, reference.time_series
        rest, ref_rest = s1._replace(time_series=None), reference._replace(time_series=None)
    else:
        series, ref_series = s1['time_series'], reference['time_series']
        rest = {k: v for k, v in s1.items() if k != 'time_series'}
        ref_rest = {k: v for k, v in reference.items() if k != 'time_series'}
    series, ref_series = np.asarray(series), np.asarray(ref_series)
    count = signed_zero_count(series, ref_series)
    if mesh == 'graded':
        assert count == 0
    print(f'B1 {name}/{mesh}/{entry}: probe samples {series.size}, -0.0 -> +0.0: {count}',
          file=sys.stderr)
    leaves, ref_leaves = bits(rest), bits(ref_rest)
    assert len(leaves) == len(ref_leaves)
    for x, y in zip(leaves, ref_leaves):
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


@pytest.mark.parametrize('actual,expected,count', [
    (np.float32([0.]), np.float32([-0.]), 1), (np.float32([1., -2.]), np.float32([1., -2.]), 0)])
def test_b1_admits_only_negative_zero(actual, expected, count):
    assert signed_zero_count(actual, expected) == count


@pytest.mark.parametrize('actual,expected', [
    (np.float32([-0.]), np.float32([0.])),
    (np.float32([np.nextafter(np.float32(1.), np.float32(2.))]), np.float32([1.])),
    (np.float32([0.]), np.float32([1e-45]))])
def test_b1_bit_comparison_is_live(actual, expected):
    with pytest.raises(AssertionError):
        signed_zero_count(actual, expected)


def test_b2_table_comparison_is_live():
    table = graded_probe_table([(3, 2, 2, 'ez'), (4, 2, 2, 'hx')], (8, 6, 6), dtype=np.float32)
    terms = table.components[0]
    moved = terms._replace(indices=(terms.indices[0]+1, *terms.indices[1:]))
    nudged = terms._replace(weights=np.nextafter(terms.weights, np.float32(2.)))
    for other in (table._replace(components=(moved, *table.components[1:])),
                  table._replace(components=(nudged, *table.components[1:])),
                  table._replace(channel_names=('hx', 'ez')),
                  table._replace(order=table.order[::-1].copy())):
        with pytest.raises(AssertionError):
            same_table(other, table)


def moved(plan, factor=3.0):
    """probe:1 one cell over in x; the first port owner's V weight times ``factor``."""
    owners, done = [], False
    for owner in plan.owners:
        if owner.id == 'probe:1':
            (channel,) = owner.channels
            (node,) = channel.nodes
            owner = replace(owner, channels=(replace(channel, nodes=(replace(node, i=node.i+1),)),))
        elif owner.kind in ('wire', 'lumped') and not done:
            v = owner.channels[0]
            assert v.name == 'V' and len(v.nodes) == 1
            owner = replace(owner, channels=(replace(v, nodes=(replace(v.nodes[0], weight=v.nodes[0].weight*factor),)),
                                             *owner.channels[1:]))
            done = True
        owners.append(owner)
    assert done
    return replace(plan, owners=tuple(owners))


def port_record(result, mesh):
    return result.sparam_time_records if mesh == 'uniform' else result['sparam_time_records']


@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
def test_b3_step_reads_plan(mesh):
    graded = mesh == 'graded'
    sim = scene('wire_pec', graded)
    seen = capture(sim, 'run', graded, edit=moved, n_steps=60)
    plan = seen['plan']
    # probe:0 sits inside this scene's PEC block; probe:1 sees the field.
    probe = next(o for o in plan.owners if o.id == 'probe:1').channels[0].nodes[0]
    port = next(o for o in plan.owners if o.kind in ('wire', 'lumped'))
    v = port.channels[0].nodes[0]
    stepper, args, kwargs = seen['calls'][0]
    s1 = stepper(*args, **kwargs)
    # Captured fields: the same stepper with no plan and two more direct probes.
    extra = [(probe.i, probe.j, probe.k, probe.component), (v.i, v.j, v.k, v.component)]
    if graded:
        probes = list(kwargs['probes']) + extra
    else:
        from rfx.simulation import ProbeSpec
        probes = list(kwargs['probes']) + [ProbeSpec(i=i, j=j, k=k, component=c) for i, j, k, c in extra]
    fields = stepper(*args, **{**{k: x for k, x in kwargs.items() if k != 'measurement'}, 'probes': probes})
    series = np.asarray(s1.time_series if not graded else s1['time_series'])
    captured = np.asarray(fields.time_series if not graded else fields['time_series'])
    n = len(kwargs['probes'])
    np.testing.assert_array_equal(series[:, 1], np.float32(probe.weight) * captured[:, n])
    np.testing.assert_array_equal(series[:, [0, *range(2, n)]], captured[:, [0, *range(2, n)]])
    record = next(r for r in port_record(s1, mesh) if r.id == port.id)
    np.testing.assert_array_equal(np.asarray(record['V']), np.float32(v.weight) * captured[:, n+1])
    assert np.any(captured[:, n+1] != 0) and np.any(captured[:, n] != captured[:, 1])
    # Both edits are visible: today's formation reads other values.
    unedited = next(r for r in port_record(fields, mesh) if r.id == port.id)
    assert np.any(np.asarray(unedited['V']) != np.asarray(record['V']))
