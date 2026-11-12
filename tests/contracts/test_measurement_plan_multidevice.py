"""Plan-read counter and seam instruments on the existing slab pipelines."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx._internal._grid_metric import field_index
from rfx.measurement import prepared
from rfx.stepping.slab import Slab
from tests.contracts.run_forward_census import (
    BUILDERS, PROBE, FrozenArray, base_scene, capture, compare, equal_request_pair, freeze,
)

PATHS = ('multi-device-uniform', 'multi-device-graded')
KINDS = ('probe', 'lumped_wire_vi', 'dft_plane', 'flux', 'ntff', 'current_moments')
OWNER_KINDS = dict(probe=('probe',), lumped_wire_vi=('lumped', 'wire'),
                   dft_plane=('dft_plane',), flux=('flux',), ntff=('ntff',),
                   current_moments=('current_moment',))
FLOOR = 2.0 ** -100
BASELINE = Path(__file__).resolve().parents[2] / 'scripts/ci/plan_read_baseline.json'


@pytest.fixture
def report(capsys):
    def emit(bar, **values):
        with capsys.disabled():
            print(bar + ' ' + json.dumps(values, sort_keys=True), file=sys.stderr, flush=True)
    return emit


def devices(n):
    found = jax.devices('cpu')
    if len(found) < n:
        pytest.skip(f'needs XLA_FLAGS=--xla_force_host_platform_device_count={n}')
    return found[:n]


def scene(builder, path):
    sim = builder(path)
    grid = sim._build_nonuniform_grid() if path.endswith('graded') else sim._build_grid()
    census = tuple(int(v) for v in field_index(grid, PROBE, 'ez'))
    probes = [('census', 'ez', census, PROBE)]
    for n in (2, 3):
        width = Slab(grid.shape[0], n).nx_per_rank
        for seam in range(1, n):
            for side, i in (('left', seam * width - 1), ('right', seam * width)):
                probes.append((f'n{n}-seam{seam}-{side}', 'ez', (i, *census[1:]), None))
    width = Slab(grid.shape[0], 3).nx_per_rank
    for rank in range(3):
        probes.append((f'n3-interior{rank}', 'hy', (rank * width + width // 2, *census[1:]), None))
    for col, (name, component, node, _) in enumerate(probes[1:], 1):
        position = tuple(float(grid.node_of(axis, index)) for axis, index in enumerate(node))
        realized = tuple(int(v) for v in field_index(grid, position, component))
        assert realized == node, (path, name, position, node, realized)
        sim.add_probe(position, component)
        probes[col] = (name, component, realized, position)
    return sim, grid, probes


def columns(result, kind):
    if kind == 'probe':
        return {'time_series': np.asarray(result.time_series)}
    if kind == 'lumped_wire_vi':
        return {f'sparam_time_records/{i}': np.asarray(a.samples)
                for i, a in enumerate(result.sparam_time_records)}
    if kind == 'dft_plane':
        return {f'dft_planes/{name}/accumulator': np.asarray(p.accumulator)
                for name, p in result.dft_planes.items()}
    if kind == 'flux':
        return {f'flux_monitors/{name}/{channel}': np.asarray(getattr(m, channel))
                for name, m in result.flux_monitors.items()
                for channel in ('e1_dft', 'e2_dft', 'h1_dft', 'h2_dft')}
    value = result.ntff_data if kind == 'ntff' else result.current_moment_data
    return {f'{kind}/{i}': np.asarray(a) for i, a in enumerate(jax.tree.leaves(value))}


def below_floor_runs(plain, changed, mask):
    """Lossless runs of excluded flat indices and both values' bytes."""
    runs = []
    for index in np.flatnonzero(~mask.ravel()):
        a, b = plain.ravel()[index].tobytes().hex(), changed.ravel()[index].tobytes().hex()
        if runs and runs[-1][1] == int(index) and runs[-1][2:] == [a, b]:
            runs[-1][1] += 1
        else:
            runs.append([int(index), int(index) + 1, a, b])
    return runs


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('kind', [
    pytest.param(kind, marks=pytest.mark.slow) if kind not in KINDS[:2] else kind
    for kind in KINDS
])
def test_measurement_plan_multidevice_counter(path, kind, report):
    baseline = json.loads(BASELINE.read_text())
    assert set(baseline) == set(PATHS)
    assert all(set(kinds) <= set(KINDS) for kinds in baseline.values())
    selected = ('P-lumped', 'P-wire-driven') if kind in KINDS[:2] else ({
        'dft_plane': 'O-dft', 'flux': 'O-flux', 'ntff': 'O-ntff', 'current_moments': 'O-cm',
    }[kind],)
    counted = []
    for declaration in selected:
        builder = BUILDERS[declaration]
        plain_sim = scene(builder, path)[0] if kind in KINDS[:2] else builder(path)
        plain = plain_sim.forward(distributed=True, devices=devices(2), n_steps=60, skip_preflight=True)
        patched_nodes = []
        original = prepared.build_measurement_plan

        def scaled(*args, **kwargs):
            plan = original(*args, **kwargs)
            owners = []
            for owner in plan.owners:
                if owner.kind in OWNER_KINDS[kind]:
                    patched_nodes.extend(node for channel in owner.channels for node in channel.nodes)
                    owner = replace(owner, channels=tuple(replace(channel, nodes=tuple(
                        replace(node, weight=2.0 * node.weight) for node in channel.nodes))
                        for channel in owner.channels))
                owners.append(owner)
            return replace(plan, owners=tuple(owners))

        changed_sim = scene(builder, path)[0] if kind in KINDS[:2] else builder(path)
        with patch.object(prepared, 'build_measurement_plan', scaled):
            changed = changed_sim.forward(distributed=True, devices=devices(2), n_steps=60, skip_preflight=True)
        assert patched_nodes, (path, kind, 'no plan node weights patched')
        before, after = columns(plain, kind), columns(changed, kind)
        assert before and before.keys() == after.keys(), (path, kind, 'missing record columns')
        a, b = True, False
        for name, values in before.items():
            altered = after[name]
            assert values.shape == altered.shape and values.dtype == altered.dtype
            mask = np.abs(values) >= FLOOR
            a &= altered[mask].tobytes() == (values[mask] * 2.0).tobytes()
            b |= bool(np.any(mask))
            report('R3-floor', path=path, kind=kind, scene=declaration, column=name,
                   dtype=values.dtype.str, shape=list(values.shape),
                   excluded_runs=below_floor_runs(values, altered, mask))
        other = 'lumped_wire_vi' if kind == 'probe' else 'probe'
        c = compare(freeze(columns(plain, other)), freeze(columns(changed, other))) == []
        counted.append(bool(a and b and c))
        report('R3', path=path, kind=kind, scene=declaration, a=bool(a), b=bool(b), c=c,
               counted=counted[-1], patched_nodes=len(patched_nodes))
    measured = all(counted)
    report('R3-cell', path=path, kind=kind, counted=measured)
    assert measured == (kind in baseline[path]), (path, kind, measured, baseline[path])


def test_measurement_plan_multidevice_legacy():
    pytest.skip('multi-device-graded legacy slab loop: 0 kinds; runner-owned sampler; report only')


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('n', (2, 3))
def test_measurement_plan_multidevice_seam(path, n, report):
    selected_devices = devices(n)
    sim, grid, probes = scene(base_scene, path)
    report('Q', path=path, n=n, nx=grid.shape[0], nx_per_rank=Slab(grid.shape[0], n).nx_per_rank,
           probes=[dict(name=name, component=component, node=node, position=position)
                   for name, component, node, position in probes])
    result = sim.run(devices=selected_devices, n_steps=60, skip_preflight=True)
    rows = []
    for column, (name, component, node, _) in enumerate(probes):
        if component != 'ez' or name == 'census':
            continue
        value = np.asarray(result.state.ez)[node]
        recorded = np.asarray(result.time_series)[-1, column]
        equal = value.dtype == recorded.dtype and value.tobytes() == recorded.tobytes()
        nonvacuous = bool(abs(value) >= FLOOR)
        report('R5', path=path, n=n, probe=name, node=node, value=float(value),
               recorded=float(recorded), value_bytes=value.tobytes().hex(),
               recorded_bytes=recorded.tobytes().hex(), equal=equal, nonvacuous=nonvacuous)
        rows.append((name, node, float(value), float(recorded), equal, nonvacuous))
    assert all(row[-1] for row in rows), ('the scene puts no field on the seam node at step 60', rows)
    assert all(row[-2] for row in rows), (path, n, rows)


@pytest.mark.slow
def test_measurement_plan_multidevice_graded_gradient(report):
    sim, grid, _ = scene(BUILDERS['P-wire-driven'], 'multi-device-graded')
    eps = jnp.full(grid.shape, 1.3, jnp.float32)
    selected_devices = devices(2)

    def objective(e):
        result = sim.forward(eps_override=e, distributed=True, devices=selected_devices,
                             n_steps=300, skip_preflight=True, checkpoint=False,
                             port_s11_freqs=np.linspace(2e9, 18e9, 9))
        assert result.s_params is not None
        return jnp.sum(result.time_series ** 2) + jnp.sum(jnp.abs(result.s_params) ** 2)

    gradient = np.asarray(jax.grad(objective)(eps))
    finite, nonzero = bool(np.isfinite(gradient).all()), bool(np.any(gradient != 0))
    peak = float(np.max(np.abs(gradient)))
    report('N2', path='multi-device-graded', scene='W', n=2, shape=list(gradient.shape),
           finite=finite, nonzero=nonzero, max_abs=peak)
    assert finite and nonzero, ('N2 gradient finite and nonzero', finite, nonzero, peak)


@pytest.mark.parametrize('declaration,direction', [
    ('B-cpml', None), ('P-lumped', 'a'), ('P-lumped', 'b'),
    ('P-wire-driven', 'a'), ('P-wire-driven', 'b'), ('P-wire', 'b'),
])
def test_measurement_plan_multidevice_assembly_tables(declaration, direction, report):
    path = 'multi-device-uniform'
    pair = (equal_request_pair(declaration, path, direction, level='b') if direction else
            {entry: capture(BUILDERS[declaration], entry, path, 'b') for entry in ('run', 'forward')})

    def leaves(value, prefix=''):
        if isinstance(value, FrozenArray):
            yield prefix
        elif isinstance(value, dict):
            for key, child in value.items():
                yield from leaves(child, f'{prefix}/{key}')
        elif isinstance(value, (tuple, list)):
            for i, child in enumerate(value):
                yield from leaves(child, f'{prefix}/{i}')

    for entry, captured in pair.items():
        assert captured['outcome'] == 'captured', captured
        paths = list(leaves(captured['value']))
        tables = {kind: [leaf for leaf in paths if f'/{kind}/' in leaf]
                  for kind in ('probe_table', 'port_table')}
        report('R4', path=path, scene=declaration, direction=direction, entry=entry, **tables)
        assert tables['probe_table'], (declaration, entry, paths)
        if direction:
            assert tables['port_table'], (declaration, entry, paths)


R2_SCENES = {'A': base_scene, 'L': BUILDERS['P-lumped'], 'W': BUILDERS['P-wire-driven']}


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('n', (2, 3))
@pytest.mark.parametrize('entry', ('run', 'forward'))
@pytest.mark.parametrize('name', tuple(R2_SCENES))
def test_measurement_plan_multidevice_plan_source(name, entry, n, path, report):
    """The cut tables and record owners are the attached plan's, compiled here."""
    from rfx.measurement.projector import sample_table
    from rfx.runners import _distributed_nu_wrapped as wrapped
    from rfx.stepping import prepared as prepared_step, slab_wrap, uniform_slab_ports
    selected_devices = devices(n)
    sim = scene(R2_SCENES[name], path)[0]
    plans, cut_tables, port_plans, setups = [], [], [], []
    advance, cut, build = prepared_step.advance, slab_wrap.cut_probe_table, wrapped.build_setup
    port_plan_type = uniform_slab_ports.UniformPortPlan

    def attached(product, *args, **kwargs):
        plans.append(product._prepared_solve.measurement)
        return advance(product, *args, **kwargs)

    def cut_spy(table, *args, **kwargs):
        cut_tables.append(table)
        return cut(table, *args, **kwargs)

    def build_spy(*args, **kwargs):
        setups.append((kwargs.get('port_plan'), build(*args, **kwargs)))
        return setups[-1][1]

    class PortPlanSpy(port_plan_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            port_plans.append(self)

    kwargs = dict(devices=selected_devices, n_steps=300, skip_preflight=True)
    with patch.object(prepared_step, 'advance', attached), \
            patch.object(slab_wrap, 'cut_probe_table', cut_spy), \
            patch.object(wrapped, 'build_setup', build_spy), \
            patch.object(uniform_slab_ports, 'UniformPortPlan', PortPlanSpy):
        sim.run(**kwargs) if entry == 'run' else sim.forward(distributed=True, **kwargs)
    assert len(plans) == 1 and plans[0] is not None, (path, name, entry, n, len(plans))
    plan = plans[0]
    if path == 'multi-device-uniform':
        assert len(port_plans) == 1 and not setups, (len(port_plans), len(setups))
        specs, owners_in_use = port_plans[0].specs, port_plans[0].owners
    else:
        assert len(setups) == 1 and not port_plans, (len(setups), len(port_plans))
        runner_plan, setup = setups[0]
        specs = () if runner_plan is None else runner_plan.specs
        owners_in_use = setup.step_fn_inv.port_owners
    by_id = {owner.id: owner for owner in plan.owners}
    port_owners = tuple(by_id[f'port:{spec[4]}:mode:0'] for spec in specs)
    expected = [sample_table(tuple(o for o in plan.owners if o.kind == 'probe'), dtype=jnp.float32)]
    if specs:
        expected.append(sample_table(tuple(replace(o, channels=tuple(
            c for c in o.channels if c.sample_stage == 'post-injection')) for o in port_owners),
            dtype=jnp.float32))
    tables = compare(freeze(cut_tables), freeze(expected))
    owners = compare(freeze(owners_in_use), freeze(port_owners))
    report('R2', path=path, n=n, scene=name, entry=entry, cut_calls=len(cut_tables),
           channels=[t.n_channels for t in cut_tables], port_owners=[o.id for o in port_owners],
           tables_equal=tables == [], owners_equal=owners == [], tables_diff=tables[:3],
           owners_diff=owners[:3])
    assert tables == [], (path, name, entry, n, tables[:3])
    assert owners == [], (path, name, entry, n, owners[:3])
