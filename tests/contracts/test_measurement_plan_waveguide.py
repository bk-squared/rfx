"""Waveguide table formation, retained-plan use, completed slots and scan cost."""
from contextlib import ExitStack
from dataclasses import replace
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.measurement import waveguide_records as compiler
from rfx.measurement.ports import waveguide_owner
from rfx.measurement.waveguide_modal import modal_voltage, modal_current, update_waveguide_port_probe
from tests.contracts.path_equivalence.builders import build
from tests.contracts.test_measurement_plan import setup
from tests.contracts.test_measurement_plan_seam import exact
from tests.oracle.test_waveguide_port_validation_battery import _build_sim

FIELDS = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')
RECORDS = ('v_probe_t', 'i_probe_t', 'v_ref_t', 'i_ref_t')
N = 24


def scene(mesh, battery=False):
    if battery:
        return _build_sim(np.linspace(4.5e9, 7e9, 3), solved_domain=True, cpml_layers=10)
    return build(('_waveguide_ports', 'waveguide_port'),
                 'run_nonuniform' if mesh == 'graded' else 'run_uniform', graded=mesh == 'graded')


def legacy_factory(module, move=False):
    """Put main's sampler back at the actual in-scan call site."""
    name = 'make_uniform_step' if module.__name__.endswith('uniform') else 'make_graded_step'
    source = inspect.getsource(getattr(module, name))
    import re
    source = re.sub(r"            waveguide_values = project\(.*\)\n", '', source)
    source = re.sub(r'write_waveguide_record\(cfg, waveguide_values\[4\*port:4\*port\+4\], frame.st.step, (dt|ctx.dt)\)',
                    r'legacy(cfg, frame.st, \1, 0.)', source)
    def legacy(cfg, state, dt, dx):
        if move:
            cfg = cfg._replace(probe_x=cfg.probe_x + 1)
        return update_waveguide_port_probe(cfg, state, dt, dx)
    namespace = dict(vars(module), legacy=legacy)
    exec(compile(source, '<legacy waveguide step>', 'exec'), namespace)
    return name, namespace[name]


def recursive_equations(jaxpr):
    if hasattr(jaxpr, 'jaxpr'):
        jaxpr = jaxpr.jaxpr
    if isinstance(jaxpr, (tuple, list)):
        return sum(recursive_equations(x) for x in jaxpr)
    if isinstance(jaxpr, dict):
        return sum(recursive_equations(x) for x in jaxpr.values())
    if not hasattr(jaxpr, 'eqns'):
        return 0
    return len(jaxpr.eqns) + sum(recursive_equations(e.params) for e in jaxpr.eqns)


def capture(mesh='uniform', entry='run', *, old=False, move_old=False, edit=None,
            disable_d3=False, move_compiler=False, battery=False, count=False, n_steps=N):
    from rfx.stepping import uniform, graded, prepared, uniform_loop
    from rfx.measurement import prepared as plan_seam
    seen = SimpleNamespace(tables=[], results=[], counts=[])
    compile_table, advance, run, builder, scan = (compiler.compile_waveguide_table, prepared.advance,
                                                uniform_loop.run, plan_seam.build_measurement_plan, jax.lax.scan)
    def compile_spy(grid, cfgs, **kwargs):
        table = compile_table(grid, cfgs, **kwargs)
        seen.tables.append((grid, cfgs, kwargs.get('measurement'), table))
        return table
    def execute(stepper, *args, **kwargs):
        if 'return_state' in kwargs:
            kwargs['return_state'] = True
        result = stepper(*args, **kwargs)
        seen.results.append(result)
        return result
    def advancing(product, stepper, *args, **kwargs):
        return advance(product, lambda *a, **k: execute(stepper, *a, **k), *args, **kwargs)
    def building(*args, **kwargs):
        plan = builder(*args, **kwargs)
        return edit(plan, kwargs['grid'] if 'grid' in kwargs else args[1]) if edit else plan
    def scanning(f, init, xs=None, *args, **kwargs):
        if count and isinstance(init, dict) and 'waveguide_port_accs' in init:
            x = jax.tree.map(lambda a: a[0], xs)
            seen.counts.append(recursive_equations(jax.make_jaxpr(f)(init, x)))
        return scan(f, init, xs, *args, **kwargs)
    def moved_owner(grid, cfg, **kwargs):
        return waveguide_owner(grid, cfg._replace(probe_x=cfg.probe_x + 1), **kwargs)
    with ExitStack() as stack:
        stack.enter_context(patch.object(compiler, 'compile_waveguide_table', compile_spy))
        stack.enter_context(patch.object(prepared, 'advance', advancing))
        stack.enter_context(patch.object(plan_seam, 'build_measurement_plan', building))
        stack.enter_context(patch.object(jax.lax, 'scan', scanning))
        if old:
            for module in (uniform, graded):
                name, factory = legacy_factory(module, move_old)
                stack.enter_context(patch.object(module, name, factory))
        if disable_d3:
            stack.enter_context(patch.object(compiler, 'check_waveguide_owner', lambda *a: None))
        if move_compiler:
            stack.enter_context(patch.object(compiler, 'waveguide_owner', moved_owner))
        sim = scene(mesh, battery)
        if battery and entry != 'calculator':
            sim._waveguide_ports = sim._waveguide_ports[:1]
        if entry == 'calculator':
            stack.enter_context(patch.object(uniform_loop, 'run', lambda *a, **k: execute(run, *a, **k)))
            sim.compute_waveguide_s_matrix(n_steps=n_steps, normalize=False, skip_preflight=True)
        elif entry == 'run':
            sim.run(n_steps=n_steps, skip_preflight=True, compute_s_params=False)
        else:
            sim.forward(n_steps=n_steps, skip_preflight=True, checkpoint=False)
    assert seen.tables and seen.results
    return seen


def configs(seen):
    result = seen.results[0]
    return result['waveguide_ports'] if isinstance(result, dict) else result.waveguide_ports


def moved_plan(plan, grid):
    from rfx.measurement.plan import build_measurement_plan
    with patch('rfx.measurement.prepared.build_measurement_plan', build_measurement_plan):
        actual = setup(scene('graded' if hasattr(grid, 'dx_arr') else 'uniform'))
    cfg = actual['waveguide_ports'][0]
    moved = waveguide_owner(grid, cfg._replace(probe_x=cfg.probe_x + 1))
    return replace(plan, owners=tuple(replace(o, channels=moved.channels) if o.kind == 'waveguide' else o
                                     for o in plan.owners))


def owner_nodes(table, owners):
    """Decode actual storage; expected weights come only from owner nodes."""
    channels = [c for o in owners for c in o.channels]
    expected = {(c, n.component, n.i, n.j, n.k): np.asarray(n.weight, dtype=np.float32).tobytes()
                for c, channel in enumerate(channels) for n in channel.nodes}
    actual = {}
    for t in table.components:
        for c, p, w in zip(t.channel_ids, zip(*t.indices), t.weights):
            actual[(int(c), t.component, *map(int, p))] = np.asarray(w).tobytes()
    for box in table.boxes:
        assert sum(s.stop - s.start >= 2 for s in box.indices) == 2
        for p in np.ndindex(box.weights.shape):
            key = (box.channel_id, box.component, *(s.start + i for s, i in zip(box.indices, p)))
            assert key not in actual
            actual[key] = np.asarray(box.weights[p]).tobytes()
    assert actual == expected
    assert table.channel_names == tuple(c.name for c in channels)


@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_b2_box_terms_match_owner_nodes(mesh, entry):
    seen = capture(mesh, entry, battery=mesh == 'uniform')
    grid, cfgs, plan, table = seen.tables[0]
    assert plan is not None
    assert table.boxes and not table.components
    assert all(cfg.h_offset == (.5, .5) for cfg in cfgs)
    owners = tuple(waveguide_owner(grid, cfg) for cfg in cfgs)
    owner_nodes(table, owners)
    without = compiler.compile_waveguide_table(grid, cfgs, dtype=np.float32)
    owner_nodes(without, owners)
    for a, b in zip(table.boxes, without.boxes, strict=True):
        assert a[:3] == b[:3]
        exact(a.weights, b.weights)


def test_b2_two_port_boxes_without_plan():
    seen = capture(entry='calculator', battery=True)
    for grid, cfgs, plan, table in seen.tables:
        assert plan is None and len(cfgs) == 2
        assert table.boxes and not table.components
        owners = tuple(waveguide_owner(grid, cfg, port_id=i) for i, cfg in enumerate(cfgs))
        owner_nodes(table, owners)
        attached = compiler.compile_waveguide_table(grid, cfgs, measurement=SimpleNamespace(owners=owners),
                                                    dtype=np.float32)
        owner_nodes(attached, owners)
        with pytest.raises(ValueError, match='waveguide_port:1:mode:0'):
            compiler.compile_waveguide_table(grid, cfgs,
                measurement=SimpleNamespace(owners=owners[::-1]), dtype=np.float32)


def test_b2_zero_h_offset():
    from rfx.model.waveguide_modes import init_waveguide_port, WaveguidePort
    actual = setup(scene('uniform'))
    grid = actual['grid']
    port = WaveguidePort(x_index=3, y_slice=(0, 8), z_slice=(0, 7), a=8*float(grid.dx), b=7*float(grid.dx))
    cfg = init_waveguide_port(port, grid.dx, freqs=jnp.asarray([8e9, 10e9]),
                             dft_total_steps=3, probe_offset=2, ref_offset=1,
                             grid=grid, h_offset=(0, 0))
    table = compiler.compile_waveguide_table(grid, (cfg,), dtype=np.float32)
    assert table.boxes
    owner_nodes(table, (waveguide_owner(grid, cfg),))


def damaged(plan, grid, kind):
    owner = next(o for o in plan.owners if o.kind == 'waveguide')
    if kind == 'count':
        return replace(plan, owners=tuple(o for o in plan.owners if o is not owner))
    channel = owner.channels[0]
    node = replace(channel.nodes[0], weight=channel.nodes[0].weight + .125)
    changed = replace(owner, channels=(replace(channel, nodes=(node, *channel.nodes[1:])), *owner.channels[1:]))
    return replace(plan, owners=tuple(changed if o is owner else o for o in plan.owners))


@pytest.mark.parametrize('kind', ['weight', 'count'])
def test_b2_d3_rejects(kind):
    with pytest.raises(ValueError, match='waveguide_port:0:mode:0' if kind == 'weight' else 'waveguide.*count'):
        capture(edit=lambda p, g: damaged(p, g, kind))


@pytest.mark.parametrize('mesh,entry', [('uniform', 'run'), ('uniform', 'forward'),
                                         ('graded', 'run'), ('graded', 'forward'),
                                         ('uniform', 'calculator')])
def test_b1_fields_and_other_records_bits(mesh, entry):
    new = capture(mesh, entry, battery=mesh == 'uniform')
    old = capture(mesh, entry, old=True, battery=mesh == 'uniform')
    a, b = new.results[0], old.results[0]
    get = lambda r, name: r.get(name) if isinstance(r, dict) else getattr(r, name, None)
    for field in FIELDS:
        exact(getattr(get(a, 'state'), field), getattr(get(b, 'state'), field))
    for name in ('time_series', 'dft_planes', 'dft_time_records', 'sparam_time_records',
                 'flux_monitors', 'wire_port_sparams',
                 'lumped_port_sparams', 'ntff_data', 'current_moment_data'):
        for x, y in zip(jax.tree.leaves(get(a, name)), jax.tree.leaves(get(b, name)), strict=True):
            exact(x, y)


def record_bar(reference, grid, cfgs):
    owners = [waveguide_owner(grid, cfg) for cfg in cfgs]
    peaks = [max(float(np.max(np.abs(getattr(cfg, name)))) for cfg in reference
                 for name in RECORDS if name.startswith(family)) for family in ('v', 'i')]
    return [[max(9., 2*np.sqrt(sum(n.weight != 0 for n in channel.nodes)))
             * np.spacing(np.float32(peaks[c % 2])) for c, channel in enumerate(o.channels)] for o in owners]


@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
def test_b4_moved_plane(mesh):
    with pytest.raises(ValueError, match='waveguide_port:0:mode:0'):
        capture(mesh, edit=moved_plan)
    new = capture(mesh, edit=moved_plan, disable_d3=True)
    old = capture(mesh, old=True, move_old=True)
    unmoved = capture(mesh, old=True)
    grid, cfgs, _, _ = new.tables[0]
    bars = record_bar(configs(old), grid, cfgs)
    for a, b, c, bounds in zip(configs(new), configs(old), configs(unmoved), bars, strict=True):
        for name, bound in zip(RECORDS, bounds):
            error = np.max(np.abs(np.asarray(getattr(a, name)) - np.asarray(getattr(b, name))))
            assert error <= bound, (name, error, bound)
        difference = np.max(np.abs(np.asarray(a.v_probe_t) - np.asarray(c.v_probe_t)))
        assert difference > 100 * bounds[0], (difference, bounds[0])


def test_p2_6_calculator_uses_no_plan_table():
    base = capture(entry='calculator', battery=True, old=True)
    moved = capture(entry='calculator', battery=True, move_compiler=True)
    assert all(plan is None for _, _, plan, _ in moved.tables)
    grid, cfgs, _, _ = moved.tables[0]
    bars = record_bar(configs(base), grid, cfgs)
    difference = np.max(np.abs(np.asarray(configs(moved)[0].v_probe_t) - np.asarray(configs(base)[0].v_probe_t)))
    assert difference > 100 * bars[0][0], (difference, bars[0][0])


def test_p1_a_completed_step_slots():
    from rfx.measurement.modal import record_waveguide
    actual = setup(scene('uniform'))
    cfg, grid = actual['waveguide_ports'][0], actual['grid']
    cfg = cfg._replace(**{name: jnp.zeros(3, jnp.float32) for name in (*RECORDS, 'v_inc_t')})
    expected: list = [[] for _ in RECORDS]
    for n in range(1, 4):
        fields = SimpleNamespace(step=jnp.asarray(n), **{c: jnp.full(grid.shape, n*(i+1.), jnp.float32)
                                                       for i, c in enumerate(FIELDS)})
        for values, plane, sampler in zip(expected, (cfg.probe_x, cfg.probe_x, cfg.ref_x, cfg.ref_x),
                                          (modal_voltage, modal_current, modal_voltage, modal_current)):
            values.append(sampler(fields, cfg, plane, 0.))
        cfg = record_waveguide(cfg, fields, float(grid.dt), 0.)
    for name, values in zip(RECORDS, expected):
        exact(getattr(cfg, name), np.asarray(values))
        assert np.asarray(getattr(cfg, name))[0] != 0
        assert np.asarray(getattr(cfg, name))[-1] != 0
    assert int(cfg.n_steps_recorded) == 3


def test_b5_recursive_scan_equations():
    new = capture(entry='calculator', battery=True, count=True)
    old = capture(entry='calculator', battery=True, old=True, count=True)
    assert len(new.counts) == len(old.counts) == 2
    assert new.counts[0] == new.counts[1] and old.counts[0] == old.counts[1]
    print(f'B5 recursive scan equations: new={new.counts[0]}, main={old.counts[0]}')
    assert new.counts[0] <= old.counts[0]
