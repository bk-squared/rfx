"""Independent monitor and pre-injection reference sampling judges."""
import ast
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from tests.contracts.measurement_oracles import COMPONENTS, assignment, jacobian_nodes, tree


def coefficients(channel):
    values = {}
    for n in channel.nodes:
        key = n.component, n.i, n.j, n.k
        values[key] = values.get(key, 0.)+n.weight
    return {k: v for k, v in values.items() if v != 0}


def judge_v_ref(owner, spec, grid):
    from rfx.probes.probes import _port_voltage_value
    channel = next(c for c in owner.channels if c.name == 'V_ref')
    wire = owner.kind == 'wire'
    name = 'v_ref' if wire else 'v_ref_l'
    idx = tuple(getattr(spec, ('mid_' if wire else '')+a) for a in 'ijk')
    def sample(fields):
        env = dict(st=fields, frame=SimpleNamespace(st=fields),
                   dx=grid.dx, wp_meta=spec, lp_meta=spec,
                   _port_voltage_value=_port_voltage_value)
        env.update(zip(('mi', 'mj', 'mk') if wire else ('li', 'lj', 'lk'), idx))
        return eval(assignment(False, name), env)
    assert coefficients(channel) == jacobian_nodes(sample, grid.shape)
    statements = list(ast.walk(tree(False)))
    reference = next(n for n in statements if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    containing = [n for n in statements if isinstance(n, ast.FunctionDef)
                  and any(child is reference for child in ast.walk(n))]
    body = min(containing, key=lambda n: n.end_lineno-n.lineno)
    sampling_order = reference.lineno
    injection_body = body
    injection_order = None
    if body.name == 'before_sources':
        # The extracted lane declares execution order and attaches functions
        # separately. Follow both mappings; keep the inline-body path below.
        builder = next(n for n in containing if n.name == 'make_uniform_step')
        mappings = {t.id: n.value for n in ast.walk(builder) if isinstance(n, ast.Assign)
                    for t in n.targets if isinstance(t, ast.Name)
                    and t.id in ('kernels', 'attachments') and isinstance(n.value, ast.Dict)}
        def entry(mapping, key):
            return next(v for k, v in zip(mapping.keys, mapping.values)
                        if isinstance(k, ast.Attribute) and k.attr == key)
        attached = entry(mappings['attachments'], 'BEFORE_SOURCES')
        assert any(isinstance(n, ast.Name) and n.id == body.name for n in ast.walk(attached))
        source = entry(mappings['kernels'], 'SOURCES')
        assert isinstance(source, ast.Name)
        injection_body = next(n for n in builder.body if isinstance(n, ast.FunctionDef)
                              and n.name == source.id)
        declared = next(n.value for n in statements if isinstance(n, ast.AnnAssign)
                        and isinstance(n.target, ast.Name) and n.target.id == 'STEP_SEQUENCE')
        phases = [n.attr for n in declared.elts]
        sampling_order = phases.index('BEFORE_SOURCES')
        injection_order = phases.index('SOURCES')
    aliases = {'inject_drives'}
    for node in ast.walk(injection_body):
        if isinstance(node, ast.Assign) and any(
                isinstance(n, ast.Attribute) and n.attr == 'inject_drives'
                for n in ast.walk(node.value)):
            aliases.update(t.id for t in node.targets if isinstance(t, ast.Name))
    injections = [n for n in ast.walk(injection_body) if isinstance(n, ast.Call)
                  and ((isinstance(n.func, ast.Name) and n.func.id in aliases)
                       or (isinstance(n.func, ast.Attribute) and n.func.attr == 'inject_drives'))
                  and len(n.args) > 1 and isinstance(n.args[1], ast.Name)
                  and n.args[1].id == 'drives']
    assert len(injections) == 1
    assert sampling_order < (injections[0].lineno if injection_order is None else injection_order)
    assert channel.sample_stage == 'pre-injection'


def judge_monitors(plan, actual):
    from rfx.farfield import accumulate_ntff, init_ntff_data
    from rfx.current_moments import (
        slab_e_snapshot, accumulate_current_moments, init_current_moment_data,
    )
    shape = actual['grid'].shape
    expected = set()
    if actual.get('ntff') is not None or actual.get('ntff_box') is not None:
        expected.add('ntff_box:0')
    if actual.get('current_moments') is not None:
        expected.add('current_moment:0')
    assert expected <= {o.id for o in plan.owners}
    for owner in plan.owners:
        if owner.kind == 'ntff':
            box = actual.get('ntff') or actual.get('ntff_box')
            assert box is not None
            expected.add('ntff_box:0')
            assert owner.frequencies == tuple(np.asarray(box.freqs))
            # Zero frequencies remove only phase, leaving the production collocation.
            zero_box = box._replace(freqs=jnp.zeros_like(box.freqs))
            initial = init_ntff_data(zero_box)
            for channel in owner.channels:
                face, component = channel.name.split(':')
                axis = 'xyz'.index(face[0])
                names = tuple(k+'xyz'[a] for k in ('e', 'h') for a in range(3) if a != axis)
                slot = names.index(component)
                def functional(fields):
                    result = accumulate_ntff(initial, fields, zero_box, 1., jnp.int32(3))
                    return jnp.sum(getattr(result, face)[0, :, :, slot].real)
                oracle = jacobian_nodes(functional, shape)
                got = coefficients(channel)
                assert got.keys() == oracle.keys()
                np.testing.assert_allclose(list(got.values()), [oracle[k] for k in got], rtol=2e-7, atol=0)
            # Actual nonzero-frequency accumulator validates both physical stamps.
            fields = SimpleNamespace(**{c: jnp.ones(shape) for c in COMPONENTS})
            result = accumulate_ntff(init_ntff_data(box), fields, box, actual['grid'].dt, jnp.int32(3))
            dt = actual['grid'].dt
            for c in owner.channels[:4]:
                face, component = c.name.split(':')
                slot = ('ey', 'ez', 'hy', 'hz').index(component)
                offset = c.e_time_offset if c.kind == 'E' else c.h_time_offset
                time = plan.time_base.time(3, c.kind)+offset*dt
                phase = np.exp(-2j*np.pi*np.asarray(box.freqs)*time)
                np.testing.assert_allclose(np.asarray(getattr(result, face)[:, 0, 0, slot])/dt, phase, rtol=3e-6)
        if owner.kind == 'current_moment':
            m = actual['current_moments']
            assert m is not None
            expected.add('current_moment:0')
            assert owner.frequencies == tuple(np.asarray(m.freqs))
            fields = SimpleNamespace(**{c: jnp.asarray(np.random.default_rng(i+7).normal(size=shape))
                                        for i, c in enumerate(COMPONENTS)})
            previous = SimpleNamespace(**{c: jnp.asarray(np.random.default_rng(i+77).normal(size=shape))
                                          for i, c in enumerate(COMPONENTS)})
            host_fields = SimpleNamespace(**{c: np.asarray(getattr(fields, c)) for c in COMPONENTS})
            host_previous = SimpleNamespace(**{c: np.asarray(getattr(previous, c)) for c in COMPONENTS})
            values = np.zeros((m.n_blocks, 3, m.n_weights))
            bounds = np.zeros_like(values)
            for channel in owner.channels:
                _, b, _, c, _, w, stage = channel.name.split(':')
                index = int(b), int(c), int(w)
                state = host_previous if stage == 'E_prev' else host_fields
                assert channel.sample_stage == ('pre-update' if stage == 'E_prev' else 'post-injection')
                terms = [n.weight*float(getattr(state, n.component)[n.i, n.j, n.k]) for n in channel.nodes]
                values[index] += sum(terms)
                bounds[index] += sum(abs(t) for t in terms)*np.finfo(np.float32).eps*max(len(terms), 1)
            dt = actual['grid'].dt
            raw, _ = accumulate_current_moments(init_current_moment_data(m), fields,
                                                slab_e_snapshot(previous, m), m, dt, jnp.int32(3))
            phase = np.exp(-2j*np.pi*np.asarray(m.freqs)*(3+m.half_step)*dt)*dt
            oracle = np.asarray(raw)/phase[:, None, None, None]
            assert np.all(abs(oracle-values) <= bounds+abs(values)*2e-6)
            # Each production weight source and block membership is checked, too.
            for channel in owner.channels[::3]:
                _, b, _, c, _, w, _ = channel.name.split(':')
                weights = np.asarray((m.w_ex, m.w_ey, m.w_ez)[int(c)])[int(w)]
                seg = np.asarray(m.seg).reshape(weights.shape[:2])
                from rfx.core.yee import EPS_0
                coef = np.float32(EPS_0)/np.float32(dt)
                expected_nodes = {('e'+'xyz'[int(c)], m.i_lo+i, m.j_lo+j, m.k_lo+k): float(-weights[i,j,k]*coef)
                                  for i,j,k in np.ndindex(weights.shape) if seg[i,j] == int(b) and weights[i,j,k] != 0}
                assert coefficients(channel) == expected_nodes
    return expected
