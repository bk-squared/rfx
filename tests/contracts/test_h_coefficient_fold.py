"""Uniform magnetic coefficient folding against the unchanged fallback steps."""
from dataclasses import replace
import inspect
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box
from rfx.boundaries import cpml
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.stepping import uniform, yee
from scripts.perf._census_scenes import base


@pytest.fixture(autouse=True)
def cpu_float32():
    with jax.default_device(jax.devices('cpu')[0]), jax.enable_x64(False):
        yield
    jax.clear_caches()


def scene(kind):
    boundary = (BoundarySpec(x=Boundary(lo='pmc', hi='pec'), y='pec', z='pec')
                if kind == 'ii' else None)
    sim = base(16, boundary=boundary).sim
    if kind == 'ii':
        sim.add_material('magnetic', mu_r=2.)
        sim.add(Box((.005, .005, .005), (.012, .012, .012)), material='magnetic')
    elif kind == 'iv':
        sim.add_material('magnetic', mu_r=2.)
        sim.add(Box((-.008, .005, .005), (.024, .012, .012)), material='magnetic')
    return sim


def clone(function, substitutions, namespace):
    function = inspect.unwrap(function)
    source = inspect.getsource(function)
    source = source[source.index('def '):]
    for old, new in substitutions:
        assert old in source
        source = source.replace(old, new, 1)
    scope = dict(function.__globals__, **namespace)
    exec(compile(source, '<frozen H fallback>', 'exec'), scope)
    return scope[function.__name__]


def legacy_kernels(invariants, frozen):
    """Use the original fallback bodies, substituting only their final products."""
    replacements, namespace = [], {}
    if frozen:
        replacements = [('= magnetic_update_coefficients(', '= frozen_product(')]
        namespace['frozen_product'] = lambda *args: invariants['h_coefficients']
    h = clone(yee.update_h, replacements, namespace)
    replacements, namespace = [], {}
    if frozen and 'cpml_h_coefficients' in invariants:
        replacements = [('    def _h_face(component, axis, lo):\n',
                         '    def _h_face(component, axis, lo):\n'
                         '        return frozen_faces[component, axis, lo]\n')]
        namespace['frozen_faces'] = invariants['cpml_h_coefficients'].faces
    absorb = clone(cpml.apply_cpml_h, replacements, namespace)

    def old_h(*args, **kwargs):
        kwargs['h_coefficients'] = None
        return h(*args, **kwargs)

    def old_absorb(*args, **kwargs):
        kwargs['h_coefficients'] = None
        return absorb(*args, **kwargs)

    return old_h, old_absorb


def window_reference(ctx, key):
    """The census uses eight CPML cells on each face of the physical grid."""
    _, axis, lo = key
    physical = ctx.grid.shape
    window = [slice(None)] * 3
    depth = ctx.grid.cpml_layers
    window[axis] = slice(0, depth) if lo else slice(physical[axis] - depth, physical[axis])
    return tuple(window)


def independent_coefficients(invariants):
    """NumPy float32 material multiply, divide, then ascending transverse factors."""
    ctx = invariants['ctx']
    factors = invariants['h_row_factors']
    mu = ctx.materials.components.mu_update
    scale = np.float32(4 * np.pi * 1e-7)

    def evaluate(component, window=None):
        value = np.float32(ctx.dt) / (np.asarray(mu[component], np.float32) * scale)
        if window is not None and value.ndim:
            value = value[window]
        for axis in range(3):
            if axis != component and factors[axis] is not None:
                factor = np.asarray(factors[axis], np.float32)
                if window is not None:
                    factor = factor[window[axis]]
                dims = [1, 1, 1]
                dims[axis] = -1
                value = value * factor.reshape(dims)
        return value

    arrays = [(f'yee.{i}', a, evaluate(i)) for i, a in enumerate(invariants['h_coefficients'])]
    if 'cpml_h_coefficients' in invariants:
        for key, value in invariants['cpml_h_coefficients'].faces.items():
            window = window_reference(ctx, key)
            arrays.append((f'cpml.{key}', value, evaluate(key[0], window)))
    return arrays


def ulp_counts(invariants, *, assert_reference=True):
    zero = one = largest = 0
    for name, actual, expected in independent_coefficients(invariants):
        actual = np.asarray(actual)
        assert actual.dtype == expected.dtype == np.float32
        assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
        difference = np.abs(actual.view(np.int32).astype(np.int64) - expected.view(np.int32))
        zero += int(np.count_nonzero(difference == 0))
        one += int(np.count_nonzero(difference == 1))
        largest = max(largest, int(difference.max()))
        if assert_reference:
            assert np.all(difference <= 1), (name, int(difference.max()), 'condition 2')
    return zero, one, largest


def solve(kind, steps, mode, receive, *, check_coefficients=False):
    original = uniform.make_uniform_step
    coefficients = []

    def instrument(ctx, invariants, **kwargs):
        if check_coefficients:
            def check(h, absorb, mu, dt, factors):
                components = ctx.materials.components._replace(mu_update=mu)
                materials = ctx.materials._replace(components=components)
                checked = dict(invariants, ctx=replace(ctx, materials=materials, dt=dt),
                               h_coefficients=h, h_row_factors=factors)
                if absorb is not None:
                    checked['cpml_h_coefficients'] = absorb
                if kind == 'iv':
                    for lo in (True, False):
                        window = window_reference(ctx, (1, 0, lo))
                        assert np.any(np.asarray(mu[1])[window] == 2.), ('magnetic CPML slab', lo)
                coefficients.append(ulp_counts(checked, assert_reference=False))
            jax.debug.callback(check, invariants['h_coefficients'],
                               invariants.get('cpml_h_coefficients'),
                               ctx.materials.components.mu_update, ctx.dt,
                               invariants['h_row_factors'], ordered=True)
        if mode != 'new':
            old_h, old_absorb = legacy_kernels(invariants, mode == 'frozen')
            with patch.object(yee, 'update_h', old_h):
                advance = original(replace(ctx, apply_cpml_h=old_absorb), invariants, **kwargs)
        else:
            advance = original(ctx, invariants, **kwargs)

        def observed(carry, index, source, magnetic):
            result = advance(carry, index, source, magnetic)
            state = result[0]['fdtd']
            jax.debug.callback(receive, index, *state[:6], ordered=True)
            return result
        return observed

    sim = scene(kind)
    kwargs = dict(n_steps=steps, checkpoint=kind.startswith('vi-'), skip_preflight=True)
    gradient = None
    with patch.object(uniform, 'make_uniform_step', instrument):
        if kind.startswith(('iii-', 'vi-')):
            shape = sim._build_grid().shape
            keyword = 'eps_override' if kind.endswith('-eps') else 'mu_r_override'

            def objective(value):
                result = sim.forward(**kwargs, **{keyword: value})
                return jnp.sum(result.time_series ** 2)

            value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.ones(shape, jnp.float32))
            jax.block_until_ready((value, gradient))
        else:
            if kind == 'v-decay':
                kwargs.pop('n_steps')
                result = sim.run(**kwargs, until_decay=1e-3, decay_max_steps=steps,
                                 decay_check_interval=50, decay_min_steps=100)
            elif kind == 'v-run':
                result = sim.run(**kwargs)
            else:
                result = sim.forward(**kwargs)
            jax.block_until_ready(result.time_series)
    jax.effects_barrier()
    return gradient, coefficients


@pytest.mark.parametrize('kind', ['i', 'ii', 'iii-eps', 'iii-mu', 'iv',
                                  'v-run', 'v-decay', 'vi-eps', 'vi-mu'])
@pytest.mark.parametrize('steps', [100, 200, 400])
def test_four_conditions(kind, steps):
    records, errors, h_errors, peaks, bit_errors = {}, {}, {}, {}, []

    def retain(index, *fields):
        records[int(index)] = tuple(np.array(field) for field in fields)

    def frozen(index, *fields):
        for actual, expected in zip(fields, records[int(index)]):
            bit_errors.append(int(np.count_nonzero(np.asarray(actual).view(np.uint32) != expected.view(np.uint32))))

    def old(index, *fields):
        index = int(index)
        expected = fields[:3]
        errors[index] = max(float(np.max(np.abs(np.asarray(a, np.float64) - b)))
                            for a, b in zip(records[index][:3], expected))
        h_errors[index] = max(float(np.max(np.abs(np.asarray(a, np.float64) - b)))
                              for a, b in zip(records[index][3:], fields[3:]))
        peaks[index] = max(float(np.max(np.abs(a))) for a in expected)

    new_gradient, counts = solve(kind, steps, 'new', retain, check_coefficients=True)
    frozen_gradient, _ = solve(kind, steps, 'frozen', frozen)
    old_gradient, _ = solve(kind, steps, 'old', old)
    completed = len(records)
    assert sorted(records) == sorted(errors) == sorted(h_errors) == sorted(peaks) == list(range(completed))
    assert 0 < completed <= steps if kind == 'v-decay' else completed == steps
    error = np.array([errors[n] for n in range(completed)])
    peak = np.array([peaks[n] for n in range(completed)])
    rhs = 1.2e-7 * np.cumsum(peak)
    scale = 1e-4 * peak.max()
    grad_error = grad_scale = 0.
    gradient_bits = True
    if new_gradient is not None:
        gradient_bits = np.array_equal(np.asarray(new_gradient).view(np.uint32),
                                       np.asarray(frozen_gradient).view(np.uint32))
        grad_error = float(np.max(np.abs(np.asarray(new_gradient, np.float64) - old_gradient)))
        grad_scale = 1e-4 * float(np.max(np.abs(old_gradient)))
        assert grad_scale > 0
    condition_1 = max(bit_errors) == 0 and gradient_bits
    condition_2 = max(c[2] for c in counts) <= 1
    condition_3 = bool(np.all(error <= rhs) and error[-1] <= scale and grad_error <= grad_scale)
    print(f'H-fold scene={kind} steps={completed} cap={steps} bits={max(bit_errors)} '
          f'max_E_delta={error.max():.17g} end_E_delta={error[-1]:.17g} '
          f'max_H_delta={max(h_errors.values()):.17g} '
          f'growth_rhs_end={rhs[-1]:.17g} growth_max_excess={(error-rhs).max():.17g} '
          f'scale_1e-4={scale:.17g} ULP_0={sum(c[0] for c in counts)} '
          f'ULP_1={sum(c[1] for c in counts)} ULP_max={max(c[2] for c in counts)} '
          f'gradient_delta={grad_error:.17g} gradient_scale={grad_scale:.17g}', file=sys.stderr)
    print(f'H-fold scene={kind} steps={completed} condition_1={"PASS" if condition_1 else "FAIL"} '
          f'condition_2={"PASS" if condition_2 else "FAIL"} '
          f'condition_3={"PASS" if condition_3 else "FAIL"}', file=sys.stderr)
    assert peak.max() > 0
    assert condition_1, ('condition 1', max(bit_errors), gradient_bits)
    assert np.all(error <= rhs), ('condition 3 growth', float((error-rhs).max()))
    assert error[-1] <= scale, ('condition 3 scale', error[-1], scale)
    assert grad_error <= grad_scale, ('condition 3 gradient', grad_error, grad_scale)
    assert max(c[2] for c in counts) <= 1, ('condition 2', counts)


def nested_equations(jaxpr):
    if hasattr(jaxpr, 'jaxpr'):
        yield from nested_equations(jaxpr.jaxpr)
    elif hasattr(jaxpr, 'eqns'):
        for equation in jaxpr.eqns:
            yield equation
            for value in equation.params.values():
                yield from nested_equations(value)
    elif isinstance(jaxpr, (tuple, list)):
        for value in jaxpr:
            yield from nested_equations(value)


def test_scan_reads_folded_arrays():
    sim = scene('i')
    graph = jax.make_jaxpr(lambda: sim.forward(n_steps=4, checkpoint=False,
                                              skip_preflight=True).time_series)()
    scans = [eq for eq in nested_equations(graph) if eq.primitive.name == 'scan']
    assert scans
    for scan in scans:
        for equation in nested_equations(scan.params['jaxpr']):
            trace = str(equation.source_info.traceback)
            assert 'magnetic_update_coefficients' not in trace, 'row-factor product inside scan body'
            assert 'cpml_h_coefficients' not in trace, 'CPML coefficient formation inside scan body'


def test_fold_reference_with_traced_materials():
    from rfx.boundaries import magnetic_coefficient as folded
    captured = []
    original = folded.uniform_h_coefficients

    def capture(invariants, dtype):
        result = original(invariants, dtype)
        captured.append(result)
        return result

    sim = scene('i')
    with patch.object(folded, 'uniform_h_coefficients', capture):
        sim.forward(n_steps=1, checkpoint=False, skip_preflight=True)
    concrete = captured[-1]
    for keyword in ('eps_override', 'mu_r_override'):
        def evaluate(value):
            with patch.object(folded, 'uniform_h_coefficients', capture):
                sim.forward(n_steps=1, checkpoint=False, skip_preflight=True,
                            **{keyword: value})
            return captured[-1]['h_coefficients'], captured[-1]['cpml_h_coefficients']
        actual = jax.jit(evaluate)(jnp.ones(sim._build_grid().shape, jnp.float32))
        check = dict(concrete, h_coefficients=actual[0], cpml_h_coefficients=actual[1])
        zero, one, largest = ulp_counts(check)
        print(f'H-fold scene=iii-{keyword} ULP_0={zero} ULP_1={one}', file=sys.stderr)
