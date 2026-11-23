"""Whole forward results preserve arrays and host metadata across JAX transforms."""
import copy
import json
import os
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.api import ForwardResult, WaveguideSMatrixResult
from rfx.model import NominalProfile
from rfx.ringdown import RingdownSpec
from tests.contracts.test_forward_result_static_state import (
    DT, DX, DY, DZ, FREQS, MM, N_STEPS, PORT_X_A, _assert_static_state,
)
from tests.contracts.test_port_bins_traced_under_jit import _scene
from tests.unit.sparams.test_port_observable_validation import _passive_reciprocal_smatrix

TRANSFORMS = ('jit', 'vmap', 'grad', 'value_and_grad', 'vjp', 'jvp', 'jacfwd', 'eval_shape')
VALUES = (1., 1.05)
SHORT_STEPS = 150
SCENES = tuple(pytest.param(name, marks=() if name in ('U', 'G', 'W') else pytest.mark.slow)
               for name in ('U', 'G', 'GT', 'W', 'R'))
EAGER_CACHE = {}
TRANSFORM_CASES = [
    pytest.param(scene, kind, marks=pytest.mark.slow if (
        (scene, kind) not in {('U', 'jit'), ('U', 'vmap'), ('U', 'grad'),
                            ('U', 'jvp'), ('U', 'eval_shape'), ('G', 'jit'), ('W', 'jit')}
    ) else ())
    for scene in ('U', 'G', 'GT', 'W', 'R') for kind in TRANSFORMS
]


def _box_scene(**mesh):
    sim = Simulation(freq_max=20e9, domain=(12*MM, 11*MM, 5*MM), dx=MM,
                     boundary='pec', **mesh)
    sim.add_material('fill', eps_r=2.2, sigma=2*np.pi*12.5e9*8.854187817e-12*2.2/300)
    sim.add(Box((0, 0, 0), (12*MM, 11*MM, 5*MM)), material='fill')
    sim.add_port(position=(PORT_X_A, 3.8*MM, .8*MM), component='ez',
                 impedance=50., extent=.6*MM,
                 waveform=GaussianPulse(f0=13e9, bandwidth=.8, cutoff=4.5))
    sim.add_probe(position=(6.1*MM, 5.3*MM, 2.1*MM), component='ez')
    return sim


def _graded(scale, *, pinned=True, ringdown=False, emit=True):
    dx = jnp.asarray(DX, dtype=jnp.float32)
    dx = dx.at[6].set(DX[6] * scale).at[7].set(float(DX[6] + DX[7]) - DX[6] * scale)
    options = dict(dt=float(DT), dt_min_cell=.6*MM) if pinned else {}
    nominal = ({'x': NominalProfile(DX, traced_cells=(6, 8))}
               if isinstance(dx, jax.core.Tracer) else None)
    sim = _box_scene(dx_profile=dx, dy_profile=DY, dz_profile=DZ,
                     nominal_profiles=nominal, **options)
    required = 2.05 * 2 * np.linalg.norm([.012, .011, .005]) * np.sqrt(2.2) / 299792458.0 / float(DT)
    upper = 2 * int(np.ceil(required))
    ringdown_steps = next(n for n in range(upper - 2, upper + 2) if n - round(n / 2) >= required)
    result = sim.forward(n_steps=ringdown_steps if ringdown else SHORT_STEPS,
                         port_s11_freqs=FREQS, skip_preflight=True,
                         ringdown=RingdownSpec() if ringdown else None, emit_time_series=emit)
    if not pinned and isinstance(scale, jax.core.Tracer):
        assert isinstance(result.dt, jax.core.Tracer)
        assert isinstance(result.grid.dt, jax.core.Tracer)
        print('TIMESTEP', type(scale).__name__, 'result.dt and grid.dt traced')
    return result


def _make_forward(scene, *, emit=True):
    if scene in ('G', 'GT', 'R'):
        return lambda scale: _graded(scale, pinned=scene != 'GT', ringdown=scene == 'R', emit=emit)
    mesh = (dict(dx_profile=DX, dy_profile=DY, dz_profile=DZ,
                 dt=float(DT), dt_min_cell=.6*MM) if scene == 'M' else {})
    sim = _scene('waveguide', 'uniform') if scene == 'W' else _box_scene(**mesh)
    grid = sim._build_nonuniform_grid() if scene == 'M' else sim._build_grid()
    eps = jnp.full(grid.shape, 1. if scene == 'W' else 2.2)
    options = {} if scene == 'W' else {'port_s11_freqs': FREQS}
    if scene == 'M':
        options.update(distributed=True, devices=jax.devices('cpu')[:2])
    return lambda scale: sim.forward(
        n_steps=SHORT_STEPS, skip_preflight=True,
        eps_override=eps * scale, emit_time_series=emit, **options)


def _objective(forward, scene):
    def loss(scale):
        result = forward(scale)
        value = (jnp.sum(next(iter(result.waveguide_ports.values())).v_probe_t ** 2)
                 if scene == 'W' else jnp.sum(jnp.abs(result.s_params) ** 2))
        return value, result
    return loss


def _transform(forward, scene, kind, *, arrays_only=False):
    scale = jnp.float32(VALUES[0])
    objective = _objective(forward, scene)
    if arrays_only:
        full_forward, full_objective = forward, objective

        def forward(value):
            return tuple(row[1] for row in _arrays(full_forward(value)))

        def objective(value):
            loss, result = full_objective(value)
            return loss, tuple(row[1] for row in _arrays(result))

    if kind == 'jit':
        return jax.jit(forward)(scale), None
    if kind == 'vmap':
        return jax.vmap(forward)(jnp.asarray(VALUES, dtype=jnp.float32)), None
    if kind == 'grad':
        return jax.grad(objective, has_aux=True)(scale)[1], None
    if kind == 'value_and_grad':
        return jax.value_and_grad(objective, has_aux=True)(scale)[0][1], None
    if kind == 'vjp':
        value, pullback, result = jax.vjp(objective, scale, has_aux=True)
        jax.block_until_ready(pullback(jnp.ones_like(value)))
        return result, None
    if kind == 'jvp':
        return jax.jvp(forward, (scale,), (jnp.ones_like(scale),))
    if kind == 'jacfwd':
        return jax.jacfwd(objective, has_aux=True)(scale)[1], None
    return jax.eval_shape(forward, scale), None


@pytest.fixture(scope='module')
def scene(request):
    yield request.param
    jax.clear_caches()


@pytest.fixture(scope='module')
def forward(scene):
    return _make_forward(scene)


def _eager(scene, value, forward):
    key = (scene, value)
    if key not in EAGER_CACHE:
        EAGER_CACHE[key] = forward(jnp.float32(value))
    return EAGER_CACHE[key]


@pytest.fixture(scope='module')
def eager(scene, forward):
    return tuple(_eager(scene, value, forward) for value in VALUES)


@pytest.fixture(autouse=True)
def quiet_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


def _leaves(result, expected, *, tangent=False, abstract=False):
    assert type(result) is type(expected)
    assert result._fields == expected._fields
    for leaf in jax.tree_util.tree_leaves(result):
        assert (isinstance(leaf, jax.ShapeDtypeStruct) if abstract else
                isinstance(leaf, jax.Array) or
                (tangent and isinstance(leaf, np.ndarray) and leaf.dtype == jax.dtypes.float0)), type(leaf)
    kind = jax.ShapeDtypeStruct if abstract else jax.Array
    arrays = {id(leaf) for value in result for leaf in jax.tree_util.tree_leaves(value)
              if isinstance(leaf, kind)}
    assert arrays <= {id(leaf) for leaf in jax.tree_util.tree_leaves(result)}, 'an array is static'
    _assert_static_state(result)


def _arrays(result, *, children_only=True):
    children = {id(leaf) for leaf in jax.tree_util.tree_leaves(result)}
    for name in ('time_series', 's_params', 'waveguide_sparams', 'settling_echo_bound'):
        value = getattr(result, name, None)
        if value is not None:
            for path, leaf in jax.tree_util.tree_flatten_with_path(value)[0]:
                if (isinstance(leaf, (jax.Array, np.ndarray)) and
                        (not children_only or id(leaf) in children)):
                    yield name + jax.tree_util.keystr(path), leaf, name in ('time_series', 'settling_echo_bound')
    for port, record in (result.waveguide_ports or {}).items():
        for name in ('v_probe_t', 'v_ref_t', 'i_probe_t', 'i_ref_t'):
            value = getattr(record, name)
            if not children_only or id(value) in children:
                yield f'waveguide_ports[{port}].{name}', value, True


def _static_compared(actual, primal):
    children = {id(leaf) for leaf in jax.tree_util.tree_leaves(actual)}
    primal_arrays = {name: value for name, value, _ in _arrays(primal, children_only=False)}
    for name, value, _ in _arrays(actual, children_only=False):
        if id(value) not in children:
            assert _value_key(value) == _value_key(primal_arrays[name]), name


def _compare(actual, expected, label, *, arrays_only=False):
    ratios = {}
    actual_arrays = list(_arrays(actual))
    expected_arrays = ([(name, value, temporal) for (name, _, temporal), value
                        in zip(actual_arrays, expected, strict=True)] if arrays_only
                       else list(_arrays(expected)))
    assert [row[0] for row in actual_arrays] == [row[0] for row in expected_arrays]
    for (name, got, temporal), (_, want, _) in zip(actual_arrays, expected_arrays):
        got, want = np.asarray(got), np.asarray(want)
        assert got.shape == want.shape, name
        if want.size == 0:
            continue
        peak = float(np.max(np.abs(want)))
        difference = float(np.max(np.abs(got - want)))
        bar = (0. if name == "settling_echo_bound['n_max']" else
               9 * float(np.spacing(np.float32(peak))) if temporal else 1e-4 * peak)
        ratio = difference / bar if bar else (0. if difference == 0 else float('inf'))
        ratios[name] = dict(difference=difference, peak=peak, bar=bar, ratio=ratio)
    print('COMPARE', label, json.dumps(ratios, sort_keys=True))
    for name, measurement in ratios.items():
        assert measurement['difference'] <= measurement['bar'], (label, name, measurement)


def _value_key(value):
    """Independent host-value comparison; unknown objects compare separately by identity."""
    if isinstance(value, np.ndarray):
        return value.dtype, value.shape, value.tobytes()
    if isinstance(value, np.generic):
        return type(value), value.dtype, value.tobytes()
    if type(value) in (str, bool, int):
        return type(value), value
    if type(value) is float:
        return float, struct.pack('!d', value)
    if type(value) is complex:
        return complex, struct.pack('!dd', value.real, value.imag)
    return None


def _static_values(result):
    return {(name, jax.tree_util.keystr(path)): _value_key(leaf)
            for name, field in zip(result._fields, result)
            for path, leaf in jax.tree_util.tree_flatten_with_path(field)[0]
            if not isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct)) and _value_key(leaf) is not None}


def _structure(result):
    out = []
    for field in result:
        leaves, definition = jax.tree_util.tree_flatten(field)
        flags = tuple(isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct)) for leaf in leaves)
        out.append((definition, flags,
                    tuple(_value_key(leaf) for leaf, flag in zip(leaves, flags) if not flag),
                    tuple(leaf.shape for leaf, flag in zip(leaves, flags) if flag)))
    return out


def _roundtrip(result):
    leaves, definition = jax.tree_util.tree_flatten(result)
    restored = jax.tree_util.tree_unflatten(definition, leaves)
    assert type(restored) is type(result)
    assert restored.realized_geometry is result.realized_geometry
    carried_leaves, carried_definition = jax.tree_util.tree_flatten(restored)
    assert carried_definition == definition
    assert all(left is right for left, right in zip(leaves, carried_leaves))
    assert carried_definition.node_data()[1] is definition.node_data()[1]
    for original, rebuilt in zip(result, restored):
        before, before_def = jax.tree_util.tree_flatten(original)
        after, after_def = jax.tree_util.tree_flatten(rebuilt)
        assert before_def == after_def
        for left, right in zip(before, after):
            if isinstance(left, jax.Array) or _value_key(left) is None:
                assert left is right
            else:
                assert _value_key(left) == _value_key(right)
    placeholders = jax.tree_util.tree_unflatten(definition, [object() for _ in leaves])
    rebuilt_leaves, rebuilt_definition = jax.tree_util.tree_flatten(placeholders)
    assert rebuilt_definition == definition
    assert len(rebuilt_leaves) == len(leaves)
    assert all(type(leaf) is object for leaf in rebuilt_leaves)


@pytest.mark.parametrize('scene', SCENES, indirect=True, scope='module')
def test_eager_leaves(scene, eager):
    for result in eager:
        _leaves(result, eager[0])


@pytest.mark.parametrize('scene, kind', TRANSFORM_CASES, indirect=['scene'], scope='module')
def test_transform(scene, forward, eager, kind):
    if scene == 'W' and kind == 'vmap':
        with pytest.raises(ValueError, match='traced override over the segment cannot be judged before the run') as error:
            _transform(forward, scene, kind)
        print('REFUSAL', scene, kind, str(error.value))
        return
    actual, tangent = _transform(forward, scene, kind)
    _leaves(actual, eager[0], abstract=kind == 'eval_shape')
    if kind == 'eval_shape':
        if scene == 'U':
            assert _structure(actual) == _structure(eager[0])
        return
    jax.block_until_ready(actual)
    reference, reference_tangent = _transform(forward, scene, kind, arrays_only=True)
    if tangent is not None:
        _leaves(tangent, eager[0], tangent=True)
        assert jax.tree_util.tree_structure(tangent) == jax.tree_util.tree_structure(actual)
        _static_compared(tangent, actual)
        _compare(tangent, reference_tangent, f'{scene}/{kind}/tangent', arrays_only=True)
    if kind == 'vmap':
        for name in ('settling_db', 'settling_witness'):
            with pytest.raises(ValueError, match='arrays are not the arrays it was built with'):
                getattr(actual, name)
        for index, expected in enumerate(eager):
            element = jax.tree_util.tree_map(lambda a: a[index], actual)
            _leaves(element, expected)
            _static_compared(element, actual)
            arrays = tuple(array[index] for array in reference)
            _compare(element, arrays, f'{scene}/{kind}/{index}', arrays_only=True)
            _settling_read(element, expected, f'{scene}/{kind}/{index}')
            if scene == 'R':
                assert element.ringdown.report.completed
                assert element.ringdown.report.witnesses
            if scene == 'U' and index == 0:
                for operation, restored in (
                        ('pickle', pickle.loads(pickle.dumps(element))),
                        ('deepcopy', copy.deepcopy(element))):
                    assert type(restored) is type(element)
                    np.testing.assert_array_equal(restored.time_series, element.time_series)
                    _leaves(restored, element)
                    print('COPY', operation, 'carried_aux', hasattr(restored, '_forward_aux'))
    else:
        _compare(actual, reference, f'{scene}/{kind}', arrays_only=True)


def _settling_read(actual, eager, label):
    db = actual.settling_db
    witness, eager_witness = actual.settling_witness, eager.settling_witness
    print('SETTLING', label, json.dumps(dict(
        db=str(db), eager_db=str(eager.settling_db),
        status=witness['status'], eager_status=eager_witness['status']), sort_keys=True))
    assert isinstance(db, float)
    assert {'status', 'db', 'worst_freq_hz'} <= witness.keys()
    assert witness['status'] == eager_witness['status']


@pytest.mark.parametrize('scene', [pytest.param(name, marks=() if name == 'U' else pytest.mark.slow)
                                 for name in ('U', 'G', 'GT', 'W', 'R')],
                         indirect=True, scope='module')
def test_fresh_trace_static_values(scene, forward):
    reused = jax.jit(forward)
    reused(jnp.float32(VALUES[0]))
    if scene == 'W':
        for call in (reused, jax.jit(lambda value: forward(value))):
            with pytest.raises(jax.errors.JaxRuntimeError, match='traced override over the segment cannot be judged before the run') as error:
                jax.block_until_ready(call(jnp.float32(VALUES[1])))
            print('REFUSAL', scene, 'fresh_trace', str(error.value))
        return
    actual = reused(jnp.float32(VALUES[1]))
    fresh = jax.jit(lambda value: forward(value))(jnp.float32(VALUES[1]))
    assert _static_values(actual) == _static_values(fresh)
    _compare(actual, fresh, f'{scene}/fresh_trace')
    _assert_static_state(actual)
    _assert_static_state(fresh)


@pytest.mark.parametrize('scene', ('U', 'G', 'W'), indirect=True, scope='module')
def test_roundtrip(scene, eager):
    _roundtrip(eager[0])


def test_waveguide_smatrix_roundtrip():
    spectrum, freqs = _passive_reciprocal_smatrix()
    result = WaveguideSMatrixResult(
        s_params=spectrum, freqs=freqs, port_names=('left', 'right'),
        port_directions=('+x', '-x'), reference_planes=np.array([.01, .09]))
    _roundtrip(result)
    assert jax.tree_util.tree_leaves(result) == []


@pytest.mark.slow
def test_equal_calls_and_scalar_jacobian():
    forward = _make_forward('U')
    left, right = forward(jnp.float32(1)), forward(jnp.float32(1))
    assert _static_values(left) == _static_values(right)
    assert _structure(left) == _structure(right)
    assert left.realized_geometry is not right.realized_geometry
    assert jax.tree_util.tree_structure(left) != jax.tree_util.tree_structure(right)
    with pytest.raises(ValueError):
        jax.tree_util.tree_map(lambda a, b: a, left, right)
    jacobian = jax.jacfwd(forward)(jnp.float32(1))
    _leaves(jacobian, left)
    _static_compared(jacobian, left)


@pytest.mark.slow
def test_changed_shapes_without_probe_series():
    forward = _make_forward('G', emit=False)
    eager = forward(jnp.float32(VALUES[0]))
    assert eager.time_series.shape == (SHORT_STEPS, 0)
    batched = jax.vmap(forward)(jnp.asarray(VALUES, dtype=jnp.float32))
    assert batched.time_series.shape == (len(VALUES), SHORT_STEPS, 0)
    for name in ('settling_db', 'settling_witness'):
        with pytest.raises(ValueError, match='arrays are not the arrays it was built with'):
            getattr(batched, name)
    element = jax.tree_util.tree_map(lambda a: a[0], batched)
    _settling_read(element, eager, 'no_series/vmap/0')
    sliced = jax.tree_util.tree_map(lambda a: a[..., :1] if a.ndim else a, element)
    with pytest.raises(ValueError, match='arrays are not the arrays it was built with'):
        _ = sliced.settling_witness
    replaced = batched._replace(time_series=None)
    assert not hasattr(replaced, '_forward_aux')


@pytest.mark.slow
@pytest.mark.parametrize('kind', ('jit', 'grad'))
def test_multi(kind):
    if os.environ.get('RFX_RESULT_TRANSFORMS_CHILD') != '1':
        pytest.skip('covered by test_distributed_subprocess')
    assert len(jax.devices('cpu')) == 2
    forward = _make_forward('M')
    eager = _eager('M', VALUES[0], forward)
    _leaves(eager, eager)
    actual, _ = _transform(forward, 'M', kind)
    _leaves(actual, eager)
    reference, _ = _transform(forward, 'M', kind, arrays_only=True)
    _compare(actual, reference, f'M/{kind}', arrays_only=True)


@pytest.mark.slow
def test_distributed_subprocess(tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root), JAX_PLATFORMS='cpu',
               PYTHONDONTWRITEBYTECODE='1', JAX_ENABLE_COMPILATION_CACHE='false',
               RFX_RESULT_TRANSFORMS_CHILD='1',
               XLA_FLAGS='--xla_force_host_platform_device_count=2')
    child = subprocess.run(
        [sys.executable, '-m', 'pytest', str(Path(__file__)), '-k', 'test_multi',
         '-q', '-s', '-m', '', '-p', 'no:cacheprovider', '--durations=0', '--durations-min=0',
         '--basetemp', str(tmp_path / 'child')],
        cwd=root, env=env, text=True, capture_output=True, timeout=240)
    print(child.stdout + child.stderr)
    assert child.returncode == 0, child.stdout + child.stderr
