"""C1: declaration setup is concrete; all three multi-device advances are staged.

The eager public call is the reference. CPU quantities are bitwise, including
advance operands. The M0 declarations use 32 steps here; three-device rows run
in a fresh interpreter when CI exposes only two devices. RFX_C1_REPORT appends
measurements before assertions, including wall times and per-leaf peak ULPs.
"""
from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, DebyePole, GaussianPulse, LorentzPole, PolylineWire, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.errors import RfxRefusal
from rfx.grid import Grid

pytestmark = pytest.mark.distributed
N_STEPS = 32
KINDS = ('pec_box', 'cpml_box', 'lumped_port', 'debye', 'lorentz',
         'thin_film', 'thin_pec_wire', 'epsilon_box')
PROFILE = np.array([1.] * 7 + [.9, 1.1, .95, 1.05, .9, 1.1, 1.05, .95]
                   + [1.] * 9) * .001
TRACED_POLE = (
    "cannot judge Lorentz/Drude time-step stability: this multi-device run's material values or "
    "cell sizes are traced by an outer jax.jit (or a nested jax.vmap), and a check cannot be raised "
    "from inside a multi-device program. Call it without jax.jit (jax.grad, jax.jvp and one "
    "jax.vmap alone keep concrete values and are judged), or run on one device.")


def scene(kind, mesh, *, eps_r=3., amplitude=None):
    boundary = (BoundarySpec(x='pec', y=Boundary('pmc', 'pec'), z='pec') if kind == 'pmc'
                else 'cpml' if kind == 'cpml_box' else 'pec')
    sim = Simulation(freq_max=10e9, domain=(.024, .012, .012), dx=.001,
                     boundary=boundary, cpml_layers=4 if boundary == 'cpml' else 0,
                     snap='declared', **({'dx_profile': PROFILE} if mesh == 'graded' else {}))
    pulse = GaussianPulse(f0=5e9, bandwidth=.8)
    waveform = pulse if amplitude is None else lambda t: amplitude * pulse(t)
    if kind == 'lumped_port':
        sim.add_port((.006, .006, .006), 'ez', impedance=50., waveform=waveform)
    else:
        sim.add_source((.006, .006, .006), 'ez', waveform=waveform, amplitude_kind='current')
    if kind in ('debye', 'lorentz', 'unstable_lorentz'):
        omega = 2*np.pi*(8e12 if kind == 'unstable_lorentz' else 8e9)
        poles = ({'debye_poles': [DebyePole(delta_eps=1., tau=1e-11)]}
                 if kind == 'debye' else {'lorentz_poles': [LorentzPole(
                     omega_0=omega, delta=2*np.pi*1e9, kappa=omega**2)]})
        sim.add_material('pole', eps_r=eps_r, **poles)
        sim.add(Box((.008, .003, .003), (.018, .009, .009)), material='pole')
    elif kind in ('f0_sheet', 'thin_film'):
        sim.add_thin_conductor(Box((.012, .003, .003), (.012, .009, .009)),
                               sigma_bulk=1000., thickness=1e-5,
                               **({'surface_impedance_f0': 5e9} if kind == 'f0_sheet' else {}))
    elif kind == 'thin_pec_wire':
        sim.add(PolylineWire(((.010, .007, .006), (.017, .007, .006)), radius=0), material='pec')
    elif kind == 'epsilon_box':
        sim.add_material('dielectric', eps_r=2.2)
        sim.add(Box((0., 0., 0.), (.024, .012, .012)), material='dielectric')
    for pos, component in (((.006, .006, .006), 'ez'),
                           ((.013, .005, .007), 'ez'), ((.019, .006, .006), 'hy')):
        sim.add_probe(pos, component)
    return sim


def leaves(value, path=''):
    """Include dataclass/static metadata as well as registered pytree leaves."""
    if isinstance(value, (jax.Array, jax.core.Tracer, np.ndarray)):
        return {path: value}
    if isinstance(value, Grid):
        value = vars(value)
    elif is_dataclass(value):
        value = {f.name: getattr(value, f.name) for f in fields(value)}
    elif hasattr(value, '_asdict'):
        value = value._asdict()
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, (tuple, list)):
        items = enumerate(value)
    else:
        return {path: value}
    result = {}
    for key, item in items:
        result.update(leaves(item, f'{path}.{key}' if path else str(key)))
    return result


def arrays(value):
    return {p: a for p, a in leaves(value).items()
            if isinstance(a, (jax.Array, jax.core.Tracer, np.ndarray))}


def snapshot(value):
    result = leaves(value)
    for path, leaf in result.items():
        assert not isinstance(leaf, jax.core.Tracer), f'traced advance operand: {path}'
        if isinstance(leaf, (jax.Array, np.ndarray)):
            result[path] = np.array(leaf, copy=True)
    return result


def compare(expected, actual, *, metadata=False):
    assert expected.keys() == actual.keys(), (expected.keys() - actual.keys(), actual.keys() - expected.keys())
    rows = {}
    for path, expected_leaf in expected.items():
        actual_leaf = actual[path]
        if not isinstance(expected_leaf, (jax.Array, np.ndarray)):
            assert (expected_leaf == actual_leaf or
                    isinstance(expected_leaf, float) and isinstance(actual_leaf, float)
                    and np.isnan(expected_leaf) and np.isnan(actual_leaf)), path
            continue
        a, b = np.asarray(expected_leaf), np.asarray(actual_leaf)
        if metadata and path.startswith('grid.') and path.endswith('_f64') and a.dtype == np.float64:
            a = a.astype(b.dtype)
        assert a.shape == b.shape and a.dtype == b.dtype, (path, a.shape, b.shape, a.dtype, b.dtype)
        equal = a.tobytes() == b.tobytes()
        ulp = 0.
        if a.size and np.issubdtype(a.dtype, np.inexact):
            delta = float(np.max(np.abs(a.astype(np.complex128) - b.astype(np.complex128))))
            peak = np.max(np.abs(a))
            spacing = float(abs(np.spacing(peak)))
            ulp = delta / spacing if spacing else (0. if delta == 0. else float('inf'))
        rows[path] = dict(bitwise=equal, peak_ulp=ulp)
    return dict(bitwise=all(r['bitwise'] for r in rows.values()),
                peak_ulp=max((r['peak_ulp'] for r in rows.values()), default=0.), arrays=len(rows), leaves=rows)


def record(row):
    print('C1 ' + json.dumps(row), flush=True)
    target = os.environ.get('RFX_C1_REPORT')
    if target:
        with open(target, 'a') as stream:
            stream.write(json.dumps(row) + '\n')


def call_sim(sim, entry, devices, **kwargs):
    options = dict(n_steps=N_STEPS, skip_preflight=True, devices=devices, **kwargs)
    return (sim.forward(distributed=True, **options) if entry == 'forward'
            else sim.run(compute_s_params=False, **options))


def is_settling(path):
    return path == 'settling_db' or path == 'settling_witness' or path.startswith('settling_witness.')


def multi_device_run_result(compiled, statics):
    """Keep the returned arrays and trace-time statics observable after jit."""
    actual = jax.block_until_ready(compiled())
    return actual, dict(statics)


def one_device_settling(kind, mesh):
    sim = scene(kind, mesh)
    statics = {}

    def invoke():
        result = call_sim(sim, 'run', jax.devices('cpu')[:1])
        values = leaves(result)
        result_arrays = arrays(result)
        statics.update({p: a for p, a in values.items()
                        if is_settling(p) and p not in result_arrays})
        return {p: a for p, a in result_arrays.items() if is_settling(p)}

    actual = jax.block_until_ready(jax.jit(invoke)())
    return {**statics, **actual}


def compare_settling(expected, actual):
    """B.2: one-device jitted run supplies the expected report, including dtype."""
    actual = {p: a for p, a in actual.items() if is_settling(p)}
    assert expected.keys() == actual.keys(), ('B.2 leaves', expected.keys(), actual.keys())
    for path, a in expected.items():
        b = actual[path]
        a_array, b_array = isinstance(a, (jax.Array, np.ndarray)), isinstance(b, (jax.Array, np.ndarray))
        assert a_array == b_array, ('B.2', path, type(a), type(b))
        if a_array:
            a, b = np.asarray(a), np.asarray(b)
            assert a.shape == b.shape and a.dtype == b.dtype, ('B.2', path, a.shape, b.shape, a.dtype, b.dtype)
            assert np.array_equal(a, b, equal_nan=True), ('B.2', path, a, b)
        else:
            assert (a == b or isinstance(a, float) and isinstance(b, float)
                    and np.isnan(a) and np.isnan(b)), ('B.2', path, a, b)
    return dict(equal=True, leaves=len(expected))


def parity(kind, mesh, count, entry):
    from rfx.stepping import sharded
    devices = jax.devices('cpu')[:count]
    assert len(devices) == count
    start = time.perf_counter()
    captures, statics, outputs = [], [], []
    original = sharded.recorded_sharded_call
    sim = scene(kind, mesh)

    def observe(run_fn, *args, **kwargs):
        captures.append(snapshot((args, kwargs)))  # The single hook, before the barrier.
        value = original(run_fn, *args, **kwargs)
        outputs.append([isinstance(a, jax.core.Tracer) for a in arrays(value).values()])
        return value

    def invoke():
        result = call_sim(sim, entry, devices)
        all_leaves = leaves(result)
        statics.append({p: a for p, a in all_leaves.items() if p not in arrays(result)})
        return arrays(result)

    row = dict(kind=kind, mesh=mesh, devices=count, entry=entry, n_steps=N_STEPS)
    try:
        with patch.object(sharded, 'recorded_sharded_call', observe):
            eager = jax.block_until_ready(invoke())
            assert len(captures) == 1 and not any(outputs[0])
            # Fresh declaration and caches: do not borrow eager setup from the reference.
            sim = scene(kind, mesh)
            jax.clear_caches()
            lowered = jax.jit(invoke).lower()
            staged = len(outputs) == 2 and bool(outputs[1]) and all(outputs[1])
            staged = staged and 'stablehlo.while' in lowered.as_text()
            compiled = lowered.compile()
            if entry == 'run':
                actual, statics[-1] = multi_device_run_result(compiled, statics[-1])
            else:
                actual = jax.block_until_ready(compiled())
        assert len(captures) == 2 and len(statics) == 2
        host = compare(captures[0], captures[1])
        # M-1 (bedca8841): base/one, head/one and head/two have identical
        # eager-vs-jit difference lists on 17 fixtures (s4-m1one-11081222-zvdfv,
        # s4-m1two-11081222-crr28). Under caller jit the settling report is
        # unformed. B.1 excludes only that report; B.2 compares it to jit/one.
        def comparable(values):
            return {p: a for p, a in values.items() if entry != 'run' or not is_settling(p)}
        returned = compare(comparable(eager), comparable(actual), metadata=True)
        compare(comparable(statics[0]), comparable(statics[1]))
        row.update(host=host, returned=returned, staged=staged)
        if entry == 'run':
            row['settling'] = compare_settling(
                one_device_settling(kind, mesh), {**statics[1], **actual})
        assert host['arrays'] > 0 and returned['arrays'] > 0
        assert host['bitwise'], host
        assert returned['bitwise'], returned
        assert staged, 'advance ran at trace time or StableHLO has no while'
        assert any(np.any(np.asarray(a)) for p, a in eager.items() if 'time_series' in p)
    except Exception as exc:
        row.update(error=type(exc).__name__, message=str(exc))
        raise
    finally:
        row['wall_seconds'] = time.perf_counter() - start
        record(row)
        jax.clear_caches()


@pytest.mark.parametrize('entry', ['forward', 'run'])
@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
@pytest.mark.parametrize('kind', KINDS)
def test_two_device_setup(kind, mesh, entry):
    parity(kind, mesh, 2, entry)


@pytest.mark.parametrize('entry', ['forward', 'run'])
def test_two_device_graded_legacy(entry):
    parity('pmc', 'graded', 2, entry)


@pytest.mark.parametrize('kind,mesh', [(k, m) for k in KINDS for m in ('uniform', 'graded')] + [('pmc', 'graded')])
@pytest.mark.parametrize('entry', ['forward', 'run'])
def test_three_device_setup(kind, mesh, entry):
    if len(jax.devices('cpu')) >= 3:
        parity(kind, mesh, 3, entry)
        return
    child_call('parity', kind, mesh, 3, entry)


def child_call(function, *args):
    # Same fresh-process pattern as the distributed runner memory contracts.
    env = dict(os.environ, JAX_PLATFORMS='cpu',
               XLA_FLAGS='--xla_force_host_platform_device_count=3', PYTHONDONTWRITEBYTECODE='1')
    code = ('from tests.contracts import test_multidevice_setup_boundary as c; '
            f'c.{function}(*{args!r})')
    child = subprocess.run([sys.executable, '-c', code], env=env,
                           cwd=Path(__file__).resolve().parents[2], text=True,
                           capture_output=True, timeout=240)
    assert child.returncode == 0, child.stdout + child.stderr


@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
@pytest.mark.parametrize('count', [2, 3])
def test_f0_sheet_refusal(mesh, count):
    if len(jax.devices('cpu')) < count:
        child_call('test_f0_sheet_refusal', mesh, count)
        return
    sim = scene('f0_sheet', mesh)
    def invoke():
        return call_sim(sim, 'forward', jax.devices('cpu')[:count]).time_series
    errors = []
    for call in (invoke, jax.jit(invoke)):
        with pytest.raises(RfxRefusal) as failure:
            call()
        errors.append(str(failure.value))
    assert errors[0] == errors[1]
    record(dict(kind='f0_sheet', mesh=mesh, devices=count, message=errors[0], status='refused'))


@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
@pytest.mark.parametrize('entry', ['forward', 'run'])
def test_unstable_lorentz_same_refusal(mesh, entry):
    errors = []
    def invoke():
        return call_sim(scene('unstable_lorentz', mesh), entry, jax.devices('cpu')[:2]).time_series
    for call in (invoke, jax.jit(invoke)):
        with pytest.raises(ValueError, match="unstable at this model's time step") as failure:
            call()
        errors.append((type(failure.value).__name__, str(failure.value)))
    assert errors[0] == errors[1]
    record(dict(kind='unstable_lorentz', mesh=mesh, entry=entry, refusal=errors[0]))


@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
def test_traced_lorentz_refusal(mesh):
    def invoke(eps_r):
        sim = scene('lorentz', mesh, eps_r=eps_r)
        return call_sim(sim, 'forward', jax.devices('cpu')[:2]).time_series
    with pytest.raises(RfxRefusal) as failure:
        jax.jit(invoke)(jnp.float32(3.))
    assert str(failure.value) == TRACED_POLE
    record(dict(kind='traced_lorentz', mesh=mesh, message=str(failure.value), status='refused'))


@pytest.mark.parametrize('entry', ['forward', 'run'])
def test_closed_box_jit_refusal(entry):
    sim = Simulation(freq_max=15e9, domain=(.024, .018, .016), dx=.001,
                     boundary='cpml', cpml_layers=4, snap='declared')
    sim.add_tfsf_source(f0=7.5e9, closed_box=True, margin=3)
    def invoke():
        return call_sim(sim, entry, jax.devices('cpu')[:2]).time_series
    with pytest.raises(NotImplementedError) as failure:
        jax.jit(invoke)()
    expected = ('Multi-device closed-box TF/SF cannot run inside a user jax.jit; '
                'use the plain call to forward() or run().' if entry == 'forward' else
                'CustomWaveform and closed_box TFSF cannot be passed through run() '
                'under external JAX transformations; differentiate forward() instead.')
    assert str(failure.value) == expected
    record(dict(kind='closed_box_tfsf', entry=entry, message=str(failure.value), status='refused'))


@pytest.mark.parametrize('entry', ['forward', 'run'])
def test_uniform_port_s_jit_refusal(entry):
    sim = scene('lumped_port', 'uniform')
    def invoke():
        options = dict(n_steps=N_STEPS, devices=jax.devices('cpu')[:2], skip_preflight=True)
        return (sim.forward(distributed=True, port_s11_freqs=np.array([5e9]), **options)
                if entry == 'forward' else sim.run(compute_s_params=True, **options)).s_params
    expected = (
        "forward(distributed=True) on a uniform mesh cannot compute port S-parameters inside jax.jit "
        "(or another JAX trace that stages it): on this path the port S-parameters are assembled "
        "on the host from the main run's port records. Omit port_s11_freqs for the fields and probe records, "
        "call forward() directly, or run on one device (omit distributed=True)." if entry == 'forward' else
        "run(devices=...) on a uniform mesh cannot compute port S-parameters inside jax.jit "
        "(or another JAX trace that stages it): the S-parameters are measured from a second "
        "multi-device run outside the trace. Pass compute_s_params=False for the fields and "
        "probe records, call run() directly, or run on one device (omit devices=).")
    with pytest.raises(NotImplementedError) as failure:
        jax.jit(invoke)()
    assert str(failure.value) == expected
    record(dict(kind='port_s', entry=entry, message=str(failure.value), status='refused'))


@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
@pytest.mark.parametrize('transform', ['jit_run', 'jit_grad', 'grad', 'jvp', 'vmap', 'remat'])
def test_source_amplitude_transforms(mesh, transform):
    """Plain grad over multi-device: runs and is finite; its value is pinned
    against one device by
    tests/unit/runners/test_distributed_graded_records_gradient.py::test_gradient
    and tests/unit/runners/test_distributed_edge_averaged_materials.py::test_the_permittivity_gradient_matches_one_device.
    """
    def objective(amplitude):
        sim = scene('epsilon_box', mesh, amplitude=amplitude)
        return jnp.sum(call_sim(sim, 'run' if transform == 'jit_run' else 'forward', jax.devices('cpu')[:2]).time_series)
    amplitude = jnp.float32(1.)
    if transform == 'jit_run':
        reference, call = objective(amplitude), jax.jit(objective)
    elif transform in ('jit_grad', 'grad'):
        reference = jax.grad(objective)(amplitude)
        call = jax.jit(jax.grad(objective)) if transform == 'jit_grad' else jax.grad(objective)
    elif transform == 'jvp':
        tangent = lambda a: jax.jvp(objective, (a,), (jnp.ones_like(a),))[1]
        reference, call = tangent(amplitude), jax.jit(tangent)
    elif transform == 'vmap':
        call = lambda a: jax.vmap(objective)(a[None])[0]
        reference = objective(amplitude)
    else:
        call = jax.jit(jax.checkpoint(objective))
        reference = objective(amplitude)
    actual = call(amplitude)
    measured = compare({'value': reference}, {'value': actual})
    record(dict(kind='epsilon_box', mesh=mesh, transform=transform, returned=measured))
    # Cross-trace bar for a quantity summed over the record: an eager-identical program change moved jit_grad by 1 ULP and main itself is not bitwise eager-vs-jit, so bitwise cannot decide this row.
    if transform == 'jit_grad':
        assert np.isfinite(actual)
        assert abs(float(actual) - float(reference)) <= 1e-4 * abs(float(reference))
    else:
        assert measured['bitwise'], measured
    assert np.isfinite(actual) and float(abs(actual)) > 0.


def test_exit_without_compile_time_switch(monkeypatch):
    from rfx.core import jax_utils
    from rfx.stepping import sharded
    monkeypatch.setattr(jax_utils, 'compile_time_switch', lambda: None)
    monkeypatch.setattr(sharded, 'compile_time_switch', lambda: None)
    seen = []
    def advance(value):
        seen.append(jax_utils._REGION.get())
        return jax.lax.scan(lambda carry, _: (carry + value, carry), jnp.zeros(()), None, length=3)[0]
    def invoke():
        with jax_utils.declaration_setup():
            result = sharded.recorded_sharded_call(advance, jnp.float32(2.))
            assert isinstance(result, jax.core.Tracer)
            assert jax_utils._REGION.get() is not None
            return result
    lowered = jax.jit(invoke).lower()
    assert 'stablehlo.while' in lowered.as_text()
    assert float(lowered.compile()()) == 6.
    assert seen == [None]
    assert jax_utils._REGION.get() is None


def plain_call(mesh, entry):
    """No caller jit: every setup leaf enters the advance as a jit argument."""
    from rfx.core.jax_utils import _REGION
    from rfx.stepping import sharded
    seen = []
    original = sharded.recorded_sharded_call

    def observe(run_fn, *args, **kwargs):
        assert _REGION.get() is None
        concrete = snapshot((args, kwargs))
        closed = jax.make_jaxpr(run_fn)(*args, **kwargs)
        inputs = jax.tree.leaves((args, kwargs))
        assert len(closed.jaxpr.invars) == len(inputs)
        for var, value in zip(closed.jaxpr.invars, inputs):
            if hasattr(value, 'dtype'):
                assert var.aval.shape == value.shape and var.aval.dtype == value.dtype
        seen.append(dict(arguments=len(inputs), arrays=len(arrays((args, kwargs))),
                         setup_array_bytes=sum(a.nbytes for a in concrete.values()
                                               if isinstance(a, np.ndarray))))
        return original(run_fn, *args, **kwargs)

    with patch.object(sharded, 'recorded_sharded_call', observe):
        result = call_sim(scene('epsilon_box', mesh), entry, jax.devices('cpu')[:2])
        returned = snapshot(result)
    assert len(seen) == 1 and seen[0]['arrays'] > 0
    record(dict(kind='epsilon_box', mesh=mesh, entry=entry, mode='plain',
                setup_arguments=True, **seen[0]))
    return returned


@pytest.mark.parametrize('mesh', ['uniform', 'graded'])
@pytest.mark.parametrize('entry', ['forward', 'run'])
def test_plain_setup_arrays_are_jit_arguments(mesh, entry):
    plain_call(mesh, entry)
