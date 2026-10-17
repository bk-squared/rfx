"""Census of requested forward bins constructed inside an outer JAX trace."""
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.ringdown import RingdownSpec

KINDS = ('lumped', 'wire', 'coax', 'waveguide', 'msl')
PATHS = ('uniform', 'graded', 'multi-uniform', 'multi-graded')
REQUEST = np.array([1.1e9, 1.7e9, 2.3e9], dtype=np.float32)


def _scene(kind, path):
    graded = path.endswith('graded')
    mesh = {'dz_profile': np.linspace(.8e-3, 1.2e-3, 8)} if graded else {}
    if kind in ('lumped', 'wire'):
        sim = Simulation(freq_max=10e9, domain=(.008,) * 3, dx=.001,
                         boundary='pec', **mesh)
        sim.add_port((.004, .004, .003), 'ez', impedance=50,
                     extent=.002 if kind == 'wire' else None,
                     waveform=GaussianPulse(f0=2e9, bandwidth=.9))
    else:
        mesh = {'dz_profile': np.linspace(.8e-3, 1.2e-3, 16)} if graded else {}
        sim = Simulation(freq_max=20e9, domain=(.024, .020, .016), dx=.001,
                         boundary=BoundarySpec(x='cpml', y='pec', z='pec'),
                         cpml_layers=2, **mesh)
        if kind == 'coax':
            sim.add_coaxial_port((.006, .006, .008), face='bottom', pin_length=.004)
        elif kind == 'waveguide':
            sim.add_waveguide_port(.006, f0=10e9, freqs=np.array([10e9]),
                                   probe_offset=3, ref_offset=2)
        else:
            sim.add_material('sub', eps_r=2.2)
            sim.add(Box((0, 0, 0), (.024, .020, .001)), material='sub')
            sim.add(Box((0, 0, 0), (.024, .020, 0)), material='pec')
            sim.add(Box((.004, .008, .001), (.024, .012, .001)), material='pec')
            sim.add_msl_port((.004, .010, 0), width=.004, height=.001,
                             direction='+x', impedance=50, eps_r_sub=2.2,
                             n_probe_offset=5, n_probe_spacing=2, n_probes=3)
    return sim


def _call(sim, path, ringdown, bins, *, eps_override=None):
    options = {'ringdown': RingdownSpec()} if ringdown else {}
    if path.startswith('multi-'):
        options.update(distributed=True, devices=jax.devices('cpu')[:2])
    if eps_override is not None:
        options['eps_override'] = eps_override
    steps = (600 if path.endswith('graded') else 300) if ringdown else 100
    result = sim.forward(n_steps=steps, skip_preflight=True,
                         port_s11_freqs=bins, **options)
    return result.s_params, result.freqs


def _evaluate(call, *, compiled=False):
    """Record actual staging so the identity mutation tests the eager control."""
    staged = []

    def invoke():
        staged.append(isinstance(jnp.asarray(0), jax.core.Tracer))
        return call()

    try:
        value = (jax.jit(invoke) if compiled else invoke)()
        return jax.block_until_ready(value), None, staged[0]
    except Exception as error:
        return None, error, staged[0]


def _success(outcome):
    value, error, _ = outcome
    if error is not None:
        raise error
    assert value is not None
    spectrum, freqs = value
    assert spectrum is not None and freqs is not None
    assert np.isfinite(spectrum).all() and np.isfinite(freqs).all()
    assert freqs.dtype == np.float32
    np.testing.assert_array_equal(freqs, REQUEST)
    return value


def _equal(eager, actual):
    s0, f0 = _success(eager)
    s1, f1 = _success(actual)
    np.testing.assert_array_equal(f1, f0)
    assert s1.shape == s0.shape
    peak = float(np.max(np.abs(s0)))
    delta = float(np.max(np.abs(np.asarray(s1) - np.asarray(s0))))
    print(f'max |dS|={delta:.9g}, peak={peak:.9g}, bar={1e-4 * peak:.9g}', file=sys.stderr)
    assert peak > 0
    assert delta <= 1e-4 * peak


def _refusal(outcome, exception, leading):
    error = outcome[1]
    assert type(error) is exception, repr(error)
    assert str(error).startswith(leading), str(error)
    print(f'{exception.__name__}: {str(error).splitlines()[0]}', file=sys.stderr)


def _same_refusal(eager, actual, exception, leading):
    _refusal(eager, exception, leading)
    _refusal(actual, exception, leading)
    assert str(actual[1]).splitlines()[0] == str(eager[1]).splitlines()[0]


def _graded_wire_gradient(sim):
    """Folded #1772 value/gradient witness, with its original bins and bar."""
    request = np.linspace(1.95e9, 2.05e9, 5)
    eps = jnp.full(sim._build_nonuniform_grid().shape, 2.0)

    def loss(eps_override):
        spectrum, freqs = _call(sim, 'graded', False, jnp.asarray(request),
                                eps_override=eps_override)
        return jnp.sum(jnp.abs(spectrum) ** 2), freqs

    (v0, f0), g0 = jax.value_and_grad(loss, has_aux=True)(eps)
    (v1, f1), g1 = jax.jit(jax.value_and_grad(loss, has_aux=True))(eps)
    np.testing.assert_array_equal(f1, f0)
    rel = float(jnp.max(jnp.abs(g1 - g0)) / jnp.max(jnp.abs(g0)))
    print(f'jit vs eager gradient rel={rel:.3g}', file=sys.stderr)
    assert abs(float(v1) - float(v0)) <= 1e-4 * abs(float(v0))
    assert rel <= 1e-4


def test_distributed_subprocess(tmp_path):
    """Always isolate the virtual CPU devices, even when the parent has two."""
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root), JAX_PLATFORMS='cpu',
               PYTHONDONTWRITEBYTECODE='1', JAX_ENABLE_COMPILATION_CACHE='false',
               RFX_TRACED_BINS_CHILD='1',
               XLA_FLAGS='--xla_force_host_platform_device_count=2')
    child = subprocess.run([sys.executable, '-m', 'pytest', str(Path(__file__)),
                            '-k', 'multi', '-q', '-s', '--basetemp', str(tmp_path / 'child')],
                           cwd=root, env=env, text=True, capture_output=True, timeout=240)
    print(child.stdout + child.stderr, file=sys.stderr)
    assert child.returncode == 0, child.stdout + child.stderr


@pytest.mark.parametrize('kind,path,ringdown', [
    pytest.param(kind, path, ringdown, id=f'{kind}-{path}' + ('-ringdown' if ringdown else ''))
    for ringdown in (False, True) for path in PATHS for kind in KINDS
])
def test_port_bins(kind, path, ringdown):
    if path.startswith('multi-'):
        if os.environ.get('RFX_TRACED_BINS_CHILD') != '1':
            pytest.skip('covered by test_distributed_subprocess')
        assert len(jax.devices('cpu')) == 2
    sim = _scene(kind, path)
    if ringdown and kind == 'wire':
        from tests.unit.sparams.test_ringdown_run import _box
        sim = _box('graded' if path.endswith('graded') else 'uniform')
    eager = _evaluate(lambda: _call(sim, path, ringdown, REQUEST))
    actual = _evaluate(lambda: _call(sim, path, ringdown, jnp.asarray(REQUEST)), compiled=True)
    print(f'REACH {kind}-{path}, ringdown={ringdown}, staged={actual[2]}', file=sys.stderr)

    if ringdown and path.startswith('multi-'):
        _same_refusal(eager, actual, NotImplementedError,
                      'forward(ringdown=...) is not supported with distributed=True')
    elif ringdown and kind != 'wire':
        _same_refusal(eager, actual, NotImplementedError,
                      'forward(ringdown=...) completes wire-port S-parameters only;')
    elif ringdown:
        concrete = _evaluate(lambda: _call(sim, path, ringdown, REQUEST), compiled=True)
        if path == 'graded' and eager[1] is not None:
            # The leader permits this reduced witness if this one small-scene
            # attempt cannot turn its pulse off before the graded fit window.
            assert type(eager[1]) is ValueError
            assert 'is still on inside the identification window' in str(eager[1])
            _same_refusal(eager, concrete, ValueError, 'forward(ringdown=...):')
            if not actual[2]:
                _same_refusal(eager, actual, ValueError, 'forward(ringdown=...):')
        else:
            _equal(eager, concrete)
            if not actual[2]:
                _equal(eager, actual)
        if actual[2]:
            assert type(actual[1]) is jax.errors.ConcretizationTypeError, repr(actual[1])
            message = ('forward(ringdown=...) identifies poles on the host and needs '
                       'port_s11_freqs as concrete values; pass a numpy array built outside '
                       'the traced function')
            assert message in str(actual[1])
            print(f'ConcretizationTypeError: {message}', file=sys.stderr)
    elif kind == 'coax':
        _same_refusal(eager, actual, NotImplementedError,
                      'add_coaxial_port() is not wired into Simulation.forward().')
    elif kind in ('waveguide', 'msl'):
        _same_refusal(eager, actual, ValueError,
                      'forward(port_s11_freqs=...) computes S11 only for add_port(...)')
        assert ('waveguide ports use compute_waveguide_s_matrix()' if kind == 'waveguide'
                else 'MSL ports use compute_msl_s_matrix()') in str(eager[1])
    elif path == 'multi-graded':
        _same_refusal(eager, actual, NotImplementedError,
                      'forward(port_s11_freqs=...) is supported only on single-device forward paths')
    elif path == 'multi-uniform' and actual[2]:
        _success(eager)
        _refusal(actual, NotImplementedError,
                 'forward(distributed=True) on a uniform mesh cannot run inside jax.jit')
    else:
        _equal(eager, actual)
        if kind == 'wire' and path == 'graded':
            _graded_wire_gradient(sim)
