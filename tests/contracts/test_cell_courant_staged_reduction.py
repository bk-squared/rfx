"""Device screening preserves the float64 cell Courant verdict and message.

Rows a/d count real host executions and compare gradient bytes. Row c observes
what the product code actually reduces, while independent float64 arithmetic
and plain calls supply the expected failures and messages for every ULP lane.
"""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, DebyePole, Simulation
from rfx.core.yee import MaterialArrays
from rfx.errors import RfxRefusal
from rfx.model import pole_stability
import rfx._traced_check as traced

C = 299792458.0
SHAPE = (6, 4, 4)
VARIANTS = ('plain', 'mu-half', 'mu-two', 'box', 'order4', 'conservative',
            'conformal', 'periodic', 'terminal', 'components', 'x-range')


def _grid(graded=False):
    widths = tuple(np.full(n, .001, np.float64) for n in SHAPE)
    if graded:
        widths = tuple(w * np.linspace(1., 1.4, w.size) for w in widths)
    return SimpleNamespace(cells=lambda a: widths[a], dt=.99e-3 / (C * np.sqrt(3.)),
                           periodic_axes=''), widths


def _materials(eps, mu=1.):
    return SimpleNamespace(eps_r=eps, mu_r=mu)


def _plain_message(eps, grid, kwargs, mu=1., box=False):
    kwargs = dict(kwargs)
    if box:
        kwargs['design_box'] = SimpleNamespace(bounds=(0, eps.shape[0], 0, eps.shape[1], 0, eps.shape[2]),
                                               eps_r=eps)
        eps = np.full(eps.shape, 2., np.float32)
    try:
        pole_stability.refuse_subluminal_cells(_materials(eps, mu), grid, **kwargs)
    except RfxRefusal as exc:
        return str(exc)
    return None


def _record_hosts(monkeypatch, *, swallow=False):
    original = traced.traced_check
    calls = []

    def stage(feature, name, check, *operands, **kwargs):
        assert (feature, name) == ('cell Courant limit', 'time-step stability')
        assert kwargs == dict(ordered=True)

        def host(*values):
            message = None
            try:
                check(*values)
            except RfxRefusal as exc:
                message = str(exc)
                if not swallow:
                    raise
            finally:
                calls.append((np.asarray(values[0][0] if isinstance(values[0], (tuple, list))
                                         else values[0]).copy(), message))

        return original(feature, name, host, *operands, **kwargs)

    monkeypatch.setattr(traced, 'traced_check', stage)
    return calls


def _loss(grid, kwargs=None):
    def loss(eps):
        pole_stability.refuse_subluminal_cells(_materials(eps), grid, **(kwargs or {}))
        return jnp.mean(jnp.sin(eps) * eps)
    return loss


@pytest.mark.parametrize('eps', [2., 1., .99])
@pytest.mark.parametrize('form', ['jit', 'jit-grad', 'jit-vmap', 'vmap'])
def test_a_admitted_host_calls(monkeypatch, eps, form):
    grid, _ = _grid()
    calls = _record_hosts(monkeypatch)
    value = jnp.full(SHAPE, eps, jnp.float32)
    loss = _loss(grid)
    if form == 'jit':
        program, argument, expected = jax.jit(loss), value, 0
    elif form == 'jit-grad':
        program, argument, expected = jax.jit(jax.grad(loss)), value, 0
    else:
        program = jax.vmap(loss)
        if form == 'jit-vmap':
            program = jax.jit(program)
        argument = jnp.stack([value] * 4)
        expected = 4
    executions = []
    count_nonzero = np.count_nonzero

    def count_host(value, *args, **kwargs):
        executions.append(1)
        return count_nonzero(value, *args, **kwargs)

    with patch.object(np, 'count_nonzero', count_host):
        jax.block_until_ready(program(argument))
        jax.effects_barrier()
    assert len(executions) == expected, (form, eps, len(executions))
    assert len(calls) == (0 if form == 'vmap' else expected)
    print(f'HOST_CALLS form={form} eps={eps} calls={len(executions)} lanes={4 if "vmap" in form else 1}', file=sys.stderr)
    if form == 'jit':
        assert 'cond[' in str(jax.make_jaxpr(loss)(value))


@pytest.mark.parametrize('eps', [.5, .8])
def test_b_refused_jit_and_each_mixed_batch_lane(monkeypatch, eps):
    grid, _ = _grid()
    value = jnp.full(SHAPE, eps, jnp.float32)
    expected = _plain_message(np.asarray(value), grid, {})
    assert expected is not None
    calls = _record_hosts(monkeypatch)
    loss = _loss(grid)
    with pytest.raises(Exception) as caught:
        jax.block_until_ready(jax.jit(loss)(value))
    assert expected in str(caught.value)
    assert calls[-1][1] == expected
    for lane in range(4):
        calls.clear()
        batch = jnp.full((4, *SHAPE), 2., jnp.float32).at[lane].set(value)
        with pytest.raises(Exception) as caught:
            jax.block_until_ready(jax.jit(jax.vmap(loss))(batch))
        assert expected in str(caught.value)
        assert [message for _, message in calls if message is not None] == [expected]


@pytest.mark.parametrize('dtype', [jnp.float16, jnp.bfloat16])
def test_b_low_precision_refused_jit(monkeypatch, dtype):
    grid, _ = _grid()
    value = jnp.full(SHAPE, .5, dtype)
    expected = _plain_message(np.asarray(value, dtype=np.float64), grid, {})
    assert expected is not None
    calls = _record_hosts(monkeypatch)
    with pytest.raises(Exception) as caught:
        jax.block_until_ready(jax.jit(_loss(grid))(value))
    assert expected in str(caught.value)
    assert calls[-1][1] == expected


def _host_failed(eps, mu, widths, step, *, periodic, factor, conservative, x_range):
    product = np.asarray(eps, np.float64) * np.asarray(mu, np.float64)
    local = np.zeros(product.shape, np.float64)
    for axis, raw in enumerate(widths):
        sizes = np.atleast_1d(np.asarray(raw, np.float64))
        size = product.shape[axis] if axis or x_range is None else x_range[1]
        if sizes.size == size - 1:
            sizes = np.append(sizes, sizes[-1])
        previous, following = np.roll(sizes, 1), np.roll(sizes, -1)
        if not periodic[axis]:
            previous[0], following[-1] = sizes[0], sizes[-1]
        near = np.minimum(sizes, np.minimum(previous, following))
        rows = slice(*x_range) if axis == 0 and x_range is not None else slice(None)
        shape = [1] * product.ndim
        shape[axis] = near[rows].size
        local += C**2 / near[rows].reshape(shape)**2
    local *= factor
    if conservative is not None and conservative != 'conformal':
        local = np.full(product.shape, float(conservative))
    failed = (product < 1.) & (float(step)**2 * local > product * (1. + 1e-6))
    if conservative == 'conformal':
        failed = product < 1.
    return failed, local


def _ulp_lanes(limit):
    middle = np.asarray(limit, np.float32)
    lanes = []
    for k in range(-8, 9):
        value = middle.copy()
        for _ in range(abs(k)):
            value = np.nextafter(value, np.float32(-np.inf if k < 0 else np.inf))
        lanes.append(value)
    return np.stack(lanes)


def _observed_masks(program, values):
    reduced = []
    original_sum = jnp.sum

    def observe(value, *args, **kwargs):
        result = original_sum(value, *args, **kwargs)
        if value.dtype == jnp.bool_:
            reduced.append((value, result))
        return result

    def run(value):
        program(value)
        return reduced[-1][0]

    with patch.object(jnp, 'sum', observe):
        return np.asarray(jax.jit(jax.vmap(run))(jnp.asarray(values)))


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('epsilon', [.25, .5, .75, 1.])
def test_c_float64_failures_are_inside_device_mask(monkeypatch, graded, variant, epsilon):
    grid, widths = _grid(graded)
    mu = .5 if variant == 'mu-half' else 2. if variant == 'mu-two' else 1.
    periodic = (variant == 'periodic',) * 3
    factor = (9 / 8 + 1 / 24)**2 if variant == 'order4' else 1.
    conservative = float(C**2 * 3 / .001**2) if variant == 'conservative' else 'conformal' if variant == 'conformal' else None
    x_range = (2, SHAPE[0]) if variant == 'x-range' else None
    shape = (SHAPE[0] - 2, *SHAPE[1:]) if x_range else SHAPE
    if variant == 'terminal':
        widths = tuple(w[:-1] for w in widths)
        grid.cells = lambda a: widths[a]
    formula = dict(periodic=periodic, factor=factor, conservative=conservative, x_range=x_range)
    _, local = _host_failed(np.ones(shape), mu, widths, grid.dt, **formula)
    step = np.sqrt(epsilon * mu * (1. + 1e-6) / np.max(local))
    limit = step**2 * local / (1. + 1e-6) / mu
    if conservative == 'conformal':
        limit = np.full(shape, 1. / mu)
    lanes = _ulp_lanes(limit)
    kwargs = dict(dt=step, periodic=periodic, stencil_order=4 if variant == 'order4' else 2,
                  conservative=conservative, x_range=x_range)
    expected_messages = [_plain_message(v, grid, kwargs, mu, box=variant == 'box') for v in lanes]
    expected_failed = np.stack([_host_failed(v, mu, widths, step, **formula)[0] for v in lanes])
    calls = _record_hosts(monkeypatch, swallow=True)

    def program(eps):
        options = dict(kwargs)
        if variant == 'box':
            options['design_box'] = SimpleNamespace(bounds=(0, shape[0], 0, shape[1], 0, shape[2]), eps_r=eps)
            eps = jnp.full(shape, 2., jnp.float32)
        if variant == 'components':
            eps = (eps, eps + jnp.float32(.125), eps + jnp.float32(.25))
        pole_stability.refuse_subluminal_cells(_materials(eps, mu), grid, **options)

    actual = _observed_masks(program, lanes)
    assert actual.shape == expected_failed.shape
    assert np.all(actual[expected_failed]), (graded, variant, epsilon, np.argwhere(expected_failed & ~actual))
    if np.min(limit) * mu <= 1.:
        assert np.any(expected_failed)
    assert len(calls) == 17
    assert [message for _, message in calls] == expected_messages


@pytest.mark.parametrize('pole', ['debye', 'lorentz'])
def test_c_pole_eps_inf_comes_from_material_assembly(monkeypatch, pole):
    from rfx.model.materials import assemble_cells
    from rfx.materials.lorentz import LorentzPole

    sim = Simulation(10e9, (.006,) * 3, dx=.001, boundary='pec')
    poles = dict(debye_poles=[DebyePole(.1, 1e-8)]) if pole == 'debye' else dict(lorentz_poles=[LorentzPole(1e6, 1e5, 1e8)])
    grid = sim._build_grid()
    widths = tuple(np.asarray(grid.cells(a), np.float64) for a in range(3))
    shape = tuple(grid.shape)
    formula = dict(periodic=(False,) * 3, factor=1., conservative=None, x_range=None)
    _, local = _host_failed(np.ones(shape), 1., widths, grid.dt, **formula)
    step = np.sqrt(.5 * (1. + 1e-6) / np.max(local))
    epsilon_lanes = _ulp_lanes(np.float64(.5))
    materials = []
    with patch.object(pole_stability, 'refuse_unstable_cells', lambda *a, **k: None):
        for eps in epsilon_lanes:
            sim.add_material('body', eps_r=float(eps), **poles)
            if not materials:
                sim.add(Box((0.,) * 3, (.006,) * 3), material='body')
            mats, debye, lorentz, *_ = assemble_cells(sim, grid)
            assert (debye if pole == 'debye' else lorentz) is not None
            assert np.any(np.asarray(mats.eps_r) == eps)
            materials.append(np.asarray(mats.eps_r))
    lanes = np.stack(materials)
    expected = [_plain_message(v, grid, dict(dt=step)) for v in lanes]
    failed = np.stack([_host_failed(v, 1., widths, step, **formula)[0] for v in lanes])
    calls = _record_hosts(monkeypatch, swallow=True)
    masks = _observed_masks(lambda eps: pole_stability.refuse_subluminal_cells(_materials(eps), grid, step), lanes)
    assert np.all(masks[failed])
    assert np.any(failed)
    assert [message for _, message in calls] == expected


def test_d_jit_gradient_is_bit_identical_without_check(monkeypatch):
    from rfx.sources.sources import CustomWaveform

    sim = Simulation(10e9, (.008,) * 3, dx=.001, boundary='pec')
    sim.add_source((.003, .004, .004), 'ez', amplitude_kind='field',
                   waveform=CustomWaveform(lambda t: jnp.where(t == 0., 1., 0.)))
    sim.add_probe((.005, .004, .004), 'ez')
    eps = jnp.full(sim._build_grid().shape, 2., jnp.float32)

    def loss(eps):
        return jnp.sum(sim.forward(eps_override=eps, n_steps=12, checkpoint=False,
                                   skip_preflight=True).time_series**2)

    calls = _record_hosts(monkeypatch)
    actual = np.asarray(jax.jit(jax.grad(loss))(eps))
    jax.effects_barrier()
    assert calls == []
    monkeypatch.setattr(pole_stability, 'refuse_subluminal_cells', lambda *a, **k: None)
    expected = np.asarray(jax.jit(jax.grad(loss))(eps))
    assert np.any(expected != 0.)
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def test_x_range_sharded_jit_reduces_globally(monkeypatch):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    devices = jax.devices('cpu')[:2]
    assert len(devices) == 2
    mesh = Mesh(np.array(devices), ('x',))
    sharding = NamedSharding(mesh, P('x'))
    grid, _ = _grid()
    values = jnp.full((4, 4, 4), 2., jnp.float32).at[-1].set(.5)
    monkeypatch.setattr(traced, 'traced_check', lambda *a, **k: None)
    sums = []
    original_sum = jnp.sum

    def observe(value, *args, **kwargs):
        result = original_sum(value, *args, **kwargs)
        if value.dtype == jnp.bool_:
            sums.append(result)
        return result

    def run(eps):
        pole_stability.refuse_subluminal_cells(_materials(eps), grid, x_range=(2, 6))
        return sums[-1]

    with patch.object(jnp, 'sum', observe):
        program = jax.jit(run, in_shardings=sharding)
        argument = jax.device_put(values, sharding)
        result = program(argument)
        assert int(result) == 16
        assert result.sharding.is_fully_replicated
        executable = program.lower(argument).compile().as_text().lower()
    assert 'all-reduce' in executable
    print('X_RANGE_SUM global=16 replicated=True all-reduce=True devices=2', file=sys.stderr)


def test_x_range_out_of_program_returns_before_reduction():
    grid, _ = _grid()

    def run(eps):
        pole_stability.refuse_subluminal_cells(_materials(eps), grid,
                                              x_range=(2, 6), in_program=False)
        return eps

    jaxpr = str(jax.make_jaxpr(run)(jnp.ones((4, 4, 4), jnp.float32)))
    assert 'cond[' not in jaxpr
    assert 'reduce_sum' not in jaxpr
    assert 'callback' not in jaxpr
