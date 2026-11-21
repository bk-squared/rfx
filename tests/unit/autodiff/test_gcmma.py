"""Conservative-trial witnesses with independent equations and literal hand values."""

from dataclasses import dataclass, field, replace
import importlib
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rfx.optimizers as optimizers
from rfx.errors import AccuracyWarning
from rfx.optimize import _run_optimizer
from rfx.optimizers import _GCMMA, _MMA, _Request, GCMMAInnerCapWarning, _gcmma_tolerance
from tests.unit.autodiff.test_mma_closed_form import _reference


_T1A_NO_CAP_MAX_NEWTON = 5000


@pytest.fixture(autouse=True)
def cpu():
    with jax.default_device(jax.devices('cpu')[0]):
        yield


@dataclass(frozen=True)
class RecordingGCMMA(_GCMMA):
    states: list = field(default_factory=list, compare=False)

    def tell(self, *args):
        state = super().tell(*args)
        self.states.append(state)
        return state


def conservative(records, width):
    """The independent approximation uses equations (4.2)-(4.5)."""
    assert records
    for record in records:
        if record.caps:
            continue
        x, g, low, upp, rho = record.x, record.gradient, record.low, record.upp, record.rho
        p = (upp - x)**2 * (1.001 * np.maximum(g, 0) + .001 * np.maximum(-g, 0) + rho[:, None] / width)
        q = (x - low)**2 * (.001 * np.maximum(g, 0) + 1.001 * np.maximum(-g, 0) + rho[:, None] / width)
        r = record.values - np.sum(p / (upp - x) + q / (x - low), axis=1)
        approximation = r + np.sum(p / (upp - record.accepted_x) + q / (record.accepted_x - low), axis=1)
        zero = (record.values == 0) & np.all(g == 0, axis=1)
        if record.stalls:
            approximation = record.values.copy()
        np.testing.assert_allclose(record.approximation, approximation, rtol=1e-12, atol=1e-12)
        assert np.all(zero | (approximation + record.tol >= record.accepted_values))


def test_t1a():
    """``e`` is T1's per-row summation error before the first GCMMA evaluation."""
    a, _, b, _, star, e_sum, _, safe, _ = _reference(1000, False)
    a32 = jnp.asarray(a, dtype=jnp.float32)

    def fun(x):
        return jnp.stack([jnp.sum((x - a32)**2), jnp.sum(x) - b[0]])

    fstar = math.fsum((star - a)**2)
    at_star = np.asarray(jax.jit(fun)(jnp.asarray(star, dtype=jnp.float32)))
    e = np.r_[abs(float(at_star[0]) - fstar), e_sum]
    opt = RecordingGCMMA(1, e=e, max_newton=_T1A_NO_CAP_MAX_NEWTON)
    x, history, visited, solves = _run_optimizer(
        opt, fun, jnp.full(1000, .5, dtype=jnp.float32), (0, 1),
        n_iters=200, m=1, jit=True, keep_iterates=True,
    )
    state = opt.states[-1]
    error = np.abs(np.asarray(x, dtype=np.float64) - star)
    assert np.max(error[safe]) <= 1e-5
    assert np.max(error) <= 1e-3
    assert math.fsum(np.asarray(x, dtype=np.float64)) - b[0] <= max(1e-6 * b[0], 10 * e_sum[0])
    assert np.all((x >= 0) & (x <= 1))
    assert state.caps_hit == state.inner_caps == 0
    assert len(history) == len(visited) == state.k == 200
    assert len(state.records) == 200 - int(state.trial is not None)
    print('T1a Newton maximum:', max(s.newton_steps_last for s in opt.states), flush=True)
    assert solves['forward'] == 200 + sum(r.inner_count for r in state.records)
    np.testing.assert_array_equal(x, state.records[-1].accepted_x)
    conservative(state.records, 1)


def test_t1c():
    center = np.array([.3, -.2, .1])
    vertices = center + np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    p32 = jnp.asarray(vertices, dtype=jnp.float32)

    def fun(x):
        return jnp.concatenate([jnp.zeros(1, dtype=x.dtype), jnp.sum((x - p32)**2, axis=1)])

    exact = np.array([math.fsum((center - p)**2) for p in vertices])
    at_star = np.asarray(jax.jit(fun)(jnp.asarray(center, dtype=jnp.float32)))
    e = np.r_[0, np.abs(at_star[1:] - exact)]
    opt = RecordingGCMMA(4, a=np.ones(4), e=e)
    x, history, _, _ = _run_optimizer(
        opt, fun, jnp.asarray(center + [.5, -.3, .2], dtype=jnp.float32),
        (center - 2, center + 2), n_iters=200, m=4, jit=True,
    )
    assert np.max(np.abs(x - center)) <= 1e-4
    assert np.max(np.sum((np.asarray(x, dtype=np.float64) - vertices)**2, axis=1)) - 1 <= 1e-4
    assert np.all((x >= center - 2) & (x <= center + 2))
    state = opt.states[-1]
    assert state.caps_hit == state.inner_caps == 0
    assert len(history) == state.k == 200
    assert len(state.records) == 200 - int(state.trial is not None)
    conservative(state.records, 4)


def test_monotone():
    with jax.enable_x64():
        fun = lambda x: jnp.sum(x**2 - jnp.cos(3 * x))
        opt = RecordingGCMMA(0, e=np.array([0.]))
        _, history, _, _ = _run_optimizer(
            opt, fun, jnp.array([1.7, -1.3, .4, 1.1, -1.9], dtype=jnp.float64),
            (-2, 2), n_iters=60, jit=True,
        )
        state = opt.states[-1]
        assert len(history) == state.k == 60
        assert state.caps_hit == state.inner_caps == 0
        assert len(state.records) >= 59
        for record in state.records:
            tau = 10 * np.spacing(abs(np.float64(record.values[0])))
            assert record.tol[0] == tau
            assert record.accepted_values[0] <= record.values[0] + tau
        conservative(state.records, 4)


def fake_solver(point):
    def solve(*args, **kwargs):
        return SimpleNamespace(x=np.array(point), y=np.zeros(len(args[8])), z=0., newton_steps=0, caps_hit=0)
    return solve


def test_hand_values(monkeypatch):
    monkeypatch.setattr(optimizers, '_mma_subproblem', fake_solver([.375, 1.5]))
    opt = _GCMMA(1, tol=np.zeros(2))
    state = opt.init(jnp.array([.25, 1.5], dtype=jnp.float32), ([0, 0], [.5, 3]))
    state = opt.tell(state, opt.ask(state), 3., np.array([2., -1.]), np.array([1.]), np.zeros((1, 2)))
    np.testing.assert_allclose(state.rho, [.2, 1e-6], rtol=1e-15)
    np.testing.assert_allclose(state.p, [[.150125, .15225], [.000000125, .00000075]], rtol=1e-14)
    np.testing.assert_allclose(state.q, [[.025125, 2.40225], [.000000125, .00000075]], rtol=1e-14)
    np.testing.assert_allclose(state.r, [.596, .999998], rtol=1e-14)
    distance = opt._distance(state, np.array([.375, 1.5]))
    assert distance == pytest.approx(1 / 3, rel=1e-15)
    delta = np.array([1 / 6, -1 / 3, 4, 0]) / distance
    np.testing.assert_allclose(delta, [.5, -1, 12, 0], rtol=1e-15)
    np.testing.assert_allclose(opt._update_rho(np.array([.2, .3, 1, .4]), delta), [.77, .3, 10, .4], rtol=1e-15)
    rejected = opt.tell(state, opt.ask(state), 0., None, np.array([1 + 1e-6 / 3 + 1 / 3]), None)
    np.testing.assert_allclose(rejected.rho, [.2, 1e-5], rtol=1e-14)
    assert rejected.k == 1 and rejected.inner_count == 1 and rejected.trial is not None
    accepted = opt.tell(rejected, opt.ask(rejected), 0., None, np.array([0.]), None)
    following = opt.tell(accepted, opt.ask(accepted), 3., np.array([2., -1.]), np.array([1.]), np.zeros((1, 2)))
    np.testing.assert_allclose(following.rho, [.02, 1e-6], rtol=1e-14)
    np.testing.assert_allclose(_gcmma_tolerance([1, 0], [0, .002], np.float32), [10 * 2**-23, .02], rtol=0, atol=0)
    with jax.enable_x64():
        np.testing.assert_array_equal(_gcmma_tolerance([1], [0], np.float64), [10 * 2**-52])
    zero_opt = _GCMMA(1, tol=np.array([-1., 0.]))
    zs = zero_opt.init(jnp.array([.25, 1.5], dtype=jnp.float32), ([0, 0], [.5, 3]))
    zs = zero_opt.tell(zs, zero_opt.ask(zs), 0., np.zeros(2), np.array([1.]), np.ones((1, 2)))
    zs = zero_opt.tell(zs, zero_opt.ask(zs), 0., None, np.array([-10.]), None)
    assert zs.trial is None and len(zs.records) == 1


@pytest.mark.parametrize('gradient,trial_value,tol,accepted', [
    ([0., 0.], 1., 0., False),
    ([1., 0.], -.1, -1., False),
    ([0., 0.], 0., -1., True),
], ids=['positive-trial', 'nonzero-gradient', 'nonpositive-trial'])
def test_zero_row_exemption(monkeypatch, gradient, trial_value, tol, accepted):
    monkeypatch.setattr(optimizers, '_mma_subproblem', fake_solver([.375, 1.5]))
    opt = _GCMMA(1, tol=np.array([0., tol]))
    state = opt.init(jnp.array([.25, 1.5], dtype=jnp.float32), ([0, 0], [.5, 3]))
    state = opt.tell(state, opt.ask(state), 1., np.zeros(2), np.array([0.]), np.array([gradient]))
    state = opt.tell(state, opt.ask(state), 1., None, np.array([trial_value]), None)
    assert (state.trial is None) == accepted
    assert len(state.records) == int(accepted)
    assert state.inner_count == 1 and state.inner_caps == 0


def test_rho_carry_over_after_rejection(monkeypatch):
    monkeypatch.setattr(optimizers, '_mma_subproblem', fake_solver([.375, 1.5]))
    opt = _GCMMA(1, tol=np.zeros(2))
    state = opt.init(jnp.array([.25, 1.5], dtype=jnp.float32), ([0, 0], [.5, 3]))
    state = opt.tell(state, opt.ask(state), 1., np.zeros(2), np.array([3.]), np.array([[2., -1.]]))
    rejected = opt.tell(state, opt.ask(state), 1., None, np.array([3.733666666666667]), None)
    np.testing.assert_allclose(rejected.rho, [1e-6, .77], rtol=1e-14, atol=0)
    assert rejected.trial is not None
    accepted = opt.tell(rejected, opt.ask(rejected), 0., None, np.array([0.]), None)
    assert accepted.trial is None
    following = opt.tell(accepted, opt.ask(accepted), 1., np.zeros(2), np.array([3.]), np.array([[2., -1.]]))
    np.testing.assert_allclose(following.rho, [1e-6, .077], rtol=1e-14, atol=0)


def test_delta_through_tell(monkeypatch):
    monkeypatch.setattr(optimizers, '_mma_subproblem', fake_solver([.375, 1.5]))
    opt = _GCMMA(0, tol=np.zeros(1))
    state = opt.init(jnp.array([.25, 1.5], dtype=jnp.float32), ([0, 0], [.5, 3]))
    state = opt.tell(state, opt.ask(state), 3., np.array([2., -1.]), None, None)
    rejected = opt.tell(state, opt.ask(state), 3.733666666666667, None, None, None)
    np.testing.assert_allclose(rejected.rho, [.77], rtol=1e-14, atol=0)
    assert rejected.trial is not None and rejected.inner_count == 1
    assert rejected.k == 1 and rejected.inner_caps == 0 and not rejected.records


def test_asymptotes(monkeypatch):
    calls = []
    solve = fake_solver([.6, .6, .6])

    def capture(*args, **kwargs):
        calls.append(tuple(v.copy() for v in args[:4]))
        return solve(*args, **kwargs)

    monkeypatch.setattr(optimizers, '_mma_subproblem', capture)
    points = ([.5, .5, .5], [.625, .625, .5], [.5, .75, .5])
    boxes = []
    for opt in (_MMA(0), _GCMMA(0, tol=np.zeros(1))):
        state = opt.init(jnp.array(points[0], dtype=jnp.float32), (0, 1))
        for k, point in enumerate(points, 1):
            request = _Request(jnp.array(point, dtype=jnp.float32), True, True)
            state = opt.tell(state, request, 3., np.array([1., -2., 0.]), None, None)
            outer_box = calls[-1]
            if isinstance(opt, _GCMMA):
                if k == 2:
                    old = state
                    state = opt.tell(state, opt.ask(state), 1e6, None, None, None)
                    for name in ('low', 'upp', 'alpha', 'beta', 'xold1', 'xold2'):
                        np.testing.assert_array_equal(getattr(state, name), getattr(old, name))
                    assert state.k == old.k and state.inner_count == 1
                state = opt.tell(state, opt.ask(state), -1e6, None, None, None)
        boxes.append(outer_box)
    for gcmma, mma in zip(boxes[1], boxes[0]):
        np.testing.assert_array_equal(gcmma, mma)


class TrialOptimizer:
    native_box = True

    def __init__(self):
        self.requests = []

    def init(self, x0, bounds):
        return SimpleNamespace(x=x0, trial=False)

    def ask(self, state):
        return _Request(state.x + .25 if state.trial else state.x, not state.trial, not state.trial)

    def tell(self, state, request, value, grad, cons_values, cons_grads):
        self.requests.append((request, grad, cons_grads))
        return SimpleNamespace(x=request.x if state.trial else state.x, trial=not state.trial)


@pytest.mark.parametrize('m,mode', [(0, 'rev'), (1, 'rev'), (1, 'fwd')])
def test_loop_trials(monkeypatch, capsys, m, mode):
    module = importlib.import_module('rfx.optimize')
    real = module._jit_with_fallback
    builds = []

    def fun(x):
        f = jnp.sum(x**2)
        return f if m == 0 else jnp.stack([f, f - 1])

    def build(fn, *args):
        if fn is fun:
            builds.append(fn)
        return real(fn, *args)

    monkeypatch.setattr(module, '_jit_with_fallback', build)
    opt, count = TrialOptimizer(), [0]
    x, history, visited, solves = _run_optimizer(
        opt, fun, jnp.zeros(2, dtype=jnp.float32), None, n_iters=3,
        m=m, jacobian=mode, jit=True, keep_iterates=True, it_count=count, verbose=True,
    )
    assert len(builds) == 1
    assert count == [3] and len(history) == len(visited) == 3
    np.testing.assert_array_equal(history, [0, .125, .5])
    np.testing.assert_array_equal(visited, [[0, 0], [.25, .25], [.5, .5]])
    np.testing.assert_array_equal(x, [.5, .5])
    assert len(opt.requests) == 5
    assert all(g is None and cg is None for req, g, cg in opt.requests if not req.iterate)
    assert solves == ({'forward': 2, 'reverse': 0, 'forward_mode': 6} if mode == 'fwd'
                      else {'forward': 5, 'reverse': 3 * (m + 1), 'forward_mode': 0})
    output = capsys.readouterr().out
    assert 'iter    0' in output and 'iter    2' in output and output.count('iter') == 2


@pytest.mark.parametrize('bad_row', [0, 1])
def test_nonfinite_trial(monkeypatch, bad_row):
    module = importlib.import_module('rfx.optimize')
    real = module._jit_with_fallback
    fun = lambda x: jnp.stack([jnp.sum(x**2), jnp.sum(x) - 1])
    trial_calls = []

    def build(fn, *args):
        if fn is fun:
            def values(x):
                trial_calls.append(x)
                v = fun(x)
                return v if len(trial_calls) == 1 else v.at[bad_row].set(jnp.nan)
            return values
        return real(fn, *args)

    monkeypatch.setattr(module, '_jit_with_fallback', build)
    opt, count = RecordingGCMMA(1, tol=np.array([100., 100.])), [0]
    with pytest.warns(AccuracyWarning, match='non-finite loss/gradient') as caught:
        x, history, visited, solves = _run_optimizer(
            opt, fun, jnp.array([.8, .7], dtype=jnp.float32), (0, 1),
            n_iters=4, m=1, jit=True, it_count=count, keep_iterates=True,
        )
    assert len(caught) == 1 and len(trial_calls) == 2
    assert len(history) == len(visited) == count[0] == 2
    assert solves == {'forward': 4, 'reverse': 4, 'forward_mode': 0}
    np.testing.assert_array_equal(x, visited[-1])
    np.testing.assert_array_equal(x, opt.states[-1].records[-1].accepted_x)
    assert not np.array_equal(x, trial_calls[-1])


def test_inner_cap(monkeypatch):
    monkeypatch.setattr(optimizers, '_mma_subproblem', fake_solver([.625]))
    opt = _GCMMA(0, tol=np.array([-1.]), _inner_cap=1)
    state = opt.init(jnp.array([.5], dtype=jnp.float32), (0, 1))
    for k in range(1, 4):
        req = _Request(jnp.array([.5], dtype=jnp.float32), True, True)
        state = opt.tell(state, req, 1., np.array([1.]), None, None)
        with pytest.warns(GCMMAInnerCapWarning, match=f'outer iteration {k}') as caught:
            state = opt.tell(state, opt.ask(state), 1., None, None, None)
        assert len(caught) == 1
        assert state.inner_caps == k and state.inner_count == 1
        assert state.records[-1].caps == 1 and state.trial is None
    assert len(state.records) == 3
    opt = _GCMMA(0, tol=np.array([-1.]))
    state = opt.init(jnp.array([.5], dtype=jnp.float32), (0, 1))
    state = opt.tell(state, opt.ask(state), 1., np.array([1.]), None, None)
    for _ in range(19):
        state = opt.tell(state, opt.ask(state), 1e30, None, None, None)
        assert state.trial is not None and state.inner_caps == 0
    with pytest.warns(GCMMAInnerCapWarning):
        state = opt.tell(state, opt.ask(state), 1e30, None, None, None)
    assert state.inner_count == 20 and state.inner_caps == 1


def test_stall(monkeypatch):
    """The solver point differs in host float64 but equals the accepted design in the design dtype."""
    monkeypatch.setattr(optimizers, '_mma_subproblem', fake_solver([.5 + 1e-10]))
    opt = RecordingGCMMA(0, tol=np.zeros(1))
    x, history, visited, solves = _run_optimizer(
        opt, lambda x: jnp.sum(x**2), jnp.array([.5], dtype=jnp.float32), (0, 1),
        n_iters=3, jit=True, keep_iterates=True,
    )
    state = opt.states[-1]
    assert state.stalls == 3 and state.inner_caps == 0
    assert state.inner_count == 0 and all(r.stalls == 1 and r.inner_count == 0 for r in state.records)
    assert len(history) == len(visited) == len(state.records) == 3
    assert solves == {'forward': 3, 'reverse': 3, 'forward_mode': 0}
    np.testing.assert_array_equal(x, [.5])
