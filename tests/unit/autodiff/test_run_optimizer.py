"""Private-loop witnesses; reference arithmetic does not use loop helpers."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.errors import AccuracyWarning, ConfigurationWarning
from rfx.optimize import _run_optimizer
from rfx.optimizers import _Adam, _MMA, _ProjectedAdam, _Request
from tests.unit.autodiff.test_mma_closed_form import _sum_value_grad


@pytest.fixture(autouse=True)
def _cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def test_projected_adam_box_and_plain_adam_identity():
    a = jnp.asarray(np.random.default_rng(0).uniform(-0.5, 1.5, 64), dtype=jnp.float32)
    x0 = jnp.full(64, 0.5, dtype=jnp.float32)
    evaluate = jax.value_and_grad(lambda x: jnp.sum((x - a)**2))
    plain, projected = _Adam(0.05), _ProjectedAdam(0.05)
    ps, bs = plain.init(x0, None), projected.init(x0, (0, 1))
    first_outside = None
    for iteration in range(1, 201):
        req = plain.ask(ps)
        value, grad = evaluate(req.x)
        ps = plain.tell(ps, req, value, grad, None, None)
        req = projected.ask(bs)
        value, grad = evaluate(req.x)
        bs = projected.tell(bs, req, value, grad, None, None)
        assert bs.x.dtype == x0.dtype and type(bs.it) is int
        assert np.all((np.asarray(bs.x) >= 0) & (np.asarray(bs.x) <= 1))
        if first_outside is None:
            if np.any((np.asarray(ps.x) < 0) | (np.asarray(ps.x) > 1)):
                first_outside = iteration
            else:
                np.testing.assert_array_equal(bs.x, ps.x)
                np.testing.assert_array_equal(bs.m, ps.m)
                np.testing.assert_array_equal(bs.v, ps.v)
    assert first_outside is not None and first_outside > 1
    assert np.any(a < -0.1) and np.any(a > 1.1)
    np.testing.assert_array_equal(np.asarray(bs.x)[a < -0.1], 0)
    np.testing.assert_array_equal(np.asarray(bs.x)[a > 1.1], 1)


def test_projected_adam_init_restart_and_runner():
    opt = _ProjectedAdam(0.05)
    x0 = jnp.asarray([-0.5, 0.5, 1.5], dtype=jnp.float32)
    assert opt.native_box is True
    with pytest.raises(ValueError, match="bounds"):
        opt.init(x0, None)
    bounds = (np.array([0, 0.1, 0.2]), np.array([0.8, 0.9, 1]))
    state = opt.init(x0, bounds)
    np.testing.assert_array_equal(state.x, np.array([0, 0.5, 1], dtype=np.float32))
    assert state.x.dtype == x0.dtype
    assert opt.restart(state) is state
    final, history, visited, counts = _run_optimizer(
        opt, lambda x: jnp.sum(x**2), x0, bounds, n_iters=4,
        jit=True, keep_iterates=True,
    )
    assert len(history) == len(visited) == 4
    np.testing.assert_array_equal(visited[0], state.x)
    assert np.all(final >= np.asarray(bounds[0], dtype=np.float32))
    assert np.all(final <= np.asarray(bounds[1], dtype=np.float32))
    assert counts == {"forward": 4, "reverse": 4, "forward_mode": 0}


def test_projected_adam_box_in_design_dtype():
    x0 = jnp.asarray([-0.1, 0.1, -0.1, 0.1], dtype=jnp.float32)
    expected_init = np.array([-0.09999999403953552, 0.09999999403953552,
                              -0.09999999403953552, 0.09999999403953552], dtype=np.float64)
    expected_final = np.array([0.09999999403953552, -0.09999999403953552,
                               0.09999999403953552, -0.09999999403953552], dtype=np.float64)
    final, history, visited, counts = _run_optimizer(
        _ProjectedAdam(0.05), lambda x: -x[0] + x[1] - x[2] + x[3],
        x0, (-0.1, 0.1), n_iters=30, m=0, jit=True, keep_iterates=True,
    )
    assert len(history) == len(visited) == 30
    np.testing.assert_array_equal(np.asarray(visited[0], dtype=np.float64), expected_init)
    np.testing.assert_array_equal(np.asarray(final, dtype=np.float64), expected_final)
    for point in [*visited, final]:
        assert point.dtype == jnp.float32
        point64 = np.asarray(point, dtype=np.float64)
        assert np.all((point64 >= -0.1) & (point64 <= 0.1))
    assert counts == {"forward": 30, "reverse": 30, "forward_mode": 0}


@pytest.mark.parametrize("bounds, message", [
    ((0.7, 0.70000001), "rounded box is empty"),
    ((0.5, 0.5), "strictly ordered bounds"),
    ((0.6, 0.5), "strictly ordered bounds"),
])
def test_projected_adam_refuses_invalid_box(bounds, message):
    with pytest.raises(ValueError, match=message):
        _ProjectedAdam(0.05).init(jnp.asarray([0.7], dtype=jnp.float32), bounds)


@pytest.mark.parametrize("bounds", [(-np.inf, 1), (0, np.inf), (np.nan, 1), (0, np.nan)])
def test_projected_adam_refuses_nonfinite_bounds(bounds):
    with pytest.raises(ValueError, match="finite bounds"):
        _ProjectedAdam(0.05).init(jnp.zeros((2, 3), dtype=jnp.float32), bounds)


def test_projected_adam_broadcasts_bounds():
    state = _ProjectedAdam(0.05).init(
        jnp.asarray([[-1, 2], [-1, 2]], dtype=jnp.float32),
        (np.array([-0.1, 0.7]), 0.9),
    )
    assert state.lower.shape == state.upper.shape == state.x.shape == (2, 2)
    assert state.lower.dtype == state.upper.dtype == state.x.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(state.lower, dtype=np.float64),
                                  [[-0.09999999403953552, 0.7000000476837158]] * 2)
    np.testing.assert_array_equal(np.asarray(state.upper, dtype=np.float64),
                                  [[0.8999999761581421, 0.8999999761581421]] * 2)
    np.testing.assert_array_equal(np.asarray(state.x, dtype=np.float64),
                                  [[-0.09999999403953552, 0.8999999761581421]] * 2)


def test_run_optimizer_refuses_bounds_for_adam_before_init(monkeypatch):
    def unexpected_init(*args):
        pytest.fail("init must not run for an unsupported box")

    monkeypatch.setattr(_Adam, "init", unexpected_init)
    with pytest.raises(ValueError, match="native_box"):
        _run_optimizer(_Adam(0.05), lambda x: jnp.sum(x**2),
                       jnp.zeros(1, dtype=jnp.float32), (0, 1), n_iters=1, jit=False)


def test_run_optimizer_refuses_trial_request_before_evaluation():
    class TrialOptimizer(_Adam):
        def ask(self, state):
            return _Request(state.x, need_grad=True, iterate=False)

    def unexpected_evaluation(x):
        pytest.fail("a trial request must not be evaluated")

    with pytest.raises(NotImplementedError, match="iterate=True"):
        _run_optimizer(TrialOptimizer(0.05), unexpected_evaluation,
                       jnp.zeros(1, dtype=jnp.float32), None, n_iters=1, jit=False)


def test_run_optimizer_refuses_value_only_request_before_evaluation():
    class ValueOnlyOptimizer(_Adam):
        def ask(self, state):
            return _Request(state.x, need_grad=False, iterate=True)

    def unexpected_evaluation(x):
        pytest.fail("a value-only request must not be evaluated")

    with pytest.raises(NotImplementedError, match="need_grad=True"):
        _run_optimizer(ValueOnlyOptimizer(0.05), unexpected_evaluation,
                       jnp.zeros(1, dtype=jnp.float32), None, n_iters=1, jit=False)


def _closed_form_test_t1a_loop(a32, x0):
    """Evaluation/update block from test_mma_closed_form.test_bounded_closed_form.

    T1a only; retain its separately jitted scalar evaluations and stacked
    constraints verbatim, recording f0 instead of the closed-form KKT checks.
    """
    b = np.array([0.3 * x0.size])
    objective = jax.jit(jax.value_and_grad(lambda x: jnp.sum((x - a32)**2)))
    opt = _MMA(m=len(b))
    state = opt.init(x0, (0, 1))
    history = []
    # The unused two-constraint branch is omitted from the T1a reference.
    for iteration in range(1, 200 + 1):
        req = opt.ask(state)
        assert req.need_grad and req.iterate and req.x.dtype == jnp.float32
        value, grad = objective(req.x)
        evaluations = [_sum_value_grad(req.x)]
        cons = jnp.stack([ev[0] for ev in evaluations]) - jnp.asarray(b, dtype=jnp.float32)
        cgrad = jnp.stack([ev[1] for ev in evaluations])
        state = opt.tell(state, req, value, grad, cons, cgrad)
        history.append(float(value))
    return state.x, history


def test_mma_t1a_matches_closed_form_test_loop_bit_for_bit():
    a = jnp.asarray(np.random.default_rng(0).uniform(-0.5, 1.5, 1000), dtype=jnp.float32)
    x0 = jnp.full(1000, 0.5, dtype=jnp.float32)

    def fun(x):
        return jnp.stack([jnp.sum((x - a)**2), jnp.sum(x) - 300])

    expected_x, expected_history = _closed_form_test_t1a_loop(a, x0)
    x, history, visited, counts = _run_optimizer(
        _MMA(m=1), fun, x0, (0, 1), n_iters=200, m=1,
        jacobian="rev", jit=True,
    )
    np.testing.assert_array_equal(x, expected_x)
    np.testing.assert_array_equal(history, expected_history)
    assert visited is None
    assert counts == {"forward": 200, "reverse": 400, "forward_mode": 0}


class _RecordingOptimizer:
    """An exact +1 step makes restoration and derivative routing observable."""
    native_box = False

    def __init__(self):
        self.evaluations = []

    def init(self, x0, bounds):
        return SimpleNamespace(x=x0)

    def ask(self, state):
        return _Request(state.x, need_grad=True, iterate=True)

    def tell(self, state, request, value, grad, cons_values, cons_grads):
        self.evaluations.append((value, grad, cons_values, cons_grads))
        return SimpleNamespace(x=request.x + 1)

    def restart(self, state):
        return state


def test_nan_constraint_restores_last_finite_iterate():
    evaluations = 0

    def fun(x):
        nonlocal evaluations
        evaluations += 1
        constraint = jnp.asarray(jnp.nan if evaluations >= 3 else 0, dtype=x.dtype)
        # The NaN is constant: both derivative rows remain finite.
        return jnp.stack([jnp.sum(x**2), constraint])

    opt, it_count = _RecordingOptimizer(), [0]
    with pytest.warns(AccuracyWarning, match="witness: non-finite loss/gradient .*iteration 2"):
        x, history, visited, counts = _run_optimizer(
            opt, fun, jnp.array([0.5], dtype=jnp.float32), None,
            n_iters=5, m=1, jit=False, it_count=it_count,
            keep_iterates=True, name="witness",
        )
    assert evaluations == 3 and len(opt.evaluations) == 2
    np.testing.assert_array_equal(x, np.array([1.5], dtype=np.float32))
    assert history == [0.25, 2.25] and it_count == [2]
    np.testing.assert_array_equal(visited, [[0.5], [1.5]])
    # The failed evaluation still consumed its transformation's solves.
    assert counts == {"forward": 3, "reverse": 6, "forward_mode": 0}


@pytest.mark.parametrize("m, mode, expected", [
    (0, "rev", {"forward": 3, "reverse": 3, "forward_mode": 0}),
    (2, "rev", {"forward": 3, "reverse": 9, "forward_mode": 0}),
    (2, "fwd", {"forward": 0, "reverse": 0, "forward_mode": 15}),
])
@pytest.mark.parametrize("jit", [False, True])
def test_solve_counter(m, mode, expected, jit):
    def fun(x):
        f = jnp.sum(x**2)
        return f if m == 0 else jnp.stack([f, jnp.sum(x), jnp.sum(2 * x)])

    _, history, visited, counts = _run_optimizer(
        _RecordingOptimizer(), fun, jnp.zeros(5, dtype=jnp.float32), None,
        n_iters=3, m=m, jacobian=mode, jit=jit,
    )
    assert len(history) == 3 and visited is None
    assert counts == expected


@pytest.mark.parametrize("jit", [False, True])
def test_vector_jacobians_match_hand_derivative(jit):
    x0 = jnp.array([0.2, -0.4, 0.7], dtype=jnp.float32)

    def fun(x):
        return jnp.stack([jnp.sum(x**2), x[0] * x[1] + 3 * x[2],
                          jnp.sin(x[0]) + x[1]**3 - x[2]])

    x = np.asarray(x0)
    expected = np.array([2 * x, [x[1], x[0], 3],
                         [np.cos(x[0]), 3 * x[1]**2, -1]], dtype=np.float32)
    jacobians = []
    for mode in ("rev", "fwd"):
        opt = _RecordingOptimizer()
        _run_optimizer(opt, fun, x0, None, n_iters=1, m=2, jacobian=mode, jit=jit)
        value, grad, cons, cgrad = opt.evaluations[0]
        np.testing.assert_allclose(jnp.concatenate([jnp.atleast_1d(value), cons]), fun(x0), rtol=1e-6)
        jacobian = np.concatenate([np.asarray(grad)[None], np.asarray(cgrad)])
        np.testing.assert_allclose(jacobian, expected, rtol=1e-6, atol=0)
        jacobians.append(jacobian)
    np.testing.assert_allclose(*jacobians, rtol=1e-6, atol=0)


@pytest.mark.parametrize("mode", ["rev", "fwd"])
def test_vector_first_call_host_read_falls_back_once(mode):
    def fun(x):
        factor = float(jnp.asarray(2.0))
        return jnp.stack([factor * jnp.sum(x**2), jnp.sum(x) - 1])

    results = []
    for jit in (False, True):
        opt = _RecordingOptimizer()
        if jit:
            with pytest.warns(ConfigurationWarning, match="objective reads a traced value") as caught:
                result = _run_optimizer(opt, fun, jnp.ones(3, dtype=jnp.float32), None,
                                        n_iters=3, m=1, jacobian=mode, jit=jit)
            assert len(caught) == 1
        else:
            result = _run_optimizer(opt, fun, jnp.ones(3, dtype=jnp.float32), None,
                                    n_iters=3, m=1, jacobian=mode, jit=jit)
        results.append(result)
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])


@pytest.mark.parametrize("mode", ["rev", "fwd"])
def test_nonfinite_constraint_jacobian_stops(mode):
    def fun(x):
        return jnp.stack([jnp.sum(x**2), jnp.sqrt(x[0])])

    opt, it_count = _RecordingOptimizer(), [0]
    x0 = jnp.zeros(1, dtype=jnp.float32)
    with pytest.warns(AccuracyWarning, match="iteration 0"):
        x, history, visited, _ = _run_optimizer(
            opt, fun, x0, None, n_iters=3, m=1, jacobian=mode,
            jit=True, keep_iterates=True, it_count=it_count,
        )
    np.testing.assert_array_equal(x, x0)
    assert history == visited == opt.evaluations == [] and it_count == [0]
