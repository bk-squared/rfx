"""Closed-form witnesses for the public few-parameter minimiser."""

from types import SimpleNamespace
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rfx
import rfx._minimize as implementation
import rfx.optimizers as optimizers
from rfx.optimize import _run_optimizer
from rfx.optimizers import _Adam


@pytest.fixture(autouse=True)
def cpu_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


@pytest.mark.parametrize("bounded,jit", [(False, False), (True, True)])
def test_adam_matches_loop(bounded, jit):
    x0 = [0.0, 0.75]
    bounds = [(0, 1), (-1, 1)] if bounded else None
    fun = lambda x: jnp.sum((x - jnp.array([0.6, -0.2])) ** 2)
    start = jnp.asarray(x0, dtype=jnp.float32)
    physical = lambda point: point
    if bounded:
        lo = jnp.array([0.0, -1.0])
        hi = jnp.array([1.0, 1.0])
        p0 = (np.array(x0, dtype=np.float64) - np.array([0, -1])) / np.array([1, 2])
        with np.errstate(divide="ignore"):
            start = jnp.asarray(np.clip(np.log(p0 / (1 - p0)), -13.8, 13.8), dtype=jnp.float32)

        def physical(point):
            raw = lo + (hi - lo) * jax.nn.sigmoid(point)
            return raw + jax.lax.stop_gradient(jnp.clip(raw, lo, hi) - raw)

    _, history, visited, solves = _run_optimizer(
        _Adam(0.01), lambda point: fun(physical(point)), start, None,
        n_iters=6, jit=jit, keep_iterates=True,
    )
    result = rfx.minimize(fun, x0, bounds=bounds, n_iters=6, jit=jit)
    assert isinstance(result, rfx.MinimizeResult)
    assert {"minimize", "MinimizeResult", "MMASubproblemCapWarning"} <= set(rfx.__all__)
    np.testing.assert_array_equal(result.history, history)
    np.testing.assert_array_equal(result.x_history, jnp.stack([physical(p) for p in visited]))
    np.testing.assert_array_equal(result.x, physical(visited[-1]))
    assert result.solves == solves == dict(forward=6, reverse=6, forward_mode=0)
    assert result.jacobian == "value_and_grad"
    assert result.method == "adam" and result.scale is None and result.setup_forwards == 0


def test_projected_bounds_convention_and_rounding():
    for bounds, x0, target, expected in [
        (((0, 1), (2, 3)), (0.5, 2.5), (1.5, 1.5), (1, 2)),
        (((-1, 0.7), (-1, 0.7)), (0, 0), (2, 2), (0.7, 0.7)),
    ]:
        seen = []

        def fun(x):
            # Capture primal evaluations as well as the returned history.
            jax.debug.callback(lambda value: seen.append(np.asarray(value)), x)
            return jnp.sum((x - jnp.asarray(target)) ** 2)

        result = rfx.minimize(fun, x0, bounds=bounds, method="PROJECTED_ADAM",
                              lr=0.1, n_iters=50, jit=False)
        result.x.block_until_ready()
        points = np.asarray(result.x_history, dtype=np.float64)
        pairs = np.asarray(bounds)
        assert np.all(points >= pairs[:, 0]) and np.all(points <= pairs[:, 1])
        assert np.all(np.asarray(seen, dtype=np.float64) >= pairs[:, 0])
        assert np.all(np.asarray(seen, dtype=np.float64) <= pairs[:, 1])
        np.testing.assert_array_equal(result.x, np.asarray(expected, dtype=np.float32))
        assert result.method == "projected_adam"
    broadcast = rfx.minimize(lambda x: jnp.sum(x**2), [0, 1],
                             bounds=SimpleNamespace(lb=0, ub=[1, 2]),
                             method="projected_adam", n_iters=1, jit=False)
    np.testing.assert_array_equal(broadcast.x, [0, 1])
    assert broadcast.x.dtype == jnp.float32

    # The first lower bound and second upper bound round outward in float32.
    bounds = ((0.7, 1.3), (-0.9, 0.1))
    pairs = np.asarray(bounds, dtype=np.float64)
    for method in ("adam", "projected_adam", "mma"):
        options = {} if method == "mma" else dict(lr=1.0 if method == "adam" else 0.1)
        for jit in (False, True):
            for target in ((0.0, -2.0), (2.0, 1.0)):
                seen = []

                def fun(x):
                    jax.debug.callback(lambda value: seen.append(np.asarray(value).copy()), x)
                    return jnp.sum((x - jnp.asarray(target)) ** 2)

                result = rfx.minimize(fun, [1.0, -0.4], bounds=bounds, method=method,
                                      n_iters=40, jit=jit, **options)
                jax.effects_barrier()
                points = np.stack(seen).astype(np.float64)
                assert points.shape == (40, 2)
                assert np.all(points >= pairs[:, 0]), (method, jit, target, points.min(axis=0))
                assert np.all(points <= pairs[:, 1]), (method, jit, target, points.max(axis=0))
                history = np.asarray(result.x_history, dtype=np.float64)
                assert np.all(history >= pairs[:, 0]) and np.all(history <= pairs[:, 1])


def test_mma_worst_case_circumcentre():
    with jax.enable_x64():
        vertices = np.array([[0, 0], [1, 0.1], [0.35, 0.9]], dtype=np.float64)
        centre = np.linalg.solve(2 * (vertices[1:] - vertices[0]),
                                 np.sum(vertices[1:]**2, axis=1) - np.sum(vertices[0]**2))
        radius2 = np.sum((centre - vertices[0])**2)
        fun = lambda x: jnp.sum((x - jnp.asarray(vertices))**2, axis=1)
        result = rfx.minimize(fun, [1.3, -0.4], bounds=[(-1, 2)] * 2,
                              method="mma", worst_case=True, n_iters=40)
        np.testing.assert_allclose(result.x, centre, rtol=0, atol=1e-4)
        assert abs(result.fun - radius2) <= 1e-6
        expected = np.max(np.sum((np.asarray(result.x_history)[:, None, :] - vertices)**2, axis=2), axis=1)
        np.testing.assert_allclose(result.history, expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(result.scale, np.max(np.sum((np.array([1.3, -0.4]) - vertices)**2, axis=1)),
                                   rtol=1e-15, atol=0)
        assert result.setup_forwards == 1 and result.nfev == 40
        assert result.subproblem_caps == 0 and not result.stopped_nonfinite
        best = rfx.minimize(fun, [1.3, -0.4], bounds=[(-1, 2)] * 2,
                            method="mma", worst_case=True, n_iters=40, best_iterate=True)
        index = int(np.argmin(expected))
        assert index != 0
        np.testing.assert_array_equal(best.x, result.x_history[index])
        assert best.fun == min(best.history)
        assert best.x.dtype == jnp.float64


@pytest.mark.parametrize("case,kwargs,message", [
    ("scalar", {"method": "BFGS"}, "scipy.optimize.minimize"),
    ("scalar", {"method": None}, "method"),
    ("scalar", {"worst_case": True}, 'method="mma"'),
    ("scalar", {"method": "projected_adam", "worst_case": True}, 'method="mma"'),
    ("scalar", {"method": "projected_adam"}, "bounds"),
    ("scalar", {"method": "mma"}, "bounds"),
    ("scalar", {"method": "mma", "bounds": [(0, 1)], "lr": 0.1}, "lr"),
    ("scalar", {"n_iters": 0}, "n_iters"),
    ("scalar", {"n_iters": -1}, "n_iters"),
    ("scalar", {"n_iters": 1.5}, "n_iters"),
    ("scalar", {"x0": [[0.5]]}, "x0"),
    ("scalar", {"x0": []}, "x0"),
    ("scalar", {"x0": [np.nan]}, "x0"),
    ("scalar", {"x0": [np.inf]}, "x0"),
    ("scalar", {"x0": [1 + 1e-8], "bounds": [(0, 1)]}, "x0"),
    ("scalar", {"x0": [-1e-8], "bounds": [(0, 1)]}, "x0"),
    ("scalar", {"bounds": [(0, np.inf)]}, "bounds"),
    ("scalar", {"bounds": [(np.nan, 1)]}, "bounds"),
    ("scalar", {"bounds": [(None, 1)]}, "bounds"),
    ("scalar", {"bounds": [(1, 1)]}, "bounds"),
    ("scalar", {"bounds": [(1, 0)]}, "bounds"),
    ("scalar", {"bounds": [(0, 1), (0, 1)]}, "bounds"),
    ("scalar", {"bounds": [0, 1]}, "bounds"),
    ("scalar", {"x0": [0.7], "bounds": [(0.7, 0.70000001)]}, "bounds"),
    ("vector", {}, "fun"),
    ("scalar", {"method": "mma", "bounds": [(0, 1)], "worst_case": True}, "fun"),
    ("matrix", {"method": "mma", "bounds": [(0, 1)], "worst_case": True}, "fun"),
    ("empty", {"method": "mma", "bounds": [(0, 1)], "worst_case": True}, "fun"),
    ("negative", {"method": "mma", "bounds": [(0, 1)], "worst_case": True}, "worst_case"),
    ("zero", {"method": "mma", "bounds": [(0, 1)], "worst_case": True}, "worst_case"),
    ("infinite", {"method": "mma", "bounds": [(0, 1)], "worst_case": True}, "worst_case"),
    ("nan", {"method": "mma", "bounds": [(0, 1)], "worst_case": True}, "worst_case"),
])
def test_refusals(case, kwargs, message):
    functions = {
        "scalar": lambda x: jnp.sum(x**2), "vector": lambda x: x**2,
        "matrix": lambda x: x[None, :], "empty": lambda x: x[:0],
        "negative": lambda x: jnp.array([-1.0, 1.0]),
        "zero": lambda x: x * 0, "infinite": lambda x: x * jnp.inf,
        "nan": lambda x: x * jnp.nan,
    }
    options = dict(x0=[0.5], n_iters=1, jit=False)
    options.update(kwargs)
    with pytest.raises(rfx.RfxRefusal, match=message):
        rfx.minimize(functions[case], **options)


@pytest.mark.parametrize("cap", ["max_newton", "max_halvings"])
def test_mma_cap_warning_and_count(monkeypatch, cap):
    original = optimizers._mma_subproblem
    counts = []
    outer = []
    einsum = np.einsum

    def force_cap(*args, **kwargs):
        outer.append(kwargs.get("outer_iteration"))
        kwargs[cap] = 1 if cap == "max_newton" else 0
        with monkeypatch.context() as patch:
            if cap == "max_halvings":
                # Reject every trial residual norm without changing its values.
                patch.setattr(np, "einsum", lambda expression, *values, **kw:
                              np.inf if expression == "i,i->" else einsum(expression, *values, **kw))
            result = original(*args, **kwargs)
        counts.append(result.caps_hit)
        return result

    monkeypatch.setattr(optimizers, "_mma_subproblem", force_cap)
    for worst_case in (False, True):
        counts.clear()
        outer.clear()
        fun = (lambda x: (x - 0.2)**2 + 1) if worst_case else (lambda x: jnp.sum((x - 0.2)**2))
        with pytest.warns(rfx.MMASubproblemCapWarning) as caught:
            result = rfx.minimize(fun, [0.8, 0.6], bounds=[(0, 1)] * 2,
                                  method="mma", worst_case=worst_case, n_iters=3, jit=False)
        assert result.subproblem_caps == sum(counts) == 3
        assert outer == [1, 2, 3]
        assert len(caught) == 3
        for iteration, warning in enumerate(caught, 1):
            assert type(warning.message) is rfx.MMASubproblemCapWarning
            assert cap in str(warning.message)
            assert f"outer iteration {iteration}" in str(warning.message)
            assert "unconverged subproblem" in str(warning.message)


def test_evaluated_result_and_nonfinite():
    fun = lambda x: jnp.sum(x**2)
    result = rfx.minimize(fun, [1, 2], lr=0.1, n_iters=4)
    assert result.fun == result.history[-1]
    np.testing.assert_array_equal(result.x, result.x_history[-1])
    assert abs(result.fun - float(fun(result.x))) <= 1e-5 * abs(result.fun)
    assert abs(result.history[-1] - result.history[-2]) > 1e-3 * abs(result.history[-1])
    assert result.nfev == 4 and not result.stopped_nonfinite
    oscillating = rfx.minimize(fun, [0.1], lr=1, n_iters=3, best_iterate=True, jit=False)
    assert np.argmin(oscillating.history) == 0
    np.testing.assert_array_equal(oscillating.x, oscillating.x_history[0])
    assert oscillating.fun == min(oscillating.history)
    for worst_case in (False, True):
        for first in (False, True):
            x0 = [0.0] if first else [0.5]
            objective = (lambda x: jnp.sqrt(x) + 1) if worst_case else (lambda x: jnp.sum(jnp.sqrt(x)))
            options = dict(method="mma", bounds=[(-1, 1)], worst_case=True) if worst_case else dict(lr=1)
            with pytest.warns(rfx.AccuracyWarning, match="minimize: non-finite"):
                stopped = rfx.minimize(objective, x0, n_iters=4, jit=False, **options)
            assert stopped.stopped_nonfinite
            assert stopped.nfev == (1 if first else 2)
            assert len(stopped.history) == (0 if first else 1)
            assert stopped.x_history.shape == (len(stopped.history), 1)
            np.testing.assert_allclose(stopped.x, x0, rtol=0, atol=0)
            assert np.isnan(stopped.fun) if first else stopped.fun == stopped.history[-1]
            expected_counts = (dict(forward=0, reverse=0, forward_mode=stopped.nfev) if worst_case
                               else dict(forward=stopped.nfev, reverse=stopped.nfev, forward_mode=0))
            assert stopped.solves == expected_counts
            assert stopped.setup_forwards == int(worst_case)
    with pytest.warns(rfx.AccuracyWarning, match="largest entry <= 0 at iteration 2"):
        rfx.minimize(lambda x: x + 0.05, [0.5], bounds=[(-1, 1)],
                     method="mma", worst_case=True, n_iters=2, jit=False)
    with pytest.warns(rfx.ConfigurationWarning, match="float64 to float32"):
        converted = rfx.minimize(fun, np.array([1.0], dtype=np.float64), n_iters=1, jit=False)
    assert converted.x.dtype == jnp.float32
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rfx.minimize(fun, [1.0], n_iters=1, jit=False)
    assert not caught


@pytest.mark.parametrize("n,m,mode", [(3, 15, "forward"), (4, 3, "forward"), (5, 3, "reverse"), (40, 3, "reverse")])
def test_jacobian_rule_and_solves(n, m, mode):
    offsets = jnp.arange(m, dtype=jnp.float32)
    fun = lambda x: jnp.sum((x - 0.3)**2) + offsets + 1
    result = rfx.minimize(fun, [0.6] * n, bounds=[(0, 1)] * n,
                          method="mma", worst_case=True, n_iters=2, jit=False)
    assert result.jacobian == mode
    expected = (dict(forward=0, reverse=0, forward_mode=2 * n) if mode == "forward"
                else dict(forward=2, reverse=2 * (m + 1), forward_mode=0))
    assert result.solves == expected
    assert result.nfev == 2 and result.setup_forwards == 1


def test_box_map_preserves_endpoint_gradient(monkeypatch):
    observed = []
    original = implementation._run_optimizer

    def inspect_map(opt, fun, x0, bounds, **kwargs):
        assert kwargs["name"] == "minimize"
        for p in (0.0, 1.0):
            point = jnp.full(2, p, dtype=jnp.float32)
            observed.append((p, jax.grad(fun)(point)))
        return original(opt, fun, x0, bounds, **kwargs)

    monkeypatch.setattr(implementation, "_run_optimizer", inspect_map)
    rfx.minimize(lambda x: jnp.sum(x**2), [0, 0], bounds=[(-1, 0.7)] * 2,
                 method="projected_adam", n_iters=1, jit=False)
    width = np.float32(0.7) - np.float32(-1)
    for p, derivative in observed:
        x = np.float32(-1) if p == 0 else np.float32(0.7)
        expected = np.full(2, width * np.float32(2) * x, dtype=np.float32)
        np.testing.assert_allclose(derivative, expected, rtol=1e-7, atol=0)


@pytest.mark.parametrize("method", ["adam", "projected_adam", "mma"])
@pytest.mark.parametrize("jit", [False, True])
def test_host_reading_objective(method, jit):
    def fun(x):
        return jnp.sum(x**2) if x[0] > 0 else jnp.sum(x**2) + 1

    options = dict(bounds=[(0, 1)] * 2) if method != "adam" else {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = rfx.minimize(fun, [0.5, 0.5], method=method, n_iters=3, jit=jit, **options)
    assert [type(w.message) for w in caught] == ([rfx.ConfigurationWarning] if jit else [])
    assert result.nfev == 3 and result.setup_forwards == 0
    assert not result.stopped_nonfinite
    assert result.history[-1] < result.history[0]
    np.testing.assert_allclose(result.fun, fun(result.x), rtol=1e-6)


@pytest.mark.parametrize("method", ["adam", "projected_adam", "mma"])
@pytest.mark.parametrize("jit", [False, True])
def test_evaluated_points_equal_history(method, jit):
    seen = []
    target = jnp.array([0.31, -1.77, 2.2])

    def fun(x):
        jax.debug.callback(lambda value: seen.append(np.asarray(value).copy()), x)
        return jnp.sum((x - target) ** 2)

    options = {} if method == "mma" else dict(lr=0.5 if method == "adam" else 0.1)
    result = rfx.minimize(fun, [0.1, 0.2, 0.3],
                          bounds=[(-0.3, 0.7), (-1.1, 0.9), (0.01, 2.0)],
                          method=method, n_iters=40, jit=jit, **options)
    jax.effects_barrier()
    points = np.stack(seen)
    history = np.asarray(result.x_history)
    assert points.shape == history.shape == (40, 3)
    assert points.dtype == history.dtype == np.float32
    difference = np.abs(points.astype(np.float64) - history.astype(np.float64))
    if jit:
        assert np.all(difference <= np.spacing(np.abs(history))), difference
    else:
        np.testing.assert_array_equal(
            points.view(np.uint32), history.view(np.uint32),
            err_msg=f"max absolute difference: {difference.max():.17g}",
        )
    np.testing.assert_array_equal(np.asarray(result.x).view(np.uint32), history[-1].view(np.uint32))
