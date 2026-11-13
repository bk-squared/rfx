"""MMA review witnesses: literal hand values from DESIGN section 8.3 (2026-11-13)."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rfx.optimizers as optimizers


@pytest.fixture(autouse=True)
def cpu_device():
    with jax.default_device(jax.devices('cpu')[0]):
        yield


def test_box_in_design_dtype():
    opt = optimizers._MMA(m=0)
    x0 = jnp.asarray([-0.1, 0.1, -0.1, 0.1], dtype=jnp.float32)
    expected = np.array([-0.09999999403953552, 0.09999999403953552,
                         -0.09999999403953552, 0.09999999403953552], dtype=np.float64)
    state = opt.init(x0, (-0.1, 0.1))
    assert state.x.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(state.x, dtype=np.float64), expected)

    solution = np.array([-0.09999999999, 0.09999999999,
                         -0.09999999999, 0.09999999999], dtype=np.float64)
    point = opt._design_point(solution, x0.dtype, x0.shape, state.xmin, state.xmax)
    assert point.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(point, dtype=np.float64), expected)

    evaluate = jax.value_and_grad(lambda x: x[0] - x[1] + x[2] - x[3])
    for _ in range(30):
        request = opt.ask(state)
        x = np.asarray(request.x, dtype=np.float64)
        assert np.all((x >= -0.1) & (x <= 0.1))
        value, grad = evaluate(request.x)
        state = opt.tell(state, request, value, grad, None, None)
        x = np.asarray(state.x, dtype=np.float64)
        assert state.x.dtype == jnp.float32
        assert np.all((x >= -0.1) & (x <= 0.1))

    with pytest.raises(ValueError, match='rounded box is empty'):
        opt.init(jnp.asarray([0.7], dtype=jnp.float32), (0.7, 0.70000001))
    # One empty component must reject the entire box.
    with pytest.raises(ValueError, match='rounded box is empty'):
        opt.init(jnp.asarray([0.5, 0.7], dtype=jnp.float32),
                 (np.array([0, 0.7]), np.array([1, 0.70000001])))


def test_state_rules(monkeypatch):
    calls = []
    real_subproblem = optimizers._mma_subproblem

    def record_subproblem(low, upp, alpha, beta, p, q, r, *args, **kwargs):
        calls.append({name: array.copy() for name, array in
                      zip(('low', 'upp', 'alpha', 'beta', 'p', 'q', 'r'),
                          (low, upp, alpha, beta, p, q, r))})
        return real_subproblem(low, upp, alpha, beta, p, q, r, *args, **kwargs)

    monkeypatch.setattr(optimizers, '_mma_subproblem', record_subproblem)
    points = ((0.5, 0.5, 0.5), (0.625, 0.625, 0.5), (0.5, 0.75, 0.5))
    value = np.array(3, dtype=np.float64)
    grad = np.array([1, -2, 0], dtype=np.float64)
    cons_values = np.array([-0.25], dtype=np.float64)
    cons_grads = np.array([[0.5, 0, -1]], dtype=np.float64)
    # These are the leader's hand-checked literals, not a second implementation.
    boxes = (
        dict(low=(0, 0, 0), upp=(1, 1, 1),
             alpha=(0.05, 0.05, 0.05), beta=(0.95, 0.95, 0.95)),
        dict(low=(0.125, 0.125, 0), upp=(1.125, 1.125, 1),
             alpha=(0.175, 0.175, 0.05), beta=(1, 1, 0.95)),
        dict(low=(0.15, 0.15, 0), upp=(0.85, 1.35, 1),
             alpha=(0.185, 0.25, 0.05), beta=(0.815, 1, 0.95)),
    )
    coefficients = dict(
        p=((1.2262372499999998e-01, 7.2360000000000024e-04, 2.5000000000000002e-06),
           (6.131247499999998e-02, 3.600000000000001e-06, 2.525000000000000e-04)),
        q=((1.2372499999999999e-04, 7.2072359999999991e-01, 2.5000000000000002e-06),
           (6.2475000000000001e-05, 3.6000000000000003e-06, 2.5025249999999999e-01)),
        r=(1.446871, -0.926379),
    )
    clamped = dict(low=(0.49, -9.25, 0), upp=(0.51, 10.75, 1),
                   alpha=(0.491, 0.25, 0.05), beta=(0.509, 1, 0.95))
    for label, opt in (('default', optimizers._MMA(m=1)),
                       ('clamped', optimizers._MMA(m=1, asydecr=0.01, asyincr=30))):
        calls.clear()
        state = opt.init(jnp.asarray(points[0], dtype=jnp.float32), (0, 1))
        for k, point in enumerate(points, start=1):
            request = replace(opt.ask(state), x=jnp.asarray(point, dtype=jnp.float32))
            state = opt.tell(state, request, value, grad, cons_values, cons_grads)
            assert len(calls) == k
            expected = dict(clamped if label == 'clamped' and k == 3 else boxes[k - 1])
            if label == 'default' and k == 3:
                expected.update(coefficients)
            for name, literal in expected.items():
                np.testing.assert_allclose(calls[-1][name], literal, rtol=1e-12, atol=1e-15,
                                           err_msg=f'{label}, k={k}, {name}')
