"""Predeclared MMA witnesses, DESIGN sections 8/8.1 (2026-11-13).

References below use only the closed-form KKT equations, never rfx helpers.
Every invocation checks all preconditions before permitting an MMA evaluation.
"""
import math
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.optimizers import _MMA, _mma_subproblem


# Section 8.1: eps/(2 |d|) <= 1e-6 away from a bound (distance >= .05);
# at a bound sqrt(eps/2) ~= 2e-4, allowing 5x gives the all-component bar.
SAFE_ERROR = 1e-5
ALL_ERROR = 1e-3
SAFE_DISTANCE = 0.05
# T1c: about 1000 times float32 evaluation error around f = 1.
MINIMAX_ERROR = 1e-4
OUTER_ITERATIONS = 200


@jax.jit
def _sum_value_grad(x):
    return jax.value_and_grad(lambda t: jnp.sum(t))(x)


@jax.jit
def _weighted_value_grad(x, w):
    return jax.value_and_grad(lambda t: jnp.sum(w * t))(x)


def _bisect_shift(a, target):
    lo, hi = -4.0, 4.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.sum(np.clip(a - mid / 2, 0, 1)) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * max(1, abs(mid)):
            break
    return (lo + hi) / 2


@lru_cache(None)
def _reference(n, two):
    # NumPy's seeded Generator, then exactly the float32 data seen by JAX.
    a = np.random.default_rng(0).uniform(-0.5, 1.5, n).astype(np.float32).astype(np.float64)
    w = np.where(np.arange(n) % 2 == 0, 1.0, 3.0)
    b = np.array([0.3 * n, 0.55 * n] if two else [0.3 * n])
    if two:
        # The two KKT equations reduce to even/odd sums: E+O=b1,
        # E+3O=b2. Independent bisections solve those two equations;
        # h_even=nu1+nu2 and h_odd=nu1+3nu2 reconstruct the multipliers.
        odd_target = (b[1] - b[0]) / 2
        even_target = b[0] - odd_target
        h_even = _bisect_shift(a[::2], even_target)
        h_odd = _bisect_shift(a[1::2], odd_target)
        nu2 = (h_odd - h_even) / 2
        nu1 = h_even - nu2
        nu = np.array([nu1, nu2])
        shifted = a - (nu1 + nu2 * w) / 2
    else:
        nu = np.array([_bisect_shift(a, b[0])])
        shifted = a - nu[0] / 2
    star = np.clip(shifted, 0, 1)
    host_sums = np.array([math.fsum(star), math.fsum(w * star)] if two else [math.fsum(star)])
    np.testing.assert_allclose(host_sums, b, rtol=1e-14, atol=0)
    star32 = jnp.asarray(star, dtype=jnp.float32)
    sums = [float(_sum_value_grad(star32)[0])]
    if two:
        sums.append(float(_weighted_value_grad(star32, jnp.asarray(w, dtype=jnp.float32))[0]))
    e_sum = np.abs(np.array(sums) - host_sums)
    free = int(np.count_nonzero((star > 0) & (star < 1)))
    safe = (np.abs(shifted) >= SAFE_DISTANCE) & (np.abs(shifted - 1) >= SAFE_DISTANCE)
    unconstrained = np.clip(a, 0, 1)
    margins = np.array([math.fsum(unconstrained) - b[0]])
    if two:
        margins = np.append(margins, math.fsum(w * unconstrained) - b[1])
    return a, w, b, nu, star, e_sum, free, safe, margins


@pytest.fixture(scope='module', autouse=True)
def preconditions():
    with jax.default_device(jax.devices('cpu')[0]):
        for two in (False, True):
            for n in (1000, 100000):
                _, _, _, nu, _, e_sum, free, _, margins = _reference(n, two)
                floor = 10 * e_sum / free + 1e-6
                print(f'PRECONDITION T1{"b" if two else "a"} n={n}: nu={nu.tolist()} '
                      f'active_margins={margins.tolist()} n_free={free} '
                      f'e_sum={e_sum.tolist()} floor={floor.tolist()}', flush=True)
                if not (np.all(nu > 1e-3) and free >= n / 10 and np.all(floor <= 0.25 * SAFE_ERROR)):
                    pytest.exit('STOP: predeclared MMA precondition failed; do not run MMA', returncode=1)
        yield


def test_00_preconditions(preconditions):
    """The module fixture prints and checks references before any MMA call."""


def _record(first, checks, iteration):
    for name, holds in checks.items():
        if holds and name not in first:
            first[name] = iteration


# n = 1e5 runs about 80-140 s on the cluster CPU lane; the weekly lane runs it, n = 1e3 runs before every merge
@pytest.mark.parametrize('n', [1000, pytest.param(100000, marks=pytest.mark.slow)])
@pytest.mark.parametrize('two', [False, True], ids=['T1a', 'T1b'])
def test_bounded_closed_form(n, two):
    a, w, b, nu, star, e_sum, _, safe, _ = _reference(n, two)
    a32, w32 = jnp.asarray(a, dtype=jnp.float32), jnp.asarray(w, dtype=jnp.float32)
    objective = jax.jit(jax.value_and_grad(lambda x: jnp.sum((x - a32)**2)))
    opt = _MMA(m=len(b))
    state = opt.init(jnp.full(n, 0.5, dtype=jnp.float32), (0, 1))
    trace, first, steps = [], {}, []
    # Measured JAX reduction error, with the 10x allowance fixed by section 8.1.
    residual_bar = np.maximum(1e-6 * b, 10 * e_sum)
    fstar = math.fsum((star - a)**2)
    for iteration in range(1, OUTER_ITERATIONS + 1):
        req = opt.ask(state)
        assert req.need_grad and req.iterate and req.x.dtype == jnp.float32
        value, grad = objective(req.x)
        evaluations = [_sum_value_grad(req.x)]
        if two:
            evaluations.append(_weighted_value_grad(req.x, w32))
        cons = jnp.stack([ev[0] for ev in evaluations]) - jnp.asarray(b, dtype=jnp.float32)
        cgrad = jnp.stack([ev[1] for ev in evaluations])
        state = opt.tell(state, req, value, grad, cons, cgrad)
        x = np.asarray(state.x, dtype=np.float64)
        errors = np.abs(x - star)
        residuals = np.array([np.sum(x), np.sum(w * x)] if two else [np.sum(x)]) - b
        f = float(np.sum((x - a)**2))
        checks = {'safe': np.max(errors[safe]) <= SAFE_ERROR,
                  'all': np.max(errors) <= ALL_ERROR,
                  'box': np.all((x >= 0) & (x <= 1)), 'caps': state.caps_hit == 0}
        checks.update({f'constraint_{i + 1}': residuals[i] <= residual_bar[i] for i in range(len(b))})
        _record(first, checks, iteration)
        _record(first, {'joint': all(checks.values())}, iteration)
        steps.append(state.newton_steps_last)
        trace.append((iteration, float(np.max(errors)), f, residuals.tolist(), state.newton_steps_last))
    print(f'T1{"b" if two else "a"} n={n}: first={first} max_error={np.max(errors):.12g} '
          f'safe_error={np.max(errors[safe]):.12g} residuals={residuals.tolist()} '
          f'objective={f:.15g} fstar={fstar:.15g} Newton_median={np.median(steps)} '
          f'Newton_max={max(steps)} caps={state.caps_hit}', flush=True)
    assert all(checks.values()), f'failed={checks}; last 10 (k, error, objective, residuals, Newton): {trace[-10:]}'


def test_minimax_closed_form():
    center = np.array([0.3, -0.2, 0.1])
    vertices = center + np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    p32 = jnp.asarray(vertices, dtype=jnp.float32)
    objective = jax.jit(jax.value_and_grad(lambda x: jnp.asarray(0, dtype=x.dtype)))
    constraints = [jax.jit(jax.value_and_grad(lambda x, p=p: jnp.sum((x - p)**2))) for p in p32]
    opt = _MMA(m=4, a0=1, a=np.ones(4), c=1000, d=1)
    state = opt.init(jnp.asarray(center + [0.5, -0.3, 0.2], dtype=jnp.float32), (center - 2, center + 2))
    trace, first, steps = [], {}, []
    for iteration in range(1, OUTER_ITERATIONS + 1):
        req = opt.ask(state)
        value, grad = objective(req.x)
        ev = [fn(req.x) for fn in constraints]
        state = opt.tell(state, req, value, grad, jnp.stack([v for v, _ in ev]), jnp.stack([g for _, g in ev]))
        x = np.asarray(state.x, dtype=np.float64)
        error = float(np.max(np.abs(x - center)))
        f = float(np.max(np.sum((x - vertices)**2, axis=1)))
        checks = {'point': error <= MINIMAX_ERROR, 'objective': f - 1 <= MINIMAX_ERROR,
                  'box': np.all((x >= center - 2) & (x <= center + 2)), 'caps': state.caps_hit == 0}
        _record(first, checks, iteration)
        _record(first, {'joint': all(checks.values())}, iteration)
        steps.append(state.newton_steps_last)
        trace.append((iteration, error, f, (np.sum((x - vertices)**2, axis=1) - 1).tolist(), state.newton_steps_last))
    print(f'T1c: first={first} max_error={error:.12g} residuals={trace[-1][3]} '
          f'objective={f:.15g} fstar=1 Newton_median={np.median(steps)} '
          f'Newton_max={max(steps)} caps={state.caps_hit}', flush=True)
    assert all(checks.values()), f'failed={checks}; last 10 (k, error, objective, residuals, Newton): {trace[-10:]}'


@pytest.mark.parametrize('n', [1, 2, 7])
def test_subproblem_relaxed_kkt(n):
    # Exercise both reductions, including n == m, with an independent witness.
    rng = np.random.default_rng(17)
    m = 2
    low, upp = -np.ones(n), 2 * np.ones(n)
    alpha, beta = np.zeros(n), np.ones(n)
    p, q = rng.uniform(0.1, 2, (2, m + 1, n))
    r = -rng.uniform(0.5, 2, m + 1) * n
    a0, a, c, d = 1.0, np.array([0.2, 0.4]), np.full(m, 1000.0), np.ones(m)
    sol = _mma_subproblem(low, upp, alpha, beta, p, q, r, a0, a, c, d)
    x, y, z = sol.x, sol.y, sol.z
    lam, xi, eta, mu, zeta, s = sol.lam, sol.xi, sol.eta, sol.mu, sol.zeta, sol.s
    eps = sol.epsilon
    g = np.sum(p[1:] / (upp - x) + q[1:] / (x - low), axis=1)
    psi = (p[0] + lam @ p[1:]) / (upp - x)**2 - (q[0] + lam @ q[1:]) / (x - low)**2
    residual = np.concatenate([psi - xi + eta, c + d * y - lam - mu,
                               [a0 - zeta - lam @ a], g - a * z - y + s + r[1:],
                               xi * (x - alpha) - eps, eta * (beta - x) - eps,
                               mu * y - eps, [zeta * z - eps], lam * s - eps])
    print(f'KKT n={n}, m=2: max_residual={np.max(np.abs(residual)):.12g} '
          f'epsilon={eps} Newton={sol.newton_steps} caps={sol.caps_hit}', flush=True)
    assert eps <= 1e-7
    assert np.max(np.abs(residual)) < 0.9e-7
    assert np.all(x > alpha) and np.all(x < beta)
    assert all(np.all(v > 0) for v in (y, z, lam, xi, eta, mu, zeta, s))
    assert sol.caps_hit == 0


def test_box_history_restart_and_unconstrained():
    opt = _MMA(m=0)
    x0 = jnp.asarray([[-0.2, 0.3], [1.2, 0.7]], dtype=jnp.float32)
    with pytest.raises(ValueError, match='box method'):
        opt.init(x0, None)
    state = opt.init(x0, (0, 1))
    np.testing.assert_array_equal(state.x, np.clip(x0, 0, 1))
    evaluate = jax.value_and_grad(lambda x: jnp.sum((x - 0.4)**2))
    for _ in range(4):
        req = opt.ask(state)
        old = np.asarray(req.x, dtype=np.float64).ravel()
        value, grad = evaluate(req.x)
        state = opt.tell(state, req, value, grad, None, None)
        np.testing.assert_array_equal(state.xold1, old)
        assert state.x.dtype == x0.dtype and state.x.shape == x0.shape
        assert state.low.dtype == np.float64 and state.upp.dtype == np.float64
    reset = opt.restart(state)
    np.testing.assert_array_equal(reset.x, state.x)
    assert reset.k == 0 and reset.xold1 is None and reset.xold2 is None
    value, grad = evaluate(reset.x)
    after = opt.tell(reset, opt.ask(reset), value, grad, None, None)
    np.testing.assert_array_equal(after.low, np.asarray(reset.x, dtype=np.float64).ravel() - 0.5)
    assert after.caps_hit == 0
