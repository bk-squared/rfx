"""Independent trial, iterate and accounting witnesses for scalar GCMMA."""

from dataclasses import dataclass, field
import itertools
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
from rfx.optimizers import _Adam, _GCMMA, _MMA, _Request


@pytest.fixture(autouse=True)
def cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


@dataclass(frozen=True)
class ReferenceGCMMA(_GCMMA):
    asks: list = field(default_factory=list, compare=False)
    decisions: list = field(default_factory=list, compare=False)

    def ask(self, state):
        request = super().ask(state)
        self.asks.append(request)
        return request

    def tell(self, state, request, value, grad, cons_values, cons_grads):
        updated = super().tell(state, request, value, grad, cons_values, cons_grads)
        if not request.iterate:
            # Rebuild the approximation without consuming p, q, r or tol.
            x = np.asarray(state.x, dtype=np.float64).ravel()
            point = np.asarray(request.x, dtype=np.float64).ravel()
            g = state.gradient
            width = state.xmax - state.xmin
            p = (state.upp - x)**2 * (1.001 * np.maximum(g, 0) + .001 * np.maximum(-g, 0) + state.rho[:, None] / width)
            q = (x - state.low)**2 * (.001 * np.maximum(g, 0) + 1.001 * np.maximum(-g, 0) + state.rho[:, None] / width)
            approximation = state.values + np.sum(p / (state.upp - point) + q / (point - state.low)
                                                  - p / (state.upp - x) - q / (x - state.low), axis=1)
            need = float(value) - approximation[0]
            tau = 50 * float(np.spacing(np.float32(abs(state.values[0]))))
            self.decisions.append((updated.trial is None, need, tau))
        return updated


def test_gcmma_matches_independent_map_and_loop():
    # A positive offset exposes float32 rounding near a quadratic's minimum.
    fun = lambda x: jnp.float32(.0328) + jnp.sum((x - jnp.float32(.3))**2)
    lo, hi = jnp.array([-2.], dtype=jnp.float32), jnp.array([2.], dtype=jnp.float32)
    start = jnp.asarray((np.array([1.]) - np.array([-2.])) / np.array([4.]), dtype=jnp.float32)

    def physical(p):
        raw = lo + (hi - lo) * p
        return raw + jax.lax.stop_gradient(jnp.clip(raw, lo, hi) - raw)

    opt = ReferenceGCMMA(m=0, e=0, _ulp_multiple=50)
    _, history, points, solves = _run_optimizer(
        opt, lambda p: fun(physical(p)), start, (0., 1.), n_iters=60,
        jit=False, keep_iterates=True,
    )
    assert any(not accepted for accepted, _, _ in opt.decisions), opt.decisions
    assert any(accepted and 0 < need <= tau for accepted, need, tau in opt.decisions), opt.decisions
    for best in (False, True):
        result = rfx.minimize(fun, [1.], bounds=[(-2, 2)], method="GCMMA",
                              n_iters=60, jit=False, best_iterate=best)
        expected_points = np.asarray(jnp.stack([physical(p) for p in points]))
        np.testing.assert_array_equal(np.asarray(result.x_history).view(np.uint32), expected_points.view(np.uint32))
        np.testing.assert_array_equal(np.asarray(result.history).view(np.uint64), np.asarray(history).view(np.uint64))
        index = int(np.argmin(history)) if best else -1
        np.testing.assert_array_equal(result.x, expected_points[index])
        assert result.fun == history[index]
        assert result.solves == solves and result.nfev == len(opt.asks)
        assert result.method == "gcmma" and result.inner_caps == 0
        assert result.scale is None and result.setup_forwards == 0
        assert not result.stopped_nonfinite


def scripted_solver(points, calls=None):
    points = itertools.chain(points, itertools.repeat(points[-1]))

    def solve(*args, **kwargs):
        point = np.float32(next(points))
        if calls is not None:
            calls.append((kwargs["outer_iteration"], float(point)))
        return SimpleNamespace(x=np.array([point]), y=np.zeros(len(args[8])),
                               z=0., newton_steps=0, caps_hit=0)
    return solve


def test_counts_and_solves_history_from_independent_asks(monkeypatch):
    asks, stalls, rejected, trace, calls = [], [], [], [], []
    original_ask, original_tell = _GCMMA.ask, _GCMMA.tell

    def ask(self, state):
        request = original_ask(self, state)
        asks.append(request)
        return request

    def tell(self, state, request, *args):
        updated = original_tell(self, state, request, *args)
        stalls.append(updated.stalls - state.stalls)
        rejected.append(not request.iterate and updated.trial is not None)
        trace.append(dict(iterate=request.iterate, x=np.asarray(request.x).tolist(),
                          value=float(args[0]), k=updated.k, rejected=rejected[-1],
                          stall=stalls[-1], trial=None if updated.trial is None
                          else np.asarray(updated.trial).tolist()))
        return updated

    with monkeypatch.context() as patch:
        patch.setattr(_GCMMA, "ask", ask)
        patch.setattr(_GCMMA, "tell", tell)
        # The kink makes f(.1)=1.09 exceed the initial model (.0014667).
        # Repeat the accepted .4 in float32 for a certain stall; the last
        # iterate's tell still solves once, but its .3 trial is unevaluated.
        patch.setattr(optimizers, "_mma_subproblem",
                      scripted_solver([.1, .4, .4, .3], calls))
        objective = lambda x: jnp.sum((x - .4)**2 + 100 * jax.nn.relu(.2 - x)**2)
        result = rfx.minimize(objective, [.5], bounds=[(0, 1)],
                              method="gcmma", n_iters=3, jit=False)
    assert any(rejected), trace
    assert any(stalls), trace
    assert sum(stalls) == sum(rejected) == 1, trace
    # Completed outer transitions, plus rejection re-solves, plus the solve
    # after the final iterate's tell. Count calls in the solver wrapper.
    outer_solves = sum(request.iterate for request in asks[:-1])
    assert len(calls) == outer_solves + sum(rejected) + 1 == 4, (calls, trace)
    assert [k for k, _ in calls] == [1, 1, 2, 3], (calls, trace)
    assert [req.iterate for req in asks] == [True, False, False, True, True]
    assert result.nfev == 5
    assert result.solves == dict(forward=5, reverse=3, forward_mode=0)
    assert result.solves_history == [2, 6, 8]
    total, expected = 0, []
    for request in asks:
        total += 2 if request.need_grad else 1
        if request.iterate:
            expected.append(total)
    assert result.solves_history == expected
    assert len(result.history) == len(result.x_history) == 3
    assert not result.stopped_nonfinite

    for method in ("adam", "projected_adam", "mma"):
        scalar = rfx.minimize(lambda x: jnp.sum(x**2), [.5], bounds=[(0, 1)],
                              method=method, n_iters=3, jit=False)
        assert scalar.solves_history == [2, 4, 6]
        assert scalar.inner_caps == 0

    original_mma_ask = _MMA.ask
    for n, m, mode in ((2, 3, "forward"), (5, 3, "reverse")):
        log = []

        def mma_ask(self, state):
            request = original_mma_ask(self, state)
            log.append(request)
            return request

        with monkeypatch.context() as patch:
            patch.setattr(_MMA, "ask", mma_ask)
            result = rfx.minimize(lambda x: jnp.sum((x - .3)**2) + jnp.arange(m) + 1,
                                  [.6] * n, bounds=[(0, 1)] * n, method="mma",
                                  worst_case=True, n_iters=3, jit=False)
        total, expected = 0, []
        for request in log:
            assert request.iterate and request.need_grad
            total += request.x.size if mode == "forward" else m + 2
            expected.append(total)
        assert len(log) == 3
        assert result.solves_history == expected
        assert all(type(count) is int for count in result.solves_history)
        assert result.solves_history[-1] == sum(result.solves.values())
        assert result.setup_forwards == 1 and result.jacobian == mode


def test_subproblem_caps_include_inner_solves(monkeypatch):
    caps, outer = [], []
    solve = optimizers._mma_subproblem

    def capped(*args, **kwargs):
        kwargs["max_newton"] = 1
        result = solve(*args, **kwargs)
        caps.append(result.caps_hit)
        outer.append(kwargs["outer_iteration"])
        return result

    monkeypatch.setattr(optimizers, "_mma_subproblem", capped)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rfx.AccuracyWarning)
        result = rfx.minimize(lambda x: jnp.sum((x - .4)**2), [.8], bounds=[(0, 1)],
                              method="gcmma", n_iters=4, jit=False)
    assert sum(caps) >= 2
    assert len(set(outer)) < len(outer), outer
    assert result.subproblem_caps == sum(caps)


def test_inner_caps_independent_of_warnings(monkeypatch):
    accept, post_init = _GCMMA._accept, implementation._RecordingGCMMA.__post_init__
    caps = []

    def forced(self):
        object.__setattr__(self, "_inner_cap", 1)
        object.__setattr__(self, "tol", -1e6)
        post_init(self)

    def counted(self, *args, **kwargs):
        caps.append(int(kwargs.get("cap", False)))
        return accept(self, *args, **kwargs)

    monkeypatch.setattr(implementation._RecordingGCMMA, "__post_init__", forced)
    monkeypatch.setattr(_GCMMA, "_accept", counted)
    monkeypatch.setattr(optimizers, "_mma_subproblem", scripted_solver([.9, .8, .7, .6]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rfx.AccuracyWarning)
        result = rfx.minimize(lambda x: jnp.sum(x**2), [.1], bounds=[(0, 1)],
                              method="gcmma", n_iters=4, jit=False)
    assert sum(caps) == 3
    assert result.inner_caps == sum(caps)
    assert result.subproblem_caps == 0


def test_gcmma_refusals():
    fun = lambda x: jnp.sum(x**2)
    cases = [
        ({}, "bounds are required"),
        ({"bounds": [(0, 1)], "worst_case": True}, 'worst_case is measured only with method="mma"'),
        ({"bounds": [(0, 1)], "lr": .1}, "lr is not accepted"),
    ]
    for kwargs, message in cases:
        with pytest.raises(rfx.RfxRefusal, match=message):
            rfx.minimize(fun, [.5], method="gcmma", n_iters=1, **kwargs)
    for method in ("adam", "projected_adam"):
        with pytest.raises(rfx.RfxRefusal, match='worst_case is measured only with method="mma"'):
            rfx.minimize(fun, [.5], bounds=[(0, 1)], method=method, worst_case=True)


def test_nonfinite_trials_iterates_and_full_runs(monkeypatch):
    for method in ("adam", "gcmma"):
        for case in ("trial", "iterate", "first", "full"):
            cls = _Adam if method == "adam" else _GCMMA
            original = cls.ask
            requests = []

            def ask(self, state):
                request = original(self, state)
                # Inject a non-finite request at the loop boundary. Adam has
                # no native trials; this exercises the result semantics for
                # the same value-only failure through either public method.
                if case == "first" or (len(requests) == 1 and case != "full"):
                    trial = case == "trial"
                    request = _Request(jnp.full_like(request.x, jnp.nan),
                                       need_grad=not trial, iterate=not trial)
                requests.append(request)
                return request

            with monkeypatch.context() as patch, warnings.catch_warnings(record=True) as caught:
                patch.setattr(cls, "ask", ask)
                warnings.simplefilter("always")
                result = rfx.minimize(lambda x: jnp.sum(x**2), [.5], bounds=[(0, 1)],
                                      method=method, n_iters=3, jit=False)
            finite = case == "full"
            assert result.stopped_nonfinite is not finite
            assert len(result.history) == (3 if finite else 0 if case == "first" else 1)
            assert result.x_history.shape == (len(result.history), 1)
            assert result.nfev == len(requests)
            assert len(result.solves_history) == len(result.history)
            assert len(caught) == (0 if finite else 1)
            if not finite:
                assert isinstance(caught[0].message, rfx.AccuracyWarning)
                assert "minimize: non-finite" in str(caught[0].message)
                np.testing.assert_array_equal(result.x, [.5])
                assert np.isnan(result.fun) if case == "first" else result.fun == result.history[-1]
            else:
                assert result.solves_history[-1] == sum(result.solves.values())


def test_tolerance_hand_values_and_decisions(monkeypatch):
    constructed = []
    post_init = implementation._RecordingGCMMA.__post_init__

    def capture(self):
        post_init(self)
        constructed.append(self)

    monkeypatch.setattr(implementation._RecordingGCMMA, "__post_init__", capture)
    monkeypatch.setattr(optimizers, "_mma_subproblem", lambda *a, **k:
                        SimpleNamespace(x=np.array([.6]), y=np.zeros(0), z=0., newton_steps=0, caps_hit=0))
    for f in (.0328, .07):
        value = np.float32(f)
        ulp = float(np.spacing(abs(value)))
        # Deliberately use a float64 design with a float32 objective value.
        with jax.enable_x64():
            rfx.minimize(lambda x: jnp.asarray(value, dtype=jnp.float32), [.5],
                         bounds=[(0, 1)], method="gcmma", n_iters=1, jit=False)
            opt = constructed[-1]
            assert opt.m == 0 and np.all(opt.e == 0)
            state = opt.init(jnp.array([.5], dtype=jnp.float64), (0, 1))
            state = opt.tell(state, opt.ask(state), value, np.zeros(1), None, None)
            assert state.tol[0] == 50 * ulp
            assert state.tol[0] == (1.862645149230957e-7 if f == .0328 else 3.725290298461914e-7)
            point = np.asarray(state.trial)
            approximation = state.r + np.sum(state.p / (state.upp - point) + state.q / (point - state.low), axis=1)
            for multiple, accepted in ((25, True), (75, False)):
                trial_value = np.float32(approximation[0] + multiple * ulp)
                need = float(trial_value) - approximation[0]
                assert (10 * ulp < need <= 50 * ulp) if accepted else (50 * ulp < need <= 100 * ulp)
                updated = opt.tell(state, opt.ask(state), trial_value, None, None, None)
                assert (updated.trial is None) is accepted
                assert updated.inner_caps == 0
