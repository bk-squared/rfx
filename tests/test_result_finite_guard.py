"""Tier-1 correctness guard: NaN/Inf must never propagate silently.

Covers ``Result.assert_finite`` + the shared ``_warn_if_nonfinite_result``
helper (rfx/api/_spec.py). A non-finite ``time_series`` / ``s_params`` almost
always means the FDTD diverged; surfacing it (warn, or raise on demand) lets
an automation loop fail fast instead of feeding garbage into a metric.

The optimizer-side ``jnp.isfinite(grad)`` guard (rfx/optimize.py,
rfx/topology.py) is the partner fix; it is exercised indirectly here via the
helper contract and directly by the gradient-coverage suites.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api._spec import (
    Result,
    _nonfinite_fields,
    _warn_if_nonfinite_result,
)


def _result(time_series, s_params=None):
    return Result(
        state=None,
        time_series=np.asarray(time_series),
        s_params=None if s_params is None else np.asarray(s_params),
        freqs=None,
    )


def test_finite_result_assert_finite_true_and_silent():
    r = _result(np.zeros((10, 2)), s_params=np.full((1, 1, 3), 0.4 + 0j))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert r.assert_finite() is True


def test_nan_time_series_warns_and_returns_false():
    ts = np.zeros((10, 2))
    ts[3, 1] = np.nan
    r = _result(ts)
    with pytest.warns(UserWarning, match="non-finite"):
        assert r.assert_finite() is False


def test_inf_sparams_raise_on_demand():
    s = np.full((1, 1, 3), 0.5 + 0j)
    s[0, 0, 1] = np.inf
    r = _result(np.zeros((4, 1)), s_params=s)
    with pytest.raises(ValueError, match="non-finite"):
        r.assert_finite(raise_on_nonfinite=True)


def test_nonfinite_fields_reports_counts():
    ts = np.zeros((5, 2))
    ts[0, 0] = np.nan
    ts[1, 1] = np.inf
    bad = _nonfinite_fields(_result(ts))
    assert dict(bad)["time_series"] == 2


def test_warn_helper_silent_when_finite():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_nonfinite_result(_result(np.ones((3, 1))), context="run")


def test_assert_finite_is_tracer_safe_under_jax_grad():
    """Under jax.grad the observables are tracers; assert_finite must be a
    no-op (never convert/raise), so it is safe to call inside forward()/run()
    on the AD path."""

    def f(x):
        r = Result(state=None, time_series=x, s_params=None, freqs=None)
        # Even though x carries non-finite-looking magnitude, tracing must skip.
        r.assert_finite()
        return jnp.sum(x)

    g = jax.grad(f)(jnp.full((4,), 3.0))
    assert bool(jnp.all(jnp.isfinite(g)))


def test_optimize_nan_gradient_warns_and_returns_last_good():
    """The optimizer NaN/Inf-gradient guard must (1) warn, (2) stop early, and
    (3) return the last FINITE design — not the divergence-causing one. Here a
    NaN objective makes the gradient NaN at iteration 0, so the loop breaks
    before any update and the returned design is the (finite) initial latent."""
    from rfx.api import Simulation
    from rfx.optimize import optimize, DesignRegion, OptimizeResult

    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015), boundary="pec")
    sim.add_port((0.005, 0.0075, 0.0075), "ez")
    sim.add_probe((0.01, 0.0075, 0.0075), "ez")
    region = DesignRegion(
        corner_lo=(0.006, 0.006, 0.006),
        corner_hi=(0.009, 0.009, 0.009),
        eps_range=(1.0, 4.4),
    )
    with pytest.warns(UserWarning, match="non-finite"):
        result = optimize(
            sim, region,
            lambda r: jnp.nan * jnp.sum(r.time_series),  # NaN loss => NaN grad
            n_iters=3, lr=0.01, verbose=False,
        )
    assert isinstance(result, OptimizeResult)
    assert len(result.loss_history) == 0          # broke at iteration 0
    assert np.all(np.isfinite(np.asarray(result.eps_design)))  # last-good is finite
