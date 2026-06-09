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
