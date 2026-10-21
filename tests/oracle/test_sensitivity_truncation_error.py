"""Closed-form sensitivity truncation oracles from arXiv:2610.28496."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import ringdown
from rfx.ringdown import required_t_over_tau, sensitivity_truncation_error


@pytest.mark.parametrize("x,d,expected", [
    (1, 0, 0.7357588823),
    (2, 0, 0.4060058497),
    (4.6, 0, 0.0562902802),
    (6.6, 0, 0.0103387971),
    (6.64, 0, 0.0099856881),
    (6.6, 1, 0.0136931703),
    (4.6, 10, 0.4657982073),
    (2, 10, 2.7369867200),
    (2, 100, 27.0701015199),
])
def test_table(x, d, expected):
    assert sensitivity_truncation_error(x, d) == pytest.approx(expected, rel=0, abs=5e-11)


@pytest.mark.parametrize("d,expected", [(0, 6.638352068), (10, 9.124878198)])
def test_required_table(d, expected):
    assert required_t_over_tau(0.01, d) == pytest.approx(expected, rel=0, abs=5e-10)


def test_evenness():
    x = np.array([0.1, 0.5, 1, 2, 6.64, 10])[:, None]
    d = np.array([0, 0.5, 3, 10, 100])[None, :]
    np.testing.assert_allclose(sensitivity_truncation_error(x, d),
                               sensitivity_truncation_error(x, -d), rtol=1e-9, atol=0)


@pytest.mark.parametrize("x", [0.1, 0.5, 1, 2, 4.6, 6.64, 10])
def test_resonance_limit(x):
    assert sensitivity_truncation_error(x, 0) == pytest.approx(
        (1 + x) * math.exp(-x), rel=1e-9, abs=0)


@pytest.mark.parametrize("error", [0.9, 0.5, 0.1, 0.01, 1e-4])
@pytest.mark.parametrize("d", [0, 0.5, 3, 10, 100])
def test_round_trip(error, d):
    x = required_t_over_tau(error, d)
    assert isinstance(x, float)
    if error == 0.9 and d == 0:
        assert x < 1
    assert sensitivity_truncation_error(x, d) == pytest.approx(error, rel=1e-9, abs=0)


@pytest.mark.parametrize("x", [0, -1, math.nan, math.inf])
def test_invalid_record_length(x):
    with pytest.raises(ValueError):
        sensitivity_truncation_error(x)


@pytest.mark.parametrize("error", [0, 1, 1.5, -0.1, math.nan, math.inf])
def test_invalid_error(error):
    with pytest.raises(ValueError):
        required_t_over_tau(error)


@pytest.mark.parametrize("d", [math.nan, math.inf, -math.inf])
def test_invalid_detuning(d):
    with pytest.raises(ValueError):
        sensitivity_truncation_error(2.0, d)
    with pytest.raises(ValueError):
        required_t_over_tau(0.01, d)


@pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])
def test_vectorized_float32(xp):
    x = xp.asarray([1, 2, 6.6], dtype=xp.float32)[:, None]
    d = xp.asarray([0, 1, 10], dtype=xp.float32)[None, :]
    expected = [[sensitivity_truncation_error(float(a), float(b))
                 for b in np.asarray(d)[0]] for a in np.asarray(x)[:, 0]]
    actual = sensitivity_truncation_error(x, d)
    assert actual.dtype == xp.float32
    assert isinstance(actual, np.ndarray if xp is np else jax.Array)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)
    assert sensitivity_truncation_error(x).dtype == xp.float32
    assert sensitivity_truncation_error(2.0, d).dtype == xp.float32


@pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])
def test_concrete_array_refusals(xp):
    with pytest.raises(ValueError):
        sensitivity_truncation_error(xp.asarray([2.0, 0.0]))
    with pytest.raises(ValueError):
        sensitivity_truncation_error(2.0, xp.asarray([0.0, math.nan]))


def test_jit_invalid_elements_are_nan():
    x = jnp.asarray([2, 0, -1, math.nan, math.inf, 2], dtype=jnp.float32)
    d = jnp.asarray([0, 0, 0, 0, 0, math.nan], dtype=jnp.float32)
    actual = jax.jit(sensitivity_truncation_error)(x, d)
    assert float(actual[0]) == pytest.approx(0.4060058497, rel=1e-6, abs=0)
    assert np.all(np.isnan(np.asarray(actual[1:])))


def test_public_exports_and_defaults():
    assert ringdown.GRADIENT_WITNESS_BAR == 1e-2
    assert "sensitivity_truncation_error" in ringdown.__all__
    assert "required_t_over_tau" in ringdown.__all__
    assert isinstance(sensitivity_truncation_error(2.0), float)
    assert sensitivity_truncation_error(2.0) == pytest.approx(0.4060058497, rel=0, abs=5e-11)
    assert required_t_over_tau() == pytest.approx(6.638352068, rel=0, abs=5e-10)
