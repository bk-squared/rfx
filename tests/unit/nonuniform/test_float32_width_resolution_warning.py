"""Declared-width oracles for the graded float32 resolution advisory (W1–W3)."""
import math
import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx._grid_metric import Float32WidthResolutionWarning
from rfx.model.graded_grid import make_nonuniform_grid

ROWS = [(100, .7e-3, 1e-6), (200, .381e-3, 1e-6), (100, .7e-3, 5e-6)]
IDS = ["100x700um-1um", "200x381um-1um", "100x700um-5um"]


def _widths(row, dtype):
    n, coarse, fine = row
    # A final coarse cell satisfies the x/y matching-boundary admission rule.
    return np.array([coarse] * n + [fine, coarse], dtype=dtype)


def _build(axis, widths, pad=0):
    boundary = float(widths[0]) if not isinstance(widths, jax.core.Tracer) else .7e-3
    profiles = {"x": np.full(3, boundary), "y": np.full(3, boundary),
                "z": np.full(3, boundary)}
    profiles[axis] = widths
    return make_nonuniform_grid(
        (3 * boundary, 3 * boundary), profiles["z"], boundary,
        cpml_layers=pad, dx_profile=profiles["x"], dy_profile=profiles["y"])


def _expected(widths):
    # Derived only from the supplied widths, independently of grid helpers.
    w = [float(v) for v in widths]
    points = [math.fsum(w[:i]) for i in range(1, len(w))]
    ratios = [float(np.finfo(np.float32).eps) / 2 * abs(x) / min(w[i - 1], w[i])
              for i, x in enumerate(points, start=1)]
    worst = max(range(len(ratios)), key=ratios.__getitem__)
    return points[worst], ratios[worst]


@pytest.mark.parametrize("row", ROWS, ids=IDS)
@pytest.mark.parametrize("axis", "xyz")
@pytest.mark.parametrize("widen,pad", [(False, 0), (True, 8)])
def test_w1_graded_float32_widths_warn_once_with_measured_risk(row, axis, widen, pad):
    widths = _widths(row, np.float32)
    if widen:
        widths = widths.astype(np.float64)
    expected_x, expected_r = _expected(widths)
    assert expected_r > .5e-3
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build(axis, widths, pad)
    found = [w for w in caught if issubclass(w.category, Float32WidthResolutionWarning)]
    assert len(found) == 1
    message = str(found[0].message)
    assert f"axis {axis}:" in message
    assert "faces declared near such nodes may move one cell" in message
    assert "Widths built in float64 avoid it" in message
    x, r, cap = map(float, re.search(r"x=([^ ]+) m have r=([^,]+), .*cap ([^;]+);", message).groups())
    assert x == pytest.approx(expected_x, rel=1e-11)
    assert r == pytest.approx(expected_r, rel=1e-11)
    assert cap == .5e-3


@pytest.mark.parametrize("q", [3, 4, 6])
@pytest.mark.parametrize("axis", "xyz")
def test_w2_issue_uniform_float32_widths_do_not_warn(q, axis):
    widths = np.full(30 * q, 1.524e-3 / q, dtype=np.float32)
    assert _expected(widths)[1] < .5e-3
    with warnings.catch_warnings():
        warnings.simplefilter("error", Float32WidthResolutionWarning)
        _build(axis, widths)


@pytest.mark.parametrize("row", ROWS, ids=IDS)
@pytest.mark.parametrize("axis", "xyz")
def test_w3_float64_widths_do_not_warn(row, axis):
    widths = _widths(row, np.float64)
    assert _expected(widths)[1] > .5e-3
    assert not np.all(widths.astype(np.float32).astype(np.float64) == widths)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Float32WidthResolutionWarning)
        _build(axis, widths)


@pytest.mark.parametrize("axis", "xyz")
def test_traced_float32_profile_does_not_warn(axis):
    widths = jnp.asarray(_widths(ROWS[0], np.float32))
    with warnings.catch_warnings():
        warnings.simplefilter("error", Float32WidthResolutionWarning)
        jax.make_jaxpr(lambda w: _build(axis, w).dt)(widths)
