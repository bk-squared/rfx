"""The cached default DFT frequency bins do not depend on whether the first
request for them ran under ``jax.disable_jit()``.

The reference is ``jnp.linspace`` called at this test's top level, outside any
trace: linspace's own compiled path, the arithmetic every run outside
``disable_jit`` publishes. (Inside a user ``jax.jit`` it would be constant-folded
and round differently again.) Under ``disable_jit`` linspace runs op by
op and its float32 bins can differ by 1 ULP; the cache must not keep those.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.measurement.monitors import _eager_frequency_grid, default_frequency_grid


@pytest.fixture
def clean_grid_cache():
    _eager_frequency_grid.cache_clear()
    try:
        yield
    finally:
        _eager_frequency_grid.cache_clear()


@pytest.mark.parametrize("freq_max, n_freqs", [(20e9, 50), (7.3e9, 37)])
def test_default_grid_same_inside_and_outside_disable_jit(clean_grid_cache, freq_max, n_freqs):
    _eager_frequency_grid.cache_clear()
    with jax.disable_jit():
        inside = np.asarray(default_frequency_grid(freq_max, n_freqs))

    _eager_frequency_grid.cache_clear()
    outside = np.asarray(default_frequency_grid(freq_max, n_freqs))

    reference = np.asarray(jnp.linspace(freq_max / 10, freq_max, n_freqs))

    assert inside.dtype == outside.dtype == reference.dtype
    assert np.array_equal(inside, outside), np.max(np.abs(inside - outside))
    assert np.array_equal(outside, reference), np.max(np.abs(outside - reference))
    assert np.array_equal(inside, reference), np.max(np.abs(inside - reference))
