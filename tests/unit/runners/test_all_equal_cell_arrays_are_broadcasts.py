"""An all-equal per-cell array is rebuilt inside the jitted step with the same bits, or left alone (rfx tracker 1606)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.jax_utils import split_loop_invariants

SHAPE = (5, 4, 3)


def _rebuilt(value):
    arrays, rebuild = split_loop_invariants({'x': value}, SHAPE)
    return arrays, np.asarray(jax.jit(lambda a: rebuild(a)['x'])(arrays)), rebuild


@pytest.mark.parametrize('value', [
    jnp.full(SHAPE, 2.5, jnp.float32), jnp.zeros(SHAPE, jnp.float32), -jnp.zeros(SHAPE, jnp.float32),
    jnp.full(SHAPE, 0.75, jnp.float16), jnp.full(SHAPE, 1 - 2j, jnp.complex64), jnp.full(SHAPE, 3, jnp.int32),
    jnp.zeros(SHAPE, bool), jnp.ones(SHAPE, bool), jnp.full((2,) + SHAPE, 0.4545, jnp.float32),
    jnp.full(SHAPE, jnp.inf, jnp.float32),
])
def test_one_bit_pattern_is_rebuilt_with_the_same_bits_and_not_closed_over(value):
    arrays, out, rebuild = _rebuilt(value)
    assert arrays == []
    assert out.dtype == value.dtype and out.shape == value.shape
    assert out.tobytes() == np.asarray(value).tobytes()
    # The rebuild function holds no array of the grid's size.
    cells = [c.cell_contents for c in (rebuild.__closure__ or ())]
    assert not any(isinstance(c, (jax.Array, np.ndarray)) and c.size > 1 for c in cells)


def test_mixed_signed_zeros_are_not_rebuilt_from_one_element():
    """+0.0 == -0.0, but a broadcast of the first element would flip the others' sign bit."""
    value = jnp.zeros(SHAPE, jnp.float32).at[1, 2, 0].set(-0.0)
    arrays, out, _ = _rebuilt(value)
    assert out.tobytes() == np.asarray(value).tobytes()


def test_a_numpy_leaf_stays_the_host_array_it_was():
    value = np.full(SHAPE, 2.0, np.float64)
    arrays, rebuild = split_loop_invariants({'x': value}, SHAPE)
    assert arrays == [] and rebuild([])['x'] is value


def test_a_varying_array_is_still_an_argument():
    value = jnp.arange(60, dtype=jnp.float32).reshape(SHAPE)
    arrays, out, _ = _rebuilt(value)
    assert len(arrays) == 1 and out.tobytes() == np.asarray(value).tobytes()
