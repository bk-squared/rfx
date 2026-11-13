"""Scoped-x64 host checks retain operand bits on worker threads, rfx#2613."""
import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx._traced_check import traced_check
from tests._x64_compat import enable_x64


@pytest.mark.parametrize('kind', ['debug', 'io'])
def test_off_thread_check_preserves_bits_and_scope(kind):
    caller = threading.get_ident()
    real = np.array([1 / 3, .1, .0005], dtype=np.float64)
    integer = np.array([2**40 + 1], dtype=np.int64)
    unsigned = np.array([2**63 + 5], dtype=np.uint64)
    complex_value = np.array([1 / 3 + .1j], dtype=np.complex128)
    expected = (real, integer, unsigned, complex_value)
    witnessed = []
    received = []
    errors = []

    def witness(*values):
        witnessed.append((threading.get_ident(), jax.config.jax_enable_x64,
                          tuple(np.asarray(v).copy() for v in values)))

    def check(*values):
        # This conversion separately pins the trace-time x64 scope (t7b).
        formed = jnp.asarray(np.asarray(values[0]), dtype=jnp.float64)
        received.append((threading.get_ident(), jax.config.jax_enable_x64,
                         tuple(np.asarray(v).copy() for v in values),
                         np.asarray(formed).copy()))
        return np.int32(1) if kind == 'io' else None

    def program(*values):
        jax.debug.callback(witness, *values)
        token = traced_check(
            'transport contract', 'operand bits', check, *values,
            result_shape=jax.ShapeDtypeStruct((), jnp.int32) if kind == 'io' else None)
        return token if kind == 'io' else jnp.int32(1)

    print(f'B_GRID_DEVICES count={len(jax.devices())} devices={jax.devices()}')
    with enable_x64():
        inputs = tuple(jnp.asarray(v) for v in expected)
        compiled = jax.jit(program).lower(*inputs).compile()

    def execute():
        try:
            assert not jax.config.jax_enable_x64
            result = compiled(*inputs)
            jax.block_until_ready(result)
            jax.effects_barrier()
            assert int(result) == 1
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=execute, name='transport-contract')
    worker.start()
    worker.join()
    if errors:
        raise errors[0]
    assert len(witnessed) == len(received) == 1
    witness_thread, witness_x64, witness_values = witnessed[0]
    print(f'WITNESS kind={kind} caller={caller} callback={witness_thread} '
          f'x64={witness_x64} dtype={witness_values[0].dtype}')
    # (a) The raw callback must reproduce narrowing before testing the fix.
    assert witness_values[0].dtype == np.dtype('float32'), 'narrowing not reproduced'
    thread, x64, values, formed = received[0]
    print(f'TRANSPORT kind={kind} caller={caller} callback={thread} x64={x64}')
    # (b) Compare bytes in the same traced dtype, without a numerical tolerance.
    for actual, reference in zip(values, expected):
        assert actual.dtype == reference.dtype
        np.testing.assert_array_equal(actual.view(np.uint8), reference.view(np.uint8))
    # (c) Array construction inside check must also retain the traced bits.
    assert formed.dtype == real.dtype
    np.testing.assert_array_equal(formed.view(np.uint8), real.view(np.uint8))
    jax.clear_caches()
