"""A traced check's host function receives its concrete operands unchanged.

Under a trace only tracer leaves pass through the host callback. A Python
float and a float64 NumPy array given beside a traced operand reach the host
function as the caller's own objects, as in an eager call; through the
callback they would arrive as float32 arrays when x64 is off. The cell Courant
refusal of a traced forward therefore prints the plain call's message
character for character. Expected values come from the caller's objects and
from the plain forward call, never from the helper under test.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx._traced_check import traced_check
from rfx.errors import RfxRefusal

NUMBER = 0.1
ARRAY = np.array([1.0 / 3.0, 0.1, 1e-300], dtype=np.float64)
ARRAY_BYTES = ARRAY.tobytes()
LANES = (1.5, 2.5, 3.5)

FORMS = {
    "debug": dict(ordered=True),
    "io": dict(ordered=False, result_shape=jax.ShapeDtypeStruct((), jnp.int32)),
}


def _recording_host():
    calls = []

    def host(number, array, traced):
        calls.append((number, array, np.asarray(traced).copy()))
        return np.int32(0)

    return calls, host


def _eager_operand_kinds():
    calls, host = _recording_host()
    traced_check("test", "concrete operands", host, NUMBER, ARRAY, np.float32(LANES[0]))
    number, array, _ = calls[0]
    return type(number), type(array), array.dtype


def _program(host, form):
    def run(value):
        token = traced_check("test", "concrete operands", host, NUMBER, ARRAY, value,
                             **FORMS[form])
        return value if token is None else value + token.astype(value.dtype)

    return run


def _assert_concrete_operands_unchanged(calls):
    number_type, array_type, array_dtype = _eager_operand_kinds()
    assert number_type is float and array_type is np.ndarray and array_dtype == np.float64
    for number, array, _ in calls:
        assert number is NUMBER
        assert type(number) is number_type
        assert array is ARRAY
        assert type(array) is array_type
        assert array.dtype == array_dtype
        assert array.tobytes() == ARRAY_BYTES


@pytest.mark.parametrize("form", list(FORMS))
def test_concrete_operands_reach_the_host_unchanged_under_jit(form):
    calls, host = _recording_host()
    out = jax.jit(_program(host, form))(jnp.float32(LANES[1]))
    np.asarray(out)
    jax.effects_barrier()
    assert len(calls) == 1
    _assert_concrete_operands_unchanged(calls)
    assert calls[0][2].dtype == np.float32
    assert float(calls[0][2]) == LANES[1]


@pytest.mark.parametrize("form", list(FORMS))
def test_concrete_operands_reach_the_host_unchanged_under_jit_vmap(form):
    calls, host = _recording_host()
    out = jax.jit(jax.vmap(_program(host, form)))(jnp.asarray(LANES, jnp.float32))
    np.asarray(out)
    jax.effects_barrier()
    assert len(calls) == len(LANES)
    _assert_concrete_operands_unchanged(calls)
    assert sorted(float(traced) for _, _, traced in calls) == sorted(LANES)


CELL = 1e-3
CELLS = 10
STEPS = 2
REFUSED_EPS = 0.8
ADMITTED_EPS = (2.0, 1.5)


def _scene():
    """The step-limit cost scene: a CPML cube with one field source and one probe."""
    side = CELLS * CELL
    sim = Simulation(freq_max=10e9, domain=(side,) * 3, dx=CELL, boundary="cpml")
    mid = round(CELLS / 2) * CELL
    sim.add_source((round(3 * CELLS / 8) * CELL, mid, mid), "ez", amplitude_kind="field")
    sim.add_probe((round(5 * CELLS / 8) * CELL, mid, mid), "ez")
    return sim


def _series(sim):
    def run(eps):
        return sim.forward(eps_override=eps, n_steps=STEPS, skip_preflight=True).time_series

    return run


@pytest.fixture(scope="module")
def courant_case():
    sim = _scene()
    shape = tuple(int(n) for n in sim._build_grid().shape)
    with pytest.raises(RfxRefusal) as caught:
        _series(sim)(jnp.full(shape, REFUSED_EPS, jnp.float32))
    plain = str(caught.value)
    assert plain.startswith("cell Courant limit: ")
    return sim, shape, plain


def _traced_error_text(program, eps):
    with pytest.raises(Exception) as caught:
        np.asarray(program(eps))
        jax.effects_barrier()
    return str(caught.value)


def test_cell_courant_refusal_under_jit_prints_the_plain_message(courant_case):
    sim, shape, plain = courant_case
    text = _traced_error_text(jax.jit(_series(sim)), jnp.full(shape, REFUSED_EPS, jnp.float32))
    assert plain in text


def test_cell_courant_refusal_under_jit_vmap_prints_the_plain_message(courant_case):
    sim, shape, plain = courant_case
    lanes = (ADMITTED_EPS[0], REFUSED_EPS, ADMITTED_EPS[1])
    eps = jnp.stack([jnp.full(shape, v, jnp.float32) for v in lanes])
    text = _traced_error_text(jax.jit(jax.vmap(_series(sim))), eps)
    assert plain in text
