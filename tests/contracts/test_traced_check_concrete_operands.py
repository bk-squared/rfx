"""A traced check's host function receives its concrete operands unchanged.

Under a trace only tracer and concrete jax.Array leaves pass through the host
callback. A Python float and a float64 NumPy array given beside a traced
operand reach the host function as the caller's own objects, as in an eager
call; through the callback they would arrive as float32 arrays when x64 is
off. A concrete device array stays staged and arrives with its dtype and
values. The cell Courant refusal of a traced forward therefore prints the
plain call's message character for character, and a periodic fill check
given one traced and one Python-float bound admits the box as the plain call
does. Expected values come from the caller's objects and from the plain
call, never from the helper under test.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx._traced_check import traced_check
from rfx.boundaries.spec import BoundarySpec
from rfx.errors import RfxRefusal
from rfx.geometry.smoothing import compute_smoothed_eps

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


DEVICE_VALUES = np.array([0.1, 1.0 / 3.0], dtype=np.float32)


def _debug_callback_equations(jaxpr):
    found = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "debug_callback":
            found.append(eqn)
        for value in eqn.params.values():
            inner = getattr(value, "jaxpr", value)
            if hasattr(inner, "eqns"):
                found.extend(_debug_callback_equations(inner))
    return found


def test_device_array_operands_stay_staged_and_arrive_unchanged_under_jit():
    device = jnp.asarray(DEVICE_VALUES)
    concrete = (NUMBER, ARRAY, device)
    expected_staged = 1 + sum(isinstance(v, jax.Array) for v in concrete)
    calls = []

    def host(number, array, device_value, traced):
        calls.append((number, array, np.asarray(device_value).copy(), np.asarray(traced).copy()))

    def run(value):
        traced_check("test", "device operands", host, *concrete, value, ordered=True)
        return value

    program = jax.jit(run)
    equations = _debug_callback_equations(jax.make_jaxpr(program)(jnp.float32(LANES[1])).jaxpr)
    assert len(equations) == 1
    assert len(equations[0].invars) == expected_staged == 2
    np.asarray(program(jnp.float32(LANES[1])))
    jax.effects_barrier()
    assert len(calls) == 1
    number, array, device_value, traced = calls[0]
    assert number is NUMBER and array is ARRAY
    assert device_value.dtype == device.dtype == np.float32
    assert device_value.tobytes() == DEVICE_VALUES.tobytes()
    assert float(traced) == LANES[1]


MM = 1e-3
PERIOD = 10 * MM
BOX_LO = float(np.float32(2.2 * MM))
BOX_HI = 5.3 * MM


def _periodic_grid():
    """A 10 mm periodic x axis between PEC y and z walls, 1 mm cells."""
    return Simulation(20e9, (PERIOD, 6 * MM, 6 * MM), dx=MM, cpml_layers=0,
                      boundary=BoundarySpec(x="periodic", y="pec", z="pec")).freeze_mesh()


def _periodic_box(lo, hi):
    """A Box between the x bounds; off-lattice faces across y and z."""
    c = 3 * MM
    return Box((lo, c - 1.63e-3, c - 1.39e-3), (hi, c + 1.83e-3, c + 2.07e-3))


def test_box_with_one_traced_corner_on_a_periodic_axis_fills_as_the_plain_call():
    """The jit fill matches the plain fill within rfx's cross-trace bar.

    The traced corner is rounded in float32 arithmetic inside the program, so
    the two fills agree to 9 float32 ULP of the peak permittivity, not bitwise.
    """
    grid = _periodic_grid()

    def fill(lo):
        return jnp.stack([jnp.asarray(a) for a in
                          compute_smoothed_eps(grid, [(_periodic_box(lo, BOX_HI), 4.)])])

    assert 0 <= (BOX_LO + BOX_HI) / 2 <= PERIOD
    traced = np.asarray(jax.block_until_ready(jax.jit(fill)(jnp.float32(BOX_LO))))
    jax.effects_barrier()
    plain = np.asarray(fill(BOX_LO))
    assert traced.shape == plain.shape
    peak = float(np.max(np.abs(plain)))
    np.testing.assert_allclose(traced, plain, rtol=0,
                               atol=9 * float(np.spacing(np.float32(peak))))
