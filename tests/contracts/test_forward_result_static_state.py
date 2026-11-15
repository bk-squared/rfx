"""A transformed ForwardResult keeps only host data in pytree static state.

Scene A is the issue's reproduction (scale 1, port at x node 4). Scene B pins
the realized-width case found in review: scale 1.05 with the port at x node 8,
next to the traced cells, so the report must read the realized mesh.
"""
from dataclasses import fields, is_dataclass
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.model import NominalProfile
from rfx.ringdown import RingdownSpec

MM = 1e-3
DX = np.array([1., .8, .6, .9, 1.2, 1.4, 1.2, 1.1, 1., .9, .9, 1.]) * MM
DY = np.array([1., 1.4, .8, .6, .8, .9, 1., 1.1, 1.2, 1.2, 1.]) * MM
DZ = np.array([.8, .6, 1.4, 1.4, .8]) * MM
DT = .9 / (299792458. * np.sqrt(3 / (.6 * MM)**2))
FREQS = np.linspace(10e9, 16e9, 7)
N_STEPS = 600  # Smallest admitted of {300, 600, 1200}; default RingdownSpec.
PORT_X_A = 3.3 * MM  # x node 4: 1.0+0.8+0.6+0.9 mm
PORT_X_B = 8.2 * MM  # x node 8: 1.0+0.8+0.6+0.9+1.2+1.4+1.2+1.1 mm, the window's upper edge
SCALE_B = 1.05


def _realized_dx(scale):
    """The float32 widths ``_forward(scale)`` builds, in NumPy float32.

    Equal to the traced ``.at[]`` arithmetic op for op (x64 off): ``DX[6] *
    scale`` multiplies the NumPy float64 scalar, rounded to float32, by the
    float32 scale in float32; ``float(DX[6] + DX[7]) - ...`` rounds the Python
    float (weakly typed) to float32 and subtracts in float32. NumPy float32
    scalar arithmetic rounds each operation the same way (IEEE round to nearest).
    """
    w = np.asarray(DX, np.float32).copy()
    w6 = np.float32(DX[6]) * np.float32(scale)
    w[6] = w6
    w[7] = np.float32(float(DX[6] + DX[7])) - w6
    return w


def _forward(scale=None, dx=DX, port_x=PORT_X_A):
    nominal = None
    if scale is not None:
        dx = jnp.asarray(DX, dtype=jnp.float32)
        dx = dx.at[6].set(DX[6] * scale).at[7].set(float(DX[6] + DX[7]) - DX[6] * scale)
        nominal = {'x': NominalProfile(DX, traced_cells=(6, 8))}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sim = Simulation(freq_max=20e9, domain=(12*MM, 11*MM, 5*MM), dx=MM,
                         boundary='pec', dx_profile=dx, dy_profile=DY, dz_profile=DZ,
                         dt=float(DT), dt_min_cell=.6*MM, nominal_profiles=nominal)
        sim.add_material('fill', eps_r=2.2,
                         sigma=2*np.pi*12.5e9*8.854187817e-12*2.2/300)
        sim.add(Box((0, 0, 0), (12*MM, 11*MM, 5*MM)), material='fill')
        sim.add_port(position=(port_x, 3.8*MM, .8*MM), component='ez',
                     impedance=50., extent=.6*MM,
                     waveform=GaussianPulse(f0=13e9, bandwidth=.8, cutoff=4.5))
        return sim.forward(n_steps=N_STEPS, port_s11_freqs=FREQS,
                           ringdown=RingdownSpec(), skip_preflight=True)


def _assert_static_state(result):
    """Inspect every node's aux, then all objects reachable from that aux.

    Independent of rfx's context/grid helpers; attributes include slots and
    dataclasses, and a seen-set permits shared objects and cycles.
    """
    seen = set()

    def static(value, path):
        assert not isinstance(value, jax.core.Tracer), f'tracer in static state: {path}'
        if id(value) in seen:
            return
        seen.add(id(value))
        if value is None or isinstance(value, (str, bytes, bool, int, float, complex,
                                               np.generic, np.ndarray, jax.Array, type)):
            return
        if isinstance(value, dict):
            for key, item in value.items():
                static(key, f'{path}.key')
                static(item, f'{path}[{key!r}]')
        elif isinstance(value, (tuple, list)):
            names = getattr(value, '_fields', range(len(value)))
            for name, item in zip(names, value):
                static(item, f'{path}.{name}')
        elif is_dataclass(value):
            for field in fields(value):
                static(getattr(value, field.name), f'{path}.{field.name}')
        else:
            for name, item in vars(value).items() if hasattr(value, '__dict__') else ():
                static(item, f'{path}.{name}')
            for cls in type(value).__mro__:
                slots = getattr(cls, '__slots__', ())
                for name in (slots,) if isinstance(slots, str) else slots:
                    if hasattr(value, name):
                        static(getattr(value, name), f'{path}.{name}')
            leaves, definition = jax.tree_util.tree_flatten(value)
            if not (len(leaves) == 1 and leaves[0] is value):
                for i, leaf in enumerate(leaves):
                    static(leaf, f'{path}.leaf[{i}]')
                aux(definition, path)

    def aux(definition, path):
        data = definition.node_data()
        if data is not None:
            static(data[1], f'{path}.aux')
        for i, child in enumerate(definition.children()):
            aux(child, f'{path}[{i}]')

    aux(jax.tree_util.tree_structure(result), 'result')


@pytest.fixture(scope='module')
def eager():
    # The realized widths at scale 1 as a concrete float32 profile, no
    # nominal_profiles: the report reads the realized mesh (Addendum 3), and
    # cell 7 at scale 1 may sit one float32 ulp from the declared width.
    return _forward(dx=_realized_dx(1.0))


def _transform(kind, scale_value, port_x):
    def objective(scale):
        result = _forward(scale, port_x=port_x)
        value = jnp.sum(jnp.abs(result.ringdown.s_params)**2)
        # Whole-ForwardResult jacfwd has a separate string-leaf limitation.
        return value, result.ringdown if kind == 'jacfwd' else result

    scale = jnp.float32(scale_value)
    if kind == 'grad':
        result = jax.grad(objective, has_aux=True)(scale)[1]
    elif kind == 'jvp':
        result = jax.jvp(objective, (scale,), (jnp.ones_like(scale),), has_aux=True)[2]
    else:
        result = jax.jacfwd(objective, has_aux=True)(scale)[1]
    return result


@pytest.fixture(scope='module', params=[
    'grad', pytest.param('jvp', marks=pytest.mark.slow),
    pytest.param('jacfwd', marks=pytest.mark.slow)])
def transformed(request):
    return _transform(request.param, 1, PORT_X_A)


@pytest.fixture(scope='module')
def eager_b():
    # The realized float32 widths at SCALE_B as a concrete profile, no
    # nominal_profiles: the concrete run of scene B's design step.
    return _forward(dx=_realized_dx(SCALE_B), port_x=PORT_X_B)


@pytest.fixture(scope='module', params=['grad', 'jvp'])
def transformed_b(request):
    return _transform(request.param, SCALE_B, PORT_X_B)


def test_no_tracer_in_forward_result_static_state(transformed):
    _assert_static_state(transformed)
    rd = getattr(transformed, 'ringdown', transformed)
    # Expectations come directly from the declared cells, without rfx helpers.
    padded = np.concatenate((DX, DX[-1:])).astype(np.float32)
    np.testing.assert_array_equal(rd._ctx.grid.dx_arr, padded)
    expected_e = np.concatenate((1/padded[:1], 2/(padded[:-1] + padded[1:])))
    expected_h = np.concatenate((1/padded[:-1], np.zeros(1, np.float32)))
    np.testing.assert_array_equal(rd._ctx.grid.inv_dx, expected_e)
    np.testing.assert_array_equal(rd._ctx.grid.inv_dx_h, expected_h)
    assert rd._ctx.plan.grid is rd._ctx.grid
    meta = rd._ctx.metas[0]
    assert meta[5] == float(padded[4])
    assert meta[9] == (float(padded[3]) + float(padded[4])) / 2


def _same(a, b):
    """Exact equality; NaN equals NaN."""
    if a == b:
        return True
    numbers = (float, np.floating)
    return isinstance(a, numbers) and isinstance(b, numbers) and np.isnan(a) and np.isnan(b)


def test_report_with_eager_context_is_identical(transformed, eager):
    """The context a transformed run keeps acts exactly as a concrete run's.

    The transformed run's own record (its children) read through an eager
    run's context gives the same report, bit for bit, as through its own.
    """
    _assert_swapped_report_identical(transformed, eager)


def _assert_swapped_report_identical(transformed, eager):
    rd = getattr(transformed, 'ringdown', transformed)
    children = rd.tree_flatten()[0]
    swapped = type(rd).tree_unflatten(eager.ringdown._ctx, children)
    actual, expected = rd.report, swapped.report
    assert actual is not None and expected is not None
    assert actual.completed == expected.completed
    assert actual.ok == expected.ok
    assert [w.name for w in actual.witnesses] == [w.name for w in expected.witnesses]
    for got, want in zip(actual.witnesses, expected.witnesses, strict=True):
        for field in ('value', 'bar', 'judged', 'ok', 'note'):
            g, w = getattr(got, field), getattr(want, field)
            assert _same(g, w), (
                f'{want.name}.{field}: own context={g!r}, eager context={w!r}')
    assert actual.summary() == expected.summary()


def test_scene_b_no_tracer_in_static_state(transformed_b):
    _assert_static_state(transformed_b)
    rd = transformed_b.ringdown
    realized = _realized_dx(SCALE_B)
    # the result carries the realized x cells of this design step as a child
    assert len(rd._cells) == 1
    np.testing.assert_array_equal(np.asarray(rd._cells[0]),
                                  np.concatenate((realized, realized[-1:])))
    # the static template stays on the declared nominal widths
    np.testing.assert_array_equal(rd._ctx.grid.dx_arr,
                                  np.concatenate((DX, DX[-1:])).astype(np.float32))


def test_scene_b_report_with_eager_context_is_identical(transformed_b, eager_b):
    """At a design step whose widths at the port differ from the nominal ones,
    the report read through the transformed run's own context equals the one
    read through the context of a concrete run on the realized widths."""
    _assert_swapped_report_identical(transformed_b, eager_b)


def test_scene_b_w0_passes(transformed_b):
    report = transformed_b.ringdown.report
    assert report is not None
    (w0,) = [w for w in report.witnesses if w.name == 'W0']
    assert w0.ok, f'W0 failed on the realized mesh: value={w0.value!r}, bar={w0.bar!r}, note={w0.note!r}'
