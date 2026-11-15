"""A transformed ForwardResult keeps only host data in pytree static state."""
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


def _forward(scale=None):
    dx = DX
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
        sim.add_port(position=(3.3*MM, 3.8*MM, .8*MM), component='ez',
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
    return _forward()


@pytest.fixture(scope='module', params=['grad', 'jvp', 'jacfwd'])
def transformed(request):
    def objective(scale):
        result = _forward(scale)
        value = jnp.sum(jnp.abs(result.ringdown.s_params)**2)
        # Whole-ForwardResult jacfwd has a separate string-leaf limitation.
        return value, result.ringdown if request.param == 'jacfwd' else result

    scale = jnp.float32(1)
    if request.param == 'grad':
        result = jax.grad(objective, has_aux=True)(scale)[1]
    elif request.param == 'jvp':
        result = jax.jvp(objective, (scale,), (jnp.ones_like(scale),), has_aux=True)[2]
    else:
        result = jax.jacfwd(objective, has_aux=True)(scale)[1]
    return result


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


def test_report_after_transform_equals_eager(transformed, eager):
    rd = getattr(transformed, 'ringdown', transformed)
    actual, expected = rd.report, eager.ringdown.report
    assert actual is not None and expected is not None
    assert actual.completed == expected.completed
    assert actual.ok == expected.ok
    assert [w.name for w in actual.witnesses] == [w.name for w in expected.witnesses]
    for got, want in zip(actual.witnesses, expected.witnesses, strict=True):
        assert (got.judged, got.ok) == (want.judged, want.ok), want.name
        if np.isnan(want.value):
            assert np.isnan(got.value), want.name
        elif np.isinf(want.value):
            assert got.value == want.value, want.name
        else:
            bound = 1e-4 * max(abs(want.value), want.bar)
            assert abs(got.value - want.value) <= bound, (
                f'{want.name}: traced={got.value}, eager={want.value}, '
                f'bar={want.bar}, difference={abs(got.value - want.value)}, bound={bound}')
