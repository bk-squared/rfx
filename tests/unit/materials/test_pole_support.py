"""Independent NumPy mask references for the static ADE support contract (E1)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import init_materials
from rfx.materials.debye import DebyePole, init_debye
from rfx.materials.lorentz import drude_pole, init_lorentz, lorentz_pole
from rfx.materials.pole_support import align_pole_states


def numpy_fractions(mask, periodic=(False,) * 3):
    """Four incident cell masks, using NumPy only, without product helpers."""
    result = []
    for component in range(3):
        axes = [axis for axis in range(3) if axis != component]
        cells = []
        for first in (0, 1):
            for second in (0, 1):
                a = np.asarray(mask, dtype=np.float32)
                for axis, offset in zip(axes, (first, second)):
                    index = np.arange(a.shape[axis]) - offset
                    index = index % a.shape[axis] if periodic[axis] else np.maximum(index, 0)
                    a = np.take(a, index, axis=axis)
                cells.append(a)
        result.append(np.sum(cells, axis=0) / 4)
    return result


def numpy_bounds(masks, shape, periodic=(False,) * 3):
    if any(m is None for m in masks):
        return tuple((0, n) for n in shape)
    reach = np.any([f > 0 for m in masks for f in numpy_fractions(m, periodic)], axis=0)
    return tuple((int(i.min()), int(i.max()) + 1) if i.size else (0, 0)
                 for i in np.nonzero(reach))


@pytest.mark.parametrize('model', ('debye', 'lorentz', 'drude'))
@pytest.mark.parametrize('kind', ('interior', 'edge', 'two_masks', 'empty', 'everywhere', 'periodic'))
def test_polarization_box_is_exact_pole_reach(model, kind):
    shape = (9, 10, 11)
    mask = np.zeros(shape, bool)
    mask[2:5, 3:6, 4:7] = True
    if kind == 'edge':
        mask[0:2, 0:2, -2:] = True
    if kind == 'periodic':
        mask[-1, -1, -1] = True
    if kind == 'empty':
        mask[:] = False
    masks = [None if kind == 'everywhere' else mask]
    if kind == 'two_masks':
        other = np.zeros(shape, bool)
        other[6:8, 7:9, 8:10] = True
        masks.append(other)
    periodic = (True,) * 3 if kind == 'periodic' else (False,) * 3
    if model == 'debye':
        poles = [DebyePole(2., 1e-10)] * len(masks)
        coeffs, state = init_debye(poles, init_materials(shape), 1e-12, masks, periodic=periodic)
        coupling = coeffs.beta
    else:
        pole = drude_pole(1e10, 1e9) if model == 'drude' else lorentz_pole(2., 1e10, 1e9)
        coeffs, state = init_lorentz([pole] * len(masks), init_materials(shape),
                                     1e-12, masks, periodic=periodic)
        coupling = coeffs.c
    expected = numpy_bounds(masks, shape, periodic)
    assert state.box.bounds == expected
    assert state.px.shape == (len(masks),) + tuple(b - a for a, b in expected)
    inside = np.zeros(shape, bool)
    inside[tuple(slice(a, b) for a, b in expected)] = True
    for component in coupling:
        assert component.shape == (len(masks),) + shape
        assert not np.any(np.asarray(component)[:, ~inside] != 0)
    for a in jax.tree.leaves(state):
        assert a.shape == state.px.shape
    print(f'E1 model={model} kind={kind} bounds={expected} outside_nonzero=0')


def test_mixed_states_share_union_box():
    shape = (9, 10, 11)
    a, b = np.zeros(shape, bool), np.zeros(shape, bool)
    a[1:3, 2:4, 3:5], b[5:7, 6:8, 7:9] = True, True
    mats = init_materials(shape)
    d = init_debye([DebyePole(2., 1e-10)], mats, 1e-12, a)
    l = init_lorentz([lorentz_pole(2., 1e10, 1e9)], mats, 1e-12, b)
    d, l = align_pole_states(d, l, shape)
    assert d[1].box == l[1].box
    assert d[1].box.bounds == numpy_bounds([a, b], shape)
    assert d[1].px.shape == l[1].px.shape


def test_traced_fractions_keep_full_state():
    shape = (4, 5, 6)
    mats = init_materials(shape)
    initialize = jax.jit(lambda mask: init_debye([DebyePole(2., 1e-10)], mats, 1e-12, mask)[1])
    state = initialize(jnp.zeros(shape, bool))
    assert state.box.bounds == tuple((0, n) for n in shape)
    assert state.px.shape == (1,) + shape


@pytest.mark.parametrize('model', ('debye', 'lorentz'))
def test_state_split_extends_original_x_and_preserves_transverse_support(model):
    from rfx.runners._distributed_common import _split_debye_state, _split_lorentz_state
    shape = (9, 10, 11)
    mask = np.zeros(shape, bool)
    mask[2:5, 3:6, 4:7] = True
    mats = init_materials(shape)
    if model == 'debye':
        _, state = init_debye([DebyePole(2., 1e-10)], mats, 1e-12, mask)
        split = _split_debye_state
    else:
        _, state = init_lorentz([lorentz_pole(2., 1e10, 1e9)], mats, 1e-12, mask)
        split = _split_lorentz_state
    bounds = numpy_bounds([mask], shape)
    values = np.arange(np.prod(state.px.shape), dtype=np.float32).reshape(state.px.shape) + 1
    state = state._replace(px=jnp.asarray(values))
    packed = split(state, 2)
    assert packed.box.bounds == ((0, 7),) + bounds[1:]
    assert packed.px.shape == (2, 1, 7, 4, 4)
    reference = np.zeros((1,) + shape, np.float32)
    reference[(slice(None),) + tuple(slice(lo, hi) for lo, hi in bounds)] = values
    for rank in range(2):
        rows = np.arange(rank * 5 - 1, rank * 5 + 6)
        expected = np.zeros((1, 7, 4, 4), np.float32)
        valid = (rows >= 0) & (rows < shape[0])
        expected[:, valid] = reference[:, rows[valid], 3:7, 4:8]
        np.testing.assert_array_equal(np.asarray(packed.px[rank]), expected)
