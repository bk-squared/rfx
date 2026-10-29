"""Concrete sheet storage scales with its plane after production assembly."""
from dataclasses import fields, is_dataclass, replace

import jax
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.pec import SheetFootprint, SheetSpec
from rfx.model.conductors import RealizedSheet, realized_conductors


def assert_plane_store(sheet):
    for field in fields(sheet):
        value = getattr(sheet, field.name)
        assert not isinstance(value, jax.Array), field.name
        assert not (isinstance(value, np.ndarray) and value.ndim == 3), field.name
    for name in ('footprint', 'unwrapped_footprint'):
        value = getattr(sheet, name)
        if value is None:
            continue
        assert value.mask.ndim == 2 and isinstance(value.mask, np.ndarray)
        assert value.mask.dtype == bool
        assert value.mask.size == np.prod([n for a, n in enumerate(value.shape)
                                          if a != sheet.normal_axis])


def test_realized_assembly_stores_only_host_planes():
    sim = Simulation(freq_max=5e9, domain=(.012, .010, .008), dx=.001,
                     boundary='pec')
    for z in (.002, .005):
        sim.add_thin_conductor(Box((.003, .003, z), (.009, .007, z)))
    root = realized_conductors(sim, sim._build_grid())
    seen, sheets = set(), []

    def walk(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, SheetSpec):
            assert_plane_store(value)
            sheets.append(value)
        if is_dataclass(value):
            for field in fields(value):
                walk(getattr(value, field.name))
        elif isinstance(value, (tuple, list)):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

    walk(root)
    assert len([s for s in sheets if isinstance(s, RealizedSheet)]) == 2


def test_storage_check_rejects_whole_grid():
    sheet = SheetSpec.from_node_ranges((7, 8, 9), normal_axis=2, plane=4,
                                      in_plane_ranges=((2, 5), (2, 5)))
    object.__setattr__(sheet, 'footprint', np.asarray(sheet.footprint))
    with pytest.raises(AssertionError, match='footprint'):
        assert_plane_store(sheet)


@pytest.mark.parametrize('normal', range(3))
@pytest.mark.parametrize('input_kind', ('numpy', 'device', 'list', 'view'))
def test_array_view_and_replace_preserve_content(normal, input_kind):
    expected = np.zeros((7, 8, 9), dtype=bool)
    index = [slice(1, 6), slice(2, 7), slice(3, 8)]
    index[normal] = 3
    expected[tuple(index)] = True
    inputs = {'numpy': lambda: expected,
              'device': lambda: jax.device_put(expected),
              'list': lambda: expected.tolist(),
              'view': lambda: SheetFootprint.normalize(expected, normal, 3)}
    sheet = SheetSpec(normal, 3, inputs[input_kind](), unwrapped_footprint=expected)
    assert_plane_store(sheet)
    np.testing.assert_array_equal(np.asarray(sheet.footprint), expected)
    np.testing.assert_array_equal(sheet.footprint[1:6], expected[1:6])
    cut = replace(sheet, plane=2 if normal == 0 else 3,
                  footprint=sheet.footprint.slab(1, 6),
                  unwrapped_footprint=sheet.unwrapped_footprint.slab(1, 6))
    np.testing.assert_array_equal(np.asarray(cut.footprint), expected[1:6])
    view = np.asarray(sheet.footprint)
    view[:] = False
    np.testing.assert_array_equal(np.asarray(sheet.footprint), expected)
    assert not np.shares_memory(sheet.footprint.mask, expected)
