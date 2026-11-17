"""Host-only identity against pre-table conformal operands and table facts."""

from dataclasses import FrozenInstanceError
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import pytest

from rfx import Sphere
from rfx.geometry import cell_table
from rfx.model.conformal_rules import conformal_fractions_from_table
from tests._cell_table_capture import record_cell_table_capture
from tests._cell_table_scenes import SCENES, cell_table_scene


REFERENCE = Path(__file__).resolve().parents[2] / 'data' / 'cell_table_56653420b.npz'
CAPTURE_REFERENCE = REFERENCE.with_name('cell_table_capture_56653420b.npz')


def _declared_sets(scene, grid, shapes):
    """Analytic closed-body membership in exact 0.1 mm units, independent of rfx geometry."""
    if scene not in ('box_nodes', 'cylinder'):
        return None

    def units(value):
        scaled = Fraction(str(value)) * 10000
        assert scaled.denominator == 1
        return int(scaled)

    # One extra owner and its endpoint allow the high bounding edges of each face.
    lines = [(np.arange(n + 2, dtype=np.int64) - getattr(grid, f'pad_{axis}_lo'))
             * units(grid.dx) for axis, n in zip('xyz', grid.shape)]
    coords = np.meshgrid(*lines, indexing='ij', sparse=True)
    body = shapes[0]
    if scene == 'box_nodes':
        lo, hi = [[units(v) for v in corner] for corner in (body.corner_lo, body.corner_hi)]
        inside = np.ones(tuple(n + 2 for n in grid.shape), dtype=bool)
        surface = np.zeros_like(inside)
        for axis, coordinate in enumerate(coords):
            inside &= (coordinate >= lo[axis]) & (coordinate <= hi[axis])
            surface |= (coordinate == lo[axis]) | (coordinate == hi[axis])
        surface &= inside
    else:
        assert body.axis == 'z'
        cx, cy, cz = map(units, body.center)
        radius, height = units(body.radius), units(body.height)
        radial = (coords[0] - cx) ** 2 + (coords[1] - cy) ** 2
        twice_z = 2 * np.abs(coords[2] - cz)
        inside = (radial <= radius ** 2) & (twice_z <= height)
        surface = inside & ((radial == radius ** 2) | (twice_z == height))

    def at_nodes(array, offsets):
        return array[tuple(slice(offset, offset + n + 1)
                           for offset, n in zip(offsets, grid.shape))]

    edges, faces = [], []
    for axis in range(3):
        end = tuple(int(a == axis) for a in range(3))
        edges.append(at_nodes(inside, (0, 0, 0)) & at_nodes(inside, end)
                     & (at_nodes(surface, (0, 0, 0)) | at_nodes(surface, end)))
        u, v = [a for a in range(3) if a != axis]
        corners = [tuple(int(a in moved) for a in range(3))
                   for moved in ((), (u,), (u, v), (v,))]
        faces.append(np.logical_and.reduce([at_nodes(inside, c) for c in corners])
                     & np.logical_or.reduce([at_nodes(surface, c) for c in corners]))

    physical = tuple(slice(0, n) for n in grid.shape)
    adjacent = []
    for normal in range(3):
        mask = np.zeros(grid.shape, dtype=bool)
        for axis in range(3):
            if axis == normal:
                continue
            transverse = 3 - axis - normal
            high = tuple(slice(int(a == transverse), n + int(a == transverse))
                         for a, n in enumerate(grid.shape))
            mask |= edges[axis][physical] | edges[axis][high]
        adjacent.append(mask)
    return ([a[physical] for a in edges], [a[physical] for a in faces], adjacent)


def _assert_frozen(value, expected, scene, rule, kind, axis, declared, *, capture=False):
    value = np.asarray(value)
    assert value.shape == expected.shape
    changed = value != expected
    print(f'cell_table {scene} {rule} {"capture" if capture else "operand"} '
          f'{kind} {"xyz"[axis]} changed={np.count_nonzero(changed)}', file=sys.stderr)
    allowed = np.zeros(value.shape, dtype=bool)
    if declared is not None:
        edges, faces, adjacent = declared
        required = (edges if kind == 'edges' else faces)[axis]
        allowed = required.copy()
        if rule == 'floor' and kind == 'faces':
            allowed |= adjacent[axis]
        if rule != 'floor' and not capture:
            np.testing.assert_array_equal(value[required], 0,
                                          err_msg=f'{scene} {rule} {kind} {axis}: D must be zero')
    np.testing.assert_array_equal(value[~allowed], expected[~allowed],
                                  err_msg=f'{scene} {rule} {kind} {axis}: outside declared set')


@pytest.mark.parametrize('scene', SCENES)
@pytest.mark.parametrize('rule', ('sc', 'raw', 'floor'))
def test_frozen_capture_calls(scene, rule):
    from rfx.geometry.conformal_cells import compute_conformal_fractions

    grid, shapes = cell_table_scene(scene)
    declared = _declared_sets(scene, grid, shapes)
    with record_cell_table_capture(grid, shapes) as calls:
        compute_conformal_fractions(
            grid, pec_shapes=shapes, clamp=rule != 'raw',
            conformal_pec='cut_cell_floor' if rule == 'floor' else None, field_dtype=np.float32)
    prefix = f'{scene}_{rule}_'
    actual = {f'{prefix}call_count': np.asarray(len(calls))}
    for index, call in enumerate(calls):
        for argument, value in call.items():
            actual[f'{prefix}{index}_{argument}'] = value
    with np.load(CAPTURE_REFERENCE) as frozen:
        assert set(actual) == {key for key in frozen.files if key.startswith(prefix)}
        for key, value in actual.items():
            assert value.dtype == frozen[key].dtype, key
            if key.endswith('_fraction'):
                call = calls[int(key[len(prefix):].split('_')[0])]
                assert bool(call['face'])
                _assert_frozen(value, frozen[key], scene, rule, 'faces',
                               int(call['component']), declared, capture=True)
            else:
                assert np.array_equal(value, frozen[key]), key


@pytest.mark.parametrize('scene', SCENES)
@pytest.mark.parametrize('rule', ('sc', 'raw', 'floor'))
def test_frozen_operands(scene, rule):
    grid, shapes = cell_table_scene(scene)
    declared = _declared_sets(scene, grid, shapes)
    table = cell_table.build_cell_table(grid, shapes, include_trailing_node=rule == "floor")
    actual = conformal_fractions_from_table(
        table, grid, 'cut_cell_floor' if rule == 'floor' else None,
        clamp=rule != 'raw', field_dtype=np.float32)
    with np.load(REFERENCE) as frozen:
        for kind, values in zip(('edges', 'faces'), actual):
            for axis, value in enumerate(values):
                assert value.shape == grid.shape
                assert value.dtype == np.float32
                _assert_frozen(value, frozen[f'{scene}_{rule}_{kind}_{axis}'],
                               scene, rule, kind, axis, declared)
    for arrays, counts, cut in ((table.lengths, table.edge_part_counts, table.cut_edges),
                               (table.areas, table.face_part_counts, table.cut_faces)):
        for axis, (array, count) in enumerate(zip(arrays, counts)):
            assert isinstance(array, np.ndarray) and array.dtype == np.float64
            assert array.shape == tuple(n + 2 for n in grid.shape)
            assert count.shape == array.shape and count.dtype == np.int8
            assert np.all(count == 1)
            assert not array.flags.writeable and not count.flags.writeable
            assert cut[axis] == np.count_nonzero((array > 0) & (array < 1))
    assert table.part_owners.shape == (0, 4) and table.part_owners.dtype == np.int32
    assert table.part_indices.shape == (0,) and table.part_indices.dtype == np.int8
    assert table.part_fractions.shape == (0,) and table.part_fractions.dtype == np.float64
    for array in (*table.edge_part_counts, *table.face_part_counts,
                  table.part_owners, table.part_indices, table.part_fractions):
        assert not array.flags.writeable
    assert len(table.topology_key) == 64
    assert int(table.topology_key, 16) >= 0
    with pytest.raises(FrozenInstanceError):
        table.topology_key = ''


@pytest.mark.parametrize('rule', ('sc', 'raw', 'floor'))
def test_node_holding_band_matches_main(rule):
    # Main's floor and default readers have different trailing-node refusal bands.
    from rfx.geometry.conformal_cells import compute_conformal_fractions

    grid, _ = cell_table_scene('sphere')
    kwargs = dict(conformal_pec='cut_cell_floor' if rule == 'floor' else None,
                  clamp=rule != 'raw')
    accepted, refused = (.014, .0155) if rule == 'floor' else (.0155, .014)
    compute_conformal_fractions(grid, [Sphere((accepted, .006, .006), .0007)], **kwargs, field_dtype=np.float32)
    with pytest.raises(NotImplementedError, match='holds no grid node'):
        compute_conformal_fractions(grid, [Sphere((refused, .006, .006), .0007)], **kwargs, field_dtype=np.float32)
