"""Host-only identity against pre-table conformal operands and table facts."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from rfx import Simulation, Sphere
from rfx.geometry import cell_table
from rfx.model.conformal_rules import conformal_fractions_from_table
from tests._cell_table_capture import record_cell_table_capture
from tests._cell_table_scenes import SCENES, cell_table_scene


REFERENCE = Path(__file__).resolve().parents[2] / 'data' / 'cell_table_56653420b.npz'
CAPTURE_REFERENCE = REFERENCE.with_name('cell_table_capture_56653420b.npz')


@pytest.mark.parametrize('scene', SCENES)
@pytest.mark.parametrize('rule', ('sc', 'raw', 'floor'))
def test_frozen_capture_calls(scene, rule):
    from rfx.geometry.conformal_cells import compute_conformal_fractions

    grid, shapes = cell_table_scene(scene)
    with record_cell_table_capture(grid, shapes) as calls:
        compute_conformal_fractions(
            grid, pec_shapes=shapes, clamp=rule != 'raw',
            conformal_pec='cut_cell_floor' if rule == 'floor' else None)
    prefix = f'{scene}_{rule}_'
    actual = {f'{prefix}call_count': np.asarray(len(calls))}
    for index, call in enumerate(calls):
        for argument, value in call.items():
            actual[f'{prefix}{index}_{argument}'] = value
    with np.load(CAPTURE_REFERENCE) as frozen:
        assert set(actual) == {key for key in frozen.files if key.startswith(prefix)}
        for key, value in actual.items():
            assert value.dtype == frozen[key].dtype, key
            assert np.array_equal(value, frozen[key]), key


@pytest.mark.parametrize('scene', SCENES)
@pytest.mark.parametrize('rule', ('sc', 'raw', 'floor'))
def test_frozen_operands(scene, rule):
    grid, shapes = cell_table_scene(scene)
    table = cell_table.build_cell_table(grid, shapes, include_trailing_node=rule == "floor")
    actual = conformal_fractions_from_table(
        table, grid, 'cut_cell_floor' if rule == 'floor' else None,
        clamp=rule != 'raw')
    with np.load(REFERENCE) as frozen:
        for kind, values in zip(('edges', 'faces'), actual):
            for axis, value in enumerate(values):
                assert value.shape == grid.shape
                assert value.dtype == np.float32
                assert np.array_equal(value, frozen[f'{scene}_{rule}_{kind}_{axis}'])
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


def test_topology_tracks_shapes_but_not_body_fractions():
    from rfx.grid import Grid

    grid, shapes = cell_table_scene('sphere')
    assert (cell_table.build_cell_table(grid, shapes).topology_key
            == cell_table.build_cell_table(grid, []).topology_key)
    keys = []
    for domain in ((.01, .012, .014), (.014, .012, .01)):
        grid = Grid(20e9, domain, dx=.001, cpml_layers=0)
        keys.append(cell_table.build_cell_table(grid, []).topology_key)
    assert keys[0] != keys[1]


def test_staircase_forms_no_table(monkeypatch):
    from rfx.core.yee import init_materials
    from rfx.model.tensor_operands import form_tensor_operands

    def fail(*args, **kwargs):
        pytest.fail('staircase operand setup built a cell table')

    monkeypatch.setattr(cell_table, 'build_cell_table', fail)
    sim = Simulation(20e9, (.012,) * 3, dx=.001, boundary='pec')
    grid = sim._build_grid()
    result = form_tensor_operands(
        sim, grid, init_materials(grid.shape), subpixel_smoothing=False,
        conformal=False, conformal_min_weight=.1,
        pec_shapes=[Sphere((.0061, .0058, .0057), .0023)])
    assert result[2] is None and result[4] is None


@pytest.mark.parametrize('rule', ('sc', 'raw', 'floor'))
def test_node_holding_band_matches_main(rule):
    # Main's floor and default readers have different trailing-node refusal bands.
    from rfx.geometry.conformal_cells import compute_conformal_fractions

    grid, _ = cell_table_scene('sphere')
    kwargs = dict(conformal_pec='cut_cell_floor' if rule == 'floor' else None,
                  clamp=rule != 'raw')
    accepted, refused = (.014, .0155) if rule == 'floor' else (.0155, .014)
    compute_conformal_fractions(grid, [Sphere((accepted, .006, .006), .0007)], **kwargs)
    with pytest.raises(NotImplementedError, match='holds no grid node'):
        compute_conformal_fractions(grid, [Sphere((refused, .006, .006), .0007)], **kwargs)
