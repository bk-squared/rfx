"""Host-only identity against pre-table conformal operands and table facts."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from rfx import Simulation, Sphere
from rfx.geometry import cell_table
from rfx.model.conformal_rules import conformal_fractions_from_table
from tests._cell_table_scenes import SCENES, cell_table_scene


REFERENCE = Path(__file__).resolve().parents[2] / 'data' / 'cell_table_fa3fcc73f.npz'


@pytest.mark.parametrize('scene', SCENES)
@pytest.mark.parametrize('rule', ('sc', 'raw', 'floor'))
def test_frozen_operands(scene, rule):
    grid, shapes = cell_table_scene(scene)
    table = cell_table.build_cell_table(grid, shapes)
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
def test_node_holding_band_is_the_default_lattice(rule):
    # The trailing halo node the floor slice needs does not decide whether a body holds a node.
    from rfx.grid import Grid

    grid = Grid(20e9, (.012,) * 3, dx=.001, cpml_layers=0)
    grid.dt = .5 * Grid.courant_dt(grid.dx, 3)
    form = lambda shape: conformal_fractions_from_table(
        cell_table.build_cell_table(grid, [shape]), grid,
        'cut_cell_floor' if rule == 'floor' else None, clamp=rule != 'raw')
    form(Sphere((.0155, .006, .006), .0007))  # 1.5 cells past the last node: admitted
    with pytest.raises(NotImplementedError, match='holds no grid node'):
        form(Sphere((.014, .006, .006), .0007))  # holds only the trailing halo node
