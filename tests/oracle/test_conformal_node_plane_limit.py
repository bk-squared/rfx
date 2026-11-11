"""Hand-counted node-plane edge sets and an exactly opaque one-cell short."""
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.errors import RfxRefusal
from tests.contracts.conformal_scenes import capture_setups, scene


@pytest.mark.parametrize('name', ['Q', 'P'])
def test_node_plane_limit(monkeypatch, name):
    rows = capture_setups(monkeypatch)
    for conformal in (True, False):
        scene(name, conformal=conformal).run(n_steps=4, compute_s_params=False)
    lo, hi, window = (((55, 6, 2), (58, 12, 6), ((54, 60), (5, 14), (1, 8))) if name == 'Q'
                      else ((28, 0, 0), (29, 13, 6), ((27, 31), (0, 13), (0, 6))))
    for component in range(3):
        indices = np.meshgrid(*(np.arange(a, b) for a, b in window), indexing='ij')
        expected = np.ones(indices[0].shape, dtype=bool)
        for axis, index in enumerate(indices):
            expected &= (index >= lo[axis]) & (index < hi[axis] + (axis != component))
        cuts = tuple(slice(a + getattr(rows[0]['grid'], f'pad_{"xyz"[axis]}_lo'),
                           b + getattr(rows[0]['grid'], f'pad_{"xyz"[axis]}_lo'))
                     for axis, (a, b) in enumerate(window))
        np.testing.assert_array_equal(rows[1]['masks'][component][cuts], expected,
                                      err_msg=f'{name}: independent staircase edge set')
        np.testing.assert_array_equal(rows[0]['weights'][component][cuts] == 0, expected,
                                      err_msg=f'{name}: conformal zero edge set')


def test_one_cell_gap_stays_open(monkeypatch):
    rows = capture_setups(monkeypatch)
    scene('G').run(n_steps=4, compute_s_params=False)
    row = rows[0]
    x = row['grid'].pad_x_lo + 56
    np.testing.assert_array_equal(row['weights'][0][x, 6:13, 2:7], np.ones((7, 5)),
                                  err_msg='G: the midpoint of the one-cell gap is vacuum')


def test_off_lattice_box_without_an_inside_node_refuses():
    sim = Simulation(24e9, (.020, .014, .010), dx=.001, boundary='pec', snap='declared')
    sim.add(Box((.018, -.01, -.01), (.0184, .024, .020)), material='pec')
    with pytest.raises(RfxRefusal, match='PEC Box.*thinner than one cell along x'):
        sim.run(n_steps=4, conformal_pec=True, compute_s_params=False, skip_preflight=True)


def test_short_has_exactly_zero_transmitted_energy():
    sim = scene('P')
    for x in (.060, .105):
        sim.add_probe((x, .020, .010), 'ez')
    series = np.asarray(sim.run(n_steps=700, compute_s_params=False).time_series, dtype=np.float64)
    energy = np.sum(series ** 2, axis=0)
    assert energy[0] > 0
    assert energy[1] / energy[0] == 0.0, f'P: behind/front = {energy[1] / energy[0]:.17g}'
