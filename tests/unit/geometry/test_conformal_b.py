"""Option B: hand-computed incidence/dt and the frozen dd2cead78 oracle."""

from pathlib import Path
import subprocess

import numpy as np
import pytest

import rfx
from rfx import Simulation, Sphere
from rfx.boundaries.spec import BoundarySpec
from rfx.geometry.conformal_cells import compute_conformal_fractions, conformal_b_floor
from rfx.grid import Grid
from rfx.stepping.step_count import steps_for_periods


@pytest.fixture(autouse=True)
def local_worktree_and_clean_gate(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    assert Path(rfx.__file__).resolve().is_relative_to(root)
    monkeypatch.delenv('RFX_CONFORMAL_B_Q', raising=False)
    monkeypatch.delenv('RFX_CONFORMAL_B_FLOOR', raising=False)
    monkeypatch.delenv('RFX_CONFORMAL_B_FLOOR_FRAC', raising=False)


@pytest.mark.parametrize('normal', range(3))
@pytest.mark.parametrize('largest_edge', range(4))
def test_one_face_floor(normal, largest_edge):
    lengths = [np.zeros((3, 3, 3)) for _ in range(3)]
    areas = [np.ones((3, 3, 3)) for _ in range(3)]
    u, v = [axis for axis in range(3) if axis != normal]
    low = (1, 1, 1)
    high_v, high_u = list(low), list(low)
    high_v[v] += 1
    high_u[u] += 1
    edges = [(u, low), (u, tuple(high_v)), (v, low), (v, tuple(high_u))]
    values = [.3, .2, .5]
    values.insert(largest_edge, 1.)
    for (axis, index), value in zip(edges, values):
        lengths[axis][index] = value
    # A normal edge is not on this face.
    lengths[normal][low] = 1.
    areas[normal][low] = .1
    areas[normal][0, 0, 0] = 0.
    expected = [a.copy() for a in areas]
    expected[normal][low] = .25  # max(.1, .5**2 * max(1, .3, .2, .5))
    before = [a.copy() for a in lengths + areas]
    actual = conformal_b_floor(lengths, areas, .5)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)
    for got, want in zip(lengths + areas, before):
        np.testing.assert_array_equal(got, want)


def test_high_boundary_face_has_its_high_edges(monkeypatch):
    from rfx import Box
    grid = Grid(12e9, (.003,) * 3, dx=.001, cpml_layers=0)
    wall = Box((-1.,) * 3, (.0039, 1., 1.))
    monkeypatch.setenv('RFX_CONFORMAL_B_Q', '.5')
    lengths, areas = compute_conformal_fractions(grid, [wall])
    # H_y at x=.003 has area .1. Its high-x E_z edge at .004 is fully
    # open, beyond the last returned x node: the four-edge maximum is 1.
    assert areas[1].shape == grid.shape
    assert float(lengths[0][3, 1, 1]) == pytest.approx(.1)
    assert float(areas[1][3, 1, 1]) == .25


@pytest.mark.parametrize('mode,ndim', [('3d', 3), ('2d_tmz', 2), ('2d_tez', 2)])
@pytest.mark.parametrize('periodic', ['', 'xy'])
def test_simulation_grid_dt_and_step_counts(monkeypatch, mode, ndim, periodic):
    boundary = BoundarySpec(x='periodic', y='periodic', z='pec') if periodic else 'pec'
    sim = Simulation(12e9, (.03,) * 3, dx=.00075, boundary=boundary, mode=mode)
    ordinary = .00075 / (299_792_458.0 * np.sqrt(float(ndim))) * .99
    assert sim._build_grid().dt == ordinary
    assert Grid.courant_dt(.00075, ndim) == ordinary
    monkeypatch.setenv('RFX_CONFORMAL_B_Q', '.5')
    grid = sim._build_grid()
    assert grid.dt == ordinary / 2
    assert Grid.courant_dt(.00075, ndim) == ordinary / 2
    expected_steps = int(np.ceil(20 * (1 / 12e9) / (ordinary / 2)))
    assert grid.num_timesteps() == expected_steps
    assert steps_for_periods(grid.dt, sim._freq_max, 20) == expected_steps
    monkeypatch.setenv('RFX_CONFORMAL_B_FLOOR', '0')
    assert sim._build_grid().dt == ordinary / 2
    monkeypatch.delenv('RFX_CONFORMAL_B_Q')
    assert sim._build_grid().dt == ordinary


def test_k3_unset_is_dd2cead78_before_and_after_gate(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    baseline_source = subprocess.check_output(
        ['git', 'show', 'dd2cead786d91258088d9546dce0ea69ff6e68b8:rfx/geometry/conformal_cells.py'],
        cwd=root, text=True)
    baseline = {}
    exec(compile(baseline_source, 'dd2cead78/conformal_cells.py', 'exec'), baseline)
    grid = Simulation(12e9, (.03,) * 3, dx=.00075, boundary='pec')._build_grid()
    shapes = [Sphere((.01512, .01493, .01507), .009)]
    expected = baseline['compute_conformal_fractions'](grid, shapes)
    raw = baseline['compute_conformal_fractions'](grid, shapes, clamp=False)

    def same_bits(actual, want):
        for got_group, want_group in zip(actual, want):
            for got, ref in zip(got_group, want_group):
                assert np.asarray(got).dtype == np.asarray(ref).dtype
                assert np.asarray(got).shape == np.asarray(ref).shape
                assert np.asarray(got).tobytes() == np.asarray(ref).tobytes()

    same_bits(compute_conformal_fractions(grid, shapes), expected)
    monkeypatch.setenv('RFX_CONFORMAL_B_Q', '.5')
    floored = compute_conformal_fractions(grid, shapes)
    same_bits((floored[0],), (raw[0],))
    assert any(np.any(np.asarray(a) > np.asarray(b)) for a, b in zip(floored[1], raw[1]))
    monkeypatch.setenv('RFX_CONFORMAL_B_FLOOR', '0')
    same_bits(compute_conformal_fractions(grid, shapes), raw)
    monkeypatch.delenv('RFX_CONFORMAL_B_Q')
    same_bits(compute_conformal_fractions(grid, shapes), expected)
    monkeypatch.delenv('RFX_CONFORMAL_B_FLOOR')
    same_bits(compute_conformal_fractions(grid, shapes), expected)


@pytest.mark.parametrize('value', ['', 'bad', '0', '1', '-.5', 'nan', 'inf'])
def test_invalid_q_refuses(monkeypatch, value):
    monkeypatch.setenv('RFX_CONFORMAL_B_Q', value)
    with pytest.raises(ValueError, match='RFX_CONFORMAL_B_Q'):
        Grid.courant_dt(.00075)


@pytest.mark.parametrize('value', ['', 'bad', '0', '-.5', '1.01', 'nan', 'inf', '-inf'])
def test_invalid_floor_fraction_refuses(monkeypatch, value):
    monkeypatch.setenv('RFX_CONFORMAL_B_FLOOR_FRAC', value)
    grid = Grid(12e9, (.003,) * 3, dx=.001, cpml_layers=0)
    with pytest.raises(ValueError, match='RFX_CONFORMAL_B_FLOOR_FRAC'):
        compute_conformal_fractions(grid, [])


@pytest.mark.parametrize('fraction,expected', [('0.0625', .1), ('0.25', .25), ('1', 1.)])
def test_floor_fraction_override_and_gates(monkeypatch, fraction, expected):
    from rfx import Box
    grid = Grid(12e9, (.003,) * 3, dx=.001, cpml_layers=0)
    wall = Box((-1.,) * 3, (.0039, 1., 1.))
    ordinary = compute_conformal_fractions(grid, [wall])
    monkeypatch.setenv('RFX_CONFORMAL_B_FLOOR_FRAC', fraction)
    inactive = compute_conformal_fractions(grid, [wall])
    for actual_group, expected_group in zip(inactive, ordinary):
        for actual, want in zip(actual_group, expected_group):
            np.testing.assert_array_equal(actual, want)
    monkeypatch.setenv('RFX_CONFORMAL_B_Q', '.25')
    _, areas = compute_conformal_fractions(grid, [wall])
    assert float(areas[1][3, 1, 1]) == pytest.approx(expected)
    monkeypatch.setenv('RFX_CONFORMAL_B_FLOOR', '0')
    _, raw = compute_conformal_fractions(grid, [wall])
    assert float(raw[1][3, 1, 1]) == pytest.approx(.1)
