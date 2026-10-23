"""Period candidates and wall envelopes, checked without the production search."""
import math

import pytest

from rfx import Simulation
from rfx.grid import C0, Grid
from rfx.runners.nonuniform import build_nonuniform_grid


def largest_period_fit(period, walls, requested):
    """Enumerate period/n independently, including the explicit-advice range."""
    candidates = (period / n for n in range(math.ceil(period / requested),
                  math.floor(period / (requested / 16)) + 1))
    expected = next(dx for dx in candidates if all(
        abs(math.floor(wall / dx + .5) * dx / wall - 1) <= .01 for wall in walls))
    for wall in walls:
        assert abs(math.floor(wall / expected + .5) * expected / wall - 1) <= .01
    return expected


def build(builder, domain, requested, *, wall_axes, cpml_layers):
    frequency = C0 / (20 * requested)
    faces = {f'{axis}_{side}' for axis in wall_axes for side in ('lo', 'hi')}
    if builder == 'simulation':
        return Simulation(frequency, domain, cpml_layers=cpml_layers,
                          boundary=dict(x='periodic', y='pec', z='cpml'))._build_grid()
    kwargs = dict(periodic_axes='x', pec_faces=faces, cpml_axes='z')
    if builder == 'uniform':
        return Grid(frequency, domain, cpml_layers=cpml_layers, **kwargs)
    return build_nonuniform_grid(frequency, domain, None, cpml_layers, None, **kwargs)


def wall_span(grid, axis):
    if hasattr(grid, 'dx_arr_f64'):
        widths = getattr(grid, {'y': 'dy_arr_f64', 'z': 'dz_f64'}[axis])
        return sum(widths[:-1])
    return (getattr(grid, f'n{axis}') - 1) * grid.dx


@pytest.mark.parametrize('builder', ['uniform', 'nonuniform', 'simulation'])
def test_period_anchor_finds_reviewers_spacing(builder):
    requested, period, wall = .00105, .010, .00995
    expected = largest_period_fit(period, (wall,), requested)
    assert expected == .001
    with pytest.warns(UserWarning, match='snapped'):
        grid = build(builder, (period, wall, .0063), requested, wall_axes='y', cpml_layers=2)
    spacing = float(grid.cells('x')[0])
    assert spacing == pytest.approx(expected, rel=2e-14, abs=0)
    assert .75 * requested <= spacing <= requested
    assert grid.shape[0] == 10 and grid.shape[1] - 1 == 10
    assert wall_span(grid, 'y') == pytest.approx(10 * expected)
    assert abs(wall_span(grid, 'y') / wall - 1) <= .01


@pytest.mark.parametrize('builder', ['uniform', 'nonuniform'])
def test_period_anchor_preserves_wall_envelope(builder):
    period, walls, requested = .0103, (.008, .006), .0005
    expected = largest_period_fit(period, walls, requested)
    with pytest.warns(UserWarning, match='snapped'):
        grid = build(builder, (period, *walls), requested, wall_axes='yz', cpml_layers=0)
    for axis, wall in zip('yz', walls):
        assert abs(wall_span(grid, axis) / wall - 1) <= .01
    assert float(grid.cells('x')[0]) == pytest.approx(expected, rel=2e-14, abs=0)
    assert grid.shape == (26, 21, 16)
    assert .75 * requested <= expected <= requested
