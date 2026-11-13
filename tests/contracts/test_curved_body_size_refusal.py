"""A material Sphere or Cylinder is admitted by its size, never by its position.

A Sphere of radius r is refused when r is below half the diagonal of the primal
cell that contains its centre; a Cylinder when its radius is below half the
diagonal of that cell's cross-section. Expected outcomes come from the drawing
and the test's own node lines, without raster or grid helpers. The positions
are seeded draws inside one cell; on the graded mesh the straddle rows cross a
width step, so each position is judged on its own containing cell. An admitted
body owns at least one cell at every position.

The preflight warning names an admitted dielectric Sphere, or a Cylinder across
its radius, whose diameter is under three containing-cell widths along an axis
across it. A traced radius is compared through the shared traced check.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Cylinder, Simulation, Sphere
from rfx.errors import RfxRefusal
from rfx.geometry.rasterize_grid import (
    _material_cell_mask, cell_sizes_from_nonuniform_grid, centres_from_nonuniform_grid,
    centres_from_uniform_grid, coords_from_nonuniform_grid, coords_from_uniform_grid,
)
from rfx.sources.sources import GaussianPulse

H = 1e-3
GRADED_Z = np.r_[np.full(6, 1.0e-3), np.full(6, 0.8e-3)]
REFUSED = 'is smaller than one cell can hold at every position'
DRAWS = np.random.default_rng(2624).random((64, 3))
STRADDLE_Z = 5e-3 + 2e-3 * np.random.default_rng(2624).random(64)


def _sim(graded=False, freq_max=30e9):
    domain = (12 * H, 12 * H, float(GRADED_Z.sum()) if graded else 12 * H)
    return Simulation(freq_max, domain, dx=H, cpml_layers=0, boundary='pec',
                      **(dict(dz_profile=GRADED_Z) if graded else {}))


def _sampling(graded):
    grid = _sim(graded)._build_nonuniform_grid() if graded else _sim()._build_grid()
    if graded:
        coords = coords_from_nonuniform_grid(grid)
        return dict(coords=coords, centres=centres_from_nonuniform_grid(grid, coords),
                    cell_sizes=cell_sizes_from_nonuniform_grid(grid))
    return dict(coords=coords_from_uniform_grid(grid), centres=centres_from_uniform_grid(grid), grid=grid)


def _widths_at(point, graded):
    """Widths of the drawing's cell holding ``point``, from the test's node lines."""
    z_nodes = np.r_[0., np.cumsum(GRADED_Z)] if graded else np.arange(13) * H
    k = int(np.searchsorted(z_nodes, point[2], side='right')) - 1
    return (H, H, float(np.diff(z_nodes)[k]) if graded else H)


def _threshold(kind, axis, widths):
    across = (0, 1, 2) if kind == 'sphere' else {'z': (0, 1), 'x': (1, 2)}[axis]
    return 0.5 * np.sqrt(sum(widths[a] ** 2 for a in across))


def _body(kind, axis, centre, radius):
    return Sphere(centre, radius) if kind == 'sphere' else Cylinder(centre, radius, 4 * H, axis)


def _outcome(shape, sampling):
    try:
        mask = _material_cell_mask(shape, sampling['coords'], sampling['centres'], name='body',
                                   grid=sampling.get('grid'), cell_sizes=sampling.get('cell_sizes'))
    except RfxRefusal as caught:
        assert REFUSED in str(caught)
        return 'refused'
    assert np.count_nonzero(np.asarray(mask)) >= 1
    return 'admitted'


def _cell_positions(kind, axis, lo, widths):
    """64 seeded centres in the cell with low corner ``lo``; a Cylinder's axial centre stays at the cell centre."""
    points = np.asarray(lo) + DRAWS * np.asarray(widths)
    if kind == 'cylinder':
        a = 'xyz'.index(axis)
        points[:, a] = lo[a] + widths[a] / 2
    return points


CASES = ([('sphere', None, k) for k in (0.577, 0.993, 1.005, 1.155)]
         + [('cylinder', axis, k) for axis in 'zx' for k in (0.495, 0.990, 1.004, 1.414)])


@pytest.mark.parametrize('kind, axis, k', CASES)
@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded-0.8mm-cell'])
def test_outcome_is_the_same_at_every_position_in_one_cell(graded, kind, axis, k):
    lo = (5 * H, 5 * H, 7.6e-3 if graded else 5 * H)
    widths = (H, H, 0.8e-3 if graded else H)
    radius = k * _threshold(kind, axis, widths)
    sampling = _sampling(graded)
    outcomes = {_outcome(_body(kind, axis, tuple(p), radius), sampling)
                for p in _cell_positions(kind, axis, lo, widths)}
    assert outcomes == {'refused' if k < 1 else 'admitted'}


@pytest.mark.parametrize('kind, axis, radius', [('sphere', None, 0.84e-3), ('cylinder', 'x', 0.68e-3)])
def test_straddle_rows_follow_the_containing_cell(kind, axis, radius):
    sampling = _sampling(True)
    seen = set()
    for z in STRADDLE_Z:
        centre = (5.5e-3 if kind == 'cylinder' else 5.81e-3, 5.77e-3, float(z))
        expected = ('refused' if radius < _threshold(kind, axis, _widths_at(centre, True))
                    else 'admitted')
        assert _outcome(_body(kind, axis, centre, radius), sampling) == expected
        seen.add(expected)
    assert seen == {'refused', 'admitted'}


def _small_body_scene(shape, smoothing=False):
    sim = _sim()
    sim.add_material('d', eps_r=4.)
    sim.add(shape, material='d')
    sim.add_source((2.5 * H,) * 3, 'ez', waveform=GaussianPulse(f0=20e9, bandwidth=0.8))
    return sim


OFFSET = np.array([0.31, 0.27, 0.19]) * H
CELL = np.full(3, 5.5 * H)
WARN_ROWS = [
    ('sphere-0.87h', Sphere(tuple(CELL + OFFSET), 0.87 * H), 'warn'),
    ('sphere-1.0h', Sphere(tuple(CELL + OFFSET), 1.0 * H), 'warn'),
    ('sphere-1.4h', Sphere(tuple(CELL + OFFSET), 1.4 * H), 'warn'),
    ('sphere-1.6h', Sphere(tuple(CELL + OFFSET), 1.6 * H), 'quiet'),
    ('cylinder-0.71h', Cylinder(tuple(CELL + OFFSET), 0.71 * H, 4 * H, 'z'), 'warn'),
    ('disc', Cylinder(tuple(CELL + np.array([0.31, 0.27, 0.05]) * H), 4 * H, 0.3 * H, 'z'), 'quiet'),
    ('box-1.5h', Box(tuple(CELL + OFFSET - 0.75 * H), tuple(CELL + OFFSET + 0.75 * H)), 'quiet'),
    ('sphere-0.5h', Sphere(tuple(CELL + OFFSET), 0.5 * H), 'refused'),
]


@pytest.mark.parametrize('label, shape, expected', WARN_ROWS, ids=[row[0] for row in WARN_ROWS])
def test_small_dielectric_body_warning(label, shape, expected):
    sim = _small_body_scene(shape)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        report = sim.preflight()
    warned = any(str(issue).startswith('Small dielectric body:') for issue in report)
    assert warned == (expected == 'warn')
    if expected == 'refused':
        with pytest.raises(RfxRefusal, match=REFUSED):
            sim.run(n_steps=1, skip_preflight=True)
    else:
        sim.run(n_steps=1, skip_preflight=True)


def traced_sphere_objective(radius):
    sim = _sim()
    sim.add_material('d', eps_r=4.)
    sim.add(Sphere(tuple(CELL + OFFSET), radius), material='d')
    sim.add_source((2.5 * H, 6.5 * H, 6.5 * H), 'ez', waveform=GaussianPulse(f0=20e9, bandwidth=0.8))
    sim.add_probe((9.5 * H, 6.5 * H, 7.5 * H), 'ez')
    result = sim.forward(n_steps=200, subpixel_smoothing=True, skip_preflight=True)
    return jnp.sum(jnp.asarray(result.time_series) ** 2)


def test_traced_radius_below_the_cell_is_refused():
    with pytest.raises(Exception, match=REFUSED):
        jax.block_until_ready(jax.jit(traced_sphere_objective)(jnp.float32(0.80 * H)))
