"""Periodic readers agree with whole-cell translations and wide image sums."""

import numpy as np
import pytest

from rfx import Box, Sphere
from rfx._internal import _periodic
from rfx.geometry.box_content import _images
from rfx.geometry.cell_table import build_cell_table
from rfx.geometry.conformal_cells import compute_conformal_fractions
from rfx.grid import Grid
from rfx.nonuniform import make_nonuniform_grid
from tests._x64_compat import enable_x64


def _uniform():
    return Grid(20e9, (.012,) * 3, dx=.001, cpml_layers=0, periodic_axes='x')


def _graded(cells):
    return make_nonuniform_grid(
        (.012, .012), cells, .001, cpml_layers=0,
        dx_profile=np.full(12, .001), dy_profile=np.full(12, .001),
        periodic_axes='z', domain=(.012, .012, float(np.sum(cells))))


@pytest.mark.parametrize('clamp', [False, True], ids=['raw', 'SC'])
def test_uniform_whole_cell_translation(clamp):
    grid = _uniform()
    seam, interior = [compute_conformal_fractions(
        grid, [Sphere((cx, .006, .006), .0008)], clamp=clamp, field_dtype=np.float32)
        for cx in (.0105, .0105 - 3 * grid.dx)]
    # Keep the record's decimal dimensions. Device operands are float32;
    # the translated operands agree bitwise despite host-coordinate roundoff.
    for original, translated in zip(seam, interior):
        for a, b in zip(original, translated):
            np.testing.assert_array_equal(a, np.roll(b, 3, axis=0))


def _chord_free(nodes, cells, centre, radius):
    """Independent 1-D chord overlap, including both adjacent images."""
    period = np.sum(cells)
    inside = np.zeros_like(nodes)
    for k in (-1, 0, 1):
        lo, hi = centre - radius + k * period, centre + radius + k * period
        inside += np.maximum(0., np.minimum(nodes + cells, hi) - np.maximum(nodes, lo))
    return 1. - inside / cells


def test_uniform_seam_edge_closed_form():
    grid = _uniform()
    table = build_cell_table(grid, [Sphere((.0105, .006, .006), .0008)])
    row = table.lengths[0][1:grid.nx + 1, 7, 7]
    cells = np.full(grid.nx, grid.dx)
    expected = _chord_free(np.arange(grid.nx) * grid.dx, cells, .0105, .0008)
    # Check before the operand's deliberate float32 cast (0.70 is not dyadic).
    np.testing.assert_allclose(row, expected, rtol=0, atol=1e-12)
    assert row[-1] == pytest.approx(.70, rel=0, abs=1e-12)


@pytest.mark.parametrize('centre', [.0101, .0091], ids=['S2', 'beyond-first-cell'])
def test_graded_translation_and_seam_distances(centre):
    cells = .001 * np.array([1., 2., 3., 2., 1.5, 2.5])
    grid = _graded(cells)
    radius = .0012
    seam = build_cell_table(grid, [Sphere((.006, .006, centre), radius)])
    # A graded mesh is invariant only when its cell widths move with the body.
    translated_grid = _graded(np.roll(cells, -3))
    interior = build_cell_table(translated_grid, [
        Sphere((.006, .006, centre - np.sum(cells[:3])), radius)])
    physical = tuple(slice(1, n + 1) for n in grid.shape)
    for field in ('lengths', 'areas'):
        for a, b in zip(getattr(seam, field), getattr(interior, field)):
            np.testing.assert_allclose(a[physical], np.roll(b[physical], 3, axis=2),
                                       rtol=0, atol=1e-12)
    # The last edge [9.5, 12] mm straddles the sphere centre at 10.1 mm.
    # Linear node-SDF interpolation is not the chord overlap on that edge:
    # its endpoint distances are -0.6 and +0.7 mm, giving 0.7 / 1.3.
    # This pins the linear endpoint-SDF limitation: 0.538..., versus the
    # exact chord's 0.28, with the nearest seam image.
    expected = [1., 1., 1., 1., .6, .7 / 1.3]
    if centre == .0091:
        # A 1.7 mm seam gap exceeds the first cell (1 mm). Here the seam
        # edge lies on one side of the centre, so chord overlap is exact.
        nodes = np.r_[0., np.cumsum(cells)[:-1]]
        expected = _chord_free(nodes, cells, centre, radius)
    np.testing.assert_allclose(seam.lengths[2][7, 7, 1:grid.nz + 1], expected,
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_box_content_enumerates_the_sdf_images(graded):
    grid = _graded(.001 * np.array([1., 2., 3., 2., 1.5, 2.5])) if graded else _uniform()
    axis = 2 if graded else 0
    lo, hi = [.004] * 3, [.008] * 3
    lo[axis], hi[axis] = (.007, .0089) if graded else (.0015, .003)
    box = Box(tuple(lo), tuple(hi))
    coords = tuple(np.zeros(1) for _ in range(3))
    visited = []

    def sample(x, y, z, shape, *, xp):
        visited.append(tuple(float(line[0]) for line in (x, y, z)))
        return xp.zeros(1)

    _periodic.periodic_sdf(grid, box, coords, sample, xp=np)
    expected = [(tuple(lo[a] - shift[a] for a in range(3)),
                 tuple(hi[a] - shift[a] for a in range(3))) for shift in visited]
    actual = _images(grid, box, (lo, hi))
    assert set(expected) < set(actual)
    # Independent integer shifts: SDF reaches only the base Box here;
    # the wider R+h reach also reads the image across the nearest seam.
    assert len(expected) == 1
    shift = [0.] * 3
    shift[axis] = -grid.domain[axis] if graded else grid.domain[axis]
    extra = (tuple(lo[a] + shift[a] for a in range(3)),
             tuple(hi[a] + shift[a] for a in range(3)))
    assert set(actual) == set(expected) | {extra}


@pytest.mark.parametrize('centre', [.0105, .0015], ids=['high', 'low'])
def test_node_tensor_change_is_confined_to_seam(monkeypatch, centre):
    from rfx.geometry.node_tensor import compute_node_tensor_eps

    grid = _uniform()
    shapes = [(Sphere((centre, .006, .006), .0008), 4.)]
    actual = compute_node_tensor_eps(grid, shapes)

    def old_ranges(grid, shape):
        return [_periodic._fill_image_range(
            grid, shape, a, padding=.5 * float(grid.cells(a)[0]))
            if 'xyz'[a] in grid.periodic_axes else (0,) for a in range(3)]

    monkeypatch.setattr(_periodic, '_sdf_image_ranges', old_ranges)
    previous = compute_node_tensor_eps(grid, shapes)
    # One-cell edge supports can touch either side of the seam. Interior
    # diagonal operands, node tensors and all eight triplets stay bitwise equal.
    for a, b in zip(actual, previous):
        np.testing.assert_array_equal(np.asarray(a)[1:-1], np.asarray(b)[1:-1])
    for field in ('xi', 'bulk', 'active', 'cross'):
        for a, b in zip(getattr(actual.nodes, field), getattr(previous.nodes, field)):
            np.testing.assert_array_equal(np.asarray(a)[1:-1], np.asarray(b)[1:-1])
    np.testing.assert_array_equal(np.asarray(actual.nodes.triplets)[:, 1:-1],
                                  np.asarray(previous.nodes.triplets)[:, 1:-1])


def _wide_images(monkeypatch):
    """Independent reference: include two whole periods beyond every bound."""
    original = _periodic._fill_image_range

    def wide(grid, shape, axis, **kwargs):
        kwargs['padding'] = 2 * float(grid.domain[axis])
        return original(grid, shape, axis, **kwargs)

    monkeypatch.setattr(_periodic, '_fill_image_range', wide)


def _assert_node_operands(actual, expected, *, roll=0, axis=0):
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, np.roll(b, roll, axis=axis), rtol=0, atol=1e-12)
    for field in ('xi', 'cross', 'cross_reverse'):
        values = getattr(actual.nodes, field)
        if values is not None:
            for a, b in zip(values, getattr(expected.nodes, field)):
                spatial_axis = axis + np.ndim(b) - 3
                np.testing.assert_allclose(a, np.roll(b, roll, axis=spatial_axis),
                                           rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual.nodes.triplets,
                               np.roll(expected.nodes.triplets, roll, axis=axis + 1),
                               rtol=0, atol=1e-12)


def test_box_blocker_reach_and_whole_cell_translation(monkeypatch):
    from rfx.geometry.node_tensor import compute_node_tensor_eps

    grid = _uniform()

    def operands(offset):
        return compute_node_tensor_eps(grid, [
            (Box((.0003 + offset, .004, .004), (.003 + offset, .008, .008)), 2.),
            (Sphere((.0097 + offset, .006, .006), .0008), 4.)], field_dtype=np.float64)

    with enable_x64(True):
        actual, translated = operands(0.), operands(.001)
        with monkeypatch.context() as reference:
            _wide_images(reference)
            expected = operands(0.)
        _assert_node_operands(actual, expected)
        _assert_node_operands(translated, actual, roll=1)


@pytest.mark.parametrize('radius', [.0003, .00002], ids=['r0.3mm', 'r0.02mm'])
@pytest.mark.parametrize('reader', ['node', 'kottke'])
def test_anisotropic_normal_reach(monkeypatch, radius, reader):
    from rfx.geometry.node_tensor import compute_node_tensor_eps
    from rfx.geometry.smoothing import compute_smoothed_eps_nonuniform

    dz = .0002
    grid = _graded(np.full(12, dz))
    # Offset from the coarse x nodes so even the tiny sphere's seam samples
    # have a normal with a coarse-axis component, rather than a pure z normal.
    sphere = Sphere((.0062, .006, grid.domain[2] - 1.05 * dz - radius), radius)

    def operands():
        if reader == 'node':
            return compute_node_tensor_eps(grid, [(sphere, 4.)], field_dtype=np.float64)
        return compute_smoothed_eps_nonuniform(grid, [(sphere, 4.)])

    with enable_x64(True):
        actual = operands()
        _wide_images(monkeypatch)
        expected = operands()
        if reader == 'node':
            _assert_node_operands(actual, expected)
        else:
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
