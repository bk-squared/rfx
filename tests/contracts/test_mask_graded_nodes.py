"""No-SDF masks use physical nodes, including smoothing's fallback (#1298)."""
from types import SimpleNamespace

import numpy as np
import pytest

from rfx.api._spec import MaterialSpec
from rfx.geometry.csg import Box, PolylineWire, _grid_coords
from rfx.geometry.curved import CurvedPatch
from rfx.geometry.via import Via
from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid, rasterize_geometry
from rfx.geometry.smoothing import compute_smoothed_eps_nonuniform
from rfx.nonuniform import make_nonuniform_grid


class BoxWithoutSDF:
    def __init__(self, lo, hi):
        self.box = Box(lo, hi)

    def bounding_box(self):
        return self.box.bounding_box()

    def mask_on_coords(self, *coords):
        return self.box.mask_on_coords(*coords)

    def mask(self, grid):
        return self.mask_on_coords(*_grid_coords(grid))


def fixture(kind, axis=2, pad=0):
    # Asymmetric profile, with a constant boundary width on both ends.
    profile = np.array([1., 1., .37, .23, .41, .29, .31, .39, 1.]) * 1e-3
    profiles = [np.full(6, 1e-3) for _ in range(3)]
    profiles[axis] = profile
    grid = make_nonuniform_grid((.006, .006), profiles[2], 1e-3,
                               cpml_layers=pad, dx_profile=profiles[0],
                               dy_profile=profiles[1])
    # Independent oracle: cumulative sum of the INPUT profiles, with padding.
    nodes = []
    for p in profiles:
        widths = np.pad(p, (pad, pad + 1), mode='edge')
        edges = np.r_[0., np.cumsum(widths)]
        nodes.append(edges[:-1] - edges[pad])
    xyz = np.meshgrid(*nodes, indexing='ij')
    lo, hi = np.array([.61, .73, .83])*1e-3, np.array([3.67, 3.53, 3.59])*1e-3
    if kind == 'box':
        shape = BoxWithoutSDF(tuple(lo), tuple(hi))
        boxes = [(lo, hi)]
    elif kind == 'mesh':
        trimesh = pytest.importorskip('trimesh')
        pytest.importorskip('rtree')
        from rfx.geometry.mesh_import import MeshShape
        mesh = trimesh.creation.box(extents=hi-lo)
        mesh.apply_translation((hi+lo)/2)
        shape = MeshShape(mesh)
        boxes = [(lo, hi)]
    elif kind == 'via':
        shape = Via(center=(.00213, .00207), drill_radius=.00111,
                    pad_radius=.00111, layers=[(.00083, .00359)])
        boxes = [b.bounding_box() for b, _ in shape.to_shapes()]
    elif kind == 'curved':
        shape = CurvedPatch(center=(.00213, .00207, .00219),
                            length=.00306, width=.00234, radius=.0067)
        boxes = [b.bounding_box() for b in shape.to_staircase(.001)]
    else:
        shape = PolylineWire(((.00061, .00123, .00137),
                              (.00367, .00291, .00359)), radius=.00069)
        start, end = np.asarray(shape.points)
        delta = end-start
        t = np.clip(sum((xyz[a]-start[a])*delta[a] for a in range(3)) /
                    np.dot(delta, delta), 0, 1)
        expected = sum((xyz[a]-start[a]-t*delta[a])**2 for a in range(3)) <= shape.radius**2
        return grid, shape, expected
    expected = np.zeros(grid.shape, dtype=bool)
    for lower, upper in boxes:
        selected = np.ones(grid.shape, dtype=bool)
        for a in range(3):
            if upper[a] == lower[a]:  # Declared sheet: nearest physical node.
                selected &= xyz[a] == nodes[a][np.argmin(abs(nodes[a]-lower[a]))]
            else:
                selected &= (xyz[a] >= lower[a]) & (xyz[a] < upper[a])
        expected |= selected
    return grid, shape, expected


@pytest.mark.parametrize('kind', ['box', 'mesh', 'curved', 'via', 'wire'])
@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('pad', [0, 2])
@pytest.mark.parametrize('subpixel_smoothing', [False, True])
def test_graded_placement(kind, axis, pad, subpixel_smoothing):
    grid, shape, expected = fixture(kind, axis, pad)
    assert expected.any()
    # Guard the fixture itself: the declared outer extents must not become
    # node/half-node witnesses when the asymmetric profile is edited.
    widths = np.array([1., 1., .37, .23, .41, .29, .31, .39, 1.])*1e-3
    nodes = np.r_[0., np.cumsum(widths)]
    ties = np.r_[nodes, (nodes[:-1] + nodes[1:])/2]
    for corner in shape.bounding_box():
        assert not np.any(np.isclose(corner[axis], ties, rtol=0, atol=1e-12))
    np.testing.assert_array_equal(shape.mask(grid), expected)
    if subpixel_smoothing:
        arrays = compute_smoothed_eps_nonuniform(grid, [(shape, 4.)])
    else:
        materials, *_ = rasterize_geometry(
            [SimpleNamespace(shape=shape, material_name='dielectric')],
            lambda _: MaterialSpec(eps_r=4.), coords_from_nonuniform_grid(grid), grid=grid)
        arrays = [materials.eps_r]
    for eps in arrays:
        np.testing.assert_array_equal(eps, np.where(expected, 4., 1.))


@pytest.mark.parametrize('kind', ['box', 'mesh', 'curved', 'via', 'wire'])
@pytest.mark.parametrize('pad', [0, 2])
def test_uniform_bit_identity(kind, pad):
    from rfx import Simulation
    from rfx.geometry.smoothing import compute_smoothed_eps

    _, shape, _ = fixture(kind)
    grid = Simulation(freq_max=20e9, domain=(.006, .006, .006),
                      dx=.001, cpml_layers=pad)._build_grid()
    # Frozen 292e3903 arithmetic, deliberately independent of the provider.
    old = tuple((np.arange(n, dtype=np.float64)-p)*grid.dx
                for n, p in zip(grid.shape, grid.axis_pads))
    for actual, baseline in zip(_grid_coords(grid), old):
        assert actual.dtype == baseline.dtype
        assert actual.tobytes() == baseline.tobytes()
    expected = np.asarray(shape.mask_on_coords(*old))
    assert np.asarray(shape.mask(grid)).tobytes() == expected.tobytes()
    for eps in compute_smoothed_eps(grid, [(shape, 4.)]):
        baseline = np.where(expected, 4., 1.).astype(np.asarray(eps).dtype)
        assert np.asarray(eps).tobytes() == baseline.tobytes()
