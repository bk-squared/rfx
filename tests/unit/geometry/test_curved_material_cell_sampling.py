"""Curved cell materials occupy the declared centre, including at E edges.

Expected occupancy uses analytic geometry and independently built coordinates;
no expected value calls the product's shape mask or centre provider (#1138).
"""
from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Cylinder, Simulation, Sphere
from rfx.core.yee import edge_averaged_materials
from rfx.geometry.csg import rasterize
from rfx.geometry.rasterize_grid import (
    GridCoords, coords_from_fine_grid, rasterize_geometry,
)
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole
from rfx.nonuniform import make_nonuniform_grid
from tests._x64_compat import enable_x64


KINDS = ['sphere', 'cylinder_x', 'cylinder_y', 'cylinder_z']


def _shape(kind, centre=(.015, .015, .015)):
    if kind == 'sphere':
        return Sphere(centre, .0063)
    return Cylinder(centre, .0052, .012, kind[-1])


def _inside(shape, axes):
    offsets = np.meshgrid(*(x - c for x, c in zip(axes, shape.center)), indexing='ij')
    if isinstance(shape, Sphere):
        return sum(x*x for x in offsets) <= shape.radius**2
    ax = 'xyz'.index(shape.axis)
    return (sum(x*x for a, x in enumerate(offsets) if a != ax) <= shape.radius**2
            ) & (np.abs(offsets[ax]) <= shape.height/2)


def _sim(shape, *, poles=False):
    sim = Simulation(freq_max=3e9, domain=(.03,)*3, dx=.001, cpml_layers=4)
    spec = dict(eps_r=4., sigma=32., mu_r=1.5, chi3=2e-4)
    if poles:
        spec.update(debye_poles=[DebyePole(delta_eps=1., tau=1e-10)],
                    lorentz_poles=[LorentzPole(omega_0=1e10, delta=1e8, kappa=2e20)])
    sim.add_material('dut', **spec)
    sim.add(shape, material='dut')
    return sim


def _centroid(values, axes):
    w = np.asarray(values, dtype=np.float64)
    return np.array([w.sum(axis=tuple(k for k in range(3) if k != a)) @ x / w.sum()
                     for a, x in enumerate(axes)])


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('precision', ['default', 'x64'])
def test_uniform_curved_cells_and_each_yee_edge_are_centred(kind, precision):
    with (enable_x64() if precision == 'x64' else nullcontext()):
        shape = _shape(kind)
        sim = _sim(shape, poles=True)
        grid = sim._build_grid()
        nodes = [(np.arange(n)-p)*grid.dx for n, p in zip(grid.shape, grid.axis_pads)]
        centres = [(np.arange(n)-p+.5)*grid.dx for n, p in zip(grid.shape, grid.axis_pads)]
        expected = _inside(shape, centres)
        eps, sigma = rasterize(grid, [(shape, 4., 32.)])
        np.testing.assert_array_equal(eps, np.where(expected, 4., 1.))
        np.testing.assert_array_equal(sigma, np.where(expected, 32., 0.))
        # Shape.mask remains its existing node API; only material sampling moves.
        np.testing.assert_array_equal(shape.mask(grid), _inside(shape, nodes))
        mats, debye, lorentz, pec, _, _, chi3 = sim._assemble_materials(grid)
        np.testing.assert_array_equal(mats.eps_r, eps)
        np.testing.assert_array_equal(mats.sigma, sigma)
        np.testing.assert_array_equal(mats.mu_r, np.where(expected, 1.5, 1.))
        for spec in (debye, lorentz):
            np.testing.assert_array_equal(spec[1][0], expected)
        np.testing.assert_array_equal(np.asarray(chi3) != 0, expected)
        assert pec is None
        np.testing.assert_allclose(_centroid(sigma, centres), shape.center,
                                   rtol=0, atol=grid.dx*1e-10)
        _, sig_edges = edge_averaged_materials(eps, sigma)
        for component, sig in enumerate(sig_edges):
            axes = [centres[a] if a == component else nodes[a] for a in range(3)]
            np.testing.assert_allclose(_centroid(sig, axes), shape.center,
                                       rtol=0, atol=grid.dx*1e-10)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('graded', [False, True])
def test_nonuniform_assembly_uses_each_axis_local_cell_centre(kind, graded):
    profiles = [np.full(n, d) for n, d in ((30, .001), (25, .0012), (40, .00075))]
    if graded:
        profiles = [p*(1+.2*np.sin(np.linspace(0, 2*np.pi, len(p)))) for p in profiles]
    grid = make_nonuniform_grid((.03, .03), profiles[2], .001, cpml_layers=4,
                               dx_profile=profiles[0], dy_profile=profiles[1])
    # The NU spine has four pad cells at either end plus one bounding node.
    axes = []
    for p in profiles:
        widths = np.pad(p, (4, 5), mode='edge')
        if graded:
            nodes = np.r_[0., np.cumsum(widths[:-1])] - 4*p[0]
            axes.append(nodes + widths/2)
        else:
            axes.append((np.arange(len(widths))-4+.5)*p[0])
    shape = _shape(kind)
    sim = _sim(shape, poles=True)
    mats, debye, lorentz, pec = sim._assemble_materials_nu(grid)
    expected = _inside(shape, axes)
    np.testing.assert_array_equal(mats.eps_r, np.where(expected, 4., 1.))
    np.testing.assert_array_equal(mats.sigma, np.where(expected, 32., 0.))
    for spec in (debye, lorentz):
        np.testing.assert_array_equal(spec[1][0], expected)
    assert pec is None


@pytest.mark.parametrize('kind', KINDS)
def test_fine_subgrid_centres_are_not_shifted_twice(kind):
    shape = _shape(kind)
    sim = _sim(shape, poles=True)
    offsets, sizes, dx = (.002, .003, .004), (28, 27, 26), .001
    coords = coords_from_fine_grid(*sizes, dx, *offsets)
    got = rasterize_geometry(sim._geometry, sim._resolve_material, coords, centres=coords)
    axes = [off+(np.arange(n)+.5)*dx for off, n in zip(offsets, sizes)]
    expected = _inside(shape, axes)
    np.testing.assert_array_equal(got[0].sigma, np.where(expected, 32., 0.))
    np.testing.assert_array_equal(np.asarray(got[-1]) != 0, expected)


@pytest.mark.parametrize('kind', KINDS)
def test_traced_coordinate_sampling_matches_analytic_cells(kind):
    sim = _sim(_shape(kind))
    nodes = np.arange(31, dtype=np.float32)*np.float32(.001)

    @jax.jit
    def assembled(scale):
        xyz = tuple(jnp.asarray(nodes)*scale[a] for a in range(3))
        coords = GridCoords(*xyz, (31,)*3)
        return rasterize_geometry(sim._geometry, sim._resolve_material, coords)[0].sigma

    scale = np.array([1.03, .97, 1.01], dtype=np.float32)
    actual_nodes = [nodes*s for s in scale]
    axes = [x + np.r_[np.diff(x), np.diff(x)[-1]]/2 for x in actual_nodes]
    expected = _inside(_shape(kind), axes)
    np.testing.assert_array_equal(assembled(scale), np.where(expected, 32., 0.))


def test_legacy_box_and_uniform_custom_shape_masks_are_preserved():
    box = Box((.009,)*3, (.021,)*3)
    sim = _sim(box)
    grid = sim._build_grid()
    nodes = [(np.arange(n)-p)*grid.dx for n, p in zip(grid.shape, grid.axis_pads)]
    inside = [(a >= .009) & (a < .021) for a in nodes]
    expected = inside[0][:, None, None] & inside[1][None, :, None] & inside[2][None, None, :]
    eps, sigma = rasterize(grid, [(box, 4., 32.)])
    np.testing.assert_array_equal(sigma, np.where(expected, 32., 0.))
    np.testing.assert_array_equal(sim._assemble_materials(grid)[0].eps_r, eps)

    class UniformOnlyShape:
        def mask(self, grid):
            return jnp.asarray(expected)

    custom = _sim(UniformOnlyShape())
    np.testing.assert_array_equal(custom._assemble_materials(grid)[0].sigma, sigma)
    np.testing.assert_array_equal(rasterize(grid, [(UniformOnlyShape(), 4., 32.)])[1], sigma)
