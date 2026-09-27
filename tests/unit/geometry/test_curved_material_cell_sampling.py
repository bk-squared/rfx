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


def test_coax_clearance_uses_centred_disk_on_both_sides_of_ground():
    """The warning's old 48-cell snapshot pinned displaced PTFE (#1138)."""
    from tests.unit.geometry.test_fidelity_topology_findings import (
        CLEAR_R, DX, EPS_COAX, JUNCTION_X, N_GND, Y_C, _junction_sim,
    )

    sim = _junction_sim(open_annulus=False)
    grid = sim._build_grid()
    # This oracle reads material cells only; collect and deliberately drop PEC.
    mats = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0]
    x = (np.arange(grid.shape[0]) - grid.pad_x_lo + .5) * DX - JUNCTION_X
    y = (np.arange(grid.shape[1]) - grid.pad_y_lo + .5) * DX - Y_C
    expected = x[:, None]**2 + y[None, :]**2 <= CLEAR_R**2
    # Half-integer disk r=4: 2*(8+8+6+4), derived before updating the snapshot.
    assert int(expected.sum()) == 52
    for k in (grid.pad_z_lo + N_GND - 1, grid.pad_z_lo + N_GND):
        np.testing.assert_array_equal(np.isclose(np.asarray(mats.eps_r)[:, :, k], EPS_COAX),
                                      expected)


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


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('field,values', [('eps_r', (4., 6.)), ('sigma', (32., 64.)),
                                        ('mu_r', (1.5, 2.))])
def test_named_material_sweep_uses_the_assembled_curved_cells(kind, field, values):
    from rfx.vmap_sweep import _build_batched_materials

    shape = _shape(kind)
    sim = _sim(shape)
    grid = sim._build_grid()
    base = sim._assemble_materials(grid)[0]
    batch = _build_batched_materials(sim, grid, base, f'dut.{field}', jnp.array(values))
    centres = [(np.arange(n)-p+.5)*grid.dx for n, p in zip(grid.shape, grid.axis_pads)]
    expected = _inside(shape, centres)
    background = 0. if field == 'sigma' else 1.
    for index, value in enumerate(values):
        np.testing.assert_array_equal(getattr(batch, field)[index],
                                      np.where(expected, value, background))
    for name in ('eps_r', 'sigma', 'mu_r'):
        np.testing.assert_array_equal(getattr(batch, name)[0], getattr(base, name))


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('with_pec', [False, True])
def test_dual_average_uses_supplied_centres_and_excludes_only_pec_cells(kind, graded, with_pec):
    from rfx.runners.nonuniform import assemble_interface_eps_nu

    profiles = [np.full(n, d) for n, d in ((30, .001), (25, .0012), (40, .00075))]
    if graded:
        profiles = [p*(1+.2*np.sin(np.linspace(0, 2*np.pi, len(p)))) for p in profiles]
    grid = make_nonuniform_grid((.03, .03), profiles[2], .001, cpml_layers=4,
                               dx_profile=profiles[0], dy_profile=profiles[1])
    widths = [np.pad(p, (4, 5), mode='edge') for p in profiles]
    axes = []
    for p, d in zip(profiles, widths):
        nodes = (np.r_[0., np.cumsum(d[:-1])] - 4*p[0] if graded
                 else (np.arange(len(d))-4)*p[0])
        centres = nodes + d/2
        centres[-1] = centres[-2]  # the bounding node owns no outgoing cell
        axes.append(centres)
    shape = _shape(kind)
    sim = Simulation(freq_max=3e9, domain=(.03,)*3, dx=.001, cpml_layers=4,
                     interface_eps='dual_average')
    sim.add_material('dut', eps_r=4.)
    sim.add(shape, material='dut')
    cell_eps = np.where(_inside(shape, axes), 4., 1.)
    live = np.ones(grid.shape, dtype=bool)
    if with_pec:
        lo, hi = (.0155, .0111, .0111), (.0187, .0188, .0188)
        sim.add(Box(lo, hi), material='pec')
        xyz = np.meshgrid(*axes, indexing='ij')
        live = ~np.logical_and.reduce([(x >= l) & (x < h) for x, l, h in zip(xyz, lo, hi)])
    mats = sim._assemble_materials_nu(grid)[0]
    got = assemble_interface_eps_nu(sim, grid, mats)
    for c in range(3):
        transverse = [a for a in range(3) if a != c]
        num, den = np.zeros(grid.shape), np.zeros(grid.shape)
        area = np.ones(grid.shape)
        for a in transverse:
            dims = [1, 1, 1]
            dims[a] = grid.shape[a]
            area *= widths[a].reshape(dims)
        for d0 in (0, -1):
            for d1 in (0, -1):
                ix = [np.arange(n) for n in grid.shape]
                for a, delta in zip(transverse, (d0, d1)):
                    ix[a] = np.maximum(ix[a]+delta, 0)
                index = np.ix_(*ix)
                weights = (area*live)[index]
                num += cell_eps[index]*weights
                den += weights
        # All-PEC fallback is separately pinned by the existing rule tests;
        # every live edge here has an independently enumerated owner average.
        expected = np.divide(num, den, out=np.ones_like(num), where=den > 0)
        np.testing.assert_allclose(np.asarray(got[c])[den > 0], expected[den > 0],
                                   rtol=2*np.finfo(np.float32).eps, atol=0)
        assert np.isfinite(got[c]).all()
