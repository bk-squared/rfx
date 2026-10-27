"""#1966 record 6: assembled cells, BASE means, hand-written wall rows."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, DebyePole, Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.core.yee import component_e_materials, component_h_materials
from rfx.geometry.rasterize_grid import extend_cpml_pad_materials, unpadded_high_walls
from rfx.model.materials import electric_cell_sizes, pole_component_weights
from rfx.stepping.slab_view import slab_views

H = 1e-3
PERIODIC = (False, False, False)


def scene(kind, span=(3, 8), *, graded=False, wall="pmc"):
    options = {"dz_profile": H * np.array([1, 1, 1, 1, 1, 1, 1, .9, 1.1, 1])} if graded else {}
    sim = Simulation(freq_max=10e9, domain=(6*H, 8*H, 10*H), dx=H,
                     boundary=BoundarySpec(x="pec", y=wall, z="cpml"),
                     cpml_layers=4, **options)
    if kind == "film":
        sim.add_thin_conductor(Box((0, span[0]*H, 6*H), (6*H, span[1]*H, 6*H)),
                               sigma_bulk=1/(377*35e-6), thickness=35e-6)
    else:
        poles = {"debye_poles": [DebyePole(3., 1e-11)]} if kind == "debye" else {}
        sim.add_material("body", eps_r=4., sigma=.2, mu_r=2., **poles)
        sim.add(Box((0, span[0]*H, 3*H), (6*H, span[1]*H, 4*H)), material="body")
    grid = sim._build_realized_grid()
    result = (sim._assemble_materials_nu(grid) if graded else
              sim._assemble_materials(grid, pec_sheets=[], pec_wires=[]))
    return grid, result


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("wall", ["pmc", "pec"])
@pytest.mark.parametrize("kind", ["film", "static"])
@pytest.mark.parametrize("span,row", [
    ((3, 8), [0, 0, 0, .5, 1, 1, 1, 1, 1]),
    ((0, 5), [1, 1, 1, 1, 1, .5, 0, 0, 0]),
    ((3, 7), [0, 0, 0, .5, 1, 1, 1, .5, 0]),
])
def test_assembled_wall_rows(graded, wall, kind, span, row):
    grid, result = scene(kind, span, graded=graded, wall=wall)
    mats = result[0]
    widths = electric_cell_sizes(grid)
    if graded:
        assert widths[2] is not None  # Ex averages over this genuinely graded axis.
        assert np.ptp(np.asarray(widths[2])) > 0
    eps, sigma = component_e_materials(mats, PERIODIC, high_wall=unpadded_high_walls(grid), cell_sizes=widths)
    mid = grid.shape[0] // 2
    if kind == "film":
        z = grid.pad_z_lo + 6
        # One plane, density = sheet conductance / the 1 mm dual width.
        np.testing.assert_allclose(np.asarray(sigma[0])[mid, :, z] * (377*H), row,
                                   rtol=2e-7, atol=0)
    else:
        z = grid.pad_z_lo + 3
        # Half of Ex's z support is dielectric; the hand-written y row
        # gives epsilon 1 + 1.5*row, conductivity .1*row.
        np.testing.assert_array_equal(np.asarray(eps[0])[mid, :, z], 1 + 1.5*np.array(row))
        np.testing.assert_allclose(np.asarray(sigma[0])[mid, :, z], .1*np.array(row),
                                   rtol=2e-7, atol=0)
        if span == (3, 8):
            h = component_h_materials(mats, PERIODIC, high_wall=unpadded_high_walls(grid))
            assert float(h[1][mid, -1, z]) == 2.
            assert float(h[1][mid, -2, z]) == 2.


def test_debye_wall_replicates_static_and_pole_edges_together():
    grid, result = scene("debye")
    mats, debye = result[:2]
    walls = unpadded_high_walls(grid)
    eps, sig = component_e_materials(mats, PERIODIC, high_wall=walls)
    weights = pole_component_weights(debye[1][0], PERIODIC, high_wall=walls)
    index = (grid.shape[0]//2, grid.shape[1]-1, grid.pad_z_lo+3)
    # #808/#627b: promoting eps_inf without the poles is forbidden.
    # Stored outside cells stay vacuum; both inside Ex cells carry the pole.
    assert float(mats.eps_r[index]) == 1.
    assert float(mats.sigma[index]) == 0.
    assert float(mats.mu_r[index]) == 1.
    assert float(eps[0][index]) == 2.5
    np.testing.assert_allclose(sig[0][index], .1, rtol=2e-7)
    np.testing.assert_allclose(component_h_materials(mats, high_wall=walls)[1][index], 2., rtol=2e-7)
    assert float(weights[0][index]) == .5


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_wall_read_preserves_air_gap_and_ignores_stored_outside(axis):
    shape = (5, 5, 5)
    eps = jnp.ones(shape)
    interior = [slice(None)]*3
    interior[axis] = -3
    eps = eps.at[tuple(interior)].set(4.)
    walls = tuple(a == axis for a in range(3))
    got, _, _ = extend_cpml_pad_materials(eps, jnp.zeros(shape), jnp.ones(shape),
                                        0, 0, 0, 0, 0, 0)
    np.testing.assert_array_equal(got, eps)  # Do not bridge a one-cell air gap.
    from rfx.core.yee import edge_mean_components
    outer = [slice(None)] * 3
    outer[axis] = -1
    for edge in edge_mean_components(eps, high_wall=walls):
        np.testing.assert_array_equal(edge[tuple(outer)], 1.)
    interior[axis] = -2
    eps = eps.at[tuple(interior)].set(4.)
    interior[axis] = -1
    eps = eps.at[tuple(interior)].set(2.)
    got, _, _ = extend_cpml_pad_materials(eps, jnp.zeros(shape), jnp.ones(shape),
                                        0, 0, 0, 0, 0, 0)
    np.testing.assert_array_equal(got, eps)
    for edge in edge_mean_components(eps, high_wall=walls):
        np.testing.assert_array_equal(edge[tuple(outer)], 4.)


@pytest.mark.parametrize("nx", [17, 19])
@pytest.mark.parametrize("devices", [2, 3])
def test_high_x_pmc_slabs_equal_whole(nx, devices):
    from rfx.runners._distributed_common import _split_materials, slab_e_component_materials
    sim = Simulation(freq_max=10e9, domain=((nx-1)*H, 6*H, 5*H), dx=H,
                     boundary=BoundarySpec(x="pmc", y="pec", z="pec"), cpml_layers=0)
    sim.add_material("body", eps_r=4., mu_r=2.)
    sim.add(Box((3*H, 2*H, H), ((nx-1)*H, 4*H, 4*H)), material="body")
    grid = sim._build_realized_grid()
    assert grid.shape[0] == nx
    mats = sim._assemble_materials(grid)[0]
    walls = unpadded_high_walls(grid)
    whole = component_e_materials(mats, high_wall=walls), component_h_materials(mats, high_wall=walls)
    assert float(whole[0][0][1][-1, 2, 2]) == 4.
    for view in slab_views(grid, devices):
        local = mats._replace(eps_r=view.rows(mats.eps_r), sigma=view.rows(mats.sigma),
                              mu_r=view.rows(mats.mu_r))
        local_walls = (walls[0] and view.last, *walls[1:])
        actual = (component_e_materials(local, high_wall=local_walls),
                  component_h_materials(local, high_wall=local_walls))
        for got, expected in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(whole)):
            assert np.asarray(view.own(got)).tobytes() == np.asarray(expected[view.x0:view.x1]).tobytes()
    # Execute the uniform runner's actual split/mean on virtual ranks as
    # well. No physical multi-device availability can skip this contract.
    split = _split_materials(mats, devices)
    per = split.eps_r.shape[1] - 2
    for rank in range(devices):
        local = jax.tree.map(lambda a: a[rank], split)
        actual = slab_e_component_materials(local, per, nx, rank=rank, high_wall=walls)
        start, stop = rank*per, min((rank+1)*per, nx)
        for got, expected in zip(jax.tree.leaves(actual), jax.tree.leaves(whole[0])):
            assert np.asarray(got[1:1+stop-start]).tobytes() == np.asarray(expected[start:stop]).tobytes()


@pytest.mark.parametrize("end,expected", [(8, 2.5), (7, 1.75)])
def test_traced_design_window_repairs_only_global_wall(end, expected):
    from rfx.core.yee import EPS_0
    from rfx.model.materials import _design_box_edge_coeffs
    sim = Simulation(freq_max=10e9, domain=(6*H, 8*H, 10*H), dx=H,
                     boundary=BoundarySpec(x="pec", y="pmc", z="cpml"), cpml_layers=4)
    grid = sim._build_realized_grid()
    mats = sim._assemble_materials(grid)[0]
    z = grid.pad_z_lo + 3
    bounds = (1, 5, 3, end, z, z+1)

    @jax.jit
    def wall_eps(value):
        _, _, cb = _design_box_edge_coeffs(
            bounds, jnp.full((4, end-3, 1), value), None, mats,
            grid.dt, grid.shape, grid=grid)
        return grid.dt / (EPS_0 * cb[0][1, -1, 0])

    np.testing.assert_allclose(wall_eps(4.), expected, rtol=2e-7)
    np.testing.assert_allclose(jax.grad(wall_eps)(4.), .5 if end == 8 else .25, rtol=3e-7)


@pytest.mark.parametrize("seeded", [False, True], ids=["vacuum-background", "geometry-eps2"])
@pytest.mark.parametrize("end", [8, 7], ids=["global-wall", "interior-end"])
@pytest.mark.parametrize("value,wall,interior", [
    (4., 2.5, 1.75), (1., 1., 1.), (1.0001, 1.00005, 1.000025),
])
def test_design_written_wall_value_and_derivative(seeded, end, value, wall, interior):
    from rfx.core.yee import EPS_0
    from rfx.model.materials import _design_box_edge_coeffs
    sim = Simulation(freq_max=10e9, domain=(6*H, 8*H, 10*H), dx=H,
                     boundary=BoundarySpec(x="pec", y="pmc", z="cpml"), cpml_layers=4)
    if seeded:
        sim.add_material("background", eps_r=2.)
        sim.add(Box((H, 3*H, 3*H), (5*H, end*H, 4*H)), material="background")
    grid = sim._build_realized_grid()
    mats = sim._assemble_materials(grid)[0]
    z = grid.pad_z_lo + 3
    assert float(mats.eps_r[2, 8, z]) == 1.
    bounds = (1, 5, 3, end, z, z+1)

    @jax.jit
    def edge_eps(v):
        _, _, cb = _design_box_edge_coeffs(
            bounds, jnp.full((4, end-3, 1), v), None, mats,
            grid.dt, grid.shape, grid=grid)
        return grid.dt / (EPS_0 * cb[0][1, -1, 0])

    # Wall: (v+v+1+1)/4. Interior plus edge: (v+1+1+1)/4,
    # the unchanged main convention. Neither reference uses a product mean.
    actual, derivative = jax.value_and_grad(edge_eps)(value)
    np.testing.assert_allclose(actual, wall if end == 8 else interior, rtol=2e-7)
    np.testing.assert_allclose(derivative, .5 if end == 8 else .25, rtol=3e-7)
    print(f"seeded={seeded} end={end} design={value}: "
          f"Ex={float(actual)} derivative={float(derivative)}")


@pytest.mark.parametrize("seeded", [False, True], ids=["vacuum-background", "geometry-eps2"])
@pytest.mark.parametrize("value", [4., 1.])
def test_design_window_corner_of_two_high_walls(seeded, value):
    """Two high PMC walls meet: the corner node is design-owned on both axes.

    The Ez edge at the corner averages four cells (x and y transverse); all
    four are the design cell after the sequential repair, so Ez = v, d/dv = 1.
    """
    from rfx.core.yee import EPS_0
    from rfx.model.materials import _design_box_edge_coeffs
    sim = Simulation(freq_max=10e9, domain=(6*H, 8*H, 10*H), dx=H,
                     boundary=BoundarySpec(x="pmc", y="pmc", z="cpml"), cpml_layers=4)
    if seeded:
        sim.add_material("background", eps_r=2.)
        sim.add(Box((H, 3*H, 3*H), (6*H, 8*H, 4*H)), material="background")
    grid = sim._build_realized_grid()
    mats = sim._assemble_materials(grid)[0]
    z = grid.pad_z_lo + 3
    bounds = (1, 6, 3, 8, z, z+1)

    @jax.jit
    def corner_eps(v):
        _, _, cb = _design_box_edge_coeffs(
            bounds, jnp.full((5, 5, 1), v), None, mats,
            grid.dt, grid.shape, grid=grid)
        return grid.dt / (EPS_0 * cb[2][-1, -1, 0])

    actual, derivative = jax.value_and_grad(corner_eps)(value)
    np.testing.assert_allclose(actual, value, rtol=2e-7)
    np.testing.assert_allclose(derivative, 1., rtol=3e-7)


def test_override_and_checkpoint_are_explicit_cell_inputs(tmp_path):
    from rfx.checkpoint import load_materials, save_materials
    from rfx.model.overrides import apply_material_overrides
    grid, result = scene("static")
    repaired = result[0]
    index = (grid.shape[0]//2, grid.shape[1]-1, grid.pad_z_lo+3)
    # These writers carry explicit values after assembly. They have no
    # geometry/pole provenance from which to decide a dropped-node repair.
    explicit = repaired._replace(eps_r=repaired.eps_r.at[:, -1, :].set(1.))
    overridden = apply_material_overrides(repaired, eps_override=explicit.eps_r)
    assert float(component_e_materials(overridden, high_wall=unpadded_high_walls(grid))[0][0][index]) == 2.5
    path = tmp_path / "wall-materials.h5"
    save_materials(path, explicit)
    loaded = load_materials(path)
    np.testing.assert_array_equal(loaded.eps_r, explicit.eps_r)
    assert float(component_e_materials(loaded, high_wall=unpadded_high_walls(grid))[0][0][index]) == 2.5


def test_periodic_terminal_is_a_real_cell():
    from rfx.geometry.rasterize_grid import unpadded_high_walls
    sim = Simulation(freq_max=10e9, domain=(6*H, 8*H, 10*H), dx=H,
                     boundary=BoundarySpec(x="pec", y="periodic", z="pec"), cpml_layers=0)
    sim.add_material("body", eps_r=4.)
    sim.add(Box((H, 5*H, H), (5*H, 7*H, 9*H)), material="body")
    grid = sim._build_realized_grid()
    assert unpadded_high_walls(grid) == (True, False, True)
    mats = sim._assemble_materials(grid)[0]
    assert float(mats.eps_r[2, -1, 2]) == 1.
    assert float(component_e_materials(mats, (False, True, False), high_wall=unpadded_high_walls(grid))[0][0][2, -1, 2]) == 2.5


@pytest.mark.parametrize("reached,last_inside", [(True, False), (False, False), (True, True)])
def test_fine_sample_requires_reached_wall_and_beyond_coordinate(reached, last_inside):
    from rfx.geometry.rasterize_grid import (
        coords_from_fine_grid, coords_from_uniform_grid, rasterize_geometry)
    from rfx.subgridding.validation import overlap_fine_extent
    sim = Simulation(freq_max=10e9, domain=(6*H, 8*H, 10*H), dx=H,
                     boundary=BoundarySpec(x="pec", y="pmc", z="pec"), cpml_layers=0)
    end = 8 if reached and not last_inside else 6
    sim.add_material("body", eps_r=4., mu_r=2.)
    sim.add(Box((0, 3*H, 3*H), (6*H, end*H, 4*H)), material="body")
    grid = sim._build_realized_grid()
    counts = tuple(overlap_fine_extent(n, 2) for n in (7, end+1, 11))
    coords = coords_from_fine_grid(*counts, H/2, 0, 0, 0)
    assert float(coords.y[-1]) == (end+.25)*H
    assert float(grid.domain[1]) == 8*H
    raw, debye, lorentz, *_ = rasterize_geometry(
        sim._geometry, sim._resolve_material, coords, centres=coords)
    global_coords = coords_from_uniform_grid(grid)
    walls = (False, unpadded_high_walls(grid)[1] and reached
             and float(coords.y[-1]) > float(global_coords.y[-1]), False)
    index = (6, counts[1]-1, 6)
    assert float(component_e_materials(raw, high_wall=(False,) * 3)[0][0][index]) == 1.75
    assert float(component_e_materials(raw, high_wall=walls)[0][0][index]) == (
        2.5 if reached and not last_inside else 1.75)
    assert float(raw.eps_r[index]) == 1.


def test_dual_average_rasterization_already_repeats_last_centre(monkeypatch):
    import rfx.geometry.rasterize_grid as raster
    from rfx.runners.nonuniform import assemble_interface_eps_nu
    grid, result = scene("static", graded=True)
    sim = Simulation(freq_max=10e9, domain=(6*H, 8*H, 10*H), dx=H,
                     boundary=BoundarySpec(x="pec", y="pmc", z="cpml"), cpml_layers=4,
                     dz_profile=H*np.array([1, 1, 1, 1, 1, 1, 1, .9, 1.1, 1]),
                     interface_eps="dual_average")
    sim.add_material("body", eps_r=4., sigma=.2, mu_r=2.)
    sim.add(Box((0, 3*H, 3*H), (6*H, 8*H, 4*H)), material="body")
    observed = []
    original = raster.rasterize_geometry

    def capture(*args, **kwargs):
        out = original(*args, **kwargs)
        observed.append((args[2], out[0]))
        return out

    monkeypatch.setattr(raster, "rasterize_geometry", capture)
    eps = assemble_interface_eps_nu(sim, grid, result[0], high_wall=unpadded_high_walls(grid))
    coords, cells = observed[0]
    assert coords.y[-1] == coords.y[-2] == 7.5*H
    np.testing.assert_array_equal(cells.eps_r[:, -1], cells.eps_r[:, -2])
    assert float(eps[0][3, -1, grid.pad_z_lo+3]) == 2.5


def test_batched_writer_leaves_stored_wall_and_producer_reads_inside():
    from rfx.vmap_sweep import _extend_batched_cpml_pad
    grid, result = scene("static")
    mats = result[0]
    eps = mats.eps_r.at[:, -1, :].set(1.)
    sig = mats.sigma.at[:, -1, :].set(0.)
    mu = mats.mu_r.at[:, -1, :].set(1.)
    eps = jnp.stack([eps, jnp.where(eps == 4., 7., eps)])
    fixed, _, _ = _extend_batched_cpml_pad(eps, jnp.stack([sig, sig]), jnp.stack([mu, mu]), grid)
    np.testing.assert_array_equal(fixed[:, 3, -1, grid.pad_z_lo+3], [1., 1.])
    from rfx.core.yee import edge_mean_components
    means = jax.vmap(lambda a: edge_mean_components(a, high_wall=unpadded_high_walls(grid)))(fixed)
    np.testing.assert_array_equal(means[0][:, 3, -1, grid.pad_z_lo+3], [2.5, 4.])
