"""Independent node cross-section expectations at the public prepared boundary."""

import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Cylinder, OrientedBox, Simulation, Sphere


class _Captured(Exception):
    pass


@pytest.fixture(autouse=True)
def mutation(monkeypatch):
    name = os.environ.get("RFX_CROSS_SECTION_MUTATION")
    if name:
        from tests.contracts.two_d_cross_section_mutations import install
        install(name, monkeypatch)


def _scene(shape, mode, **options):
    sim = Simulation(freq_max=10e9, domain=(.030, .030, .001), dx=.001,
                     boundary="cpml", cpml_layers=2, mode=mode, **options)
    sim.add_material("body", eps_r=4.)
    sim.add(shape, material="body")
    return sim


def _capture(monkeypatch, shape, mode, entry, path):
    captured = []

    def stop(product, *args, **kwargs):
        captured.append(product)
        raise _Captured

    monkeypatch.setattr("rfx.stepping.prepared.advance", stop)
    smoothing = {"off": False, "kottke": True, "node": "full_tensor",
                 "full": "full_tensor", "kottke_pec": "kottke_pec"}[path]
    with pytest.raises(_Captured):
        getattr(_scene(shape, mode), entry)(n_steps=1, skip_preflight=True,
                                          subpixel_smoothing=smoothing)
    return captured[0]


EXTENTS = [(-2, 2), (-.7, 2), (-.3, 2), (-.1, 2), (-2, .3),
           (-2, .1), (0, 2), (-2, 0), (.1, 2), (.3, 2), (-2, -.1)]


@pytest.mark.parametrize("extent", EXTENTS, ids=lambda e: f"z{e[0]:g}_{e[1]:g}")
@pytest.mark.parametrize("path", ["off", "kottke", "node", "full"])
@pytest.mark.parametrize("mode", ["2d_tez", "2d_tmz"])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_box_extent_table(monkeypatch, extent, path, mode, entry):
    lo, hi = (v * .001 for v in extent)
    shape = Box((.010, .010, lo), (.020, .020, hi))
    if lo > 0 or hi < 0:
        message = (
            "Box: in a 2-D model the geometry is its cross-section at z = 0 m; "
            f"this shape's z extent [{lo:g}, {hi:g}] m does not contain it. "
            "Move the shape so its z extent contains z = 0 m.")
        with pytest.raises(NotImplementedError, match=re.escape(message)):
            _capture(monkeypatch, shape, mode, entry, path)
        return
    product = _capture(monkeypatch, shape, mode, entry, path)
    components = product.materials.components
    values = (components.eps_update if path == "off" else
              components.tensor.node_smoothing.xi[:3] if path == "node" else
              components.tensor.eps)
    index = (round(.015 / .001) + product.grid.pad_x_lo,
             round(.015 / .001) + product.grid.pad_y_lo, 0)
    expected = 1 / 4 if path == "node" else 4.
    np.testing.assert_allclose([v[index] for v in values], expected, rtol=1e-6)
    if path == "full":
        nodes = components.tensor.node_smoothing
        np.testing.assert_allclose(nodes.triplets[(slice(None),) + index],
                                   np.broadcast_to(np.eye(3) / 4, (8, 3, 3)), rtol=1e-6)
        np.testing.assert_array_equal([v[index] for v in nodes.cross], 0.)


@pytest.mark.parametrize("mode", ["2d_tez", "2d_tmz"])
@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("kind", ["sphere", "cylinder_x"])
def test_curved_footprint(monkeypatch, mode, entry, kind):
    center = (.015506, .0155, .0003)
    shape = (Sphere(center, .005) if kind == "sphere" else
             Cylinder(center, .005, .010, "x"))
    product = _capture(monkeypatch, shape, mode, entry, "off")
    grid = product.grid
    x = (np.arange(grid.shape[0]) - grid.pad_x_lo + .5) * .001
    y = (np.arange(grid.shape[1]) - grid.pad_y_lo + .5) * .001
    radius_squared = .005**2 - .0003**2
    if kind == "sphere":
        expected = (x[:, None] - center[0])**2 + (y[None, :] - center[1])**2 <= radius_squared
        wrong = (x[:, None] - center[0])**2 + (y[None, :] - center[1])**2 <= .005**2 - .0002**2
        assert np.any(expected != wrong)
    else:
        expected = ((x[:, None] >= center[0] - .005)
                    & (x[:, None] < center[0] + .005)
                    & (abs(y[None, :] - center[1]) <= np.sqrt(radius_squared)))
    np.testing.assert_array_equal(np.asarray(product.materials.eps_r)[:, :, 0],
                                  np.where(expected, 4., 1.))


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("mode", ["2d_tez", "2d_tmz"])
def test_sphere_off_node(monkeypatch, entry, mode):
    with pytest.raises(NotImplementedError, match="Sphere: in a 2-D model.*does not contain it"):
        _capture(monkeypatch, Sphere((.015, .015, .006), .005), mode, entry, "off")


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("mode", ["2d_tez", "2d_tmz"])
def test_tilted_oriented_box(monkeypatch, entry, mode):
    c, s = np.cos(.2), np.sin(.2)
    shape = OrientedBox((.015, .015, 0.), (.010, .010, .010),
                        ((1., 0., 0.), (0., c, -s), (0., s, c)))
    with pytest.raises(NotImplementedError, match="OrientedBox: tilted out of the 2-D plane"):
        _capture(monkeypatch, shape, mode, entry, "off")


def test_traced_radius_and_presence():
    from rfx.geometry.cross_section import cross_section_2d
    grid = _scene(Sphere((.015, .015, 0.), .005), "2d_tmz").freeze_mesh()

    def radius(r):
        return cross_section_2d(Sphere((.015, .015, 0.), r), grid).radius

    assert float(jax.jit(jax.grad(radius))(.005)) == 1.
    with pytest.raises(NotImplementedError, match="Sphere: a traced z bound or z centre"):
        jax.jit(lambda z: cross_section_2d(Sphere((.015, .015, z), .005), grid).radius)(jnp.array(0.))


@pytest.mark.parametrize("offset,present", [(5e-13, True), (2e-12, False)])
def test_closed_node_tie(offset, present):
    from rfx.geometry.cross_section import cross_section_2d
    shape = Box((.010, .010, offset), (.020, .020, .002))
    grid = _scene(shape, "2d_tez").freeze_mesh()
    if present:
        result = cross_section_2d(shape, grid)
        assert result.corner_lo[2] < 0 < result.corner_hi[2]
    else:
        with pytest.raises(NotImplementedError, match="Box: in a 2-D model.*does not contain it"):
            cross_section_2d(shape, grid)


def test_sheet_and_legacy_conformal():
    from rfx.geometry.conformal import compute_conformal_weights
    from rfx.geometry.cross_section import cross_section_2d
    sheet = Cylinder((.015, .015, 0.), .005, 0., "z")
    grid = _scene(sheet, "2d_tmz").freeze_mesh()
    assert cross_section_2d(sheet, grid) is sheet
    with pytest.raises(NotImplementedError, match="Legacy conformal subsampling.*2-D"):
        compute_conformal_weights(grid, [sheet])


@pytest.mark.parametrize("mode", ["2d_tez", "2d_tmz"])
def test_named_sweep_uses_node_footprint(mode):
    from rfx.vmap_sweep import _build_batched_materials
    center = (.015506, .0155, .0003)
    sim = _scene(Sphere(center, .005), mode)
    grid = sim.freeze_mesh()
    base = sim._assemble_materials(grid)[0]
    batch = _build_batched_materials(sim, grid, base, "body.eps_r", jnp.array([4., 6.]))
    x = (np.arange(grid.shape[0]) - grid.pad_x_lo + .5) * .001
    y = (np.arange(grid.shape[1]) - grid.pad_y_lo + .5) * .001
    expected = (x[:, None] - center[0])**2 + (y[None, :] - center[1])**2 <= .005**2 - .0003**2
    for i, value in enumerate((4., 6.)):
        np.testing.assert_array_equal(np.asarray(batch.eps_r)[i, :, :, 0],
                                      np.where(expected, value, 1.))


@pytest.mark.parametrize("mode", ["2d_tez", "2d_tmz"])
def test_via_children_keep_cell_centres(monkeypatch, mode):
    from rfx.geometry.via import Via
    center, radius = (.0153, .0154), .0032
    shape = Via(center=center, drill_radius=radius, pad_radius=.0053, layers=[(0., .002)])
    product = _capture(monkeypatch, shape, mode, "forward", "off")
    grid = product.grid
    x = (np.arange(grid.shape[0]) - grid.pad_x_lo + .5) * .001
    y = (np.arange(grid.shape[1]) - grid.pad_y_lo + .5) * .001
    expected = ((x[:, None] >= center[0] - radius) & (x[:, None] < center[0] + radius)
                & (y[None, :] >= center[1] - radius) & (y[None, :] < center[1] + radius))
    np.testing.assert_array_equal(np.asarray(product.materials.eps_r)[:, :, 0],
                                  np.where(expected, 4., 1.))


def _prepared(monkeypatch, sim, **options):
    products = []

    def stop(product, *args, **kwargs):
        products.append(product)
        raise _Captured

    monkeypatch.setattr("rfx.stepping.prepared.advance", stop)
    with pytest.raises(_Captured):
        sim.run(n_steps=1, skip_preflight=True, **options)
    return products[0]


def _coordinates(grid, offset):
    return tuple((np.arange(grid.shape[a]) - getattr(grid, f"pad_{'xy'[a]}_lo")
                  + offset) * .001 for a in range(2))


@pytest.mark.parametrize("mode", ["2d_tez", "2d_tmz"])
@pytest.mark.parametrize("material", ["body", "pec"])
@pytest.mark.parametrize("z", [.0003, .005])
def test_off_node_sheet_presence(monkeypatch, mode, material, z):
    shape = Box((.010, .010, z), (.020, .020, z))
    sim = _scene(shape, mode)
    sim._geometry.clear()
    sim.add(shape, material=material)
    with pytest.raises(NotImplementedError, match="Box: in a 2-D model.*does not contain it"):
        _prepared(monkeypatch, sim)


@pytest.mark.parametrize("material", ["body", "pec"])
def test_node_sheet_keeps_sampling(monkeypatch, material):
    shape = Box((.010, .010, 0.), (.020, .020, 0.))
    sim = _scene(shape, "2d_tez", snap="declared")
    sim._geometry.clear()
    sim.add(shape, material=material)
    product = _prepared(monkeypatch, sim)
    if material == "body":
        x, y = _coordinates(product.grid, .5)
        expected = ((x[:, None] >= .010) & (x[:, None] < .020)
                    & (y[None, :] >= .010) & (y[None, :] < .020))
        np.testing.assert_array_equal(np.asarray(product.materials.eps_r)[:, :, 0],
                                      np.where(expected, 4., 1.))
    else:
        assert product.pec_cells is None
        assert len(product.sheets) == 1
        x, y = _coordinates(product.grid, 0.)
        for axis, edge in enumerate(product.pec_edges):
            expected = ((x[:, None] >= .010) & (x[:, None] <= .020)
                        & (y[None, :] >= .010) & (y[None, :] <= .020))
            if axis == 0:
                expected &= x[:, None] < .020
            elif axis == 1:
                expected &= y[None, :] < .020
            np.testing.assert_array_equal(np.asarray(edge)[:, :, 0], expected)


@pytest.mark.parametrize("extent", [(-.002, 0.), (-.0005, .0005), (0., .0009)])
@pytest.mark.parametrize("mode", ["2d_tez", "2d_tmz"])
def test_pec_volume_and_e_occupancy(monkeypatch, extent, mode):
    sim = _scene(Box((.010, .010, extent[0]), (.020, .020, extent[1])), mode)
    sim._geometry.clear()
    sim.add(Box((.010, .010, extent[0]), (.020, .020, extent[1])), material="pec")
    product = _prepared(monkeypatch, sim)
    x, y = _coordinates(product.grid, .5)
    expected = ((x[:, None] >= .010) & (x[:, None] < .020)
                & (y[None, :] >= .010) & (y[None, :] < .020))
    np.testing.assert_array_equal(np.asarray(product.pec_cells)[:, :, 0], expected)
    x, y = _coordinates(product.grid, 0.)
    for axis, edge in enumerate(product.pec_edges):
        expected = ((x[:, None] >= .010) & (x[:, None] <= .020)
                    & (y[None, :] >= .010) & (y[None, :] <= .020))
        if axis == 0:
            expected &= x[:, None] < .020
        elif axis == 1:
            expected &= y[None, :] < .020
        np.testing.assert_array_equal(np.asarray(edge)[:, :, 0], expected)


@pytest.mark.parametrize("mode", ["2d_tez", "2d_tmz"])
def test_conformal_face_on_slice(monkeypatch, mode):
    sim = _scene(Box((.010, .010, 0.), (.020, .020, .004)), mode)
    sim._geometry.clear()
    sim.add(Box((.010, .010, 0.), (.020, .020, .004)), material="pec")
    product = _prepared(monkeypatch, sim, conformal_pec=True)
    weights = product.materials.components.tensor.conformal_weights
    inside = (15 + product.grid.pad_x_lo, 15 + product.grid.pad_y_lo, 0)
    outside = (5 + product.grid.pad_x_lo, 5 + product.grid.pad_y_lo, 0)
    np.testing.assert_array_equal([w[inside] for w in weights], 0.)
    np.testing.assert_array_equal([w[outside] for w in weights], 1.)


@pytest.mark.parametrize("extent", [(-.0003, .002), (-.002, .0001)])
@pytest.mark.parametrize("mode", ["2d_tez", "2d_tmz"])
def test_kottke_pec_interior(monkeypatch, extent, mode):
    product = _capture(monkeypatch, Box((.010, .010, extent[0]), (.020, .020, extent[1])),
                       mode, "run", "kottke_pec")
    index = (15 + product.grid.pad_x_lo, 15 + product.grid.pad_y_lo, 0)
    np.testing.assert_allclose([v[index] for v in product.materials.components.tensor.inv_eps],
                               1 / 4, rtol=1e-6)


@pytest.mark.parametrize("mode", ["2d_tez", "2d_tmz"])
def test_debye_and_mu_section(mode):
    from rfx.materials.debye import DebyePole
    sim = Simulation(freq_max=10e9, domain=(.030, .030, .001), dx=.001,
                     boundary="cpml", cpml_layers=2, mode=mode)
    sim.add_material("body", eps_r=2., mu_r=3.,
                     debye_poles=[DebyePole(delta_eps=2., tau=1e-11)])
    sim.add(Box((.010, .010, -.002), (.020, .020, .0001)), material="body")
    grid = sim.freeze_mesh()
    materials, poles = sim._assemble_materials(grid)[:2]
    x, y = _coordinates(grid, .5)
    expected = ((x[:, None] >= .010) & (x[:, None] < .020)
                & (y[None, :] >= .010) & (y[None, :] < .020))
    np.testing.assert_array_equal(np.asarray(materials.eps_r)[:, :, 0], np.where(expected, 2., 1.))
    np.testing.assert_array_equal(np.asarray(materials.mu_r)[:, :, 0], np.where(expected, 3., 1.))
    for mask in poles[1]:
        np.testing.assert_array_equal(np.asarray(mask)[:, :, 0], expected)


@pytest.mark.parametrize("kind", ["oriented_box", "cylinder_z", "mesh"])
@pytest.mark.parametrize("mode", ["2d_tez", "2d_tmz"])
def test_other_sections(monkeypatch, kind, mode):
    c, s = np.cos(.3), np.sin(.3)
    center = (.0153, .0151, .0004)
    if kind == "oriented_box":
        shape = OrientedBox(center, (.010, .006, .002),
                            ((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))
    elif kind == "cylinder_z":
        shape = Cylinder(center, .005, .002, "z")
    else:
        trimesh = pytest.importorskip("trimesh")
        from rfx.geometry.mesh_import import MeshShape
        mesh = trimesh.creation.box(extents=(.010, .006, .002))
        mesh.apply_translation(center)
        shape = MeshShape(mesh)
    product = _capture(monkeypatch, shape, mode, "run", "off")
    x, y = _coordinates(product.grid, 0. if kind == "mesh" else .5)
    dx, dy = x[:, None] - center[0], y[None, :] - center[1]
    if kind == "cylinder_z":
        expected = dx**2 + dy**2 <= .005**2
    elif kind == "oriented_box":
        u, v = c * dx + s * dy, -s * dx + c * dy
        expected = (u >= -.005) & (u < .005) & (v >= -.003) & (v < .003)
    else:
        expected = (dx >= -.005) & (dx < .005) & (dy >= -.003) & (dy < .003)
    np.testing.assert_array_equal(np.asarray(product.materials.eps_r)[:, :, 0],
                                  np.where(expected, 4., 1.))


def test_curved_patch_children_remain_sheets(monkeypatch):
    from rfx import CurvedPatch
    shape = CurvedPatch(center=(.015, .015, 0.), length=.009, width=.008, radius=.02, axis="x")
    with pytest.raises(ValueError, match="CurvedPatch.*ZERO cells"):
        _capture(monkeypatch, shape, "2d_tez", "run", "off")


@pytest.mark.parametrize("angle,accepted", [(2e-8, False), (2e-15, True)])
def test_oriented_box_orthogonality_tolerance(angle, accepted):
    from rfx.geometry.cross_section import cross_section_2d
    c, s = np.cos(angle), np.sin(angle)
    shape = OrientedBox((.015, .015, 0.), (.010, .010, .010),
                        ((1., 0., 0.), (0., c, -s), (0., s, c)))
    grid = _scene(shape, "2d_tez").freeze_mesh()
    if accepted:
        assert cross_section_2d(shape, grid).size[2] > .010
    else:
        with pytest.raises(NotImplementedError, match="OrientedBox: tilted out of the 2-D plane"):
            cross_section_2d(shape, grid)


def test_traced_uniform_dx_public_admission():
    from rfx._traced_refusal import TracedStructureRefusal
    with pytest.raises(TracedStructureRefusal, match=r"Simulation\(dx=\.\.\.\) is traced"):
        jax.jit(lambda dx: Simulation(freq_max=10e9, domain=(.030, .030, .001),
                                      dx=dx, mode="2d_tez").freeze_mesh().dt)(jnp.array(.001))


def test_section_cache_miss_recomputes():
    from rfx.geometry.cross_section import section_entries
    sim = _scene(Sphere((.015, .015, .003), .005), "2d_tmz")
    grid = sim.freeze_mesh()
    section = section_entries(sim, grid)[0].shape
    assert isinstance(section, Cylinder)
    np.testing.assert_allclose(section.radius, np.sqrt(.005**2 - .003**2))
    sim.add(Box((.010, .010, .005), (.020, .020, .006)), material="body")
    with pytest.raises(NotImplementedError, match="Box: in a 2-D model.*does not contain it"):
        section_entries(sim, grid)


@pytest.mark.parametrize("kind", ["sphere", "cylinder_x", "cylinder_y"])
@pytest.mark.parametrize("material", ["body", "pec"])
def test_extent_verdict_uses_section(monkeypatch, kind, material):
    center = (.015, .015, .003)
    shape = (Sphere(center, .005) if kind == "sphere" else
             Cylinder(center, .005, .010, kind[-1]))
    sim = _scene(shape, "2d_tmz")
    sim._geometry.clear()
    sim.add(shape, material=material)
    product = _prepared(monkeypatch, sim)
    row = next(row for row in product.realization_table.rows if row.label == "geometry[0]")
    radius = np.sqrt(.005**2 - .003**2)
    half = [radius, radius]
    if kind != "sphere":
        half["xy".index(kind[-1])] = .010 / 2
    np.testing.assert_allclose(np.asarray(row.declared_bounds_m)[:, :2],
                               [np.asarray(center[:2]) - half, np.asarray(center[:2]) + half])
    np.testing.assert_allclose([a.declared_span_m for a in row.axes[:2]], 2 * np.array(half))


@pytest.mark.parametrize("kind", ["sphere", "cylinder_x"])
@pytest.mark.parametrize("mode", ["2d_tez", "2d_tmz"])
def test_pad_fill_uses_section_bounds(monkeypatch, kind, mode):
    """A 3-D bound touching a pad does not extend an interior section."""
    center = (.015, .025, .004)
    shape = (Sphere(center, .005) if kind == "sphere" else
             Cylinder(center, .005, .010, "x"))
    product = _capture(monkeypatch, shape, mode, "run", "off")
    grid = product.grid
    x = (np.arange(grid.shape[0]) - grid.pad_x_lo + .5) * .001
    y = (np.arange(grid.shape[1]) - grid.pad_y_lo + .5) * .001
    radius_squared = .005**2 - .004**2
    expected = (((x[:, None] - center[0])**2 + (y[None, :] - center[1])**2
                 <= radius_squared) if kind == "sphere" else
                ((x[:, None] >= center[0] - .005)
                 & (x[:, None] < center[0] + .005)
                 & (abs(y[None, :] - center[1]) <= np.sqrt(radius_squared))))
    np.testing.assert_array_equal(np.asarray(product.materials.eps_r)[:, :, 0],
                                  np.where(expected, 4., 1.))


def test_in_plane_sheet_retains_its_rules():
    from rfx.geometry.cross_section import cross_section_2d
    shape = Box((.015, .010, .001), (.015, .020, .002))
    grid = _scene(shape, "2d_tez").freeze_mesh()
    assert cross_section_2d(shape, grid) is shape


def test_traced_uniform_dx_transform_guard():
    import copy
    from rfx.geometry.cross_section import cross_section_2d
    shape = Sphere((.015, .015, 0.), .005)
    grid = _scene(shape, "2d_tez").freeze_mesh()

    def section(dx):
        traced_grid = copy.copy(grid)
        traced_grid.dx = dx
        return cross_section_2d(shape, traced_grid).radius

    with pytest.raises(NotImplementedError, match="2-D cross-section: traced uniform dx"):
        jax.jit(section)(jnp.array(.001))


@pytest.mark.parametrize("kind", ["sphere", "via"])
@pytest.mark.parametrize("mode", ["2d_tez", "2d_tmz"])
def test_conductor_message_names_declared_shape(kind, mode):
    from rfx.geometry.via import Via
    shape = (Sphere((.0155, .0155, 0.), .0006) if kind == "sphere" else
             Via(center=(.0155, .0155), drill_radius=.0005, pad_radius=.005,
                 layers=[(-.001, .002)]))
    sim = _scene(shape, mode, snap="declared")
    sim._geometry.clear()
    sim.add(shape, material="pec")
    findings = sim.preflight().by_code("pec_box_one_cell")
    assert len(findings) == 1
    declared_name = "Sphere" if kind == "sphere" else "Via"
    assert f"geometry[0] 'pec' ({declared_name})" in str(findings[0])


@pytest.mark.parametrize("mode", ["2d_tez", "2d_tmz"])
def test_via_keeps_on_node_pad_sheet(monkeypatch, mode):
    from rfx.geometry.via import Via
    shape = Via(center=(.015, .015), drill_radius=.002,
                pad_radius=.005, layers=[(0., .002)])
    sim = _scene(shape, mode)
    sim._geometry.clear()
    sim.add(shape, material="pec")
    product = _prepared(monkeypatch, sim)
    x, y = _coordinates(product.grid, .5)
    expected = ((x[:, None] >= .010) & (x[:, None] < .020)
                & (y[None, :] >= .010) & (y[None, :] < .020))
    assert np.count_nonzero(expected) == 100
    np.testing.assert_array_equal(np.asarray(product.pec_cells)[:, :, 0], expected)


@pytest.mark.parametrize("mode", ["2d_tez", "2d_tmz"])
def test_via_excludes_off_node_pad_sheets(monkeypatch, mode):
    from rfx.geometry.via import Via
    center, radius = (.015, .015), .002
    shape = Via(center=center, drill_radius=radius,
                pad_radius=.005, layers=[(-.001, .002)])
    sim = _scene(shape, mode)
    sim._geometry.clear()
    sim.add(shape, material="pec")
    product = _prepared(monkeypatch, sim)
    x, y = _coordinates(product.grid, .5)
    expected = ((x[:, None] >= center[0] - radius)
                & (x[:, None] < center[0] + radius)
                & (y[None, :] >= center[1] - radius)
                & (y[None, :] < center[1] + radius))
    assert np.count_nonzero(expected) == 16
    np.testing.assert_array_equal(np.asarray(product.pec_cells)[:, :, 0], expected)
