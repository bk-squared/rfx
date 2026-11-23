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


def _scene(shape, mode):
    sim = Simulation(freq_max=10e9, domain=(.030, .030, .001), dx=.001,
                     boundary="cpml", cpml_layers=2, mode=mode)
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
                 "full": "full_tensor"}[path]
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


def test_via_children_keep_cell_centres(monkeypatch):
    from rfx.geometry.via import Via
    center, radius = (.0153, .0154), .0053
    shape = Via(center=center, drill_radius=.0032, pad_radius=radius, layers=[(0., .002)])
    product = _capture(monkeypatch, shape, "2d_tmz", "forward", "off")
    grid = product.grid
    x = (np.arange(grid.shape[0]) - grid.pad_x_lo + .5) * .001
    y = (np.arange(grid.shape[1]) - grid.pad_y_lo + .5) * .001
    expected = ((x[:, None] >= center[0] - radius) & (x[:, None] < center[0] + radius)
                & (y[None, :] >= center[1] - radius) & (y[None, :] < center[1] + radius))
    np.testing.assert_array_equal(np.asarray(product.materials.eps_r)[:, :, 0],
                                  np.where(expected, 4., 1.))
