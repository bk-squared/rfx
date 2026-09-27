"""Static geometry setup oracles; no FDTD or angular-accuracy claim.

F0 is fixed before implementation. Its referee evaluates world-space plane
inequalities on independently constructed physical coordinates, rather than
calling the product's rotation, mask, or node/centre helpers.
"""
from dataclasses import FrozenInstanceError
from contextlib import nullcontext
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import OrientedBox, Simulation
from rfx.core.yee import edge_averaged_materials
from rfx.geometry.csg import rasterize
from rfx.geometry.rasterize_grid import coords_from_fine_grid, rasterize_geometry
from rfx.nonuniform import make_nonuniform_grid
from tests._x64_compat import enable_x64

Q = np.array([[1, 2, 2], [2, 1, -2], [-2, 2, -1]], dtype=np.float64) / 3
R0 = np.array([10000.125, -20000.25, 30000.375])
Q0 = np.array([.04, .045, .05])
C = R0 + np.array([.0037, -.0041, .0023])
SIZE = (.021, .017, .013)
DOMAIN = (.08, .09, .10)
DX = .002
PAD = 4


def _shape():
    return OrientedBox(Q0 + Q.T @ (C - R0), SIZE, Q.T)


def _sim(shape=None, **kwargs):
    sim = Simulation(freq_max=6e9, domain=DOMAIN, dx=DX, cpml_layers=PAD, **kwargs)
    sim.add_material("body", eps_r=3.25, sigma=.125)
    sim.add(_shape() if shape is None else shape, material="body")
    return sim


def _world_inside(axes):
    x, y, z = np.meshgrid(*axes, indexing="ij")
    x, y, z = x - Q0[0], y - Q0[1], z - Q0[2]
    # Explicit fixed world-frame plane equations; no product rotation helper.
    world = (R0[0] + (x + 2*y + 2*z)/3,
             R0[1] + (2*x + y - 2*z)/3,
             R0[2] + (-2*x + 2*y - z)/3)
    return np.logical_and.reduce([
        (w >= c - s/2) & (w < c + s/2)
        for w, c, s in zip(world, C, SIZE)])


def _edge_means(cells):
    """Enumerate each E edge's four incident primal cells, boundary-replicated."""
    index = np.indices(cells.shape)
    result = []
    for component in range(3):
        transverse = [a for a in range(3) if a != component]
        values = []
        for offsets in product((0, -1), repeat=2):
            sample = index.copy()
            for axis, offset in zip(transverse, offsets):
                sample[axis] = np.maximum(sample[axis] + offset, 0)
            values.append(cells[tuple(sample)])
        result.append(np.sum(values, axis=0) / 4)
    return result


def _assert_materials(mats, expected):
    eps, sigma = np.where(expected, 3.25, 1.), np.where(expected, .125, 0.)
    np.testing.assert_array_equal(mats.eps_r, eps)
    np.testing.assert_array_equal(mats.sigma, sigma)
    for got, want in zip(edge_averaged_materials(mats.eps_r, mats.sigma),
                         (_edge_means(eps), _edge_means(sigma))):
        for actual, target in zip(got, want):
            np.testing.assert_array_equal(actual, target)


@pytest.mark.parametrize("x64", [False, True])
def test_f0_world_planes_match_every_material_cell_and_e_edge(x64):
    with enable_x64() if x64 else nullcontext():
        sim = _sim()
        grid = sim._build_grid()
        centres = [(np.arange(n)-p+.5)*DX for n, p in zip(grid.shape, grid.axis_pads)]
        expected = _world_inside(centres)
        assert expected.any()
        mats = sim._assemble_materials(grid)[0]
        _assert_materials(mats, expected)
        eps, sigma = rasterize(grid, [(_shape(), 3.25, .125)])
        np.testing.assert_array_equal(eps, mats.eps_r)
        np.testing.assert_array_equal(sigma, mats.sigma)
        # The future closed-box source's four stencil planes on every axis
        # remain vacuum. This is setup geometry, not a source integration run.
        for axis, n in enumerate(grid.shape):
            lo, hi = PAD + 4, n - PAD - 4 - 1
            assert not np.take(expected, (lo-1, lo, hi, hi+1), axis=axis).any()


def test_f0_node_contract_and_mutations_are_distinguishable():
    sim = _sim()
    grid = sim._build_grid()
    nodes = [(np.arange(n)-p)*DX for n, p in zip(grid.shape, grid.axis_pads)]
    centres = [x + DX/2 for x in nodes]
    node_mask, expected = _world_inside(nodes), _world_inside(centres)
    np.testing.assert_array_equal(_shape().mask(grid), node_mask)
    assert np.count_nonzero(node_mask != expected) > 0
    transposed = OrientedBox(_shape().center, SIZE, Q)
    assert np.count_nonzero(np.asarray(transposed.mask_on_coords(*centres)) != expected) > 0
    with pytest.raises(ValueError, match="proper orthogonal"):
        OrientedBox(_shape().center, SIZE, Q.T @ np.diag([-1., 1., 1.]))


@pytest.mark.parametrize("graded", [False, True])
def test_nonuniform_material_centres_are_sampled_once(graded):
    profiles = [np.full(n, DX) for n in (40, 45, 50)]
    if graded:
        profiles = [p*(1+.1*np.sin(np.linspace(0, 2*np.pi, len(p)))) for p in profiles]
    grid = make_nonuniform_grid(DOMAIN[:2], profiles[2], DX, cpml_layers=PAD,
                               dx_profile=profiles[0], dy_profile=profiles[1])
    axes = []
    for p in profiles:
        widths = np.pad(p, (PAD, PAD+1), mode="edge")
        nodes = (np.r_[0., np.cumsum(widths[:-1])] - PAD*p[0] if graded
                 else (np.arange(len(widths))-PAD)*p[0])
        axes.append(nodes + widths/2)
    _assert_materials(_sim()._assemble_materials_nu(grid)[0], _world_inside(axes))


def test_fine_region_coordinates_are_already_cell_centres():
    offsets, counts, dx = (.015, .016, .020), (43, 44, 45), .001
    coords = coords_from_fine_grid(*counts, dx, *offsets)
    sim = _sim()
    mats = rasterize_geometry(sim._geometry, sim._resolve_material, coords, centres=coords)[0]
    axes = [off+(np.arange(n)+.5)*dx for off, n in zip(offsets, counts)]
    _assert_materials(mats, _world_inside(axes))


def test_nonuniform_dual_average_uses_the_same_cell_centres():
    from rfx.runners.nonuniform import assemble_interface_eps_nu
    sim = _sim(dz_profile=np.full(50, DX), interface_eps="dual_average")
    grid = sim._build_nonuniform_grid()
    mats = sim._assemble_materials_nu(grid)[0]
    centres = [(np.arange(n)-PAD+.5)*DX for n in grid.shape]
    expected = np.where(_world_inside(centres), 3.25, 1.)
    for got, want in zip(assemble_interface_eps_nu(sim, grid, mats), _edge_means(expected)):
        np.testing.assert_array_equal(got, want)


def test_parameters_are_immutable_and_bounds_enclose_all_eight_vertices():
    center, size, rotation = np.array([1., 2., 3.]), np.array(SIZE), Q.copy()
    shape = OrientedBox(center, size, rotation)
    center[:] = 0
    size[:] = 0
    rotation[:] = 0
    assert shape.center == (1., 2., 3.) and shape.size == SIZE
    with pytest.raises(FrozenInstanceError):
        shape.size = (1., 2., 3.)
    vertices = np.array([np.asarray(shape.center) + Q @ (np.array(sign)*np.array(SIZE)/2)
                         for sign in product((-1, 1), repeat=3)])
    np.testing.assert_allclose(shape.bounding_box(), (vertices.min(0), vertices.max(0)),
                               rtol=0, atol=4*np.finfo(float).eps)


@pytest.mark.parametrize("field,value", [
    ("size", (1., 0., 1.)), ("size", (-1., 1., 1.)),
    ("size", (1., np.inf, 1.)), ("center", (0., np.nan, 0.)),
    ("center", (1., 2.)), ("rotation", np.eye(2)),
    ("rotation", np.diag([1., 1., 2.])), ("rotation", np.diag([1., 1., -1.])),
    ("rotation", np.eye(3).astype(complex)),
])
def test_invalid_geometry_is_refused(field, value):
    values = dict(center=(0., 0., 0.), size=(1., 1., 1.), rotation=np.eye(3))
    values[field] = value
    with pytest.raises(ValueError, match="OrientedBox"):
        OrientedBox(**values)


def test_traced_geometry_or_coordinates_are_refused():
    with pytest.raises(NotImplementedError, match="concrete geometry"):
        jax.jit(lambda center: OrientedBox(center, SIZE).mask_on_coords([0.], [0.], [0.]))(jnp.ones(3))
    with pytest.raises(NotImplementedError, match="concrete mesh"):
        jax.jit(lambda x: _shape().mask_on_coords(x, x, x))(jnp.ones(3))


def test_existing_pec_volume_rule_uses_centres_and_incident_edges():
    from rfx.boundaries.pec import realized_pec_edge_masks
    sim = Simulation(freq_max=6e9, domain=DOMAIN, dx=DX, cpml_layers=PAD)
    sim.add(_shape(), material="pec")
    grid = sim._build_grid()
    centres = [(np.arange(n)-p+.5)*DX for n, p in zip(grid.shape, grid.axis_pads)]
    expected = _world_inside(centres)
    mats, _, _, pec, *_ = sim._assemble_materials(grid)
    np.testing.assert_array_equal(pec, expected)
    np.testing.assert_array_equal(mats.sigma, np.zeros(grid.shape))
    for actual, count in zip(realized_pec_edge_masks(pec, periodic=(False,)*3), _edge_means(expected)):
        np.testing.assert_array_equal(actual, count > 0)


@pytest.mark.parametrize("operator", ["smooth", "inverse", "conformal", "legacy_conformal"])
def test_unsupported_special_operator_never_falls_back(operator):
    from rfx.geometry.smoothing import compute_smoothed_eps, compute_inv_eps_tensor_diag
    from rfx.geometry.conformal import compute_conformal_weights, compute_conformal_weights_sdf
    grid = _sim()._build_grid()
    calls = {
        "smooth": lambda: compute_smoothed_eps(grid, [(_shape(), 3.25)]),
        "inverse": lambda: compute_inv_eps_tensor_diag(grid, pec_shapes=[_shape()]),
        "conformal": lambda: compute_conformal_weights_sdf(grid, [_shape()]),
        "legacy_conformal": lambda: compute_conformal_weights(grid, [_shape()]),
    }
    with pytest.raises(NotImplementedError, match="OrientedBox"):
        calls[operator]()


@pytest.mark.parametrize("sigma,f0", [(5.8e7, None), (100., None), (5.8e7, 3e9)])
def test_sheet_conversion_refuses_every_thin_conductor_mode(sigma, f0):
    with pytest.raises(NotImplementedError, match="OrientedBox"):
        _sim().add_thin_conductor(_shape(), sigma_bulk=sigma, surface_impedance_f0=f0)


def test_geometric_conductor_continuation_is_explicitly_refused():
    sim = Simulation(freq_max=6e9, domain=DOMAIN, dx=DX, cpml_layers=PAD)
    sim.add(OrientedBox((0., .045, .05), SIZE, Q), material="pec")
    with pytest.raises(NotImplementedError, match="continuation"):
        sim._assemble_materials(sim._build_grid())


@pytest.mark.parametrize("operator,graded", [("smooth", False), ("smooth", True),
                                            ("inverse", False), ("conformal", False)])
def test_public_runner_refuses_unimplemented_geometry_operator_before_stepping(operator, graded):
    sim = _sim(**({"dz_profile": np.full(50, DX)} if graded else {}))
    if operator == "conformal":
        sim = Simulation(freq_max=6e9, domain=DOMAIN, dx=DX, cpml_layers=PAD)
        sim.add(_shape(), material="pec")
    options = ({"conformal_pec": True} if operator == "conformal" else
               {"subpixel_smoothing": "kottke_pec" if operator == "inverse" else True})
    with pytest.raises(NotImplementedError, match="OrientedBox"):
        sim.run(n_steps=1, skip_preflight=True, **options)
