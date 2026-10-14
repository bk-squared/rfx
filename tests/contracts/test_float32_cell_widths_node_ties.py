"""Declared node faces survive float32-valued cell widths (DESIGN E1-E5)."""

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.nonuniform import position_to_index


U = 1.524e-3


def _model(q, width, lane):
    h = U / q
    w = float(np.float32(h)) if width == "w32" else h
    profile = np.full(30 * q, w)
    domain = (30 * U, 30 * U, 10 * U)
    kwargs = {}
    if lane == "graded":
        domain = (float(profile.sum()), float(profile.sum()), 10 * U)
        kwargs.update(dx_profile=profile, dy_profile=profile)
    sim = Simulation(freq_max=4e9, domain=domain, dx=w,
                     boundary="cpml", cpml_layers=2, **kwargs)
    return sim, h


@pytest.fixture(params=[(q, width, lane) for q in (1, 3, 4, 6)
                       for width in ("w32", "w64") for lane in ("uniform", "graded")],
                ids=lambda p: f"q{p[0]}-{p[1]}-{p[2]}")
def scene(request):
    q, width, lane = request.param
    sim, h = _model(q, width, lane)
    return sim, h, q, lane


def _entity(sim):
    record = sim.realized_geometry()
    assert not record.refused and not record.refused_thin_conductors
    entity, = record.entities
    assert entity.occupancy_role == "solved"
    assert entity.error is None and entity.mask_error is None
    return entity


def test_e1_material_faces(scene):
    sim, h, _, _ = scene
    lo, hi = (13 * U, 13 * U, 2 * U), (17 * U, 17 * U, 3 * U)
    sim.add_material("substrate", eps_r=3.48)
    sim.add(Box(lo, hi), material="substrate")
    entity = _entity(sim)
    for axis, lower, upper in zip(entity.axes, lo, hi):
        np.testing.assert_allclose(axis.bounds_m, (lower, upper), rtol=0, atol=1e-3 * h)


@pytest.mark.parametrize("route", ["pec", "thin"])
def test_e2_sheet_plane(scene, route):
    sim, h, q, _ = scene
    shape = Box((13 * U, 13 * U, 3 * U), (17 * U, 17 * U, 3 * U))
    if route == "pec":
        sim.add(shape, material="pec")
    else:
        sim.add_thin_conductor(shape)
    entity = _entity(sim)
    assert entity.plane is not None
    normal, index, coordinate = entity.plane
    assert normal == 2
    assert index == 3 * q + 2
    assert abs(coordinate - 3 * U) <= 1e-3 * h


def test_e3_half_cell_point_uses_lower_node(scene):
    sim, h, q, lane = scene
    grid = sim._build_realized_grid()
    point = ((13 * q + .5) * h, 13 * U, 3 * U)
    index = position_to_index(grid, point) if lane == "graded" else grid.position_to_index(point)
    assert index[0] == 13 * q + 2


@pytest.mark.parametrize("q", [1, 3, 4, 6])
@pytest.mark.parametrize("width", ["w32", "w64"])
def test_e4_declared_domain_cell_count(q, width):
    sim, _ = _model(q, width, "uniform")
    grid = sim._build_grid()
    assert grid.nx - grid.pad_x_lo - grid.pad_x_hi - 1 == 30 * q


@pytest.mark.parametrize("route", ["volume", "thin"])
def test_e5_one_cell_box_is_admitted_and_has_one_row(scene, route):
    sim, h, q, _ = scene
    # Two U in z, so x stays the one thinnest axis at q = 1 (1.524 mm cells).
    shape = Box((13 * U, 13 * U, 2 * U), (13 * U + h, 17 * U, 4 * U))
    if route == "volume":
        sim.add(shape, material="pec")
    else:
        sim.add_thin_conductor(shape)
    entity = _entity(sim)
    rows = np.flatnonzero(np.any(entity.mask, axis=(1, 2)))
    np.testing.assert_array_equal(rows, [13 * q + 2])
