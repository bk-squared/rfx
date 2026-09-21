"""Build-only checks of the conducting geometry at absorbing faces."""
from types import SimpleNamespace

import numpy as np
import pytest

from rfx import Box, Cylinder, Simulation
from rfx.geometry.smoothing import continued_conductor_shape


def _sim(domain=(8., 8., 8.)):
    return Simulation(domain=domain, dx=1., freq_max=1e6,
                      boundary="cpml", cpml_layers=2)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("side", (0, 1))
@pytest.mark.parametrize("kind", ("volume", "sheet", "thin"))
def test_reached_face_is_filled(axis, side, kind):
    sim = _sim()
    lo, hi = [2., 2., 2.], [6., 6., 6.]
    (lo if side == 0 else hi)[axis] = 0. if side == 0 else 8.
    if kind != "volume":
        normal = (axis + 1) % 3
        lo[normal] = hi[normal] = 4.
    declared = Box(tuple(lo), tuple(hi))
    if kind == "thin":
        sim.add_thin_conductor(declared, sigma_bulk=5.8e7, thickness=0.01)
    else:
        sim.add(declared, material="pec")
    grid = sim._build_grid()
    sheets = []
    _, _, _, cells, shapes, _, _ = sim._assemble_materials(
        grid, pec_sheets=sheets, pec_wires=[])
    actual = np.asarray(cells if kind == "volume" else sheets[0].footprint)
    # Read solved cells/footprints, independent of the continuation helper.
    face = 2 if side == 0 else grid.shape[axis] - 3 - (kind == "volume")
    pad = range(2) if side == 0 else range(grid.shape[axis]-3, grid.shape[axis]-1)
    reference = np.take(actual, face, axis=axis)
    assert reference.any()
    for layer in pad:
        np.testing.assert_array_equal(np.take(actual, layer, axis=axis), reference)
    if kind != "volume":
        assert shapes[0].corner_lo[normal] == shapes[0].corner_hi[normal] == 4.
    assert declared.corner_lo == tuple(lo) and declared.corner_hi == tuple(hi)


def test_corner_pad_zero_and_fractional_declared_face():
    sim = _sim((535.43, 8., 8.))
    grid = sim._build_grid()
    grid.pad_z_lo = 0
    shape = Box((0., 0., 0.), (535.43, 4., 4.))
    solved = continued_conductor_shape(sim, grid, shape)
    assert solved.corner_lo[0] < -2. and solved.corner_lo[1] < -2.
    assert solved.corner_hi[0] > 538.
    assert solved.corner_lo[2] == 0.


def test_cylinder_axis_continues_and_cross_axis_is_reported():
    sim = _sim()
    grid = sim._build_grid()
    shape = Cylinder((4., 1., 4.), radius=1., height=8., axis="x")
    findings = []
    solved = continued_conductor_shape(sim, grid, shape, unextendable=findings)
    assert solved.height > 8.
    assert solved.radius == 1.
    assert [(u.axis, u.side, u.conductor) for u in findings] == [(1, "lo", True)]


@pytest.mark.parametrize("direction", ("+x", "-x", "+y", "-y"))
def test_port_signal_stays_declared_and_ground_continues(direction):
    sim = _sim()
    sim._msl_ports = [SimpleNamespace(position=(4., 4., 2.), width=2.,
                                     height=1., direction=direction)]
    grid = sim._build_grid()
    signal = Box((0., 0., 3.), (8., 8., 3.))
    ground = Box((0., 0., 2.), (8., 8., 2.))
    strip = continued_conductor_shape(sim, grid, signal)
    plane = continued_conductor_shape(sim, grid, ground)
    axis = "xy".index(direction[-1])
    if direction[0] == "+":
        assert strip.corner_lo[axis] == 0.
        assert plane.corner_lo[axis] < -2.
    else:
        assert strip.corner_hi[axis] == 8.
        assert plane.corner_hi[axis] > 10.
    assert strip.corner_lo[2] == strip.corner_hi[2] == 3.
    assert plane.corner_lo[2] == plane.corner_hi[2] == 2.


@pytest.mark.parametrize("axis", range(3))
def test_mirror_cells(axis):
    def build(lo, hi):
        sim = _sim()
        sim.add(Box(tuple(lo), tuple(hi)), material="pec")
        grid = sim._build_grid()
        mask = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[3]
        return np.asarray(mask)[:-1, :-1, :-1]
    lo, hi = [2., 2., 2.], [6., 6., 6.]
    lo[axis] = 0.
    actual = build(lo, hi)
    reflected_lo, reflected_hi = lo.copy(), hi.copy()
    reflected_lo[axis], reflected_hi[axis] = 8-hi[axis], 8-lo[axis]
    np.testing.assert_array_equal(np.flip(actual, axis), build(reflected_lo, reflected_hi))
    assert np.take(actual, 0, axis=axis).any()
