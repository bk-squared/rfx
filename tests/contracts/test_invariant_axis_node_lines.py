"""Port and geometry diagnostics on the single node of an invariant axis."""
import warnings

import jax
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.geometry.port_termination import lattice_intersects_aperture
from rfx.geometry.rasterize_grid import coords_from_uniform_grid
from rfx.geometry.smoothing import refuse_pmc_smoothing
from rfx.preflight.line_stub import _intervals


@pytest.fixture(autouse=True)
def cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


@pytest.mark.parametrize("mode,entry,faces", [
    ("2d_tez", "run", "x-lo, x-hi"),
    ("2d_tez", "forward", "x-lo, x-hi"),
    ("2d_tmz", "run", None),
    ("2d_tmz", "forward", None),
    ("3d", "forward", "x-lo, x-hi, z-lo, z-hi"),
])
def test_port_termination(mode, entry, faces):
    domain_z, box_lo, box_hi, port_z = (
        (0.006, 0.0, 0.006, 0.003) if mode == "3d" else (0.001, -0.001, 0.002, 0.0))
    sim = Simulation(freq_max=10e9, domain=(0.060, 0.024, domain_z), dx=0.001,
                     mode=mode, boundary="cpml", cpml_layers=16)
    sim.add(Box((0.0, 0.010, box_lo), (0.060, 0.011, box_hi)), material="pec")
    if mode == "2d_tmz":
        component = "ez"
        sim.add_port((0.030, 0.012, port_z), component, impedance=50.0)
    else:
        component = "ey"
        sim.add(Box((0.0, 0.013, box_lo), (0.060, 0.014, box_hi)), material="pec")
        sim.add_port((0.030, 0.011, port_z), component, impedance=50.0, extent=0.002)
    sim.add_probe((0.045, 0.012, port_z), component)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(sim, entry)(n_steps=2)
    verdicts = [line for warning in caught for line in str(warning.message).splitlines()
                if "names no terminated conductor" in line]
    if faces is None:
        assert verdicts == []
    else:
        assert len(verdicts) == 1
        assert (f"names no terminated conductor; 'pec' (entry 0) under its terminal "
                f"continues into the {faces} absorber; 'pec' (entry 1) under its terminal "
                f"continues into the {faces} absorber;") in verdicts[0]
        if mode == "2d_tez":
            assert "z-lo" not in verdicts[0] and "z-hi" not in verdicts[0]


@pytest.mark.parametrize("cell_axis", [True, False])
@pytest.mark.parametrize("node,lo,hi,expected", [
    (0., -1., 0., True), (0., 1e-30, 1., False),
    (1., 1.0000000000000018, 2., True), (1., 1.000000000000002, 2., False),
])
def test_aperture_on_single_node(cell_axis, node, lo, hi, expected):
    nodes = (np.array([0., 1.]), np.array([0., 1.]), np.array([node]))
    lattice = [(np.ones((2, 2, 1), dtype=bool), (False, False, cell_axis))]
    assert lattice_intersects_aperture(lattice, nodes, (0., 0., lo), (0., 0., hi)) is expected


def test_stub_intervals_on_single_node():
    sim = Simulation(freq_max=1e9, domain=(2., 2., 0.125), dx=0.125,
                     mode="2d_tmz", boundary="pec")
    shape = Box((0.25, 0.875, -0.125), (1.5, 1.125, 0.125))
    sim.add(shape, material="pec")
    grid = sim._build_grid()
    coords = coords_from_uniform_grid(grid)
    assert _intervals(sim, grid, shape, coords, 0, (0., 1., 0.), (0., 1., 0.), {}) == [
        (0.25, 1.5, 0.25, 1.5)]
    assert _intervals(sim, grid, shape, coords, 0, (0., 1., 0.01), (0., 1., 0.02), {}) == []


def test_wire_port_center_on_single_node():
    sim = Simulation(freq_max=10e9, domain=(0.060, 0.024, 0.001), dx=0.001,
                     mode="2d_tmz", boundary="pec")
    sim.add_port((0.030, 0.012, 0.0), "ez", impedance=50., extent=0.001)
    assert sim._wire_port_cell_centers(sim._ports[0]) == ([(0.030, 0.012, 0.0)], 0)


def test_corner_offsets_on_single_node():
    sim = Simulation(freq_max=1e9, domain=(2., 2., 0.125), dx=0.125,
                     mode="2d_tez", boundary="pec")
    assert sim._campaign_ctx().sub_lattice_offsets((0.1875, 0.28125, -0.125)) == [0.5, 0.25, 0.0]


def test_pmc_smoothing_on_single_node():
    sim = Simulation(freq_max=10e9, domain=(0.060, 0.024, 0.001), dx=0.001,
                     mode="2d_tez", boundary=BoundarySpec(x="pec", y="pec", z="pmc"))
    sim.add_material("substrate", eps_r=4)
    sim.add(Box((0.010, 0.010, 0.), (0.020, 0.020, 0.001)), material="substrate")
    assert refuse_pmc_smoothing(sim, sim._build_grid()) is None
