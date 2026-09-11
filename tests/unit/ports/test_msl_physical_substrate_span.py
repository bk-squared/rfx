"""An MSL source/termination must use the actual ground-to-trace edge span.

The physical bounds come from #931's realized conductor walls, independently
of the source profile's own dimensions. No FDTD solve or fitted RF tolerance.
"""
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes
from rfx.sources.msl_port import (
    MSLPort,
    _msl_yz_cells,
    compute_msl_mode_profile,
    msl_axis_roles,
    msl_cross_section_span,
    msl_physical_point,
    setup_msl_port,
)


def _board(direction, n_sub, ground_cells):
    dx = 254e-6 / n_sub
    lo, hi = ground_cells * dx, (ground_cells + n_sub) * dx
    sim = Simulation(freq_max=5e9, domain=(32*dx, 32*dx, 24*dx),
                     dx=dx, cpml_layers=0, boundary="pec")
    sim.add_material("substrate", eps_r=3.66)
    sim.add(Box((0, 0, lo), (32*dx, 32*dx, hi)), material="substrate")
    sim.add(Box((0, 0, lo), (32*dx, 32*dx, lo)), material="pec")
    trace_lo = msl_physical_point(direction, 0, 8*dx, hi)
    trace_hi = msl_physical_point(direction, 32*dx, 16*dx, hi)
    sim.add(Box(trace_lo, trace_hi), material="pec")
    grid = sim._build_grid()
    sheets, wires = [], []
    materials, _, _, mask, _, _, _ = sim._assemble_materials(
        grid, pec_sheets=sheets, pec_wires=wires)
    edges = realized_pec_edge_masks(mask, sheets=tuple(sheets), wires=tuple(wires))
    port = MSLPort(feed_x=4*dx, y_lo=8*dx, y_hi=16*dx, z_lo=lo, z_hi=hi,
                   direction=direction, impedance=50., excitation=None)
    centre = grid.position_to_index(msl_physical_point(direction, 4*dx, 12*dx, lo))
    planes = realized_wall_planes(edges, 2, ij=centre[:2])
    assert list(planes) == [ground_cells, ground_cells+n_sub]
    return grid, port, materials, centre, tuple(map(int, planes))


@pytest.mark.parametrize("direction,n_sub,ground_cells", [
    ("+x", 3, 0), ("+x", 4, 8), ("-x", 6, 5),
    ("+y", 4, 8), ("-y", 4, 8),
])
def test_laplace_source_uses_wall_span_for_voltage_and_load(direction, n_sub, ground_cells):
    grid, port, materials, centre, (lower, upper) = _board(direction, n_sub, ground_cells)
    profile = compute_msl_mode_profile(grid, port, 3.66)
    assert profile["n_z_sub"] == upper-lower
    assert {c[2] for c in profile["cell_indices"]} == set(range(lower, upper))
    assert (upper-lower)*grid.dx == pytest.approx(port.z_hi-port.z_lo)
    width_axis = {"x": 0, "y": 1}[msl_axis_roles(direction)[1]]
    iw = centre[width_axis]-profile["j_grid_lo"]
    ez = np.asarray(profile["ez_profile"])
    # Use the wall-derived voltage interval, not profile['n_z_sub'].
    voltage = float(np.sum(ez[iw, :upper-lower])*grid.dx)
    assert voltage == pytest.approx(1., rel=1e-12)
    loaded = setup_msl_port(grid, port, materials, mode_profile=profile)
    sigma = np.asarray(loaded.sigma)-np.asarray(materials.sigma)
    power = sum(
        sigma[cell] * ez[cell[width_axis]-profile["j_grid_lo"], cell[2]-lower]**2
        * grid.dx**3 for cell in profile["cell_indices"])
    # Existing physical load identity; float32 sigma limits arithmetic precision.
    assert voltage**2/power == pytest.approx(port.impedance, rel=1e-6)
    assert not np.any(sigma[:, :, :lower])
    assert not np.any(sigma[:, :, upper:])


def test_uniform_source_cells_and_span_metadata_distinguish_edges_from_nodes():
    grid, port, materials, _, (lower, upper) = _board("+x", 4, 8)
    cells = _msl_yz_cells(grid, port)
    assert {c[2] for c in cells} == set(range(lower, upper))
    span = msl_cross_section_span(grid, port)
    # Extraction uses n_hi as the trace plane, an exclusive edge bound.
    assert (span["n_lo"], span["n_hi"]) == (lower, upper)
    loaded = setup_msl_port(grid, port, materials)
    sigma = np.asarray(loaded.sigma)-np.asarray(materials.sigma)
    assert not np.any(sigma[:, :, upper:])
