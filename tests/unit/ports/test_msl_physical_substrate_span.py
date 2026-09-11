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
    return grid, port, materials, centre, tuple(map(int, planes)), sim


@pytest.mark.parametrize("direction,n_sub,ground_cells", [
    ("+x", 3, 0), ("+x", 4, 8), ("-x", 6, 5),
    ("+y", 4, 8), ("-y", 4, 8),
])
def test_laplace_source_uses_wall_span_for_voltage_and_load(direction, n_sub, ground_cells):
    grid, port, materials, centre, (lower, upper), _ = _board(direction, n_sub, ground_cells)
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
    grid, port, materials, _, (lower, upper), _ = _board("+x", 4, 8)
    cells = _msl_yz_cells(grid, port)
    assert {c[2] for c in cells} == set(range(lower, upper))
    span = msl_cross_section_span(grid, port)
    # Extraction uses n_hi as the trace plane, an exclusive edge bound.
    assert (span["n_lo"], span["n_hi"]) == (lower, upper)
    loaded = setup_msl_port(grid, port, materials)
    sigma = np.asarray(loaded.sigma)-np.asarray(materials.sigma)
    assert not np.any(sigma[:, :, upper:])


def _graded_board(direction, z_ratio):
    """Unequal grading on all axes, with two independently realized sheets."""
    profiles = (
        1e-3*np.array([.2]*4 + [.4]*6 + [.2]*4),
        1e-3*np.array([.15]*4 + [.45]*6 + [.15]*4),
        1e-3*np.array([.1]*2 + [.1*z_ratio]*2 + [.2]*10),
    )
    nodes = tuple(np.r_[0., np.cumsum(p)] for p in profiles)
    prop_axis = 0 if direction.endswith("x") else 1
    width_axis = 1-prop_axis
    lower, upper = 2, 6
    lo, hi = nodes[2][[lower, upper]]
    domain = tuple(float(n[-1]) for n in nodes)
    sim = Simulation(freq_max=5e9, domain=domain, dx=profiles[0][0],
                     dx_profile=profiles[0], dy_profile=profiles[1],
                     dz_profile=profiles[2], cpml_layers=0, boundary="pec")
    sim.add_material("substrate", eps_r=3.66)
    sim.add(Box((0, 0, lo), (domain[0], domain[1], hi)), material="substrate")
    sim.add(Box((0, 0, lo), (domain[0], domain[1], lo)), material="pec")
    trace_lo = msl_physical_point(direction, 0, nodes[width_axis][4], hi)
    trace_hi = msl_physical_point(direction, domain[prop_axis], nodes[width_axis][10], hi)
    sim.add(Box(trace_lo, trace_hi), material="pec")
    grid = sim._build_nonuniform_grid()
    sheets, wires = [], []
    materials, _, _, mask = sim._assemble_materials_nu(
        grid, pec_sheets=sheets, pec_wires=wires)
    edges = realized_pec_edge_masks(mask, sheets=tuple(sheets), wires=tuple(wires))
    centre = (5, 7, lower) if prop_axis == 0 else (7, 5, lower)
    planes = realized_wall_planes(edges, 2, ij=centre[:2])
    assert list(planes) == [lower, upper]
    port = MSLPort(feed_x=nodes[prop_axis][5], y_lo=nodes[width_axis][4],
                   y_hi=nodes[width_axis][10], z_lo=lo, z_hi=hi,
                   direction=direction, impedance=50., excitation=None)
    return grid, port, materials, centre, tuple(map(int, planes))


@pytest.mark.parametrize("direction,z_ratio", [
    ("+x", 2.), ("-x", 4.), ("+y", 2.), ("-y", 4.),
])
def test_graded_source_voltage_and_load_use_actual_walls(direction, z_ratio):
    """The upper integration bound comes from metal, not profile dimensions.

    Read Joule volumes from the E-update metrics independently of the
    source's dual-spacing helper. This checks the modal load identity;
    it does not claim that the static profile is an exact NU eigenmode.
    """
    grid, port, materials, centre, (lower, upper) = _graded_board(direction, z_ratio)
    profile = compute_msl_mode_profile(grid, port, 3.66)
    assert profile["n_z_sub"] == upper-lower
    assert {c[2] for c in profile["cell_indices"]} == set(range(lower, upper))
    width_axis = 1 if direction.endswith("x") else 0
    ez = np.asarray(profile["ez_profile"])
    row = centre[width_axis]-profile["j_grid_lo"]
    dz = np.asarray(grid.dz, dtype=float)
    voltage = float(np.sum(ez[row, :upper-lower]*dz[lower:upper]))
    assert voltage == pytest.approx(1., rel=1e-12)
    loaded = setup_msl_port(grid, port, materials, mode_profile=profile)
    sigma = np.asarray(loaded.sigma)-np.asarray(materials.sigma)
    inv_x, inv_y = np.asarray(grid.inv_dx), np.asarray(grid.inv_dy)
    power = 0.
    for cell in profile["cell_indices"]:
        field = ez[cell[width_axis]-profile["j_grid_lo"], cell[2]-lower]
        volume = dz[cell[2]]/(float(inv_x[cell[0]])*float(inv_y[cell[1]]))
        power += float(sigma[cell])*field**2*volume
    assert voltage**2/power == pytest.approx(port.impedance, rel=1e-6)
    assert not np.any(sigma[:, :, :lower])
    assert not np.any(sigma[:, :, upper:])


@pytest.mark.parametrize("direction", ["+x", "-x", "+y", "-y"])
def test_run_and_forward_build_the_same_physical_port(direction, monkeypatch):
    import rfx.sources.msl_port as source_module

    grid, expected, _, _, _, sim = _board(direction, 4, 8)
    pos = msl_physical_point(direction, expected.feed_x,
                              (expected.y_lo+expected.y_hi)/2, expected.z_lo)
    sim.add_msl_port(position=pos, width=expected.y_hi-expected.y_lo,
                     height=expected.z_hi-expected.z_lo, direction=direction,
                     mode="laplace", impedance=50.)
    seen = []

    class Captured(Exception):
        pass

    def capture(grid_arg, port, eps, **kwargs):
        seen.append((port, eps))
        raise Captured

    monkeypatch.setattr(source_module, "compute_msl_mode_profile", capture)
    # Stop at the real mode-setup boundary, before any solve or FDTD scan.
    with pytest.raises(Captured):
        sim.run(n_steps=1, compute_s_params=False, skip_preflight=True)
    with pytest.raises(Captured):
        sim.forward(n_steps=1, skip_preflight=True)
    for port, eps in seen:
        assert port.feed_x == pytest.approx(expected.feed_x)
        assert (port.y_lo, port.y_hi) == pytest.approx((expected.y_lo, expected.y_hi))
        assert (port.z_lo, port.z_hi) == pytest.approx((8*grid.dx, 12*grid.dx))
        assert port.direction == direction
        assert eps == pytest.approx(3.66, rel=1e-6)
