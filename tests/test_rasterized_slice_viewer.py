"""The rasterization viewer must show what the SOLVE builds, not what was asked for.

Each test here pins one way the previous answer to "show me my geometry"
(``plot_geometry_2d_slice``) is silent about the thing that actually goes
wrong on an RF board.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.visualize import (_edges_from_nodes, plot_rasterized_slice,
                           plot_stack_profile)

DX = 100e-6
H_SUB = 400e-6           # 4 cells
DOM = (3e-3, 3e-3, 2e-3)


def _board(*, f0=None, dz_profile=None):
    """Substrate + a ONE-CELL metal sheet on top of it."""
    sim = Simulation(freq_max=20e9, domain=DOM, dx=DX, cpml_layers=4,
                     boundary="cpml", dz_profile=dz_profile)
    sim.add_material("sub", eps_r=4.0, sigma=0.0)
    sim.add(Box((0, 0, 0), (DOM[0], DOM[1], H_SUB)), material="sub")
    trace = Box((0.5e-3, 1.0e-3, H_SUB), (2.5e-3, 2.0e-3, H_SUB + DX))
    if f0 is None:
        sim.add(trace, material="pec")
    else:
        sim.add_thin_conductor(trace, sigma_bulk=5.8e7, thickness=35e-6,
                               surface_impedance_f0=f0)
    return sim


def _assembled(sim):
    from rfx.nonuniform import NonUniformGrid
    is_nu = sim._dz_profile is not None
    grid = sim._build_nonuniform_grid() if is_nu else sim._build_grid()
    cond = np.asarray(sim.conductor_mask(grid), dtype=bool)
    eps = np.asarray(
        (sim._assemble_materials_nu(grid) if isinstance(grid, NonUniformGrid)
         else sim._assemble_materials(grid))[0].eps_r, dtype=float)
    return grid, cond, eps


def test_one_cell_metal_is_invisible_in_permittivity_but_present_in_the_mask():
    """The reason an eps_r plot cannot answer 'where is my metal'.

    A one-cell PEC sheet contributes no permittivity contrast of its own, so
    on an eps_r slice its cells are indistinguishable from the air (or
    dielectric) around them. The conductor mask is the only place it exists.
    """
    sim = _board()
    _, cond, eps = _assembled(sim)
    assert cond.any(), "the sheet produced no conductor cells at all"
    k = int(np.argmax(cond.reshape(-1, cond.shape[2]).sum(axis=0)))
    metal = cond[:, :, k]
    assert metal.any()
    inside = eps[:, :, k][metal]
    outside = eps[:, :, k][~metal]
    assert np.isclose(inside.min(), inside.max()), "sheet cells not uniform in eps"
    assert np.isclose(float(inside[0]), float(np.median(outside))), (
        "this fixture is supposed to make the metal INVISIBLE in eps_r; if the "
        "sheet cell now carries its own permittivity the premise moved")


def test_surface_impedance_sheet_is_in_neither_pec_mask_nor_sigma():
    """#677 made the f0 sheet a node-thin operator.

    A viewer that draws ``pec_mask | (sigma > thr)`` shows NOTHING for a board
    whose traces are all surface-impedance sheets. conductor_mask() is the
    spelling that covers it, and the viewer must use that one.
    """
    sim = _board(f0=10e9)
    grid = sim._build_grid()
    specs: list = []
    mats, _, _, pec, *_ = sim._assemble_materials(grid, sheet_specs=specs)
    naive = np.asarray(pec, dtype=bool) | (np.asarray(mats.sigma) > 1e3)
    full = np.asarray(sim.conductor_mask(grid), dtype=bool)
    assert full.sum() > 0, "no conductor found at all — fixture broken"
    assert naive.sum() < full.sum(), (
        "the naive spelling found as much as conductor_mask(); this test can "
        "no longer prove the viewer needs the accessor")
    fig = plot_rasterized_slice(sim)
    assert "conductor cells" in fig.axes[0].get_title()
    assert int(fig.axes[0].get_title().split("—")[1].split()[0]) > 0


def test_edges_follow_a_graded_axis_instead_of_one_uniform_extent():
    """imshow(extent=[0, n*dx]) redraws a graded mesh as a uniform one."""
    nodes = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0])
    e = _edges_from_nodes(nodes)
    assert e.size == nodes.size + 1
    widths = np.diff(e)
    assert widths[1] < widths[-2], "graded widths were flattened"
    # every node sits inside its own cell
    assert np.all((e[:-1] <= nodes) & (nodes <= e[1:]))
    uniform = _edges_from_nodes(np.arange(5) * 3.0)
    assert np.allclose(np.diff(uniform), 3.0)


def test_position_lands_on_the_plane_that_holds_the_sheet():
    """A one-cell sheet's mask sits on the LOWER node of its cell.

    Asking for the sheet's geometric centre and taking the nearest node lands
    one plane high and draws whatever else is there — the exact mistake that
    put a via column under the label 'a driven patch' while this was being
    developed.
    """
    sim = _board()
    _, cond, _ = _assembled(sim)
    per_plane = cond.reshape(-1, cond.shape[2]).sum(axis=0)
    k_true = int(np.argmax(per_plane))
    centre = H_SUB + 0.5 * DX
    fig = plot_rasterized_slice(sim, axis=2, position=centre)
    shown = int(fig.axes[0].get_title().split("—")[1].split()[0])
    assert shown == int(per_plane[k_true]), (
        f"asked for the sheet centre and got {shown} conductor cells; the "
        f"plane holding the sheet has {int(per_plane[k_true])}")


def test_index_and_position_are_mutually_exclusive_and_axis_is_checked():
    sim = _board()
    with pytest.raises(ValueError, match="not both"):
        plot_rasterized_slice(sim, index=3, position=1e-3)
    with pytest.raises(ValueError, match="axis must be"):
        plot_rasterized_slice(sim, axis=3)
    with pytest.raises(IndexError):
        plot_rasterized_slice(sim, index=10**6)


def test_stack_profile_reports_the_conductor_cells_on_its_column():
    sim = _board()
    fig = plot_stack_profile(sim)
    t = fig.axes[0].get_title()
    assert "conductor cell" in t
    n = int(t.split("bod(y/ies),")[1].split()[0])
    assert n >= 1, f"stack column found no conductor: {t}"


def test_viewer_runs_on_a_graded_z_simulation():
    """The uniform-only path silently draws the wrong grid on a NU sim."""
    nz = int(round(DOM[2] / DX))
    prof = np.full(nz, DX)
    prof[nz // 2:] = DX * 1.5          # grade the upper half
    sim = _board(dz_profile=prof)
    fig = plot_rasterized_slice(sim, axis=1)
    assert fig.axes[0].get_ylabel() == "z (mm)"
    fig2 = plot_stack_profile(sim)
    assert "conductor cell" in fig2.axes[0].get_title()
