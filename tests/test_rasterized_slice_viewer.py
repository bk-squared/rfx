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
from rfx.visualize import (_axis_edges, plot_rasterized_slice,
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


def test_edges_are_cell_lower_edges_not_node_midpoints():
    """The convention: node[k] is the LOWER EDGE of cell k.

    Verified against the grid's own spacing arrays — node[k+1] - node[k] ==
    dz[k] on a graded axis. Building edges as node MIDPOINTS instead shifts
    every drawn cell by half a cell and distorts widths where the grading
    changes, which is the same node-vs-cell confusion that puts `position=`
    one plane above a one-cell sheet.
    """
    nz = 20
    prof = np.full(nz, 100e-6)
    prof[10:] = 250e-6
    sim = _board(dz_profile=prof)
    grid = sim._build_nonuniform_grid()
    from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid
    nodes = np.asarray(coords_from_nonuniform_grid(grid).z, dtype=float)
    dz = np.asarray(grid.dz, dtype=float)

    # the premise this test rests on
    assert np.allclose(np.diff(nodes), dz[:nodes.size - 1], rtol=1e-5), (
        "the grid no longer places node[k] at the lower edge of cell k; the "
        "edge construction below must be revisited, not just re-pinned")

    e = _axis_edges(grid, 2, nodes)
    assert e.size == nodes.size + 1
    assert np.allclose(e[:-1], nodes, rtol=1e-5), "edges are not the nodes"
    assert np.allclose(np.diff(e)[:nodes.size - 1], dz[:nodes.size - 1],
                       rtol=1e-5), "cell widths do not match the spacing array"
    # a midpoint construction would fail the above; show it explicitly
    mid = 0.5 * (nodes[:-1] + nodes[1:])
    assert not np.allclose(mid[:5], nodes[:5], rtol=1e-3), (
        "this fixture has no grading, so midpoints and nodes coincide and the "
        "test cannot tell the two constructions apart")


def test_uniform_lane_edges_match_the_scalar_cell_size():
    sim = _board()
    grid = sim._build_grid()
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid
    nodes = np.asarray(coords_from_uniform_grid(grid).x, dtype=float)
    e = _axis_edges(grid, 0, nodes)
    assert e.size == nodes.size + 1
    assert np.allclose(np.diff(e), float(grid.dx), rtol=1e-5)


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


# --- regressions for the defects an adversarial review found ---------------

def test_thin_conductor_bodies_are_outlined_not_silently_omitted():
    """add_thin_conductor() bodies live in a DIFFERENT registry.

    They are appended to sim._thin_conductors, not sim._geometry, so a viewer
    that walks only the latter drew a board of red cells with no outline and a
    title asserting nothing was omitted — on a PCB, that is every trace.
    """
    sim = _board(f0=10e9)
    assert len(sim._thin_conductors) == 1, "fixture no longer exercises the path"
    from rfx.visualize import _declared_entries
    assert len(_declared_entries(sim)) == len(sim._geometry) + 1
    fig = plot_rasterized_slice(sim)
    ax = fig.axes[0]
    assert len(ax.patches) >= 1, (
        "the thin conductor was neither outlined nor reported as skipped")


def test_a_uniform_slice_does_not_invent_a_permittivity_range():
    """matplotlib's `nonsingular` turns a degenerate range into +/-10%.

    An all-vacuum slice drew a colorbar running 0.900 to 1.100, i.e. eps_r < 1
    printed on the axis of an air plane.
    """
    sim = _board()
    grid = sim._build_grid()
    eps = np.asarray(sim._assemble_materials(grid)[0].eps_r, dtype=float)
    k = next(i for i in range(eps.shape[2])
             if np.isclose(eps[:, :, i].min(), eps[:, :, i].max()))
    fig = plot_rasterized_slice(sim, axis=2, index=k)
    labels = [t.get_text() for t in fig.axes[-1].get_yticklabels()]
    assert any("uniform" in t for t in labels), (
        f"a uniform slice must name its single value; got {labels}")


def test_position_outside_the_axis_raises_instead_of_clamping():
    sim = _board()
    with pytest.raises(ValueError, match="outside axis"):
        plot_rasterized_slice(sim, axis=2, position=50.0)


def test_a_relocated_plane_says_so_in_the_title():
    """Silently answering about a different plane turns 'is this plane clear
    of metal?' into a picture of the ground plane."""
    sim = _board()
    _, cond, _ = _assembled(sim)
    per_plane = cond.reshape(-1, cond.shape[2]).sum(axis=0)
    empty = [k for k in range(cond.shape[2]) if per_plane[k] == 0
             and any(per_plane[max(k - 1, 0):k + 2])]
    if not empty:
        pytest.skip("this fixture has no empty plane adjacent to a sheet")
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid
    z = np.asarray(coords_from_uniform_grid(sim._build_grid()).z, dtype=float)
    fig = plot_rasterized_slice(sim, axis=2, position=float(z[empty[0]]))
    assert "showing the neighbouring plane" in fig.axes[0].get_title()


def test_stack_profile_refuses_a_model_with_nothing_to_read():
    """No conductor and no permittivity structure: argmax on an all-zero array
    returns 0, i.e. the CPML corner, and the figure comes back blank."""
    sim = Simulation(freq_max=20e9, domain=DOM, dx=DX, cpml_layers=4,
                     boundary="cpml")
    with pytest.raises(ValueError, match="no column worth reading"):
        plot_stack_profile(sim)


def test_stack_shading_is_absolute_so_two_columns_compare():
    """Per-column normalisation made a solid eps_r=9 column pixel-identical to
    a vacuum one, while the docstring promises the tint carries permittivity."""
    import inspect
    from rfx.visualize import plot_stack_profile as f
    src = inspect.getsource(f)
    assert "np.ptp(eps_col)" not in src, (
        "the column shading is normalised per column again")
    assert "eps_hi" in src, "no absolute permittivity anchor in the shading"
