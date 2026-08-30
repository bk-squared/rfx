"""The rasterization viewer must show what the SOLVE builds, not what was asked for.

Each test here pins one way the previous answer to "show me my geometry"
(``plot_geometry_2d_slice``) is silent about the thing that actually goes
wrong on an RF board.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    This premise is then held against the VIEWER itself: an eps_r-only
    reading of the same plane would show nothing, so plot_rasterized_slice's
    conductor overlay must be what makes the metal visible there.
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
    fig = plot_rasterized_slice(sim, axis=2, index=k)
    n_shown = int(fig.axes[0].get_title().split("—")[1].split()[0])
    assert n_shown == int(metal.sum()), (
        f"the viewer's conductor overlay must show the same {int(metal.sum())} "
        f"metal cells the mask has on this eps-invisible plane; got {n_shown} "
        "-- this test is a premise pin AND a viewer regression test")


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

    The CLOSING edge is pinned separately: ``dz``'s trailing entry is the
    #562 node-provider duplicate (rfx/nonuniform.py:64-69), a copy of the
    last real cell's width with no physical extent of its own. Every width
    this test checked used to come from ``np.diff(nodes)`` alone, which
    never touches that duplicate -- so a version of ``_axis_edges`` that
    used the scalar boundary ``dx`` for EVERY closing edge (ignoring the
    per-axis array entirely) passed here unnoticed. The explicit check
    below on ``e[-1]`` closes that gap.
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
    assert np.isclose(e[-1], nodes[-1], rtol=1e-6), (
        f"closing edge {e[-1]:g} must equal the last node {nodes[-1]:g} m -- "
        "the array's trailing entry is a node-provider duplicate with no "
        "physical extent, so nothing should be added past the last node")


def test_uniform_lane_edges_match_the_scalar_cell_size():
    sim = _board()
    grid = sim._build_grid()
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid
    nodes = np.asarray(coords_from_uniform_grid(grid).x, dtype=float)
    e = _axis_edges(grid, 0, nodes)
    assert e.size == nodes.size + 1
    assert np.allclose(np.diff(e), float(grid.dx), rtol=1e-5)
    # Pin the ORIGIN too: a midpoint construction (edges at node +/- dx/2)
    # has the same size and the same constant spacing as the correct one,
    # differing only by a dx/2 offset that the two checks above cannot see.
    assert np.isclose(e[0], nodes[0]), (
        f"the first edge {e[0]:g} must be the first node {nodes[0]:g} "
        "itself, not the node's midpoint")


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
    """The uniform-only path silently draws the wrong grid on a NU sim.

    Pinned against the ACTUAL graded extent, not just a smoke assertion:
    half this fixture's z cells are widened 1.5x, so the real (non-uniform)
    grid's z range is measurably taller than the uniform grid
    ``sim._build_grid()`` would produce for the same nominal cell count --
    a mutant that hardcodes the uniform-only path (``is_nu = False``)
    passed the previous version of this test (it only checked the ylabel
    and a title substring, both true on either grid).
    """
    nz = int(round(DOM[2] / DX))
    prof = np.full(nz, DX)
    prof[nz // 2:] = DX * 1.5          # grade the upper half
    sim = _board(dz_profile=prof)

    grid_nu = sim._build_nonuniform_grid()
    from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid
    z_nu_max = float(np.asarray(
        coords_from_nonuniform_grid(grid_nu).z, dtype=float).max())

    fig = plot_rasterized_slice(sim, axis=1)
    assert fig.axes[0].get_ylabel() == "z (mm)"
    ylim = fig.axes[0].get_ylim()
    assert ylim[1] >= z_nu_max * 1e3 - 1e-6, (
        f"the drawn z-axis tops out at {ylim[1]:.4f} mm but the graded "
        f"grid's real extent is {z_nu_max * 1e3:.4f} mm -- a uniform grid "
        "built from sim._build_grid() (ignoring dz_profile) would be "
        "shorter than this because it does not know about the widened cells")

    fig2 = plot_stack_profile(sim)
    assert "conductor cell" in fig2.axes[0].get_title()


def test_orientation_is_not_transposed_on_a_rectangular_domain():
    """A square in-plane domain makes a `keep` axis reversal dimensionally
    legal and silently wrong -- a transposed picture is still (39, 39) on a
    (39, 39) fixture. Use a RECTANGULAR domain so a transpose changes the
    drawn axis RANGES, not just which pixel holds which value.
    """
    dom = (4e-3, 2e-3, 2e-3)
    sim = Simulation(freq_max=20e9, domain=dom, dx=DX, cpml_layers=4,
                     boundary="cpml")
    sim.add_material("sub", eps_r=4.0, sigma=0.0)
    sim.add(Box((0, 0, 0), (dom[0], dom[1], H_SUB)), material="sub")
    trace = Box((0.5e-3, 0.5e-3, H_SUB), (3.5e-3, 1.0e-3, H_SUB + DX))
    sim.add(trace, material="pec")
    fig = plot_rasterized_slice(sim, axis=2)
    ax = fig.axes[0]
    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    assert x_span > 1.5 * y_span, (
        f"the x-domain (4 mm) must draw roughly twice as wide as the "
        f"y-domain (2 mm); got x_span={x_span:.3f} mm y_span={y_span:.3f} mm "
        "-- a `keep` axis reversal would transpose these ranges")


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


# --- regressions for the SECOND adversarial review pass (#11-24) -----------

def test_two_plane_wall_is_not_shown_and_the_title_says_so():
    """conductor_mask() is a CELL footprint; #706's ``two_plane=True`` opt-in
    only zeroes an extra tangential-E EDGE at the body's far node plane
    during the solve, so there is no cell for a cell-based viewer to show.
    ``conductor_mask()`` is verified bit-identical regardless of the flag --
    the premise the viewer's skip-note rests on -- and the title must count
    flagged bodies instead of drawing them identically to an unflagged one
    with no comment.
    """
    dom = (3e-3, 3e-3, 2e-3)

    def _make(two_plane):
        sim = Simulation(freq_max=20e9, domain=dom, dx=DX, cpml_layers=4,
                         boundary="cpml")
        sim.add(Box((0.5e-3, 0.5e-3, 0.4e-3), (2.5e-3, 2.5e-3, 0.5e-3)),
                material="pec", two_plane=two_plane)
        return sim

    s0, s1 = _make(False), _make(True)
    grid = s0._build_grid()
    c0 = np.asarray(s0.conductor_mask(grid), dtype=bool)
    c1 = np.asarray(s1.conductor_mask(grid), dtype=bool)
    assert np.array_equal(c0, c1), (
        "conductor_mask() must be bit-identical regardless of two_plane -- "
        "if this ever changes, the viewer's skip-note claim (and this test) "
        "must change with it, because it would mean the wall IS now visible")

    fig1 = plot_rasterized_slice(s1)
    assert "two-plane" in fig1.axes[0].get_title(), (
        "a two_plane=True body must be counted in the title since its "
        "second wall cannot be drawn")
    fig0 = plot_rasterized_slice(s0)
    assert "two-plane" not in fig0.axes[0].get_title(), (
        "an unflagged body must not trigger the two-plane note")


def test_absorber_pad_is_hatched_past_the_declared_domain():
    """The material assembly legitimately extends into the CPML pad -- CPML
    needs a real material to absorb into -- but nothing marked where the
    declared domain ends and the pad begins, so that legitimate overhang
    read as "the rasterizer grew my substrate."
    """
    sim = _board()
    fig = plot_rasterized_slice(sim, axis=2)
    ax = fig.axes[0]
    hatched = [p for p in ax.patches
              if getattr(p, "get_hatch", lambda: None)() == "////"
              and not p.get_fill()]
    assert hatched, "no CPML absorber patch drawn for a cpml_layers=4 sim"
    past_lo = any(p.get_x() < -1e-6 or p.get_y() < -1e-6 for p in hatched)
    past_hi = any(
        (p.get_x() + p.get_width()) > DOM[0] * 1e3 + 1e-6
        or (p.get_y() + p.get_height()) > DOM[1] * 1e3 + 1e-6
        for p in hatched)
    assert past_lo and past_hi, (
        "the absorber patches must sit past BOTH declared domain edges "
        f"(0 and {DOM[0] * 1e3:g} mm); got {[(p.get_x(), p.get_y(), p.get_width(), p.get_height()) for p in hatched]}")


def test_a_caller_supplied_axes_figure_is_not_reflowed():
    """`fig.tight_layout()` reflows EVERY axes in the figure, not just the
    one being drawn into -- measured moving a sibling subplot's bounds on
    every call, so a two-panel comparison ends up at two different scales
    depending on which panel was drawn last.
    """
    sim = _board()
    fig, axes = plt.subplots(1, 2)
    before = axes[1].get_position().bounds
    plot_rasterized_slice(sim, ax=axes[0])
    after = axes[1].get_position().bounds
    assert before == after, (
        f"drawing into axes[0] must not move axes[1]'s bounds: "
        f"{before} -> {after}")
    plt.close(fig)


def test_aspect_is_not_forced_on_an_extreme_board():
    """`set_aspect('equal', adjustable='box')` on an extreme (30 x ~1 mm
    edge-on) slice shrank the axes box to a sub-pixel sliver (measured:
    0.045 of the figure height on this exact fixture, down from 0.77 with
    the fix)."""
    dom = (30e-3, 5e-3, 0.8e-3)
    sim = Simulation(freq_max=20e9, domain=dom, dx=DX, cpml_layers=4,
                     boundary="cpml")
    sim.add_material("sub", eps_r=4.4, sigma=0.0)
    sim.add(Box((0, 0, 0), (dom[0], dom[1], 0.6e-3)), material="sub")
    sim.add(Box((0, 0, 0), (dom[0], dom[1], 0.2e-3)), material="pec")
    fig = plot_rasterized_slice(sim, axis=1)
    bbox = fig.axes[0].get_position()
    assert bbox.height > 0.3, (
        f"axes box height {bbox.height:.4f} is a sliver -- equal aspect was "
        "forced on an extreme-aspect slice (30 mm wide x ~1.6 mm tall "
        "including CPML pad)")


def test_conductor_overlay_is_not_opaque():
    """An opaque conductor overlay paints over the exact cell whose eps
    ``plot_stack_profile``'s docstring promises stays readable."""
    from matplotlib.collections import QuadMesh
    sim = _board()
    fig = plot_rasterized_slice(sim, axis=2)
    meshes = [c for c in fig.axes[0].collections if isinstance(c, QuadMesh)]
    assert len(meshes) >= 2, "expected an eps mesh and a conductor overlay mesh"
    cond_mesh = meshes[-1]
    alpha = cond_mesh.get_alpha()
    assert alpha is not None and alpha < 1.0, (
        f"the conductor overlay must be semi-transparent (alpha < 1); got {alpha!r}")


def test_legend_explains_the_conductor_color():
    """Only the title's cell count implied what red meant; nothing in the
    figure said so directly."""
    sim = _board()
    fig = plot_rasterized_slice(sim, axis=2)
    leg = fig.axes[0].get_legend()
    assert leg is not None, "no legend drawn even though conductor cells exist"
    labels = [t.get_text() for t in leg.get_texts()]
    assert any("conductor" in l for l in labels), (
        f"nothing in the legend explains what red means; got {labels}")


def test_position_search_reaches_the_same_physical_distance_on_a_graded_axis():
    """A fixed +/-1 INDEX window reaches a different PHYSICAL distance on
    each side of a grading transition.

    Fixture: 50 um cells below index 10, 200 um cells from index 10 on.
    Querying `position` at the transition node makes the search radius the
    WIDER touching cell (200 um). A one-cell sheet placed three 50 um cells
    before the transition (150 um away -- more than 1 INDEX, less than the
    200 um radius) must be found; a fixed +/-1 index window would only
    reach the immediate neighbour index and miss it (measured: index 14 /
    0 conductor cells with a hardcoded +/-1 window on this exact fixture).
    """
    nz = 20
    prof = np.full(nz, 50e-6)
    prof[10:] = 200e-6
    dom = (3e-3, 3e-3, 3e-3)
    sim = Simulation(freq_max=20e9, domain=dom, dx=DX, cpml_layers=4,
                     boundary="cpml", dz_profile=prof)
    sim.add_material("sub", eps_r=1.0, sigma=0.0)
    grid = sim._build_nonuniform_grid()
    from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid
    nodes = np.asarray(coords_from_nonuniform_grid(grid).z, dtype=float)
    t_idx = grid.pad_z_lo + 10
    sheet_idx = t_idx - 3
    z_lo, z_hi = float(nodes[sheet_idx]), float(nodes[sheet_idx + 1])
    sim.add(Box((0, 0, z_lo), (dom[0], dom[1], z_hi)), material="pec")

    pos = float(nodes[t_idx])
    fig = plot_rasterized_slice(sim, axis=2, position=pos)
    title = fig.axes[0].get_title()
    shown = int(title.split("—")[1].split()[0])
    assert shown > 0, (
        f"a sheet 150 um away (three 50 um cells, inside the 200 um "
        f"coarse-side cell touching the query node) must be found; a fixed "
        f"+/-1 index window only reaches 50 um on the fine side. title: "
        f"{title!r}")
