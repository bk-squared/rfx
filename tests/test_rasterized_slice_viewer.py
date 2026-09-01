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

from rfx import Box, Cylinder, Simulation
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

    ``_axis_edges`` does NOT append a closing edge at all: a NonUniformGrid
    already carries N+1 nodes for N real cells (``_append_bounding_node``,
    ``rfx/nonuniform.py`` -- explicitly "the node count the uniform Grid
    allocates for N cells"), so ``nodes`` itself IS the N+1-edge array.
    Appending one more (this function's PREVIOUS behaviour, whether by
    re-reading ``dz`` or by duplicating the last node) either overshoots
    past the grid's real extent or makes the last REAL data column
    zero-width and invisible -- see
    ``test_last_real_cells_extent_and_data_are_not_lost_past_the_fence_post``
    for the visible-picture consequence of both.
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
    assert e.size == nodes.size, (
        f"got {e.size} edges for {nodes.size} nodes -- NonUniformGrid "
        "already carries N+1 nodes for N cells, nothing should be appended")
    assert np.allclose(e, nodes, rtol=1e-5), "edges are not the nodes"
    assert np.allclose(np.diff(e), dz[:nodes.size - 1], rtol=1e-5), (
        "cell widths do not match the spacing array")


def test_uniform_lane_edges_match_the_scalar_cell_size():
    sim = _board()
    grid = sim._build_grid()
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid
    nodes = np.asarray(coords_from_uniform_grid(grid).x, dtype=float)
    e = _axis_edges(grid, 0, nodes)
    # Grid.nx already carries the "+1 fence-post correction" (rfx/grid.py
    # :151, "N cells need N+1 nodes so that PEC walls at index 0 and index
    # N span exactly N*dx") baked directly into the grid dimension --
    # `nodes` (built from `nx`) IS the N+1-edge array pcolormesh wants.
    # Appending one more `dx` here (this function's PREVIOUS behaviour)
    # drew the axis one full cell past the grid's real extent (measured,
    # 3 mm domain + 4 CPML at dx=100 um: -400..3500 um drawn against
    # true walls at -400..3400 um).
    assert e.size == nodes.size, (
        f"got {e.size} edges for {nodes.size} nodes -- the uniform Grid's "
        "own fence-post +1 already supplies the N+1th edge")
    assert np.allclose(np.diff(e), float(grid.dx), rtol=1e-5)
    # Pin the ORIGIN too: a midpoint construction (edges at node +/- dx/2)
    # has the same size and the same constant spacing as the correct one,
    # differing only by a dx/2 offset that the two checks above cannot see.
    assert np.isclose(e[0], nodes[0]), (
        f"the first edge {e[0]:g} must be the first node {nodes[0]:g} "
        "itself, not the node's midpoint")
    assert np.isclose(e[-1], nodes[-1]), (
        f"the last edge {e[-1]:g} must be the fence-post node "
        f"{nodes[-1]:g} itself, not one more dx past it")


def test_last_real_cells_extent_and_data_are_not_lost_past_the_fence_post():
    """The two ways to get the fence-post edge wrong, and what each looks
    like in the actual picture, not just in ``_axis_edges``'s return value.

    Appending ``+dx`` past the fence-post node overshoots the drawn extent
    (measured: 3 mm domain + 4 CPML at dx=100 um drew -400..3500 um against
    true walls -400..3400 um -- a full 100 um / 1 cell past the real wall).
    Appending a DUPLICATE of the fence-post node instead keeps the extent
    right but gives the last REAL data column zero width, so a body filling
    the whole domain would go invisible exactly there. Both are ruled out
    at once: the drawn x-extent must match the true walls (from
    ``grid.pad_x_lo/hi`` and ``grid.nx``, computed independently of
    ``_axis_edges``) AND the last real column's material must still show
    up as data (checked one column short of ``eps``'s raw array length,
    which is exactly the fence-post slot this guards).
    """
    dom = (3e-3, 3e-3, 3e-3)
    sim = Simulation(freq_max=20e9, domain=dom, dx=DX, cpml_layers=4,
                     boundary="cpml")
    sim.add_material("sub", eps_r=9.0, sigma=0.0)
    sim.add(Box((0, 0, 0), (dom[0], dom[1], dom[2])), material="sub")
    grid = sim._build_grid()
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid
    nodes = np.asarray(coords_from_uniform_grid(grid).x, dtype=float)
    true_lo = (0 - grid.pad_x_lo) * DX
    true_hi = (grid.nx - 1 - grid.pad_x_lo) * DX
    fig = plot_rasterized_slice(sim, axis=2)
    ax = fig.axes[0]
    xlim = ax.get_xlim()
    assert np.isclose(xlim[0], true_lo * 1e3, atol=1e-6), (
        f"drawn x lo {xlim[0]:.4f} mm != true wall {true_lo * 1e3:.4f} mm")
    assert np.isclose(xlim[1], true_hi * 1e3, atol=1e-6), (
        f"drawn x hi {xlim[1]:.4f} mm != true wall {true_hi * 1e3:.4f} mm "
        "-- overshoot means the axis was extended past the fence-post node")

    eps = np.asarray(sim._assemble_materials(grid)[0].eps_r, dtype=float)
    # The fence-post/last array index (grid.nx - 1) is the degenerate slot;
    # the last REAL cell is one short of that. A domain-filling box must
    # still show eps=9 there in the DRAWN data, not just in the raw array.
    last_real = eps[grid.nx - 2, grid.ny - 2, 5]
    assert np.isclose(last_real, 9.0), (
        f"fixture sanity: last real cell should carry eps=9, got {last_real}")
    xe = _axis_edges(grid, 0, nodes) * 1e3
    ye = _axis_edges(grid, 1, np.asarray(coords_from_uniform_grid(grid).y, dtype=float)) * 1e3
    n_x, n_y = xe.size - 1, ye.size - 1
    assert n_x == grid.nx - 1 and n_y == grid.ny - 1, (
        "drawn cell counts must be one short of the raw array length -- "
        f"got n_x={n_x} (grid.nx-1={grid.nx - 1}), n_y={n_y} "
        f"(grid.ny-1={grid.ny - 1})")


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


def test_a_patterned_body_is_counted_as_skipped_not_silently_undrawn():
    """A patterned body (no ``corner_lo``/``corner_hi`` -- a Cylinder,
    Sphere, or MeshShape) is deliberately not drawn as its bounding box
    (that would assert a solid rectangle the model never declared), but
    the honesty mechanism only means something if the skip is COUNTED.
    Untested before this: nothing pinned that ``n_skipped`` actually
    increments for a real patterned-body fixture.
    """
    sim = Simulation(freq_max=20e9, domain=DOM, dx=DX, cpml_layers=4,
                     boundary="cpml")
    sim.add_material("sub", eps_r=4.0, sigma=0.0)
    sim.add(Box((0, 0, 0), (DOM[0], DOM[1], H_SUB)), material="sub")
    sim.add(Cylinder(center=(1.5e-3, 1.5e-3, H_SUB + 0.05e-3), radius=0.5e-3,
                     height=0.1e-3, axis="z"), material="pec")
    fig = plot_rasterized_slice(sim, axis=2)
    t = fig.axes[0].get_title()
    assert "patterned" in t and "1" in t.split("patterned")[0], (
        f"expected a '1 patterned bod(y/ies)' skip note; got title {t!r}")
    ax = fig.axes[0]
    # No dashed declared-outline rectangle for the cylinder -- it has no
    # corner_lo/corner_hi, so drawing one would be a bounding-box lie.
    dashed = [p for p in ax.patches if p.get_linestyle() not in
             (None, "solid", "-")]
    assert not dashed, (
        "a patterned body must not be drawn as a bounding-box outline")


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


def test_stack_profile_falls_back_to_the_column_with_eps_structure_not_the_cpml_corner():
    """The POSITIVE half of the item-6 fix, previously untested: a
    dielectric-only model (no conductor, but real permittivity structure
    somewhere) must pick THAT column, not silently fall through to
    ``argmax`` on an all-zero conductor-count array (which returns index 0
    -- the CPML corner). ``test_stack_profile_refuses_a_model_with_nothing_
    to_read`` only pins the case with NEITHER conductor NOR permittivity
    structure (raises); a mutant that hardcoded ``i0=i1=0`` whenever
    ``counts.max()==0`` -- skipping the ``var``/eps-variance fallback
    entirely -- would still raise nothing and would still pass that test.
    """
    sim = Simulation(freq_max=20e9, domain=DOM, dx=DX, cpml_layers=4,
                     boundary="cpml")
    sim.add_material("die", eps_r=9.0, sigma=0.0)
    # An off-center resonator, well away from the (0, 0) CPML corner.
    sim.add(Box((1.0e-3, 1.0e-3, 0.4e-3), (2.0e-3, 2.0e-3, 0.8e-3)),
           material="die")
    fig = plot_stack_profile(sim)
    t = fig.axes[0].get_title()
    assert "no conductor in this model" in t
    x_at = float(t.split("x=")[1].split(" mm")[0])
    y_at = float(t.split("y=")[1].split(" mm")[0])
    assert 0.9 <= x_at <= 2.1 and 0.9 <= y_at <= 2.1, (
        f"the auto-picked column (x={x_at}, y={y_at}) must fall inside the "
        "resonator's declared span [1.0, 2.0] mm, not at/near the CPML "
        f"corner; title: {t!r}")


def test_stack_shading_is_absolute_so_two_columns_compare():
    """Per-column normalisation made a solid eps_r=9 column pixel-identical to
    a vacuum one, while the docstring promises the tint carries permittivity."""
    import inspect
    from rfx.visualize import plot_stack_profile as f
    src = inspect.getsource(f)
    assert "np.ptp(eps_col)" not in src, (
        "the column shading is normalised per column again")
    assert "eps_hi" in src, "no absolute permittivity anchor in the shading"


def test_add_refinement_warns_because_the_drawn_grid_is_coarse_only():
    """The commit that first claimed this was fixed never actually
    implemented it (``git log -p -- rfx/visualize.py`` shows the string
    "refinement" never appeared in this file before this test's own
    commit) -- the claim was false, and both docstrings' "the grid this
    simulation would RUN on" was silently wrong whenever a refinement
    region was set. ``add_refinement()`` (SBP-SAT subgridding, #90,
    EXPERIMENTAL) runs its fine z-region on a separate runner this
    module's grid-builders (``_build_grid`` / ``_build_nonuniform_grid``)
    know nothing about.
    """
    sim = _board()
    sim.add_refinement((0.4e-3, 1.2e-3), ratio=4)
    with pytest.warns(UserWarning, match="add_refinement"):
        plot_rasterized_slice(sim, axis=1)
    with pytest.warns(UserWarning, match="add_refinement"):
        plot_stack_profile(sim)


# --- regressions for the SECOND adversarial review pass (#11-24) -----------

def test_two_plane_wall_is_drawn_plane_locally_on_its_own_wall_plane():
    """conductor_mask() is a CELL footprint; #706's ``two_plane=True`` opt-in
    zeroes an extra tangential-E EDGE at the body's far node plane during
    the solve, and ``conductor_mask()`` is verified bit-identical regardless
    of the flag -- so there is genuinely no CONDUCTOR CELL for a
    cell-based overlay to show on the wall's own plane. But the edge
    operator's footprint IS independently computable
    (``two_plane_extension_masks``, the same function the solve uses), so
    it is drawn as its own marker rather than only being explained away.

    Reproduces the repo's own two_plane fixture (a 2x2 mm, 1-cell-thick
    box, dx=100 um): measured wall_mask nonzero at z=8 (the body's own
    plane, the interior normal-E edge) AND z=9 (the far face) -- BEFORE
    the marker fix, the wall's own plane showed "0 conductor cells" with
    no note at all.

    Re-pinned at the exact-coordinate fix (#802/#807): the old body plane
    9 was the float32 thin-branch tie artifact; on exact float64 nodes the
    face-registered one-cell box realizes on its lo-face node (plane 8,
    the node its own half-open [lo, hi) window keeps), and the transverse
    span gains its convention-owed node (361 -> 400 cells per plane).
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
        "the premise that the wall marker (not conductor_mask itself) is "
        "what has to carry this information")

    # The body's own plane (index 8): a conductor cell IS present, plus the
    # interior-normal-E-edge component of the wall operator.
    fig_body = plot_rasterized_slice(s1, axis=2, index=8)
    t_body = fig_body.axes[0].get_title()
    assert int(t_body.split("—")[1].split()[0]) == 400, "fixture sanity"
    assert "sealing wall" in t_body

    # The wall's OWN plane (index 9): flag ON -> marker present, title
    # says so, even though there are 0 conductor cells there.
    fig_wall_on = plot_rasterized_slice(s1, axis=2, index=9)
    ax_wall_on = fig_wall_on.axes[0]
    t_wall_on = ax_wall_on.get_title()
    assert int(t_wall_on.split("—")[1].split()[0]) == 0
    assert "sealing wall" in t_wall_on, (
        f"plane 9 must note the sealing wall even with 0 conductor "
        f"cells there; got title {t_wall_on!r}")
    from matplotlib.collections import QuadMesh
    wall_meshes = [c for c in ax_wall_on.collections if isinstance(c, QuadMesh)]
    assert len(wall_meshes) >= 3, (
        "expected an eps mesh, a conductor overlay mesh, AND a wall "
        f"marker mesh; got {len(wall_meshes)}")
    leg = ax_wall_on.get_legend()
    assert leg is not None and any(
        "two-plane wall" in t.get_text() for t in leg.get_texts())

    # Flag OFF: the wall plane has neither a marker nor a note.
    fig_wall_off = plot_rasterized_slice(s0, axis=2, index=9)
    ax10_off = fig_wall_off.axes[0]
    assert "two-plane" not in ax10_off.get_title()
    wall_meshes_off = [c for c in ax10_off.collections if isinstance(c, QuadMesh)]
    assert len(wall_meshes_off) == 2, (
        "an unflagged body must draw only the eps + conductor meshes, no "
        f"wall marker; got {len(wall_meshes_off)}")


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


def test_absorber_pad_edges_match_the_true_pad_node_coordinates():
    """PLACEMENT, not just presence: a hatch rect existing somewhere near
    the boundary does not prove it covers the right cells. A mutant that
    zeroed the hi-side pad count on one axis still drew SOME hatch (from
    the lo side and the other axis) and passed the presence-only check
    above; this pins every one of the four lo/hi x/y edges against the
    pad node coordinate computed independently from
    ``coords_from_uniform_grid`` + ``grid.pad_{x,y}_{lo,hi}`` (NOT from
    ``_axis_edges`` or ``_absorber_rects`` themselves, so a shared bug in
    either cannot cancel out here), on both the uniform slice and the
    non-uniform stack lane.
    """
    from rfx.geometry.rasterize_grid import (coords_from_uniform_grid,
                                             coords_from_nonuniform_grid)

    # --- uniform lane: plot_rasterized_slice, both in-plane axes ---
    sim = _board()
    grid = sim._build_grid()
    xn = np.asarray(coords_from_uniform_grid(grid).x, dtype=float)
    yn = np.asarray(coords_from_uniform_grid(grid).y, dtype=float)
    # four named boundaries per axis: far edge, lo-pad/interior seam,
    # interior/hi-pad seam, far edge -- NOT reused across the two rects.
    x_lo_edge, x_lo_seam = xn[0], xn[grid.pad_x_lo]
    x_hi_seam, x_hi_edge = xn[grid.nx - 1 - grid.pad_x_hi], xn[-1]
    y_lo_edge, y_lo_seam = yn[0], yn[grid.pad_y_lo]
    y_hi_seam, y_hi_edge = yn[grid.ny - 1 - grid.pad_y_hi], yn[-1]
    fig = plot_rasterized_slice(sim, axis=2)
    hatched = [p for p in fig.axes[0].patches
              if getattr(p, "get_hatch", lambda: None)() == "////"
              and not p.get_fill()]

    def _has_rect(x0, x1, y0, y1, tol=1e-6):
        return any(np.isclose(p.get_x(), x0 * 1e3, atol=tol)
                  and np.isclose(p.get_x() + p.get_width(), x1 * 1e3, atol=tol)
                  and np.isclose(p.get_y(), y0 * 1e3, atol=tol)
                  and np.isclose(p.get_y() + p.get_height(), y1 * 1e3, atol=tol)
                  for p in hatched)

    assert _has_rect(x_lo_edge, x_lo_seam, y_lo_edge, y_hi_edge), (
        f"x-lo pad rect missing/misplaced; want x=[{x_lo_edge * 1e3:.4f}, "
        f"{x_lo_seam * 1e3:.4f}] y=[{y_lo_edge * 1e3:.4f}, {y_hi_edge * 1e3:.4f}] mm")
    assert _has_rect(x_hi_seam, x_hi_edge, y_lo_edge, y_hi_edge), (
        "x-hi pad rect missing/misplaced -- would be dropped by a "
        "hardcoded hi=0 mutation on this axis")
    assert _has_rect(x_lo_edge, x_hi_edge, y_lo_edge, y_lo_seam), (
        "y-lo pad rect missing/misplaced")
    assert _has_rect(x_lo_edge, x_hi_edge, y_hi_seam, y_hi_edge), (
        "y-hi pad rect missing/misplaced -- would be dropped by a "
        "hardcoded hi=0 mutation on this axis")

    # --- non-uniform lane: plot_stack_profile, the stack axis itself ---
    nz = 20
    prof = np.full(nz, 100e-6)
    prof[10:] = 150e-6
    sim2 = _board(dz_profile=prof)
    grid2 = sim2._build_nonuniform_grid()
    zn = np.asarray(coords_from_nonuniform_grid(grid2).z, dtype=float)
    z_lo_edge, z_lo_seam = zn[0], zn[grid2.pad_z_lo]
    z_hi_seam, z_hi_edge = zn[grid2.nz - 1 - grid2.pad_z_hi], zn[-1]
    fig2 = plot_stack_profile(sim2)
    hatched2 = [p for p in fig2.axes[0].patches
               if getattr(p, "get_hatch", lambda: None)() == "////"
               and not p.get_fill()]

    def _has_span(z0, z1, tol=1e-6):
        return any(np.isclose(p.get_y(), z0 * 1e3, atol=tol)
                  and np.isclose(p.get_y() + p.get_height(), z1 * 1e3, atol=tol)
                  for p in hatched2)

    assert _has_span(z_lo_edge, z_lo_seam), (
        f"z-lo pad span missing/misplaced; true [{z_lo_edge * 1e3:.4f}, "
        f"{z_lo_seam * 1e3:.4f}] mm, got "
        f"{[(p.get_y(), p.get_y() + p.get_height()) for p in hatched2]}")
    assert _has_span(z_hi_seam, z_hi_edge), (
        f"z-hi pad span missing/misplaced -- would be dropped by a "
        f"hardcoded hi=0 mutation; true [{z_hi_seam * 1e3:.4f}, "
        f"{z_hi_edge * 1e3:.4f}] mm, got "
        f"{[(p.get_y(), p.get_y() + p.get_height()) for p in hatched2]}")


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


def test_colorbar_on_a_caller_supplied_axes_does_not_move_siblings():
    """`fig.colorbar(mesh, ax=ax)` steals space from ITS OWN target axes
    (an expected, matplotlib-standard mechanism -- and, combined with
    `adjustable="box"`, that target axes' box can shrink a lot to keep
    equal aspect in the space that's left) but must not touch any OTHER
    axes in a caller-supplied multi-panel figure -- checked in isolation
    from the `tight_layout()` guard (see the test above), with THREE
    panels and TWO separate calls so a colorbar added by the second call
    cannot be mistaken for moving the first call's own axes either.
    """
    fig, axes = plt.subplots(1, 3)
    before = [ax.get_position().bounds for ax in axes]
    plot_rasterized_slice(_board(), ax=axes[0])
    after_first = [ax.get_position().bounds for ax in axes]
    assert before[1] == after_first[1] and before[2] == after_first[2], (
        "adding a colorbar to axes[0] must not move axes[1] or axes[2]")
    plot_rasterized_slice(_board(), ax=axes[1])
    after_second = [ax.get_position().bounds for ax in axes]
    assert after_first[0] == after_second[0], (
        "drawing a second colorbar into axes[1] must not move axes[0]'s "
        "already-established bounds")
    assert before[2] == after_second[2], (
        "axes[2], never drawn into, must stay at its original bounds "
        "through both calls")
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


def test_title_with_every_honesty_note_fits_the_default_canvas():
    """The two-plane note alone measured ~1300 px wide on a single title
    line (default 10pt font); two-plane + moved-plane together ~2340 px --
    both well past the 800 px default figsize=(8,6) canvas, so exactly the
    notes this honesty mechanism exists to surface were the part getting
    clipped off-canvas. Trigger BOTH notes at once (a two_plane body AND a
    position= that lands on an empty plane next to a sheet) and measure the
    actual rendered title bounding box against the actual canvas width --
    not a character count, which does not account for font metrics.
    """
    sim = Simulation(freq_max=20e9, domain=DOM, dx=DX, cpml_layers=4,
                     boundary="cpml")
    sim.add_material("sub", eps_r=4.0, sigma=0.0)
    sim.add(Box((0, 0, 0), (DOM[0], DOM[1], H_SUB)), material="sub")
    sim.add(Box((0.5e-3, 1.0e-3, H_SUB), (2.5e-3, 2.0e-3, H_SUB + DX)),
           material="pec", two_plane=True)
    grid = sim._build_grid()
    cond = np.asarray(sim.conductor_mask(grid), dtype=bool)
    per_plane = cond.reshape(-1, cond.shape[2]).sum(axis=0)
    empty = [k for k in range(cond.shape[2]) if per_plane[k] == 0
             and any(per_plane[max(k - 1, 0):k + 2])]
    if not empty:
        pytest.skip("this fixture has no empty plane adjacent to a sheet")
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid
    z = np.asarray(coords_from_uniform_grid(grid).z, dtype=float)
    fig = plot_rasterized_slice(sim, axis=2, position=float(z[empty[0]]))
    ax = fig.axes[0]
    title_text = ax.get_title()
    assert "two-plane" in title_text
    assert "showing the neighbouring plane" in title_text
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.title.get_window_extent(renderer=renderer)
    canvas_px = fig.get_size_inches()[0] * fig.dpi
    assert bbox.width <= canvas_px, (
        f"rendered title is {bbox.width:.0f} px wide on a {canvas_px:.0f} "
        f"px canvas -- the honesty notes would be clipped off-canvas. "
        f"title:\n{title_text!r}")
