"""One tie rule for points and conductors on the non-uniform lane (#1295, #1342).

A source, probe or port declared exactly halfway between two grid nodes is
equally near both. A PEC sheet or wire vertex declared there goes to the LOWER
node (#931). The non-uniform lane now resolves a point feature with the same
function, on the same node line the conductors are rasterized on, so a port
declared on a sheet or a wire lands on it. Before this change the NU lane
also took the lower node, but through its own float32 edge search, which put
a coordinate within float32 roundoff of a half node on either node. The
uniform lane rounds a point's tie to the EVEN node (``round(x/dx)``) until
#1342; the two lanes agree off ties, and agreement at ties is a strict xfail
against #1342 so it turns into an XPASS when #1342 lands.

What is pinned here:

* on equal cells, every NU lookup (``index_of``, ``position_to_index``,
  ``pos_to_nu_index``) lands on the node the sheet snap gives the same
  coordinate, and off ties on the uniform lane's node;
* on a graded axis, a tie goes to the lower node and every other coordinate
  keeps its nearest node;
* the other sites that resolve through the rule: waveguide aperture ends,
  preflight's model of the placement, the traced run-time check.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.geometry.rasterize_grid import _nearest_plane, coords_from_nonuniform_grid
from rfx.nonuniform import make_nonuniform_grid, position_to_index
from rfx.runners.nonuniform import pos_to_nu_index

DX = 1e-3
AXES = ("x", "y", "z")
ISSUE_1342 = ("#1342: the uniform lane rounds a point feature's tie to the even "
              "node; the non-uniform lane and every conductor round it down")

# (cells per axis, boundary). 239 cells is the kernel-timing 13M y axis whose
# half-domain source exposed the split; the CPML case puts a pad in front of
# the interior so the padded index is checked as well.
LANE_CASES = {
    "pec_239": ((239, 12, 10), "pec"),
    "pec_odd_even": ((20, 19, 17), "pec"),
    "cpml_pads": ((13, 21, 9), "cpml"),
}


def _both_lanes(cells, boundary):
    """The uniform ``Grid`` and the NU grid ``Simulation`` builds for one
    declaration, the NU one from constant profiles on all three axes."""
    domain = tuple(n * DX for n in cells)
    kw = dict(freq_max=10e9, domain=domain, dx=DX, boundary=boundary,
              cpml_layers=8 if boundary == "cpml" else 0)
    uniform = Simulation(**kw)._build_grid()
    nonuniform = Simulation(
        **kw, dx_profile=np.full(cells[0], DX), dy_profile=np.full(cells[1], DX),
        dz_profile=np.full(cells[2], DX))._build_nonuniform_grid()
    assert uniform.shape == nonuniform.shape
    return uniform, nonuniform, domain


def _sweep(n_cells, extent):
    """Every node, every half node spelled three ways and one float step to
    each side, the fractions of the domain the kernel-timing model uses, and
    a scatter of ordinary coordinates."""
    k = np.arange(n_cells + 1, dtype=float)
    half = (k[:-1] + 0.5) * DX
    xs = np.concatenate([
        k * DX, half, k[:-1] * DX + 0.5 * DX, 0.5 * (2 * k[:-1] + 1) * DX,
        np.nextafter(half, np.inf), np.nextafter(half, -np.inf),
        np.array([0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75]) * extent,
        np.random.default_rng(n_cells).uniform(0.0, extent, 400),
    ])
    return xs[(xs >= 0.0) & (xs <= extent)]


def _tie_lower(line, x, cell):
    """The tie rule spelled here: the nearest node on ``line``, and the lower
    of two whose distances agree to 1e-9 of the cell. Returns (node, tie)."""
    dist = np.abs(np.asarray(line, dtype=np.float64) - float(x))
    k = int(np.argmin(dist))
    if k >= 1 and abs(dist[k - 1] - dist[k]) <= 1e-9 * cell:
        return k - 1, True
    return k, bool(k + 1 < dist.size and abs(dist[k + 1] - dist[k]) <= 1e-9 * cell)


def _nu_nodes(nu, ax, axis, x):
    pos = [0.0, 0.0, 0.0]
    pos[ax] = float(x)
    return (nu.index_of(axis, float(x)), position_to_index(nu, tuple(pos))[ax],
            pos_to_nu_index(nu, tuple(pos))[ax])


@pytest.mark.parametrize("case", sorted(LANE_CASES))
def test_a_point_lands_on_the_node_a_sheet_at_the_same_coordinate_takes(case):
    """Lane B's point lookups against lane B's sheet snap, on the node line
    the rasterizer uses; and off ties, against the uniform lane."""
    cells, boundary = LANE_CASES[case]
    uniform, nu, domain = _both_lanes(cells, boundary)
    coords = coords_from_nonuniform_grid(nu)
    counts = dict(off_tie=0, tie=0, tie_lower_odd=0)
    for ax, axis in enumerate(AXES):
        line = np.asarray(getattr(coords, axis), dtype=np.float64)
        for x in _sweep(cells[ax], domain[ax]):
            got = _nu_nodes(nu, ax, axis, x)
            sheet = _nearest_plane(line, float(x), DX, axis=ax)
            assert got == (sheet, sheet, sheet), (case, axis, float(x), got, sheet)
            want, tie = _tie_lower(line, x, DX)
            assert got[0] == want, (case, axis, float(x), got[0], want)
            if tie:
                counts["tie"] += 1
                pad = (nu.pad_x_lo, nu.pad_y_lo, nu.pad_z_lo)[ax]
                counts["tie_lower_odd"] += (want - pad) % 2 == 1
            else:
                counts["off_tie"] += 1
                assert uniform.index_of(axis, float(x)) == got[0], (
                    case, axis, float(x))
    print(f"{case}: {counts}", file=sys.stderr)
    # The rule only shows where the lower node of a tie is odd.
    assert counts["off_tie"] and counts["tie_lower_odd"]


@pytest.mark.xfail(strict=True, reason=ISSUE_1342)
def test_at_a_tie_both_lanes_put_a_point_on_the_same_node():
    for case in sorted(LANE_CASES):
        cells, boundary = LANE_CASES[case]
        uniform, nu, domain = _both_lanes(cells, boundary)
        for ax, axis in enumerate(AXES):
            for x in _sweep(cells[ax], domain[ax]):
                assert _nu_nodes(nu, ax, axis, x)[0] == uniform.index_of(
                    axis, float(x)), (case, axis, float(x))


def test_float_quotient_cases():
    """``0.1195 / 1e-3 = 119.49999999999999`` is a tie on the node line, and
    both lanes put it on 119. The kernel-timing source at ``0.5 * (239 *
    1e-3)`` and a feed at 9.5 mm go to the lower node (119, 9) on the NU
    lane, where a sheet at the same coordinate lands; the uniform lane puts
    them on 120 and 10 until #1342."""
    uniform, nu, domain = _both_lanes((239, 12, 10), "pec")
    line = np.asarray(coords_from_nonuniform_grid(nu).x, dtype=np.float64)
    for x, node in ((0.1195, 119), (0.5 * domain[0], 119), (0.0095, 9),
                    (0.0105, 10)):
        assert _nu_nodes(nu, 0, "x", x) == (node, node, node), x
        assert _nearest_plane(line, x, DX, axis=0) == node, x
    assert uniform.position_to_index((0.1195, 0.0, 0.0))[0] == 119
    assert uniform.position_to_index((0.0105, 0.0, 0.0))[0] == 10


def _kernel_timing_graded(n_cells, t=4):
    """The kernel-timing record's graded z profile: DX outside, DX/2 over the
    middle third, ``t`` geometric cells each side."""
    n_f = int(round(n_cells / 3))
    n_lo = (n_cells - n_f - 2 * t) // 2
    n_hi = n_cells - n_f - 2 * t - n_lo
    r = 2.0 ** (1.0 / (t + 1))
    up = [DX / 2 * r ** (i + 1) for i in range(t)]
    return np.asarray([DX] * n_lo + up[::-1] + [DX / 2] * n_f + up + [DX] * n_hi)


GRADED = {
    "kernel_timing_z": lambda: make_nonuniform_grid(
        (0.02, 0.02), _kernel_timing_graded(60), DX, cpml_layers=8),
    "random_cells_xyz": lambda: make_nonuniform_grid(
        (0, 0), np.random.default_rng(3).uniform(0.3e-3, 1.2e-3, 40), DX,
        cpml_layers=4,
        dx_profile=np.r_[DX, np.random.default_rng(4).uniform(0.2e-3, 1e-3, 30), DX],
        dy_profile=np.r_[np.full(3, 0.5e-3), np.full(6, 0.25e-3), np.full(3, 0.5e-3)]),
}


@pytest.mark.parametrize("name", sorted(GRADED))
def test_graded_axis_ties_go_to_the_lower_node_and_nothing_else_moves(name):
    """On the node line the rasterizer uses, a tie goes to the lower node and
    every other coordinate to its nearest node (the ``argmin`` main's
    ``index_of`` took). A coordinate one float step above a midpoint is a
    tie; the argmin put it on the upper node."""
    grid = GRADED[name]()
    coords = coords_from_nonuniform_grid(grid)
    ties = moved = 0
    for ax, axis in enumerate(AXES):
        if grid.is_constant(axis):
            continue
        pad_lo = (grid.pad_x_lo, grid.pad_y_lo, grid.pad_z_lo)[ax]
        pad_hi = (grid.pad_x_hi, grid.pad_y_hi, grid.pad_z_hi)[ax]
        line = np.asarray(getattr(coords, axis), dtype=np.float64)
        cells = np.asarray(grid.cells(axis), dtype=np.float64)
        nodes = line[pad_lo:line.size - pad_hi]
        mids = 0.5 * (nodes[:-1] + nodes[1:])
        xs = np.concatenate([
            nodes, mids, np.nextafter(mids, np.inf), np.nextafter(mids, -np.inf),
            mids + 1e-12, mids - 1e-12, mids + 1e-9, mids - 1e-9,
            np.random.default_rng(ax).uniform(nodes[0], nodes[-1], 400)])
        for x in xs[(xs >= nodes[0]) & (xs <= nodes[-1])]:
            k = int(np.clip(np.searchsorted(line, x, side="right") - 1,
                            0, cells.size - 1))
            want, tie = _tie_lower(line, x, cells[k])
            before = int(np.argmin(np.abs(line - x)))
            ties += tie
            moved += want != before
            assert not (want != before and not tie), (name, axis, float(x))
            got = _nu_nodes(grid, ax, axis, x)
            assert got == (want, want, want), (
                f"{name}/{axis} x={float(x)!r}: expected node {want} "
                f"(argmin {before}, tie {tie}), got {got}")
    assert ties > moved > 0


def _kernel_timing_box(nonuniform, cells=(20, 19, 17), steps=400):
    """Board A of the kernel-timing record at 20 x 19 x 17 cells: a PEC box,
    an eps_r 3.66 block with its corners 0.3 cell off the node planes, one
    Ez current source at (0.3, 0.5, 0.7) of the domain and one probe at
    (0.6, 0.4, 0.3). With 19 cells along y the source sits at y = 9.5 mm,
    exactly midway between nodes 9 and 10."""
    cx, cy, cz = cells
    lx, ly, lz = cx * DX, cy * DX, cz * DX
    kw = dict(freq_max=10e9, dx=DX, stencil_order=2, precision="float32",
              boundary="pec")
    if nonuniform:
        kw.update(dx_profile=np.full(cx, DX), dy_profile=np.full(cy, DX),
                  dz_profile=np.full(cz, DX))
    sim = Simulation(domain=(lx, ly, lz), **kw)
    sim.add_material("block", eps_r=3.66)
    off = 0.3 * DX
    sim.add(Box((0.25 * lx + off, 0.25 * ly + off, 0.20 * lz + off),
                (0.75 * lx + off, 0.75 * ly + off, 0.50 * lz + off)),
            material="block")
    src = (0.30 * lx, 0.50 * ly, 0.70 * lz)
    sim.add_source(position=src, component="ez", amplitude_kind="current")
    sim.add_probe(position=(0.60 * lx, 0.40 * ly, 0.30 * lz), component="ez")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sim.run(n_steps=steps, skip_preflight=True)
    return (np.asarray(result.time_series, dtype=np.float64)[:, 0],
            result.grid, src)


def _lane_gap(cells):
    a, grid_a, src = _kernel_timing_box(nonuniform=False, cells=cells)
    b, grid_b, _ = _kernel_timing_box(nonuniform=True, cells=cells)
    peak = float(np.max(np.abs(a)))
    assert peak > 0.0
    gap = float(np.max(np.abs(a - b))) / peak
    print(f"cells {cells}: source nodes uniform {grid_a.position_to_index(src)} "
          f"NU {pos_to_nu_index(grid_b, src)}, trace gap / peak {gap:.3e}",
          file=sys.stderr)
    return src, grid_a.position_to_index(src), pos_to_nu_index(grid_b, src), gap


def test_a_source_on_a_node_gives_the_same_trace_on_both_lanes():
    """Control: with 18 cells along y the source is on node 9 on both lanes
    and the traces differ by the float32 difference between the kernels,
    1.5e-5 of the peak."""
    src, node_u, node_nu, gap = _lane_gap((20, 18, 17))
    assert src[1] / DX == 9.0 and node_u == node_nu and node_nu[1] == 9
    assert gap <= 3e-5


def test_a_half_node_source_goes_to_the_lower_node_on_the_nu_lane():
    """With 19 cells along y the source is at y = 9.5 mm, midway between
    nodes 9 and 10. The NU lane puts it on 9, where a sheet at 9.5 mm lands;
    the uniform lane on 10 (#1342)."""
    uniform, nu, domain = _both_lanes((20, 19, 17), "pec")
    src = (0.30 * domain[0], 0.50 * domain[1], 0.70 * domain[2])
    assert src[1] / DX == 9.5
    assert pos_to_nu_index(nu, src)[1] == 9
    assert uniform.position_to_index(src)[1] == 10


@pytest.mark.xfail(strict=True, reason=ISSUE_1342)
def test_a_half_node_source_gives_the_same_trace_on_both_lanes():
    """The kernel-timing box scaled down, with its source on the half node.
    On main the gap was 8.3e-2 of the peak, the one-cell feed shift; it
    comes down to the control's level when both lanes round a tie alike."""
    _src, _nu, _u, gap = _lane_gap((20, 19, 17))
    assert gap <= 3e-5


# ---------------------------------------------------------------------------
# The sites that resolve through the same rule besides the three lookups.
# ---------------------------------------------------------------------------

def _waveguide_cfg(lane, y_range, z_range, cells=(30, 48, 13)):
    import jax.numpy as jnp
    domain = tuple(n * DX for n in cells)
    kw = dict(freq_max=20e9, domain=domain, dx=DX, boundary="cpml",
              cpml_layers=8)
    if lane == "nu":
        kw.update(dx_profile=np.full(cells[0], DX),
                  dy_profile=np.full(cells[1], DX),
                  dz_profile=np.full(cells[2], DX))
    sim = Simulation(**kw)
    freqs = jnp.linspace(8e9, 12e9, 3)
    sim.add_waveguide_port(5.5e-3, y_range=y_range, z_range=z_range,
                           direction="+x", freqs=freqs)
    entry = sim._waveguide_ports[0]
    if lane == "nu":
        from rfx.runners.nonuniform import _build_waveguide_port_config_nu
        grid = sim._build_nonuniform_grid()
        cfg = _build_waveguide_port_config_nu(sim, entry, grid, freqs, 200)
    else:
        grid = sim._build_grid()
        cfg = sim._build_waveguide_port_config(entry, grid, freqs, 200)
    return cfg, grid


def _up(v):
    return float(np.nextafter(v, np.inf))


@pytest.mark.parametrize("y_range,z_range,y_on,z_on", [
    # every end on a half node: the NU aperture takes the lower nodes
    ((2.5e-3, 12.5e-3), (1.5e-3, 6.5e-3), (2e-3, 12e-3), (1e-3, 6e-3)),
    # one float step above: still a tie, where an argmin takes the upper node
    ((_up(3.5e-3), _up(13.5e-3)), (_up(2.5e-3), _up(7.5e-3)),
     (3e-3, 13e-3), (2e-3, 7e-3)),
    # above y = 33 mm main's float32 edge search put every half node on the
    # UPPER node of a 1 mm axis (its cumulative sums drift past the tie)
    ((35.5e-3, 45.5e-3), (1.5e-3, 6.5e-3), (35e-3, 45e-3), (1e-3, 6e-3)),
    # control: every end on a node
    ((2e-3, 12e-3), (1e-3, 6e-3), (2e-3, 12e-3), (1e-3, 6e-3)),
])
def test_waveguide_aperture_ends_take_the_lower_node_at_a_tie(
        y_range, z_range, y_on, z_on):
    """A waveguide port's aperture ends are coordinates too: between two
    sheet walls at the same half-node coordinates, the aperture must fill
    the guide. The NU aperture declared on half nodes equals the uniform
    lane's aperture declared on the lower nodes. Compared on the interior
    index (the NU lane pads the transverse faces, the uniform lane does
    not). On main a z_range of (1.5, 6.5) mm gave the NU aperture 5 mm
    against the uniform lane's 4 mm; here both give 5 cells."""
    cn, gn = _waveguide_cfg("nu", y_range, z_range)
    cu, gu = _waveguide_cfg("uniform", y_on, z_on)
    assert (cn.u_lo - gn.pad_y_lo, cn.u_hi - gn.pad_y_lo) == (
        cu.u_lo - gu.pad_y_lo, cu.u_hi - gu.pad_y_lo)
    assert (cn.v_lo - gn.pad_z_lo, cn.v_hi - gn.pad_z_lo) == (
        cu.v_lo - gu.pad_z_lo, cu.v_hi - gu.pad_z_lo)
    # The NU spans are still read from the float32 store, whose cumulative
    # sums drift: 1.7e-6 relative on the aperture above y = 33 mm.
    np.testing.assert_allclose(cn.a, cu.a, rtol=1e-5)
    np.testing.assert_allclose(cn.b, cu.b, rtol=1e-5)


PREFLIGHT_PROFILES = {
    "const_1mm": np.full(19, DX),
    "const_0p3": np.full(23, 0.3e-3),
    "graded": np.r_[np.full(5, DX), [0.8e-3, 0.6e-3, 0.45e-3],
                    np.full(9, 0.3e-3), [0.45e-3, 0.6e-3, 0.8e-3],
                    np.full(5, DX)],
    "graded_mil": np.r_[np.full(6, 0.508e-3), np.full(10, 0.127e-3),
                        np.full(6, 0.508e-3)],
}


@pytest.mark.parametrize("name", sorted(PREFLIGHT_PROFILES))
def test_preflight_puts_a_half_node_coordinate_on_the_grids_node(name):
    """``profile_node_at`` is preflight's model of where a port is stamped;
    at every half node it must name the node the grid's own lookup picks."""
    from rfx.preflight._common import profile_node_at
    prof = PREFLIGHT_PROFILES[name]
    grid = make_nonuniform_grid((10e-3, 10e-3), prof, float(prof[0]),
                                cpml_layers=8)
    pad = grid.pad_z_lo
    nodes = np.r_[0.0, np.cumsum(prof)]
    grid_nodes = np.r_[0.0, np.cumsum(grid.cells("z")[pad:])]
    mids = 0.5 * (nodes[:-1] + nodes[1:])
    xs = np.r_[mids, np.nextafter(mids, 1), np.nextafter(mids, -1),
               (np.arange(len(prof)) + 0.5) * float(prof[0])]
    moved = 0
    for x in xs[(xs >= 0) & (xs <= nodes[-1])]:
        k = position_to_index(grid, (0.0, 0.0, float(x)))[2] - pad
        assert profile_node_at(float(prof[0]), prof, float(x)) == pytest.approx(
            float(grid_nodes[k]), abs=1e-15), (name, float(x), k)
        moved += int(np.argmin(np.abs(nodes - x))) != k
    # On every profile some half node is one the argmin put on the other node.
    assert moved > 0


@pytest.mark.parametrize("step_above", [0.0, 1.0])
def test_the_graded_node_report_reads_the_node_the_port_is_stamped_on(
        step_above):
    """Three 1 mm cells and then 0.5 mm cells (binary sizes, so the midpoint
    is exact): node 3 is the step. A port midway between node 3 and node 4,
    or one float step above that midpoint, is stamped on node 3, so the
    report names the step. An argmin puts the second one on node 4, where
    there is no step."""
    coarse, fine = 2.0 ** -10, 2.0 ** -11
    dz = np.r_[np.full(3, coarse), np.full(8, fine)]
    sim = Simulation(freq_max=10e9, domain=(8 * coarse, 8 * coarse, 0.0),
                     dx=coarse, dz_profile=dz, boundary="pec")
    z = 3 * coarse + 0.5 * fine
    if step_above:
        z = float(np.nextafter(z, np.inf))
    grid = sim._build_nonuniform_grid()
    assert position_to_index(grid, (0.0, 0.0, z))[2] - grid.pad_z_lo == 3
    report = sim._graded_node_report(2, z)
    assert report is not None
    node_pos, d_below, d_above, _dual, ratio = report
    assert (node_pos, d_below, d_above, ratio) == (3 * coarse, coarse, fine, 2.0)


def _traced_half_nodes_refused(cell, n, profile=None):
    """Count run-time refusals of the traced-axis check over every half node
    of a traced z column (the nominal mesh is ``n`` cells of ``cell``)."""
    import jax
    import jax.numpy as jnp
    if profile is None:
        profile = np.full(n, cell)
    profile = jnp.asarray(profile, dtype=jnp.float32)
    refused = 0
    for z in [s for k in range(n) for s in ((k + 0.5) * cell,
                                            k * cell + 0.5 * cell)]:

        def f(p, z=z):
            grid = make_nonuniform_grid((4 * cell, 4 * cell), p, cell,
                                        cpml_layers=0)
            position_to_index(grid, (0.0, 0.0, z))
            return jnp.sum(p)

        try:
            jax.block_until_ready(jax.jit(f)(profile))
            jax.effects_barrier()
        except Exception as exc:  # the callback's ValueError, as XLA raises it
            assert "on the mesh the run built" in str(exc)
            refused += 1
    return refused


@pytest.mark.parametrize("cell,n", [(5e-4, 20), (0.3e-3, 24), (0.254e-3, 30)])
def test_a_traced_axis_accepts_a_half_node_coordinate(cell, n):
    """On a traced profile equal to its nominal mesh, float32 cumulative
    sums put one of the two nodes of a half node a few ulp nearer, either
    one; the run-time check must not read that as the node having moved.
    The PR review counted refusals at 4 of 40, 28 of 48 and 51 of 60 lookups
    on main, and at 19 of 40, 18 of 48 and 30 of 60 on this PR's first head,
    on these columns (two spellings of each half node)."""
    assert _traced_half_nodes_refused(cell, n) == 0


def test_a_traced_axis_still_refuses_a_node_that_moved():
    """Control: cells 30 % wider than the nominal mesh over the bottom half
    move every node line above it by several cells."""
    cell, n = 5e-4, 20
    profile = np.r_[np.full(10, 1.3 * cell), np.full(10, cell)]
    profile[0] = cell       # the boundary cell stays the nominal one
    assert _traced_half_nodes_refused(cell, n, profile) >= 16
