"""A coordinate midway between two nodes lands on the same node in both lanes (#1295).

A source, probe or port declared exactly halfway between two grid nodes has
two equally near nodes. The uniform ``Grid`` resolves it with
``round(x / dx)``, round-half-to-even on the float quotient; the non-uniform
grid took the first of the two in an ``argmin`` over its node line, i.e.
always the lower node. On 1 mm cells a feed declared at y = 9.5 mm sat on
node 10 in one lane and node 9 in the other, a 1 mm shift of the feed. In the
kernel-timing record (rfx-archive ``records/20260925-nu-kernel-timing``) that
put the 13.4 M-cell probe traces of the two lanes 8e-2 of the peak apart,
against <= 1.4e-5 at the sizes with no half-node source.

What is pinned here:

* on equal cells, every NU lookup (``index_of``, ``position_to_index``,
  ``pos_to_nu_index``) returns the uniform lane's index bit for bit, float
  quotient cases included -- lane A against lane B, both grids built by
  ``Simulation`` from the same declaration;
* on a graded axis, an exact tie goes to the even node, and every coordinate
  that is not an exact tie keeps the node the ``argmin`` gave;
* end to end, the scaled-down kernel-timing box with its source on a half
  node gives the same probe trace on both lanes.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.nonuniform import interior_cells, make_nonuniform_grid, position_to_index
from rfx.runners.nonuniform import pos_to_nu_index

DX = 1e-3
AXES = ("x", "y", "z")

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


@pytest.mark.parametrize("case", sorted(LANE_CASES))
def test_both_lanes_put_every_coordinate_on_the_same_node(case):
    cells, boundary = LANE_CASES[case]
    uniform, nu, domain = _both_lanes(cells, boundary)
    upper_half_ties = 0
    for ax, axis in enumerate(AXES):
        for x in _sweep(cells[ax], domain[ax]):
            pos = [0.0, 0.0, 0.0]
            pos[ax] = float(x)
            want = uniform.index_of(axis, float(x))
            assert uniform.position_to_index(tuple(pos))[ax] == want
            got = (nu.index_of(axis, float(x)),
                   position_to_index(nu, tuple(pos))[ax],
                   pos_to_nu_index(nu, tuple(pos))[ax])
            assert got == (want, want, want), (
                f"{case}/{axis} x={float(x)!r}: uniform lane node {want}, "
                f"NU lane (index_of, position_to_index, pos_to_nu_index) {got}")
            pad = (uniform.pad_x_lo, uniform.pad_y_lo, uniform.pad_z_lo)[ax]
            q = float(x) / DX
            if q - np.floor(q) == 0.5 and want - pad == np.floor(q) + 1:
                upper_half_ties += 1
    # The split only shows where the uniform lane rounds an exact half UP (odd
    # lower node); a sweep without such a coordinate would pass on main.
    assert upper_half_ties > 0


def test_float_quotient_cases_match_the_uniform_lane():
    """The cases the float quotient decides, with the uniform lane's own
    answers: ``0.1195 / 1e-3 = 119.49999999999999`` goes to 119, while the
    kernel-timing source at ``0.5 * (239 * 1e-3)`` gives
    ``119.50000000000001`` and goes to 120. An exact 9.5 goes to the even
    node 10, where the NU lane used to answer 9."""
    uniform, nu, domain = _both_lanes((239, 12, 10), "pec")
    for x, node in ((0.1195, 119), (0.5 * domain[0], 120), (0.0095, 10),
                    (0.0105, 10)):
        pos = (x, 0.0, 0.0)
        assert uniform.position_to_index(pos)[0] == node, x
        assert nu.index_of("x", x) == node, x
        assert position_to_index(nu, pos)[0] == node, x
        assert pos_to_nu_index(nu, pos)[0] == node, x


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
def test_graded_axis_ties_go_to_the_even_node_and_nothing_else_moves(name):
    grid = GRADED[name]()
    ties = moved = 0
    for ax, axis in enumerate(AXES):
        if grid.is_constant(axis):
            continue
        pad_lo = (grid.pad_x_lo, grid.pad_y_lo, grid.pad_z_lo)[ax]
        pad_hi = (grid.pad_x_hi, grid.pad_y_hi, grid.pad_z_hi)[ax]
        nodes = np.insert(
            np.cumsum(interior_cells(grid.cells(axis), pad_lo, pad_hi)), 0, 0.0)
        mids = 0.5 * (nodes[:-1] + nodes[1:])
        xs = np.concatenate([
            nodes, mids, np.nextafter(mids, np.inf), np.nextafter(mids, -np.inf),
            mids + 1e-12, mids - 1e-12, mids + 1e-9, mids - 1e-9,
            np.random.default_rng(ax).uniform(nodes[0], nodes[-1], 400)])
        for x in xs[(xs >= nodes[0]) & (xs <= nodes[-1])]:
            dist = np.abs(nodes - x)
            before = int(np.argmin(dist))  # the lookup this change replaces
            tied = np.flatnonzero(dist == dist[before])
            want = before if tied.size == 1 else int(tied[tied % 2 == 0][0])
            ties += tied.size > 1
            moved += want != before
            pos = [0.0, 0.0, 0.0]
            pos[ax] = float(x)
            got = grid.index_of(axis, float(x))
            assert got == want + pad_lo, (
                f"{name}/{axis} x={float(x)!r}: expected node {want} "
                f"(argmin {before}, tied {tied.tolist()}), got {got - pad_lo}")
            assert position_to_index(grid, tuple(pos))[ax] == got
            assert pos_to_nu_index(grid, tuple(pos))[ax] == got
    # Both kinds of tie occur: one the argmin already sent to the even node,
    # and one it sent to the odd node, which is the one that moves.
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
    assert pos_to_nu_index(grid_b, src) == grid_a.position_to_index(src)
    peak = float(np.max(np.abs(a)))
    assert peak > 0.0
    return src, grid_a.position_to_index(src), float(np.max(np.abs(a - b))) / peak


def test_half_node_source_gives_the_same_trace_on_both_lanes():
    """With 19 cells along y the source is on an exact half node that the
    uniform lane rounds UP, the case the argmin rounded down; with 18 it is
    on a node, the control. On main 057af7a7 the half-node box gave a trace
    gap of 8.3e-2 of the peak, the one-cell feed shift, and the control
    1.5e-5. The two kernels differ by float32 roundoff, so the half-node box
    must come down to the control's level, not to zero."""
    src, node, tie_gap = _lane_gap((20, 19, 17))
    assert src[1] / DX == 9.5 and node[1] == 10
    src, node, control_gap = _lane_gap((20, 18, 17))
    assert src[1] / DX == 9.0 and node[1] == 9
    print(f"trace gap / peak: half-node source {tie_gap:.3e}, "
          f"node source {control_gap:.3e}", file=sys.stderr)
    assert control_gap <= 3e-5
    assert tie_gap <= 3e-5
