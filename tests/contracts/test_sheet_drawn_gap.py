"""Drawn extents gate joins; expected edges come from node coordinates.

K3's main oracle removes ``drawn`` from every spec: None restores the
pre-fix union by construction. K6 field stepping is outside this phase.
"""

from rfx.geometry.rasterize_grid import unpadded_high_walls
from dataclasses import replace
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, PolylineWire, Simulation
from rfx.boundaries.pec import SheetSpec, realized_pec_edge_masks
from rfx.boundaries.spec import BoundarySpec
from rfx.geometry.rasterize_grid import (
    coords_from_uniform_grid, coords_from_nonuniform_grid,
    cell_sizes_from_uniform_grid, cell_sizes_from_nonuniform_grid,
    sheet_spec_from_shape,
)
from rfx.materials.thin_conductor import (
    ThinConductor, apply_thin_conductor, build_sheet_impedance_ctx,
)
from rfx.model import conductors as C
from rfx.model.materials import e_update_material_at, electric_grid_kwargs
from rfx.runners import distributed_v2 as D

PROFILE = np.array([1] + [.9, 1.1] * 9 + [1]) * .001
GAPS = (0., 1e-9, .00005, .0003, .0005, .00095, .001, .0013)


def model(graded=False, periodic=False, domain=(.008, .020, .008)):
    return Simulation(domain=domain, dx=.001, freq_max=5e9, snap='declared',
                      boundary=BoundarySpec(x='pec', y='periodic' if periodic else 'pec', z='pec'),
                      **({'dy_profile': PROFILE} if graded else {}))


@lru_cache(None)
def lattice(graded, periodic):
    sim = model(graded, periodic)
    grid = sim._build_realized_grid()
    coords = (coords_from_nonuniform_grid if graded else coords_from_uniform_grid)(grid)
    sizes = (cell_sizes_from_nonuniform_grid if graded else cell_sizes_from_uniform_grid)(grid)
    materials = C.realized_conductors(sim, grid, nonuniform=graded).materials
    return grid, coords, sizes, materials


def specs(intervals, graded=False, periodic=False, kind='pec'):
    grid, coords, sizes, materials = lattice(graded, periodic)
    result = []
    for lo, hi in intervals:
        shape = Box((.004, lo, .002), (.004, hi, .006))
        if kind == 'pec':
            result.append(sheet_spec_from_shape(shape, coords, sizes, grid=None if graded else grid))
        else:
            apply_thin_conductor(grid, ThinConductor(shape, sigma_bulk=5.8e7,
                thickness=1e-5, surface_impedance_f0=5e9), materials, sheet_specs=result)
    return result


def masks(rows, periodic=False, kind='pec'):
    flags = (False, periodic, False)
    if kind == 'pec':
        return tuple(np.asarray(m) for m in realized_pec_edge_masks(None, rows, periodic=flags))
    ctx = build_sheet_impedance_ctx(rows, periodic=flags)
    return tuple(np.asarray(m) for m in (ctx.mask_ex, ctx.mask_ey, ctx.mask_ez))


def same(actual, expected):
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def coordinate_masks(intervals, graded, periodic):
    """Independent closed-node/segment oracle for the x-normal K1 boxes."""
    grid, coords, _, _ = lattice(graded, periodic)
    y, z = np.asarray(coords.y), np.asarray(coords.z)
    next_y = np.append(y[1:], .020 if periodic else np.inf)
    next_z = np.append(z[1:], np.inf)
    y_nodes, y_edges = np.zeros(y.shape, bool), np.zeros(y.shape, bool)
    for lo, hi in intervals:
        for shift in ((0., .020, -.020) if periodic else (0.,)):
            # Only absorb host-coordinate roundoff, not the drawn-gap tolerance.
            y_nodes |= (y >= lo+shift-1e-12) & (y <= hi+shift+1e-12)
            y_edges |= (y >= lo+shift-1e-12) & (next_y <= hi+shift+1e-12)
    z_nodes = (z >= .002-1e-12) & (z <= .006+1e-12)
    z_edges = (z >= .002-1e-12) & (next_z <= .006+1e-12)
    expected = [np.zeros(grid.shape, bool) for _ in range(3)]
    expected[1][4] = y_edges[:, None] & z_nodes[None, :]
    expected[2][4] = y_nodes[:, None] & z_edges[None, :]
    return expected


@pytest.mark.parametrize('kind', ['pec', 'f0'])
@pytest.mark.parametrize('lane', ['uniform', 'graded', 'seam'])
@pytest.mark.parametrize('k', range(21))
@pytest.mark.parametrize('gap', GAPS)
def test_k1_drawn_gap(kind, lane, k, gap):
    graded, periodic = lane == 'graded', lane == 'seam'
    b = .010 + k * .0000473
    intervals = ((.005, b), (b + gap, .016))
    whole = ((.005, .016),)
    if periodic:
        intervals = ((.014, b + .009), (b + .009 + gap - .020, .005))
        whole = ((.014, .025),)
    pair = specs(intervals, graded, periodic, kind)
    actual = masks(pair, periodic, kind)
    tol = 1e-6 * min(np.asarray(lattice(graded, periodic)[2][1]))
    same(actual, coordinate_masks(whole if gap <= tol else intervals, graded, periodic))
    if gap <= tol:
        same(actual, masks(specs(whole, graded, periodic, kind), periodic, kind))
    else:
        alone = [masks([sp], periodic, kind) for sp in pair]
        same(actual, tuple(a | b for a, b in zip(*alone)))
        # Coordinates supply each sheet's nodes, independently of the rule.
        nodes = np.asarray(lattice(graded, periodic)[1].y)
        own = [np.zeros(nodes.shape, bool) for _ in pair]
        for covered, (lo, hi) in zip(own, intervals):
            for shift in ((0., .020, -.020) if periodic else (0.,)):
                covered |= (nodes >= lo + shift - 1e-12) & (nodes <= hi + shift + 1e-12)
        candidates = own[0][:-1] & own[1][1:] | own[1][:-1] & own[0][1:]
        assert not actual[1][4, :-1, 4][candidates].any()


def named_intervals(graded, case):
    y = np.asarray(lattice(graded, False)[1].y)
    return [((y[5], y[10]), (y[11], y[15])),
            ((y[5], y[10]), ((y[10] + y[11])/2, y[15])),
            ((y[5], y[10]-.1*(y[10]-y[9])), (y[10]+.1*(y[11]-y[10]), y[15])),
            ((y[5], (y[9]+y[10])/2), ((y[9]+y[10])/2, y[15])),
            ((y[5], y[10]), (y[10], y[15]))][case]


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('case,expected', [
    (0, [5, 6, 7, 8, 9, 11, 12, 13, 14]),
    (1, [5, 6, 7, 8, 9, 11, 12, 13, 14]),
    (2, [5, 6, 7, 8, 11, 12, 13, 14]),
    (3, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
    (4, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
])
def test_k2_mid_domain(graded, case, expected):
    actual = masks(specs(named_intervals(graded, case), graded))
    assert np.flatnonzero(actual[1][4, :, 4]).tolist() == expected


@pytest.mark.parametrize('bounds,expected', [
    ((14, 19, 0, 4), [0, 1, 2, 3, 14, 15, 16, 17, 18]),
    ((14, 19, -.5, 4), [0, 1, 2, 3, 14, 15, 16, 17, 18]),
    ((14, 18.9, -.9, 4), [0, 1, 2, 3, 14, 15, 16, 17]),
    ((14, 18.5, -1.5, 4), [0, 1, 2, 3, 14, 15, 16, 17, 18, 19]),
    ((14, 19, -1, 4), [0, 1, 2, 3, 14, 15, 16, 17, 18, 19]),
])
def test_k2_seam(bounds, expected):
    a, b, c, d = np.asarray(bounds)*.001
    actual = masks(specs(((a, b), (c, d)), periodic=True), True)
    assert np.flatnonzero(actual[1][4, :, 4]).tolist() == expected


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('cover', [False, True])
def test_k2_three_sheets(graded, cover):
    y = np.asarray(lattice(graded, False)[1].y)
    intervals = [(y[5], y[10]), (y[11], y[15]),
                 (y[9], y[12]) if cover else (y[16], y[18])]
    expected = ([5, 6, 7, 8, 9, 10, 11, 12, 13, 14] if cover else
                [5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17])
    assert np.flatnonzero(masks(specs(intervals, graded))[1][4, :, 4]).tolist() == expected


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('kind', ['volume', 'wire', 'lumped', 'stack', 'crossing', 'indices'])
def test_k3_unchanged(graded, kind):
    sim = model(graded, kind == 'crossing')
    grid = sim._build_realized_grid()
    y = lambda j: grid.node_of(1, j)
    sim.add(Box((.002, y(5), .004), (.006, y(10), .004)), material='pec')
    if kind == 'volume':
        sim.add(Box((.002, y(11), .004), (.006, y(15), .006)), material='pec')
    elif kind == 'wire':
        sim.add(PolylineWire(((.004, y(11), .004), (.004, y(15), .004)), radius=0), material='pec')
    elif kind == 'lumped':
        sim.add(Box((.002, y(11), .004), (.006, y(15), .004)), material='pec')
        sim.add_port((.004, y(10), .004), component='ey')
    elif kind == 'stack':
        sim.add(Box((.002, y(5), .005), (.006, y(10), .005)), material='pec')
    elif kind == 'crossing':
        sim.add(Box((.002, -.003, .004), (.006, .003, .004)), material='pec')
    root = C.realized_conductors(sim, grid, nonuniform=graded)
    if kind == 'indices':
        rows = [SheetSpec.from_node_ranges(grid.shape, normal_axis=2, plane=4,
                in_plane_ranges=((2, 6), interval)) for interval in ((5, 9), (10, 15))]
        same(masks(rows), masks([replace(sp, drawn=None) for sp in rows]))
        return
    legacy = realized_pec_edge_masks(root.pec_cells,
        [replace(sp, drawn=None) for sp in root.sheets], root.wires, root.periodic)
    old = replace(root, pec_edges=legacy)
    if kind == 'lumped':
        # A one-cell drawn feed gap: the old rule shorted it at every edge but the port's own.
        root, old = C.preview_port_stages(sim, root), C.preview_port_stages(sim, old)
        new_ey, old_ey = np.asarray(root.pec_edges[1]), np.array(old.pec_edges[1])
        assert not new_ey[2:7, 10, 4].any()
        assert old_ey[2:7, 10, 4].tolist() == [True, True, False, True, True]
        old_ey[2:7, 10, 4] = False
        same(root.pec_edges, (old.pec_edges[0], old_ey, old.pec_edges[2]))
        return
    same(root.pec_edges, old.pec_edges)


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('case', range(5))
def test_k7_dc_film(graded, case):
    sim = model(graded)
    for lo, hi in named_intervals(graded, case):
        sim.add_thin_conductor(Box((.002, lo, .004), (.006, hi, .004)),
                               sigma_bulk=1000., thickness=1e-5)
    grid = sim._build_realized_grid()
    root = C.realized_conductors(sim, grid, nonuniform=graded)
    values = [float(e_update_material_at(root.materials, (4, j, 4), 'ey', root.periodic,
              **electric_grid_kwargs(grid), high_wall=unpadded_high_walls(grid))[1]) for j in (9, 10)]
    # Survey's five rows: the first three have a gap at row 10; last two touch. A touching
    # pair is one continuous film: its joint edge carries the film's interior edge value.
    assert values[0] > 0.
    assert values[1] == (0. if case < 3 else values[0])


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('location', ['inside', 'seam'])
@pytest.mark.parametrize('k', range(21))
@pytest.mark.parametrize('gap', [0., .0005])
def test_k4_staged_slabs(monkeypatch, graded, location, k, gap):
    if len(jax.devices('cpu')) < 2:
        pytest.skip('requires two CPU devices')
    sim = model(graded, domain=(.020, .020, .008))
    grid = sim._build_realized_grid()
    per = (grid.shape[0] + 1)//2
    b = (.010 if location == 'seam' else .005) + k*.0000473
    for lo, hi in ((b-.005, b), (b+gap, b+.005)):
        sim.add_thin_conductor(Box((lo, .004, .004), (hi, .015, .004)),
                               sigma_bulk=5.8e7, thickness=1e-5)
    expected = C.solve_conductors(sim, grid, nonuniform=graded).pec_edges
    staged = []
    original = C.stage_distributed_edges if graded else D.cut

    class Staged(Exception):
        pass

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        if graded or args[2] == 'pec_edge':
            staged.extend(result)
            raise Staged
        return result

    monkeypatch.setattr(C if graded else D, 'stage_distributed_edges' if graded else 'cut', capture)
    sim.add_source((.003, .003, .002), 'ez', amplitude_kind='field')
    with pytest.raises(Staged):
        sim.run(n_steps=2, devices=jax.devices('cpu')[:2], compute_s_params=False, skip_preflight=True)
    for actual, reference in zip(staged, expected):
        actual, reference = np.asarray(actual), np.asarray(reference)
        local = actual.shape[0]//2
        oracle = np.zeros((2, local, *actual.shape[1:]), bool)
        for rank in range(2):
            for li in range(local):
                gi = rank*per + li - 1
                if 0 <= gi < grid.shape[0]:
                    oracle[rank, li] = reference[gi]
                elif grid.shape[0] <= gi < 2*per:
                    oracle[rank, li] = True
        np.testing.assert_array_equal(actual, oracle.reshape(actual.shape))
        owned = actual.reshape(2, local, *actual.shape[1:])[:, 1:-1]
        np.testing.assert_array_equal(owned.reshape(-1, *actual.shape[1:])[:grid.shape[0]], reference)


@pytest.mark.parametrize('intervals', [
    ((.005, .010), (.011, .015)), ((.005, .010), (.0105, .015)),
    ((.005, .0095), (.0095, .015)), ((.005, .0094), (.0094, .015)),
    ((.005, .0096), (.0096, .015)), ((.005, .010), (.010, .015)),
    ((.005, .011), (.010, .015)), ((.005, .010), (.0115, .015)),
    ((.005, .010), (.012, .015)), ((.003, .006), (.007, .010), (.011, .014)),
    ((.011, .015), (.005, .010)), ((.005, .0099), (.0101, .015)),
])
@pytest.mark.parametrize('route', ['thin', 'add', 'mixed'])
def test_realization_stopgap_cases(intervals, route):
    sim = model(domain=(.020, .020, .020))
    for i, (lo, hi) in enumerate(intervals):
        box = Box((.010, lo, .005), (.010, hi, .015))
        if route == 'add' or route == 'mixed' and i == 1:
            sim.add(box, material='pec')
        else:
            sim.add_thin_conductor(box, sigma_bulk=5.8e7, thickness=35e-6)
    grid = sim._build_realized_grid()
    root = C.realized_conductors(sim, grid)
    y = np.array([grid.node_of(1, j) for j in range(grid.shape[1])])
    # Independent interval segments: an edge is wholly covered by a single
    # connected drawn interval. Merge touching/overlapping metric intervals.
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([lo, hi])
    expected = np.zeros(len(y), bool)
    for lo, hi in merged:
        expected[:-1] |= (y[:-1] >= lo-1e-12) & (y[1:] <= hi+1e-12)
    np.testing.assert_array_equal(root.pec_edges[1][10, :, 10], expected)


@pytest.mark.parametrize('cover', [(.009, .012), (.010, .011), (.004, .016), 'partial'])
def test_realization_cover(cover):
    sim = model(domain=(.020, .020, .020))
    for lo, hi in ((.005, .010), (.011, .015)):
        sim.add(Box((.010, lo, .005), (.010, hi, .015)), material='pec')
    lo, hi = (.009, .012) if cover == 'partial' else cover
    sim.add(Box((.010, lo, .005), (.010, hi, .010 if cover == 'partial' else .015)), material='pec')
    root = C.realized_conductors(sim, sim._build_realized_grid())
    assert np.flatnonzero(np.asarray(root.pec_edges[1])[10, 10]).tolist() == (
        [5, 6, 7, 8, 9, 10] if cover == 'partial' else list(range(5, 16)))


@pytest.mark.parametrize('normal,tangent', [(0, 1), (1, 2), (2, 0)])
@pytest.mark.parametrize('different_plane', [False, True])
def test_realization_axes(normal, tangent, different_plane):
    grid, coords, sizes, _ = lattice(False, False)
    rows = []
    for i, (lo, hi) in enumerate(((.001, .003), (.004, .006))):
        low, high = [.001]*3, [.006]*3
        low[normal] = high[normal] = .003 + (i*.001 if different_plane else 0.)
        low[tangent], high[tangent] = lo, hi
        rows.append(sheet_spec_from_shape(Box(tuple(low), tuple(high)), coords, sizes, grid=grid))
    actual = masks(rows)
    same(actual, tuple(a | b for a, b in zip(masks(rows[:1]), masks(rows[1:]))))


def test_realization_swapped_box_remains_refused():
    grid, coords, sizes, _ = lattice(False, False)
    with pytest.raises(ValueError, match='footprint rasterizes to ZERO nodes'):
        sheet_spec_from_shape(Box((.004, .015, .006), (.004, .011, .002)), coords, sizes, grid=grid)


def test_traced_bounds_touch_without_python_branch():
    from rfx.geometry.rasterize_grid import sheet_footprint_traced
    from rfx.materials.thin_conductor import SheetImpedanceSpec
    _, coords, sizes, _ = lattice(False, False)

    @jax.jit
    def joined(gap):
        rows = []
        for lo, hi in ((.005, .0104), (.0104+gap, .015)):
            mask, drawn = sheet_footprint_traced(
                Box((.004, lo, .002), (.004, hi, .006)), coords, 0, sizes, with_drawn=True)
            rows.append(SheetImpedanceSpec(mask, 0, 1., mask.astype(float), drawn=drawn))
        return build_sheet_impedance_ctx(rows).mask_ey[4, 10, 4]

    assert bool(joined(0.))
    assert not bool(joined(.0005))


def test_realization_former_seam_case():
    sim = model(periodic=True, domain=(.020, .020, .020))
    for lo, hi in ((.015, .020), (.001, .005)):
        sim.add_thin_conductor(Box((.010, lo, .005), (.010, hi, .015)),
                               sigma_bulk=5.8e7, thickness=35e-6)
    root = C.realized_conductors(sim, sim._build_realized_grid())
    assert np.flatnonzero(np.asarray(root.pec_edges[1])[10, :, 10]).tolist() == [
        1, 2, 3, 4, 15, 16, 17, 18, 19]


def test_realization_different_normals():
    sim = model(domain=(.020, .020, .020))
    sim.add(Box((.010, .005, .005), (.010, .010, .015)), material='pec')
    sim.add(Box((.005, .011, .005), (.015, .011, .015)), material='pec')
    root = C.realized_conductors(sim, sim._build_realized_grid())
    expected = tuple(a | b for a, b in zip(masks(root.sheets[:1]), masks(root.sheets[1:])))
    same(root.pec_edges, expected)


@pytest.mark.parametrize('graded', [False, True])
def test_drawn_carrier_and_copies(graded):
    sim = model(graded)
    sim.add(Box((.004, .005, .002), (.004, .010, .006)), material='pec')
    root = C.realized_conductors(sim, sim._build_realized_grid(), nonuniform=graded)
    sp, = root.sheets
    assert sp.drawn[0] is None
    assert sp.drawn[1][:2] == (.005, .010)
    assert all(isinstance(value, float) for row in sp.drawn if row is not None for value in row)
    assert sp.drawn[1][2] == 1e-6 * min(lattice(graded, False)[2][1])
    assert replace(sp, footprint=sp.footprint).drawn == sp.drawn
    interior = sim._campaign_ctx().interior_pec_entries()
    assert interior[0].sheet.drawn == sp.drawn


@pytest.mark.parametrize('kind', ['pec', 'f0'])
@pytest.mark.parametrize('case', ['gap', 'touch', 'different_planes'])
def test_f1_2d_run(kind, case):
    sim = Simulation(freq_max=30e9, domain=(.020, .020, .001), dx=.001,
                     boundary='pec', mode='2d_tmz')
    for i, (lo, hi) in enumerate(((.003, .008), (.008 if case == 'touch' else .009, .015))):
        x = .015 if case == 'different_planes' and i else .010
        shape = Box((x, lo, 0), (x, hi, .001))
        if kind == 'pec':
            sim.add(shape, material='pec')
        else:
            sim.add_thin_conductor(shape, sigma_bulk=5.8e7, surface_impedance_f0=15e9)
    sim.add_source((.006, .010, 0), 'ez', amplitude_kind='field')
    sim.add_probe((.007, .012, 0), 'ez')
    grid = sim._build_realized_grid()
    root = C.realized_conductors(sim, grid)
    if kind == 'pec':
        edges = realized_pec_edge_masks(None, root.sheets, periodic=(False, False, True))
    else:
        ctx = build_sheet_impedance_ctx(root.sheet_specs, periodic=(False, False, True))
        edges = (ctx.mask_ex, ctx.mask_ey, ctx.mask_ez)
    assert bool(edges[1][10, 8, 0]) == (case == 'touch')
    sim.run(n_steps=3, skip_preflight=True)


def test_f1_tfsf_forward():
    sim = Simulation(freq_max=10e9, domain=(.030, .006, .006), dx=.001,
                     boundary='cpml', cpml_layers=8, mode='3d')
    for x in (.012, .018):
        sim.add(Box((x, -1, -1), (x, 1, 1)), material='pec')
    sim.add_tfsf_source(f0=5e9, bandwidth=.3, amplitude=1., polarization='ez', direction='+x')
    sim.add_probe((.008, .003, .003), component='ez')
    sim.forward(n_steps=3, skip_preflight=True)


@pytest.mark.parametrize('touch', [False, True])
def test_f1_tfsf_wrap_masks(touch):
    from rfx.boundaries.tfsf import boundary_flags
    sim = Simulation(freq_max=10e9, domain=(.030, .006, .006), dx=.001,
                     boundary='cpml', cpml_layers=8)
    sim.add_tfsf_source(f0=5e9, bandwidth=.3, polarization='ez', direction='+x')
    grid = sim._build_realized_grid()
    coords, sizes = coords_from_uniform_grid(grid), cell_sizes_from_uniform_grid(grid)
    # TFSF wraps the stored padded index line, not just the physical domain.
    period = grid.shape[1] * grid.dx
    y = np.asarray(coords.y)
    split = .5 * grid.dx if touch else 0.
    intervals = ((y[-4], y[-1] + split), (y[0] - split, y[3]))
    rows = [sheet_spec_from_shape(Box((.012, lo, .002), (.012, hi, .004)),
            coords, sizes, grid=grid) for lo, hi in intervals]
    from rfx.sources.tfsf import init_tfsf
    cfg, _ = init_tfsf(grid.nx, grid.dx, grid.dt, ny=grid.ny, nz=grid.nz,
                       cpml_layers=8, f0=5e9, bandwidth=.3, polarization='ez', direction='+x')
    flags, _ = boundary_flags(cfg, grid)
    assert flags == (False, True, True)
    assert 'y' not in grid.periodic_axes
    assert rows[0].drawn[1][3] == pytest.approx(period, rel=1e-14)
    assert y[-1] + grid.dx == pytest.approx(y[0] + period, rel=1e-14)
    edges = realized_pec_edge_masks(None, rows, periodic=flags)
    ix, iz = int(np.flatnonzero(np.isclose(coords.x, .012))[0]), int(np.flatnonzero(np.isclose(coords.z, .003))[0])
    assert bool(edges[1][ix, -1, iz]) == touch


def test_f1_declared_periodic_carrier():
    grid, _, _, _ = lattice(False, True)
    row, = specs(((.005, .010),), periodic=True)
    for axis in grid.periodic_axes:
        t = 'xyz'.index(axis)
        assert row.drawn[t][3] == pytest.approx(grid.domain[t], rel=1e-14)


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('case,ends', [('gap', (True, True)), ('touch', (False, False)),
                                      ('off_mesh_touch', (False, False)), ('indices', (False, False))])
def test_f1_solved_spans(graded, case, ends):
    from rfx.mesh_edges import solved_sheet_span
    sim = model(graded)
    grid = sim._build_realized_grid()
    y = np.asarray(lattice(graded, False)[1].y)
    split = .0095 if case == 'off_mesh_touch' else .010
    intervals = ((.005, split), (.011 if case == 'gap' else split, .015))
    if case == 'indices':
        rows = [SheetSpec.from_node_ranges(grid.shape, normal_axis=0, plane=4,
                in_plane_ranges=(interval, (2, 6))) for interval in ((5, 10), (11, 15))]
        from rfx.boundaries.pec import _sheet_edge_masks
        edges = _sheet_edge_masks(rows, grid.shape, (False, False, False))
        spans = [solved_sheet_span(row.footprint, 1, y, lo, hi, .020, union=edges)
                 for row, (lo, hi) in zip(rows, ((y[5], y[10]), (y[11], y[15])))]
    else:
        for lo, hi in intervals:
            sim.add(Box((.004, lo, .002), (.004, hi, .006)), material='pec')
        root = C.realized_conductors(sim, grid, nonuniform=graded)
        spans = [sheet.solved_spans[1] for sheet in root.sheets]
    assert (spans[0].free_hi, spans[1].free_lo) == ends
