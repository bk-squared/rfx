"""Per-slab setup producers equal the cut of the whole producer, bit for bit.

The contract of :mod:`rfx.stepping.slab_view` (rfx #1402, W1 staging). A
multi-device run realizes the model per x slab; the one-device run realizes it
on the whole grid. Both must hand the time step the same numbers, so every
grid-shaped producer is run here on the whole grid and on each slab view (two
and three devices), and each slab's owned rows are compared byte for byte with
the same rows of the whole result. No tolerance: a difference is a stencil the
halo does not cover, an outside-the-lattice rule applied at an interior seam,
or an arithmetic path chosen from the slab instead of the whole axis.

The scenes put a material change at every seam (independent random cells),
give the x axis one stretch of equal cells inside a graded axis (so a slab can
look uniform while the axis is not), and leave the last slab of three short.

To plug in a producer, append a ``Producer`` to ``PRODUCERS``: ``whole(scene)``
returns its result on the whole grid, ``slab(scene, view)`` its result on one
view, taking its inputs with ``view.rows`` and its global values from
``view.globals``. Results are pytrees; every array leaf whose ``x_axis`` has
the grid's x length is cut, every other leaf must be equal as a whole.
"""
from dataclasses import fields, replace
from types import SimpleNamespace
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation, Box, DebyePole
from rfx.boundaries.pec import SheetSpec, WireSpec, realized_pec_edge_masks, wire_path_edge_masks
from rfx.core.yee import MaterialArrays, cell_owned_component_materials, component_h_materials, edge_averaged_materials
from rfx.materials.lorentz import lorentz_pole
from rfx.model.materials import (ComponentCells, electric_cell_sizes, pole_component_weights,
                                 realize_components, with_components, validate_components)
from rfx.model.pole_stability import (CourantBudget, pole_set_extremes, combine_pole_extremes,
                                     judge_pole_sets, refuse_unstable_cells, courant_budget)
from rfx.model.occupancy import build_edge_keep, publish_occupancy, with_edge_keep
from rfx.stepping.slab_view import slab_views
from rfx.model.source_coefficients import drive_columns as _slab_drive_columns
from tests._multi_device_scene import host_devices

PERIODIC = (False, False, False)
DEVICES = (2, 3)


class Scene(NamedTuple):
    name: str
    grid: object
    eps_r: np.ndarray
    sigma: np.ndarray
    mu_r: np.ndarray
    occupancy: np.ndarray
    pec_mask: np.ndarray
    debye: object
    lorentz: object
    globals: dict


def _build(name):
    nx, ny, nz = 18, 6, 5
    # Equal cells over the first 11, a graded stretch after (the end cell
    # must equal the boundary dx): the first slab of two or three devices is
    # all equal cells inside a graded axis.
    dx = np.full(nx, 1e-3)
    dx[11:15] = 1e-3 * np.array([.9, .7, .6, .8])
    dz = np.full(nz, 1e-3)
    dz[:2] *= .8
    cpml = name == "cpml"
    sim = Simulation(freq_max=15e9, domain=(float(dx.sum()), ny * 1e-3, float(dz.sum())),
                     dx=1e-3, dx_profile=dx, dz_profile=dz,
                     boundary="cpml" if cpml else "pec", cpml_layers=3 if cpml else 0)
    if name == "poles":
        sim.add_material("disp", eps_r=3.0, debye_poles=[DebyePole(1., 1e-11)],
                         lorentz_poles=[lorentz_pole(1., 2 * np.pi * 3e9, 1e9)])
        sim.add(Box((0, 0, 0), (float(dx.sum()), ny * 1e-3, float(dz.sum()))), material="disp")
    grid = sim._build_nonuniform_grid()
    materials, debye, lorentz, _ = sim._assemble_materials_nu(grid)
    rng = np.random.default_rng(1402 + len(name))
    shape = tuple(int(s) for s in grid.shape)
    eps = rng.uniform(1., 6., shape).astype(np.float32)
    sig = rng.uniform(0., .1, shape).astype(np.float32)
    mu = rng.uniform(1., 2., shape).astype(np.float32)
    occ = np.where(rng.random(shape) < .4, rng.uniform(0., 1., shape), 0.).astype(np.float32)
    pec = rng.random(shape) < .05

    def random_masks(spec):
        if spec is None:
            return None
        poles, masks = spec
        return poles, [rng.random(shape) < .6 for _ in masks]

    return Scene(name, grid, eps, sig, mu, occ, pec, random_masks(debye), random_masks(lorentz),
                 {"dt": grid.dt})


SCENES = {name: _build(name) for name in ("material_seam", "cpml", "poles")}


def _assert_realized_seams(scene):
    """Every seam separates different cells and has conductor on both sides;
    the last of three slabs is short (the runtime pads it)."""
    for n in DEVICES:
        views = slab_views(scene.grid, n)
        for view in views[1:]:
            assert np.any(scene.eps_r[view.x0 - 1] != scene.eps_r[view.x0]), (scene.name, n, view.x0)
            assert scene.occupancy[view.x0 - 1].any() and scene.occupancy[view.x0].any(), (scene.name, n)
        if n == 3:
            assert views[-1].x1 - views[-1].x0 < views[0].x1 - views[0].x0, (scene.name, views)


class Producer(NamedTuple):
    name: str
    whole: Callable
    slab: Callable
    scenes: tuple = tuple(SCENES)
    x_axis: int = 0


def _x_widths(scene):
    # The global decision: None only if the WHOLE x axis has equal cells.
    return electric_cell_sizes(scene.grid)


def _cut_widths(widths, view):
    return (None if widths[0] is None else view.rows(np.asarray(widths[0])),) + tuple(widths[1:])


def _h_widths(grid):
    out = []
    for d, n in zip((grid.dx_arr, grid.dy_arr, grid.dz), grid.shape):
        d = np.asarray(d)
        out.append(np.concatenate((d, d[-1:])) if d.shape[0] == n - 1 else d)
    return tuple(out)


def _mu(view_or_none, scene):
    mu = scene.mu_r if view_or_none is None else view_or_none.rows(scene.mu_r)
    return SimpleNamespace(mu_r=jax.numpy.asarray(mu), mu_r_wire=None)


_CONDUCTORS = {}


def _conductors(scene):
    if scene.name in _CONDUCTORS:
        return _CONDUCTORS[scene.name]
    shape = scene.grid.shape
    nx, ny, nz = shape
    seams = {n: [v.x0 for v in slab_views(scene.grid, n)[1:]] for n in DEVICES}
    release_rows = set(np.asarray(R)[:, 0])
    assert all({s - 1, s} <= release_rows for rows in seams.values() for s in rows)
    sheets = [SheetSpec.from_node_ranges(
        shape, normal_axis=2, plane=1, in_plane_ranges=((1, nx - 2), (1, 2)))]
    for plane, seam in enumerate(seams[3], start=3):
        pair = [SheetSpec.from_node_ranges(
            shape, normal_axis=2, plane=plane, in_plane_ranges=ranges)
            for ranges in (((1, seam - 1), (1, 3)), ((seam, nx - 2), (2, 4)))]
        # At this row neither individual sheet owns Ex; their union joins.
        joined = realized_pec_edge_masks(None, pair, periodic=PERIODIC)[0]
        assert bool(joined[seam - 1, 2, plane])
        assert all(not bool(realized_pec_edge_masks(None, [sp], periodic=PERIODIC)[0][seam - 1, 2, plane])
                   for sp in pair)
        sheets.extend(pair)
    for n in DEVICES:
        for plane in (seams[n][0], seams[n][0] - 1):
            sheets.append(SheetSpec.from_node_ranges(
                shape, normal_axis=0, plane=plane,
                in_plane_ranges=((1, ny - 2), (1, nz - 2))))
    wires = tuple(WireSpec(edges=wire_path_edge_masks(nodes, shape)) for nodes in (
        ((1, 0, 0), (nx - 2, 0, 0)),
        ((seams[3][0], 0, 1), (seams[3][0], 0, nz - 2))))
    sheet_masks = realized_pec_edge_masks(None, sheets, periodic=PERIODIC)
    for seam in set(seams[2] + seams[3]):
        assert np.asarray(sheet_masks[0][seam - 1]).any(), (scene.name, seam)
    _CONDUCTORS[scene.name] = tuple(sheets), wires
    return _CONDUCTORS[scene.name]


class _Mats(NamedTuple):
    eps_r: object
    edge_keep: object = None


# Global cells on both sides of the layouts' seams, plus exterior rows.
R = jnp.asarray([(1, 0, 2), (5, 0, 2), (6, 0, 2), (7, 0, 2),
                 (8, 0, 2), (9, 0, 2), (10, 0, 2), (11, 0, 2),
                 (12, 0, 2), (13, 0, 2), (14, 0, 2), (16, 0, 2),
                 (17, 0, 2), (18, 0, 2)], dtype=jnp.int32)


def _cells(scene, view=None):
    take = (lambda x: x) if view is None else view.rows
    materials = MaterialArrays(*(jax.numpy.asarray(take(x)) for x in
                                (scene.eps_r, scene.sigma, scene.mu_r)))
    def cut(spec):
        if spec is None:
            return None
        poles, masks = spec
        return poles, [None if m is None else take(m) for m in masks]
    return ComponentCells(materials, cut(scene.debye), cut(scene.lorentz))


def _component_numbers(components):
    # Identity fields cells_view/grid_key are excluded. All other fields are
    # compared, including static periodic/source_upml and pole tuples/dt.
    # ADE coefficients store (pole, x, y, z): expose x first to the harness.
    out = {}
    for field in fields(components):
        if field.name in ("cells_view", "grid_key"):
            continue
        value = getattr(components, field.name)
        if field.name in ("debye", "lorentz") and value is not None:
            value = value._replace(coefficients=jax.tree.map(
                lambda a: jax.numpy.moveaxis(a, 1, 0), value.coefficients))
        out[field.name] = value
    return out


def _realized_whole(scene):
    return _component_numbers(realize_components(_cells(scene), scene.grid, periodic=PERIODIC))


def _realized_slab(scene, view):
    # Model the staging owner: judge whole-model reductions before granting
    # the token; the producer itself may only realize the input slab rows.
    whole = _cells(scene)
    refuse_unstable_cells(whole.materials, whole.lorentz_spec, scene.grid.dt,
                          courant_budget(scene.grid), grid_kind=type(scene.grid).__name__)
    view = view._replace(globals={**view.globals, "pole_sets_judged": True})
    return _component_numbers(realize_components(_cells(scene, view), scene.grid,
                                                 periodic=PERIODIC, view=view))


_DRIVE_SOURCE_CELLS = {}


def _drive_sources(scene):
    """Asymmetric seam, material-interior and partial-occupancy source edges."""
    if scene.name in _DRIVE_SOURCE_CELLS:
        return _DRIVE_SOURCE_CELLS[scene.name]
    seam = slab_views(scene.grid, 3)[1].x0
    keep = np.asarray(build_edge_keep(jnp.asarray(scene.occupancy))[2])
    candidates = np.argwhere((keep[11:16, 1:-1, 1:-1] > 0.) &
                            (keep[11:16, 1:-1, 1:-1] < 1.) &
                            (keep[11:16, 1:-1, 1:-1] * 16 % 1 != 0))
    assert len(candidates), 'scene needs a non-dyadic occupancy edge'
    partial = tuple(int(a + b) for a, b in zip(candidates[0], (11, 1, 1)))
    _DRIVE_SOURCE_CELLS[scene.name] = ((1, (seam, 2, 2)), (0, (3, 3, 2)), (2, partial))
    return _DRIVE_SOURCE_CELLS[scene.name]


def _drive_operands(scene):
    from rfx.nonuniform import current_source_volume
    cells = _drive_sources(scene)
    volumes = tuple(current_source_volume(scene.grid, cell, ('ex', 'ey', 'ez')[axis])[0]
                    for axis, cell in cells)
    samples = jnp.asarray([[.13, .23, .37], [.71, .89, .59],
                           [1.37, 1.19, 1.73], [.83, .47, .31]], jnp.float32)
    return cells, volumes, samples


def _drive_cb(components, cells, dt):
    from rfx.materials.debye import debye_e_coeffs
    from rfx.materials.lorentz import lorentz_e_coeffs, mixed_e_component_coeffs
    coefficients = []
    for family, builder in (('debye', debye_e_coeffs), ('lorentz', lorentz_e_coeffs)):
        pole = components[family]
        terms = None if pole is None else jax.tree.map(
            lambda a: jnp.moveaxis(a, 0, 1), pole.coefficients)
        coefficients.append(None if terms is None else builder(
            (components['eps_update'], components['sigma_update']), dt, *terms))
    d, l = coefficients
    if d is None and l is None:
        return None
    arrays = (tuple(mixed_e_component_coeffs(d, l, axis, dt)[1] for axis in range(3))
              if d is not None and l is not None else (d if d is not None else l).cb)
    return jnp.stack([arrays[axis][cell] for axis, cell in cells])


def _drive_whole(scene, *, with_keep=True):
    from rfx.model.source_coefficients import drive_columns
    components = _realized_whole(scene)
    cells, volumes, samples = _drive_operands(scene)
    keep = build_edge_keep(jnp.asarray(scene.occupancy)) if with_keep else None
    return drive_columns(components['eps_update'], components['sigma_update'], keep,
        cells, jnp.ones(len(cells), dtype=bool), samples, volumes, scene.grid.dt,
        cb=_drive_cb(components, cells, scene.grid.dt))


def _drive_slab(scene, view, *, with_keep=True):
    components = _realized_slab(scene, view)
    global_cells, volumes, samples = _drive_operands(scene)
    if isinstance(view, _UniformHarnessView):
        lo = view.indices[0]
        x0, x1 = view.indices[view.halo_lo], view.indices[-view.halo_hi - 1] + 1
    else:
        lo, x0, x1 = view.lo, view.x0, view.x1
    cells = tuple((axis, (i - lo, j, k)) for axis, (i, j, k) in global_cells)
    owned = jnp.asarray([(x0 <= i) & (i < x1) for _, (i, j, k) in global_cells])
    keep = build_edge_keep(view.rows(scene.occupancy), view=view) if with_keep else None
    return _slab_drive_columns(components['eps_update'], components['sigma_update'], keep,
        cells, owned, samples, volumes, scene.grid.dt,
        cb=_drive_cb(components, cells, scene.grid.dt))


def _assert_drive_columns(scene, n, expected, actual):
    expected = np.asarray(expected)
    sources = _drive_sources(scene)
    for view, got in zip(slab_views(scene.grid, n), actual):
        got = np.asarray(got)
        owned = np.asarray([view.x0 <= cell[0] < view.x1 for _, cell in sources])
        assert got.shape == expected.shape and got.dtype == expected.dtype == np.float32
        assert got[:, owned].tobytes() == expected[:, owned].tobytes(), 'drive owned columns differ'
        assert np.all(got[:, ~owned] == 0), 'drive non-owned columns are nonzero'


# Reference entries: the stencil functions under the realized-materials and
# occupancy producers. Owners add the producers themselves (realize_components,
# build_edge_keep with sheets, realized_pec_edge_masks, drive_table, CPML state).
PRODUCERS = [
    Producer('drive_table', _drive_whole, _drive_slab, ('material_seam', 'poles')),
    Producer("realize_components", _realized_whole, _realized_slab),
    Producer(
        "edge_averaged_materials",
        lambda s: edge_averaged_materials(s.eps_r, s.sigma, PERIODIC, cell_sizes=_x_widths(s)),
        lambda s, v: edge_averaged_materials(v.rows(s.eps_r), v.rows(s.sigma), PERIODIC,
                                             cell_sizes=_cut_widths(_x_widths(s), v))),
    Producer(
        "component_h_materials",
        lambda s: component_h_materials(_mu(None, s), PERIODIC, cell_sizes=_h_widths(s.grid)),
        lambda s, v: component_h_materials(_mu(v, s), PERIODIC, cell_sizes=(
            v.rows(_h_widths(s.grid)[0]),) + _h_widths(s.grid)[1:])),
    Producer(
        "pole_component_weights",
        lambda s: [pole_component_weights(m, PERIODIC, cell_sizes=_x_widths(s)) for m in s.debye[1]],
        lambda s, v: [pole_component_weights(v.rows(m), PERIODIC, cell_sizes=_cut_widths(_x_widths(s), v))
                      for m in s.debye[1]],
        scenes=("poles",)),
    Producer(
        "build_edge_keep",
        lambda s: build_edge_keep(s.occupancy, periodic=PERIODIC),
        lambda s, v: build_edge_keep(v.rows(s.occupancy), periodic=PERIODIC)),
    Producer(
        "realized_pec_edge_masks",
        lambda s: realized_pec_edge_masks(s.pec_mask, *_conductors(s), PERIODIC),
        lambda s, v: realized_pec_edge_masks(v.rows(s.pec_mask), *_conductors(s), PERIODIC, view=v)),
    Producer(
        "realized_pec_edge_masks_sheets_only",
        lambda s: realized_pec_edge_masks(None, *_conductors(s), PERIODIC),
        lambda s, v: realized_pec_edge_masks(None, *_conductors(s), PERIODIC, view=v)),
    Producer(
        "publish_occupancy",
        lambda s: publish_occupancy(_Mats(s.eps_r), s.occupancy, dtype=jnp.float32,
            periodic=PERIODIC, sheets=_conductors(s)[0], wires=_conductors(s)[1], released_cells=R).edge_keep,
        lambda s, v: publish_occupancy(_Mats(v.rows(s.eps_r)), v.rows(s.occupancy), dtype=jnp.float32,
            periodic=PERIODIC, sheets=_conductors(s)[0], wires=_conductors(s)[1], released_cells=R, view=v).edge_keep),
    Producer(
        "with_edge_keep_static",
        lambda s: with_edge_keep(_Mats(s.eps_r), s.occupancy, dtype=jnp.float16, periodic=PERIODIC,
            sheet_edge_masks=realized_pec_edge_masks(None, *_conductors(s), PERIODIC)).edge_keep,
        lambda s, v: with_edge_keep(_Mats(v.rows(s.eps_r)), v.rows(s.occupancy), dtype=jnp.float16,
            periodic=PERIODIC, view=v,
            sheet_edge_masks=realized_pec_edge_masks(None, *_conductors(s), PERIODIC, view=v)).edge_keep),
]


def assert_slab_view(producer, scene, n_devices, **view_kwargs):
    if producer.name == 'drive_table':
        expected = producer.whole(scene)
        actual = [producer.slab(scene, view) for view in slab_views(scene.grid, n_devices, **view_kwargs)]
        _assert_drive_columns(scene, n_devices, expected, actual)
        return
    whole = jax.tree.leaves(producer.whole(scene))
    nx = int(scene.grid.nx)
    for view in slab_views(scene.grid, n_devices, globals=scene.globals, **view_kwargs):
        local = jax.tree.leaves(producer.slab(scene, view))
        assert len(local) == len(whole), (producer.name, view.rank)
        for leaf, (w, s) in enumerate(zip(whole, local)):
            w, s = np.asarray(w), np.asarray(s)
            if w.ndim > producer.x_axis and w.shape[producer.x_axis] == nx:
                w = np.take(w, np.arange(view.x0, view.x1), axis=producer.x_axis)
                s = view.own(s, producer.x_axis)
            label = (producer.name, scene.name, n_devices, view.rank, leaf)
            assert w.shape == s.shape and w.dtype == s.dtype, (label, w.shape, s.shape, w.dtype, s.dtype)
            if w.tobytes() != s.tobytes():
                rows = np.unique(np.nonzero(w != s)[producer.x_axis]) + view.x0
                raise AssertionError(f"{label}: differs at global x rows {rows.tolist()}")


_CASES = [pytest.param(p, SCENES[s], n, id=f"{p.name}-{s}-{n}")
          for p in PRODUCERS for s in p.scenes for n in DEVICES]


@pytest.mark.parametrize("scene", list(SCENES.values()), ids=list(SCENES))
def test_scene_has_a_material_change_at_every_seam(scene):
    _assert_realized_seams(scene)


@pytest.mark.parametrize("producer,scene,n_devices", _CASES)
def test_slab_view_equals_cut_of_whole(producer, scene, n_devices):
    assert_slab_view(producer, scene, n_devices)


@pytest.mark.parametrize("name", ["edge_averaged_materials", "component_h_materials", "build_edge_keep",
                                 "realized_pec_edge_masks", "publish_occupancy"])
def test_mutation_missing_halo_is_caught(name):
    """Without the low-x halo the first owned row takes the outside rule."""
    producer = next(p for p in PRODUCERS if p.name == name)
    with pytest.raises(AssertionError, match="differs at global x rows"):
        assert_slab_view(producer, SCENES["material_seam"], 2, halo_lo=0)


def test_mutation_slab_local_metric_decision_is_caught():
    """Deciding 'equal cells' on the slab instead of the axis changes the bits."""
    def slab(s, v):
        x = v.rows(np.asarray(_x_widths(s)[0]))
        local = (None if np.all(x == x[0]) else x,) + tuple(_x_widths(s)[1:])
        return edge_averaged_materials(v.rows(s.eps_r), v.rows(s.sigma), PERIODIC, cell_sizes=local)
    producer = next(p for p in PRODUCERS if p.name == "edge_averaged_materials")._replace(slab=slab)
    with pytest.raises(AssertionError, match="differs at global x rows"):
        assert_slab_view(producer, SCENES["material_seam"], 2)


@pytest.mark.parametrize("n_devices", DEVICES)
def test_mutation_sheet_forward_row_is_caught(monkeypatch, n_devices):
    import rfx.boundaries.pec as pec
    monkeypatch.setattr(pec, "_SHEET_X_READ_AHEAD", 0)
    producer = next(p for p in PRODUCERS if p.name == "realized_pec_edge_masks_sheets_only")
    with pytest.raises(AssertionError, match="differs at global x rows"):
        assert_slab_view(producer, SCENES["material_seam"], n_devices)


@pytest.mark.parametrize("n_devices", DEVICES)
def test_mutation_released_cells_not_shifted_is_caught(monkeypatch, n_devices):
    import rfx.model.occupancy as occupancy
    monkeypatch.setattr(occupancy, "_local_released",
                        lambda cells, view: cells[(cells[:, 0] >= view.lo) & (cells[:, 0] < view.hi)])
    producer = next(p for p in PRODUCERS if p.name == "publish_occupancy")
    with pytest.raises(AssertionError, match="differs at global x rows"):
        assert_slab_view(producer, SCENES["material_seam"], n_devices)


def test_view_refuses_periodic_x():
    scene = SCENES["material_seam"]
    view = slab_views(scene.grid, 2)[0]
    with pytest.raises(ValueError, match="periodic x"):
        realized_pec_edge_masks(view.rows(scene.pec_mask), periodic=(True, False, False), view=view)
    with pytest.raises(NotImplementedError, match="periodic x"):
        build_edge_keep(view.rows(scene.occupancy), periodic=(True, False, False), view=view)


def test_view_refuses_design():
    scene = SCENES["material_seam"]
    view = slab_views(scene.grid, 2)[0]
    with pytest.raises(NotImplementedError, match="design"):
        build_edge_keep(view.rows(scene.occupancy), design=object(), view=view)


def test_view_refuses_traced_x_sheet_plane():
    scene = SCENES["material_seam"]
    view = slab_views(scene.grid, 2)[0]
    sheet = next(s for s in _conductors(scene)[0] if s.normal_axis == 0)

    def traced(plane):
        copy = replace(sheet)
        object.__setattr__(copy, "plane", plane)
        return realized_pec_edge_masks(None, [copy], view=view)

    with pytest.raises(ValueError, match="traced sheet plane"):
        jax.jit(traced)(jnp.asarray(sheet.plane))


def test_view_without_local_conductors_keeps_shape_dtype_and_container():
    scene = SCENES["material_seam"]
    view = slab_views(scene.grid, 3)[0]
    sheet = SheetSpec.from_node_ranges(scene.grid.shape, normal_axis=0, plane=view.nx - 2,
                                       in_plane_ranges=((1, 3), (1, 3)))
    whole = realized_pec_edge_masks(None, [sheet])
    local = realized_pec_edge_masks(None, [sheet], view=view)
    assert type(local) is type(whole)
    for w, m in zip(whole, local):
        assert type(w) is type(m) and w.dtype == m.dtype == jnp.bool_
        assert m.shape == (view.hi - view.lo,) + tuple(scene.grid.shape[1:])
        assert not np.asarray(m).any()
    with pytest.raises(ValueError, match="no cell mask, sheets or wires"):
        realized_pec_edge_masks(None, view=view)


def test_view_cuts_sheet_metadata():
    import rfx.boundaries.pec as pec
    scene = SCENES["material_seam"]
    view = slab_views(scene.grid, 3)[1]
    sheet = _conductors(scene)[0][0]
    sheet = replace(sheet, unwrapped_footprint=sheet.footprint,
                    end_rows=(np.arange(view.nx) % 2 == 0, None, None),
                    drawn=((.001, .017, 1e-9), (.001, .002, 1e-9), None))
    end = min(view.hi + 1, view.nx)
    cut, = pec._slab_sheets([sheet], view, end)
    assert cut.drawn is sheet.drawn
    for name in ("footprint", "unwrapped_footprint"):
        np.testing.assert_array_equal(getattr(cut, name), getattr(sheet, name)[view.lo:end])
    np.testing.assert_array_equal(cut.end_rows[0], sheet.end_rows[0][view.lo:end])
    whole = realized_pec_edge_masks(None, [sheet])
    local = realized_pec_edge_masks(None, [sheet], view=view)
    for w, m in zip(whole, local):
        assert np.asarray(w[view.x0:view.x1]).tobytes() == np.asarray(view.own(m)).tobytes()


@pytest.mark.parametrize("released", [jnp.empty((0, 3), dtype=jnp.int32),
                                    jnp.asarray([[0, 0, 0]], dtype=jnp.int32)])
def test_view_without_local_releases(released):
    scene = SCENES["material_seam"]
    view = slab_views(scene.grid, 3)[-1]
    cells = view.rows(scene.occupancy)
    actual = build_edge_keep(cells, released_cells=released, view=view)
    expected = build_edge_keep(cells)
    for a, e in zip(actual, expected):
        assert np.asarray(a).tobytes() == np.asarray(e).tobytes()


@pytest.mark.parametrize("released", [jnp.zeros((1, 3)), jnp.zeros((0, 2), dtype=jnp.int32)])
def test_view_validates_global_releases(released):
    scene = SCENES["material_seam"]
    view = slab_views(scene.grid, 3)[-1]
    with pytest.raises(ValueError, match="integer array of shape"):
        build_edge_keep(view.rows(scene.occupancy), released_cells=released, view=view)


def test_view_refuses_a_one_row_slab_and_gives_an_empty_slab_no_rows():
    """A one-row slab would wrap in the neighbour rules (length-1 axis = 2-D lane)."""
    grid = SimpleNamespace(nx=3)
    with pytest.raises(ValueError, match="single x row"):
        slab_views(grid, 3)
    views = slab_views(SimpleNamespace(nx=9), 4)   # ceil layout: owned rows 3, 3, 3, 0
    assert views[-1].x0 == views[-1].x1 and views[-1].hi == views[-1].lo
    assert [v.hi - v.lo for v in views] == [3, 4, 4, 0]


def _refusing_scene():
    # Fine cells/high epsilon in the first slab; low epsilon on coarse cells
    # in the last. The whole pattern combines their extrema and occupied axes.
    shape = (18, 6, 5)
    widths = (np.r_[np.full(9, .25e-3), np.full(9, 1e-3)],
              np.full(6, 1e-3), np.full(5, 1e-3))
    budget = CourantBudget(widths)
    dt = .9 / np.sqrt(float(budget))
    grid = SimpleNamespace(nx=18, shape=shape, dt=dt, cells=lambda a: widths[a])
    eps = np.full(shape, 4., np.float32)
    eps[15:] = 1.
    sigma = np.full(shape, .1, np.float32)
    sigma[15:] = 0.
    mask = np.zeros(shape, bool)
    mask[:3] = True
    mask[15:] = True
    return Scene("refusing", grid, eps, sigma, np.ones(shape, np.float32),
                 np.zeros(shape), np.zeros(shape, bool), None,
                 ([(0., 0., 1. / dt**2)], [mask]), {"courant_budget": budget})


def _assert_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


def _assert_records(whole, combined):
    assert len(whole) == len(combined)
    for a, b in zip(whole, combined):
        assert a.code == b.code
        assert a.count == b.count, "cell count"
        for name in ("poles", "eps_low", "sigma_low", "sigma_high", "occupied"):
            aa, bb = jax.tree.leaves(getattr(a, name)), jax.tree.leaves(getattr(b, name))
            assert len(aa) == len(bb)
            for x, y in zip(aa, bb):
                _assert_bits(x, y)


def _records(scene, n):
    whole = _cells(scene)
    slabs = []
    for view in slab_views(scene.grid, n):
        cells = _cells(scene, view)
        slabs.append(pole_set_extremes(cells.materials, cells.lorentz_spec, view=view))
    return pole_set_extremes(whole.materials, whole.lorentz_spec), slabs


def _verdict(call):
    try:
        call()
    except ValueError as error:
        return str(error)
    return None


@pytest.mark.parametrize("n", DEVICES)
@pytest.mark.parametrize("name", ("poles", "refusing"))
def test_pole_extremes_and_refusal_equal_whole(name, n):
    scene = SCENES[name] if name == "poles" else _refusing_scene()
    whole, slabs = _records(scene, n)
    combined = combine_pole_extremes(slabs)
    _assert_records(whole, combined)
    cells = _cells(scene)
    budget = courant_budget(scene.grid)
    kwargs = dict(grid_kind=type(scene.grid).__name__)
    expected = _verdict(lambda: refuse_unstable_cells(cells.materials, cells.lorentz_spec,
                                                     scene.grid.dt, budget, **kwargs))
    assert _verdict(lambda: judge_pole_sets(combined, scene.grid.dt, budget, **kwargs)) == expected
    if name == "refusing":
        assert expected is not None
        mask = cells.lorentz_spec[1][0]
        eps, sigma = np.asarray(cells.materials.eps_r), np.asarray(cells.materials.sigma)
        eps_rows = np.where(np.any(mask & (eps == eps[mask].min()), axis=(1, 2)))[0]
        sigma_rows = np.where(np.any(mask & (sigma == sigma[mask].max()), axis=(1, 2)))[0]
        views = slab_views(scene.grid, n)
        eps_slabs = {v.rank for v in views if np.any((eps_rows >= v.x0) & (eps_rows < v.x1))}
        sigma_slabs = {v.rank for v in views if np.any((sigma_rows >= v.x0) & (sigma_rows < v.x1))}
        assert eps_slabs.isdisjoint(sigma_slabs)
        for records in slabs:
            assert _verdict(lambda: judge_pole_sets(records, scene.grid.dt, budget, **kwargs)) is None
    else:
        assert expected is None


def test_slab_lorentz_requires_global_judgement_and_view_identity():
    scene = SCENES["poles"]
    view, other = slab_views(scene.grid, 2)
    cells = _cells(scene, view)
    with pytest.raises(ValueError, match="pole_set_extremes / combine_pole_extremes / judge_pole_sets"):
        realize_components(cells, scene.grid, periodic=PERIODIC, view=view)
    view = view._replace(globals={"pole_sets_judged": True})
    ready = with_components(cells.materials, scene.grid, periodic=PERIODIC, view=view,
                            debye_spec=cells.debye_spec, lorentz_spec=cells.lorentz_spec)
    assert with_components(ready, scene.grid, periodic=PERIODIC, view=view) is ready
    for wrong in (None, other, view._replace(halo_hi=1)):
        with pytest.raises(ValueError, match="grid differs"):
            validate_components(ready, periodic=PERIODIC, grid=scene.grid, view=wrong)


def test_upml_is_pointwise_with_stamps_without_halo():
    scene = SCENES["material_seam"]
    materials = _cells(scene).materials
    stamp = tuple(materials.sigma * f for f in (.1, .2, .3))
    materials = materials._replace(eps_r_lumped=stamp, sigma_lumped=stamp, sigma_film=stamp)
    whole = cell_owned_component_materials(materials)
    for view in slab_views(scene.grid, 3, halo_lo=0):
        local = jax.tree.map(view.rows, materials)
        for a, b in zip(jax.tree.leaves(whole), jax.tree.leaves(cell_owned_component_materials(local))):
            _assert_bits(view.rows(a), b)


def test_realized_mutation_missing_halo_is_caught():
    producer = next(p for p in PRODUCERS if p.name == "realize_components")
    with pytest.raises(AssertionError, match="differs at global x rows"):
        assert_slab_view(producer, SCENES["material_seam"], 2, halo_lo=0)


def test_realized_mutation_local_metric_is_caught(monkeypatch):
    import rfx.model.materials as model
    original = model.edge_averaged_materials
    def wrong(*args, cell_sizes, **kwargs):
        x = cell_sizes[0]
        if x is not None and np.all(np.asarray(x) == x[0]):
            cell_sizes = (None,) + cell_sizes[1:]
        return original(*args, cell_sizes=cell_sizes, **kwargs)
    monkeypatch.setattr(model, "edge_averaged_materials", wrong)
    producer = next(p for p in PRODUCERS if p.name == "realize_components")
    with pytest.raises(AssertionError, match="differs at global x rows"):
        assert_slab_view(producer, SCENES["material_seam"], 2)


def test_slab_realization_refuses_x_periodic():
    """A slab's own rows cannot supply the wrapped x neighbour (global row nx-1)."""
    scene = SCENES["material_seam"]
    view = slab_views(scene.grid, 2)[0]
    with pytest.raises(ValueError, match="x-periodic"):
        realize_components(_cells(scene, view), scene.grid,
                           periodic=(True, False, False), view=view)


def test_extremes_mutation_halo_count_is_caught(monkeypatch):
    from rfx.stepping.slab_view import SlabView
    # Return all input rows and expand the matching ownership range: the
    # numerical reduction succeeds but duplicate halo cells must be caught.
    monkeypatch.setattr(SlabView, "own", lambda self, a, axis=0: a)
    scene = SCENES["poles"]
    whole = pole_set_extremes(_cells(scene).materials, scene.lorentz)
    slabs = []
    for view in slab_views(scene.grid, 2):
        cells = _cells(scene, view)
        wrong = view._replace(x0=view.lo, halo_lo=0)
        slabs.append(pole_set_extremes(cells.materials, cells.lorentz_spec, view=wrong))
    with pytest.raises(AssertionError, match="cell count"):
        _assert_records(whole, combine_pole_extremes(slabs))


class _UniformHarnessView:
    """Bind pre-stacked scene inputs; metric gathers use the runtime view."""
    def __init__(self, view, inputs):
        self.view, self.inputs = view, inputs

    def __getattr__(self, key):
        return getattr(self.view, key)

    def _replace(self, **kwargs):
        return type(self)(self.view._replace(**kwargs), self.inputs)

    def rows(self, value, axis=0):
        for original, local in self.inputs:
            if value is original:
                return local
        return self.view.rows(value, axis=axis)


def _uniform_producer(producer, scene, n, *, missing_extra=False, zero_material=False,
                      local_metric=False, metadata_through=False, negative_release=False, releases=R):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from rfx.stepping.slab import Slab
    from rfx.stepping.slab_view import (SlabView, UniformSlabView,
                                      uniform_slab_rows, stack_and_place, uniform_apply)
    layout = Slab(int(scene.grid.nx), n)
    rows = uniform_slab_rows(layout)
    if missing_extra:
        rows = type(rows)(*(a[:, 1:] for a in rows))
    mesh = Mesh(np.asarray(host_devices(n)), ('x',))
    originals = [scene.eps_r, scene.sigma, scene.mu_r, scene.occupancy, scene.pec_mask]
    for spec in (scene.debye, scene.lorentz):
        if spec is not None:
            originals.extend(spec[1])
    arrays = [stack_and_place(a, rows, mesh,
              outside=0 if index in (3, 4) or zero_material else 'replicate')
              for index, a in enumerate(originals)]
    static, released = [], []
    sheets, wires = _conductors(scene)
    for rank, indices in enumerate(rows.indices):
        lo, hi = max(0, int(indices[0])), min(layout.nx, int(indices[-1]) + 1)
        view = SlabView(rank, n, lo, hi, 0, 0, scene.grid, {})
        masks = realized_pec_edge_masks(None, sheets, wires, PERIODIC, view=view)
        static.append(tuple(np.pad(np.asarray(a),
            ((lo - int(indices[0]), int(indices[-1]) + 1 - hi), (0, 0), (0, 0))) for a in masks))
        cells = np.asarray(releases).copy()
        cells[:, 0] -= indices[0]
        cells[(cells[:, 0] < 0) | (cells[:, 0] >= len(indices)), 0] = -1 if negative_release else len(indices)
        released.append(cells)
    static = tuple(jax.device_put(jnp.asarray(np.stack([a[c] for a in static])),
                                 NamedSharding(mesh, P('x'))) for c in range(3))
    released = jax.device_put(jnp.asarray(np.stack(released)), NamedSharding(mesh, P('x')))
    indices = jax.device_put(jnp.asarray(rows.indices), NamedSharding(mesh, P('x')))

    def local(index, inputs, masks, release):
        globals = dict(scene.globals, pole_sets_judged=True,
                       pec_static_edge_masks=masks, released_cells=release)
        view = UniformSlabView(index, scene.grid, globals)
        view = _UniformHarnessView(view, list(zip(originals, inputs)))
        if local_metric:
            # Deliberate mutation: the first slab has constant x metrics.
            widths = view.rows(np.asarray(_x_widths(scene)[0]))
            uniform = jnp.all(widths == widths[0])
            graded = edge_averaged_materials(inputs[0], inputs[1], PERIODIC,
                    cell_sizes=(widths,) + tuple(_x_widths(scene)[1:]))
            equal = edge_averaged_materials(inputs[0], inputs[1], PERIODIC,
                    cell_sizes=(None,) + tuple(_x_widths(scene)[1:]))
            return jax.tree.map(lambda a, b: jnp.where(uniform, a, b), equal, graded)
        return producer.slab(scene, view)
    apply = (lambda f, *args, **kw: jax.vmap(f)(*args)) if metadata_through else uniform_apply
    count = _drive_operands(scene)[2].shape[0] if producer.name == 'drive_table' else rows.indices.shape[1]
    return rows.indices, apply(local, indices, arrays, static, released,
                              mesh=mesh, row_count=count, row_axis=producer.x_axis)


def _assert_uniform(producer, scene, n, **kwargs):
    if producer.name == 'drive_table':
        expected = producer.whole(scene)
        _, actual = _uniform_producer(producer, scene, n, **kwargs)
        _assert_drive_columns(scene, n, expected, actual)
        return
    rows, actual = _uniform_producer(producer, scene, n, **kwargs)
    expected = jax.tree.leaves(producer.whole(scene))
    leaves = jax.tree.leaves(actual)
    assert len(expected) == len(leaves)
    for leaf, (want, got) in enumerate(zip(expected, leaves)):
        if not isinstance(want, (jax.Array, np.ndarray)) or np.ndim(want) == 0:
            assert type(want) is type(got) and want == got, 'uniform metadata differs'
            continue
        want, got = np.asarray(want), np.asarray(got)
        for rank in range(n):
            if want.ndim > producer.x_axis and want.shape[producer.x_axis] == int(scene.grid.nx):
                valid = (rows[rank] >= 0) & (rows[rank] < int(scene.grid.nx))
                if not kwargs.get('missing_extra'):
                    valid[0] = False
                w = np.take(want, rows[rank, valid], axis=producer.x_axis)
                s = np.take(got[rank], np.flatnonzero(valid), axis=producer.x_axis)
            else:
                w, s = want, got[rank]
            label = (producer.name, scene.name, n, rank, leaf)
            assert w.shape == s.shape and w.dtype == s.dtype, label
            assert w.tobytes() == s.tobytes(), f'{label}: uniform rows differ'


@pytest.mark.parametrize('producer,scene,n_devices', _CASES)
def test_uniform_producer_equals_whole_on_all_real_rows(producer, scene, n_devices):
    _assert_uniform(producer, scene, n_devices)


@pytest.mark.parametrize('mode', ('view', 'uniform'))
@pytest.mark.parametrize('mutation', ('M1', 'M2', 'M3'))
def test_drive_column_mutations_are_caught(monkeypatch, mode, mutation):
    import inspect
    import sys
    from functools import partial
    from rfx.model.source_coefficients import drive_columns
    producer = next(p for p in PRODUCERS if p.name == 'drive_table')
    if mutation == 'M1':
        # Without occupancy, step 3 alone must suppress the finite dummy Cb.
        # Keep both calls unchanged between the control and mutated checks.
        producer = producer._replace(whole=partial(_drive_whole, with_keep=False),
                                     slab=partial(_drive_slab, with_keep=False))
    check = assert_slab_view if mode == 'view' else _assert_uniform
    check(producer, SCENES['material_seam'], 3)
    source = inspect.getsource(drive_columns)
    if mutation == 'M1':
        source = source.replace('scale = jnp.where(owned[n], coefficient / volume, 0.)',
                                'scale = coefficient / volume')
    elif mutation == 'M2':
        source = source.replace('column = scale * samples[:, n]',
            'column = (scale * jnp.where(owned[n], keep[axis][cell], 0.)) * samples[:, n]')
        source = source.replace('if keep is not None:', 'if keep is not None and volume is None:')
    else:
        source = source.replace('if volume is None:',
            'cell = (cell[0] - 1, cell[1], cell[2])\n        if volume is None:', 1)
    namespace = {}
    exec(compile(source, '<drive-column-mutation>', 'exec'), namespace)
    monkeypatch.setattr(sys.modules[__name__], '_slab_drive_columns', namespace['drive_columns'])
    with pytest.raises(AssertionError, match='drive .* columns'):
        check(producer, SCENES['material_seam'], 3)


@pytest.mark.parametrize('mutation', ('missing_extra', 'zero_material', 'local_metric'))
def test_uniform_mutations_are_caught(mutation):
    producer = next(p for p in PRODUCERS if p.name == 'edge_averaged_materials')
    with pytest.raises(AssertionError, match='uniform rows differ'):
        _assert_uniform(producer, SCENES['material_seam'], 2, **{mutation: True})


@pytest.mark.parametrize('scene', list(SCENES.values()), ids=list(SCENES))
def test_cell_and_static_pec_masks_combine_by_or(scene):
    sheets, wires = _conductors(scene)
    both = realized_pec_edge_masks(scene.pec_mask, sheets, wires, PERIODIC)
    cells = realized_pec_edge_masks(scene.pec_mask, periodic=PERIODIC)
    static = realized_pec_edge_masks(None, sheets, wires, PERIODIC)
    for want, cell, fixed in zip(both, cells, static):
        _assert_bits(want, cell | fixed)


def test_uniform_metadata_mutation_is_caught():
    producer = next(p for p in PRODUCERS if p.name == 'realize_components')
    with pytest.raises(AssertionError, match='uniform metadata differs'):
        _assert_uniform(producer, SCENES['poles'], 2, metadata_through=True)


@pytest.mark.parametrize('negative', (False, True))
def test_uniform_absent_release_uses_out_of_bounds_sentinel(negative):
    # The release exists only in slab 0. A -1 sentinel would clear the last
    # input row of slab 1, which is its real high ghost, rather than dropping.
    scene = SCENES['material_seam']
    release = jnp.asarray([[1, 2, 2]], dtype=jnp.int32)
    occupancy = np.full(scene.grid.shape, .5, dtype=np.float32)
    scene = scene._replace(occupancy=occupancy)
    producer = Producer('released_keep',
        lambda s: build_edge_keep(s.occupancy, released_cells=release),
        lambda s, v: build_edge_keep(v.rows(s.occupancy), released_cells=release, view=v))
    if negative:
        with pytest.raises(AssertionError, match='uniform rows differ'):
            _assert_uniform(producer, scene, 3, releases=release, negative_release=True)
    else:
        _assert_uniform(producer, scene, 3, releases=release)


@pytest.mark.parametrize('n_devices', DEVICES)
def test_cut_drives_without_sources_keeps_rank_shards(n_devices):
    """No source: the per-rank table is (n_steps, 0); every shard is one rank's (n_steps, 1, 0)."""
    from jax.sharding import Mesh
    from rfx.model.electric_metrics import stage_forward_materials
    from rfx.simulation import ProbeSpec
    from rfx.stepping.slab import Slab
    from rfx.stepping.slab_setup import build_setup, finish, stage_pec
    if len(jax.devices('cpu')) < n_devices:
        pytest.skip(f'needs {n_devices} host devices')
    scene = SCENES['material_seam']
    grid, n_steps = scene.grid, 3
    mesh = Mesh(np.asarray(jax.devices('cpu')[:n_devices]), ('x',))
    layout = Slab(grid.nx, n_devices)
    material = stage_forward_materials(_cells(scene).materials, layout, mesh, uniform=True)
    setup = build_setup(grid, material, layout, mesh, n_steps, [], [ProbeSpec(4, 2, 2, 'ez')],
                        pec_mask=stage_pec(None, (), (), grid, layout, mesh))
    result, _ = finish(setup.invariants, setup.carry_init, layout, mesh,
                       waveforms=setup.src_waveforms, dt=grid.dt)
    waves = result['drives'].waves
    assert {s.data.shape for s in waves.addressable_shards} == {(n_steps, 1, 0)}
    assert np.asarray(waves).shape == (n_steps, n_devices, 0)


def _two_rank_drives(n_devices):
    from jax.sharding import Mesh
    from rfx.core.drives import drive_layout
    from rfx.stepping.slab import Slab
    if len(jax.devices('cpu')) < n_devices:
        pytest.skip(f'needs {n_devices} host devices')
    layout = Slab(SCENES['material_seam'].grid.nx, n_devices)
    mesh = Mesh(np.asarray(jax.devices('cpu')[:n_devices]), ('x',))
    # Source 0 on rank 0, source 1 on the last rank.
    meta = [(1, 2, 2, 'ez'), (layout.nx - 2, 2, 2, 'ez')]
    return drive_layout(meta, jnp.float32), layout, mesh


@pytest.mark.parametrize('n_devices', DEVICES)
def test_cut_drives_per_rank_refuses_a_wave_read_on_two_ranks(n_devices):
    """A rank block holds only its own sources: a wave read by rows on two ranks would read zeros."""
    from rfx.stepping.slab_wrap import cut_drives
    drives, layout, mesh = _two_rank_drives(n_devices)
    shared = drives._replace(wave_id={'ez': np.asarray([0, 0], np.int32)})
    table = jnp.ones((3, n_devices * 2), jnp.float32)
    with pytest.raises(ValueError, match='waves read on several ranks'):
        cut_drives(shared, layout, mesh, waveforms=table, per_rank=True)
    cut_drives(drives, layout, mesh, waveforms=table, per_rank=True)


@pytest.mark.parametrize('n_devices', DEVICES)
def test_cut_drives_form_is_declared_not_guessed(n_devices):
    """A per-rank table of the wrong width raises instead of taking the replicated branch."""
    from rfx.stepping.slab_wrap import cut_drives
    drives, layout, mesh = _two_rank_drives(n_devices)
    with pytest.raises(ValueError, match='expected n_devices \\* n_src'):
        cut_drives(drives, layout, mesh, waveforms=jnp.ones((3, 2), jnp.float32), per_rank=True)
    # The replicated form (the cut-of-whole caller) is selected by the keyword, not the width.
    replicated = cut_drives(drives, layout, mesh, waveforms=jnp.ones((3, n_devices * 2), jnp.float32)[:, :2])
    assert np.asarray(replicated.waves).shape[0] == 3
