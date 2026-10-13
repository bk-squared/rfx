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
from types import SimpleNamespace
from typing import Callable, NamedTuple

import jax
import numpy as np
import pytest

from rfx import Simulation, Box, DebyePole
from rfx.core.yee import component_h_materials, edge_averaged_materials
from rfx.materials.lorentz import lorentz_pole
from rfx.model.materials import electric_cell_sizes, pole_component_weights
from rfx.model.occupancy import build_edge_keep
from rfx.stepping.slab_view import slab_views

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


# Reference entries: the stencil functions under the realized-materials and
# occupancy producers. Owners add the producers themselves (realize_components,
# build_edge_keep with sheets, realized_pec_edge_masks, drive_table, CPML state).
PRODUCERS = [
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
]


def assert_slab_view(producer, scene, n_devices, **view_kwargs):
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


@pytest.mark.parametrize("name", ["edge_averaged_materials", "component_h_materials", "build_edge_keep"])
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
    producer = PRODUCERS[0]._replace(slab=slab)
    with pytest.raises(AssertionError, match="differs at global x rows"):
        assert_slab_view(producer, SCENES["material_seam"], 2)
