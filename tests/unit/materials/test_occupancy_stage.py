"""Published occupancy factors match the unmodified lane operators.

The numerical oracle below owns its four-cell product; it never calls a
product mask/weight helper. Ghost values are not part of the slab contract.
"""
import inspect
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from rfx.core.yee import MaterialArrays, init_state
from rfx.model.occupancy import (
    build_edge_keep, publish_occupancy, publish_slab_occupancy,
)


def reference(occupancy, periodic=(False, False, False)):
    cells = np.clip(np.asarray(occupancy, np.float64), 0., 1.)
    result = []
    for c in range(3):
        axes = [a for a in range(3) if a != c]
        product = np.ones(cells.shape, np.float64)
        for da in (0, 1):
            for db in (0, 1):
                value = cells.copy()
                for axis, shift in zip(axes, (da, db)):
                    if shift:
                        value = np.roll(value, 1, axis=axis)
                        if not periodic[axis] and cells.shape[axis] != 1:
                            face = [slice(None)] * 3
                            face[axis] = 0
                            value[tuple(face)] = 0.
                product *= 1. - value
        result.append(product)
    return result


def material(shape):
    return MaterialArrays(jnp.ones(shape), jnp.zeros(shape), jnp.ones(shape))


@pytest.mark.parametrize('periodic', [(False, False, False), (True, False, True)])
@pytest.mark.parametrize('binary', [False, True])
@pytest.mark.parametrize('shape', [(7, 5, 9), (6, 4, 1)])
def test_independent_four_cell_product(shape, binary, periodic):
    cells = np.random.default_rng(194).uniform(-.1, 1.1, shape).astype(np.float32)
    if binary:
        cells = (cells > .5).astype(np.float32)
    actual = build_edge_keep(jnp.asarray(cells), periodic=periodic)
    for got, expected in zip(actual, reference(cells, periodic)):
        if binary:
            np.testing.assert_array_equal(got, expected)
        else:
            np.testing.assert_allclose(got, expected, rtol=0, atol=2.4e-7)


@pytest.mark.parametrize('lane', ['uniform', 'graded'])
@pytest.mark.parametrize('sheet', [False, True])
def test_published_matches_lane_operator(lane, sheet, monkeypatch):
    from rfx import Simulation, Box
    import rfx.simulation as sm
    import rfx.nonuniform as nu
    module, setup = (sm, '_build_step_setup') if lane == 'uniform' else (nu, '_build_nu_scan')
    kw = {} if lane == 'uniform' else dict(dy_profile=np.array([.001, .0009, .0011, .001, .001, .001]))
    sim = Simulation(freq_max=15e9, domain=(.008, .006, .004), dx=.001, boundary='pec', **kw)
    if sheet:
        sim.add(Box((.004, .001, .001), (.004, .005, .003)), material='pec')
    sim.add_source((.002, .003, .002), 'ez')
    sim.add_probe((.006, .004, .002), 'ez')
    grid = sim._build_grid() if lane == 'uniform' else sim._build_nonuniform_grid()
    cells = jnp.asarray(np.random.default_rng(32).uniform(0, .7, grid.shape).astype(np.float32))
    published = []; applied = []
    old_setup = getattr(module, setup)
    def record_setup(*a, **k):
        b = inspect.signature(old_setup).bind_partial(*a, **k).arguments
        # No run publishes the weight yet (its consumers are unswitched); build
        # it here from the arguments this lane's setup really receives.
        built = publish_occupancy(
            b['materials'], b['pec_occupancy'], periodic=b.get('periodic') or (False,) * 3,
            sheets=b.get('pec_sheets') or (), wires=b.get('pec_wires') or (),
            edge_masks=b.get('pec_edge_masks'))
        published.append(tuple(np.asarray(v) for v in built.edge_keep))
        return old_setup(*a, **k)
    # The uniform step body lives in rfx.stepping since S3-2 stage A.
    import rfx.stepping.uniform as step_uniform
    step_module = step_uniform if lane == 'uniform' else module
    old_apply = step_module.apply_pec_occupancy
    def record_apply(st, *a, **k):
        one = st._replace(ex=jnp.ones_like(st.ex), ey=jnp.ones_like(st.ey), ez=jnp.ones_like(st.ez))
        result = old_apply(one, *a, **k)
        jax.debug.callback(lambda *v: applied.append(tuple(np.asarray(x) for x in v)), result.ex, result.ey, result.ez)
        return old_apply(st, *a, **k)
    monkeypatch.setattr(module, setup, record_setup)
    monkeypatch.setattr(step_module, 'apply_pec_occupancy', record_apply)
    sim.forward(n_steps=3, pec_occupancy_override=cells, skip_preflight=True, checkpoint=False).time_series.block_until_ready()
    jax.effects_barrier()
    assert published and applied
    for row in applied:
        for expected, actual in zip(row, published[0]):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('n_devices', [2, 3])
def test_slab_owned_rows_match_operator_and_reference(n_devices):
    if len(jax.devices()) < n_devices:
        pytest.skip('requires three virtual CPU devices configured before import')
    from rfx.runners.distributed_nu import _apply_pec_occupancy_nu_shmap
    from rfx.runners._rank import mesh_ranks
    from rfx.stepping.slab import Slab, cut
    shape = (11, 7, 5)
    cells = np.random.default_rng(202).uniform(0, .85, shape).astype(np.float32)
    cells[0, 2:5, 1:4] = .5
    layout = Slab(shape[0], n_devices)
    cells[layout.nx_per_rank-1:layout.nx_per_rank+1, 1:4, 2:4] = .75
    mesh = Mesh(np.asarray(jax.devices()[:n_devices]), ('x',))
    slabs = cut(jnp.asarray(cells), layout, 'pec_occupancy', mesh=mesh)
    result = publish_slab_occupancy(material(slabs.shape), slabs, mesh)
    st = init_state(slabs.shape)
    ones = jnp.ones(slabs.shape)
    st = st._replace(ex=ones, ey=ones, ez=ones)
    effective = _apply_pec_occupancy_nu_shmap(st, slabs, mesh, n_devices, layout.nx_local, ranks=mesh_ranks(mesh))
    refs = reference(cells)
    for component, keep, ref in zip((effective.ex, effective.ey, effective.ez), result.edge_keep, refs):
        for rank, (lo, hi) in enumerate(layout.owned):
            owned = slice(rank * layout.nx_local + 1, rank * layout.nx_local + 1 + hi - lo)
            np.testing.assert_array_equal(np.asarray(keep)[owned], np.asarray(component)[owned])
            np.testing.assert_allclose(np.asarray(keep)[owned], ref[lo:hi], rtol=0, atol=2.4e-7)


def test_released_cells_and_empty_are_exact():
    cells = jnp.full((7, 5, 9), .5)
    released = jnp.array([[0, 1, 2], [3, 2, 5]], dtype=jnp.int32)
    result = build_edge_keep(cells, released_cells=released)
    cleared = np.asarray(cells).copy()
    cleared[tuple(np.asarray(released).T)] = 0.
    for actual, expected in zip(result, reference(cleared)):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(build_edge_keep(cells, released_cells=jnp.empty((0, 3), jnp.int32)), build_edge_keep(cells)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('bad', [jnp.zeros((0, 2), jnp.int32), jnp.zeros((2, 3), jnp.float32)])
def test_released_cells_require_integer_triples(bad):
    with pytest.raises(ValueError, match='integer array of shape'):
        build_edge_keep(jnp.zeros((3, 4, 5)), released_cells=bad)


def test_absent_input_publishes_none():
    mat = material((4, 5, 6))
    assert publish_occupancy(mat, None).edge_keep is None
    assert build_edge_keep(None, sheet_edge_masks=(jnp.ones((4, 5, 6)),)*3) is None


def test_design_replaces_the_preweight_window_and_is_traced():
    from rfx.simulation import DesignOccupancySpec
    from rfx.boundaries.pec import apply_pec_occupancy, apply_pec_occupancy_box, pec_occupancy_box_keep
    shape = (7, 6, 5); bounds = (2, 4, 1, 4, 2, 3)
    background = jnp.full(shape, .25); x = jnp.full((2, 3, 1), .5)
    def f(v):
        return build_edge_keep(background, design=DesignOccupancySpec(bounds, v))
    one = jnp.ones(shape); st = init_state(shape)._replace(ex=one, ey=one, ez=one)
    write, keep = pec_occupancy_box_keep(bounds, x, shape=shape, dtype=jnp.float32, pec_occupancy=background)
    expected = apply_pec_occupancy_box(apply_pec_occupancy(st, background), st, write, keep)
    for a, b in zip(f(x), (expected.ex, expected.ey, expected.ez)):
        np.testing.assert_array_equal(a, b)
    merged = np.asarray(background).copy()
    merged[2:4, 1:4, 2:3] = np.asarray(x)
    for actual, independent in zip(f(x), reference(merged)):
        np.testing.assert_allclose(actual, independent, rtol=0, atol=2.4e-7)
    grad = jax.grad(lambda v: sum(jnp.sum(a) for a in f(v)))(x)
    assert np.isfinite(grad).all() and np.any(np.asarray(grad) != 0)


def test_one_publication_builder_definition():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[3] / 'rfx'
    definitions = [(str(p.relative_to(root)), n.lineno)
                   for p in root.rglob('*.py')
                   for n in ast.walk(ast.parse(p.read_text()))
                   if isinstance(n, ast.FunctionDef) and n.name == 'build_edge_keep']
    assert len(definitions) == 1
    assert definitions[0][0] == 'model/occupancy.py'
