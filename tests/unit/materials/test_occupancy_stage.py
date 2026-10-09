"""The published occupancy factor: one value in every compile context.

The factor is prod(1 - o) over the four cells of an E edge, written as one
subtraction per cell followed by multiplies only. ``reference32`` below is
the same association in NumPy float32 — IEEE arithmetic with no compiler —
and the builder must equal it BITWISE eager, under jit, inside a scan and per
slab. ``reference`` is the float64 product, the independent value. Against
the steps' own in-loop formers (``1 - M``, whose rounding depends on where
they are compiled) the bar is one float32 ULP of 1.0. Ghost values are not
part of the slab contract.
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


ULP1 = float(np.spacing(np.float32(1.0)))       # 1.19e-7; the two forms differ by at most half of it per stage


def reference32(occupancy, periodic=(False, False, False), sheets=None):
    """((1-o)(1-o_t1)) * ((1-o_t2)(1-o_t1t2)), t1 < t2, in NumPy float32."""
    occ = np.clip(np.asarray(occupancy, np.float32), np.float32(0), np.float32(1))

    def back(arr, axis):
        out = np.roll(arr, 1, axis=axis)
        if not periodic[axis] and arr.shape[axis] != 1:
            face = [slice(None)] * 3
            face[axis] = 0
            out[tuple(face)] = np.float32(0)
        return out
    one = np.float32(1)
    result = []
    for c in range(3):
        t1, t2 = (t for t in range(3) if t != c)
        b1 = back(occ, t1)
        k = ((one - occ) * (one - b1)).astype(np.float32) * (
            (one - back(occ, t2)) * (one - back(b1, t2))).astype(np.float32)
        k = k.astype(np.float32)
        if sheets is not None:
            k = np.where(np.asarray(sheets[c], bool), np.float32(0), k)
        result.append(k)
    return result


NON_DYADIC = [0.2, 0.37, 'random']


def cells_for(value, shape=(13, 9, 7)):
    if value == 'random':
        return np.random.default_rng(202).uniform(0, .85, shape).astype(np.float32)
    cells = np.zeros(shape, np.float32)
    cells[5:7, 3:6, 2:4] = value
    cells[0, 1:4, 0:2] = value          # on two walls
    return cells


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
    for got, expected in zip(actual, reference32(cells, periodic)):
        assert np.array_equal(np.asarray(got), expected)


def _contexts(cells, periodic, sheets):
    x = jnp.asarray(cells)
    masks = None if sheets is None else tuple(jnp.asarray(m) for m in sheets)

    def f(v):
        return build_edge_keep(v, periodic=periodic, sheet_edge_masks=masks)

    def in_scan():
        def body(carry, _):
            return carry, f(x + 0.0 * carry)
        return tuple(k[-1] for k in jax.lax.scan(body, jnp.float32(0.), None, length=2)[1])

    def into_field():
        field = jnp.full(x.shape, jnp.float32(0.731))

        def body(carry, _):
            return tuple(c * k for c, k in zip(carry, f(x))), None
        return jax.lax.scan(body, (field,) * 3, None, length=1)[0]
    return {'eager': f(x), 'jit': jax.jit(f)(x), 'in a scan': jax.jit(in_scan)(),
            'into a field in a scan': jax.jit(into_field)()}


def _assert_one_value_in_every_context(value, periodic, with_sheet):
    cells = cells_for(value)
    sheets = None
    if with_sheet:
        sheets = [np.zeros(cells.shape, bool) for _ in range(3)]
        sheets[1][6, 2:7, 1:5] = True
        sheets[2][6, 2:7, 1:5] = True
    expected = reference32(cells, periodic, sheets)
    got = _contexts(cells, periodic, sheets)
    for name in ('eager', 'jit', 'in a scan'):
        for actual, ref in zip(got[name], expected):
            assert np.array_equal(np.asarray(actual), ref), name
    for actual, ref in zip(got['into a field in a scan'], expected):
        assert np.array_equal(np.asarray(actual), (np.float32(0.731) * ref).astype(np.float32))
    for actual, ref in zip(got['eager'], reference(cells, periodic)):
        keep = np.asarray(actual, np.float64)
        if sheets is None:
            np.testing.assert_allclose(keep, ref, rtol=0, atol=2.4e-7)


@pytest.mark.parametrize('with_sheet', [False, True])
@pytest.mark.parametrize('periodic', [(False, False, False), (False, True, True)])
@pytest.mark.parametrize('value', NON_DYADIC)
def test_one_value_in_every_compile_context(value, periodic, with_sheet):
    _assert_one_value_in_every_context(value, periodic, with_sheet)


@pytest.mark.gpu_gate
def test_one_value_in_every_compile_context_on_this_backend():
    """The same bitwise comparison, run on a GPU by the merge train's gate."""
    for value in NON_DYADIC:
        _assert_one_value_in_every_context(value, (False, False, False), False)


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
    # The step's in-loop former is 1 - M; its last bit depends on the compile
    # context, so the published product is held to one ULP of 1.0 against it.
    for row in applied:
        for expected, actual in zip(row, published[0]):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=ULP1)
    # Exactness is a statement about the OCCUPANCY: an edge whose four cells
    # are all 0, or one of which is 1, has factor exactly 1 or 0. (Where the
    # operator rounds a tiny product to 0 the published factor keeps it.)
    exact = reference32(np.asarray(cells))
    full = reference(np.asarray(cells))
    for actual, e32, e64 in zip(published[0], exact, full):
        if not sheet:
            binary = (e64 == 0) | (e64 == 1)
            assert np.array_equal(actual[binary], e64[binary].astype(np.float32))
            assert np.array_equal(actual, e32)


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
    whole = [np.asarray(k) for k in build_edge_keep(jnp.asarray(cells))]
    exact = reference32(cells)
    for c, (component, keep, ref) in enumerate(zip((effective.ex, effective.ey, effective.ez), result.edge_keep, refs)):
        for rank, (lo, hi) in enumerate(layout.owned):
            owned = slice(rank * layout.nx_local + 1, rank * layout.nx_local + 1 + hi - lo)
            got = np.asarray(keep)[owned]
            # per slab == whole domain == NumPy float32, bit for bit
            assert np.array_equal(got, whole[c][lo:hi])
            assert np.array_equal(got, exact[c][lo:hi])
            np.testing.assert_allclose(got, np.asarray(component)[owned], rtol=0, atol=ULP1)
            np.testing.assert_allclose(got, ref[lo:hi], rtol=0, atol=2.4e-7)

    # Gradient through the cut and the per-slab builder against the whole domain
    # (a derivative of a product is a sum of products: equal to rounding, not bitwise).
    def slab_total(v):
        keep = publish_slab_occupancy(material(slabs.shape), cut(v, layout, 'pec_occupancy', mesh=mesh), mesh).edge_keep
        total = 0.
        for k in keep:
            for rank, (lo, hi) in enumerate(layout.owned):
                total = total + jnp.sum(k[rank * layout.nx_local + 1: rank * layout.nx_local + 1 + hi - lo] ** 2)
        return total
    g_slab = np.asarray(jax.grad(slab_total)(jnp.asarray(cells)))
    g_whole = np.asarray(jax.grad(lambda v: sum(jnp.sum(k ** 2) for k in build_edge_keep(v)))(jnp.asarray(cells)))
    assert np.isfinite(g_slab).all() and np.any(g_slab != 0)
    np.testing.assert_allclose(g_slab, g_whole, rtol=0, atol=1e-6 * np.max(np.abs(g_whole)))


@pytest.mark.parametrize('n_ranks', [2, 3, 4])
def test_a_slab_with_its_low_ghost_row_gives_the_whole_domain_bits(n_ranks):
    """The per-slab arithmetic with no devices: any slab count, PR lane.

    A slab reads its own cells and one low ghost row; rank 0's ghost is the
    zero outside the lattice. (The compile context of the real shard_map is
    covered by the two-device case above.)
    """
    shape = (13, 7, 5)
    cells = np.random.default_rng(31).uniform(0, .9, shape).astype(np.float32)
    whole = [np.asarray(k) for k in build_edge_keep(jnp.asarray(cells))]
    edges = np.linspace(0, shape[0], n_ranks + 1).astype(int)
    for lo, hi in zip(edges[:-1], edges[1:]):
        ghost = cells[lo - 1:lo] if lo else np.zeros((1,) + shape[1:], np.float32)
        local = [np.asarray(k) for k in build_edge_keep(jnp.asarray(np.concatenate((ghost, cells[lo:hi]))))]
        for c in range(3):
            assert np.array_equal(local[c][1:], whole[c][lo:hi])


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


def test_design_window_equals_the_whole_domain_product_bit_for_bit():
    from rfx.simulation import DesignOccupancySpec
    from rfx.boundaries.pec import apply_pec_occupancy, apply_pec_occupancy_box, pec_occupancy_box_keep
    shape = (7, 6, 5)
    rng = np.random.default_rng(7)
    background_np = rng.uniform(0, .6, shape).astype(np.float32)
    background = jnp.asarray(background_np)
    for bounds in ((2, 4, 1, 4, 2, 3), (0, 2, 0, 3, 0, 2), (4, 7, 3, 6, 2, 5)):
        box = tuple(slice(bounds[2 * a], bounds[2 * a + 1]) for a in range(3))
        x_np = rng.uniform(0, .9, tuple(bounds[2 * a + 1] - bounds[2 * a] for a in range(3))).astype(np.float32)
        x = jnp.asarray(x_np)

        def f(v):
            return build_edge_keep(background, design=DesignOccupancySpec(bounds, v))
        merged = background_np.copy()
        merged[box] = x_np
        for mode in (f, jax.jit(f)):
            for actual, exact in zip(mode(x), reference32(merged)):
                assert np.array_equal(np.asarray(actual), exact), bounds
        # and within one ULP of 1.0 of what the step's window rewrite gives today
        one = jnp.ones(shape); st = init_state(shape)._replace(ex=one, ey=one, ez=one)
        write, keep = pec_occupancy_box_keep(bounds, x, shape=shape, dtype=jnp.float32, pec_occupancy=background)
        today = apply_pec_occupancy_box(apply_pec_occupancy(st, background), st, write, keep)
        for a, b in zip(f(x), (today.ex, today.ey, today.ez)):
            np.testing.assert_allclose(a, b, rtol=0, atol=ULP1)
        grad = jax.grad(lambda v: sum(jnp.sum(a) for a in f(v)))(x)
        assert np.isfinite(grad).all() and np.any(np.asarray(grad) != 0)


def test_gradient_is_the_same_whole_domain_and_under_jit():
    cells = jnp.asarray(cells_for('random'))

    def total(v):
        return sum(jnp.sum(k * k) for k in build_edge_keep(v))
    eager = np.asarray(jax.grad(total)(cells))
    jitted = np.asarray(jax.jit(jax.grad(total))(cells))
    assert np.isfinite(eager).all() and np.any(eager != 0)
    np.testing.assert_allclose(jitted, eager, rtol=1e-6, atol=1e-6)


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
