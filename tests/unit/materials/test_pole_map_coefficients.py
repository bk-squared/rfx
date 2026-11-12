"""Independent cell-centre and edge-weight witnesses for spatial poles."""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import LorentzPoleMap, Simulation, drude_pole_map, lorentz_pole_map
from rfx.core.yee import init_materials
from rfx.materials.lorentz import drude_pole, init_lorentz, pole_update_coeffs
from rfx.model.materials import EdgePoles, e_update_coefficient_at, with_components

EPS0 = 8.8541878128e-12
WP = 2 * np.pi * 3e9


def _capture_c2(kind, monkeypatch):
    """Observe public setup before its scan, as in C1's coefficient measurement."""
    import sys
    from functools import wraps
    import rfx.materials.lorentz as lorentz
    import rfx.materials.pole_map as pole_map
    import rfx.model.materials as materials
    from tests.unit.materials.test_pole_map_runs import capture_scan, scene

    records = {name: [] for name in ('realize', 'map', 'electric')}
    locals_at_return = {}
    targets = [('realize', materials, '_realize_poles'), ('map', pole_map, 'map_terms'),
               ('electric', lorentz, 'lorentz_e_coeffs')]
    codes = {getattr(mod, attr).__code__ for _, mod, attr in targets}
    def profile(frame, event, arg):
        if event == 'return' and frame.f_code in codes:
            locals_at_return[frame.f_code] = frame.f_locals.copy()
    def wrap(name, original):
        @wraps(original)
        def spy(*args, **kwargs):
            offsets = {k: len(v) for k, v in records.items()}
            result = original(*args, **kwargs)
            records[name].append(dict(local=locals_at_return[original.__code__], result=result, offsets=offsets))
            return result
        return spy
    with monkeypatch.context() as patch:
        for name, mod, attr in targets:
            patch.setattr(mod, attr, wrap(name, getattr(mod, attr)))
        sim, grid, _ = scene(True, kind, mapped=True, precision='float64')
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            capture_scan(sim, 1, patch)
        finally:
            sys.setprofile(previous)
    electric = [r for r in records['electric'] if np.shape(r['result'].c[0])[-3:] == grid.shape][-1]
    c = electric['result'].c
    realized = [r for r in records['realize'] if r['result'] is not None
                and r['local']['kind'] == 'Lorentz' and r['result'].coefficients is not None
                and all(np.array_equal(a, b) for a, b in zip(r['result'].coefficients[2], c))][-1]
    local = realized['local']
    assert not any(local['periodic'])
    widths = tuple(np.asarray(local['grid'].cells(a)) for a in range(3))
    assert all(len(d) == grid.shape[a] for a, d in enumerate(widths))
    maps = records['map'][realized['offsets']['map']:]
    assert len(maps) == 1
    m = maps[0]['local']
    return dict(w=tuple(np.asarray(v) for v in m['weights']), c=tuple(np.asarray(v) for v in c),
                k=np.asarray(m['k']), delta=np.asarray(m['delta']), ks=float(m['ks']), dt=float(m['dt']),
                widths=widths, high_wall=local['high_wall'])


def _four_cell_mean(values, widths, component, high_wall):
    """Independent four-area NumPy float64 rule, including boundary replication."""
    shape = values.shape
    axes = [a for a in range(3) if a != component]
    base = [np.minimum(np.arange(n), n - 2 if high_wall[a] else n - 1)
            for a, n in enumerate(shape)]
    cells, areas = [], []
    for backs in ((0, 0), (1, 0), (0, 1), (1, 1)):
        ix = [v.copy() for v in base]
        for a, back in zip(axes, backs):
            ix[a] = np.maximum(np.arange(shape[a]) - back, 0)
            if high_wall[a]:
                ix[a] = np.minimum(ix[a], shape[a] - 2)
        cells.append(values[np.ix_(*ix)])
        ds = []
        for a in axes:
            sh = [1, 1, 1]
            sh[a] = shape[a]
            ds.append(np.asarray(widths[a], np.float64)[ix[a]].reshape(sh))
        areas.append(ds[0] * ds[1])
    return sum(np.asarray(v, np.float64) * a for v, a in zip(cells, areas)) / sum(areas)


def _float64_distance(actual, expected):
    active = expected != 0
    a, b = np.asarray(actual)[active], expected[active]
    assert np.all(np.isfinite(a)) and np.all(np.signbit(a) == np.signbit(b))
    ai, bi = a.view(np.uint64), b.view(np.uint64)
    return int((np.maximum(ai, bi) - np.minimum(ai, bi)).max())


@pytest.mark.parametrize('kind', ['P2a', 'P2b'])
def test_c2_graded_float64_coefficients(kind, monkeypatch):
    from tests._x64_compat import enable_x64
    from tests.unit.materials.test_pole_map_runs import record

    with enable_x64():
        m = _capture_c2(kind, monkeypatch)
        dtypes = dict(w=[str(v.dtype) for v in m['w']], c=[str(v.dtype) for v in m['c']])
        record(f'C2-{kind}-precondition', dtypes)
        assert all(v.dtype == np.float64 for v in (*m['w'], *m['c'])), dtypes
        rows = []
        for comp in range(3):
            w64 = _four_cell_mean(m['k'], m['widths'], comp, m['high_wall'])
            kd64 = _four_cell_mean(m['k'] * m['delta'], m['widths'], comp, m['high_wall'])
            de = np.divide(kd64, w64, out=np.zeros_like(w64), where=w64 != 0)
            ke = m['ks'] * w64
            c64 = EPS0 * ke * m['dt']**2 / (1 + de * m['dt'])
            rows.append(dict(component=('Ex', 'Ey', 'Ez')[comp],
                             w=_float64_distance(m['w'][comp], w64),
                             c=_float64_distance(m['c'][comp][0], c64)))
        record(f'C2-{kind}', dict(rows=rows))
        for row in rows:
            assert row['w'] <= 17, row
            assert row['c'] <= 28, row


def rig(graded=False):
    widths = np.array([.5, .45, .4, .35, .4, .45, .55, .6, .65, .6, .55, .5]) * .001
    sim = Simulation(freq_max=6e9, domain=(.006,) * 3, dx=.0005, cpml_layers=0, boundary='pec',
                     **(dict(dx_profile=widths) if graded else {}))
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    assert grid.shape == (13, 13, 13)
    axes = []
    for a in range(3):
        d = np.asarray(grid.cells(a), dtype=float)
        nodes = np.r_[0., np.cumsum(d[:-1])]
        assert nodes[12] == pytest.approx(.006)
        axes.append(nodes + d / 2)
    xyz = np.meshgrid(*axes, indexing='ij', sparse=True)
    mask = np.ones(grid.shape, bool)
    mask[:2] = mask[-2:] = False
    mask[:, :2] = mask[:, -2:] = False
    mask[:, :, :2] = mask[:, :, -2:] = False
    return grid, xyz, mask


def init(pole, grid, mask):
    return init_lorentz([pole], init_materials(grid.shape), grid.dt, mask=mask, grid=grid)


def hand_mean(values, grid, comp, cell):
    """Four incident primal areas, with independent NumPy weight construction."""
    trans = [a for a in range(3) if a != comp]
    total = 0.
    norm = 0.
    for offsets in itertools.product((0, -1), repeat=2):
        index = list(cell)
        area = 1.
        for a, off in zip(trans, offsets):
            index[a] += off
            area *= float(grid.cells(a)[index[a]])
        total += area * float(values[tuple(index)])
        norm += area
    return total / norm


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
@pytest.mark.parametrize('callables', [False, True], ids=['numbers', 'callables'])
def test_p1_constant_coefficients(graded, callables):
    grid, xyz, mask = rig(graded)
    pole = drude_pole_map((lambda x, y, z: WP + 0 * x) if callables else WP,
                          (lambda x, y, z: 2e9 + 0 * y) if callables else 2e9)
    actual, state = init(pole, grid, mask)
    expected, _ = init(drude_pole(WP, 2e9), grid, mask)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert a.dtype == e.dtype
        np.testing.assert_array_equal(a, e)
        assert np.asarray(a).tobytes() == np.asarray(e).tobytes()
    assert all(v.shape == (1,) + grid.shape for v in state)


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_p2c_touching_damping_and_p8_sampling(graded, monkeypatch):
    import rfx.materials.lorentz as lorentz
    from tests.unit.materials.test_pole_map_runs import record
    grid, xyz, mask = rig(graded)
    x, y, z = xyz
    kappa = np.broadcast_to(WP**2 * (1 + 40 * x + 20 * y), grid.shape)
    delta = np.broadcast_to(np.where(x < .0025, 1e9, 2e9) + 1e10 * y, grid.shape)
    pole = LorentzPoleMap(0., lambda x, y, z: np.where(x < .0025, 1e9, 2e9) + 1e10 * y,
                         lambda x, y, z: WP**2 * (1 + 40 * x + 20 * y))
    observed_delta = []
    def observe(omega_0, delta, kappa, dt, field_dtype=None):
        if np.ndim(delta) == 3:
            observed_delta.append(np.asarray(delta))
        return pole_update_coeffs(omega_0, delta, kappa, dt, field_dtype)
    monkeypatch.setattr(lorentz, 'pole_update_coeffs', observe)
    coeffs, _ = init(pole, grid, mask)
    assert len(observed_delta) == 3
    assert isinstance(coeffs.damping, tuple) and isinstance(coeffs.restoring, tuple)
    worst_delta = 0.
    worst_damping = 0
    worst_restoring = 0.
    worst_c = 0
    for comp, cell in itertools.product(range(3), itertools.product(range(3, 10), repeat=3)):
        ke = hand_mean(mask * kappa, grid, comp, cell)
        de = hand_mean(mask * kappa * delta, grid, comp, cell) / ke
        want_delta = np.float32(de)
        delta_ulp = abs(float(np.float32(observed_delta[comp][cell])) - float(want_delta)) / abs(float(np.spacing(want_delta)))
        worst_delta = max(worst_delta, delta_ulp)
        assert delta_ulp <= 2, (comp, cell, observed_delta[comp][cell], de, delta_ulp)
        expected = pole_update_coeffs(0., de, ke, grid.dt)
        d = np.float64(np.float32(observed_delta[comp][cell]))
        t = np.float64(np.float32(grid.dt))
        want_damping = np.float32(2 * d * t / (1 + d * t))
        got_damping = np.float32(np.asarray(coeffs.damping[comp])[(0,) + cell])
        assert np.signbit(got_damping) == np.signbit(want_damping)
        damping_spacings = abs(int(got_damping.view(np.uint32)) - int(want_damping.view(np.uint32)))
        worst_damping = max(worst_damping, damping_spacings)
        assert damping_spacings <= 4, (comp, cell, got_damping, want_damping, damping_spacings)
        want = np.float32(expected[1])
        got = np.asarray(coeffs.restoring[comp])[(0,) + cell]
        ulp = abs(float(got) - float(want)) / abs(float(np.spacing(want)))
        worst_restoring = max(worst_restoring, ulp)
        assert ulp <= 2, (comp, cell, got, want, ulp)
        # A8b: independent float64 closed form, rounded once to float32.
        want_c = np.float32(EPS0 * ke * float(grid.dt)**2 / (1 + de * float(grid.dt)))
        got_c = np.float32(np.asarray(coeffs.c[comp])[(0,) + cell])
        assert np.signbit(got_c) == np.signbit(want_c)
        c_spacings = abs(int(got_c.view(np.uint32)) - int(want_c.view(np.uint32)))
        worst_c = max(worst_c, c_spacings)
        assert c_spacings <= 12, (comp, cell, got_c, want_c, c_spacings)
    interface = int(np.flatnonzero(x[:, 0, 0] >= .0025)[0])
    record(f'P2c-coefficients-{graded}', dict(delta_e_worst_ulp=worst_delta, damping_worst_spacings=worst_damping, restoring_worst_ulp=worst_restoring, c_worst_spacings=worst_c,
           interface_index=interface,
           interface_delta_e=[float(v[interface, 6, 6]) for v in observed_delta]))


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_p2c_float64_edge_mean(graded):
    from tests._x64_compat import enable_x64
    from tests.unit.materials.test_pole_map_runs import record

    with enable_x64():
        grid, xyz, mask = rig(graded)
        x, y, z = xyz
        kappa = np.broadcast_to(WP**2 * (1 + 40 * x + 20 * y), grid.shape)
        delta = np.broadcast_to(np.where(x < .0025, 1e9, 2e9) + 1e10 * y, grid.shape)
        pole = LorentzPoleMap(0., lambda x, y, z: np.where(x < .0025, 1e9, 2e9) + 1e10 * y,
                             lambda x, y, z: WP**2 * (1 + 40 * x + 20 * y))
        coeffs, _ = init(pole, grid, mask)
        assert all(v.dtype == np.float64 for v in coeffs.c), [v.dtype for v in coeffs.c]
        worst = 0
        edges = 0
        bar = 30 if graded else 25
        for comp, cell in itertools.product(range(3), itertools.product(range(3, 10), repeat=3)):
            ke = hand_mean(mask * kappa, grid, comp, cell)
            de = hand_mean(mask * kappa * delta, grid, comp, cell) / ke
            expected = np.float64(EPS0 * ke * float(grid.dt)**2 / (1 + de * float(grid.dt)))
            actual = np.asarray(coeffs.c[comp])[(0,) + cell]
            assert np.signbit(actual) == np.signbit(expected)
            spacings = abs(int(actual.view(np.uint64)) - int(expected.view(np.uint64)))
            worst = max(worst, spacings)
            edges += 1
            assert spacings <= bar, (comp, cell, actual, expected, spacings)
        record(f'A8c-P2c-{graded}', dict(c_dtype=str(coeffs.c[0].dtype), edges=edges,
                                        worst_spacings=worst, bar=bar))


def test_p3_one_state_identity_and_layout():
    grid, xyz, mask = rig()
    profile = lambda x, y, z: WP**2 * (1 + 40 * x)
    pole = LorentzPoleMap(0., 1e9, profile)
    coeffs, state = init(pole, grid, mask)
    assert not isinstance(coeffs.damping, tuple)
    assert not isinstance(coeffs.restoring, tuple)
    assert all(v.shape == (1,) + grid.shape for v in state), [v.shape for v in state]
    values = np.unique(np.broadcast_to(profile(*xyz), grid.shape)[mask])
    from rfx.materials.lorentz import LorentzPole
    masks = [(np.broadcast_to(profile(*xyz), grid.shape) == value) & mask for value in values]
    _, staircase = init_lorentz([LorentzPole(0., 1e9, v) for v in values], init_materials(grid.shape),
                                grid.dt, mask=masks, grid=grid)
    assert all(v.shape == (len(values),) + grid.shape for v in staircase)
    from rfx.geometry._pole_keying import _accumulate_pole_mask, _spec_from_pole_masks
    entries = {}
    _accumulate_pole_mask(entries, pole, jnp.asarray(mask))
    _accumulate_pole_mask(entries, pole, ~jnp.asarray(mask))
    poles, masks = _spec_from_pole_masks(entries)
    assert len(poles) == 1
    assert np.all(masks[0])
    _accumulate_pole_mask(entries, LorentzPoleMap(0., 1e9, profile), jnp.asarray(mask))
    assert len(entries) == 2
    assert lorentz_pole_map(2., 3., 4.).kappa == 18.


def test_source_edge_tuple_readers():
    grid, xyz, mask = rig(True)
    pole = LorentzPoleMap(0., lambda x, y, z: 1e9 * (1 + 30 * x), WP**2)
    materials = init_materials(grid.shape)
    spec = ([pole], [mask])
    realized = with_components(materials, grid, periodic=(False,) * 3, lorentz_spec=spec)
    local = materials._replace(components=EdgePoles(None, spec, grid))
    for component in ('ex', 'ey', 'ez'):
        got = e_update_coefficient_at(realized, (6, 6, 6), component, grid.dt)
        cut = e_update_coefficient_at(local, (6, 6, 6), component, grid.dt)
        np.testing.assert_array_equal(got, cut)
        assert float(got) == pytest.approx(grid.dt / EPS0, rel=2e-7)


@pytest.mark.parametrize('all_zero', [False, True], ids=['band', 'all'])
def test_p2c_prime_zero_strength_gradient(all_zero):
    grid, xyz, mask = rig(True)
    def objective(scale):
        pole = LorentzPoleMap(0., lambda x, y, z: 1e9 * (1 + 20 * y),
                             lambda x, y, z: scale * jnp.where((x < .003) | all_zero, 0., WP**2))
        coeffs, state = init(pole, grid, mask)
        for v in jax.tree.leaves(coeffs):
            if not isinstance(v, jax.core.Tracer):
                assert np.all(np.isfinite(v))
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in jax.tree.leaves(coeffs)]))
        return sum(jnp.sum((v / EPS0)**2) for v in coeffs.c), finite
    (value, finite), grad = jax.value_and_grad(objective, has_aux=True)(1.)
    assert finite and np.isfinite(value) and np.isfinite(grad)
    concrete_value, concrete_finite = objective(1.)
    assert concrete_finite and np.isfinite(concrete_value)
    if all_zero:
        assert value == grad == 0.


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_p8_linear_cell_centre_sampling(graded):
    from tests.unit.materials.test_pole_map_runs import record
    grid, xyz, mask = rig(graded)
    slope = 40.
    kappa = np.broadcast_to(WP**2 * (1 + slope * xyz[0]), grid.shape)
    coeffs, _ = init(LorentzPoleMap(0., 1e9, lambda x, y, z: WP**2 * (1 + slope*x)), grid, mask)
    worst = 0.
    for comp, i in itertools.product(range(3), range(3, 10)):
        cell = (i, 6, 6)
        ke = hand_mean(kappa, grid, comp, cell)
        if not graded:
            physical_x = (i + .5) * .0005 if comp == 0 else i * .0005
            assert ke == pytest.approx(WP**2 * (1 + slope * physical_x), rel=1e-14)
        expected = np.float32(EPS0 * ke * grid.dt**2 / (1 + 1e9 * grid.dt))
        actual = np.asarray(coeffs.c[comp])[(0,) + cell]
        assert np.signbit(actual) == np.signbit(expected)
        spacings = abs(int(actual.view(np.uint32)) - int(expected.view(np.uint32)))
        worst = max(worst, spacings)
        bar = 19 if graded else 14
        assert spacings <= bar, (comp, cell, actual, expected, spacings)
    record(f'P8-{graded}', dict(worst_spacings=worst))


def test_callable_sampled_once_and_outside_mask_ignored():
    grid, xyz, mask = rig(True)
    count = [0, 0]
    def kappa(x, y, z):
        count[0] += 1
        return np.where((x >= xyz[0][2, 0, 0]) & (x < xyz[0][-2, 0, 0]), WP**2, np.nan)
    def delta(x, y, z):
        count[1] += 1
        return 1e9 + 0*y
    coeffs, _ = init(LorentzPoleMap(0., delta, kappa), grid, mask)
    assert count == [1, 1]
    assert all(np.isfinite(v).all() for v in jax.tree.leaves(coeffs))


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
@pytest.mark.parametrize('scene_name', ['P2c', 'common'])
@pytest.mark.parametrize('vary_delta', [False, True], ids=['constant_delta', 'varying_delta'])
def test_exterior_increment_convention(graded, scene_name, vary_delta):
    from rfx.materials.debye import pole_reach
    from rfx.model.materials import electric_cell_sizes, pole_component_weights
    from rfx.geometry.rasterize_grid import unpadded_high_walls
    from tests.unit.materials.test_pole_map_runs import HI, LO, record, scene

    if scene_name == 'P2c':
        grid, xyz, mask = rig(graded)
    else:
        _, grid, _ = scene(graded, 'P1', mapped=True)
        axes = []
        for axis in range(3):
            widths = np.asarray(grid.cells(axis), dtype=float)
            nodes = np.r_[0., np.cumsum(widths[:-1])]
            nodes -= nodes[8]
            axes.append(nodes + widths / 2)
        xyz = np.meshgrid(*axes, indexing='ij', sparse=True)
        mask = np.broadcast_to(np.logical_and.reduce([
            np.broadcast_to((v >= lo) & (v < hi), grid.shape)
            for v, lo, hi in zip(xyz, LO, HI)]), grid.shape)
    delta = (lambda x, y, z: 1e9 * (1 + 30 * x)) if vary_delta else 1e9
    pole = LorentzPoleMap(0., delta, lambda x, y, z: WP**2 * (1 + 40 * x + 20 * y))
    coeffs, _ = init(pole, grid, mask)
    assert isinstance(coeffs.damping, tuple) == vary_delta
    fractions = pole_component_weights(mask, high_wall=unpadded_high_walls(grid),
                                       cell_sizes=electric_cell_sizes(grid))
    exterior = ~np.asarray(pole_reach(fractions))
    assert np.any(exterior)
    counts = []
    for component in range(3):
        c = np.asarray(coeffs.c[component])[0]
        forbidden = np.asarray(fractions[component]) <= 0
        assert np.count_nonzero(c[forbidden]) == 0
        assert np.count_nonzero(c[exterior]) == 0
        for values in (coeffs.damping, coeffs.restoring):
            array = values[component] if isinstance(values, tuple) else values
            np.testing.assert_array_equal(np.asarray(array)[0][exterior], 1.)
        counts.append(dict(c_nonzero=int(np.count_nonzero(c)), forbidden=int(forbidden.sum()),
                           exterior_c_nonzero=int(np.count_nonzero(c[exterior]))))
    record(f'A7-exterior-{scene_name}-{graded}-{vary_delta}',
           dict(exterior_cells=int(exterior.sum()), components=counts))
