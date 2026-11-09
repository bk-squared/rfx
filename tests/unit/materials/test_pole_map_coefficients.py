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
    assert isinstance(coeffs.a, tuple) and isinstance(coeffs.b, tuple)
    worst = 0.
    worst_delta = 0.
    for comp, cell in itertools.product(range(3), itertools.product(range(3, 10), repeat=3)):
        ke = hand_mean(mask * kappa, grid, comp, cell)
        de = hand_mean(mask * kappa * delta, grid, comp, cell) / ke
        want_delta = np.float32(de)
        delta_ulp = abs(float(np.float32(observed_delta[comp][cell])) - float(want_delta)) / abs(float(np.spacing(want_delta)))
        worst_delta = max(worst_delta, delta_ulp)
        assert delta_ulp <= 2, (comp, cell, observed_delta[comp][cell], de, delta_ulp)
        expected = pole_update_coeffs(0., de, ke, grid.dt)
        for arr, value in zip((coeffs.a[comp], coeffs.b[comp], coeffs.c[comp]), expected):
            want = np.float32(value)
            got = np.asarray(arr)[(0,) + cell]
            ulp = abs(float(got) - float(want)) / abs(float(np.spacing(want)))
            worst = max(worst, ulp)
            assert ulp <= 2, (comp, cell, got, want, ulp)
    interface = int(np.flatnonzero(x[:, 0, 0] >= .0025)[0])
    record(f'P2c-coefficients-{graded}', dict(worst_ulp=worst, delta_e_worst_ulp=worst_delta,
           interface_index=interface,
           interface_delta_e=[float(v[interface, 6, 6]) for v in observed_delta]))


def test_p3_one_state_identity_and_layout():
    grid, xyz, mask = rig()
    profile = lambda x, y, z: WP**2 * (1 + 40 * x)
    pole = LorentzPoleMap(0., 1e9, profile)
    coeffs, state = init(pole, grid, mask)
    assert not isinstance(coeffs.a, tuple)
    assert not isinstance(coeffs.b, tuple)
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
        ulp = abs(float(actual) - float(expected)) / abs(float(np.spacing(expected)))
        worst = max(worst, ulp)
        assert ulp <= 2, (comp, cell, actual, expected, ulp)
    record(f'P8-{graded}', dict(worst_ulp=worst))


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
