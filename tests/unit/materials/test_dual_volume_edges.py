"""Independent four-cell references for the graded constitutive rule."""
from itertools import product
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import MaterialArrays, edge_mean_components, cell_component_e_materials
from rfx.model.materials import (ComponentCells, EdgePoles, electric_cell_sizes,
                                 e_update_material_at, realize_components)
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import lorentz_pole


class MetricGrid:
    def __init__(self, widths):
        self.widths = tuple(np.asarray(d) for d in widths)
        self.shape = tuple(len(d) for d in widths)
        self.dt = 1e-12
        self.dx_arr, self.dy_arr, self.dz = self.widths

    def cells(self, axis):
        return self.widths[axis]

    def is_constant(self, axis):
        return np.all(self.widths[axis] == self.widths[axis][0])


def reference(values, widths, component, cell, periodic):
    """NumPy cell selection and primal-area sum; no product mean helper."""
    transverse = [a for a in range(3) if a != component]
    numerator = denominator = 0.0
    gradients = np.zeros_like(values, dtype=float)
    for offsets in product((0, 1), repeat=2):
        index = list(cell)
        for axis, offset in zip(transverse, offsets):
            i = index[axis] - offset
            index[axis] = i % values.shape[axis] if periodic[axis] else max(i, 0)
        index = tuple(index)
        area = np.prod([widths[a][index[a]] for a in transverse])
        numerator += area * values[index]
        denominator += area
        gradients[index] += area
    return numerator / denominator, gradients / denominator


@pytest.mark.parametrize('periodic', [(False,)*3, (True,)*3, (True, False, True)])
@pytest.mark.parametrize('shape', [(3, 4, 5), (1, 4, 3)])
def test_volume_sigma_and_poles_against_hand_reference(periodic, shape):
    rng = np.random.default_rng(733)
    g = MetricGrid([np.arange(n, dtype=float) + 1 for n in shape])
    eps = rng.uniform(1, 8, shape).astype(np.float32)
    sigma = rng.uniform(0, .8, shape).astype(np.float32)
    mask = rng.random(shape) > .4
    m = MaterialArrays(jnp.asarray(eps), jnp.asarray(sigma), jnp.ones(shape))
    c = realize_components(ComponentCells(m, ([DebyePole(2., 1e-11)], mask)),
                           g, periodic=periodic)
    # beta = eps0 * delta_eps * dt/(2*tau+dt) * occupancy.
    beta_scale = 8.8541878128e-12 * 2 * g.dt / (2e-11 + g.dt)
    for a, cell in product(range(3), np.ndindex(shape)):
        for array, edge in ((eps, c.eps_update[a]), (sigma, c.sigma_update[a]),
                            (mask, c.debye.coefficients[1][a][0] / beta_scale)):
            expected, _ = reference(array, g.widths, a, cell, periodic)
            np.testing.assert_allclose(float(edge[cell]), expected, rtol=3e-7, atol=1e-7)
        point = cell_component_e_materials(m, cell, 'e'+'xyz'[a], periodic,
                                          cell_sizes=electric_cell_sizes(g))
        np.testing.assert_allclose(point, [c.eps_update[a][cell], c.sigma_update[a][cell]], rtol=3e-7)


@pytest.mark.parametrize('axis', range(3))
def test_derivative_is_primal_area_fraction(axis):
    g = MetricGrid(([1., 2., 4.], [3., 2., 1.], [2., 5., 3.]))
    values = np.arange(27, dtype=np.float32).reshape(g.shape) + 1
    cell = (1, 1, 1)
    derivative = jax.grad(lambda a: edge_mean_components(
        a, cell_sizes=electric_cell_sizes(g))[axis][cell])(jnp.asarray(values))
    _, expected = reference(values, g.widths, axis, cell, (False,)*3)
    np.testing.assert_allclose(derivative, expected, rtol=1e-7, atol=1e-8)


def test_equal_cells_are_bit_identical_per_transverse_component():
    rng = np.random.default_rng(512)
    values = jnp.asarray(rng.uniform(1, 8, (3, 4, 5)).astype(np.float32))
    plain = edge_mean_components(values)
    for widths in (([2.]*3, [3.]*4, [7.]*5), ([1., 2., 3.], [3.]*4, [7.]*5)):
        g = MetricGrid(widths)
        actual = edge_mean_components(values, cell_sizes=electric_cell_sizes(g))
        axes = range(3) if g.is_constant(0) else (0,)
        for axis in axes:
            assert np.asarray(actual[axis]).tobytes() == np.asarray(plain[axis]).tobytes()


@pytest.mark.parametrize('tensor', [False, True])
def test_raw_nu_update_requires_primal_metrics(tensor):
    from rfx.core.yee import init_state, update_e_nu, update_e_nu_aniso
    g = MetricGrid(([1.,2.,4.],[3.,2.,1.],[2.,5.,3.]))
    m = MaterialArrays(jnp.arange(27,dtype=jnp.float32).reshape(g.shape)+1,
                       jnp.full(g.shape,.1),jnp.ones(g.shape))
    inv = tuple(jnp.asarray(1/d) for d in g.widths)
    state = init_state(g.shape)._replace(ex=jnp.ones(g.shape),ey=jnp.ones(g.shape),ez=jnp.ones(g.shape))
    c = realize_components(m,g,periodic=(False,)*3)
    def update(mat, metric=inv, **kwargs):
        if tensor:
            return update_e_nu_aniso(state,mat,*c.eps_update,g.dt,*metric,**kwargs)
        return update_e_nu(state,mat,g.dt,*metric,**kwargs)
    with pytest.raises(ValueError,match='requires realized component materials or primal'):
        update(m)
    with pytest.raises(ValueError,match='requires realized component materials or primal'):
        jax.jit(lambda d:update(m,metric=d))(inv)
    got=update(m,cell_sizes=electric_cell_sizes(g))
    expected=update(m._replace(components=c))
    for a,b in zip(jax.tree.leaves(got),jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('kind', ['debye', 'lorentz'])
def test_distributed_drive_window_reads_weighted_poles(kind):
    g = MetricGrid(([1., 2., 4.], [3., 2., 1.], [2., 5., 3.]))
    mask = np.indices(g.shape)[0] >= 1
    db = ([DebyePole(2., 1e-11)], jnp.asarray(mask)) if kind == 'debye' else None
    lr = ([lorentz_pole(2., 2e10, 1e9)], jnp.asarray(mask)) if kind == 'lorentz' else None
    m = MaterialArrays(jnp.asarray(1 + 3*mask, dtype=jnp.float32),
                       jnp.asarray(.2*mask, dtype=jnp.float32), jnp.ones(g.shape))
    c = realize_components(ComponentCells(m, db, lr), g, periodic=(False,)*3)
    drive = m._replace(components=EdgePoles(db, lr, g))
    for cell in ((1, 1, 1), (0, 0, 0)):
        for a in range(3):
            expected = reference(np.asarray(m.eps_r), g.widths, a, cell, (False,)*3)[0]
            actual = e_update_material_at(drive, cell, 'e'+'xyz'[a])[0]
            np.testing.assert_allclose(actual, expected, rtol=2e-7)
        terms = drive.components.at(cell, g.dt)[0 if kind == 'debye' else 1]
        full = (c.debye if kind == 'debye' else c.lorentz).coefficients
        for local, entire in zip(jax.tree.leaves(terms), jax.tree.leaves(full)):
            np.testing.assert_allclose(local, entire[(slice(None),)+cell], rtol=2e-7)


def test_slab_widths_and_existing_ghost_match_global_reference():
    from rfx.model.electric_metrics import slab_cell_sizes
    from rfx.runners._distributed_common import slab_e_component_materials, slab_pole_fractions
    g = MetricGrid(([1., 2., 3., 4., 5., 6.], [2., 3., 4.], [3., 4., 2.]))
    values = np.arange(np.prod(g.shape), dtype=np.float32).reshape(g.shape)+1
    mask = values % 3 == 0
    sg = SimpleNamespace(e_cell_sizes=electric_cell_sizes(g), nx=6, nx_per_rank=3, nx_local=5)
    for rank in (0, 1):
        indices = np.clip(rank*3 - 1 + np.arange(5), 0, 5)
        slab = jnp.asarray(values[indices])
        sizes = slab_cell_sizes(sg, rank)
        actual = slab_e_component_materials(MaterialArrays(slab, slab, slab), 3, 6,
                                            rank=rank, cell_sizes=sizes)[0]
        fractions = slab_pole_fractions(jnp.asarray(mask[indices]), 3, 6,
                                        rank=rank, cell_sizes=sizes)
        for a, i, j, k in product(range(3), range(1, 4), range(3), range(3)):
            cell = (rank*3+i-1, j, k)
            for array, result in ((values, actual), (mask, fractions)):
                expected = reference(array, g.widths, a, cell, (False,)*3)[0]
                np.testing.assert_allclose(result[a][i,j,k], expected, rtol=2e-7, atol=1e-7)


@pytest.mark.parametrize('kind', ['plain', 'lossy', 'debye', 'lorentz'])
@pytest.mark.parametrize('drive', ['current', 'port'])
def test_first_e_sample_at_unequal_interface(kind, drive):
    from rfx import Simulation, Box
    widths = np.array([1., 1., .9, 1.1, .9, 1.1, 1., 1.]) * .001
    sim = Simulation(freq_max=8e9, domain=(.008,)*3, dx=.001,
                     dx_profile=widths, boundary='pec', cpml_layers=0)
    extra = {}
    if kind == 'debye':
        extra['debye_poles'] = [DebyePole(2., 1e-11)]
    if kind == 'lorentz':
        extra['lorentz_poles'] = [lorentz_pole(2., 2e10, 1e9)]
    conductivity = .2 if kind == 'lossy' else 0.
    sim.add_material('slab', eps_r=4., sigma=conductivity, **extra)
    sim.add(Box((.004, 0, 0), (.008,)*3), material='slab')
    position = (.004, .004, .004)
    if drive == 'current':
        sim.add_source(position, 'ez', waveform=jnp.ones_like, amplitude_kind='current')
    elif drive == 'port':
        sim.add_port(position, 'ez', waveform=jnp.ones_like)
    else:
        sim.add_lumped_rlc(position, 'ez', R=50., C=1e-14, topology='parallel')
        sim.add_source(position, 'ez', waveform=jnp.ones_like, amplitude_kind='current')
    sim.add_probe(position, 'ez')
    grid = sim._build_nonuniform_grid()
    dt = float(grid.dt)
    # The edge sits between dx=.0011 (vacuum) and dx=.0009 (slab).
    fraction = .9 / (1.1 + .9)
    eps = 1 + 3*fraction + (1e-14/(8.8541878128e-12*.001) if drive == 'rlc' else 0.)
    sigma = conductivity*fraction + (20. if drive in ('port','rlc') else 0.)
    beta = 8.8541878128e-12*2*fraction*dt/(2e-11+dt) if kind == 'debye' else 0.
    cb = dt / (8.8541878128e-12*eps + beta + sigma*dt/2)
    current_density = 20./.001 if drive == 'port' else 1e9
    result = sim.run(n_steps=2, skip_preflight=True, compute_s_params=False)
    measured = np.asarray(result.time_series).reshape(-1)[0]
    np.testing.assert_allclose(measured, cb*current_density, rtol=4e-7)


def test_design_window_and_held_edge_use_primal_widths():
    from rfx.model.materials import _design_box_edge_coeffs
    from rfx.core.yee import EPS_0
    g = MetricGrid(([1., 2., 3., 4., 5.],)*3)
    m = MaterialArrays(jnp.ones(g.shape), jnp.zeros(g.shape), jnp.ones(g.shape))
    bounds = (1, 3, 1, 3, 1, 3)
    values = np.ones(g.shape, dtype=np.float32)
    values[1:3, 1:3, 1:3] = 4.
    write, _, cb = _design_box_edge_coeffs(bounds, jnp.full((2,)*3, 4.), None,
                                          m, g.dt, g.shape, grid=g)
    for axis in range(3):
        expected = reference(values, g.widths, axis, (1,1,1), (False,)*3)[0]
        cell = tuple(1 - write[2*a] for a in range(3))
        np.testing.assert_allclose(cb[axis][cell], g.dt/(EPS_0*expected), rtol=2e-7)
        _, _, held = _design_box_edge_coeffs(bounds, jnp.full((2,)*3, 4.), None,
            m, g.dt, g.shape, held_edges=((axis,1,1,1),), grid=g)
        np.testing.assert_allclose(held[axis][cell], g.dt/EPS_0, rtol=2e-7)


def test_eps_override_gradient_at_graded_interface_matches_fd():
    from rfx import Simulation, Box
    widths = np.array([1., 1., .9, 1.1, .9, 1.1, 1., 1.]) * .001
    sim = Simulation(freq_max=8e9, domain=(.008,)*3, dx=.001,
                     dx_profile=widths, boundary='pec', cpml_layers=0)
    sim.add_material('slab', eps_r=4.)
    sim.add(Box((.004, 0, 0), (.008,)*3), material='slab')
    position = (.004, .004, .004)
    sim.add_source(position, 'ez', waveform=jnp.ones_like, amplitude_kind='current')
    sim.add_probe(position, 'ez')
    shape = sim._build_nonuniform_grid().shape
    mask = jnp.indices(shape)[0] >= 4

    def objective(epsilon):
        values = jnp.where(mask, epsilon, 1.)
        samples = sim.forward(eps_override=values, n_steps=3,
                              skip_preflight=True).time_series
        return jnp.sum(samples) * 1e-9

    actual = jax.grad(objective)(4.)
    delta = .01
    finite = (objective(4.+delta) - objective(4.-delta)) / (2*delta)
    np.testing.assert_allclose(actual, finite, rtol=2e-4, atol=1e-8)


def test_plain_distributed_drive_keeps_equal_cell_host_bits():
    from rfx.model.materials import e_update_coefficient_at
    g = MetricGrid(([.001]*3,)*3)
    eps = jnp.arange(27, dtype=jnp.float32).reshape(g.shape)/17+1
    sigma = jnp.arange(27, dtype=jnp.float32).reshape(g.shape)/71
    raw = MaterialArrays(eps, sigma, jnp.ones(g.shape))
    carried = raw._replace(components=EdgePoles(None, None, g))
    for component in ('ex','ey','ez'):
        old = e_update_coefficient_at(raw, (1,1,1), component, g.dt, host=True)
        new = e_update_coefficient_at(carried, (1,1,1), component, g.dt, host=True)
        assert np.asarray(old).tobytes() == np.asarray(new).tobytes()


@pytest.mark.parametrize('kind', ['plain','lossy'])
def test_first_e_sample_with_graded_lumped_rc(kind):
    test_first_e_sample_at_unequal_interface(kind, 'rlc')


@pytest.mark.parametrize('periodic', [(False, False, False), (True, True, True)])
@pytest.mark.parametrize('term', ['fraction', 'coupling'])
def test_lorentz_occupancy_and_recurrence_against_hand_reference(periodic, term):
    from rfx.model.materials import pole_component_weights
    shape = (3, 4, 5)
    widths = (np.array([1., 2., 4.]), np.array([3., 1., 2., 5.]),
              np.array([2., 4., 1., 3., 6.]))
    grid = MetricGrid(widths)
    mask = np.random.default_rng(829).random(shape) > .45
    delta_eps, omega0 = 2., 2*np.pi*15e9
    delta = omega0/20
    pole = lorentz_pole(delta_eps, omega0, delta)
    eps = jnp.ones(shape, dtype=jnp.float32)*2
    raw = MaterialArrays(eps, jnp.zeros_like(eps), jnp.ones_like(eps))
    realized = realize_components(ComponentCells(raw, None, ([pole], mask)),
                                  grid, periodic=periodic)
    fractions = pole_component_weights(mask, periodic, cell_sizes=electric_cell_sizes(grid))
    expected = np.empty((3, *shape))
    for axis in range(3):
        for cell in np.ndindex(shape):
            expected[(axis, *cell)] = reference(mask, widths, axis, cell, periodic)[0]
    # Independent central-difference / CN Lorentz recurrence coupling.
    scale = 8.8541878128e-12 * (delta_eps*omega0**2) * grid.dt**2 / (1 + delta*grid.dt)
    coupling = np.asarray(realized.lorentz.coefficients[2])[:, 0]
    if term == 'fraction':
        np.testing.assert_allclose(fractions, expected, rtol=3e-7, atol=1e-7)
    else:
        np.testing.assert_allclose(coupling, scale*expected, rtol=4e-7, atol=0)


@pytest.mark.parametrize('tensor', [False, True])
def test_rounded_inverse_duals_cannot_decide_primal_constancy(tensor):
    from rfx.core.yee import init_state, update_e_nu, update_e_nu_aniso
    # Unequal float64 cells become indistinguishable as float32 inverse duals.
    grid = MetricGrid(([1., 1.+1e-9, 1.],)*3)
    assert not any(grid.is_constant(a) for a in range(3))
    widths = electric_cell_sizes(grid)
    assert all(d is not None for d in widths)
    inverse = tuple(jnp.asarray(1 / np.r_[d[0], (d[:-1]+d[1:])/2],
                                dtype=jnp.float32) for d in grid.widths)
    assert all(np.all(np.asarray(d) == 1.) for d in inverse)
    eps = jnp.arange(27, dtype=jnp.float32).reshape(grid.shape)+1
    raw = MaterialArrays(eps, eps*.01, jnp.ones_like(eps))
    state = init_state(grid.shape)._replace(ex=jnp.ones_like(eps))
    components = realize_components(raw, grid, periodic=(False,)*3)

    def step(materials, **kwargs):
        if tensor:
            return update_e_nu_aniso(state, materials, *components.eps_update,
                                     grid.dt, *inverse, **kwargs)
        return update_e_nu(state, materials, grid.dt, *inverse, **kwargs)

    with pytest.raises(ValueError, match='requires realized component materials or primal'):
        step(raw)
    explicit = step(raw, cell_sizes=widths)
    stored = step(raw._replace(components=components))
    for a, b in zip(jax.tree.leaves(explicit), jax.tree.leaves(stored)):
        np.testing.assert_array_equal(a, b)
