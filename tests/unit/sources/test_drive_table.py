"""Stored edge-weighted drives, including the occupancy derivative.

Mutation witness: with the drive weight removed the gradient misses the drive
term. The forward records in H3/H4 also pin the weighted primal independently.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.model.source_coefficients import drive_table
from tests._x64_compat import enable_x64


def test_absent_occupancy_is_the_exact_stack_program():
    samples = jnp.asarray(np.random.default_rng(1579).normal(size=(3, 17)).astype(np.float32))
    meta = [(2, 2, 2, c) for c in ('ex', 'ez', 'hx')]
    def actual(x):
        return drive_table(list(x), meta, 17)
    def expected(x):
        return jnp.stack(list(x), axis=-1)
    np.testing.assert_array_equal(jax.jit(actual)(samples), expected(samples))
    actual_jaxpr = str(jax.make_jaxpr(jax.jit(actual))(samples))
    assert 'mul ' not in actual_jaxpr
    assert str(jax.make_jaxpr(actual)(samples)) == str(jax.make_jaxpr(expected)(samples))
    empty = drive_table([], [], 17)
    assert empty.shape == (17, 0) and empty.dtype == jnp.float32


def incident_product(occupancy, component, cell):
    """Plain float32 four-cell product, paired over the two transverse axes.

    build_edge_keep documents chained noisy-OR: first combine the edge cell
    with its backward neighbour on the first transverse axis, then combine
    that pair with the pair one cell back on the second transverse axis.
    """
    first, second = [axis for axis in range(3) if axis != 'xyz'.index(component[1])]
    pairs = []
    for back in (0, 1):
        here = list(cell)
        here[second] -= back
        previous = here.copy()
        previous[first] -= 1
        pairs.append((np.float32(1) - occupancy[tuple(here)]) *
                     (np.float32(1) - occupancy[tuple(previous)]))
    return pairs[0] * pairs[1]


@pytest.mark.parametrize('component', ['ex', 'ey', 'ez'])
@pytest.mark.parametrize('value', [None, 0., 1.])
def test_weight_against_independent_incident_cells(component, value):
    cells = np.array([0., .13, .27, .5, .71, .83, .92, 1.] * 8,
                     dtype=np.float32).reshape(4, 4, 4)
    if value is not None:
        cells.fill(value)
    cell = (2, 2, 2)
    result = drive_table([jnp.ones(3, jnp.float32)], [(*cell, component)], 3,
                         occupancy=jnp.asarray(cells))
    expected = incident_product(cells, component, cell)
    np.testing.assert_allclose(result, expected, rtol=0, atol=np.spacing(np.float32(1)))
    if value is not None:
        np.testing.assert_array_equal(result, np.full((3, 1), 1. - value, np.float32))


def test_released_port_edge_has_exactly_unit_weight():
    from rfx._grid_metric import field_index
    from rfx.model.conductors import release_port_occupancy, solve_conductors, preview_port_stages
    sim = Simulation(freq_max=15e9, domain=(.008, .006, .004), dx=.001, boundary='pec')
    sim.add_port((.004, .003, .002), 'ez', impedance=50.)
    grid = sim._build_grid()
    conductors = preview_port_stages(sim, solve_conductors(sim, grid))
    with pytest.warns(UserWarning, match='Port occupancy release'):
        released = release_port_occupancy(jnp.full(grid.shape, .5), conductors)
    cell = field_index(grid, (.004, .003, .002), 'ez')
    samples = jnp.asarray([1., -3., .125], jnp.float32)
    actual = drive_table([samples], [(*cell, 'ez')], 3, occupancy=released)
    np.testing.assert_array_equal(actual[:, 0], samples)


@pytest.mark.parametrize('component', ['hx', 'hy', 'hz'])
def test_magnetic_column_is_untouched(component):
    samples = jnp.asarray([1., -3., .125], jnp.float32)
    result = drive_table([samples, samples], [(2, 2, 2, 'ez'), (2, 2, 2, component)],
                         3, occupancy=jnp.ones((4, 4, 4)))
    np.testing.assert_array_equal(result[:, 0], 0.)
    np.testing.assert_array_equal(result[:, 1], samples)


def test_h7_uniform_probe_energy_gradient(record_property):
    """Removing the drive weight misses its contribution to the gradient."""
    with enable_x64():
        sim = Simulation(freq_max=15e9, domain=(.020, .016, .012), dx=.001,
                         boundary='pec', precision='float64')
        sim.add_source((.006, .007, .005), 'ez', amplitude_kind='field',
                       waveform=lambda t: jnp.exp(-((t.astype(jnp.float64) - 2e-11) / 1e-11)**2))
        sim.add_probe((.010, .007, .005), 'ez')
        grid = sim._build_grid()
        def objective(value):
            occupancy = jnp.zeros(grid.shape, jnp.float64).at[5:7, 6:8, 4:6].set(value)
            record = sim.forward(n_steps=60, pec_occupancy_override=occupancy,
                                 checkpoint=False, skip_preflight=True).time_series
            assert record.dtype == jnp.float64
            return jnp.sum(record**2)
        loss = jax.jit(objective)
        value = jnp.asarray(.2, jnp.float64)
        automatic = float(jax.grad(loss)(value))
        finite = float((loss(value + 1e-3) - loss(value - 1e-3)) / 2e-3)
        relative = abs(automatic - finite) / abs(finite)
    record_property('autodiff', automatic)
    record_property('finite_difference', finite)
    record_property('relative_error', relative)
    assert np.isfinite(automatic) and abs(finite) > 0
    assert relative <= 1e-4


def test_design_window_weight_replaces_the_background():
    """The window's own value weights a source inside it (uniform forward).

    Mutation: with ``design=`` not passed the background value is used and
    the first assertion fails.
    """
    from types import SimpleNamespace
    background = np.full((6, 6, 6), np.float32(.5))
    window = SimpleNamespace(bounds=(2, 4, 2, 4, 2, 4),
                             occupancy=jnp.full((2, 2, 2), .3, jnp.float32))
    table = drive_table([jnp.ones(3, jnp.float32)], [(3, 3, 3, 'ez')], 3,
                        occupancy=jnp.asarray(background), design=window,
                        shape=background.shape)
    inside = (np.float32(1) - np.float32(.3)) ** 4
    assert abs(float(table[0, 0]) - float(inside)) <= np.spacing(np.float32(1))
    without = drive_table([jnp.ones(3, jnp.float32)], [(3, 3, 3, 'ez')], 3,
                          occupancy=jnp.asarray(background), shape=background.shape)
    assert abs(float(without[0, 0]) - float(np.float32(.5) ** 4)) <= np.spacing(np.float32(1))


def test_periodic_axis_wraps_the_incident_cell():
    """On a periodic x axis the last cell is incident on an Ez edge at i = 0.

    Mutation: with ``periodic=`` not passed the edge reads no wrapped cell
    and the weight is 1.
    """
    cells = np.zeros((6, 6, 6), np.float32)
    cells[5, 3, 3] = .3
    kwargs = dict(occupancy=jnp.asarray(cells), shape=cells.shape)
    wrapped = drive_table([jnp.ones(3, jnp.float32)], [(0, 3, 3, 'ez')], 3,
                          periodic=(True, False, False), **kwargs)
    assert abs(float(wrapped[0, 0]) - float(np.float32(1) - np.float32(.3))) <= np.spacing(np.float32(1))
    assert float(drive_table([jnp.ones(3, jnp.float32)], [(0, 3, 3, 'ez')], 3, **kwargs)[0, 0]) == 1.0
