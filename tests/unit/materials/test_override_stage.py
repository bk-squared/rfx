"""Cell overrides keep each lane's material convention and edge-owned loads.

References use cells and one-dimensional widths, never a product mean helper.
Main-versus-move bit identity (including the radius model) is captured separately.
"""
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import MaterialArrays
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import lorentz_pole
from rfx.model.materials import ComponentCells, _design_box_edge_coeffs, realize_components
from rfx.model.overrides import apply_material_overrides
from tests.unit.materials.test_dual_volume_edges import MetricGrid, reference


def _grid(graded=False, n=4):
    return MetricGrid([np.arange(1, n + 1, dtype=float) if graded else
                       np.ones(n) for _ in range(3)])


def _cells(shape):
    return MaterialArrays(jnp.ones(shape), jnp.zeros(shape), jnp.ones(shape))


def _stamp(m, field, axis, cell, value):
    stamp = jnp.zeros_like(getattr(m, field)).at[cell].set(value)
    record = [None] * 3
    record[axis] = stamp
    name = {"eps_r": "eps_r_lumped", "sigma": "sigma_lumped", "mu_r": "mu_r_wire"}[field]
    # Wire contours live only in their H record, not in cell permeability.
    return m._replace(**{field: getattr(m, field) if field == "mu_r" else
                         getattr(m, field) + stamp, name: tuple(record)})


def _face_reference(values, widths, axis, cell, periodic):
    lo = list(cell)
    lo[axis] = (lo[axis] - 1) % values.shape[axis] if periodic[axis] else max(lo[axis] - 1, 0)
    lo = tuple(lo)
    dl, dh = widths[axis][lo[axis]], widths[axis][cell[axis]]
    den = dl / values[lo] + dh / values[cell]
    grad = np.zeros_like(values, dtype=float)
    grad[lo] += (dl + dh) * dl / (values[lo] ** 2 * den ** 2)
    grad[cell] += (dl + dh) * dh / (values[cell] ** 2 * den ** 2)
    return (dl + dh) / den, grad


@pytest.mark.parametrize("field,keyword,component", [
    ("eps_r", "eps_override", "eps_update"),
    ("sigma", "sigma_override", "sigma_update"),
    ("mu_r", "mu_r_override", "mu_update"),
])
@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("periodic", [(False,) * 3, (True,) * 3])
def test_cells_are_overridden_before_realization_and_stamps_stay_on_their_edges(
        field, keyword, component, graded, periodic):
    grid = _grid(graded)
    values = (np.arange(64, dtype=np.float32).reshape(grid.shape) + 8) / 8
    point = (2, 2, 2)

    def consume(a):
        m = apply_material_overrides(_cells(grid.shape), **{keyword: a})
        m = _stamp(m, field, 2, point, 2.)
        return getattr(realize_components(m, grid, periodic=periodic), component)

    realized = consume(jnp.asarray(values))
    oracle = _face_reference if field == "mu_r" else reference
    for axis, cell in product(range(3), np.ndindex(grid.shape)):
        expected, _ = oracle(values, grid.widths, axis, cell, periodic)
        expected += 2. if axis == 2 and cell == point else 0.
        np.testing.assert_allclose(realized[axis][cell], expected, rtol=3e-7)
    _, expected_grad = oracle(values, grid.widths, 2, point, periodic)
    grad = jax.grad(lambda a: consume(a)[2][point])(jnp.asarray(values))
    np.testing.assert_allclose(grad, expected_grad, rtol=3e-7, atol=1e-8)
    direction = np.linspace(.25, 1., values.size, dtype=np.float32).reshape(grid.shape)
    _, tangent = jax.jvp(lambda a: consume(a)[2][point],
                         (jnp.asarray(values),), (jnp.asarray(direction),))
    np.testing.assert_allclose(tangent, np.sum(expected_grad * direction), rtol=3e-7)


@pytest.mark.parametrize("field,keyword,operand", [
    ("eps_r", "eps_override", "upml_eps"),
    ("sigma", "sigma_override", "upml_sigma"),
])
def test_upml_override_keeps_the_cell_owned_convention(field, keyword, operand):
    grid = _grid()
    values = jnp.arange(64, dtype=jnp.float32).reshape(grid.shape) / 8 + 1
    def consume(a):
        m = apply_material_overrides(_cells(grid.shape), **{keyword: a})
        return getattr(realize_components(_stamp(m, field, 2, (2, 2, 2), 2.),
                                          grid, periodic=(False,) * 3), operand)
    actual = consume(values)
    for axis in range(3):
        expected = values.at[2, 2, 2].add(2.) if axis == 2 else values
        np.testing.assert_array_equal(actual[axis], expected)
    grad = jax.grad(lambda a: consume(a)[2][2, 2, 2])(values)
    np.testing.assert_array_equal(grad, jnp.zeros_like(values).at[2, 2, 2].set(1.))


def test_override_keeps_debye_and_lorentz_occupancy():
    grid = _grid(True)
    mask = np.indices(grid.shape).sum(axis=0) % 3 == 0
    debye = ([DebyePole(.5, 1e-11)], mask)
    lorentz = ([lorentz_pole(.4, 2 * np.pi * 3e9, 1e9)], ~mask)
    cells = _cells(grid.shape)
    base = realize_components(ComponentCells(cells, debye, lorentz), grid, periodic=(False,) * 3)
    def consume(eps):
        changed = apply_material_overrides(cells, eps_override=eps,
                                           sigma_override=jnp.ones(grid.shape))
        c = realize_components(ComponentCells(changed, debye, lorentz), grid, periodic=(False,) * 3)
        return c.debye.coefficients, c.lorentz.coefficients
    actual, tangent = jax.jvp(consume, (jnp.full(grid.shape, 3.),), (jnp.ones(grid.shape),))
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves((base.debye.coefficients, base.lorentz.coefficients))):
        np.testing.assert_array_equal(a, b)
    assert all(np.count_nonzero(a) == 0 for a in jax.tree.leaves(tangent))


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("per_edge", [False, True])
def test_design_box_cells_held_edges_and_edge_sigma(graded, held, per_edge):
    grid = _grid(graded, n=6)
    bounds = (2, 4, 2, 4, 2, 4)
    sl = (slice(2, 4),) * 3
    m = _stamp(_stamp(_cells(grid.shape), "eps_r", 2, (3, 3, 3), 2.),
               "sigma", 2, (3, 3, 3), 2.)
    eps = jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2) / 4 + 2
    sigma = jnp.full((2, 2, 2), .125)
    sigma_arg = (sigma, sigma * 2, sigma * 3) if per_edge else sigma
    held_edges = ((2, 3, 3, 3),) if held else ()
    write, ca, cb = _design_box_edge_coeffs(bounds, eps, sigma_arg, m,
                                           grid.dt, grid.shape, held_edges, grid=grid)
    full_eps = np.ones(grid.shape, np.float32)
    full_eps[sl] = eps
    full_sigma = np.zeros(grid.shape, np.float32)
    if not per_edge:
        full_sigma[sl] = sigma
    for axis, cell in product(range(3), np.ndindex(ca[0].shape)):
        index = tuple(cell[t] + write[2 * t] for t in range(3))
        ep, _ = reference(full_eps, grid.widths, axis, index, (False,) * 3)
        sg, _ = reference(full_sigma, grid.widths, axis, index, (False,) * 3)
        at_stamp = axis == 2 and index == (3, 3, 3)
        ep += 2. if at_stamp else 0.
        sg += 2. if at_stamp else 0.
        if per_edge and all(2 <= i < 4 for i in index):
            sg = .125 * (axis + 1)
        if held and at_stamp:
            ep, sg = 3., 2.
        den = ep * 8.8541878128e-12 + sg * grid.dt / 2
        np.testing.assert_allclose(ca[axis][cell], (ep * 8.8541878128e-12 - sg * grid.dt / 2) / den,
                                   rtol=3e-7, atol=1e-7)
        np.testing.assert_allclose(cb[axis][cell], grid.dt / den, rtol=3e-7)
    derivative = jax.grad(lambda a: _design_box_edge_coeffs(
        bounds, a, sigma_arg, m, grid.dt, grid.shape, held_edges, grid=grid)[2][2][1, 1, 1])(eps)
    assert bool(jnp.all(derivative == 0)) == held


def test_whole_grid_override_preserves_the_input_buffer_for_slab_staging():
    cells = _cells((4, 3, 2))
    override = jnp.arange(24, dtype=jnp.float32).reshape(4, 3, 2)
    result = apply_material_overrides(cells, eps_override=override)
    assert result.eps_r is override
    assert result.sigma is cells.sigma and result.mu_r is cells.mu_r
    assert result.components is None
    assert apply_material_overrides(cells) is cells
