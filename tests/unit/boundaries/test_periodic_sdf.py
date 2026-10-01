"""Translate a body, source and probe together through a periodic seam.

The compared tensors/weights belong to the selected physical option. The
cell has unequal y faces and no absorber. Array tolerances allow float32
arithmetic; the 300-step field comparison allows its accumulated residual.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation, Sphere
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.conformal import compute_conformal_weights_sdf
from rfx.geometry.smoothing import compute_inv_eps_tensor_diag, compute_smoothed_eps
from tests.unit.boundaries.test_periodic_fills import DX, L, model

SHIFT = 8


def grid_model(periodic=True):
    return Simulation(20e9, (L, 7.3 * DX, 5.6 * DX), dx=DX, cpml_layers=0,
                      boundary=BoundarySpec(x='periodic' if periodic else 'pec',
                                            y=Boundary(lo='pec', hi='pmc'), z='pec'))


def declaration(kind, x):
    if kind == 'box':
        return Box((x - 3.2 * DX, .9 * DX, .6 * DX),
                   (x + 2.4 * DX, 5.3 * DX, 4.1 * DX))
    return Sphere((x - .6 * DX, 3.1 * DX, 2.7 * DX), 1.8 * DX)


def configured(kind, option, shift=0):
    sim = model()
    sim.add_material('dielectric', eps_r=3.7)
    sim.add(declaration(kind, L - shift * DX),
            material='dielectric' if option == 'subpixel' else 'pec')
    return sim


def option_arrays(sim, option):
    grid = sim._build_grid()
    built = sim._assemble_materials(grid)
    if option == 'subpixel':
        return compute_smoothed_eps(grid, [(e.shape, 3.7) for e in sim._geometry])
    if option == 'conformal':
        return compute_conformal_weights_sdf(grid, built[4])
    return compute_inv_eps_tensor_diag(grid, pec_shapes=built[4])


def option_trace(kind, option, shift=0):
    sim = configured(kind, option, shift)
    sim.add_source(((.31 * L - shift * DX) % L, 2.2 * DX, 3.3 * DX),
                   'ez', amplitude_kind='field')
    sim.add_probe(((.68 * L - shift * DX) % L, 4.3 * DX, 1.2 * DX), 'ez')
    flags = {'conformal_pec': True} if option == 'conformal' else {
        'subpixel_smoothing': True if option == 'subpixel' else 'kottke_pec'}
    return np.asarray(sim.run(n_steps=300, compute_s_params=False,
                             skip_preflight=True, **flags).time_series)


@pytest.mark.parametrize('option', ['subpixel', 'conformal', 'kottke_pec'])
@pytest.mark.parametrize('kind', ['box', 'sphere'])
def test_periodic_sdf_option_realizes_translated_body(kind, option):
    seam = option_arrays(configured(kind, option), option)
    interior = option_arrays(configured(kind, option, SHIFT), option)
    for a, b in zip(seam, interior):
        np.testing.assert_allclose(np.roll(a, -SHIFT, axis=0), b, rtol=0,
                                   atol=32 * np.finfo(np.float32).eps * max(1., float(np.max(b))))


@pytest.mark.parametrize('option', ['subpixel', 'conformal', 'kottke_pec'])
@pytest.mark.parametrize('kind', ['box', 'sphere'])
def test_periodic_sdf_option_trace_translates_with_body(kind, option):
    seam = option_trace(kind, option)
    interior = option_trace(kind, option, SHIFT)
    peak = np.max(np.abs(interior))
    assert peak > 0
    # 300 updates can accumulate roundoff from translated SDF coordinates.
    np.testing.assert_allclose(seam, interior, rtol=0, atol=1e-4 * peak)


def traced_smoothing(lo, hi, *, periodic=True):
    sim = grid_model(periodic)
    sim.add_material('dielectric', eps_r=3.7)
    # Unequal face offsets also avoid a nearest-normal tie between x and y.
    sim.add(Box((lo, .87 * DX, .64 * DX), (hi, 5.23 * DX, 4.17 * DX)), material='dielectric')
    return compute_smoothed_eps(sim._build_grid(), [(sim._geometry[0].shape, 3.7)])


def test_traced_negative_box_sdf_keeps_negative_image_and_gradient():
    lo, hi = np.float32(-.9 * L), np.float32(-.2 * L)
    traced = jax.jit(traced_smoothing)(lo, hi)
    # Independent translated drawing entirely in the base period.
    reference = traced_smoothing(float(lo) + L, float(hi) + L)
    for a, b in zip(traced, reference):
        np.testing.assert_allclose(a, b, rtol=0, atol=32 * np.finfo(np.float32).eps * 3.7)
    def total(a):
        return sum(jnp.sum(e) for e in traced_smoothing(a, hi))
    grad = float(jax.grad(total)(lo))
    h = np.float32(DX / 100)
    difference = float((total(lo + h) - total(lo - h)) / (2 * h))
    assert abs(grad) > 1.
    np.testing.assert_allclose(grad, difference, rtol=.01, atol=0)


@pytest.mark.parametrize('outside', ['low', 'high'])
def test_traced_box_sdf_checks_runtime_image_bounds(outside):
    lo, hi = jnp.float32(-.9 * L), jnp.float32(-.2 * L)
    with pytest.raises(Exception, match="axis 'x'.*bounds.*L=0.0137"):
        values = jax.jit(traced_smoothing)(
            jnp.float32(-1.3 * L) if outside == 'low' else lo,
            jnp.float32(2.3 * L) if outside == 'high' else hi)
        jax.block_until_ready(values)


@pytest.mark.parametrize('periodic', [False, True])
def test_staircase_box_corner_gradient_is_refused(periodic):
    def total(lo):
        sim = grid_model(periodic)
        sim.add_material('dielectric', eps_r=3.7)
        sim.add(Box((lo, .9 * DX, .6 * DX), (8.1 * DX, 5.3 * DX, 4.1 * DX)),
                material='dielectric')
        return jnp.sum(sim._build_materials(sim._build_grid())[0].eps_r)
    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.grad(total)(jnp.float32(3.3 * DX))


def test_periodic_custom_grid_only_shape_names_coordinate_protocol():
    class GridOnly:
        def mask(self, grid):
            return jnp.ones(grid.shape, dtype=bool)
    sim = model()
    sim.add_material('dielectric', eps_r=3.7)
    sim.add(GridOnly(), material='dielectric')
    with pytest.raises(NotImplementedError, match=r'GridOnly.*mask_on_coords\(x, y, z\)'):
        sim._build_materials(sim._build_grid())


@pytest.mark.parametrize('option', ['subpixel', 'conformal', 'kottke_pec'])
def test_sdf_transition_band_reaches_across_periodic_face(option):
    def arrays(shift):
        sim = model()
        sim.add_material('dielectric', eps_r=3.7)
        # Body ends 0.2dx before L; its half-cell SDF transition crosses L.
        sim.add(Box(((11.2 - shift) * DX, .87 * DX, .64 * DX),
                    ((16.8 - shift) * DX, 5.23 * DX, 4.17 * DX)),
                material='dielectric' if option == 'subpixel' else 'pec')
        return option_arrays(sim, option)
    for a, b in zip(arrays(0), arrays(SHIFT)):
        np.testing.assert_allclose(np.roll(a, -SHIFT, axis=0), b, rtol=0,
                                   atol=32 * np.finfo(np.float32).eps * max(1., float(np.max(b))))
