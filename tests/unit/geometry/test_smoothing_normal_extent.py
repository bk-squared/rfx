"""Flat eps_r=4 slab: independent Yee averaging-cell fill fractions.

FACT: f = clip(0.5 - signed_distance/L, 0, 1); eps_normal =
1/(f/4 + 1-f), eps_tangent = 4*f + 1-f. L is primal along the
component and dual across it. Host float64 tolerance: 1e-12 absolute;
traced float64 tolerance: 1e-12 absolute. Cubic float32 is bit-identical
with the closed-form scalar-dx result on main (no stored oracle).
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation, Sphere
from rfx.geometry.node_tensor import compute_node_tensor_eps
from rfx.geometry.rasterize_grid import (
    centres_from_nonuniform_grid, coords_from_nonuniform_grid,
)
from rfx.geometry.smoothing import (
    compute_inv_eps_tensor_diag, compute_smoothed_eps, compute_smoothed_eps_nonuniform,
)
from rfx.grid import Grid
from rfx.model.graded_grid import make_nonuniform_grid
from tests._x64_compat import enable_x64


def slab_case(axis, structure):
    profiles = [np.full(8, d) for d in (0.2e-3, 0.25e-3, 0.3e-3)]
    if structure == 'cubic':
        profiles = [np.full(8, 0.2e-3) for _ in range(3)]
    elif structure.startswith('normal'):
        profiles[axis] = np.r_[np.full(4, 0.15e-3), np.full(3, 0.25e-3), 0.15e-3]
    elif structure == 'tangent':
        profiles[(axis + 1) % 3] = np.r_[np.full(4, 0.15e-3), np.full(3, 0.25e-3), 0.15e-3]
    widths = profiles[axis]
    face = widths[:4].sum() + (0 if structure == 'normal_joint' else 0.3 * widths[4])
    high = widths[:7].sum() + 0.15 * widths[7]
    lo, hi = [-1.] * 3, [1.] * 3
    lo[axis], hi[axis] = face, high
    grid = make_nonuniform_grid(
        tuple(float(p.sum()) for p in profiles[:2]), profiles[2], profiles[0][0],
        dx_profile=profiles[0], dy_profile=profiles[1], cpml_layers=0, cpml_axes='')
    return profiles, grid, Box(tuple(lo), tuple(hi))


def closed_form(profiles, axis, box, component):
    # Non-periodic grids store the high boundary node with its own last width.
    primal = np.r_[profiles[axis], profiles[axis][-1]]
    nodes = np.r_[0., np.cumsum(profiles[axis])]
    dual = np.r_[primal[0], (primal[:-1] + primal[1:]) / 2]
    position = nodes + primal / 2 if component == axis else nodes
    length = primal if component == axis else dual
    distance = np.maximum(box.corner_lo[axis] - position, position - box.corner_hi[axis])
    fill = np.clip(0.5 - distance / length, 0, 1)
    eps = 1 / (fill / 4 + (1 - fill)) if component == axis else 4 * fill + (1 - fill)
    shape = tuple(len(primal) if a == axis else 1 for a in range(3))
    return eps.reshape(shape), fill.reshape(shape), distance.reshape(shape)


@pytest.mark.parametrize('axis', range(3), ids=list('xyz'))
@pytest.mark.parametrize('structure', ['noncubic', 'normal_region', 'normal_joint', 'tangent'])
@pytest.mark.parametrize('traced', [False, True], ids=['host', 'traced'])
def test_profile_flat_slab(axis, structure, traced):
    profiles, grid, box = slab_case(axis, structure)
    with enable_x64(True):
        def evaluate(face):
            lo = list(box.corner_lo)
            lo[axis] = face
            return compute_smoothed_eps_nonuniform(grid, [(Box(tuple(lo), box.corner_hi), 4.)])
        actual = (jax.jit(evaluate)(box.corner_lo[axis]) if traced else evaluate(box.corner_lo[axis]))
        for component, value in enumerate(actual):
            expected, _, _ = closed_form(profiles, axis, box, component)
            expected = np.broadcast_to(expected, grid.shape)
            np.testing.assert_allclose(value, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize('axis', range(3), ids=list('xyz'))
def test_cubic_scalar_main_closed_form_bit_identical(axis):
    profiles, _, box = slab_case(axis, 'cubic')
    grid = Grid(freq_max=30e9, domain=(1.6e-3,) * 3, dx=0.2e-3, cpml_layers=0)
    with enable_x64(False):
        dielectric = compute_smoothed_eps(grid, [(box, 4.)])
        inverse = compute_inv_eps_tensor_diag(grid, dielectric_shapes=[(box, 4.)])
        pec = compute_inv_eps_tensor_diag(grid, pec_shapes=[box])
        for component in range(3):
            expected, fill, distance = closed_form(profiles, axis, box, component)
            def full(value):
                return np.broadcast_to(value, grid.shape).astype(np.float32)
            np.testing.assert_array_equal(dielectric[component], full(expected))
            np.testing.assert_array_equal(inverse[component], full(1 / expected))
            pec_expected = (1 - fill) if component == axis else np.where(fill > 0, 0., 1.)
            pec_expected = np.where(distance <= 0, 0., pec_expected)
            np.testing.assert_array_equal(pec[component], full(pec_expected))


@pytest.mark.parametrize('axis', range(3), ids=list('xyz'))
def test_full_tensor_node_dual_fill_and_midpoint_fallback(axis):
    profiles, grid, box = slab_case(axis, 'normal_region')
    with enable_x64(True):
        # One padded transverse slice exercises the diagonal midpoint fallback.
        grid = grid._replace(**{f'pad_{"xyz"[(axis + 1) % 3]}_hi': 1})
        actual = compute_node_tensor_eps(grid, [(box, 4.)], field_dtype=jnp.float64)
        for component in range(3):
            expected, _, _ = closed_form(profiles, axis, box, component)
            index = [slice(None)] * 3
            index[(axis + 1) % 3] = -1
            np.testing.assert_allclose(np.asarray(actual[component])[tuple(index)],
                                       np.broadcast_to(expected, grid.shape)[tuple(index)],
                                       rtol=0, atol=1e-12)
        # At nodes every extent is dual. Tangential E samples share that position.
        tangent = (axis + 1) % 3
        _, fill, _ = closed_form(profiles, axis, box, tangent)
        for component in range(3):
            expected = fill / 4 + 1 - fill if component == axis else 1 / (4 * fill + 1 - fill)
            np.testing.assert_allclose(actual.nodes.xi[component],
                                       np.broadcast_to(expected, grid.shape), rtol=0, atol=1e-12)


@pytest.mark.parametrize('node_tensor', [False, True], ids=['yee-ez', 'node-tensor'])
def test_zero_normal_radius_gradient(node_tensor):
    _, grid, _ = slab_case(2, 'noncubic')
    with enable_x64(True):
        nodes = coords_from_nonuniform_grid(grid)
        centres = centres_from_nonuniform_grid(grid, nodes)
        center = (float(nodes[0][4]), float(nodes[1][4]),
                  float(nodes[2][3] if node_tensor else centres[2][2]))

        def build(radius):
            shapes = [(Sphere(center, radius), 4.)]
            if node_tensor:
                return compute_node_tensor_eps(grid, shapes, field_dtype=jnp.float64)
            return compute_smoothed_eps_nonuniform(grid, shapes)

        def loss(radius):
            return sum(jnp.sum(a) for a in build(radius))

        radius, step = 0.35e-3, 1e-8
        gradient = jax.grad(loss)(radius)
        assert np.isfinite(gradient)
        # Independent central finite difference of the public host build;
        # no normal or cell-length implementation helper supplies the oracle.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', RuntimeWarning)
            host = build(radius)
            upper = sum(np.sum(np.asarray(a)) for a in build(radius + step))
            lower = sum(np.sum(np.asarray(a)) for a in build(radius - step))
        assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert all(np.isfinite(np.asarray(a)).all() for a in host)
        np.testing.assert_allclose(gradient, (upper - lower) / (2 * step), rtol=1e-3, atol=0)


def test_forward_with_traced_shape_on_profile_mesh():
    def forward(radius):
        sim = Simulation(freq_max=30e9, domain=(.0016, .002, 0), dx=.0002,
                         dx_profile=np.full(8, .0002), dy_profile=np.full(8, .00025),
                         dz_profile=np.r_[np.full(4, .00015), np.full(4, .00025)],
                         boundary='pec', cpml_layers=0)
        sim.add_material('sphere', eps_r=4.)
        sim.add(Sphere((.0008, .001, .0008), radius), material='sphere')
        sim.add_source((.0006, .00075, .0004), 'ex',
                       waveform=GaussianPulse(f0=15e9, bandwidth=.8), amplitude_kind='current')
        sim.add_probe((.0008, .001, .0008), 'ex')
        return sim.forward(n_steps=12, skip_preflight=True, subpixel_smoothing=True).time_series
    evaluate = jax.jit(forward)
    traced = np.asarray(evaluate(.00035))
    changed = np.asarray(evaluate(.0004))
    assert np.isfinite(traced).all() and np.any(traced != 0)
    assert np.isfinite(changed).all() and not np.array_equal(changed, traced)
