"""#2397: declared normal-line content, independent laminate and area oracles."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Cylinder, Simulation, Sphere
from rfx.geometry.node_tensor import compute_node_tensor_eps
from rfx.geometry.smoothing import (
    _depth_cdf, compute_inv_eps_tensor_diag, compute_smoothed_eps,
    compute_smoothed_eps_nonuniform,
)


def grid_of(graded=False):
    profiles = dict(dx_profile=np.full(8, .0005), dy_profile=np.full(8, .0005),
                    dz_profile=np.array([.5, .5, .6, .4, .5, .6, .4, .5]) * .001) if graded else {}
    return Simulation(domain=(.004,) * 3, dx=.0005, freq_max=12e9,
                      boundary='pec', cpml_layers=0, **profiles).freeze_mesh()


def pair(faces, eps=4., cylinder=False, xhi=.1):
    a, b = faces
    if cylinder:
        return [(Cylinder((0., 0., (-.1 + a) / 2), .1, a + .1, axis='z'), 4.),
                (Cylinder((0., 0., (b + .1) / 2), .1, .1 - b, axis='z'), eps)]
    return [(Box((-.1, -.1, -.1), (.1, .1, a)), 4.),
            (Box((-.1, -.1, b), (xhi, .1, .1)), eps)]


def samples(result):
    return jnp.stack((result[2][4, 4, 4], result[0][4, 4, 4], result[0][4, 4, 5],
                      result[1][4, 4, 4], result[1][4, 4, 5]))


@pytest.mark.parametrize('eps', [4., 6.])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('traced', [False, True])
@pytest.mark.parametrize('mode', ['yee', 'pec'])
def test_b1_and_b1prime(eps, reverse, traced, mode):
    grid = grid_of()

    def evaluate(faces):
        shapes = pair(faces, eps)
        if reverse:
            shapes.reverse()
        return samples(compute_inv_eps_tensor_diag(grid, dielectric_shapes=shapes)
                       if mode == 'pec' else compute_smoothed_eps(grid, shapes))

    faces = np.array([.00218, .00228])
    result = jax.jit(evaluate)(jnp.asarray(faces)) if traced else evaluate(faces)
    expected = np.array([.5 / (.18 / 4 + .10 + .22 / eps), 3.58,
                         (.03 + .47 * eps) / .5, 3.58, (.03 + .47 * eps) / .5])
    if mode == 'pec':
        expected = 1 / expected
    np.testing.assert_allclose(result, expected, rtol=2e-5 if traced else 1e-6)


@pytest.mark.parametrize('traced', [False, True])
def test_b1doubleprime_paint(traced):
    grid = grid_of()

    def evaluate(z):
        shapes = [(Box((-.1, -.1, z), (.1, .1, .1)), 4.),
                  (Box((-.1, -.1, z), (.1, .1, .003)), 10.),
                  (Box((-.1, -.1, -.1), (.1, .1, .00218)), 2.)]
        return compute_smoothed_eps(grid, shapes)[2][4, 4, 4]

    result = jax.jit(evaluate)(jnp.float32(.00228)) if traced else evaluate(.00228)
    np.testing.assert_allclose(result, .5 / (.18 / 2 + .10 + .22 / 10), rtol=2e-5 if traced else 1e-6)


def test_b1g_graded():
    result = compute_smoothed_eps_nonuniform(grid_of(True), pair((.00218, .00228)))
    np.testing.assert_allclose(samples(result), [2.5, 3.7, 3.7, 3.7, 3.7], rtol=1e-6)


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('cylinder', [False, True], ids=['B2b', 'B2'])
def test_b2_planar_and_box_scope(graded, cylinder):
    result = compute_node_tensor_eps(grid_of(graded), pair((.00218, .00228), cylinder=cylinder))
    low, high = ((.38 * 4 + .07) / .45, (.03 + .52 * 4) / .55) if graded else (3.58, 3.82)
    np.testing.assert_allclose(samples(result), [2.5, low, high, low, high], rtol=1e-6)
    if not graded:
        np.testing.assert_allclose([result.nodes.xi[a][4, 4, 4] for a in range(3)],
                                   [1 / 3.58, 1 / 3.58, (.43 / 4 + .07) / .5], rtol=1e-6)


@pytest.mark.parametrize('normal', [(0.6, 0.8, 0.), (0.48, 0.6, 0.64), (1., 0., 0.)])
def test_b2c_depth_cdf(normal):
    # Row x, transverse widths 0.5 and 0.4 mm, midpoint quadrature.
    a, b = sorted((abs(.0005 * normal[1]) / 2, abs(.0004 * normal[2]) / 2), reverse=True)
    points = (np.arange(2000) + .5) / 2000 - .5
    depths = .0005 * normal[1] * points[:, None] + .0004 * normal[2] * points[None, :]
    for t in (-a-b, -a+b/2, 0., a-b/2, a+b):
        expected = np.count_nonzero(depths < t) / 2000**2
        np.testing.assert_allclose(_depth_cdf(np, np.asarray(t), a, b), expected, atol=1e-5, rtol=0)


def test_b6_far_wall_gradient():
    grid = grid_of()
    derivative = jax.grad(lambda z: compute_smoothed_eps(grid, pair((.00218, z)))[2][4, 4, 4])(jnp.float32(.00228))
    np.testing.assert_allclose(derivative, -9375., rtol=1e-4)


def floating_arrays(result):
    if hasattr(result, 'nodes'):
        return (*result.diagonal, *result.nodes.xi, result.nodes.triplets, *result.nodes.cross)
    return result


@pytest.mark.parametrize('case', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('eps', [4., 6.])
@pytest.mark.parametrize('mode', ['yee', 'cylinder'])
def test_b8_continuity(case, eps, mode):
    grid = grid_of()
    base = {1: .00218, 2: .00218 + 1e-12, 3: .00228, 4: .00225, 5: .00228}[case]
    results = []
    for delta in (-1e-12, 1e-12):
        shapes = pair((.00218, base + (delta if case != 5 else 0)), eps,
                      cylinder=mode == 'cylinder', xhi=.0024 + delta if case == 5 else .1)
        results.append(floating_arrays(compute_node_tensor_eps(grid, shapes) if mode == 'cylinder'
                                       else compute_smoothed_eps(grid, shapes)))
    for a, b in zip(*results):
        a, b = np.asarray(a), np.asarray(b)
        peak = np.float32(max(np.max(np.abs(a)), np.max(np.abs(b))))
        error = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
        assert error <= np.spacing(peak)


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('scene', ['cylinder-x', 'cylinder-y', 'cylinder-z', 'box-sphere', 'bisector'])
def test_b8_case6_known_limitations(graded, scene):
    """The step-0 nonbox record's rim, nearby sphere and bisector geometries."""
    profile = np.array([.001] * 4 + [.0008] * 5 + [.001] * 4)
    grid = Simulation(domain=(.012,) * 3, dx=.001, freq_max=20e9, boundary='pec',
                      **({'dz_profile': profile} if graded else {})).freeze_mesh()
    results = []
    for delta in (-1e-12, 1e-12):
        if scene.startswith('cylinder'):
            axis = scene[-1]
            center = [.006, .006, .0064 if graded else .006]
            center['xyz'.index(axis)] += delta
            radius = .0016 if graded and axis != 'z' else .002
            height = .0032 if graded and axis == 'z' else .004
            shapes = [(Cylinder(tuple(center), radius, height, axis), 10.)]
        elif scene == 'box-sphere':
            shapes = [(Box((.005 + delta, .004, .004), (.009, .008, .008)), 10.),
                      (Sphere((.0025, .004, .006), .0015), 10.)]
        else:
            shapes = [(Box((.0046 + delta, .0036, .0033), (.009, .0084, .0083)), 10.),
                      (Sphere((.0021, .0036, .006), .0015), 10.)]
        results.append(compute_smoothed_eps_nonuniform(grid, shapes) if graded
                       else compute_smoothed_eps(grid, shapes))
    for a, b in zip(*results):
        a, b = np.asarray(a), np.asarray(b)
        peak = np.float32(max(np.max(np.abs(a)), np.max(np.abs(b))))
        assert np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))) <= np.spacing(peak)
