"""Decision record 3: material composition and traced node geometry."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.geometry import Box, Cylinder
from rfx.geometry.node_tensor import compute_node_tensor_eps
from rfx.grid import Grid


def _grid():
    return Grid(domain=(.01, .01, .01), dx=.001, freq_max=10e9,
                cpml_axes='')


def _matrices(xi):
    xx, yy, zz, xy, xz, yz = (np.asarray(a) for a in xi)
    return np.stack([np.stack(row, axis=-1) for row in
                     ((xx, xy, xz), (xy, yy, yz), (xz, yz, zz))], axis=-2)


def test_touching_substrate_and_rod_compose_spd():
    grid = _grid()
    substrate = Box((-.1, -.1, -.1), (.1, .1, .004))
    rod = Cylinder((.005, .005, .006), .0024, .004, 'z')
    prior = compute_node_tensor_eps(grid, [(substrate, 4.)])
    result = compute_node_tensor_eps(grid, [(substrate, 4.), (rod, 38.)])
    matrices = _matrices(result.nodes.xi)
    assert np.array_equal(matrices, matrices.swapaxes(-1, -2))
    assert np.linalg.eigvalsh(matrices.astype(np.float64)).min() > 0
    # At the touching face both volumes have f=1/2 and normal along z.
    index = (5, 5, 4)
    old = _matrices(prior.nodes.xi)[index]
    assert old[0, 0] != old[2, 2]  # prior is an interface, not isotropic
    outside = 1. / np.mean(np.diag(old))
    expected = np.diag([1. / (.5 * 38. + .5 * outside)] * 2
                       + [.5 / 38. + .5 / outside])
    np.testing.assert_allclose(matrices[index], expected, rtol=1e-6, atol=0)
    reverse = compute_node_tensor_eps(grid, [(rod, 38.), (substrate, 4.)])
    assert not np.array_equal(matrices[index], _matrices(reverse.nodes.xi)[index])


def test_traced_radius_gradient_matches_concrete_central_fd():
    grid = _grid()

    def objective(radius):
        rod = Cylinder((.005, .005, .005), radius, .008, 'z')
        xi = compute_node_tensor_eps(grid, [(rod, 38.)]).nodes.xi
        index = (7, 6, 5)  # partial fill, away from clipping and normal ties
        return xi[0][index] + .2 * xi[1][index] + .3 * xi[3][index]

    radius, h = .0024, 1e-6
    gradient = float(jax.grad(objective)(jnp.float32(radius)))
    fd = (float(objective(radius + h)) - float(objective(radius - h))) / (2 * h)
    assert np.isfinite(gradient) and gradient != 0
    assert gradient == pytest.approx(fd, rel=1e-3, abs=0)
    print(f'node radius gradient={gradient}, concrete FD={fd}, '
          f'relative error={abs(gradient - fd) / abs(fd)}')
