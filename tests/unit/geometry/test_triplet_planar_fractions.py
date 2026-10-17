"""DR6 independent hand cuts and 2000 x 2000 midpoint area counts."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.geometry.triplet_tensor import build_triplets, fractions

# A unit cubic cell, asked per axis as the builder asks a uniform Grid.
UNIT_GRID = SimpleNamespace(cells=lambda axis: np.ones(4), is_constant=lambda axis: True)


@pytest.mark.parametrize('xp', [np, jnp])
def test_hand_computed_edges_and_parallel_derivative(xp):
    s = np.array([-.25, .25, -.75, .75, -.25, .25, 0.])
    nx = np.array([1., -1., 1., -1., 0., 0., 0.])
    normal = (nx, np.sqrt(1 - nx**2), np.zeros(7))
    lengths, _ = fractions(xp, UNIT_GRID, None, None,
                          signed_distance=s, normal=normal)
    np.testing.assert_array_equal(lengths[0][0], [.25, .75, .75, .25, 1., 0., 0.])
    np.testing.assert_array_equal(lengths[0][1], [1., 0., 1., 0., 1., 0., 0.])
    if xp is jnp:
        def edge(s):
            return fractions(jnp, UNIT_GRID, None, None,
                             signed_distance=s, normal=(0., 1., 0.))[0][0][0]
        assert np.isfinite(jax.grad(edge)(jnp.float32(-.25)))


@pytest.mark.parametrize('xp', [np, jnp])
def test_faces_against_independent_grid_and_continuity(xp):
    rng = np.random.default_rng(1632)
    normal = rng.normal(size=(20, 3))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    s = rng.uniform(-.8, .8, size=20)
    args = dict(signed_distance=s, normal=tuple(normal.T))
    lengths, areas = fractions(xp, UNIT_GRID, None, None, **args)
    # Independent brute-force midpoint count, absolute tolerance 1e-5.
    u = (np.arange(2000) + .5) / 2000 - .5
    for i in range(20):
        for axis in range(3):
            other = [a for a in range(3) if a != axis]
            for index, sign in enumerate((1, -1)):
                phi = (s[i] + sign * normal[i, axis] / 2
                       + u[:, None] * normal[i, other[0]]
                       + u[None, :] * normal[i, other[1]])
                expected = np.count_nonzero(phi < 0) / 2000**2
                assert abs(float(areas[axis][index][i]) - expected) <= 1e-5
    shifted = fractions(xp, UNIT_GRID, None, None,
                        signed_distance=s + 1e-9, normal=tuple(normal.T))
    for before, after in zip((lengths, areas), shifted):
        assert np.max(np.abs(np.asarray(before) - np.asarray(after))) <= 1e-6


@pytest.mark.parametrize('xp', [np, jnp])
@pytest.mark.parametrize('s,eps', [(-2., 38.), (2., 2.5)])
def test_planar_builder_preserves_exact_bulk(xp, s, eps):
    tensor, guards, _ = build_triplets(
        xp, UNIT_GRID, None, None, (.6, .8, 0.),
        38., 2.5, xp.eye(3), signed_distance=s)
    np.testing.assert_array_equal(tensor, xp.broadcast_to(xp.eye(3) / eps, (8, 3, 3)))
    assert not np.any(guards)
