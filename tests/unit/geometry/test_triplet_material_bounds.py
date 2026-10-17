"""Decision 3 bounds, with hand-built expectations independent of helpers."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.geometry.triplet_tensor import guarded_tensor


@pytest.mark.parametrize('xp', [np, jnp])
def test_addendum2_nonsymmetric_pi_is_spectrally_clipped(xp):
    # a/16 node (6,15): displacement (-2,5), +++ lengths zero,
    # a_x=1/16, a_y=a_z=0. Pi's only nonidentity row is its x row.
    normal = np.array([-2., 5., 0.]) / np.sqrt(29.)
    p = 1 + 99 * 25 / (16 * 29)
    q = 99 * 10 / (16 * 29)
    acc = np.array([[1 / p, -q / p, 0.], [0., 1., 0.], [0., 0., 1.]])
    raw = (acc + acc.T) / 2
    w, v = np.linalg.eigh(raw)
    assert w[-1] > 1
    expected = (v * np.clip(w, .01, 1)) @ v.T
    actual, singular = guarded_tensor(
        xp, xp.asarray(normal), xp.zeros(3), xp.asarray([1 / 16, 0., 0.]),
        100., 1., xp.eye(3) * .5)
    tol = 3e-7 if xp is jnp else 1e-14
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)
    assert not singular
    assert np.linalg.eigvalsh(actual)[0] >= .01 - tol
    assert np.linalg.eigvalsh(actual)[-1] <= 1 + tol


@pytest.mark.parametrize('xp', [np, jnp])
@pytest.mark.parametrize('bulk', [False, True])
def test_in_range_and_bulk_keep_exact_bits(xp, bulk):
    # Axis normal: Gamma_x=1/2, Pi_x=1; transverse Pi=4.
    # Different y/z lengths do not matter since their Gamma rows are I.
    lengths = xp.ones(3) if bulk else xp.asarray([1., .25, .75])
    areas = xp.ones(3) if bulk else xp.asarray([.25, 0., 0.])
    expected = xp.eye(3) / 2 if bulk else xp.diag(xp.asarray([.5, .25, .25]))
    actual, singular = guarded_tensor(xp, xp.asarray([1., 0., 0.]),
                                      lengths, areas, 2., 4., xp.eye(3))
    assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
    assert not singular


def test_a16_eps100_assembled_operator_material_bounds():
    from rfx.core.node_tensor import off_diagonal
    from rfx.geometry import Cylinder
    from rfx.geometry.node_tensor import compute_node_tensor_eps
    from rfx.grid import Grid

    dx = .01 / 16
    grid = Grid(domain=(.01, .0125, dx), dx=dx, freq_max=60e9,
                mode='2d_tez', periodic_axes='xy', cpml_axes='')
    rod = Cylinder((.005, .00625, dx / 2), .003, 1., 'z')
    eps = compute_node_tensor_eps(grid, [(rod, 100.)])

    def apply(v):
        curls = tuple(v.reshape((3,) + grid.shape))
        cross = off_diagonal(curls, eps.nodes, periodic=(True, True, True))
        return jnp.stack([c / e + x for c, e, x in zip(curls, eps, cross)]).ravel()

    xi = np.asarray(jax.jit(jax.vmap(apply))(jnp.eye(3 * np.prod(grid.shape)))).T
    np.testing.assert_array_equal(xi, xi.T)
    w = np.linalg.eigvalsh(xi.astype(np.float64))
    assert w[-1] <= 1 + 1e-6
    assert w[0] >= .01 * (1 - 1e-5)
