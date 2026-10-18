"""Sec. 7 tensors checked against independently integrated boundary fields."""
import numpy as np
import pytest

from rfx.geometry.triplet_tensor import accurate_tensor, guarded_tensor


def _inside_face_fraction(center, axis, normal, cut):
    # Clip a unit dual-face square against n.r <= cut (polygon area).
    other = [a for a in range(3) if a != axis]
    polygon = [np.array(center, copy=True) for _ in range(4)]
    for point, uv in zip(polygon, ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5))):
        point[other] += uv
    clipped = []
    for p, q in zip(polygon, polygon[1:] + polygon[:1]):
        dp, dq = p @ normal - cut, q @ normal - cut
        if dp <= 0:
            clipped.append(p)
        if (dp <= 0) != (dq <= 0):
            clipped.append(p + dp / (dp - dq) * (q - p))
    if not clipped:
        return 0.
    uv = np.array(clipped)[:, other]
    return abs(np.sum(uv[:, 0] * np.roll(uv[:, 1], -1)
                      - uv[:, 1] * np.roll(uv[:, 0], -1))) / 2


def test_planar_slab_30_degrees_exact_infinite_wavelength():
    normal = np.array([np.cos(np.pi / 6), np.sin(np.pi / 6), 0.])
    cut, eps1, eps2 = .21, 38., 2.5
    # Slab -10 < n.r < cut: its other face lies beyond all integration domains.
    tangent = np.array([-normal[1], normal[0], 0.])
    for signs in ((1, 1, 1), (-1, 1, -1), (1, -1, 1)):
        lengths, areas = [], []
        for axis, sign in enumerate(signs):
            end = sign * np.eye(3)[axis]
            projection = end @ normal
            if projection > 0:
                length = np.clip(cut / projection, 0, 1)
            elif projection < 0:
                length = 1 - np.clip(cut / projection, 0, 1)
            else:
                length = float(cut > 0)
            lengths.append(length)
            areas.append(_inside_face_fraction(end / 2, axis, normal, cut))
        lengths, areas = np.array(lengths), np.array(areas)
        acc, singular = accurate_tensor(np, normal, lengths, areas, eps1, eps2)
        assert not singular
        # Three independent boundary solutions span the map D -> E. Tangential
        # E and normal D are continuous; E_normal = D_normal / eps in each slab.
        for et, ez, dn in ((1.2, 0., 0.), (0., -.7, 0.), (0., 0., 2.3)):
            e1 = et * tangent + np.array([0., 0., ez]) + dn / eps1 * normal
            e2 = et * tangent + np.array([0., 0., ez]) + dn / eps2 * normal
            expected_e = lengths * e1 + (1 - lengths) * e2
            integrated_d = areas * eps1 * e1 + (1 - areas) * eps2 * e2
            np.testing.assert_allclose(acc @ integrated_d, expected_e, atol=2e-14, rtol=2e-14)
        tensor, guard = guarded_tensor(np, normal, lengths, areas, eps1, eps2, np.eye(3))
        assert np.array_equal(tensor, tensor.T)
        assert np.linalg.eigvalsh(tensor)[0] > 0
        assert not guard


@pytest.mark.parametrize('fraction,eps', [(0., 2.5), (1., 38.)])
def test_bulk_triplet_is_exact_isotropic_inverse(fraction, eps):
    tensor, guard = guarded_tensor(np, np.array([.6, .8, 0.]),
                                   np.full(3, fraction), np.full(3, fraction),
                                   38., 2.5, np.eye(3))
    assert np.array_equal(tensor, np.eye(3) / eps)
    assert not guard


def test_nonpositive_triplets_are_clipped_not_replaced_by_fallback():
    normal = np.array([2 ** -.5, 2 ** -.5, 0.])
    lengths = np.array([[1., 0., 0.], [0., 0., 0.]])
    areas = np.array([[0., 1., 0.], [0., 0., 0.]])
    fallback = np.diag([.3, .3, .2])
    tensor, guard = guarded_tensor(np, normal, lengths, areas, 38., 1., fallback)
    assert np.count_nonzero(guard) == 0
    assert not np.array_equal(tensor[0], fallback)
    assert np.linalg.eigvalsh(tensor[0])[0] >= 1 / 38 - 1e-15
    assert np.linalg.eigvalsh(tensor[0])[-1] <= 1 + 1e-15
    assert np.array_equal(tensor[1], np.eye(3))


def test_singular_pi_uses_node_fallback():
    # Synthetic singular Pi, to exercise the guard independently of geometry.
    fallback = np.eye(3) * .25
    tensor, guard = guarded_tensor(np, np.array([1., 0., 0.]),
                                   np.full(3, .5), np.full(3, .5),
                                   1., -1., fallback)
    assert guard
    assert np.array_equal(tensor, fallback)


def test_builder_guard_count_and_traced_triplets(monkeypatch):
    import jax
    import jax.numpy as jnp
    from rfx.geometry import Box
    from rfx.geometry import triplet_tensor as triplet
    from rfx.geometry.node_tensor import compute_node_tensor_eps
    from rfx.grid import Grid

    grid = Grid(domain=(.003, .003, .003), dx=.001, freq_max=10e9, cpml_axes='')

    def build(face):
        return compute_node_tensor_eps(grid, [(Box((-.01, -.01, -.01),
                                                   (face, .01, .01)), 38.)])

    concrete = build(.0012)
    traced = jax.jit(build)(jnp.float32(.0012))
    assert int(concrete.nodes.guard_count) == int(traced.nodes.guard_count) == 0
    np.testing.assert_allclose(traced.nodes.triplets, concrete.nodes.triplets,
                               rtol=1e-5, atol=1e-7)
    original = triplet.guarded_tensor
    counted = []

    def force_fallback(xp, normal, lengths, areas, eps1, eps2, fallback):
        tensor, guard = original(xp, normal, lengths, areas, eps1, eps2, fallback)
        touched = xp.any(lengths > 0, axis=-1) | xp.any(areas > 0, axis=-1)
        counted.append(int(np.sum(touched)))
        return xp.broadcast_to(fallback, tensor.shape), xp.ones_like(guard)

    monkeypatch.setattr(triplet, 'guarded_tensor', force_fallback)
    forced = build(.0012)
    assert int(forced.nodes.guard_count) == sum(counted) > 0


def test_rod_operator_equals_literal_eq10_and_is_exactly_symmetric():
    from itertools import product

    import jax
    import jax.numpy as jnp
    from rfx.core.node_tensor import off_diagonal
    from rfx.geometry import Cylinder
    from rfx.geometry.node_tensor import compute_node_tensor_eps
    from rfx.grid import Grid

    dx = .01 / 8
    grid = Grid(domain=(.01, .0125, dx), dx=dx, freq_max=60e9,
                mode='2d_tez', periodic_axes='xy', cpml_axes='')
    rod = Cylinder((.005, .00625, dx / 2), .003, 1., 'z')
    eps = compute_node_tensor_eps(grid, [(rod, 38.)])
    size = int(np.prod(grid.shape))

    def apply(v):
        curls = tuple(v.reshape((3,) + grid.shape))
        cross = off_diagonal(curls, eps.nodes, periodic=(True, True, True))
        return jnp.stack([c / e + x for c, e, x in zip(curls, eps, cross)]).ravel()

    assembled = np.asarray(jax.jit(jax.vmap(apply))(jnp.eye(3 * size))).T
    # Independently scatter each local 3x3 tensor onto its three periodic
    # edges, with weight 1/8; do not use coefficient or mask helpers.
    literal = np.zeros(assembled.shape, dtype=np.float64)
    triplets = np.asarray(eps.nodes.triplets)
    for node in np.ndindex(grid.shape):
        for t, signs in enumerate(product((1, -1), repeat=3)):
            ids = []
            for axis, sign in enumerate(signs):
                edge = list(node)
                edge[axis] = (edge[axis] - (sign < 0)) % grid.shape[axis]
                ids.append(axis * size + np.ravel_multi_index(tuple(edge), grid.shape))
            literal[np.ix_(ids, ids)] += triplets[(t,) + node].astype(np.float64) / 8
    # The epsilon-form diagonal and float32 coefficient storage round the
    # float64 literal sum; equality is checked at float32 roundoff.
    np.testing.assert_allclose(assembled, literal, rtol=np.finfo(np.float32).eps, atol=0)
    np.testing.assert_array_equal(assembled, assembled.T)
    np.testing.assert_array_equal(literal, literal.T)
    assert int(eps.nodes.guard_count) == 0
