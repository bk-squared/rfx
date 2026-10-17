"""#1632 T4/T5/T7: independent clip derivative and PEC symmetry judges."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.geometry import Cylinder
from rfx.geometry.node_tensor import compute_node_tensor_eps
from rfx.geometry import triplet_tensor as triplet
from rfx.grid import Grid


def numpy_clip(tensor, lo, hi):
    # Independent host oracle; never calls the implementation's clip helpers.
    w, v = np.linalg.eigh(np.asarray(tensor, dtype=np.float64))
    return (v * np.minimum(np.maximum(w, lo), hi)) @ v.T


@pytest.mark.parametrize('tied', [True, False], ids=['tie', 'separated'])
def test_spectral_clip_jvp(tied):
    with jax.enable_x64():
        # A 45-degree rotation of diag(1.25, 1.25, .5), with an exact tie.
        tensor = np.array([[.875, 0., .375], [0., 1.25, 0.], [.375, 0., .875]])
        if not tied:
            tensor[1, 1] = .8
        direction = np.array([[.2, .1, -.3], [.1, -.4, .15], [-.3, .15, .5]])
        lo, hi, dlo, dhi = .6, 1., .07, -.09
        args = tuple(jnp.asarray(a) for a in (tensor, lo, hi))
        tangents = tuple(jnp.asarray(a) for a in (direction, dlo, dhi))
        _, actual = jax.jvp(triplet._spectral_clip, args, tangents)
        h = 1e-5
        expected = (numpy_clip(tensor + h * direction, lo + h * dlo, hi + h * dhi)
                    - numpy_clip(tensor - h * direction, lo - h * dlo, hi - h * dhi)) / (2 * h)
        assert np.all(np.isfinite(actual)), actual
        relative = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        print(f'clip tied={tied}, FD relative error={relative:.17g}')
        assert relative <= 1e-6
        if not tied:
            def plain(t, l, u):
                w, v = jnp.linalg.eigh(t)
                return (v * jnp.clip(w, l, u)) @ v.T
            _, derivative = jax.jvp(plain, args, tangents)
            relative = np.linalg.norm(actual - derivative) / np.linalg.norm(derivative)
            print(f'clip separated eigh relative error={relative:.17g}')
            assert relative <= 1e-6


def test_pec_wall_operator_is_symmetric():
    from rfx.core.node_tensor import off_diagonal
    from rfx.core.yee import CurlBoundary

    with jax.enable_x64():
        dx = .001
        grid = Grid(domain=(.004, .005, dx), dx=dx, freq_max=10e9,
                    mode='2d_tez', cpml_axes='')
        # The circle cuts x=0 obliquely. Both normal components are nonzero
        # at the wall; periodic y/z remove unrelated terminal faces.
        rod = Cylinder((.0004, .0023, dx / 2), .0017, 1., 'z')
        eps = compute_node_tensor_eps(grid, [(rod, 38.)], field_dtype=jnp.float64)
        boundary = CurlBoundary(pec_faces=frozenset({'x_lo'}), periodic=(False, True, True))
        count = 3 * int(np.prod(grid.shape))
        def apply(vector):
            curls = tuple(vector.reshape((3,) + grid.shape))
            cross = off_diagonal(curls, eps.nodes, periodic=boundary.periodic, boundary=boundary)
            # Output PEC projection is the step's wall condition, independent
            # of the input mask being tested. x-lo tangential edges are y,z.
            result = jnp.stack(cross)
            return result.at[1:, 0, :, :].set(0.).ravel()
        operator = np.asarray(jax.jit(jax.vmap(apply))(jnp.eye(count))).T
        maximum = np.max(np.abs(operator))
        assert maximum > 0
        relative = np.max(np.abs(operator - operator.T)) / maximum
        print(f'PEC symmetry max|A-A.T|/max|A|={relative:.17g}')
        assert relative <= 1e-12


def test_chunked_traced_build_and_jvp_equal_unchunked(monkeypatch):
    grid = Grid(domain=(.005, .005, .003), dx=.001, freq_max=10e9, cpml_axes='')
    def build(radius):
        rod = Cylinder((.0024, .0026, .0015), radius, .008, 'z')
        return compute_node_tensor_eps(grid, [(rod, 38.)])
    def evaluate():
        return jax.jit(lambda r: jax.jvp(build, (r,), (jnp.ones_like(r),)))(jnp.float32(.0017))
    expected = evaluate()
    jax.clear_caches()
    monkeypatch.setattr(triplet, 'MATRIX_BATCH_SIZE', 64)
    actual = evaluate()
    for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(value, reference)
    print('chunked build: 0 ULP primal and radius JVP')
