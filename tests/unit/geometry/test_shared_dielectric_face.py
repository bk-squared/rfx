"""Bulk media on both sides of a shared face, without time stepping."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.geometry.node_tensor import compute_node_tensor_eps
from rfx.geometry.smoothing import (
    compute_inv_eps_tensor_diag, compute_smoothed_eps, compute_smoothed_eps_nonuniform,
)


@pytest.mark.parametrize('path', ['uniform', 'graded', 'kottke_pec', 'nodes', 'graded_nodes'])
@pytest.mark.parametrize('traced', [False, True], ids=['host', 'traced'])
@pytest.mark.parametrize('reverse', [False, True], ids=['AB', 'BA'])
def test_shared_face_operands(path, traced, reverse):
    graded = path.startswith('graded')
    profiles = dict(dx_profile=np.full(8, .0005), dy_profile=np.full(8, .0005),
                    dz_profile=np.r_[np.full(4, .0004), np.full(4, .0006)]) if graded else {}
    sim = Simulation(domain=(.004, .004, .004), dx=.0005, freq_max=12e9,
                     boundary='pec', cpml_layers=0, **profiles)
    grid = sim.freeze_mesh()
    nodes = np.asarray(sim.realized_geometry().nodes[2])
    widths = np.r_[np.diff(nodes), np.diff(nodes)[-1]]
    # Different E components sample different z positions. Include half fill
    # and samples on either side of the shared face.
    faces = [nodes[4], nodes[4]+widths[4]/2, nodes[4]-.1*widths[4]]
    for face in faces:
        def evaluate(z):
            pairs = [(Box((-.1, -.1, -.1), (.1, .1, z)), 3.38),
                     (Box((-.1, -.1, z), (.1, .1, .1)), 10.2)]
            if reverse:
                pairs.reverse()
            if path.endswith('nodes'):
                return compute_node_tensor_eps(grid, pairs).nodes.xi[:3]
            if path == 'kottke_pec':
                return tuple(1/a for a in compute_inv_eps_tensor_diag(grid, dielectric_shapes=pairs))
            return (compute_smoothed_eps_nonuniform(grid, pairs) if graded
                    else compute_smoothed_eps(grid, pairs))
        coordinate = jnp.float32(face) if traced else np.float64(face)
        # Amendment 3: host SDF arithmetic is float64 (1e-6 relative);
        # traced float32 subtracts box coordinates before dividing by cell
        # length. Amendment 4 corrects |x|/L to ~100 for the +/-0.1 m
        # corners: one rounding gives ~6e-6 in fill, consistent with the
        # observed 7.5e-6 relative error. The 2e-5 bound is unchanged.
        # Returned arrays alone do not identify the
        # arithmetic dtype: the host path also casts its output to float32.
        rtol = 2e-5 if coordinate.dtype == np.dtype('float32') else 1e-6
        result = jax.jit(evaluate)(coordinate) if traced else evaluate(coordinate)
        for axis, values in enumerate(result):
            is_node = path.endswith('nodes')
            position = nodes[4] + (widths[4]/2 if axis == 2 and not is_node else 0.)
            length = (widths[4] if axis == 2 and not is_node else
                      (widths[3]+widths[4])/2)
            f = np.clip(.5+(position-face)/length, 0., 1.)
            expected = 1/(f/10.2+(1-f)/3.38) if axis == 2 else f*10.2+(1-f)*3.38
            if is_node:
                expected = 1/expected
            np.testing.assert_allclose(np.asarray(values)[4, 4, 4], expected, rtol=rtol)


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('traced', [False, True])
def test_mask_only_outside_is_clamped_primal_cell(graded, traced):
    profiles = dict(dz_profile=np.r_[np.full(4, .0004), np.full(4, .0006)]) if graded else {}
    sim = Simulation(domain=(.004, .004, .004), dx=.0005, freq_max=12e9,
                     boundary='pec', cpml_layers=0, **profiles)
    grid = sim.freeze_mesh()
    class MaskOnly:
        def __init__(self, enabled):
            self.enabled = enabled
        def mask(self, grid):
            # The low boundary cell owns points clamped from below the grid.
            return jnp.broadcast_to((jnp.arange(grid.shape[2]) == 0)[None, None, :], grid.shape) & self.enabled
    def evaluate(enabled):
        shapes = [(MaskOnly(enabled), 10.2),
                  (Box((-.1, -.1, 0.), (.1, .1, .003)), 3.38)]
        return (compute_smoothed_eps_nonuniform(grid, shapes) if graded
                else compute_smoothed_eps(grid, shapes))
    result = jax.jit(evaluate)(True) if traced else evaluate(True)
    np.testing.assert_allclose(np.asarray(result[0])[4, 4, 0], (3.38+10.2)/2, rtol=1e-6)


@pytest.mark.parametrize('path', ['uniform', 'nodes'])
@pytest.mark.parametrize('traced', [False, True], ids=['host', 'traced'])
def test_same_permittivity_mask_body_across_sdf_face(path, traced):
    """Review finding 1: the SDF surface lies inside a homogeneous eps=4 body."""
    sim = Simulation(domain=(.004, .004, .004), dx=.0005, freq_max=12e9,
                     boundary='pec', cpml_layers=0)
    grid = sim.freeze_mesh()

    class MaskAbove:
        def mask(self, grid):
            return np.broadcast_to((np.arange(grid.shape[2]) >= 4)[None, None, :],
                                   grid.shape).copy()

    def evaluate(top):
        pairs = [(Box((-.1, -.1, -.1), (.1, .1, top)), 4.0), (MaskAbove(), 4.0)]
        if path == 'nodes':
            return tuple(1 / a for a in compute_node_tensor_eps(grid, pairs).nodes.xi[:3])
        return compute_smoothed_eps(grid, pairs)

    coordinate = jnp.float32(.0022) if traced else np.float64(.0022)
    # Host-f64 SDF arithmetic and traced-f32 arithmetic, per Amendment 4.
    rtol = 2e-5 if coordinate.dtype == np.dtype('float32') else 1e-6
    result = jax.jit(evaluate)(coordinate) if traced else evaluate(coordinate)
    for values in result:
        np.testing.assert_allclose(np.asarray(values)[4, 4, 4], 4.0, rtol=rtol)


@pytest.mark.parametrize('traced', [False, True], ids=['host', 'traced'])
def test_gap_reads_same_medium_as_nearest_box_alone(traced):
    """A half-cell beyond the nearest face is still in the 0.45 mm gap."""
    sim = Simulation(domain=(.004, .004, .004), dx=.0005, freq_max=12e9,
                     boundary='pec', cpml_layers=0)
    grid = sim.freeze_mesh()
    nodes = np.asarray(sim.realized_geometry().nodes[2])[:grid.shape[2]]

    def evaluate(faces):
        lower = (Box((-.1, -.1, -.1), (.1, .1, faces[0])), 4.0)
        upper = (Box((-.1, -.1, faces[1]), (.1, .1, .1)), 6.0)
        return tuple(compute_smoothed_eps(grid, pairs)
                     for pairs in ([lower, upper], [lower], [upper]))

    faces = (jnp.asarray([.00202, .00247], dtype=jnp.float32) if traced
             else np.asarray([.00202, .00247], dtype=np.float64))
    # As above, tolerance follows the SDF arithmetic dtype, not the returned
    # array dtype: host float64 uses 1e-6; traced float32 uses 2e-5.
    rtol = 2e-5 if faces.dtype == np.dtype('float32') else 1e-6
    combined, lower, upper = jax.jit(evaluate)(faces) if traced else evaluate(faces)
    lo, hi = np.asarray(faces)
    for axis in range(3):
        positions = nodes + (grid.dx / 2 if axis == 2 else 0.)
        near_lower = np.abs(positions - lo) < np.abs(positions - hi)
        distance = np.where(near_lower, positions - lo, hi - positions)
        normal = np.where(near_lower, 1., -1.)
        q = positions + (grid.dx / 2 - distance) * normal
        selected = (np.abs(distance) < grid.dx / 2) & (q > lo) & (q < hi)
        assert np.any(selected)
        expected = np.where(near_lower[None, None, :], np.asarray(lower[axis]),
                            np.asarray(upper[axis]))[..., selected]
        actual = np.asarray(combined[axis])[..., selected]
        print(f'GAP traced={traced} axis={axis} z={positions[selected].tolist()} '
              f'actual={actual[4, 4].tolist()} expected={expected[4, 4].tolist()}')
        np.testing.assert_allclose(actual, expected, rtol=rtol)
