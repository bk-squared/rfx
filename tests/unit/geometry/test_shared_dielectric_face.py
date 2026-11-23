"""Bulk media on both sides of a shared face, without time stepping.
The host SDF arithmetic is float64 even though its output is cast to float32."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.errors import RfxUnsupported
from rfx.geometry.node_tensor import compute_node_tensor_eps
from rfx.geometry.smoothing import (
    compute_inv_eps_tensor_diag, compute_smoothed_eps, compute_smoothed_eps_nonuniform,
)


@pytest.mark.parametrize("reverse,traced,path", [
    pytest.param(r, t, p, id=f"{'BA' if r else 'AB'}-{'traced' if t else 'host'}-{p}")
    for r in (False, True) for t in (False, True)
    for p in ('uniform', 'graded', 'kottke_pec', 'nodes', 'graded_nodes')
    if r == t or (p in ('nodes', 'graded_nodes') and not t)
])
def test_shared_face_operands(path, traced, reverse):
    """The 2e-5 traced-float32 bound covers an earlier observed 7.5e-6 relative error from subtracting the +/-0.1 m box corners."""
    graded = path.startswith('graded')
    profiles = dict(dx_profile=np.full(8, .0005), dy_profile=np.full(8, .0005),
                    dz_profile=np.r_[np.full(4, .0004), np.full(4, .0006)]) if graded else {}
    sim = Simulation(domain=(.004, .004, .004), dx=.0005, freq_max=12e9,
                     boundary='pec', cpml_layers=0, **profiles)
    grid = sim.freeze_mesh()
    nodes = np.asarray(sim.realized_geometry().nodes[2])
    widths = np.r_[np.diff(nodes), np.diff(nodes)[-1]]
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
                return tuple(1/a for a in compute_inv_eps_tensor_diag(grid, dielectric_shapes=pairs, dtype='float32'))
            return (compute_smoothed_eps_nonuniform(grid, pairs, dtype='float32') if graded
                    else compute_smoothed_eps(grid, pairs, dtype='float32'))
        coordinate = jnp.float32(face) if traced else np.float64(face)
        rtol = 2e-5 if coordinate.dtype == np.dtype('float32') else 1e-6
        if traced and path in {'nodes', 'graded_nodes'}:
            with pytest.raises(RfxUnsupported, match="traced shape coordinate"):
                jax.jit(evaluate)(coordinate)
            return
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
    """The low boundary cell owns points clamped from below the grid."""
    profiles = dict(dz_profile=np.r_[np.full(4, .0004), np.full(4, .0006)]) if graded else {}
    sim = Simulation(domain=(.004, .004, .004), dx=.0005, freq_max=12e9,
                     boundary='pec', cpml_layers=0, **profiles)
    grid = sim.freeze_mesh()
    class MaskOnly:
        def __init__(self, enabled):
            self.enabled = enabled
        def mask(self, grid):
            return jnp.broadcast_to((jnp.arange(grid.shape[2]) == 0)[None, None, :], grid.shape) & self.enabled
    def evaluate(enabled):
        shapes = [(MaskOnly(enabled), 10.2),
                  (Box((-.1, -.1, 0.), (.1, .1, .003)), 3.38)]
        return (compute_smoothed_eps_nonuniform(grid, shapes, dtype='float32') if graded
                else compute_smoothed_eps(grid, shapes, dtype='float32'))
    result = jax.jit(evaluate)(True) if traced else evaluate(True)
    np.testing.assert_allclose(np.asarray(result[0])[4, 4, 0], (3.38+10.2)/2, rtol=1e-6)


@pytest.mark.parametrize('path', ['uniform', 'nodes'])
@pytest.mark.parametrize('traced', [False, True], ids=['host', 'traced'])
def test_same_permittivity_mask_body_across_sdf_face(path, traced):
    """Review finding 1: the SDF surface lies inside a homogeneous eps=4 body.
    Tolerance follows host-float64 or traced-float32 SDF arithmetic."""
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
        return compute_smoothed_eps(grid, pairs, dtype='float32')

    coordinate = jnp.float32(.0022) if traced else np.float64(.0022)
    rtol = 2e-5 if coordinate.dtype == np.dtype('float32') else 1e-6
    if traced and path in {'nodes', 'graded_nodes'}:
        with pytest.raises(RfxUnsupported, match="traced shape coordinate"):
            jax.jit(evaluate)(coordinate)
        return
    result = jax.jit(evaluate)(coordinate) if traced else evaluate(coordinate)
    for values in result:
        np.testing.assert_allclose(np.asarray(values)[4, 4, 4], 4.0, rtol=rtol)


@pytest.mark.parametrize('traced', [False, True], ids=['host', 'traced'])
def test_cell_holding_both_walls_reads_content(traced):
    """A cell holding both walls reads its content; other cells keep one wall.
    Tolerance follows the SDF arithmetic dtype (1e-6 for host float64, 2e-5 for traced float32) rather than the returned array dtype."""
    sim = Simulation(domain=(.004, .004, .004), dx=.0005, freq_max=12e9,
                     boundary='pec', cpml_layers=0)
    grid = sim.freeze_mesh()
    nodes = np.asarray(sim.realized_geometry().nodes[2])[:grid.shape[2]]

    def evaluate(faces):
        lower = (Box((-.1, -.1, -.1), (.1, .1, faces[0])), 4.0)
        upper = (Box((-.1, -.1, faces[1]), (.1, .1, .1)), 6.0)
        return tuple(compute_smoothed_eps(grid, pairs, dtype='float32')
                     for pairs in ([lower, upper], [lower], [upper]))

    faces = (jnp.asarray([.00202, .00247], dtype=jnp.float32) if traced
             else np.asarray([.00202, .00247], dtype=np.float64))
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
        if axis == 2:
            expected[..., np.flatnonzero(selected) == 4] = 1 / (.04/4 + .90 + .06/6)
        actual = np.asarray(combined[axis])[..., selected]
        print(f'GAP traced={traced} axis={axis} z={positions[selected].tolist()} '
              f'actual={actual[4, 4].tolist()} expected={expected[4, 4].tolist()}')
        np.testing.assert_allclose(actual, expected, rtol=rtol)
